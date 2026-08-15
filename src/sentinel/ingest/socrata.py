"""Minimal, explicit Socrata (SODA 2.1) client.

Design notes
------------
Pagination is written out in the open rather than delegated to a third-party
SDK (e.g. ``sodapy``). The paging contract is the single most important thing
to get right in this component: if pages silently overlap or skip, every
downstream model trains on corrupted data. Keeping it visible makes it
reviewable and directly unit-testable.

Two behaviours are load-bearing and were verified against the live API
(see docs/api/socrata_findings.md):

1. ``$order`` is mandatory. Socrata does not guarantee a stable row order
   across requests. Without an explicit total order, ``$offset`` paging can
   duplicate or drop rows between pages.
2. Errors are loud. Anything other than a retryable transient condition raises
   immediately with the status, URL and response body attached.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Socrata exposes the dataset's field names and declared types as response
# headers on every request. Capturing them gives us the source-declared schema
# without a second metadata call.
FIELDS_HEADER = "X-SODA2-Fields"
TYPES_HEADER = "X-SODA2-Types"

# 429 (throttled) and 5xx are transient. Everything else in 4xx means our
# request was wrong, and retrying it would just hide the bug.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class SocrataError(RuntimeError):
    """Base class for Socrata client failures."""


class SocrataRequestError(SocrataError):
    """A request failed permanently (non-retryable, or retry budget exhausted)."""

    def __init__(
        self,
        message: str,
        *,
        url: str,
        status_code: int | None = None,
        body: str | None = None,
    ) -> None:
        self.url = url
        self.status_code = status_code
        self.body = body
        detail = f"{message} | url={url}"
        if status_code is not None:
            detail += f" | status={status_code}"
        if body:
            detail += f" | body={body[:500]}"
        super().__init__(detail)


class SocrataResponseError(SocrataError):
    """The response was not the JSON array of records we require."""


@dataclass
class Page:
    """One page of records plus the source-declared schema that came with it."""

    records: list[dict[str, Any]]
    offset: int
    # The $limit actually requested for this page, recorded so the manifest can
    # replay the exact request sequence without re-deriving it.
    limit: int = 0
    field_names: list[str] = field(default_factory=list)
    field_types: list[str] = field(default_factory=list)


def build_params(
    *,
    limit: int,
    offset: int,
    order_column: str,
    select: Sequence[str] | None = None,
) -> dict[str, str]:
    """Build the SODA query parameters for one page.

    Pure function with no I/O so the exact wire contract can be asserted in
    tests. ``$order`` is always included; see the module docstring.

    ``select`` is optional and, when given, becomes ``$select``. It exists
    because this endpoint omits its ``:@computed_region_*`` columns whenever
    ``$order`` is present unless they are explicitly selected. See
    docs/api/socrata_findings.md.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    if not order_column:
        raise ValueError("order_column is required for stable pagination")

    params = {
        "$limit": str(limit),
        "$offset": str(offset),
        "$order": order_column,
    }
    if select:
        params["$select"] = ",".join(select)
    return params


def _parse_header_list(raw: str | None) -> list[str]:
    """Parse an X-SODA2-* header, which is a JSON array of strings."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Could not parse SODA2 schema header: %r", raw)
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


class SocrataClient:
    """Paginating client for a single Socrata resource."""

    def __init__(
        self,
        *,
        resource_url: str,
        page_size: int,
        order_column: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        app_token: str | None = None,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.resource_url = resource_url
        self.page_size = page_size
        self.order_column = order_column
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        # Injected so retry tests run instantly instead of actually sleeping.
        self._sleep = sleep

        headers = {"Accept": "application/json"}
        if app_token:
            headers["X-App-Token"] = app_token

        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout, headers=headers)

        # Schema declared by the most recent response. Retained on the client
        # because a request that returns zero records still carries the
        # X-SODA2-* headers, and `iter_pages` does not yield empty pages.
        self.last_field_names: list[str] = []
        self.last_field_types: list[str] = []

    # --- lifecycle -------------------------------------------------------

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> SocrataClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- requests --------------------------------------------------------

    def _request_with_retry(self, params: dict[str, str]) -> httpx.Response:
        """Issue one GET with bounded retries on transient failures only.

        Retries: 429, 5xx, timeouts, transport errors.
        Does NOT retry: 4xx other than 429. That is a defect in our query and
        must surface immediately rather than being masked by three attempts.
        """
        attempts = self.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = self._client.get(self.resource_url, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt == attempts:
                    raise SocrataRequestError(
                        f"Request failed after {attempts} attempt(s): {exc!r}",
                        url=self.resource_url,
                    ) from exc
                self._backoff(attempt, reason=type(exc).__name__)
                continue

            if response.status_code in RETRYABLE_STATUS_CODES:
                if attempt == attempts:
                    raise SocrataRequestError(
                        f"Transient HTTP {response.status_code} persisted after "
                        f"{attempts} attempt(s)",
                        url=str(response.request.url),
                        status_code=response.status_code,
                        body=response.text,
                    )
                self._backoff(attempt, reason=f"HTTP {response.status_code}")
                continue

            if response.status_code >= 400:
                # Non-retryable. Fail loudly with the server's explanation.
                raise SocrataRequestError(
                    f"Non-retryable HTTP {response.status_code}",
                    url=str(response.request.url),
                    status_code=response.status_code,
                    body=response.text,
                )

            return response

        # Defensive: every path above either returns or raises.
        raise SocrataRequestError(
            f"Request loop exited without a response: {last_error!r}",
            url=self.resource_url,
        )

    def _backoff(self, attempt: int, *, reason: str) -> None:
        delay = self.retry_backoff * (2 ** (attempt - 1))
        logger.warning(
            "Retrying after %s (attempt %d/%d), sleeping %.1fs",
            reason,
            attempt,
            self.max_retries + 1,
            delay,
        )
        self._sleep(delay)

    def discover_fields(self) -> list[str]:
        """Ask the API which fields the dataset currently has.

        Deliberately issues an unordered single-row request: this endpoint
        reports its full field list (including the ``:@computed_region_*``
        spatial annotations) only when ``$order`` is absent. Discovering the
        list rather than hardcoding it means a new upstream column is picked up
        automatically instead of being silently dropped by a stale ``$select``.
        """
        response = self._request_with_retry({"$limit": "1"})
        fields = _parse_header_list(response.headers.get(FIELDS_HEADER))
        logger.info("Discovered %d declared fields from the source", len(fields))
        return fields

    def fetch_page(
        self,
        *,
        offset: int,
        limit: int,
        select: Sequence[str] | None = None,
    ) -> Page:
        """Fetch exactly one page of records."""
        params = build_params(
            limit=limit,
            offset=offset,
            order_column=self.order_column,
            select=select,
        )
        logger.info("Requesting page: offset=%d limit=%d", offset, limit)

        response = self._request_with_retry(params)

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise SocrataResponseError(
                f"Response was not valid JSON (offset={offset}): {exc}"
            ) from exc

        if not isinstance(payload, list):
            raise SocrataResponseError(
                f"Expected a JSON array of records, got {type(payload).__name__} (offset={offset})"
            )
        for index, record in enumerate(payload):
            if not isinstance(record, dict):
                raise SocrataResponseError(
                    f"Record {index} at offset={offset} is {type(record).__name__}, expected object"
                )

        page = Page(
            records=payload,
            offset=offset,
            limit=limit,
            field_names=_parse_header_list(response.headers.get(FIELDS_HEADER)),
            field_types=_parse_header_list(response.headers.get(TYPES_HEADER)),
        )
        if page.field_names:
            self.last_field_names = page.field_names
        if page.field_types:
            self.last_field_types = page.field_types

        logger.info("Received %d rows at offset=%d", len(page.records), offset)
        return page

    def iter_pages(
        self,
        *,
        total_limit: int | None = None,
        select: Sequence[str] | None = None,
    ) -> Iterator[Page]:
        """Yield pages until the dataset (or ``total_limit``) is exhausted.

        A generator rather than a list so memory stays bounded: a full pull is
        ~314k records and we never need more than one page resident at a time.

        Termination conditions, in order of evaluation:
          1. ``total_limit`` reached. The final page is sized to fit exactly.
          2. An empty page: we have run off the end of the dataset.
          3. A short page (fewer rows than requested). Socrata returns fewer
             rows than ``$limit`` only when no more remain.
        """
        offset = 0
        fetched = 0

        while True:
            if total_limit is not None:
                remaining = total_limit - fetched
                if remaining <= 0:
                    return
                limit = min(self.page_size, remaining)
            else:
                limit = self.page_size

            page = self.fetch_page(offset=offset, limit=limit, select=select)
            received = len(page.records)

            if received == 0:
                logger.info("Empty page at offset=%d; pagination complete", offset)
                return

            yield page

            fetched += received
            offset += received

            if received < limit:
                logger.info(
                    "Short page (%d < %d) at offset=%d; pagination complete",
                    received,
                    limit,
                    page.offset,
                )
                return
