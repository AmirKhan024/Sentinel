"""Live smoke test against the real Chicago Data Portal.

Marked ``live`` and deselected by default (see pyproject.toml ``addopts``), so
neither the normal test run nor CI depends on an external service.

Run explicitly with:

    uv run pytest -m live -v

Purpose: the mocked unit tests prove our code behaves correctly against the API
contract *as we understand it*. This test is the check on that understanding.
If Chicago changes the endpoint, the field list, or the string encoding, this
is what catches it.
"""

from __future__ import annotations

import pytest

from sentinel.config import Settings
from sentinel.ingest.socrata import SocrataClient

pytestmark = pytest.mark.live

SMOKE_ROWS = 5


@pytest.fixture
def live_client() -> SocrataClient:
    settings = Settings()
    return SocrataClient(
        resource_url=settings.resource_url,
        page_size=SMOKE_ROWS,
        order_column=settings.order_column,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
        retry_backoff=settings.retry_backoff,
        app_token=settings.socrata_app_token,
    )


def test_live_endpoint_returns_records_and_schema(live_client: SocrataClient) -> None:
    with live_client as client:
        page = client.fetch_page(offset=0, limit=SMOKE_ROWS)

    assert len(page.records) == SMOKE_ROWS
    assert page.field_names, "X-SODA2-Fields header missing; schema capture would break"
    assert len(page.field_names) == len(page.field_types)

    # The columns this project depends on must still exist.
    for required in ("inspection_id", "license_", "inspection_date", "results"):
        assert required in page.field_names


def test_live_values_are_still_string_encoded(live_client: SocrataClient) -> None:
    """The all-Utf8 raw contract rests on this. If it changes, we must know."""
    with live_client as client:
        page = client.fetch_page(offset=0, limit=1)

    record = page.records[0]
    assert isinstance(record["inspection_id"], str)
    assert isinstance(record["inspection_date"], str)


def test_live_pagination_pages_do_not_overlap(live_client: SocrataClient) -> None:
    """The core correctness property of $order + $offset paging."""
    with live_client as client:
        first = client.fetch_page(offset=0, limit=SMOKE_ROWS)
        second = client.fetch_page(offset=SMOKE_ROWS, limit=SMOKE_ROWS)

    first_ids = [r["inspection_id"] for r in first.records]
    second_ids = [r["inspection_id"] for r in second.records]

    assert not set(first_ids) & set(second_ids)
