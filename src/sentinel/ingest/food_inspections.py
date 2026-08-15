"""Ingest the Chicago Food Inspections dataset into the raw data layer.

Raw means raw. The one and only transformation applied here is the unavoidable
one: JSON records become Parquet columns. Specifically:

* Every column is written as ``Utf8``. The Socrata API returns every value as a
  JSON string, including columns it declares as ``number`` and
  ``calendar_date`` (verified: ``"inspection_id":"2641210"``). Casting here
  would turn any malformed value into a silent null and would mean the raw file
  no longer matches the source. Typing is an explicit, tested step in a later
  component.
* The nested ``location`` object is serialized back to its JSON string rather
  than being flattened or dropped, so no information is lost.
* Missing keys become nulls. Nothing is ever defaulted or invented.

No Sentinel-specific filtering, deduplication, or business logic happens here.
That belongs to Component 2 onward.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from sentinel import __version__
from sentinel.config import Settings
from sentinel.ingest.manifest import (
    IngestionManifest,
    compute_sha256,
    manifest_path_for,
    write_manifest,
)
from sentinel.ingest.socrata import Page, SocrataClient, build_params

logger = logging.getLogger(__name__)

DATASET_SLUG = "food_inspections"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


@dataclass
class IngestionResult:
    """Outcome of one ingestion run."""

    parquet_path: Path
    manifest_path: Path
    manifest: IngestionManifest

    @property
    def row_count(self) -> int:
        return self.manifest.row_count


def _stringify(value: Any) -> str | None:
    """Coerce one JSON value to its raw string form.

    Scalars pass through as-is (they already arrive as strings). Nested
    structures such as ``location`` are re-serialized to compact JSON so the
    original content survives round-trip without being flattened.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    # bool/int/float should not occur given the API's string encoding, but if
    # the source ever changes we preserve rather than reject the value.
    return str(value)


def records_to_frame(
    records: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str],
) -> pl.DataFrame:
    """Build an all-``Utf8`` Polars frame with a fixed, explicit column order.

    Passing ``columns`` explicitly (rather than inferring from the records)
    keeps the schema stable even if a given page happens to omit a key that is
    null for every row on that page.
    """
    data: dict[str, list[str | None]] = {
        name: [_stringify(record.get(name)) for record in records] for name in columns
    }
    return pl.DataFrame(data, schema={name: pl.Utf8 for name in columns})


def _resolve_columns(
    observed: list[str],
    declared: list[str],
) -> list[str]:
    """Decide the output column set, complaining loudly about any divergence.

    ``declared`` is what the API's X-SODA2-Fields header says the dataset
    contains. ``observed`` is what actually appeared in the records. A mismatch
    means the upstream schema moved, which is exactly the sort of thing that
    must not pass silently.
    """
    if not declared:
        logger.warning("No X-SODA2-Fields header seen; falling back to observed keys")
        return observed

    declared_set = set(declared)
    observed_set = set(observed)

    missing = [name for name in declared if name not in observed_set]
    extra = [name for name in observed if name not in declared_set]

    if missing:
        # Common and benign: sparse columns absent from every fetched record.
        # Still worth stating, because it is also how a dropped column looks.
        logger.warning(
            "Declared fields absent from all fetched records (kept as null columns): %s",
            ", ".join(missing),
        )
    if extra:
        logger.warning(
            "Records contain fields not declared by X-SODA2-Fields (kept anyway): %s",
            ", ".join(extra),
        )

    # Declared order first (it is the dataset's canonical order), then any
    # undeclared extras. Nothing is ever dropped.
    return list(declared) + extra


def ingest_food_inspections(
    settings: Settings,
    *,
    row_limit: int | None,
    output_dir: Path | None = None,
    client: SocrataClient | None = None,
) -> IngestionResult:
    """Pull food inspection records and write raw Parquet plus a manifest.

    Args:
        settings: Resolved configuration.
        row_limit: Maximum rows to fetch, or ``None`` for the whole dataset.
        output_dir: Override the destination directory (used by tests).
        client: Pre-built client (used by tests); otherwise one is created and
            closed here.

    Returns:
        The paths written and the manifest describing them.
    """
    retrieved_at = datetime.now(UTC)
    mode = "full" if row_limit is None else "dev"
    destination = output_dir or settings.food_inspections_raw_dir

    logger.info(
        "Starting ingestion | dataset=%s (%s) mode=%s row_limit=%s page_size=%d",
        settings.dataset_name,
        settings.dataset_id,
        mode,
        row_limit if row_limit is not None else "none (full dataset)",
        settings.page_size,
    )
    logger.info("Source: %s", settings.resource_url)

    owns_client = client is None
    active_client = client or SocrataClient(
        resource_url=settings.resource_url,
        page_size=settings.page_size,
        order_column=settings.order_column,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
        retry_backoff=settings.retry_backoff,
        app_token=settings.socrata_app_token,
    )

    # Pages are held as raw record lists until the full column set is known,
    # then converted in one pass. Holding dicts rather than per-page frames
    # avoids having to re-project mismatched schemas afterwards.
    pages: list[list[dict[str, Any]]] = []
    request_params: list[dict[str, str]] = []
    declared_fields: list[str] = []
    declared_types: list[str] = []
    observed_order: list[str] = []
    seen: set[str] = set()
    pages_fetched = 0
    total_rows = 0

    try:
        select = _resolve_select(active_client, settings)

        for page in active_client.iter_pages(total_limit=row_limit, select=select):
            pages_fetched += 1
            total_rows += len(page.records)

            request_params.append(
                build_params(
                    limit=page.limit,
                    offset=page.offset,
                    order_column=active_client.order_column,
                    select=select,
                )
            )

            _capture_schema(page, declared_fields, declared_types)
            for record in page.records:
                for key in record:
                    if key not in seen:
                        seen.add(key)
                        observed_order.append(key)

            pages.append(page.records)

            logger.info(
                "Page %d complete | rows_this_page=%d rows_total=%d",
                pages_fetched,
                len(page.records),
                total_rows,
            )
        # A zero-row result yields no pages, but the response still declared a
        # schema. Fall back to it so an empty raw file still has real columns.
        if not declared_fields:
            declared_fields = list(active_client.last_field_names)
        if not declared_types:
            declared_types = list(active_client.last_field_types)
    finally:
        if owns_client:
            active_client.close()

    columns = _resolve_columns(observed_order, declared_fields)

    if pages:
        frame = pl.concat(
            [records_to_frame(records, columns=columns) for records in pages],
            how="vertical",
        )
    else:
        logger.warning("No records returned; writing an empty raw file")
        frame = records_to_frame([], columns=columns)

    destination.mkdir(parents=True, exist_ok=True)
    stamp = retrieved_at.strftime(TIMESTAMP_FORMAT)
    parquet_path = destination / f"{DATASET_SLUG}_{stamp}.parquet"

    # Timestamped filename: a re-run never overwrites previously downloaded raw
    # data. Raw data is append-only by construction.
    frame.write_parquet(parquet_path, compression="zstd")
    logger.info("Wrote raw Parquet: %s (%d rows)", parquet_path, frame.height)

    manifest = IngestionManifest(
        source="Chicago Data Portal (Socrata SODA 2.1)",
        source_url=settings.resource_url,
        dataset_id=settings.dataset_id,
        dataset_name=settings.dataset_name,
        retrieved_at=retrieved_at,
        code_version=__version__,
        mode=mode,
        row_limit=row_limit,
        page_size=settings.page_size,
        pages_fetched=pages_fetched,
        order_column=settings.order_column,
        request_params=request_params,
        row_count=frame.height,
        column_names=list(frame.columns),
        socrata_field_names=declared_fields,
        socrata_field_types=declared_types,
        parquet_schema={name: str(dtype) for name, dtype in frame.schema.items()},
        output_path=str(parquet_path),
        output_bytes=parquet_path.stat().st_size,
        sha256=compute_sha256(parquet_path),
    )
    manifest_path = write_manifest(manifest, manifest_path_for(parquet_path))

    logger.info(
        "Ingestion complete | rows=%d pages=%d parquet=%s manifest=%s",
        manifest.row_count,
        pages_fetched,
        parquet_path,
        manifest_path,
    )
    return IngestionResult(
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _resolve_select(client: SocrataClient, settings: Settings) -> list[str] | None:
    """Decide whether to send an explicit ``$select``, and with which fields.

    Returns ``None`` to let the API choose the projection (which, with $order
    present, silently excludes the ``:@computed_region_*`` columns).
    """
    if not settings.include_computed_regions:
        logger.info("include_computed_regions=False; letting the API choose the projection")
        return None

    fields = client.discover_fields()
    if not fields:
        logger.warning("Field discovery returned nothing; falling back to no $select")
        return None

    computed = [name for name in fields if name.startswith(":@computed_region_")]
    logger.info(
        "Selecting %d discovered fields explicitly (%d computed-region columns)",
        len(fields),
        len(computed),
    )
    return fields


def _capture_schema(page: Page, fields: list[str], types: list[str]) -> None:
    """Record the source-declared schema from the first page that carries it."""
    if page.field_names and not fields:
        fields.extend(page.field_names)
    if page.field_types and not types:
        types.extend(page.field_types)
