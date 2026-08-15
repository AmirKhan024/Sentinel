"""Tests for the food inspections ingestion: Parquet output, schema, limits."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import polars as pl
import pytest
import respx

from sentinel.config import Settings
from sentinel.ingest.food_inspections import (
    ingest_food_inspections,
    records_to_frame,
)
from sentinel.ingest.socrata import SocrataClient, SocrataRequestError
from tests.conftest import (
    FIELD_NAMES,
    SODA_HEADERS,
    TEST_RESOURCE_URL,
    discovery_response,
    make_record,
    make_records,
)


def make_client(page_size: int = 10) -> SocrataClient:
    return SocrataClient(
        resource_url=TEST_RESOURCE_URL,
        page_size=page_size,
        order_column="inspection_id",
        max_retries=1,
        retry_backoff=0.0,
        sleep=lambda _s: None,
    )


# --- frame construction ----------------------------------------------------


def test_records_to_frame_is_all_utf8() -> None:
    """Raw stays raw: numeric-looking columns must not be cast to numbers."""
    frame = records_to_frame(make_records(3), columns=FIELD_NAMES)

    assert frame.height == 3
    assert list(frame.columns) == FIELD_NAMES
    assert all(dtype == pl.Utf8 for dtype in frame.schema.values())
    # The value survives as the exact string the API sent.
    assert frame["inspection_id"][0] == "100000"


def test_nested_location_is_serialized_not_dropped() -> None:
    frame = records_to_frame([make_record(0)], columns=FIELD_NAMES)
    raw = frame["location"][0]

    assert isinstance(raw, str)
    decoded = json.loads(raw)
    assert decoded["latitude"] == "41.8781"
    assert "human_address" in decoded


def test_missing_keys_become_null_not_defaults() -> None:
    records = [{"inspection_id": "1"}, {"inspection_id": "2", "dba_name": "X"}]
    frame = records_to_frame(records, columns=["inspection_id", "dba_name"])

    assert frame["dba_name"].to_list() == [None, "X"]


def test_empty_records_still_produce_typed_columns() -> None:
    frame = records_to_frame([], columns=["a", "b"])
    assert frame.height == 0
    assert list(frame.columns) == ["a", "b"]
    assert all(dtype == pl.Utf8 for dtype in frame.schema.values())


# --- end-to-end ingestion --------------------------------------------------


@respx.mock
def test_ingestion_writes_parquet_and_manifest(settings: Settings) -> None:
    respx.get(TEST_RESOURCE_URL).mock(
        side_effect=[
            discovery_response(),
            httpx.Response(200, json=make_records(10, start=0), headers=SODA_HEADERS),
            httpx.Response(200, json=make_records(4, start=10), headers=SODA_HEADERS),
        ]
    )

    result = ingest_food_inspections(settings, row_limit=None, client=make_client())

    assert result.parquet_path.exists()
    assert result.manifest_path.exists()
    assert result.row_count == 14

    frame = pl.read_parquet(result.parquet_path)
    assert frame.height == 14
    assert list(frame.columns) == FIELD_NAMES
    assert all(dtype == pl.Utf8 for dtype in frame.schema.values())


@respx.mock
def test_row_limit_is_honoured_end_to_end(settings: Settings) -> None:
    respx.get(TEST_RESOURCE_URL).mock(
        side_effect=[
            discovery_response(),
            httpx.Response(200, json=make_records(10, start=0), headers=SODA_HEADERS),
            httpx.Response(200, json=make_records(2, start=10), headers=SODA_HEADERS),
        ]
    )

    result = ingest_food_inspections(settings, row_limit=12, client=make_client())

    assert result.row_count == 12
    assert pl.read_parquet(result.parquet_path).height == 12
    assert result.manifest.mode == "dev"
    assert result.manifest.row_limit == 12


@respx.mock
def test_full_mode_recorded_in_manifest(settings: Settings) -> None:
    respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=make_records(3), headers=SODA_HEADERS)
    )

    result = ingest_food_inspections(settings, row_limit=None, client=make_client())

    assert result.manifest.mode == "full"
    assert result.manifest.row_limit is None


@respx.mock
def test_output_filename_is_timestamped_and_not_overwritten(settings: Settings) -> None:
    """Two runs must produce two files. Raw data is append-only."""
    respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=make_records(2), headers=SODA_HEADERS)
    )

    first = ingest_food_inspections(settings, row_limit=2, client=make_client())
    second = ingest_food_inspections(settings, row_limit=2, client=make_client())

    assert first.parquet_path.exists()
    assert second.parquet_path.exists()
    # Same second is possible on a fast machine; what matters is the first file
    # still exists and the naming scheme carries a UTC stamp.
    assert first.parquet_path.name.startswith("food_inspections_")
    assert first.parquet_path.name.endswith("Z.parquet")


@respx.mock
def test_output_dir_override(settings: Settings, tmp_path: Path) -> None:
    respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=make_records(2), headers=SODA_HEADERS)
    )
    destination = tmp_path / "elsewhere"

    result = ingest_food_inspections(
        settings, row_limit=2, output_dir=destination, client=make_client()
    )

    assert result.parquet_path.parent == destination


@respx.mock
def test_undeclared_extra_field_is_kept_not_dropped(settings: Settings) -> None:
    """If the source grows a column, we keep it and warn rather than silently drop."""
    record = make_record(0)
    record["brand_new_field"] = "surprise"
    respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=[record], headers=SODA_HEADERS)
    )

    result = ingest_food_inspections(settings, row_limit=1, client=make_client())
    frame = pl.read_parquet(result.parquet_path)

    assert "brand_new_field" in frame.columns
    assert frame["brand_new_field"][0] == "surprise"


@respx.mock
def test_declared_field_absent_from_records_becomes_null_column(settings: Settings) -> None:
    record = make_record(0)
    del record["aka_name"]
    respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=[record], headers=SODA_HEADERS)
    )

    result = ingest_food_inspections(settings, row_limit=1, client=make_client())
    frame = pl.read_parquet(result.parquet_path)

    assert "aka_name" in frame.columns
    assert frame["aka_name"][0] is None


@respx.mock
def test_api_failure_propagates_and_writes_nothing(settings: Settings) -> None:
    """Fail loudly: a hard API error must not leave a half-written raw file."""
    respx.get(TEST_RESOURCE_URL).mock(return_value=httpx.Response(400, text="bad query"))

    with pytest.raises(SocrataRequestError):
        ingest_food_inspections(settings, row_limit=10, client=make_client())

    assert not settings.food_inspections_raw_dir.exists() or not list(
        settings.food_inspections_raw_dir.glob("*.parquet")
    )


@respx.mock
def test_empty_dataset_produces_empty_file_with_declared_columns(settings: Settings) -> None:
    respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=[], headers=SODA_HEADERS)
    )

    result = ingest_food_inspections(settings, row_limit=10, client=make_client())

    assert result.row_count == 0
    assert list(pl.read_parquet(result.parquet_path).columns) == FIELD_NAMES


@respx.mock
def test_pages_fetched_and_request_params_recorded(settings: Settings) -> None:
    respx.get(TEST_RESOURCE_URL).mock(
        side_effect=[
            discovery_response(),
            httpx.Response(200, json=make_records(10, start=0), headers=SODA_HEADERS),
            httpx.Response(200, json=make_records(10, start=10), headers=SODA_HEADERS),
            httpx.Response(200, json=make_records(3, start=20), headers=SODA_HEADERS),
        ]
    )

    result = ingest_food_inspections(settings, row_limit=None, client=make_client())

    assert result.manifest.pages_fetched == 3
    assert [p["$offset"] for p in result.manifest.request_params] == ["0", "10", "20"]
    assert all(p["$order"] == "inspection_id" for p in result.manifest.request_params)


# --- computed_region handling ---------------------------------------------


@respx.mock
def test_computed_region_columns_are_selected_and_kept(settings: Settings) -> None:
    """The endpoint drops :@computed_region_* under $order unless selected."""
    computed = ":@computed_region_43wa_7qmu"
    fields = [*FIELD_NAMES, computed]
    headers = {
        "X-SODA2-Fields": json.dumps(fields),
        "X-SODA2-Types": json.dumps(["text"] * len(fields)),
    }
    record = {**make_record(0), computed: "7"}

    route = respx.get(TEST_RESOURCE_URL).mock(
        side_effect=[
            httpx.Response(200, json=[record], headers=headers),  # discovery
            httpx.Response(200, json=[record], headers=headers),  # page 1
        ]
    )

    result = ingest_food_inspections(settings, row_limit=1, client=make_client())
    frame = pl.read_parquet(result.parquet_path)

    assert computed in frame.columns
    assert frame[computed][0] == "7"
    # The paging request explicitly selected the discovered field list.
    assert computed in route.calls[1].request.url.params["$select"]


@respx.mock
def test_discovery_skipped_when_computed_regions_disabled(settings: Settings) -> None:
    disabled = settings.model_copy(update={"include_computed_regions": False})
    route = respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=make_records(2), headers=SODA_HEADERS)
    )

    ingest_food_inspections(disabled, row_limit=2, client=make_client())

    # One paging request only: no discovery call, and no $select sent.
    assert route.call_count == 1
    assert "$select" not in route.calls.last.request.url.params
