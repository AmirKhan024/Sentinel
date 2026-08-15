"""Tests for the DuckDB query layer over raw Parquet."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from sentinel.query import duckdb_queries
from tests.conftest import FIELD_NAMES, make_records


@pytest.fixture
def raw_parquet(tmp_path: Path) -> Path:
    """A small raw-shaped Parquet file: all columns Utf8, as ingestion writes."""
    from sentinel.ingest.food_inspections import records_to_frame

    frame = records_to_frame(make_records(7), columns=FIELD_NAMES)
    path = tmp_path / "food_inspections_20260815T120000Z.parquet"
    frame.write_parquet(path)
    return path


def test_row_count(raw_parquet: Path) -> None:
    result = duckdb_queries.run_named_query(raw_parquet, "row_count")
    assert result.columns == ["row_count"]
    assert result.rows == [(7,)]


def test_unique_licenses(raw_parquet: Path) -> None:
    result = duckdb_queries.run_named_query(raw_parquet, "unique_licenses")
    unique, nulls = result.rows[0]
    assert unique == 7
    assert nulls == 0


def test_inspection_types_grouped(raw_parquet: Path) -> None:
    result = duckdb_queries.run_named_query(raw_parquet, "inspection_types")
    assert result.columns == ["inspection_type", "n"]
    assert result.rows == [("Canvass", 7)]


def test_schema_query_reports_all_varchar(raw_parquet: Path) -> None:
    """Raw Parquet is all strings, so DuckDB must see VARCHAR everywhere."""
    result = duckdb_queries.describe(raw_parquet)
    types = {row[1] for row in result.rows}
    assert types == {"VARCHAR"}
    assert len(result.rows) == len(FIELD_NAMES)


def test_unknown_query_name_raises(raw_parquet: Path) -> None:
    with pytest.raises(KeyError, match="Unknown query"):
        duckdb_queries.run_named_query(raw_parquet, "definitely_not_a_query")


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        duckdb_queries.run_named_query(tmp_path / "absent.parquet", "row_count")


def test_latest_parquet_picks_newest_by_sortable_name(tmp_path: Path) -> None:
    frame = pl.DataFrame({"a": ["1"]})
    for stamp in ("20260101T000000Z", "20260815T120000Z", "20260301T000000Z"):
        frame.write_parquet(tmp_path / f"food_inspections_{stamp}.parquet")

    assert duckdb_queries.latest_parquet(tmp_path).name == (
        "food_inspections_20260815T120000Z.parquet"
    )


def test_latest_parquet_raises_with_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sentinel ingest"):
        duckdb_queries.latest_parquet(tmp_path)
