"""Schema and determinism contract for the writer module."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from sentinel.review import writer


@pytest.mark.parametrize("table", list(writer.SCHEMAS))
def test_empty_returns_the_correct_schema(table: str) -> None:
    frame = writer.empty(table)
    assert frame.is_empty()
    assert list(frame.schema.keys()) == list(writer.SCHEMAS[table].keys())


def test_finalize_of_no_rows_returns_typed_empty() -> None:
    frame = writer.finalize([], "review_resolution_log")
    assert frame.is_empty()
    assert list(frame.schema.keys()) == list(writer.REVIEW_RESOLUTION_LOG_SCHEMA.keys())


def _advisory_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "code": "b",
        "severity": "warn",
        "scope": "run",
        "n_cases": 1,
        "detail": "",
        "review_definition_version": "v1",
    }
    base.update(overrides)
    return base


def test_finalize_sorts_by_the_declared_key() -> None:
    rows = [_advisory_row(code="b"), _advisory_row(code="a")]
    frame = writer.finalize(rows, "review_advisories")
    assert frame["code"].to_list() == ["a", "b"]


def test_finalize_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        writer.finalize([{"code": "a"}], "review_advisories")


def test_finalize_rejects_unknown_columns() -> None:
    rows = [
        {
            **{k: None for k in writer.REVIEW_ADVISORIES_SCHEMA},
            "surprise": "extra",
        }
    ]
    with pytest.raises(ValueError, match="unknown columns"):
        writer.finalize(rows, "review_advisories")


def test_unknown_table_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        writer.finalize([], "not_a_real_table")
    with pytest.raises(KeyError):
        writer.empty("not_a_real_table")


def test_write_table_round_trips(tmp_path: Path) -> None:
    frame = writer.empty("human_review_queue")
    path = writer.write_table(frame, tmp_path / "human_review_queue_20260101T000000Z.parquet")
    assert path.exists()
    read_back = pl.read_parquet(path)
    assert read_back.schema == frame.schema


def test_every_table_lands_in_the_review_layer() -> None:
    assert set(writer.LAYERS.values()) == {"review"}
