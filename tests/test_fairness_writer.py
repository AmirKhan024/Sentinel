"""The artifact contract: ten schemas, total sort orders, and nulls that stay nulls.

Component 12 emits more nulls than any other component in this project -- an unsupported
group's value, a ratio whose denominator vanished, a capture rate for a group with no
positives. So the schema is passed to the ``DataFrame`` constructor rather than cast
afterwards, and these tests pin that: a column that is null for its first two hundred rows
must still arrive as ``Float64`` rather than ``Null``.
"""

from __future__ import annotations

import polars as pl
import pytest

from sentinel.fairness import writer
from sentinel.fairness.definitions import FAIRNESS_DEFINITION_VERSION

TABLES = tuple(writer.SCHEMAS)


# --- 1. the schema set ---------------------------------------------------------


def test_there_are_ten_tables_all_in_one_layer() -> None:
    """Unlike Component 9, which had to write to three: a group metric is not a prediction
    and not a trial, so there is nowhere else it could legitimately go.
    """
    assert len(writer.SCHEMAS) == 10
    assert set(writer.LAYERS.values()) == {"fairness"}


def test_the_manifest_is_keyed_to_the_metrics_table() -> None:
    assert writer.DATASET_SLUG == "fairness_group_metrics"
    assert writer.DATASET_SLUG in writer.SCHEMAS


@pytest.mark.parametrize("table", TABLES)
def test_every_table_has_a_sort_key_drawn_from_its_own_columns(table: str) -> None:
    assert table in writer.SORT_KEYS
    for key in writer.SORT_KEYS[table]:
        assert key in writer.SCHEMAS[table], f"{table} sorts on a column it does not have"


@pytest.mark.parametrize("table", TABLES)
def test_every_table_carries_the_definition_version(table: str) -> None:
    """So a row can never be read without knowing which contract produced it."""
    assert "fairness_definition_version" in writer.SCHEMAS[table]


@pytest.mark.parametrize("table", TABLES)
def test_an_empty_table_still_has_the_right_columns(table: str) -> None:
    """A run restricted to one model may legitimately write an empty attribution table."""
    frame = writer.empty(table)
    assert frame.height == 0
    assert list(frame.columns) == list(writer.SCHEMAS[table])


def test_an_unknown_table_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown table"):
        writer.empty("fairness_score")


# --- 2. no output column may make the artifact joinable -------------------------


@pytest.mark.parametrize("table", TABLES)
def test_no_schema_carries_an_outcome_or_a_probability(table: str) -> None:
    """These tables are keyed by group precisely so they cannot become features.

    A `target` or `score` column would undo that, turning a group summary into a per-row
    table one join away from re-entering a model as an input. ADR 0032.
    """
    forbidden = {"target", "score", "base_score", "calibrated_probability", "base_probability"}
    assert not forbidden & set(writer.SCHEMAS[table])


# --- 3. finalize: typing, ordering and rejection ---------------------------------


def _metric_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "model_name": "xgboost_platt",
        "stage": "calibrated",
        "group_definition": "community_area",
        "group_value": "1",
        "grain": "fold_set",
        "fold_set": "quarterly",
        "fold_id": "",
        "metric": "roc_auc",
        "metric_kind": "ranking",
        "k_name": "",
        "k": 0,
        "value": 0.61,
        "n_rows": 400,
        "n_positive": 200,
        "n_negative": 200,
        "group_status": "supported",
        "insufficient_reason": "",
        "fairness_definition_version": FAIRNESS_DEFINITION_VERSION,
    }
    row.update(overrides)
    return row


def test_a_column_that_is_null_on_every_row_is_still_typed_from_the_schema() -> None:
    """Inference would type it `Null`, and a reader would meet a column with no dtype."""
    rows = [_metric_row(group_value=str(i), value=None) for i in range(3)]
    frame = writer.finalize(rows, "fairness_group_metrics")
    assert frame.schema["value"] == pl.Float64


def test_rows_are_sorted_into_the_declared_total_order() -> None:
    rows = [_metric_row(group_value=v) for v in ("9", "1", "5")]
    frame = writer.finalize(rows, "fairness_group_metrics")
    assert frame["group_value"].to_list() == ["1", "5", "9"]


def test_the_same_rows_in_a_different_order_produce_an_identical_frame() -> None:
    """Which is what makes two runs byte-comparable."""
    rows = [_metric_row(group_value=v) for v in ("9", "1", "5")]
    assert writer.finalize(rows, "fairness_group_metrics").equals(
        writer.finalize(list(reversed(rows)), "fairness_group_metrics")
    )


def test_a_row_missing_a_contract_column_is_rejected() -> None:
    row = _metric_row()
    del row["n_positive"]
    with pytest.raises(ValueError, match="missing columns: n_positive"):
        writer.finalize([row], "fairness_group_metrics")


def test_a_row_carrying_an_undeclared_column_is_rejected() -> None:
    """A column nobody declared is a column no contract describes."""
    with pytest.raises(ValueError, match="unknown columns: fairness_score"):
        writer.finalize([_metric_row(fairness_score=0.9)], "fairness_group_metrics")


def test_no_rows_produces_the_typed_empty_frame() -> None:
    assert writer.finalize([], "fairness_group_metrics").equals(
        writer.empty("fairness_group_metrics")
    )


# --- 4. round trip ---------------------------------------------------------------


def test_a_written_table_reads_back_identically(tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    rows = [_metric_row(group_value=str(i), value=None if i else 0.6) for i in range(4)]
    frame = writer.finalize(rows, "fairness_group_metrics")
    path = writer.write_table(frame, tmp_path / "fairness_group_metrics_20260825T000000Z.parquet")
    assert pl.read_parquet(path).equals(frame)


def test_the_schema_record_names_every_column_and_its_dtype() -> None:
    frame = writer.finalize([_metric_row()], "fairness_group_metrics")
    schema = writer.schema_of(frame)
    assert schema["value"] == "Float64"
    assert set(schema) == set(writer.SCHEMAS["fairness_group_metrics"])


# --- 5. the bootstrap table carries two rows per group ---------------------------


def test_the_bootstrap_sort_key_includes_the_scheme() -> None:
    """Both resampling schemes are run for every group, so without it the key is not unique."""
    assert "scheme" in writer.SORT_KEYS["fairness_bootstrap"]
