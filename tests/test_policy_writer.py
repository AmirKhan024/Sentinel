"""The artifact contract: schemas, column order, total sort keys, typed empties.

Column order is part of the data contract, and a sort key that is not a total order is the
defect that quietly breaks byte-comparison between two runs. Component 12 shipped that once.
"""

from __future__ import annotations

import polars as pl
import pytest

from sentinel.policy import writer


def test_every_table_has_a_schema_a_layer_and_a_sort_key() -> None:
    """Three registries that must agree, and each of which is edited separately."""
    assert set(writer.SCHEMAS) == set(writer.LAYERS)
    assert set(writer.SCHEMAS) == set(writer.SORT_KEYS)


def test_every_sort_key_names_only_columns_that_exist() -> None:
    for table, keys in writer.SORT_KEYS.items():
        missing = [key for key in keys if key not in writer.SCHEMAS[table]]
        assert not missing, f"{table}: sort key names {missing}"


def test_the_manifest_is_keyed_to_the_recommendation_table() -> None:
    """The component's answer is the queue, so that is what the provenance record describes."""
    assert writer.DATASET_SLUG in writer.SCHEMAS
    assert writer.DATASET_SLUG == "inspection_recommendations"


def test_every_table_lands_in_the_one_new_layer() -> None:
    """A recommendation is not a prediction, not a metric and not a group measurement."""
    assert set(writer.LAYERS.values()) == {"policy"}


def test_the_recommendation_grain_is_a_total_order() -> None:
    """Policy, model, fold and capacity name the cell; the id names the row inside it.

    Anything less would leave ties resolved by append order, which is not a contract.
    """
    assert writer.SORT_KEYS["inspection_recommendations"] == [
        "policy_id",
        "model_name",
        "fold_set",
        "fold_id",
        "k_name",
        "target_inspection_id",
    ]


def test_no_decision_table_carries_an_outcome_column() -> None:
    """The structural statement that the policy did not read the answer.

    A future edit that wanted to use the label would have to add a column and change the
    contract to do it, rather than reaching for one that was already there.
    """
    for table, schema in writer.SCHEMAS.items():
        assert "target" not in schema, table
        assert "target_status" not in schema, table


def test_an_empty_table_carries_the_full_schema() -> None:
    """A run with no overrides must leave a reader the columns, not a missing file."""
    frame = writer.empty("policy_override_log")
    assert frame.height == 0
    assert frame.columns == list(writer.SCHEMAS["policy_override_log"])


def test_an_unknown_table_is_refused_by_name() -> None:
    with pytest.raises(KeyError, match="policy_imaginary"):
        writer.empty("policy_imaginary")


def test_finalize_sorts_into_the_declared_order() -> None:
    rows = [
        {
            "policy_id": "b",
            "reserve_mechanism": "none",
            "reserve_share": 0.0,
            "is_baseline": False,
            "rationale": "r",
            "policy_definition_version": "v1",
        },
        {
            "policy_id": "a",
            "reserve_mechanism": "floor",
            "reserve_share": 0.1,
            "is_baseline": True,
            "rationale": "r",
            "policy_definition_version": "v1",
        },
    ]
    frame = writer.finalize(rows, "policy_configurations")
    assert frame["policy_id"].to_list() == ["a", "b"]


def test_finalize_preserves_the_declared_column_order() -> None:
    """Column order is the contract; a dict-ordered frame would silently change it."""
    rows = [
        {
            "policy_definition_version": "v1",
            "rationale": "r",
            "is_baseline": True,
            "reserve_share": 0.0,
            "reserve_mechanism": "none",
            "policy_id": "a",
        }
    ]
    frame = writer.finalize(rows, "policy_configurations")
    assert frame.columns == list(writer.SCHEMAS["policy_configurations"])


def test_a_missing_column_is_named_rather_than_filled_with_a_null() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        writer.finalize([{"policy_id": "a"}], "policy_configurations")


def test_an_unknown_column_is_named_rather_than_dropped() -> None:
    """A dropped column is a contract change nobody was told about."""
    rows = [
        {
            "policy_id": "a",
            "reserve_mechanism": "none",
            "reserve_share": 0.0,
            "is_baseline": True,
            "rationale": "r",
            "policy_definition_version": "v1",
            "surprise": 1,
        }
    ]
    with pytest.raises(ValueError, match="unknown columns"):
        writer.finalize(rows, "policy_configurations")


def test_an_all_null_metric_column_keeps_its_declared_dtype() -> None:
    """The reason the schema is passed to the constructor rather than cast afterwards.

    This component emits nulls in quantity -- a capture rate for a window with no positives, a
    rank for a row nobody selected. Inference would type such a column ``Null`` and the file
    would stop matching its contract.
    """
    rows = [
        {
            "policy_id": "a",
            "model_name": "m",
            "fold_set": "quarterly",
            "k_name": "k_1_day",
            "positives_selected": None,
            "eligible_selected": None,
            "is_dominated": False,
            "dominated_by": "",
            "policy_definition_version": "v1",
        }
    ]
    frame = writer.finalize(rows, "policy_frontier")
    assert frame.schema["positives_selected"] == pl.Float64


def test_a_written_table_round_trips_byte_stably(tmp_path: object) -> None:
    """Two writes of the same frame produce the same bytes, which is what repro rests on."""
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    frame = writer.finalize(
        [
            {
                "policy_id": "a",
                "reserve_mechanism": "none",
                "reserve_share": 0.0,
                "is_baseline": True,
                "rationale": "r",
                "policy_definition_version": "v1",
            }
        ],
        "policy_configurations",
    )
    first = writer.write_table(frame, tmp_path / "one.parquet")
    second = writer.write_table(frame, tmp_path / "two.parquet")
    assert first.read_bytes() == second.read_bytes()


def test_schema_of_reports_every_column() -> None:
    frame = writer.empty("policy_comparison")
    assert set(writer.schema_of(frame)) == set(writer.SCHEMAS["policy_comparison"])
