"""The artifact's schema, its determinism, and the writes it refuses.

Column order is part of the data contract, and row order is the last place non-determinism
could hide after everything upstream has been seeded. Both are asserted here rather than
left to the round-trip test, because a table that is *correct* but ordered differently on
two runs is still an artifact nobody can checksum.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from sentinel.explain import writer
from sentinel.explain.definitions import EXPLAIN_DEFINITION_VERSION


def _value_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "model_name": "xgboost",
        "model_version": "v1",
        "family": "boosted",
        "fold_set": "quarterly",
        "fold_id": "quarterly-2026Q2",
        "target_inspection_id": "100",
        "feature_name": "prior_canvass_count",
        "original_feature_name": "prior_canvass_count",
        "derived_from": "prior_canvass_count",
        "feature_kind": "feature",
        "feature_value": 3.0,
        "transformed_value": 3.0,
        "shap_value": 0.25,
        "output_space": "log_odds",
        "explanation_method": "tree_shap",
        "is_exact": True,
        "base_value": 0.1,
        "prediction_value": 0.35,
        "trained_through": date(2025, 12, 31),
        "explain_definition_version": EXPLAIN_DEFINITION_VERSION,
    }
    row.update(overrides)
    return row


# --- 1. the schemas ----------------------------------------------------------


def test_seven_tables_are_declared_and_every_one_has_a_sort_key() -> None:
    assert set(writer.SCHEMAS) == set(writer.SORT_KEYS)
    assert len(writer.SCHEMAS) == 7
    assert writer.DATASET_SLUG in writer.SCHEMAS


def test_every_table_lives_in_the_explanations_layer() -> None:
    """Unlike Component 9 there is no second home: nothing here is a prediction."""
    assert set(writer.LAYERS.values()) == {"explanations"}


def test_every_sort_key_names_columns_the_schema_actually_has() -> None:
    for table, keys in writer.SORT_KEYS.items():
        for key in keys:
            assert key in writer.SCHEMAS[table], f"{table}.{key}"


def test_the_values_sort_key_determines_a_row_uniquely() -> None:
    """A partial key leaves ties whose order polars is free to choose."""
    assert writer.SORT_KEYS["explanation_values"] == [
        "model_name",
        "fold_set",
        "fold_id",
        "target_inspection_id",
        "feature_name",
    ]


def test_the_values_table_carries_both_names_and_both_values() -> None:
    """Section 7 of the brief: the transformed representation and the original, together."""
    schema = writer.SCHEMAS["explanation_values"]
    for column in (
        "feature_name",
        "original_feature_name",
        "derived_from",
        "feature_value",
        "transformed_value",
    ):
        assert column in schema


def test_the_cases_table_carries_three_distinct_horizons() -> None:
    """A single trained_through would conflate the estimator's horizon with the calibrator's."""
    schema = writer.SCHEMAS["explanation_cases"]
    assert schema["base_model_trained_through"] == pl.Date()
    assert schema["calibrator_fitted_through"] == pl.Date()
    assert schema["prediction_available_from"] == pl.Date()


def test_the_cases_table_connects_the_base_score_to_the_calibrated_probability() -> None:
    schema = writer.SCHEMAS["explanation_cases"]
    assert schema["base_score"] == pl.Float64()
    assert schema["calibrated_probability"] == pl.Float64()
    assert schema["calibration_method"] == pl.Utf8()


def test_the_cases_table_records_the_full_sampling_provenance() -> None:
    schema = writer.SCHEMAS["explanation_cases"]
    for column in (
        "sample_strategy",
        "sample_size",
        "sampling_seed",
        "sampling_population",
        "background_strategy",
        "background_size",
        "background_seed",
        "background_max_date",
        "permutation_rounds",
    ):
        assert column in schema


def test_the_support_table_can_express_an_unsupported_model() -> None:
    """Nullable method and space, and a reason column. Zeros would read as 'used nothing'."""
    schema = writer.SCHEMAS["explanation_support"]
    assert schema["explanation_method"] == pl.Utf8()
    assert schema["unsupported_reason"] == pl.Utf8()


# --- 2. finalize -------------------------------------------------------------


def test_finalize_casts_to_the_contract_schema() -> None:
    frame = writer.finalize([_value_row()], "explanation_values")
    assert list(frame.columns) == list(writer.SCHEMAS["explanation_values"])
    assert frame.schema["trained_through"] == pl.Date()
    assert frame.schema["is_exact"] == pl.Boolean()


def test_finalize_sorts_deterministically_whatever_order_it_is_handed() -> None:
    rows = [
        _value_row(target_inspection_id="200", feature_name="zip_something"),
        _value_row(target_inspection_id="100", feature_name="prior_canvass_count"),
        _value_row(target_inspection_id="100", feature_name="days_since_last_canvass"),
    ]
    forward = writer.finalize(rows, "explanation_values")
    backward = writer.finalize(list(reversed(rows)), "explanation_values")
    assert forward.equals(backward)
    assert forward["target_inspection_id"].to_list() == ["100", "100", "200"]
    assert forward["feature_name"].to_list()[:2] == [
        "days_since_last_canvass",
        "prior_canvass_count",
    ]


def test_finalize_rejects_a_missing_column() -> None:
    row = _value_row()
    del row["shap_value"]
    with pytest.raises(ValueError, match="missing columns: shap_value"):
        writer.finalize([row], "explanation_values")


def test_finalize_rejects_an_unknown_column() -> None:
    """A column nobody declared is a contract change arriving by accident."""
    with pytest.raises(ValueError, match="unknown columns: surprise"):
        writer.finalize([_value_row(surprise=1)], "explanation_values")


def test_finalize_rejects_an_unknown_table() -> None:
    with pytest.raises(KeyError, match="Unknown table"):
        writer.finalize([], "explanation_nonsense")


def test_a_null_feature_value_survives_finalize() -> None:
    """NaN in the source is a real observation for the tree models; it must not become 0.0."""
    frame = writer.finalize([_value_row(feature_value=None)], "explanation_values")
    assert frame["feature_value"].to_list() == [None]
    assert frame.schema["feature_value"] == pl.Float64()


def test_empty_returns_a_correctly_typed_zero_row_frame() -> None:
    for table in writer.SCHEMAS:
        frame = writer.empty(table)
        assert frame.height == 0
        assert list(frame.columns) == list(writer.SCHEMAS[table])


def test_finalize_of_no_rows_is_the_empty_frame() -> None:
    assert writer.finalize([], "explanation_values").equals(writer.empty("explanation_values"))


# --- 3. round trip -----------------------------------------------------------


def test_a_written_table_reads_back_identically(tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    frame = writer.finalize(
        [_value_row(target_inspection_id=str(index)) for index in range(5)],
        "explanation_values",
    )
    path = writer.write_table(frame, tmp_path / "explanation_values_x.parquet")
    assert pl.read_parquet(path).equals(frame)


def test_two_writes_of_the_same_frame_are_byte_identical(tmp_path: object) -> None:
    """The determinism claim, at the file level rather than the frame level."""
    from pathlib import Path

    from sentinel.manifest import compute_sha256

    assert isinstance(tmp_path, Path)
    frame = writer.finalize([_value_row()], "explanation_values")
    first = writer.write_table(frame, tmp_path / "a.parquet")
    second = writer.write_table(frame, tmp_path / "b.parquet")
    assert compute_sha256(first) == compute_sha256(second)


def test_schema_of_reports_dtypes_as_strings_for_the_manifest() -> None:
    frame = writer.finalize([_value_row()], "explanation_values")
    schema = writer.schema_of(frame)
    assert schema["shap_value"] == "Float64"
    assert schema["trained_through"] == "Date"
