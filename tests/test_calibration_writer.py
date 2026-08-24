"""Output schemas, artifact layering, and the seam back to Component 5.

The most important test here is ``test_the_calibrated_artifact_is_readable_by_component_5``.
The whole reason the calibrated predictions live under ``predictions/`` with the contract's
column set is that ``sentinel evaluate --predictions <file>`` must work with **no change to
Component 5** -- so the headline PR-AUC, ROC-AUC and NDE for a calibrated model come from
the same evaluator that produced every earlier number.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from sentinel.calibration import writer
from sentinel.calibration.definitions import CALIBRATION_DEFINITION_VERSION
from sentinel.evaluation.contract import (
    PREDICTION_COLUMNS,
    PREDICTION_METADATA_COLUMNS,
    read_predictions,
    validate_predictions,
)
from sentinel.evaluation.models import FoldSpec

FOLD = FoldSpec(
    fold_set="quarterly",
    fold_id="quarterly-2022Q2",
    train_start=date(2018, 7, 1),
    train_end=date(2021, 12, 31),
    calibration_start=date(2022, 1, 1),
    calibration_end=date(2022, 3, 31),
    test_start=date(2022, 4, 1),
    test_end=date(2022, 6, 30),
)


def _prediction_rows(n: int = 40) -> list[dict[str, object]]:
    return [
        {
            "target_inspection_id": f"row-{i:04d}",
            "score": (i + 0.5) / n,
            "model_name": "logistic_regression_platt",
            "model_version": "v1",
            "fold_set": FOLD.fold_set,
            "fold_id": FOLD.fold_id,
            "trained_through": FOLD.calibration_end,
            "is_probability": True,
            "base_model_name": "logistic_regression",
            "base_model_version": "v1",
            "base_score": (i + 0.5) / n * 0.9,
            "base_model_trained_through": FOLD.train_end,
            "calibrator_fitted_through": FOLD.calibration_end,
            "calibrated_prediction_available_from": FOLD.test_start,
            "method": "platt",
            "is_experimental": False,
            "calibration_definition_version": CALIBRATION_DEFINITION_VERSION,
        }
        for i in range(n)
    ]


# --- schemas -----------------------------------------------------------------


def test_every_table_declares_a_schema_a_sort_key_and_a_layer() -> None:
    assert set(writer.SCHEMAS) == set(writer.SORT_KEYS) == set(writer.LAYERS)
    assert len(writer.SCHEMAS) == 9


def test_the_three_layers_are_the_ones_the_adrs_named() -> None:
    """ADR 0014 named predictions/, ADR 0018 named tuning/, ADR 0024 added calibration/."""
    assert writer.LAYERS["calibrated_predictions"] == "predictions"
    assert writer.LAYERS["calibrator_selection"] == "tuning"
    assert set(writer.LAYERS.values()) == {"predictions", "tuning", "calibration"}
    diagnostics = [t for t, layer in writer.LAYERS.items() if layer == "calibration"]
    assert len(diagnostics) == 7


def test_only_the_selection_log_carries_a_duration() -> None:
    """``modeling/writer.py``'s rule, with ADR 0018's narrow exception and no other."""
    for table, schema in writer.SCHEMAS.items():
        has_seconds = "seconds" in schema
        assert has_seconds == (table == "calibrator_selection"), table
        assert not any("timestamp" in column for column in schema), table


def test_every_table_carries_the_definition_version() -> None:
    for table, schema in writer.SCHEMAS.items():
        assert "calibration_definition_version" in schema, table


def test_sort_keys_are_real_columns_of_their_table() -> None:
    for table, keys in writer.SORT_KEYS.items():
        for key in keys:
            assert key in writer.SCHEMAS[table], f"{table}.{key}"


def test_empty_returns_a_correctly_typed_zero_row_frame() -> None:
    for table in writer.SCHEMAS:
        frame = writer.empty(table)
        assert frame.height == 0
        assert list(frame.columns) == list(writer.SCHEMAS[table])


def test_finalize_sorts_deterministically() -> None:
    rows = _prediction_rows()
    forward = writer.finalize(list(rows), "calibrated_predictions")
    backward = writer.finalize(list(reversed(rows)), "calibrated_predictions")
    assert forward.to_dicts() == backward.to_dicts()


def test_finalize_rejects_a_missing_column() -> None:
    rows = _prediction_rows()
    del rows[0]["method"]
    with pytest.raises(ValueError, match="missing columns"):
        writer.finalize(rows, "calibrated_predictions")


def test_finalize_rejects_an_unknown_column() -> None:
    """A column nobody declared would silently widen the contract."""
    rows = _prediction_rows()
    rows[0]["surprise"] = 1.0
    with pytest.raises(ValueError, match="unknown columns"):
        writer.finalize(rows, "calibrated_predictions")


def test_an_unknown_table_is_rejected() -> None:
    with pytest.raises(KeyError):
        writer.finalize([], "calibration_vibes")


# --- the seam back to Component 5 ---------------------------------------------


def test_the_calibrated_schema_carries_the_contract_columns() -> None:
    schema = writer.CALIBRATED_PREDICTIONS_SCHEMA
    for column in (*PREDICTION_COLUMNS, *PREDICTION_METADATA_COLUMNS):
        assert column in schema, column
    assert "trained_through" in schema
    assert "is_probability" in schema


def test_the_calibrated_artifact_is_readable_by_component_5(tmp_path: object) -> None:
    """``sentinel evaluate --predictions`` must work with no change to Component 5.

    ``read_predictions`` selects only the contract's two columns into a ``PredictionSet``,
    so Component 9's extra columns are read-safe: ``validate_predictions`` never sees them
    and its "unexpected column" rejection cannot fire.
    """
    from pathlib import Path

    frame = writer.finalize(_prediction_rows(), "calibrated_predictions")
    path = Path(str(tmp_path)) / "calibrated_predictions_20260824T000000Z.parquet"
    writer.write_table(frame, path)

    sets = read_predictions(path)
    assert len(sets) == 1
    predictions = sets[0]
    assert predictions.model_name == "logistic_regression_platt"
    assert predictions.is_probability is True
    assert predictions.trained_through == FOLD.calibration_end
    assert list(predictions.frame.columns) == list(PREDICTION_COLUMNS)

    expected = [f"row-{i:04d}" for i in range(40)]
    validate_predictions(predictions, FOLD, expected)


def test_a_calibrated_row_can_never_be_confused_with_its_base_model() -> None:
    """``model_name`` is ``<base>_<method>``, so both can sit in one results table."""
    frame = writer.finalize(_prediction_rows(), "calibrated_predictions")
    names = set(frame["model_name"].to_list())
    bases = set(frame["base_model_name"].to_list())
    assert names == {"logistic_regression_platt"}
    assert bases == {"logistic_regression"}
    assert not names & bases


def test_the_base_score_is_carried_so_the_correction_is_visible() -> None:
    """A consumer never has to join back to Component 6/7/8 to see what changed."""
    frame = writer.finalize(_prediction_rows(), "calibrated_predictions")
    assert "base_score" in frame.columns
    assert frame["base_score"].null_count() == 0
    assert frame["score"].to_list() != frame["base_score"].to_list()


def test_the_three_horizons_are_all_present_and_ordered() -> None:
    """One of them alone would be misleading; ``trained_through`` is the max of the pair."""
    frame = writer.finalize(_prediction_rows(), "calibrated_predictions")
    row = frame.to_dicts()[0]
    assert row["base_model_trained_through"] == FOLD.train_end
    assert row["calibrator_fitted_through"] == FOLD.calibration_end
    assert row["calibrated_prediction_available_from"] == FOLD.test_start
    assert row["trained_through"] == max(
        row["base_model_trained_through"], row["calibrator_fitted_through"]
    )


def test_no_output_table_carries_a_component_4_feature_column() -> None:
    """Nothing in these layers may be joinable onto a feature table (ADR 0024)."""
    from sentinel.features.definitions import FEATURE_SPECS

    feature_names = {spec.name for spec in FEATURE_SPECS}
    for table, schema in writer.SCHEMAS.items():
        overlap = feature_names & set(schema)
        assert not overlap, f"{table} carries feature column(s) {overlap}"


def test_the_base_scores_table_records_whether_it_matched_the_committed_artifact() -> None:
    """The bit-identity result is persisted per row, not only summarised in the manifest."""
    schema = writer.BASE_SCORES_SCHEMA
    assert schema["reproduces_committed_artifact"] == pl.Boolean()
    assert "native_margin" in schema
    assert "base_logit" in schema
    assert "inner_portion" in schema
