"""The prediction contract: what a future model must hand over, and what is refused.

This is the seam between Component 5 and every modelling component after it, so
these tests are really a specification for Component 6. Two of the rules are
arithmetic; the rest are about honesty.

The rule worth understanding is **exact coverage**. A model that quietly drops
the establishments it finds hard would post a better precision@k for a reason
that has nothing to do with being better at the task, and nothing downstream
would notice. So the score set must cover the fold's test window exactly, and a
missing score is a rejection rather than a zero.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from sentinel.evaluation.contract import (
    PREDICTION_COLUMNS,
    PREDICTION_METADATA_COLUMNS,
    PredictionContractError,
    prediction_frame,
    read_predictions,
    validate_predictions,
)
from sentinel.evaluation.models import FoldSpec, PredictionSet

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

IDS = ["1001", "1002", "1003"]


def _predictions(frame: pl.DataFrame, **overrides: object) -> PredictionSet:
    kwargs: dict[str, object] = {
        "model_name": "example",
        "model_version": "v1",
        "fold_id": FOLD.fold_id,
        "frame": frame,
    }
    kwargs.update(overrides)
    return PredictionSet(**kwargs)  # type: ignore[arg-type]


# --- 1. the happy path ------------------------------------------------------


def test_a_well_formed_prediction_set_is_accepted() -> None:
    validate_predictions(_predictions(prediction_frame(IDS, [0.9, 0.5, 0.1])), FOLD, IDS)


def test_prediction_frame_produces_exactly_the_contract_columns() -> None:
    frame = prediction_frame(IDS, [0.9, 0.5, 0.1])
    assert tuple(frame.columns) == PREDICTION_COLUMNS
    assert frame["score"].dtype == pl.Float64
    assert frame["target_inspection_id"].dtype == pl.Utf8


def test_the_order_of_the_rows_does_not_matter() -> None:
    """Scores are joined by id, so a model may emit them in any order."""
    shuffled = prediction_frame(list(reversed(IDS)), [0.1, 0.5, 0.9])
    validate_predictions(_predictions(shuffled), FOLD, IDS)


# --- 2. shape and type ------------------------------------------------------


def test_a_missing_required_column_is_rejected() -> None:
    frame = pl.DataFrame({"target_inspection_id": IDS})
    with pytest.raises(PredictionContractError, match="missing required column"):
        validate_predictions(_predictions(frame), FOLD, IDS)


def test_an_extra_column_is_rejected() -> None:
    """An id and a score, nothing else -- so a label cannot ride along."""
    frame = prediction_frame(IDS, [0.9, 0.5, 0.1]).with_columns(pl.lit(1).alias("target"))
    with pytest.raises(PredictionContractError, match="unexpected column"):
        validate_predictions(_predictions(frame), FOLD, IDS)


def test_a_null_score_is_rejected_rather_than_imputed() -> None:
    """Filling the gap would silently rank that establishment last."""
    frame = pl.DataFrame(
        {"target_inspection_id": IDS, "score": [0.9, None, 0.1]},
        schema={"target_inspection_id": pl.Utf8, "score": pl.Float64},
    )
    with pytest.raises(PredictionContractError, match="never imputes"):
        validate_predictions(_predictions(frame), FOLD, IDS)


def test_a_null_identifier_is_rejected() -> None:
    frame = pl.DataFrame(
        {"target_inspection_id": ["1001", None, "1003"], "score": [0.9, 0.5, 0.1]},
        schema={"target_inspection_id": pl.Utf8, "score": pl.Float64},
    )
    with pytest.raises(PredictionContractError, match="null target_inspection_id"):
        validate_predictions(_predictions(frame), FOLD, IDS)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_score_is_rejected(bad: float) -> None:
    frame = prediction_frame(IDS, [0.9, bad, 0.1])
    with pytest.raises(PredictionContractError, match="NaN or infinite"):
        validate_predictions(_predictions(frame), FOLD, IDS)


def test_duplicate_identifiers_are_rejected() -> None:
    frame = prediction_frame(["1001", "1001", "1003"], [0.9, 0.5, 0.1])
    with pytest.raises(PredictionContractError, match="duplicate"):
        validate_predictions(_predictions(frame), FOLD, ["1001", "1003"])


def test_mismatched_id_and_score_lengths_are_rejected() -> None:
    with pytest.raises(PredictionContractError, match="differ in length"):
        prediction_frame(IDS, [0.9, 0.5])


# --- 3. exact coverage ------------------------------------------------------


def test_an_unscored_test_row_is_rejected() -> None:
    frame = prediction_frame(["1001", "1002"], [0.9, 0.5])
    with pytest.raises(PredictionContractError, match="1 test row\\(s\\) unscored"):
        validate_predictions(_predictions(frame), FOLD, IDS)


def test_a_score_for_a_row_outside_the_test_window_is_rejected() -> None:
    frame = prediction_frame([*IDS, "9999"], [0.9, 0.5, 0.1, 0.2])
    with pytest.raises(PredictionContractError, match="not in the test window"):
        validate_predictions(_predictions(frame), FOLD, IDS)


# --- 4. provenance ----------------------------------------------------------


def test_a_prediction_set_for_a_different_fold_is_rejected() -> None:
    frame = prediction_frame(IDS, [0.9, 0.5, 0.1])
    with pytest.raises(PredictionContractError, match="declare fold"):
        validate_predictions(_predictions(frame, fold_id="quarterly-2023Q1"), FOLD, IDS)


def test_a_training_horizon_past_the_calibration_end_is_rejected() -> None:
    frame = prediction_frame(IDS, [0.9, 0.5, 0.1])
    with pytest.raises(PredictionContractError, match="after fold"):
        validate_predictions(_predictions(frame, trained_through=date(2022, 4, 2)), FOLD, IDS)


def test_a_training_horizon_inside_the_fold_is_accepted() -> None:
    frame = prediction_frame(IDS, [0.9, 0.5, 0.1])
    validate_predictions(_predictions(frame, trained_through=FOLD.calibration_end), FOLD, IDS)


def test_an_undeclared_training_horizon_is_permitted() -> None:
    """Optional, because a rule-based baseline has no training horizon at all."""
    frame = prediction_frame(IDS, [0.9, 0.5, 0.1])
    validate_predictions(_predictions(frame, trained_through=None), FOLD, IDS)


# --- 5. persisted artifacts -------------------------------------------------


def _write(path: Path, **overrides: object) -> Path:
    data: dict[str, object] = {
        "target_inspection_id": IDS,
        "score": [0.9, 0.5, 0.1],
        "model_name": ["m"] * 3,
        "model_version": ["v1"] * 3,
        "fold_id": [FOLD.fold_id] * 3,
    }
    data.update(overrides)
    pl.DataFrame(data).write_parquet(path)
    return path


def test_a_persisted_artifact_round_trips_into_prediction_sets(tmp_path: Path) -> None:
    path = _write(tmp_path / "predictions.parquet")
    (predictions,) = read_predictions(path)
    assert predictions.model_name == "m"
    assert predictions.fold_id == FOLD.fold_id
    assert tuple(predictions.frame.columns) == PREDICTION_COLUMNS
    validate_predictions(predictions, FOLD, IDS)


def test_a_persisted_artifact_missing_its_metadata_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bare.parquet"
    prediction_frame(IDS, [0.9, 0.5, 0.1]).write_parquet(path)
    with pytest.raises(PredictionContractError, match="must carry"):
        read_predictions(path)


def test_one_file_may_hold_several_models_and_is_split_by_producer(tmp_path: Path) -> None:
    path = tmp_path / "many.parquet"
    pl.DataFrame(
        {
            "target_inspection_id": IDS * 2,
            "score": [0.9, 0.5, 0.1, 0.2, 0.3, 0.4],
            "model_name": ["a"] * 3 + ["b"] * 3,
            "model_version": ["v1"] * 6,
            "fold_id": [FOLD.fold_id] * 6,
        }
    ).write_parquet(path)
    sets = read_predictions(path)
    assert [s.model_name for s in sets] == ["a", "b"]
    for predictions in sets:
        validate_predictions(predictions, FOLD, IDS)


def test_reading_can_be_filtered_to_one_fold(tmp_path: Path) -> None:
    path = tmp_path / "folds.parquet"
    pl.DataFrame(
        {
            "target_inspection_id": IDS * 2,
            "score": [0.9, 0.5, 0.1, 0.2, 0.3, 0.4],
            "model_name": ["a"] * 6,
            "model_version": ["v1"] * 6,
            "fold_id": [FOLD.fold_id] * 3 + ["quarterly-2022Q3"] * 3,
        }
    ).write_parquet(path)
    sets = read_predictions(path, fold_id=FOLD.fold_id)
    assert len(sets) == 1
    assert sets[0].fold_id == FOLD.fold_id


def test_a_missing_artifact_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        read_predictions(tmp_path / "absent.parquet")


def test_the_declared_metadata_columns_are_the_ones_the_reader_requires() -> None:
    assert PREDICTION_METADATA_COLUMNS == ("model_name", "model_version", "fold_id")
