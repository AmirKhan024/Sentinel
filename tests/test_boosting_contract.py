"""The seam: Component 7's artifact must be something Component 5 accepts unchanged.

Component 5 owns evaluation and Component 7 owns prediction, and the only thing crossing
between them is a Parquet file. So the tests here exercise the real contract functions --
``read_predictions`` and ``validate_predictions`` -- against a real artifact, rather than
asserting that the schema dictionary looks right.

They also drive the contract's rejections. A rejection that has never been observed is
indistinguishable from one that cannot happen, and Component 5 shipped exactly that
defect once (``scores_respect_the_decision_point``, declared and unreachable, fixed in
ADR 0014). So each way a boosted artifact could be wrong gets a test that builds it wrong
and confirms the evaluator refuses it.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from sentinel.boosting import build, writer
from sentinel.config import Settings
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.contract import (
    PREDICTION_COLUMNS,
    PredictionContractError,
    PredictionHorizonError,
    prediction_frame,
    read_predictions,
    validate_predictions,
)
from sentinel.evaluation.models import PredictionSet
from tests.conftest import spanning_model_features


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, pl.DataFrame]:
    """A real boosted prediction artifact on disk, produced by the real command."""
    tmp = tmp_path_factory.mktemp("boosting_contract")
    features = tmp / "as_of_features_20260101T000000Z.parquet"
    frame = spanning_model_features(days=1900)
    frame.write_parquet(features)

    result = build.train_boosting(
        Settings(data_dir=tmp),
        features_path=features,
        output_dir=tmp,
        models=["xgboost", "lightgbm"],
    )
    assert result.predictions_path is not None
    return result.predictions_path, frame.with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )


# --- 1. the evaluator reads it -------------------------------------------------


def test_component_5_reads_the_artifact(artifact: tuple[Path, pl.DataFrame]) -> None:
    path, _ = artifact
    sets = read_predictions(path)
    assert sets
    assert {p.model_name for p in sets} == {"xgboost", "lightgbm"}


def test_every_prediction_set_declares_a_horizon(artifact: tuple[Path, pl.DataFrame]) -> None:
    """A null ``trained_through`` silently skips the evaluator's horizon check."""
    path, _ = artifact
    for pset in read_predictions(path):
        assert pset.trained_through is not None


def test_every_prediction_set_declares_a_probability(
    artifact: tuple[Path, pl.DataFrame],
) -> None:
    """A null would downgrade the model to ranking-only and suppress ECE and MCE."""
    path, _ = artifact
    assert all(p.is_probability for p in read_predictions(path))


def test_the_frame_carries_exactly_the_contracts_two_columns(
    artifact: tuple[Path, pl.DataFrame],
) -> None:
    path, _ = artifact
    for pset in read_predictions(path):
        assert tuple(pset.frame.columns) == PREDICTION_COLUMNS


def test_no_label_rides_along_in_the_artifact(artifact: tuple[Path, pl.DataFrame]) -> None:
    path, _ = artifact
    columns = set(pl.read_parquet(path).columns)
    assert "target" not in columns
    assert "target_status" not in columns


def test_the_artifact_passes_validation_for_every_fold(
    artifact: tuple[Path, pl.DataFrame],
) -> None:
    """The whole seam, exercised: real folds, real windows, the real validator."""
    path, frame = artifact
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    assert start is not None and end is not None
    folds = [
        *folds_module.quarterly_folds(data_start=start, data_end=end),
        *folds_module.covid_shift_fold(data_end=end),
    ]
    checked = 0
    for fold in folds:
        window = folds_module.window_frame(frame, fold)
        if window.height == 0:
            continue
        for pset in read_predictions(path, fold_id=fold.fold_id):
            validate_predictions(pset, fold, window["target_inspection_id"].to_list())
            checked += 1
    assert checked >= 2, "too few (model, fold) pairs validated; the loop would be weak"


def test_the_declared_horizon_is_the_training_end_not_the_calibration_end(
    artifact: tuple[Path, pl.DataFrame],
) -> None:
    """Strictly inside the contract's ceiling, which is what ADR 0014 requires."""
    path, frame = artifact
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    assert start is not None and end is not None
    folds = {f.fold_id: f for f in folds_module.quarterly_folds(data_start=start, data_end=end)}
    for pset in read_predictions(path):
        fold = folds.get(pset.fold_id)
        if fold is None:
            continue
        assert pset.trained_through == fold.train_end
        assert pset.trained_through < fold.calibration_end


# --- 2. the rejections, driven ------------------------------------------------


def _fold() -> object:
    from sentinel.evaluation.models import FoldSpec

    return FoldSpec(
        fold_set="quarterly",
        fold_id="quarterly-2022Q2",
        train_start=date(2018, 7, 1),
        train_end=date(2021, 12, 31),
        calibration_start=date(2022, 1, 1),
        calibration_end=date(2022, 3, 31),
        test_start=date(2022, 4, 1),
        test_end=date(2022, 6, 30),
    )


def _pset(**overrides: object) -> PredictionSet:
    defaults: dict[str, object] = {
        "model_name": "xgboost",
        "model_version": "v1",
        "fold_id": "quarterly-2022Q2",
        "frame": prediction_frame(["a", "b"], [0.7, 0.2]),
        "is_probability": True,
        "trained_through": date(2021, 12, 31),
    }
    defaults.update(overrides)
    return PredictionSet(**defaults)  # type: ignore[arg-type]


def test_a_horizon_past_the_calibration_end_is_rejected() -> None:
    """The check that stops a model claiming it learned from the calibration window."""
    fold = _fold()
    late = _pset(trained_through=fold.calibration_end + timedelta(days=1))  # type: ignore[attr-defined]
    with pytest.raises(PredictionHorizonError):
        validate_predictions(late, fold, ["a", "b"])  # type: ignore[arg-type]


def test_the_horizon_this_component_declares_is_accepted() -> None:
    validate_predictions(_pset(), _fold(), ["a", "b"])  # type: ignore[arg-type]


def test_a_missing_row_is_rejected() -> None:
    with pytest.raises(PredictionContractError):
        validate_predictions(_pset(), _fold(), ["a", "b", "c"])  # type: ignore[arg-type]


def test_an_extra_row_is_rejected() -> None:
    with pytest.raises(PredictionContractError):
        validate_predictions(_pset(), _fold(), ["a"])  # type: ignore[arg-type]


def test_a_duplicate_row_is_rejected() -> None:
    duplicated = _pset(frame=prediction_frame(["a", "a"], [0.7, 0.2]))
    with pytest.raises(PredictionContractError):
        validate_predictions(duplicated, _fold(), ["a", "b"])  # type: ignore[arg-type]


def test_a_mismatched_fold_id_is_rejected() -> None:
    with pytest.raises(PredictionContractError):
        validate_predictions(_pset(fold_id="quarterly-2023Q1"), _fold(), ["a", "b"])  # type: ignore[arg-type]


def test_an_extra_column_is_rejected() -> None:
    """A label could not ride along even if someone tried."""
    wide = prediction_frame(["a", "b"], [0.7, 0.2]).with_columns(pl.lit(1).alias("target"))
    with pytest.raises(PredictionContractError):
        validate_predictions(_pset(frame=wide), _fold(), ["a", "b"])  # type: ignore[arg-type]


# --- 3. the artifact's own schema ---------------------------------------------


def test_the_prediction_schema_carries_the_contracts_columns() -> None:
    for column in PREDICTION_COLUMNS:
        assert column in writer.PREDICTIONS_SCHEMA


def test_the_prediction_schema_types_the_horizon_as_a_date() -> None:
    assert writer.PREDICTIONS_SCHEMA["trained_through"] == pl.Date()
    assert writer.PREDICTIONS_SCHEMA["is_probability"] == pl.Boolean()


def test_the_boosted_slug_differs_from_component_6s() -> None:
    """C6's benchmark must stay visible; a shared slug would overwrite it."""
    from sentinel.modeling.build import DATASET_SLUG as BASELINE_SLUG

    assert build.PREDICTIONS_SLUG == "boosted_predictions"
    assert build.PREDICTIONS_SLUG != BASELINE_SLUG


def test_the_boosted_and_baseline_prediction_schemas_are_shaped_alike() -> None:
    """Different shapes would mean the evaluator treats the two components differently."""
    from sentinel.modeling.writer import PREDICTIONS_SCHEMA as BASELINE_SCHEMA

    shared = set(BASELINE_SCHEMA) & set(writer.PREDICTIONS_SCHEMA)
    assert shared >= set(PREDICTION_COLUMNS) | {
        "model_name",
        "model_version",
        "fold_set",
        "fold_id",
        "trained_through",
        "is_probability",
    }
    for column in shared:
        assert BASELINE_SCHEMA[column] == writer.PREDICTIONS_SCHEMA[column]
