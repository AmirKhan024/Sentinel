"""Fitting one booster to one fold: determinism, direction, horizon, and refusals.

The property this file exists for is the one Component 6's profiler quantified and this
component's profiler re-measured: **row order is load-bearing**. Fitting the same rows in
a shuffled order moved a real prediction by 1.12e-01 for XGBoost and 1.23e-01 for
LightGBM -- seven orders of magnitude larger than the 7.049e-09 Component 6 saw in its
coefficients, because a booster subsamples rows and columns in row order rather than
merely summing over them. So the canonical sort is not tidiness here; it is the only
reason two runs agree, and the tests below assert bit-identity rather than closeness.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from sentinel.boosting import predict
from sentinel.boosting.definitions import (
    BOOSTING_REGISTRY,
    Estimator,
    estimator_params,
    n_estimators_of,
    spec_for,
)
from sentinel.boosting.preprocess import matrix_columns
from sentinel.boosting.train import BoostingTrainError, build_estimator, fit_fold, fold_labels
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling.train import training_frame
from tests.conftest import make_model_feature_row, model_feature_scenario, spanning_model_features

PRIMARY = spec_for("xgboost")
SECONDARY = spec_for("lightgbm")
ABLATION = spec_for("xgboost_class_weighted")

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


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return spanning_model_features(days=1900).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )


@pytest.fixture(scope="module")
def training(frame: pl.DataFrame) -> pl.DataFrame:
    return training_frame(frame, FOLD)


# --- 1. construction ---------------------------------------------------------


def test_an_xgboost_estimator_constructs() -> None:
    estimator = build_estimator(PRIMARY, estimator_params(PRIMARY, "quarterly"))
    assert type(estimator).__name__ == "XGBClassifier"


def test_a_lightgbm_estimator_constructs() -> None:
    estimator = build_estimator(SECONDARY, estimator_params(SECONDARY, "quarterly"))
    assert type(estimator).__name__ == "LGBMClassifier"


def test_an_unsupported_estimator_raises() -> None:
    from dataclasses import replace

    broken = replace(PRIMARY, estimator="decision_tree")  # type: ignore[arg-type]
    with pytest.raises(BoostingTrainError, match="unsupported estimator"):
        build_estimator(broken, {})


@pytest.mark.parametrize("spec", BOOSTING_REGISTRY, ids=lambda s: s.name)
def test_every_registered_model_fits(spec: object, training: pl.DataFrame) -> None:
    fitted = fit_fold(spec, training, FOLD)  # type: ignore[arg-type]
    assert fitted.train_rows == training.height
    assert fitted.trees_built >= 1
    assert len(fitted.importances) == len(fitted.matrix_columns)


# --- 2. determinism ----------------------------------------------------------


@pytest.mark.parametrize("spec", [PRIMARY, SECONDARY], ids=lambda s: s.name)
def test_two_fits_of_the_same_window_are_bit_identical(
    spec: object, training: pl.DataFrame, frame: pl.DataFrame
) -> None:
    test = folds_module.window_frame(frame, FOLD)
    first = predict.score_window(fit_fold(spec, training, FOLD), test)  # type: ignore[arg-type]
    second = predict.score_window(fit_fold(spec, training, FOLD), test)  # type: ignore[arg-type]
    assert first == second


@pytest.mark.parametrize("spec", [PRIMARY, SECONDARY], ids=lambda s: s.name)
def test_shuffling_the_training_rows_is_bit_identical(
    spec: object, training: pl.DataFrame, frame: pl.DataFrame
) -> None:
    """``fit_fold`` re-sorts, so a caller cannot break reproducibility by passing rows badly."""
    test = folds_module.window_frame(frame, FOLD)
    shuffled = training.sample(fraction=1.0, shuffle=True, seed=3)
    assert shuffled["target_inspection_id"].to_list() != training["target_inspection_id"].to_list()
    reference = predict.score_window(fit_fold(spec, training, FOLD), test)  # type: ignore[arg-type]
    after = predict.score_window(fit_fold(spec, shuffled, FOLD), test)  # type: ignore[arg-type]
    assert after == reference


def test_the_two_libraries_do_not_produce_the_same_scores(
    training: pl.DataFrame, frame: pl.DataFrame
) -> None:
    """Sanity: if they agreed exactly, one of them is not being fitted."""
    test = folds_module.window_frame(frame, FOLD)
    _, left = predict.score_window(fit_fold(PRIMARY, training, FOLD), test)
    _, right = predict.score_window(fit_fold(SECONDARY, training, FOLD), test)
    assert left != right


# --- 3. the horizon ----------------------------------------------------------


@pytest.mark.parametrize("spec", BOOSTING_REGISTRY, ids=lambda s: s.name)
def test_a_fit_declares_the_training_end_not_the_calibration_end(
    spec: object, training: pl.DataFrame
) -> None:
    fitted = fit_fold(spec, training, FOLD)  # type: ignore[arg-type]
    assert fitted.trained_through == FOLD.train_end
    assert fitted.trained_through != FOLD.calibration_end
    assert fitted.calibration_end_unused == FOLD.calibration_end


def test_a_final_fit_carries_no_early_stopping_parameter(training: pl.DataFrame) -> None:
    """Early stopping would require reading a window later than ``train_end``."""
    fitted = fit_fold(PRIMARY, training, FOLD)
    assert "early_stopping_rounds" not in fitted.params
    assert "eval_set" not in fitted.params


def test_a_fit_runs_the_frozen_number_of_rounds(training: pl.DataFrame) -> None:
    fitted = fit_fold(PRIMARY, training, FOLD)
    assert fitted.n_estimators == n_estimators_of(PRIMARY, FOLD.fold_set)
    assert fitted.trees_built <= fitted.n_estimators


# --- 4. score direction ------------------------------------------------------


@pytest.mark.parametrize("spec", [PRIMARY, SECONDARY], ids=lambda s: s.name)
def test_higher_score_means_higher_predicted_risk(
    spec: object, training: pl.DataFrame, frame: pl.DataFrame
) -> None:
    """The fixture correlates ``prior_canvass_priority_rate`` with the target at 0.6."""
    fitted = fit_fold(spec, training, FOLD)  # type: ignore[arg-type]
    test = folds_module.window_frame(frame, FOLD)
    _, scores = predict.score_window(fitted, test)
    labels = test["target"].to_list()
    positives = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    negatives = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    assert sum(positives) / len(positives) > sum(negatives) / len(negatives)


def test_every_score_is_a_probability(training: pl.DataFrame, frame: pl.DataFrame) -> None:
    fitted = fit_fold(PRIMARY, training, FOLD)
    _, scores = predict.score_window(fitted, folds_module.window_frame(frame, FOLD))
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_every_test_row_gets_exactly_one_score(training: pl.DataFrame, frame: pl.DataFrame) -> None:
    fitted = fit_fold(PRIMARY, training, FOLD)
    test = folds_module.window_frame(frame, FOLD)
    ids, scores = predict.score_window(fitted, test)
    assert len(ids) == len(scores) == test.height
    assert len(set(ids)) == len(ids)
    assert set(ids) == set(test["target_inspection_id"].to_list())


# --- 5. NULL routing ---------------------------------------------------------


def test_a_fit_records_how_many_nan_cells_reached_the_estimator(
    training: pl.DataFrame,
) -> None:
    """The observable that proves no imputation happened. Zero would be the alarm."""
    fitted = fit_fold(PRIMARY, training, FOLD)
    assert fitted.train_nan_cells > 0


# --- 6. the class-weighting ablation ------------------------------------------


def test_the_ablation_weights_from_its_own_training_window(training: pl.DataFrame) -> None:
    fitted = fit_fold(ABLATION, training, FOLD)
    rate = fitted.train_positive_rate
    assert rate is not None
    assert fitted.scale_pos_weight == pytest.approx((1 - rate) / rate, rel=1e-6)
    assert fitted.params["scale_pos_weight"] == fitted.scale_pos_weight


def test_the_unweighted_models_carry_no_weight(training: pl.DataFrame) -> None:
    for spec in (PRIMARY, SECONDARY):
        fitted = fit_fold(spec, training, FOLD)
        assert fitted.scale_pos_weight == 1.0
        assert "scale_pos_weight" not in fitted.params


def test_the_ablation_differs_from_its_donor_only_by_the_weight(
    training: pl.DataFrame, frame: pl.DataFrame
) -> None:
    donor = fit_fold(PRIMARY, training, FOLD)
    ablation = fit_fold(ABLATION, training, FOLD)
    shared = {k: v for k, v in donor.params.items()}
    weighted = {k: v for k, v in ablation.params.items() if k != "scale_pos_weight"}
    assert weighted == shared

    test = folds_module.window_frame(frame, FOLD)
    _, left = predict.score_window(donor, test)
    _, right = predict.score_window(ablation, test)
    assert left != right, "the weighting changed nothing, so the ablation measures nothing"


def test_class_weighting_is_refused_for_lightgbm() -> None:
    """``is_unbalance`` and ``scale_pos_weight`` are not the same thing."""
    from dataclasses import replace

    broken = replace(SECONDARY, name="lightgbm", class_weighted=True)
    rows = spanning_model_features(days=1900).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )
    with pytest.raises(BoostingTrainError, match="XGBoost only"):
        fit_fold(broken, training_frame(rows, FOLD), FOLD)


# --- 7. refusals --------------------------------------------------------------


def test_fitting_an_empty_window_raises() -> None:
    empty = model_feature_scenario([]).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )
    with pytest.raises(BoostingTrainError, match="no training rows"):
        fit_fold(PRIMARY, empty, FOLD)


def test_fitting_a_single_class_window_raises() -> None:
    rows = model_feature_scenario(
        [make_model_feature_row(i, inspection_date="2020-01-15", target=1) for i in range(10)]
    ).with_columns(pl.col("inspection_date").str.to_date().alias("rd"))
    with pytest.raises(BoostingTrainError, match="single class"):
        fit_fold(PRIMARY, rows, FOLD)


def test_a_null_target_raises_rather_than_being_dropped() -> None:
    rows = model_feature_scenario(
        [
            make_model_feature_row(0, inspection_date="2020-01-15", target=None),
            make_model_feature_row(1, inspection_date="2020-01-16", target=0),
        ]
    )
    with pytest.raises(BoostingTrainError, match="null target"):
        fold_labels(rows, "xgboost", FOLD.fold_id)


def test_a_missing_target_column_raises() -> None:
    rows = model_feature_scenario([make_model_feature_row(0)]).drop("target")
    with pytest.raises(BoostingTrainError, match="no target"):
        fold_labels(rows, "xgboost", FOLD.fold_id)


def test_scoring_an_empty_window_raises(training: pl.DataFrame) -> None:
    fitted = fit_fold(PRIMARY, training, FOLD)
    empty = model_feature_scenario([])
    with pytest.raises(predict.BoostingPredictError, match="test window is empty"):
        predict.score_window(fitted, empty)


def test_scoring_a_frame_without_ids_raises(training: pl.DataFrame, frame: pl.DataFrame) -> None:
    fitted = fit_fold(PRIMARY, training, FOLD)
    test = folds_module.window_frame(frame, FOLD).drop("target_inspection_id")
    with pytest.raises(predict.BoostingPredictError, match="no target_inspection_id"):
        predict.score_window(fitted, test)


# --- 8. importances -----------------------------------------------------------


def test_importances_line_up_with_the_matrix_columns(training: pl.DataFrame) -> None:
    fitted = fit_fold(PRIMARY, training, FOLD)
    assert len(fitted.importances) == len(matrix_columns(PRIMARY)) == 30
    assert fitted.matrix_columns == matrix_columns(PRIMARY)


def test_the_estimator_is_recorded_on_the_fit(training: pl.DataFrame) -> None:
    assert fit_fold(PRIMARY, training, FOLD).spec.estimator is Estimator.XGBOOST
    assert fit_fold(SECONDARY, training, FOLD).spec.estimator is Estimator.LIGHTGBM


# --- 9. params_fold_set override (Component 18) --------------------------------
#
# Component 18's operational FoldSpec carries fold_set="operational", which
# TUNED_PARAMS has no study for. ``params_fold_set`` lets a caller supply a real fold
# set's tuned hyperparameters without relabelling the fold itself. Every evaluation
# call site in this repository omits the parameter, so its default must be provably
# inert.


def test_omitting_params_fold_set_is_bit_identical_to_before_the_parameter_existed(
    training: pl.DataFrame,
) -> None:
    default = fit_fold(PRIMARY, training, FOLD)
    explicit = fit_fold(PRIMARY, training, FOLD, params_fold_set=FOLD.fold_set)
    assert default.params == explicit.params
    assert default.n_estimators == explicit.n_estimators
    assert default.trees_built == explicit.trees_built
    assert default.importances == explicit.importances


def test_params_fold_set_overrides_which_tuned_study_is_read(training: pl.DataFrame) -> None:
    from dataclasses import replace

    operational_fold = replace(FOLD, fold_set="operational", fold_id="operational-2027-01-01")
    with pytest.raises(KeyError, match="Known fold sets"):
        # No override: an unstudied fold_set is refused, exactly as before this
        # parameter existed -- this is ``tuned_params``'s own pre-existing refusal,
        # unwrapped, since ``params_fold_set`` changes nothing about it when omitted.
        fit_fold(PRIMARY, training, operational_fold)

    borrowed = fit_fold(PRIMARY, training, operational_fold, params_fold_set="quarterly")
    quarterly_native = fit_fold(PRIMARY, training, FOLD)
    assert borrowed.params == quarterly_native.params
    assert borrowed.n_estimators == quarterly_native.n_estimators
    # The fit itself still describes the operational fold, not "quarterly" -- only the
    # hyperparameter *lookup* was borrowed.
    assert borrowed.fold_set == "operational"
    assert borrowed.fold_id == "operational-2027-01-01"
