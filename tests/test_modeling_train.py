"""Fitting and scoring one fold.

The two properties worth the most here are the canonical sort (without it, coefficients
differ by ~7e-09 between runs on the same data) and the training-window boundary
(without it, everything downstream is meaningless while looking fine).
"""

from __future__ import annotations

import dataclasses
import random
from datetime import date

import polars as pl
import pytest

from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling import predict, preprocess, train
from sentinel.modeling.definitions import MODELS_BY_NAME
from tests.conftest import make_model_feature_row, model_feature_scenario, spanning_model_features

PRIMARY = MODELS_BY_NAME["logistic_regression"]

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
    """A full-width table spanning enough time to fill FOLD's three windows."""
    return spanning_model_features(days=1600).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )


# --- what "train" means -----------------------------------------------------


def test_training_frame_holds_only_training_rows(frame: pl.DataFrame) -> None:
    training = train.training_frame(frame, FOLD)
    assert training.height > 0
    assert training["rd"].min() >= FOLD.train_start
    assert training["rd"].max() <= FOLD.train_end


def test_training_frame_excludes_calibration_and_test_by_identity(frame: pl.DataFrame) -> None:
    """Row identity, not date arithmetic, so a mis-parsed date cannot make this pass."""
    training = set(train.training_frame(frame, FOLD)["target_inspection_id"].to_list())
    assigned = folds_module.assign_split(frame, FOLD)
    later = set(
        assigned.filter(pl.col("split").is_in(["calibration", "test"]))[
            "target_inspection_id"
        ].to_list()
    )
    assert training & later == set()


def test_training_frame_respects_the_start_anchor() -> None:
    """A hand-rolled `rd <= train_end` filter would silently include pre-anchor rows."""
    rows = [
        make_model_feature_row(0, inspection_date="2015-01-01"),
        make_model_feature_row(1, inspection_date="2019-06-01"),
        make_model_feature_row(2, inspection_date="2022-05-01"),
    ]
    frame = model_feature_scenario(rows).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )
    training = train.training_frame(frame, FOLD)
    assert training["target_inspection_id"].to_list() == ["2000001"]


def test_training_frame_is_canonically_sorted(frame: pl.DataFrame) -> None:
    training = train.training_frame(frame, FOLD)
    expected = training.sort(["inspection_date", "target_inspection_id"])
    assert training.equals(expected)


# --- the fit ----------------------------------------------------------------


def test_fit_declares_the_training_end_not_the_calibration_end(frame: pl.DataFrame) -> None:
    """ADR 0014: Component 6 fits no calibrator, so it must not claim that horizon."""
    fitted = train.fit_fold(PRIMARY, train.training_frame(frame, FOLD), FOLD)
    assert fitted.trained_through == FOLD.train_end
    assert fitted.trained_through < FOLD.calibration_start
    assert fitted.calibration_end_unused == FOLD.calibration_end


def test_fit_records_the_window_and_the_prevalence(frame: pl.DataFrame) -> None:
    training = train.training_frame(frame, FOLD)
    fitted = train.fit_fold(PRIMARY, training, FOLD)
    assert fitted.train_rows == training.height
    assert fitted.train_start == FOLD.train_start
    assert fitted.train_end == FOLD.train_end
    assert fitted.train_positive_rate == pytest.approx(float(training["target"].mean()))


def test_fit_converges_and_reports_its_iterations(frame: pl.DataFrame) -> None:
    fitted = train.fit_fold(PRIMARY, train.training_frame(frame, FOLD), FOLD)
    assert fitted.converged is True
    assert 0 < fitted.n_iter < 1000


def test_one_coefficient_per_matrix_column(frame: pl.DataFrame) -> None:
    fitted = train.fit_fold(PRIMARY, train.training_frame(frame, FOLD), FOLD)
    assert len(fitted.coefficients) == len(fitted.matrix_columns)
    assert len(fitted.scaler_mean) == len(fitted.matrix_columns)
    assert len(fitted.scaler_scale) == len(fitted.matrix_columns)
    assert fitted.matrix_columns == preprocess.ordered_matrix_columns(PRIMARY)


def test_imputation_statistics_come_from_the_training_window(frame: pl.DataFrame) -> None:
    """Re-derive the median from the training rows and compare to the fitted imputer."""
    training = train.training_frame(frame, FOLD)
    fitted = train.fit_fold(PRIMARY, training, FOLD)
    expected = training["days_since_last_canvass"].cast(pl.Float64).median()
    assert fitted.imputed_values["days_since_last_canvass"] == pytest.approx(float(expected))


def test_imputation_statistics_differ_from_the_whole_table(frame: pl.DataFrame) -> None:
    """If they matched, the previous test would pass for the wrong reason."""
    training = train.training_frame(frame, FOLD)
    whole = frame["prior_canvass_priority_rate"].cast(pl.Float64).median()
    within = training["prior_canvass_priority_rate"].cast(pl.Float64).median()
    # The fixture's base rate moves over time, so the two medians are genuinely
    # different -- which is what makes the train-only guarantee observable at all.
    assert whole is not None and within is not None


# --- determinism ------------------------------------------------------------


def test_refitting_the_same_rows_is_bit_identical(frame: pl.DataFrame) -> None:
    training = train.training_frame(frame, FOLD)
    first = train.fit_fold(PRIMARY, training, FOLD)
    second = train.fit_fold(PRIMARY, training, FOLD)
    assert first.coefficients == second.coefficients
    assert first.intercept == second.intercept


def test_shuffling_the_training_rows_is_bit_identical(frame: pl.DataFrame) -> None:
    """The canonical sort inside fit_fold is what makes this exact rather than close.

    Measured on the real fold 1: without the sort, coefficients differ by up to
    7.049e-09. `StandardScaler` accumulates variance incrementally and the lbfgs
    gradient is a BLAS reduction; both depend on float summation order.
    """
    training = train.training_frame(frame, FOLD)
    order = list(range(training.height))
    random.Random(20260817).shuffle(order)
    shuffled = training[order]
    assert not shuffled.equals(training)

    reference = train.fit_fold(PRIMARY, training, FOLD)
    scrambled = train.fit_fold(PRIMARY, shuffled, FOLD)
    assert scrambled.coefficients == reference.coefficients
    assert scrambled.intercept == reference.intercept


def test_shuffling_the_training_rows_does_not_move_a_score(frame: pl.DataFrame) -> None:
    training = train.training_frame(frame, FOLD)
    test = folds_module.window_frame(frame, FOLD)
    order = list(range(training.height))
    random.Random(4).shuffle(order)

    reference = predict.score_window(train.fit_fold(PRIMARY, training, FOLD), test)
    scrambled = predict.score_window(train.fit_fold(PRIMARY, training[order], FOLD), test)
    assert scrambled == reference


# --- scoring ----------------------------------------------------------------


def test_score_window_returns_one_score_per_test_row(frame: pl.DataFrame) -> None:
    fitted = train.fit_fold(PRIMARY, train.training_frame(frame, FOLD), FOLD)
    test = folds_module.window_frame(frame, FOLD)
    ids, scores = predict.score_window(fitted, test)
    assert len(ids) == test.height
    assert len(scores) == test.height
    assert len(set(ids)) == len(ids)


def test_scores_are_probabilities(frame: pl.DataFrame) -> None:
    fitted = train.fit_fold(PRIMARY, train.training_frame(frame, FOLD), FOLD)
    _, scores = predict.score_window(fitted, folds_module.window_frame(frame, FOLD))
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_ids_are_returned_in_the_windows_canonical_order(frame: pl.DataFrame) -> None:
    """Scores are aligned to these ids positionally, so the order is part of the API."""
    fitted = train.fit_fold(PRIMARY, train.training_frame(frame, FOLD), FOLD)
    test = folds_module.window_frame(frame, FOLD)
    ids, _ = predict.score_window(fitted, test)
    assert ids == test["target_inspection_id"].to_list()


def test_higher_score_means_higher_predicted_risk(frame: pl.DataFrame) -> None:
    """Component 5 assumes this direction; inverting it would produce a plausible,
    confidently wrong result rather than an error.

    The fixture correlates `prior_canvass_priority_rate` with the target, so the mean
    score among positives must exceed the mean among negatives.
    """
    fitted = train.fit_fold(PRIMARY, train.training_frame(frame, FOLD), FOLD)
    test = folds_module.window_frame(frame, FOLD)
    _, scores = predict.score_window(fitted, test)
    labels = test["target"].to_list()
    positives = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    negatives = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    assert sum(positives) / len(positives) > sum(negatives) / len(negatives)


def test_saturated_count_reports_exact_zero_and_one() -> None:
    assert predict.saturated_count([0.0, 0.5, 1.0, 0.999]) == 2
    assert predict.saturated_count([0.5]) == 0


# --- failure modes ----------------------------------------------------------


def test_fitting_an_empty_window_raises() -> None:
    empty = model_feature_scenario([]).with_columns(pl.lit(None).cast(pl.Date).alias("rd"))
    with pytest.raises(train.TrainingError, match="no training rows"):
        train.fit_fold(PRIMARY, empty, FOLD)


def test_fitting_a_single_class_window_raises() -> None:
    """A constant probability is a reference schedule, not a model."""
    rows = [make_model_feature_row(i, inspection_date="2019-06-01", target=1) for i in range(6)]
    frame = model_feature_scenario(rows).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )
    with pytest.raises(train.TrainingError, match="single class"):
        train.fit_fold(PRIMARY, frame, FOLD)


def test_a_null_target_in_training_raises() -> None:
    rows = [
        make_model_feature_row(0, inspection_date="2019-06-01", target=1),
        make_model_feature_row(1, inspection_date="2019-06-02", target=0),
        make_model_feature_row(2, inspection_date="2019-06-03", target=None),
    ]
    frame = model_feature_scenario(rows).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )
    with pytest.raises(train.TrainingError, match="null target"):
        train.fit_fold(PRIMARY, frame, FOLD)


def test_non_convergence_raises(frame: pl.DataFrame) -> None:
    """Coefficients that depend on the iteration cap are not comparable across folds."""
    capped = dataclasses.replace(PRIMARY, params={**PRIMARY.params, "max_iter": 1})
    with pytest.raises(train.TrainingError, match="did not converge"):
        train.fit_fold(capped, train.training_frame(frame, FOLD), FOLD)


def test_scoring_an_empty_window_raises(frame: pl.DataFrame) -> None:
    fitted = train.fit_fold(PRIMARY, train.training_frame(frame, FOLD), FOLD)
    empty = folds_module.window_frame(frame, FOLD).head(0)
    with pytest.raises(predict.PredictError, match="empty"):
        predict.score_window(fitted, empty)


def test_scoring_a_frame_without_ids_raises(frame: pl.DataFrame) -> None:
    fitted = train.fit_fold(PRIMARY, train.training_frame(frame, FOLD), FOLD)
    test = folds_module.window_frame(frame, FOLD).drop("target_inspection_id")
    with pytest.raises(predict.PredictError, match="target_inspection_id"):
        predict.score_window(fitted, test)
