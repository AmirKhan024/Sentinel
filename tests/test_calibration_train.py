"""Fitting the two calibrators, and the selection protocol that chooses between them.

The fitting tests check that each method does what its name claims on data where the answer
is known by construction. The selection tests check ADR 0025's protocol: the expanding
prefix, the pre-declared tie rule, and the refusal to decide on a window too small.
"""

from __future__ import annotations

import dataclasses
import random
from datetime import date

import pytest

from sentinel.calibration import predict, train
from sentinel.calibration.definitions import TIE_THRESHOLD, Method
from sentinel.calibration.preprocess import expit, logit
from sentinel.calibration.train import CalibrationTrainError
from sentinel.evaluation.metrics import brier, log_loss
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


def _population(n: int = 1500, slope: float = 0.5, seed: int = 3) -> tuple[list[int], list[float]]:
    rng = random.Random(seed)
    labels: list[int] = []
    probabilities: list[float] = []
    for _ in range(n):
        z = rng.gauss(0.0, 2.0)
        labels.append(1 if rng.random() < expit(slope * z) else 0)
        probabilities.append(expit(z))
    return labels, probabilities


def _fit(method: Method, labels: list[int], probabilities: list[float]) -> object:
    return train.fit_method(
        method,
        labels,
        probabilities,
        model_name="synthetic",
        fold=FOLD,
        fit_start=FOLD.calibration_start,
        fit_end=FOLD.calibration_end,
    )


# --- Platt --------------------------------------------------------------------


def test_platt_recovers_a_planted_overconfidence() -> None:
    """The fitted slope IS the correction: a model claiming twice the log-odds gets ~0.5."""
    labels, probabilities = _population(n=8000, slope=0.5, seed=5)
    calibrator = _fit(Method.PLATT, labels, probabilities)
    assert calibrator.coefficient == pytest.approx(0.5, abs=0.06)  # type: ignore[attr-defined]
    assert calibrator.intercept == pytest.approx(0.0, abs=0.15)  # type: ignore[attr-defined]


def test_platt_leaves_an_already_calibrated_model_almost_alone() -> None:
    """Slope 1, intercept 0 is inside the model family -- which is why the logit is the input.

    On a probability scale the identity map would not be reachable, so a calibrated model
    could not be left alone by its own calibrator.
    """
    labels, probabilities = _population(n=8000, slope=1.0, seed=9)
    calibrator = _fit(Method.PLATT, labels, probabilities)
    assert calibrator.coefficient == pytest.approx(1.0, abs=0.12)  # type: ignore[attr-defined]

    calibrated = predict.apply(calibrator, probabilities)  # type: ignore[arg-type]
    assert max(abs(a - b) for a, b in zip(probabilities, calibrated, strict=True)) < 0.05


def test_platt_improves_brier_on_the_window_it_was_fitted_on() -> None:
    labels, probabilities = _population()
    calibrated = predict.apply(_fit(Method.PLATT, labels, probabilities), probabilities)  # type: ignore[arg-type]
    assert brier(labels, calibrated) < brier(labels, probabilities)


def test_the_platt_mapping_is_the_two_persisted_parameters() -> None:
    """``sigmoid(a * logit(p) + b)`` and nothing else, so the artifact reproduces it."""
    labels, probabilities = _population()
    calibrator = _fit(Method.PLATT, labels, probabilities)
    a, b = calibrator.coefficient, calibrator.intercept  # type: ignore[attr-defined]

    probes = [(i + 0.5) / 100 for i in range(100)]
    expected = [expit(a * logit(p) + b) for p in probes]
    assert predict.apply(calibrator, probes) == expected  # type: ignore[arg-type]


# --- isotonic -----------------------------------------------------------------


def test_isotonic_produces_a_reproducible_breakpoint_table() -> None:
    labels, probabilities = _population()
    calibrator = _fit(Method.ISOTONIC, labels, probabilities)
    assert calibrator.breakpoint_count > 1  # type: ignore[attr-defined]
    assert len(calibrator.x_thresholds) == len(calibrator.y_thresholds)  # type: ignore[attr-defined]
    assert list(calibrator.x_thresholds) == sorted(calibrator.x_thresholds)  # type: ignore[attr-defined]
    assert list(calibrator.y_thresholds) == sorted(calibrator.y_thresholds)  # type: ignore[attr-defined]


def test_isotonic_clips_rather_than_returning_nan_outside_its_fitted_range() -> None:
    """A NaN here would be rejected by the prediction contract as a null score."""
    labels, probabilities = _population()
    calibrator = _fit(Method.ISOTONIC, labels, probabilities)
    below, above = predict.apply(calibrator, [1e-9, 1.0 - 1e-9])  # type: ignore[arg-type]
    assert 0.0 <= below <= 1.0
    assert 0.0 <= above <= 1.0
    assert below == calibrator.y_thresholds[0]  # type: ignore[attr-defined]
    assert above == calibrator.y_thresholds[-1]  # type: ignore[attr-defined]


def test_isotonic_output_is_monotone_non_decreasing() -> None:
    labels, probabilities = _population()
    calibrator = _fit(Method.ISOTONIC, labels, probabilities)
    probes = [(i + 0.5) / 300 for i in range(300)]
    mapped = predict.apply(calibrator, probes)  # type: ignore[arg-type]
    assert all(a <= b for a, b in zip(mapped, mapped[1:], strict=False))


def test_isotonic_is_more_flexible_and_that_is_visible_in_the_breakpoint_count() -> None:
    """Two parameters against many: the whole variance argument in one comparison."""
    labels, probabilities = _population()
    platt = _fit(Method.PLATT, labels, probabilities)
    isotonic = _fit(Method.ISOTONIC, labels, probabilities)
    assert platt.breakpoint_count == 0  # type: ignore[attr-defined]
    assert isotonic.breakpoint_count > 10  # type: ignore[attr-defined]


# --- refusals -----------------------------------------------------------------


@pytest.mark.parametrize("method", list(Method), ids=lambda m: m.value)
def test_a_single_class_window_is_refused_not_fitted(method: Method) -> None:
    """A calibrator fitted on one class maps everything to a constant, which is not a correction."""
    with pytest.raises(CalibrationTrainError, match="single class"):
        _fit(method, [1] * 200, [0.4] * 200)


@pytest.mark.parametrize("method", list(Method), ids=lambda m: m.value)
def test_an_empty_window_is_refused(method: Method) -> None:
    with pytest.raises(CalibrationTrainError, match="nothing to fit"):
        _fit(method, [], [])


@pytest.mark.parametrize("method", list(Method), ids=lambda m: m.value)
def test_mismatched_lengths_are_refused(method: Method) -> None:
    with pytest.raises(CalibrationTrainError, match="labels against"):
        _fit(method, [0, 1, 0], [0.1, 0.2])


# --- the selection protocol ---------------------------------------------------


def _trial(method: Method, loss: float, fold_index: int = 0) -> object:
    """A hand-built trial, so the selection logic is tested rather than the fitters."""
    labels, probabilities = _population(n=400)
    real = train.trial(
        method,
        model_name="synthetic",
        fold=FOLD,
        fold_index=fold_index,
        inner_split_date=date(2022, 3, 10),
        fit_labels=labels[:300],
        fit_probabilities=probabilities[:300],
        select_labels=labels[300:],
        select_probabilities=probabilities[300:],
    )
    return dataclasses.replace(real, inner_select_log_loss=loss)


def test_isotonic_wins_only_when_it_clears_the_threshold() -> None:
    history = [{
        Method.PLATT: _trial(Method.PLATT, 0.700),
        Method.ISOTONIC: _trial(Method.ISOTONIC, 0.700 - TIE_THRESHOLD - 0.001),
    }]
    outcome = train.select_method(history)  # type: ignore[arg-type]
    assert outcome.method is Method.ISOTONIC
    assert not outcome.declared_tie
    assert "clearing the pre-declared" in outcome.reason


def test_a_margin_inside_the_threshold_goes_to_platt() -> None:
    """Not-worse-by-enough is not good enough: the simpler method keeps the fold."""
    history = [{
        Method.PLATT: _trial(Method.PLATT, 0.700),
        Method.ISOTONIC: _trial(Method.ISOTONIC, 0.700 - TIE_THRESHOLD / 2),
    }]
    outcome = train.select_method(history)  # type: ignore[arg-type]
    assert outcome.method is Method.PLATT
    assert outcome.declared_tie
    # The per-fold winner is still recorded, so the alternative stays auditable.
    assert outcome.per_fold_winner is Method.ISOTONIC


def test_a_clearly_worse_isotonic_goes_to_platt() -> None:
    history = [{
        Method.PLATT: _trial(Method.PLATT, 0.685),
        Method.ISOTONIC: _trial(Method.ISOTONIC, 1.103),
    }]
    outcome = train.select_method(history)  # type: ignore[arg-type]
    assert outcome.method is Method.PLATT
    assert outcome.per_fold_winner is Method.PLATT
    assert outcome.gap > 0


def test_the_decision_is_the_mean_over_the_expanding_prefix() -> None:
    """Fold 3's choice averages folds 1-3, so one bad fold does not flip the series."""
    def pair(index: int, isotonic_loss: float) -> dict[Method, object]:
        return {
            Method.PLATT: _trial(Method.PLATT, 0.70, index),
            Method.ISOTONIC: _trial(Method.ISOTONIC, isotonic_loss, index),
        }

    history = [pair(0, 0.60), pair(1, 0.60), pair(2, 0.90)]
    outcome = train.select_method(history)  # type: ignore[arg-type]
    assert outcome.prefix_mean_log_loss[Method.PLATT] == pytest.approx(0.70)
    assert outcome.prefix_mean_log_loss[Method.ISOTONIC] == pytest.approx(0.70)
    # Tied on the prefix mean even though isotonic lost this fold badly, so Platt keeps it.
    assert outcome.method is Method.PLATT
    assert outcome.per_fold_winner is Method.PLATT


def test_the_decision_always_belongs_to_the_last_fold_in_the_history() -> None:
    """The API cannot express re-deciding an earlier fold with later evidence."""
    first = {
        Method.PLATT: _trial(Method.PLATT, 0.70, 0),
        Method.ISOTONIC: _trial(Method.ISOTONIC, 0.60, 0),
    }
    second = {
        Method.PLATT: _trial(Method.PLATT, 0.70, 1),
        Method.ISOTONIC: _trial(Method.ISOTONIC, 0.60, 1),
    }
    assert train.select_method([first]).fold_index == 0  # type: ignore[arg-type]
    assert train.select_method([first, second]).fold_index == 1  # type: ignore[arg-type]


def test_a_history_missing_a_method_is_refused() -> None:
    """Both methods must be fitted for every fold, or the counterfactual is lost."""
    with pytest.raises(CalibrationTrainError, match="both methods must be fitted"):
        train.select_method([{Method.PLATT: _trial(Method.PLATT, 0.7)}])  # type: ignore[arg-type]


def test_an_empty_history_is_refused() -> None:
    with pytest.raises(CalibrationTrainError, match="no trials"):
        train.select_method([])


def test_the_override_is_recorded_as_having_bypassed_the_rule() -> None:
    """``--method`` is diagnostic, and the artifact has to say so."""
    history = [{
        Method.PLATT: _trial(Method.PLATT, 0.60),
        Method.ISOTONIC: _trial(Method.ISOTONIC, 0.90),
    }]
    outcome = train.select_method(history, override=Method.ISOTONIC)  # type: ignore[arg-type]
    assert outcome.method is Method.ISOTONIC
    assert "forced on the command line" in outcome.reason
    assert "pre-registered rule was not applied" in outcome.reason.replace("\n", " ")


def test_a_trial_records_diagnostics_that_do_not_decide_anything() -> None:
    """ECE, MCE and Brier are logged on the inner-select window and ignored by the rule."""
    entry = _trial(Method.PLATT, 0.7)
    for field in ("inner_select_ece", "inner_select_mce", "inner_select_brier"):
        assert getattr(entry, field) >= 0.0
    assert entry.inner_select_rows > 0  # type: ignore[attr-defined]
    assert entry.inner_fit_rows > 0  # type: ignore[attr-defined]


def test_the_trial_scores_the_select_window_not_the_fit_window() -> None:
    """If it scored its own fitting rows the comparison would favour the flexible method."""
    labels, probabilities = _population(n=800)
    entry = train.trial(
        Method.ISOTONIC,
        model_name="synthetic",
        fold=FOLD,
        fold_index=0,
        inner_split_date=date(2022, 3, 10),
        fit_labels=labels[:600],
        fit_probabilities=probabilities[:600],
        select_labels=labels[600:],
        select_probabilities=probabilities[600:],
    )
    assert entry.inner_fit_rows == 600
    assert entry.inner_select_rows == 200

    refitted = _fit(Method.ISOTONIC, labels[:600], probabilities[:600])
    in_sample = log_loss(labels[:600], predict.apply(refitted, probabilities[:600]))  # type: ignore[arg-type]
    assert entry.inner_select_log_loss != in_sample
