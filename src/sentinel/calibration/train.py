"""Fitting a calibrator, and choosing between the two. Pure -- no filesystem, no clock.

Two things happen here, and only one of them is allowed to see a label from anywhere other
than the calibration window.

**Fitting.** Platt is a two-parameter logistic on the recovered logit; isotonic is a
non-parametric monotone step function on the probability (ADR 0027). Both are always
fitted, for every fold, including where one of them loses -- so the counterfactual stays
answerable from the artifact rather than by re-running with a different flag, which is how
a selection quietly becomes a test-set selection.

**Selecting.** ADR 0025. Each calibration window is cut chronologically; both methods are
fitted on the earlier portion and compared on the later one; the winner over the
*expanding prefix* of folds 1..k is frozen and refitted on the full window.

The one subtlety worth reading twice: the selection is an expanding prefix rather than a
pool over all folds, and that is not a stylistic choice. **Fold N's calibration window is
fold N-1's test window.** Pooling every fold's inner-select result to choose fold 1's
method would choose it using fold 1's own test period. The prefix keeps every input at or
before ``fold_k.calibration_end``, which is exactly the horizon
``evaluation.contract._training_horizon`` already enforces.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np

from sentinel.calibration.definitions import (
    INPUT_TRANSFORM,
    ISOTONIC_PARAMS,
    PLATT_PARAMS,
    TIE_PREFERENCE,
    TIE_THRESHOLD,
    Method,
)
from sentinel.calibration.models import (
    FittedCalibrator,
    MethodTrial,
    SelectionOutcome,
)
from sentinel.calibration.predict import apply
from sentinel.calibration.preprocess import logits_of
from sentinel.evaluation.metrics import brier, ece, log_loss, mce
from sentinel.evaluation.models import FoldSpec

logger = logging.getLogger(__name__)


class CalibrationTrainError(RuntimeError):
    """Raised when a calibrator cannot be fitted, or a method cannot be chosen."""


def _positive_rate(labels: Sequence[int]) -> float | None:
    return sum(labels) / len(labels) if labels else None


def _check_fittable(labels: Sequence[int], probabilities: Sequence[float], what: str) -> None:
    if len(labels) != len(probabilities):
        raise CalibrationTrainError(
            f"{what}: {len(labels)} labels against {len(probabilities)} probabilities"
        )
    if not labels:
        raise CalibrationTrainError(f"{what}: nothing to fit on")
    if len(set(labels)) < 2:
        raise CalibrationTrainError(
            f"{what}: the fitting window contains a single class ({labels[0]}). A calibrator "
            "fitted on one class maps every score to a constant, which is not a correction."
        )


def fit_platt(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    model_name: str,
    fold: FoldSpec,
    fit_start: date,
    fit_end: date,
) -> FittedCalibrator:
    """Platt scaling: a logistic regression of the label on ``logit(p)``.

    Deliberately unpenalised (``C = 1e10``). Platt is a two-parameter maximum-likelihood
    fit, and scikit-learn's default ``C = 1.0`` would shrink the slope toward zero -- which
    would *cause* the under-confidence the calibrator exists to remove.

    Slope 1.0 and intercept 0.0 mean the base model was already calibrated, which is why
    the logit is the right input scale: on that scale the identity map is inside the model
    family, so a well-calibrated model can be left alone by its own calibrator.
    """
    from sklearn.linear_model import LogisticRegression

    what = f"{model_name}/{fold.fold_id}/platt"
    _check_fittable(labels, probabilities, what)

    x = np.asarray(logits_of(probabilities), dtype=np.float64).reshape(-1, 1)
    y = np.asarray(labels, dtype=np.int64)
    estimator = LogisticRegression(**dict(PLATT_PARAMS))
    estimator.fit(x, y)

    return FittedCalibrator(
        model_name=model_name,
        fold_set=fold.fold_set,
        fold_id=fold.fold_id,
        method=Method.PLATT,
        estimator=estimator,
        input_transform=INPUT_TRANSFORM[Method.PLATT],
        fit_rows=len(labels),
        fit_positive_rate=_positive_rate(labels),
        fit_start=fit_start,
        fit_end=fit_end,
        coefficient=float(estimator.coef_[0][0]),
        intercept=float(estimator.intercept_[0]),
    )


def fit_isotonic(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    model_name: str,
    fold: FoldSpec,
    fit_start: date,
    fit_end: date,
) -> FittedCalibrator:
    """Isotonic regression: a non-parametric monotone step function on the probability.

    Fitted on ``p`` rather than ``logit(p)``. Isotonic is invariant to any strictly
    monotone reparametrisation, so the choice is free, and ``p`` keeps the persisted
    breakpoints readable.

    ``out_of_bounds="clip"`` is mandatory: a test-window score outside the calibration
    window's observed range would otherwise map to NaN, which the prediction contract
    rejects as a null score.

    The fitted ``X_thresholds_`` / ``y_thresholds_`` are extracted here so the mapping is
    reproducible from the artifact alone -- with ``np.interp`` they determine it exactly.
    """
    from sklearn.isotonic import IsotonicRegression

    what = f"{model_name}/{fold.fold_id}/isotonic"
    _check_fittable(labels, probabilities, what)

    x = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    estimator = IsotonicRegression(**dict(ISOTONIC_PARAMS))
    estimator.fit(x, y)

    x_thresholds = tuple(float(v) for v in np.asarray(estimator.X_thresholds_))
    y_thresholds = tuple(float(v) for v in np.asarray(estimator.y_thresholds_))
    if len(x_thresholds) != len(y_thresholds) or not x_thresholds:
        raise CalibrationTrainError(
            f"{what}: isotonic produced {len(x_thresholds)} x-thresholds and "
            f"{len(y_thresholds)} y-thresholds; the mapping would not be reproducible"
        )

    return FittedCalibrator(
        model_name=model_name,
        fold_set=fold.fold_set,
        fold_id=fold.fold_id,
        method=Method.ISOTONIC,
        estimator=estimator,
        input_transform=INPUT_TRANSFORM[Method.ISOTONIC],
        fit_rows=len(labels),
        fit_positive_rate=_positive_rate(labels),
        fit_start=fit_start,
        fit_end=fit_end,
        x_thresholds=x_thresholds,
        y_thresholds=y_thresholds,
        x_min=float(x.min()),
        x_max=float(x.max()),
    )


def fit_method(
    method: Method,
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    model_name: str,
    fold: FoldSpec,
    fit_start: date,
    fit_end: date,
) -> FittedCalibrator:
    """Dispatch to the named method's fitter."""
    fitter = fit_platt if method is Method.PLATT else fit_isotonic
    return fitter(
        labels,
        probabilities,
        model_name=model_name,
        fold=fold,
        fit_start=fit_start,
        fit_end=fit_end,
    )


def trial(
    method: Method,
    *,
    model_name: str,
    fold: FoldSpec,
    fold_index: int,
    inner_split_date: date,
    fit_labels: Sequence[int],
    fit_probabilities: Sequence[float],
    select_labels: Sequence[int],
    select_probabilities: Sequence[float],
) -> MethodTrial:
    """Fit one method on the inner-fit portion and score it on the inner-select portion.

    The scored numbers all come from a window carved out of the **calibration** period.
    None of them is a result, and only ``inner_select_log_loss`` decides anything -- ECE,
    MCE and Brier are recorded as diagnostics (ADR 0025).
    """
    started = time.perf_counter()
    calibrator = fit_method(
        method,
        fit_labels,
        fit_probabilities,
        model_name=model_name,
        fold=fold,
        fit_start=fold.calibration_start,
        fit_end=inner_split_date,
    )
    calibrated = apply(calibrator, select_probabilities)
    elapsed = time.perf_counter() - started

    return MethodTrial(
        model_name=model_name,
        fold_set=fold.fold_set,
        fold_id=fold.fold_id,
        fold_index=fold_index,
        method=method,
        inner_fit_rows=len(fit_labels),
        inner_select_rows=len(select_labels),
        inner_split_date=inner_split_date,
        inner_fit_positive_rate=_positive_rate(fit_labels),
        inner_select_positive_rate=_positive_rate(select_labels),
        inner_select_log_loss=log_loss(select_labels, calibrated),
        inner_select_brier=brier(select_labels, calibrated),
        inner_select_ece=ece(select_labels, calibrated),
        inner_select_mce=mce(select_labels, calibrated),
        seconds=elapsed,
    )


def select_method(
    history: Sequence[Mapping[Method, MethodTrial]],
    *,
    override: Method | None = None,
) -> SelectionOutcome:
    """Freeze one method for the most recent fold in ``history``.

    ``history`` is folds 1..k of ONE fold set, in calibration order, each mapping both
    methods to their trial on that fold. The decision is the mean inner-select log-loss
    over the whole prefix, which is why the earlier folds are passed in at all.

    Every input therefore has ``rd <= fold_k.calibration_end``. That is the property that
    makes this legal, and ``validate.method_selection_reads_no_future_fold`` re-derives it
    from the artifact rather than trusting this docstring.

    ``override`` is the CLI's ``--method`` diagnostic escape. It is recorded in the
    manifest when used, and the production run does not pass it.
    """
    if not history:
        raise CalibrationTrainError("cannot select a method with no trials")
    current = history[-1]
    missing = [m for m in Method if m not in current]
    if missing:
        raise CalibrationTrainError(
            f"fold has no trial for {', '.join(m.value for m in missing)}; both methods must "
            "be fitted for every fold so the counterfactual stays auditable"
        )

    sample = current[TIE_PREFERENCE]
    prefix_mean = {
        method: float(np.mean([fold[method].inner_select_log_loss for fold in history]))
        for method in Method
    }
    per_fold_winner = min(Method, key=lambda m: current[m].inner_select_log_loss)

    # Lower log-loss is better. The gap is signed so that a positive value means isotonic
    # is worse, which is the direction the tie rule is written in.
    gap = prefix_mean[Method.ISOTONIC] - prefix_mean[TIE_PREFERENCE]
    beats_threshold = gap < -TIE_THRESHOLD
    declared_tie = abs(gap) <= TIE_THRESHOLD

    if override is not None:
        method = override
        reason = (
            f"--method {override.value} forced on the command line; the pre-registered rule "
            "was not applied. Diagnostic only."
        )
    elif beats_threshold:
        method = Method.ISOTONIC
        reason = (
            f"isotonic beat {TIE_PREFERENCE.value} by {-gap:.4f} nats of mean inner-select "
            f"log-loss over folds 1..{len(history)}, clearing the pre-declared "
            f"{TIE_THRESHOLD} threshold"
        )
    else:
        method = TIE_PREFERENCE
        reason = (
            f"{TIE_PREFERENCE.value} preferred: isotonic's mean inner-select log-loss over "
            f"folds 1..{len(history)} is {gap:+.4f} nats against it, which does not clear "
            f"the pre-declared {TIE_THRESHOLD} threshold"
            + (" (declared tie)" if declared_tie else "")
        )

    logger.info("%s/%s: frozen method %s -- %s", sample.model_name, sample.fold_id, method, reason)
    return SelectionOutcome(
        model_name=sample.model_name,
        fold_set=sample.fold_set,
        fold_id=sample.fold_id,
        fold_index=sample.fold_index,
        method=method,
        per_fold_winner=per_fold_winner,
        prefix_mean_log_loss=prefix_mean,
        gap=gap,
        declared_tie=declared_tie,
        reason=reason,
    )


__all__ = [
    "CalibrationTrainError",
    "fit_isotonic",
    "fit_method",
    "fit_platt",
    "select_method",
    "trial",
]
