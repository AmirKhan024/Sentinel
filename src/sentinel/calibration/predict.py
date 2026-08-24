"""Applying a frozen calibrator to a window. Pure, and deliberately incapable of fitting.

The separation is the point. Every function here takes an already-``FittedCalibrator`` and
a sequence of base probabilities; none of them takes a label, so no code path in this module
can learn anything from the window it is scoring. A calibrator applied to the test window
must be a frozen object, and the cheapest way to guarantee that is to make the module that
applies it unable to do anything else.

The mapping is reproduced from the *extracted* parameters rather than by calling the
scikit-learn estimator, so that the arithmetic exercised here is the same arithmetic a
consumer would use reading ``calibrator_parameters_*.parquet`` and
``calibrator_isotonic_breakpoints_*.parquet`` from disk. If the two ever disagreed, the
persisted calibrator would be a description of something other than what ran, and
``validate.py`` asserts they do not.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from sentinel.calibration.definitions import Method
from sentinel.calibration.models import FittedCalibrator
from sentinel.calibration.preprocess import expit, logit

logger = logging.getLogger(__name__)


class CalibrationPredictError(ValueError):
    """Raised when a frozen calibrator cannot score a window."""


def apply(calibrator: FittedCalibrator, probabilities: Sequence[float]) -> list[float]:
    """Map uncalibrated probabilities through a frozen calibrator.

    Input is always the base model's **probability**. The logit transform Platt needs is
    applied here rather than by the caller, so there is exactly one place the sigmoid can be
    applied and no caller can apply it twice (ADR 0027).
    """
    if not probabilities:
        raise CalibrationPredictError(
            f"{calibrator.model_name}/{calibrator.fold_id}: nothing to score. Component 5 "
            "skips an empty window; a calibrator should never be asked to score one."
        )

    if calibrator.method is Method.PLATT:
        calibrated = _apply_platt(calibrator, probabilities)
    elif calibrator.method is Method.ISOTONIC:
        calibrated = _apply_isotonic(calibrator, probabilities)
    else:  # pragma: no cover - the enum is exhaustive
        raise CalibrationPredictError(f"unknown calibration method {calibrator.method!r}")

    bad = [v for v in calibrated if not np.isfinite(v)]
    if bad:
        raise CalibrationPredictError(
            f"{calibrator.model_name}/{calibrator.fold_id}: {len(bad)} non-finite calibrated "
            "score(s). The evaluator rejects these rather than imputing them."
        )
    outside = [v for v in calibrated if v < 0.0 or v > 1.0]
    if outside:
        raise CalibrationPredictError(
            f"{calibrator.model_name}/{calibrator.fold_id}: {len(outside)} calibrated "
            "score(s) outside [0, 1], which cannot be a probability"
        )
    return calibrated


def _apply_platt(calibrator: FittedCalibrator, probabilities: Sequence[float]) -> list[float]:
    """``sigmoid(a * logit(p) + b)`` from the two extracted parameters."""
    if calibrator.coefficient is None or calibrator.intercept is None:
        raise CalibrationPredictError(
            f"{calibrator.model_name}/{calibrator.fold_id}: Platt calibrator has no fitted "
            "parameters, so its mapping cannot be reproduced"
        )
    a, b = calibrator.coefficient, calibrator.intercept
    return [expit(a * logit(p) + b) for p in probabilities]


def _apply_isotonic(calibrator: FittedCalibrator, probabilities: Sequence[float]) -> list[float]:
    """Linear interpolation between the fitted breakpoints, clipped at the fitted range.

    ``out_of_bounds="clip"`` is what ``np.interp`` does natively at the ends, so the two
    agree by construction: a test-window score below the calibration window's minimum takes
    the first breakpoint's value, and one above its maximum takes the last. Without the clip
    such a score would be NaN, which the prediction contract rejects as a null.
    """
    if not calibrator.x_thresholds:
        raise CalibrationPredictError(
            f"{calibrator.model_name}/{calibrator.fold_id}: isotonic calibrator has no "
            "breakpoints, so its mapping cannot be reproduced"
        )
    values = np.interp(
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(calibrator.x_thresholds, dtype=np.float64),
        np.asarray(calibrator.y_thresholds, dtype=np.float64),
    )
    return [float(v) for v in values]


def is_monotone(calibrator: FittedCalibrator, *, probes: int = 200) -> bool:
    """Whether the mapping is non-decreasing over a grid spanning ``(0, 1)``.

    Checked by probing rather than by inspecting parameters, because it is the *applied*
    mapping that must be monotone. Note this is monotone in the weak sense: isotonic is
    expected to pass while producing plateaus, and its ties are counted separately.
    """
    grid = [(i + 0.5) / probes for i in range(probes)]
    mapped = apply(calibrator, grid)
    return all(a <= b for a, b in zip(mapped, mapped[1:], strict=False))


def creates_ties(calibrator: FittedCalibrator, *, probes: int = 200) -> bool:
    """Whether distinct inputs can leave with the same calibrated value.

    True for isotonic on any window where pool-adjacent-violators pooled anything, false for
    Platt, which is strictly monotone.
    """
    grid = [(i + 0.5) / probes for i in range(probes)]
    mapped = apply(calibrator, grid)
    return len(set(mapped)) < len(mapped)


__all__ = ["CalibrationPredictError", "apply", "creates_ties", "is_monotone"]
