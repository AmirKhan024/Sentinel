"""Window construction and the score-to-logit transform. Pure -- no filesystem, no clock.

Two things live here, and both are deliberately thin.

**The calibration window.** ``evaluation.folds.window_frame`` returns a fold's *test* rows
and takes no split argument, so Component 9 needs its own accessor. It is written as a
filter over ``assign_split`` rather than as a hand-rolled ``rd.is_between``, for exactly the
reason ``modeling.train.training_frame`` gives: there is then one definition of every window
in the repository -- the one Component 5's own validator independently re-derives -- and a
second one could drift from it without any test noticing.

**The logit.** ADR 0027. The calibrator is fed the logit recovered from the committed
probability rather than the base model's native decision margin, so that applying a frozen
calibrator to the test window is a pure function of an artifact that already exists on disk.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from datetime import date

import polars as pl

from sentinel.calibration.definitions import (
    INNER_SELECT_FRACTION,
    LOGIT_EPSILON,
    MIN_INNER_FIT_ROWS,
    MIN_INNER_SELECT_ROWS,
)
from sentinel.calibration.models import InnerSplit
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec

logger = logging.getLogger(__name__)

#: The order every window in this project is built in. Identical to ``window_frame``'s, so
#: scores returned positionally line up with the ids the evaluator expects.
WINDOW_SORT_KEYS: tuple[str, ...] = ("rd", "target_inspection_id")


class CalibrationPreprocessError(ValueError):
    """Raised when a window cannot be built or a probability cannot be transformed."""


def calibration_frame(
    frame: pl.DataFrame, fold: FoldSpec, *, date_column: str = "rd"
) -> pl.DataFrame:
    """The rows of one fold's calibration window, in deterministic order.

    The mirror of ``evaluation.folds.window_frame``, differing only in which split it keeps.
    Training and test rows are excluded by construction, not by subtraction: only rows whose
    split is ``calibration`` survive.
    """
    labelled = folds_module.assign_split(frame, fold, date_column=date_column)
    return (
        labelled.filter(pl.col("split") == "calibration")
        .sort([date_column, "target_inspection_id"])
    )


def inner_split_date(window: pl.DataFrame, fraction: float = INNER_SELECT_FRACTION) -> date:
    """The first date belonging to the inner-select portion.

    Delegates to ``neural.train.inner_split_date`` rather than reimplementing it: the
    whole-day property is the point, and two implementations of it would be two chances to
    lose it. Component 8's argument carries over unchanged -- two inspections of the same
    establishment days apart share almost all of their as-of history, so a row quantile
    would land mid-day and split rows that are not independent.
    """
    from sentinel.neural.train import NeuralTrainError
    from sentinel.neural.train import inner_split_date as _inner_split_date

    try:
        return _inner_split_date(window, fraction)
    except NeuralTrainError as exc:
        raise CalibrationPreprocessError(f"cannot split the calibration window: {exc}") from exc


def split_calibration_window(
    window: pl.DataFrame, fold: FoldSpec, *, fraction: float = INNER_SELECT_FRACTION
) -> InnerSplit:
    """Cut a calibration window into inner-fit and inner-select portions (ADR 0025).

    Returns positional indices into ``window`` rather than two frames, so the caller can
    subset the aligned score, margin and label sequences with the same indices and no
    re-join is ever needed.

    A window that cannot produce a usable split is **refused, not degraded**. Two folds
    calibrated under different selection rules are not comparable, and the whole protocol
    is a comparison -- the same posture ``neural.train.split_training_window`` takes.
    """
    if window.height == 0:
        raise CalibrationPreprocessError(
            f"fold {fold.fold_id}: calibration window is empty. Component 5 builds this "
            "window from the data; an empty one is a defect upstream."
        )

    cut = inner_split_date(window, fraction)
    days = window["rd"].to_list()
    fit_index = tuple(i for i, day in enumerate(days) if day < cut)
    select_index = tuple(i for i, day in enumerate(days) if day >= cut)

    if len(fit_index) < MIN_INNER_FIT_ROWS or len(select_index) < MIN_INNER_SELECT_ROWS:
        raise CalibrationPreprocessError(
            f"fold {fold.fold_id}: calibration window splits into {len(fit_index)} inner-fit "
            f"and {len(select_index)} inner-select rows at {cut}, against minimums of "
            f"{MIN_INNER_FIT_ROWS} and {MIN_INNER_SELECT_ROWS}. The fold is refused rather "
            "than calibrated on a window too small to choose a method on; a fold selected "
            "under a different rule would not be comparable with the others."
        )

    split = InnerSplit(cut=cut, fit_index=fit_index, select_index=select_index)
    logger.info(
        "Fold %s inner split at %s: %d fit, %d select",
        fold.fold_id,
        cut,
        split.fit_rows,
        split.select_rows,
    )
    return split


def logit(p: float) -> float:
    """``log(p / (1 - p))``, computed so neither tail loses precision.

    ``log(p) - log1p(-p)`` rather than ``log(p / (1 - p))``: for p near 1 the subtraction
    ``1 - p`` cancels catastrophically, while ``log1p`` is accurate there by construction.

    Clamped by ``LOGIT_EPSILON`` so a saturated probability yields a large finite number
    rather than an infinity. On this snapshot no score of the 34,261 per model sits at
    exactly 0 or 1, so the clamp never fires -- ``clamped_count`` exists to make a future
    snapshot that does saturate visible rather than silently corrected.
    """
    if not math.isfinite(p):
        raise CalibrationPreprocessError(f"cannot take the logit of a non-finite score: {p}")
    if p < 0.0 or p > 1.0:
        raise CalibrationPreprocessError(
            f"cannot take the logit of {p}, which is not a probability"
        )
    clamped = min(max(p, LOGIT_EPSILON), 1.0 - LOGIT_EPSILON)
    return math.log(clamped) - math.log1p(-clamped)


def expit(z: float) -> float:
    """The inverse of :func:`logit`, written to avoid overflow for large negative ``z``."""
    if not math.isfinite(z):
        raise CalibrationPreprocessError(f"cannot invert a non-finite logit: {z}")
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exponential = math.exp(z)
    return exponential / (1.0 + exponential)


def logits_of(probabilities: Sequence[float]) -> list[float]:
    """Vectorised :func:`logit` over a window's scores."""
    return [logit(p) for p in probabilities]


def clamped_count(probabilities: Sequence[float]) -> int:
    """How many scores the logit clamp would actually move.

    Reported rather than assumed to be zero: it is zero on this snapshot, and a non-zero
    value later means the base model started saturating, which is worth seeing.
    """
    return sum(1 for p in probabilities if p < LOGIT_EPSILON or p > 1.0 - LOGIT_EPSILON)


__all__ = [
    "WINDOW_SORT_KEYS",
    "CalibrationPreprocessError",
    "calibration_frame",
    "clamped_count",
    "expit",
    "inner_split_date",
    "logit",
    "logits_of",
    "split_calibration_window",
]
