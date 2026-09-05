"""Scoring a fitted model over one fold's test window.

The score is ``P(target = 1)`` from ``predict_proba``, so **higher means higher
predicted risk** of a Priority or Priority Foundation citation. Component 5 assumes that
direction (``SCORE_DIRECTION``, and a live two-row probe in its validator), and
inverting it here would produce a plausible, confidently wrong result rather than an
error.

The test frame must be ``folds.window_frame(frame, fold)`` output. That is not a
convenience: ``window_frame`` sorts by ``(rd, target_inspection_id)``, which is the
canonical order Component 5 builds its window in, so scores returned positionally line
up with the ids the evaluator expects. Aligning by anything other than the id column
would be a latent mis-join, so the ids are returned alongside the scores and the caller
never has to reconstruct them.

Nothing here touches the calibration window. The probabilities are **uncalibrated** and
Component 9 owns calibration; nothing in this component may describe them otherwise.
"""

from __future__ import annotations

import logging
import math

import polars as pl

from sentinel.modeling import preprocess
from sentinel.modeling.models import FittedModel

logger = logging.getLogger(__name__)


class PredictError(ValueError):
    """Raised when a fitted model cannot score a window."""


def score_window(fitted: FittedModel, test: pl.DataFrame) -> tuple[list[str], list[float]]:
    """Score one test window. Returns ids and scores, aligned positionally.

    Both lists are returned so the caller pairs an id with its own score rather than
    re-deriving the order.
    """
    if test.height == 0:
        raise PredictError(
            f"{fitted.spec.name}: fold {fitted.fold_id} test window is empty. Component "
            "5 skips an empty window; a model should never be asked to score one."
        )
    if "target_inspection_id" not in test.columns:
        raise PredictError(
            f"{fitted.spec.name}: test frame has no target_inspection_id, so scores "
            "could not be attributed to a row."
        )

    matrix = preprocess.to_matrix(test, fitted.spec)
    proba = fitted.pipeline.predict_proba(matrix)
    # Explicit conversion at the sklearn boundary: strict mypy would otherwise let an
    # ``Any`` escape into a declared ``list[float]``. Column 1 is P(class 1) because
    # ``classes_`` is sorted and the target is 0/1.
    scores = [float(v) for v in proba[:, 1]]
    ids = [str(v) for v in test["target_inspection_id"].to_list()]

    if len(ids) != len(scores):
        raise PredictError(
            f"{fitted.spec.name}: {len(ids)} ids but {len(scores)} scores for fold {fitted.fold_id}"
        )
    bad = [s for s in scores if not math.isfinite(s)]
    if bad:
        raise PredictError(
            f"{fitted.spec.name}: {len(bad)} non-finite score(s) for fold "
            f"{fitted.fold_id}. The evaluator rejects these rather than imputing them, "
            "so failing here gives a better message."
        )
    outside = [s for s in scores if s < 0.0 or s > 1.0]
    if outside:
        raise PredictError(
            f"{fitted.spec.name}: {len(outside)} score(s) outside [0, 1] for fold "
            f"{fitted.fold_id}, which cannot be a probability"
        )
    return ids, scores


def saturated_count(scores: list[float]) -> int:
    """How many scores sit exactly at 0.0 or 1.0.

    Not an error: saturation is harmless for ranking, and ``log_loss`` clamps. Reported
    as a warning-severity observation because it means the linear predictor is extreme
    enough that the probability lost resolution, which is worth seeing.
    """
    return sum(1 for s in scores if s in (0.0, 1.0))


__all__ = ["PredictError", "saturated_count", "score_window"]
