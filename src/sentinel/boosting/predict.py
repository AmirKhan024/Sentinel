"""Scoring a fitted booster over one fold's test window.

The score is ``P(target = 1)`` from ``predict_proba``, so **higher means higher predicted
risk** of a Priority or Priority Foundation citation. Component 5 assumes that direction
(``SCORE_DIRECTION``, plus a live two-row probe in its validator), and inverting it here
would produce a plausible, confidently wrong result rather than an error.

The test frame must be ``folds.window_frame(frame, fold)`` output, whose sort is the
canonical order Component 5 builds its window in. Ids are returned alongside the scores
so the caller pairs each score with its own id rather than reconstructing the order.

These probabilities are **raw**, and the word matters. A boosted ensemble's
``predict_proba`` is typically *worse* calibrated than a penalised GLM's -- boosting
drives the margin, not the likelihood, and subsampling adds variance the logistic model
does not have. Nothing here corrects that, and no document produced by this component
may call these calibrated. Component 9 owns calibration.
"""

from __future__ import annotations

import logging
import math

import polars as pl

from sentinel.boosting import preprocess
from sentinel.boosting.models import FittedBooster

logger = logging.getLogger(__name__)


class BoostingPredictError(ValueError):
    """Raised when a fitted booster cannot score a window."""


def score_window(fitted: FittedBooster, test: pl.DataFrame) -> tuple[list[str], list[float]]:
    """Score one test window. Returns ids and scores, aligned positionally."""
    if test.height == 0:
        raise BoostingPredictError(
            f"{fitted.spec.name}: fold {fitted.fold_id} test window is empty. Component "
            "5 skips an empty window; a model should never be asked to score one."
        )
    if "target_inspection_id" not in test.columns:
        raise BoostingPredictError(
            f"{fitted.spec.name}: test frame has no target_inspection_id, so scores "
            "could not be attributed to a row."
        )

    matrix = preprocess.tree_matrix(test, fitted.spec)
    proba = fitted.estimator.predict_proba(matrix)
    # Explicit conversion at the library boundary: strict mypy would otherwise let an
    # ``Any`` escape into a declared ``list[float]``. Column 1 is P(class 1) because
    # ``classes_`` is sorted and the target is 0/1.
    scores = [float(v) for v in proba[:, 1]]
    ids = [str(v) for v in test["target_inspection_id"].to_list()]

    if len(ids) != len(scores):
        raise BoostingPredictError(
            f"{fitted.spec.name}: {len(ids)} ids but {len(scores)} scores for fold {fitted.fold_id}"
        )
    bad = [s for s in scores if not math.isfinite(s)]
    if bad:
        raise BoostingPredictError(
            f"{fitted.spec.name}: {len(bad)} non-finite score(s) for fold "
            f"{fitted.fold_id}. The evaluator rejects these rather than imputing them, "
            "so failing here gives a better message."
        )
    outside = [s for s in scores if s < 0.0 or s > 1.0]
    if outside:
        raise BoostingPredictError(
            f"{fitted.spec.name}: {len(outside)} score(s) outside [0, 1] for fold "
            f"{fitted.fold_id}, which cannot be a probability"
        )
    return ids, scores


def saturated_count(scores: list[float]) -> int:
    """How many scores sit exactly at 0.0 or 1.0.

    Not an error: saturation is harmless for ranking, and ``log_loss`` clamps. Worth
    reporting because a boosted ensemble saturates far more readily than a penalised
    GLM, and a saturated score is one Component 9 will have the least room to calibrate.
    """
    return sum(1 for s in scores if s in (0.0, 1.0))


__all__ = ["BoostingPredictError", "saturated_count", "score_window"]
