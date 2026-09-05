"""Scoring a fitted network over one fold's test window.

The score is ``sigmoid(logit)`` = ``P(target = 1)``, so **higher means higher predicted
risk** of a Priority or Priority Foundation citation. Component 5 assumes that direction
(``SCORE_DIRECTION``, plus a live two-row probe in its validator), and inverting it here
would produce a plausible, confidently wrong result rather than an error.

The sigmoid is applied here and nowhere else. The network emits a logit, the loss consumes
a logit, and the one place a probability is required is the artifact Component 5 reads.

The test frame must be ``folds.window_frame(frame, fold)`` output, for the same reason
Components 6 and 7 require it: ``window_frame`` sorts by ``(rd, target_inspection_id)``,
which is the canonical order Component 5 builds its window in, so scores returned
positionally line up with the ids the evaluator expects. Ids are returned alongside the
scores so the caller never reconstructs the order.

Nothing here touches the calibration window, and nothing here fits anything. The
preprocessor and the vocabularies arrive already fitted, from ``fit_fold``, and are
*applied* -- there is deliberately no code path in this module that could fit a statistic
on a test window.

The probabilities are **uncalibrated**. A network trained under BCE is typically
overconfident, and Component 8 measures that through Component 5's ECE and MCE and
corrects none of it. Component 9 owns calibration.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import polars as pl
import torch

from sentinel.neural import preprocess, train
from sentinel.neural.definitions import CategoricalEncoding
from sentinel.neural.models import FittedNetwork

logger = logging.getLogger(__name__)


class NeuralPredictError(ValueError):
    """Raised when a fitted network cannot score a window."""


def score_window(
    fitted: FittedNetwork,
    test: pl.DataFrame,
    *,
    categoricals: pl.DataFrame | None = None,
) -> tuple[list[str], list[float]]:
    """Score one test window. Returns ids and scores, aligned positionally."""
    if test.height == 0:
        raise NeuralPredictError(
            f"{fitted.spec.name}: fold {fitted.fold_id} test window is empty. Component 5 "
            "skips an empty window; a model should never be asked to score one."
        )
    if "target_inspection_id" not in test.columns:
        raise NeuralPredictError(
            f"{fitted.spec.name}: test frame has no target_inspection_id, so scores could "
            "not be attributed to a row."
        )

    model, preprocessor = train.scorer_for(fitted)
    frame = _with_categoricals(fitted, test, categoricals)

    dense = preprocess.dense_matrix(frame, fitted.spec, preprocessor, fitted.encoding)
    codes = preprocess.code_matrix(frame, fitted.spec, fitted.encoding)

    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(np.ascontiguousarray(dense, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(codes, dtype=np.int64)),
        )
        probabilities = torch.sigmoid(logits).numpy()

    scores = [float(v) for v in probabilities]
    ids = [str(v) for v in frame["target_inspection_id"].to_list()]

    if len(ids) != len(scores):
        raise NeuralPredictError(
            f"{fitted.spec.name}: {len(ids)} ids but {len(scores)} scores for fold {fitted.fold_id}"
        )
    bad = [s for s in scores if not math.isfinite(s)]
    if bad:
        raise NeuralPredictError(
            f"{fitted.spec.name}: {len(bad)} non-finite score(s) for fold "
            f"{fitted.fold_id}. The evaluator rejects these rather than imputing them, so "
            "failing here gives a better message."
        )
    outside = [s for s in scores if s < 0.0 or s > 1.0]
    if outside:
        raise NeuralPredictError(
            f"{fitted.spec.name}: {len(outside)} score(s) outside [0, 1] for fold "
            f"{fitted.fold_id}, which cannot be a probability"
        )
    return ids, scores


def _with_categoricals(
    fitted: FittedNetwork, test: pl.DataFrame, categoricals: pl.DataFrame | None
) -> pl.DataFrame:
    """Attach categorical columns to a test window without disturbing its order."""
    if fitted.spec.encoding is CategoricalEncoding.NONE:
        return test
    if categoricals is None:
        raise NeuralPredictError(
            f"{fitted.spec.name}: scoring requires the categorical table its vocabularies "
            "were fitted against"
        )
    wanted = [
        c
        for c in ("chain_key", "facility_type", "community_area", "zip", "establishment_id")
        if c in categoricals.columns
    ]
    joined = test.join(
        categoricals.select("target_inspection_id", *wanted),
        on="target_inspection_id",
        how="left",
        suffix="_cat",
    )
    if joined.height != test.height:
        raise NeuralPredictError(
            f"{fitted.spec.name}: categorical join changed the test row count from "
            f"{test.height} to {joined.height}"
        )
    return joined


def saturated_count(scores: list[float]) -> int:
    """How many scores sit exactly at 0.0 or 1.0.

    Not an error: saturation is harmless for ranking, and ``log_loss`` clamps. Reported as
    a warning-severity observation, and it is worth more attention here than in Component
    6 -- a sigmoid over an unbounded logit saturates far more readily than a penalised
    GLM's, and a saturated score is one a calibrator cannot recover.
    """
    return sum(1 for s in scores if s in (0.0, 1.0))


__all__ = ["NeuralPredictError", "saturated_count", "score_window"]
