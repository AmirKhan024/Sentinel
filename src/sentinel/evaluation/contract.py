"""The model-agnostic prediction contract.

This is the seam between Component 5 and every modelling component after it.
The evaluator does not know, and must not know, whether a score came from a
heuristic, a logistic regression, XGBoost, a neural network or an ensemble. It
knows only that something claims a risk ordering over one fold's test window,
and it checks that claim before scoring it.

Two of these checks are about arithmetic -- right columns, no nulls, unique ids.
The other three are about honesty:

**Exact coverage.** The score set must cover the fold's test window exactly:
no row missing, no row extra. A model that quietly drops the establishments it
finds hard would post a better precision@k for a reason that has nothing to do
with being better.

**No imputation.** A missing score is a rejection, never a zero. Filling a gap
would silently place that establishment at the bottom of the queue and call it
a prediction.

**Declared training horizon.** ``trained_through`` is the last reference date
the producer was allowed to learn from, and it must not run past the fold's
calibration end. This is the check that makes retrospective cheating hard to do
by accident: a Component 6 model trained on everything through 2025 and pointed
at the 2023 test window is rejected at the door rather than quietly producing an
excellent, meaningless number.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import polars as pl

from sentinel.evaluation.models import FoldSpec, PredictionSet

logger = logging.getLogger(__name__)

#: The only two columns a prediction artifact may carry. Anything else -- a
#: label, a feature, a date -- is refused, so a model cannot smuggle the answer
#: back in alongside its guess.
PREDICTION_COLUMNS: tuple[str, ...] = ("target_inspection_id", "score")

#: Columns a persisted prediction file carries in addition, identifying which
#: model produced it and for which fold. Held in the file so an artifact on
#: disk is self-describing.
PREDICTION_METADATA_COLUMNS: tuple[str, ...] = (
    "model_name",
    "model_version",
    "fold_id",
)


class PredictionContractError(ValueError):
    """Raised when a prediction artifact cannot be trusted enough to score."""


class PredictionHorizonError(PredictionContractError):
    """Raised when a producer declares a training horizon past the fold's.

    A subclass, so every ``except PredictionContractError`` keeps catching it, while a
    caller that wants to distinguish "this model saw the future" from "this model
    miscounted its rows" can. The distinction matters because the two failures have
    different remedies and the validator reports them as different checks.
    """


def validate_predictions(
    predictions: PredictionSet,
    fold: FoldSpec,
    expected_ids: Sequence[str],
) -> None:
    """Reject a prediction set that would produce a dishonest number.

    Raises rather than returns, because every failure here is a defect in the
    producer that a warning would let through into a results table.
    """
    frame = predictions.frame

    if predictions.fold_id != fold.fold_id:
        raise PredictionContractError(
            f"predictions declare fold {predictions.fold_id!r} but were offered "
            f"fold {fold.fold_id!r}"
        )

    missing_columns = [c for c in PREDICTION_COLUMNS if c not in frame.columns]
    if missing_columns:
        raise PredictionContractError(
            f"{predictions.model_name}: missing required column(s) {', '.join(missing_columns)}"
        )
    extra_columns = [c for c in frame.columns if c not in PREDICTION_COLUMNS]
    if extra_columns:
        raise PredictionContractError(
            f"{predictions.model_name}: unexpected column(s) {', '.join(extra_columns)}; "
            "a prediction artifact carries an id and a score, nothing else"
        )

    if frame["target_inspection_id"].null_count():
        raise PredictionContractError(f"{predictions.model_name}: null target_inspection_id")
    if frame["score"].null_count():
        raise PredictionContractError(
            f"{predictions.model_name}: null score. The evaluator never imputes a missing "
            "score -- a gap would silently rank that establishment last."
        )

    scores = frame["score"].to_list()
    bad = [s for s in scores if s is None or math.isnan(s) or math.isinf(s)]
    if bad:
        raise PredictionContractError(
            f"{predictions.model_name}: {len(bad)} score(s) are NaN or infinite"
        )

    ids = frame["target_inspection_id"].to_list()
    if len(set(ids)) != len(ids):
        raise PredictionContractError(
            f"{predictions.model_name}: duplicate target_inspection_id in predictions"
        )

    expected = set(expected_ids)
    got = set(ids)
    if got != expected:
        absent = len(expected - got)
        surplus = len(got - expected)
        raise PredictionContractError(
            f"{predictions.model_name}: prediction set does not cover fold {fold.fold_id} "
            f"exactly -- {absent} test row(s) unscored, {surplus} row(s) not in the test window"
        )

    if predictions.trained_through is not None:
        horizon = _training_horizon(fold)
        if predictions.trained_through > horizon:
            raise PredictionHorizonError(
                f"{predictions.model_name}: declares trained_through="
                f"{predictions.trained_through}, which is after fold {fold.fold_id}'s "
                f"calibration end {horizon}. Scoring this would evaluate a model on a "
                "period it had already seen."
            )


def _training_horizon(fold: FoldSpec) -> date:
    """The last date a producer may legitimately have learned from for this fold.

    The calibration end, not the training end: fitting a calibrator on the
    calibration window is exactly what that window exists for. Anything past it
    is the test period.
    """
    return fold.calibration_end


def prediction_frame(ids: Sequence[str], scores: Sequence[float]) -> pl.DataFrame:
    """Assemble a contract-shaped frame from aligned ids and scores."""
    if len(ids) != len(scores):
        raise PredictionContractError(
            f"ids ({len(ids)}) and scores ({len(scores)}) differ in length"
        )
    return pl.DataFrame(
        {
            "target_inspection_id": pl.Series(list(ids), dtype=pl.Utf8),
            "score": pl.Series([float(s) for s in scores], dtype=pl.Float64),
        }
    )


def read_predictions(path: Path, *, fold_id: str | None = None) -> list[PredictionSet]:
    """Load prediction artifacts written by a later component.

    A file may hold several models and several folds; it is split into one
    ``PredictionSet`` per ``(model_name, model_version, fold_id)`` so each is
    validated against its own fold. Nothing is validated here -- that happens
    when a set is offered against a specific fold.
    """
    if not path.exists():
        raise FileNotFoundError(f"Prediction artifact not found: {path}")
    frame = pl.read_parquet(path)

    required = (*PREDICTION_COLUMNS, *PREDICTION_METADATA_COLUMNS)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise PredictionContractError(
            f"{path.name}: persisted predictions must carry {', '.join(required)}; "
            f"missing {', '.join(missing)}"
        )
    if fold_id is not None:
        frame = frame.filter(pl.col("fold_id") == fold_id)

    sets: list[PredictionSet] = []
    keys = (
        frame.select(["model_name", "model_version", "fold_id"])
        .unique()
        .sort(["model_name", "model_version", "fold_id"])
    )
    for row in keys.iter_rows(named=True):
        subset = frame.filter(
            (pl.col("model_name") == row["model_name"])
            & (pl.col("model_version") == row["model_version"])
            & (pl.col("fold_id") == row["fold_id"])
        )
        trained_through = None
        if "trained_through" in subset.columns:
            raw = subset["trained_through"].drop_nulls()
            if raw.len():
                value = raw[0]
                trained_through = value if isinstance(value, date) else date.fromisoformat(value)
        sets.append(
            PredictionSet(
                model_name=row["model_name"],
                model_version=row["model_version"],
                fold_id=row["fold_id"],
                frame=subset.select(list(PREDICTION_COLUMNS)),
                is_probability=bool(subset["is_probability"][0])
                if "is_probability" in subset.columns
                else False,
                trained_through=trained_through,
            )
        )
    logger.info("Read %d prediction set(s) from %s", len(sets), path.name)
    return sets


__all__ = [
    "PREDICTION_COLUMNS",
    "PREDICTION_METADATA_COLUMNS",
    "PredictionContractError",
    "PredictionHorizonError",
    "prediction_frame",
    "read_predictions",
    "validate_predictions",
]
