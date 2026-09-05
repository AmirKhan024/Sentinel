"""Fitting one booster to one fold's training window. Pure -- no filesystem, no clock.

Three decisions carry this module's integrity.

**What "train" means is not redefined here.** The training frame comes from
``modeling.train.training_frame``, which delegates to ``evaluation.folds.assign_split``.
There is therefore exactly one definition of the training window in the repository --
the one Component 5's ``future_rows_never_enter_training`` check independently
re-derives -- and Component 7 inherits it rather than reimplementing it.

**The final fit does no early stopping.** This is the difference that lets
``trained_through`` stay honest. Early stopping needs a validation window later than the
training data, and the only such windows available at fold-fit time are the fold's own
calibration and test windows. Reading either would mean the model had learned from a
date later than ``train_end``, which is precisely what ``trained_through`` denies. So
the boosting-round count is fixed in advance, taken from a search that ran on a region
strictly earlier than any test window (see ``tuning`` and ADR 0017), and the fit here
runs exactly that many rounds and stops. Nothing later than ``fold.train_end`` is read.

**Row order is load-bearing.** Component 6 measured coefficients moving by 7.049e-09
under a re-ordering of the same rows, because float summation is not associative.
Boosters have the same exposure through their histogram construction, and additionally
draw row and column subsamples in row order. So training rows are canonically sorted
before fitting, using Component 6's own sort keys. That sort, together with the pinned
single thread, is what makes a re-run reproducible.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from sentinel.boosting import preprocess
from sentinel.boosting.definitions import (
    BoostingSpec,
    Estimator,
    estimator_params,
    n_estimators_of,
)
from sentinel.boosting.models import FittedBooster
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling.train import TRAIN_SORT_KEYS, training_frame

logger = logging.getLogger(__name__)


class BoostingTrainError(RuntimeError):
    """Raised when a fold cannot be fitted."""


def build_estimator(spec: BoostingSpec, params: dict[str, object]) -> Any:
    """Construct an unfitted estimator. Returned as ``Any`` -- see ``models`` docstring.

    Isolated from :func:`fit_fold` so a test can construct an estimator without fitting
    one, and so the two libraries are dispatched in exactly one place.
    """
    # Splatted through an ``Any`` because both constructors declare narrow per-parameter
    # types that a ``dict[str, object]`` cannot satisfy positionally. The dictionary's
    # contents are guarded at import by ``_guard_registry`` instead, which is the check
    # that actually matters: it is the *set* of keys that must be right, not their
    # inferred types at this call site.
    kwargs: Any = params
    if spec.estimator is Estimator.XGBOOST:
        from xgboost import XGBClassifier

        return XGBClassifier(**kwargs)
    if spec.estimator is Estimator.LIGHTGBM:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**kwargs)
    raise BoostingTrainError(f"{spec.name}: unsupported estimator {spec.estimator!r}")


def fit_fold(
    spec: BoostingSpec, train: pl.DataFrame, fold: FoldSpec, *, params_fold_set: str | None = None
) -> FittedBooster:
    """Fit one boosted model on one fold's training window.

    ``train`` must already be a training frame for ``fold`` -- pass the output of
    ``modeling.train.training_frame``. It is re-sorted here anyway, because the sort is
    a correctness requirement rather than a caller's courtesy.

    ``params_fold_set`` overrides which fold set's frozen tuned hyperparameters are used
    (``TUNED_PARAMS`` is keyed by fold set, per ADR 0017), while every other value fitted
    -- and every value recorded on the returned :class:`FittedBooster` -- still describes
    ``fold`` itself. Every evaluation call site omits it and gets ``fold.fold_set``,
    exactly as before this parameter existed. It exists for callers whose fold has no
    tuned study of its own to read: Component 18's operational ``FoldSpec`` uses
    ``fold_set="operational"``, which ``TUNED_PARAMS`` was never asked about and never
    will be, and the alternative -- inventing an operational tuning study, or silently
    mislabelling the fold as ``"quarterly"`` -- would be worse than naming the borrowed
    fold set explicitly here.
    """
    if train.height == 0:
        raise BoostingTrainError(
            f"{spec.name}: fold {fold.fold_id} has no training rows. A model is never "
            "fitted on an empty window, and a fold that produces one is a defect "
            "upstream rather than something to work around."
        )

    ordered = train.sort(list(TRAIN_SORT_KEYS))
    labels = fold_labels(ordered, spec.name, fold.fold_id)
    if len(set(labels.tolist())) < 2:
        raise BoostingTrainError(
            f"{spec.name}: fold {fold.fold_id} training window contains a single class "
            f"({int(labels[0])}). A fitted probability would be a constant, which is a "
            "reference schedule rather than a model."
        )

    resolved_params_fold_set = fold.fold_set if params_fold_set is None else params_fold_set
    params = estimator_params(spec, resolved_params_fold_set)
    weight = 1.0
    if spec.class_weighted:
        # From this fold's own training labels, never a global constant. The ablation is
        # only interpretable if the weight is what that window actually implies.
        weight = preprocess.positive_weight(labels)
        if spec.estimator is not Estimator.XGBOOST:
            raise BoostingTrainError(
                f"{spec.name}: class weighting is implemented for XGBoost only; "
                "LightGBM's is_unbalance and scale_pos_weight do not mean the same "
                "thing and silently substituting one would make the ablation "
                "uninterpretable"
            )
        params["scale_pos_weight"] = weight

    matrix = preprocess.tree_matrix(ordered, spec)
    columns = preprocess.matrix_columns(spec)
    if matrix.shape[1] != len(columns):
        raise BoostingTrainError(
            f"{spec.name}: matrix has {matrix.shape[1]} columns for {len(columns)} "
            "declared names. Every importance would be mislabelled."
        )

    estimator = build_estimator(spec, params)
    estimator.fit(matrix, labels)

    n_estimators = n_estimators_of(spec, resolved_params_fold_set)
    trees = _trees_built(spec, estimator)
    importances = _importances(estimator, len(columns), spec.name, fold.fold_id)

    fitted = FittedBooster(
        spec=spec,
        fold_set=fold.fold_set,
        fold_id=fold.fold_id,
        estimator=estimator,
        matrix_columns=columns,
        params=params,
        n_estimators=n_estimators,
        trees_built=trees,
        importances=importances,
        scale_pos_weight=weight,
        train_rows=ordered.height,
        train_positive_rate=_positive_rate(ordered),
        train_nan_cells=int(np.count_nonzero(np.isnan(matrix))),
        train_start=fold.train_start,
        train_end=fold.train_end,
        # The training end, not the calibration end. This fit performs no early stopping
        # and reads no window later than train_end; its round count came from a search
        # confined to an earlier region entirely. ADR 0014, ADR 0017.
        trained_through=fold.train_end,
        calibration_end_unused=fold.calibration_end,
    )

    logger.info(
        "Fitted %s on %s: %d rows, %d features, prevalence %.4f, %d trees",
        spec.name,
        fold.fold_id,
        fitted.train_rows,
        len(spec.feature_columns),
        fitted.train_positive_rate or 0.0,
        fitted.trees_built,
    )
    return fitted


def fold_labels(frame: pl.DataFrame, model_name: str, fold_id: str) -> NDArray[np.int64]:
    """The target column as an integer array, with the two ways it can be wrong refused.

    Shared with ``tuning`` so the search and the final fit read the label identically.
    """
    if "target" not in frame.columns:
        raise BoostingTrainError(f"{model_name}: fold {fold_id} frame has no target")
    values = frame["target"].to_list()
    if any(v is None for v in values):
        raise BoostingTrainError(
            f"{model_name}: fold {fold_id} window contains a null target. Component 3 "
            "emits a target only for eligible rows; a null here means an ineligible row "
            "reached the model."
        )
    return np.asarray(values, dtype=np.int64)


def _positive_rate(frame: pl.DataFrame) -> float | None:
    if frame.height == 0:
        return None
    mean = frame["target"].mean()
    # polars aggregates are typed as a wide union; the target is Int8 so the mean is a
    # float, but the narrowing has to be explicit under strict mode.
    return float(mean) if isinstance(mean, (int, float)) else None


def _trees_built(spec: BoostingSpec, estimator: Any) -> int:
    """How many trees the fitted ensemble actually contains.

    Recorded rather than assumed equal to ``n_estimators``: a booster can stop early on
    its own when a round produces no usable split, and a silent gap between the frozen
    round count and the built count would make two folds incomparable without saying so.
    """
    if spec.estimator is Estimator.XGBOOST:
        booster = estimator.get_booster()
        return len(booster.get_dump())
    return int(estimator.booster_.num_trees())


def _importances(estimator: Any, width: int, model_name: str, fold_id: str) -> tuple[float, ...]:
    """Native split importances, converted explicitly at the library boundary.

    These are a **diagnostic**, not an attribution and not a causal statement. Component
    6's findings measured a condition number of 71.8 and a feature pair correlated at
    0.9888, and a tree ensemble distributes credit between collinear features according
    to which one it happened to split on first. Component 11 owns attribution.
    """
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        raise BoostingTrainError(f"{model_name}: fold {fold_id} estimator exposed no importances")
    converted = tuple(float(v) for v in values)
    if len(converted) != width:
        raise BoostingTrainError(
            f"{model_name}: fold {fold_id} produced {len(converted)} importances for "
            f"{width} matrix columns. Every one would be mislabelled."
        )
    return converted


__all__ = [
    "BoostingTrainError",
    "build_estimator",
    "fit_fold",
    "fold_labels",
    "training_frame",
]
