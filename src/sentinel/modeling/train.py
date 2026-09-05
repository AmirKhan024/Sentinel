"""Fitting one model to one fold's training window. Pure -- no filesystem, no clock.

Two decisions in this module carry the component's integrity.

**What "train" means.** The training frame is obtained from
``evaluation.folds.assign_split`` and filtered to ``split == "train"``, not from a
hand-rolled ``rd <= train_end`` predicate. There is then exactly one definition of the
training window in the repository -- the one Component 5's
``future_rows_never_enter_training`` check independently re-derives. A hand-rolled
filter would also silently omit the ``train_start`` anchor, which is harmless on this
data and would stop being harmless the moment the anchor moved.

**Row order is load-bearing.** Fitting the same 23,346 rows in a different order
produces coefficients differing by up to 7.049e-09: ``StandardScaler`` accumulates
variance incrementally and the lbfgs gradient is a BLAS reduction, and both depend on
float summation order. So training rows are sorted canonically before fitting, and that
sort -- not ``random_state``, which ``lbfgs`` ignores entirely -- is what makes a re-run
reproducible.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import polars as pl
from numpy.typing import NDArray

from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling import preprocess
from sentinel.modeling.definitions import ModelSpec, max_iter_of
from sentinel.modeling.models import FittedModel

logger = logging.getLogger(__name__)

#: Canonical training order. Matches the order Component 5 sorts a test window into, so
#: the whole component agrees on one row ordering.
TRAIN_SORT_KEYS: tuple[str, ...] = ("inspection_date", "target_inspection_id")


class TrainingError(RuntimeError):
    """Raised when a fold cannot be fitted, or was fitted without converging."""


def training_frame(frame: pl.DataFrame, fold: FoldSpec) -> pl.DataFrame:
    """The fold's training rows, canonically ordered.

    Calibration and test rows are excluded by construction, not by subtraction: only
    rows whose split is ``train`` survive.
    """
    assigned = folds_module.assign_split(frame, fold)
    return assigned.filter(pl.col("split") == "train").sort(list(TRAIN_SORT_KEYS))


def fit_fold(spec: ModelSpec, train: pl.DataFrame, fold: FoldSpec) -> FittedModel:
    """Fit one model on one fold's training window.

    ``train`` must already be a training frame for ``fold`` -- pass the output of
    :func:`training_frame`. It is re-sorted here anyway, because the sort is a
    correctness requirement rather than a caller's courtesy.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    if train.height == 0:
        raise TrainingError(
            f"{spec.name}: fold {fold.fold_id} has no training rows. A model is never "
            "fitted on an empty window, and a fold that produces one is a defect "
            "upstream rather than something to work around."
        )

    ordered = train.sort(list(TRAIN_SORT_KEYS))
    labels = _labels(ordered, spec, fold)
    if len(set(labels.tolist())) < 2:
        raise TrainingError(
            f"{spec.name}: fold {fold.fold_id} training window contains a single class "
            f"({int(labels[0])}). A fitted probability would be a constant, which is a "
            "reference schedule rather than a model."
        )

    matrix = preprocess.to_matrix(ordered, spec)
    pipeline = Pipeline(
        [
            ("preprocess", preprocess.build_preprocessor(spec)),
            ("model", LogisticRegression(**dict(spec.params), random_state=spec.seed)),
        ]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(matrix, labels)
    saw_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)

    estimator = pipeline.named_steps["model"]
    scaler = pipeline.named_steps["preprocess"].named_steps["scale"]
    n_iter = int(estimator.n_iter_[0])
    max_iter = max_iter_of(spec)
    converged = n_iter < max_iter and not saw_warning

    fitted = FittedModel(
        spec=spec,
        fold_set=fold.fold_set,
        fold_id=fold.fold_id,
        pipeline=pipeline,
        matrix_columns=preprocess.ordered_matrix_columns(spec),
        coefficients=tuple(float(v) for v in estimator.coef_[0]),
        intercept=float(estimator.intercept_[0]),
        scaler_mean=tuple(float(v) for v in scaler.mean_),
        scaler_scale=tuple(float(v) for v in scaler.scale_),
        imputed_values=preprocess.imputed_values(pipeline.named_steps["preprocess"], spec),
        n_iter=n_iter,
        converged=converged,
        train_rows=ordered.height,
        train_positive_rate=_positive_rate(ordered),
        train_start=fold.train_start,
        train_end=fold.train_end,
        # The training end, not the calibration end. Component 6 fits no calibrator and
        # never reads the calibration window, so declaring its end would claim a horizon
        # this model did not use. ADR 0014.
        trained_through=fold.train_end,
        calibration_end_unused=fold.calibration_end,
    )

    if len(fitted.coefficients) != len(fitted.matrix_columns):
        raise TrainingError(
            f"{spec.name}: fold {fold.fold_id} produced {len(fitted.coefficients)} "
            f"coefficients for {len(fitted.matrix_columns)} matrix columns. Every "
            "coefficient would be mislabelled."
        )
    if not converged:
        raise TrainingError(
            f"{spec.name}: fold {fold.fold_id} did not converge ({n_iter} of "
            f"{max_iter} iterations, ConvergenceWarning={saw_warning}). Coefficients "
            "that depend on the iteration cap are not comparable across folds."
        )

    logger.info(
        "Fitted %s on %s: %d rows, %d features, prevalence %.4f, %d iterations",
        spec.name,
        fold.fold_id,
        fitted.train_rows,
        len(spec.feature_columns),
        fitted.train_positive_rate or 0.0,
        n_iter,
    )
    return fitted


def _labels(frame: pl.DataFrame, spec: ModelSpec, fold: FoldSpec) -> NDArray[np.int64]:
    if "target" not in frame.columns:
        raise TrainingError(f"{spec.name}: fold {fold.fold_id} training frame has no target")
    values = frame["target"].to_list()
    if any(v is None for v in values):
        raise TrainingError(
            f"{spec.name}: fold {fold.fold_id} training window contains a null target. "
            "Component 3 emits a target only for eligible rows; a null here means an "
            "ineligible row reached the model."
        )
    return np.asarray(values, dtype=np.int64)


def _positive_rate(frame: pl.DataFrame) -> float | None:
    if frame.height == 0:
        return None
    mean = frame["target"].mean()
    # polars aggregates are typed as a wide union; the target is Int8 so the mean is a
    # float, but the narrowing has to be explicit under strict mode.
    return float(mean) if isinstance(mean, (int, float)) else None


__all__ = ["TRAIN_SORT_KEYS", "TrainingError", "fit_fold", "training_frame"]
