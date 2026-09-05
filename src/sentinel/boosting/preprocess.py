"""The tree matrix: the same 30 columns Component 6 builds, and nothing done to them.

This module is short on purpose, and the shortness is the design.

Component 6's preprocessor is a two-stage ``Pipeline``: a ``ColumnTransformer`` that
imputes each column by its declared rule, then a ``StandardScaler``. A gradient-boosted
tree needs neither stage, and applying them would actively cost accuracy and honesty:

**No imputation.** Both libraries route NaN to a learned default direction at every
split, so a NULL is a value the model can branch on. Component 4's NULLs are not missing
measurements -- ``prior_canvass_fail_rate`` is NULL because there was no prior canvass,
which is a fact about the establishment. Filling that with the training-window median
would replace a fact with the average of a population the establishment is not in. The
logistic model had no choice; a booster does, and takes it.

**No scaling.** A tree splits on thresholds, so it is invariant to any monotone
transform of a feature. Standardising would change nothing about the fitted model and
would add a fitted statistic that has to be carried, validated and explained.

**The four family indicators stay anyway.** A NaN-native learner does not need them --
it can already distinguish NULL from any real value. They are kept because Component 6
has them, and dropping them would mean the two model classes see different matrices,
which would make every C6-versus-C7 comparison ambiguous between "the estimator is
better" and "the matrix is different". They are declared columns, not observed ones, so
keeping them costs four columns and buys a clean comparison.

The consequence for validation: there are no fitted preprocessing statistics, so
Component 6's ``preprocessing_comes_from_train`` check has no analogue here. It is
replaced by a stronger one -- ``validate`` re-derives the NULL mask from the source frame
and asserts it equals the NaN mask of the matrix the estimator actually received. That
catches an accidentally reintroduced imputation, which is the failure this module exists
to prevent.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from numpy.typing import NDArray

from sentinel.boosting.definitions import BoostingSpec
from sentinel.modeling import preprocess as baseline_preprocess
from sentinel.modeling.definitions import ModelSpec, indicator_columns

logger = logging.getLogger(__name__)


class TreePreprocessError(ValueError):
    """Raised when a tree matrix cannot be built from a frame."""


def _as_model_spec(spec: BoostingSpec) -> ModelSpec:
    """Adapt a :class:`BoostingSpec` to the shape ``modeling.preprocess`` reads.

    ``modeling.preprocess`` reads exactly two attributes -- ``name`` and
    ``feature_columns`` -- and reusing it is what guarantees the boosted matrix has the
    same columns in the same order as the baseline's. Rebuilding the column list here
    would be a second definition of the matrix, and the two would drift.
    """
    return ModelSpec(
        name=spec.name,
        version=spec.version,
        description=spec.description,
        feature_columns=spec.feature_columns,
        is_probability=spec.is_probability,
        seed=spec.seed,
        params={},
    )


def matrix_columns(spec: BoostingSpec) -> tuple[str, ...]:
    """The matrix's columns in order: the spec's features, then the four indicators.

    Identical to Component 6's, deliberately. Note this is ``modeling.preprocess``'s
    ``matrix_columns`` and *not* its ``ordered_matrix_columns``: the latter reorders
    columns to match the ``ColumnTransformer``'s branch order, which exists only because
    a ``ColumnTransformer`` emits its branches in sequence. There is no transformer here,
    so the natural order survives and the matrix reads the way the spec is written.
    """
    return baseline_preprocess.matrix_columns(_as_model_spec(spec))


def tree_matrix(frame: pl.DataFrame, spec: BoostingSpec) -> NDArray[np.float64]:
    """Build the float matrix one booster sees over one window.

    NULLs arrive as NaN and stay NaN. Booleans are cast to float in the one place
    Component 6 already casts them, so there is a single representation change in the
    repository and a test can see it.
    """
    try:
        return baseline_preprocess.to_matrix(frame, _as_model_spec(spec))
    except baseline_preprocess.PreprocessError as exc:
        raise TreePreprocessError(str(exc)) from exc


def null_mask(frame: pl.DataFrame, spec: BoostingSpec) -> NDArray[np.bool_]:
    """The NULL mask the matrix *should* have, in matrix column order.

    Computed straight from the source frame's own null flags rather than by inspecting
    the matrix, so ``validate`` compares two independently derived answers instead of
    comparing the matrix with itself. The four indicators are all-False by construction:
    an indicator is computed from another column's null mask and is therefore never null
    itself, which is a property worth asserting rather than assuming.
    """
    indicators = set(indicator_columns())
    height = frame.height
    stacked: list[NDArray[np.bool_]] = []
    for name in matrix_columns(spec):
        if name in indicators:
            stacked.append(np.zeros(height, dtype=np.bool_))
            continue
        if name not in frame.columns:
            raise TreePreprocessError(f"{spec.name}: frame is missing matrix column {name}")
        stacked.append(frame[name].is_null().to_numpy().astype(np.bool_, copy=False))
    return np.column_stack(stacked)


def positive_weight(labels: NDArray[np.int64]) -> float:
    """``scale_pos_weight`` for the class-weighting ablation: negatives over positives.

    Computed from the training window's own labels, never from a global constant, so the
    ablation weights by what that fold actually saw. Returns 1.0 for a degenerate window
    rather than dividing by zero -- ``train`` refuses a single-class window before this
    is reached, so the guard is a belt on top of a brace.
    """
    positives = int(np.count_nonzero(labels == 1))
    negatives = int(np.count_nonzero(labels == 0))
    if positives == 0 or negatives == 0:
        return 1.0
    return negatives / positives


__all__ = [
    "TreePreprocessError",
    "matrix_columns",
    "null_mask",
    "positive_weight",
    "tree_matrix",
]
