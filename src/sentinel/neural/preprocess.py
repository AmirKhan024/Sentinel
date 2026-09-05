"""The network's input matrix: Component 6's preprocessing, plus a categorical block.

The numeric half is **not reimplemented here**. It calls
``modeling.preprocess.build_preprocessor`` through the same ``_as_model_spec`` adapter
Component 7 uses, which is what guarantees the network sees the same 30 columns, in the
same order, imputed by the same rules and standardised by statistics fitted on the same
rows. Rebuilding any of that here would be a third definition of the matrix, and three
definitions drift faster than two.

Why a network needs the scaling Component 7 skipped: a tree splits on thresholds and is
invariant to any monotone transform, so scaling would have changed nothing for a booster.
A network is not. Its first layer computes a weighted sum, and a feature measured in
thousands of days would dominate the gradient of one measured as a rate in [0, 1]
regardless of which carries more signal. ``StandardScaler`` is what stops the optimiser
from spending its early epochs undoing the units.

**Missingness semantics are inherited, not redesigned.** The project specification warns
against blindly copying the logistic model's preprocessing, and against silently changing
what a NULL means. Those pull in opposite directions here, and the resolution is:

* The *encoding* is copied deliberately, not blindly. Component 6's rules -- train-window
  median for a nullable numeric, constant ``0.0`` for a nullable boolean, passthrough for
  a never-null column -- carry a measured justification (a boolean median sits 0.0056
  from flipping across folds), and that justification is about the data, not about
  logistic regression. It applies unchanged to a network.
* The *information* is preserved by the four null-rule family indicators, which survive
  into the network exactly as they survive into the booster. A network can learn an
  interaction between an indicator and its imputed column, which is the thing Component
  6's findings flagged as an open question and could not do itself.
* What is **not** done is Component 7's NaN-native routing. There is no such thing for a
  dense layer: a NaN propagates through every weight and destroys the fit. So the network
  imputes, and the indicator is how the fact survives. That is a real difference between
  C7 and C8, it is stated in the manifest, and it is one reason a booster may be better
  suited to this data.

The categorical block is appended *after* the scaled numerics and is never scaled:
embedding rows are learned parameters, and one-hot indicators are already on [0, 1].
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from sentinel.modeling import preprocess as baseline_preprocess
from sentinel.modeling.definitions import ModelSpec
from sentinel.neural import encode
from sentinel.neural.definitions import CategoricalEncoding, NeuralSpec
from sentinel.neural.encode import FoldEncoding

logger = logging.getLogger(__name__)


class NeuralPreprocessError(ValueError):
    """Raised when a frame cannot be turned into a network input matrix."""


def _as_model_spec(spec: NeuralSpec) -> ModelSpec:
    """Adapt a :class:`NeuralSpec` to the shape ``modeling.preprocess`` reads.

    ``modeling.preprocess`` reads exactly two attributes -- ``name`` and
    ``feature_columns`` -- so the adapter is the whole of the reuse. Note it passes only
    the *numeric* half: ``entity_columns`` is deliberately not visible to Component 6's
    code, which has no concept of a categorical and must not acquire one.
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


def matrix_columns(spec: NeuralSpec) -> tuple[str, ...]:
    """The numeric matrix's columns in order: features, then the four indicators.

    ``matrix_columns`` and not ``ordered_matrix_columns``: the latter permutes columns to
    match a ``ColumnTransformer``'s branch order. That permutation *does* apply here,
    because this component uses the same transformer -- but it applies to the fitted
    output, and :func:`transformed_columns` is where it is named. This function is the
    input order.
    """
    return baseline_preprocess.matrix_columns(_as_model_spec(spec))


def transformed_columns(spec: NeuralSpec, encoding: FoldEncoding) -> tuple[str, ...]:
    """Every input column the network sees, in the order it sees it.

    Numeric columns in ``ColumnTransformer`` branch order, then -- for the one-hot control
    only -- the indicator columns. The embedding block has no column names here because
    its width is a learned representation rather than a set of features; ``embed`` names
    those dimensions when it writes them out.
    """
    numeric = baseline_preprocess.ordered_matrix_columns(_as_model_spec(spec))
    if spec.encoding is CategoricalEncoding.ONE_HOT:
        return (*numeric, *encode.one_hot_columns(spec, encoding))
    return numeric


def numeric_matrix(frame: pl.DataFrame, spec: NeuralSpec) -> NDArray[np.float64]:
    """The unimputed, unscaled float matrix. NULLs arrive as NaN."""
    try:
        return baseline_preprocess.to_matrix(frame, _as_model_spec(spec))
    except baseline_preprocess.PreprocessError as exc:
        raise NeuralPreprocessError(str(exc)) from exc


def build_preprocessor(spec: NeuralSpec) -> Any:
    """Imputation then standardisation, to be fitted on training rows only.

    Returned as ``Any`` for the same reason Component 6 returns one: it is a scikit-learn
    object and pretending otherwise would be a false annotation.
    """
    return baseline_preprocess.build_preprocessor(_as_model_spec(spec))


def apply_preprocessor(
    preprocessor: Any, frame: pl.DataFrame, spec: NeuralSpec
) -> NDArray[np.float64]:
    """Apply already-fitted statistics to any window.

    Deliberately separate from fitting, and never a ``fit_transform``. The one-call
    convenience is exactly the shape a leak takes: called on a test window it would fit a
    scaler on test rows and produce a plausible, better-looking, wrong number.
    """
    matrix = numeric_matrix(frame, spec)
    transformed = preprocessor.transform(matrix)
    out = np.asarray(transformed, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise NeuralPreprocessError(
            f"{spec.name}: preprocessed matrix contains non-finite values. A network "
            "cannot be fitted on NaN -- every downstream weight would become NaN in one "
            "backward pass -- so this fails here rather than producing a dead model."
        )
    return out


def imputed_values(preprocessor: Any, spec: NeuralSpec) -> dict[str, float]:
    """The fill value actually fitted for each nullable column.

    Extracted so ``validate`` can re-derive each median from the training frame and
    confirm it matches: the mechanical proof that preprocessing statistics came from the
    training window and nowhere else.
    """
    return baseline_preprocess.imputed_values(preprocessor, _as_model_spec(spec))


def scaler_statistics(preprocessor: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The fitted mean and scale vectors, as plain typed Python.

    Carried on ``FittedNetwork`` so a re-run can be checked without unpickling an
    estimator, and so ``validate`` can compare them against statistics it re-derives.
    """
    scaler = preprocessor.named_steps["scale"]
    return (
        tuple(float(v) for v in scaler.mean_),
        tuple(float(v) for v in scaler.scale_),
    )


def dense_matrix(
    frame: pl.DataFrame,
    spec: NeuralSpec,
    preprocessor: Any,
    encoding: FoldEncoding,
) -> NDArray[np.float64]:
    """The full dense input: scaled numerics, then one-hot indicators if any.

    For an embedding spec this returns the numerics alone -- the categorical block enters
    the network as integer codes through :func:`code_matrix`, not as dense columns.
    """
    numeric = apply_preprocessor(preprocessor, frame, spec)
    if spec.encoding is not CategoricalEncoding.ONE_HOT:
        return numeric
    indicators = encode.one_hot_matrix(frame, spec, encoding)
    if indicators.shape[0] != numeric.shape[0]:
        raise NeuralPreprocessError(
            f"{spec.name}: {numeric.shape[0]} numeric rows but {indicators.shape[0]} "
            "indicator rows; the categorical join is misaligned"
        )
    return np.hstack([numeric, indicators]).astype(np.float64, copy=False)


def code_matrix(frame: pl.DataFrame, spec: NeuralSpec, encoding: FoldEncoding) -> NDArray[np.int64]:
    """Integer category codes for the embedding block, or a zero-width array."""
    if spec.encoding is not CategoricalEncoding.EMBEDDING:
        return np.zeros((frame.height, 0), dtype=np.int64)
    return encode.index_matrix(frame, spec, encoding)


def dense_width(spec: NeuralSpec, encoding: FoldEncoding) -> int:
    """How many dense input columns the network's first layer receives."""
    width = len(matrix_columns(spec))
    if spec.encoding is CategoricalEncoding.ONE_HOT:
        width += len(encode.one_hot_columns(spec, encoding))
    return width


def nan_cells(frame: pl.DataFrame, spec: NeuralSpec) -> int:
    """Cells that were NULL before imputation.

    Component 7 records this to prove imputation did *not* happen. Component 8 records the
    same number to say exactly how much imputation did -- the two components disagree
    about what to do with a NULL, and the artifact should make the disagreement visible
    rather than leave it to prose.
    """
    matrix = numeric_matrix(frame, spec)
    return int(np.count_nonzero(np.isnan(matrix)))


__all__ = [
    "NeuralPreprocessError",
    "apply_preprocessor",
    "build_preprocessor",
    "code_matrix",
    "dense_matrix",
    "dense_width",
    "imputed_values",
    "matrix_columns",
    "nan_cells",
    "numeric_matrix",
    "scaler_statistics",
    "transformed_columns",
]
