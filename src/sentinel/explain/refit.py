"""Re-executing the frozen fits so their attributions can be computed. Pure -- no I/O.

This module exists for the reason ``calibration.basescores`` exists, and the argument is
ADR 0026's restated for a different need (ADR 0029).

A Shapley value is a property of a *model*, not of a score. TreeSHAP walks the trees;
linear SHAP reads the coefficients and the reference; the permutation game calls the
network. None of that can be done from a Parquet file of probabilities, and **no fitted
model object is persisted anywhere in this repository**. So the fits run again.

Like Component 9, this module imports Components 6, 7 and 8's fit functions **unchanged**:
same registry spec, same seed, same hyperparameters, same canonical row order, same
training frame -- ``modeling.train.training_frame``, the repository's one definition of
"train". Nothing is tuned, no hyperparameter is touched, no feature is added. That is a
re-execution, not a re-training, and the difference is proved rather than asserted:

Every re-executed fit scores the **test window**, and
``validate.regenerated_scores_reproduce_the_committed_artifact`` compares it to the
committed Component 6/7/8 artifact with ``==`` -- bit-identity, not ``math.isclose``. The
gate itself is not reimplemented here: ``calibration.basescores.committed_test_scores`` and
``reproduction_mismatches`` already own that comparison and are called directly, so there is
one definition of "the same model" in the project rather than two that could drift.

``build.py`` refuses to write an artifact if the gate fails, because an attribution computed
on a model no committed artifact contains would be an explanation of nothing.

**On what is *not* here.** ``xgboost_chain_embeddings`` has no path in this module. Its
fitted booster is reachable only through ``neural.embed._scorer_for``, a private
process-local stash, and Component 8 is closed. See ADR 0031 and
:data:`definitions.EMBEDDING_BOOSTER_UNSUPPORTED_REASON`.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from sentinel.calibration.definitions import Family
from sentinel.calibration.models import BaseScores
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.explain.definitions import ExplanationSpec, ExplanationStatus
from sentinel.explain.models import RefitModel, ReproductionOutcome

logger = logging.getLogger(__name__)


class RefitError(RuntimeError):
    """Raised when a model cannot be re-executed for a fold."""


def _network_logits(network: Any, dense: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pre-sigmoid output of a numeric-only network.

    ``codes`` is zero-width because the supported neural candidate embeds nothing;
    ``net.EmbeddingNet.forward`` documents that shape as legitimate rather than degenerate.
    """
    import torch

    network.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(dense, dtype=np.float32))
        codes = torch.zeros((dense.shape[0], 0), dtype=torch.int64)
        out = network(tensor, codes)
    return np.asarray(out.numpy(), dtype=np.float64)


def _logistic(
    spec: ExplanationSpec,
    training: pl.DataFrame,
    window: pl.DataFrame,
    background_window: pl.DataFrame,
    fold: FoldSpec,
) -> tuple[Any, tuple[str, ...], Any, Any, Any, Any, Any]:
    """Re-execute Component 6's fit and return everything the linear attribution needs."""
    from sentinel.modeling import predict as modeling_predict
    from sentinel.modeling import preprocess as modeling_preprocess
    from sentinel.modeling import train as modeling_train
    from sentinel.modeling.definitions import spec_for as modeling_spec_for

    model_spec = modeling_spec_for(spec.name)
    fitted = modeling_train.fit_fold(model_spec, training, fold)

    # ``ordered_matrix_columns``, not ``matrix_columns``. The two disagree at 19 of 30
    # positions and only this one matches what the fitted ColumnTransformer emits.
    columns = modeling_preprocess.ordered_matrix_columns(model_spec)
    raw = modeling_preprocess.to_matrix(window, model_spec)
    transform = fitted.pipeline.named_steps["preprocess"]
    transformed = np.asarray(transform.transform(raw), dtype=np.float64)
    background = np.asarray(
        transform.transform(modeling_preprocess.to_matrix(background_window, model_spec)),
        dtype=np.float64,
    )
    output = np.asarray(fitted.pipeline.decision_function(raw), dtype=np.float64)
    ids, scores = modeling_predict.score_window(fitted, window)
    # ``to_matrix`` emits the natural order, so the raw block is permuted into the fitted
    # pipeline's order before it is carried alongside values named in that order.
    natural = modeling_preprocess.matrix_columns(model_spec)
    permutation = [natural.index(name) for name in columns]
    return fitted, columns, raw[:, permutation], transformed, output, background, (ids, scores)


def _boosted(
    spec: ExplanationSpec,
    training: pl.DataFrame,
    window: pl.DataFrame,
    background_window: pl.DataFrame,
    fold: FoldSpec,
) -> tuple[Any, tuple[str, ...], Any, Any, Any, Any, Any]:
    """Re-execute Component 7's fit and return everything TreeSHAP needs.

    The background block is empty on purpose. TreeSHAP under the tree-path-dependent
    algorithm takes its conditional expectation over the *cover* recorded in the trees at
    fit time, so it needs no reference rows -- and supplying some would misdescribe what
    the base value is.
    """
    from sentinel.boosting import predict as boosting_predict
    from sentinel.boosting import preprocess as boosting_preprocess
    from sentinel.boosting import train as boosting_train
    from sentinel.boosting.definitions import Estimator
    from sentinel.boosting.definitions import spec_for as boosting_spec_for

    model_spec = boosting_spec_for(spec.name)
    fitted = boosting_train.fit_fold(model_spec, training, fold)

    # ``matrix_columns``, not ``ordered_matrix_columns``. Component 7 fits no
    # ColumnTransformer, so its matrix is in the natural order and permuting it here would
    # attach every value to the wrong feature without changing any sum.
    columns = boosting_preprocess.matrix_columns(model_spec)
    matrix = boosting_preprocess.tree_matrix(window, model_spec)
    estimator = fitted.estimator
    if model_spec.estimator is Estimator.XGBOOST:
        output = np.asarray(estimator.predict(matrix, output_margin=True), dtype=np.float64)
    else:
        output = np.asarray(estimator.predict_proba(matrix, raw_score=True), dtype=np.float64)
    ids, scores = boosting_predict.score_window(fitted, window)
    empty = np.zeros((0, matrix.shape[1]), dtype=np.float64)
    # Nothing is imputed or scaled, so the estimator's matrix *is* the raw one.
    return fitted, columns, matrix, matrix, output, empty, (ids, scores)


def _neural(
    spec: ExplanationSpec,
    training: pl.DataFrame,
    window: pl.DataFrame,
    background_window: pl.DataFrame,
    fold: FoldSpec,
) -> tuple[Any, tuple[str, ...], Any, Any, Any, Any, Any]:
    """Re-execute Component 8's fit and return everything the permutation game needs.

    The live module is reached through ``neural.train.scorer_for``, which is public. That
    is the whole difference between this model and ``xgboost_chain_embeddings``.
    """
    from sentinel.neural import predict as neural_predict
    from sentinel.neural import preprocess as neural_preprocess
    from sentinel.neural import train as neural_train
    from sentinel.neural.definitions import spec_for as neural_spec_for

    model_spec = neural_spec_for(spec.name)
    fitted = neural_train.fit_fold(model_spec, training, fold)
    network, preprocessor = neural_train.scorer_for(fitted)

    columns = neural_preprocess.transformed_columns(model_spec, fitted.encoding)
    transformed = neural_preprocess.apply_preprocessor(preprocessor, window, model_spec)
    background = neural_preprocess.apply_preprocessor(preprocessor, background_window, model_spec)
    raw = neural_preprocess.numeric_matrix(window, model_spec)
    natural = neural_preprocess.matrix_columns(model_spec)
    permutation = [natural.index(name) for name in columns]
    output = _network_logits(network, transformed)
    ids, scores = neural_predict.score_window(fitted, window)
    return (
        (fitted, network),
        columns,
        raw[:, permutation],
        transformed,
        output,
        background,
        (ids, scores),
    )


def _background_window(
    frame: pl.DataFrame, fold: FoldSpec, spec: ExplanationSpec, size: int, seed: int
) -> pl.DataFrame:
    """Reference rows for a method that needs them, drawn from the training window only.

    Delegated to ``background.select`` so there is one definition of a temporally safe
    reference set. Imported here rather than at module scope to keep the dependency
    one-directional: ``background`` knows nothing about refitting.
    """
    from sentinel.explain.background import select_background

    if spec.family is Family.BOOSTED:
        # TreeSHAP needs no reference rows, and drawing some would imply it did.
        return frame.head(0)
    return select_background(frame, fold, size=size, seed=seed)


def regenerate_fold(
    spec: ExplanationSpec,
    frame: pl.DataFrame,
    fold: FoldSpec,
    explained_ids: tuple[str, ...],
    *,
    background_size: int,
    background_seed: int,
) -> RefitModel:
    """Re-execute one model's fit on one fold and assemble its attribution inputs.

    ``frame`` is the full Component 4 feature table with a parsed ``rd`` column. The
    training frame comes from ``modeling.train.training_frame`` for every family, because
    that is the repository's one definition of "train" -- the one Component 5's
    ``future_rows_never_enter_training`` check independently re-derives.

    The model is fitted and scored over the **whole** test window, not over
    ``explained_ids``. Two reasons, and the first is not negotiable: the bit-identity gate
    compares against a committed artifact that covers every test row, and a partial
    comparison would be a partial proof. The second is that the tree and linear
    attributions are free over the full window, so restricting them would buy nothing.
    """
    from sentinel.modeling.train import training_frame

    if spec.status is not ExplanationStatus.SUPPORTED:
        raise RefitError(
            f"{spec.name} is not supported by Component 11 and must not be refitted: "
            f"{spec.unsupported_reason}"
        )

    started = time.perf_counter()
    training = training_frame(frame, fold)
    window = folds_module.window_frame(frame, fold)
    if training.height == 0:
        raise RefitError(f"{spec.name}: fold {fold.fold_id} has an empty training window")
    if window.height == 0:
        raise RefitError(f"{spec.name}: fold {fold.fold_id} has an empty test window")

    background_window = _background_window(frame, fold, spec, background_size, background_seed)

    if spec.family is Family.LOGISTIC:
        handler = _logistic
    elif spec.family is Family.BOOSTED:
        handler = _boosted
    elif spec.family is Family.NEURAL_MLP:
        handler = _neural
    else:  # pragma: no cover - the registry guard makes this unreachable
        raise RefitError(f"{spec.name}: no re-execution path for family {spec.family}")

    estimator, columns, raw, transformed, output, background, scored = handler(
        spec, training, window, background_window, fold
    )
    ids, scores = scored

    if len(columns) != transformed.shape[1]:
        raise RefitError(
            f"{spec.name}/{fold.fold_id}: {len(columns)} column names for a "
            f"{transformed.shape[1]}-wide matrix. A name list and a matrix that disagree "
            "would mislabel every attribution without changing any sum."
        )
    if list(ids) != window["target_inspection_id"].to_list():
        raise RefitError(
            f"{spec.name}/{fold.fold_id}: the scorer's row order differs from the test "
            "window's, so attributions could not be aligned to ids by position"
        )

    # Narrowed rather than annotated: a polars aggregate is typed as a wide union, and
    # the manifest records this value as a date. Component 5's ``folds`` narrows the same
    # way for the same reason.
    background_max: date | None = None
    if background_window.height:
        latest = background_window["rd"].max()
        background_max = latest if isinstance(latest, date) else None

    elapsed = time.perf_counter() - started
    logger.info(
        "Re-executed %s on %s: %d test rows, %d background rows in %.1fs",
        spec.name,
        fold.fold_id,
        window.height,
        background.shape[0],
        elapsed,
    )
    return RefitModel(
        spec=spec,
        fold_set=fold.fold_set,
        fold_id=fold.fold_id,
        estimator=estimator,
        matrix_columns=tuple(columns),
        matrix=np.asarray(transformed, dtype=np.float64),
        raw_matrix=np.asarray(raw, dtype=np.float64),
        row_ids=tuple(str(v) for v in ids),
        output=np.asarray(output, dtype=np.float64),
        probability=np.asarray(scores, dtype=np.float64),
        # The BASE model's horizon. Identical to what Components 6, 7 and 8 declare, and
        # joinable back to their ``calibration_end_unused`` column.
        trained_through=fold.train_end,
        train_start=fold.train_start,
        train_end=fold.train_end,
        background=np.asarray(background, dtype=np.float64),
        background_max_date=background_max,
        fit_seconds=elapsed,
    )


def check_reproduction(model: RefitModel, committed: pl.DataFrame) -> ReproductionOutcome:
    """Compare a re-executed fit's test scores to the committed artifact, bit for bit.

    The comparison is Component 9's, called rather than copied. ``!=`` on floats,
    deliberately: a tolerance here would convert the one check that makes ADR 0029 safe
    into a check that passes when the models differ.

    A ``BaseScores`` is assembled to carry the scores because that is the type
    ``reproduction_mismatches`` accepts. Only the test-window fields are populated -- this
    component never touches a calibration window -- and the empty ones are exactly that
    rather than fabricated, which is why they are empty tuples and not zeros.
    """
    from sentinel.calibration.basescores import committed_test_scores, reproduction_mismatches

    scores = BaseScores(
        model_name=model.spec.name,
        fold_set=model.fold_set,
        fold_id=model.fold_id,
        calibration_ids=(),
        calibration_scores=(),
        calibration_margins=(),
        calibration_labels=(),
        calibration_dates=(),
        test_ids=model.row_ids,
        test_scores=tuple(float(v) for v in model.probability),
        test_margins=tuple(float(v) for v in model.output),
        test_labels=(),
        base_model_trained_through=model.trained_through,
        fit_seconds=model.fit_seconds,
    )
    reference = committed_test_scores(committed, model.spec.name, model.fold_id)
    mismatches, offenders = reproduction_mismatches(scores, reference)
    return ReproductionOutcome(
        model_name=model.spec.name,
        fold_id=model.fold_id,
        rows=len(model.row_ids),
        mismatches=mismatches,
        offenders=tuple(offenders),
    )


__all__ = ["RefitError", "check_reproduction", "regenerate_fold"]
