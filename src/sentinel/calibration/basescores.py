"""Regenerating the base-model scores Component 9 calibrates. Pure -- no filesystem, no clock.

This module exists because of a missing artifact, not a missing capability, and the
distinction is the whole of ADR 0026.

A calibrator is fitted on a base model's scores over the fold's **calibration** window.
Those scores do not exist. Every prediction artifact on disk covers exactly the test window
-- 41,536 rows per model over 18 folds, which is ``sum(test_rows)`` to the row -- and the
34,261 calibration-window rows were never scored. Components 6, 7 and 8 each say so, in a
column literally named ``calibration_end_unused``. Nor can the scores be produced by loading
a model: **no fitted model object is persisted anywhere in this repository.**

So the fits run again. This module is the only place that imports Component 6, 7 or 8's fit
functions, and it imports them **unchanged**: same registry spec, same seed, same
hyperparameters, same canonical row order, same training frame. Nothing is tuned, no
hyperparameter is touched, no feature is added. This is a re-execution, not a re-training,
and the difference is not rhetorical -- it is proved rather than asserted:

Every fit scores **both** windows. The calibration window is what Component 9 wanted. The
test window is the **control**: ``validate.base_scores_reproduce_the_committed_artifact``
compares it to the committed artifact with ``==``, bit-identity rather than
``math.isclose``, and ``build.py`` refuses to fit a single calibrator if it does not match.
A calibrator fitted on scores no committed artifact contains would be a correction to
nothing.

The native decision margin is captured alongside the probability. It is never what the
calibrator is fed (ADR 0027) -- it is the cross-check that turns "recovering the logit is
lossless" from a claim into a measurement.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from sentinel.calibration.definitions import EMBEDDING_DONOR, CandidateSpec, Family
from sentinel.calibration.models import BaseScores
from sentinel.calibration.preprocess import calibration_frame
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec

logger = logging.getLogger(__name__)


class BaseScoreError(RuntimeError):
    """Raised when a base model's scores cannot be regenerated for a fold."""


def _labels_and_dates(window: pl.DataFrame, ids: list[str]) -> tuple[list[int], list[Any]]:
    """Labels and reference dates for ``ids``, in the order the scorer returned them.

    Aligned by id rather than by position. ``score_window`` returns its own ids precisely so
    a caller never has to reconstruct the order, and re-deriving it here would reintroduce
    the mis-join that guarantee exists to prevent.
    """
    if "target" not in window.columns:
        raise BaseScoreError("calibration window has no target column")
    lookup = dict(
        zip(
            (str(v) for v in window["target_inspection_id"].to_list()),
            zip(
                (int(v) for v in window["target"].to_list()),
                window["rd"].to_list(),
                strict=True,
            ),
            strict=True,
        )
    )
    missing = [row_id for row_id in ids if row_id not in lookup]
    if missing:
        raise BaseScoreError(
            f"{len(missing)} scored id(s) are not in the window they were scored from; "
            f"first: {', '.join(missing[:5])}"
        )
    pairs = [lookup[row_id] for row_id in ids]
    return [label for label, _ in pairs], [day for _, day in pairs]


def _finite(values: NDArray[Any] | list[float], what: str, model: str, fold_id: str) -> list[float]:
    """Narrow a numpy result to a list of floats, rejecting anything non-finite.

    A non-finite margin is a defect in the fit rather than a value to carry: it would reach
    the manifest as a NaN maximum and read as "no discrepancy measured".
    """
    out = [float(v) for v in np.asarray(values).ravel()]
    bad = sum(1 for v in out if not np.isfinite(v))
    if bad:
        raise BaseScoreError(f"{model}: {bad} non-finite {what} for fold {fold_id}")
    return out


# --- one family per function, so the dispatch reads as a list of call chains ---


def _logistic(
    spec: CandidateSpec, training: pl.DataFrame, windows: dict[str, pl.DataFrame], fold: FoldSpec
) -> dict[str, tuple[list[str], list[float], list[float]]]:
    from sentinel.modeling import predict as modeling_predict
    from sentinel.modeling import preprocess as modeling_preprocess
    from sentinel.modeling import train as modeling_train
    from sentinel.modeling.definitions import spec_for as modeling_spec_for

    model_spec = modeling_spec_for(spec.name)
    fitted = modeling_train.fit_fold(model_spec, training, fold)

    out: dict[str, tuple[list[str], list[float], list[float]]] = {}
    for split, window in windows.items():
        ids, scores = modeling_predict.score_window(fitted, window)
        matrix = modeling_preprocess.to_matrix(window, model_spec)
        # The exact linear predictor. sklearn's Pipeline delegates decision_function to the
        # final LogisticRegression, so this is the logit that predict_proba's sigmoid maps.
        raw = fitted.pipeline.decision_function(matrix)
        margins = _finite(raw, "margin", spec.name, fold.fold_id)
        out[split] = (ids, scores, margins)
    return out


def _boosted(
    spec: CandidateSpec, training: pl.DataFrame, windows: dict[str, pl.DataFrame], fold: FoldSpec
) -> dict[str, tuple[list[str], list[float], list[float]]]:
    from sentinel.boosting import predict as boosting_predict
    from sentinel.boosting import preprocess as boosting_preprocess
    from sentinel.boosting import train as boosting_train
    from sentinel.boosting.definitions import Estimator
    from sentinel.boosting.definitions import spec_for as boosting_spec_for

    model_spec = boosting_spec_for(spec.name)
    fitted = boosting_train.fit_fold(model_spec, training, fold)

    out: dict[str, tuple[list[str], list[float], list[float]]] = {}
    for split, window in windows.items():
        ids, scores = boosting_predict.score_window(fitted, window)
        matrix = boosting_preprocess.tree_matrix(window, model_spec)
        # Both objectives are the logistic one, so the raw output is the logit rather than
        # a score on some other scale. XGBoost and LightGBM spell the same request
        # differently, which is the only reason this branches.
        raw: Any = (
            fitted.estimator.predict(matrix, output_margin=True)
            if model_spec.estimator is Estimator.XGBOOST
            else fitted.estimator.predict_proba(matrix, raw_score=True)
        )
        out[split] = (ids, scores, _finite(raw, "margin", spec.name, fold.fold_id))
    return out


def _neural_mlp(
    spec: CandidateSpec, training: pl.DataFrame, windows: dict[str, pl.DataFrame], fold: FoldSpec
) -> dict[str, tuple[list[str], list[float], list[float]]]:
    import torch

    from sentinel.neural import predict as neural_predict
    from sentinel.neural import preprocess as neural_preprocess
    from sentinel.neural import train as neural_train
    from sentinel.neural.definitions import spec_for as neural_spec_for

    model_spec = neural_spec_for(spec.name)
    # categoricals=None is legal for an encoding of NONE; neural.train raises only when the
    # spec actually names entity columns. Determinism comes free: fit_fold calls
    # net.seed_everything, which sets deterministic algorithms and a single torch thread.
    fitted = neural_train.fit_fold(model_spec, training, fold)
    network, preprocessor = neural_train.scorer_for(fitted)

    out: dict[str, tuple[list[str], list[float], list[float]]] = {}
    for split, window in windows.items():
        ids, scores = neural_predict.score_window(fitted, window)
        dense = neural_preprocess.dense_matrix(window, model_spec, preprocessor, fitted.encoding)
        codes = neural_preprocess.code_matrix(window, model_spec, fitted.encoding)
        network.eval()
        with torch.no_grad():
            # The pre-sigmoid logit. predict.score_window applies torch.sigmoid to exactly
            # this tensor and nowhere else, so the two are the same quantity.
            logits = network(
                torch.from_numpy(np.ascontiguousarray(dense, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(codes, dtype=np.int64)),
            )
        out[split] = (ids, scores, _finite(logits.numpy(), "logit", spec.name, fold.fold_id))
    return out


def _neural_embedding_booster(
    spec: CandidateSpec,
    training: pl.DataFrame,
    windows: dict[str, pl.DataFrame],
    fold: FoldSpec,
    categoricals: pl.DataFrame | None,
) -> dict[str, tuple[list[str], list[float], list[float]]]:
    from sentinel.neural import embed
    from sentinel.neural import train as neural_train
    from sentinel.neural.definitions import spec_for as neural_spec_for

    if categoricals is None:
        raise BaseScoreError(
            f"{spec.name}: needs Component 8's categorical table. Pass --categoricals, or "
            "drop this experimental candidate with --models."
        )

    # The donor network is re-fitted rather than reconstructed from the persisted embedding
    # table. The table WOULD reproduce it exactly, but the booster half is not persisted
    # either, so the reconstruction saves only the donor fit while requiring a 30-field
    # FittedNetwork to be fabricated -- inside an object whose purpose is to be readable by
    # a leakage test. The persisted table is used as a validation check instead (ADR 0026).
    donor = neural_train.fit_fold(
        neural_spec_for(EMBEDDING_DONOR), training, fold, categoricals=categoricals
    )
    model_spec = neural_spec_for(spec.name)
    # embed.fit_fold re-checks that the donor was fitted on THIS fold. That check is the
    # experiment's entire temporal guarantee, and it is left to run.
    fitted = embed.fit_fold(model_spec, training, fold, donor=donor, categoricals=categoricals)

    out: dict[str, tuple[list[str], list[float], list[float]]] = {}
    for split, window in windows.items():
        ids, scores = embed.score_window(fitted, window, donor=donor, categoricals=categoricals)
        # No native margin for this family. Reaching it would need ``embed``'s private join
        # helper, and Component 8 is closed -- widening its API for a warn-severity
        # diagnostic is not a trade worth making. NaN rather than a fabricated value, and
        # the margin cross-check reports this candidate as "not available" rather than as
        # passing. The check it would perform is also nearly circular here: an XGBoost
        # binary:logistic margin IS the logit of its own predict_proba.
        out[split] = (ids, scores, [float("nan")] * len(ids))
    return out


def regenerate_fold(
    spec: CandidateSpec,
    frame: pl.DataFrame,
    fold: FoldSpec,
    *,
    categoricals: pl.DataFrame | None = None,
) -> BaseScores:
    """Re-execute one candidate's fit on one fold, scoring both windows.

    ``frame`` is the full Component 4 feature table with a parsed ``rd`` column. The
    training frame comes from ``modeling.train.training_frame`` for every family, because
    that is the repository's one definition of "train" -- the one Component 5's
    ``future_rows_never_enter_training`` check independently re-derives.
    """
    from sentinel.modeling.train import training_frame

    started = time.perf_counter()
    training = training_frame(frame, fold)
    windows = {
        "calibration": calibration_frame(frame, fold),
        "test": folds_module.window_frame(frame, fold),
    }
    for split, window in windows.items():
        if window.height == 0:
            raise BaseScoreError(f"{spec.name}: fold {fold.fold_id} has an empty {split} window")

    if spec.family is Family.LOGISTIC:
        scored = _logistic(spec, training, windows, fold)
    elif spec.family is Family.BOOSTED:
        scored = _boosted(spec, training, windows, fold)
    elif spec.family is Family.NEURAL_MLP:
        scored = _neural_mlp(spec, training, windows, fold)
    elif spec.family is Family.NEURAL_EMBEDDING_BOOSTER:
        scored = _neural_embedding_booster(spec, training, windows, fold, categoricals)
    else:  # pragma: no cover - the enum is exhaustive and guarded at import
        raise BaseScoreError(f"{spec.name}: no regeneration path for family {spec.family}")

    calibration_ids, calibration_scores, calibration_margins = scored["calibration"]
    test_ids, test_scores, test_margins = scored["test"]
    calibration_labels, calibration_dates = _labels_and_dates(
        windows["calibration"], calibration_ids
    )
    test_labels, _ = _labels_and_dates(windows["test"], test_ids)

    elapsed = time.perf_counter() - started
    logger.info(
        "Regenerated %s on %s: %d calibration + %d test rows in %.1fs",
        spec.name,
        fold.fold_id,
        len(calibration_ids),
        len(test_ids),
        elapsed,
    )
    return BaseScores(
        model_name=spec.name,
        fold_set=fold.fold_set,
        fold_id=fold.fold_id,
        calibration_ids=tuple(calibration_ids),
        calibration_scores=tuple(calibration_scores),
        calibration_margins=tuple(calibration_margins),
        calibration_labels=tuple(calibration_labels),
        calibration_dates=tuple(calibration_dates),
        test_ids=tuple(test_ids),
        test_scores=tuple(test_scores),
        test_margins=tuple(test_margins),
        test_labels=tuple(test_labels),
        # The BASE model's horizon, not the calibrator's. Identical to what Components 6, 7
        # and 8 declare, and joinable back to their calibration_end_unused column.
        base_model_trained_through=fold.train_end,
        fit_seconds=elapsed,
    )


def committed_test_scores(
    predictions: pl.DataFrame, model_name: str, fold_id: str
) -> dict[str, float]:
    """The committed artifact's test-window scores for one (model, fold), by id.

    Read straight from the Component 6/7/8 Parquet rather than through
    ``evaluation.contract.read_predictions``: the comparison is on raw stored floats, and a
    round trip through the contract's ``PredictionSet`` would add a select and a filter
    between the file and the assertion for no benefit.
    """
    subset = predictions.filter(
        (pl.col("model_name") == model_name) & (pl.col("fold_id") == fold_id)
    )
    if subset.height == 0:
        raise BaseScoreError(
            f"{model_name}: the committed artifact has no rows for fold {fold_id}, so the "
            "bit-identity gate has nothing to compare against"
        )
    return dict(
        zip(
            (str(v) for v in subset["target_inspection_id"].to_list()),
            (float(v) for v in subset["score"].to_list()),
            strict=True,
        )
    )


def reproduction_mismatches(
    scores: BaseScores, committed: dict[str, float], *, limit: int = 20
) -> tuple[int, list[str]]:
    """Rows where the regenerated test score is not bit-identical to the committed one.

    ``!=`` on floats, deliberately. A tolerance here would convert the one check that makes
    ADR 0026 safe into a check that passes when the models differ, so the comparison is
    exact and a single differing bit is a failure.
    """
    if set(scores.test_ids) != set(committed):
        absent = len(set(committed) - set(scores.test_ids))
        surplus = len(set(scores.test_ids) - set(committed))
        raise BaseScoreError(
            f"{scores.model_name}/{scores.fold_id}: regenerated test coverage differs from "
            f"the committed artifact -- {absent} committed row(s) unscored, {surplus} extra"
        )

    offenders: list[str] = []
    mismatches = 0
    for row_id, score in zip(scores.test_ids, scores.test_scores, strict=True):
        if score != committed[row_id]:
            mismatches += 1
            if len(offenders) < limit:
                offenders.append(
                    f"{scores.model_name}/{scores.fold_id}/{row_id}: "
                    f"regenerated {score!r} != committed {committed[row_id]!r}"
                )
    return mismatches, offenders


__all__ = [
    "BaseScoreError",
    "committed_test_scores",
    "regenerate_fold",
    "reproduction_mismatches",
]
