"""The three attribution methods. Pure -- no I/O, no filesystem, no clock.

One model family, one method, chosen because of what the estimator *is* rather than to make
the code uniform. Forcing all three through one model-agnostic explainer would have been
tidier and would have thrown away two exact computations to buy that tidiness.

===================  =========================================  =========
family               method                                     exact
===================  =========================================  =========
logistic             closed form under an interventional ref.   yes
boosted              the booster's own TreeSHAP                 yes
neural MLP           antithetic permutation sampling            **no**
===================  =========================================  =========

**Why the ``shap`` package is not imported here.** Two of the three methods are already
present in libraries this project depends on: ``xgboost`` and ``lightgbm`` each ship an
exact TreeSHAP behind ``pred_contribs``/``pred_contrib``, and a linear model's Shapley
values under an interventional reference are three lines of arithmetic. Only the network
needs an approximation, and a permutation game over thirty columns is arithmetic rather
than an algorithm -- ADR 0015's own dividing line for what earns a runtime dependency. So
the values are computed here and the test suite cross-checks every one of them against
``shap.TreeExplainer`` and ``shap.LinearExplainer``, which are dev-only, exactly as
Component 5 cross-checks its hand-rolled metrics against scikit-learn. ADR 0030.

**Everything here is in log-odds.** ``decision_function`` for the linear model,
``output_margin``/``raw_score`` for the boosters, the pre-sigmoid logit for the network.
That is not a convenience: additivity holds in the margin and not in probability, because
``sigmoid`` is not linear. A probability-space table would have to abandon additivity or
fabricate it, and this component does neither.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sentinel.calibration.definitions import Family
from sentinel.explain.definitions import (
    ExplanationMethod,
    ExplanationStatus,
    OutputSpace,
)
from sentinel.explain.models import FoldAttribution, RefitModel

logger = logging.getLogger(__name__)


class AttributionError(RuntimeError):
    """Raised when attributions cannot be computed for a model."""


# --- exact: linear -----------------------------------------------------------


def linear_attributions(
    matrix: NDArray[np.float64],
    background: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    intercept: float,
) -> tuple[NDArray[np.float64], float]:
    """Exact Shapley values for a linear model under an interventional reference.

    For ``f(z) = intercept + coef @ z``, the Shapley value of column *j* at row *z* with
    reference distribution *R* is::

        phi_j = coef_j * (z_j - E_R[z_j])

    and the base value is ``intercept + coef @ E_R[z]``. There is nothing to sample: the
    feature-order average that defines a Shapley value collapses because a linear model has
    no interactions, so every ordering contributes the same marginal.

    The reference mean is **computed from the background**, never assumed to be zero. It is
    very nearly zero here -- ``StandardScaler`` centres each column on its training mean, and
    the profiling script measured the residual at 4.2e-16 -- but assuming it would make every
    attribution silently wrong the moment a background narrower than the full training window
    is used, which is exactly what a fold-local reference is.
    """
    if background.shape[0] == 0:
        raise AttributionError(
            "linear attribution needs a reference distribution; an empty background would "
            "make the base value the intercept alone, which is a different quantity"
        )
    reference = background.mean(axis=0)
    values = (matrix - reference) * coefficients
    base = intercept + float(reference @ coefficients)
    return values, base


# --- exact: trees ------------------------------------------------------------


def tree_attributions(
    estimator: Any, matrix: NDArray[np.float64], *, is_xgboost: bool
) -> tuple[NDArray[np.float64], float]:
    """Exact TreeSHAP, from the booster's own implementation.

    Both libraries return an ``[n, M + 1]`` block: one column per feature, and the expected
    value in the last position. The expected value is constant across rows -- the profiling
    script measured its standard deviation at 0.0 for xgboost and 5.6e-17 for lightgbm --
    which is what makes it an expected value rather than a per-row intercept, and it is
    checked here rather than assumed because a per-row base would silently break every
    downstream aggregate.

    No background is passed, and none is wanted. The tree-path-dependent algorithm takes its
    conditional expectations over the *cover* recorded in the trees at fit time, so the
    reference distribution is the training data the model already saw -- which is both
    temporally safe by construction and not something a caller could override without
    changing what the values mean.
    """
    if is_xgboost:
        import xgboost as xgb

        booster = estimator.get_booster()
        block = np.asarray(
            booster.predict(xgb.DMatrix(matrix), pred_contribs=True), dtype=np.float64
        )
    else:
        block = np.asarray(estimator.predict_proba(matrix, pred_contrib=True), dtype=np.float64)

    if block.ndim != 2 or block.shape[1] != matrix.shape[1] + 1:
        raise AttributionError(
            f"expected an [n, {matrix.shape[1] + 1}] contribution block, got {block.shape}"
        )
    bases = block[:, -1]
    spread = float(bases.max() - bases.min()) if len(bases) else 0.0
    if spread > 1e-9:
        raise AttributionError(
            f"the booster returned a base value that varies across rows by {spread:.3e}. "
            "TreeSHAP's last column is an expected value and must be constant; a per-row "
            "base would make every aggregate in this component meaningless."
        )
    return block[:, :-1], float(bases[0]) if len(bases) else 0.0


# --- approximate: permutation ------------------------------------------------


def permutation_attributions(
    predict: Any,
    matrix: NDArray[np.float64],
    background: NDArray[np.float64],
    *,
    rounds: int,
    seed: int,
) -> tuple[NDArray[np.float64], float]:
    """Antithetic permutation SHAP for an arbitrary model. **Approximate.**

    A permutation of the columns defines a path from a background row to the explained row:
    overwrite one column at a time, and attribute each step's change in output to the column
    that moved. Averaged over every ordering, those marginals are the Shapley values by
    definition; averaged over a *sample* of orderings, they are an unbiased estimate of them.

    Each permutation is walked forwards and then backwards -- the antithetic partner -- which
    cancels the first-order ordering bias for free, since a column drawn early in one walk is
    drawn late in the other.

    **The path telescopes, so additivity is exact at one round.** Summing the steps of any
    single walk gives ``f(row) - f(background)`` identically, and averaging walks preserves
    that. This is worth stating plainly because it is a trap: an additivity check on this
    method passes at *any* round count and is therefore **not evidence that the values are
    accurate**. What is approximate is how the credit is divided among columns, which
    additivity cannot see and which the ``permutation_convergence`` profile measures instead.

    Determinism is the seeded generator plus the fixed background order. The generator is
    consumed in a fixed sequence -- one permutation per round per row, rows in matrix order --
    so the same inputs and seed produce the same values bit for bit.
    """
    if background.shape[0] == 0:
        raise AttributionError("permutation attribution needs a non-empty background")
    if rounds <= 0:
        raise AttributionError(f"rounds must be positive, got {rounds}")

    n_rows, n_features = matrix.shape
    n_background = background.shape[0]
    rng = np.random.default_rng(seed)
    values = np.zeros((n_rows, n_features), dtype=np.float64)
    base = float(np.mean(predict(background)))

    for i in range(n_rows):
        row = matrix[i]
        total = np.zeros(n_features, dtype=np.float64)
        for _ in range(rounds):
            order = rng.permutation(n_features)
            for forward in (True, False):
                walk = order if forward else order[::-1]
                current = background.copy() if forward else np.tile(row, (n_background, 1))
                # The whole path is materialised and scored in one call. A per-step call
                # would be (M + 1) times the Python overhead for the same arithmetic, and
                # for the network it is the difference between minutes and an hour.
                path = np.empty((n_features + 1, n_background, n_features), dtype=np.float64)
                path[0] = current
                for step, column in enumerate(walk):
                    current[:, column] = row[column] if forward else background[:, column]
                    path[step + 1] = current
                outputs = (
                    np.asarray(predict(path.reshape(-1, n_features)), dtype=np.float64)
                    .reshape(n_features + 1, n_background)
                    .mean(axis=1)
                )
                deltas = np.diff(outputs)
                for step, column in enumerate(walk):
                    total[column] += deltas[step] if forward else -deltas[step]
        values[i] = total / (2 * rounds)
    return values, base


# --- dispatch ----------------------------------------------------------------


def _network_predictor(network: Any) -> Any:
    """A ``matrix -> logits`` callable over a numeric-only network."""
    import torch

    def predict(block: NDArray[np.float64]) -> NDArray[np.float64]:
        network.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32))
            codes = torch.zeros((block.shape[0], 0), dtype=torch.int64)
            out = network(tensor, codes)
        return np.asarray(out.numpy(), dtype=np.float64)

    return predict


def attribute_fold(
    model: RefitModel, positions: list[int], *, rounds: int, seed: int
) -> FoldAttribution:
    """Attributions for one model on one fold, over the sampled rows.

    ``positions`` indexes the sampled rows into the model's full test-window matrix. The
    tree and linear methods are computed over the sample too, rather than over the full
    window and then subset: the values are identical either way -- both are row-wise -- and
    computing what is written keeps one code path instead of two that could diverge.
    """
    spec = model.spec
    if spec.status is not ExplanationStatus.SUPPORTED or spec.method is None:
        raise AttributionError(f"{spec.name} is unsupported and has no attribution path")

    started = time.perf_counter()
    matrix = model.matrix[positions]
    output = model.output[positions]
    row_ids = tuple(model.row_ids[p] for p in positions)

    if spec.family is Family.LOGISTIC:
        fitted = model.estimator
        values, base = linear_attributions(
            matrix,
            model.background,
            np.asarray(fitted.coefficients, dtype=np.float64),
            float(fitted.intercept),
        )
    elif spec.family is Family.BOOSTED:
        from sentinel.boosting.definitions import Estimator
        from sentinel.boosting.definitions import spec_for as boosting_spec_for

        is_xgboost = boosting_spec_for(spec.name).estimator is Estimator.XGBOOST
        values, base = tree_attributions(model.estimator.estimator, matrix, is_xgboost=is_xgboost)
    elif spec.family is Family.NEURAL_MLP:
        _, network = model.estimator
        values, base = permutation_attributions(
            _network_predictor(network),
            matrix,
            model.background,
            rounds=rounds,
            seed=seed,
        )
    else:  # pragma: no cover - the registry guard makes this unreachable
        raise AttributionError(f"{spec.name}: no attribution path for family {spec.family}")

    if values.shape != (len(positions), len(model.matrix_columns)):
        raise AttributionError(
            f"{spec.name}/{model.fold_id}: attributions are {values.shape}, expected "
            f"{(len(positions), len(model.matrix_columns))}. A transposed block would "
            "attribute every value to the wrong feature."
        )
    if not np.isfinite(values).all():
        raise AttributionError(
            f"{spec.name}/{model.fold_id}: attribution block contains a non-finite value"
        )

    elapsed = time.perf_counter() - started
    logger.info(
        "Attributed %s on %s: %d rows x %d features via %s in %.1fs",
        spec.name,
        model.fold_id,
        values.shape[0],
        values.shape[1],
        spec.method.value,
        elapsed,
    )
    return FoldAttribution(
        model_name=spec.name,
        fold_set=model.fold_set,
        fold_id=model.fold_id,
        method=spec.method,
        output_space=spec.output_space or OutputSpace.LOG_ODDS,
        is_exact=spec.is_exact,
        row_ids=row_ids,
        feature_names=model.matrix_columns,
        values=values,
        base_value=base,
        output=output,
        seconds=elapsed,
    )


__all__ = [
    "AttributionError",
    "ExplanationMethod",
    "attribute_fold",
    "linear_attributions",
    "permutation_attributions",
    "tree_attributions",
]
