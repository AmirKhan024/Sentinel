"""Are the attributions *right*?

Three methods, three different standards of proof, and none of them is "the numbers look
plausible".

**Tree and linear SHAP are cross-checked against the ``shap`` package.** It is a dev-only
dependency used exactly as scikit-learn was used through Component 5: the values ship
computed in-house, and an independent implementation asserts they are the same numbers.
Agreement to 1e-9 with a library nobody in this project wrote is far stronger evidence than
any internal consistency check.

**The permutation method is checked against the Shapley definition itself.** For a model
small enough to enumerate, the exact interventional Shapley value is a sum over all 2^M
subsets, which is the definition rather than another implementation of it. That is the
strongest oracle available and it is what
``test_permutation_shap_converges_on_brute_force_shapley`` uses.

**Additivity is tested, and its weakness is tested too.** The permutation path telescopes,
so additivity holds at one round and at sixty-four alike -- and
``test_additivity_holds_at_one_round_and_is_therefore_not_evidence_of_accuracy`` asserts
precisely that, so nobody later reads a passing additivity check as a passing accuracy
check.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from sentinel.explain.attribute import (
    AttributionError,
    linear_attributions,
    permutation_attributions,
    tree_attributions,
)

shap = pytest.importorskip("shap", reason="the dev-only attribution oracle")


FEATURES = 6
ROWS = 120
SEED = 20260825


def _dataset() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """A deterministic, mildly non-linear dataset. No RNG state leaks between tests."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(ROWS, FEATURES))
    logits = 1.4 * x[:, 0] - 0.9 * x[:, 1] + 0.5 * x[:, 2] * x[:, 3]
    return x, (logits > 0).astype(np.int64)


# --- 1. linear SHAP is exact, and matches the oracle -------------------------


def test_linear_shap_matches_the_shap_package_exactly() -> None:
    x, y = _dataset()
    model = LogisticRegression(max_iter=1000).fit(x, y)
    coefficients = np.asarray(model.coef_[0], dtype=np.float64)

    values, base = linear_attributions(x, x, coefficients, float(model.intercept_[0]))

    oracle = shap.LinearExplainer(model, shap.maskers.Independent(x, max_samples=ROWS))
    assert np.abs(values - oracle.shap_values(x)).max() < 1e-9
    assert abs(base - float(oracle.expected_value)) < 1e-9


def test_linear_shap_reconstructs_the_decision_function() -> None:
    x, y = _dataset()
    model = LogisticRegression(max_iter=1000).fit(x, y)
    values, base = linear_attributions(
        x, x, np.asarray(model.coef_[0], dtype=np.float64), float(model.intercept_[0])
    )
    assert np.abs(base + values.sum(axis=1) - model.decision_function(x)).max() < 1e-10


def test_the_linear_base_value_follows_the_background_it_was_given() -> None:
    """The reference is computed, never assumed to be zero -- the point of the docstring."""
    x, y = _dataset()
    model = LogisticRegression(max_iter=1000).fit(x, y)
    coefficients = np.asarray(model.coef_[0], dtype=np.float64)
    intercept = float(model.intercept_[0])

    _, full = linear_attributions(x, x, coefficients, intercept)
    _, narrow = linear_attributions(x, x[:10], coefficients, intercept)
    assert full != narrow, "a different reference distribution is a different base value"


def test_an_empty_background_is_refused_rather_than_treated_as_the_intercept() -> None:
    x, y = _dataset()
    model = LogisticRegression(max_iter=1000).fit(x, y)
    with pytest.raises(AttributionError, match="reference distribution"):
        linear_attributions(
            x,
            np.zeros((0, FEATURES)),
            np.asarray(model.coef_[0], dtype=np.float64),
            float(model.intercept_[0]),
        )


# --- 2. tree SHAP is exact, and matches the oracle ---------------------------


def _xgboost(x: NDArray[np.float64], y: NDArray[np.int64]) -> object:
    import xgboost as xgb

    return xgb.XGBClassifier(n_estimators=12, max_depth=3, n_jobs=1, random_state=SEED).fit(x, y)


def _lightgbm(x: NDArray[np.float64], y: NDArray[np.int64]) -> object:
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=12, max_depth=3, n_jobs=1, random_state=SEED, verbose=-1
    ).fit(x, y)


@pytest.mark.parametrize("is_xgboost", [True, False], ids=["xgboost", "lightgbm"])
def test_tree_shap_matches_the_shap_package_exactly(is_xgboost: bool) -> None:
    x, y = _dataset()
    model = _xgboost(x, y) if is_xgboost else _lightgbm(x, y)
    values, _ = tree_attributions(model, x, is_xgboost=is_xgboost)
    oracle = np.asarray(shap.TreeExplainer(model).shap_values(x), dtype=np.float64)
    assert np.abs(values - oracle).max() < 1e-9


@pytest.mark.parametrize("is_xgboost", [True, False], ids=["xgboost", "lightgbm"])
def test_tree_shap_reconstructs_the_native_margin(is_xgboost: bool) -> None:
    x, y = _dataset()
    model = _xgboost(x, y) if is_xgboost else _lightgbm(x, y)
    values, base = tree_attributions(model, x, is_xgboost=is_xgboost)
    margin = (
        np.asarray(model.predict(x, output_margin=True), dtype=np.float64)  # type: ignore[attr-defined]
        if is_xgboost
        else np.asarray(model.predict_proba(x, raw_score=True), dtype=np.float64)  # type: ignore[attr-defined]
    )
    assert np.abs(base + values.sum(axis=1) - margin).max() < 1e-5


@pytest.mark.parametrize("is_xgboost", [True, False], ids=["xgboost", "lightgbm"])
def test_the_tree_base_value_is_constant_across_rows(is_xgboost: bool) -> None:
    """What makes it an expected value rather than a per-row intercept."""
    x, y = _dataset()
    model = _xgboost(x, y) if is_xgboost else _lightgbm(x, y)
    _, base = tree_attributions(model, x, is_xgboost=is_xgboost)
    _, half = tree_attributions(model, x[:20], is_xgboost=is_xgboost)
    assert base == pytest.approx(half, abs=1e-12)


def test_a_per_row_base_value_would_be_refused() -> None:
    """The rejection, driven. A varying base would silently break every aggregate."""

    class _Wobbly:
        def predict_proba(
            self, matrix: NDArray[np.float64], pred_contrib: bool = False
        ) -> NDArray[np.float64]:
            block = np.zeros((matrix.shape[0], matrix.shape[1] + 1))
            block[:, -1] = np.arange(matrix.shape[0], dtype=np.float64)
            return block

    with pytest.raises(AttributionError, match="base value that varies across rows"):
        tree_attributions(_Wobbly(), np.zeros((4, FEATURES)), is_xgboost=False)


def test_a_misshapen_contribution_block_is_refused() -> None:
    class _Narrow:
        def predict_proba(
            self, matrix: NDArray[np.float64], pred_contrib: bool = False
        ) -> NDArray[np.float64]:
            return np.zeros((matrix.shape[0], matrix.shape[1]))

    with pytest.raises(AttributionError, match=r"expected an \[n, 7\] contribution block"):
        tree_attributions(_Narrow(), np.zeros((4, FEATURES)), is_xgboost=False)


# --- 3. permutation SHAP, against the definition itself ----------------------


def _brute_force_shapley(
    predict: object, row: NDArray[np.float64], background: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Exact interventional Shapley values by enumerating all 2^M subsets.

    The definition, not another implementation of it:

        phi_j = sum over S excluding j of
                |S|! (M - |S| - 1)! / M! * (v(S + j) - v(S))

    where ``v(S)`` is the model's mean output when the columns in ``S`` are taken from the
    explained row and the rest from the background. Feasible only for a handful of columns,
    which is exactly why the production path samples instead.
    """
    n_features = len(row)
    columns = range(n_features)

    def value(subset: frozenset[int]) -> float:
        block = background.copy()
        for column in subset:
            block[:, column] = row[column]
        return float(np.mean(predict(block)))  # type: ignore[operator]

    cache = {
        frozenset(combination): value(frozenset(combination))
        for size in range(n_features + 1)
        for combination in itertools.combinations(columns, size)
    }

    out = np.zeros(n_features, dtype=np.float64)
    for j in columns:
        others = [c for c in columns if c != j]
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(n_features - size - 1)
            weight /= math.factorial(n_features)
            for combination in itertools.combinations(others, size):
                subset = frozenset(combination)
                out[j] += weight * (cache[subset | {j}] - cache[subset])
    return out


def test_permutation_shap_converges_on_brute_force_shapley() -> None:
    """The definitional oracle. A wrong weighting or a lost antithetic pass fails here."""
    rng = np.random.default_rng(SEED)
    weights = rng.normal(size=FEATURES)
    interaction = rng.normal(size=(FEATURES, FEATURES)) * 0.3

    def predict(block: NDArray[np.float64]) -> NDArray[np.float64]:
        # Deliberately non-additive: a purely linear model would be matched by any
        # attribution scheme that happens to sum correctly.
        return block @ weights + np.einsum("ij,jk,ik->i", block, interaction, block) * 0.1

    background = rng.normal(size=(24, FEATURES))
    rows = rng.normal(size=(3, FEATURES))

    values, base = permutation_attributions(predict, rows, background, rounds=400, seed=SEED)
    assert abs(base - float(np.mean(predict(background)))) < 1e-12

    for index in range(rows.shape[0]):
        exact = _brute_force_shapley(predict, rows[index], background)
        assert np.abs(values[index] - exact).max() < 0.02 * np.abs(exact).max()


def test_additivity_holds_at_one_round_and_is_therefore_not_evidence_of_accuracy() -> None:
    """The trap, asserted so nobody later reads a green additivity check as an accuracy check.

    One round reconstructs the output exactly and is still visibly far from the converged
    answer. Both halves are asserted; the second is what stops the first being misread.
    """
    rng = np.random.default_rng(SEED)
    weights = rng.normal(size=FEATURES)

    def predict(block: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.tanh(block @ weights) * 3.0

    background = rng.normal(size=(16, FEATURES))
    rows = rng.normal(size=(8, FEATURES))
    output = predict(rows)

    coarse, base_coarse = permutation_attributions(predict, rows, background, rounds=1, seed=1)
    fine, base_fine = permutation_attributions(predict, rows, background, rounds=200, seed=1)

    assert np.abs(base_coarse + coarse.sum(axis=1) - output).max() < 1e-9
    assert np.abs(base_fine + fine.sum(axis=1) - output).max() < 1e-9
    assert np.abs(coarse - fine).max() > 1e-3, (
        "one round must still be visibly wrong; if it were not, this assertion would be "
        "the only thing standing between a reader and 'additivity proves accuracy'"
    )


def test_permutation_shap_is_reproducible_at_a_fixed_seed() -> None:
    rng = np.random.default_rng(SEED)
    weights = rng.normal(size=FEATURES)

    def predict(block: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.tanh(block @ weights)

    background = rng.normal(size=(12, FEATURES))
    rows = rng.normal(size=(5, FEATURES))
    first, base_a = permutation_attributions(predict, rows, background, rounds=4, seed=7)
    second, base_b = permutation_attributions(predict, rows, background, rounds=4, seed=7)
    third, _ = permutation_attributions(predict, rows, background, rounds=4, seed=8)

    assert np.array_equal(first, second), "same seed must give bit-identical values"
    assert base_a == base_b
    assert not np.array_equal(first, third), "a different seed must actually differ"


def test_permutation_shap_refuses_an_empty_background_or_zero_rounds() -> None:
    def predict(block: NDArray[np.float64]) -> NDArray[np.float64]:
        return block.sum(axis=1)

    rows = np.ones((2, FEATURES))
    with pytest.raises(AttributionError, match="non-empty background"):
        permutation_attributions(predict, rows, np.zeros((0, FEATURES)), rounds=2, seed=1)
    with pytest.raises(AttributionError, match="rounds must be positive"):
        permutation_attributions(predict, rows, np.zeros((3, FEATURES)), rounds=0, seed=1)


# --- 4. direction and separation ---------------------------------------------


def test_a_feature_that_raises_the_output_gets_a_positive_attribution() -> None:
    """Sign is not a convention here: a flipped sign would invert every local explanation."""
    background = np.zeros((8, FEATURES))
    row = np.zeros((1, FEATURES))
    row[0, 0] = 5.0

    def predict(block: NDArray[np.float64]) -> NDArray[np.float64]:
        return block[:, 0] * 2.0

    values, base = permutation_attributions(predict, row, background, rounds=4, seed=1)
    assert base == pytest.approx(0.0, abs=1e-12)
    assert values[0, 0] == pytest.approx(10.0, abs=1e-9)
    assert np.abs(values[0, 1:]).max() < 1e-9


def test_positive_and_negative_contributions_separate_cleanly() -> None:
    background = np.zeros((8, FEATURES))
    row = np.zeros((1, FEATURES))
    row[0, 0], row[0, 1] = 3.0, 3.0

    def predict(block: NDArray[np.float64]) -> NDArray[np.float64]:
        return block[:, 0] - block[:, 1]

    values, _ = permutation_attributions(predict, row, background, rounds=4, seed=1)
    assert values[0, 0] > 0 > values[0, 1]
    assert float(np.clip(values, 0, None).sum()) == pytest.approx(3.0, abs=1e-9)
    assert float(np.clip(values, None, 0).sum()) == pytest.approx(-3.0, abs=1e-9)
