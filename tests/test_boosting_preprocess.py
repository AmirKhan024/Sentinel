"""The tree matrix must be the baseline's matrix with nothing done to it.

Component 6's risk was that a fill value came from the wrong window. Component 7's risk
is the opposite and larger: that a fill happened at all. A silently imputed NULL does not
raise, does not change the matrix shape and does not show up in any metric -- it just
quietly replaces "this establishment has no prior canvass" with the average of a
population it is not in, and the model gets slightly worse for a reason nobody can see.

So these tests assert absence: no imputer, no scaler, and a NaN pattern identical to the
source frame's NULL pattern, cell for cell.

They also assert the matrix is *the same* as Component 6's, because every C6-versus-C7
comparison depends on the two models having seen the same columns in the same order.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sentinel.boosting.definitions import spec_for
from sentinel.boosting.preprocess import (
    TreePreprocessError,
    matrix_columns,
    null_mask,
    positive_weight,
    tree_matrix,
)
from sentinel.modeling import preprocess as baseline_preprocess
from sentinel.modeling.definitions import MODELS_BY_NAME, indicator_columns, nullable_columns
from tests.conftest import make_model_feature_row, model_feature_scenario

PRIMARY = spec_for("xgboost")
SECONDARY = spec_for("lightgbm")
BASELINE = MODELS_BY_NAME["logistic_regression"]


def _frame(*histories: str) -> pl.DataFrame:
    return model_feature_scenario(
        [make_model_feature_row(i, history=h) for i, h in enumerate(histories)]
    )


# --- 1. the matrix is the baseline's matrix ----------------------------------


def test_the_matrix_is_thirty_columns_wide() -> None:
    """26 Component 4 features plus the four null-rule family indicators."""
    assert len(matrix_columns(PRIMARY)) == 30
    assert len(PRIMARY.feature_columns) == 26
    assert len(indicator_columns()) == 4


def test_the_boosted_and_baseline_matrices_have_identical_columns() -> None:
    """The premise of every C6-versus-C7 comparison, asserted rather than assumed."""
    assert matrix_columns(PRIMARY) == baseline_preprocess.matrix_columns(BASELINE)


def test_both_boosters_see_the_same_matrix() -> None:
    assert matrix_columns(PRIMARY) == matrix_columns(SECONDARY)


def test_the_four_indicators_come_last_and_are_the_declared_four() -> None:
    columns = matrix_columns(PRIMARY)
    assert columns[-4:] == indicator_columns()
    assert columns[:26] == PRIMARY.feature_columns


def test_the_matrix_uses_the_declared_order_not_the_transformer_branch_order() -> None:
    """Component 6 reorders to match its ColumnTransformer; there is no transformer here."""
    assert matrix_columns(PRIMARY) != baseline_preprocess.ordered_matrix_columns(BASELINE)
    assert set(matrix_columns(PRIMARY)) == set(baseline_preprocess.ordered_matrix_columns(BASELINE))


# --- 2. nothing is imputed ---------------------------------------------------


def test_a_null_survives_as_nan() -> None:
    frame = _frame("none")
    matrix = tree_matrix(frame, PRIMARY)
    columns = matrix_columns(PRIMARY)
    index = columns.index("prior_canvass_priority_rate")
    assert np.isnan(matrix[0, index])


def test_a_null_boolean_becomes_nan_not_false() -> None:
    """Component 6 fills these with 0.0. Doing so here would delete the distinction."""
    frame = _frame("none")
    matrix = tree_matrix(frame, PRIMARY)
    index = matrix_columns(PRIMARY).index("priority_at_last_canvass")
    assert np.isnan(matrix[0, index])
    assert matrix[0, index] != 0.0


def test_the_nan_mask_equals_the_frames_null_mask_cell_for_cell() -> None:
    """The check ``validate`` runs per fold, exercised directly on a mixed frame."""
    frame = _frame("full", "none", "no_code_era", "no_inspected_canvass", "full")
    assert np.array_equal(np.isnan(tree_matrix(frame, PRIMARY)), null_mask(frame, PRIMARY))


def test_every_nullable_column_can_actually_be_nan() -> None:
    """Otherwise the mask comparison above could pass on an all-present frame for free."""
    frame = _frame("none")
    matrix = tree_matrix(frame, PRIMARY)
    columns = matrix_columns(PRIMARY)
    for name in nullable_columns():
        assert np.isnan(matrix[0, columns.index(name)]), f"{name} was not NaN"


def test_the_indicators_are_never_nan() -> None:
    """An indicator is computed from another column's null mask, so it always has a value."""
    frame = _frame("none", "full")
    matrix = tree_matrix(frame, PRIMARY)
    columns = matrix_columns(PRIMARY)
    for name in indicator_columns():
        assert not np.isnan(matrix[:, columns.index(name)]).any()


def test_the_indicators_still_fire_even_though_a_booster_does_not_need_them() -> None:
    """Kept for comparability with Component 6, so they must still be correct."""
    frame = _frame("none", "full")
    matrix = tree_matrix(frame, PRIMARY)
    index = matrix_columns(PRIMARY).index("missing_no_prior_canvass")
    assert matrix[0, index] == 1.0
    assert matrix[1, index] == 0.0


# --- 3. nothing is scaled ----------------------------------------------------


def test_a_feature_reaches_the_matrix_on_its_own_scale() -> None:
    """Standardising would change nothing for a tree and would add a fitted statistic."""
    frame = model_feature_scenario([make_model_feature_row(0, days_since_last_canvass=1234)])
    matrix = tree_matrix(frame, PRIMARY)
    index = matrix_columns(PRIMARY).index("days_since_last_canvass")
    assert matrix[0, index] == 1234.0


def test_two_frames_with_different_spreads_map_a_shared_value_identically() -> None:
    """A scaler would make the same input produce different matrix values per frame."""
    narrow = model_feature_scenario(
        [make_model_feature_row(i, days_since_last_canvass=v) for i, v in enumerate([10, 11, 12])]
    )
    wide = model_feature_scenario(
        [make_model_feature_row(i, days_since_last_canvass=v) for i, v in enumerate([10, 500, 900])]
    )
    index = matrix_columns(PRIMARY).index("days_since_last_canvass")
    assert tree_matrix(narrow, PRIMARY)[0, index] == tree_matrix(wide, PRIMARY)[0, index]


# --- 4. errors ---------------------------------------------------------------


def test_a_missing_column_is_refused_rather_than_filled() -> None:
    frame = _frame("full").drop("prior_canvass_priority_rate")
    with pytest.raises(TreePreprocessError, match="missing required column"):
        tree_matrix(frame, PRIMARY)


def test_the_null_mask_refuses_a_frame_missing_a_matrix_column() -> None:
    frame = _frame("full").drop("prior_canvass_fail_rate")
    with pytest.raises(TreePreprocessError, match="missing matrix column"):
        null_mask(frame, PRIMARY)


# --- 5. the class-weighting ablation's weight --------------------------------


def test_positive_weight_is_negatives_over_positives() -> None:
    labels = np.asarray([1, 1, 1, 0], dtype=np.int64)
    assert positive_weight(labels) == pytest.approx(1 / 3)


def test_positive_weight_is_one_at_a_balanced_window() -> None:
    labels = np.asarray([1, 0, 1, 0], dtype=np.int64)
    assert positive_weight(labels) == 1.0


def test_positive_weight_refuses_to_divide_by_zero() -> None:
    """``train`` rejects a single-class window first; this is the belt on that brace."""
    assert positive_weight(np.asarray([1, 1, 1], dtype=np.int64)) == 1.0
    assert positive_weight(np.asarray([0, 0, 0], dtype=np.int64)) == 1.0
