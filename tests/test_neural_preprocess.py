"""The network's input matrix, and the guarantee that it is Component 6's.

The claim these tests exist to protect is narrow and load-bearing: the numeric half of
Component 8's matrix is *the same object* Components 6 and 7 build, produced by the same
code with the same rules. If that stopped being true, every C6/C7/C8 comparison would be
confounded between "the estimator changed" and "the matrix changed", and nothing in a
metric would reveal it.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sentinel.boosting import preprocess as tree_preprocess
from sentinel.boosting.definitions import spec_for as boosting_spec_for
from sentinel.modeling import preprocess as baseline_preprocess
from sentinel.modeling.definitions import indicator_columns
from sentinel.modeling.definitions import spec_for as modeling_spec_for
from sentinel.neural import encode, preprocess
from sentinel.neural.definitions import spec_for
from tests.conftest import (
    make_model_feature_row,
    model_feature_scenario,
    neural_categoricals_for,
    spanning_model_features,
)

PRIMARY = spec_for("neural_embeddings")
ONEHOT = spec_for("neural_onehot")
NUMERIC_ONLY = spec_for("neural_numeric_only")


def _frame() -> pl.DataFrame:
    rows = [make_model_feature_row(i, history="full") for i in range(20)]
    rows.extend(make_model_feature_row(100 + i, history="none") for i in range(6))
    rows.extend(make_model_feature_row(200 + i, history="no_code_era") for i in range(6))
    rows.extend(make_model_feature_row(300 + i, history="no_inspected_canvass") for i in range(6))
    return model_feature_scenario(rows)


def _with_categoricals(frame: pl.DataFrame) -> pl.DataFrame:
    cats = neural_categoricals_for(frame)
    return frame.join(
        cats.select("target_inspection_id", "chain_key", "facility_type", "community_area", "zip"),
        on="target_inspection_id",
        how="left",
    )


# --- 1. the matrix is Component 6's and Component 7's -------------------------


def test_the_numeric_matrix_columns_are_component_6s() -> None:
    """Same names, same order, from the same function."""
    assert preprocess.matrix_columns(NUMERIC_ONLY) == baseline_preprocess.matrix_columns(
        modeling_spec_for("logistic_regression")
    )


def test_the_numeric_matrix_columns_are_component_7s() -> None:
    assert preprocess.matrix_columns(NUMERIC_ONLY) == tree_preprocess.matrix_columns(
        boosting_spec_for("xgboost")
    )


def test_the_matrix_is_26_features_plus_4_family_indicators() -> None:
    columns = preprocess.matrix_columns(PRIMARY)
    assert len(columns) == 30
    assert columns[-4:] == indicator_columns()


def test_the_four_family_indicators_survive_into_the_network() -> None:
    """They are how missingness survives imputation.

    Component 7 keeps them so the matrices match; Component 8 keeps them for that reason
    AND because they are the only way a NULL's meaning reaches a dense layer -- a network
    cannot route a NaN the way a tree can.
    """
    frame = _frame()
    matrix = preprocess.numeric_matrix(frame, PRIMARY)
    columns = preprocess.matrix_columns(PRIMARY)
    for name in indicator_columns():
        index = columns.index(name)
        values = matrix[:, index]
        assert set(np.unique(values)) <= {0.0, 1.0}
    # And at least one indicator actually fires, or the test is vacuous.
    indicator_block = matrix[:, -4:]
    assert indicator_block.sum() > 0, "no missingness indicator fired; the fixture is too clean"


def test_nulls_arrive_as_nan_before_imputation() -> None:
    frame = _frame()
    matrix = preprocess.numeric_matrix(frame, PRIMARY)
    assert np.isnan(matrix).any(), "the fixture produced no NULLs; later tests are vacuous"
    assert preprocess.nan_cells(frame, PRIMARY) == int(np.count_nonzero(np.isnan(matrix)))


# --- 2. fitting is separate from applying ------------------------------------


def test_the_preprocessor_must_be_fitted_before_it_is_applied() -> None:
    """There is deliberately no ``fit_transform`` path in this module.

    A one-call convenience is exactly the shape a leak takes: called on a test window it
    would fit a scaler on test rows and produce a plausible, better, wrong number.
    """
    from sklearn.exceptions import NotFittedError

    assert not hasattr(preprocess, "fit_transform")
    frame = _frame()
    unfitted = preprocess.build_preprocessor(PRIMARY)
    with pytest.raises(NotFittedError):
        preprocess.apply_preprocessor(unfitted, frame, PRIMARY)


def test_applying_a_fitted_preprocessor_leaves_no_nan() -> None:
    """A NaN reaching a dense layer turns every downstream weight to NaN in one step."""
    frame = _frame()
    fitted = preprocess.build_preprocessor(PRIMARY)
    fitted.fit(preprocess.numeric_matrix(frame, PRIMARY))
    out = preprocess.apply_preprocessor(fitted, frame, PRIMARY)
    assert np.all(np.isfinite(out))


def test_statistics_come_from_the_window_the_preprocessor_was_fitted_on() -> None:
    """Fit on one half, and confirm the medians are that half's."""
    frame = _frame()
    first = frame.head(20)
    fitted = preprocess.build_preprocessor(PRIMARY)
    fitted.fit(preprocess.numeric_matrix(first, PRIMARY))
    values = preprocess.imputed_values(fitted, PRIMARY)

    checked = 0
    for column, fill in values.items():
        if column not in first.columns:
            continue
        strategy = baseline_preprocess.strategy_for(column)
        if strategy.value == "constant_false":
            assert fill == 0.0
            checked += 1
            continue
        median = first[column].cast(pl.Float64).median()
        if median is None:
            continue
        assert abs(fill - float(median)) < 1e-9
        checked += 1
    assert checked >= 4


def test_the_scaler_standardises() -> None:
    frame = spanning_model_features(days=200, per_day=3)
    fitted = preprocess.build_preprocessor(NUMERIC_ONLY)
    fitted.fit(preprocess.numeric_matrix(frame, NUMERIC_ONLY))
    out = preprocess.apply_preprocessor(fitted, frame, NUMERIC_ONLY)
    means = out.mean(axis=0)
    # Constant columns stay at zero variance and therefore at mean zero after centring.
    assert np.allclose(means, 0.0, atol=1e-8)


def test_scaler_statistics_are_plain_typed_python() -> None:
    """Carried on the fit so a re-run can be checked without unpickling an estimator."""
    frame = _frame()
    fitted = preprocess.build_preprocessor(PRIMARY)
    fitted.fit(preprocess.numeric_matrix(frame, PRIMARY))
    mean, scale = preprocess.scaler_statistics(fitted)
    assert len(mean) == len(scale) == 30
    assert all(isinstance(v, float) for v in mean)


# --- 3. the categorical block ------------------------------------------------


def test_an_embedding_spec_gets_codes_and_no_extra_dense_columns() -> None:
    """Embedding rows are parameters, so they must not appear as dense inputs."""
    frame = _with_categoricals(_frame())
    encoding = encode.fit_encoding(frame, PRIMARY)
    fitted = preprocess.build_preprocessor(PRIMARY)
    fitted.fit(preprocess.numeric_matrix(frame, PRIMARY))

    dense = preprocess.dense_matrix(frame, PRIMARY, fitted, encoding)
    codes = preprocess.code_matrix(frame, PRIMARY, encoding)
    assert dense.shape[1] == 30
    assert codes.shape == (frame.height, len(PRIMARY.entity_columns))
    assert preprocess.dense_width(PRIMARY, encoding) == 30


def test_a_onehot_spec_gets_dense_indicators_and_no_codes() -> None:
    frame = _with_categoricals(_frame())
    encoding = encode.fit_encoding(frame, ONEHOT)
    fitted = preprocess.build_preprocessor(ONEHOT)
    fitted.fit(preprocess.numeric_matrix(frame, ONEHOT))

    dense = preprocess.dense_matrix(frame, ONEHOT, fitted, encoding)
    codes = preprocess.code_matrix(frame, ONEHOT, encoding)
    assert codes.shape[1] == 0
    assert dense.shape[1] == 30 + sum(encoding.sizes.values())
    assert preprocess.dense_width(ONEHOT, encoding) == dense.shape[1]


def test_a_numeric_only_spec_gets_neither() -> None:
    frame = _frame()
    encoding = encode.fit_encoding(frame, NUMERIC_ONLY)
    fitted = preprocess.build_preprocessor(NUMERIC_ONLY)
    fitted.fit(preprocess.numeric_matrix(frame, NUMERIC_ONLY))
    dense = preprocess.dense_matrix(frame, NUMERIC_ONLY, fitted, encoding)
    assert dense.shape[1] == 30
    assert preprocess.code_matrix(frame, NUMERIC_ONLY, encoding).shape[1] == 0


def test_the_indicator_block_is_never_scaled_away() -> None:
    """A constant indicator would scale to NaN or to zero; neither may reach the network."""
    frame = _with_categoricals(_frame())
    encoding = encode.fit_encoding(frame, ONEHOT)
    fitted = preprocess.build_preprocessor(ONEHOT)
    fitted.fit(preprocess.numeric_matrix(frame, ONEHOT))
    dense = preprocess.dense_matrix(frame, ONEHOT, fitted, encoding)
    indicators = dense[:, 30:]
    assert np.all(np.isfinite(indicators))
    assert set(np.unique(indicators)) <= {0.0, 1.0}


# --- 4. column naming --------------------------------------------------------


def test_transformed_columns_follow_the_transformer_branch_order() -> None:
    """The permutation is real, and getting it wrong mislabels silently."""
    frame = _frame()
    encoding = encode.fit_encoding(frame, NUMERIC_ONLY)
    named = preprocess.transformed_columns(NUMERIC_ONLY, encoding)
    assert named == baseline_preprocess.ordered_matrix_columns(
        modeling_spec_for("logistic_regression")
    )
    assert set(named) == set(preprocess.matrix_columns(NUMERIC_ONLY))


def test_transformed_columns_append_the_indicator_names_for_the_onehot_control() -> None:
    frame = _with_categoricals(_frame())
    encoding = encode.fit_encoding(frame, ONEHOT)
    named = preprocess.transformed_columns(ONEHOT, encoding)
    assert len(named) == 30 + sum(encoding.sizes.values())
    assert named[30].startswith("chain=")


def test_a_frame_missing_a_feature_column_is_refused() -> None:
    frame = _frame().drop("prior_canvass_count")
    with pytest.raises(preprocess.NeuralPreprocessError, match="missing required column"):
        preprocess.numeric_matrix(frame, PRIMARY)


def test_the_matrix_is_deterministic() -> None:
    frame = _frame()
    first = preprocess.numeric_matrix(frame, PRIMARY)
    second = preprocess.numeric_matrix(frame, PRIMARY)
    assert np.array_equal(first, second, equal_nan=True)
