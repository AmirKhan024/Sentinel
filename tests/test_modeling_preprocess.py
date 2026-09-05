"""Matrix construction and the train-only preprocessing pipeline.

The property these tests exist to protect: the matrix width and the meaning of every
column are declared, not observed. `SimpleImputer(add_indicator=True)` would decide the
indicator set by looking at which columns happen to have nulls in a given training
window, so on a small fixture the width would change between folds and every coefficient
term would silently be renamed. Several tests below would fail if anyone reintroduced it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl
import pytest

from sentinel.features.definitions import FEATURE_COLUMNS, NullRule
from sentinel.modeling import preprocess
from sentinel.modeling.definitions import (
    MODELS_BY_NAME,
    MissingStrategy,
    boolean_columns,
    columns_in_family,
    family_indicator_name,
    indicator_columns,
    nullable_columns,
)
from tests.conftest import make_model_feature_row, model_feature_scenario

PRIMARY = MODELS_BY_NAME["logistic_regression"]
CDPH = MODELS_BY_NAME["cdph_2015_approximation"]


def _frame(*rows: dict[str, object]) -> pl.DataFrame:
    return model_feature_scenario(list(rows))


# --- matrix shape -----------------------------------------------------------


def test_matrix_is_features_plus_four_indicators() -> None:
    columns = preprocess.matrix_columns(PRIMARY)
    assert len(columns) == len(FEATURE_COLUMNS) + 4
    assert columns[: len(FEATURE_COLUMNS)] == FEATURE_COLUMNS
    assert columns[len(FEATURE_COLUMNS) :] == indicator_columns()


def test_every_model_gets_all_four_indicators_whatever_its_features() -> None:
    """The CDPH subset drops nullable columns but keeps the full indicator set.

    Matrix width is a property of the rule set, not of the feature subset or the fold. A
    varying width would mean coefficient terms shifting between models and folds.
    """
    for spec in (PRIMARY, MODELS_BY_NAME["logistic_regression_no_scheduling"], CDPH):
        assert set(indicator_columns()) <= set(preprocess.matrix_columns(spec))
        assert len(preprocess.matrix_columns(spec)) == len(spec.feature_columns) + 4


def test_matrix_width_is_the_same_for_a_frame_with_no_nulls_at_all() -> None:
    """This is the add_indicator test: `features="missing-only"` would emit zero
    indicator columns here and four on a frame that has nulls."""
    complete = _frame(make_model_feature_row(0), make_model_feature_row(1))
    assert complete.select(nullable_columns()).null_count().sum_horizontal()[0] == 0
    matrix = preprocess.to_matrix(complete, PRIMARY)
    assert matrix.shape == (2, len(FEATURE_COLUMNS) + 4)


def test_matrix_rows_match_frame_rows() -> None:
    frame = _frame(*[make_model_feature_row(i) for i in range(7)])
    assert preprocess.to_matrix(frame, PRIMARY).shape[0] == 7


# --- null to NaN conversion, in one place ------------------------------------


def test_nulls_survive_as_nan_and_are_not_imputed_here() -> None:
    """`to_matrix` only converts. Imputation is the pipeline's job, fitted on train."""
    frame = _frame(make_model_feature_row(0, history="none"))
    matrix = preprocess.to_matrix(frame, PRIMARY)
    columns = preprocess.matrix_columns(PRIMARY)
    position = columns.index("days_since_last_canvass")
    assert np.isnan(matrix[0, position])


def test_booleans_become_floats() -> None:
    frame = _frame(
        make_model_feature_row(0, fail_at_last_canvass=True),
        make_model_feature_row(1, fail_at_last_canvass=False),
    )
    matrix = preprocess.to_matrix(frame, PRIMARY)
    position = preprocess.matrix_columns(PRIMARY).index("fail_at_last_canvass")
    assert matrix[0, position] == 1.0
    assert matrix[1, position] == 0.0
    assert matrix.dtype == np.float64


def test_null_boolean_becomes_nan_not_false() -> None:
    """A null boolean must be distinguishable from an observed False at this stage;
    collapsing them here would make the indicator meaningless."""
    frame = _frame(make_model_feature_row(0, history="none"))
    matrix = preprocess.to_matrix(frame, PRIMARY)
    position = preprocess.matrix_columns(PRIMARY).index("fail_at_last_canvass")
    assert np.isnan(matrix[0, position])


# --- the four family indicators ---------------------------------------------


@pytest.mark.parametrize(
    ("history", "expected_families"),
    [
        ("full", set()),
        (
            "none",
            {
                NullRule.NO_PRIOR_CANVASS,
                NullRule.NO_CODE_ERA_CANVASS,
                NullRule.NO_INSPECTED_CANVASS,
                NullRule.NO_PRIOR_INSPECTION,
            },
        ),
        ("no_code_era", {NullRule.NO_CODE_ERA_CANVASS}),
        ("no_inspected_canvass", {NullRule.NO_INSPECTED_CANVASS}),
    ],
)
def test_each_null_rule_family_lights_its_own_indicator(
    history: str, expected_families: set[NullRule]
) -> None:
    """Every one of Component 4's four NULL patterns, exercised."""
    frame = _frame(make_model_feature_row(0, history=history))
    matrix = preprocess.to_matrix(frame, PRIMARY)
    columns = preprocess.matrix_columns(PRIMARY)
    for rule in (
        NullRule.NO_PRIOR_CANVASS,
        NullRule.NO_CODE_ERA_CANVASS,
        NullRule.NO_INSPECTED_CANVASS,
        NullRule.NO_PRIOR_INSPECTION,
    ):
        position = columns.index(family_indicator_name(rule))
        expected = 1.0 if rule in expected_families else 0.0
        assert matrix[0, position] == expected, f"{rule.name} indicator"


def test_indicator_is_identical_for_every_member_of_a_family() -> None:
    """The four-indicator design depends on this; validate re-asserts it per run."""
    frame = _frame(
        make_model_feature_row(0, history="none"),
        make_model_feature_row(1),
        make_model_feature_row(2, history="no_code_era"),
    )
    for rule in (NullRule.NO_PRIOR_CANVASS, NullRule.NO_CODE_ERA_CANVASS):
        members = columns_in_family(rule)
        masks = [frame[member].is_null().to_list() for member in members]
        assert all(mask == masks[0] for mask in masks)


def test_indicator_works_for_a_model_that_dropped_the_family_member() -> None:
    """The CDPH model has no CONTEXT features, so its NO_PRIOR_INSPECTION indicator is
    computed from a column it does not use. The mask is identical within the family, so
    it gains no information by it -- but it must still be correct."""
    frame = _frame(make_model_feature_row(0, history="none"))
    assert "days_since_any_inspection" not in CDPH.feature_columns
    matrix = preprocess.to_matrix(frame, CDPH)
    position = preprocess.matrix_columns(CDPH).index(
        family_indicator_name(NullRule.NO_PRIOR_INSPECTION)
    )
    assert matrix[0, position] == 1.0


# --- branch ordering, which labels the coefficients -------------------------


def test_ordered_columns_are_a_permutation_of_the_matrix_columns() -> None:
    """A mismatch here would mislabel every coefficient while predicting identically."""
    for spec in MODELS_BY_NAME.values():
        assert sorted(preprocess.ordered_matrix_columns(spec)) == sorted(
            preprocess.matrix_columns(spec)
        )


def test_ordered_columns_follow_the_declared_branch_order() -> None:
    ordered = preprocess.ordered_matrix_columns(PRIMARY)
    strategies = [preprocess.strategy_for(name) for name in ordered]
    boundaries = [preprocess.BRANCH_ORDER.index(s) for s in strategies]
    assert boundaries == sorted(boundaries)


def test_indicators_are_treated_as_never_null() -> None:
    for name in indicator_columns():
        assert preprocess.strategy_for(name) is MissingStrategy.PASSTHROUGH


# --- the fitted pipeline ----------------------------------------------------


def _fit_preprocessor(frame: pl.DataFrame, spec=PRIMARY):  # type: ignore[no-untyped-def]
    pipeline = preprocess.build_preprocessor(spec)
    pipeline.fit(preprocess.to_matrix(frame, spec))
    return pipeline


def test_median_is_taken_from_the_frame_it_was_fitted_on() -> None:
    rows = [
        make_model_feature_row(i, days_since_last_canvass=v) for i, v in enumerate([10, 20, 30])
    ]
    rows.append(make_model_feature_row(3, history="none"))
    pipeline = _fit_preprocessor(_frame(*rows))
    values = preprocess.imputed_values(pipeline, PRIMARY)
    assert values["days_since_last_canvass"] == pytest.approx(20.0)


def test_a_different_training_frame_gives_a_different_median() -> None:
    """The statistic is a property of the training rows, which is the whole guarantee."""
    low = _frame(
        *[make_model_feature_row(i, days_since_last_canvass=v) for i, v in enumerate([1, 2, 3])],
        make_model_feature_row(3, history="none"),
    )
    high = _frame(
        *[
            make_model_feature_row(i, days_since_last_canvass=v)
            for i, v in enumerate([100, 200, 300])
        ],
        make_model_feature_row(3, history="none"),
    )
    assert preprocess.imputed_values(_fit_preprocessor(low), PRIMARY)[
        "days_since_last_canvass"
    ] == pytest.approx(2.0)
    assert preprocess.imputed_values(_fit_preprocessor(high), PRIMARY)[
        "days_since_last_canvass"
    ] == pytest.approx(200.0)


def test_nullable_booleans_fill_with_constant_zero() -> None:
    """Not the median: a near-50/50 boolean's median fill flips between folds."""
    rows = [make_model_feature_row(i, fail_at_last_canvass=True) for i in range(5)]
    rows.append(make_model_feature_row(9, history="none"))
    values = preprocess.imputed_values(_fit_preprocessor(_frame(*rows)), PRIMARY)
    nullable_booleans = set(boolean_columns()) & set(nullable_columns())
    for column in nullable_booleans:
        assert values[column] == 0.0, column


def test_median_imputation_is_order_invariant() -> None:
    """The median is partition-based, unlike the mean. This is why it was chosen."""
    values = [
        make_model_feature_row(i, days_since_last_canvass=v) for i, v in enumerate([5, 1, 9, 3])
    ]
    values.append(make_model_feature_row(4, history="none"))
    forward = preprocess.imputed_values(_fit_preprocessor(_frame(*values)), PRIMARY)
    backward = preprocess.imputed_values(_fit_preprocessor(_frame(*reversed(values))), PRIMARY)
    assert forward == backward


def test_transformed_matrix_has_no_nan_left() -> None:
    frame = _frame(
        make_model_feature_row(0, history="none"),
        make_model_feature_row(1, history="no_code_era"),
        make_model_feature_row(2),
        make_model_feature_row(3, history="no_inspected_canvass"),
    )
    pipeline = _fit_preprocessor(frame)
    transformed = pipeline.transform(preprocess.to_matrix(frame, PRIMARY))
    assert not np.isnan(transformed).any()


def test_transformed_width_matches_the_declared_column_count() -> None:
    frame = _frame(*[make_model_feature_row(i) for i in range(4)])
    pipeline = _fit_preprocessor(frame)
    transformed = pipeline.transform(preprocess.to_matrix(frame, PRIMARY))
    assert transformed.shape[1] == len(preprocess.ordered_matrix_columns(PRIMARY))


# --- failure modes ----------------------------------------------------------


def test_a_missing_declared_feature_fails_loudly() -> None:
    frame = _frame(make_model_feature_row(0)).drop("prior_canvass_count")
    with pytest.raises(preprocess.PreprocessError, match="missing required column"):
        preprocess.to_matrix(frame, PRIMARY)


def test_a_missing_indicator_source_fails_loudly() -> None:
    """The CDPH model does not use days_since_any_inspection, but its indicator does."""
    frame = _frame(make_model_feature_row(0)).drop("days_since_any_inspection")
    with pytest.raises(preprocess.PreprocessError, match="missing required column"):
        preprocess.to_matrix(frame, CDPH)


def test_a_spec_naming_an_unknown_column_fails_loudly() -> None:
    rogue = dataclasses.replace(PRIMARY, feature_columns=("no_such_feature",))
    with pytest.raises(preprocess.PreprocessError, match="missing required column"):
        preprocess.to_matrix(_frame(make_model_feature_row(0)), rogue)
