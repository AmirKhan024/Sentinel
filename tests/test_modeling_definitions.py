"""The model registry and the feature partitions derived from Component 4.

These tests pin the partition to measured facts about the real feature table: 10
nullable features in 4 families, 16 never-null. If Component 4 adds a feature or
changes a null rule, these fail -- which is the point. The preprocessing depends on the
partition, and a silent change there would alter what every coefficient means without
altering a single line of Component 6.
"""

from __future__ import annotations

import dataclasses

import pytest

from sentinel.features.definitions import (
    FEATURE_COLUMNS,
    KEY_COLUMNS,
    LABEL_COLUMNS,
    PROVENANCE_COLUMNS,
    NullRule,
)
from sentinel.modeling.definitions import (
    ABLATED_FEATURE,
    CDPH_2015_UNREACHABLE_INPUTS,
    CDPH_APPROXIMATION_NOTE,
    FORBIDDEN_COLUMNS,
    MODEL_DEFINITION_VERSION,
    MODEL_REGISTRY,
    MODELS_BY_NAME,
    MissingStrategy,
    ModelSpec,
    _guard_registry,
    boolean_columns,
    columns_in_family,
    family_indicator_name,
    indicator_columns,
    indicator_source_column,
    max_iter_of,
    missing_strategy_for,
    never_null_columns,
    null_families,
    nullable_columns,
    spec_for,
)

# --- the derived partition --------------------------------------------------

EXPECTED_NULLABLE = {
    "days_since_last_canvass",
    "fail_at_last_canvass",
    "name_changed_since_last_canvass",
    "prior_canvass_fail_rate",
    "prior_canvass_priority_count",
    "prior_canvass_priority_foundation_count",
    "prior_canvass_priority_rate",
    "priority_at_last_canvass",
    "days_since_any_inspection",
    "days_since_first_inspection",
}


def test_nullable_partition_is_the_measured_ten() -> None:
    """Measured on all 57,727 rows: exactly 10 of the 26 features can be NULL."""
    assert set(nullable_columns()) == EXPECTED_NULLABLE
    assert len(nullable_columns()) == 10


def test_never_null_partition_is_the_remaining_sixteen() -> None:
    assert len(never_null_columns()) == 16
    assert set(never_null_columns()) | EXPECTED_NULLABLE == set(FEATURE_COLUMNS)
    assert not set(never_null_columns()) & EXPECTED_NULLABLE


def test_there_are_exactly_four_null_families() -> None:
    """Four rules, so four indicators -- not ten, one per nullable column."""
    assert len(null_families()) == 4
    assert set(null_families()) == {
        NullRule.NO_PRIOR_CANVASS,
        NullRule.NO_CODE_ERA_CANVASS,
        NullRule.NO_INSPECTED_CANVASS,
        NullRule.NO_PRIOR_INSPECTION,
    }


def test_null_families_are_ordered_stably() -> None:
    """Indicator column order must be a property of the rule set, not of spec order."""
    assert list(null_families()) == sorted(null_families(), key=lambda rule: rule.value)
    assert null_families() == null_families()


def test_every_nullable_column_belongs_to_exactly_one_family() -> None:
    seen: list[str] = []
    for rule in null_families():
        seen.extend(columns_in_family(rule))
    assert sorted(seen) == sorted(nullable_columns())
    assert len(seen) == len(set(seen))


def test_indicator_names_are_four_and_do_not_collide_with_features() -> None:
    names = indicator_columns()
    assert len(names) == 4
    assert len(set(names)) == 4
    assert not set(names) & set(FEATURE_COLUMNS)


def test_indicator_name_comes_from_the_enum_member_not_its_value() -> None:
    """The values are prose with spaces and a date in them; the names are identifiers."""
    assert family_indicator_name(NullRule.NO_CODE_ERA_CANVASS) == "missing_no_code_era_canvass"
    assert " " not in family_indicator_name(NullRule.NO_PRIOR_CANVASS)


def test_indicator_source_is_a_member_of_its_own_family() -> None:
    for rule in null_families():
        assert indicator_source_column(rule) in columns_in_family(rule)


def test_indicator_source_rejects_a_rule_with_no_members() -> None:
    with pytest.raises(KeyError):
        indicator_source_column(NullRule.NEVER)


# --- missing-value strategy -------------------------------------------------


def test_nullable_booleans_use_a_constant_not_a_median() -> None:
    """Measured: priority_at_last_canvass drifts 0.6310 -> 0.5056 across the 17 folds.

    A median fill sits 0.0056 from flipping, which would reverse the encoding of
    "unknown" mid-sequence for no substantive reason.
    """
    nullable_booleans = set(boolean_columns()) & set(nullable_columns())
    assert nullable_booleans == {
        "fail_at_last_canvass",
        "priority_at_last_canvass",
        "name_changed_since_last_canvass",
    }
    for column in nullable_booleans:
        assert missing_strategy_for(column) is MissingStrategy.CONSTANT_FALSE


def test_nullable_numerics_use_the_median() -> None:
    for column in set(nullable_columns()) - set(boolean_columns()):
        assert missing_strategy_for(column) is MissingStrategy.MEDIAN


def test_never_null_columns_pass_through() -> None:
    """0 is a true observation for a count and must not be imputed over."""
    for column in never_null_columns():
        assert missing_strategy_for(column) is MissingStrategy.PASSTHROUGH


def test_missing_strategy_rejects_an_unknown_column() -> None:
    with pytest.raises(KeyError, match="Unknown feature"):
        missing_strategy_for("not_a_feature")


# --- the registry -----------------------------------------------------------


def test_registry_holds_the_three_declared_baselines() -> None:
    assert [spec.name for spec in MODEL_REGISTRY] == [
        "logistic_regression",
        "logistic_regression_no_scheduling",
        "cdph_2015_approximation",
    ]


def test_model_definition_version_is_stable() -> None:
    """A bump means two runs are not comparable, so it is asserted rather than assumed."""
    assert MODEL_DEFINITION_VERSION == "v1"


def test_every_model_version_is_stable() -> None:
    assert {spec.name: spec.version for spec in MODEL_REGISTRY} == {
        "logistic_regression": "v1",
        "logistic_regression_no_scheduling": "v1",
        "cdph_2015_approximation": "v1",
    }


def test_specs_are_frozen() -> None:
    """A spec mutated after import would make the manifest a record of nothing."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        MODEL_REGISTRY[0].name = "renamed"  # type: ignore[misc]


def test_hyperparameters_are_fixed_and_identical_across_models() -> None:
    """No tuning in Component 6, and the ablation differs only in its features."""
    params = {tuple(sorted(spec.params.items())) for spec in MODEL_REGISTRY}
    assert len(params) == 1
    assert dict(MODEL_REGISTRY[0].params) == {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
    }


def test_no_model_sets_a_class_weight() -> None:
    """Prevalence is 52.52%; there is no imbalance to correct and resampling would
    corrupt the probability scale Component 9 depends on."""
    for spec in MODEL_REGISTRY:
        assert "class_weight" not in spec.params


def test_every_model_declares_a_seed_and_produces_probabilities() -> None:
    for spec in MODEL_REGISTRY:
        assert spec.seed == 42
        assert spec.is_probability is True


def test_max_iter_of_narrows_the_params_mapping() -> None:
    assert max_iter_of(MODEL_REGISTRY[0]) == 1000
    bare = dataclasses.replace(MODEL_REGISTRY[0], params={})
    assert max_iter_of(bare) == 0


# --- the feature contract per model -----------------------------------------


def test_primary_model_uses_every_feature() -> None:
    assert MODELS_BY_NAME["logistic_regression"].feature_columns == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 26


def test_ablation_differs_by_exactly_one_feature() -> None:
    primary = set(MODELS_BY_NAME["logistic_regression"].feature_columns)
    ablated = set(MODELS_BY_NAME["logistic_regression_no_scheduling"].feature_columns)
    assert primary - ablated == {ABLATED_FEATURE}
    assert not ablated - primary


def test_cdph_model_is_a_strict_subset_and_is_labelled_an_approximation() -> None:
    spec = MODELS_BY_NAME["cdph_2015_approximation"]
    assert set(spec.feature_columns) < set(FEATURE_COLUMNS)
    assert len(spec.feature_columns) == 19
    assert spec.approximation_note is not None
    assert "APPROXIMATION" in spec.approximation_note


def test_the_approximation_note_names_every_unreachable_input() -> None:
    """The caveat must travel with the artifact, not live only in a document."""
    assert len(CDPH_2015_UNREACHABLE_INPUTS) == 7
    for unreachable in CDPH_2015_UNREACHABLE_INPUTS:
        assert unreachable in CDPH_APPROXIMATION_NOTE
    for topic in ("inspector", "311", "burglary", "licence", "weather", "risk category"):
        assert topic in CDPH_APPROXIMATION_NOTE


def test_only_the_cdph_model_is_an_approximation() -> None:
    labelled = [s.name for s in MODEL_REGISTRY if s.approximation_note is not None]
    assert labelled == ["cdph_2015_approximation"]


@pytest.mark.parametrize("column", sorted(FORBIDDEN_COLUMNS))
def test_no_model_uses_a_forbidden_column(column: str) -> None:
    for spec in MODEL_REGISTRY:
        assert column not in spec.feature_columns


def test_forbidden_columns_are_keys_labels_and_provenance() -> None:
    assert set(KEY_COLUMNS) | set(LABEL_COLUMNS) | set(PROVENANCE_COLUMNS) == FORBIDDEN_COLUMNS
    assert "target" in FORBIDDEN_COLUMNS
    assert "target_inspection_id" in FORBIDDEN_COLUMNS
    assert "inspection_date" in FORBIDDEN_COLUMNS


def test_no_model_input_is_a_feature_component_4_does_not_have() -> None:
    for spec in MODEL_REGISTRY:
        assert set(spec.feature_columns) <= set(FEATURE_COLUMNS)


def test_spec_for_rejects_an_unknown_name_and_lists_the_options() -> None:
    with pytest.raises(KeyError, match="Unknown model"):
        spec_for("xgboost")


# --- the import-time guard --------------------------------------------------


def test_guard_passes_on_the_shipped_registry() -> None:
    _guard_registry()


def test_guard_raises_rather_than_asserts(monkeypatch: pytest.MonkeyPatch) -> None:
    """`assert` is stripped under `python -O`; a guard that vanishes is not a guard."""
    leaky = ModelSpec(
        name="leaky",
        version="v1",
        description="uses the label as a feature",
        feature_columns=("prior_canvass_count", "target"),
        is_probability=True,
        seed=42,
        params={"max_iter": 1000},
    )
    monkeypatch.setattr("sentinel.modeling.definitions.MODEL_REGISTRY", (leaky,))
    with pytest.raises(ValueError, match="non-feature column"):
        _guard_registry()


def test_guard_rejects_an_unknown_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    bogus = dataclasses.replace(
        MODEL_REGISTRY[0], name="bogus", feature_columns=("not_a_component_4_column",)
    )
    monkeypatch.setattr("sentinel.modeling.definitions.MODEL_REGISTRY", (bogus,))
    with pytest.raises(ValueError, match="absent from Component"):
        _guard_registry()


def test_guard_rejects_a_duplicate_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sentinel.modeling.definitions.MODEL_REGISTRY",
        (MODEL_REGISTRY[0], MODEL_REGISTRY[0]),
    )
    with pytest.raises(ValueError, match="duplicate model name"):
        _guard_registry()


def test_guard_rejects_an_empty_feature_list(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = dataclasses.replace(MODEL_REGISTRY[0], feature_columns=())
    monkeypatch.setattr("sentinel.modeling.definitions.MODEL_REGISTRY", (empty,))
    with pytest.raises(ValueError, match="feature_columns is empty"):
        _guard_registry()


def test_guard_rejects_a_duplicated_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    doubled = dataclasses.replace(
        MODEL_REGISTRY[0], feature_columns=("prior_canvass_count", "prior_canvass_count")
    )
    monkeypatch.setattr("sentinel.modeling.definitions.MODEL_REGISTRY", (doubled,))
    with pytest.raises(ValueError, match="duplicate"):
        _guard_registry()
