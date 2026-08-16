"""Feature specifications.

These are cheap tests over metadata, but they enforce the property that makes the
rest of the component maintainable: a column cannot exist without a
specification, and a specification cannot exist without a stated missing-value
rule and a reason for the feature to be there at all.
"""

from __future__ import annotations

import pytest

from sentinel.features.definitions import (
    FEATURE_COLUMNS,
    FEATURE_DEFINITION_VERSION,
    FEATURE_SPECS,
    KEY_COLUMNS,
    LABEL_COLUMNS,
    PROVENANCE_COLUMNS,
    WINDOW_DAYS,
    Family,
    FeatureSpec,
    NullRule,
    spec_by_name,
    specs_in_family,
)


@pytest.mark.parametrize("spec", FEATURE_SPECS, ids=lambda s: s.name)
def test_every_spec_is_complete(spec: FeatureSpec) -> None:
    assert spec.name
    assert spec.dtype in {"int32", "float64", "bool"}
    assert isinstance(spec.family, Family)
    assert isinstance(spec.null_rule, NullRule)
    assert spec.sources


@pytest.mark.parametrize("spec", FEATURE_SPECS, ids=lambda s: s.name)
def test_every_spec_explains_itself(spec: FeatureSpec) -> None:
    """A feature that cannot say why it might predict the target should not exist."""
    assert len(spec.description) > 80, f"{spec.name} needs a real description"


@pytest.mark.parametrize("spec", FEATURE_SPECS, ids=lambda s: s.name)
def test_feature_names_are_explicit_not_vague(spec: FeatureSpec) -> None:
    vague = {"score", "risk", "recent_behavior", "history_score", "feature"}
    assert spec.name not in vague
    assert spec.name.islower()
    assert " " not in spec.name


def test_names_are_unique() -> None:
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_feature_count_is_stable() -> None:
    """A change here is a contract change and must be deliberate."""
    assert len(FEATURE_SPECS) == 26


def test_features_and_labels_are_disjoint() -> None:
    assert not set(FEATURE_COLUMNS) & set(LABEL_COLUMNS)


def test_features_and_keys_are_disjoint() -> None:
    assert not set(FEATURE_COLUMNS) & set(KEY_COLUMNS)


def test_features_and_provenance_are_disjoint() -> None:
    assert not set(FEATURE_COLUMNS) & set(PROVENANCE_COLUMNS)


def test_inspection_date_is_a_key_not_a_feature() -> None:
    """It is the as-of boundary, so a model must not consume it directly."""
    assert "inspection_date" in KEY_COLUMNS
    assert "inspection_date" not in FEATURE_COLUMNS


def test_code_era_phase_is_provenance_not_a_feature() -> None:
    """It describes the regulatory regime, not the establishment."""
    assert "code_era_phase" in PROVENANCE_COLUMNS
    assert "code_era_phase" not in FEATURE_COLUMNS


def test_every_family_has_at_least_one_feature() -> None:
    for family in Family:
        assert specs_in_family(family), f"{family.value} is empty"


def test_counts_are_never_nullable() -> None:
    """Rule 1: 0 is a true observation for a count."""
    for spec in FEATURE_SPECS:
        is_count = (
            spec.dtype == "int32"
            and not spec.name.startswith("days_")
            and spec.family is not Family.PRIORITY_HISTORY
        )
        if is_count:
            assert spec.null_rule is NullRule.NEVER, spec.name


def test_recency_features_are_nullable() -> None:
    """Rule 2: 0 would mean 'today', which is the opposite of 'never'."""
    for spec in FEATURE_SPECS:
        if spec.name.startswith("days_since_"):
            assert spec.null_rule is not NullRule.NEVER, spec.name


def test_rate_features_are_nullable() -> None:
    """Rule 3: 0/0 is not 0."""
    for spec in FEATURE_SPECS:
        if spec.name.endswith("_rate"):
            assert spec.dtype == "float64"
            assert spec.null_rule is not NullRule.NEVER, spec.name


def test_priority_features_depend_on_code_era_history() -> None:
    """Priority did not exist before 2018-07-01 (ADR 0009)."""
    for spec in specs_in_family(Family.PRIORITY_HISTORY):
        assert spec.null_rule is NullRule.NO_CODE_ERA_CANVASS, spec.name


def test_every_window_has_a_paired_canvass_count() -> None:
    """A priority-event count of 0 is only interpretable beside its denominator."""
    for days in WINDOW_DAYS:
        assert f"canvasses_last_{days}d" in FEATURE_COLUMNS
        assert f"canvass_priority_events_last_{days}d" in FEATURE_COLUMNS


def test_windows_are_ordered_and_distinct() -> None:
    assert list(WINDOW_DAYS) == sorted(set(WINDOW_DAYS))


def test_window_sizes_match_the_documented_choice() -> None:
    assert WINDOW_DAYS == (365, 730, 1095)


def test_spec_lookup_works() -> None:
    assert spec_by_name("prior_canvass_count").family is Family.CANVASS_HISTORY


def test_spec_lookup_raises_for_unknown() -> None:
    with pytest.raises(KeyError):
        spec_by_name("not_a_feature")


def test_definition_version_is_set() -> None:
    assert FEATURE_DEFINITION_VERSION == "v1"


def test_no_model_derived_features() -> None:
    """Component 6+ territory; Component 4 is deterministic history only."""
    banned = ("predicted_", "model_", "embedding", "score", "proba")
    for spec in FEATURE_SPECS:
        assert not spec.name.startswith(banned), spec.name
        assert "embedding" not in spec.name


def test_no_demographic_features() -> None:
    """Spec §8: demographics are audit-only, never model inputs."""
    banned = ("income", "race", "ethnic", "poverty", "demographic", "acs_")
    for spec in FEATURE_SPECS:
        assert not any(b in spec.name for b in banned), spec.name
