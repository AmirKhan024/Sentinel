"""Component 12's frozen declarations, and the guard that refuses a dishonest one.

The registry is where this component states which geographies it may audit and which it
refuses. Every test here drives a *rejection*: a refused definition with no reason, a
definition claiming to be a model feature, a calibration floor below the ranking floor. A
guard whose failure path has never been observed is indistinguishable from one that cannot
fire, and Component 5 shipped exactly that defect once (``scores_respect_the_decision_point``,
declared and unreachable, fixed in ADR 0014).
"""

from __future__ import annotations

import dataclasses

import pytest

from sentinel.calibration.definitions import CANDIDATE_REGISTRY
from sentinel.evaluation.metrics import DEFAULT_CALIBRATION_BINS
from sentinel.fairness import definitions as d

# --- 1. the group registry ---------------------------------------------------


def test_exactly_two_geographies_are_audited() -> None:
    assert d.AUDITED_GROUP_DEFINITIONS == ("community_area", "zip")


def test_ward_is_refused_and_the_reason_carries_the_measurement() -> None:
    """A refusal that states no measurement is indistinguishable from an omission."""
    spec = d.GROUP_DEFINITIONS_BY_NAME["ward"]
    assert spec.status is d.GroupDefinitionStatus.REFUSED
    assert "56,451" in spec.refusal_reason
    assert "boundary version" in spec.refusal_reason


def test_every_refused_definition_names_a_reason() -> None:
    for spec in d.GROUP_DEFINITION_REGISTRY:
        if spec.status is d.GroupDefinitionStatus.REFUSED:
            assert spec.refusal_reason, f"{spec.name} refused without a reason"


def test_census_tract_and_point_geography_are_refused_for_stated_reasons() -> None:
    assert "41 rows" in d.GROUP_DEFINITIONS_BY_NAME["census_tract"].refusal_reason
    assert "less legible" in d.GROUP_DEFINITIONS_BY_NAME["point_geography"].refusal_reason


def test_asking_for_a_refused_definition_raises_with_its_reason() -> None:
    """A caller who typed `ward` is told why not, rather than that they may not."""
    with pytest.raises(d.FairnessDefinitionError, match="boundary version"):
        d.group_definition_for("ward")


def test_asking_for_an_unknown_definition_lists_the_known_ones() -> None:
    with pytest.raises(d.FairnessDefinitionError, match="community_area"):
        d.group_definition_for("neighbourhood")


def test_no_group_definition_claims_to_be_a_model_feature() -> None:
    """No geography reaches Component 4's table, and the registry may not imply one does.

    The claim matters because it is the premise of "the model does not use community area,
    therefore it is fair" -- the argument this whole component exists to be able to refute.
    """
    for spec in d.GROUP_DEFINITION_REGISTRY:
        assert spec.is_model_feature is False


def test_community_area_provenance_refuses_to_name_neighbourhoods() -> None:
    """The region id is not the official community-area number, and the spec says so."""
    provenance = d.GROUP_DEFINITIONS_BY_NAME["community_area"].provenance
    assert "computed region" in provenance
    assert "NOT necessarily" in provenance
    assert "no neighbourhood name is printed" in provenance


# --- 2. the guard --------------------------------------------------------------


def test_a_refused_definition_without_a_reason_is_rejected() -> None:
    bad = dataclasses.replace(d.GROUP_DEFINITIONS_BY_NAME["ward"], refusal_reason="")
    with pytest.raises(d.FairnessDefinitionError, match="refused without a reason"):
        _guard_with((bad,))


def test_an_audited_definition_carrying_a_refusal_reason_is_rejected() -> None:
    bad = dataclasses.replace(
        d.GROUP_DEFINITIONS_BY_NAME["community_area"], refusal_reason="changed my mind"
    )
    with pytest.raises(d.FairnessDefinitionError, match="carries a refusal reason"):
        _guard_with((bad,))


def test_a_definition_claiming_to_be_a_model_feature_is_rejected() -> None:
    bad = dataclasses.replace(d.GROUP_DEFINITIONS_BY_NAME["community_area"], is_model_feature=True)
    with pytest.raises(d.FairnessDefinitionError, match="declared as a model feature"):
        _guard_with((bad,))


def test_a_duplicate_definition_is_rejected() -> None:
    spec = d.GROUP_DEFINITIONS_BY_NAME["zip"]
    with pytest.raises(d.FairnessDefinitionError, match="duplicate group definition"):
        _guard_with((spec, spec))


def _guard_with(registry: tuple[d.GroupDefinitionSpec, ...]) -> None:
    """Run the import-time guard against a substituted registry.

    The guard reads the module global, so the substitution is made and restored rather than
    passed -- which also proves the guard reads the registry rather than a copy handed to it.
    """
    original = d.GROUP_DEFINITION_REGISTRY
    try:
        d.GROUP_DEFINITION_REGISTRY = registry  # type: ignore[misc]
        d._guard_registry()  # noqa: SLF001
    finally:
        d.GROUP_DEFINITION_REGISTRY = original  # type: ignore[misc]


# --- 3. the support policy ------------------------------------------------------


def test_the_support_floors_are_the_measured_ones() -> None:
    """Frozen from `scripts/profile_fairness.py`, before any disparity was computed."""
    assert d.SUPPORT_MIN_ROWS == 200
    assert d.SUPPORT_MIN_POSITIVE == 20
    assert d.SUPPORT_MIN_NEGATIVE == 20


def test_the_calibration_floor_is_twenty_rows_per_bin() -> None:
    """Arithmetic, not taste: 15 equal-mass bins at 20 rows each."""
    assert d.CALIBRATION_MIN_ROWS == d.GROUP_CALIBRATION_BINS * 20
    assert d.CALIBRATION_MIN_ROWS == 300


def test_the_group_bin_count_equals_component_fives() -> None:
    """A different bin count would make a group ECE incomparable with C9's global one.

    Which is the exact comparison this component exists to make, so loosening the threshold
    that way would destroy the thing it was loosened to measure.
    """
    assert d.GROUP_CALIBRATION_BINS == DEFAULT_CALIBRATION_BINS


def test_the_calibration_floor_may_not_drop_below_the_ranking_floor() -> None:
    assert d.CALIBRATION_MIN_ROWS >= d.SUPPORT_MIN_ROWS


def test_probability_metrics_are_gated_by_the_stricter_floor() -> None:
    assert d.support_floor_for(d.MetricKind.PROBABILITY) == d.CALIBRATION_MIN_ROWS
    assert d.support_floor_for(d.MetricKind.RANKING) == d.SUPPORT_MIN_ROWS
    assert d.support_floor_for(d.MetricKind.THRESHOLD_AUDIT) == d.SUPPORT_MIN_ROWS


# --- 4. the rest of the frozen surface -------------------------------------------


def test_no_probability_threshold_is_offered() -> None:
    """Component 13 owns decision policy; every cutoff here is a rank position."""
    assert "descriptive threshold audit" in d.THRESHOLD_POLICY
    assert "never at a probability threshold" in d.THRESHOLD_POLICY
    assert all(name.startswith("k_") for name in d.K_LEVELS)


def test_the_disparity_reference_is_the_population_and_never_a_group() -> None:
    assert "never a nominated group" in d.DISPARITY_REFERENCE


def test_there_are_four_disparity_measures_and_no_single_score() -> None:
    """A scalar would be a hidden weighting of mutually incompatible criteria."""
    assert len(list(d.DisparityMeasure)) == 4
    assert not hasattr(d, "FAIRNESS_SCORE")


def test_the_bootstrap_seed_is_an_integer_literal() -> None:
    """Never hash() of a string: Python salts str hashing per process. Invariant 92."""
    assert isinstance(d.BOOTSTRAP_SEED, int)
    assert d.BOOTSTRAP_SEED == 20260826


def test_both_resampling_schemes_are_run() -> None:
    """Establishments recur inside a group, so an i.i.d. row bootstrap understates the SE."""
    assert d.BOOTSTRAP_SCHEMES == ("row", "establishment_block")


def test_a_drift_trend_needs_at_least_three_folds() -> None:
    """Two points are a line through any two numbers."""
    assert d.DRIFT_MIN_FOLDS >= 3


def test_the_unknown_group_is_audited_rather_than_dropped() -> None:
    assert d.UNKNOWN_GROUP == "__UNKNOWN__"
    assert "first-class group value" in d.UNKNOWN_IS_A_GROUP
    assert "no prior inspection" in d.UNKNOWN_IS_A_GROUP


# --- 5. the boundary --------------------------------------------------------------


def test_the_boundary_list_names_every_claim_the_component_cannot_make() -> None:
    """ADR 0035. Carried in every manifest and printed on every run."""
    joined = " ".join(d.DOES_NOT_ESTABLISH).lower()
    for claim in (
        "causality",
        "discrimination",
        "absence of bias",
        "legal",
        "ethical acceptability",
        "equal treatment",
        "optimal fairness policy",
    ):
        assert claim in joined, f"the boundary list does not mention {claim}"


def test_the_blocked_list_names_the_inspector_gap_and_the_missing_demographics() -> None:
    joined = " ".join(d.BLOCKED_EXPERIMENTS)
    assert "ADR 0019" in joined
    assert "protected-class" in joined
    assert "group-specific recalibration" in joined
    assert "model selection" in joined


def test_the_calibration_candidates_are_the_models_this_component_can_audit() -> None:
    """The five models Component 9 calibrated are the five with both stages on one row."""
    assert len(CANDIDATE_REGISTRY) == 5
