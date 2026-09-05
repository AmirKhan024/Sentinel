"""The frozen policy contracts, and the guard that stops them drifting apart.

``definitions.py`` runs ``_guard_registry()`` at import time, so most of these tests work by
constructing a contradictory registry and asserting the guard would have caught it. A guard
whose failure path has never been observed is indistinguishable from one that cannot fire.
"""

from __future__ import annotations

import pytest

from sentinel.policy import definitions
from sentinel.policy.definitions import (
    BASELINE_POLICY_ID,
    CANDIDATE_MODELS,
    K_LEVELS,
    MECHANISM_REASONS,
    POLICY_GRID,
    PRIMARY_K_LEVEL,
    REFUSED_MODELS,
    RESERVE_SHARES,
    SELECTION_AXES,
    DecisionMechanism,
    DecisionReason,
    PolicyDefinitionError,
    PolicySpec,
    PolicyWarning,
    ReserveMechanism,
    policy_for,
)

# --- 1. the grid ---------------------------------------------------------------


def test_the_grid_holds_the_baseline_and_both_mechanisms_at_every_share() -> None:
    """Seven policies: the null one, and each mechanism at each of the three shares.

    Both mechanisms must be measured at identical shares or the comparison between them is
    not a comparison -- it is two different experiments reported in one table.
    """
    assert len(POLICY_GRID) == 1 + 2 * len(RESERVE_SHARES)
    for mechanism in (ReserveMechanism.FLOOR, ReserveMechanism.FORCED):
        shares = sorted(s.reserve_share for s in POLICY_GRID if s.mechanism is mechanism)
        assert shares == sorted(RESERVE_SHARES)


def test_the_baseline_reserves_nothing() -> None:
    """Every opportunity cost is measured against it, so it must be the null policy."""
    baseline = policy_for(BASELINE_POLICY_ID)
    assert baseline.mechanism is ReserveMechanism.NONE
    assert baseline.reserve_share == 0.0


def test_the_population_share_policy_sits_at_the_measured_share() -> None:
    """The grid is anchored on a measurement, not on round numbers.

    Profile 2 measured 3,410 of 32,696 quarterly test rows coverage-eligible. The grid is
    half that, that, and twice that; if the anchor ever drifts away from the grid the two
    stop meaning what the rationale says they mean.
    """
    assert pytest.approx(0.1043, abs=0.005) == definitions.ELIGIBLE_POPULATION_SHARE
    assert 0.10 in RESERVE_SHARES
    assert min(RESERVE_SHARES) == pytest.approx(0.10 / 2)
    assert max(RESERVE_SHARES) == pytest.approx(0.10 * 2)


def test_every_policy_states_why_it_exists() -> None:
    assert all(spec.rationale for spec in POLICY_GRID)


def test_an_unknown_policy_names_the_ones_that_exist() -> None:
    with pytest.raises(PolicyDefinitionError, match="pure_risk"):
        policy_for("coverage_whatever")


# --- 2. the registry guard, driven red ------------------------------------------


def test_a_duplicate_policy_id_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    duplicate = (*POLICY_GRID, POLICY_GRID[0])
    monkeypatch.setattr(definitions, "POLICY_GRID", duplicate)
    with pytest.raises(PolicyDefinitionError, match="duplicate policy_id"):
        definitions._guard_registry()


def test_a_baseline_that_reserves_capacity_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The comparison is meaningless if the thing being compared against is itself a policy."""
    broken = tuple(
        PolicySpec(
            policy_id=spec.policy_id,
            mechanism=ReserveMechanism.FLOOR,
            reserve_share=0.10,
            rationale=spec.rationale,
        )
        if spec.policy_id == BASELINE_POLICY_ID
        else spec
        for spec in POLICY_GRID
    )
    monkeypatch.setattr(definitions, "POLICY_GRID", broken)
    with pytest.raises(PolicyDefinitionError, match="must reserve nothing"):
        definitions._guard_registry()


def test_a_mechanism_and_share_that_disagree_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A policy claiming no mechanism while declaring a share reserves capacity by accident."""
    broken = (
        PolicySpec(
            policy_id="contradiction",
            mechanism=ReserveMechanism.NONE,
            reserve_share=0.10,
            rationale="a mechanism and a share that disagree",
        ),
        *POLICY_GRID,
    )
    monkeypatch.setattr(definitions, "POLICY_GRID", broken)
    with pytest.raises(PolicyDefinitionError, match="disagree"):
        definitions._guard_registry()


def test_a_mechanism_no_policy_exercises_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A code path no run takes is indistinguishable from one that is broken."""
    floor_only = tuple(
        spec for spec in POLICY_GRID if spec.mechanism is not ReserveMechanism.FORCED
    )
    monkeypatch.setattr(definitions, "POLICY_GRID", floor_only)
    with pytest.raises(PolicyDefinitionError, match="no policy exercises"):
        definitions._guard_registry()


def test_a_reserve_of_the_whole_window_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = (
        PolicySpec(
            policy_id="everything",
            mechanism=ReserveMechanism.FORCED,
            reserve_share=1.0,
            rationale="reserving the entire window",
        ),
        *POLICY_GRID,
    )
    monkeypatch.setattr(definitions, "POLICY_GRID", broken)
    with pytest.raises(PolicyDefinitionError, match="not a share"):
        definitions._guard_registry()


def test_an_empty_boundary_list_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """It travels in every manifest precisely so it cannot be dropped."""
    monkeypatch.setattr(definitions, "DOES_NOT_ESTABLISH", ())
    with pytest.raises(PolicyDefinitionError, match="boundary list is empty"):
        definitions._guard_registry()


def test_an_empty_forbidden_column_list_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The validator checks eligibility against it; empty turns that check into a formality."""
    monkeypatch.setattr(definitions, "FORBIDDEN_POLICY_COLUMNS", ())
    with pytest.raises(PolicyDefinitionError, match="forbidden-column list is empty"):
        definitions._guard_registry()


def test_a_selection_rule_that_cannot_terminate_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the name terminator two identical models leave the rule undecided."""
    monkeypatch.setattr(definitions, "SELECTION_AXES", SELECTION_AXES[:-1])
    with pytest.raises(PolicyDefinitionError, match="terminate"):
        definitions._guard_registry()


def test_a_primary_capacity_outside_the_reported_cutoffs_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(definitions, "PRIMARY_K_LEVEL", "k_pct_99")
    with pytest.raises(PolicyDefinitionError, match="not one of the reported"):
        definitions._guard_registry()


def test_an_orphan_reason_code_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reason code no mechanism accepts can be written and never checked."""
    trimmed = {
        DecisionMechanism.RISK_PRIORITY: MECHANISM_REASONS[DecisionMechanism.RISK_PRIORITY],
        DecisionMechanism.COVERAGE_RESERVE: MECHANISM_REASONS[DecisionMechanism.COVERAGE_RESERVE],
        DecisionMechanism.NOT_SELECTED: frozenset({DecisionReason.NOT_SELECTED_CAPACITY_EXHAUSTED}),
    }
    monkeypatch.setattr(definitions, "MECHANISM_REASONS", trimmed)
    with pytest.raises(PolicyDefinitionError, match="belong to no mechanism"):
        definitions._guard_registry()


def test_refusing_every_model_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A component that cannot produce a queue should say so, not emit an empty one."""
    monkeypatch.setattr(definitions, "CANDIDATE_MODELS", ())
    with pytest.raises(PolicyDefinitionError, match="cannot produce a queue"):
        definitions._guard_registry()


# --- 3. the vocabularies ---------------------------------------------------------


def test_every_reason_code_belongs_to_exactly_one_mechanism() -> None:
    seen: list[str] = []
    for reasons in MECHANISM_REASONS.values():
        seen.extend(reasons)
    assert sorted(seen) == sorted(set(seen))
    assert set(seen) == {str(reason) for reason in DecisionReason}


def test_no_warning_contains_the_separator() -> None:
    """The warning column is a joined set, so a code containing the join char is unparseable."""
    for warning in PolicyWarning:
        assert definitions.WARNING_SEPARATOR not in warning
    assert definitions.WARNING_SEPARATOR not in definitions.NO_WARNING


def test_the_experimental_model_is_refused_with_a_reason() -> None:
    """ADR 0031: a model Component 11 could not explain is not a deployment candidate."""
    assert REFUSED_MODELS == ("xgboost_chain_embeddings_platt",)
    refused = next(c for c in definitions.MODEL_CANDIDATES if not c.admissible)
    assert "ADR 0031" in refused.reason
    assert refused.model_name not in CANDIDATE_MODELS


def test_the_capacity_levels_are_component_twelves() -> None:
    """Imported, not restated: a policy number and an audit number must describe one point."""
    from sentinel.fairness.definitions import K_LEVELS as AUDIT_K_LEVELS

    assert K_LEVELS == AUDIT_K_LEVELS
    assert PRIMARY_K_LEVEL in K_LEVELS


def test_no_probability_threshold_is_offered_anywhere() -> None:
    """Component 12 refused one in prose and named this component as the one that owns policy.

    Capacity here is a rank position. If a probability cutoff ever appeared it would be the
    single easiest thing in this project to mistake for a validated decision rule.
    """
    assert "rank position" in definitions.CAPACITY_SEMANTICS
    assert not hasattr(definitions, "PROBABILITY_THRESHOLD")
