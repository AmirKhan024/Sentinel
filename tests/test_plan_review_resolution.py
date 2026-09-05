"""Component 21: parsing and applying supervisor plan decisions."""

from __future__ import annotations

import pytest

from sentinel.plan_review import resolution
from sentinel.plan_review.resolution import PlanReviewGovernanceError

PLANNING_DATE = "2026-08-28"


def _decision(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "decision_id": "DEC-0001",
        "planning_date": PLANNING_DATE,
        "target_inspection_id": "CANDIDATE::2026-08-28::EST-000001",
        "decision_action": "keep_selected",
        "reason_code": "no_concern",
        "actor": "supervisor.jsmith",
        "decided_at": "2026-09-02T15:00:00Z",
    }
    row.update(overrides)
    return row


# --- parsing --------------------------------------------------------------------------


def test_a_well_formed_decision_parses() -> None:
    decisions = resolution.parse_decisions([_decision()])
    assert len(decisions) == 1
    assert decisions[0].decision_action == "keep_selected"


def test_blank_required_field_is_refused() -> None:
    with pytest.raises(PlanReviewGovernanceError, match="actor"):
        resolution.parse_decisions([_decision(actor="")])


def test_unknown_decision_action_is_refused() -> None:
    with pytest.raises(PlanReviewGovernanceError, match="unknown decision_action"):
        resolution.parse_decisions([_decision(decision_action="approve")])


def test_move_to_later_workday_without_revised_date_is_refused() -> None:
    with pytest.raises(PlanReviewGovernanceError, match="revised_planned_date"):
        resolution.parse_decisions([_decision(decision_action="move_to_later_workday")])


def test_move_to_later_workday_with_revised_date_parses() -> None:
    decisions = resolution.parse_decisions(
        [
            _decision(
                decision_action="move_to_later_workday",
                revised_planned_date="2026-09-04",
            )
        ]
    )
    assert decisions[0].revised_planned_date == "2026-09-04"


def test_duplicate_decision_id_is_refused() -> None:
    with pytest.raises(PlanReviewGovernanceError, match="duplicate decision_id"):
        resolution.parse_decisions([_decision(), _decision()])


def test_one_bad_row_refuses_the_whole_batch() -> None:
    good = _decision()
    bad = _decision(decision_id="DEC-0002", actor="")
    with pytest.raises(PlanReviewGovernanceError):
        resolution.parse_decisions([good, bad])


def test_extra_field_is_refused() -> None:
    with pytest.raises(PlanReviewGovernanceError):
        resolution.parse_decisions([_decision(unexpected_field="x")])


# --- applying ---------------------------------------------------------------------------


def test_a_decision_for_an_establishment_in_the_plan_applies() -> None:
    decisions = resolution.parse_decisions([_decision()])
    outcomes, final = resolution.apply_decisions(
        ["CANDIDATE::2026-08-28::EST-000001"], decisions
    )
    assert outcomes["DEC-0001"].outcome == resolution.OUTCOME_APPLIED
    assert final["CANDIDATE::2026-08-28::EST-000001"].decision_id == "DEC-0001"


def test_a_decision_for_an_establishment_not_in_the_plan_is_rejected() -> None:
    decisions = resolution.parse_decisions([_decision()])
    outcomes, final = resolution.apply_decisions(["CANDIDATE::2026-08-28::EST-999999"], decisions)
    assert outcomes["DEC-0001"].outcome == resolution.OUTCOME_ESTABLISHMENT_NOT_IN_PLAN
    assert final == {}


def test_two_decisions_for_the_same_establishment_the_lower_id_wins_by_apply_order() -> None:
    d1 = _decision(decision_id="DEC-0001")
    d2 = _decision(decision_id="DEC-0002", decision_action="do_not_proceed_as_planned")
    decisions = resolution.parse_decisions([d2, d1])  # deliberately out of id order
    outcomes, final = resolution.apply_decisions(
        ["CANDIDATE::2026-08-28::EST-000001"], decisions
    )
    # Applied in decision_id order regardless of file order: DEC-0001 wins.
    assert outcomes["DEC-0001"].outcome == resolution.OUTCOME_APPLIED
    assert outcomes["DEC-0002"].outcome == resolution.OUTCOME_NO_OP_ALREADY_DECIDED
    assert final["CANDIDATE::2026-08-28::EST-000001"].decision_id == "DEC-0001"


def test_apply_never_mutates_the_target_ids_input() -> None:
    ids = ["CANDIDATE::2026-08-28::EST-000001"]
    decisions = resolution.parse_decisions([_decision()])
    resolution.apply_decisions(ids, decisions)
    assert ids == ["CANDIDATE::2026-08-28::EST-000001"]
