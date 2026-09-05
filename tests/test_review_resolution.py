"""The human resolution layer: all-or-nothing parsing, id-order application."""

from __future__ import annotations

import pytest

from sentinel.review.models import ReviewCase
from sentinel.review.resolution import (
    OUTCOME_APPLIED,
    OUTCOME_CASE_NOT_IN_QUEUE,
    OUTCOME_NO_OP_ALREADY_RESOLVED,
    ReviewGovernanceError,
    apply_resolutions,
    parse_resolutions,
)

VALID = {
    "review_id": "R1",
    "policy_id": "pure_risk",
    "fold_id": "2026Q1",
    "k_name": "k_1_day",
    "target_inspection_id": "t1",
    "resolution_action": "acknowledge",
    "reason_code": "reviewed",
    "actor": "alice",
    "decided_at": "2026-01-06T00:00:00Z",
}


def _case(target_inspection_id: str = "t1", **overrides: object) -> ReviewCase:
    fields: dict[str, object] = {
        "policy_id": "pure_risk",
        "model_name": "lightgbm_platt",
        "fold_set": "quarterly",
        "fold_id": "2026Q1",
        "k_name": "k_1_day",
        "target_inspection_id": target_inspection_id,
        "establishment_id": "e1",
        "final_policy_rank": 1,
        "decision_mechanism": "risk_priority",
        "decision_reason": "selected_by_risk_rank",
        "warnings": "limited_history",
        "schedule_config_id": None,
        "planning_run_id": None,
        "replan_index": None,
        "scheduled_date": None,
        "trigger_reasons": ("policy_warning_present",),
    }
    fields.update(overrides)
    return ReviewCase(**fields)


def test_a_valid_resolution_parses() -> None:
    parsed = parse_resolutions([VALID])
    assert parsed[0].review_id == "R1"


def test_an_unknown_action_is_refused() -> None:
    with pytest.raises(ReviewGovernanceError, match="unknown resolution_action"):
        parse_resolutions([{**VALID, "resolution_action": "delete_everything"}])


def test_a_blank_required_field_is_refused() -> None:
    with pytest.raises(ReviewGovernanceError, match="blank"):
        parse_resolutions([{**VALID, "actor": ""}])


def test_refer_to_override_requires_the_pointer() -> None:
    with pytest.raises(ReviewGovernanceError, match="requires a non-blank referenced_override_id"):
        parse_resolutions([{**VALID, "resolution_action": "refer_to_override"}])


def test_refer_to_override_with_pointer_parses() -> None:
    parsed = parse_resolutions(
        [{**VALID, "resolution_action": "refer_to_override", "referenced_override_id": "O1"}]
    )
    assert parsed[0].referenced_override_id == "O1"


def test_acknowledge_must_not_carry_a_pointer() -> None:
    with pytest.raises(ReviewGovernanceError, match="must not carry"):
        parse_resolutions([{**VALID, "referenced_override_id": "O1"}])


def test_refer_to_override_must_not_also_carry_adjustment_pointer() -> None:
    with pytest.raises(ReviewGovernanceError, match="must not carry"):
        parse_resolutions(
            [
                {
                    **VALID,
                    "resolution_action": "refer_to_override",
                    "referenced_override_id": "O1",
                    "referenced_adjustment_id": "A1",
                }
            ]
        )


def test_duplicate_review_id_is_refused() -> None:
    with pytest.raises(ReviewGovernanceError, match="duplicate review_id"):
        parse_resolutions([VALID, {**VALID, "actor": "bob"}])


def test_one_bad_row_refuses_the_whole_file() -> None:
    good = VALID
    bad = {**VALID, "review_id": "R2", "resolution_action": "not_a_real_action"}
    with pytest.raises(ReviewGovernanceError):
        parse_resolutions([good, bad])


def test_an_extra_field_is_refused_by_pydantic() -> None:
    with pytest.raises(ReviewGovernanceError):
        parse_resolutions([{**VALID, "unexpected_field": "x"}])


# --- apply_resolutions -----------------------------------------------------------


def test_a_resolution_addressing_a_case_applies() -> None:
    cases = [_case("t1")]
    resolutions = parse_resolutions([VALID])
    outcomes, final_status = apply_resolutions(cases, resolutions)
    assert outcomes["R1"].outcome == OUTCOME_APPLIED
    assert final_status[("pure_risk", "2026Q1", "k_1_day", "t1")] == "resolved"


def test_a_resolution_addressing_a_case_not_in_the_queue_is_reported() -> None:
    cases = [_case("t1")]
    resolutions = parse_resolutions([{**VALID, "target_inspection_id": "t99"}])
    outcomes, _ = apply_resolutions(cases, resolutions)
    assert outcomes["R1"].outcome == OUTCOME_CASE_NOT_IN_QUEUE


def test_a_second_resolution_for_an_already_resolved_case_is_a_no_op() -> None:
    cases = [_case("t1")]
    first = {**VALID, "review_id": "R1"}
    second = {**VALID, "review_id": "R2", "actor": "bob"}
    outcomes, final_status = apply_resolutions(cases, parse_resolutions([first, second]))
    assert outcomes["R1"].outcome == OUTCOME_APPLIED
    assert outcomes["R2"].outcome == OUTCOME_NO_OP_ALREADY_RESOLVED
    assert final_status[("pure_risk", "2026Q1", "k_1_day", "t1")] == "resolved"


def test_resolutions_apply_in_review_id_order_not_file_order() -> None:
    """Two resolutions addressing the same case: whichever sorts first by id wins, regardless
    of the order they appear in the file."""
    cases = [_case("t1")]
    forward = [{**VALID, "review_id": "R1"}, {**VALID, "review_id": "R2", "actor": "bob"}]
    backward = list(reversed(forward))
    outcomes_forward, _ = apply_resolutions(cases, parse_resolutions(forward))
    outcomes_backward, _ = apply_resolutions(cases, parse_resolutions(backward))
    assert outcomes_forward["R1"].outcome == OUTCOME_APPLIED
    assert outcomes_forward["R2"].outcome == OUTCOME_NO_OP_ALREADY_RESOLVED
    assert outcomes_backward["R1"].outcome == OUTCOME_APPLIED
    assert outcomes_backward["R2"].outcome == OUTCOME_NO_OP_ALREADY_RESOLVED


def test_no_resolutions_leaves_every_case_flagged() -> None:
    cases = [_case("t1"), _case("t2")]
    _, final_status = apply_resolutions(cases, [])
    assert set(final_status.values()) == {"flagged"}


def test_a_case_with_no_trigger_reasons_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="at least one trigger"):
        _case("t1", trigger_reasons=())
