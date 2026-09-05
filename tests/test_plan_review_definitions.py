"""Component 21: frozen contracts and their guard-registry checks."""

from __future__ import annotations

from sentinel.plan_review.definitions import (
    PLAN_REVIEW_REQUIRED_FIELDS,
    REQUIRED_FIELD_FOR_ACTION,
    PlanApprovalStatus,
    PlanDecisionAction,
    derive_plan_approval_status,
)
from sentinel.policy.definitions import OverrideAction
from sentinel.review.definitions import ReviewResolutionAction
from sentinel.scheduling.definitions import AdjustmentAction


def test_plan_decision_verbs_are_disjoint_from_every_other_human_layer() -> None:
    plan_verbs = {str(a) for a in PlanDecisionAction}
    assert plan_verbs.isdisjoint({str(a) for a in OverrideAction})
    assert plan_verbs.isdisjoint({str(a) for a in AdjustmentAction})
    assert plan_verbs.isdisjoint({str(a) for a in ReviewResolutionAction})


def test_no_plan_review_vocabulary_reuses_the_word_defer() -> None:
    for value in (*PlanDecisionAction, *PlanApprovalStatus):
        assert "defer" not in str(value)


def test_required_field_for_action_covers_every_action_exactly_once() -> None:
    assert set(REQUIRED_FIELD_FOR_ACTION) == set(PlanDecisionAction)


def test_move_to_later_workday_requires_a_revised_date() -> None:
    assert REQUIRED_FIELD_FOR_ACTION[PlanDecisionAction.MOVE_TO_LATER_WORKDAY] == (
        "revised_planned_date"
    )


def test_keep_selected_and_do_not_proceed_require_no_extra_field() -> None:
    assert REQUIRED_FIELD_FOR_ACTION[PlanDecisionAction.KEEP_SELECTED] is None
    assert REQUIRED_FIELD_FOR_ACTION[PlanDecisionAction.DO_NOT_PROCEED_AS_PLANNED] is None


def test_actor_and_reason_code_are_required_fields() -> None:
    assert "actor" in PLAN_REVIEW_REQUIRED_FIELDS
    assert "reason_code" in PLAN_REVIEW_REQUIRED_FIELDS


def test_derive_status_draft_when_nothing_decided() -> None:
    assert derive_plan_approval_status(total=30, decided=0) == PlanApprovalStatus.DRAFT


def test_derive_status_under_review_when_partially_decided() -> None:
    assert (
        derive_plan_approval_status(total=30, decided=5)
        == PlanApprovalStatus.UNDER_SUPERVISOR_REVIEW
    )


def test_derive_status_adjusted_when_fully_decided() -> None:
    assert derive_plan_approval_status(total=30, decided=30) == PlanApprovalStatus.ADJUSTED


def test_derive_status_draft_for_an_empty_plan() -> None:
    assert derive_plan_approval_status(total=0, decided=0) == PlanApprovalStatus.DRAFT
