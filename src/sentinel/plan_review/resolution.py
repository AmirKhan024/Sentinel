"""The human decision layer for Component 21. Pure -- no filesystem, no clock.

Mirrors Component 16's ``resolution.py``: an all-or-nothing parser at the boundary, and an
apply step that never edits the plan it is deciding about -- it only decides each decision's
outcome and the establishment's final status.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sentinel.plan_review.definitions import (
    PLAN_REVIEW_REQUIRED_FIELDS,
    REQUIRED_FIELD_FOR_ACTION,
    PlanDecisionAction,
)
from sentinel.plan_review.models import PlanDecision, PlanDecisionOutcome

OUTCOME_APPLIED = "applied"
OUTCOME_NO_OP_ALREADY_DECIDED = "no_op_already_decided"
OUTCOME_ESTABLISHMENT_NOT_IN_PLAN = "establishment_not_in_plan"

STATUS_UNDECIDED = "undecided"
STATUS_DECIDED = "decided"


class PlanReviewGovernanceError(ValueError):
    """Raised when a supervisor plan decision cannot be trusted enough to apply."""


def parse_decisions(records: Sequence[Mapping[str, object]]) -> list[PlanDecision]:
    """Validate a decoded decision file at the boundary, or refuse the whole run.

    Every required field must be present and non-blank, the action must be a known
    ``PlanDecisionAction``, and ``REQUIRED_FIELD_FOR_ACTION`` names for that action -- if
    any -- must be present and non-blank. Refusing the whole file rather than skipping bad
    rows matches Component 13/14/16's own parsers: a partially applied decision file
    produces a plan-review state nobody authorised.
    """
    actions = {a.value for a in PlanDecisionAction}
    decisions: list[PlanDecision] = []
    for position, raw in enumerate(records):
        try:
            decision = PlanDecision.model_validate(dict(raw))
        except Exception as exc:  # pydantic raises its own error type
            raise PlanReviewGovernanceError(f"decision {position}: {exc}") from exc

        blank = [
            name
            for name in PLAN_REVIEW_REQUIRED_FIELDS
            if not str(getattr(decision, name, "")).strip()
        ]
        if blank:
            raise PlanReviewGovernanceError(
                f"decision {decision.decision_id or position}: {', '.join(sorted(blank))} is "
                "blank. Every required plan-review field is required"
            )
        if decision.decision_action not in actions:
            raise PlanReviewGovernanceError(
                f"decision {decision.decision_id}: unknown decision_action "
                f"{decision.decision_action!r}; known: {', '.join(sorted(actions))}"
            )

        required_field = REQUIRED_FIELD_FOR_ACTION[decision.decision_action]
        if required_field is not None:
            value = getattr(decision, required_field, None)
            if not (value and str(value).strip()):
                raise PlanReviewGovernanceError(
                    f"decision {decision.decision_id}: {decision.decision_action} requires a "
                    f"non-blank {required_field}"
                )
        decisions.append(decision)

    ids = [d.decision_id for d in decisions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise PlanReviewGovernanceError(
            f"duplicate decision_id: {', '.join(duplicates)}. The id is how a decision is "
            "referred to afterwards, so two decisions cannot share one"
        )
    return decisions


def apply_decisions(
    target_inspection_ids: Sequence[str], decisions: Sequence[PlanDecision]
) -> tuple[dict[str, PlanDecisionOutcome], dict[str, PlanDecision]]:
    """Apply every decision addressed to this plan, in ``decision_id`` order.

    Applied in id order rather than file order, matching ADR 0047's rule for Component 13's
    overrides, Component 14's adjustments, and Component 16's resolutions: re-serialising the
    file cannot change which decision "wins" an establishment two decisions both address.

    Returns a map from a decision's own id to its ``PlanDecisionOutcome``, and a map from
    each *decided* establishment's ``target_inspection_id`` to the winning ``PlanDecision`` --
    the second map is what a reader joins onto the plan frame to show the supervisor's
    decision beside Sentinel's own recommendation, never in place of it.
    """
    in_plan = set(target_inspection_ids)
    final_decision: dict[str, PlanDecision] = {}
    outcomes: dict[str, PlanDecisionOutcome] = {}

    for decision in sorted(decisions, key=lambda d: d.decision_id):
        if decision.target_inspection_id not in in_plan:
            outcomes[decision.decision_id] = PlanDecisionOutcome(
                decision=decision, outcome=OUTCOME_ESTABLISHMENT_NOT_IN_PLAN
            )
            continue
        if decision.target_inspection_id in final_decision:
            outcomes[decision.decision_id] = PlanDecisionOutcome(
                decision=decision,
                outcome=OUTCOME_NO_OP_ALREADY_DECIDED,
                original_status=STATUS_DECIDED,
                final_status=STATUS_DECIDED,
            )
            continue
        final_decision[decision.target_inspection_id] = decision
        outcomes[decision.decision_id] = PlanDecisionOutcome(
            decision=decision,
            outcome=OUTCOME_APPLIED,
            original_status=STATUS_UNDECIDED,
            final_status=STATUS_DECIDED,
        )

    return outcomes, final_decision


__all__ = [
    "OUTCOME_APPLIED",
    "OUTCOME_ESTABLISHMENT_NOT_IN_PLAN",
    "OUTCOME_NO_OP_ALREADY_DECIDED",
    "STATUS_DECIDED",
    "STATUS_UNDECIDED",
    "PlanReviewGovernanceError",
    "apply_decisions",
    "parse_decisions",
]
