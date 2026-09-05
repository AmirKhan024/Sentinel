"""Response/request shapes for Component 21's artifacts.

Field lists mirror ``sentinel.plan_review.writer``'s schema exactly. The column contract
lives in one place -- the writer -- and this module is a typed view onto it for JSON.
"""

from __future__ import annotations

from pydantic import BaseModel

from sentinel.api.schemas.establishment import RiskHistoryFactorsOut
from sentinel.plan_review.models import PlanApprovalRequest, PlanDecision


class PlanRowOut(BaseModel):
    """One establishment: Sentinel's own recommendation, Component 20's geography, and any
    recorded supervisor decision, all visible at once."""

    planning_date: str
    establishment_id: str
    target_inspection_id: str
    canonical_name: str | None = None
    canonical_address: str | None = None
    establishment_name: str | None = None
    establishment_address: str | None = None

    # Sentinel's own recommendation -- never overwritten by a supervisor decision.
    calibrated_score: float
    base_score: float
    rank: int
    policy_rank: int | None
    selection_reason: str
    selection_mechanism: str

    #: Display-only field-work ordering: the supervisor's ADJUST_OPERATIONAL_PRIORITY value
    #: where recorded, else exactly ``policy_rank``. Never a substitute for ``rank``/
    #: ``policy_rank``, both of which are always present above, unedited.
    operational_priority: int | None = None

    # Component 20's geographic organization.
    location_status: str
    work_block_id: str
    work_block_label: str
    suggested_order_in_block: int | None
    organization_mode: str
    highest_sentinel_rank_in_block: int | None

    # The supervisor's decision, if any -- nullable, and always additional to, never a
    # replacement for, the fields above.
    supervisor_decision_id: str | None = None
    supervisor_decision_action: str | None = None
    supervisor_decision_reason_code: str | None = None
    supervisor_decision_actor: str | None = None
    supervisor_decision_decided_at: str | None = None
    supervisor_revised_planned_date: str | None = None
    supervisor_revised_work_block_id: str | None = None
    supervisor_revised_operational_priority: int | None = None

    #: A curated slice of Component 17's own as-of feature row for this candidate -- the exact
    #: same fields Component 4's `RiskHistoryFactorsOut` already surfaces for the historical
    #: (Side-A) establishment detail page, reused verbatim, never recomputed. `None` only when
    #: the operational candidate table has no row for this `target_inspection_id` (should not
    #: happen for a row that made it into a plan, but never assumed).
    history_factors: RiskHistoryFactorsOut | None = None


class WorkBlockOut(BaseModel):
    """One geographic work block, aggregated from ``PlanRowOut`` rows for display."""

    work_block_id: str
    work_block_label: str
    size: int
    highest_sentinel_rank: int | None
    rank_range: list[int] | None
    is_unmapped: bool
    decisions_recorded: int


class PlanSummaryOut(BaseModel):
    planning_date: str
    selected_inspection_workload: int
    location_available_count: int
    location_unavailable_count: int
    work_block_count: int
    decisions_recorded: int
    approval_status: str


class PlanDecisionLogRowOut(BaseModel):
    decision_id: str
    planning_date: str
    target_inspection_id: str
    decision_action: str
    reason_code: str
    actor: str
    decided_at: str
    revised_planned_date: str | None = None
    revised_work_block_id: str | None = None
    outcome: str
    plan_review_definition_version: str
    #: Always "committed" -- read endpoints read only the committed decision log, never the
    #: staging store, matching Component 16's own status-field convention.
    status: str = "committed"


class PlanDecisionIn(BaseModel):
    """Exactly Component 21's ``PlanDecision`` contract -- the API adds no field of its own.

    Validated by running the payload through ``sentinel.plan_review.resolution.parse_decisions``
    as a one-element list, so a staged request can never be accepted in a shape the batch CLI
    would go on to refuse.
    """

    decision_id: str
    planning_date: str
    target_inspection_id: str
    decision_action: str
    reason_code: str
    actor: str
    decided_at: str
    revised_planned_date: str | None = None
    revised_work_block_id: str | None = None
    revised_operational_priority: int | None = None

    model_config = {"extra": "forbid"}

    def to_decision(self) -> PlanDecision:
        return PlanDecision(**self.model_dump())


class PlanApprovalIn(BaseModel):
    """Exactly Component 21's ``PlanApprovalRequest`` contract -- the API adds no field.

    Staged, never applied (ADR 0049), matching ``PlanDecisionIn``: an operator later commits
    it through ``sentinel approve-plan``, which re-runs the full readiness checklist before
    writing anything.
    """

    approval_id: str
    planning_date: str
    approved_by: str
    approved_at: str
    note: str | None = None

    model_config = {"extra": "forbid"}

    def to_request(self) -> PlanApprovalRequest:
        return PlanApprovalRequest(**self.model_dump())


class PlanApprovalOut(BaseModel):
    """The latest committed approval for a planning date, if any."""

    approval_id: str
    planning_date: str
    approved_by: str
    approved_at: str
    note: str | None = None
    final_selected_count: int
    final_active_count: int
    final_deferred_count: int
    final_not_proceeding_count: int
    final_undecided_count: int
    source_plan_review_path: str
    source_plan_review_sha256: str


__all__ = [
    "PlanApprovalIn",
    "PlanApprovalOut",
    "PlanDecisionIn",
    "PlanDecisionLogRowOut",
    "PlanRowOut",
    "PlanSummaryOut",
    "WorkBlockOut",
]
