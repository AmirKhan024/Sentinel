"""Response/request shapes for Component 14's artifacts.

Field lists mirror ``sentinel.scheduling.writer`` schemas exactly; see
``sentinel.api.schemas.policy`` for why that mirroring is deliberate rather than incidental.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from sentinel.scheduling.models import Adjustment, ExecutionEvent


class ScheduleRowOut(BaseModel):
    schedule_config_id: str
    policy_id: str
    model_name: str
    fold_set: str
    fold_id: str
    k_name: str
    k: int
    target_inspection_id: str
    establishment_id: str
    establishment_name: str | None = None
    establishment_address: str | None = None
    recommendation_date: date
    base_score: float
    score: float
    model_rank: int
    final_policy_rank: int
    decision_mechanism: str
    decision_reason: str
    coverage_eligible: bool
    warnings: str
    recommendation_override_id: str
    policy_definition_version: str
    planning_run_id: str
    replan_index: int
    schedule_status: str
    schedule_reason: str
    inversion_reason: str
    scheduled_date: date | None
    day_index: int | None
    slot_index: int | None
    schedule_rank: int | None
    wait_operating_days: int | None
    original_scheduled_date: date | None
    original_schedule_rank: int | None
    adjustment_id: str
    is_scenario: bool
    schedule_definition_version: str


class BacklogRowOut(BaseModel):
    schedule_config_id: str
    policy_id: str
    model_name: str
    fold_set: str
    fold_id: str
    k_name: str
    k: int
    target_inspection_id: str
    establishment_id: str
    establishment_name: str | None = None
    establishment_address: str | None = None
    final_policy_rank: int
    decision_mechanism: str
    decision_reason: str
    coverage_eligible: bool
    backlog_position: int
    backlog_reason: str
    horizon_slots: int
    slots_short: int
    would_fit_on_day_index: int | None
    first_available_date: date | None
    planning_run_id: str
    replan_index: int
    is_scenario: bool
    schedule_definition_version: str


class ReplanningRunOut(BaseModel):
    schedule_config_id: str
    policy_id: str
    fold_set: str
    fold_id: str
    k_name: str
    planning_run_id: str
    replan_index: int
    parent_replan_index: int | None
    replan_from_date: date | None
    trigger: str
    n_preserved_completed: int
    n_preserved_past: int
    n_returned_to_queue: int
    n_cancelled: int
    n_newly_scheduled: int
    n_still_backlog: int
    remaining_slots: int
    schedule_definition_version: str


class AdjustmentLogRowOut(BaseModel):
    adjustment_id: str
    schedule_config_id: str
    policy_id: str
    fold_id: str
    k_name: str
    target_inspection_id: str
    action: str
    target_date: str
    reason_code: str
    actor: str
    decided_at: str
    original_status: str
    final_status: str
    displaced_target_inspection_id: str
    outcome: str
    planning_run_id: str
    replan_index: int
    schedule_definition_version: str
    status: str = "committed"


class ExecutionLogRowOut(BaseModel):
    execution_id: str
    schedule_config_id: str
    policy_id: str
    fold_id: str
    k_name: str
    target_inspection_id: str
    scheduled_date: date | None
    plan_scheduled_date: date | None
    execution_status: str
    reason_code: str
    actor: str
    observed_at: str
    outcome: str
    triggers_replan: bool
    applied_at_replan_index: int
    schedule_definition_version: str
    status: str = "committed"


class ExecutionSummaryOut(BaseModel):
    schedule_config_id: str
    policy_id: str
    model_name: str
    fold_set: str
    fold_id: str
    k_name: str
    n_scheduled: int
    n_completed: int
    n_not_performed: int
    n_cancelled_in_field: int
    n_no_execution_record: int
    completion_rate: float
    final_replan_index: int
    schedule_definition_version: str


class AdjustmentIn(BaseModel):
    """Exactly Component 14's ``Adjustment`` contract."""

    adjustment_id: str
    schedule_config_id: str
    policy_id: str
    fold_id: str
    k_name: str
    target_inspection_id: str
    action: str
    target_date: str
    reason_code: str
    actor: str
    decided_at: str

    model_config = {"extra": "forbid"}

    def to_adjustment(self) -> Adjustment:
        return Adjustment(**self.model_dump())


class ExecutionEventIn(BaseModel):
    """Exactly Component 14's ``ExecutionEvent`` contract."""

    execution_id: str
    schedule_config_id: str
    policy_id: str
    fold_id: str
    k_name: str
    target_inspection_id: str
    scheduled_date: str
    execution_status: str
    reason_code: str
    actor: str
    observed_at: str

    model_config = {"extra": "forbid"}

    def to_event(self) -> ExecutionEvent:
        return ExecutionEvent(**self.model_dump())


__all__ = [
    "AdjustmentIn",
    "AdjustmentLogRowOut",
    "BacklogRowOut",
    "ExecutionEventIn",
    "ExecutionLogRowOut",
    "ExecutionSummaryOut",
    "ReplanningRunOut",
    "ScheduleRowOut",
]
