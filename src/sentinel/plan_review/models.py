"""Typed structures for Component 21. No behaviour, no I/O, no clock."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
from pydantic import BaseModel, Field

from sentinel.features.models import ValidationCheck

__all__ = [
    "ApprovedPlanManifest",
    "ArtifactRecord",
    "PlanApprovalRequest",
    "PlanApprovalResult",
    "PlanDecision",
    "PlanDecisionOutcome",
    "PlanReviewManifest",
    "PlanReviewResult",
    "PlanReviewSummary",
    "ValidationCheck",
]


class PlanDecision(BaseModel):
    """One supervisor decision about one establishment in a Component 20 plan.

    A pydantic model, not a dataclass, because this is the only input to the component a
    human types. ``decided_at`` is the supervisor's own timestamp, not the run's.
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
    #: Set only by ADJUST_OPERATIONAL_PRIORITY. A display-only field ordering the plan for
    #: field work; never written into, or read from, ``rank``/``policy_rank``.
    revised_operational_priority: int | None = None

    model_config = {"extra": "forbid"}


@dataclass(frozen=True, slots=True)
class PlanDecisionOutcome:
    """What applying one decision did, including when it did nothing."""

    decision: PlanDecision
    outcome: str
    original_status: str = ""
    final_status: str = ""


class ArtifactRecord(BaseModel):
    """Provenance for one written file -- identical shape to other components."""

    path: str
    bytes: int
    sha256: str
    row_count: int
    schema_: dict[str, str] = Field(alias="schema")

    model_config = {"populate_by_name": True}


@dataclass
class PlanReviewSummary:
    """High-level supervisor-facing summary for one plan review."""

    planning_date: str
    selected_inspection_workload: int
    location_available_count: int
    location_unavailable_count: int
    work_block_count: int
    decisions_recorded: int
    approval_status: str


class PlanReviewManifest(BaseModel):
    """Self-contained provenance and QA record for one plan-review run."""

    component: str = "plan_review"
    code_version: str
    plan_review_definition_version: str
    built_at: str

    planning_date: str

    plan_artifact_path: str
    plan_artifact_sha256: str
    geographic_organization_definition_version: str

    decisions_path: str | None = None
    decisions_sha256: str | None = None

    selected_inspection_workload: int
    location_available_count: int
    location_unavailable_count: int
    work_block_count: int
    decisions_recorded: int
    approval_status: str

    five_human_layers: str
    plan_review_cannot: str
    does_not_establish: list[str]
    inherited_limitations: list[str]

    warnings: list[str]
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    checks: list[dict[str, str]] = Field(default_factory=list)


@dataclass
class PlanReviewResult:
    """Everything a caller needs after a plan-review build."""

    plan_frame: pl.DataFrame
    summary: PlanReviewSummary
    checks: list[ValidationCheck]
    manifest: PlanReviewManifest
    plan_review_path: Path | None = None
    manifest_path: Path | None = None


class PlanApprovalRequest(BaseModel):
    """The one human-typed input to plan approval: who, and when.

    Deliberately as small as ``PlanDecision``'s own required core: this is not a second place
    to restate reasons for individual establishment decisions (those already live in the
    decision log) -- it is the single fact that a named supervisor elected to proceed with
    the plan as currently decided.
    """

    approval_id: str
    planning_date: str
    approved_by: str
    approved_at: str
    note: str | None = None

    model_config = {"extra": "forbid"}


class ApprovedPlanManifest(BaseModel):
    """Self-contained provenance and QA record for one plan approval.

    Answers, from the artifact alone: which exact supervisor plan review (and therefore
    which exact Component 18/19/20 chain) was approved, by whom, when, under what readiness
    checks, and with what final counts -- everything Component 22 needs without re-deriving
    any of it.
    """

    component: str = "approved_operational_plan"
    code_version: str
    plan_review_definition_version: str
    built_at: str

    approval_id: str
    planning_date: str
    approved_by: str
    approved_at: str
    note: str | None = None

    source_plan_review_path: str
    source_plan_review_sha256: str
    source_decision_log_path: str | None = None
    source_decision_log_sha256: str | None = None

    final_selected_count: int
    final_active_count: int
    final_deferred_count: int
    final_not_proceeding_count: int
    final_undecided_count: int

    plan_review_cannot: str
    does_not_establish: list[str]
    inherited_limitations: list[str]

    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    readiness_checks: list[dict[str, str]] = Field(default_factory=list)


@dataclass
class PlanApprovalResult:
    """Everything a caller needs after an approval build."""

    approved_frame: pl.DataFrame
    checks: list[ValidationCheck]
    manifest: ApprovedPlanManifest
    approved_path: Path | None = None
    manifest_path: Path | None = None
