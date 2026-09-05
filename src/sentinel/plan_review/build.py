"""Orchestration: a Component 20 geographic plan in, a supervisor plan review out.

The only module in the package that touches the filesystem or the clock. Component 21 never
reads Component 19's or Component 18's output directly -- only Component 20's, so a plan
decision can never bypass geographic organization or capacity/policy selection.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from sentinel import __version__
from sentinel.config import Settings
from sentinel.geographic_organization.models import GeographicOrganizationManifest
from sentinel.manifest import compute_sha256, manifest_path_for, read_manifest_as, write_manifest
from sentinel.plan_review import approval, inputs, resolution, validate, writer
from sentinel.plan_review.definitions import (
    APPROVED_PLAN_DATASET_SLUG,
    DOES_NOT_ESTABLISH,
    FIVE_HUMAN_LAYERS,
    INHERITED_LIMITATIONS,
    PLAN_REVIEW_CANNOT,
    PLAN_REVIEW_DEFINITION_VERSION,
    PlanDecisionAction,
    derive_plan_approval_status,
)
from sentinel.plan_review.models import (
    ApprovedPlanManifest,
    ArtifactRecord,
    PlanApprovalRequest,
    PlanApprovalResult,
    PlanReviewManifest,
    PlanReviewResult,
    PlanReviewSummary,
)
from sentinel.plan_review.resolution import PlanReviewGovernanceError

logger = logging.getLogger(__name__)

DATASET_SLUG = "supervisor_plan_review"
DECISION_LOG_SLUG = "plan_decision_log"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

REQUIRED_REVIEW_COLUMNS_FOR_APPROVAL = (
    "target_inspection_id",
    "planning_date",
    "policy_rank",
    "selection_reason",
    "work_block_id",
    "location_status",
    "supervisor_decision_action",
)

REQUIRED_PLAN_COLUMNS = (
    "target_inspection_id",
    "planning_date",
    "policy_rank",
    "selection_reason",
    "geographic_group_id",
    "work_block_id",
)


class PlanReviewBuildError(RuntimeError):
    """Raised when a plan review cannot be built at all."""


def build_plan_review(
    settings: Settings,
    *,
    plan_path: Path,
    decisions_path: Path | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> PlanReviewResult:
    """Summarize Component 20's plan for a supervisor, and join any human decisions.

    Never edits a Component 19/20 field -- enforced by ``validate.run_all_checks`` below,
    not merely by convention. ``decisions_path``, if given, is the operator's own
    accumulated JSON file of every supervisor decision to date (matching Component 16's
    ``--resolutions`` convention): this component does not itself merge across runs.
    """
    if not plan_path.exists():
        raise FileNotFoundError(f"Component 20 geographic plan not found: {plan_path}")
    plan_manifest_path = manifest_path_for(plan_path)
    if not plan_manifest_path.exists():
        raise FileNotFoundError(f"Component 20 manifest not found: {plan_manifest_path}")

    started = datetime.now(UTC)

    plan_frame = pl.read_parquet(plan_path)
    missing = [c for c in REQUIRED_PLAN_COLUMNS if c not in plan_frame.columns]
    if missing:
        raise PlanReviewBuildError(
            f"{plan_path.name}: not a Component 20 geographic plan, missing {', '.join(missing)}"
        )

    plan_manifest = read_manifest_as(GeographicOrganizationManifest, plan_manifest_path)
    planning_date = plan_manifest.planning_date

    try:
        raw_decisions = inputs.read_decisions_file(decisions_path)
        decisions = resolution.parse_decisions(raw_decisions)
    except PlanReviewGovernanceError as exc:
        raise PlanReviewBuildError(f"decisions file rejected: {exc}") from exc

    outcomes, final_decision = resolution.apply_decisions(
        plan_frame["target_inspection_id"].to_list(), decisions
    )

    decision_by_target_id: dict[str, dict[str, object]] = {
        tid: d.model_dump() for tid, d in final_decision.items()
    }
    review_frame = writer.finalize(
        plan_frame,
        plan_review_definition_version=PLAN_REVIEW_DEFINITION_VERSION,
        decision_by_target_id=decision_by_target_id,
    )

    checks = validate.run_all_checks(plan_frame, review_frame)

    selected_inspection_workload = review_frame.height
    location_available_count = int(
        review_frame.filter(pl.col("location_status") == "location_available").height
    )
    location_unavailable_count = selected_inspection_workload - location_available_count
    work_block_count = int(
        review_frame.filter(pl.col("work_block_id") != "unmapped")["work_block_id"].n_unique()
    )
    decisions_recorded = len(final_decision)
    approval_status = derive_plan_approval_status(
        total=selected_inspection_workload, decided=decisions_recorded
    ).value

    decision_log_rows: list[dict[str, object]] = []
    for _decision_id, outcome in sorted(outcomes.items()):
        d = outcome.decision
        decision_log_rows.append(
            {
                "decision_id": d.decision_id,
                "planning_date": d.planning_date,
                "target_inspection_id": d.target_inspection_id,
                "decision_action": d.decision_action,
                "reason_code": d.reason_code,
                "actor": d.actor,
                "decided_at": d.decided_at,
                "revised_planned_date": d.revised_planned_date,
                "revised_work_block_id": d.revised_work_block_id,
                "revised_operational_priority": d.revised_operational_priority,
                "outcome": outcome.outcome,
                "plan_review_definition_version": PLAN_REVIEW_DEFINITION_VERSION,
            }
        )
    decision_log_frame = writer.finalize_decision_log(decision_log_rows)

    destination = output_dir or settings.plan_review_processed_dir
    stamp = started.strftime(TIMESTAMP_FORMAT)

    artifacts: list[ArtifactRecord] = []
    review_path: Path | None = None
    decision_log_path: Path | None = None
    if not dry_run:
        review_path = destination / f"{DATASET_SLUG}_{planning_date}_{stamp}.parquet"
        writer.write_table(review_frame, review_path)
        artifacts.append(
            ArtifactRecord(
                path=review_path.name,
                bytes=review_path.stat().st_size,
                sha256=compute_sha256(review_path),
                row_count=review_frame.height,
                schema=writer.schema_of(review_frame),
            )
        )
        if not decision_log_frame.is_empty():
            decision_log_path = destination / f"{DECISION_LOG_SLUG}_{planning_date}_{stamp}.parquet"
            writer.write_table(decision_log_frame, decision_log_path)
            artifacts.append(
                ArtifactRecord(
                    path=decision_log_path.name,
                    bytes=decision_log_path.stat().st_size,
                    sha256=compute_sha256(decision_log_path),
                    row_count=decision_log_frame.height,
                    schema=writer.schema_of(decision_log_frame),
                )
            )

    warnings: list[str] = []
    rejected = [o for o in outcomes.values() if o.outcome != resolution.OUTCOME_APPLIED]
    if rejected:
        warnings.append(
            f"{len(rejected)} decision(s) in the decisions file did not apply (already "
            "decided, or the establishment is not in this plan) -- see the decision log"
        )

    manifest = PlanReviewManifest(
        code_version=__version__,
        plan_review_definition_version=PLAN_REVIEW_DEFINITION_VERSION,
        built_at=started.isoformat(),
        planning_date=planning_date,
        plan_artifact_path=plan_path.name,
        plan_artifact_sha256=compute_sha256(plan_path),
        geographic_organization_definition_version=plan_manifest.geographic_organization_definition_version,
        decisions_path=decisions_path.name if decisions_path else None,
        decisions_sha256=compute_sha256(decisions_path) if decisions_path else None,
        selected_inspection_workload=selected_inspection_workload,
        location_available_count=location_available_count,
        location_unavailable_count=location_unavailable_count,
        work_block_count=work_block_count,
        decisions_recorded=decisions_recorded,
        approval_status=approval_status,
        five_human_layers=FIVE_HUMAN_LAYERS,
        plan_review_cannot=PLAN_REVIEW_CANNOT,
        does_not_establish=list(DOES_NOT_ESTABLISH),
        inherited_limitations=list(INHERITED_LIMITATIONS),
        warnings=warnings,
        artifacts=artifacts,
        checks=[
            {"name": c.name, "severity": c.severity, "passed": str(c.passed), "detail": c.detail}
            for c in checks
        ],
    )

    manifest_path: Path | None = None
    if not dry_run and review_path is not None:
        manifest_path = manifest_path_for(review_path)
        write_manifest(manifest, manifest_path)

    logger.info(
        "Plan review for planning_date=%s: %d establishment(s), %d work block(s), "
        "%d decision(s) recorded, status=%s",
        planning_date,
        selected_inspection_workload,
        work_block_count,
        decisions_recorded,
        approval_status,
    )

    summary = PlanReviewSummary(
        planning_date=planning_date,
        selected_inspection_workload=selected_inspection_workload,
        location_available_count=location_available_count,
        location_unavailable_count=location_unavailable_count,
        work_block_count=work_block_count,
        decisions_recorded=decisions_recorded,
        approval_status=approval_status,
    )

    return PlanReviewResult(
        plan_frame=review_frame,
        summary=summary,
        checks=checks,
        manifest=manifest,
        plan_review_path=review_path,
        manifest_path=manifest_path,
    )


def summarize(result: PlanReviewResult) -> str:
    """One-screen summary of a build, printed by the CLI."""
    m = result.manifest
    lines = [
        f"planning date:                {m.planning_date}",
        f"approval status:              {m.approval_status}",
        f"selected inspection workload: {m.selected_inspection_workload}",
        f"  with coordinates:           {m.location_available_count}",
        f"  without coordinates:        {m.location_unavailable_count}",
        f"geographic work blocks:       {m.work_block_count}",
        f"supervisor decisions recorded: {m.decisions_recorded}",
    ]
    if m.warnings:
        lines.append("warnings:")
        lines.extend(f"  - {w}" for w in m.warnings)
    if result.plan_review_path is not None:
        lines.append(f"plan review:                  {result.plan_review_path}")
        lines.append(f"manifest:                     {result.manifest_path}")
    return "\n".join(lines)


class PlanApprovalBuildError(RuntimeError):
    """Raised when a plan cannot be approved at all."""


def build_approved_plan(
    settings: Settings,
    *,
    review_path: Path,
    approval_request: PlanApprovalRequest,
    decision_log_path: Path | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> PlanApprovalResult:
    """Approve exactly the supervisor plan review named by ``review_path``.

    Refuses outright -- never partially -- if ``approval.check_readiness`` finds a blocking
    problem. The written artifact is the plan-review frame at the moment of approval, plus
    the approval's own identity; it is never edited afterward. A later amendment produces a
    new ``supervisor_plan_review`` snapshot and, if the supervisor chooses, a new approval
    event through this same function -- this one's output is untouched either way.
    """
    if not review_path.exists():
        raise FileNotFoundError(f"Supervisor plan review not found: {review_path}")

    started = datetime.now(UTC)

    review_frame = pl.read_parquet(review_path)
    missing = [c for c in REQUIRED_REVIEW_COLUMNS_FOR_APPROVAL if c not in review_frame.columns]
    if missing:
        raise PlanApprovalBuildError(
            f"{review_path.name}: not a Component 21 supervisor plan review, missing "
            f"{', '.join(missing)}"
        )

    if approval_request.planning_date and review_frame.height:
        actual_dates = set(review_frame["planning_date"].to_list())
        if actual_dates != {approval_request.planning_date}:
            raise PlanApprovalBuildError(
                f"approval names planning_date={approval_request.planning_date!r} but "
                f"{review_path.name} carries {sorted(actual_dates)!r}"
            )

    checks = approval.check_readiness(review_frame)
    if approval.has_blocking_failures(checks):
        raise PlanApprovalBuildError(
            "plan approval blocked -- one or more readiness checks failed:\n"
            + approval.format_readiness_report(checks)
        )

    deferred = int(
        review_frame.filter(
            pl.col("supervisor_decision_action") == PlanDecisionAction.MOVE_TO_LATER_WORKDAY
        ).height
    )
    not_proceeding = int(
        review_frame.filter(
            pl.col("supervisor_decision_action") == PlanDecisionAction.DO_NOT_PROCEED_AS_PLANNED
        ).height
    )
    undecided = int(review_frame.filter(pl.col("supervisor_decision_action").is_null()).height)
    total = review_frame.height
    active = total - deferred - not_proceeding

    approved_frame = review_frame.with_columns(
        pl.lit(approval_request.approval_id).alias("approval_id"),
        pl.lit(approval_request.approved_by).alias("approved_by"),
        pl.lit(approval_request.approved_at).alias("approved_at"),
    ).sort(
        [pl.col("operational_priority").is_null(), "operational_priority", "target_inspection_id"]
    )

    destination = output_dir or settings.plan_review_processed_dir
    stamp = started.strftime(TIMESTAMP_FORMAT)

    artifacts: list[ArtifactRecord] = []
    approved_path: Path | None = None
    if not dry_run:
        approved_name = (
            f"{APPROVED_PLAN_DATASET_SLUG}_{approval_request.planning_date}_{stamp}.parquet"
        )
        approved_path = destination / approved_name
        writer.write_table(approved_frame, approved_path)
        artifacts.append(
            ArtifactRecord(
                path=approved_path.name,
                bytes=approved_path.stat().st_size,
                sha256=compute_sha256(approved_path),
                row_count=approved_frame.height,
                schema=writer.schema_of(approved_frame),
            )
        )

    manifest = ApprovedPlanManifest(
        code_version=__version__,
        plan_review_definition_version=PLAN_REVIEW_DEFINITION_VERSION,
        built_at=started.isoformat(),
        approval_id=approval_request.approval_id,
        planning_date=approval_request.planning_date,
        approved_by=approval_request.approved_by,
        approved_at=approval_request.approved_at,
        note=approval_request.note,
        source_plan_review_path=review_path.name,
        source_plan_review_sha256=compute_sha256(review_path),
        source_decision_log_path=decision_log_path.name if decision_log_path else None,
        source_decision_log_sha256=(
            compute_sha256(decision_log_path) if decision_log_path else None
        ),
        final_selected_count=total,
        final_active_count=active,
        final_deferred_count=deferred,
        final_not_proceeding_count=not_proceeding,
        final_undecided_count=undecided,
        plan_review_cannot=PLAN_REVIEW_CANNOT,
        does_not_establish=list(DOES_NOT_ESTABLISH),
        inherited_limitations=list(INHERITED_LIMITATIONS),
        artifacts=artifacts,
        readiness_checks=[
            {"name": c.name, "severity": c.severity, "passed": str(c.passed), "detail": c.detail}
            for c in checks
        ],
    )

    manifest_path: Path | None = None
    if not dry_run and approved_path is not None:
        manifest_path = manifest_path_for(approved_path)
        write_manifest(manifest, manifest_path)

    logger.info(
        "Approved plan for planning_date=%s by %s: %d total (%d active, %d deferred, "
        "%d not proceeding, %d undecided)",
        approval_request.planning_date,
        approval_request.approved_by,
        total,
        active,
        deferred,
        not_proceeding,
        undecided,
    )

    return PlanApprovalResult(
        approved_frame=approved_frame,
        checks=checks,
        manifest=manifest,
        approved_path=approved_path,
        manifest_path=manifest_path,
    )


def summarize_approval(result: PlanApprovalResult) -> str:
    """One-screen summary of an approval, printed by the CLI."""
    m = result.manifest
    lines = [
        f"planning date:           {m.planning_date}",
        f"approved by:             {m.approved_by}",
        f"approved at:             {m.approved_at}",
        f"final selected:          {m.final_selected_count}",
        f"  active:                {m.final_active_count}",
        f"  deferred:              {m.final_deferred_count}",
        f"  not proceeding:        {m.final_not_proceeding_count}",
        f"  undecided (as proposed): {m.final_undecided_count}",
        f"source plan review:      {m.source_plan_review_path}",
    ]
    if result.approved_path is not None:
        lines.append(f"approved plan:           {result.approved_path}")
        lines.append(f"manifest:                {result.manifest_path}")
    return "\n".join(lines)


__all__ = [
    "PlanApprovalBuildError",
    "PlanReviewBuildError",
    "build_approved_plan",
    "build_plan_review",
    "summarize",
    "summarize_approval",
]
