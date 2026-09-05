"""Read and stage access to Component 21's artifacts. No triggering, no deciding, no scoring.

Reads only the latest **committed** ``supervisor_plan_review`` table -- the joined artifact
``sentinel review-plan`` writes. Matches the rest of this API's discipline (ADR 0048): this
module computes nothing new about risk, geography, or a decision's validity; it only reads,
paginates, and aggregates already-computed columns for display, and stages writes it never
applies itself (ADR 0049).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from sentinel.api.errors import ArtifactNotFound, RowNotFound, ValidationRefused
from sentinel.api.schemas.common import Page, PageMeta, StagedRequestReceipt
from sentinel.api.schemas.plan_review import (
    PlanApprovalIn,
    PlanApprovalOut,
    PlanDecisionIn,
    PlanDecisionLogRowOut,
    PlanRowOut,
    PlanSummaryOut,
    WorkBlockOut,
)
from sentinel.api.services.artifacts import read_table, resolve_latest, run_info
from sentinel.api.services.entity_service import join_establishment_identity
from sentinel.api.services.establishment_service import _HISTORY_FACTOR_FIELDS
from sentinel.api.services.pagination import PageParams, slice_frame
from sentinel.api.services.staging_service import StagingService
from sentinel.config import Settings
from sentinel.geographic_organization.definitions import UNMAPPED_GROUP_ID
from sentinel.manifest import manifest_path_for, read_manifest_as
from sentinel.plan_review.approval import PlanApprovalGovernanceError, parse_approval
from sentinel.plan_review.definitions import derive_plan_approval_status
from sentinel.plan_review.models import ApprovedPlanManifest
from sentinel.plan_review.resolution import PlanReviewGovernanceError, parse_decisions


def _resolve_plan_review(settings: Settings, planning_date: str | None) -> Path:
    directory = settings.plan_review_processed_dir
    if planning_date is not None:
        candidates = sorted(directory.glob(f"supervisor_plan_review_{planning_date}_*.parquet"))
        if not candidates:
            raise ArtifactNotFound(
                f"No supervisor_plan_review artifact for planning_date={planning_date!r}. "
                "Run `sentinel review-plan` for that date first.",
                component="supervisor_plan_review",
            )
        return candidates[-1]
    return resolve_latest(directory, prefix="supervisor_plan_review")


def _latest_approval_path(settings: Settings, planning_date: str) -> Path | None:
    directory = settings.plan_review_processed_dir
    candidates = sorted(directory.glob(f"approved_operational_plan_{planning_date}_*.parquet"))
    return candidates[-1] if candidates else None


def get_plan_summary(settings: Settings, planning_date: str | None) -> PlanSummaryOut:
    path = _resolve_plan_review(settings, planning_date)
    frame = read_table(path)
    total = frame.height
    decided = int(frame.filter(pl.col("supervisor_decision_action").is_not_null()).height)
    resolved_date = str(frame["planning_date"][0]) if total else (planning_date or "")
    is_approved = bool(resolved_date) and _latest_approval_path(settings, resolved_date) is not None
    status = derive_plan_approval_status(total=total, decided=decided, is_approved=is_approved)
    return PlanSummaryOut(
        planning_date=resolved_date,
        selected_inspection_workload=total,
        location_available_count=int(
            frame.filter(pl.col("location_status") == "location_available").height
        ),
        location_unavailable_count=int(
            frame.filter(pl.col("location_status") == "location_unavailable").height
        ),
        work_block_count=int(
            frame.filter(pl.col("work_block_id") != UNMAPPED_GROUP_ID)["work_block_id"].n_unique()
        ),
        decisions_recorded=decided,
        approval_status=status.value,
    )


def _history_factors_lookup(settings: Settings) -> dict[str, dict[str, object]]:
    """Component 17's own as-of feature row, reused verbatim -- see `PlanRowOut.history_factors`.

    Reads, never computes: `compute_operational_features` calls Component 4's own
    `historical.aggregate_sql`/`features_sql` unmodified, so this table carries the same
    `_HISTORY_FACTOR_FIELDS` columns under the same names as `as_of_features`. Absence of the
    artifact (not built yet) degrades to no history factors, never an error -- this is a
    supplementary "why" panel, not a required field of the plan row.
    """
    try:
        path = resolve_latest(
            settings.operational_candidates_processed_dir, prefix="operational_candidates"
        )
    except ArtifactNotFound:
        return {}
    frame = read_table(path)
    if "target_inspection_id" not in frame.columns:
        return {}
    present_fields = [f for f in _HISTORY_FACTOR_FIELDS if f in frame.columns]
    frame = frame.select(["target_inspection_id", *present_fields])
    lookup: dict[str, dict[str, object]] = {}
    for row in frame.iter_rows(named=True):
        tiid = row.pop("target_inspection_id")
        lookup[str(tiid)] = row
    return lookup


def list_plan_rows(
    settings: Settings, planning_date: str | None, page: PageParams
) -> Page[PlanRowOut]:
    path = _resolve_plan_review(settings, planning_date)
    frame = read_table(path)
    frame = join_establishment_identity(frame, settings)
    frame = frame.sort(page.sort_column or "suggested_order_in_block", descending=page.descending)
    rows, total = slice_frame(frame, page)
    history_factors = _history_factors_lookup(settings)
    for row in rows:
        row["history_factors"] = history_factors.get(str(row["target_inspection_id"]))
    return Page(
        data=[PlanRowOut.model_validate(row) for row in rows],
        page=PageMeta(offset=page.offset, limit=page.limit, total=total),
        run=run_info(path),
    )


def get_plan_row(
    settings: Settings, target_inspection_id: str, planning_date: str | None
) -> PlanRowOut:
    path = _resolve_plan_review(settings, planning_date)
    frame = read_table(path)
    frame = frame.filter(pl.col("target_inspection_id") == target_inspection_id)
    if frame.height == 0:
        raise RowNotFound(f"No plan row for target_inspection_id={target_inspection_id!r}.")
    frame = join_establishment_identity(frame, settings)
    row = dict(frame.row(0, named=True))
    row["history_factors"] = _history_factors_lookup(settings).get(str(target_inspection_id))
    return PlanRowOut.model_validate(row)


def list_work_blocks(settings: Settings, planning_date: str | None) -> list[WorkBlockOut]:
    path = _resolve_plan_review(settings, planning_date)
    frame = read_table(path)
    blocks: list[WorkBlockOut] = []
    for block_id, group in frame.group_by("work_block_id", maintain_order=True):
        bid = block_id[0] if isinstance(block_id, tuple) else block_id
        ranks = sorted(r for r in group["policy_rank"].drop_nulls().to_list())
        decided = int(group.filter(pl.col("supervisor_decision_action").is_not_null()).height)
        blocks.append(
            WorkBlockOut(
                work_block_id=str(bid),
                work_block_label=str(group["work_block_label"][0]),
                size=group.height,
                highest_sentinel_rank=ranks[0] if ranks else None,
                rank_range=[ranks[0], ranks[-1]] if ranks else None,
                is_unmapped=bid == UNMAPPED_GROUP_ID,
                decisions_recorded=decided,
            )
        )
    blocks.sort(key=lambda b: (b.is_unmapped, b.highest_sentinel_rank or 10**9))
    return blocks


def get_plan_decision_log(
    settings: Settings, planning_date: str | None, page: PageParams
) -> Page[PlanDecisionLogRowOut]:
    directory = settings.plan_review_processed_dir
    if planning_date is not None:
        candidates = sorted(directory.glob(f"plan_decision_log_{planning_date}_*.parquet"))
        if not candidates:
            raise ArtifactNotFound(
                f"No plan_decision_log artifact for planning_date={planning_date!r}.",
                component="plan_decision_log",
            )
        path = candidates[-1]
    else:
        path = resolve_latest(directory, prefix="plan_decision_log")
    frame = read_table(path).with_columns(pl.lit("committed").alias("status"))
    frame = frame.sort("decision_id")
    rows, total = slice_frame(frame, page)
    return Page(
        data=[PlanDecisionLogRowOut.model_validate(row) for row in rows],
        page=PageMeta(offset=page.offset, limit=page.limit, total=total),
        run=run_info(path),
    )


def stage_plan_decision(
    settings: Settings, payload: PlanDecisionIn, staging: StagingService
) -> StagedRequestReceipt:
    """Validate against Component 21's own contract, then append -- never apply.

    Runs the payload through ``parse_decisions`` (Component 21's real parser) as a
    one-element list: if the batch CLI would refuse this row, the API refuses it too, with
    the parser's own message. Same discipline as ``review_service.stage_resolution``.
    """
    record = payload.model_dump()
    try:
        parse_decisions([record])
    except PlanReviewGovernanceError as exc:
        raise ValidationRefused(str(exc)) from exc

    try:
        committed_path = resolve_latest(
            settings.plan_review_processed_dir, prefix="plan_decision_log"
        )
        committed_ids = set(read_table(committed_path).get_column("decision_id").to_list())
    except ArtifactNotFound:
        committed_ids = set()
    return staging.append(
        kind="plan_decision",
        natural_id=payload.decision_id,
        record=record,
        committed_ids=committed_ids,
    )


def get_plan_approval(settings: Settings, planning_date: str | None) -> PlanApprovalOut:
    """The latest committed approval for a planning date. Read-only, like every other GET."""
    resolved_date = planning_date
    if resolved_date is None:
        path = _resolve_plan_review(settings, None)
        resolved_date = str(read_table(path)["planning_date"][0])
    approval_path = _latest_approval_path(settings, resolved_date)
    if approval_path is None:
        raise ArtifactNotFound(
            f"No approved_operational_plan for planning_date={resolved_date!r}. "
            "Run `sentinel approve-plan` for that date first.",
            component="approved_operational_plan",
        )
    manifest = read_manifest_as(ApprovedPlanManifest, manifest_path_for(approval_path))
    return PlanApprovalOut(
        approval_id=manifest.approval_id,
        planning_date=manifest.planning_date,
        approved_by=manifest.approved_by,
        approved_at=manifest.approved_at,
        note=manifest.note,
        final_selected_count=manifest.final_selected_count,
        final_active_count=manifest.final_active_count,
        final_deferred_count=manifest.final_deferred_count,
        final_not_proceeding_count=manifest.final_not_proceeding_count,
        final_undecided_count=manifest.final_undecided_count,
        source_plan_review_path=manifest.source_plan_review_path,
        source_plan_review_sha256=manifest.source_plan_review_sha256,
    )


def stage_plan_approval(
    settings: Settings, payload: PlanApprovalIn, staging: StagingService
) -> StagedRequestReceipt:
    """Validate against Component 21's own approval contract, then append -- never apply.

    This stages *intent* only. The actual readiness checklist (every row carries the
    machine recommendation, every decision has a reason, ...) runs when an operator commits
    the staged approval through ``sentinel approve-plan``, exactly like a plan decision.
    """
    record = payload.model_dump()
    try:
        parse_approval(record)
    except PlanApprovalGovernanceError as exc:
        raise ValidationRefused(str(exc)) from exc

    committed_ids: set[str] = set()
    approval_path = _latest_approval_path(settings, payload.planning_date)
    if approval_path is not None:
        committed_ids = {
            str(v)
            for v in read_table(approval_path)
            .get_column("approval_id")
            .drop_nulls()
            .unique()
            .to_list()
        }
    return staging.append(
        kind="plan_approval",
        natural_id=payload.approval_id,
        record=record,
        committed_ids=committed_ids,
    )


__all__ = [
    "get_plan_approval",
    "get_plan_decision_log",
    "get_plan_row",
    "get_plan_summary",
    "list_plan_rows",
    "list_work_blocks",
    "stage_plan_approval",
    "stage_plan_decision",
]
