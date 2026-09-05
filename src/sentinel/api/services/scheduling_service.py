"""Read and stage access to Component 14's artifacts. No allocation, no placement, no replanning."""

from __future__ import annotations

from datetime import date

import polars as pl

from sentinel.api.errors import ArtifactNotFound, RowNotFound, ValidationRefused
from sentinel.api.schemas.common import DecisionScope, Page, PageMeta, StagedRequestReceipt
from sentinel.api.schemas.scheduling import (
    AdjustmentIn,
    AdjustmentLogRowOut,
    BacklogRowOut,
    ExecutionEventIn,
    ExecutionLogRowOut,
    ExecutionSummaryOut,
    ReplanningRunOut,
    ScheduleRowOut,
)
from sentinel.api.services.artifacts import (
    apply_scope_filter,
    read_table,
    require_scope,
    resolve_latest,
    run_info,
)
from sentinel.api.services.entity_service import join_establishment_identity
from sentinel.api.services.pagination import PageParams, slice_frame
from sentinel.api.services.staging_service import StagingService
from sentinel.config import Settings
from sentinel.scheduling.adjustments import AdjustmentError, parse_adjustments
from sentinel.scheduling.execution import ExecutionError, parse_execution_events

SCHEDULE_SCOPE = ("schedule_config_id", "policy_id", "fold_set", "fold_id", "k_name")


def pin_to_latest_replan(frame: pl.DataFrame, scope: DecisionScope) -> pl.DataFrame:
    """Default ``planning_run_id``/``replan_index`` to the cell's latest planning run."""
    if scope.replan_index is not None:
        return frame.filter(pl.col("replan_index") == scope.replan_index)
    if scope.planning_run_id is not None:
        return frame.filter(pl.col("planning_run_id") == scope.planning_run_id)
    if frame.height == 0:
        return frame
    latest_index = frame.get_column("replan_index").max()
    return frame.filter(pl.col("replan_index") == latest_index)


def get_schedule(
    settings: Settings,
    scope: DecisionScope,
    page: PageParams,
    *,
    establishment_id: str | None = None,
    schedule_status: str | None = None,
    scheduled_date: date | None = None,
) -> Page[ScheduleRowOut]:
    require_scope(scope, required=SCHEDULE_SCOPE)
    path = resolve_latest(settings.scheduling_processed_dir, prefix="inspection_schedule")
    frame = apply_scope_filter(read_table(path), scope)
    frame = pin_to_latest_replan(frame, scope)
    if establishment_id is not None:
        frame = frame.filter(pl.col("establishment_id") == establishment_id)
    if schedule_status is not None:
        frame = frame.filter(pl.col("schedule_status") == schedule_status)
    if scheduled_date is not None:
        frame = frame.filter(pl.col("scheduled_date") == scheduled_date)
    frame = frame.sort(page.sort_column or "final_policy_rank", descending=page.descending)
    frame = join_establishment_identity(frame, settings)
    rows, total = slice_frame(frame, page)
    return Page(
        data=[ScheduleRowOut.model_validate(row) for row in rows],
        page=PageMeta(offset=page.offset, limit=page.limit, total=total),
        run=run_info(path),
    )


def get_schedule_dates(settings: Settings, scope: DecisionScope) -> list[dict[str, object]]:
    """Distinct planned dates in this plan, each with how many establishments occupy it.

    A display aggregation over rows the caller could otherwise page through and count itself --
    not a new inference (ADR 0048: the API computes nothing beyond what's already decided). Exists
    so the frontend can offer a real day picker (only dates that actually have inspections) instead
    of a generic calendar the user could pick an empty day from.
    """
    require_scope(scope, required=SCHEDULE_SCOPE)
    path = resolve_latest(settings.scheduling_processed_dir, prefix="inspection_schedule")
    frame = apply_scope_filter(read_table(path), scope)
    frame = pin_to_latest_replan(frame, scope)
    frame = frame.filter(pl.col("scheduled_date").is_not_null())
    counts = (
        frame.group_by("scheduled_date")
        .agg(pl.len().alias("n_establishments"))
        .sort("scheduled_date")
    )
    return counts.to_dicts()


def get_backlog(settings: Settings, scope: DecisionScope, page: PageParams) -> Page[BacklogRowOut]:
    require_scope(scope, required=SCHEDULE_SCOPE)
    path = resolve_latest(settings.scheduling_processed_dir, prefix="schedule_backlog")
    frame = apply_scope_filter(read_table(path), scope)
    frame = pin_to_latest_replan(frame, scope)
    frame = frame.sort(page.sort_column or "backlog_position", descending=page.descending)
    frame = join_establishment_identity(frame, settings)
    rows, total = slice_frame(frame, page)
    return Page(
        data=[BacklogRowOut.model_validate(row) for row in rows],
        page=PageMeta(offset=page.offset, limit=page.limit, total=total),
        run=run_info(path),
    )


def get_small_table(
    settings: Settings, scope: DecisionScope, *, table: str
) -> list[dict[str, object]]:
    """Non-paginated read for the small per-cell tables (summary, utilization, preservation)."""
    require_scope(scope, required=SCHEDULE_SCOPE)
    path = resolve_latest(settings.scheduling_processed_dir, prefix=table)
    frame = apply_scope_filter(read_table(path), scope)
    return frame.to_dicts()


def get_replanning_runs(settings: Settings, scope: DecisionScope) -> list[ReplanningRunOut]:
    require_scope(scope, required=SCHEDULE_SCOPE)
    path = resolve_latest(settings.scheduling_processed_dir, prefix="replanning_runs")
    frame = apply_scope_filter(read_table(path), scope).sort("replan_index")
    return [ReplanningRunOut.model_validate(row) for row in frame.to_dicts()]


def get_adjustment_log(
    settings: Settings,
    scope: DecisionScope,
    page: PageParams,
    staging: StagingService,
    *,
    target_inspection_id: str | None = None,
) -> Page[AdjustmentLogRowOut]:
    require_scope(scope, required=SCHEDULE_SCOPE)
    path = resolve_latest(settings.scheduling_processed_dir, prefix="schedule_adjustment_log")
    frame = apply_scope_filter(read_table(path), scope)
    if target_inspection_id is not None:
        frame = frame.filter(pl.col("target_inspection_id") == target_inspection_id)
    frame = frame.with_columns(pl.lit("committed").alias("status")).sort("adjustment_id")
    rows, total = slice_frame(frame, page)
    return Page(
        data=[AdjustmentLogRowOut.model_validate(row) for row in rows],
        page=PageMeta(offset=page.offset, limit=page.limit, total=total),
        run=run_info(path),
    )


def get_execution_events(
    settings: Settings,
    scope: DecisionScope,
    page: PageParams,
    *,
    target_inspection_id: str | None = None,
) -> Page[ExecutionLogRowOut]:
    require_scope(scope, required=SCHEDULE_SCOPE)
    path = resolve_latest(settings.scheduling_processed_dir, prefix="execution_log")
    frame = apply_scope_filter(read_table(path), scope)
    if target_inspection_id is not None:
        frame = frame.filter(pl.col("target_inspection_id") == target_inspection_id)
    frame = frame.with_columns(pl.lit("committed").alias("status")).sort("execution_id")
    rows, total = slice_frame(frame, page)
    return Page(
        data=[ExecutionLogRowOut.model_validate(row) for row in rows],
        page=PageMeta(offset=page.offset, limit=page.limit, total=total),
        run=run_info(path),
    )


def get_execution_summary(settings: Settings, scope: DecisionScope) -> ExecutionSummaryOut:
    require_scope(scope, required=SCHEDULE_SCOPE)
    path = resolve_latest(settings.scheduling_processed_dir, prefix="execution_summary")
    frame = apply_scope_filter(read_table(path), scope)
    if frame.height == 0:
        raise RowNotFound("No execution summary for the given scope.")
    return ExecutionSummaryOut.model_validate(frame.row(0, named=True))


def get_execution_contract(settings: Settings) -> list[dict[str, object]]:
    path = resolve_latest(settings.scheduling_processed_dir, prefix="execution_contract")
    return read_table(path).to_dicts()


def stage_adjustment(
    settings: Settings, payload: AdjustmentIn, staging: StagingService
) -> StagedRequestReceipt:
    record = payload.model_dump()
    try:
        parse_adjustments([record])
    except AdjustmentError as exc:
        raise ValidationRefused(str(exc)) from exc

    try:
        committed_path = resolve_latest(
            settings.scheduling_processed_dir, prefix="schedule_adjustment_log"
        )
        committed_ids = set(read_table(committed_path).get_column("adjustment_id").to_list())
    except ArtifactNotFound:
        committed_ids = set()
    return staging.append(
        kind="adjustment",
        natural_id=payload.adjustment_id,
        record=record,
        committed_ids=committed_ids,
    )


def stage_execution_event(
    settings: Settings, payload: ExecutionEventIn, staging: StagingService
) -> StagedRequestReceipt:
    record = payload.model_dump()
    try:
        parse_execution_events([record])
    except ExecutionError as exc:
        raise ValidationRefused(str(exc)) from exc

    try:
        committed_path = resolve_latest(settings.scheduling_processed_dir, prefix="execution_log")
        committed_ids = set(read_table(committed_path).get_column("execution_id").to_list())
    except ArtifactNotFound:
        committed_ids = set()
    return staging.append(
        kind="execution_event",
        natural_id=payload.execution_id,
        record=record,
        committed_ids=committed_ids,
    )


__all__ = [
    "get_adjustment_log",
    "get_backlog",
    "get_execution_contract",
    "get_execution_events",
    "get_execution_summary",
    "get_replanning_runs",
    "get_schedule",
    "get_schedule_dates",
    "get_small_table",
    "pin_to_latest_replan",
    "stage_adjustment",
    "stage_execution_event",
]
