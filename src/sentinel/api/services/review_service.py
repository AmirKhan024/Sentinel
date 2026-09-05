"""Read and stage access to Component 16's artifacts. No triggering, no resolving, no scoring."""

from __future__ import annotations

import polars as pl

from sentinel.api.errors import ArtifactNotFound, RowNotFound, ValidationRefused
from sentinel.api.schemas.common import DecisionScope, Page, PageMeta, StagedRequestReceipt
from sentinel.api.schemas.review import ResolutionIn, ResolutionLogRowOut, ReviewCaseOut
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
from sentinel.review.resolution import ReviewGovernanceError, parse_resolutions

REVIEW_SCOPE = ("policy_id", "fold_set", "fold_id", "k_name")


def get_review_queue(
    settings: Settings, scope: DecisionScope, page: PageParams, *, trigger: str | None = None
) -> Page[ReviewCaseOut]:
    """``trigger``, when given, keeps only cases whose ``trigger_reasons`` contains that exact
    code (e.g. ``policy_warning_present``) -- a literal substring match on the existing
    pipe-joined column, not a new classification. The set of trigger codes a case can carry
    remains entirely Component 16's; this only lets a caller ask for one of them."""
    require_scope(scope, required=REVIEW_SCOPE)
    path = resolve_latest(settings.review_processed_dir, prefix="human_review_queue")
    frame = apply_scope_filter(read_table(path), scope)
    if trigger is not None:
        frame = frame.filter(pl.col("trigger_reasons").str.contains(trigger, literal=True))
    frame = frame.with_columns(pl.lit("committed").alias("status"))
    frame = frame.sort(page.sort_column or "target_inspection_id", descending=page.descending)
    frame = join_establishment_identity(frame, settings)
    rows, total = slice_frame(frame, page)
    return Page(
        data=[ReviewCaseOut.model_validate(row) for row in rows],
        page=PageMeta(offset=page.offset, limit=page.limit, total=total),
        run=run_info(path),
    )


def get_review_case(
    settings: Settings, target_inspection_id: str, scope: DecisionScope
) -> ReviewCaseOut:
    require_scope(scope, required=REVIEW_SCOPE)
    path = resolve_latest(settings.review_processed_dir, prefix="human_review_queue")
    frame = apply_scope_filter(read_table(path), scope)
    frame = frame.filter(pl.col("target_inspection_id") == target_inspection_id)
    if frame.height == 0:
        raise RowNotFound(
            f"No review case for target_inspection_id={target_inspection_id!r} under the "
            "given scope."
        )
    frame = join_establishment_identity(frame, settings)
    row = dict(frame.row(0, named=True))
    row["status"] = "committed"
    return ReviewCaseOut.model_validate(row)


def get_resolution_log(
    settings: Settings,
    scope: DecisionScope,
    page: PageParams,
    *,
    target_inspection_id: str | None = None,
) -> Page[ResolutionLogRowOut]:
    require_scope(scope, required=REVIEW_SCOPE)
    path = resolve_latest(settings.review_processed_dir, prefix="review_resolution_log")
    frame = apply_scope_filter(read_table(path), scope)
    if target_inspection_id is not None:
        frame = frame.filter(pl.col("target_inspection_id") == target_inspection_id)
    frame = frame.with_columns(pl.lit("committed").alias("status"))
    frame = frame.sort("review_id")
    rows, total = slice_frame(frame, page)
    return Page(
        data=[ResolutionLogRowOut.model_validate(row) for row in rows],
        page=PageMeta(offset=page.offset, limit=page.limit, total=total),
        run=run_info(path),
    )


def stage_resolution(
    settings: Settings, payload: ResolutionIn, staging: StagingService
) -> StagedRequestReceipt:
    """Validate against Component 16's own contract, then append -- never apply.

    Runs the payload through ``parse_resolutions`` (Component 16's real parser) as a
    one-element list purely for validation: if the batch CLI would refuse this row, the API
    refuses it too, with the parser's own message. Same discipline as
    ``policy_service.stage_override``. See ADR 0049.
    """
    record = payload.model_dump()
    try:
        parse_resolutions([record])
    except ReviewGovernanceError as exc:
        raise ValidationRefused(str(exc)) from exc

    try:
        committed_path = resolve_latest(
            settings.review_processed_dir, prefix="review_resolution_log"
        )
        committed_ids = set(read_table(committed_path).get_column("review_id").to_list())
    except ArtifactNotFound:
        committed_ids = set()
    return staging.append(
        kind="review_resolution",
        natural_id=payload.review_id,
        record=record,
        committed_ids=committed_ids,
    )


__all__ = [
    "get_resolution_log",
    "get_review_case",
    "get_review_queue",
    "stage_resolution",
]
