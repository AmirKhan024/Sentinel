"""Read and stage access to Component 13's artifacts. No allocation, no selection, no scoring."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from sentinel.api.errors import ArtifactNotFound, RowNotFound, ValidationRefused
from sentinel.api.schemas.common import DecisionScope, Page, PageMeta, StagedRequestReceipt
from sentinel.api.schemas.policy import (
    AllocationOut,
    OverrideIn,
    OverrideLogRowOut,
    RecommendationOut,
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
from sentinel.policy.governance import GovernanceError, parse_overrides

RECOMMENDATION_SCOPE = ("policy_id", "fold_set", "fold_id", "k_name")


def _recommendations_frame(settings: Settings) -> tuple[pl.DataFrame, Path]:
    path = resolve_latest(settings.policy_processed_dir, prefix="inspection_recommendations")
    return read_table(path), path


def get_recommendations(
    settings: Settings,
    scope: DecisionScope,
    page: PageParams,
    *,
    establishment_id: str | None = None,
    is_selected: bool | None = None,
) -> Page[RecommendationOut]:
    require_scope(scope, required=RECOMMENDATION_SCOPE)
    frame, path = _recommendations_frame(settings)
    frame = apply_scope_filter(frame, scope)
    if establishment_id is not None:
        frame = frame.filter(pl.col("establishment_id") == establishment_id)
    if is_selected is not None:
        frame = frame.filter(pl.col("is_selected") == is_selected)
    frame = frame.sort(page.sort_column or "final_policy_rank", descending=page.descending)
    frame = join_establishment_identity(frame, settings)
    rows, total = slice_frame(frame, page)
    return Page(
        data=[RecommendationOut.model_validate(row) for row in rows],
        page=PageMeta(offset=page.offset, limit=page.limit, total=total),
        run=run_info(path),
    )


def get_recommendation(
    settings: Settings, target_inspection_id: str, scope: DecisionScope
) -> RecommendationOut:
    require_scope(scope, required=RECOMMENDATION_SCOPE)
    frame, _path = _recommendations_frame(settings)
    frame = apply_scope_filter(frame, scope)
    frame = frame.filter(pl.col("target_inspection_id") == target_inspection_id)
    if frame.height == 0:
        raise RowNotFound(
            f"No recommendation for target_inspection_id={target_inspection_id!r} under the "
            "given scope."
        )
    frame = join_establishment_identity(frame, settings)
    return RecommendationOut.model_validate(frame.row(0, named=True))


def get_selection_allocation(settings: Settings, scope: DecisionScope) -> list[AllocationOut]:
    require_scope(scope, required=RECOMMENDATION_SCOPE)
    path = resolve_latest(settings.policy_processed_dir, prefix="policy_selection_allocation")
    frame = apply_scope_filter(read_table(path), scope)
    return [AllocationOut.model_validate(row) for row in frame.sort("policy_id").to_dicts()]


def get_override_log(
    settings: Settings,
    scope: DecisionScope,
    page: PageParams,
    *,
    target_inspection_id: str | None = None,
) -> Page[OverrideLogRowOut]:
    require_scope(scope, required=RECOMMENDATION_SCOPE)
    path = resolve_latest(settings.policy_processed_dir, prefix="policy_override_log")
    frame = apply_scope_filter(read_table(path), scope)
    if target_inspection_id is not None:
        frame = frame.filter(pl.col("target_inspection_id") == target_inspection_id)
    frame = frame.with_columns(pl.lit("committed").alias("status"))
    frame = frame.sort("override_id")
    rows, total = slice_frame(frame, page)
    return Page(
        data=[OverrideLogRowOut.model_validate(row) for row in rows],
        page=PageMeta(offset=page.offset, limit=page.limit, total=total),
        run=run_info(path),
    )


def stage_override(
    settings: Settings, payload: OverrideIn, staging: StagingService
) -> StagedRequestReceipt:
    """Validate against Component 13's own contract, then append -- never apply.

    Runs the payload through ``parse_overrides`` (Component 13's real parser) as a one-element
    list purely for validation: if the batch CLI would refuse this row, the API refuses it too,
    with the parser's own message. See ADR 0049.
    """
    record = payload.model_dump()
    try:
        parse_overrides([record])
    except GovernanceError as exc:
        raise ValidationRefused(str(exc)) from exc

    try:
        committed_path = resolve_latest(settings.policy_processed_dir, prefix="policy_override_log")
        committed_ids = set(read_table(committed_path).get_column("override_id").to_list())
    except ArtifactNotFound:
        committed_ids = set()
    return staging.append(
        kind="override",
        natural_id=payload.override_id,
        record=record,
        committed_ids=committed_ids,
    )


__all__ = [
    "get_override_log",
    "get_recommendation",
    "get_recommendations",
    "get_selection_allocation",
    "stage_override",
]
