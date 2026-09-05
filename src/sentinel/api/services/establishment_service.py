"""The establishment-detail bundle: one product view composed from four independent artifacts.

Composes, never computes. Every field here was already written by Components 11, 13 or 14; this
module's only job is to find the right rows under an unambiguous scope and put them beside each
other without merging what they mean (ADR 0042, ADR 0050).
"""

from __future__ import annotations

import polars as pl

from sentinel.api.errors import AmbiguousScope, ArtifactNotFound, RowNotFound
from sentinel.api.schemas.common import DecisionScope
from sentinel.api.schemas.establishment import EstablishmentHistoryOut, RiskHistoryFactorsOut
from sentinel.api.schemas.policy import RecommendationOut
from sentinel.api.schemas.scheduling import ScheduleRowOut
from sentinel.api.services.artifacts import (
    apply_scope_filter,
    candidate_values,
    read_table,
    require_scope,
    resolve_latest,
)
from sentinel.api.services.entity_service import establishment_identity_row
from sentinel.api.services.explain_service import base_model_name_of, get_explanation
from sentinel.api.services.policy_service import RECOMMENDATION_SCOPE
from sentinel.api.services.scheduling_service import pin_to_latest_replan
from sentinel.config import Settings

_HISTORY_FACTOR_FIELDS: tuple[str, ...] = (
    "prior_canvass_count_code_era",
    "prior_canvass_priority_count",
    "prior_canvass_priority_rate",
    "prior_canvass_fail_rate",
    "fail_at_last_canvass",
    "priority_at_last_canvass",
    "days_since_last_canvass",
    "days_since_any_inspection",
    "prior_inspection_count_any_type",
    "name_changed_since_last_canvass",
)


def _load_history_factors(
    settings: Settings, target_inspection_id: str
) -> tuple[RiskHistoryFactorsOut | None, str | None]:
    """A curated slice of Component 4's own feature row for this prediction opportunity.

    Reads, never computes: every value returned already existed in the feature table before
    this function ran. Absence (no feature table built yet, or this row outside it) is reported
    as a reason rather than silently omitted, matching the explanation section's pattern.
    """
    try:
        path = resolve_latest(settings.features_processed_dir, prefix="as_of_features")
    except ArtifactNotFound as exc:
        return None, str(exc)

    frame = read_table(path).filter(pl.col("target_inspection_id") == target_inspection_id)
    if frame.height == 0:
        return None, (
            f"No feature history found for target_inspection_id={target_inspection_id!r} in "
            "the current feature table."
        )
    row = frame.row(0, named=True)
    factors = {field: row.get(field) for field in _HISTORY_FACTOR_FIELDS}
    return RiskHistoryFactorsOut(**factors), None


def get_establishment_history(
    settings: Settings, establishment_id: str, scope: DecisionScope
) -> EstablishmentHistoryOut:
    require_scope(scope, required=RECOMMENDATION_SCOPE)

    rec_path = resolve_latest(settings.policy_processed_dir, prefix="inspection_recommendations")
    rec_frame = apply_scope_filter(read_table(rec_path), scope)
    rec_frame = rec_frame.filter(pl.col("establishment_id") == establishment_id)
    if rec_frame.height == 0:
        raise RowNotFound(
            f"No recommendation for establishment_id={establishment_id!r} under the given scope."
        )
    if rec_frame.height > 1:
        raise AmbiguousScope(
            f"establishment_id={establishment_id!r} matches {rec_frame.height} recommendation "
            "rows under the given scope. Add target_inspection_id, or query "
            "/v1/recommendations/{target_inspection_id} directly.",
            candidate_values=candidate_values(rec_frame, "target_inspection_id"),
        )
    recommendation = RecommendationOut(**rec_frame.row(0, named=True))

    schedule: ScheduleRowOut | None = None
    if scope.schedule_config_id is not None:
        sched_path = resolve_latest(settings.scheduling_processed_dir, prefix="inspection_schedule")
        sched_scope = scope.model_copy(update={"model_name": recommendation.model_name})
        sched_frame = apply_scope_filter(read_table(sched_path), sched_scope)
        sched_frame = sched_frame.filter(pl.col("establishment_id") == establishment_id)
        sched_frame = pin_to_latest_replan(sched_frame, scope)
        if sched_frame.height > 1:
            raise AmbiguousScope(
                f"establishment_id={establishment_id!r} matches {sched_frame.height} schedule "
                "rows under the given scope. Add planning_run_id.",
                candidate_values=candidate_values(sched_frame, "planning_run_id"),
            )
        if sched_frame.height == 1:
            schedule = ScheduleRowOut(**sched_frame.row(0, named=True))

    explanation = None
    explanation_unavailable_reason: str | None = None
    try:
        explanation = get_explanation(
            settings,
            recommendation.target_inspection_id,
            DecisionScope(
                # Component 13 carries Component 9's *calibrated* name (e.g. "xgboost_platt");
                # Component 11's tables carry the *base* name ("xgboost") and never a calibrated
                # one -- see docs/data_contracts/explanations.md 0a. Resolve it here, once.
                model_name=base_model_name_of(recommendation.model_name),
                fold_set=recommendation.fold_set,
                fold_id=recommendation.fold_id,
            ),
        )
    except (RowNotFound, ArtifactNotFound) as exc:
        # Explanations are an optional, sampled-subset layer (ADR 0028): a model that has never
        # been explained, or a fold nobody explained yet, must not sink the whole bundle -- the
        # reason travels in the field a caller actually reads instead.
        explanation_unavailable_reason = str(exc)

    history_factors, history_factors_unavailable_reason = _load_history_factors(
        settings, recommendation.target_inspection_id
    )

    establishment_name, establishment_address = establishment_identity_row(
        settings, establishment_id
    )

    return EstablishmentHistoryOut(
        establishment_id=establishment_id,
        establishment_name=establishment_name,
        establishment_address=establishment_address,
        recommendation=recommendation,
        schedule=schedule,
        explanation=explanation,
        explanation_unavailable_reason=explanation_unavailable_reason,
        history_factors=history_factors,
        history_factors_unavailable_reason=history_factors_unavailable_reason,
    )


__all__ = ["get_establishment_history"]
