"""The cross-artifact establishment-detail bundle: one product view over four layers."""

from __future__ import annotations

from pydantic import BaseModel

from sentinel.api.schemas.explain import ExplanationCaseOut
from sentinel.api.schemas.policy import RecommendationOut
from sentinel.api.schemas.scheduling import ScheduleRowOut


class RiskHistoryFactorsOut(BaseModel):
    """A handful of Component 4's as-of feature values, surfaced for one establishment.

    These are not a new computation: every field here is a column that already existed in
    ``as_of_features_*.parquet`` and already fed the model that produced the recommendation
    being explained. Selected because each one is independently meaningful to a non-technical
    reader without a model in front of them -- unlike a SHAP value, which only means something
    relative to a background distribution. This is a deliberately small, curated subset of the
    26-feature table, not the full feature vector (that stays in the model explanation section).
    """

    prior_canvass_count_code_era: int | None = None
    prior_canvass_priority_count: int | None = None
    prior_canvass_priority_rate: float | None = None
    prior_canvass_fail_rate: float | None = None
    fail_at_last_canvass: bool | None = None
    priority_at_last_canvass: bool | None = None
    days_since_last_canvass: int | None = None
    days_since_any_inspection: int | None = None
    prior_inspection_count_any_type: int | None = None
    name_changed_since_last_canvass: bool | None = None


class EstablishmentHistoryOut(BaseModel):
    """Risk, policy selection, schedule assignment and explanation, kept as distinct fields.

    Never merged into one ambiguous "reason": ``recommendation.decision_reason`` is Component
    13's, ``schedule.schedule_reason`` is Component 14's, and ``explanation`` is Component 11's,
    per ADR 0042. A caller that wants "why was this establishment scheduled for Thursday" reads
    ``schedule.schedule_reason``; "why was it selected at all" reads
    ``recommendation.decision_reason``. Neither answers the other. ``history_factors`` answers a
    third, narrower question -- "what does Sentinel actually know about this establishment's
    inspection history" -- and is present even when ``explanation`` is not, since it does not
    depend on the sampled explanation layer.
    """

    establishment_id: str
    #: Display-only, from Component 2's entity resolution -- see `entity_service`. ``None`` only
    #: when that artifact hasn't been built in this environment.
    establishment_name: str | None = None
    establishment_address: str | None = None
    recommendation: RecommendationOut
    schedule: ScheduleRowOut | None = None
    explanation: ExplanationCaseOut | None = None
    explanation_unavailable_reason: str | None = None
    history_factors: RiskHistoryFactorsOut | None = None
    history_factors_unavailable_reason: str | None = None


__all__ = ["EstablishmentHistoryOut", "RiskHistoryFactorsOut"]
