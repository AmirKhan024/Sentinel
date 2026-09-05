"""Response/request shapes for Component 13's artifacts.

Field lists mirror ``sentinel.policy.writer.RECOMMENDATIONS_SCHEMA`` /
``OVERRIDE_LOG_SCHEMA`` / ``SELECTION_ALLOCATION_SCHEMA`` exactly. The column contract lives in
one place -- the writer -- and this module is a typed view onto it for JSON, not a second
definition of it.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from sentinel.policy.models import Override


class RecommendationOut(BaseModel):
    policy_id: str
    model_name: str
    fold_set: str
    fold_id: str
    k_name: str
    k: int
    target_inspection_id: str
    establishment_id: str
    #: Display-only, joined in from Component 2's entity resolution (`entity_service`). ``None``
    #: only when that artifact hasn't been built in this environment -- never a data quality flag.
    establishment_name: str | None = None
    establishment_address: str | None = None
    inspection_date: date
    base_score: float
    score: float
    model_rank: int
    final_policy_rank: int | None
    is_selected: bool
    decision_mechanism: str
    decision_reason: str
    coverage_eligible: bool
    secondary_no_history: bool
    warnings: str
    group_value: str
    group_status: str
    policy_definition_version: str


class AllocationOut(BaseModel):
    policy_id: str
    model_name: str
    fold_set: str
    fold_id: str
    k_name: str
    k: int
    n_universe: int
    reserve_mechanism: str
    reserve_share: float
    reserve_target: int
    n_eligible_available: int
    n_eligible_in_risk_top_k: int
    n_risk: int
    n_reserve: int
    n_selected: int
    reserve_inert: bool
    policy_definition_version: str


class OverrideLogRowOut(BaseModel):
    override_id: str
    policy_id: str
    fold_set: str
    fold_id: str
    k_name: str
    target_inspection_id: str
    action: str
    reason_code: str
    actor: str
    decided_at: str
    original_is_selected: bool
    original_mechanism: str
    original_reason: str
    original_policy_rank: int
    final_is_selected: bool
    displaced_target_inspection_id: str
    outcome: str
    policy_definition_version: str
    #: Always "committed" -- `get_override_log` reads only `policy_override_log`, never the
    #: staging store. A still-pending override is visible only via `GET /v1/staged-requests`;
    #: see "Correction: the four log-read endpoints do not merge staged rows" in
    #: docs/data_contracts/sentinel_api.md.
    status: str = "committed"


class OverrideIn(BaseModel):
    """Exactly Component 13's ``Override`` contract -- the API adds no field of its own.

    Validated by constructing the real ``Override`` model (and, before that, by running the
    payload through ``sentinel.policy.governance.parse_overrides`` as a one-element list) so a
    staged request can never be accepted in a shape the batch CLI would go on to refuse.
    """

    override_id: str
    policy_id: str
    fold_id: str
    k_name: str
    target_inspection_id: str
    action: str
    reason_code: str
    actor: str
    decided_at: str

    model_config = {"extra": "forbid"}

    def to_override(self) -> Override:
        return Override(**self.model_dump())


__all__ = [
    "AllocationOut",
    "OverrideIn",
    "OverrideLogRowOut",
    "RecommendationOut",
]
