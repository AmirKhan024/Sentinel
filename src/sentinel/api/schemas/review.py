"""Response/request shapes for Component 16's artifacts.

Field lists mirror ``sentinel.review.writer.HUMAN_REVIEW_QUEUE_SCHEMA`` /
``REVIEW_RESOLUTION_LOG_SCHEMA`` exactly. The column contract lives in one place -- the writer --
and this module is a typed view onto it for JSON, not a second definition of it.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from sentinel.review.models import ReviewResolution


class ReviewCaseOut(BaseModel):
    policy_id: str
    model_name: str
    fold_set: str
    fold_id: str
    k_name: str
    target_inspection_id: str
    establishment_id: str
    establishment_name: str | None = None
    establishment_address: str | None = None
    final_policy_rank: int | None
    decision_mechanism: str
    decision_reason: str
    warnings: str
    trigger_reasons: str
    schedule_config_id: str
    planning_run_id: str
    replan_index: int | None
    scheduled_date: date | None
    review_status: str
    review_id: str
    resolution_action: str
    review_definition_version: str
    #: Always "committed" -- `get_review_queue`/`get_review_case` read only `human_review_queue`,
    #: never the staging store. A "pending_review" presentation status (a flagged case with a
    #: resolution already staged but not yet committed) is not implemented; a still-pending
    #: resolution is visible only via `GET /v1/staged-requests`. See "Correction: the four
    #: log-read endpoints do not merge staged rows" in docs/data_contracts/sentinel_api.md.
    status: str = "committed"


class ResolutionLogRowOut(BaseModel):
    review_id: str
    policy_id: str
    fold_id: str
    k_name: str
    target_inspection_id: str
    resolution_action: str
    reason_code: str
    actor: str
    decided_at: str
    referenced_override_id: str
    referenced_adjustment_id: str
    escalation_note: str
    original_status: str
    final_status: str
    outcome: str
    review_definition_version: str
    #: Always "committed" -- `get_resolution_log` reads only `review_resolution_log`, never the
    #: staging store. A still-pending resolution is visible only via `GET /v1/staged-requests`.
    status: str = "committed"


class ResolutionIn(BaseModel):
    """Exactly Component 16's ``ReviewResolution`` contract -- the API adds no field of its own.

    Validated by running the payload through ``sentinel.review.resolution.parse_resolutions`` as
    a one-element list, so a staged request can never be accepted in a shape the batch CLI would
    go on to refuse.
    """

    review_id: str
    policy_id: str
    fold_id: str
    k_name: str
    target_inspection_id: str
    resolution_action: str
    reason_code: str
    actor: str
    decided_at: str
    referenced_override_id: str | None = None
    referenced_adjustment_id: str | None = None
    escalation_note: str | None = None

    model_config = {"extra": "forbid"}

    def to_resolution(self) -> ReviewResolution:
        return ReviewResolution(**self.model_dump())


__all__ = [
    "ResolutionIn",
    "ResolutionLogRowOut",
    "ReviewCaseOut",
]
