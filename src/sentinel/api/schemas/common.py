"""Shapes shared across every router: decision scope, pagination, run provenance."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DecisionScope(BaseModel):
    """Every field a caller might need to pick out exactly one artifact cell or row.

    All fields are optional here -- which ones are *required* is a per-endpoint decision, made
    by ``sentinel.api.services.artifacts.require_scope``, never by this model. An establishment
    can appear in many folds, policies, capacities and planning runs; a scope with a field left
    out is a scope that has not said which one it means, and the service layer refuses to guess.
    """

    policy_id: str | None = None
    model_name: str | None = None
    fold_set: str | None = None
    fold_id: str | None = None
    k_name: str | None = None
    schedule_config_id: str | None = None
    planning_run_id: str | None = None
    replan_index: int | None = None


class RunInfo(BaseModel):
    """Which artifact file a response was read from, so a caller can tell two runs apart."""

    path: str
    manifest_path: str | None = None
    built_at: str | None = None


class PageMeta(BaseModel):
    offset: int
    limit: int
    total: int


class Page[T](BaseModel):
    """The envelope every list endpoint returns: rows, pagination, and where they came from."""

    data: list[T]
    page: PageMeta
    run: RunInfo


class ErrorResponse(BaseModel):
    error: str
    detail: str


class StagedRequestReceipt(BaseModel):
    """What a write endpoint returns. Never a recomputed artifact -- the request is pending
    until an operator drains the staging store through the batch CLI. See ADR 0049."""

    request_id: str
    kind: str
    natural_id: str
    status: str = "pending"
    staged_at: str


class StagedRequestStatus(BaseModel):
    """One staged request, reconciled against the latest committed log for its layer."""

    request_id: str
    kind: str
    natural_id: str
    status: str = Field(description='"pending" or "applied_in_run:<timestamp>"')
    staged_at: str
    payload: dict[str, object]


__all__ = [
    "DecisionScope",
    "ErrorResponse",
    "Page",
    "PageMeta",
    "RunInfo",
    "StagedRequestReceipt",
    "StagedRequestStatus",
]
