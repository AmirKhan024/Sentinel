"""FastAPI dependency wiring: settings, pagination, decision scope, staging.

``Settings`` is resolved once per app (via ``create_app``'s closure) and overridden per-test with
FastAPI's ``dependency_overrides`` -- no new configuration abstraction, per the plan's "reuse
``sentinel.config.Settings`` directly" rule.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, Query

from sentinel.api.schemas.common import DecisionScope
from sentinel.api.services.pagination import PageParams
from sentinel.api.services.staging_service import StagingService
from sentinel.config import Settings, load_settings


def get_settings() -> Settings:
    """Overridden in tests via ``app.dependency_overrides[get_settings]``."""
    return load_settings()


def get_staging_service(settings: Settings = Depends(get_settings)) -> StagingService:
    return StagingService(settings.staging_dir)


def get_decision_scope(
    policy_id: str | None = Query(None),
    model_name: str | None = Query(None),
    fold_set: str | None = Query(None),
    fold_id: str | None = Query(None),
    k_name: str | None = Query(None),
    schedule_config_id: str | None = Query(None),
    planning_run_id: str | None = Query(None),
    replan_index: int | None = Query(None),
) -> DecisionScope:
    return DecisionScope(
        policy_id=policy_id,
        model_name=model_name,
        fold_set=fold_set,
        fold_id=fold_id,
        k_name=k_name,
        schedule_config_id=schedule_config_id,
        planning_run_id=planning_run_id,
        replan_index=replan_index,
    )


def make_page_params_dependency(*, default_sort: str | None = None) -> Callable[..., PageParams]:
    """Build a pagination dependency bounded by ``Settings.api_max_page_size``.

    A factory rather than one shared function because each list endpoint's default sort column
    is its own -- ``final_policy_rank`` for recommendations, ``backlog_position`` for the
    backlog -- and the sort whitelist is a per-endpoint contract, never an arbitrary column name.
    """

    def _dependency(
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1),
        descending: bool = Query(False),
        settings: Settings = Depends(get_settings),
    ) -> PageParams:
        bounded_limit = min(limit, settings.api_max_page_size)
        return PageParams(
            offset=offset, limit=bounded_limit, sort_column=default_sort, descending=descending
        )

    return _dependency


__all__ = [
    "get_decision_scope",
    "get_settings",
    "get_staging_service",
    "make_page_params_dependency",
]
