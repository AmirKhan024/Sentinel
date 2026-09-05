from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Query

from sentinel.api.deps import get_decision_scope, get_settings, make_page_params_dependency
from sentinel.api.schemas.common import DecisionScope, Page
from sentinel.api.schemas.scheduling import BacklogRowOut, ReplanningRunOut, ScheduleRowOut
from sentinel.api.services import scheduling_service
from sentinel.api.services.pagination import PageParams
from sentinel.config import Settings

router = APIRouter(prefix="/v1/schedule", tags=["schedule"])

_schedule_page = make_page_params_dependency(default_sort="final_policy_rank")
_backlog_page = make_page_params_dependency(default_sort="backlog_position")


@router.get("", response_model=Page[ScheduleRowOut])
def list_schedule(
    establishment_id: str | None = Query(None),
    schedule_status: str | None = Query(None),
    scheduled_date: date | None = Query(None),
    scope: DecisionScope = Depends(get_decision_scope),
    page: PageParams = Depends(_schedule_page),
    settings: Settings = Depends(get_settings),
) -> Page[ScheduleRowOut]:
    return scheduling_service.get_schedule(
        settings,
        scope,
        page,
        establishment_id=establishment_id,
        schedule_status=schedule_status,
        scheduled_date=scheduled_date,
    )


@router.get("/dates", response_model=list[dict[str, object]])
def list_schedule_dates(
    scope: DecisionScope = Depends(get_decision_scope), settings: Settings = Depends(get_settings)
) -> list[dict[str, object]]:
    return scheduling_service.get_schedule_dates(settings, scope)


@router.get("/backlog", response_model=Page[BacklogRowOut])
def list_backlog(
    scope: DecisionScope = Depends(get_decision_scope),
    page: PageParams = Depends(_backlog_page),
    settings: Settings = Depends(get_settings),
) -> Page[BacklogRowOut]:
    return scheduling_service.get_backlog(settings, scope, page)


@router.get("/summary", response_model=list[dict[str, object]])
def get_summary(
    scope: DecisionScope = Depends(get_decision_scope), settings: Settings = Depends(get_settings)
) -> list[dict[str, object]]:
    return scheduling_service.get_small_table(settings, scope, table="schedule_summary")


@router.get("/capacity-utilization", response_model=list[dict[str, object]])
def get_capacity_utilization(
    scope: DecisionScope = Depends(get_decision_scope), settings: Settings = Depends(get_settings)
) -> list[dict[str, object]]:
    return scheduling_service.get_small_table(settings, scope, table="capacity_utilization")


@router.get("/priority-preservation", response_model=list[dict[str, object]])
def get_priority_preservation(
    scope: DecisionScope = Depends(get_decision_scope), settings: Settings = Depends(get_settings)
) -> list[dict[str, object]]:
    return scheduling_service.get_small_table(settings, scope, table="priority_preservation")


@router.get("/replanning-runs", response_model=list[ReplanningRunOut])
def get_replanning_runs(
    scope: DecisionScope = Depends(get_decision_scope), settings: Settings = Depends(get_settings)
) -> list[ReplanningRunOut]:
    return scheduling_service.get_replanning_runs(settings, scope)


__all__ = ["router"]
