from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status

from sentinel.api.deps import (
    get_decision_scope,
    get_settings,
    get_staging_service,
    make_page_params_dependency,
)
from sentinel.api.schemas.common import DecisionScope, Page, StagedRequestReceipt
from sentinel.api.schemas.scheduling import (
    ExecutionEventIn,
    ExecutionLogRowOut,
    ExecutionSummaryOut,
)
from sentinel.api.services import scheduling_service
from sentinel.api.services.pagination import PageParams
from sentinel.api.services.staging_service import StagingService
from sentinel.config import Settings

router = APIRouter(prefix="/v1/execution", tags=["execution"])

_page_params = make_page_params_dependency(default_sort="execution_id")


@router.get("/events", response_model=Page[ExecutionLogRowOut])
def list_execution_events(
    target_inspection_id: str | None = Query(None),
    scope: DecisionScope = Depends(get_decision_scope),
    page: PageParams = Depends(_page_params),
    settings: Settings = Depends(get_settings),
) -> Page[ExecutionLogRowOut]:
    return scheduling_service.get_execution_events(
        settings, scope, page, target_inspection_id=target_inspection_id
    )


@router.post("/events", response_model=StagedRequestReceipt, status_code=status.HTTP_201_CREATED)
def submit_execution_event(
    payload: ExecutionEventIn,
    settings: Settings = Depends(get_settings),
    staging: StagingService = Depends(get_staging_service),
) -> StagedRequestReceipt:
    return scheduling_service.stage_execution_event(settings, payload, staging)


@router.get("/summary", response_model=ExecutionSummaryOut)
def get_execution_summary(
    scope: DecisionScope = Depends(get_decision_scope), settings: Settings = Depends(get_settings)
) -> ExecutionSummaryOut:
    return scheduling_service.get_execution_summary(settings, scope)


@router.get("/contract", response_model=list[dict[str, object]])
def get_execution_contract(settings: Settings = Depends(get_settings)) -> list[dict[str, object]]:
    return scheduling_service.get_execution_contract(settings)


__all__ = ["router"]
