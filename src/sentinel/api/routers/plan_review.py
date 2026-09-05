from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status

from sentinel.api.deps import get_settings, get_staging_service, make_page_params_dependency
from sentinel.api.schemas.common import Page, StagedRequestReceipt
from sentinel.api.schemas.plan_review import (
    PlanApprovalIn,
    PlanApprovalOut,
    PlanDecisionIn,
    PlanDecisionLogRowOut,
    PlanRowOut,
    PlanSummaryOut,
    WorkBlockOut,
)
from sentinel.api.services import plan_review_service
from sentinel.api.services.pagination import PageParams
from sentinel.api.services.staging_service import StagingService
from sentinel.config import Settings

router = APIRouter(prefix="/v1/plan-review", tags=["plan-review"])

_row_page_params = make_page_params_dependency(default_sort="suggested_order_in_block")
_decision_page_params = make_page_params_dependency(default_sort="decision_id")


@router.get("/summary", response_model=PlanSummaryOut)
def get_plan_summary(
    planning_date: str | None = Query(None),
    settings: Settings = Depends(get_settings),
) -> PlanSummaryOut:
    return plan_review_service.get_plan_summary(settings, planning_date)


@router.get("/rows", response_model=Page[PlanRowOut])
def list_plan_rows(
    planning_date: str | None = Query(None),
    page: PageParams = Depends(_row_page_params),
    settings: Settings = Depends(get_settings),
) -> Page[PlanRowOut]:
    return plan_review_service.list_plan_rows(settings, planning_date, page)


@router.get("/rows/{target_inspection_id}", response_model=PlanRowOut)
def get_plan_row(
    target_inspection_id: str,
    planning_date: str | None = Query(None),
    settings: Settings = Depends(get_settings),
) -> PlanRowOut:
    return plan_review_service.get_plan_row(settings, target_inspection_id, planning_date)


@router.get("/work-blocks", response_model=list[WorkBlockOut])
def list_work_blocks(
    planning_date: str | None = Query(None),
    settings: Settings = Depends(get_settings),
) -> list[WorkBlockOut]:
    return plan_review_service.list_work_blocks(settings, planning_date)


@router.get("/decisions", response_model=Page[PlanDecisionLogRowOut])
def list_decisions(
    planning_date: str | None = Query(None),
    page: PageParams = Depends(_decision_page_params),
    settings: Settings = Depends(get_settings),
) -> Page[PlanDecisionLogRowOut]:
    return plan_review_service.get_plan_decision_log(settings, planning_date, page)


@router.post("/decisions", response_model=StagedRequestReceipt, status_code=status.HTTP_201_CREATED)
def submit_decision(
    payload: PlanDecisionIn,
    settings: Settings = Depends(get_settings),
    staging: StagingService = Depends(get_staging_service),
) -> StagedRequestReceipt:
    return plan_review_service.stage_plan_decision(settings, payload, staging)


@router.get("/approval", response_model=PlanApprovalOut)
def get_plan_approval(
    planning_date: str | None = Query(None),
    settings: Settings = Depends(get_settings),
) -> PlanApprovalOut:
    return plan_review_service.get_plan_approval(settings, planning_date)


@router.post("/approve", response_model=StagedRequestReceipt, status_code=status.HTTP_201_CREATED)
def submit_approval(
    payload: PlanApprovalIn,
    settings: Settings = Depends(get_settings),
    staging: StagingService = Depends(get_staging_service),
) -> StagedRequestReceipt:
    """Stage a plan approval. Never applied here -- `sentinel approve-plan` commits it,
    running the full readiness checklist first (ADR 0049)."""
    return plan_review_service.stage_plan_approval(settings, payload, staging)


__all__ = ["router"]
