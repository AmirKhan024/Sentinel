from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status

from sentinel.api.deps import (
    get_decision_scope,
    get_settings,
    get_staging_service,
    make_page_params_dependency,
)
from sentinel.api.schemas.common import DecisionScope, Page, StagedRequestReceipt
from sentinel.api.schemas.review import ResolutionIn, ResolutionLogRowOut, ReviewCaseOut
from sentinel.api.services import review_service
from sentinel.api.services.pagination import PageParams
from sentinel.api.services.staging_service import StagingService
from sentinel.config import Settings

router = APIRouter(prefix="/v1/review", tags=["review"])

_queue_page_params = make_page_params_dependency(default_sort="target_inspection_id")
_resolution_page_params = make_page_params_dependency(default_sort="review_id")


@router.get("/queue", response_model=Page[ReviewCaseOut])
def list_review_queue(
    trigger: str | None = Query(None),
    scope: DecisionScope = Depends(get_decision_scope),
    page: PageParams = Depends(_queue_page_params),
    settings: Settings = Depends(get_settings),
) -> Page[ReviewCaseOut]:
    return review_service.get_review_queue(settings, scope, page, trigger=trigger)


@router.get("/queue/{target_inspection_id}", response_model=ReviewCaseOut)
def get_review_case(
    target_inspection_id: str,
    scope: DecisionScope = Depends(get_decision_scope),
    settings: Settings = Depends(get_settings),
) -> ReviewCaseOut:
    return review_service.get_review_case(settings, target_inspection_id, scope)


@router.get("/resolutions", response_model=Page[ResolutionLogRowOut])
def list_resolutions(
    target_inspection_id: str | None = Query(None),
    scope: DecisionScope = Depends(get_decision_scope),
    page: PageParams = Depends(_resolution_page_params),
    settings: Settings = Depends(get_settings),
) -> Page[ResolutionLogRowOut]:
    return review_service.get_resolution_log(
        settings, scope, page, target_inspection_id=target_inspection_id
    )


@router.post(
    "/resolutions", response_model=StagedRequestReceipt, status_code=status.HTTP_201_CREATED
)
def submit_resolution(
    payload: ResolutionIn,
    settings: Settings = Depends(get_settings),
    staging: StagingService = Depends(get_staging_service),
) -> StagedRequestReceipt:
    return review_service.stage_resolution(settings, payload, staging)


__all__ = ["router"]
