from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status

from sentinel.api.deps import (
    get_decision_scope,
    get_settings,
    get_staging_service,
    make_page_params_dependency,
)
from sentinel.api.schemas.common import DecisionScope, Page, StagedRequestReceipt
from sentinel.api.schemas.policy import OverrideIn, OverrideLogRowOut
from sentinel.api.services import policy_service
from sentinel.api.services.pagination import PageParams
from sentinel.api.services.staging_service import StagingService
from sentinel.config import Settings

router = APIRouter(prefix="/v1/policy/overrides", tags=["overrides"])

_page_params = make_page_params_dependency(default_sort="override_id")


@router.get("", response_model=Page[OverrideLogRowOut])
def list_overrides(
    target_inspection_id: str | None = Query(None),
    scope: DecisionScope = Depends(get_decision_scope),
    page: PageParams = Depends(_page_params),
    settings: Settings = Depends(get_settings),
) -> Page[OverrideLogRowOut]:
    return policy_service.get_override_log(
        settings, scope, page, target_inspection_id=target_inspection_id
    )


@router.post("", response_model=StagedRequestReceipt, status_code=status.HTTP_201_CREATED)
def submit_override(
    payload: OverrideIn,
    settings: Settings = Depends(get_settings),
    staging: StagingService = Depends(get_staging_service),
) -> StagedRequestReceipt:
    return policy_service.stage_override(settings, payload, staging)


__all__ = ["router"]
