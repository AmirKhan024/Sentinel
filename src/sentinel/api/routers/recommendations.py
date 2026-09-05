from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from sentinel.api.deps import get_decision_scope, get_settings, make_page_params_dependency
from sentinel.api.schemas.common import DecisionScope, Page
from sentinel.api.schemas.policy import AllocationOut, RecommendationOut
from sentinel.api.services import policy_service
from sentinel.api.services.pagination import PageParams
from sentinel.config import Settings

router = APIRouter(tags=["recommendations"])

_page_params = make_page_params_dependency(default_sort="final_policy_rank")


@router.get("/v1/recommendations", response_model=Page[RecommendationOut])
def list_recommendations(
    establishment_id: str | None = Query(None),
    is_selected: bool | None = Query(None),
    scope: DecisionScope = Depends(get_decision_scope),
    page: PageParams = Depends(_page_params),
    settings: Settings = Depends(get_settings),
) -> Page[RecommendationOut]:
    return policy_service.get_recommendations(
        settings, scope, page, establishment_id=establishment_id, is_selected=is_selected
    )


@router.get("/v1/recommendations/{target_inspection_id}", response_model=RecommendationOut)
def get_recommendation(
    target_inspection_id: str,
    scope: DecisionScope = Depends(get_decision_scope),
    settings: Settings = Depends(get_settings),
) -> RecommendationOut:
    return policy_service.get_recommendation(settings, target_inspection_id, scope)


@router.get("/v1/policy/selection-allocation", response_model=list[AllocationOut])
def get_selection_allocation(
    scope: DecisionScope = Depends(get_decision_scope),
    settings: Settings = Depends(get_settings),
) -> list[AllocationOut]:
    return policy_service.get_selection_allocation(settings, scope)


__all__ = ["router"]
