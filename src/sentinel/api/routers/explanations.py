from __future__ import annotations

from fastapi import APIRouter, Depends

from sentinel.api.deps import get_decision_scope, get_settings
from sentinel.api.schemas.common import DecisionScope
from sentinel.api.schemas.explain import ExplanationCaseOut, SupportOut
from sentinel.api.services import explain_service
from sentinel.config import Settings

router = APIRouter(prefix="/v1/explanations", tags=["explanations"])


@router.get("/support", response_model=list[SupportOut])
def get_support(settings: Settings = Depends(get_settings)) -> list[SupportOut]:
    return explain_service.get_support(settings)


@router.get("/{target_inspection_id}", response_model=ExplanationCaseOut)
def get_explanation(
    target_inspection_id: str,
    scope: DecisionScope = Depends(get_decision_scope),
    settings: Settings = Depends(get_settings),
) -> ExplanationCaseOut:
    return explain_service.get_explanation(settings, target_inspection_id, scope)


__all__ = ["router"]
