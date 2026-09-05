from __future__ import annotations

from fastapi import APIRouter, Depends

from sentinel.api.deps import get_decision_scope, get_settings
from sentinel.api.schemas.common import DecisionScope
from sentinel.api.schemas.establishment import EstablishmentHistoryOut
from sentinel.api.services import establishment_service
from sentinel.config import Settings

router = APIRouter(prefix="/v1/establishments", tags=["establishments"])


@router.get("/{establishment_id}", response_model=EstablishmentHistoryOut)
def get_establishment(
    establishment_id: str,
    scope: DecisionScope = Depends(get_decision_scope),
    settings: Settings = Depends(get_settings),
) -> EstablishmentHistoryOut:
    return establishment_service.get_establishment_history(settings, establishment_id, scope)


__all__ = ["router"]
