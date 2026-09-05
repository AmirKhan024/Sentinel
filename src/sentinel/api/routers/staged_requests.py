from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from sentinel.api.deps import get_settings, get_staging_service
from sentinel.api.schemas.common import StagedRequestStatus
from sentinel.api.services.staged_requests_service import list_staged_requests
from sentinel.api.services.staging_service import StagingService
from sentinel.config import Settings

router = APIRouter(prefix="/v1/staged-requests", tags=["staged-requests"])


@router.get("", response_model=list[StagedRequestStatus])
def get_staged_requests(
    kind: str | None = Query(None),
    status: str | None = Query(None),
    settings: Settings = Depends(get_settings),
    staging: StagingService = Depends(get_staging_service),
) -> list[StagedRequestStatus]:
    return list_staged_requests(settings, staging, kind=kind, status=status)


__all__ = ["router"]
