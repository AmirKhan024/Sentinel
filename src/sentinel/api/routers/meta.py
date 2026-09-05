from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from sentinel.api.deps import get_settings
from sentinel.api.services import meta_service
from sentinel.config import Settings

router = APIRouter(tags=["meta"])


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness only. No artifact is read, so this never fails because of upstream data."""
    return {"status": "ok"}


@router.get("/v1/manifests/{component}", response_model=dict[str, object])
def get_manifest(component: str, settings: Settings = Depends(get_settings)) -> dict[str, object]:
    return meta_service.get_manifest(settings, component)


@router.get("/v1/runs", response_model=list[dict[str, object]])
def list_runs(
    component: str | None = Query(None), settings: Settings = Depends(get_settings)
) -> list[dict[str, object]]:
    return meta_service.list_runs(settings, component)


__all__ = ["router"]
