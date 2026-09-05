"""Reconciling staged requests against the committed logs each layer already writes."""

from __future__ import annotations

from sentinel.api.errors import ArtifactNotFound
from sentinel.api.schemas.common import StagedRequestStatus
from sentinel.api.services.artifacts import read_table, resolve_latest
from sentinel.api.services.staging_service import StagingService
from sentinel.config import Settings

#: (committed-table directory attribute, table prefix, id column), per staged kind.
_COMMITTED: dict[str, tuple[str, str, str]] = {
    "override": ("policy_processed_dir", "policy_override_log", "override_id"),
    "adjustment": ("scheduling_processed_dir", "schedule_adjustment_log", "adjustment_id"),
    "execution_event": ("scheduling_processed_dir", "execution_log", "execution_id"),
    "review_resolution": ("review_processed_dir", "review_resolution_log", "review_id"),
}


def _committed_ids(settings: Settings, kind: str) -> set[str]:
    dir_attr, prefix, id_column = _COMMITTED[kind]
    directory = getattr(settings, dir_attr)
    try:
        path = resolve_latest(directory, prefix=prefix)
    except ArtifactNotFound:
        return set()
    return set(read_table(path).get_column(id_column).to_list())


def list_staged_requests(
    settings: Settings,
    staging: StagingService,
    *,
    kind: str | None = None,
    status: str | None = None,
) -> list[StagedRequestStatus]:
    kinds = [kind] if kind is not None else list(_COMMITTED)
    out: list[StagedRequestStatus] = []
    for one_kind in kinds:
        out.extend(staging.reconcile(one_kind, _committed_ids(settings, one_kind)))
    if status is not None:
        out = [entry for entry in out if entry.status == status]
    return out


__all__ = ["list_staged_requests"]
