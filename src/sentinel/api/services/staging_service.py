"""Append-only staging for the three human-input contracts. Never applies anything.

Writes are staged, never applied (ADR 0049): a validated override/adjustment/execution-event
request is appended as one JSON line to a file this module owns, in exactly the list-of-objects
shape the batch CLI's ``--overrides``/``--adjustments``/``--execution`` flags already read. This
module never calls ``sentinel.policy.select``, ``sentinel.scheduling.build.run_schedule``,
``replan``, ``apply_adjustments`` or ``record_execution`` -- turning a staged request into a new
artifact stays a human operator's job, run through the existing CLI.

The file is append-only in the literal sense: a line is added, never rewritten in place, so
the staging store itself carries the same "history is not edited" property as every other layer
in this project.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sentinel.api.errors import DuplicateKey
from sentinel.api.schemas.common import StagedRequestReceipt, StagedRequestStatus

#: The natural-id field and sub-path per staged kind. One entry per human-input contract;
#: adding a fourth kind means adding a fourth line here, never a generic "type" column.
_KIND_CONFIG: dict[str, tuple[str, str]] = {
    "override": ("override_id", "policy/overrides_pending.jsonl"),
    "adjustment": ("adjustment_id", "scheduling/adjustments_pending.jsonl"),
    "execution_event": ("execution_id", "scheduling/execution_events_pending.jsonl"),
    "review_resolution": ("review_id", "review/resolutions_pending.jsonl"),
    "plan_decision": ("decision_id", "plan_review/decisions_pending.jsonl"),
    "plan_approval": ("approval_id", "plan_review/approvals_pending.jsonl"),
}


class StagingService:
    def __init__(self, staging_dir: Path) -> None:
        self._staging_dir = staging_dir

    def _path_for(self, kind: str) -> Path:
        _, relative = _KIND_CONFIG[kind]
        return self._staging_dir / relative

    def _read_all(self, kind: str) -> list[dict[str, Any]]:
        path = self._path_for(kind)
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def append(
        self,
        *,
        kind: str,
        natural_id: str,
        record: dict[str, object],
        committed_ids: set[str],
    ) -> StagedRequestReceipt:
        """Stage one request, or refuse a colliding id.

        Idempotent on an exact re-post: the same natural id with byte-identical payload returns
        the original receipt rather than erroring, so a caller's retry after a dropped response
        is safe. A colliding id with a *different* payload is refused (409) -- silently
        overwriting a pending human decision is exactly the failure an audit trail exists to
        prevent.
        """
        if natural_id in committed_ids:
            raise DuplicateKey(
                f"{natural_id!r} already exists in the committed log for this layer. Staging "
                "it again would be indistinguishable from re-deciding something already decided."
            )
        existing = self._read_all(kind)
        for entry in existing:
            if entry["record"].get(_KIND_CONFIG[kind][0]) == natural_id:
                if entry["record"] == record:
                    return StagedRequestReceipt(
                        request_id=str(entry["request_id"]),
                        kind=kind,
                        natural_id=natural_id,
                        status="pending",
                        staged_at=str(entry["staged_at"]),
                    )
                raise DuplicateKey(
                    f"{natural_id!r} is already staged with a different payload. Submit a new "
                    "id, or resolve the conflict before re-submitting."
                )

        request_id = str(uuid4())
        staged_at = datetime.now(UTC).isoformat()
        entry = {"request_id": request_id, "kind": kind, "staged_at": staged_at, "record": record}
        path = self._path_for(kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return StagedRequestReceipt(
            request_id=request_id,
            kind=kind,
            natural_id=natural_id,
            status="pending",
            staged_at=staged_at,
        )

    def list_pending(self, kind: str | None = None) -> list[StagedRequestStatus]:
        kinds = [kind] if kind is not None else list(_KIND_CONFIG)
        out: list[StagedRequestStatus] = []
        for one_kind in kinds:
            id_field, _ = _KIND_CONFIG[one_kind]
            for entry in self._read_all(one_kind):
                out.append(
                    StagedRequestStatus(
                        request_id=str(entry["request_id"]),
                        kind=one_kind,
                        natural_id=str(entry["record"].get(id_field, "")),
                        status="pending",
                        staged_at=str(entry["staged_at"]),
                        payload=entry["record"],
                    )
                )
        return out

    def reconcile(self, kind: str, committed_ids: set[str]) -> list[StagedRequestStatus]:
        """Pending requests, relabelled ``applied_in_run`` where the id is now committed."""
        id_field, _ = _KIND_CONFIG[kind]
        out: list[StagedRequestStatus] = []
        for entry in self._read_all(kind):
            natural_id = str(entry["record"].get(id_field, ""))
            status = "applied" if natural_id in committed_ids else "pending"
            out.append(
                StagedRequestStatus(
                    request_id=str(entry["request_id"]),
                    kind=kind,
                    natural_id=natural_id,
                    status=status,
                    staged_at=str(entry["staged_at"]),
                    payload=entry["record"],
                )
            )
        return out


__all__ = ["StagingService"]
