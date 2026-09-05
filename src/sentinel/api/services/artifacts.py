"""The shared spine every read service uses: find the latest run, read a table, check scope.

Genuinely new code -- no reader like this exists elsewhere in the repository, because every
existing loader (``policy/inputs.py``, ``scheduling/inputs.py``) takes an explicit path for one
pipeline run's internal use. This module is what turns "the artifacts on disk" into "a table a
product endpoint can page through," and it is the only place that touches a filesystem path
directly; every service function above it works with a resolved ``Path`` or an already-read
``pl.DataFrame``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from sentinel.api.errors import AmbiguousScope, ArtifactNotFound
from sentinel.api.schemas.common import DecisionScope, RunInfo
from sentinel.manifest import manifest_path_for
from sentinel.query.duckdb_queries import latest_parquet


def resolve_latest(directory: Path, *, prefix: str) -> Path:
    """The most recent timestamped Parquet file for one table, or a 404-shaped error.

    Thin wrapper over ``sentinel.query.duckdb_queries.latest_parquet``: same lexicographic-max
    logic (filenames embed a sortable UTC timestamp), but raised as ``ArtifactNotFound`` rather
    than ``FileNotFoundError`` so a router never has to translate a filesystem exception itself.
    """
    try:
        return latest_parquet(directory, prefix=f"{prefix}_")
    except FileNotFoundError as exc:
        raise ArtifactNotFound(
            f"No {prefix} artifact found. The upstream component has not been run yet.",
            component=prefix,
        ) from exc


def run_info(path: Path) -> RunInfo:
    """Provenance for a response: which file, which manifest, when it was built."""
    manifest_path = manifest_path_for(path)
    built_at: str | None = None
    if manifest_path.exists():
        import json

        try:
            built_at = json.loads(manifest_path.read_text(encoding="utf-8")).get("built_at")
        except (OSError, ValueError):
            built_at = None
    return RunInfo(
        path=str(path),
        manifest_path=str(manifest_path) if manifest_path.exists() else None,
        built_at=built_at,
    )


def read_table(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise ArtifactNotFound(f"Artifact file not found: {path.name}")
    return pl.read_parquet(path)


def require_scope(scope: DecisionScope, *, required: Sequence[str]) -> None:
    """Refuse an ambiguous request rather than defaulting to "latest" or "first".

    ADR 0050: an establishment, a policy or a schedule cell is only unambiguous once every
    field in ``required`` is set. A missing field is reported by name so a caller knows exactly
    what to add, rather than receiving an arbitrarily chosen row.
    """
    missing = [field for field in required if getattr(scope, field) is None]
    if missing:
        raise AmbiguousScope(
            f"Decision scope is missing required field(s): {', '.join(missing)}. "
            "This endpoint will not guess which run, fold, policy or capacity you mean.",
            missing_scope_fields=missing,
        )


def apply_scope_filter(frame: pl.DataFrame, scope: DecisionScope) -> pl.DataFrame:
    """Filter a frame to the caller's scope, on whichever scope columns the frame carries."""
    result = frame
    for field in (
        "policy_id",
        "model_name",
        "fold_set",
        "fold_id",
        "k_name",
        "schedule_config_id",
        "planning_run_id",
        "replan_index",
    ):
        value = getattr(scope, field)
        if value is not None and field in result.columns:
            result = result.filter(pl.col(field) == value)
    return result


def candidate_values(frame: pl.DataFrame, column: str) -> list[object]:
    """Distinct values a caller could add to disambiguate a scope, for an error body."""
    if column not in frame.columns:
        return []
    return sorted(frame.get_column(column).unique().drop_nulls().to_list())


__all__ = [
    "apply_scope_filter",
    "candidate_values",
    "read_table",
    "require_scope",
    "resolve_latest",
    "run_info",
]
