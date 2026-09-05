"""Loading the authoritative artifacts. The one module here that touches Parquet on the way in.

This component fits nothing and re-derives nothing it can read. It reads Component 13's and
Component 14's own artifacts and produces nothing but typed structures, so every later module can
be pure.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from sentinel.review.models import ReviewResolution
from sentinel.review.resolution import parse_resolutions


class ReviewInputError(ValueError):
    """Raised when an input artifact cannot be trusted enough to build a review queue from."""


def read_recommendations(path: Path) -> pl.DataFrame:
    """Component 13's recommendation universe."""
    if not path.exists():
        raise FileNotFoundError(f"Recommendation table not found: {path}")
    frame = pl.read_parquet(path)
    required = ("target_inspection_id", "establishment_id", "is_selected", "warnings")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ReviewInputError(f"{path.name}: recommendation table is missing {', '.join(missing)}")
    return frame


def read_schedule(path: Path | None) -> pl.DataFrame | None:
    """Component 14's schedule, or None if it was not supplied.

    Optional by design: without it, only the warning trigger runs and the execution-gap trigger
    is skipped, recorded as an advisory rather than an error.
    """
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Schedule table not found: {path}")
    frame = pl.read_parquet(path)
    required = ("target_inspection_id", "schedule_status", "replan_index", "schedule_config_id")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ReviewInputError(f"{path.name}: schedule table is missing {', '.join(missing)}")
    return frame


def read_execution_log(path: Path | None) -> pl.DataFrame | None:
    """Component 14's accumulated execution log, or None if it was not supplied."""
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Execution log not found: {path}")
    frame = pl.read_parquet(path)
    required = ("schedule_config_id", "policy_id", "fold_id", "k_name", "target_inspection_id")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ReviewInputError(f"{path.name}: execution log is missing {', '.join(missing)}")
    return frame


def read_resolutions_file(path: Path | None) -> list[ReviewResolution]:
    """Decode and validate a human resolution file, or return nothing.

    JSON rather than Parquet because a person edits it. The whole file is refused if any row is
    malformed, matching Component 13's and Component 14's own external contracts.
    """
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Resolution file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReviewInputError(f"{path.name}: not valid JSON -- {exc}") from exc
    if not isinstance(payload, list):
        raise ReviewInputError(
            f"{path.name}: the resolution contract is a JSON list of resolution objects, got "
            f"{type(payload).__name__}"
        )
    return parse_resolutions(payload)


__all__ = [
    "ReviewInputError",
    "read_execution_log",
    "read_recommendations",
    "read_resolutions_file",
    "read_schedule",
]
