"""Decode the human-typed decisions file. JSON, because a person edits it."""

from __future__ import annotations

import json
from pathlib import Path


class PlanReviewInputError(ValueError):
    """Raised when the decisions file cannot be decoded at all."""


def read_decisions_file(path: Path | None) -> list[dict[str, object]]:
    """Decode a human decisions file, or return nothing if none was given.

    The whole file is refused if it is not a JSON list of objects, matching
    ``review.inputs.read_resolutions_file``. Per-row validation happens in
    ``resolution.parse_decisions``, not here.
    """
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Decisions file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanReviewInputError(f"{path.name}: not valid JSON -- {exc}") from exc
    if not isinstance(payload, list):
        raise PlanReviewInputError(f"{path.name}: expected a JSON list of decision objects")
    for i, row in enumerate(payload):
        if not isinstance(row, dict):
            raise PlanReviewInputError(f"{path.name}: row {i} is not a JSON object")
    return payload


__all__ = ["PlanReviewInputError", "read_decisions_file"]
