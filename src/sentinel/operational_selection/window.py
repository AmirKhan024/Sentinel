"""Building a `PolicyWindow` from Component 18's scored output. No allocation logic here.

`PolicyWindow` is Component 13's, imported unchanged. Every field either carries real
operational data (`ids`, `scores`, `base_scores`, `eligible`, `secondary_no_history`,
`dates`) or an explicit, documented sentinel for the two fields `allocate()`/`decide()`
never read (`labels`, `median_daily_capacity`) -- verified directly against
`policy/allocation.py` rather than assumed.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from sentinel.operational_selection.definitions import (
    NOT_APPLICABLE_MEDIAN_DAILY_CAPACITY,
    UNKNOWN_LABEL,
)
from sentinel.policy.models import PolicyWindow

REQUIRED_COLUMNS = (
    "target_inspection_id",
    "scoring_status",
    "base_score",
    "calibrated_score",
    "coverage_eligible",
    "secondary_no_history",
    "planning_date",
)


class SelectionWindowError(ValueError):
    """Raised when a `PolicyWindow` cannot be built from the priority frame offered."""


def build_selection_window(
    priority_frame: pl.DataFrame, *, fold_set: str, fold_id: str
) -> PolicyWindow:
    """The scored (never excluded) rows of a priority set, as a `PolicyWindow`.

    ``fold_set``/``fold_id`` are carried through only as labels on the window -- they
    are never real evaluation-fold identifiers here, and callers should pass Component
    18's own ``operational_fold_set``/``operational_fold_id`` so a reader who opens the
    resulting ``Allocation`` sees the same operational identity Component 18 already
    established.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in priority_frame.columns]
    if missing:
        raise SelectionWindowError(
            f"priority frame is missing required column(s): {', '.join(missing)}"
        )

    scored = priority_frame.filter(pl.col("scoring_status") == "scored")
    if scored.is_empty():
        raise SelectionWindowError(
            "no scored candidates in the priority set -- there is nothing to allocate"
        )

    dates: tuple[date, ...] = tuple(
        date.fromisoformat(v) for v in scored["planning_date"].to_list()
    )

    return PolicyWindow(
        fold_set=fold_set,
        fold_id=fold_id,
        ids=tuple(scored["target_inspection_id"].to_list()),
        scores=tuple(float(v) for v in scored["calibrated_score"].to_list()),
        base_scores=tuple(float(v) for v in scored["base_score"].to_list()),
        labels=tuple(UNKNOWN_LABEL for _ in range(scored.height)),
        dates=dates,
        eligible=tuple(bool(v) for v in scored["coverage_eligible"].to_list()),
        secondary_no_history=tuple(bool(v) for v in scored["secondary_no_history"].to_list()),
        median_daily_capacity=NOT_APPLICABLE_MEDIAN_DAILY_CAPACITY,
    )


__all__ = ["REQUIRED_COLUMNS", "SelectionWindowError", "build_selection_window"]
