"""The two deterministic triggers. Pure -- no filesystem, no clock, no threshold.

Each trigger is a boolean predicate over a column an upstream component already wrote. Neither
reads a score, a probability or a rank: ``warning_triggered_rows`` reads Component 13's
``warnings`` column, and ``execution_gap_rows`` is an anti-join against Component 14's own
execution log. Nothing here compares a number to a cutoff.
"""

from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl

from sentinel.review.definitions import (
    NO_TRIGGER,
    TRIGGER_SEPARATOR,
    ReviewTriggerReason,
)
from sentinel.review.models import ReviewCase
from sentinel.scheduling.definitions import OCCUPYING_STATUSES


def warning_triggered_rows(recommendations: pl.DataFrame) -> pl.DataFrame:
    """Selected rows carrying at least one Component 13 policy warning.

    Restricted to ``is_selected``: a warning on a row nobody was going to inspect was never an
    operational decision, and flagging it would blur "not selected" with "needs review".
    """
    if recommendations.is_empty():
        return recommendations.head(0)
    return recommendations.filter(pl.col("is_selected") & (pl.col("warnings") != "none"))


#: The columns that identify one execution-log row. ``model_name`` and ``fold_set`` are not
#: part of Component 14's execution contract (``EXECUTION_REQUIRED_FIELDS``), so the anti-join
#: must key on exactly these five -- not the full schedule cell key -- or a target inspection id
#: that recurs under a different capacity mode or model would be matched to the wrong report.
EXECUTION_JOIN_KEYS: tuple[str, ...] = (
    "schedule_config_id",
    "policy_id",
    "fold_id",
    "k_name",
    "target_inspection_id",
)


def execution_gap_rows(schedule: pl.DataFrame, execution_log: pl.DataFrame) -> pl.DataFrame:
    """Occupying schedule rows, at each cell's latest replan index, with no execution report.

    An anti-join against ``execution_log`` on the execution contract's own key -- the row-level
    version of the cell-grain count Component 14 already computes as ``NO_EXECUTION_RECORD``. No
    date arithmetic and no notion of "today" is introduced: the predicate is exactly "does an
    execution event exist for this row in the log as currently accumulated".
    """
    if schedule.is_empty():
        return schedule.head(0)
    cell_keys = ["schedule_config_id", "policy_id", "model_name", "fold_set", "fold_id", "k_name"]
    latest = schedule.group_by(cell_keys).agg(
        pl.col("replan_index").max().alias("_latest_replan_index")
    )
    at_latest = schedule.join(latest, on=cell_keys, how="inner").filter(
        pl.col("replan_index") == pl.col("_latest_replan_index")
    )
    occupying = at_latest.filter(pl.col("schedule_status").is_in(sorted(OCCUPYING_STATUSES)))
    if execution_log.is_empty():
        return occupying.drop("_latest_replan_index")
    return occupying.join(
        execution_log.select(list(EXECUTION_JOIN_KEYS)).unique(),
        on=list(EXECUTION_JOIN_KEYS),
        how="anti",
    ).drop("_latest_replan_index")


def build_review_cases(
    recommendations: pl.DataFrame,
    schedule: pl.DataFrame | None,
    execution_log: pl.DataFrame | None,
) -> list[ReviewCase]:
    """Union both trigger populations into one case per flagged establishment cell.

    ``schedule``/``execution_log`` are optional: when either is absent, only the warning trigger
    runs. A row hit by both triggers carries both reasons, sorted and pipe-joined -- the same
    convention Component 13 uses for multiple warnings on one row.
    """
    triggers: dict[tuple[str, str, str, str, str, str], set[str]] = {}
    schedule_info: dict[tuple[str, str, str, str, str, str], dict[str, object]] = {}

    warned = warning_triggered_rows(recommendations)
    for row in warned.iter_rows(named=True):
        key = (
            row["policy_id"],
            row["model_name"],
            row["fold_set"],
            row["fold_id"],
            row["k_name"],
            row["target_inspection_id"],
        )
        triggers.setdefault(key, set()).add(ReviewTriggerReason.POLICY_WARNING_PRESENT)

    if schedule is not None and not schedule.is_empty():
        gapped = execution_gap_rows(
            schedule, execution_log if execution_log is not None else schedule.head(0)
        )
        for row in gapped.iter_rows(named=True):
            key = (
                row["policy_id"],
                row["model_name"],
                row["fold_set"],
                row["fold_id"],
                row["k_name"],
                row["target_inspection_id"],
            )
            triggers.setdefault(key, set()).add(
                ReviewTriggerReason.NO_EXECUTION_RECORD_ON_SCHEDULED_ROW
            )
            schedule_info[key] = {
                "schedule_config_id": row["schedule_config_id"],
                "planning_run_id": row["planning_run_id"],
                "replan_index": row["replan_index"],
                "scheduled_date": row["scheduled_date"],
            }

    by_key = {
        (
            row["policy_id"],
            row["model_name"],
            row["fold_set"],
            row["fold_id"],
            row["k_name"],
            row["target_inspection_id"],
        ): row
        for row in recommendations.iter_rows(named=True)
    }

    cases: list[ReviewCase] = []
    for key, reasons in triggers.items():
        source = by_key.get(key)
        establishment_id = source["establishment_id"] if source is not None else ""
        final_policy_rank = source["final_policy_rank"] if source is not None else None
        decision_mechanism = source["decision_mechanism"] if source is not None else ""
        decision_reason = source["decision_reason"] if source is not None else ""
        warnings = source["warnings"] if source is not None else "none"
        info = schedule_info.get(key, {})
        cases.append(
            ReviewCase(
                policy_id=key[0],
                model_name=key[1],
                fold_set=key[2],
                fold_id=key[3],
                k_name=key[4],
                target_inspection_id=key[5],
                establishment_id=establishment_id,
                final_policy_rank=final_policy_rank,
                decision_mechanism=decision_mechanism,
                decision_reason=decision_reason,
                warnings=warnings,
                schedule_config_id=cast("str | None", info.get("schedule_config_id")),
                planning_run_id=cast("str | None", info.get("planning_run_id")),
                replan_index=cast("int | None", info.get("replan_index")),
                scheduled_date=cast("date | None", info.get("scheduled_date")),
                trigger_reasons=tuple(sorted(reasons)),
            )
        )
    return cases


def trigger_column(case: ReviewCase) -> str:
    """The sorted, pipe-joined trigger set, or the no-trigger token. Never called on a real
    queue row without at least one trigger -- ``ReviewCase`` enforces that at construction."""
    if not case.trigger_reasons:
        return NO_TRIGGER
    return TRIGGER_SEPARATOR.join(sorted(case.trigger_reasons))


__all__ = [
    "build_review_cases",
    "execution_gap_rows",
    "trigger_column",
    "warning_triggered_rows",
]
