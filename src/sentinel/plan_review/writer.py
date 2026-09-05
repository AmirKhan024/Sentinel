"""Build and write the Component 21 output table: the supervisor plan review.

The grain is exactly Component 20's plan-frame grain (one row per selected, geographically
organized establishment). This module only adds columns; it never removes or edits one it did
not add. The joined decision columns are nullable: an establishment with no recorded decision
carries nulls in every ``supervisor_decision_*`` column, never a fabricated default action.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

#: Additive columns Component 21 attaches to Component 20's plan-frame schema. Every column
#: Component 20 wrote is preserved verbatim; nothing here is subtracted from it.
DECISION_COLUMNS: dict[str, pl.DataType] = {
    "plan_review_definition_version": pl.Utf8(),
    "supervisor_decision_id": pl.Utf8(),
    "supervisor_decision_action": pl.Utf8(),
    "supervisor_decision_reason_code": pl.Utf8(),
    "supervisor_decision_actor": pl.Utf8(),
    "supervisor_decision_decided_at": pl.Utf8(),
    "supervisor_revised_planned_date": pl.Utf8(),
    "supervisor_revised_work_block_id": pl.Utf8(),
    "supervisor_revised_operational_priority": pl.Int64(),
    #: Sentinel's own ``policy_rank`` unless a supervisor recorded
    #: ADJUST_OPERATIONAL_PRIORITY, in which case the supervisor's value -- a display-only
    #: field work order, never a substitute for ``rank``/``policy_rank``, both of which are
    #: still present, unedited, on every row.
    "operational_priority": pl.Int64(),
}

#: Every decision-log row, one per applied/rejected decision -- the audit trail itself.
DECISION_LOG_SCHEMA: dict[str, pl.DataType] = {
    "decision_id": pl.Utf8(),
    "planning_date": pl.Utf8(),
    "target_inspection_id": pl.Utf8(),
    "decision_action": pl.Utf8(),
    "reason_code": pl.Utf8(),
    "actor": pl.Utf8(),
    "decided_at": pl.Utf8(),
    "revised_planned_date": pl.Utf8(),
    "revised_work_block_id": pl.Utf8(),
    "revised_operational_priority": pl.Int64(),
    "outcome": pl.Utf8(),
    "plan_review_definition_version": pl.Utf8(),
}


def finalize(
    plan_frame: pl.DataFrame,
    *,
    plan_review_definition_version: str,
    decision_by_target_id: dict[str, dict[str, object]],
) -> pl.DataFrame:
    """Join decision columns onto Component 20's plan frame, additive only."""
    target_ids = plan_frame["target_inspection_id"].to_list()

    def _col(field: str) -> list[object]:
        return [decision_by_target_id.get(tid, {}).get(field) for tid in target_ids]

    revised_priority = _col("revised_operational_priority")
    with_decisions = plan_frame.with_columns(
        pl.lit(plan_review_definition_version).alias("plan_review_definition_version"),
        pl.Series("supervisor_decision_id", _col("decision_id"), dtype=pl.Utf8),
        pl.Series("supervisor_decision_action", _col("decision_action"), dtype=pl.Utf8),
        pl.Series("supervisor_decision_reason_code", _col("reason_code"), dtype=pl.Utf8),
        pl.Series("supervisor_decision_actor", _col("actor"), dtype=pl.Utf8),
        pl.Series("supervisor_decision_decided_at", _col("decided_at"), dtype=pl.Utf8),
        pl.Series(
            "supervisor_revised_planned_date",
            _col("revised_planned_date"),
            dtype=pl.Utf8,
        ),
        pl.Series(
            "supervisor_revised_work_block_id",
            _col("revised_work_block_id"),
            dtype=pl.Utf8,
        ),
        pl.Series("supervisor_revised_operational_priority", revised_priority, dtype=pl.Int64),
    )
    # operational_priority: the supervisor's value where ADJUST_OPERATIONAL_PRIORITY was
    # recorded, else Sentinel's own policy_rank, unedited. Never the reverse.
    return with_decisions.with_columns(
        pl.coalesce(pl.col("supervisor_revised_operational_priority"), pl.col("policy_rank")).alias(
            "operational_priority"
        )
    )


def finalize_decision_log(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=DECISION_LOG_SCHEMA)
    ordered = [{name: row.get(name) for name in DECISION_LOG_SCHEMA} for row in rows]
    return pl.DataFrame(ordered, schema=DECISION_LOG_SCHEMA).sort("decision_id")


def write_table(frame: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")
    logger.info("Wrote %s (%d rows, %d columns)", path, frame.height, frame.width)
    return path


def schema_of(frame: pl.DataFrame) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in frame.schema.items()}


__all__ = [
    "DECISION_COLUMNS",
    "DECISION_LOG_SCHEMA",
    "finalize",
    "finalize_decision_log",
    "schema_of",
    "write_table",
]
