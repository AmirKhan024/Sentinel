"""Parquet schemas and deterministic writing for Component 16.

Column order is part of the data contract; changing it is a contract change. See
``docs/data_contracts/human_review.md``.

Three tables in one new processed layer.

**The queue is rebuilt fresh each run, not accumulated.** ``human_review_queue`` reflects
current upstream state: a row whose trigger condition no longer holds (an execution record
arrives, closing the gap) drops off the next run's queue. History is not lost -- if a human ever
recorded a resolution for that case, it stays in ``review_resolution_log`` permanently. The split
mirrors ``inspection_recommendations`` (rebuilt) beside ``policy_override_log`` (append-only).

**The resolution log is its own table because a review decision is an external human input, not
reproducible computation.** Same discipline as Component 13's override log and Component 14's
adjustment and execution logs: the queue is written unchanged, the resolution sits beside it, and
the manifest pins the resolutions file by checksum rather than claiming a human decision is
reproducible.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

#: The table the manifest is keyed to. The review queue is the component's answer.
DATASET_SLUG = "human_review_queue"


#: One row per (policy, model, fold, capacity, flagged establishment). The component's answer:
#: which rows need a human, and why.
HUMAN_REVIEW_QUEUE_SCHEMA: dict[str, pl.DataType] = {
    "policy_id": pl.Utf8(),
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "target_inspection_id": pl.Utf8(),
    "establishment_id": pl.Utf8(),
    "final_policy_rank": pl.Int64(),
    "decision_mechanism": pl.Utf8(),
    "decision_reason": pl.Utf8(),
    "warnings": pl.Utf8(),
    # A sorted, pipe-joined set of ReviewTriggerReason values. Never blank on a real row.
    "trigger_reasons": pl.Utf8(),
    # Populated only for a case reached through the execution-gap trigger; blank otherwise.
    "schedule_config_id": pl.Utf8(),
    "planning_run_id": pl.Utf8(),
    "replan_index": pl.Int64(),
    "scheduled_date": pl.Date(),
    # FLAGGED or RESOLVED, joined from the accumulated resolution log by scope + target id.
    "review_status": pl.Utf8(),
    "review_id": pl.Utf8(),
    "resolution_action": pl.Utf8(),
    "review_definition_version": pl.Utf8(),
}

#: One row per resolution offered, applied or not. Typed empty on runs nobody supplied a file
#: for -- a typed empty table, not a missing file, so a reader meets the right columns and can
#: tell "nobody resolved anything" from "this run does not support resolutions".
REVIEW_RESOLUTION_LOG_SCHEMA: dict[str, pl.DataType] = {
    "review_id": pl.Utf8(),
    "policy_id": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "target_inspection_id": pl.Utf8(),
    "resolution_action": pl.Utf8(),
    "reason_code": pl.Utf8(),
    "actor": pl.Utf8(),
    "decided_at": pl.Utf8(),
    "referenced_override_id": pl.Utf8(),
    "referenced_adjustment_id": pl.Utf8(),
    "escalation_note": pl.Utf8(),
    "original_status": pl.Utf8(),
    "final_status": pl.Utf8(),
    "outcome": pl.Utf8(),
    "review_definition_version": pl.Utf8(),
}

#: One row per advisory finding. Advisories never fail a run and there is no flag to change
#: that; they are emitted as data so a finding travels with the artifact.
REVIEW_ADVISORIES_SCHEMA: dict[str, pl.DataType] = {
    "code": pl.Utf8(),
    "severity": pl.Utf8(),
    "scope": pl.Utf8(),
    "n_cases": pl.Int64(),
    "detail": pl.Utf8(),
    "review_definition_version": pl.Utf8(),
}


SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "human_review_queue": HUMAN_REVIEW_QUEUE_SCHEMA,
    "review_resolution_log": REVIEW_RESOLUTION_LOG_SCHEMA,
    "review_advisories": REVIEW_ADVISORIES_SCHEMA,
}

#: Every table lands in the one new layer. A review case is not a recommendation and not a
#: schedule slot, so there is nowhere else it could go.
LAYERS: dict[str, str] = dict.fromkeys(SCHEMAS, "review")

#: Full keys, so a sort is a total order and two runs over identical inputs produce
#: byte-identical files.
SORT_KEYS: dict[str, list[str]] = {
    "human_review_queue": [
        "policy_id",
        "model_name",
        "fold_set",
        "fold_id",
        "k_name",
        "target_inspection_id",
    ],
    "review_resolution_log": ["review_id"],
    "review_advisories": ["code", "scope"],
}


def finalize(rows: list[dict[str, object]], table: str) -> pl.DataFrame:
    """Cast to the contract schema and order deterministically.

    The schema is passed to the ``DataFrame`` constructor rather than cast afterwards, for the
    reason every earlier component recorded: nulls appear in quantity here -- a scheduled_date
    for a case reached only through the warning trigger, a policy rank for none of the queue's
    rows if it is ever empty -- and inference would type a column from whichever value happened
    to arrive first.
    """
    if table not in SCHEMAS:
        raise KeyError(f"Unknown table: {table}")
    schema = SCHEMAS[table]
    if not rows:
        return empty(table)
    missing = [c for c in schema if c not in rows[0]]
    if missing:
        raise ValueError(f"{table} rows are missing columns: {', '.join(missing)}")
    extra = [c for c in rows[0] if c not in schema]
    if extra:
        raise ValueError(f"{table} rows carry unknown columns: {', '.join(extra)}")
    return pl.DataFrame(rows, schema=schema).sort(SORT_KEYS[table])


def empty(table: str) -> pl.DataFrame:
    """A correctly typed zero-row frame, so a reader meets the right columns.

    A real outcome rather than a defensive one: ``review_resolution_log`` is empty on every run
    nobody supplied resolutions for, which is most of them, and ``human_review_queue`` is empty
    on a run where nothing matched either trigger.
    """
    if table not in SCHEMAS:
        raise KeyError(f"Unknown table: {table}")
    return pl.DataFrame(schema=SCHEMAS[table])


def write_table(frame: pl.DataFrame, path: Path) -> Path:
    """Write a table as zstd Parquet, matching project convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")
    logger.info("Wrote %s (%d rows, %d columns)", path, frame.height, frame.width)
    return path


def schema_of(frame: pl.DataFrame) -> dict[str, str]:
    """Column name to dtype string, for recording in the manifest."""
    return {name: str(dtype) for name, dtype in frame.schema.items()}


__all__ = [
    "DATASET_SLUG",
    "LAYERS",
    "SCHEMAS",
    "SORT_KEYS",
    "empty",
    "finalize",
    "schema_of",
    "write_table",
]
