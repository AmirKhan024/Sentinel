"""Parquet schemas and deterministic writing for Component 14.

Column order is part of the data contract; changing it is a contract change. See
``docs/data_contracts/inspection_schedule.md``.

Thirteen tables in one new processed layer (ADR 0041). Four of the splits are load-bearing
rather than tidy.

**The schedule carries the queue, not the universe.** Component 13 already owns a
universe-grained artifact and already answers *why was this establishment not inspected*. This
layer answers a different question -- *when is this approved recommendation being worked* --
and its population is therefore the approved queue. Writing the universe again here would copy
1.4 million rows to restate an answer that exists one layer up.

**The horizon is its own table and is policy-independent.** The operating days and their
volumes depend on the fold, the capacity level and the capacity mode, and on nothing else.
Duplicating them per policy would write one measured fact seven times and invite the seven
copies to disagree; Component 13 makes the same argument for its model-independent eligibility
table.

**The backlog is its own table because it is a population, not a flag.** It carries the rank,
the mechanism, the shortfall and the fold's next available day -- the columns that turn "ten
rows did not fit" into something an operations manager can act on. A boolean on the schedule
row could not carry any of them.

**The contract is emitted as data.** ``execution_contract`` holds the two external file formats
field by field, so a reader who opens the Parquet layer rather than the markdown can still
construct a valid adjustment or execution file. A contract that lives only in prose is a
contract that drifts from the parser.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

#: The table the manifest is keyed to. The schedule is the component's answer.
DATASET_SLUG = "inspection_schedule"


#: One row per scheduling configuration. Frozen before any plan was built, and emitted so that
#: a run's grid travels with its results -- including which configuration is a scenario, which
#: is the label that must never be dropped.
SCHEDULE_CONFIGURATIONS_SCHEMA: dict[str, pl.DataType] = {
    "schedule_config_id": pl.Utf8(),
    "strategy_id": pl.Utf8(),
    "capacity_mode": pl.Utf8(),
    "is_scenario": pl.Boolean(),
    "is_default": pl.Boolean(),
    "preserves_priority_exactly": pl.Boolean(),
    "horizon_rule": pl.Utf8(),
    "capacity_rule": pl.Utf8(),
    "rationale": pl.Utf8(),
    "schedule_definition_version": pl.Utf8(),
}

#: One row per (config, fold, capacity level, operating day). The capacity grid, and the only
#: place a slot count is stated. Day-grained rather than slot-grained deliberately: a row per
#: individual slot would be a Cartesian product against nothing, and every question it could
#: answer is answerable from this table joined to the schedule's ``slot_index``.
SCHEDULE_SLOTS_SCHEMA: dict[str, pl.DataType] = {
    "schedule_config_id": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "k": pl.Int64(),
    "median_daily_capacity": pl.Int64(),
    "horizon_days": pl.Int64(),
    "day_index": pl.Int64(),
    "slot_date": pl.Date(),
    "n_slots": pl.Int64(),
    # Where this day's slot count came from. On the row rather than inferred from the mode, so
    # a reader looking at one day can tell an observation from an assumption.
    "capacity_source": pl.Utf8(),
    "cumulative_slots": pl.Int64(),
    "is_scenario": pl.Boolean(),
    "horizon_was_clamped": pl.Boolean(),
    "schedule_definition_version": pl.Utf8(),
}

#: The component's answer. Component 13's provenance is carried verbatim in the first block and
#: Component 14's scheduling provenance in the second, and the two never mix -- so a reader can
#: point at any row and ask "did the policy decide this or did the calendar?" and the artifact
#: answers without anyone reading source code. ADR 0037's ``model_rank`` beside
#: ``final_policy_rank`` is the pattern, one layer further out.
#:
#: There is deliberately **no execution_status column**. An execution outcome must never
#: retroactively change a plan, and the strongest form of that guarantee is not to give it a
#: column to write into.
INSPECTION_SCHEDULE_SCHEMA: dict[str, pl.DataType] = {
    "schedule_config_id": pl.Utf8(),
    "policy_id": pl.Utf8(),
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "k": pl.Int64(),
    "target_inspection_id": pl.Utf8(),
    "establishment_id": pl.Utf8(),
    # --- Component 13, verbatim. Never recomputed, never overwritten. ---
    "recommendation_date": pl.Date(),
    "base_score": pl.Float64(),
    "score": pl.Float64(),
    "model_rank": pl.Int64(),
    "final_policy_rank": pl.Int64(),
    "decision_mechanism": pl.Utf8(),
    "decision_reason": pl.Utf8(),
    "coverage_eligible": pl.Boolean(),
    "warnings": pl.Utf8(),
    "recommendation_override_id": pl.Utf8(),
    "policy_definition_version": pl.Utf8(),
    # --- Component 14. Kept separate. ---
    "planning_run_id": pl.Utf8(),
    "replan_index": pl.Int64(),
    "schedule_status": pl.Utf8(),
    "schedule_reason": pl.Utf8(),
    "inversion_reason": pl.Utf8(),
    "scheduled_date": pl.Date(),
    "day_index": pl.Int64(),
    "slot_index": pl.Int64(),
    "schedule_rank": pl.Int64(),
    "wait_operating_days": pl.Int64(),
    "original_scheduled_date": pl.Date(),
    "original_schedule_rank": pl.Int64(),
    "adjustment_id": pl.Utf8(),
    "is_scenario": pl.Boolean(),
    "schedule_definition_version": pl.Utf8(),
}

#: Recommended, approved, and not reached. A population rather than a flag.
SCHEDULE_BACKLOG_SCHEMA: dict[str, pl.DataType] = {
    "schedule_config_id": pl.Utf8(),
    "policy_id": pl.Utf8(),
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "k": pl.Int64(),
    "target_inspection_id": pl.Utf8(),
    "establishment_id": pl.Utf8(),
    "final_policy_rank": pl.Int64(),
    "decision_mechanism": pl.Utf8(),
    "decision_reason": pl.Utf8(),
    "coverage_eligible": pl.Boolean(),
    "backlog_position": pl.Int64(),
    "backlog_reason": pl.Utf8(),
    "horizon_slots": pl.Int64(),
    "slots_short": pl.Int64(),
    # Null when the fold's remaining calendar cannot reach this row at all, which is a
    # different and worse answer than a large number.
    "would_fit_on_day_index": pl.Int64(),
    "first_available_date": pl.Date(),
    "planning_run_id": pl.Utf8(),
    "replan_index": pl.Int64(),
    "is_scenario": pl.Boolean(),
    "schedule_definition_version": pl.Utf8(),
}

#: The schedule's account of its own arithmetic. ``n_recommended`` restates Component 13's k so
#: the three counts below it can be checked against it without a join.
SCHEDULE_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "schedule_config_id": pl.Utf8(),
    "policy_id": pl.Utf8(),
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "k": pl.Int64(),
    "median_daily_capacity": pl.Int64(),
    "horizon_days": pl.Int64(),
    "horizon_start_date": pl.Date(),
    "horizon_end_date": pl.Date(),
    "horizon_slots": pl.Int64(),
    "n_recommended": pl.Int64(),
    # n_scheduled + n_backlog + n_cancelled == n_recommended. n_deferred is a breakdown of
    # n_scheduled, not a fourth bucket: a deferred row still holds a slot, and adding it as a
    # sibling would double-count exactly the rows somebody moved.
    "n_scheduled": pl.Int64(),
    "n_backlog": pl.Int64(),
    "n_deferred": pl.Int64(),
    "n_cancelled": pl.Int64(),
    "idle_slots": pl.Int64(),
    "capacity_utilization": pl.Float64(),
    "horizon_was_clamped": pl.Boolean(),
    "n_adjustments_applied": pl.Int64(),
    "n_execution_events": pl.Int64(),
    "planning_run_id": pl.Utf8(),
    "replan_index": pl.Int64(),
    "is_scenario": pl.Boolean(),
    "schedule_definition_version": pl.Utf8(),
}

#: One row per operating day per cell: what it held, what it used, and by which mechanism.
CAPACITY_UTILIZATION_SCHEMA: dict[str, pl.DataType] = {
    "schedule_config_id": pl.Utf8(),
    "policy_id": pl.Utf8(),
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "day_index": pl.Int64(),
    "slot_date": pl.Date(),
    "n_slots": pl.Int64(),
    "n_scheduled": pl.Int64(),
    "idle_slots": pl.Int64(),
    "utilization": pl.Float64(),
    "n_risk_scheduled": pl.Int64(),
    "n_reserve_scheduled": pl.Int64(),
    "capacity_source": pl.Utf8(),
    "is_scenario": pl.Boolean(),
    "schedule_definition_version": pl.Utf8(),
}

#: The component's evidence, in four column groups that are never summed together. The last
#: group is the headline: ``reserve_slots_lost`` is what a strict-priority schedule costs the
#: coverage allocation, because Component 13 places the reserve at the tail of the rank order.
PRIORITY_PRESERVATION_SCHEMA: dict[str, pl.DataType] = {
    "schedule_config_id": pl.Utf8(),
    "policy_id": pl.Utf8(),
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "k": pl.Int64(),
    # --- priority ---
    "n_scheduled": pl.Int64(),
    "n_inversions": pl.Int64(),
    "max_inversion_depth": pl.Int64(),
    "rank_spearman": pl.Float64(),
    "strict_priority_preserved": pl.Boolean(),
    "n_rows_with_inversion_reason": pl.Int64(),
    # --- queue coverage ---
    "queue_coverage": pl.Float64(),
    "worst_scheduled_policy_rank": pl.Int64(),
    "best_backlogged_policy_rank": pl.Int64(),
    # --- wait, in operating days. Never calendar days: the dataset has no clock. ---
    "mean_wait_operating_days": pl.Float64(),
    "median_wait_operating_days": pl.Float64(),
    "max_wait_operating_days": pl.Int64(),
    # --- mechanism preservation: the headline ---
    "n_risk_recommended": pl.Int64(),
    "n_reserve_recommended": pl.Int64(),
    "n_risk_scheduled": pl.Int64(),
    "n_reserve_scheduled": pl.Int64(),
    "reserve_share_recommended": pl.Float64(),
    "reserve_share_scheduled": pl.Float64(),
    "reserve_share_delta": pl.Float64(),
    "reserve_slots_lost": pl.Int64(),
    # --- external impact, kept in its own category so it is never read as a schedule effect ---
    "n_adjusted": pl.Int64(),
    "n_displaced_by_adjustment": pl.Int64(),
    "n_execution_completed": pl.Int64(),
    "n_execution_not_performed": pl.Int64(),
    "n_execution_cancelled": pl.Int64(),
    "n_no_execution_record": pl.Int64(),
    "is_scenario": pl.Boolean(),
    "schedule_definition_version": pl.Utf8(),
}

#: One row per adjustment offered, applied or not. Typed empty on runs nobody supplied a file
#: for -- a typed empty table, not a missing file, so a reader meets the right columns and can
#: tell "nobody adjusted anything" from "this run did not support adjustments".
SCHEDULE_ADJUSTMENT_LOG_SCHEMA: dict[str, pl.DataType] = {
    "adjustment_id": pl.Utf8(),
    "schedule_config_id": pl.Utf8(),
    "policy_id": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "target_inspection_id": pl.Utf8(),
    "action": pl.Utf8(),
    "target_date": pl.Utf8(),
    "reason_code": pl.Utf8(),
    "actor": pl.Utf8(),
    "decided_at": pl.Utf8(),
    "original_status": pl.Utf8(),
    "original_scheduled_date": pl.Date(),
    "original_schedule_rank": pl.Int64(),
    "final_status": pl.Utf8(),
    "final_scheduled_date": pl.Date(),
    # Always a risk-priority row, never the coverage reserve. The validator checks it.
    "displaced_target_inspection_id": pl.Utf8(),
    "displaced_landed_status": pl.Utf8(),
    "outcome": pl.Utf8(),
    "planning_run_id": pl.Utf8(),
    "replan_index": pl.Int64(),
    "schedule_definition_version": pl.Utf8(),
}

#: The two external file formats, as data. Always non-empty: it is the frozen contract, and a
#: reader who never opens the markdown must still be able to construct a valid file.
EXECUTION_CONTRACT_SCHEMA: dict[str, pl.DataType] = {
    "contract_name": pl.Utf8(),
    "field_name": pl.Utf8(),
    "required": pl.Boolean(),
    "dtype": pl.Utf8(),
    "allowed_values": pl.Utf8(),
    "meaning": pl.Utf8(),
    "schedule_definition_version": pl.Utf8(),
}

#: One row per execution event offered. Typed empty when nobody supplied a file.
#:
#: ``scheduled_date`` and ``plan_scheduled_date`` are both kept and never merged. A field log
#: that disagrees with the plan is a fact about operations, and overwriting either value would
#: destroy the only evidence that it happened.
EXECUTION_LOG_SCHEMA: dict[str, pl.DataType] = {
    "execution_id": pl.Utf8(),
    "schedule_config_id": pl.Utf8(),
    "policy_id": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "target_inspection_id": pl.Utf8(),
    "scheduled_date": pl.Date(),
    "plan_scheduled_date": pl.Date(),
    "execution_status": pl.Utf8(),
    "reason_code": pl.Utf8(),
    "actor": pl.Utf8(),
    "observed_at": pl.Utf8(),
    "outcome": pl.Utf8(),
    "triggers_replan": pl.Boolean(),
    "applied_at_replan_index": pl.Int64(),
    "schedule_definition_version": pl.Utf8(),
}

#: Execution counts per cell, with "nobody reported" as its own visible category rather than
#: silently folded into "not completed".
EXECUTION_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "schedule_config_id": pl.Utf8(),
    "policy_id": pl.Utf8(),
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "n_scheduled": pl.Int64(),
    "n_completed": pl.Int64(),
    "n_not_performed": pl.Int64(),
    "n_cancelled_in_field": pl.Int64(),
    "n_no_execution_record": pl.Int64(),
    "completion_rate": pl.Float64(),
    "final_replan_index": pl.Int64(),
    "execution_log_sha256": pl.Utf8(),
    "schedule_definition_version": pl.Utf8(),
}

#: One row per planning run, starting at the original plan. Never empty: the original plan is a
#: planning run, and saying so is more honest than emitting nothing and leaving a reader to
#: infer that a plan existed.
REPLANNING_RUNS_SCHEMA: dict[str, pl.DataType] = {
    "schedule_config_id": pl.Utf8(),
    "policy_id": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "planning_run_id": pl.Utf8(),
    "replan_index": pl.Int64(),
    "parent_replan_index": pl.Int64(),
    "replan_from_date": pl.Date(),
    "trigger": pl.Utf8(),
    "n_preserved_completed": pl.Int64(),
    "n_preserved_past": pl.Int64(),
    "n_returned_to_queue": pl.Int64(),
    "n_cancelled": pl.Int64(),
    "n_newly_scheduled": pl.Int64(),
    "n_still_backlog": pl.Int64(),
    "remaining_slots": pl.Int64(),
    "execution_log_sha256": pl.Utf8(),
    "schedule_definition_version": pl.Utf8(),
}

#: One row per advisory finding. Advisories never fail a run and there is no flag to change
#: that; they are emitted as data so a finding travels with the artifact.
SCHEDULE_ADVISORIES_SCHEMA: dict[str, pl.DataType] = {
    "code": pl.Utf8(),
    "severity": pl.Utf8(),
    "scope": pl.Utf8(),
    "n_cells": pl.Int64(),
    "detail": pl.Utf8(),
    "schedule_definition_version": pl.Utf8(),
}


SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "schedule_configurations": SCHEDULE_CONFIGURATIONS_SCHEMA,
    "schedule_slots": SCHEDULE_SLOTS_SCHEMA,
    "inspection_schedule": INSPECTION_SCHEDULE_SCHEMA,
    "schedule_backlog": SCHEDULE_BACKLOG_SCHEMA,
    "schedule_summary": SCHEDULE_SUMMARY_SCHEMA,
    "capacity_utilization": CAPACITY_UTILIZATION_SCHEMA,
    "priority_preservation": PRIORITY_PRESERVATION_SCHEMA,
    "schedule_adjustment_log": SCHEDULE_ADJUSTMENT_LOG_SCHEMA,
    "execution_contract": EXECUTION_CONTRACT_SCHEMA,
    "execution_log": EXECUTION_LOG_SCHEMA,
    "execution_summary": EXECUTION_SUMMARY_SCHEMA,
    "replanning_runs": REPLANNING_RUNS_SCHEMA,
    "schedule_advisories": SCHEDULE_ADVISORIES_SCHEMA,
}

#: Every sort key is a **total order**: two runs over identical inputs produce byte-identical
#: files. A partial key would leave ties resolved by append order, which is not a contract --
#: Component 12 shipped exactly that defect once and Component 13 recorded the lesson.
_CELL = ["schedule_config_id", "policy_id", "model_name", "fold_set", "fold_id", "k_name"]

SORT_KEYS: dict[str, list[str]] = {
    "schedule_configurations": ["schedule_config_id"],
    "schedule_slots": ["schedule_config_id", "fold_set", "fold_id", "k_name", "slot_date"],
    # ``replan_index`` is part of the grain: a re-plan appends a whole new plan beside the
    # old one rather than editing it, so one scored inspection legitimately holds one row per
    # planning run. Without the index in the key the sort is not a total order and two plans
    # would be indistinguishable in the artifact.
    "inspection_schedule": [*_CELL, "replan_index", "target_inspection_id"],
    "schedule_backlog": [*_CELL, "replan_index", "target_inspection_id"],
    "schedule_summary": _CELL,
    "capacity_utilization": [*_CELL, "slot_date"],
    "priority_preservation": _CELL,
    "schedule_adjustment_log": ["adjustment_id"],
    "execution_contract": ["contract_name", "field_name"],
    "execution_log": ["execution_id"],
    "execution_summary": _CELL,
    "replanning_runs": [
        "schedule_config_id",
        "policy_id",
        "fold_set",
        "fold_id",
        "k_name",
        "replan_index",
    ],
    "schedule_advisories": ["code", "scope"],
}

#: The tables in the order the manifest lists them, schedule first.
LAYERS: tuple[str, ...] = (
    "inspection_schedule",
    "schedule_backlog",
    "schedule_slots",
    "schedule_summary",
    "capacity_utilization",
    "priority_preservation",
    "schedule_configurations",
    "schedule_adjustment_log",
    "execution_contract",
    "execution_log",
    "execution_summary",
    "replanning_runs",
    "schedule_advisories",
)


def finalize(rows: list[dict[str, object]], table: str) -> pl.DataFrame:
    """Cast to the contract schema and order deterministically.

    The schema is passed to the ``DataFrame`` constructor rather than cast afterwards, for the
    reason Component 12 recorded and Component 13 restated: this component emits nulls in
    quantity -- a slot date for a backlogged row, a wait for a row nobody scheduled, a
    completion rate for a cell with no execution record -- and inference would type a column
    from whichever value happened to arrive first. A column that is null for the first two
    hundred rows would arrive as ``Null`` rather than ``Date``, and the file would no longer
    match its contract.
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

    The adjustment log and the execution log are empty on every run nobody supplied a file for,
    which is most of them. A missing file would be ambiguous between "nobody adjusted anything"
    and "this build does not support adjustments"; a typed empty table says the first.
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
