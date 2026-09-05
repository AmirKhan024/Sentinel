"""Parquet schemas and deterministic writing for Component 13.

Column order is part of the data contract; changing it is a contract change. See
``docs/data_contracts/policy_decisions.md``.

Eleven tables in one new processed layer (ADR 0036). Three of the splits are load-bearing
rather than tidy.

**The recommendation table carries the whole prediction universe, not only the queue.** It
would be smaller to write the selected rows and call that the recommendation, and it would
make the most important question unanswerable: *why was this establishment not inspected?* A
queue-only artifact can say who was chosen; only a universe-grained one can say who was
considered and what happened to them, which is what an establishment owner or an alderman is
actually asking.

**Allocation is its own table because it is the policy's own account of itself.** How many
slots the reserve was offered, how many the risk block had already filled, and how many were
finally granted are three different numbers, and only together do they distinguish "the floor
was satisfied" from "the floor was ignored" from "there were not enough eligible
establishments to satisfy it". Reconstructing them by aggregating the recommendation table
would be possible and would be a second implementation of the allocator.

**Opportunity cost sits on the comparison row, not in a separate table.** The single most
misreadable thing this component could produce is a coverage number without its price beside
it, so the price is a column on the same row rather than a join away.

**Every model-selection candidate is a row, including the refused one.** The refusal is data
so that a reader who opens the Parquet instead of ADR 0039 still finds out why there are four
candidates and not five, and finds the measurement that decided it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

#: The table the manifest is keyed to. The recommendation universe is the component's answer.
DATASET_SLUG = "inspection_recommendations"


#: One row per candidate policy. Frozen before any comparison ran, and emitted so that a run's
#: grid travels with its results rather than being looked up in a version of the source that
#: may have moved on.
POLICY_CONFIGURATIONS_SCHEMA: dict[str, pl.DataType] = {
    "policy_id": pl.Utf8(),
    "reserve_mechanism": pl.Utf8(),
    "reserve_share": pl.Float64(),
    # True for exactly one policy. Every opportunity-cost column in `policy_comparison` is
    # differenced against it, so which policy it is must not be inferable only from the name.
    "is_baseline": pl.Boolean(),
    "rationale": pl.Utf8(),
    "policy_definition_version": pl.Utf8(),
}

#: One row per calibrated model, admissible or refused, with every axis of the rule and both
#: outcomes -- the rule's, and the discarded tie band's. The second is here because the tie
#: rule decides which model is deployed and it was fixed after its inputs were first read;
#: recording only the outcome would hide that a different defensible rule gives another answer.
MODEL_SELECTION_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "admissible": pl.Boolean(),
    "admissibility_reason": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "nde": pl.Float64(),
    "nde_p05": pl.Float64(),
    "nde_p95": pl.Float64(),
    "ece": pl.Float64(),
    "precision_at_k_1_day": pl.Float64(),
    "tied_on_nde": pl.Boolean(),
    "is_selected": pl.Boolean(),
    "selected_under_discarded_band": pl.Boolean(),
    "decided_on_axis": pl.Utf8(),
    "policy_definition_version": pl.Utf8(),
}

#: One row per (grain, fold). Model-independent: which establishments have no code-era history
#: is a property of Component 4's table, not of any estimator, so repeating it per model would
#: copy one measured fact and invite the copies to disagree.
COVERAGE_ELIGIBILITY_SCHEMA: dict[str, pl.DataType] = {
    "grain": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "n_rows": pl.Int64(),
    "n_eligible": pl.Int64(),
    "eligible_share": pl.Float64(),
    "n_secondary_no_history": pl.Int64(),
    "n_positive": pl.Int64(),
    "n_eligible_positive": pl.Int64(),
    "base_rate": pl.Float64(),
    # Reported beside the window's own rate on purpose. The eligible population's outcome rate
    # runs *above* the window's in every quarterly fold, which is the fact that turns "the
    # model neglects establishments it knows nothing about" into a claim the data refutes.
    "eligible_base_rate": pl.Float64(),
    "eligible_share_of_positives": pl.Float64(),
    "eligibility_column": pl.Utf8(),
    "policy_definition_version": pl.Utf8(),
}

#: One row per (policy, capacity, fold, scored establishment). The component's answer, at the
#: grain a person asks about: *this establishment, this quarter, this much capacity*.
RECOMMENDATIONS_SCHEMA: dict[str, pl.DataType] = {
    "policy_id": pl.Utf8(),
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "k": pl.Int64(),
    "target_inspection_id": pl.Utf8(),
    "establishment_id": pl.Utf8(),
    "inspection_date": pl.Date(),
    # The uncalibrated score is carried beside the calibrated one, as Component 9 does, so a
    # reader can see that the policy ranks on the calibrated value without performing a join.
    "base_score": pl.Float64(),
    "score": pl.Float64(),
    # The pair that makes the component's central question answerable per row. Where these two
    # agree the model decided; where they differ the policy did.
    "model_rank": pl.Int64(),
    "final_policy_rank": pl.Int64(),
    "is_selected": pl.Boolean(),
    "decision_mechanism": pl.Utf8(),
    "decision_reason": pl.Utf8(),
    "coverage_eligible": pl.Boolean(),
    "secondary_no_history": pl.Boolean(),
    # A sorted, pipe-joined set, or the literal token "none". Advisory: proven by the validator
    # to have no effect on either rank.
    "warnings": pl.Utf8(),
    # Component 12's as-of group label and support status, read onto the row for a reviewer's
    # benefit and never read back into an allocation.
    "group_value": pl.Utf8(),
    "group_status": pl.Utf8(),
    "policy_definition_version": pl.Utf8(),
}

#: One row per (policy, capacity, fold). The policy's account of its own arithmetic.
SELECTION_ALLOCATION_SCHEMA: dict[str, pl.DataType] = {
    "policy_id": pl.Utf8(),
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "k": pl.Int64(),
    "n_universe": pl.Int64(),
    "reserve_mechanism": pl.Utf8(),
    "reserve_share": pl.Float64(),
    # Offered, already-satisfied, and granted. Three numbers, because two of them cannot
    # distinguish a floor that was met from a floor that could not be met.
    "reserve_target": pl.Int64(),
    "n_eligible_available": pl.Int64(),
    "n_eligible_in_risk_top_k": pl.Int64(),
    "n_risk": pl.Int64(),
    "n_reserve": pl.Int64(),
    "n_selected": pl.Int64(),
    "reserve_inert": pl.Boolean(),
    "policy_definition_version": pl.Utf8(),
}

#: One row per (policy, model, capacity, fold). Effectiveness, coverage and the price of the
#: coverage, on one row. The ``delta_`` columns are all measured against ``pure_risk`` at the
#: identical model, fold and capacity.
COMPARISON_SCHEMA: dict[str, pl.DataType] = {
    "policy_id": pl.Utf8(),
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "k": pl.Int64(),
    "n_universe": pl.Int64(),
    "n_selected": pl.Int64(),
    "n_risk": pl.Int64(),
    "n_reserve": pl.Int64(),
    "reserve_target": pl.Int64(),
    "reserve_inert": pl.Boolean(),
    "n_positive": pl.Int64(),
    "positives_selected": pl.Int64(),
    "precision_at_k": pl.Float64(),
    "capture_rate": pl.Float64(),
    "lift_at_k": pl.Float64(),
    "nde": pl.Float64(),
    "n_eligible_available": pl.Int64(),
    "n_eligible_in_risk_top_k": pl.Int64(),
    "eligible_selected": pl.Int64(),
    "eligible_selected_share": pl.Float64(),
    "eligible_positives_selected": pl.Int64(),
    "eligible_capture_rate": pl.Float64(),
    # The opportunity cost, in the unit that means something to a department first.
    "delta_positives": pl.Float64(),
    "delta_precision": pl.Float64(),
    "delta_capture": pl.Float64(),
    "delta_nde": pl.Float64(),
    "delta_eligible_selected": pl.Float64(),
    "delta_eligible_capture": pl.Float64(),
    "policy_definition_version": pl.Utf8(),
}

#: One row per (policy, model, fold set, capacity), pooled. Marks dominated policies and stops
#: there: choosing among the survivors needs an exchange rate between a missed citation and an
#: uninspected establishment with no history, and nothing in this project measures one.
FRONTIER_SCHEMA: dict[str, pl.DataType] = {
    "policy_id": pl.Utf8(),
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "k_name": pl.Utf8(),
    "positives_selected": pl.Float64(),
    "eligible_selected": pl.Float64(),
    "is_dominated": pl.Boolean(),
    "dominated_by": pl.Utf8(),
    "policy_definition_version": pl.Utf8(),
}

#: One row per (policy, capacity, fold, group). Descriptive. Component 12's ``group_status``
#: travels with every row, so an unsupported group stays unsupported here rather than becoming
#: a number somebody quotes.
GROUP_AUDIT_SCHEMA: dict[str, pl.DataType] = {
    "policy_id": pl.Utf8(),
    "model_name": pl.Utf8(),
    "group_definition": pl.Utf8(),
    "group_value": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "n_rows": pl.Int64(),
    "n_positive": pl.Int64(),
    "population_share": pl.Float64(),
    "n_selected": pl.Int64(),
    "selected_share": pl.Float64(),
    "selection_rate": pl.Float64(),
    "positives_selected": pl.Int64(),
    "capture_rate": pl.Float64(),
    "group_status": pl.Utf8(),
    "policy_definition_version": pl.Utf8(),
}

#: One row per (policy, capacity, fold, mechanism, reason, warning set). The distribution of
#: *why*, which is the table a reviewer reads before reading any individual recommendation.
DECISION_REASONS_SCHEMA: dict[str, pl.DataType] = {
    "policy_id": pl.Utf8(),
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "decision_mechanism": pl.Utf8(),
    "decision_reason": pl.Utf8(),
    "warnings": pl.Utf8(),
    "n_rows": pl.Int64(),
    "policy_definition_version": pl.Utf8(),
}

#: One row per advisory finding. Component 12 kept its advisories in the manifest only; this
#: component tabulates them as well, because a policy run produces many more of them and
#: "which cells were inert, and which gave up citations" is a question with a shape that a
#: list of formatted strings answers badly.
ADVISORIES_SCHEMA: dict[str, pl.DataType] = {
    "code": pl.Utf8(),
    "severity": pl.Utf8(),
    "scope": pl.Utf8(),
    "n_cells": pl.Int64(),
    "detail": pl.Utf8(),
    "policy_definition_version": pl.Utf8(),
}

#: One row per override offered, applied or not. The original recommendation and the final
#: decision sit side by side because an audit never asks only what happened -- it asks what
#: would have happened, what happened instead, and who decided.
OVERRIDE_LOG_SCHEMA: dict[str, pl.DataType] = {
    "override_id": pl.Utf8(),
    "policy_id": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "target_inspection_id": pl.Utf8(),
    "action": pl.Utf8(),
    "reason_code": pl.Utf8(),
    "actor": pl.Utf8(),
    "decided_at": pl.Utf8(),
    "original_is_selected": pl.Boolean(),
    "original_mechanism": pl.Utf8(),
    "original_reason": pl.Utf8(),
    "original_policy_rank": pl.Int64(),
    "final_is_selected": pl.Boolean(),
    "displaced_target_inspection_id": pl.Utf8(),
    "outcome": pl.Utf8(),
    "policy_definition_version": pl.Utf8(),
}


SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "policy_configurations": POLICY_CONFIGURATIONS_SCHEMA,
    "policy_model_selection": MODEL_SELECTION_SCHEMA,
    "policy_coverage_eligibility": COVERAGE_ELIGIBILITY_SCHEMA,
    "inspection_recommendations": RECOMMENDATIONS_SCHEMA,
    "policy_selection_allocation": SELECTION_ALLOCATION_SCHEMA,
    "policy_comparison": COMPARISON_SCHEMA,
    "policy_frontier": FRONTIER_SCHEMA,
    "policy_group_audit": GROUP_AUDIT_SCHEMA,
    "policy_decision_reasons": DECISION_REASONS_SCHEMA,
    "policy_advisories": ADVISORIES_SCHEMA,
    "policy_override_log": OVERRIDE_LOG_SCHEMA,
}

#: Every table lands in the one new layer. A recommendation is not a prediction, not a metric
#: and not a group measurement, so there is nowhere else it could go (ADR 0036).
LAYERS: dict[str, str] = dict.fromkeys(SCHEMAS, "policy")

#: Full keys, so a sort is a total order and two runs over identical inputs produce
#: byte-identical files. A partial key would leave ties resolved by whatever order the rows
#: happened to be appended in, which is not a contract -- and Component 12 shipped exactly
#: that defect once, in its equal-mass binning, before a test caught it.
SORT_KEYS: dict[str, list[str]] = {
    "policy_configurations": ["policy_id"],
    "policy_model_selection": ["model_name"],
    "policy_coverage_eligibility": ["grain", "fold_set", "fold_id"],
    "inspection_recommendations": [
        "policy_id",
        "model_name",
        "fold_set",
        "fold_id",
        "k_name",
        "target_inspection_id",
    ],
    "policy_selection_allocation": [
        "policy_id",
        "model_name",
        "fold_set",
        "fold_id",
        "k_name",
    ],
    "policy_comparison": ["policy_id", "model_name", "fold_set", "fold_id", "k_name"],
    "policy_frontier": ["model_name", "fold_set", "k_name", "policy_id"],
    "policy_group_audit": [
        "policy_id",
        "model_name",
        "group_definition",
        "fold_set",
        "fold_id",
        "k_name",
        "group_value",
    ],
    "policy_decision_reasons": [
        "policy_id",
        "model_name",
        "fold_set",
        "fold_id",
        "k_name",
        "decision_mechanism",
        "decision_reason",
        "warnings",
    ],
    "policy_advisories": ["code", "scope"],
    "policy_override_log": ["override_id"],
}


def finalize(rows: list[dict[str, object]], table: str) -> pl.DataFrame:
    """Cast to the contract schema and order deterministically.

    The schema is passed to the ``DataFrame`` constructor rather than cast afterwards, for the
    reason Component 12 recorded: this component emits nulls in quantity -- a capture rate for
    a window with no positives, a delta for a metric that was undefined on one side, a rank for
    a row nobody selected -- and inference would type a column from whichever value happened to
    arrive first. A column that is null for the first two hundred rows would arrive as ``Null``
    rather than ``Float64``, and the file would no longer match its contract.
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

    A real outcome rather than a defensive one: ``policy_override_log`` is empty on every run
    nobody supplied overrides for, which is most of them, and ``policy_group_audit`` is empty
    when the as-of group frame is absent. In both cases the reader should find the schema and
    conclude "no rows", not find a missing file and conclude "no such thing".
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
