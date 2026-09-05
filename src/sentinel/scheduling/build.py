"""Orchestration for Component 14: read, plan, measure, validate, write.

The only module with a clock, a filesystem and a run identifier. Everything it calls is pure,
which is what makes the pure parts testable without a temporary directory.

**The planning run id is a content hash, never a timestamp.** A run id derived from the clock
would make two runs over identical inputs differ in a column, and the byte-identity contract
would be unmeetable for a reason that has nothing to do with scheduling. It is derived from the
inputs and the configuration, so the same inputs produce the same id and a changed input
produces a visibly different one.

**The deterministic plan is always written, whatever else is supplied.** Adjustments and
execution events produce additional planning runs and their own logs; the plan at index 0 is
written unchanged beside them, and the validator rebuilds it with the external files withheld
to prove it. Component 13 writes its queue unchanged beside its override log for the same
reason and the pattern is inherited deliberately.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from sentinel import __version__ as code_version
from sentinel.config import Settings
from sentinel.manifest import compute_sha256, manifest_path_for, write_manifest
from sentinel.scheduling import figures as figure_module
from sentinel.scheduling import validate as checks
from sentinel.scheduling.adjustments import (
    OUTCOME_APPLIED,
    adjustment_log_rows,
    apply_adjustments,
)
from sentinel.scheduling.allocation import place
from sentinel.scheduling.backlog import backlog_rows
from sentinel.scheduling.definitions import (
    ADJUSTMENT_CANNOT,
    ADJUSTMENT_REQUIRED_FIELDS,
    ALLOCATION_CLAIM,
    BLOCKED,
    CALENDAR_IS_OBSERVED,
    CAPACITY_IS_INHERITED,
    CAPACITY_MODE_SCENARIO_CLAIM,
    CONFIG_GRID,
    DEFAULT_CAPACITY_MODE,
    DETERMINISM_SCOPE,
    DOES_NOT_ESTABLISH,
    EXECUTION_CANNOT,
    EXECUTION_REQUIRED_FIELDS,
    GREEN_RUN_MEANS,
    HORIZON_RULE,
    INHERITED_LIMITATIONS,
    INVERSION_REASON_REQUIRED,
    K_LEVELS,
    LAYER_SEPARATION,
    NO_CONSTRAINT_AWARE_STRATEGY,
    NO_SOLVER,
    PRIMARY_K_LEVEL,
    PRIORITY_IS_INHERITED,
    REPLAN_BACKFILL_RULE,
    REPLAN_PRESERVES,
    SCHEDULE_CLAIM,
    SCHEDULE_DEFINITION_VERSION,
    SCHEDULING_SEMANTICS,
    STRATEGY_BY_ID,
    TEMPORAL_BOUNDARY,
    THREE_HUMAN_LAYERS,
    AdjustmentAction,
    CapacityMode,
    ExecutionStatus,
    ScheduleConfigSpec,
)
from sentinel.scheduling.evaluate import (
    capacity_utilization_rows,
    preservation_row,
    summary_row,
)
from sentinel.scheduling.execution import (
    execution_log_rows,
    execution_summary_row,
    observed_date,
    record_execution,
)
from sentinel.scheduling.horizon import build_horizon
from sentinel.scheduling.inputs import (
    cell_frames,
    median_daily_by_fold,
    observed_calendars,
    queue_rows,
    read_adjustments,
    read_execution_events,
    read_folds,
    read_override_log,
    read_recommendations,
    validate_folds_against_recommendations,
    validate_recommendations,
)
from sentinel.scheduling.models import (
    Adjustment,
    ArtifactRecord,
    ExecutionEvent,
    QueueRow,
    ScheduleManifest,
    SchedulePlan,
    ScheduleStats,
    ValidationCheck,
)
from sentinel.scheduling.replan import original_run_row, replan, replan_point
from sentinel.scheduling.writer import (
    DATASET_SLUG,
    LAYERS,
    empty,
    finalize,
    schema_of,
    write_table,
)

logger = logging.getLogger(__name__)


@dataclass
class ScheduleResult:
    """Everything a caller needs to report on, or write, one scheduling run."""

    tables: dict[str, pl.DataFrame]
    checks: list[ValidationCheck]
    stats: ScheduleStats
    manifest: ScheduleManifest | None = None
    written: list[Path] = field(default_factory=list)


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _planning_run_id(*parts: str) -> str:
    """A stable run id from the inputs and the configuration. Never a clock, never random."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"PR-{digest}"


def _contract_rows() -> list[dict[str, object]]:
    """The two external file formats, as data.

    Emitted so a reader who opens the Parquet layer rather than the markdown can still build a
    valid file. A contract that lives only in prose is a contract that drifts from its parser.
    """
    meanings = {
        "adjustment_id": "unique id; adjustments are applied in this order, never file order",
        "execution_id": "unique id; events are recorded in this order, never file order",
        "schedule_config_id": "which configuration this change applies to",
        "policy_id": "Component 13 policy the queue came from",
        "fold_id": "the operating window",
        "k_name": "the capacity level",
        "target_inspection_id": "the scored inspection this change is about",
        "action": "defer_to_date, advance_to_date or cancel",
        "target_date": "an operating day inside the horizon; empty for cancel",
        "scheduled_date": "the day the field believes the inspection was planned for",
        "execution_status": "completed, not_performed or cancelled_in_field",
        "reason_code": "why; a controlled string chosen by the department, never generated",
        "actor": "who decided or reported. Never blank: an anonymous change is not auditable",
        "decided_at": "the supervisor's timestamp, never the run's clock",
        "observed_at": "the field's timestamp; the temporal boundary is measured from its date",
    }
    allowed = {
        "action": "|".join(sorted(str(a) for a in AdjustmentAction)),
        "execution_status": "|".join(sorted(str(s) for s in ExecutionStatus)),
    }
    rows: list[dict[str, object]] = []
    for contract, fields in (
        ("scheduling_adjustment", ADJUSTMENT_REQUIRED_FIELDS),
        ("execution_event", EXECUTION_REQUIRED_FIELDS),
    ):
        for name in fields:
            rows.append(
                {
                    "contract_name": contract,
                    "field_name": name,
                    "required": True,
                    "dtype": "string",
                    "allowed_values": allowed.get(name, ""),
                    "meaning": meanings.get(name, ""),
                    "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
                }
            )
    return rows


def _count(row: dict[str, object], key: str) -> int:
    """One integer out of a measured row, narrowly enough for strict typing.

    The measurement functions return ``dict[str, object]`` because their columns are a mix of
    counts, rates and dates. Coercing at the call site keeps that flexibility without letting a
    non-numeric value into a counter, where it would surface as a wrong total rather than an
    error.
    """
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} is {type(value).__name__}, expected a count")
    return value


def _configuration_rows(configs: Sequence[ScheduleConfigSpec]) -> list[dict[str, object]]:
    return [
        {
            "schedule_config_id": spec.schedule_config_id,
            "strategy_id": spec.strategy_id,
            "capacity_mode": str(spec.capacity_mode),
            "is_scenario": spec.is_scenario,
            "is_default": spec.is_default,
            "preserves_priority_exactly": STRATEGY_BY_ID[
                spec.strategy_id
            ].preserves_priority_exactly,
            "horizon_rule": HORIZON_RULE,
            "capacity_rule": (
                CALENDAR_IS_OBSERVED
                if spec.capacity_mode is CapacityMode.OBSERVED_CALENDAR
                else CAPACITY_MODE_SCENARIO_CLAIM
            ),
            "rationale": spec.rationale,
            "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
        }
        for spec in configs
    ]


def _schedule_rows(
    plan: SchedulePlan,
    queue: Sequence[QueueRow],
    cell: dict[str, str],
    spec: ScheduleConfigSpec,
    override_ids: dict[str, str],
    policy_version: str,
) -> list[dict[str, object]]:
    day_index = {day.slot_date: day.day_index for day in plan.horizon.days}
    by_id = {row.target_inspection_id: row for row in queue}
    out: list[dict[str, object]] = []
    for placement in plan.placements:
        row = by_id[placement.target_inspection_id]
        wait = (
            day_index[placement.slot_date] - 1
            if placement.slot_date is not None and placement.slot_date in day_index
            else None
        )
        out.append(
            {
                "schedule_config_id": spec.schedule_config_id,
                "policy_id": cell["policy_id"],
                "model_name": cell["model_name"],
                "fold_set": cell["fold_set"],
                "fold_id": cell["fold_id"],
                "k_name": cell["k_name"],
                "k": plan.k,
                "target_inspection_id": placement.target_inspection_id,
                "establishment_id": row.establishment_id,
                "recommendation_date": row.recommendation_date,
                "base_score": row.base_score,
                "score": row.score,
                "model_rank": row.model_rank,
                "final_policy_rank": row.final_policy_rank,
                "decision_mechanism": row.decision_mechanism,
                "decision_reason": row.decision_reason,
                "coverage_eligible": row.coverage_eligible,
                "warnings": row.warnings,
                "recommendation_override_id": override_ids.get(placement.target_inspection_id, ""),
                "policy_definition_version": policy_version,
                "planning_run_id": plan.planning_run_id,
                "replan_index": plan.replan_index,
                "schedule_status": placement.status,
                "schedule_reason": placement.reason,
                "inversion_reason": placement.inversion_reason,
                "scheduled_date": placement.slot_date,
                "day_index": placement.day_index,
                "slot_index": placement.slot_index,
                "schedule_rank": placement.schedule_rank,
                "wait_operating_days": wait,
                "original_scheduled_date": placement.original_slot_date,
                "original_schedule_rank": placement.original_schedule_rank,
                "adjustment_id": placement.adjustment_id,
                "is_scenario": spec.is_scenario,
                "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
            }
        )
    return out


def run_schedule(
    settings: Settings,
    *,
    recommendations_path: Path,
    folds_path: Path,
    override_log_path: Path | None = None,
    adjustments_path: Path | None = None,
    execution_path: Path | None = None,
    configs: Sequence[ScheduleConfigSpec] | None = None,
    policies: Sequence[str] | None = None,
    k_names: Sequence[str] | None = None,
    output_dir: Path | None = None,
    figures_dir: Path | None = None,
    no_figures: bool = False,
    dry_run: bool = False,
) -> ScheduleResult:
    """Plan every requested cell, measure it, validate it, and write the layer."""
    started = time.perf_counter()
    selected_configs = list(configs) if configs else list(CONFIG_GRID)

    recommendations = read_recommendations(recommendations_path)
    validate_recommendations(recommendations, source=recommendations_path.name)
    folds = read_folds(folds_path)
    medians = median_daily_by_fold(folds)
    validate_folds_against_recommendations(
        recommendations, medians, source=recommendations_path.name
    )
    override_ids = read_override_log(override_log_path)
    adjustments = read_adjustments(adjustments_path)
    events = read_execution_events(execution_path)

    calendars = observed_calendars(recommendations)
    cells = cell_frames(recommendations, policies=policies, k_names=k_names)
    policy_version = str(recommendations["policy_definition_version"][0])
    model_name = str(recommendations["model_name"][0])
    evaluation_version = str(folds["evaluation_definition_version"][0])

    adjustments_by_cell: dict[tuple[str, str, str, str], list[Adjustment]] = {}
    for adjustment in adjustments:
        key = (
            adjustment.schedule_config_id,
            adjustment.policy_id,
            adjustment.fold_id,
            adjustment.k_name,
        )
        adjustments_by_cell.setdefault(key, []).append(adjustment)
    events_by_cell: dict[tuple[str, str, str, str], list[ExecutionEvent]] = {}
    for event in events:
        key = (event.schedule_config_id, event.policy_id, event.fold_id, event.k_name)
        events_by_cell.setdefault(key, []).append(event)

    execution_sha = compute_sha256(execution_path) if execution_path else ""

    schedule_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    slot_rows: list[dict[str, object]] = []
    backlog_out: list[dict[str, object]] = []
    summary_out: list[dict[str, object]] = []
    utilization_out: list[dict[str, object]] = []
    preservation_out: list[dict[str, object]] = []
    adjustment_out: list[dict[str, object]] = []
    execution_out: list[dict[str, object]] = []
    execution_summary_out: list[dict[str, object]] = []
    runs_out: list[dict[str, object]] = []

    stats = ScheduleStats(
        recommendation_rows=recommendations.height,
        configs=len(selected_configs),
        model_name=model_name,
        policies=recommendations["policy_id"].n_unique(),
        folds=recommendations["fold_id"].n_unique(),
        fold_sets=sorted(recommendations["fold_set"].unique().to_list()),
    )
    seen_slots: set[tuple[str, str, str]] = set()

    for spec in selected_configs:
        for cell, frame in cells:
            queue = queue_rows(frame)
            if not queue:
                continue
            fold_id = cell["fold_id"]
            k = int(frame["k"][0])
            horizon = build_horizon(
                fold_set=cell["fold_set"],
                fold_id=fold_id,
                k_name=cell["k_name"],
                k=k,
                median_daily_capacity=medians[fold_id],
                calendar=calendars[fold_id],
                capacity_mode=spec.capacity_mode,
            )
            run_id = _planning_run_id(
                spec.schedule_config_id,
                cell["policy_id"],
                cell["fold_id"],
                cell["k_name"],
                str(k),
                "0",
            )
            plan = SchedulePlan(
                schedule_config_id=spec.schedule_config_id,
                policy_id=cell["policy_id"],
                model_name=cell["model_name"],
                fold_set=cell["fold_set"],
                fold_id=fold_id,
                k_name=cell["k_name"],
                k=k,
                horizon=horizon,
                placements=place(queue, horizon),
                planning_run_id=run_id,
            )

            baseline_rows.extend(
                _schedule_rows(plan, queue, cell, spec, override_ids, policy_version)
            )
            plans_for_cell = [plan]
            cell_key = (
                spec.schedule_config_id,
                cell["policy_id"],
                fold_id,
                cell["k_name"],
            )
            cell_identity = {
                "schedule_config_id": spec.schedule_config_id,
                "policy_id": cell["policy_id"],
                "fold_set": cell["fold_set"],
                "fold_id": fold_id,
                "k_name": cell["k_name"],
            }
            runs_out.append(
                {
                    **cell_identity,
                    **original_run_row(plan),
                    "execution_log_sha256": execution_sha,
                    "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
                }
            )

            n_adjusted = 0
            n_displaced = 0
            next_index = 1
            if cell_key in adjustments_by_cell:
                # An applied adjustment appends its own planning run rather than editing the
                # deterministic one. Both plans are written, so the plan a supervisor changed
                # stays readable beside the plan they changed it to -- which is what makes the
                # change auditable rather than merely recorded.
                outcomes, adjusted = apply_adjustments(
                    plan,
                    adjustments_by_cell[cell_key],
                    queue,
                )
                rows = adjustment_log_rows(outcomes)
                n_adjusted = sum(1 for r in rows if r["outcome"] == OUTCOME_APPLIED)
                n_displaced = sum(1 for r in rows if r["displaced_target_inspection_id"])
                stats.adjustments_applied += n_adjusted

                if n_adjusted:
                    adjusted_id = _planning_run_id(
                        spec.schedule_config_id,
                        cell["policy_id"],
                        cell["fold_id"],
                        cell["k_name"],
                        str(k),
                        str(next_index),
                    )
                    plan = SchedulePlan(
                        schedule_config_id=adjusted.schedule_config_id,
                        policy_id=adjusted.policy_id,
                        model_name=adjusted.model_name,
                        fold_set=adjusted.fold_set,
                        fold_id=adjusted.fold_id,
                        k_name=adjusted.k_name,
                        k=adjusted.k,
                        horizon=adjusted.horizon,
                        placements=adjusted.placements,
                        planning_run_id=adjusted_id,
                        replan_index=next_index,
                    )
                    schedule_rows.extend(
                        _schedule_rows(plan, queue, cell, spec, override_ids, policy_version)
                    )
                    plans_for_cell.append(plan)
                    runs_out.append(
                        {
                            **cell_identity,
                            "planning_run_id": adjusted_id,
                            "replan_index": next_index,
                            "parent_replan_index": next_index - 1,
                            "replan_from_date": None,
                            "trigger": "scheduling_adjustment",
                            "n_preserved_completed": 0,
                            "n_preserved_past": 0,
                            "n_returned_to_queue": 0,
                            "n_cancelled": plan.n_cancelled,
                            "n_newly_scheduled": n_adjusted,
                            "n_still_backlog": plan.n_backlog,
                            "remaining_slots": plan.idle_slots,
                            "execution_log_sha256": execution_sha,
                            "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
                        }
                    )
                    next_index += 1

                for entry in rows:
                    entry["planning_run_id"] = plan.planning_run_id
                    entry["replan_index"] = plan.replan_index
                    entry["schedule_definition_version"] = SCHEDULE_DEFINITION_VERSION
                adjustment_out.extend(rows)

            statuses: dict[str, str] = {}
            cell_events = events_by_cell.get(cell_key, [])
            if cell_events:
                event_outcomes, statuses = record_execution(plan, cell_events)
                event_rows = execution_log_rows(event_outcomes)
                for entry in event_rows:
                    entry["schedule_definition_version"] = SCHEDULE_DEFINITION_VERSION
                execution_out.extend(event_rows)
                stats.execution_events += len(event_rows)

                observed = {e.target_inspection_id: observed_date(e) for e in cell_events}
                boundary = replan_point(plan, statuses, observed)
                if boundary is not None:
                    next_id = _planning_run_id(
                        spec.schedule_config_id,
                        cell["policy_id"],
                        cell["fold_id"],
                        cell["k_name"],
                        str(k),
                        str(next_index),
                    )
                    plan, run = replan(
                        plan,
                        queue,
                        statuses,
                        from_date=boundary,
                        planning_run_id=next_id,
                        replan_index=next_index,
                    )
                    runs_out.append(
                        {
                            **cell_identity,
                            **run,
                            "execution_log_sha256": execution_sha,
                            "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
                        }
                    )
                    stats.replanning_runs += 1
                    schedule_rows.extend(
                        _schedule_rows(plan, queue, cell, spec, override_ids, policy_version)
                    )
                    plans_for_cell.append(plan)
                    next_index += 1

            execution_summary_out.append(
                {
                    "schedule_config_id": spec.schedule_config_id,
                    "policy_id": cell["policy_id"],
                    "model_name": cell["model_name"],
                    "fold_set": cell["fold_set"],
                    "fold_id": fold_id,
                    "k_name": cell["k_name"],
                    **execution_summary_row(plan, statuses),
                    "final_replan_index": plan.replan_index,
                    "execution_log_sha256": execution_sha,
                    "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
                }
            )

            slot_key = (spec.schedule_config_id, fold_id, cell["k_name"])
            if slot_key not in seen_slots:
                seen_slots.add(slot_key)
                cumulative = horizon.cumulative_slots
                for day, running in zip(horizon.days, cumulative, strict=True):
                    slot_rows.append(
                        {
                            "schedule_config_id": spec.schedule_config_id,
                            "fold_set": cell["fold_set"],
                            "fold_id": fold_id,
                            "k_name": cell["k_name"],
                            "k": k,
                            "median_daily_capacity": horizon.median_daily_capacity,
                            "horizon_days": horizon.n_days,
                            "day_index": day.day_index,
                            "slot_date": day.slot_date,
                            "n_slots": day.n_slots,
                            "capacity_source": day.capacity_source,
                            "cumulative_slots": running,
                            "is_scenario": spec.is_scenario,
                            "horizon_was_clamped": horizon.was_clamped,
                            "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
                        }
                    )

            identity = {
                "schedule_config_id": spec.schedule_config_id,
                "policy_id": cell["policy_id"],
                "model_name": cell["model_name"],
                "fold_set": cell["fold_set"],
                "fold_id": fold_id,
                "k_name": cell["k_name"],
            }
            # One backlog block per planning run. A re-plan appends a plan rather than
            # editing one, so each plan has its own backlog and comparing the two is how a
            # reader sees what a not-performed day actually cost.
            for emitted in plans_for_cell:
                for entry in backlog_rows(emitted, queue, calendars[fold_id]):
                    backlog_out.append(
                        {
                            **identity,
                            "k": k,
                            **entry,
                            "planning_run_id": emitted.planning_run_id,
                            "replan_index": emitted.replan_index,
                            "is_scenario": spec.is_scenario,
                            "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
                        }
                    )
            summary_out.append(
                {
                    **identity,
                    "k": k,
                    **summary_row(plan),
                    "n_adjustments_applied": n_adjusted,
                    "n_execution_events": len(cell_events),
                    "planning_run_id": plan.planning_run_id,
                    "replan_index": plan.replan_index,
                    "is_scenario": spec.is_scenario,
                    "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
                }
            )
            for entry in capacity_utilization_rows(plan, queue):
                utilization_out.append(
                    {
                        **identity,
                        **entry,
                        "is_scenario": spec.is_scenario,
                        "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
                    }
                )
            measured = preservation_row(plan, queue)
            counts = {
                str(ExecutionStatus.COMPLETED): 0,
                str(ExecutionStatus.NOT_PERFORMED): 0,
                str(ExecutionStatus.CANCELLED_IN_FIELD): 0,
            }
            for status in statuses.values():
                counts[status] = counts.get(status, 0) + 1
            preservation_out.append(
                {
                    **identity,
                    "k": k,
                    **measured,
                    "n_adjusted": n_adjusted,
                    "n_displaced_by_adjustment": n_displaced,
                    "n_execution_completed": counts[str(ExecutionStatus.COMPLETED)],
                    "n_execution_not_performed": counts[str(ExecutionStatus.NOT_PERFORMED)],
                    "n_execution_cancelled": counts[str(ExecutionStatus.CANCELLED_IN_FIELD)],
                    "n_no_execution_record": plan.n_scheduled - len(statuses),
                    "is_scenario": spec.is_scenario,
                    "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
                }
            )

            stats.cells += 1
            stats.queue_rows += len(queue)
            stats.scheduled_rows += plan.n_scheduled
            stats.backlog_rows += plan.n_backlog
            stats.idle_slots += plan.idle_slots
            stats.cells_with_backlog += 1 if plan.n_backlog else 0
            stats.cells_with_idle += 1 if plan.idle_slots else 0
            stats.reserve_recommended += _count(measured, "n_reserve_recommended")
            stats.reserve_scheduled += _count(measured, "n_reserve_scheduled")
            stats.reserve_slots_lost += _count(measured, "reserve_slots_lost")
            stats.inversions += _count(measured, "n_inversions")

    schedule_all = baseline_rows + schedule_rows
    tables: dict[str, pl.DataFrame] = {
        "inspection_schedule": finalize(schedule_all, "inspection_schedule"),
        "schedule_backlog": finalize(backlog_out, "schedule_backlog"),
        "schedule_slots": finalize(slot_rows, "schedule_slots"),
        "schedule_summary": finalize(summary_out, "schedule_summary"),
        "capacity_utilization": finalize(utilization_out, "capacity_utilization"),
        "priority_preservation": finalize(preservation_out, "priority_preservation"),
        "schedule_configurations": finalize(
            _configuration_rows(selected_configs), "schedule_configurations"
        ),
        "schedule_adjustment_log": finalize(adjustment_out, "schedule_adjustment_log"),
        "execution_contract": finalize(_contract_rows(), "execution_contract"),
        "execution_log": finalize(execution_out, "execution_log"),
        "execution_summary": finalize(execution_summary_out, "execution_summary"),
        "replanning_runs": finalize(runs_out, "replanning_runs"),
        "schedule_advisories": empty("schedule_advisories"),
    }

    rebuilt = (
        finalize(baseline_rows, "inspection_schedule").filter(pl.col("replan_index") == 0)
        if (adjustments or events)
        else None
    )
    if rebuilt is not None:
        rebuilt = rebuilt.with_columns(pl.lit("").alias("adjustment_id"))

    before = {
        "recommendations": compute_sha256(recommendations_path),
        "evaluation_folds": compute_sha256(folds_path),
    }
    # Validated against the cells actually requested, not the whole artifact. A run
    # restricted with --policies or --k-names schedules a subset by design, and comparing it
    # against every cell Component 13 wrote would report a shortfall the user asked for.
    requested = recommendations
    if policies:
        requested = requested.filter(pl.col("policy_id").is_in(list(policies)))
    if k_names:
        requested = requested.filter(pl.col("k_name").is_in(list(k_names)))
    report = _validate(tables, requested, medians, selected_configs, override_ids, rebuilt)
    tables["schedule_advisories"] = finalize(
        checks.advisory_rows(report, SCHEDULE_DEFINITION_VERSION), "schedule_advisories"
    )
    stats.advisories = len(checks.advisory_findings(report))
    stats.seconds = time.perf_counter() - started

    result = ScheduleResult(tables=tables, checks=report, stats=stats)
    if dry_run or checks.has_failures(report):
        return result

    destination = output_dir or settings.scheduling_processed_dir
    destination.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    records: list[ArtifactRecord] = []
    for name in LAYERS:
        path = destination / f"{name}_{stamp}.parquet"
        write_table(tables[name], path)
        result.written.append(path)
        records.append(
            ArtifactRecord(
                path=str(path),
                bytes=path.stat().st_size,
                sha256=compute_sha256(path),
                row_count=tables[name].height,
                schema=schema_of(tables[name]),
            )
        )

    after = {
        "recommendations": compute_sha256(recommendations_path),
        "evaluation_folds": compute_sha256(folds_path),
    }
    unchanged = checks.inputs_were_not_modified(before, after)
    report.append(unchanged)
    stats.inputs_unchanged = unchanged.passed

    # The reserve figures are reported on the observed calendar alone. The flat_median scenario
    # supplies exactly k slots for a queue of k, so it loses nothing by construction; pooling
    # the two would divide the finding by the number of modes that happened to run.
    observed_preservation = tables["priority_preservation"].filter(~pl.col("is_scenario"))
    bearing = observed_preservation.filter(pl.col("n_reserve_recommended") > 0)
    result.manifest = ScheduleManifest(
        code_version=code_version,
        schedule_definition_version=SCHEDULE_DEFINITION_VERSION,
        built_at=datetime.now(UTC).isoformat(),
        recommendations_path=str(recommendations_path),
        recommendations_sha256=before["recommendations"],
        policy_definition_version=policy_version,
        selection_allocation_path="",
        selection_allocation_sha256="",
        policy_comparison_path="",
        policy_comparison_sha256="",
        evaluation_folds_path=str(folds_path),
        evaluation_folds_sha256=before["evaluation_folds"],
        evaluation_definition_version=evaluation_version,
        override_log_path=str(override_log_path) if override_log_path else None,
        override_log_sha256=compute_sha256(override_log_path) if override_log_path else None,
        adjustments_path=str(adjustments_path) if adjustments_path else None,
        adjustments_sha256=compute_sha256(adjustments_path) if adjustments_path else None,
        execution_path=str(execution_path) if execution_path else None,
        execution_sha256=execution_sha or None,
        inputs_unchanged=unchanged.passed,
        input_sha256_after=after,
        layer_separation=LAYER_SEPARATION,
        scheduling_semantics=SCHEDULING_SEMANTICS,
        capacity_is_inherited=CAPACITY_IS_INHERITED,
        priority_is_inherited=PRIORITY_IS_INHERITED,
        calendar_is_observed=CALENDAR_IS_OBSERVED,
        horizon_rule=HORIZON_RULE,
        allocation_claim=ALLOCATION_CLAIM,
        no_solver=NO_SOLVER,
        no_constraint_aware_strategy=NO_CONSTRAINT_AWARE_STRATEGY,
        capacity_mode_scenario_claim=CAPACITY_MODE_SCENARIO_CLAIM,
        default_capacity_mode=str(DEFAULT_CAPACITY_MODE),
        capacity_modes=[str(m) for m in CapacityMode],
        strategies=sorted(STRATEGY_BY_ID),
        config_grid=_configuration_rows(selected_configs),
        k_levels=list(K_LEVELS),
        primary_k_level=PRIMARY_K_LEVEL,
        model_name=model_name,
        policies=sorted(tables["inspection_schedule"]["policy_id"].unique().to_list()),
        three_human_layers=THREE_HUMAN_LAYERS,
        adjustment_cannot=ADJUSTMENT_CANNOT,
        execution_cannot=EXECUTION_CANNOT,
        replan_preserves=REPLAN_PRESERVES,
        replan_backfill_rule=REPLAN_BACKFILL_RULE,
        temporal_boundary=TEMPORAL_BOUNDARY,
        adjustments_applied=stats.adjustments_applied,
        execution_events=stats.execution_events,
        replanning_runs=tables["replanning_runs"].height,
        schedule_claim=SCHEDULE_CLAIM,
        green_run_means=GREEN_RUN_MEANS,
        determinism_scope=DETERMINISM_SCOPE,
        inversion_reason_required=INVERSION_REASON_REQUIRED,
        scheduled_rows=stats.scheduled_rows,
        backlog_rows=stats.backlog_rows,
        idle_slots=stats.idle_slots,
        cells_with_backlog=stats.cells_with_backlog,
        cells_with_idle=stats.cells_with_idle,
        total_cells=stats.cells,
        reserve_recommended=int(bearing["n_reserve_recommended"].sum() or 0),
        reserve_scheduled=int(bearing["n_reserve_scheduled"].sum() or 0),
        reserve_slots_lost=int(bearing["reserve_slots_lost"].sum() or 0),
        cells_losing_reserve=int(bearing.filter(pl.col("reserve_slots_lost") > 0).height),
        cells_losing_all_reserve=int(bearing.filter(pl.col("n_reserve_scheduled") == 0).height),
        inversions=stats.inversions,
        does_not_establish=list(DOES_NOT_ESTABLISH),
        blocked=list(BLOCKED),
        inherited_limitations=list(INHERITED_LIMITATIONS),
        checks=checks.check_rows(report),
        advisories=checks.advisory_findings(report),
        artifacts=records,
        row_counts={name: tables[name].height for name in LAYERS},
        seconds=stats.seconds,
    )
    manifest_target = manifest_path_for(destination / f"{DATASET_SLUG}_{stamp}.parquet")
    write_manifest(result.manifest, manifest_target)
    result.written.append(manifest_target)

    if not no_figures:
        target = figures_dir or (Path("docs") / "analysis" / "figures")
        result.written.extend(figure_module.render(tables, destination=target))
    return result


def _validate(
    tables: dict[str, pl.DataFrame],
    recommendations: pl.DataFrame,
    medians: dict[str, int],
    configs: Sequence[ScheduleConfigSpec],
    override_ids: dict[str, str],
    rebuilt: pl.DataFrame | None,
) -> list[ValidationCheck]:
    schedule = tables["inspection_schedule"]
    baseline = schedule.filter(pl.col("replan_index") == 0)
    return [
        checks.tables_are_deterministically_sorted(tables),
        checks.schedule_rows_originate_in_the_recommendation_universe(schedule, recommendations),
        checks.every_selected_recommendation_is_accounted_for(
            baseline, recommendations, len(configs)
        ),
        checks.no_slot_is_double_booked(schedule),
        checks.no_inspection_occupies_two_slots(baseline),
        checks.no_day_exceeds_its_capacity(
            baseline, tables["schedule_slots"], tables["capacity_utilization"]
        ),
        checks.capacity_matches_the_declared_mode(
            tables["schedule_slots"], recommendations, medians
        ),
        checks.horizon_is_ordered_contiguous_and_real(
            tables["schedule_slots"], recommendations, medians
        ),
        checks.schedule_ranks_are_unique_and_contiguous(baseline),
        checks.schedule_order_follows_policy_rank(baseline, tables["priority_preservation"]),
        checks.no_inversion_without_a_reason_code(baseline),
        checks.every_row_declares_a_valid_status(schedule),
        checks.backlog_is_exactly_the_unscheduled_remainder(schedule, tables["schedule_backlog"]),
        checks.counts_add_up(tables["schedule_summary"]),
        checks.capacity_never_exceeds_the_recommendation_k(
            tables["schedule_summary"], recommendations
        ),
        checks.c13_provenance_is_preserved(schedule, recommendations),
        checks.no_outcome_column_reaches_the_schedule(tables),
        checks.completed_rows_are_never_rescheduled(schedule, tables["execution_log"]),
        checks.execution_never_alters_a_recommendation(schedule, recommendations),
        checks.no_execution_event_changes_an_earlier_schedule(
            tables["replanning_runs"], schedule, tables["execution_log"]
        ),
        checks.planning_runs_are_unique_and_chained(tables["replanning_runs"]),
        checks.adjustments_are_not_overrides(
            tables["schedule_adjustment_log"], list(override_ids.values())
        ),
        checks.adjustments_preserve_the_original_assignment(
            schedule, tables["schedule_adjustment_log"]
        ),
        checks.adjustments_never_displace_a_coverage_reserve_row(
            tables["schedule_adjustment_log"], schedule
        ),
        checks.external_changes_are_fully_attributed(
            tables["schedule_adjustment_log"], tables["execution_log"]
        ),
        checks.the_deterministic_plan_is_intact(schedule, rebuilt),
        checks.configurations_match_the_frozen_grid(
            tables["schedule_configurations"], [c.schedule_config_id for c in configs]
        ),
        checks.capacity_is_fully_utilized(tables["schedule_summary"]),
        checks.every_recommendation_was_scheduled(tables["schedule_summary"]),
        checks.the_coverage_reserve_survived_scheduling(tables["priority_preservation"]),
        checks.the_scenario_is_not_observed_fact(schedule),
        checks.an_execution_record_was_supplied(tables["execution_log"]),
        checks.the_horizon_opens_on_a_full_day(tables["schedule_slots"]),
        checks.an_establishment_recurs_within_a_horizon(baseline),
    ]


def summarize(result: ScheduleResult) -> str:
    """The CLI summary. Leads with the headline, and labels the scenario.

    The reserve line is measured on the observed calendar alone. The flat_median scenario loses
    no reserve slots by construction -- it supplies exactly k slots for a queue of k -- so
    pooling the two modes would average a tautology into the component's headline and halve it.
    """
    s = result.stats
    preservation = result.tables["priority_preservation"]
    observed = preservation.filter(~pl.col("is_scenario"))
    offered = int(observed["n_reserve_recommended"].sum() or 0)
    lost = int(observed["reserve_slots_lost"].sum() or 0)
    reserve_share = (lost / offered) if offered else 0.0
    summary = result.tables["schedule_summary"]
    obs_summary = summary.filter(~pl.col("is_scenario"))
    obs_backlog = int(obs_summary["n_backlog"].sum() or 0)
    obs_cells = int(obs_summary.filter(pl.col("n_backlog") > 0).height)
    lines = [
        f"  cells                {s.cells} ({s.configs} configuration(s), {s.policies} "
        f"policies, {s.folds} folds)",
        f"  model                {s.model_name}",
        f"  queue rows           {s.queue_rows}",
        f"  scheduled            {s.scheduled_rows}",
        f"  backlog              {s.backlog_rows} in {s.cells_with_backlog} cell(s) "
        f"({obs_backlog} in {obs_cells} on the observed calendar; the scenario fits by "
        "construction)",
        f"  idle slots           {s.idle_slots} in {s.cells_with_idle} cell(s)",
        f"  inversions           {s.inversions}",
        "",
        f"  coverage reserve     {lost} of {offered} slot(s) lost to the horizon "
        f"({reserve_share:.3f}) -- observed calendar only",
        "",
        f"  adjustments applied  {s.adjustments_applied}",
        f"  execution events     {s.execution_events}",
        f"  planning runs        {s.replanning_runs}",
        f"  advisories           {s.advisories}",
        f"  seconds              {s.seconds:.1f}",
    ]
    return "\n".join(lines)


__all__ = ["ScheduleResult", "run_schedule", "summarize"]
