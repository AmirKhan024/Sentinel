"""The execution layer: what a person reports actually happened. External, and not computation.

**Real-world execution is not deterministic, and this module never pretends otherwise.** The
reproducibility claim it participates in is scoped precisely: *identical recommendation
artifact + identical scheduling configuration + identical execution log produces byte-identical
output*. The manifest pins the execution file by checksum rather than claiming that what
inspectors did last Tuesday is reproducible computation. Overstating that would be the easiest
lie in this component, so ``DETERMINISM_SCOPE`` states it in words and the validator states it
in a check.

**An execution event records and never edits.** It cannot touch a score, a rank, a decision
mechanism, a decision reason, an original assignment, or the plan at replan index 0. That is
enforced structurally rather than by convention: ``inspection_schedule`` has no
``execution_status`` column at all. A consumer who wants both joins the two tables on the key,
and the join is the moment where a reader can see that they are two different facts.

**The temporal boundary is arithmetic, not an intention.** An event observed on day *D* may
only affect operating days strictly after *D*. Nothing in this module reaches backwards, and
``validate.no_execution_event_changes_an_earlier_schedule`` compares plan *n* against plan
*n+1* row by row to prove it.

**"We do not know" is a category.** A scheduled row that no event mentions is counted as
``no_execution_record`` rather than folded into "not completed". Silently treating an absent
report as a failure would manufacture a completion rate out of missing data.

**Nothing here is ever generated.** The engine never infers an event, never fills one in for a
row with no report, and never writes a file. Every row in the execution log came from a person.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime

from sentinel.scheduling.definitions import (
    EXECUTION_REQUIRED_FIELDS,
    NO_EXECUTION_RECORD,
    REPLAN_TRIGGERING_STATUSES,
    ExecutionStatus,
)
from sentinel.scheduling.models import ExecutionEvent, ExecutionOutcome, SchedulePlan

OUTCOME_RECORDED = "recorded"
OUTCOME_NO_OP_NOT_SCHEDULED = "no_op_not_scheduled"
OUTCOME_ROW_NOT_IN_PLAN = "row_not_in_plan"


class ExecutionError(ValueError):
    """Raised when an execution file cannot be trusted enough to record any of it."""


def parse_execution_events(payload: Sequence[Mapping[str, object]]) -> list[ExecutionEvent]:
    """Decode and validate a whole execution file, or refuse all of it.

    The same all-or-nothing rule as the adjustment and override contracts, for a slightly
    different reason: a half-read execution log produces a completion rate computed over an
    arbitrary subset, and a rate over an unknown denominator is worse than no rate at all.
    """
    parsed: list[ExecutionEvent] = []
    seen: set[str] = set()
    statuses = {str(status) for status in ExecutionStatus}

    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ExecutionError(
                f"execution event {index}: expected an object, got {type(raw).__name__}"
            )
        missing = [
            field for field in EXECUTION_REQUIRED_FIELDS if not str(raw.get(field, "")).strip()
        ]
        if missing:
            raise ExecutionError(
                f"execution event {index}: missing or blank {', '.join(missing)}. Every field "
                "is required and the whole file is refused rather than partly recorded -- a "
                "completion rate over an arbitrary subset is worse than none"
            )
        status = str(raw["execution_status"])
        # The derived category is checked first, so somebody who tries to file one gets the
        # reason rather than a generic "unknown status" that does not explain why it is absent
        # from a list they can see in the contract table.
        if status == NO_EXECUTION_RECORD:
            raise ExecutionError(
                f"execution event {index}: {NO_EXECUTION_RECORD!r} is a derived summary "
                "category for rows nobody reported on, not a status a person can supply"
            )
        if status not in statuses:
            raise ExecutionError(
                f"execution event {index}: unknown execution_status {status!r}. Known: "
                f"{', '.join(sorted(statuses))}"
            )
        try:
            event = ExecutionEvent(**{k: str(v) for k, v in raw.items()})
        except Exception as exc:  # pragma: no cover - pydantic message passthrough
            raise ExecutionError(f"execution event {index}: {exc}") from exc
        if event.execution_id in seen:
            raise ExecutionError(
                f"duplicate execution_id {event.execution_id!r}. Events are recorded in id "
                "order, so a repeated id makes the order -- and the result -- ambiguous"
            )
        seen.add(event.execution_id)
        parsed.append(event)

    return sorted(parsed, key=lambda e: e.execution_id)


def observed_date(event: ExecutionEvent) -> date:
    """The operating day an event was observed on, from its timestamp.

    The date rather than the instant, because the dataset has no clock: every capacity in this
    project is a count per day, so a boundary finer than a day would be a precision the data
    cannot support.
    """
    text = event.observed_at.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).date()
    except ValueError as exc:
        raise ExecutionError(
            f"{event.execution_id}: observed_at {event.observed_at!r} is not an ISO timestamp"
        ) from exc


def triggers_replan(status: str) -> bool:
    """Whether a status returns its row to the queue.

    Only ``not_performed``. A completed inspection needs no re-planning, and a field
    cancellation is a removal -- and removals are not backfilled, on the same rule that stops
    Component 13 backfilling a ``force_exclude``.
    """
    return status in REPLAN_TRIGGERING_STATUSES


def record_execution(
    plan: SchedulePlan, events: Sequence[ExecutionEvent]
) -> tuple[list[ExecutionOutcome], dict[str, str]]:
    """Record a set of events against a plan. Returns the log and a status per row.

    The plan is not touched. What comes back is a mapping the re-planner consumes, and a log
    the writer emits; the plan itself is written unchanged, exactly as Component 13 writes its
    queue unchanged beside its override log.
    """
    placements = plan.by_id()
    outcomes: list[ExecutionOutcome] = []
    statuses: dict[str, str] = {}

    for event in events:
        placement = placements.get(event.target_inspection_id)
        if placement is None:
            outcomes.append(ExecutionOutcome(event=event, outcome=OUTCOME_ROW_NOT_IN_PLAN))
            continue
        if not placement.occupies_a_slot:
            outcomes.append(
                ExecutionOutcome(
                    event=event,
                    outcome=OUTCOME_NO_OP_NOT_SCHEDULED,
                    plan_slot_date=placement.slot_date,
                )
            )
            continue
        statuses[event.target_inspection_id] = event.execution_status
        outcomes.append(
            ExecutionOutcome(
                event=event,
                outcome=OUTCOME_RECORDED,
                plan_slot_date=placement.slot_date,
                triggers_replan=triggers_replan(event.execution_status),
                applied_at_replan_index=plan.replan_index,
            )
        )
    return outcomes, statuses


def execution_log_rows(outcomes: Sequence[ExecutionOutcome]) -> list[dict[str, object]]:
    """The execution log, one row per event offered.

    ``scheduled_date`` and ``plan_scheduled_date`` are both written and never merged. A field
    log that disagrees with the plan is a fact about operations; overwriting either value would
    destroy the only evidence that it happened, so the disagreement is an advisory rather than
    a correction.
    """
    return [
        {
            "execution_id": outcome.event.execution_id,
            "schedule_config_id": outcome.event.schedule_config_id,
            "policy_id": outcome.event.policy_id,
            "fold_id": outcome.event.fold_id,
            "k_name": outcome.event.k_name,
            "target_inspection_id": outcome.event.target_inspection_id,
            "scheduled_date": _maybe_date(outcome.event.scheduled_date),
            "plan_scheduled_date": outcome.plan_slot_date,
            "execution_status": outcome.event.execution_status,
            "reason_code": outcome.event.reason_code,
            "actor": outcome.event.actor,
            "observed_at": outcome.event.observed_at,
            "outcome": outcome.outcome,
            "triggers_replan": outcome.triggers_replan,
            "applied_at_replan_index": outcome.applied_at_replan_index,
        }
        for outcome in outcomes
    ]


def _maybe_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def execution_summary_row(plan: SchedulePlan, statuses: Mapping[str, str]) -> dict[str, object]:
    """Counts per cell, with "nobody reported" as its own visible category."""
    scheduled = [p for p in plan.placements if p.occupies_a_slot]
    counts = {
        str(ExecutionStatus.COMPLETED): 0,
        str(ExecutionStatus.NOT_PERFORMED): 0,
        str(ExecutionStatus.CANCELLED_IN_FIELD): 0,
        NO_EXECUTION_RECORD: 0,
    }
    for placement in scheduled:
        status = statuses.get(placement.target_inspection_id, NO_EXECUTION_RECORD)
        counts[status] = counts.get(status, 0) + 1

    n_scheduled = len(scheduled)
    completed = counts[str(ExecutionStatus.COMPLETED)]
    return {
        "n_scheduled": n_scheduled,
        "n_completed": completed,
        "n_not_performed": counts[str(ExecutionStatus.NOT_PERFORMED)],
        "n_cancelled_in_field": counts[str(ExecutionStatus.CANCELLED_IN_FIELD)],
        "n_no_execution_record": counts[NO_EXECUTION_RECORD],
        "completion_rate": (completed / n_scheduled) if n_scheduled else None,
    }


__all__ = [
    "OUTCOME_NO_OP_NOT_SCHEDULED",
    "OUTCOME_RECORDED",
    "OUTCOME_ROW_NOT_IN_PLAN",
    "ExecutionError",
    "execution_log_rows",
    "execution_summary_row",
    "observed_date",
    "parse_execution_events",
    "record_execution",
    "triggers_replan",
]
