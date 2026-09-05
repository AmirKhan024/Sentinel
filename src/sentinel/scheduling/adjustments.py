"""The scheduling adjustment layer: a human changing *when* an approved row is worked.

Shaped as a near-mirror of ``policy/governance.py`` so that a reader who knows Component 13's
override contract knows this one, and so that the *differences* are the visible part.

**This is not an override, and the two must never be confusable.** An override changes who is
in the approved queue; an adjustment changes when an approved row is scheduled. They have
different id namespaces, disjoint verb vocabularies -- enforced at import time by
``definitions._guard_registry`` -- and different tables. Merging them would collapse the chain
*original recommendation -> approved recommendation -> planned schedule -> scheduling change ->
execution outcome* into a single ambiguous "somebody changed something".

**All or nothing.** Every field is required and a blank one refuses the whole file, because a
partially applied adjustment file produces a schedule nobody authorised. That is Component 13's
rule verbatim, and it matters more here: an adjustment names a date, and a file half-applied
leaves some rows on days a supervisor chose and others on days nobody did.

**Applied in ``adjustment_id`` order, never file order**, so re-serialising the JSON cannot
change the schedule.

**Capacity is fixed, so a move costs a displacement, and the displaced row is named.** The row
that gives up its slot is always the lowest-ranked ``risk_priority`` row on the target day --
never a ``coverage_reserve`` row, for the reason ``policy/governance.py`` gives about override
inclusions: taking the slot from the coverage allocation would quietly convert every scheduling
adjustment into a coverage cut, which is a policy change nobody made.

**The deterministic plan is written unchanged.** This module returns a second plan that sits
beside it, and ``validate.the_deterministic_plan_is_intact`` proves the first was not touched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

from sentinel.policy.definitions import DecisionMechanism
from sentinel.scheduling.definitions import (
    ADJUSTMENT_REQUIRED_FIELDS,
    AdjustmentAction,
    InversionReason,
    ScheduleReason,
    ScheduleStatus,
)
from sentinel.scheduling.models import (
    Adjustment,
    AdjustmentOutcome,
    Horizon,
    Placement,
    QueueRow,
    SchedulePlan,
)

OUTCOME_APPLIED = "applied"
OUTCOME_NO_OP_ALREADY_ON_DATE = "no_op_already_on_date"
OUTCOME_NO_OP_NOT_SCHEDULED = "no_op_not_scheduled"
OUTCOME_ROW_NOT_IN_PLAN = "row_not_in_plan"


class AdjustmentError(ValueError):
    """Raised when an adjustment file cannot be trusted enough to apply any of it."""


def parse_adjustments(payload: Sequence[Mapping[str, object]]) -> list[Adjustment]:
    """Decode and validate a whole adjustment file, or refuse all of it.

    The refusal is total on purpose. A file with one malformed row is a file somebody was in
    the middle of editing, and applying the well-formed half produces a schedule that no
    supervisor reviewed and nobody can reconstruct.
    """
    parsed: list[Adjustment] = []
    seen: set[str] = set()
    actions = {str(action) for action in AdjustmentAction}

    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise AdjustmentError(
                f"adjustment {index}: expected an object, got {type(raw).__name__}"
            )
        missing = [
            field
            for field in ADJUSTMENT_REQUIRED_FIELDS
            if field != "target_date" and not str(raw.get(field, "")).strip()
        ]
        if missing:
            raise AdjustmentError(
                f"adjustment {index}: missing or blank {', '.join(missing)}. Every field is "
                "required, and the whole file is refused rather than partly applied -- a "
                "half-applied adjustment file produces a schedule nobody authorised"
            )
        if "target_date" not in raw:
            raise AdjustmentError(
                f"adjustment {index}: target_date is required. It is empty for a cancel and a "
                "date for a move, but it is never absent"
            )
        action = str(raw["action"])
        if action not in actions:
            raise AdjustmentError(
                f"adjustment {index}: unknown action {action!r}. Known: "
                f"{', '.join(sorted(actions))}"
            )
        target = str(raw["target_date"]).strip()
        if action == AdjustmentAction.CANCEL and target:
            raise AdjustmentError(
                f"adjustment {index}: a cancel carries no target_date, and one that does is "
                "ambiguous between striking the row and moving it"
            )
        if action != AdjustmentAction.CANCEL and not target:
            raise AdjustmentError(f"adjustment {index}: {action} requires a target_date")

        try:
            adjustment = Adjustment(**{k: str(v) for k, v in raw.items()})
        except Exception as exc:  # pragma: no cover - pydantic message passthrough
            raise AdjustmentError(f"adjustment {index}: {exc}") from exc
        if adjustment.adjustment_id in seen:
            raise AdjustmentError(
                f"duplicate adjustment_id {adjustment.adjustment_id!r}. Adjustments are applied "
                "in id order, so a repeated id makes the order -- and the result -- ambiguous"
            )
        seen.add(adjustment.adjustment_id)
        parsed.append(adjustment)

    return sorted(parsed, key=lambda a: a.adjustment_id)


def _parse_date(value: str, adjustment_id: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise AdjustmentError(f"{adjustment_id}: {value!r} is not an ISO date") from exc


def _lowest_standing_risk(
    placements: Mapping[str, Placement],
    queue: Mapping[str, QueueRow],
    *,
    on_day: date,
    protected: str,
) -> str | None:
    """The displaceable row on a day: lowest-ranked ``risk_priority``, never the reserve.

    Returning None means the day holds nothing that may be displaced, and the caller refuses
    the run rather than taking a reserve slot. That refusal is the point: a scheduler that fell
    back to the coverage allocation when it ran out of risk rows would convert an adjustment
    into a coverage cut silently, and the coverage allocation is the one thing in this pipeline
    that was priced in forgone citations.
    """
    candidates = [
        (queue[pid].final_policy_rank, pid)
        for pid, placement in placements.items()
        if placement.occupies_a_slot
        and placement.slot_date == on_day
        and pid != protected
        and queue[pid].decision_mechanism == DecisionMechanism.RISK_PRIORITY
    ]
    if not candidates:
        return None
    return max(candidates)[1]


def _free_slot_on(
    placements: Mapping[str, Placement], horizon: Horizon, *, day: date
) -> int | None:
    """The next unused slot index on a day, or None when it is full."""
    used = {
        placement.slot_index
        for placement in placements.values()
        if placement.occupies_a_slot and placement.slot_date == day
    }
    for candidate in horizon.days:
        if candidate.slot_date != day:
            continue
        for index in range(1, candidate.n_slots + 1):
            if index not in used:
                return index
        return None
    return None


def apply_adjustments(
    plan: SchedulePlan,
    adjustments: Sequence[Adjustment],
    queue: Sequence[QueueRow],
) -> tuple[list[AdjustmentOutcome], SchedulePlan]:
    """Apply an adjustment file to a plan, returning the log and a **new** plan.

    The input plan is never mutated. Every outcome is logged whether or not it changed
    anything, because "the supervisor asked and it made no difference" is an audit fact, and a
    log that recorded only effective adjustments would make a no-op indistinguishable from a
    request that was never made.
    """
    by_id = {row.target_inspection_id: row for row in queue}
    current = dict(plan.by_id())
    day_index = {day.slot_date: day.day_index for day in plan.horizon.days}
    outcomes: list[AdjustmentOutcome] = []

    for adjustment in adjustments:
        pid = adjustment.target_inspection_id
        placement = current.get(pid)
        if placement is None:
            outcomes.append(
                AdjustmentOutcome(adjustment=adjustment, outcome=OUTCOME_ROW_NOT_IN_PLAN)
            )
            continue

        if adjustment.action == AdjustmentAction.CANCEL:
            outcomes.append(
                AdjustmentOutcome(
                    adjustment=adjustment,
                    outcome=OUTCOME_APPLIED,
                    original_status=placement.status,
                    original_slot_date=placement.slot_date,
                    original_schedule_rank=placement.schedule_rank,
                    final_status=ScheduleStatus.CANCELLED,
                )
            )
            current[pid] = Placement(
                target_inspection_id=pid,
                status=ScheduleStatus.CANCELLED,
                reason=ScheduleReason.CANCELLED_BY_ADJUSTMENT,
                inversion_reason=placement.inversion_reason,
                original_slot_date=placement.original_slot_date,
                original_schedule_rank=placement.original_schedule_rank,
                adjustment_id=adjustment.adjustment_id,
                replan_index=placement.replan_index,
            )
            continue

        if not placement.occupies_a_slot:
            outcomes.append(
                AdjustmentOutcome(
                    adjustment=adjustment,
                    outcome=OUTCOME_NO_OP_NOT_SCHEDULED,
                    original_status=placement.status,
                )
            )
            continue

        target = _parse_date(adjustment.target_date, adjustment.adjustment_id)
        if target not in day_index:
            raise AdjustmentError(
                f"{adjustment.adjustment_id}: {target} is not an operating day in this cell's "
                "horizon. Moving a row outside the horizon would extend it, which is a "
                "capacity increase by another name"
            )
        if target == placement.slot_date:
            outcomes.append(
                AdjustmentOutcome(
                    adjustment=adjustment,
                    outcome=OUTCOME_NO_OP_ALREADY_ON_DATE,
                    original_status=placement.status,
                    original_slot_date=placement.slot_date,
                    original_schedule_rank=placement.schedule_rank,
                    final_status=placement.status,
                    final_slot_date=placement.slot_date,
                )
            )
            continue
        if adjustment.action == AdjustmentAction.DEFER_TO_DATE and target < (
            placement.slot_date or target
        ):
            raise AdjustmentError(
                f"{adjustment.adjustment_id}: defer_to_date moves a row later, and {target} is "
                f"before {placement.slot_date}. Use advance_to_date, so the log says which the "
                "supervisor meant"
            )
        if adjustment.action == AdjustmentAction.ADVANCE_TO_DATE and target > (
            placement.slot_date or target
        ):
            raise AdjustmentError(
                f"{adjustment.adjustment_id}: advance_to_date moves a row earlier, and {target} "
                f"is after {placement.slot_date}. Use defer_to_date"
            )

        vacated = placement.slot_date
        displaced_id = ""
        displaced_status = ""

        free = _free_slot_on(current, plan.horizon, day=target)
        if free is None:
            displaced_id = _lowest_standing_risk(current, by_id, on_day=target, protected=pid) or ""
            if not displaced_id:
                raise AdjustmentError(
                    f"{adjustment.adjustment_id}: {target} is full and holds no risk-priority "
                    "row that may be displaced. The coverage reserve is an allocation with a "
                    "measured price, not slack for a scheduling change to spend"
                )
            displaced = current[displaced_id]
            free = displaced.slot_index
            # The displaced row takes the slot the mover just vacated. A swap rather than a
            # search: capacity is fixed, so the only slot guaranteed to be free is the one that
            # has just been given up, and re-searching the horizon could displace a third row
            # and turn one supervisor's request into a cascade nobody asked for.
            landing = placement.slot_index
            landing_day = vacated
            if landing is not None and landing_day is not None:
                displaced_status = ScheduleStatus.DEFERRED
                current[displaced_id] = Placement(
                    target_inspection_id=displaced_id,
                    status=ScheduleStatus.DEFERRED,
                    reason=ScheduleReason.DISPLACED_BY_ADJUSTMENT,
                    inversion_reason=InversionReason.DISPLACED_BY_ADJUSTMENT,
                    slot_date=landing_day,
                    day_index=day_index[landing_day],
                    slot_index=landing,
                    schedule_rank=displaced.schedule_rank,
                    original_slot_date=displaced.original_slot_date,
                    original_schedule_rank=displaced.original_schedule_rank,
                    adjustment_id=adjustment.adjustment_id,
                    replan_index=displaced.replan_index,
                )
            else:
                displaced_status = ScheduleStatus.BACKLOG
                current[displaced_id] = Placement(
                    target_inspection_id=displaced_id,
                    status=ScheduleStatus.BACKLOG,
                    reason=ScheduleReason.DISPLACED_BY_ADJUSTMENT,
                    inversion_reason=InversionReason.DISPLACED_BY_ADJUSTMENT,
                    original_slot_date=displaced.original_slot_date,
                    original_schedule_rank=displaced.original_schedule_rank,
                    adjustment_id=adjustment.adjustment_id,
                    replan_index=displaced.replan_index,
                )

        deferring = adjustment.action == AdjustmentAction.DEFER_TO_DATE
        moved_status = ScheduleStatus.DEFERRED if deferring else ScheduleStatus.SCHEDULED
        moved_reason = (
            ScheduleReason.DEFERRED_BY_ADJUSTMENT
            if deferring
            else ScheduleReason.ADVANCED_BY_ADJUSTMENT
        )
        moved_inversion = (
            InversionReason.DEFERRED_BY_ADJUSTMENT
            if deferring
            else InversionReason.ADVANCED_BY_ADJUSTMENT
        )
        current[pid] = Placement(
            target_inspection_id=pid,
            status=moved_status,
            reason=moved_reason,
            inversion_reason=moved_inversion,
            slot_date=target,
            day_index=day_index[target],
            slot_index=free,
            schedule_rank=placement.schedule_rank,
            original_slot_date=placement.original_slot_date,
            original_schedule_rank=placement.original_schedule_rank,
            adjustment_id=adjustment.adjustment_id,
            replan_index=placement.replan_index,
        )
        outcomes.append(
            AdjustmentOutcome(
                adjustment=adjustment,
                outcome=OUTCOME_APPLIED,
                original_status=placement.status,
                original_slot_date=placement.slot_date,
                original_schedule_rank=placement.schedule_rank,
                final_status=moved_status,
                final_slot_date=target,
                displaced_target_inspection_id=displaced_id,
                displaced_landed_status=displaced_status,
            )
        )

    adjusted = SchedulePlan(
        schedule_config_id=plan.schedule_config_id,
        policy_id=plan.policy_id,
        model_name=plan.model_name,
        fold_set=plan.fold_set,
        fold_id=plan.fold_id,
        k_name=plan.k_name,
        k=plan.k,
        horizon=plan.horizon,
        placements=tuple(current[row.target_inspection_id] for row in queue),
        planning_run_id=plan.planning_run_id,
        replan_index=plan.replan_index,
    )
    return outcomes, adjusted


def adjustment_log_rows(outcomes: Sequence[AdjustmentOutcome]) -> list[dict[str, object]]:
    """The adjustment log, one row per adjustment offered, applied or not."""
    return [
        {
            "adjustment_id": outcome.adjustment.adjustment_id,
            "schedule_config_id": outcome.adjustment.schedule_config_id,
            "policy_id": outcome.adjustment.policy_id,
            "fold_id": outcome.adjustment.fold_id,
            "k_name": outcome.adjustment.k_name,
            "target_inspection_id": outcome.adjustment.target_inspection_id,
            "action": outcome.adjustment.action,
            "target_date": outcome.adjustment.target_date,
            "reason_code": outcome.adjustment.reason_code,
            "actor": outcome.adjustment.actor,
            "decided_at": outcome.adjustment.decided_at,
            "original_status": outcome.original_status,
            "original_scheduled_date": outcome.original_slot_date,
            "original_schedule_rank": outcome.original_schedule_rank,
            "final_status": outcome.final_status,
            "final_scheduled_date": outcome.final_slot_date,
            "displaced_target_inspection_id": outcome.displaced_target_inspection_id,
            "displaced_landed_status": outcome.displaced_landed_status,
            "outcome": outcome.outcome,
        }
        for outcome in outcomes
    ]


__all__ = [
    "OUTCOME_APPLIED",
    "OUTCOME_NO_OP_ALREADY_ON_DATE",
    "OUTCOME_NO_OP_NOT_SCHEDULED",
    "OUTCOME_ROW_NOT_IN_PLAN",
    "AdjustmentError",
    "adjustment_log_rows",
    "apply_adjustments",
    "parse_adjustments",
]
