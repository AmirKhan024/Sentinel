"""Rolling re-planning. Appends a planning run; never mutates one.

**History is preserved structurally, not by discipline.** A ``SchedulePlan`` is frozen, this
module returns a new one, and both are written. There is no code path that edits a plan in
place, so "we do not rewrite history" is a property of the types rather than a promise in a
docstring. ``validate.no_execution_event_changes_an_earlier_schedule`` compares plan *n*
against plan *n+1* row by row and proves it from the artifacts.

**What is frozen, and why each thing is frozen.**

* A **completed** row keeps its slot forever. Rescheduling something that already happened is
  not a plan, it is a contradiction.
* Every row on an operating day **before the re-plan point** is frozen whatever its status. A
  re-plan is forward-looking by definition; one that moved yesterday's inspection would be
  rewriting the record of what was planned, which is the exact failure the temporal boundary
  exists to prevent.
* ``original_slot_date`` and ``original_schedule_rank`` are copied forward untouched at every
  index, so *"where was this originally going to be?"* stays answerable after any number of
  re-plans.

**What comes back to the queue, and what does not.** A ``not_performed`` row returns at its
original rank -- the establishment still needs an inspection. Backlogged rows compete for the
freed capacity, because that capacity is real and stranding it would be a worse answer than
using it. A ``cancelled_in_field`` row does **not** come back: a cancellation is a removal.

**Why a re-plan backfills where an override does not.** Component 13 deliberately refuses to
backfill a ``force_exclude``, and this module deliberately does backfill a day that did not
happen. The two look like the same operation and are not. An excluded row is a human decision
that the slot should not be used, and re-filling it would overturn that decision. A day that
did not happen is capacity that still exists, and refusing to re-plan it would strand it for no
reason anybody chose. ``REPLAN_BACKFILL_RULE`` states the distinction in the manifest so it
travels with the artifact.

**Ordering is ``final_policy_rank`` at every index.** Nothing is re-ranked, re-scored or
re-prioritised, ever, at any re-plan depth. Component 13 owns the ordering and a re-planner
that produced a second one would be an unowned policy layer that only appeared on the second
run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

from sentinel.scheduling.definitions import (
    ExecutionStatus,
    InversionReason,
    ScheduleReason,
    ScheduleStatus,
)
from sentinel.scheduling.models import Placement, QueueRow, SchedulePlan


class ReplanError(ValueError):
    """Raised when a re-plan would have to rewrite something it is not allowed to touch."""


def replan_point(
    plan: SchedulePlan, statuses: Mapping[str, str], observed: Mapping[str, date]
) -> date | None:
    """The first operating day a re-plan may touch, or None when there is nothing to do.

    The day after the latest observation that triggers a re-plan. Derived from the events
    rather than supplied, so the boundary cannot be set to a date that makes a convenient
    amount of the plan editable.
    """
    triggering = [
        observed[pid]
        for pid, status in statuses.items()
        if status == ExecutionStatus.NOT_PERFORMED and pid in observed
    ]
    if not triggering:
        return None
    latest = max(triggering)
    for day in plan.horizon.days:
        if day.slot_date > latest:
            return day.slot_date
    return None


def replan(
    previous: SchedulePlan,
    queue: Sequence[QueueRow],
    statuses: Mapping[str, str],
    *,
    from_date: date,
    planning_run_id: str,
    replan_index: int,
) -> tuple[SchedulePlan, dict[str, object]]:
    """Produce plan *n+1* from plan *n* and a set of execution statuses.

    Returns the new plan and a row describing the run. The previous plan is untouched and is
    written beside the new one; a reader can diff them, and the validator does.
    """
    if replan_index <= previous.replan_index:
        raise ReplanError(
            f"a re-plan must advance the index, got {replan_index} after "
            f"{previous.replan_index}. A repeated index would make two different plans "
            "indistinguishable in the artifact"
        )

    by_id = {row.target_inspection_id: row for row in queue}
    current = previous.by_id()
    horizon = previous.horizon
    day_index = {day.slot_date: day.day_index for day in horizon.days}

    frozen: dict[str, Placement] = {}
    needs_placement: list[str] = []
    cancelled: dict[str, Placement] = {}

    n_preserved_completed = 0
    n_preserved_past = 0
    n_returned = 0
    n_cancelled = 0

    for pid, placement in current.items():
        status = statuses.get(pid, "")

        if status == ExecutionStatus.COMPLETED:
            frozen[pid] = placement
            n_preserved_completed += 1
            continue
        if status == ExecutionStatus.CANCELLED_IN_FIELD:
            cancelled[pid] = Placement(
                target_inspection_id=pid,
                status=ScheduleStatus.CANCELLED,
                reason=ScheduleReason.CANCELLED_IN_FIELD,
                inversion_reason=placement.inversion_reason,
                original_slot_date=placement.original_slot_date,
                original_schedule_rank=placement.original_schedule_rank,
                adjustment_id=placement.adjustment_id,
                replan_index=replan_index,
            )
            n_cancelled += 1
            continue
        if placement.status == ScheduleStatus.CANCELLED:
            cancelled[pid] = placement
            continue
        if status == ExecutionStatus.NOT_PERFORMED:
            # Returned to the queue even though its day has already passed. That is the whole
            # point of the report: the inspection did not happen and the establishment still
            # needs one. Freezing it because its slot is in the past would strand exactly the
            # row the field took the trouble to tell us about.
            needs_placement.append(pid)
            n_returned += 1
            continue
        if placement.occupies_a_slot and placement.slot_date is not None:
            if placement.slot_date < from_date:
                frozen[pid] = placement
                n_preserved_past += 1
                continue
            needs_placement.append(pid)
            continue
        needs_placement.append(pid)

    used_by_day: dict[date, set[int]] = {}
    for placement in frozen.values():
        if placement.occupies_a_slot and placement.slot_date is not None:
            used_by_day.setdefault(placement.slot_date, set()).add(placement.slot_index or 0)

    open_slots: list[tuple[date, int]] = []
    for day in horizon.days:
        if day.slot_date < from_date:
            continue
        used = used_by_day.get(day.slot_date, set())
        for index in range(1, day.n_slots + 1):
            if index not in used:
                open_slots.append((day.slot_date, index))

    ordered = sorted(
        needs_placement,
        key=lambda pid: (by_id[pid].final_policy_rank, pid),
    )

    replanned: dict[str, Placement] = {}
    n_newly_scheduled = 0
    for position, pid in enumerate(ordered):
        original = current[pid]
        if position < len(open_slots):
            slot_date, slot_index = open_slots[position]
            moved = original.slot_date != slot_date
            replanned[pid] = Placement(
                target_inspection_id=pid,
                status=ScheduleStatus.SCHEDULED,
                reason=(
                    ScheduleReason.RESCHEDULED_BY_REPLAN
                    if moved
                    else ScheduleReason.PLACED_IN_PRIORITY_ORDER
                ),
                inversion_reason=(
                    InversionReason.RESCHEDULED_BY_REPLAN if moved else original.inversion_reason
                ),
                slot_date=slot_date,
                day_index=day_index[slot_date],
                slot_index=slot_index,
                schedule_rank=original.schedule_rank,
                original_slot_date=original.original_slot_date,
                original_schedule_rank=original.original_schedule_rank,
                adjustment_id=original.adjustment_id,
                replan_index=replan_index,
            )
            if moved:
                n_newly_scheduled += 1
        else:
            replanned[pid] = Placement(
                target_inspection_id=pid,
                status=ScheduleStatus.BACKLOG,
                reason=ScheduleReason.CAPACITY_EXHAUSTED_IN_HORIZON,
                inversion_reason=original.inversion_reason,
                original_slot_date=original.original_slot_date,
                original_schedule_rank=original.original_schedule_rank,
                adjustment_id=original.adjustment_id,
                replan_index=replan_index,
            )

    merged = {**frozen, **cancelled, **replanned}
    missing = {row.target_inspection_id for row in queue} - set(merged)
    if missing:
        raise ReplanError(
            f"{len(missing)} approved row(s) vanished during the re-plan, first "
            f"{sorted(missing)[:3]}. Every row must survive a re-plan in some status; one that "
            "disappears is an establishment nobody is accountable for"
        )

    plan = SchedulePlan(
        schedule_config_id=previous.schedule_config_id,
        policy_id=previous.policy_id,
        model_name=previous.model_name,
        fold_set=previous.fold_set,
        fold_id=previous.fold_id,
        k_name=previous.k_name,
        k=previous.k,
        horizon=horizon,
        placements=tuple(merged[row.target_inspection_id] for row in queue),
        planning_run_id=planning_run_id,
        replan_index=replan_index,
    )
    run = {
        "planning_run_id": planning_run_id,
        "replan_index": replan_index,
        "parent_replan_index": previous.replan_index,
        "replan_from_date": from_date,
        "trigger": "execution_not_performed",
        "n_preserved_completed": n_preserved_completed,
        "n_preserved_past": n_preserved_past,
        "n_returned_to_queue": n_returned,
        "n_cancelled": n_cancelled,
        "n_newly_scheduled": n_newly_scheduled,
        "n_still_backlog": plan.n_backlog,
        "remaining_slots": len(open_slots),
    }
    return plan, run


def original_run_row(plan: SchedulePlan) -> dict[str, object]:
    """The row describing the original plan.

    The original plan *is* a planning run, and saying so is more honest than emitting an empty
    table on runs where nobody supplied an execution file. A reader looking at
    ``replanning_runs`` should see the lineage start at index 0 rather than see nothing and
    have to infer that a plan existed.
    """
    return {
        "planning_run_id": plan.planning_run_id,
        "replan_index": plan.replan_index,
        "parent_replan_index": None,
        "replan_from_date": None,
        "trigger": "original_plan",
        "n_preserved_completed": 0,
        "n_preserved_past": 0,
        "n_returned_to_queue": 0,
        "n_cancelled": 0,
        "n_newly_scheduled": plan.n_scheduled,
        "n_still_backlog": plan.n_backlog,
        "remaining_slots": plan.idle_slots,
    }


__all__ = ["ReplanError", "original_run_row", "replan", "replan_point"]
