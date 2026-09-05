"""Deterministic slot allocation. Pure functions over typed structures, no I/O.

**This module is the component, and it is deliberately small.** Filling a horizon from an
approved queue in rank order is a prefix operation, not a search: walk the days in date order,
walk the queue in ``final_policy_rank`` order, and stop when one of them runs out. That is the
whole algorithm, and its smallness is the argument against the alternative -- an optimiser here
would have to trade off travel time, duration and inspector availability, none of which this
dataset contains, so it would be optimising over invented parameters and reporting an
optimality claim with fabricated inputs.

**Nothing here reads a score, a probability, a mechanism or a geography.** The type system
enforces most of it -- ``Placement`` carries no score to read -- and the rest is enforced by
the validator. A scheduler that preferred a ``coverage_reserve`` row for a slot, or preferred
against one, would be a second policy layer with no ADR behind it, and the fact that this
component *measures* the reserve being lost is precisely why it must not be the component that
fixes it.

**Insufficient capacity is an outcome, not an error.** A queue longer than its horizon returns
rows with status ``backlog`` and the run stays green. In 44 of 90 cells that is what happens,
and a component that raised on it would be refusing to report the thing it was built to find.

**The tie-break is unreachable and is implemented anyway.** ``final_policy_rank`` is unique and
contiguous within a cell, checked by Component 13 at error severity, so ordering by rank alone
is already total. The secondary key on ``target_inspection_id`` costs nothing and is tested,
because "unique by construction" is a claim about code that was correct when it was written --
the same argument ``policy/allocation.py`` makes about reserve disjointness.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from sentinel.scheduling.definitions import (
    InversionReason,
    ScheduleReason,
    ScheduleStatus,
)
from sentinel.scheduling.models import Horizon, Placement, QueueRow


class ScheduleAllocationError(ValueError):
    """Raised when an allocation violates an invariant it is supposed to guarantee."""


def queue_order(rows: Sequence[QueueRow]) -> tuple[int, ...]:
    """Indices into ``rows``, in the order the queue must be worked.

    ``(final_policy_rank, target_inspection_id)`` ascending. Component 13 owns the first key
    and this module never produces a second ordering; the second key is a total-order guarantee
    that the data says is unreachable.

    Returning indices rather than a sorted copy is what makes the function usable as a proof:
    a test can hand in a deliberately shuffled sequence and compare the resulting index order
    against the same sequence sorted, without the builder having quietly re-sorted anything.
    """
    return tuple(
        sorted(
            range(len(rows)),
            key=lambda i: (rows[i].final_policy_rank, rows[i].target_inspection_id),
        )
    )


def place(rows: Sequence[QueueRow], horizon: Horizon) -> tuple[Placement, ...]:
    """Fill the horizon day by day, in date order, from the queue in policy-rank order.

    Returns one ``Placement`` per input row, in the input's own order, so a caller can zip the
    two together without re-sorting. Rows the horizon does not reach carry status ``backlog``
    and reason ``capacity_exhausted_in_horizon``; they keep their rank, and the backlog module
    turns them into an artifact that says what they were waiting for.
    """
    if not rows:
        return ()

    order = queue_order(rows)
    placements: list[Placement | None] = [None] * len(rows)

    position = 0
    day_cursor = 0
    slot_in_day = 0
    day = horizon.days[0] if horizon.days else None

    for rank_index, row_index in enumerate(order, start=1):
        row = rows[row_index]
        while day is not None and slot_in_day >= day.n_slots:
            day_cursor += 1
            slot_in_day = 0
            day = horizon.days[day_cursor] if day_cursor < horizon.n_days else None

        if day is None:
            placements[row_index] = Placement(
                target_inspection_id=row.target_inspection_id,
                status=ScheduleStatus.BACKLOG,
                reason=ScheduleReason.CAPACITY_EXHAUSTED_IN_HORIZON,
                inversion_reason=InversionReason.NONE,
            )
            continue

        slot_in_day += 1
        position = rank_index
        placements[row_index] = Placement(
            target_inspection_id=row.target_inspection_id,
            status=ScheduleStatus.SCHEDULED,
            reason=ScheduleReason.PLACED_IN_PRIORITY_ORDER,
            inversion_reason=InversionReason.NONE,
            slot_date=day.slot_date,
            day_index=day.day_index,
            slot_index=slot_in_day,
            schedule_rank=position,
            original_slot_date=day.slot_date,
            original_schedule_rank=position,
        )

    out = tuple(p for p in placements if p is not None)
    if len(out) != len(rows):
        raise ScheduleAllocationError(
            f"{horizon.fold_id}/{horizon.k_name}: {len(rows)} queue rows produced {len(out)} "
            "placements. Every approved row must be placed or backlogged; a row that produced "
            "no placement would silently disappear from the plan"
        )
    reconcile_capacity(out, horizon)
    return out


def reconcile_capacity(placements: Sequence[Placement], horizon: Horizon) -> None:
    """Assert the post-conditions rather than assume them.

    Every one of these is guaranteed by the loop above, and every one of them is checked here,
    because the loop is what would be edited. The validator re-checks them from the written
    artifact as well: this catches a broken allocator, and that catches a broken writer.
    """
    per_day: dict[int, set[int]] = {}
    seen: set[str] = set()
    ranks: list[int] = []

    for placement in placements:
        if placement.target_inspection_id in seen:
            raise ScheduleAllocationError(
                f"{horizon.fold_id}/{horizon.k_name}: {placement.target_inspection_id} was "
                "placed twice. One approved inspection is one slot"
            )
        seen.add(placement.target_inspection_id)
        if not placement.occupies_a_slot:
            continue
        if placement.day_index is None or placement.slot_index is None:
            raise ScheduleAllocationError(
                f"{placement.target_inspection_id}: scheduled without a day or slot index"
            )
        slots = per_day.setdefault(placement.day_index, set())
        if placement.slot_index in slots:
            raise ScheduleAllocationError(
                f"{horizon.fold_id}/{horizon.k_name}: day {placement.day_index} slot "
                f"{placement.slot_index} is double-booked"
            )
        slots.add(placement.slot_index)
        ranks.append(placement.schedule_rank or 0)

    for day in horizon.days:
        used = per_day.get(day.day_index, set())
        if len(used) > day.n_slots:
            raise ScheduleAllocationError(
                f"{horizon.fold_id}/{horizon.k_name}: {day.slot_date} holds {day.n_slots} "
                f"slots and {len(used)} inspections were placed on it. Capacity is inherited "
                "from the window's own measured rate and is never raised"
            )
        if used and max(used) > day.n_slots:
            raise ScheduleAllocationError(
                f"{horizon.fold_id}/{horizon.k_name}: {day.slot_date} has a slot index above "
                f"its capacity of {day.n_slots}"
            )

    if sorted(ranks) != list(range(1, len(ranks) + 1)):
        raise ScheduleAllocationError(
            f"{horizon.fold_id}/{horizon.k_name}: schedule ranks are not unique and contiguous "
            "from 1. A gapped or duplicated rank makes the plan's own order unreadable"
        )


def has_inversion(rows: Sequence[QueueRow], placements: Sequence[Placement]) -> bool:
    """True when any scheduled row sits out of ``final_policy_rank`` order.

    Under strict priority with no external files this is always False, and the validator
    requires it. The function exists so that the claim is *measured* on every run rather than
    asserted once in a docstring -- and so that when an adjustment or a re-plan does create an
    inversion, the same code path detects it.
    """
    by_id = {row.target_inspection_id: row for row in rows}
    scheduled = sorted(
        (p for p in placements if p.occupies_a_slot),
        key=lambda p: (p.slot_date or date.min, p.slot_index or 0),
    )
    previous = 0
    for placement in scheduled:
        rank = by_id[placement.target_inspection_id].final_policy_rank
        if rank < previous:
            return True
        previous = rank
    return False


__all__ = [
    "ScheduleAllocationError",
    "has_inversion",
    "place",
    "queue_order",
    "reconcile_capacity",
]
