"""The backlog: recommended, approved, and not reached by this horizon.

**"Not scheduled" is never redefined as "not recommended".** That sentence is the whole module.
An establishment that Component 13 selected and Component 14 could not fit is still selected --
it keeps its rank, its mechanism, its reason code and its place in the queue -- and the only
thing that changed is that the calendar ran out. A component that dropped those rows, or
recoloured them as unselected, would be answering "who did we inspect" with a number that
quietly excluded everyone it failed to reach.

**Insufficient capacity is not an error.** It is the measurement: in 44 of 90 (fold, capacity)
cells the observed calendar supplies fewer slots than the approved queue needs, for 784
inspections in total. A run that went red on that would be a run that goes green only when the
scheduler is lying about the calendar.

**The columns exist to make the backlog answerable rather than merely present.** Knowing that
ten rows did not fit is much less useful than knowing they were ranks 131 to 140, that the
horizon was ten slots short, and that the fold's next operating day was 2026-04-09. The first
is a count; the second is something an operations manager can act on.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from sentinel.scheduling.definitions import ScheduleStatus
from sentinel.scheduling.models import Horizon, Placement, QueueRow, SchedulePlan


def _next_operating_days(
    calendar: Sequence[tuple[date, int]], horizon: Horizon
) -> tuple[tuple[date, int], ...]:
    """The fold's operating days after the horizon ends, with their observed volumes."""
    return tuple((day, count) for day, count in calendar if day > horizon.end_date)


def backlog_rows(
    plan: SchedulePlan,
    queue: Sequence[QueueRow],
    calendar: Sequence[tuple[date, int]],
) -> list[dict[str, object]]:
    """One row per approved recommendation the horizon did not reach.

    ``would_fit_on_day_index`` answers the question an operations manager actually asks --
    *how much longer would this have taken?* -- by walking the fold's own remaining operating
    days at their own observed volumes. It is null when the fold's calendar cannot reach the
    row at all, which is a different and worse answer than a large number.

    Nothing here re-orders anything. ``backlog_position`` is ascending ``final_policy_rank``,
    so the backlog is the queue's tail in the queue's own order, and the row that would be
    worked first if one more day existed is position 1.
    """
    by_id = {row.target_inspection_id: row for row in queue}
    unplaced = [
        placement for placement in plan.placements if placement.status in (ScheduleStatus.BACKLOG,)
    ]
    if not unplaced:
        return []

    ordered = sorted(
        unplaced,
        key=lambda p: (
            by_id[p.target_inspection_id].final_policy_rank,
            p.target_inspection_id,
        ),
    )
    remaining = _next_operating_days(calendar, plan.horizon)
    horizon_slots = plan.horizon.total_slots

    out: list[dict[str, object]] = []
    day_cursor = 0
    slot_in_day = 0
    for position, placement in enumerate(ordered, start=1):
        row = by_id[placement.target_inspection_id]

        would_fit: int | None = None
        first_available: date | None = None
        while day_cursor < len(remaining):
            _, capacity = remaining[day_cursor]
            if slot_in_day < capacity:
                slot_in_day += 1
                would_fit = plan.horizon.n_days + day_cursor + 1
                first_available = remaining[day_cursor][0]
                break
            day_cursor += 1
            slot_in_day = 0

        out.append(
            {
                "target_inspection_id": placement.target_inspection_id,
                "establishment_id": row.establishment_id,
                "final_policy_rank": row.final_policy_rank,
                "decision_mechanism": row.decision_mechanism,
                "decision_reason": row.decision_reason,
                "coverage_eligible": row.coverage_eligible,
                "backlog_position": position,
                "backlog_reason": placement.reason,
                "horizon_slots": horizon_slots,
                "slots_short": max(0, plan.k - horizon_slots),
                "would_fit_on_day_index": would_fit,
                "first_available_date": first_available,
            }
        )
    return out


def backlog_by_mechanism(plan: SchedulePlan, queue: Sequence[QueueRow]) -> dict[str, int]:
    """Backlogged rows counted by the mechanism that put them in the queue.

    The input to the component's headline. Component 13 fills the risk block first and the
    coverage reserve after it, so the reserve is always the tail of the rank order -- which
    means a horizon that falls short takes the reserve first, every time. This function is what
    turns that structural fact into a number per cell.
    """
    by_id = {row.target_inspection_id: row for row in queue}
    counts: dict[str, int] = {}
    for placement in plan.placements:
        if placement.status != ScheduleStatus.BACKLOG:
            continue
        mechanism = by_id[placement.target_inspection_id].decision_mechanism
        counts[mechanism] = counts.get(mechanism, 0) + 1
    return counts


def unplaced_placements(placements: Sequence[Placement]) -> tuple[Placement, ...]:
    """Every placement that does not occupy a slot, whatever the reason.

    Used by the validator to prove that the backlog table and the plan agree. They are derived
    from the same object, so they cannot disagree -- and the check runs anyway, because the two
    are written by different code and only one of them is read by a person.
    """
    return tuple(p for p in placements if not p.occupies_a_slot)


__all__ = ["backlog_by_mechanism", "backlog_rows", "unplaced_placements"]
