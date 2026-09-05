"""Operational measurements of a schedule. No model, no label, no ML metric.

**There is no precision, no capture rate, no NDE and no lift anywhere in this module, and that
is a design decision rather than an omission.** Component 5 owns discovery metrics and
Component 13 already reports them per policy. A scheduling table carrying ``precision_at_k``
would invite exactly one conclusion -- that a schedule improved a model -- and it cannot: this
component re-orders an approved queue in time and never changes which establishments are in it.
The set of inspections is Component 13's; only their dates are this component's.

**Six families, kept in separate column groups and never summed.** Capacity utilisation,
priority preservation, queue coverage, wait, mechanism preservation, and adjustment/execution
impact measure different things and trade off against each other. A single "schedule quality"
score would hide which one moved, and would invite tuning against a number nobody agreed to.

**The mechanism-preservation group is the component's headline.** ``reserve_slots_lost`` is the
number of coverage-reserve slots a strict-priority schedule drops because Component 13 places
the reserve at the tail of the rank order. It is measured here, reported at advisory severity,
and deliberately not corrected.

**Wait is measured in operating days, never in calendar days.** A calendar-day count would
include weekends the dataset never worked and would make a Friday-to-Monday gap look like a
three-day delay. The dataset has no clock at all, so no measurement here is finer than a day.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from sentinel.policy.definitions import DecisionMechanism
from sentinel.scheduling.definitions import ScheduleStatus
from sentinel.scheduling.models import Placement, QueueRow, SchedulePlan


def count_inversions(values: Sequence[int]) -> int:
    """Pairs out of order in ``values``, by a Fenwick tree in O(n log n).

    Arithmetic rather than an algorithm, which is ADR 0015's dividing line, so no dependency
    arrives for it -- Component 5 hand-rolled every metric in this project on the same
    reasoning. The brute-force O(n^2) version is the test oracle.
    """
    n = len(values)
    if n < 2:
        return 0
    ranks = {value: index for index, value in enumerate(sorted(set(values)), start=1)}
    size = len(ranks)
    tree = [0] * (size + 1)

    def add(position: int) -> None:
        while position <= size:
            tree[position] += 1
            position += position & -position

    def prefix(position: int) -> int:
        total = 0
        while position > 0:
            total += tree[position]
            position -= position & -position
        return total

    inversions = 0
    for value in reversed(values):
        rank = ranks[value]
        inversions += prefix(rank - 1)
        add(rank)
    return inversions


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Rank correlation between two equal-length sequences, or None when undefined.

    None rather than 1.0 or 0.0 for a constant or single-element sequence. A correlation over
    one point is not 1.0; it is a question with no answer, and returning a number would put a
    fabricated value in a column a reader would average.
    """
    n = len(left)
    if n != len(right):
        raise ValueError(f"spearman needs equal lengths, got {n} and {len(right)}")
    if n < 2:
        return None
    mean_left = sum(left) / n
    mean_right = sum(right) / n
    cov = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True))
    var_left = sum((a - mean_left) ** 2 for a in left)
    var_right = sum((b - mean_right) ** 2 for b in right)
    if var_left <= 0.0 or var_right <= 0.0:
        return None
    return float(cov / (var_left * var_right) ** 0.5)


def _scheduled_in_slot_order(plan: SchedulePlan) -> list[Placement]:
    return sorted(
        (p for p in plan.placements if p.occupies_a_slot),
        key=lambda p: (p.slot_date or date.min, p.slot_index or 0),
    )


def capacity_utilization_rows(
    plan: SchedulePlan, queue: Sequence[QueueRow]
) -> list[dict[str, object]]:
    """One row per operating day: what it held, what it used, and by which mechanism.

    Utilisation is bounded above by one and the validator fails if it is not. A day above one
    means the allocator placed more inspections than the city worked, which is the failure this
    component's whole capacity contract exists to prevent.
    """
    by_id = {row.target_inspection_id: row for row in queue}
    used: dict[int, list[Placement]] = {}
    for placement in plan.placements:
        if placement.occupies_a_slot and placement.day_index is not None:
            used.setdefault(placement.day_index, []).append(placement)

    out: list[dict[str, object]] = []
    for day in plan.horizon.days:
        placements = used.get(day.day_index, [])
        risk = sum(
            1
            for p in placements
            if by_id[p.target_inspection_id].decision_mechanism == DecisionMechanism.RISK_PRIORITY
        )
        reserve = sum(
            1
            for p in placements
            if by_id[p.target_inspection_id].decision_mechanism
            == DecisionMechanism.COVERAGE_RESERVE
        )
        out.append(
            {
                "day_index": day.day_index,
                "slot_date": day.slot_date,
                "n_slots": day.n_slots,
                "n_scheduled": len(placements),
                "idle_slots": max(0, day.n_slots - len(placements)),
                "utilization": len(placements) / day.n_slots if day.n_slots else 0.0,
                "n_risk_scheduled": risk,
                "n_reserve_scheduled": reserve,
                "capacity_source": day.capacity_source,
            }
        )
    return out


def preservation_row(plan: SchedulePlan, queue: Sequence[QueueRow]) -> dict[str, object]:
    """One row per cell: priority, coverage, wait and mechanism preservation together.

    Together rather than in four tables because they are four readings of one event, and a
    reader comparing them across policies wants them on one line. They stay in separate column
    *groups* so that no arithmetic can accidentally combine them.
    """
    by_id = {row.target_inspection_id: row for row in queue}
    scheduled = _scheduled_in_slot_order(plan)
    policy_ranks = [by_id[p.target_inspection_id].final_policy_rank for p in scheduled]
    schedule_ranks = [float(p.schedule_rank or 0) for p in scheduled]

    inversions = count_inversions(policy_ranks)
    max_depth = 0
    best = 0
    for rank in policy_ranks:
        if rank < best:
            max_depth = max(max_depth, best - rank)
        best = max(best, rank)

    day_index_by_date = {day.slot_date: day.day_index for day in plan.horizon.days}
    waits = [
        day_index_by_date[p.slot_date] - 1
        for p in scheduled
        if p.slot_date is not None and p.slot_date in day_index_by_date
    ]

    backlogged = [
        by_id[p.target_inspection_id].final_policy_rank
        for p in plan.placements
        if p.status == ScheduleStatus.BACKLOG
    ]

    n_risk_rec = sum(
        1 for row in queue if row.decision_mechanism == DecisionMechanism.RISK_PRIORITY
    )
    n_reserve_rec = sum(
        1 for row in queue if row.decision_mechanism == DecisionMechanism.COVERAGE_RESERVE
    )
    n_risk_sch = sum(
        1
        for p in scheduled
        if by_id[p.target_inspection_id].decision_mechanism == DecisionMechanism.RISK_PRIORITY
    )
    n_reserve_sch = sum(
        1
        for p in scheduled
        if by_id[p.target_inspection_id].decision_mechanism == DecisionMechanism.COVERAGE_RESERVE
    )
    n_scheduled = len(scheduled)

    return {
        "n_scheduled": n_scheduled,
        "n_inversions": inversions,
        "max_inversion_depth": max_depth,
        "rank_spearman": spearman([float(r) for r in policy_ranks], schedule_ranks),
        "strict_priority_preserved": inversions == 0,
        "n_rows_with_inversion_reason": sum(
            1 for p in plan.placements if p.inversion_reason != "none"
        ),
        "queue_coverage": n_scheduled / plan.k if plan.k else 0.0,
        "worst_scheduled_policy_rank": max(policy_ranks) if policy_ranks else None,
        "best_backlogged_policy_rank": min(backlogged) if backlogged else None,
        "mean_wait_operating_days": (sum(waits) / len(waits)) if waits else None,
        "median_wait_operating_days": _median(waits),
        "max_wait_operating_days": max(waits) if waits else None,
        "n_risk_recommended": n_risk_rec,
        "n_reserve_recommended": n_reserve_rec,
        "n_risk_scheduled": n_risk_sch,
        "n_reserve_scheduled": n_reserve_sch,
        "reserve_share_recommended": n_reserve_rec / plan.k if plan.k else 0.0,
        "reserve_share_scheduled": (n_reserve_sch / n_scheduled) if n_scheduled else 0.0,
        "reserve_share_delta": (
            ((n_reserve_sch / n_scheduled) if n_scheduled else 0.0)
            - (n_reserve_rec / plan.k if plan.k else 0.0)
        ),
        "reserve_slots_lost": n_reserve_rec - n_reserve_sch,
    }


def _median(values: Sequence[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def summary_row(plan: SchedulePlan) -> dict[str, object]:
    """One row per cell: the schedule's account of its own arithmetic.

    ``n_recommended`` restates Component 13's ``k`` rather than deriving a new number, so a
    reader can check the two against each other without a join. The three counts below it must
    add up to it, and the validator says so.
    """
    return {
        "median_daily_capacity": plan.horizon.median_daily_capacity,
        "horizon_days": plan.horizon.n_days,
        "horizon_start_date": plan.horizon.start_date,
        "horizon_end_date": plan.horizon.end_date,
        "horizon_slots": plan.horizon.total_slots,
        "n_recommended": plan.k,
        "n_scheduled": plan.n_scheduled,
        "n_backlog": plan.n_backlog,
        "n_deferred": plan.n_deferred,
        "n_cancelled": plan.n_cancelled,
        "idle_slots": plan.idle_slots,
        "capacity_utilization": plan.capacity_utilization,
        "horizon_was_clamped": plan.horizon.was_clamped,
    }


__all__ = [
    "capacity_utilization_rows",
    "count_inversions",
    "preservation_row",
    "spearman",
    "summary_row",
]
