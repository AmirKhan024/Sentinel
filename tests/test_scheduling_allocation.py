"""Deterministic slot allocation, and the invariants it must guarantee.

The two properties that matter most are order-independence and the absence of silent
disappearance. A scheduler that quietly depended on Parquet row order would produce a different
plan on a re-read of the same file, and one that dropped a row would answer "who did we
inspect" with a number that excluded everyone it failed to reach.
"""

from __future__ import annotations

import random
from datetime import date

import pytest

from sentinel.scheduling.allocation import (
    ScheduleAllocationError,
    has_inversion,
    place,
    queue_order,
    reconcile_capacity,
)
from sentinel.scheduling.definitions import (
    InversionReason,
    ScheduleReason,
    ScheduleStatus,
)
from sentinel.scheduling.models import Placement

from .conftest import make_horizon, make_queue, make_queue_row


class TestQueueOrder:
    def test_orders_by_policy_rank(self) -> None:
        rows = [make_queue_row(2), make_queue_row(0), make_queue_row(1)]
        assert queue_order(rows) == (1, 2, 0)

    def test_is_independent_of_input_order(self) -> None:
        rows = list(make_queue(20))
        expected = [rows[i].target_inspection_id for i in queue_order(rows)]
        for seed in range(10):
            shuffled = list(rows)
            random.Random(seed).shuffle(shuffled)
            got = [shuffled[i].target_inspection_id for i in queue_order(shuffled)]
            assert got == expected

    def test_the_tie_break_is_the_inspection_id(self) -> None:
        """Unreachable on real data -- Component 13 guarantees unique ranks -- and tested anyway.

        "Unique by construction" is a claim about code that was correct when it was written.
        """
        rows = [
            make_queue_row(0, target_inspection_id="T-b", final_policy_rank=1),
            make_queue_row(1, target_inspection_id="T-a", final_policy_rank=1),
        ]
        assert [rows[i].target_inspection_id for i in queue_order(rows)] == ["T-a", "T-b"]


class TestPlacement:
    def test_fills_days_in_date_order_from_the_queue_in_rank_order(self) -> None:
        queue = make_queue(6)
        placements = place(queue, make_horizon([2, 2, 2], k=6, median=2))
        landed = {p.target_inspection_id: (p.slot_date, p.slot_index) for p in placements}
        assert landed["T00000"] == (date(2026, 4, 1), 1)
        assert landed["T00001"] == (date(2026, 4, 1), 2)
        assert landed["T00002"] == (date(2026, 4, 2), 1)
        assert landed["T00005"] == (date(2026, 4, 3), 2)

    def test_a_day_boundary_falls_where_the_volume_says(self) -> None:
        queue = make_queue(6)
        placements = place(queue, make_horizon([1, 5], k=6, median=3))
        by_id = {p.target_inspection_id: p for p in placements}
        assert by_id["T00000"].slot_date == date(2026, 4, 1)
        assert by_id["T00001"].slot_date == date(2026, 4, 2)

    def test_schedule_ranks_are_contiguous_from_one(self) -> None:
        placements = place(make_queue(7), make_horizon([3, 4], k=7, median=4))
        ranks = sorted(p.schedule_rank for p in placements if p.occupies_a_slot)
        assert ranks == [1, 2, 3, 4, 5, 6, 7]

    def test_returns_one_placement_per_input_row_in_input_order(self) -> None:
        queue = make_queue(9)
        placements = place(queue, make_horizon([4, 4], k=9, median=5))
        assert [p.target_inspection_id for p in placements] == [
            row.target_inspection_id for row in queue
        ]

    def test_the_original_assignment_is_recorded_at_placement(self) -> None:
        placements = place(make_queue(4), make_horizon([4], k=4, median=4))
        assert all(p.original_slot_date == p.slot_date for p in placements if p.occupies_a_slot)
        assert all(
            p.original_schedule_rank == p.schedule_rank for p in placements if p.occupies_a_slot
        )

    def test_an_empty_queue_places_nothing(self) -> None:
        assert place([], make_horizon([3], k=3, median=3)) == ()


class TestInsufficientCapacity:
    def test_the_overflow_becomes_backlog_rather_than_an_error(self) -> None:
        """44 of 90 real cells do this. A component that raised would refuse to report it."""
        placements = place(make_queue(10), make_horizon([2, 2], k=10, median=5))
        assert sum(1 for p in placements if p.occupies_a_slot) == 4
        assert sum(1 for p in placements if p.status == ScheduleStatus.BACKLOG) == 6

    def test_the_backlog_is_the_worst_ranked_suffix(self) -> None:
        queue = make_queue(10)
        placements = place(queue, make_horizon([2, 2], k=10, median=5))
        backlogged = {
            p.target_inspection_id for p in placements if p.status == ScheduleStatus.BACKLOG
        }
        assert backlogged == {f"T{i:05d}" for i in range(4, 10)}

    def test_a_backlogged_row_names_the_exhausted_horizon(self) -> None:
        placements = place(make_queue(4), make_horizon([1], k=4, median=4))
        backlog = [p for p in placements if p.status == ScheduleStatus.BACKLOG]
        assert all(p.reason == ScheduleReason.CAPACITY_EXHAUSTED_IN_HORIZON for p in backlog)

    def test_a_backlogged_row_holds_no_slot_and_no_rank(self) -> None:
        placements = place(make_queue(4), make_horizon([1], k=4, median=4))
        for placement in placements:
            if placement.status == ScheduleStatus.BACKLOG:
                assert placement.slot_date is None
                assert placement.slot_index is None
                assert placement.schedule_rank is None

    def test_the_coverage_reserve_is_lost_first(self) -> None:
        """The component's headline, in miniature.

        Component 13 puts the reserve at the tail of the rank order, so a short horizon takes
        it first -- every time, and without the scheduler ever looking at a mechanism.
        """
        queue = make_queue(10, reserve_tail=3)
        placements = place(queue, make_horizon([5], k=10, median=10))
        by_id = {row.target_inspection_id: row for row in queue}
        scheduled = [p for p in placements if p.occupies_a_slot]
        reserve_scheduled = [
            p
            for p in scheduled
            if by_id[p.target_inspection_id].decision_mechanism == "coverage_reserve"
        ]
        assert len(scheduled) == 5
        assert reserve_scheduled == []

    def test_idle_slots_appear_when_the_calendar_is_generous(self) -> None:
        placements = place(make_queue(3), make_horizon([9], k=3, median=3))
        assert sum(1 for p in placements if p.occupies_a_slot) == 3

    def test_a_queue_exactly_equal_to_capacity_saturates(self) -> None:
        placements = place(make_queue(6), make_horizon([3, 3], k=6, median=3))
        assert all(p.occupies_a_slot for p in placements)

    def test_a_single_slot_horizon_schedules_exactly_one(self) -> None:
        placements = place(make_queue(5), make_horizon([1], k=5, median=5))
        assert sum(1 for p in placements if p.occupies_a_slot) == 1


class TestShuffleInvariance:
    @pytest.mark.parametrize("seed", range(8))
    def test_shuffled_input_produces_an_identical_plan(self, seed: int) -> None:
        queue = list(make_queue(24))
        horizon = make_horizon([5, 3, 8, 6], k=24, median=6)
        expected = {
            p.target_inspection_id: (p.slot_date, p.slot_index, p.status)
            for p in place(queue, horizon)
        }
        shuffled = list(queue)
        random.Random(seed).shuffle(shuffled)
        got = {
            p.target_inspection_id: (p.slot_date, p.slot_index, p.status)
            for p in place(shuffled, horizon)
        }
        assert got == expected

    def test_shuffling_does_not_change_the_backlog_membership(self) -> None:
        queue = list(make_queue(20))
        horizon = make_horizon([3, 3], k=20, median=10)
        expected = {p.target_inspection_id for p in place(queue, horizon) if not p.occupies_a_slot}
        shuffled = list(queue)
        random.Random(99).shuffle(shuffled)
        got = {p.target_inspection_id for p in place(shuffled, horizon) if not p.occupies_a_slot}
        assert got == expected


class TestInversions:
    def test_strict_priority_produces_none(self) -> None:
        queue = make_queue(15)
        assert not has_inversion(queue, place(queue, make_horizon([4, 5, 6], k=15, median=5)))

    def test_a_swapped_placement_is_detected(self) -> None:
        queue = make_queue(2)
        placements = (
            Placement(
                target_inspection_id="T00000",
                status=ScheduleStatus.SCHEDULED,
                reason=ScheduleReason.PLACED_IN_PRIORITY_ORDER,
                slot_date=date(2026, 4, 2),
                day_index=2,
                slot_index=1,
                schedule_rank=2,
            ),
            Placement(
                target_inspection_id="T00001",
                status=ScheduleStatus.SCHEDULED,
                reason=ScheduleReason.PLACED_IN_PRIORITY_ORDER,
                slot_date=date(2026, 4, 1),
                day_index=1,
                slot_index=1,
                schedule_rank=1,
            ),
        )
        assert has_inversion(queue, placements)


class TestPostConditions:
    def test_reconcile_accepts_a_well_formed_plan(self) -> None:
        horizon = make_horizon([3, 3], k=6, median=3)
        reconcile_capacity(place(make_queue(6), horizon), horizon)

    def test_a_double_booked_slot_is_refused(self) -> None:
        horizon = make_horizon([2], k=2, median=2)
        doubled = tuple(
            Placement(
                target_inspection_id=f"T{i:05d}",
                status=ScheduleStatus.SCHEDULED,
                reason=ScheduleReason.PLACED_IN_PRIORITY_ORDER,
                slot_date=date(2026, 4, 1),
                day_index=1,
                slot_index=1,
                schedule_rank=i + 1,
            )
            for i in range(2)
        )
        with pytest.raises(ScheduleAllocationError, match="double-booked"):
            reconcile_capacity(doubled, horizon)

    def test_an_over_capacity_day_is_refused(self) -> None:
        horizon = make_horizon([1], k=1, median=1)
        over = tuple(
            Placement(
                target_inspection_id=f"T{i:05d}",
                status=ScheduleStatus.SCHEDULED,
                reason=ScheduleReason.PLACED_IN_PRIORITY_ORDER,
                slot_date=date(2026, 4, 1),
                day_index=1,
                slot_index=i + 1,
                schedule_rank=i + 1,
            )
            for i in range(2)
        )
        with pytest.raises(ScheduleAllocationError, match="Capacity is inherited"):
            reconcile_capacity(over, horizon)

    def test_a_row_placed_twice_is_refused(self) -> None:
        horizon = make_horizon([2], k=2, median=2)
        twice = tuple(
            Placement(
                target_inspection_id="T00000",
                status=ScheduleStatus.SCHEDULED,
                reason=ScheduleReason.PLACED_IN_PRIORITY_ORDER,
                slot_date=date(2026, 4, 1),
                day_index=1,
                slot_index=i + 1,
                schedule_rank=i + 1,
            )
            for i in range(2)
        )
        with pytest.raises(ScheduleAllocationError, match="placed twice"):
            reconcile_capacity(twice, horizon)

    def test_a_gapped_schedule_rank_is_refused(self) -> None:
        horizon = make_horizon([2], k=2, median=2)
        gapped = (
            Placement(
                target_inspection_id="T00000",
                status=ScheduleStatus.SCHEDULED,
                reason=ScheduleReason.PLACED_IN_PRIORITY_ORDER,
                slot_date=date(2026, 4, 1),
                day_index=1,
                slot_index=1,
                schedule_rank=1,
            ),
            Placement(
                target_inspection_id="T00001",
                status=ScheduleStatus.SCHEDULED,
                reason=ScheduleReason.PLACED_IN_PRIORITY_ORDER,
                slot_date=date(2026, 4, 1),
                day_index=1,
                slot_index=2,
                schedule_rank=7,
            ),
        )
        with pytest.raises(ScheduleAllocationError, match="unique and contiguous"):
            reconcile_capacity(gapped, horizon)


class TestThePlacementType:
    def test_a_scheduled_row_must_carry_a_date(self) -> None:
        with pytest.raises(ValueError, match="disagree"):
            Placement(
                target_inspection_id="T",
                status=ScheduleStatus.SCHEDULED,
                reason=ScheduleReason.PLACED_IN_PRIORITY_ORDER,
            )

    def test_a_backlogged_row_must_not_carry_a_date(self) -> None:
        with pytest.raises(ValueError, match="disagree"):
            Placement(
                target_inspection_id="T",
                status=ScheduleStatus.BACKLOG,
                reason=ScheduleReason.CAPACITY_EXHAUSTED_IN_HORIZON,
                slot_date=date(2026, 4, 1),
                day_index=1,
                slot_index=1,
            )

    def test_a_deferred_row_holds_a_slot(self) -> None:
        placement = Placement(
            target_inspection_id="T",
            status=ScheduleStatus.DEFERRED,
            reason=ScheduleReason.DEFERRED_BY_ADJUSTMENT,
            inversion_reason=InversionReason.DEFERRED_BY_ADJUSTMENT,
            slot_date=date(2026, 4, 2),
            day_index=2,
            slot_index=1,
        )
        assert placement.occupies_a_slot
