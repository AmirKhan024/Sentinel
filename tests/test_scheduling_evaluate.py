"""Backlog accounting and the operational measurements over a plan.

Two things this file is careful about. The backlog must stay a *population* -- rank, mechanism,
shortfall, next available day -- rather than collapsing into a count, because "ten rows did not
fit" is much less useful than "they were ranks 131 to 140 and the next operating day was the
9th". And no measurement here may be an ML metric: the set of inspections is Component 13's, and
only their dates are this component's, so a precision or a capture rate on a scheduling table
would invite a conclusion the component cannot support.
"""

from __future__ import annotations

import itertools
import random
from datetime import date

import pytest

from sentinel.scheduling.backlog import (
    backlog_by_mechanism,
    backlog_rows,
    unplaced_placements,
)
from sentinel.scheduling.evaluate import (
    capacity_utilization_rows,
    count_inversions,
    preservation_row,
    spearman,
    summary_row,
)

from .conftest import make_calendar, make_plan, make_queue


def _brute_force_inversions(values: list[int]) -> int:
    return sum(1 for i, j in itertools.combinations(range(len(values)), 2) if values[i] > values[j])


class TestCountInversions:
    @pytest.mark.parametrize("seed", range(25))
    def test_matches_a_brute_force_oracle(self, seed: int) -> None:
        values = random.Random(seed).sample(range(30), 30)
        assert count_inversions(values) == _brute_force_inversions(values)

    def test_a_sorted_sequence_has_none(self) -> None:
        assert count_inversions([1, 2, 3, 4, 5]) == 0

    def test_a_reversed_sequence_is_the_maximum(self) -> None:
        assert count_inversions([5, 4, 3, 2, 1]) == 10

    @pytest.mark.parametrize("values", [[], [1]])
    def test_a_degenerate_sequence_has_none(self, values: list[int]) -> None:
        assert count_inversions(values) == 0


class TestSpearman:
    def test_a_perfect_agreement_is_one(self) -> None:
        assert spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_a_perfect_reversal_is_minus_one(self) -> None:
        assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_a_constant_sequence_is_undefined_rather_than_one(self) -> None:
        """None, not a number.

        A correlation over a constant is a question with no answer, and returning 1.0 would put
        a fabricated value in a column a reader would average.
        """
        assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None

    def test_a_single_point_is_undefined(self) -> None:
        assert spearman([1.0], [1.0]) is None

    def test_unequal_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="equal lengths"):
            spearman([1.0, 2.0], [1.0])


class TestBacklogRows:
    def test_nothing_is_emitted_when_everything_fits(self) -> None:
        plan = make_plan([9], k=3, median=3)
        assert backlog_rows(plan, make_queue(3), make_calendar([9])) == []

    def test_positions_run_in_policy_rank_order(self) -> None:
        plan = make_plan([2, 2], k=10, median=5)
        rows = backlog_rows(plan, make_queue(10), make_calendar([2, 2, 9, 9]))
        assert [r["backlog_position"] for r in rows] == [1, 2, 3, 4, 5, 6]
        assert [r["final_policy_rank"] for r in rows] == [5, 6, 7, 8, 9, 10]

    def test_component_13_provenance_is_carried_unchanged(self) -> None:
        """A backlogged row is still recommended. It keeps its mechanism and its reason."""
        plan = make_plan([1], k=4, median=4, reserve_tail=2)
        rows = backlog_rows(plan, make_queue(4, reserve_tail=2), make_calendar([1, 9]))
        assert {r["decision_mechanism"] for r in rows} == {"risk_priority", "coverage_reserve"}

    def test_the_shortfall_is_recorded(self) -> None:
        plan = make_plan([2, 2], k=10, median=5)
        rows = backlog_rows(plan, make_queue(10), make_calendar([2, 2]))
        assert rows[0]["horizon_slots"] == 4
        assert rows[0]["slots_short"] == 6

    def test_the_next_available_day_comes_from_the_folds_own_calendar(self) -> None:
        plan = make_plan([2, 2], k=6, median=3)
        rows = backlog_rows(plan, make_queue(6), make_calendar([2, 2, 5, 5]))
        assert rows[0]["first_available_date"] == date(2026, 4, 3)
        assert rows[0]["would_fit_on_day_index"] == 3

    def test_a_row_the_calendar_cannot_reach_gets_a_null_rather_than_a_guess(self) -> None:
        """Null is a different and worse answer than a large number, and it is the true one."""
        plan = make_plan([1, 1], k=10, median=5)
        rows = backlog_rows(plan, make_queue(10), make_calendar([1, 1]))
        assert all(r["would_fit_on_day_index"] is None for r in rows)
        assert all(r["first_available_date"] is None for r in rows)

    def test_backlog_by_mechanism_counts_the_reserve_separately(self) -> None:
        queue = make_queue(10, reserve_tail=4)
        plan = make_plan([5], k=10, median=10, reserve_tail=4)
        counts = backlog_by_mechanism(plan, queue)
        assert counts["coverage_reserve"] == 4
        assert counts["risk_priority"] == 1

    def test_unplaced_placements_returns_everything_without_a_slot(self) -> None:
        plan = make_plan([2], k=5, median=5)
        assert len(unplaced_placements(plan.placements)) == 3


class TestCapacityUtilization:
    def test_one_row_per_operating_day(self) -> None:
        plan = make_plan([3, 4], k=7, median=4)
        assert len(capacity_utilization_rows(plan, make_queue(7))) == 2

    def test_utilization_never_exceeds_one(self) -> None:
        plan = make_plan([3, 4], k=20, median=4)
        assert all(r["utilization"] <= 1.0 for r in capacity_utilization_rows(plan, make_queue(20)))

    def test_idle_slots_are_reported_when_the_queue_is_short(self) -> None:
        plan = make_plan([9], k=3, median=3)
        rows = capacity_utilization_rows(plan, make_queue(3))
        assert rows[0]["idle_slots"] == 6
        assert rows[0]["utilization"] == pytest.approx(3 / 9)

    def test_the_mechanism_split_is_carried_per_day(self) -> None:
        queue = make_queue(6, reserve_tail=2)
        plan = make_plan([6], k=6, median=6, reserve_tail=2)
        row = capacity_utilization_rows(plan, queue)[0]
        assert row["n_risk_scheduled"] == 4
        assert row["n_reserve_scheduled"] == 2


class TestPreservationRow:
    def test_strict_priority_reports_no_inversions(self) -> None:
        plan = make_plan([4, 5], k=9, median=5)
        row = preservation_row(plan, make_queue(9))
        assert row["n_inversions"] == 0
        assert row["strict_priority_preserved"] is True
        assert row["rank_spearman"] == pytest.approx(1.0)

    def test_queue_coverage_is_the_scheduled_share(self) -> None:
        plan = make_plan([2, 2], k=10, median=5)
        row = preservation_row(plan, make_queue(10))
        assert row["queue_coverage"] == pytest.approx(0.4)

    def test_wait_is_measured_in_operating_days(self) -> None:
        """Never calendar days: a Friday-to-Monday gap is one operating day, not three."""
        plan = make_plan([1, 1, 1], k=3, median=1)
        row = preservation_row(plan, make_queue(3))
        assert row["max_wait_operating_days"] == 2
        assert row["mean_wait_operating_days"] == pytest.approx(1.0)

    def test_the_best_backlogged_rank_is_reported(self) -> None:
        plan = make_plan([3], k=8, median=8)
        row = preservation_row(plan, make_queue(8))
        assert row["worst_scheduled_policy_rank"] == 3
        assert row["best_backlogged_policy_rank"] == 4

    def test_nothing_backlogged_leaves_the_rank_null(self) -> None:
        plan = make_plan([9], k=3, median=3)
        assert preservation_row(plan, make_queue(3))["best_backlogged_policy_rank"] is None

    def test_the_reserve_loss_is_measured(self) -> None:
        """The component's headline number, on a hand-built cell."""
        queue = make_queue(10, reserve_tail=4)
        plan = make_plan([5], k=10, median=10, reserve_tail=4)
        row = preservation_row(plan, queue)
        assert row["n_reserve_recommended"] == 4
        assert row["n_reserve_scheduled"] == 0
        assert row["reserve_slots_lost"] == 4
        assert row["reserve_share_scheduled"] == pytest.approx(0.0)

    def test_a_reserve_that_survives_reports_no_loss(self) -> None:
        queue = make_queue(10, reserve_tail=4)
        plan = make_plan([10], k=10, median=10, reserve_tail=4)
        row = preservation_row(plan, queue)
        assert row["reserve_slots_lost"] == 0


class TestSummaryRow:
    def test_the_counts_account_for_every_approved_row(self) -> None:
        plan = make_plan([2, 2], k=10, median=5)
        row = summary_row(plan)
        assert row["n_recommended"] == 10
        assert row["n_scheduled"] + row["n_backlog"] + row["n_cancelled"] == 10  # type: ignore[operator]

    def test_the_horizon_bounds_are_reported(self) -> None:
        row = summary_row(make_plan([3, 3, 3], k=9, median=3))
        assert row["horizon_days"] == 3
        assert row["horizon_start_date"] == date(2026, 4, 1)
        assert row["horizon_end_date"] == date(2026, 4, 3)
        assert row["horizon_slots"] == 9

    def test_utilization_is_scheduled_over_supplied(self) -> None:
        row = summary_row(make_plan([9], k=3, median=3))
        assert row["capacity_utilization"] == pytest.approx(3 / 9)


class TestNoModelMetricAppears:
    def test_the_measurement_surface_carries_no_ml_metric(self) -> None:
        """No precision, no capture, no NDE, no lift.

        Component 5 owns discovery metrics and Component 13 already reports them per policy. A
        scheduling table carrying one would invite exactly one conclusion -- that a schedule
        improved a model -- and it cannot: the set of inspections never changes here.
        """
        plan = make_plan([4, 5], k=9, median=5)
        keys = set(preservation_row(plan, make_queue(9))) | set(summary_row(plan))
        # Matched on whole underscore-separated tokens. A substring test would fire on
        # "recommended", which contains "nde", and a test that cries wolf gets deleted.
        tokens = {token for key in keys for token in key.lower().split("_")}
        banned = {
            "precision",
            "capture",
            "nde",
            "lift",
            "recall",
            "auc",
            "roc",
            "target",
            "positives",
            "label",
        }
        assert not tokens & banned
