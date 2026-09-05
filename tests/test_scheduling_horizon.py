"""The planning horizon: which days, how many slots, and where the numbers come from.

The two capacity modes must produce the **same calendar** and differ only in slot counts. If a
mode changed the horizon as well, the two would not be comparable and the measurement the whole
component exists to make -- what the real calendar costs against the assumed one -- would be
confounded by a second difference nobody asked for.
"""

from __future__ import annotations

from datetime import date

import pytest

from sentinel.scheduling.definitions import CapacityMode
from sentinel.scheduling.horizon import (
    HorizonError,
    build_horizon,
    clamp_detail,
    observed_calendar_from_dates,
)
from sentinel.scheduling.models import Horizon, OperatingDay

from .conftest import make_calendar, make_horizon


def _build(
    counts: list[int], *, k: int, median: int = 5, mode: CapacityMode | None = None
) -> Horizon:
    return build_horizon(
        fold_set="quarterly",
        fold_id="quarterly-2026Q2",
        k_name="k_1_week",
        k=k,
        median_daily_capacity=median,
        calendar=make_calendar(counts),
        capacity_mode=mode or CapacityMode.OBSERVED_CALENDAR,
    )


class TestTheObservedCalendar:
    def test_slot_counts_are_the_observed_volumes(self) -> None:
        horizon = _build([3, 7, 2, 9], k=20, median=5)
        assert [day.n_slots for day in horizon.days] == [3, 7, 2, 9]

    def test_the_capacity_source_names_the_observation(self) -> None:
        horizon = _build([3, 7], k=10, median=5)
        assert all(day.capacity_source == "observed_inspection_count" for day in horizon.days)

    def test_the_calendar_is_derived_by_counting_real_dates(self) -> None:
        dates = [date(2026, 4, 2), date(2026, 4, 1), date(2026, 4, 2), date(2026, 4, 3)]
        assert observed_calendar_from_dates(dates) == (
            (date(2026, 4, 1), 1),
            (date(2026, 4, 2), 2),
            (date(2026, 4, 3), 1),
        )

    def test_a_weekend_date_survives_because_nothing_generates_the_calendar(self) -> None:
        """2026-04-04 is a Saturday. The dataset holds weekend inspections and so does this.

        A synthesised Monday-to-Friday calendar would silently drop them, and would need a
        holiday list this project has no way to verify.
        """
        saturday = date(2026, 4, 4)
        calendar = observed_calendar_from_dates([saturday, saturday])
        assert calendar == ((saturday, 2),)


class TestTheFlatMedianScenario:
    def test_every_day_carries_the_window_median(self) -> None:
        horizon = _build([3, 7, 2], k=15, median=5, mode=CapacityMode.FLAT_MEDIAN)
        assert [day.n_slots for day in horizon.days] == [5, 5, 5]

    def test_the_capacity_source_names_the_assumption(self) -> None:
        horizon = _build([3, 7], k=10, median=5, mode=CapacityMode.FLAT_MEDIAN)
        assert all(day.capacity_source == "test_median_daily_capacity" for day in horizon.days)

    def test_the_scenario_is_saturated_by_construction_at_a_day_denominated_cutoff(self) -> None:
        """The tautology, asserted so it is a documented property rather than a surprise.

        ``k_1_week`` is five median days, the horizon is five days, so the scenario supplies
        exactly k slots -- a backlog of zero and a utilisation of exactly 1.0 before anything is
        measured. That is precisely why the observed calendar is the default.
        """
        horizon = _build([1, 1, 1, 1, 1], k=25, median=5, mode=CapacityMode.FLAT_MEDIAN)
        assert horizon.total_slots == horizon.k == 25

    def test_the_observed_calendar_is_not_saturated_on_the_same_cell(self) -> None:
        horizon = _build([1, 1, 1, 1, 1], k=25, median=5)
        assert horizon.total_slots == 5
        assert horizon.total_slots < horizon.k


class TestTheModesShareOneCalendar:
    def test_both_modes_span_the_same_days(self) -> None:
        """A mode change moves capacity, never the calendar."""
        counts = [3, 7, 2, 9, 4]
        observed = _build(counts, k=20, median=5)
        flat = _build(counts, k=20, median=5, mode=CapacityMode.FLAT_MEDIAN)
        assert [day.slot_date for day in observed.days] == [day.slot_date for day in flat.days]
        assert observed.n_days == flat.n_days


class TestHorizonLength:
    @pytest.mark.parametrize(
        ("k", "median", "expected_days"),
        [(5, 5, 1), (10, 5, 2), (11, 5, 3), (25, 5, 5), (1, 5, 1)],
    )
    def test_length_follows_the_rule(self, k: int, median: int, expected_days: int) -> None:
        horizon = _build([9] * 10, k=k, median=median)
        assert horizon.n_days == expected_days

    def test_the_horizon_is_a_prefix_of_the_calendar(self) -> None:
        horizon = _build([9, 9, 9, 9, 9, 9], k=10, median=5)
        assert [day.slot_date for day in horizon.days] == [date(2026, 4, 1), date(2026, 4, 2)]

    def test_day_indices_are_contiguous_from_one(self) -> None:
        horizon = _build([9] * 6, k=25, median=5)
        assert [day.day_index for day in horizon.days] == [1, 2, 3, 4, 5]


class TestClamping:
    def test_a_horizon_longer_than_the_calendar_is_clamped_and_flagged(self) -> None:
        horizon = _build([4, 4], k=100, median=5)
        assert horizon.n_days == 2
        assert horizon.was_clamped

    def test_an_ordinary_horizon_is_not_flagged(self) -> None:
        assert not _build([4, 4, 4], k=10, median=5).was_clamped

    def test_clamp_detail_is_none_when_the_rule_fits(self) -> None:
        assert clamp_detail(10, 5, 60) is None

    def test_clamp_detail_explains_the_consequence(self) -> None:
        detail = clamp_detail(1000, 5, 60)
        assert detail is not None
        assert "trivially saturated" in detail


class TestRefusals:
    def test_an_empty_calendar_is_refused(self) -> None:
        with pytest.raises(HorizonError, match="no operating days"):
            build_horizon(
                fold_set="quarterly",
                fold_id="quarterly-2026Q2",
                k_name="k_1_week",
                k=10,
                median_daily_capacity=5,
                calendar=(),
                capacity_mode=CapacityMode.OBSERVED_CALENDAR,
            )

    def test_an_unsorted_calendar_is_refused(self) -> None:
        """Placement walks the days in order; an unsorted calendar would place rows arbitrarily."""
        with pytest.raises(HorizonError, match="ascending"):
            build_horizon(
                fold_set="quarterly",
                fold_id="quarterly-2026Q2",
                k_name="k_1_week",
                k=10,
                median_daily_capacity=5,
                calendar=((date(2026, 4, 3), 4), (date(2026, 4, 1), 4)),
                capacity_mode=CapacityMode.OBSERVED_CALENDAR,
            )

    def test_a_zero_volume_day_is_refused(self) -> None:
        """A day with no inspections is not an operating day; it arrived through a defect."""
        with pytest.raises(HorizonError, match="not an operating day"):
            build_horizon(
                fold_set="quarterly",
                fold_id="quarterly-2026Q2",
                k_name="k_1_week",
                k=10,
                median_daily_capacity=5,
                calendar=((date(2026, 4, 1), 0),),
                capacity_mode=CapacityMode.OBSERVED_CALENDAR,
            )

    def test_a_repeated_date_is_refused_by_the_horizon_type(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            Horizon(
                fold_set="quarterly",
                fold_id="f",
                k_name="k_1_week",
                k=2,
                median_daily_capacity=1,
                capacity_mode="observed_calendar",
                days=(
                    OperatingDay(1, date(2026, 4, 1), 1, "observed_inspection_count"),
                    OperatingDay(2, date(2026, 4, 1), 1, "observed_inspection_count"),
                ),
            )

    def test_a_gapped_day_index_is_refused_by_the_horizon_type(self) -> None:
        with pytest.raises(ValueError, match="contiguous"):
            Horizon(
                fold_set="quarterly",
                fold_id="f",
                k_name="k_1_week",
                k=2,
                median_daily_capacity=1,
                capacity_mode="observed_calendar",
                days=(
                    OperatingDay(1, date(2026, 4, 1), 1, "observed_inspection_count"),
                    OperatingDay(3, date(2026, 4, 2), 1, "observed_inspection_count"),
                ),
            )


class TestHorizonArithmetic:
    def test_total_slots_sums_the_days(self) -> None:
        assert make_horizon([3, 7, 2], k=15, median=5).total_slots == 12

    def test_cumulative_slots_runs_forward(self) -> None:
        assert make_horizon([3, 7, 2], k=15, median=5).cumulative_slots == (3, 10, 12)

    def test_day_for_position_locates_a_rank(self) -> None:
        horizon = make_horizon([3, 7, 2], k=15, median=5)
        assert horizon.day_for_position(1).day_index == 1  # type: ignore[union-attr]
        assert horizon.day_for_position(3).day_index == 1  # type: ignore[union-attr]
        assert horizon.day_for_position(4).day_index == 2  # type: ignore[union-attr]
        assert horizon.day_for_position(12).day_index == 3  # type: ignore[union-attr]

    def test_day_for_position_returns_none_past_the_horizon(self) -> None:
        assert make_horizon([3], k=5, median=5).day_for_position(99) is None

    def test_the_start_and_end_dates_bound_the_horizon(self) -> None:
        horizon = make_horizon([3, 7, 2], k=15, median=5)
        assert horizon.start_date == date(2026, 4, 1)
        assert horizon.end_date == date(2026, 4, 3)
