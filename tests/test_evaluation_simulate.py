"""The re-ordering simulation: the claims the headline numbers rest on.

Four of these are load-bearing and everything else in the project depends on
them holding.

**The business-as-usual identity.** Filling slots in date order returns every
inspection to its own real date. If that ever stops being true, "days earlier
than business as usual" stops meaning "days earlier than what really happened",
and the project's central result quietly changes meaning.

**Capacity conservation.** Every schedule consumes the same slots. If it did
not, a schedule could win by inspecting more rather than by inspecting better,
which is a different intervention from the one Sentinel proposes.

**The analytic bounds.** Optimal is exactly 1, worst is exactly -1, random is 0
in expectation. These are derived rather than sampled, so they are exact and a
drift of even 1e-9 means the area formula has changed.

**Labels never move.** The simulation reorders; it does not relabel. Asserted
directly, because a causal simulator is a different and much weaker thing.
"""

from __future__ import annotations

import statistics
from datetime import date, timedelta

import pytest

from sentinel.evaluation.models import ScheduleName
from sentinel.evaluation.simulate import (
    DEFAULT_RANDOM_SEED,
    SimulationError,
    analytic_optimal_area,
    analytic_random_area,
    build_window,
    business_as_usual_order,
    capacity_k_values,
    days_earlier,
    discovery_curve,
    evaluate_schedule,
    first_half_discovery,
    model_order,
    normalized_area,
    normalized_discovery_efficiency,
    optimal_order,
    random_order,
    simulated_dates,
    worst_order,
)

START = date(2022, 4, 1)


def _window(labels: list[int], *, per_day: int = 1):
    """A window with ``len(labels)`` inspections, ``per_day`` on each date."""
    ids = [f"{i:04d}" for i in range(len(labels))]
    dates = [START + timedelta(days=i // per_day) for i in range(len(labels))]
    return build_window(ids, labels, dates)


# --- 1. the business-as-usual identity -------------------------------------


def test_business_as_usual_reproduces_the_observed_dates_exactly() -> None:
    """The identity the whole days-earlier metric depends on."""
    window = _window([1, 0, 1, 0, 0, 1, 0])
    assert simulated_dates(window, business_as_usual_order(window)) == list(window.dates)


def test_the_identity_holds_when_many_inspections_share_a_date() -> None:
    """Real capacity is ~29 a day, so the interesting case is heavy ties."""
    window = _window([1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1], per_day=4)
    assert simulated_dates(window, business_as_usual_order(window)) == list(window.dates)


def test_business_as_usual_moves_nobody_so_days_earlier_is_all_zero() -> None:
    window = _window([1, 0, 1, 1, 0, 0])
    measured = days_earlier(window, business_as_usual_order(window))
    assert measured.mean == 0.0
    assert measured.fraction_unchanged == 1.0
    assert measured.fraction_improved == 0.0
    assert measured.fraction_worse == 0.0


# --- 2. capacity conservation ----------------------------------------------


@pytest.mark.parametrize("schedule", ["optimal", "worst", "random", "model"])
def test_every_schedule_consumes_exactly_the_observed_slots(schedule: str) -> None:
    window = _window([1, 0, 1, 0, 0, 1, 1, 0], per_day=3)
    orders = {
        "optimal": optimal_order(window),
        "worst": worst_order(window),
        "random": random_order(window, seed=DEFAULT_RANDOM_SEED),
        "model": model_order(window, [0.5, 0.9, 0.1, 0.7, 0.2, 0.8, 0.3, 0.4]),
    }
    assert sorted(simulated_dates(window, orders[schedule])) == sorted(window.dates)


def test_the_number_of_slots_equals_the_number_of_inspections() -> None:
    window = _window([1, 0, 1, 0, 0], per_day=2)
    assert len(window.slots) == window.n == 5


def test_an_order_that_is_not_a_permutation_is_rejected() -> None:
    window = _window([1, 0, 1])
    with pytest.raises(SimulationError, match="permutation"):
        simulated_dates(window, [0, 0, 1])


def test_a_score_list_of_the_wrong_length_is_rejected() -> None:
    window = _window([1, 0, 1])
    with pytest.raises(SimulationError, match="expected 3 scores"):
        model_order(window, [0.1, 0.2])


# --- 3. the analytic bounds -------------------------------------------------


def test_the_optimal_schedule_has_efficiency_one() -> None:
    """Exact to floating-point tolerance: the area is a sum over n trapezoids,
    so the bound is analytic but its evaluation accumulates rounding."""
    window = _window([1, 0, 0, 1, 0, 1, 0, 0, 1, 0])
    result = evaluate_schedule(
        window, optimal_order(window), schedule=ScheduleName.OPTIMAL, label="optimal"
    )
    assert result.normalized_discovery_efficiency == pytest.approx(1.0, abs=1e-12)


def test_the_worst_schedule_has_efficiency_minus_one() -> None:
    window = _window([1, 0, 0, 1, 0, 1, 0, 0, 1, 0])
    result = evaluate_schedule(
        window, worst_order(window), schedule=ScheduleName.WORST, label="worst"
    )
    assert result.normalized_discovery_efficiency == pytest.approx(-1.0, abs=1e-12)


def test_the_bounds_hold_across_many_shapes_of_window() -> None:
    for n, positives in ((10, 1), (10, 9), (50, 25), (100, 3), (7, 4)):
        labels = [1] * positives + [0] * (n - positives)
        window = _window(labels)
        optimal = evaluate_schedule(
            window, optimal_order(window), schedule=ScheduleName.OPTIMAL, label="o"
        )
        worst = evaluate_schedule(
            window, worst_order(window), schedule=ScheduleName.WORST, label="w"
        )
        assert optimal.normalized_discovery_efficiency == pytest.approx(1.0, abs=1e-12)
        assert worst.normalized_discovery_efficiency == pytest.approx(-1.0, abs=1e-12)


def test_the_analytic_random_area_is_exactly_one_half() -> None:
    """Derived, not sampled: a uniformly random permutation's expected curve is
    the diagonal, whose area is 0.5 regardless of n or prevalence."""
    assert analytic_random_area() == 0.5


def test_the_analytic_optimal_area_matches_the_measured_one() -> None:
    for n, positives in ((10, 4), (25, 7), (100, 51)):
        labels = [1] * positives + [0] * (n - positives)
        window = _window(labels)
        cumulative = discovery_curve(window, optimal_order(window))
        measured = normalized_area(cumulative, n=n, positives=positives)
        assert measured == pytest.approx(analytic_optimal_area(n=n, positives=positives), abs=1e-12)


def test_random_schedules_average_to_an_efficiency_near_zero() -> None:
    """The empirical check on the analytic denominator."""
    window = _window([1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0])
    values = [
        evaluate_schedule(
            window, random_order(window, seed=s), schedule=ScheduleName.RANDOM, label="r", seed=s
        ).normalized_discovery_efficiency
        for s in range(500)
    ]
    assert statistics.mean([v for v in values if v is not None]) == pytest.approx(0.0, abs=0.02)


# --- 4. the discovery curve -------------------------------------------------


def test_the_curve_starts_at_zero_and_ends_at_every_positive() -> None:
    window = _window([1, 0, 1, 1, 0])
    cumulative = discovery_curve(window, business_as_usual_order(window))
    assert len(cumulative) == window.n + 1
    assert cumulative[0] == 0
    assert cumulative[-1] == window.positives == 3


def test_the_curve_never_decreases() -> None:
    window = _window([1, 0, 1, 0, 0, 1, 1])
    cumulative = discovery_curve(window, random_order(window, seed=7))
    assert all(b >= a for a, b in zip(cumulative, cumulative[1:], strict=False))


def test_the_optimal_curve_climbs_every_slot_until_the_positives_run_out() -> None:
    window = _window([1, 1, 1, 0, 0, 0, 0])
    cumulative = discovery_curve(window, optimal_order(window))
    assert cumulative == [0, 1, 2, 3, 3, 3, 3, 3]


def test_first_half_discovery_is_one_for_optimal_and_zero_for_worst() -> None:
    window = _window([1, 0, 1, 0, 1, 0, 0, 0])
    optimal_curve = discovery_curve(window, optimal_order(window))
    worst_curve = discovery_curve(window, worst_order(window))
    assert first_half_discovery(optimal_curve, n=window.n, positives=window.positives) == 1.0
    assert first_half_discovery(worst_curve, n=window.n, positives=window.positives) == 0.0


# --- 5. days earlier --------------------------------------------------------


def test_days_earlier_is_computed_against_the_real_dates() -> None:
    """One positive on day 4 moved to day 0 is four days earlier, exactly."""
    window = _window([0, 0, 0, 0, 1])
    order = optimal_order(window)
    measured = days_earlier(window, order, positives_only=True)
    assert measured.count == 1
    assert measured.mean == 4.0
    assert measured.fraction_improved == 1.0


def test_days_earlier_reports_the_rows_made_worse_not_only_the_mean() -> None:
    """The 2015 precedent reported 7.438 days with SD 25.156 and never said how
    many establishments were found later. This asserts the fractions exist."""
    window = _window([1, 0, 0, 0, 1])
    measured = days_earlier(window, worst_order(window), positives_only=True)
    assert measured.fraction_worse is not None
    assert measured.fraction_worse > 0
    assert measured.mean is not None
    assert measured.mean < 0


def test_days_earlier_over_all_rows_differs_from_positives_only() -> None:
    window = _window([1, 0, 0, 1, 0, 0])
    order = optimal_order(window)
    positives = days_earlier(window, order, positives_only=True)
    everything = days_earlier(window, order, positives_only=False)
    assert positives.count == 2
    assert everything.count == 6
    assert everything.mean == pytest.approx(0.0)  # a permutation moves rows both ways


def test_days_earlier_on_a_window_with_no_positives_is_empty_not_zero() -> None:
    window = _window([0, 0, 0])
    measured = days_earlier(window, optimal_order(window), positives_only=True)
    assert measured.count == 0
    assert measured.mean is None
    assert measured.fraction_improved is None


def test_the_quartiles_bracket_the_median() -> None:
    window = _window([1, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    measured = days_earlier(window, optimal_order(window))
    assert measured.p25 is not None and measured.median is not None and measured.p75 is not None
    assert measured.p25 <= measured.median <= measured.p75
    assert measured.minimum <= measured.p25  # type: ignore[operator]
    assert measured.maximum >= measured.p75  # type: ignore[operator]


# --- 6. labels never move ---------------------------------------------------


def test_reordering_does_not_change_a_single_label() -> None:
    """A reordering simulation, not a causal simulator."""
    window = _window([1, 0, 1, 0, 0, 1])
    before = list(window.labels)
    for order in (
        optimal_order(window),
        worst_order(window),
        random_order(window, seed=3),
        model_order(window, [0.2, 0.9, 0.4, 0.1, 0.8, 0.3]),
    ):
        evaluate_schedule(window, order, schedule=ScheduleName.MODEL, label="m")
    assert list(window.labels) == before


def test_the_positive_count_is_identical_under_every_schedule() -> None:
    window = _window([1, 0, 1, 0, 0, 1, 1])
    for order in (optimal_order(window), worst_order(window), random_order(window, seed=1)):
        assert discovery_curve(window, order)[-1] == window.positives


# --- 7. score direction and tie-breaking ------------------------------------


def test_a_higher_score_is_inspected_first() -> None:
    window = _window([0, 1])
    order = model_order(window, [0.10, 0.90])
    assert window.ids[order[0]] == "0001"


def test_tied_scores_break_on_the_identifier_deterministically() -> None:
    window = _window([1, 0, 1, 0])
    order = model_order(window, [0.5, 0.5, 0.5, 0.5])
    assert [window.ids[i] for i in order] == ["0000", "0001", "0002", "0003"]


def test_a_constant_scorer_produces_the_same_result_every_run() -> None:
    window = _window([1, 0, 1, 0, 1])
    first = evaluate_schedule(
        window, model_order(window, [0.0] * 5), schedule=ScheduleName.MODEL, label="c"
    )
    second = evaluate_schedule(
        window, model_order(window, [0.0] * 5), schedule=ScheduleName.MODEL, label="c"
    )
    assert first.order == second.order
    assert first.cumulative == second.cumulative


def test_random_order_is_reproducible_from_its_seed() -> None:
    window = _window([1, 0, 1, 0, 1, 0])
    assert random_order(window, seed=99) == random_order(window, seed=99)
    assert random_order(window, seed=99) != random_order(window, seed=100)


def test_window_construction_is_independent_of_input_row_order() -> None:
    ids = ["c", "a", "b"]
    labels = [1, 0, 1]
    dates = [START + timedelta(days=2), START, START + timedelta(days=1)]
    forward = build_window(ids, labels, dates)
    backward = build_window(list(reversed(ids)), list(reversed(labels)), list(reversed(dates)))
    assert forward == backward


# --- 8. degenerate windows --------------------------------------------------


def test_a_window_with_no_positives_has_no_efficiency() -> None:
    """Undefined, not zero: with nothing to separate, every schedule is identical."""
    window = _window([0, 0, 0, 0])
    result = evaluate_schedule(
        window, optimal_order(window), schedule=ScheduleName.OPTIMAL, label="o"
    )
    assert result.normalized_discovery_efficiency is None
    assert result.first_half_discovery is None
    assert result.area is None


def test_a_window_where_everything_is_positive_has_no_efficiency() -> None:
    window = _window([1, 1, 1, 1])
    result = evaluate_schedule(
        window, optimal_order(window), schedule=ScheduleName.OPTIMAL, label="o"
    )
    assert result.normalized_discovery_efficiency is None


def test_a_single_inspection_window_is_handled() -> None:
    window = _window([1])
    result = evaluate_schedule(
        window, business_as_usual_order(window), schedule=ScheduleName.BUSINESS_AS_USUAL, label="b"
    )
    assert result.cumulative == (0, 1)
    assert result.normalized_discovery_efficiency is None  # positives == n


def test_duplicate_identifiers_in_a_window_are_rejected() -> None:
    with pytest.raises(SimulationError, match="unique"):
        build_window(["a", "a"], [1, 0], [START, START])


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(SimulationError, match="same length"):
        build_window(["a", "b"], [1], [START, START])


def test_normalized_efficiency_is_none_when_the_area_is_none() -> None:
    assert normalized_discovery_efficiency(None, n=10, positives=5) is None


# --- 9. capacity-derived k --------------------------------------------------


def test_k_values_are_derived_from_measured_capacity() -> None:
    window = _window([1, 0] * 100)
    k = capacity_k_values(window, median_daily=29)
    assert k["k_1_day"] == 29
    assert k["k_1_week"] == 29 * 5
    assert k["k_1_month"] == min(29 * 21, window.n)


def test_k_never_exceeds_the_window_size() -> None:
    window = _window([1, 0, 1])
    for value in capacity_k_values(window, median_daily=29).values():
        assert 1 <= value <= window.n


def test_a_capacity_below_one_is_rejected() -> None:
    window = _window([1, 0])
    with pytest.raises(SimulationError, match="at least 1"):
        capacity_k_values(window, median_daily=0)
