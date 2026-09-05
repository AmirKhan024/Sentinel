"""Time-invariance sensitivity: turning an assumption into a measurement.

The re-ordering simulation assumes an establishment cited on 14 June would also
have been cited on 2 May. The published audit of Chicago's 2015 model named that
assumption as a flaw, because temperature-related violations are seasonal.

Two properties are tested hardest.

**De-trending works.** This dataset's base rate falls from 0.876 to 0.391 for
reasons unrelated to the seasons, so a naive month effect would mostly measure
that drift. A pure secular trend with no seasonal component must therefore
produce month effects near zero, and a pure seasonal pattern with no trend must
be recovered intact.

**The coupling preserves the observed world.** When a schedule leaves a row on
its own date, its re-drawn label must equal the observed one exactly -- so
business as usual is untouched by the sensitivity analysis and the comparison
stays fair.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from sentinel.evaluation.models import ScheduleName
from sentinel.evaluation.sensitivity import (
    DEFAULT_REPLICATIONS,
    DEFAULT_SENSITIVITY_SEED,
    TEMPERATURE_STATUS,
    month_effects,
    redraw_sensitivity,
    seasonal_amplitude,
)
from sentinel.evaluation.simulate import (
    build_window,
    business_as_usual_order,
    evaluate_schedule,
    model_order,
    optimal_order,
)


def _frame(rows: list[tuple[date, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {"rd": [d for d, _ in rows], "target": [t for _, t in rows]},
        schema={"rd": pl.Date, "target": pl.Int8},
    )


def _year_of(year: int, *, rate_by_month: dict[int, float], per_month: int = 100):
    """Deterministic rows: ``rate_by_month[m] * per_month`` positives each month."""
    rows: list[tuple[date, int]] = []
    for month in range(1, 13):
        positives = round(rate_by_month[month] * per_month)
        for i in range(per_month):
            rows.append((date(year, month, 1) + timedelta(days=i % 27), 1 if i < positives else 0))
    return rows


# --- 1. de-trending ---------------------------------------------------------


def test_a_pure_secular_trend_produces_no_seasonal_effect() -> None:
    """The whole point of the year term. Three years, each flat within itself,
    but with rates 0.9, 0.6 and 0.3 -- a large drift and zero seasonality.
    """
    rows: list[tuple[date, int]] = []
    for year, rate in ((2019, 0.9), (2020, 0.6), (2021, 0.3)):
        rows.extend(_year_of(year, rate_by_month=dict.fromkeys(range(1, 13), rate)))
    effects = month_effects(_frame(rows))

    assert len(effects) == 12
    for effect in effects:
        assert abs(effect.log_odds_effect) < 1e-9
    assert seasonal_amplitude(effects) == pytest.approx(0.0, abs=1e-6)


def test_a_pure_seasonal_pattern_is_recovered() -> None:
    """Summer high, winter low, no trend: the effect must have the right sign."""
    seasonal = {m: (0.7 if m in (6, 7, 8) else 0.3) for m in range(1, 13)}
    rows: list[tuple[date, int]] = []
    for year in (2019, 2020, 2021):
        rows.extend(_year_of(year, rate_by_month=seasonal))
    effects = {e.month: e for e in month_effects(_frame(rows))}

    assert effects[7].log_odds_effect > 0
    assert effects[1].log_odds_effect < 0
    assert effects[7].log_odds_effect > effects[1].log_odds_effect


def test_seasonality_is_detected_through_a_simultaneous_trend() -> None:
    """The realistic case: a strong drift *and* a seasonal component together."""
    rows: list[tuple[date, int]] = []
    for year, base in ((2019, 0.8), (2020, 0.6), (2021, 0.4)):
        rates = {m: min(0.95, base + (0.1 if m in (6, 7, 8) else -0.05)) for m in range(1, 13)}
        rows.extend(_year_of(year, rate_by_month=rates))
    effects = {e.month: e for e in month_effects(_frame(rows))}
    assert effects[7].log_odds_effect > 0
    assert effects[2].log_odds_effect < 0


def test_an_empty_frame_produces_no_effects_rather_than_an_error() -> None:
    assert month_effects(_frame([])) == []
    assert seasonal_amplitude([]) is None


def test_a_month_absent_from_the_data_is_omitted_not_invented() -> None:
    rows = [(date(2020, 1, 5), 1), (date(2020, 1, 6), 0), (date(2020, 2, 5), 1)]
    months = {e.month for e in month_effects(_frame(rows))}
    assert months == {1, 2}


def test_the_amplitude_is_a_peak_to_trough_spread_in_percentage_points() -> None:
    seasonal = {m: (0.7 if m in (6, 7, 8) else 0.3) for m in range(1, 13)}
    rows: list[tuple[date, int]] = []
    for year in (2019, 2020):
        rows.extend(_year_of(year, rate_by_month=seasonal))
    amplitude = seasonal_amplitude(month_effects(_frame(rows)))
    assert amplitude is not None
    assert amplitude > 10  # a large, real seasonal swing


# --- 2. the label re-draw ---------------------------------------------------


def _window(n: int = 40):
    ids = [f"{i:04d}" for i in range(n)]
    labels = [1 if i % 2 == 0 else 0 for i in range(n)]
    dates = [date(2022, 1, 1) + timedelta(days=i * 9) for i in range(n)]
    return build_window(ids, labels, dates)


def _effects():
    seasonal = {m: (0.7 if m in (6, 7, 8) else 0.3) for m in range(1, 13)}
    rows: list[tuple[date, int]] = []
    for year in (2019, 2020, 2021):
        rows.extend(_year_of(year, rate_by_month=seasonal))
    return month_effects(_frame(rows))


def test_business_as_usual_labels_are_never_re_drawn() -> None:
    """The coupling's defining property: a row left on its own date keeps its
    label exactly, so the comparator is untouched by the sensitivity analysis.
    """
    window = _window()
    band = redraw_sensitivity(
        window, business_as_usual_order(window), _effects(), replications=200, seed=1
    )
    assert band.label_flip_rate == 0.0


def test_moving_rows_across_the_calendar_does_flip_some_labels() -> None:
    window = _window()
    band = redraw_sensitivity(window, optimal_order(window), _effects(), replications=200, seed=1)
    assert band.label_flip_rate is not None
    assert band.label_flip_rate > 0.0


def test_the_result_is_a_distribution_not_a_point_estimate() -> None:
    window = _window()
    order = model_order(window, [float(i % 7) for i in range(window.n)])
    observed = evaluate_schedule(
        window, order, schedule=ScheduleName.MODEL, label="m"
    ).normalized_discovery_efficiency
    band = redraw_sensitivity(
        window, order, _effects(), replications=300, seed=1, observed_nde=observed
    )
    assert band.p05 is not None and band.p95 is not None and band.mean is not None
    assert band.p05 <= band.p50 <= band.p95  # type: ignore[operator]
    assert band.std is not None and band.std >= 0.0
    assert band.observed == observed


def test_the_band_is_reproducible_from_its_seed() -> None:
    window = _window()
    order = optimal_order(window)
    first = redraw_sensitivity(window, order, _effects(), replications=100, seed=99)
    second = redraw_sensitivity(window, order, _effects(), replications=100, seed=99)
    assert first == second


def test_a_different_seed_produces_a_different_draw() -> None:
    window = _window()
    order = optimal_order(window)
    first = redraw_sensitivity(window, order, _effects(), replications=100, seed=1)
    second = redraw_sensitivity(window, order, _effects(), replications=100, seed=2)
    assert first.mean != second.mean


def test_the_replication_count_is_recorded() -> None:
    window = _window()
    band = redraw_sensitivity(window, optimal_order(window), _effects(), replications=17, seed=1)
    assert band.replications == 17


def test_zero_replications_is_rejected() -> None:
    window = _window()
    with pytest.raises(ValueError, match="at least 1"):
        redraw_sensitivity(window, optimal_order(window), _effects(), replications=0)


def test_with_no_measured_seasonality_there_is_nothing_to_re_draw() -> None:
    window = _window()
    band = redraw_sensitivity(window, optimal_order(window), [], replications=100)
    assert band.mean is None
    assert band.replications == 0


def test_an_empty_window_produces_an_empty_band() -> None:
    window = build_window([], [], [])
    band = redraw_sensitivity(window, [], _effects(), replications=10)
    assert band.mean is None


# --- 3. what is blocked -----------------------------------------------------


def test_the_temperature_limitation_is_stated_in_the_module_not_only_in_a_document() -> None:
    """The limitation must travel attached to the number, not live in a report."""
    assert "BLOCKED" in TEMPERATURE_STATUS
    assert "NOAA" in TEMPERATURE_STATUS
    assert "Component 1" in TEMPERATURE_STATUS


def test_the_defaults_match_the_project_specification() -> None:
    assert DEFAULT_REPLICATIONS == 1000
    assert isinstance(DEFAULT_SENSITIVITY_SEED, int)
