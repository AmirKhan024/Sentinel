"""Time-invariance sensitivity -- audit Finding 2, measured rather than assumed.

The re-ordering simulation quietly assumes **time invariance**: that an
establishment cited on 14 June would also have been cited had it been inspected
on 2 May. Kannan, Shapiro and Bilgic named exactly this as a flaw in the 2015
Chicago evaluation, because temperature-related violations spike in summer, so
moving an inspection across the calendar changes the probability of citation.

That is a measurement, not an assumption, and this module makes it one.

**De-trending is the whole difficulty.** The raw citation rate by month is
useless here, because this dataset's base rate falls from 0.876 in 2018 to 0.391
in 2026 for reasons that have nothing to do with the seasons. A naive month
effect would mostly be reading that drift. So the model is fitted on each
month's deviation **from its own year**, in log-odds, weighted by cell size:

.. math::

    \\text{effect}_m = \\frac{\\sum_y n_{ym}\\,
        (\\operatorname{logit} p_{ym} - \\operatorname{logit} p_{y})}
        {\\sum_y n_{ym}}

The year term absorbs the secular trend; what is left is seasonal.

**Re-drawing labels uses a monotone coupling.** For each row draw one uniform
``u``, conditioned on the label actually observed: if the establishment was
cited, ``u`` is drawn below its real-date probability; if it was not, above.
The counterfactual label at the simulated date is then ``1`` when ``u`` falls
below the simulated-date probability. Two properties make this the right
construction:

* when a schedule leaves a row on its own date, the label is unchanged exactly,
  so business-as-usual is untouched and the comparison stays fair;
* a positive moved from a high-risk month to a low-risk one can flip to zero,
  and a negative moved the other way can flip to one, in proportion to the
  measured effect and nothing else.

**What is BLOCKED.** The project spec asks for citation rate against day-of-year
*and temperature*, controlling for chain. No weather data is ingested anywhere
in this repository, and Component 5 does not fabricate a data source. So the
temperature covariate is unavailable and the seasonal effect measured here is a
*proxy* that will contain temperature, daylight, holiday scheduling and staffing
patterns without separating them. Ingesting NOAA GHCN station USW00094846 is a
Component 1 extension; this module's interface takes the covariate the moment it
exists.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import polars as pl

from sentinel.evaluation.simulate import (
    Window,
    normalized_area,
    normalized_discovery_efficiency,
)

logger = logging.getLogger(__name__)

#: Replications of the label re-draw. The project spec asks for 1,000.
DEFAULT_REPLICATIONS = 1000

#: Seed for the re-draw. Recorded in the manifest; the whole distribution is
#: reproducible from it.
DEFAULT_SENSITIVITY_SEED = 20260816

#: Rates are clamped away from 0 and 1 before taking a logit, so a small cell
#: that happened to be all-positive does not contribute an infinite effect.
RATE_CLAMP = 1e-6

#: Stated in the manifest and the contract so the limitation travels with the
#: number rather than living only in a findings document.
TEMPERATURE_STATUS = (
    "BLOCKED: no weather data is ingested. The seasonal effect is a proxy that "
    "confounds temperature with daylight, holidays and staffing. Separating them "
    "requires NOAA GHCN USW00094846, a Component 1 extension."
)


def _logit(p: float) -> float:
    clamped = min(max(p, RATE_CLAMP), 1.0 - RATE_CLAMP)
    return math.log(clamped / (1.0 - clamped))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True, slots=True)
class MonthEffect:
    """One month's de-trended seasonal effect on the log-odds of citation."""

    month: int
    rows: int
    raw_rate: float
    log_odds_effect: float
    #: The effect expressed as a percentage-point shift applied at the overall
    #: base rate, which is the form a human can actually read.
    pct_point_shift_at_base: float


def month_effects(
    frame: pl.DataFrame,
    *,
    date_column: str = "rd",
    label_column: str = "target",
) -> list[MonthEffect]:
    """Fit the de-trended month-of-year effect on citation rate.

    Only the rows passed in are used, so a caller fitting per fold can pass that
    fold's **training** window and keep the test period unseen. That matters:
    fitting the seasonal model on the test period would be the same class of
    error this component exists to prevent.
    """
    if frame.height == 0:
        return []

    prepared = frame.select(
        pl.col(date_column).dt.year().alias("yr"),
        pl.col(date_column).dt.month().alias("mo"),
        pl.col(label_column).cast(pl.Float64).alias("y"),
    )
    yearly = prepared.group_by("yr").agg(pl.col("y").mean().alias("year_rate"))
    monthly = (
        prepared.group_by(["yr", "mo"])
        .agg(pl.col("y").mean().alias("month_rate"), pl.len().alias("n"))
        .join(yearly, on="yr", how="inner")
    )

    mean_label = prepared["y"].mean()
    overall = float(mean_label) if isinstance(mean_label, int | float) else 0.0
    base_logit = _logit(overall)

    effects: list[MonthEffect] = []
    for month in range(1, 13):
        cells = monthly.filter(pl.col("mo") == month)
        if cells.height == 0:
            continue
        weight_total = 0.0
        weighted_effect = 0.0
        positives = 0.0
        rows = 0
        for cell in cells.iter_rows(named=True):
            n = float(cell["n"])
            deviation = _logit(float(cell["month_rate"])) - _logit(float(cell["year_rate"]))
            weighted_effect += n * deviation
            weight_total += n
            positives += n * float(cell["month_rate"])
            rows += int(cell["n"])
        effect = weighted_effect / weight_total if weight_total else 0.0
        effects.append(
            MonthEffect(
                month=month,
                rows=rows,
                raw_rate=round(positives / weight_total, 6) if weight_total else 0.0,
                log_odds_effect=round(effect, 6),
                pct_point_shift_at_base=round(100.0 * (_sigmoid(base_logit + effect) - overall), 4),
            )
        )
    return effects


def seasonal_amplitude(effects: Sequence[MonthEffect]) -> float | None:
    """Peak-to-trough spread in percentage points at the base rate.

    The single number that answers "does time invariance hold?". A value near
    zero would mean the simulation's assumption is harmless; anything large
    means a schedule that moves inspections across the calendar is also moving
    them across a real change in citation probability.
    """
    if not effects:
        return None
    shifts = [e.pct_point_shift_at_base for e in effects]
    return round(max(shifts) - min(shifts), 4)


def _probabilities(
    dates: Sequence[date],
    effects_by_month: dict[int, float],
    base_rate: float,
) -> list[float]:
    base = _logit(base_rate)
    return [_sigmoid(base + effects_by_month.get(day.month, 0.0)) for day in dates]


@dataclass(frozen=True, slots=True)
class SensitivityResult:
    """Distribution of a headline metric under re-drawn labels."""

    metric: str
    replications: int
    observed: float | None
    mean: float | None
    std: float | None
    p05: float | None
    p50: float | None
    p95: float | None
    label_flip_rate: float | None


def _quantile(values: Sequence[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[int(position)]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def redraw_sensitivity(
    window: Window,
    order: Sequence[int],
    effects: Sequence[MonthEffect],
    *,
    replications: int = DEFAULT_REPLICATIONS,
    seed: int = DEFAULT_SENSITIVITY_SEED,
    observed_nde: float | None = None,
) -> SensitivityResult:
    """Re-run one schedule's efficiency with labels re-drawn under seasonality.

    Returns a distribution, not a point estimate. That is the deliverable: an
    unstated assumption converted into a measured uncertainty band.

    The ordering is fixed across replications -- the scores did not change, only
    the counterfactual labels -- so each replication is a linear pass over the
    window and the whole thing stays cheap.
    """
    if replications < 1:
        raise ValueError(f"replications must be at least 1, got {replications}")
    if window.n == 0 or not effects:
        return SensitivityResult(
            "normalized_discovery_efficiency", 0, observed_nde, None, None, None, None, None, None
        )

    effects_by_month = {e.month: e.log_odds_effect for e in effects}
    base_rate = window.positives / window.n

    simulated: list[date] = [window.dates[0]] * window.n
    for slot, row in enumerate(order):
        simulated[row] = window.slots[slot]

    p_real = _probabilities(list(window.dates), effects_by_month, base_rate)
    p_sim = _probabilities(simulated, effects_by_month, base_rate)

    rng = random.Random(seed)
    values: list[float] = []
    flips = 0
    total = 0

    for _ in range(replications):
        labels: list[int] = []
        for i in range(window.n):
            if window.labels[i] == 1:
                u = rng.random() * p_real[i]
            else:
                u = p_real[i] + rng.random() * (1.0 - p_real[i])
            drawn = 1 if u < p_sim[i] else 0
            labels.append(drawn)
            flips += drawn != window.labels[i]
            total += 1

        positives = sum(labels)
        if positives == 0 or positives == window.n:
            continue
        cumulative = [0]
        running = 0
        for row in order:
            running += labels[row]
            cumulative.append(running)
        area = normalized_area(cumulative, n=window.n, positives=positives)
        nde = normalized_discovery_efficiency(area, n=window.n, positives=positives)
        if nde is not None:
            values.append(nde)

    if not values:
        return SensitivityResult(
            "normalized_discovery_efficiency",
            replications,
            observed_nde,
            None,
            None,
            None,
            None,
            None,
            round(flips / total, 6) if total else None,
        )

    ordered = sorted(values)
    mean = sum(ordered) / len(ordered)
    variance = (
        sum((v - mean) ** 2 for v in ordered) / (len(ordered) - 1) if len(ordered) > 1 else 0.0
    )
    return SensitivityResult(
        metric="normalized_discovery_efficiency",
        replications=replications,
        observed=observed_nde,
        mean=round(mean, 6),
        std=round(math.sqrt(variance), 6),
        p05=round(_quantile(ordered, 0.05), 6),
        p50=round(_quantile(ordered, 0.50), 6),
        p95=round(_quantile(ordered, 0.95), 6),
        label_flip_rate=round(flips / total, 6) if total else None,
    )


__all__ = [
    "DEFAULT_REPLICATIONS",
    "DEFAULT_SENSITIVITY_SEED",
    "TEMPERATURE_STATUS",
    "MonthEffect",
    "SensitivityResult",
    "month_effects",
    "redraw_sensitivity",
    "seasonal_amplitude",
]
