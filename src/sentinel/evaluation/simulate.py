"""The retrodictive re-ordering simulation -- the centrepiece of Component 5.

The question is not *did the model predict correctly*. It is:

    If the same historically observed inspection opportunities had been ordered
    as Sentinel suggests, how much sooner would the known positive outcomes have
    appeared?

**The slot model.** A test window's capacity profile is the exact multiset of
``(date -> count)`` the city actually worked. Sort those dates and you have a
*slot calendar*: slot *i* has a real date, and there are exactly as many slots
as inspections. A schedule is a permutation of the window's inspections;
position *i* takes slot *i*, which hands every inspection a **simulated date**.

Capacity is therefore held constant by construction rather than by assumption.
The system changes the *order*, never the *number* -- which is Sentinel's whole
identity. A simulation that quietly added inspectors would beat business as
usual for a reason that has nothing to do with the model.

**The business-as-usual identity.** Sorting by actual date and filling slots in
date order returns every inspection to a slot on its own real date: the sorted
sequence holds exactly ``count(d)`` consecutive entries for date *d*, and the
slot array holds exactly ``count(d)`` consecutive positions for date *d*, in the
same order. So ``bau_simulated_date == actual inspection_date``, always, and
"days earlier than business as usual" means "days earlier than what really
happened". ``tests/test_evaluation_simulate.py`` asserts this rather than
trusting the argument.

**What this is not.** Labels never move. An establishment cited on 14 June is
still cited when the schedule moves it to 2 May -- this is a *reordering*
simulation, not a causal simulator, and §28 of the project spec is right that
the assumption is questionable. ``sensitivity.py`` measures how questionable.

**Intra-day order is unrecoverable.** ``inspection_date`` carries no time
component anywhere in the source, so within one day the order is arbitrary. It
is settled deterministically on ``target_inspection_id`` and cannot bias any
date-based metric, because every row tied that way shares a date.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from sentinel.evaluation.models import ScheduleName

logger = logging.getLogger(__name__)

#: Default seed for the random reference schedule. Recorded in the manifest.
DEFAULT_RANDOM_SEED = 42

#: A single shuffle can be lucky. The random schedule is replicated across this
#: many seeds and reported as mean +/- SD, so the model is never compared
#: against one favourable draw.
DEFAULT_RANDOM_REPLICATIONS = 20

#: Stated in the manifest so a consumer never has to infer it from the code.
CAPACITY_SEMANTICS = (
    "slots = the observed multiset of inspection dates in the window; "
    "every schedule is a permutation over the same slots, so capacity is identical"
)


class SimulationError(ValueError):
    """Raised when a window cannot be simulated as described."""


@dataclass(frozen=True, slots=True)
class Window:
    """One test window, in canonical order.

    Canonical order is ``(inspection_date, target_inspection_id)`` ascending --
    which is also the business-as-usual schedule, so the observed world is the
    natural index and every other schedule is a permutation away from it.
    """

    ids: tuple[str, ...]
    labels: tuple[int, ...]
    dates: tuple[date, ...]

    def __post_init__(self) -> None:
        if not (len(self.ids) == len(self.labels) == len(self.dates)):
            raise SimulationError("ids, labels and dates must be the same length")
        if len(set(self.ids)) != len(self.ids):
            raise SimulationError("target_inspection_id must be unique within a window")

    @property
    def n(self) -> int:
        """Total inspection slots in the window."""
        return len(self.ids)

    @property
    def positives(self) -> int:
        """Number of inspections that cited a Priority or Priority Foundation violation."""
        return sum(self.labels)

    @property
    def slots(self) -> tuple[date, ...]:
        """The slot calendar: the window's real dates, sorted ascending.

        Identical to ``dates`` because the window is already in canonical order.
        Named separately because the two ideas are different -- one is what
        happened, the other is the capacity any schedule must fit inside.
        """
        return self.dates


def build_window(
    ids: Sequence[str],
    labels: Sequence[int],
    dates: Sequence[date],
) -> Window:
    """Sort rows into canonical order and wrap them as a ``Window``."""
    if not (len(ids) == len(labels) == len(dates)):
        raise SimulationError("ids, labels and dates must be the same length")
    order = sorted(range(len(ids)), key=lambda i: (dates[i], ids[i]))
    return Window(
        ids=tuple(ids[i] for i in order),
        labels=tuple(int(labels[i]) for i in order),
        dates=tuple(dates[i] for i in order),
    )


# --- 1. the five schedules -------------------------------------------------


def business_as_usual_order(window: Window) -> list[int]:
    """The actual historical order: by inspection date, ties on id.

    The comparator that matters. Chicago does not inspect at random -- it works
    a statutory cycle and targets high-risk categories -- so this is a genuine
    baseline, not a straw man.
    """
    return list(range(window.n))


def optimal_order(window: Window) -> list[int]:
    """Every positive first. Unreachable upper bound, not a target.

    Requires knowing the answer before inspecting, so no schedule can achieve
    it. It exists to give the normalized efficiency a ceiling.
    """
    return sorted(range(window.n), key=lambda i: (-window.labels[i], window.ids[i]))


def worst_order(window: Window) -> list[int]:
    """Every positive last. Lower bound, and the mirror image of optimal."""
    return sorted(range(window.n), key=lambda i: (window.labels[i], window.ids[i]))


def random_order(window: Window, *, seed: int) -> list[int]:
    """A seeded shuffle. Deterministic given the seed and the canonical order."""
    order = list(range(window.n))
    random.Random(seed).shuffle(order)
    return order


def model_order(window: Window, scores: Sequence[float]) -> list[int]:
    """Descending score, ties settled on ``target_inspection_id`` ascending.

    Higher score means higher predicted risk, so it is inspected sooner. The
    direction is asserted by a test rather than assumed by a reader.
    """
    if len(scores) != window.n:
        raise SimulationError(f"expected {window.n} scores for this window, received {len(scores)}")
    return sorted(range(window.n), key=lambda i: (-scores[i], window.ids[i]))


# --- 2. simulated dates and discovery --------------------------------------


def simulated_dates(window: Window, order: Sequence[int]) -> list[date]:
    """Date each inspection would have happened under ``order``.

    Returned aligned to the window's canonical index, not to ``order``, so it
    joins straight back onto the observed world.
    """
    if len(order) != window.n:
        raise SimulationError(f"order has {len(order)} entries, window has {window.n} slots")
    if sorted(order) != list(range(window.n)):
        raise SimulationError("order must be a permutation of the window's rows")
    assigned = [window.slots[0]] * window.n if window.n else []
    for slot, row in enumerate(order):
        assigned[row] = window.slots[slot]
    return assigned


def discovery_curve(window: Window, order: Sequence[int]) -> list[int]:
    """Cumulative positives found after each slot, length ``n + 1``, starting at 0.

    The x-axis is inspection slots consumed, the y-axis is violations
    discovered. The faster it climbs, the sooner the city learns what it needs
    to know.
    """
    cumulative = [0]
    total = 0
    for row in order:
        total += window.labels[row]
        cumulative.append(total)
    return cumulative


def normalized_area(cumulative: Sequence[int], *, n: int, positives: int) -> float | None:
    """Trapezoid area under the discovery curve in normalized coordinates.

    Both axes are scaled to [0, 1] -- x by slots, y by positives -- so folds of
    different size and prevalence are directly comparable. Undefined when the
    window has no positives, because the y-axis has no scale.
    """
    if n == 0 or positives == 0:
        return None
    total = 0.0
    for i in range(1, n + 1):
        y1 = cumulative[i - 1] / positives
        y2 = cumulative[i] / positives
        total += (y1 + y2) / 2.0
    return total / n


def analytic_random_area() -> float:
    """Expected area for a uniformly random permutation: exactly one half.

    Derivation, not simulation: after *i* slots a random schedule has found
    ``P*i/N`` positives in expectation, so the expected curve is the diagonal
    and its area is 0.5 exactly. Using the analytic value rather than a sampled
    seed removes the last source of noise from the denominator -- the empirical
    multi-seed band is still reported, as a check on this claim.
    """
    return 0.5


def analytic_optimal_area(*, n: int, positives: int) -> float | None:
    """Area when every positive comes first: ``1 - P/(2N)``."""
    if n == 0 or positives == 0:
        return None
    return 1.0 - positives / (2.0 * n)


def normalized_discovery_efficiency(
    area: float | None,
    *,
    n: int,
    positives: int,
) -> float | None:
    """Where a schedule sits between random and perfect.

    .. math:: NDE = \\frac{A_{model} - A_{random}}{A_{optimal} - A_{random}}

    with ``A_random = 0.5`` and ``A_optimal = 1 - P/(2N)``, both analytic. So
    the denominator is ``(N - P) / 2N`` and the scale is fixed: **1** is
    perfect, **0** is random, **-1** is worst-possible. Essentially a CAP/Gini
    coefficient for the schedule, and base-rate invariant, which makes it the
    headline number to average across folds whose prevalence drifts.

    ``None`` when the window is all-positive or all-negative: with nothing to
    separate, every schedule is identical and an efficiency would be a fiction.
    """
    if area is None or n == 0 or positives == 0 or positives == n:
        return None
    optimal = analytic_optimal_area(n=n, positives=positives)
    if optimal is None:
        return None
    denominator = optimal - analytic_random_area()
    if denominator <= 0:
        return None
    return (area - analytic_random_area()) / denominator


def first_half_discovery(cumulative: Sequence[int], *, n: int, positives: int) -> float | None:
    """Share of the window's positives found in the first half of the schedule.

    The metric the 2015 Chicago evaluation reported as 69% for the model versus
    55% for business as usual. Reproduced here for conceptual comparability
    only -- those numbers came from a different food code, a different target
    and a 14.1% base rate, so they are not a benchmark this project can be
    measured against.

    The half is ``n // 2`` slots. Base-rate dependent, so always read it beside
    the fold's prevalence.
    """
    if n == 0 or positives == 0:
        return None
    return cumulative[n // 2] / positives


# --- 3. days-earlier detection ---------------------------------------------


@dataclass(frozen=True, slots=True)
class DaysEarlier:
    """Distribution of the timing change, not just its mean.

    Reporting the mean alone would repeat the weakness of the 2015 result --
    7.438 days earlier with a standard deviation of 25.156, meaning a great many
    establishments were found *later* and the headline never said so. The
    fractions improved / unchanged / worse are the honest part.
    """

    count: int
    mean: float | None
    median: float | None
    std: float | None
    p25: float | None
    p75: float | None
    minimum: float | None
    maximum: float | None
    fraction_improved: float | None
    fraction_unchanged: float | None
    fraction_worse: float | None


def _quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile over a sorted sequence."""
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[int(position)]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def days_earlier(
    window: Window,
    order: Sequence[int],
    *,
    positives_only: bool = True,
) -> DaysEarlier:
    """How many days sooner each inspection happens than it really did.

    Positive means earlier. The baseline is the business-as-usual simulated
    date, which is provably the real observed date, so this is measured against
    the world rather than against another simulation.

    ``positives_only`` because the operational quantity is how much sooner a
    *violation* is discovered; moving a clean establishment forward buys
    nothing. The all-rows view is reported separately as a secondary.
    """
    simulated = simulated_dates(window, order)
    deltas = [
        float((window.dates[i] - simulated[i]).days)
        for i in range(window.n)
        if not positives_only or window.labels[i] == 1
    ]
    if not deltas:
        return DaysEarlier(0, None, None, None, None, None, None, None, None, None, None)

    ordered = sorted(deltas)
    count = len(ordered)
    mean = sum(ordered) / count
    variance = sum((d - mean) ** 2 for d in ordered) / (count - 1) if count > 1 else 0.0
    return DaysEarlier(
        count=count,
        mean=mean,
        median=_quantile(ordered, 0.5),
        std=math.sqrt(variance),
        p25=_quantile(ordered, 0.25),
        p75=_quantile(ordered, 0.75),
        minimum=ordered[0],
        maximum=ordered[-1],
        fraction_improved=sum(1 for d in ordered if d > 0) / count,
        fraction_unchanged=sum(1 for d in ordered if d == 0) / count,
        fraction_worse=sum(1 for d in ordered if d < 0) / count,
    )


# --- 4. one schedule, fully measured ---------------------------------------


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    """Everything measured about one schedule over one window."""

    schedule: ScheduleName
    label: str
    seed: int | None
    order: tuple[int, ...]
    cumulative: tuple[int, ...]
    area: float | None
    normalized_discovery_efficiency: float | None
    first_half_discovery: float | None
    days_earlier_positives: DaysEarlier
    days_earlier_all: DaysEarlier


def evaluate_schedule(
    window: Window,
    order: Sequence[int],
    *,
    schedule: ScheduleName,
    label: str,
    seed: int | None = None,
) -> ScheduleResult:
    """Run every simulation metric over one ordering of one window."""
    cumulative = discovery_curve(window, order)
    area = normalized_area(cumulative, n=window.n, positives=window.positives)
    return ScheduleResult(
        schedule=schedule,
        label=label,
        seed=seed,
        order=tuple(order),
        cumulative=tuple(cumulative),
        area=area,
        normalized_discovery_efficiency=normalized_discovery_efficiency(
            area, n=window.n, positives=window.positives
        ),
        first_half_discovery=first_half_discovery(
            cumulative, n=window.n, positives=window.positives
        ),
        days_earlier_positives=days_earlier(window, order, positives_only=True),
        days_earlier_all=days_earlier(window, order, positives_only=False),
    )


def capacity_k_values(window: Window, *, median_daily: int) -> dict[str, int]:
    """The ``k`` values precision@k is reported at, derived from real capacity.

    One day, one week and one month of inspections at the window's own measured
    median daily rate, plus a fractional grid. Nothing here is a round number
    chosen for convenience -- inventing ``k = 50`` would make the headline
    operational metric an assumption.
    """
    if median_daily < 1:
        raise SimulationError(f"median daily capacity must be at least 1, got {median_daily}")
    candidates = {
        "k_1_day": median_daily,
        "k_1_week": median_daily * 5,
        "k_1_month": median_daily * 21,
        "k_pct_01": max(1, round(window.n * 0.01)),
        "k_pct_05": max(1, round(window.n * 0.05)),
        "k_pct_10": max(1, round(window.n * 0.10)),
        "k_pct_25": max(1, round(window.n * 0.25)),
        "k_pct_50": max(1, round(window.n * 0.50)),
    }
    return {name: min(k, window.n) for name, k in candidates.items() if window.n > 0}


__all__ = [
    "CAPACITY_SEMANTICS",
    "DEFAULT_RANDOM_REPLICATIONS",
    "DEFAULT_RANDOM_SEED",
    "DaysEarlier",
    "ScheduleResult",
    "SimulationError",
    "Window",
    "analytic_optimal_area",
    "analytic_random_area",
    "build_window",
    "business_as_usual_order",
    "capacity_k_values",
    "days_earlier",
    "discovery_curve",
    "evaluate_schedule",
    "first_half_discovery",
    "model_order",
    "normalized_area",
    "normalized_discovery_efficiency",
    "optimal_order",
    "random_order",
    "simulated_dates",
    "worst_order",
]
