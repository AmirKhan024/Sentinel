"""The planning horizon and the capacity contract. The only module allowed to decide either.

**The horizon is not a new constant.** ``evaluation.simulate.capacity_k_values`` produced every
capacity cutoff in this project from a window's measured median daily rate: ``k_1_day`` is that
rate and ``k_1_week`` is five times it. Reading that rule backwards gives the horizon --
``ceil(k / median_daily)`` -- and it reproduces both day-denominated names exactly, one day and
five days. That is the reason to use this rule rather than any other: it introduces no number
that was not already in the project, so there is nothing here for a reader to have to accept.

**The calendar is read, never generated.** An operating day is a date Component 13's universe
carries for the fold. No working-week rule, no holiday list, no synthesised date appears
anywhere. The alternative was to generate Monday-to-Friday and subtract holidays, and profile 4
measured why not: three inspections in the snapshot fall on a weekend, so the generated calendar
would be wrong at the edges, and the holiday list it would need is something this project has no
way to verify.

**Capacity is never raised here or anywhere.** There is no parameter on any function in this
module that could increase a slot count, no flag that reaches one, and no branch that extends a
horizon past the days the rule allows. That is deliberate and it is the reason the module is
small: a schedule that fitted more inspections than the city worked would beat every
alternative for that reason alone.

**The two modes move capacity, never the calendar.** ``observed_calendar`` and ``flat_median``
produce the same operating days in the same order, and differ only in how many slots each day
holds. If a mode changed the horizon as well, the two would not be comparable and the
measurement the component exists to make would be confounded.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from sentinel.scheduling.definitions import (
    CAPACITY_SOURCES,
    CapacityMode,
    horizon_days,
)
from sentinel.scheduling.models import Horizon, OperatingDay


class HorizonError(ValueError):
    """Raised when a horizon cannot be built from the calendar and the capacity rule."""


def build_horizon(
    *,
    fold_set: str,
    fold_id: str,
    k_name: str,
    k: int,
    median_daily_capacity: int,
    calendar: Sequence[tuple[date, int]],
    capacity_mode: CapacityMode,
) -> Horizon:
    """The operating days for one cell, and the slots each of them holds.

    ``calendar`` is the fold's observed operating days as ``(date, inspections_performed)``
    pairs, ascending. It is passed in rather than derived here because it is a property of the
    fold and is shared by every policy and every capacity level in it; deriving it per cell
    would recompute one measurement 70 times per fold.

    The horizon is the first ``ceil(k / median_daily_capacity)`` of those days. In
    ``observed_calendar`` mode each day keeps its observed volume; in ``flat_median`` mode each
    day is assigned the window's median rate, which is what the capacity cutoff already
    assumed and is labelled a scenario everywhere it appears.
    """
    if not calendar:
        raise HorizonError(
            f"{fold_id}/{k_name}: the fold has no operating days. A horizon cannot be built "
            "from an empty calendar, and inventing one would invent the whole schedule"
        )
    dates = [day for day, _ in calendar]
    if dates != sorted(set(dates)):
        raise HorizonError(
            f"{fold_id}: the operating calendar is not strictly ascending. Slot assignment "
            "walks it in order, so a repeated or unsorted date would place rows arbitrarily"
        )
    if any(count < 1 for _, count in calendar):
        raise HorizonError(
            f"{fold_id}: an operating day with no inspections is not an operating day. It "
            "reached the calendar through a grouping defect, not through the data"
        )

    wanted = horizon_days(k, median_daily_capacity)
    was_clamped = wanted > len(calendar)
    span = min(wanted, len(calendar))

    source = CAPACITY_SOURCES[capacity_mode]
    days = tuple(
        OperatingDay(
            day_index=index,
            slot_date=slot_date,
            n_slots=(
                observed
                if capacity_mode is CapacityMode.OBSERVED_CALENDAR
                else median_daily_capacity
            ),
            capacity_source=source,
        )
        for index, (slot_date, observed) in enumerate(calendar[:span], start=1)
    )

    return Horizon(
        fold_set=fold_set,
        fold_id=fold_id,
        k_name=k_name,
        k=k,
        median_daily_capacity=median_daily_capacity,
        capacity_mode=str(capacity_mode),
        days=days,
        was_clamped=was_clamped,
    )


def clamp_detail(k: int, median_daily_capacity: int, available_days: int) -> str | None:
    """A message when the capacity rule wants more days than the fold has, else None.

    Measured to be unreachable: across all 90 (fold, capacity) cells the widest horizon is 41
    days against a minimum of 58 available. The branch exists anyway and records a check,
    because "unreachable" is a claim about a snapshot, and the next snapshot is a different
    one. A clamp means the horizon swallows the whole test window, at which point capacity and
    universe coincide and every scheduling ratio is trivially one -- so it is reported rather
    than absorbed.
    """
    wanted = horizon_days(k, median_daily_capacity)
    if wanted <= available_days:
        return None
    return (
        f"capacity {k} at {median_daily_capacity}/day wants {wanted} operating days and the "
        f"fold has {available_days}. The horizon is clamped to the whole window, so every "
        "utilisation and coverage ratio in this cell is trivially saturated"
    )


def observed_calendar_from_dates(dates: Sequence[date]) -> tuple[tuple[date, int], ...]:
    """The operating calendar from a fold's raw inspection dates.

    One pass, counting per date, then sorted. The counting is the measurement -- the number of
    inspections Chicago performed on a date is that date's real capacity -- and the sort is what
    makes the horizon a prefix rather than a sample.
    """
    counts: dict[date, int] = {}
    for value in dates:
        counts[value] = counts.get(value, 0) + 1
    return tuple(sorted(counts.items()))


__all__ = [
    "HorizonError",
    "build_horizon",
    "clamp_detail",
    "observed_calendar_from_dates",
]
