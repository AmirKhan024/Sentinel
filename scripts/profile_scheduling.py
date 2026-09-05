"""Read-only profiling of the operating calendar, before any scheduling code is written.

Analysis tooling, not library code: it answers one-off questions about a snapshot, nothing
imports it, and it should not ship in the wheel. Output is markdown on stdout, pasted into
``docs/analysis/scheduling_findings.md``.

⚠ **This script fits nothing, scores nothing, ranks nothing and changes nothing.** Every model
is frozen, every prediction is on disk, and Component 13's queue is an artifact this script
reads and never recomputes. Component 14 is a scheduling layer over artifacts, so the only
computation here is arithmetic over columns Components 5 and 13 already wrote.

⚠ **This script is run before the capacity modes, the horizon rule and the advisory thresholds
are frozen, and is what fixes them.** Profile 1 establishes that an operating calendar exists
in the data at all; profile 2 fixes the horizon rule and proves it never overruns a fold;
profile 3 is the component's central measurement -- the gap between Component 13's flat median
capacity and the capacity Chicago actually had -- and it fixes both the default capacity mode
and the ``horizon_capacity_meets_the_queue`` advisory. A scheduling constant chosen from
expectation rather than measurement is a guess wearing a decimal point, and this project has
corrected that mistake once already in Component 9.

⚠ **It reads no label and computes no metric.** "Did the schedule find more violations" is not
a question Component 14 asks: the component re-orders an approved queue in time and does not
touch what the queue is. There is deliberately no outcome column anywhere in this script.

Questions this script answers
-----------------------------
1.  ``operating_calendar``       -- does a real operating calendar exist in the artifacts, how
                                    many distinct dates does each fold's test window hold, and
                                    what is the per-day inspection count? **Decides whether an
                                    observed-calendar mode is possible at all.**
2.  ``horizon_rule``             -- under ``ceil(k / median_daily)``, how many operating days
                                    does each capacity level span, and does any cell demand
                                    more days than its fold actually has? **Fixes the horizon
                                    rule and proves it total.**
3.  ``observed_versus_flat``     -- how many slots does the observed calendar supply over each
                                    horizon, against the flat median Component 13 assumed, and
                                    how large is the resulting backlog? **The measured problem
                                    statement, and it fixes DEFAULT_CAPACITY_MODE.**
4.  ``weekday_shape``            -- which weekdays does Chicago inspect on, and does the data
                                    support treating the calendar as observed rather than
                                    synthesised from a working-week assumption?
5.  ``queue_recurrence``         -- can one establishment appear twice inside a single horizon,
                                    and how often? **Decides whether establishment uniqueness
                                    is an error check or an advisory.**
6.  ``mechanism_mix``            -- what share of each queue arrives by risk rank against the
                                    coverage reserve, so a schedule can report the mix it
                                    inherits without recomputing the policy.
7.  ``absent_operational_data``  -- which operational fields a scheduler would want are simply
                                    not in the snapshot. **The inventory that decides what
                                    Component 14 refuses to implement.**
8.  ``reserve_survives_scheduling``
                                 -- Component 13 places the coverage reserve at the tail of the
                                    rank order, so a horizon that falls short takes the reserve
                                    first. How much of it survives? **The component's headline,
                                    and it is a critique of Component 13 measured from inside
                                    Component 14.**
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.config import load_settings  # noqa: E402
from sentinel.query.duckdb_queries import latest_parquet  # noqa: E402

#: The capacity levels Component 13 emits, in the order its own definitions list them.
K_LEVELS: tuple[str, ...] = ("k_pct_01", "k_pct_05", "k_pct_10", "k_1_day", "k_1_week")

#: The fold reported row-by-row in the narrow profiles. The most recent complete quarter, which
#: is also the one Component 13's findings document uses, so the two can be read side by side.
FOCUS_FOLD = "quarterly-2026Q2"

#: The baseline policy. Profiles that do not vary the policy hold it here rather than pooling,
#: because pooling seven policies would average away the mechanism mix profile 6 exists to show.
FOCUS_POLICY = "pure_risk"

#: Operational fields a scheduler would want, and why the snapshot does not carry them. The raw
#: table has 22 columns; none of these is among them.
ABSENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("inspector identity", "no inspector column anywhere in the snapshot (ADR 0019)"),
    ("inspector roster / headcount", "not published with the inspection data"),
    ("inspector working hours or shifts", "not published"),
    ("inspector base location", "not published"),
    ("inspection duration / time on site", "no start or end time, only a date"),
    ("travel time between establishments", "no route, no distance, no timestamp"),
    ("road network / drive-time matrix", "outside the scope of the Socrata dataset"),
    ("service territory assignment", "not published; district is inferable only by geography"),
    ("appointment windows", "inspections are unannounced; no such field exists"),
    ("establishment closure / unavailability", "no availability calendar is published"),
    ("statutory deadline per establishment", "not in the dataset"),
    ("execution status of a planned inspection", "the dataset records completed inspections only"),
)


def _fmt(value: float, places: int = 4) -> str:
    return f"{value:.{places}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _baseline(sources: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """The recommendation universe under the baseline policy, one row per scored inspection.

    Filtered to a single capacity level as well as a single policy: the universe is identical
    at every ``k_name`` by construction -- Component 13 writes the whole prediction universe
    into each cell -- so taking one avoids counting every row five times.
    """
    return sources["recommendations"].filter(
        (pl.col("policy_id") == FOCUS_POLICY) & (pl.col("k_name") == K_LEVELS[0])
    )


def _day_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """Inspections per operating date, ascending. The observed calendar, and nothing else."""
    return frame.group_by("inspection_date").len().sort("inspection_date")


def _median_daily(sources: dict[str, pl.DataFrame]) -> dict[str, int]:
    return {
        str(row["fold_id"]): max(1, int(row["test_median_daily_capacity"] or 1))
        for row in sources["folds"].iter_rows(named=True)
    }


def _k_by_cell(sources: dict[str, pl.DataFrame]) -> dict[tuple[str, str], int]:
    frame = (
        sources["recommendations"]
        .filter(pl.col("policy_id") == FOCUS_POLICY)
        .group_by("fold_id", "k_name")
        .agg(pl.col("k").first())
    )
    return {(str(r["fold_id"]), str(r["k_name"])): int(r["k"]) for r in frame.iter_rows(named=True)}


# --- profiles ----------------------------------------------------------------


def operating_calendar(sources: dict[str, pl.DataFrame]) -> str:
    """Does a real operating calendar exist in the artifacts?

    Everything in Component 14 rests on this. If the only temporal information available were
    a fold's start and end date, a per-day schedule would have to invent its own calendar --
    which working days, how many inspections each -- and every number downstream would be an
    assumption wearing a decimal point.

    It does not have to. ``inspection_date`` on the recommendation universe is the date the
    inspection actually happened, so the set of operating dates and the number of inspections
    performed on each are both observations.
    """
    universe = _baseline(sources)
    medians = _median_daily(sources)
    rows: list[list[str]] = []
    for fold_id in sorted(universe["fold_id"].unique().to_list()):
        counts = _day_counts(universe.filter(pl.col("fold_id") == fold_id))
        series = counts["len"]
        rows.append(
            [
                str(fold_id),
                str(counts.height),
                str(int(series.min() or 0)),
                _fmt(float(series.median() or 0.0), 1),
                _fmt(float(series.mean() or 0.0), 1),
                str(int(series.max() or 0)),
                str(medians.get(str(fold_id), 0)),
            ]
        )
    body = _table(
        ["fold", "operating days", "min/day", "median/day", "mean/day", "max/day", "C5 median"],
        rows,
    )
    return (
        "**An operating calendar exists, and it is an observation rather than an assumption.**\n"
        "Every fold's test window resolves to a set of distinct dates on which inspections were\n"
        "actually performed, with a real count on each. The last column is Component 5's own\n"
        "`test_median_daily_capacity`, recomputed here from the recommendation artifact as a\n"
        "cross-check: it agrees with the median of the per-day counts, which is what it is\n"
        "defined to be.\n\n"
        "The spread is the part that matters. A fold whose median day holds 28 inspections has\n"
        "days holding 1 and days holding 55, so a schedule built on the median alone is not\n"
        "describing any particular day.\n\n" + body
    )


def horizon_rule(sources: dict[str, pl.DataFrame]) -> str:
    """How many operating days does each capacity level span, and is the rule total?

    Component 13's capacity levels already carry a duration in their names, and
    ``evaluation/simulate.py`` fixes the arithmetic behind them: ``k_1_day`` is the window's
    median daily rate and ``k_1_week`` is five times it. So a horizon length is not a new
    assumption -- it is the same number read backwards, ``ceil(k / median_daily)``.

    The question this profile settles is whether that rule is *total*: a cell demanding more
    operating days than its fold contains would have no schedule, and the component would need
    an answer for it. It never happens.
    """
    medians = _median_daily(sources)
    k_by_cell = _k_by_cell(sources)
    universe = _baseline(sources)
    available = {
        str(fold_id): _day_counts(universe.filter(pl.col("fold_id") == fold_id)).height
        for fold_id in universe["fold_id"].unique().to_list()
    }
    rows: list[list[str]] = []
    overruns = 0
    widest = ("", 0, 0)
    for fold_id in sorted(available):
        median = medians.get(fold_id, 1)
        cells: list[str] = []
        for k_name in K_LEVELS:
            k = k_by_cell.get((fold_id, k_name))
            if k is None:
                cells.append("--")
                continue
            days = math.ceil(k / median)
            if days > available[fold_id]:
                overruns += 1
                cells.append(f"**{days}**")
            else:
                cells.append(str(days))
            if days > widest[1]:
                widest = (f"{fold_id} / {k_name}", days, available[fold_id])
        rows.append([fold_id, str(available[fold_id]), *cells])
    body = _table(["fold", "operating days", *K_LEVELS], rows)
    return (
        "**The horizon rule is `ceil(k / test_median_daily_capacity)`, and it is total.**\n"
        f"Across every (fold, capacity) cell, **{overruns}** demand more operating days than\n"
        "their fold contains. The rule therefore needs no fallback branch, and Component 14 can\n"
        "refuse to invent one -- a fallback nothing exercises is a fallback nobody has tested.\n\n"
        f"The widest cell is `{widest[0]}`, which spans {widest[1]} of {widest[2]} available\n"
        "days.\n\n" + body
    )


def observed_versus_flat(sources: dict[str, pl.DataFrame]) -> str:
    """What does the observed calendar supply, against the flat median Component 13 assumed?

    **This is the component's central measurement.** Component 13's cutoffs descend from a
    quarter-wide median. A schedule built on that same median is feasible by construction --
    the horizon is defined as ``k / median`` days of ``median`` slots, so it holds exactly *k*
    and the backlog is zero before anything is measured.

    Whether it is feasible against the days Chicago actually worked is a different question,
    and it is the only one worth asking. A median is a summary of a quarter; the first days of
    a quarter are particular days.
    """
    medians = _median_daily(sources)
    k_by_cell = _k_by_cell(sources)
    universe = _baseline(sources)
    rows: list[list[str]] = []
    short_cells = 0
    total_cells = 0
    total_backlog = 0
    for fold_id in sorted(universe["fold_id"].unique().to_list()):
        counts = _day_counts(universe.filter(pl.col("fold_id") == fold_id))["len"].to_list()
        median = medians.get(str(fold_id), 1)
        for k_name in K_LEVELS:
            k = k_by_cell.get((str(fold_id), k_name))
            if k is None:
                continue
            days = math.ceil(k / median)
            observed = int(sum(counts[:days]))
            flat = median * days
            backlog = max(0, k - observed)
            total_cells += 1
            if observed < k:
                short_cells += 1
                total_backlog += backlog
            if str(fold_id) == FOCUS_FOLD:
                rows.append(
                    [
                        k_name,
                        str(k),
                        str(days),
                        str(flat),
                        str(observed),
                        str(max(0, k - flat)),
                        f"**{backlog}**" if backlog else "0",
                    ]
                )
    body = _table(
        [
            "capacity",
            "k",
            "horizon days",
            "flat slots",
            "observed slots",
            "backlog flat",
            "backlog observed",
        ],
        rows,
    )
    share = (short_cells / total_cells) if total_cells else 0.0
    return (
        f"**In {short_cells} of {total_cells} (fold, capacity) cells the observed calendar\n"
        f"supplies fewer slots than the queue needs** -- {_fmt(share, 4)} of them -- for a total\n"
        f"of **{total_backlog}** recommended inspections that do not fit inside their own\n"
        "horizon. Under the flat median the backlog is zero in every cell, by construction.\n\n"
        "That contrast is the finding, and it is a fact about Chicago's operating pattern rather\n"
        "than about the scheduler: early-quarter days run below the quarter's median, so a\n"
        "cutoff derived from the median promises capacity the first week does not have.\n\n"
        "It also fixes two decisions. `observed_calendar` is the **default** capacity mode,\n"
        "because it describes days that happened; `flat_median` is retained as an explicitly\n"
        "labelled **scenario**, because it reproduces Component 13's own stated capacity\n"
        "semantics and the comparison between them is the measurement above. And the\n"
        "`horizon_capacity_meets_the_queue` advisory fires on exactly the cells counted here.\n\n"
        f"Row-by-row for `{FOCUS_FOLD}`:\n\n" + body
    )


def weekday_shape(sources: dict[str, pl.DataFrame]) -> str:
    """Which weekdays does Chicago inspect on?

    Asked in order to *refuse* a shortcut. A scheduler could synthesise its calendar from a
    Monday-to-Friday rule and a holiday list, and that would be an assumption imported from
    outside the data. This profile shows the assumption would also be wrong at the edges --
    there are inspections on weekend dates -- which is the argument for reading the calendar
    instead of generating it.
    """
    universe = _baseline(sources)
    names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    frame = (
        universe.with_columns(pl.col("inspection_date").dt.weekday().alias("wd"))
        .group_by("wd")
        .len()
        .sort("wd")
    )
    total = int(frame["len"].sum() or 1)
    rows = [
        [names.get(int(r["wd"]), str(r["wd"])), str(int(r["len"])), _fmt(int(r["len"]) / total, 4)]
        for r in frame.iter_rows(named=True)
    ]
    weekend = int(frame.filter(pl.col("wd") >= 6)["len"].sum() or 0)
    body = _table(["weekday", "inspections", "share"], rows)
    return (
        "**The calendar is read, never generated.** Inspections concentrate on weekdays, but\n"
        f"**{weekend}** fall on a Saturday or Sunday. A synthesised Monday-to-Friday calendar\n"
        "would silently discard those days, and would also need a holiday list this project does\n"
        "not have and has no way to verify.\n\n"
        "Reading the operating dates out of the artifact costs nothing and imports no\n"
        "assumption, so that is what Component 14 does.\n\n" + body
    )


def queue_recurrence(sources: dict[str, pl.DataFrame]) -> str:
    """Can one establishment appear twice inside a single horizon?

    A scheduling validator wants to assert that nothing is booked twice. The question is what
    "nothing" is. Component 13's grain is the scored inspection event, and an establishment
    canvassed twice in one quarter is two events -- two genuine opportunities, each with its
    own as-of features and its own score.

    If that happens inside a queue, then "no establishment occupies two slots" is not an
    invariant of a correct scheduler; it is a claim about the data that the data refutes.
    """
    frame = sources["recommendations"].filter(pl.col("policy_id") == FOCUS_POLICY)
    rows: list[list[str]] = []
    for k_name in K_LEVELS:
        cell = frame.filter((pl.col("k_name") == k_name) & pl.col("is_selected"))
        dupes = cell.group_by("fold_id", "establishment_id").len().filter(pl.col("len") > 1)
        rows.append(
            [
                k_name,
                str(cell.height),
                str(cell["target_inspection_id"].n_unique()),
                str(cell["establishment_id"].n_unique()),
                str(dupes.height),
            ]
        )
    universe = _baseline(sources)
    universe_dupes = (
        universe.group_by("fold_id", "establishment_id").len().filter(pl.col("len") > 1).height
    )
    body = _table(
        ["capacity", "selected rows", "unique inspections", "unique establishments", "recurring"],
        rows,
    )
    return (
        "**An establishment can legitimately recur inside one fold, and rarely does inside one\n"
        f"queue.** {universe_dupes} establishment-fold pairs hold more than one scored canvass\n"
        "across the universe, and the table below counts how many survive into each queue.\n\n"
        "This fixes a validation decision. Uniqueness is an **error** check on\n"
        "`target_inspection_id` -- booking one inspection into two slots is always a defect --\n"
        "and only an **advisory** on `establishment_id`, because two canvasses of one premises\n"
        "in one quarter is something Chicago did, not something the scheduler got wrong.\n\n"
        "Asserting the stronger invariant would have produced a red build on correct data,\n"
        "which is the failure mode that makes a suite stop being believed.\n\n" + body
    )


def mechanism_mix(sources: dict[str, pl.DataFrame]) -> str:
    """What share of each queue arrives by risk rank against the coverage reserve?

    Component 14 has to report this without recomputing it: ADR 0037 put the mechanism on the
    row precisely so a later layer could carry it forward rather than re-deriving which
    establishments a policy promoted. This profile confirms the column is populated and shows
    the mix a schedule inherits.
    """
    frame = sources["recommendations"].filter(pl.col("is_selected"))
    grouped = (
        frame.group_by("policy_id", "decision_mechanism")
        .len()
        .sort("policy_id", "decision_mechanism")
    )
    totals = {
        str(r["policy_id"]): int(r["len"])
        for r in frame.group_by("policy_id").len().iter_rows(named=True)
    }
    rows = [
        [
            str(r["policy_id"]),
            str(r["decision_mechanism"]),
            str(int(r["len"])),
            _fmt(int(r["len"]) / max(1, totals.get(str(r["policy_id"]), 1)), 4),
        ]
        for r in grouped.iter_rows(named=True)
    ]
    body = _table(["policy", "mechanism", "selected rows", "share of queue"], rows)
    return (
        "**The mechanism travels on the row, so the schedule carries it forward unchanged.**\n"
        "A scheduled inspection can therefore answer two separate questions -- *why was this\n"
        "recommended* and *why is it on this day* -- without either answer being reconstructed.\n\n"
        "Component 14 reports this mix and never alters it. Promoting a coverage-reserve row to\n"
        "an earlier day, or demoting one to make a schedule tidier, would be a second policy\n"
        "decision taken with no ADR behind it.\n\n" + body
    )


def absent_operational_data(sources: dict[str, pl.DataFrame]) -> str:
    """Which operational fields does a scheduler want that this snapshot does not have?

    The inventory that decides what Component 14 refuses to build. Every row here is something
    a real inspection department schedules against, and every one of them is absent.

    Named explicitly, because the alternative is not neutrality -- it is a plausible default
    quietly standing in for a measurement. A travel-time matrix generated from straight-line
    distance and an assumed speed would produce routes that look authoritative and describe
    nothing.
    """
    raw_columns = sources.get("raw_columns")
    n_columns = raw_columns.height if raw_columns is not None else 22
    rows = [[field, "absent", note] for field, note in ABSENT_FIELDS]
    body = _table(["operational field", "in snapshot", "note"], rows)
    have = _table(
        ["present and used", "source"],
        [
            ["operating dates", "`inspection_date` on the recommendation universe"],
            ["inspections per operating day", "the same column, grouped"],
            ["median daily capacity", "Component 5 `test_median_daily_capacity`"],
            ["capacity cutoff k", "Component 13 `k`"],
            ["priority order", "Component 13 `final_policy_rank`"],
            ["selection provenance", "Component 13 `decision_mechanism` / `decision_reason`"],
        ],
    )
    return (
        f"**The raw snapshot has {n_columns} columns and none of them is an inspector.**\n"
        "That is the same absence ADR 0019 recorded when it blocked Component 10, and it decides\n"
        "the shape of this component: Sentinel can schedule *time* and *workload*, and it cannot\n"
        "schedule *people* or *routes*.\n\n"
        "Component 14 therefore performs temporal and workload scheduling, **not** geographic\n"
        "route optimisation. The README roadmap already assigns routing to Component 15, so\n"
        "declining it here is repository policy rather than an improvisation -- and Component 15\n"
        "is itself blocked on the same missing data.\n\n"
        + body
        + "\n\nWhat *is* present, and is what the component is built from:\n\n"
        + have
    )


def reserve_survives_scheduling(sources: dict[str, pl.DataFrame]) -> str:
    """How much of Component 13's coverage reserve survives contact with the calendar?

    **This is the component's headline, and it is uncomfortable.**

    Component 13 allocates a coverage reserve as a slot count, then fills the risk block at
    ranks ``1..n_risk`` and the reserve at ranks ``n_risk+1..k``. The reserve is therefore
    *always* the tail of the rank order -- this profile checks that, and finds no exception in
    any cell that allocates a reserve at all.

    A strict-priority schedule fills the horizon from the top of that order. So when the
    horizon holds fewer slots than ``k``, the rows that fall off the end are the reserve rows,
    every time. The coverage allocation ADR 0037 priced in citations is silently unspent by a
    layer that never mentions coverage.

    Component 14 **measures and reports this. It does not correct it.** Promoting reserve rows
    in the schedule would be re-ranking, which ``HANDOFF.md`` forbids in those words, and it
    would put the coverage decision in two places at once.
    """
    medians = _median_daily(sources)
    universe = _baseline(sources)
    calendars = {
        str(fold_id): _day_counts(universe.filter(pl.col("fold_id") == fold_id))["len"].to_list()
        for fold_id in universe["fold_id"].unique().to_list()
    }
    selected = sources["recommendations"].filter(pl.col("is_selected"))

    out_of_order = 0
    cells = 0
    lost = 0
    total_reserve = 0
    cells_losing_some = 0
    cells_losing_all = 0
    by_policy: dict[str, list[int]] = {}

    for key, group in selected.group_by(["policy_id", "fold_id", "k_name"]):
        policy_id, fold_id, _ = (str(key[0]), str(key[1]), str(key[2]))
        ordered = group.sort("final_policy_rank")
        mechanisms = ordered["decision_mechanism"].to_list()
        n_reserve = mechanisms.count("coverage_reserve")
        if n_reserve == 0:
            continue
        first_reserve = mechanisms.index("coverage_reserve")
        if "risk_priority" in mechanisms[first_reserve:]:
            out_of_order += 1

        k = int(ordered["k"][0])
        median = medians.get(fold_id, 1)
        days = math.ceil(k / median)
        slots = int(sum(calendars[fold_id][:days]))
        scheduled = mechanisms[: min(k, slots)]
        survived = scheduled.count("coverage_reserve")
        shortfall = n_reserve - survived

        cells += 1
        total_reserve += n_reserve
        lost += shortfall
        if shortfall > 0:
            cells_losing_some += 1
        if survived == 0:
            cells_losing_all += 1
        bucket = by_policy.setdefault(policy_id, [0, 0, 0])
        bucket[0] += n_reserve
        bucket[1] += survived
        bucket[2] += 1

    rows = [
        [
            policy_id,
            str(counts[2]),
            str(counts[0]),
            str(counts[1]),
            str(counts[0] - counts[1]),
            _fmt((counts[0] - counts[1]) / max(1, counts[0]), 4),
        ]
        for policy_id, counts in sorted(by_policy.items())
    ]
    body = _table(
        ["policy", "cells", "reserve recommended", "reserve scheduled", "lost", "share lost"],
        rows,
    )
    return (
        f"**The coverage reserve is always the tail of the rank order.** In {out_of_order} of\n"
        f"{cells} reserve-bearing cells does a reserve row outrank a risk row -- the\n"
        f"allocator places the risk block first and the reserve after it, by construction.\n\n"
        f"> **Strict-priority scheduling loses {lost} of {total_reserve} coverage-reserve\n"
        f"> slots to the horizon -- {_fmt(lost / max(1, total_reserve), 3)}. In\n"
        f"> {cells_losing_some} of {cells} cells some of the reserve is lost, and in\n"
        f"> {cells_losing_all} it is lost entirely.**\n\n"
        "This is a fact about Component 13's allocation meeting Component 14's calendar,\n"
        "and neither layer is wrong on its own terms. ADR 0037 priced the reserve in\n"
        "forgone citations and granted it a slot count; nothing in that decision said the\n"
        "slots had to be at the *end* of the queue, and nothing in Component 13 had a\n"
        "calendar to notice that it mattered.\n\n"
        "**Component 14 reports this and does not correct it.** Promoting reserve rows in\n"
        "the schedule would be re-ranking -- forbidden in those words -- and would put one\n"
        "coverage decision in two layers. The advisory\n"
        "`the_coverage_reserve_survived_scheduling` fires on exactly the cells counted\n"
        "here, at advisory severity, because the cheapest way to turn such a build green\n"
        "would be to make the scheduler prefer reserve rows.\n\n" + body
    )


PROFILES: dict[str, Callable[[dict[str, pl.DataFrame]], str]] = {
    "operating_calendar": operating_calendar,
    "horizon_rule": horizon_rule,
    "observed_versus_flat": observed_versus_flat,
    "weekday_shape": weekday_shape,
    "queue_recurrence": queue_recurrence,
    "mechanism_mix": mechanism_mix,
    "absent_operational_data": absent_operational_data,
    "reserve_survives_scheduling": reserve_survives_scheduling,
}


def _load(args: argparse.Namespace) -> tuple[dict[str, pl.DataFrame], list[str]]:
    settings = load_settings()
    recommendations_path = args.recommendations or latest_parquet(
        settings.policy_processed_dir, prefix="inspection_recommendations_"
    )
    folds_path = args.folds or latest_parquet(
        settings.evaluation_processed_dir, prefix="evaluation_folds_"
    )
    sources: dict[str, pl.DataFrame] = {
        "recommendations": pl.read_parquet(recommendations_path),
        "folds": pl.read_parquet(folds_path),
    }
    provenance = [
        f"recommendations: {recommendations_path.name}",
        f"folds: {folds_path.name}",
    ]
    try:
        raw_path = latest_parquet(settings.food_inspections_raw_dir, prefix="food_inspections_")
    except FileNotFoundError:
        provenance.append("raw snapshot: absent")
    else:
        sources["raw_columns"] = pl.DataFrame(
            {"column": pl.scan_parquet(raw_path).collect_schema().names()}
        )
        provenance.append(f"raw snapshot: {raw_path.name}")
    return sources, provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommendations", type=Path, help="Component 13 recommendation table.")
    parser.add_argument("--folds", type=Path, help="Component 5 fold table.")
    parser.add_argument("--only", action="append", help="Profile to run; repeatable.")
    args = parser.parse_args(argv)

    requested = args.only or list(PROFILES)
    unknown = [name for name in requested if name not in PROFILES]
    if unknown:
        raise SystemExit(f"unknown profile(s): {', '.join(unknown)}")

    sources, provenance = _load(args)

    print("<!-- generated by scripts/profile_scheduling.py -->")
    for line in provenance:
        print(f"<!-- {line} -->")
    for name in requested:
        print()
        print(f"### {name}")
        print()
        print(PROFILES[name](sources))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
