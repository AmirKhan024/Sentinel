"""Integrity and temporal checks over the scheduling artifacts.

**The severity split is this module's whole design, and it is inherited from ADR 0034 through
ADR 0041.** A defect in the *computation* is an error and fails the run. A finding about what
the schedule *costs* is an advisory: it is recorded, printed, and the run exits zero.

The sharpest case is this component's headline. A schedule that loses coverage-reserve slots to
a short horizon must never turn a build red, because the cheapest way to make such a build
green is to have the scheduler prefer reserve rows -- which is re-ranking, which is forbidden,
and which would move a coverage decision into a layer that does not own it. The same logic
covers the backlog: 44 of 90 cells cannot fit their queue, and a build that went red on that
would be a build that goes green only when the scheduler lies about the calendar.

**Checks read the written artifacts, not the objects that produced them.** The allocator
asserts its own post-conditions, and this module re-derives them from the frames. That is
deliberate duplication: the first catches a broken allocator and the second catches a broken
writer, and a single check placed in one of the two places would miss half the failures.

**Two checks re-run the computation, because their property is not observable from a table.**
Shuffle-invariance and "the plan is unchanged with the external files withheld" are both claims
about what the code would do under different inputs, so they are proved by doing it. Component
13's ``warnings_do_not_change_the_queue`` is the same shape.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from sentinel.policy.definitions import DecisionMechanism, OverrideAction
from sentinel.scheduling.definitions import (
    ADVISORY_BACKLOG_ROWS,
    ADVISORY_IDLE_SLOTS,
    ADVISORY_LOW_VOLUME_OPENING_DAY,
    ADVISORY_RESERVE_SLOTS_LOST,
    GREEN_RUN_MEANS,
    OCCUPYING_STATUSES,
    STATUS_REASONS,
    AdjustmentAction,
    CapacityMode,
    InversionReason,
    ScheduleStatus,
    horizon_days,
)
from sentinel.scheduling.models import (
    MAX_OFFENDERS,
    SEVERITY_ERROR,
    SEVERITY_WARN,
    ValidationCheck,
)
from sentinel.scheduling.writer import SCHEMAS, SORT_KEYS

#: Statuses that hold a slot, as a list for polars ``is_in``. Deferred rows count: a deferral
#: moves an inspection rather than removing it, so it still consumes the day's capacity.
_OCCUPYING: list[str] = sorted(str(status) for status in OCCUPYING_STATUSES)

#: The cell key, repeated often enough to be worth a name.
CELL_KEYS: list[str] = [
    "schedule_config_id",
    "policy_id",
    "model_name",
    "fold_set",
    "fold_id",
    "k_name",
]


def _check(
    name: str,
    passed: bool,
    detail: str,
    offenders: Sequence[str] = (),
    severity: str = SEVERITY_ERROR,
) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        passed=passed,
        severity=severity,
        detail=detail,
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


# --- error-severity checks: was the schedule built correctly? ------------------


def tables_are_deterministically_sorted(tables: dict[str, pl.DataFrame]) -> ValidationCheck:
    """Every table in its declared total order, with no duplicated sort key."""
    offenders: list[str] = []
    for name, frame in tables.items():
        if frame.is_empty():
            continue
        keys = SORT_KEYS[name]
        if not frame.equals(frame.sort(keys)):
            offenders.append(f"{name}: not sorted by {', '.join(keys)}")
        if frame.select(keys).is_duplicated().any():
            offenders.append(f"{name}: duplicate sort key")
    return _check(
        "tables_are_deterministically_sorted",
        not offenders,
        f"{len(offenders)} table(s) unsorted or carrying a duplicate key",
        offenders,
    )


def schedule_rows_originate_in_the_recommendation_universe(
    schedule: pl.DataFrame, recommendations: pl.DataFrame
) -> ValidationCheck:
    """Every scheduled row is a row Component 13 selected. No row is invented here."""
    approved = recommendations.filter(pl.col("is_selected")).select(
        "policy_id", "model_name", "fold_set", "fold_id", "k_name", "target_inspection_id"
    )
    joined = schedule.join(
        approved.with_columns(pl.lit(True).alias("_approved")),
        on=["policy_id", "model_name", "fold_set", "fold_id", "k_name", "target_inspection_id"],
        how="left",
    )
    stray = joined.filter(pl.col("_approved").is_null())
    return _check(
        "schedule_rows_originate_in_the_recommendation_universe",
        stray.is_empty(),
        f"{stray.height} scheduled row(s) are not in Component 13's approved queue",
        stray.head(MAX_OFFENDERS)["target_inspection_id"].to_list(),
    )


def every_selected_recommendation_is_accounted_for(
    schedule: pl.DataFrame, recommendations: pl.DataFrame, n_configs: int
) -> ValidationCheck:
    """Nothing silently disappears: every approved row appears once per configuration."""
    approved = (
        recommendations.filter(pl.col("is_selected"))
        .group_by(["policy_id", "model_name", "fold_set", "fold_id", "k_name"])
        .len()
        .rename({"len": "n_approved"})
    )
    planned = (
        schedule.group_by(["policy_id", "model_name", "fold_set", "fold_id", "k_name"])
        .len()
        .rename({"len": "n_planned"})
    )
    merged = approved.join(
        planned,
        on=["policy_id", "model_name", "fold_set", "fold_id", "k_name"],
        how="left",
    ).with_columns(pl.col("n_planned").fill_null(0))
    wrong = merged.filter(pl.col("n_planned") != pl.col("n_approved") * n_configs)
    return _check(
        "every_selected_recommendation_is_accounted_for",
        wrong.is_empty(),
        f"{wrong.height} cell(s) plan a different number of rows than were approved",
        [
            f"{r['policy_id']}/{r['fold_id']}/{r['k_name']}: {r['n_planned']} vs "
            f"{r['n_approved']}x{n_configs}"
            for r in wrong.head(MAX_OFFENDERS).iter_rows(named=True)
        ],
    )


def no_slot_is_double_booked(schedule: pl.DataFrame) -> ValidationCheck:
    """One slot holds one inspection."""
    occupied = schedule.filter(pl.col("schedule_status").is_in(_OCCUPYING))
    if occupied.is_empty():
        return _check("no_slot_is_double_booked", True, "no scheduled rows")
    # Grouped by planning run as well as by cell. A re-plan writes a second plan beside the
    # first, so the same slot legitimately appears once per run; double-booking is two rows in
    # one slot *within* one run.
    dupes = (
        occupied.group_by([*CELL_KEYS, "replan_index", "scheduled_date", "slot_index"])
        .len()
        .filter(pl.col("len") > 1)
    )
    return _check(
        "no_slot_is_double_booked",
        dupes.is_empty(),
        f"{dupes.height} slot(s) hold more than one inspection",
        [
            f"{r['fold_id']}/{r['k_name']} {r['scheduled_date']} slot {r['slot_index']}"
            for r in dupes.head(MAX_OFFENDERS).iter_rows(named=True)
        ],
    )


def no_inspection_occupies_two_slots(schedule: pl.DataFrame) -> ValidationCheck:
    """One approved inspection is one slot.

    Asserted on ``target_inspection_id``, not on ``establishment_id``. Profile 5 measured why:
    1,573 establishment-fold pairs hold more than one scored canvass, and two canvasses of one
    premises in a quarter are two real opportunities with two as-of feature vectors. The
    stronger invariant would have failed on correct data, and a suite that goes red on correct
    data stops being believed.
    """
    occupied = schedule.filter(pl.col("schedule_status").is_in(_OCCUPYING))
    dupes = (
        occupied.group_by([*CELL_KEYS, "replan_index", "target_inspection_id"])
        .len()
        .filter(pl.col("len") > 1)
    )
    return _check(
        "no_inspection_occupies_two_slots",
        dupes.is_empty(),
        f"{dupes.height} inspection(s) occupy more than one slot",
        dupes.head(MAX_OFFENDERS)["target_inspection_id"].to_list(),
    )


def no_day_exceeds_its_capacity(
    schedule: pl.DataFrame, slots: pl.DataFrame, utilization: pl.DataFrame
) -> ValidationCheck:
    """No day holds more inspections than the city worked, and no utilisation exceeds one.

    The check the whole capacity contract exists for. A schedule that fitted more inspections
    than the calendar supplied would beat every alternative for that reason alone.
    """
    occupied = (
        schedule.filter(pl.col("schedule_status").is_in(_OCCUPYING))
        .group_by([*CELL_KEYS, "scheduled_date"])
        .agg(pl.len().alias("n_used"), pl.col("slot_index").max().alias("max_slot"))
    )
    merged = occupied.join(
        slots.select("schedule_config_id", "fold_set", "fold_id", "k_name", "slot_date", "n_slots"),
        left_on=["schedule_config_id", "fold_set", "fold_id", "k_name", "scheduled_date"],
        right_on=["schedule_config_id", "fold_set", "fold_id", "k_name", "slot_date"],
        how="left",
    )
    over = merged.filter(
        pl.col("n_slots").is_null()
        | (pl.col("n_used") > pl.col("n_slots"))
        | (pl.col("max_slot") > pl.col("n_slots"))
    )
    hot = utilization.filter(pl.col("utilization") > 1.0)
    offenders = [
        f"{r['fold_id']}/{r['k_name']} {r['scheduled_date']}: {r['n_used']} in {r['n_slots']}"
        for r in over.head(MAX_OFFENDERS).iter_rows(named=True)
    ]
    return _check(
        "no_day_exceeds_its_capacity",
        over.is_empty() and hot.is_empty(),
        f"{over.height} day(s) over capacity, {hot.height} utilisation(s) above 1.0",
        offenders,
    )


def capacity_matches_the_declared_mode(
    slots: pl.DataFrame, recommendations: pl.DataFrame, medians: dict[str, int]
) -> ValidationCheck:
    """Re-derive every slot count from its declared mode and compare.

    Observed counts are re-derived from Component 13's own ``inspection_date`` column and flat
    counts from Component 5's median. This is what stops a mode label drifting away from the
    arithmetic it names -- a ``flat_median`` row carrying observed volumes would be a scenario
    presenting itself as a measurement, which is the one thing the labelling exists to prevent.
    """
    first_policy = sorted(recommendations["policy_id"].unique().to_list())[0]
    first_k = sorted(recommendations["k_name"].unique().to_list())[0]
    observed = (
        recommendations.filter(
            (pl.col("policy_id") == first_policy) & (pl.col("k_name") == first_k)
        )
        .group_by("fold_id", "inspection_date")
        .len()
        .rename({"len": "expected_observed", "inspection_date": "slot_date"})
    )
    merged = slots.join(observed, on=["fold_id", "slot_date"], how="left").with_columns(
        pl.col("fold_id").replace_strict(medians, default=None).alias("expected_flat")
    )
    wrong = merged.filter(
        pl.when(pl.col("schedule_config_id").str.ends_with(str(CapacityMode.FLAT_MEDIAN)))
        .then(pl.col("n_slots") != pl.col("expected_flat"))
        .otherwise(pl.col("n_slots") != pl.col("expected_observed"))
    )
    return _check(
        "capacity_matches_the_declared_mode",
        wrong.is_empty(),
        f"{wrong.height} operating day(s) carry a slot count its mode does not produce",
        [
            f"{r['schedule_config_id']} {r['fold_id']} {r['slot_date']}: {r['n_slots']}"
            for r in wrong.head(MAX_OFFENDERS).iter_rows(named=True)
        ],
    )


def horizon_is_ordered_contiguous_and_real(
    slots: pl.DataFrame, recommendations: pl.DataFrame, medians: dict[str, int]
) -> ValidationCheck:
    """Day indices contiguous from 1, dates strictly increasing and real, length by the rule."""
    offenders: list[str] = []
    first_policy = sorted(recommendations["policy_id"].unique().to_list())[0]
    first_k = sorted(recommendations["k_name"].unique().to_list())[0]
    real_days = {
        (str(r["fold_id"]), r["inspection_date"])
        for r in recommendations.filter(
            (pl.col("policy_id") == first_policy) & (pl.col("k_name") == first_k)
        )
        .select("fold_id", "inspection_date")
        .unique()
        .iter_rows(named=True)
    }
    for key, group in slots.group_by(
        ["schedule_config_id", "fold_set", "fold_id", "k_name"], maintain_order=True
    ):
        ordered = group.sort("slot_date")
        label = f"{key[0]}/{key[2]}/{key[3]}"
        indices = ordered["day_index"].to_list()
        if indices != list(range(1, len(indices) + 1)):
            offenders.append(f"{label}: day_index not contiguous from 1")
        dates = ordered["slot_date"].to_list()
        if dates != sorted(set(dates)):
            offenders.append(f"{label}: operating days not strictly increasing")
        for value in dates:
            if (str(key[2]), value) not in real_days:
                offenders.append(f"{label}: {value} is not an observed operating day")
                break
        k = int(ordered["k"][0])
        median = medians.get(str(key[2]), 1)
        expected = horizon_days(k, median)
        clamped = bool(ordered["horizon_was_clamped"][0])
        if len(dates) != expected and not clamped:
            offenders.append(f"{label}: {len(dates)} days, rule wants {expected}")
    return _check(
        "horizon_is_ordered_contiguous_and_real",
        not offenders,
        f"{len(offenders)} horizon defect(s)",
        offenders,
    )


def schedule_ranks_are_unique_and_contiguous(schedule: pl.DataFrame) -> ValidationCheck:
    """Ranks 1..n over scheduled rows, and no rank on a row that occupies no slot."""
    offenders: list[str] = []
    stray = schedule.filter(
        ~pl.col("schedule_status").is_in(_OCCUPYING) & pl.col("schedule_rank").is_not_null()
    )
    if stray.height:
        offenders.append(f"{stray.height} unscheduled row(s) carry a schedule_rank")
    occupied = schedule.filter(pl.col("schedule_status").is_in(_OCCUPYING))
    if occupied.height:
        agg = occupied.group_by(CELL_KEYS).agg(
            pl.len().alias("n"),
            pl.col("schedule_rank").min().alias("lo"),
            pl.col("schedule_rank").max().alias("hi"),
            pl.col("schedule_rank").n_unique().alias("distinct"),
        )
        bad = agg.filter(
            (pl.col("lo") != 1)
            | (pl.col("hi") != pl.col("n"))
            | (pl.col("distinct") != pl.col("n"))
        )
        offenders.extend(
            f"{r['fold_id']}/{r['k_name']}: {r['lo']}..{r['hi']} over {r['distinct']}"
            for r in bad.head(MAX_OFFENDERS).iter_rows(named=True)
        )
    return _check(
        "schedule_ranks_are_unique_and_contiguous",
        not offenders,
        f"{len(offenders)} rank defect(s)",
        offenders,
    )


def schedule_order_follows_policy_rank(
    schedule: pl.DataFrame, preservation: pl.DataFrame
) -> ValidationCheck:
    """Slot order equals policy-rank order in every cell no external file touched.

    The reference property of the whole component. Under strict priority with no adjustments
    this is exact, and a violation means the scheduler produced an ordering Component 13 did
    not authorise.
    """
    untouched = schedule.filter(pl.col("adjustment_id") == "").filter(
        pl.col("schedule_status") == ScheduleStatus.SCHEDULED
    )
    offenders: list[str] = []
    for key, group in untouched.group_by(CELL_KEYS, maintain_order=True):
        ordered = group.sort("scheduled_date", "slot_index")
        ranks = ordered["final_policy_rank"].to_list()
        if ranks != sorted(ranks):
            offenders.append(f"{key[1]}/{key[4]}/{key[5]}")
        if len(offenders) >= MAX_OFFENDERS:
            break
    claimed = preservation.filter(
        (pl.col("n_inversions") > 0) & pl.col("strict_priority_preserved")
    )
    if claimed.height:
        offenders.append(f"{claimed.height} cell(s) claim preservation while counting inversions")
    return _check(
        "schedule_order_follows_policy_rank",
        not offenders,
        f"{len(offenders)} cell(s) place a lower-priority row before a higher-priority one",
        offenders,
    )


def no_inversion_without_a_reason_code(schedule: pl.DataFrame) -> ValidationCheck:
    """Every out-of-order row names the mechanism that moved it. No silent inversions."""
    offenders: list[str] = []
    occupied = schedule.filter(pl.col("schedule_status").is_in(_OCCUPYING))
    for key, group in occupied.group_by(CELL_KEYS, maintain_order=True):
        ordered = group.sort("scheduled_date", "slot_index")
        best = 0
        for row in ordered.iter_rows(named=True):
            rank = int(row["final_policy_rank"])
            if rank < best and row["inversion_reason"] == InversionReason.NONE:
                offenders.append(f"{key[1]}/{key[4]}/{key[5]}: {row['target_inspection_id']}")
                break
            best = max(best, rank)
        if len(offenders) >= MAX_OFFENDERS:
            break
    blank = schedule.filter(
        pl.col("inversion_reason").is_null() | (pl.col("inversion_reason") == "")
    )
    if blank.height:
        offenders.append(f"{blank.height} row(s) carry no inversion_reason token at all")
    return _check(
        "no_inversion_without_a_reason_code",
        not offenders,
        f"{len(offenders)} silent inversion(s)",
        offenders,
    )


def every_row_declares_a_valid_status(schedule: pl.DataFrame) -> ValidationCheck:
    """Status and reason pair only in ways the frozen vocabulary permits."""
    offenders: list[str] = []
    pairs = schedule.select("schedule_status", "schedule_reason").unique()
    for status, reason in pairs.iter_rows():
        allowed = STATUS_REASONS.get(str(status))
        if allowed is None:
            offenders.append(f"unknown status {status!r}")
        elif str(reason) not in allowed:
            offenders.append(f"status {status!r} carries reason {reason!r}")
    placed = schedule.filter(
        pl.col("schedule_status").is_in(_OCCUPYING)
        & (pl.col("scheduled_date").is_null() | pl.col("slot_index").is_null())
    )
    if placed.height:
        offenders.append(f"{placed.height} scheduled row(s) carry no date or slot")
    unplaced = schedule.filter(
        ~pl.col("schedule_status").is_in(_OCCUPYING) & pl.col("scheduled_date").is_not_null()
    )
    if unplaced.height:
        offenders.append(f"{unplaced.height} unscheduled row(s) carry a date")
    return _check(
        "every_row_declares_a_valid_status",
        not offenders,
        f"{len(offenders)} status or reason defect(s)",
        offenders,
    )


def backlog_is_exactly_the_unscheduled_remainder(
    schedule: pl.DataFrame, backlog: pl.DataFrame
) -> ValidationCheck:
    """The backlog table holds every backlogged row and nothing else.

    A deferred or backlogged row that vanished from both tables would be an establishment
    nobody is accountable for, which is the failure this pair of tables exists to prevent.
    """
    keys = [*CELL_KEYS, "replan_index", "target_inspection_id"]
    expected = schedule.filter(pl.col("schedule_status") == ScheduleStatus.BACKLOG).select(keys)
    got = backlog.select(keys) if not backlog.is_empty() else expected.head(0)
    missing = expected.join(got, on=keys, how="anti")
    surplus = got.join(expected, on=keys, how="anti")
    return _check(
        "backlog_is_exactly_the_unscheduled_remainder",
        missing.is_empty() and surplus.is_empty(),
        f"{missing.height} backlogged row(s) absent from the table, {surplus.height} stray",
        missing.head(MAX_OFFENDERS)["target_inspection_id"].to_list(),
    )


def counts_add_up(summary: pl.DataFrame) -> ValidationCheck:
    """Every approved row is in exactly one of three places, and nothing is counted twice.

    The identity is ``n_scheduled + n_backlog + n_cancelled == n_recommended``. ``n_deferred``
    is deliberately **not** a fourth term: a deferred row still holds a slot, so it is already
    inside ``n_scheduled``, and adding it again would double-count exactly the rows a supervisor
    moved. It is reported as a breakdown of the scheduled block rather than as a sibling of it,
    and the check asserts that relationship rather than assuming it.
    """
    wrong = summary.filter(
        (pl.col("n_scheduled") + pl.col("n_backlog") + pl.col("n_cancelled"))
        != pl.col("n_recommended")
    )
    nested = summary.filter(pl.col("n_deferred") > pl.col("n_scheduled"))
    if nested.height:
        wrong = pl.concat([wrong, nested])
    idle = summary.filter(
        pl.col("idle_slots") != (pl.col("horizon_slots") - pl.col("n_scheduled")).clip(0)
    )
    return _check(
        "counts_add_up",
        wrong.is_empty() and idle.is_empty(),
        f"{wrong.height} cell(s) do not account for every approved row, {idle.height} idle "
        "count(s) disagree with the horizon",
        [
            f"{r['policy_id']}/{r['fold_id']}/{r['k_name']}"
            for r in wrong.head(MAX_OFFENDERS).iter_rows(named=True)
        ],
    )


def capacity_never_exceeds_the_recommendation_k(
    summary: pl.DataFrame, recommendations: pl.DataFrame
) -> ValidationCheck:
    """The plan never schedules more than Component 13 approved."""
    approved = (
        recommendations.filter(pl.col("is_selected"))
        .group_by(["policy_id", "model_name", "fold_set", "fold_id", "k_name"])
        .len()
        .rename({"len": "n_approved"})
    )
    merged = summary.join(
        approved, on=["policy_id", "model_name", "fold_set", "fold_id", "k_name"], how="left"
    )
    over = merged.filter(
        (pl.col("n_scheduled") > pl.col("n_approved"))
        | (pl.col("n_recommended") != pl.col("n_approved"))
    )
    return _check(
        "capacity_never_exceeds_the_recommendation_k",
        over.is_empty(),
        f"{over.height} cell(s) schedule more than the approved queue",
        [
            f"{r['policy_id']}/{r['fold_id']}/{r['k_name']}"
            for r in over.head(MAX_OFFENDERS).iter_rows(named=True)
        ],
    )


def c13_provenance_is_preserved(
    schedule: pl.DataFrame, recommendations: pl.DataFrame
) -> ValidationCheck:
    """Every Component 13 column on the schedule row equals Component 13's own value.

    The strongest form of "this component never adjusts a score". A drifted value here would
    mean the schedule is annotated with numbers that no longer describe the decision it came
    from, and every downstream reading of provenance would be wrong.
    """
    carried = [
        "base_score",
        "score",
        "model_rank",
        "final_policy_rank",
        "decision_mechanism",
        "decision_reason",
        "coverage_eligible",
        "warnings",
    ]
    source = recommendations.filter(pl.col("is_selected")).select(
        "policy_id",
        "model_name",
        "fold_set",
        "fold_id",
        "k_name",
        "target_inspection_id",
        *[pl.col(c).alias(f"_src_{c}") for c in carried],
    )
    merged = schedule.join(
        source,
        on=["policy_id", "model_name", "fold_set", "fold_id", "k_name", "target_inspection_id"],
        how="inner",
    )
    offenders: list[str] = []
    for column in carried:
        drift = merged.filter(pl.col(column) != pl.col(f"_src_{column}"))
        if drift.height:
            offenders.append(f"{column}: {drift.height} row(s) differ")
    return _check(
        "c13_provenance_is_preserved",
        not offenders,
        f"{len(offenders)} provenance column(s) drifted from Component 13",
        offenders,
    )


def no_outcome_column_reaches_the_schedule(tables: dict[str, pl.DataFrame]) -> ValidationCheck:
    """No label anywhere. A scheduler that could read the outcome could order by it."""
    offenders = [
        f"{name}.{column}"
        for name, frame in tables.items()
        for column in ("target", "target_status")
        if column in frame.columns
    ]
    return _check(
        "no_outcome_column_reaches_the_schedule",
        not offenders,
        f"{len(offenders)} outcome column(s) present in a scheduling artifact",
        offenders,
    )


def completed_rows_are_never_rescheduled(
    schedule: pl.DataFrame, execution_log: pl.DataFrame
) -> ValidationCheck:
    """A completed inspection keeps its slot at every later planning index."""
    if execution_log.is_empty():
        return _check(
            "completed_rows_are_never_rescheduled", True, "no execution events were supplied"
        )
    completed = execution_log.filter(pl.col("execution_status") == "completed").select(
        "policy_id", "fold_id", "k_name", "target_inspection_id"
    )
    if completed.is_empty():
        return _check("completed_rows_are_never_rescheduled", True, "no completed events")
    joined = schedule.join(
        completed, on=["policy_id", "fold_id", "k_name", "target_inspection_id"], how="inner"
    )
    moved = (
        joined.group_by(
            ["schedule_config_id", "policy_id", "fold_id", "k_name", "target_inspection_id"]
        )
        .agg(pl.col("scheduled_date").n_unique().alias("dates"))
        .filter(pl.col("dates") > 1)
    )
    return _check(
        "completed_rows_are_never_rescheduled",
        moved.is_empty(),
        f"{moved.height} completed inspection(s) were moved by a later planning run",
        moved.head(MAX_OFFENDERS)["target_inspection_id"].to_list(),
    )


def execution_never_alters_a_recommendation(
    schedule: pl.DataFrame, recommendations: pl.DataFrame
) -> ValidationCheck:
    """An execution outcome cannot retroactively change a recommendation.

    Enforced structurally as well as here: ``inspection_schedule`` has no ``execution_status``
    column, so there is nothing for an execution event to write into. This check proves the
    consequence -- that provenance is identical at every planning index.
    """
    if "execution_status" in schedule.columns:
        return _check(
            "execution_never_alters_a_recommendation",
            False,
            "the schedule carries an execution_status column, which gives an execution outcome "
            "a place to write into a plan",
        )
    per_index = (
        schedule.group_by(
            ["schedule_config_id", "policy_id", "fold_id", "k_name", "target_inspection_id"]
        )
        .agg(
            pl.col("score").n_unique().alias("scores"),
            pl.col("final_policy_rank").n_unique().alias("ranks"),
            pl.col("decision_mechanism").n_unique().alias("mechanisms"),
        )
        .filter((pl.col("scores") > 1) | (pl.col("ranks") > 1) | (pl.col("mechanisms") > 1))
    )
    del recommendations
    return _check(
        "execution_never_alters_a_recommendation",
        per_index.is_empty(),
        f"{per_index.height} row(s) carry different provenance across planning runs",
        per_index.head(MAX_OFFENDERS)["target_inspection_id"].to_list(),
    )


def no_execution_event_changes_an_earlier_schedule(
    runs: pl.DataFrame, schedule: pl.DataFrame, execution_log: pl.DataFrame | None = None
) -> ValidationCheck:
    """Nothing before a re-plan point moves, unless the field reported it did not happen.

    The temporal boundary, proved from the artifacts rather than asserted. The exemption is
    load-bearing and narrow: a row the field explicitly reported as ``not_performed`` is the
    one thing a re-plan exists to move, and freezing it because its slot is in the past would
    strand the inspection the report was filed to rescue. Every *other* row on a day before the
    boundary must be byte-identical between the two plans -- including completed ones, and
    including ones nobody reported on at all.
    """
    if runs.filter(pl.col("replan_index") > 0).is_empty():
        return _check(
            "no_execution_event_changes_an_earlier_schedule", True, "no re-planning run occurred"
        )
    reported: set[str] = set()
    if execution_log is not None and not execution_log.is_empty():
        reported = set(
            execution_log.filter(pl.col("execution_status") == "not_performed")[
                "target_inspection_id"
            ].to_list()
        )
    offenders: list[str] = []
    for run in runs.filter(pl.col("replan_index") > 0).iter_rows(named=True):
        boundary = run["replan_from_date"]
        if boundary is None and run["trigger"] == "scheduling_adjustment":
            # An adjustment run has no re-plan point: it is a human moving one named row, and
            # the adjustment log is what audits it.
            continue
        if boundary is None:
            offenders.append(f"{run['planning_run_id']}: no re-plan point recorded")
            continue
        cell = schedule.filter(
            (pl.col("schedule_config_id") == run["schedule_config_id"])
            & (pl.col("policy_id") == run["policy_id"])
            & (pl.col("fold_id") == run["fold_id"])
            & (pl.col("k_name") == run["k_name"])
        )
        # Compare this run against its parent, row by row. A row whose assignment in the
        # PARENT plan fell before the boundary must be identical in the child; anything else
        # is a re-plan reaching backwards into a day that has already been worked.
        parent = cell.filter(pl.col("replan_index") == run["parent_replan_index"]).select(
            "target_inspection_id",
            pl.col("scheduled_date").alias("_parent_date"),
            pl.col("schedule_status").alias("_parent_status"),
        )
        child = cell.filter(pl.col("replan_index") == run["replan_index"]).select(
            "target_inspection_id", "scheduled_date", "schedule_status"
        )
        joined = parent.join(child, on="target_inspection_id", how="inner")
        if reported:
            joined = joined.filter(~pl.col("target_inspection_id").is_in(list(reported)))
        before = joined.filter(
            pl.col("_parent_date").is_not_null()
            & (pl.col("_parent_date") < boundary)
            & (
                (pl.col("scheduled_date") != pl.col("_parent_date"))
                | (pl.col("schedule_status") != pl.col("_parent_status"))
            )
        )
        if before.height:
            offenders.append(f"{run['planning_run_id']}: {before.height} row(s) before {boundary}")
    return _check(
        "no_execution_event_changes_an_earlier_schedule",
        not offenders,
        f"{len(offenders)} re-plan(s) touched an operating day before their own boundary",
        offenders,
    )


def planning_runs_are_unique_and_chained(runs: pl.DataFrame) -> ValidationCheck:
    """Indices start at zero, advance by one, and no id is used twice."""
    offenders: list[str] = []
    if runs.is_empty():
        return _check(
            "planning_runs_are_unique_and_chained",
            False,
            "no planning run was recorded. The original plan is a planning run and must appear",
        )
    dupes = runs.group_by("planning_run_id").len().filter(pl.col("len") > 1)
    if dupes.height:
        offenders.append(f"{dupes.height} duplicated planning_run_id")
    for key, group in runs.group_by(
        ["schedule_config_id", "policy_id", "fold_id", "k_name"], maintain_order=True
    ):
        ordered = group.sort("replan_index")
        indices = ordered["replan_index"].to_list()
        if indices != list(range(len(indices))):
            offenders.append(f"{key[1]}/{key[2]}/{key[3]}: indices {indices[:5]}")
            continue
        parents = ordered["parent_replan_index"].to_list()
        for index, parent in enumerate(parents):
            if index == 0 and parent is not None:
                offenders.append(f"{key[1]}/{key[2]}/{key[3]}: run 0 names a parent")
            elif index > 0 and parent != index - 1:
                offenders.append(f"{key[1]}/{key[2]}/{key[3]}: run {index} parent {parent}")
        if len(offenders) >= MAX_OFFENDERS:
            break
    return _check(
        "planning_runs_are_unique_and_chained",
        not offenders,
        f"{len(offenders)} planning-run lineage defect(s)",
        offenders,
    )


def adjustments_are_not_overrides(
    adjustment_log: pl.DataFrame, override_ids: Sequence[str]
) -> ValidationCheck:
    """A scheduling adjustment is never a recommendation override.

    Three separate guarantees: the id namespaces do not collide, the verbs do not collide, and
    the adjustment log carries no selection column. Merging the two layers is the most plausible
    way to lose the chain from a human decision to a planned date, so it is checked rather than
    assumed.
    """
    offenders: list[str] = []
    if not adjustment_log.is_empty():
        collisions = set(adjustment_log["adjustment_id"].to_list()) & set(override_ids)
        if collisions:
            offenders.append(f"id collision: {', '.join(sorted(collisions)[:3])}")
        verbs = set(adjustment_log["action"].unique().to_list())
        stray = verbs & {str(a) for a in OverrideAction}
        if stray:
            offenders.append(f"override verb in an adjustment: {', '.join(sorted(stray))}")
        unknown = verbs - {str(a) for a in AdjustmentAction}
        if unknown:
            offenders.append(f"unknown adjustment verb: {', '.join(sorted(unknown))}")
    for column in ("is_selected", "final_policy_rank", "decision_mechanism"):
        if column in adjustment_log.columns:
            offenders.append(f"the adjustment log carries {column}, which is a queue column")
    return _check(
        "adjustments_are_not_overrides",
        not offenders,
        f"{len(offenders)} confusion(s) between the two human layers",
        offenders,
    )


def adjustments_preserve_the_original_assignment(
    schedule: pl.DataFrame, adjustment_log: pl.DataFrame
) -> ValidationCheck:
    """Every adjusted row still names where it was originally going to be."""
    if adjustment_log.is_empty():
        return _check(
            "adjustments_preserve_the_original_assignment", True, "no adjustments were supplied"
        )
    touched = schedule.filter(pl.col("adjustment_id") != "")
    lost = touched.filter(
        pl.col("original_scheduled_date").is_null() & pl.col("original_schedule_rank").is_null()
    )
    applied = adjustment_log.filter(pl.col("outcome") == "applied")
    blank = applied.filter(pl.col("original_status").is_null() | (pl.col("original_status") == ""))
    return _check(
        "adjustments_preserve_the_original_assignment",
        lost.is_empty() and blank.is_empty(),
        f"{lost.height} adjusted row(s) lost their original assignment, {blank.height} log "
        "row(s) record no original status",
        lost.head(MAX_OFFENDERS)["target_inspection_id"].to_list(),
    )


def adjustments_never_displace_a_coverage_reserve_row(
    adjustment_log: pl.DataFrame, schedule: pl.DataFrame
) -> ValidationCheck:
    """A scheduling change never spends the coverage allocation.

    Component 13's override contract refuses to raid the reserve for the same reason: taking
    the slot from the coverage allocation would quietly convert every scheduling change into a
    coverage cut, and the coverage allocation is the one thing in this pipeline that was priced
    in forgone citations.
    """
    if adjustment_log.is_empty():
        return _check(
            "adjustments_never_displace_a_coverage_reserve_row", True, "no adjustments supplied"
        )
    displaced = adjustment_log.filter(pl.col("displaced_target_inspection_id") != "").select(
        pl.col("displaced_target_inspection_id").alias("target_inspection_id")
    )
    if displaced.is_empty():
        return _check(
            "adjustments_never_displace_a_coverage_reserve_row", True, "nothing was displaced"
        )
    reserve = schedule.filter(
        pl.col("decision_mechanism") == DecisionMechanism.COVERAGE_RESERVE
    ).select("target_inspection_id")
    bad = displaced.join(reserve, on="target_inspection_id", how="inner").unique()
    return _check(
        "adjustments_never_displace_a_coverage_reserve_row",
        bad.is_empty(),
        f"{bad.height} coverage-reserve row(s) were displaced by a scheduling adjustment",
        bad.head(MAX_OFFENDERS)["target_inspection_id"].to_list(),
    )


def external_changes_are_fully_attributed(
    adjustment_log: pl.DataFrame, execution_log: pl.DataFrame
) -> ValidationCheck:
    """Every external change names an actor, a reason and a time.

    An external change with no attribution is an anonymous change to who gets inspected when,
    which is the precise thing an audit trail exists to prevent.
    """
    offenders: list[str] = []
    for frame, name, stamp in (
        (adjustment_log, "adjustment", "decided_at"),
        (execution_log, "execution", "observed_at"),
    ):
        if frame.is_empty():
            continue
        for column in ("actor", "reason_code", stamp):
            blank = frame.filter(pl.col(column).is_null() | (pl.col(column) == ""))
            if blank.height:
                offenders.append(f"{name}: {blank.height} row(s) with no {column}")
    return _check(
        "external_changes_are_fully_attributed",
        not offenders,
        f"{len(offenders)} attribution gap(s)",
        offenders,
    )


def the_deterministic_plan_is_intact(
    schedule: pl.DataFrame, rebuilt: pl.DataFrame | None
) -> ValidationCheck:
    """The plan built with the external files withheld matches the artifact at index 0.

    Rebuilds rather than inspects, because the property is a claim about what the code does
    under different inputs. Component 13's ``warnings_do_not_change_the_queue`` is the same
    shape and exists for the same reason.
    """
    if rebuilt is None:
        return _check(
            "the_deterministic_plan_is_intact",
            True,
            "no external file was supplied, so the written plan is the deterministic plan",
        )
    baseline = schedule.filter(pl.col("replan_index") == 0).drop("adjustment_id")
    comparable = rebuilt.drop("adjustment_id")
    return _check(
        "the_deterministic_plan_is_intact",
        baseline.equals(comparable),
        (
            "the plan is byte-identical with the adjustment and execution inputs withheld"
            if baseline.equals(comparable)
            else "an external file changed the deterministic plan, which must be written "
            "unchanged beside the logs"
        ),
    )


def inputs_were_not_modified(before: dict[str, str], after: dict[str, str]) -> ValidationCheck:
    """Every input checksum, re-read after the last table was written."""
    moved = [name for name, digest in before.items() if after.get(name) != digest]
    return _check(
        "inputs_were_not_modified",
        not moved,
        f"{len(moved)} input artifact(s) changed during the run",
        moved,
    )


def configurations_match_the_frozen_grid(
    configurations: pl.DataFrame, expected: Sequence[str]
) -> ValidationCheck:
    """The emitted grid is the frozen grid."""
    got = sorted(configurations["schedule_config_id"].to_list())
    want = sorted(expected)
    return _check(
        "configurations_match_the_frozen_grid",
        got == want,
        f"{len(set(got) ^ set(want))} configuration(s) differ from the frozen grid",
        sorted(set(got) ^ set(want)),
    )


# --- advisory-severity checks: what did the schedule cost? --------------------


def capacity_is_fully_utilized(summary: pl.DataFrame) -> ValidationCheck:
    """Idle slots the queue did not fill. Never an error: a thin queue is not a defect."""
    idle = summary.filter(pl.col("idle_slots") >= ADVISORY_IDLE_SLOTS)
    total = int(idle["idle_slots"].sum() or 0)
    return _check(
        "capacity_is_fully_utilized",
        idle.is_empty(),
        f"{idle.height} cell(s) left {total} slot(s) idle. The calendar was more generous than "
        "the cutoff, which is the mirror image of a backlog and a different problem",
        severity=SEVERITY_WARN,
    )


def every_recommendation_was_scheduled(summary: pl.DataFrame) -> ValidationCheck:
    """Approved rows the horizon could not reach.

    Advisory by design. Insufficient capacity is a measured property of the city's calendar,
    and a run that went red on it would be a run that goes green only when the scheduler is
    lying about the calendar.
    """
    short = summary.filter(pl.col("n_backlog") >= ADVISORY_BACKLOG_ROWS)
    total = int(short["n_backlog"].sum() or 0)
    return _check(
        "every_recommendation_was_scheduled",
        short.is_empty(),
        f"{short.height} cell(s) could not fit {total} approved inspection(s) inside their own "
        "horizon. Reported rather than hidden, and never corrected by extending the horizon",
        severity=SEVERITY_WARN,
    )


def the_coverage_reserve_survived_scheduling(preservation: pl.DataFrame) -> ValidationCheck:
    """Coverage-reserve slots lost to a short horizon. **The component's headline.**

    Advisory, and it must stay advisory. Component 13 places the reserve at the tail of the
    rank order, so a horizon that falls short takes the reserve first, every time. The cheapest
    way to turn a red build green here would be to make the scheduler prefer reserve rows --
    which is re-ranking, which is forbidden, and which would move a coverage decision into a
    layer that does not own it.
    """
    lost = preservation.filter(pl.col("reserve_slots_lost") >= ADVISORY_RESERVE_SLOTS_LOST)
    # Measured on the observed calendar only. The flat_median scenario loses nothing by
    # construction -- it supplies k slots for a queue of k -- so pooling the two would average
    # a tautology into the finding and halve it. The scenario is reported on its own line.
    observed = preservation.filter(~pl.col("is_scenario"))
    bearing = observed.filter(pl.col("n_reserve_recommended") > 0)
    offered = int(bearing["n_reserve_recommended"].sum() or 0)
    total = int(bearing["reserve_slots_lost"].sum() or 0)
    wiped = bearing.filter(pl.col("n_reserve_scheduled") == 0)
    hit = bearing.filter(pl.col("reserve_slots_lost") > 0)
    share = (total / offered) if offered else 0.0
    scenario_lost = int(preservation.filter(pl.col("is_scenario"))["reserve_slots_lost"].sum() or 0)
    return _check(
        "the_coverage_reserve_survived_scheduling",
        lost.is_empty(),
        f"on the observed calendar, {total} of {offered} coverage-reserve slot(s) ({share:.3f}) "
        f"were lost to the horizon across {hit.height} of {bearing.height} reserve-bearing "
        f"cell(s); {wiped.height} lost the reserve entirely. The flat_median scenario lost "
        f"{scenario_lost}, which is zero by construction and is why the two are not pooled. A "
        "measured consequence of the reserve sitting at the tail of Component 13's rank order, "
        "reported rather than corrected",
        severity=SEVERITY_WARN,
    )


def the_scenario_is_not_observed_fact(schedule: pl.DataFrame) -> ValidationCheck:
    """Fires whenever a scenario row was written. Always, when the mode ran."""
    scenario = schedule.filter(pl.col("is_scenario"))
    return _check(
        "the_scenario_is_not_observed_fact",
        scenario.is_empty(),
        f"{scenario.height} row(s) come from the flat_median scenario, which assigns every day "
        "the window's median rate. At k_1_day and k_1_week it holds exactly k slots and reports "
        "zero backlog by construction; those numbers describe the assumption, not a day",
        severity=SEVERITY_WARN,
    )


def an_execution_record_was_supplied(execution_log: pl.DataFrame) -> ValidationCheck:
    """Fires when nobody supplied an execution file, so every execution number is a typed zero."""
    return _check(
        "an_execution_record_was_supplied",
        not execution_log.is_empty(),
        f"{execution_log.height} execution event(s) were supplied. With none, every execution "
        "and completion number in this run is a typed zero rather than a measurement, and no "
        "row in this repository describes anything that happened in Chicago",
        severity=SEVERITY_WARN,
    )


def the_horizon_opens_on_a_full_day(slots: pl.DataFrame) -> ValidationCheck:
    """Fires when a horizon's first operating day is far below the window's median rate.

    Measured rather than chosen: quarter-opening days are systematically thin -- 2024Q1 and
    2026Q1 both open on a single inspection against medians of 34 and 35 -- and every
    ``k_1_day`` number is dominated by that one day. The alternative was to start the horizon on
    the first day above a volume floor, which would import an arbitrary constant and a selection
    effect at once. Anchoring on the window's own start and reporting the thinness is the
    honest version.
    """
    opening = slots.filter(pl.col("day_index") == 1)
    if opening.is_empty():
        return _check(
            "the_horizon_opens_on_a_full_day", True, "no horizons", severity=SEVERITY_WARN
        )
    thin = opening.filter(
        pl.col("n_slots") < (pl.col("median_daily_capacity") * ADVISORY_LOW_VOLUME_OPENING_DAY)
    )
    return _check(
        "the_horizon_opens_on_a_full_day",
        thin.is_empty(),
        f"{thin.height} of {opening.height} horizon(s) open on a day holding less than "
        f"{ADVISORY_LOW_VOLUME_OPENING_DAY:.0%} of the window's median rate. Every one-day "
        "capacity number in those cells is dominated by an unrepresentative day",
        severity=SEVERITY_WARN,
    )


def an_establishment_recurs_within_a_horizon(schedule: pl.DataFrame) -> ValidationCheck:
    """Fires when one premises holds two scored canvasses inside one queue.

    Advisory rather than error, and profile 5 is the reason. Component 13's grain is the scored
    inspection event; two canvasses of one premises in a quarter are two real opportunities with
    two as-of feature vectors. Asserting uniqueness on the establishment would have produced a
    red build on correct data.
    """
    occupied = schedule.filter(pl.col("schedule_status").is_in(_OCCUPYING))
    if occupied.is_empty():
        return _check(
            "an_establishment_recurs_within_a_horizon",
            True,
            "no scheduled rows",
            severity=SEVERITY_WARN,
        )
    dupes = occupied.group_by([*CELL_KEYS, "establishment_id"]).len().filter(pl.col("len") > 1)
    return _check(
        "an_establishment_recurs_within_a_horizon",
        dupes.is_empty(),
        f"{dupes.height} establishment(s) hold more than one scheduled canvass inside a single "
        "horizon. A data fact about repeat canvasses, not a scheduling defect",
        severity=SEVERITY_WARN,
    )


# --- reporting ----------------------------------------------------------------


def format_report(checks: Sequence[ValidationCheck]) -> str:
    """A human-readable report, failures first, with the boundary line at the top."""
    lines = [f"  {GREEN_RUN_MEANS}", ""]
    order = sorted(checks, key=lambda c: (c.passed, c.severity != SEVERITY_ERROR, c.name))
    for check in order:
        mark = "PASS" if check.passed else ("FAIL" if check.severity == SEVERITY_ERROR else "NOTE")
        lines.append(f"  [{mark}] {check.name} ({check.severity})")
        lines.append(f"         {check.detail}")
        for offender in check.offenders:
            lines.append(f"           - {offender}")
    errors = sum(1 for c in checks if not c.passed and c.severity == SEVERITY_ERROR)
    warns = sum(1 for c in checks if not c.passed and c.severity == SEVERITY_WARN)
    lines.append("")
    lines.append(f"  {len(checks)} checks, {errors} error(s), {warns} advisory finding(s)")
    return "\n".join(lines)


def has_failures(checks: Sequence[ValidationCheck]) -> bool:
    """True when any error-severity check failed. Advisories never fail a run."""
    return any(not c.passed and c.severity == SEVERITY_ERROR for c in checks)


def advisory_findings(checks: Sequence[ValidationCheck]) -> list[str]:
    """The advisory findings, for the manifest."""
    return [f"{c.name}: {c.detail}" for c in checks if not c.passed and c.severity == SEVERITY_WARN]


def advisory_rows(checks: Sequence[ValidationCheck], version: str) -> list[dict[str, object]]:
    """The advisory table: one row per advisory finding."""
    rows: list[dict[str, object]] = []
    for check in checks:
        if check.passed or check.severity != SEVERITY_WARN:
            continue
        rows.append(
            {
                "code": check.name,
                "severity": check.severity,
                "scope": "run",
                "n_cells": len(check.offenders),
                "detail": check.detail,
                "schedule_definition_version": version,
            }
        )
    return rows


def check_rows(checks: Sequence[ValidationCheck]) -> list[dict[str, object]]:
    """The checks, for the manifest."""
    return [
        {
            "name": check.name,
            "passed": check.passed,
            "severity": check.severity,
            "detail": check.detail,
        }
        for check in checks
    ]


def _known_tables() -> frozenset[str]:
    return frozenset(SCHEMAS)


__all__ = [
    "CELL_KEYS",
    "adjustments_are_not_overrides",
    "adjustments_never_displace_a_coverage_reserve_row",
    "adjustments_preserve_the_original_assignment",
    "advisory_findings",
    "advisory_rows",
    "an_establishment_recurs_within_a_horizon",
    "an_execution_record_was_supplied",
    "backlog_is_exactly_the_unscheduled_remainder",
    "c13_provenance_is_preserved",
    "capacity_is_fully_utilized",
    "capacity_matches_the_declared_mode",
    "capacity_never_exceeds_the_recommendation_k",
    "check_rows",
    "completed_rows_are_never_rescheduled",
    "configurations_match_the_frozen_grid",
    "counts_add_up",
    "every_recommendation_was_scheduled",
    "every_row_declares_a_valid_status",
    "every_selected_recommendation_is_accounted_for",
    "execution_never_alters_a_recommendation",
    "external_changes_are_fully_attributed",
    "format_report",
    "has_failures",
    "horizon_is_ordered_contiguous_and_real",
    "inputs_were_not_modified",
    "no_day_exceeds_its_capacity",
    "no_execution_event_changes_an_earlier_schedule",
    "no_inspection_occupies_two_slots",
    "no_inversion_without_a_reason_code",
    "no_outcome_column_reaches_the_schedule",
    "no_slot_is_double_booked",
    "planning_runs_are_unique_and_chained",
    "schedule_order_follows_policy_rank",
    "schedule_ranks_are_unique_and_contiguous",
    "schedule_rows_originate_in_the_recommendation_universe",
    "tables_are_deterministically_sorted",
    "the_coverage_reserve_survived_scheduling",
    "the_deterministic_plan_is_intact",
    "the_horizon_opens_on_a_full_day",
    "the_scenario_is_not_observed_fact",
]
