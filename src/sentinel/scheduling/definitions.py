"""Frozen contracts for Component 14. Every scheduling constant lives here and nowhere else.

**The separation this module exists to hold.** Component 13 decides *who* should be inspected
at a stated capacity, and why. This component decides *when* the approved queue is executed.
Those are different questions with different owners: a queue changes when a department changes
its mind about coverage, a schedule changes when a Tuesday turns out to hold sixteen
inspections instead of twenty-eight. A reader who wants to know why an establishment is in the
queue looks at Component 13. A reader who wants to know why it is on Thursday looks here.

**Nothing here is a policy, and nothing here is a model.** No score is produced, adjusted or
read into an ordering; no rank is recomputed; no capacity is raised; no probability threshold
exists and there is no flag to add one. The scheduler consumes ``final_policy_rank`` and the
observed calendar, and it has no other inputs that could move an establishment. Every one of
those refusals is enumerated in ``BLOCKED`` and travels inside every manifest, so the boundary
arrives with the artifact rather than living only in documentation.

**Every number in this file came from a measurement.** ``scripts/profile_scheduling.py`` ran
first, over the frozen artifacts, and ``docs/analysis/scheduling_findings.md`` holds its
output. The horizon rule, the default capacity mode, the advisory thresholds and the decision
to treat establishment recurrence as an advisory rather than an error are all set from that
output. Component 9 set three thresholds from expectation and had to correct all three; a
scheduling constant chosen the same way would be worse, because a scheduling constant decides
who waits.

**One vocabulary decision is an omission, and it is deliberate.** There is no
``constraint_adjusted`` reason code, because no real or explicitly-configured operational
constraint exists in this dataset -- no closure calendar, no deadline, no availability window.
A reason code no run can emit is indistinguishable from one that is broken, which is the rule
``policy/definitions.py`` states for mechanisms and this module inherits for reasons.
``NO_CONSTRAINT_AWARE_STRATEGY`` records the omission and why.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sentinel.policy.definitions import K_LEVELS as POLICY_K_LEVELS
from sentinel.policy.definitions import PRIMARY_K_LEVEL as POLICY_PRIMARY_K_LEVEL
from sentinel.policy.definitions import DecisionMechanism, OverrideAction

#: Bumped whenever the horizon rule, the capacity contract, the placement semantics or either
#: external contract changes in a way that makes two runs incomparable.
SCHEDULE_DEFINITION_VERSION = "v1"


class SchedulingDefinitionError(ValueError):
    """Raised when the frozen scheduling contracts contradict each other."""


# --- 1. the layer separation ---------------------------------------------------

#: The five layers, and the four boundaries that must never be crossed. Printed in the manifest
#: because the whole component is an argument that these are different things.
LAYER_SEPARATION = (
    "model -> policy -> recommendation -> schedule -> execution. A calibrated probability "
    "never becomes a date without passing through a policy rank; a scheduling rule never "
    "edits a probability, a rank or a mechanism; an execution outcome never retroactively "
    "changes a recommendation or a plan that was already written"
)

#: What this component does, in the words it must be described in. The negative half is
#: load-bearing: a reader who sees "scheduling" and assumes routing has been misled by the
#: name alone.
SCHEDULING_SEMANTICS = (
    "Component 14 performs temporal and workload scheduling, not geographic route "
    "optimisation. The dataset has no inspector, no shift, no duration, no travel time and no "
    "road network, so a route here is not underdetermined -- it is unrepresented. Every slot "
    "is a position on an operating day, and every operating day and its slot count is measured "
    "from the fold's own observed inspection calendar"
)

#: Capacity is inherited, never created. Stated separately from the semantics because it is
#: the single easiest way for a scheduler to produce a flattering number.
CAPACITY_IS_INHERITED = (
    "every capacity in this component descends from Component 5's measured median daily "
    "inspection rate by way of Component 13's k. There is no flag that raises a slot count, "
    "extends a horizon or adds a day, because a schedule that fitted more inspections than the "
    "city worked would beat every alternative for that reason alone"
)

#: Priority is inherited, never recomputed.
PRIORITY_IS_INHERITED = (
    "final_policy_rank is the only ordering key this component has. Nothing here reads a "
    "score, a probability, a mechanism or a geography to decide who goes first, and a "
    "scheduler that preferred -- or preferred against -- a coverage-reserve row for a slot "
    "would be a second policy layer with no ADR behind it"
)


# --- 2. the horizon and the capacity contract ----------------------------------


class CapacityMode(StrEnum):
    """Where a horizon day's slot count comes from.

    Two modes, and the contrast between them is the component's central measurement. They are
    not two spellings of one thing: one is an observation and the other is an assumption, and
    the difference between them is 784 recommended inspections.
    """

    #: The number of inspections Chicago actually performed on that date, read from the
    #: recommendation universe's ``inspection_date`` column. Measured, and the default.
    OBSERVED_CALENDAR = "observed_calendar"

    #: Component 5's ``test_median_daily_capacity`` on every horizon day. This reproduces
    #: Component 13's own stated capacity semantics exactly, and it is a **scenario**: it is
    #: what the capacity cutoffs assume, not what any particular day held. Retained because
    #: the assumption is worth being able to see, and because at ``k_1_day`` and ``k_1_week``
    #: it is provably tautological -- which is the most useful thing it demonstrates.
    FLAT_MEDIAN = "flat_median"


#: Measured, so it is the default. Profile 3 is the argument: under the flat median the
#: backlog is zero in every one of 90 cells by construction, and under the observed calendar
#: it is non-zero in 44 of them.
DEFAULT_CAPACITY_MODE = CapacityMode.OBSERVED_CALENDAR

#: The label that must never be dropped from a scenario number. Carried on every row of every
#: table that has a mode column, restated in the manifest, and printed in the CLI summary.
CAPACITY_MODE_SCENARIO_CLAIM = (
    "flat_median is a labelled scenario, never an observed Chicago operational fact. It "
    "assigns every horizon day the window's median daily rate, which is what Component 13's "
    "capacity cutoffs already assume -- so at k_1_day and k_1_week it holds exactly k slots, "
    "reports a backlog of zero and a utilisation of exactly 1.000, and is describing its own "
    "arithmetic rather than any day that happened"
)

#: The horizon rule. Not a new constant: the inverse of the rule
#: ``evaluation.simulate.capacity_k_values`` used to produce ``k`` in the first place, read
#: backwards. It reproduces both day-denominated cutoff names exactly -- ``k_1_day`` is one day
#: and ``k_1_week`` is five -- which is why it is one rule rather than a table of cases.
HORIZON_RULE = (
    "horizon_days = ceil(k / test_median_daily_capacity), taken as a prefix of the fold's own "
    "observed operating days. The horizon is identical in both capacity modes; only the "
    "per-day slot count differs, so that a mode change moves capacity and never the calendar"
)

#: The calendar is read, never generated. Profile 4 is the argument: three inspections fall on
#: a weekend, so a synthesised Monday-to-Friday calendar would be wrong at the edges, and a
#: holiday list is something this project has no way to verify.
CALENDAR_IS_OBSERVED = (
    "operating days are the distinct inspection_date values Component 13's universe carries "
    "for the fold, ascending. No working-week rule, no holiday calendar and no generated date "
    "appears anywhere in this component; a day is an operating day because Chicago inspected "
    "on it"
)

#: Component 13's capacity levels, by identity rather than by copy. Restating them here would
#: create a second list that could silently disagree with the first.
K_LEVELS: tuple[str, ...] = POLICY_K_LEVELS
PRIMARY_K_LEVEL: str = POLICY_PRIMARY_K_LEVEL


def horizon_days(k: int, median_daily_capacity: int) -> int:
    """Operating days needed to work a queue of ``k`` at the window's median daily rate.

    Integer arithmetic rather than ``math.ceil`` over floats, so the exact-multiple boundary
    -- which is where ``k_1_day`` and ``k_1_week`` both live -- cannot depend on a floating
    point representation.
    """
    if k < 1:
        raise SchedulingDefinitionError(f"capacity k must be at least 1, got {k}")
    if median_daily_capacity < 1:
        raise SchedulingDefinitionError(
            f"median daily capacity must be at least 1, got {median_daily_capacity}"
        )
    return -(-k // median_daily_capacity)


# --- 3. the scheduling strategies ----------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """One way of turning an approved queue into a sequence of slots."""

    strategy_id: str
    preserves_priority_exactly: bool
    rationale: str


#: Exactly one strategy, and the count is a finding rather than an omission.
#:
#: A constraint-aware strategy would need a constraint. The spec that asked for one made it
#: conditional on real or explicitly-configured operational constraints existing, and profile 7
#: is the inventory showing that none do: no closure calendar, no deadline, no availability
#: window, no inspector. A second strategy here would be a strategy over invented inputs.
STRATEGY_GRID: tuple[StrategySpec, ...] = (
    StrategySpec(
        strategy_id="strict_priority",
        preserves_priority_exactly=True,
        rationale=(
            "fill the horizon day by day, in date order, from the approved queue in "
            "final_policy_rank order. The reference schedule, and the only one this data "
            "supports: Component 13 owns the ordering and a scheduler that produced a second "
            "one would be an unowned policy layer"
        ),
    ),
)

#: Why there is no strategy B, recorded so the absence reads as a decision rather than a gap.
NO_CONSTRAINT_AWARE_STRATEGY = (
    "no constraint-aware strategy exists because no operational constraint exists in this "
    "dataset to be aware of. Establishment closures, statutory deadlines, appointment windows "
    "and inspector availability are all absent (profile 7), and a strategy that traded off "
    "invented parameters would produce a schedule that looks considered and is arbitrary. The "
    "external adjustment contract is the supported way to move a row, because it carries an "
    "actor, a reason code and a timestamp -- a constraint somebody is accountable for"
)

#: Why no solver arrives with this component, answering the promise pyproject.toml made.
#:
#: The promise was that a dependency would arrive with the component that needed one. It is
#: kept by checking rather than by assuming: strict priority preservation has a closed form
#: (fill days in date order from a queue in rank order), ``final_policy_rank`` is unique and
#: contiguous so there is nothing to search over, and every constraint an optimiser would
#: trade off is absent from the data. A solver would also be non-deterministic in the one way
#: that matters -- equal objective values return a search-order-dependent solution.
NO_SOLVER = (
    "Component 14 adds no solver and no dependency. Strict priority preservation is a prefix "
    "operation with a closed form, not a search: final_policy_rank is unique and contiguous, "
    "so the assignment is unique and there is nothing to optimise over. Every constraint an "
    "optimiser would need -- travel time, duration, inspector count, skill matching -- is "
    "absent, so the model would be a model over invented parameters and its output would be an "
    "optimality claim with fabricated inputs"
)

#: What the allocation actually is, in the words it must be described in. ADR 0043's line.
ALLOCATION_CLAIM = (
    "deterministic greedy slot allocation down an approved rank order. Not optimal, not "
    "optimised, and no objective function is defined or solved anywhere in this component"
)


@dataclass(frozen=True, slots=True)
class ScheduleConfigSpec:
    """One (strategy, capacity mode) pair. The unit a run is parameterised by."""

    schedule_config_id: str
    strategy_id: str
    capacity_mode: CapacityMode
    is_scenario: bool
    is_default: bool
    rationale: str


#: The frozen configuration grid. Two configurations, emitted together by default so the
#: scenario's divergence from the observed calendar is always visible rather than opt-in.
CONFIG_GRID: tuple[ScheduleConfigSpec, ...] = (
    ScheduleConfigSpec(
        schedule_config_id="strict_priority__observed_calendar",
        strategy_id="strict_priority",
        capacity_mode=CapacityMode.OBSERVED_CALENDAR,
        is_scenario=False,
        is_default=True,
        rationale=(
            "the approved queue laid against the days Chicago actually worked, at the volumes "
            "it actually worked them. The only configuration with contact with a real calendar"
        ),
    ),
    ScheduleConfigSpec(
        schedule_config_id="strict_priority__flat_median",
        strategy_id="strict_priority",
        capacity_mode=CapacityMode.FLAT_MEDIAN,
        is_scenario=True,
        is_default=False,
        rationale=(
            "the same queue against Component 13's own capacity assumption. Retained because "
            "the assumption is worth being able to see: it is feasible by construction, and "
            "the 784 inspections it fits that the real calendar does not are the measurement"
        ),
    ),
)

CONFIG_BY_ID: dict[str, ScheduleConfigSpec] = {
    spec.schedule_config_id: spec for spec in CONFIG_GRID
}

STRATEGY_BY_ID: dict[str, StrategySpec] = {spec.strategy_id: spec for spec in STRATEGY_GRID}


def config_for(schedule_config_id: str) -> ScheduleConfigSpec:
    """The frozen configuration with this id, or a message naming the ones that exist."""
    try:
        return CONFIG_BY_ID[schedule_config_id]
    except KeyError:
        known = ", ".join(sorted(CONFIG_BY_ID))
        raise SchedulingDefinitionError(
            f"unknown schedule configuration {schedule_config_id!r}. Known: {known}"
        ) from None


# --- 4. controlled vocabularies ------------------------------------------------


class ScheduleStatus(StrEnum):
    """What the **plan** says about a recommended row.

    Four values, and two conspicuous absences.

    ``recommended`` is not here: it is Component 13's ``is_selected``, it is carried on the row
    verbatim, and giving it a second home would give one fact two owners. ``completed`` is not
    here either: it is an execution fact, and a plan column that execution writes into is
    precisely the retroactive edit the temporal boundary exists to prevent.
    """

    #: Assigned to a slot on an operating day inside the horizon.
    SCHEDULED = "scheduled"

    #: Recommended and approved, but the horizon held no slot. Not an error, and never
    #: redefined as "not recommended" -- the row keeps its rank, its mechanism and its reason.
    BACKLOG = "backlog"

    #: Had a slot and was moved to a later operating day by an adjustment or a re-plan. Still
    #: in the plan; a deferred row that vanished would be the failure this status exists to
    #: make visible.
    DEFERRED = "deferred"

    #: Removed from the plan by a human adjustment or a field cancellation. The freed slot is
    #: not backfilled.
    CANCELLED = "cancelled"


class ScheduleReason(StrEnum):
    """Why the status. Deterministic codes, never generated prose.

    No language model writes any value in this enum. A reason code that varied between runs
    would make the audit trail unreproducible, and one a reader cannot enumerate is one nobody
    can check. Component 13 makes the same argument about its decision reasons.
    """

    #: Placed by strict priority: the queue in rank order, the horizon in date order.
    PLACED_IN_PRIORITY_ORDER = "placed_in_priority_order"

    #: The horizon ran out of slots before the queue ran out of rows.
    CAPACITY_EXHAUSTED_IN_HORIZON = "capacity_exhausted_in_horizon"

    #: A human adjustment moved this row to a later day.
    DEFERRED_BY_ADJUSTMENT = "deferred_by_adjustment"

    #: A human adjustment moved this row to an earlier day. Kept distinct from a deferral
    #: because the two are different requests -- one buys time and the other spends it -- and a
    #: log that recorded both as "moved" could not tell a supervisor what they had asked for.
    ADVANCED_BY_ADJUSTMENT = "advanced_by_adjustment"

    #: A human adjustment moved another row onto this row's slot. Capacity is fixed, so an
    #: inclusion costs a displacement, and the displaced row is named rather than absorbed.
    DISPLACED_BY_ADJUSTMENT = "displaced_by_adjustment"

    #: A re-planning run moved this row after an execution event freed or consumed capacity.
    RESCHEDULED_BY_REPLAN = "rescheduled_by_replan"

    #: A human adjustment struck this row from the plan.
    CANCELLED_BY_ADJUSTMENT = "cancelled_by_adjustment"

    #: The field reported the inspection cancelled. Not backfilled.
    CANCELLED_IN_FIELD = "cancelled_in_field"


#: The reason each status is allowed to carry. Checked by the validator, so a row that claims
#: to be scheduled while citing an exhausted horizon is an error rather than a curiosity.
STATUS_REASONS: dict[str, frozenset[str]] = {
    ScheduleStatus.SCHEDULED: frozenset(
        {
            ScheduleReason.PLACED_IN_PRIORITY_ORDER,
            ScheduleReason.RESCHEDULED_BY_REPLAN,
            ScheduleReason.ADVANCED_BY_ADJUSTMENT,
        }
    ),
    ScheduleStatus.BACKLOG: frozenset(
        {
            ScheduleReason.CAPACITY_EXHAUSTED_IN_HORIZON,
            ScheduleReason.DISPLACED_BY_ADJUSTMENT,
        }
    ),
    ScheduleStatus.DEFERRED: frozenset(
        {
            ScheduleReason.DEFERRED_BY_ADJUSTMENT,
            ScheduleReason.DISPLACED_BY_ADJUSTMENT,
            ScheduleReason.RESCHEDULED_BY_REPLAN,
        }
    ),
    ScheduleStatus.CANCELLED: frozenset(
        {
            ScheduleReason.CANCELLED_BY_ADJUSTMENT,
            ScheduleReason.CANCELLED_IN_FIELD,
        }
    ),
}

#: Statuses that occupy a slot on an operating day.
#:
#: ``DEFERRED`` is in here, and that is the whole point of having it as a separate status. A
#: deferred row has been *moved*, not removed: it still holds a slot, still counts against a
#: day's capacity, and still appears in the plan. What distinguishes it from a plainly scheduled
#: row is that somebody moved it and the row says who. A backlogged or cancelled row holds no
#: slot and must never be counted against capacity.
OCCUPYING_STATUSES: frozenset[str] = frozenset({ScheduleStatus.SCHEDULED, ScheduleStatus.DEFERRED})


class InversionReason(StrEnum):
    """Why a row sits out of ``final_policy_rank`` order. Never blank, never null.

    ``NONE`` is a token rather than a null for the reason Component 13's ``NO_WARNING`` is: an
    empty cell is ambiguous between "no inversion" and "inversions were not computed", and only
    one of those is a statement about the schedule.
    """

    NONE = "none"
    DEFERRED_BY_ADJUSTMENT = "deferred_by_adjustment"
    ADVANCED_BY_ADJUSTMENT = "advanced_by_adjustment"
    DISPLACED_BY_ADJUSTMENT = "displaced_by_adjustment"
    RESCHEDULED_BY_REPLAN = "rescheduled_by_replan"


#: The rule the validator enforces at error severity. Stated as a string so it travels in the
#: manifest beside the check that proves it.
INVERSION_REASON_REQUIRED = (
    "a row whose slot position disagrees with its final_policy_rank position must carry an "
    "inversion_reason other than none. There are no silent inversions and no flag that permits "
    "one; under strict priority with no external files, every row carries none and the "
    "validator requires it"
)


class AdjustmentAction(StrEnum):
    """What a human may do to a **schedule**. Three verbs, all auditable.

    Disjoint from Component 13's ``OverrideAction`` by construction and by import-time guard.
    That disjointness is the mechanical form of a distinction the component depends on: an
    override changes *who* is in the queue, an adjustment changes *when* an approved row is
    worked, and an execution event records *what happened*. There is no generic "override".
    """

    #: Move a scheduled row to a later operating day inside the horizon.
    DEFER_TO_DATE = "defer_to_date"

    #: Move a scheduled row to an earlier operating day inside the horizon.
    ADVANCE_TO_DATE = "advance_to_date"

    #: Strike a row from the plan. The freed slot is **not** backfilled: filling it would be
    #: the scheduler making a second decision on the back of a human one, and the supervisor
    #: who struck a row did not ask for a replacement. Component 13's ``force_exclude`` refuses
    #: to backfill for the same reason.
    CANCEL = "cancel"


#: Outcomes an adjustment can have, in one-to-one correspondence with Component 13's override
#: outcomes. An adjustment that changed nothing is logged as loudly as one that did, because
#: "the supervisor asked and it made no difference" is an audit fact.
ADJUSTMENT_OUTCOMES: tuple[str, ...] = (
    "applied",
    "no_op_already_on_date",
    "no_op_not_scheduled",
    "row_not_in_plan",
)


class ExecutionStatus(StrEnum):
    """What the field reports happened. External, and not reproducible computation."""

    #: The inspection was carried out. Needs no re-planning.
    COMPLETED = "completed"

    #: The inspection did not happen and the establishment still needs one. Returns to the
    #: queue at its original rank; this is the only status that triggers a re-plan.
    NOT_PERFORMED = "not_performed"

    #: The field cancelled the inspection. A removal, and removals are not backfilled.
    CANCELLED_IN_FIELD = "cancelled_in_field"


#: The derived status for a scheduled row no execution event mentions. Never supplied in a
#: file; it appears only in summary counts, so that "we do not know" is a visible category
#: rather than being silently folded into "not completed".
NO_EXECUTION_RECORD = "no_execution_record"

#: Outcomes an execution event can have.
EXECUTION_OUTCOMES: tuple[str, ...] = (
    "recorded",
    "no_op_not_scheduled",
    "row_not_in_plan",
)

#: The only status that returns a row to the queue.
REPLAN_TRIGGERING_STATUSES: frozenset[str] = frozenset({ExecutionStatus.NOT_PERFORMED})

#: Where a horizon day's slot count came from, as a value on the row.
CAPACITY_SOURCES: dict[str, str] = {
    CapacityMode.OBSERVED_CALENDAR: "observed_inspection_count",
    CapacityMode.FLAT_MEDIAN: "test_median_daily_capacity",
}

#: Why a planning run exists. Three reasons, and each appends a run rather than editing one:
#: the deterministic plan, a human moving an approved row, and the field reporting that a
#: planned day did not happen. An adjustment gets its own run for the same reason a re-plan
#: does -- the plan before it must stay readable beside the plan after it.
REPLAN_TRIGGERS: tuple[str, ...] = (
    "original_plan",
    "scheduling_adjustment",
    "execution_not_performed",
)


# --- 5. the three human layers, kept apart -------------------------------------

#: The distinction the component is built around, stated in the manifest because merging any
#: two of these is the most plausible way to lose the audit trail.
THREE_HUMAN_LAYERS = (
    "a recommendation override is Component 13's and changes who is in the approved queue; a "
    "scheduling adjustment is Component 14's and changes when an approved row is worked; an "
    "execution deviation records what a person reports actually happened. Three contracts, "
    "three id namespaces, three disjoint action vocabularies, three tables. They are never "
    "merged into one generic override, and the reconstruction original recommendation -> "
    "approved recommendation -> planned schedule -> scheduling change -> execution outcome is "
    "available end to end because of it"
)

#: Every field a scheduling adjustment must carry. Absent or blank refuses the whole file.
ADJUSTMENT_REQUIRED_FIELDS: tuple[str, ...] = (
    "adjustment_id",
    "schedule_config_id",
    "policy_id",
    "fold_id",
    "k_name",
    "target_inspection_id",
    "action",
    "target_date",
    "reason_code",
    "actor",
    "decided_at",
)

#: Every field an execution event must carry. Same all-or-nothing rule.
EXECUTION_REQUIRED_FIELDS: tuple[str, ...] = (
    "execution_id",
    "schedule_config_id",
    "policy_id",
    "fold_id",
    "k_name",
    "target_inspection_id",
    "scheduled_date",
    "execution_status",
    "reason_code",
    "actor",
    "observed_at",
)

#: What an adjustment may never do.
ADJUSTMENT_CANNOT = (
    "a scheduling adjustment changes when an approved row is worked, and nothing else. It "
    "never edits a score, a rank, a decision mechanism, a decision reason or the deterministic "
    "plan, all of which are written unchanged beside the adjustment log so the original "
    "assignment stays recoverable. It never moves a row outside the horizon, because that "
    "would be a capacity increase by another name, and it never displaces a coverage-reserve "
    "row, because that would quietly convert an adjustment into a coverage cut"
)

#: What an execution event may never do.
EXECUTION_CANNOT = (
    "an execution event records what happened and changes nothing about what was recommended "
    "or planned. It never edits a score, a rank, a mechanism, a reason, an original assignment "
    "or the plan at replan index 0. It may only affect assignments on operating days strictly "
    "after the day it was observed, which is the temporal boundary stated as an arithmetic "
    "rule rather than an intention"
)

#: What a re-plan preserves. The list a reader should be able to check against the artifact.
REPLAN_PRESERVES = (
    "a re-plan appends a planning run and never mutates one. Completed rows keep their slot "
    "and are never rescheduled; every row on an operating day before the re-plan point is "
    "frozen whatever its status; original_scheduled_date and original_schedule_rank are copied "
    "forward untouched; and placement is ordered by final_policy_rank at every index, because "
    "that is the only ordering key this component has"
)

#: Why a re-plan backfills where an override does not. The two look like the same operation and
#: are not, so the difference is written down rather than left to be inferred.
REPLAN_BACKFILL_RULE = (
    "a re-plan fills capacity freed by a not-performed inspection from the backlog in rank "
    "order, where Component 13's force_exclude deliberately does not backfill. The difference "
    "is what the freed capacity means: an excluded row is a decision that the slot should not "
    "be used, and re-filling it would overturn a human decision; a day that did not happen is "
    "capacity that still exists, and refusing to re-plan it would strand it. A cancellation -- "
    "by adjustment or in the field -- is a removal and is never backfilled, on the first rule"
)


# --- 6. advisory thresholds ----------------------------------------------------

#: One backlogged row is worth reporting. Not a tuned threshold: the question the advisory
#: answers is "did every approved inspection fit", and the answer is binary.
ADVISORY_BACKLOG_ROWS = 1

#: One idle slot is worth reporting, for the same reason in the other direction.
ADVISORY_IDLE_SLOTS = 1

#: One lost reserve slot is worth reporting. The component's headline advisory, and the
#: threshold is one because the coverage reserve is a stated allocation with a measured price.
ADVISORY_RESERVE_SLOTS_LOST = 1

#: A horizon whose opening day carries less than half the window's median rate. Measured, not
#: chosen: quarter-opening days are systematically thin -- 2024Q1 and 2026Q1 both open on a
#: single inspection against medians of 34 and 35 -- and every k_1_day number is dominated by
#: that day. Half the median is the point at which the opening day stops being representative
#: of the window the cutoff was derived from.
ADVISORY_LOW_VOLUME_OPENING_DAY = 0.5


# --- 7. determinism, the temporal boundary, and the boundary lists -------------

#: The scope of the reproducibility claim, stated precisely because overstating it would be the
#: easiest lie in this component.
DETERMINISM_SCOPE = (
    "the scheduling computation is deterministic: identical inputs produce byte-identical "
    "tables, and shuffling the input rows changes nothing. Adjustments and execution events "
    "are external human and operational inputs, so a run is byte-identical only given the "
    "identical files, and the manifest pins them by checksum rather than claiming that what "
    "inspectors did last Tuesday is reproducible computation"
)

#: The temporal boundary, as an arithmetic rule rather than an intention.
TEMPORAL_BOUNDARY = (
    "an execution event observed on day D may only affect assignments on operating days "
    "strictly after D. A historical plan is never rewritten: a re-plan produces a new planning "
    "run beside the old one, both are written, and the validator compares them row by row to "
    "prove that nothing before the re-plan point moved"
)

#: The claim this component makes about its own output, scoped hard.
SCHEDULE_CLAIM = (
    "an operating plan produced by one stated strategy from one approved queue under one "
    "measured calendar. Revisable. NOT a claim that this is the right schedule, NOT a claim "
    "that it is optimal, and NOT a forecast -- every day in it is a day that has already "
    "happened"
)

#: What a green run does not mean. Printed at the top of every validation report.
GREEN_RUN_MEANS = (
    "a green run means the plan was built correctly. It does not mean the city has enough "
    "capacity, it does not mean the schedule is the right one, and it does not mean the "
    "coverage reserve survived"
)

#: What this component does not establish. Every line is a claim somebody could reasonably
#: think the artifact supports, and does not.
DOES_NOT_ESTABLISH: tuple[str, ...] = (
    "that this is the correct schedule. It is the schedule one stated strategy produces from "
    "one approved queue under one measured calendar",
    "that the schedule is optimal. It is deterministic greedy allocation down an approved rank "
    "order; no objective function is defined anywhere in this component and none is solved",
    "that 784 inspections were missed, or that any establishment went uninspected. Every "
    "number here re-orders inspections that already happened, and a backlog is what a stated "
    "capacity rule would not have fitted",
    "that the observed calendar is knowable in advance. It is measured from the window it "
    "schedules, so it states what capacity existed rather than what a planner could have known "
    "on day one. A live deployment would need a forecast this project has not built",
    "that the lost coverage reserve is Component 13's error. It is what two individually "
    "correct layers do when composed, and whether the reserve belongs at the head of the queue "
    "is a policy question this component is not entitled to answer",
    "that any establishment can be reached in the time the schedule allows. There is no travel "
    "time in this dataset, so a day's slots are a workload count and never a route",
    "any claim about inspectors. The dataset names none, and nothing here infers one",
)

#: What this component refuses to do, and where the refusal comes from.
BLOCKED: tuple[str, ...] = (
    "re-ranking the queue by risk, geography, mechanism or anything else. Component 13 owns "
    "the ordering (HANDOFF 16g)",
    "adjusting a score. No component after 9 writes one (ADR 0037)",
    "raising capacity, extending a horizon or adding an operating day. Every cutoff descends "
    "from the window's own measured median daily rate",
    "introducing a probability threshold. Refused by Component 12 in prose and by Component "
    "13's CAPACITY_SEMANTICS; there is no flag to add one",
    "treating the coverage reserve as slack. It is an allocation with a measured price, and a "
    "scheduler that raided it to fit a route would silently be changing the policy",
    "geographic route optimisation, inspector assignment or travel-time estimation. The "
    "dataset has no inspector (ADR 0019) and routing is Component 15",
    "joining anything in data/processed/scheduling/ onto a feature table. It holds the "
    "system's own past decisions one layer further out than Component 13 already refused to "
    "close",
    "letting an execution outcome edit a recommendation, a rank or a plan that was already written",
)

#: Limitations inherited whole from upstream components. Restated rather than referenced,
#: because a manifest that pointed at another manifest would not travel.
INHERITED_LIMITATIONS: tuple[str, ...] = (
    "Component 5: this is a re-ordering study over inspections that actually happened. A "
    "schedule here is a counterfactual arrangement of real historical inspections, not a "
    "forecast of future ones",
    "Component 13: the queue is the queue one stated policy produces from one selected model "
    "under one capacity assumption, and no policy winner was declared",
    "Component 13: the production model is an operating choice from a rule whose tie band "
    "decides the answer, and two defensible bands pick two different models",
    "Component 3: the target is that a Priority violation was cited, not that an establishment "
    "was unsafe",
    "Component 12: geographic differences are confounded with inspection practice by "
    "construction, because Chicago assigns inspectors by district and the dataset names none",
)


def _guard_registry() -> None:
    """Check the frozen constants against each other, at import time.

    Every one of these has a way of drifting during an edit, and every one of them would fail
    silently and plausibly. A configuration grid with two defaults, a status with no reason, a
    reason no status accepts, an adjustment verb that collides with an override verb -- each
    produces a run that finishes green and schedules the wrong establishments, or an audit
    trail in which two different human decisions are indistinguishable.
    """
    ids = [spec.schedule_config_id for spec in CONFIG_GRID]
    if len(set(ids)) != len(ids):
        raise SchedulingDefinitionError("duplicate schedule_config_id in CONFIG_GRID")

    defaults = [spec for spec in CONFIG_GRID if spec.is_default]
    if len(defaults) != 1:
        raise SchedulingDefinitionError(
            f"exactly one configuration must be the default, found {len(defaults)}"
        )
    if defaults[0].capacity_mode is not DEFAULT_CAPACITY_MODE:
        raise SchedulingDefinitionError(
            f"the default configuration uses {defaults[0].capacity_mode}, but "
            f"DEFAULT_CAPACITY_MODE is {DEFAULT_CAPACITY_MODE}"
        )
    if defaults[0].is_scenario:
        raise SchedulingDefinitionError(
            "the default configuration is a scenario. A scenario is what the capacity cutoffs "
            "assume; defaulting to it would make every headline number describe an assumption"
        )

    for spec in CONFIG_GRID:
        if spec.strategy_id not in STRATEGY_BY_ID:
            raise SchedulingDefinitionError(
                f"{spec.schedule_config_id}: unknown strategy {spec.strategy_id!r}"
            )
        if spec.is_scenario != (spec.capacity_mode is CapacityMode.FLAT_MEDIAN):
            raise SchedulingDefinitionError(
                f"{spec.schedule_config_id}: is_scenario disagrees with the capacity mode. The "
                "flat median is the assumption and the observed calendar is the measurement"
            )
        if not spec.rationale:
            raise SchedulingDefinitionError(
                f"{spec.schedule_config_id}: every configuration states why it exists"
            )

    modes = {spec.capacity_mode for spec in CONFIG_GRID}
    if modes != set(CapacityMode):
        missing = ", ".join(sorted(set(CapacityMode) - modes))
        raise SchedulingDefinitionError(
            f"no configuration exercises {missing}. A capacity mode with no configuration is a "
            "code path no run ever takes, which is indistinguishable from one that is broken"
        )

    for strategy in STRATEGY_GRID:
        if not strategy.rationale:
            raise SchedulingDefinitionError(f"{strategy.strategy_id}: every strategy states why")
    if not any(spec.preserves_priority_exactly for spec in STRATEGY_GRID):
        raise SchedulingDefinitionError(
            "no strategy preserves priority exactly. The reference schedule is the one thing "
            "this component must always be able to produce"
        )

    for status, reasons in STATUS_REASONS.items():
        if status not in set(ScheduleStatus):
            raise SchedulingDefinitionError(f"STATUS_REASONS names unknown status {status!r}")
        if not reasons:
            raise SchedulingDefinitionError(
                f"{status}: a status with no reason cannot be written to a row"
            )
    missing_statuses = set(ScheduleStatus) - set(STATUS_REASONS)
    if missing_statuses:
        raise SchedulingDefinitionError(
            f"no reason is declared for {', '.join(sorted(missing_statuses))}"
        )
    declared = {reason for reasons in STATUS_REASONS.values() for reason in reasons}
    orphaned = set(ScheduleReason) - declared
    if orphaned:
        raise SchedulingDefinitionError(
            f"reason(s) {', '.join(sorted(orphaned))} belong to no status. A reason no status "
            "accepts is a reason no run can emit, which is indistinguishable from a broken one"
        )

    if InversionReason.NONE not in set(InversionReason):
        raise SchedulingDefinitionError("InversionReason must carry an explicit NONE token")
    if not set(ScheduleStatus) >= OCCUPYING_STATUSES:
        raise SchedulingDefinitionError("OCCUPYING_STATUSES names a status that does not exist")

    overlap = {str(a) for a in AdjustmentAction} & {str(o) for o in OverrideAction}
    if overlap:
        raise SchedulingDefinitionError(
            f"adjustment and override verbs collide on {', '.join(sorted(overlap))}. A "
            "recommendation override and a scheduling adjustment must never be confusable: "
            "one changes who is in the queue and the other changes when they are worked"
        )
    if ADJUSTMENT_REQUIRED_FIELDS[0] == EXECUTION_REQUIRED_FIELDS[0]:
        raise SchedulingDefinitionError(
            "the two external contracts share an id field name, so a log row could not say "
            "which contract it came from"
        )
    for fields, name in (
        (ADJUSTMENT_REQUIRED_FIELDS, "adjustment"),
        (EXECUTION_REQUIRED_FIELDS, "execution"),
    ):
        if len(set(fields)) != len(fields):
            raise SchedulingDefinitionError(f"the {name} contract repeats a field")
        for required in ("actor", "reason_code"):
            if required not in fields:
                raise SchedulingDefinitionError(
                    f"the {name} contract has no {required!r} field. An external change with "
                    "no attribution is an anonymous change to who gets inspected when"
                )

    if set(CAPACITY_SOURCES) != set(CapacityMode):
        raise SchedulingDefinitionError("every capacity mode must declare its capacity source")
    if not set(ExecutionStatus) >= REPLAN_TRIGGERING_STATUSES:
        raise SchedulingDefinitionError("REPLAN_TRIGGERING_STATUSES names an unknown status")
    if NO_EXECUTION_RECORD in set(ExecutionStatus):
        raise SchedulingDefinitionError(
            "no_execution_record is a derived summary category and must not be a status a "
            "person can supply in a file"
        )

    if PRIMARY_K_LEVEL not in K_LEVELS:
        raise SchedulingDefinitionError(
            f"the primary capacity level {PRIMARY_K_LEVEL!r} is not in the inherited grid"
        )
    if str(DecisionMechanism.COVERAGE_RESERVE) not in {str(m) for m in DecisionMechanism}:
        raise SchedulingDefinitionError("the inherited mechanism vocabulary has moved")

    for name, value in (
        ("DOES_NOT_ESTABLISH", DOES_NOT_ESTABLISH),
        ("BLOCKED", BLOCKED),
        ("INHERITED_LIMITATIONS", INHERITED_LIMITATIONS),
    ):
        if not value:
            raise SchedulingDefinitionError(
                f"{name} is empty. It travels in every manifest so the boundary arrives with "
                "the artifact; an empty list silently drops it"
            )

    if not 0.0 < ADVISORY_LOW_VOLUME_OPENING_DAY < 1.0:
        raise SchedulingDefinitionError(
            "the low-volume opening-day threshold is a fraction of the window's median rate"
        )
    for name, threshold in (
        ("ADVISORY_BACKLOG_ROWS", ADVISORY_BACKLOG_ROWS),
        ("ADVISORY_IDLE_SLOTS", ADVISORY_IDLE_SLOTS),
        ("ADVISORY_RESERVE_SLOTS_LOST", ADVISORY_RESERVE_SLOTS_LOST),
    ):
        if threshold < 1:
            raise SchedulingDefinitionError(
                f"{name} must be at least 1; a threshold of zero fires on every run and "
                "carries no information"
            )


_guard_registry()


__all__ = [
    "ADJUSTMENT_CANNOT",
    "ADJUSTMENT_OUTCOMES",
    "ADJUSTMENT_REQUIRED_FIELDS",
    "ADVISORY_BACKLOG_ROWS",
    "ADVISORY_IDLE_SLOTS",
    "ADVISORY_LOW_VOLUME_OPENING_DAY",
    "ADVISORY_RESERVE_SLOTS_LOST",
    "ALLOCATION_CLAIM",
    "BLOCKED",
    "CALENDAR_IS_OBSERVED",
    "CAPACITY_IS_INHERITED",
    "CAPACITY_MODE_SCENARIO_CLAIM",
    "CAPACITY_SOURCES",
    "CONFIG_BY_ID",
    "CONFIG_GRID",
    "DEFAULT_CAPACITY_MODE",
    "DETERMINISM_SCOPE",
    "DOES_NOT_ESTABLISH",
    "EXECUTION_CANNOT",
    "EXECUTION_OUTCOMES",
    "EXECUTION_REQUIRED_FIELDS",
    "GREEN_RUN_MEANS",
    "HORIZON_RULE",
    "INHERITED_LIMITATIONS",
    "INVERSION_REASON_REQUIRED",
    "K_LEVELS",
    "LAYER_SEPARATION",
    "NO_CONSTRAINT_AWARE_STRATEGY",
    "NO_EXECUTION_RECORD",
    "NO_SOLVER",
    "OCCUPYING_STATUSES",
    "PRIMARY_K_LEVEL",
    "PRIORITY_IS_INHERITED",
    "REPLAN_BACKFILL_RULE",
    "REPLAN_PRESERVES",
    "REPLAN_TRIGGERING_STATUSES",
    "REPLAN_TRIGGERS",
    "SCHEDULE_CLAIM",
    "SCHEDULE_DEFINITION_VERSION",
    "SCHEDULING_SEMANTICS",
    "STATUS_REASONS",
    "STRATEGY_BY_ID",
    "STRATEGY_GRID",
    "TEMPORAL_BOUNDARY",
    "THREE_HUMAN_LAYERS",
    "AdjustmentAction",
    "CapacityMode",
    "ExecutionStatus",
    "InversionReason",
    "ScheduleConfigSpec",
    "ScheduleReason",
    "ScheduleStatus",
    "SchedulingDefinitionError",
    "StrategySpec",
    "config_for",
    "horizon_days",
]
