"""Typed structures for Component 14. No behaviour, no I/O, no clock.

Three of these carry most of the weight.

``Horizon`` is the calendar a cell is scheduled against: the operating days, in order, each
with the number of slots it holds and where that number came from. It is built once per (fold,
capacity level, capacity mode) and is policy-independent, because the calendar is a fact about
the fold rather than about a policy.

``QueueRow`` is one approved recommendation, carried verbatim. Every field on it is Component
13's and none is ever recomputed. It is a separate type from the placement precisely so that
the compiler enforces what the prose asks for: a function that places rows cannot reach a
score, because the placement type does not carry one.

``Placement`` is where one row landed, and it is deliberately verbose. It keeps the current
assignment beside the *original* one and beside the reason the two differ. Those three facts
are what distinguish "this was always Thursday" from "this moved to Thursday because a
supervisor asked" from "this moved to Thursday because Tuesday did not happen", and a table
that recorded only the final date could not tell them apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, Field

from sentinel.scheduling.definitions import (
    OCCUPYING_STATUSES,
    InversionReason,
    ScheduleStatus,
)

#: A defect in the scheduling computation. Fails the run.
SEVERITY_ERROR = "error"

#: A finding about what the schedule cost. Recorded, printed, exit zero. ADR 0034's line, held
#: one layer further out.
SEVERITY_WARN = "warn"

#: Offenders listed on a failing check before the list is truncated. Matching Components 9, 11,
#: 12 and 13: a report that prints ten thousand ids is a report nobody reads.
MAX_OFFENDERS = 20


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One assertion about the scheduling run or the artifacts holding it.

    ``severity`` decides whether a failure stops the run, and the line is ADR 0034's, inherited
    unchanged through two components now: a defect in the *computation* is an error, and a
    finding about what the schedule *costs* is an advisory.

    The sharpest case is this component's headline. A schedule that loses coverage-reserve
    slots to a short horizon must never turn a build red, because the cheapest way to make such
    a build green is to have the scheduler prefer reserve rows -- which is re-ranking, which is
    forbidden, and which would move a coverage decision into a layer that does not own it.
    """

    name: str
    passed: bool
    severity: str
    detail: str
    offenders: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OperatingDay:
    """One day of the planning horizon, and the capacity it holds.

    ``capacity_source`` is on the row rather than inferred from the mode, so a reader looking
    at a single day can tell whether its slot count is something Chicago did or something a
    scenario assumed.
    """

    day_index: int
    slot_date: date
    n_slots: int
    capacity_source: str

    def __post_init__(self) -> None:
        if self.day_index < 1:
            raise ValueError(f"day_index is 1-based, got {self.day_index}")
        if self.n_slots < 1:
            raise ValueError(
                f"{self.slot_date}: an operating day holds at least one slot. A zero-slot day "
                "is not an operating day and must not be in the horizon"
            )


@dataclass(frozen=True, slots=True)
class Horizon:
    """The planning horizon for one (fold, capacity level, capacity mode).

    Policy-independent by construction, and that is load-bearing: the calendar and its volumes
    are facts about the fold, so deriving them per policy would produce seven copies of one
    measurement and invite the copies to disagree. Component 13 makes the same argument for its
    model-independent eligibility table.
    """

    fold_set: str
    fold_id: str
    k_name: str
    k: int
    median_daily_capacity: int
    capacity_mode: str
    days: tuple[OperatingDay, ...]
    was_clamped: bool = False

    def __post_init__(self) -> None:
        if not self.days:
            raise ValueError(f"{self.fold_id}/{self.k_name}: a horizon holds at least one day")
        indices = [day.day_index for day in self.days]
        if indices != list(range(1, len(self.days) + 1)):
            raise ValueError(
                f"{self.fold_id}/{self.k_name}: day_index must be contiguous from 1, got "
                f"{indices[:5]}"
            )
        dates = [day.slot_date for day in self.days]
        if dates != sorted(set(dates)):
            raise ValueError(
                f"{self.fold_id}/{self.k_name}: operating days must be strictly increasing. A "
                "repeated or out-of-order date would make slot assignment ambiguous"
            )

    @property
    def n_days(self) -> int:
        return len(self.days)

    @property
    def total_slots(self) -> int:
        """Every slot the horizon holds. The number the queue is measured against."""
        return sum(day.n_slots for day in self.days)

    @property
    def cumulative_slots(self) -> tuple[int, ...]:
        """Running slot total by day, for locating a rank without walking the days twice."""
        running = 0
        out: list[int] = []
        for day in self.days:
            running += day.n_slots
            out.append(running)
        return tuple(out)

    @property
    def start_date(self) -> date:
        return self.days[0].slot_date

    @property
    def end_date(self) -> date:
        return self.days[-1].slot_date

    def day_for_position(self, position: int) -> OperatingDay | None:
        """The day holding the ``position``-th slot, 1-based, or None past the horizon."""
        if position < 1:
            raise ValueError(f"slot positions are 1-based, got {position}")
        for day, cumulative in zip(self.days, self.cumulative_slots, strict=True):
            if position <= cumulative:
                return day
        return None


@dataclass(frozen=True, slots=True)
class QueueRow:
    """One approved recommendation, carried verbatim from Component 13.

    Every field here is Component 13's and none is recomputed. ``recommendation_date`` is the
    as-of date the row was scored on -- it is emphatically **not** a schedule date, and the two
    are kept under different names because conflating them is the single most plausible way for
    this component to produce a wrong artifact that looks right.
    """

    target_inspection_id: str
    establishment_id: str
    final_policy_rank: int
    model_rank: int
    decision_mechanism: str
    decision_reason: str
    score: float
    base_score: float
    coverage_eligible: bool
    warnings: str
    recommendation_date: date

    def __post_init__(self) -> None:
        if self.final_policy_rank < 1:
            raise ValueError(
                f"{self.target_inspection_id}: final_policy_rank is 1-based, got "
                f"{self.final_policy_rank}"
            )


@dataclass(frozen=True, slots=True)
class Placement:
    """Where one approved row landed, and why.

    ``original_slot_date`` and ``original_schedule_rank`` are **write-once**. They are set when
    the row is first placed, at replan index 0, and every later run copies them forward
    untouched. That is what makes the question "where was this originally going to be?"
    answerable after any number of adjustments and re-plans, and the validator proves it rather
    than trusting it.
    """

    target_inspection_id: str
    status: str
    reason: str
    inversion_reason: str = InversionReason.NONE
    slot_date: date | None = None
    day_index: int | None = None
    slot_index: int | None = None
    schedule_rank: int | None = None
    original_slot_date: date | None = None
    original_schedule_rank: int | None = None
    adjustment_id: str = ""
    replan_index: int = 0

    def __post_init__(self) -> None:
        scheduled = self.status in OCCUPYING_STATUSES
        placed = self.slot_date is not None
        if scheduled != placed:
            raise ValueError(
                f"{self.target_inspection_id}: status {self.status!r} and slot_date "
                f"{self.slot_date!r} disagree about whether this row occupies a slot"
            )
        if placed and (self.slot_index is None or self.day_index is None):
            raise ValueError(
                f"{self.target_inspection_id}: a placed row carries a day and a slot index"
            )

    @property
    def occupies_a_slot(self) -> bool:
        """True for a row holding a slot on an operating day.

        Deferred rows count. A deferral moves an inspection to a later day; it does not remove
        it, so it still consumes capacity there, and a capacity check that ignored deferrals
        would let a day be overbooked by exactly the rows somebody moved onto it.
        """
        return self.status in OCCUPYING_STATUSES


@dataclass(frozen=True, slots=True)
class SchedulePlan:
    """One cell's plan: the horizon, and where every approved row landed in it.

    A plan is immutable and a re-plan produces a new one beside it. Nothing in this component
    edits a plan in place, which is the structural form of the promise that history is not
    rewritten.
    """

    schedule_config_id: str
    policy_id: str
    model_name: str
    fold_set: str
    fold_id: str
    k_name: str
    k: int
    horizon: Horizon
    placements: tuple[Placement, ...]
    planning_run_id: str
    replan_index: int = 0

    @property
    def n_scheduled(self) -> int:
        """Rows holding a slot, deferred ones included: they are still in the plan."""
        return sum(1 for p in self.placements if p.occupies_a_slot)

    @property
    def n_backlog(self) -> int:
        return sum(1 for p in self.placements if p.status == ScheduleStatus.BACKLOG)

    @property
    def n_deferred(self) -> int:
        return sum(1 for p in self.placements if p.status == ScheduleStatus.DEFERRED)

    @property
    def n_cancelled(self) -> int:
        return sum(1 for p in self.placements if p.status == ScheduleStatus.CANCELLED)

    @property
    def idle_slots(self) -> int:
        """Slots the horizon held and the queue did not fill.

        Non-zero whenever the calendar was more generous than the cutoff, which happens in 44
        of 90 cells -- the mirror image of the backlog, and reported beside it so a reader can
        see that a short horizon and a thin queue are different problems.
        """
        return max(0, self.horizon.total_slots - self.n_scheduled)

    @property
    def capacity_utilization(self) -> float:
        total = self.horizon.total_slots
        return self.n_scheduled / total if total else 0.0

    def by_id(self) -> dict[str, Placement]:
        return {p.target_inspection_id: p for p in self.placements}


@dataclass(frozen=True, slots=True)
class CellKey:
    """The identity of one scheduled cell. Hashable, so it can key a dict of plans."""

    schedule_config_id: str
    policy_id: str
    model_name: str
    fold_set: str
    fold_id: str
    k_name: str


class Adjustment(BaseModel):
    """One human decision about *when* an approved row is worked.

    A pydantic model rather than a dataclass because this is one of only two inputs to the
    component that a human types. Everything else on the way in is an artifact another
    component wrote and checksummed; this arrives as a JSON file somebody edited, so it is
    validated at the boundary and rejected with a message rather than trusted and half-applied.

    ``decided_at`` is the supervisor's timestamp, not the run's. A run that stamped its own
    clock onto a human decision would be recording when the file was processed and labelling it
    when the decision was made.
    """

    adjustment_id: str
    schedule_config_id: str
    policy_id: str
    fold_id: str
    k_name: str
    target_inspection_id: str
    action: str
    target_date: str
    reason_code: str
    actor: str
    decided_at: str

    model_config = {"extra": "forbid"}


class ExecutionEvent(BaseModel):
    """One report of what actually happened to a planned inspection.

    Separate from ``Adjustment`` in every respect that matters -- separate id namespace,
    separate verb vocabulary, separate table -- because a plan change and a report of reality
    are different kinds of fact. An adjustment is a decision somebody made about the future; an
    execution event is an observation about the past, and it can never edit the plan it
    describes.

    ``observed_at`` is the field's timestamp, and it is what the temporal boundary is measured
    against: an event observed on day D may only affect operating days strictly after D.
    """

    execution_id: str
    schedule_config_id: str
    policy_id: str
    fold_id: str
    k_name: str
    target_inspection_id: str
    scheduled_date: str
    execution_status: str
    reason_code: str
    actor: str
    observed_at: str

    model_config = {"extra": "forbid"}


@dataclass(frozen=True, slots=True)
class AdjustmentOutcome:
    """What applying one adjustment did, including when it did nothing."""

    adjustment: Adjustment
    outcome: str
    original_status: str = ""
    original_slot_date: date | None = None
    original_schedule_rank: int | None = None
    final_status: str = ""
    final_slot_date: date | None = None
    displaced_target_inspection_id: str = ""
    displaced_landed_status: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """What recording one execution event did, including when it did nothing."""

    event: ExecutionEvent
    outcome: str
    plan_slot_date: date | None = None
    triggers_replan: bool = False
    applied_at_replan_index: int = 0


@dataclass
class ScheduleStats:
    """Counts accumulated during a run, surfaced in the manifest and the CLI summary."""

    recommendation_rows: int = 0
    queue_rows: int = 0
    cells: int = 0
    configs: int = 0
    folds: int = 0
    fold_sets: list[str] = field(default_factory=list)
    policies: int = 0
    model_name: str = ""
    scheduled_rows: int = 0
    backlog_rows: int = 0
    idle_slots: int = 0
    cells_with_backlog: int = 0
    cells_with_idle: int = 0
    reserve_recommended: int = 0
    reserve_scheduled: int = 0
    reserve_slots_lost: int = 0
    inversions: int = 0
    adjustments_applied: int = 0
    execution_events: int = 0
    replanning_runs: int = 1
    advisories: int = 0
    inputs_unchanged: bool = True
    seconds: float = 0.0


class ArtifactRecord(BaseModel):
    """Provenance for one written file."""

    path: str
    bytes: int
    sha256: str
    row_count: int
    schema_: dict[str, str] = Field(alias="schema")

    model_config = {"populate_by_name": True}


class ScheduleManifest(BaseModel):
    """Self-contained provenance and QA record for one scheduling run.

    The question this manifest must answer without reading any source code is *"when was each
    approved inspection planned for, under what calendar, and what did that calendar cost the
    policy?"* -- so the horizon rule, the two capacity modes with the scenario labelled, the
    reserve-survival numbers, the three human layers and the whole boundary travel inside it
    rather than only in documentation.

    ``reserve_slots_lost`` is here for one reason: it is the component's headline, it is a
    finding about an upstream component measured from inside this one, and a manifest that
    recorded only the schedule would let a reader conclude the coverage policy survived
    contact with the calendar. It did not.
    """

    component: str = "operational_scheduling"
    code_version: str
    schedule_definition_version: str
    built_at: str

    recommendations_path: str
    recommendations_sha256: str
    policy_definition_version: str
    selection_allocation_path: str
    selection_allocation_sha256: str
    policy_comparison_path: str
    policy_comparison_sha256: str
    evaluation_folds_path: str
    evaluation_folds_sha256: str
    evaluation_definition_version: str
    override_log_path: str | None = None
    override_log_sha256: str | None = None
    adjustments_path: str | None = None
    adjustments_sha256: str | None = None
    execution_path: str | None = None
    execution_sha256: str | None = None

    #: Every input checksum re-read after the last table was written. This component is a pure
    #: observer of thirteen closed components; if any input moved, it was not.
    inputs_unchanged: bool
    input_sha256_after: dict[str, str]

    layer_separation: str
    scheduling_semantics: str
    capacity_is_inherited: str
    priority_is_inherited: str
    calendar_is_observed: str
    horizon_rule: str
    allocation_claim: str
    no_solver: str
    no_constraint_aware_strategy: str
    capacity_mode_scenario_claim: str
    default_capacity_mode: str
    capacity_modes: list[str]
    strategies: list[str]
    config_grid: list[dict[str, object]]
    k_levels: list[str]
    primary_k_level: str
    model_name: str
    policies: list[str]

    three_human_layers: str
    adjustment_cannot: str
    execution_cannot: str
    replan_preserves: str
    replan_backfill_rule: str
    temporal_boundary: str
    adjustments_applied: int
    execution_events: int
    replanning_runs: int

    schedule_claim: str
    green_run_means: str
    determinism_scope: str
    inversion_reason_required: str

    scheduled_rows: int
    backlog_rows: int
    idle_slots: int
    cells_with_backlog: int
    cells_with_idle: int
    total_cells: int
    #: Measured on the observed calendar only. The flat_median scenario supplies exactly k
    #: slots for a queue of k, so it loses no reserve by construction, and pooling the two
    #: would divide the finding by however many modes happened to run.
    reserve_recommended: int
    reserve_scheduled: int
    reserve_slots_lost: int
    cells_losing_reserve: int
    cells_losing_all_reserve: int
    inversions: int

    does_not_establish: list[str]
    blocked: list[str]
    inherited_limitations: list[str]

    checks: list[dict[str, object]]
    advisories: list[str]
    artifacts: list[ArtifactRecord]
    row_counts: dict[str, int]
    seconds: float


__all__ = [
    "MAX_OFFENDERS",
    "SEVERITY_ERROR",
    "SEVERITY_WARN",
    "Adjustment",
    "AdjustmentOutcome",
    "ArtifactRecord",
    "CellKey",
    "ExecutionEvent",
    "ExecutionOutcome",
    "Horizon",
    "OperatingDay",
    "Placement",
    "QueueRow",
    "ScheduleManifest",
    "SchedulePlan",
    "ScheduleStats",
    "ValidationCheck",
]
