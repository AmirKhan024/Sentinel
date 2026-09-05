"""Typed structures for Component 13. No behaviour, no I/O, no clock.

Two of these carry most of the weight.

``PolicyWindow`` is the unit every allocation runs over: one fold's test window, already in
Component 5's canonical order, carrying the score, the label, the eligibility flag and the
dates as parallel tuples. Tuples rather than a frame, because an allocation is index
arithmetic and a frame invites a reorder halfway through it -- which is exactly the defect
Component 12 found in its own equal-mass binning.

``Allocation`` is the record of one policy's decision at one capacity, and it is deliberately
verbose. It keeps the reserve *target* beside the reserve *granted*, and the eligible count
already inside the risk cutoff beside both. Those three numbers are what distinguish "the
floor was satisfied" from "the floor was ignored" from "there were not enough eligible
establishments to satisfy it", and a table that recorded only the granted count could not tell
them apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, Field

#: A defect in the policy computation. Fails the run.
SEVERITY_ERROR = "error"

#: A finding about what the policy did. Recorded, printed, exit zero. ADR 0034's line, held.
SEVERITY_WARN = "warn"

#: Offenders listed on a failing check before the list is truncated. Matching Components 9,
#: 11 and 12: a report that prints ten thousand ids is a report nobody reads.
MAX_OFFENDERS = 20


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One assertion about the policy run or the artifact holding it.

    ``severity`` decides whether a failure stops the run. The line is the one ADR 0034 drew
    for Component 12 and this component inherits unchanged: a defect in the *computation* is
    an error, and a finding about what the policy *costs* is an advisory. A coverage reserve
    that gives up citations must never turn a build red, because the cheapest way to make such
    a build green is to delete the reserve -- which is a policy decision, and not one a CI
    runner is entitled to take.
    """

    name: str
    passed: bool
    severity: str
    detail: str
    offenders: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PolicyWindow:
    """One fold's scored test window, in canonical order, with everything a policy needs.

    Canonical order is ``(inspection_date, target_inspection_id)`` ascending -- Component 5's,
    reused rather than redefined, so an index here means the same row it means there.

    ``eligible`` and ``secondary_no_history`` are precomputed booleans rather than columns to
    be filtered later. Deciding eligibility once, at the edge, is what makes it checkable: the
    validator can re-derive the whole tuple from the feature table and compare, which it could
    not do if eligibility were a predicate scattered through the allocator.
    """

    fold_set: str
    fold_id: str
    ids: tuple[str, ...]
    scores: tuple[float, ...]
    base_scores: tuple[float, ...]
    labels: tuple[int, ...]
    dates: tuple[date, ...]
    eligible: tuple[bool, ...]
    secondary_no_history: tuple[bool, ...]
    median_daily_capacity: int

    def __post_init__(self) -> None:
        lengths = {
            len(self.ids),
            len(self.scores),
            len(self.base_scores),
            len(self.labels),
            len(self.dates),
            len(self.eligible),
            len(self.secondary_no_history),
        }
        if len(lengths) != 1:
            raise ValueError(f"{self.fold_id}: window columns differ in length: {sorted(lengths)}")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f"{self.fold_id}: target_inspection_id is not unique in the window")

    @property
    def n(self) -> int:
        """Rows in the window: the total capacity any schedule must fit inside."""
        return len(self.ids)

    @property
    def positives(self) -> int:
        return sum(self.labels)

    @property
    def n_eligible(self) -> int:
        return sum(self.eligible)


@dataclass(frozen=True, slots=True)
class Allocation:
    """One policy's decision for one window at one capacity.

    ``risk_indices`` and ``reserve_indices`` index into the window and are disjoint by
    construction -- the reserve is filled from rows the risk block did not take. The validator
    checks the disjointness anyway, because "by construction" is a claim about code that was
    correct when it was written.
    """

    policy_id: str
    fold_set: str
    fold_id: str
    k_name: str
    k: int
    n_universe: int
    reserve_target: int
    n_eligible_available: int
    n_eligible_in_risk_top_k: int
    risk_indices: tuple[int, ...]
    reserve_indices: tuple[int, ...]

    @property
    def n_risk(self) -> int:
        return len(self.risk_indices)

    @property
    def n_reserve(self) -> int:
        return len(self.reserve_indices)

    @property
    def n_selected(self) -> int:
        return self.n_risk + self.n_reserve

    @property
    def reserve_inert(self) -> bool:
        """The policy asked for a reserve and the allocation granted none.

        A real and expected outcome, not a defect: a floor is inert whenever the risk ranking
        already clears it, and a small share at a small cutoff floors to zero slots. Reported
        as an advisory so an inert mechanism is visible rather than merely absent.
        """
        return self.reserve_target > 0 and self.n_reserve == 0


@dataclass(frozen=True, slots=True)
class SelectionAxis:
    """One model's value on one axis of the selection rule, with how it was read."""

    model_name: str
    axis: str
    value: float | None
    direction: str
    band_low: float | None
    band_high: float | None


class Override(BaseModel):
    """One human decision about one recommendation.

    A pydantic model rather than a dataclass because this is the only input to the component
    that a human types. Everything else on the way in is an artifact another component wrote
    and checksummed; this arrives as a JSON file somebody edited, so it is validated at the
    boundary and rejected with a message rather than trusted and half-applied.

    ``decided_at`` is the reviewer's timestamp, not the run's. A run that stamped its own
    clock onto a human decision would be recording when the file was processed and labelling
    it when the decision was made.
    """

    override_id: str
    policy_id: str
    fold_id: str
    k_name: str
    target_inspection_id: str
    action: str
    reason_code: str
    actor: str
    decided_at: str

    model_config = {"extra": "forbid"}


@dataclass
class PolicyStats:
    """Counts accumulated during a run, surfaced in the manifest and the CLI summary."""

    prediction_rows: int = 0
    universe_rows: int = 0
    folds: int = 0
    fold_sets: list[str] = field(default_factory=list)
    policies: int = 0
    selected_model: str = ""
    selection_axis: str = ""
    eligible_rows: int = 0
    queue_rows: int = 0
    reserve_rows: int = 0
    inert_cells: int = 0
    overrides_applied: int = 0
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


class PolicyManifest(BaseModel):
    """Self-contained provenance and QA record for one policy run.

    The question this manifest must answer without reading any source code is *"who did
    Sentinel recommend inspecting, under what rule, using whose model, and what did that rule
    cost?"* -- so the frozen grid, the selection rule, the selected model, the winner (or the
    absence of one) and the boundary all travel inside it rather than only in documentation.

    ``selected_model_under_discarded_band`` is here for one reason: the tie rule decides which
    model is deployed, the rule was fixed after its inputs were first read, and a manifest that
    recorded only the outcome would hide that a different defensible rule gives a different
    answer. See ADR 0039.
    """

    component: str = "decision_policy"
    code_version: str
    policy_definition_version: str
    built_at: str

    features_path: str
    features_sha256: str
    feature_definition_version: str
    calibrated_predictions_path: str
    calibrated_predictions_sha256: str
    calibration_definition_version: str
    evaluation_folds_path: str
    evaluation_folds_sha256: str
    evaluation_definition_version: str
    simulation_summary_path: str
    simulation_summary_sha256: str
    evaluation_metrics_path: str
    evaluation_metrics_sha256: str
    sensitivity_path: str
    sensitivity_sha256: str
    fairness_support_path: str | None = None
    fairness_support_sha256: str | None = None
    categoricals_path: str | None = None
    categoricals_sha256: str | None = None
    overrides_path: str | None = None
    overrides_sha256: str | None = None

    #: Every input checksum re-read after the last table was written. This component is a pure
    #: observer of nine closed components; if any input moved, it was not.
    inputs_unchanged: bool
    input_sha256_after: dict[str, str]

    eligibility_column: str
    eligibility_rule: str
    eligibility_is_not_geography: str
    secondary_flag_column: str
    eligible_population_share: float
    capacity_semantics: str
    k_levels: list[str]
    primary_k_level: str
    reserve_shares: list[float]
    policy_grid: list[dict[str, object]]

    candidate_models: list[str]
    refused_models: list[str]
    selection_axes: list[list[str]]
    selection_tie_rule: str
    selection_fold_set: str
    discarded_tie_band: float
    selected_model: str
    selection_decided_on_axis: str
    selected_model_under_discarded_band: str
    production_model_claim: str

    policy_winner: str | None
    policy_winner_rule: str
    no_winner_statement: str | None

    abstention_policy: str
    override_cannot: str
    determinism_scope: str
    overrides_applied: int

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
    "Allocation",
    "ArtifactRecord",
    "Override",
    "PolicyManifest",
    "PolicyStats",
    "PolicyWindow",
    "SelectionAxis",
    "ValidationCheck",
]
