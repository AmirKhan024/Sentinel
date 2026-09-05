"""Typed structures for Component 16. No behaviour, no I/O, no clock.

``ReviewCase`` is the unit the trigger layer produces: one flagged row, carrying enough of
Component 13's and Component 14's own fields that a reviewer never has to join back to either
artifact to see why it was flagged.

``ReviewResolution`` is the only input to this component a human types. Everything else on the
way in is an artifact another component wrote and checksummed; this arrives as a JSON file
somebody edited, so it is validated at the boundary and rejected with a message rather than
trusted and half-applied. The same discipline Component 13's ``Override`` and Component 14's
``Adjustment``/``ExecutionEvent`` already apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, Field

#: A defect in the review computation. Fails the run.
SEVERITY_ERROR = "error"

#: A finding about the review run. Recorded, printed, exit zero. ADR 0034's line, held.
SEVERITY_WARN = "warn"

#: Offenders listed on a failing check before the list is truncated. Matching every other
#: component: a report that prints ten thousand ids is a report nobody reads.
MAX_OFFENDERS = 20


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One assertion about the review run or the artifact holding it.

    ``severity`` decides whether a failure stops the run, inheriting the line ADR 0034 drew and
    every later component has held: a defect in the *computation* is an error, and a finding
    about the run (how many cases were flagged, whether a pointer resolves yet) is an advisory.
    """

    name: str
    passed: bool
    severity: str
    detail: str
    offenders: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReviewCase:
    """One flagged row, computed fresh each run.

    Carries enough of Component 13's and Component 14's own fields that a reviewer never has to
    join back to either artifact to see why the row was flagged. Nothing here is recomputed --
    every field is read verbatim from the source it names.

    ``trigger_reasons`` is non-empty by construction: a row with no trigger is not a case, and
    the invariant is enforced here rather than only checked downstream.
    """

    policy_id: str
    model_name: str
    fold_set: str
    fold_id: str
    k_name: str
    target_inspection_id: str
    establishment_id: str
    final_policy_rank: int | None
    decision_mechanism: str
    decision_reason: str
    warnings: str
    schedule_config_id: str | None
    planning_run_id: str | None
    replan_index: int | None
    scheduled_date: date | None
    trigger_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.trigger_reasons:
            raise ValueError(
                f"{self.target_inspection_id}: a review case must carry at least one trigger "
                "reason. A row with no trigger is not a case"
            )


class ReviewResolution(BaseModel):
    """One human decision about one flagged case.

    A pydantic model rather than a dataclass because this is the only input to the component a
    human types. ``decided_at`` is the reviewer's own timestamp, not the run's -- a run that
    stamped its own clock onto a human decision would be recording when the file was processed
    and labelling it when the decision was made.

    The two pointer fields are optional on the model itself; the parser enforces which one (if
    any) a given ``resolution_action`` requires, from ``POINTER_FIELD_FOR_ACTION``.
    """

    review_id: str
    policy_id: str
    fold_id: str
    k_name: str
    target_inspection_id: str
    resolution_action: str
    reason_code: str
    actor: str
    decided_at: str
    referenced_override_id: str | None = None
    referenced_adjustment_id: str | None = None
    escalation_note: str | None = None

    model_config = {"extra": "forbid"}


@dataclass(frozen=True, slots=True)
class ResolutionOutcome:
    """What applying one resolution did, including when it did nothing."""

    resolution: ReviewResolution
    outcome: str
    original_status: str = ""
    final_status: str = ""


@dataclass
class ReviewStats:
    """Counts accumulated during a run, surfaced in the manifest and the CLI summary."""

    recommendation_rows: int = 0
    schedule_rows: int = 0
    execution_rows: int = 0
    cases_flagged: int = 0
    cases_resolved: int = 0
    resolutions_applied: int = 0
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


class ReviewManifest(BaseModel):
    """Self-contained provenance and QA record for one review run.

    The question this manifest must answer without reading any source code is *"which cases did
    Sentinel flag for a human, why, and what has a human decided about them?"* -- so the trigger
    set, the no-threshold statement, the resolution vocabulary and the boundary all travel inside
    it rather than only in documentation.
    """

    component: str = "human_review"
    code_version: str
    review_definition_version: str
    built_at: str

    recommendations_path: str
    recommendations_sha256: str
    policy_definition_version: str
    schedule_path: str | None = None
    schedule_sha256: str | None = None
    schedule_definition_version: str | None = None
    execution_log_path: str | None = None
    execution_log_sha256: str | None = None
    resolutions_path: str | None = None
    resolutions_sha256: str | None = None

    #: Every input checksum re-read after the last table was written. This component is a pure
    #: observer of Components 13 and 14; if any input moved, it was not.
    inputs_unchanged: bool
    input_sha256_after: dict[str, str]

    four_human_layers: str
    deferral_is_not_scheduling_deferral: str
    review_cannot: str
    abstention_policy_inherited: str
    no_threshold: str
    determinism_scope: str

    does_not_establish: list[str]
    blocked: list[str]
    inherited_limitations: list[str]

    cases_flagged: int
    cases_resolved: int
    resolutions_applied: int

    checks: list[dict[str, object]]
    advisories: list[str]
    artifacts: list[ArtifactRecord]
    row_counts: dict[str, int]
    seconds: float


__all__ = [
    "MAX_OFFENDERS",
    "SEVERITY_ERROR",
    "SEVERITY_WARN",
    "ArtifactRecord",
    "ResolutionOutcome",
    "ReviewCase",
    "ReviewManifest",
    "ReviewResolution",
    "ReviewStats",
    "ValidationCheck",
]
