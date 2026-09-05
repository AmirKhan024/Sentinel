"""Data structures for operational candidate generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

# Reused rather than redefined: Component 17's checks run in the same list as
# Component 4's reused ones (see candidates.build), so both must share one type
# or the combined list has no single element type. The shape has no
# component-specific meaning to diverge on -- it is a name, a pass/fail and a
# severity -- so importing it here is reuse, not a layering violation.
from sentinel.features.models import ValidationCheck

__all__ = [
    "ArtifactRecord",
    "CandidateManifest",
    "CandidateStats",
    "PlanningContext",
    "ValidationCheck",
]


@dataclass(frozen=True, slots=True)
class PlanningContext:
    """The minimal operational input contract for Component 17.

    Deliberately just a date. Capacity, policy, model selection, geographic
    clustering and route assignment are later components' responsibility (18
    onward) and are refused here on purpose: this component answers "which
    establishments, and what do we know about them" only.
    """

    planning_date: str  # ISO 'YYYY-MM-DD'


@dataclass
class CandidateStats:
    """Counts accumulated during a build, surfaced in the manifest and the CLI."""

    candidate_count: int = 0
    cold_start_candidates: int = 0
    candidates_missing_location: int = 0
    min_supported_planning_date: str = ""
    max_ingested_inspection_date: str = ""
    days_beyond_ingested_data: int = 0
    null_rates: dict[str, float] = field(default_factory=dict)


class ArtifactRecord(BaseModel):
    """Provenance for one written file."""

    path: str
    bytes: int
    sha256: str
    row_count: int
    schema_: dict[str, str] = Field(alias="schema")

    model_config = {"populate_by_name": True}


class CandidateManifest(BaseModel):
    """Self-contained provenance and QA record for one candidate-generation run.

    Pins the same inputs Component 4 pins (raw + Component 2 assignments), plus
    Component 2's establishments table for display metadata, plus the planning
    date itself -- the one input Component 4 never has, because Component 4's
    reference date always comes from a real row in one of the pinned files.
    """

    component: str = "operational_candidates"
    code_version: str
    candidate_definition_version: str
    feature_definition_version: str
    built_at: str

    planning_date: str

    source_path: str
    source_sha256: str
    assignments_path: str
    assignments_sha256: str
    establishments_path: str
    establishments_sha256: str

    temporal_boundary: str
    candidate_eligibility_rule: str

    min_supported_planning_date: str
    max_ingested_inspection_date: str
    days_beyond_ingested_data: int

    candidate_count: int
    feature_count: int
    cold_start_candidates: int
    candidates_missing_location: int
    null_rates: dict[str, float]

    warnings: list[str]
    artifacts: list[ArtifactRecord]
    checks: list[dict[str, str]]
