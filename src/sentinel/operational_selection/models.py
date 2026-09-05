"""Data structures for Component 19."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from sentinel.features.models import ValidationCheck  # reused, see candidates.models

__all__ = [
    "ArtifactRecord",
    "OperationalCapacityRequest",
    "OperationalSelectionManifest",
    "ValidationCheck",
]


@dataclass(frozen=True, slots=True)
class OperationalCapacityRequest:
    """The operational planning input Component 19 actually needs.

    Deliberately just a planning date and a slot count. This is **inspection
    capacity**, not an inspector count: nothing in this repository defines how many
    establishments one inspector can cover, and inventing that ratio here would be
    exactly the fabricated staffing fact the architecture forbids. A future component
    may let a supervisor derive ``maximum_inspections`` from
    ``teams x inspections_per_team``, but that conversion belongs where the assumption
    can be stated honestly -- not here.
    """

    planning_date: str  # ISO 'YYYY-MM-DD', must match the priority set's own date
    maximum_inspections: int
    policy_id: str = ""  # "" means: use operational_selection.definitions.DEFAULT_POLICY_ID


class ArtifactRecord(BaseModel):
    """Provenance for one written file."""

    path: str
    bytes: int
    sha256: str
    row_count: int
    schema_: dict[str, str] = Field(alias="schema")

    model_config = {"populate_by_name": True}


class OperationalSelectionManifest(BaseModel):
    """Self-contained provenance and QA record for one capacity-constrained selection.

    Answers, from the artifact alone: which priority set was selected from, at what
    requested capacity, under which policy, using which allocation engine, and how many
    establishments that produced.
    """

    component: str = "operational_selection"
    code_version: str
    operational_selection_definition_version: str
    built_at: str

    planning_date: str

    priority_set_path: str
    priority_set_sha256: str
    operational_scoring_definition_version: str
    composite_model_name: str

    requested_capacity: int
    policy_id: str
    policy_mechanism: str
    policy_reserve_share: float
    allocation_source: str

    ranked_candidate_count: int
    selectable_candidate_count: int
    unscorable_count: int
    selected_count: int
    reserve_selected_count: int
    risk_selected_count: int
    unfilled_capacity: int
    capacity_utilization: float | None
    coverage_eligible_selected_count: int

    warnings: list[str]
    artifacts: list[ArtifactRecord]
    checks: list[dict[str, str]]
