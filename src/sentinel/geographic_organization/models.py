"""Data structures for Component 20.

Follows the same Pydantic + dataclass pattern as the other operational components
(18, 19): Pydantic for the JSON manifest sidecar, dataclass for the in-memory result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from pydantic import BaseModel, Field

from sentinel.features.models import ValidationCheck
from sentinel.geographic_organization.metrics import GeographicGroupMetrics

__all__ = [
    "ArtifactRecord",
    "GeographicOrganizationManifest",
    "GeographicOrganizationResult",
    "GeographicPlanSummary",
]


class ArtifactRecord(BaseModel):
    """Provenance for one written file -- identical shape to other components."""

    path: str
    bytes: int
    sha256: str
    row_count: int
    schema_: dict[str, str] = Field(alias="schema")

    model_config = {"populate_by_name": True}


@dataclass
class GeographicPlanSummary:
    """High-level geographic coverage summary for one planning run."""

    planning_date: str
    selected_count: int
    location_available_count: int
    location_unavailable_count: int
    location_coverage_pct: float
    # Count of *mapped* proximity groups -- the "unmapped" pseudo-group is not counted.
    geographic_group_count: int
    threshold_km: float
    group_metrics: list[GeographicGroupMetrics] = field(default_factory=list)
    organization_mode: str = "risk_first"
    threshold_preset: str | None = None
    notes: list[str] = field(default_factory=list)


class GeographicOrganizationManifest(BaseModel):
    """Self-contained provenance and QA record for one geographic organization run.

    Answers from the artifact alone: which selection set was organized, using
    which algorithm and threshold, producing how many geographic proximity groups.
    """

    component: str = "geographic_organization"
    code_version: str
    geographic_organization_definition_version: str
    built_at: str

    planning_date: str

    selection_artifact_path: str
    selection_artifact_sha256: str
    operational_selection_definition_version: str
    composite_model_name: str

    geographic_algorithm: str
    threshold_km: float

    selected_count: int
    location_available_count: int
    location_unavailable_count: int
    location_coverage_pct: float
    # Mapped proximity groups only; "unmapped" is not counted here.
    geographic_group_count: int

    #: One entry per group (mapped groups, then "unmapped" if non-empty), so "for
    #: planning date X, which establishments were grouped together, and why" is
    #: answerable from the manifest alone -- not only from the wider parquet table.
    group_metrics: list[dict[str, object]]

    #: How the suggested order within each block was produced. See OrganizationMode.
    organization_mode: str = "risk_first"
    #: The named threshold label used, if any ("tight"/"balanced"/"broad"); None if an
    #: explicit threshold_km was given instead.
    threshold_preset: str | None = None
    #: One entry per work block -- the operational planning view of group_metrics,
    #: including highest_sentinel_rank, rank_range, and suggested_order.
    work_blocks: list[dict[str, object]] = Field(default_factory=list)
    #: Honest, computed observations about this run (e.g. "most work blocks are
    #: singletons at this threshold") -- never hidden, always derived from the plan itself.
    notes: list[str] = Field(default_factory=list)

    warnings: list[str]
    artifacts: list[ArtifactRecord]
    checks: list[dict[str, str]]


@dataclass
class GeographicOrganizationResult:
    """Everything a caller needs after a geographic organization build."""

    plan_frame: pl.DataFrame
    summary: GeographicPlanSummary
    checks: list[ValidationCheck]
    manifest: GeographicOrganizationManifest
    plan_path: Path | None = None
    manifest_path: Path | None = None
