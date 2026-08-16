"""Data structures for as-of feature construction."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One post-build assertion about the feature table."""

    name: str
    passed: bool
    severity: str
    detail: str
    offenders: tuple[str, ...] = ()


@dataclass
class FeatureStats:
    """Counts accumulated during a build, surfaced in the manifest and the CLI."""

    eligible_target_rows: int = 0
    feature_rows: int = 0
    history_rows: int = 0
    rows_without_any_history: int = 0
    rows_without_prior_canvass: int = 0
    rows_without_code_era_canvass: int = 0
    rows_with_name_change: int = 0
    null_rates: dict[str, float] = field(default_factory=dict)


class ArtifactRecord(BaseModel):
    """Provenance for one written file."""

    path: str
    bytes: int
    sha256: str
    row_count: int
    schema_: dict[str, str] = Field(alias="schema")

    model_config = {"populate_by_name": True}


class FeatureManifest(BaseModel):
    """Self-contained provenance and QA record for one feature build.

    Pins **three** inputs by checksum. The features are a function of the raw
    snapshot, Component 2's identities and Component 3's targets together, so
    recording fewer would leave the output unreproducible.
    """

    component: str = "as_of_features"
    code_version: str
    feature_definition_version: str
    built_at: str

    source_path: str
    source_sha256: str
    assignments_path: str
    assignments_sha256: str
    targets_path: str
    targets_sha256: str

    temporal_boundary: str
    window_days: list[int]
    code_era_start: str

    eligible_target_rows: int
    feature_rows: int
    feature_count: int
    rows_without_any_history: int
    rows_without_prior_canvass: int
    rows_without_code_era_canvass: int
    rows_with_name_change: int
    null_rates: dict[str, float]

    artifacts: list[ArtifactRecord]
    checks: list[dict[str, str]]
