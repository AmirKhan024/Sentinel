"""Data structures for baseline model training.

``FittedModel`` is the typed facade over an otherwise opaque scikit-learn pipeline.
scikit-learn ships no ``py.typed`` marker, so under mypy's strict mode every symbol it
exports is ``Any``; rather than let that spread through the package, everything the rest
of Component 6 needs from a fit is extracted once, converted explicitly, and carried
here as ordinary typed Python. See ADR 0015.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from sentinel.modeling.definitions import ModelSpec


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One post-training assertion about a model or its predictions."""

    name: str
    passed: bool
    severity: str
    detail: str
    offenders: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FittedModel:
    """One model fitted to one fold's training window.

    ``pipeline`` is deliberately typed ``Any``: it is a scikit-learn estimator and
    treating it as opaque is the honest annotation. Every value downstream needs is
    already extracted into a typed field below, so nothing else in the package has to
    reach into it.

    ``trained_through`` is the last reference date this fit was allowed to learn from,
    and it is the fold's **training** end. Component 6 fits no calibrator and never
    reads the calibration window; declaring the calibration end would claim a horizon
    the model did not use. See ADR 0014.
    """

    spec: ModelSpec
    fold_set: str
    fold_id: str
    pipeline: Any
    matrix_columns: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    imputed_values: dict[str, float]
    n_iter: int
    converged: bool
    train_rows: int
    train_positive_rate: float | None
    train_start: date
    train_end: date
    trained_through: date
    calibration_end_unused: date


@dataclass
class BaselineStats:
    """Counts accumulated during a run, surfaced in the manifest and the CLI."""

    feature_rows: int = 0
    folds: int = 0
    fold_sets: dict[str, int] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)
    fits: int = 0
    prediction_rows: int = 0
    coefficient_rows: int = 0
    training_log_rows: int = 0
    fit_seconds_total: float = 0.0


class ArtifactRecord(BaseModel):
    """Provenance for one written file."""

    path: str
    bytes: int
    sha256: str
    row_count: int
    schema_: dict[str, str] = Field(alias="schema")

    model_config = {"populate_by_name": True}


class BaselineModelManifest(BaseModel):
    """Self-contained provenance and QA record for one baseline training run.

    Pins the feature table by checksum and records the library versions and BLAS thread
    count, because the coefficients depend on all three and nothing else in the
    repository pins them. The determinism claim this manifest supports is narrow and
    deliberate: identical predictions for a fixed input, a fixed row order, a fixed
    library set and a fixed thread count -- not across library versions. See ADR 0015.
    """

    component: str = "baseline_models"
    code_version: str
    model_definition_version: str
    built_at: str

    features_path: str
    features_sha256: str
    feature_definition_version: str

    sklearn_version: str
    numpy_version: str
    blas_threads: str

    score_direction: str
    trained_through_semantics: str
    missing_value_rules: dict[str, str]
    indicator_columns: list[str]

    fold_sets: dict[str, int]
    folds: int
    models: list[str]
    model_feature_counts: dict[str, int]
    approximation_notes: dict[str, str]
    seeds: dict[str, int]
    model_params: dict[str, str]

    feature_rows: int
    fits: int
    prediction_rows: int
    coefficient_rows: int
    fit_seconds_total: float

    blocked: list[str]
    artifacts: list[ArtifactRecord]
    checks: list[dict[str, str]]
