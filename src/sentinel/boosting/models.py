"""Data structures for Component 7.

``FittedBooster`` is the typed facade over an otherwise opaque estimator, for the same
reason ``modeling.models.FittedModel`` is: xgboost and lightgbm ship partial or absent
type information, so under mypy's strict mode everything they export is ``Any``. Rather
than let that spread, every value the rest of the package needs is extracted once,
converted explicitly, and carried here as ordinary typed Python. See ADR 0016.

The difference from ``FittedModel`` is what is *absent*. There is no ``scaler_mean``, no
``scaler_scale`` and no ``imputed_values``, because a boosted model fits no preprocessing
statistics -- see the ``preprocess`` docstring. In their place is ``null_columns``, the
count of matrix cells that reached the estimator as NaN, which is the observable that
proves the imputation really did not happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from sentinel.boosting.definitions import BoostingSpec


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One post-training assertion about a booster or its predictions."""

    name: str
    passed: bool
    severity: str
    detail: str
    offenders: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FittedBooster:
    """One boosted model fitted to one fold's training window.

    ``estimator`` is deliberately typed ``Any``: it is an xgboost or lightgbm object and
    treating it as opaque is the honest annotation.

    ``trained_through`` is the fold's **training** end, exactly as in Component 6. This
    component fits no calibrator, and -- unlike a naive booster implementation -- it does
    no early stopping in the final fit either, so the calibration window is untouched by
    model fitting. ``n_estimators`` came from the frozen tuning result, which was
    selected on data strictly earlier than any test window. Declaring ``calibration_end``
    would claim a horizon this model did not use. See ADR 0014 and ADR 0017.
    """

    spec: BoostingSpec
    fold_set: str
    fold_id: str
    estimator: Any
    matrix_columns: tuple[str, ...]
    params: dict[str, object]
    n_estimators: int
    trees_built: int
    importances: tuple[float, ...]
    scale_pos_weight: float
    train_rows: int
    train_positive_rate: float | None
    train_nan_cells: int
    train_start: date
    train_end: date
    trained_through: date
    calibration_end_unused: date


@dataclass(frozen=True, slots=True)
class InnerFoldScore:
    """One trial's result on one inner validation window."""

    fold_id: str
    train_rows: int
    validation_rows: int
    pr_auc: float
    best_iteration: int


@dataclass(frozen=True, slots=True)
class TrialResult:
    """One Optuna trial: what it tried, and what it scored on the inner folds.

    ``mean_pr_auc`` is the objective. It is a *validation* number over windows that are
    training data for every fold the chosen parameters will be used on, so it is never a
    result and never comparable with a Component 5 metric. The distinction is enforced by
    where it is written -- ``data/processed/tuning/``, not ``evaluation/``. See ADR 0018.
    """

    study: str
    model_name: str
    fold_set: str
    number: int
    params: dict[str, object]
    inner_scores: tuple[InnerFoldScore, ...]
    mean_pr_auc: float
    n_estimators: int
    seconds: float
    failed: bool = False
    failure: str = ""


@dataclass(frozen=True, slots=True)
class StudyResult:
    """One completed search: every trial, and the one that won."""

    study: str
    model_name: str
    fold_set: str
    region_start: date
    region_end: date
    inner_folds: tuple[str, ...]
    trials: tuple[TrialResult, ...]
    best: TrialResult
    sampler_seed: int
    seconds: float


@dataclass
class BoostingStats:
    """Counts accumulated during a run, surfaced in the manifest and the CLI."""

    feature_rows: int = 0
    folds: int = 0
    fold_sets: dict[str, int] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)
    fits: int = 0
    prediction_rows: int = 0
    importance_rows: int = 0
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


class BoostedModelManifest(BaseModel):
    """Self-contained provenance and QA record for one boosted training run.

    Pins the feature table by checksum and records every library version and the thread
    count, because a boosted ensemble depends on all of them. The determinism claim is
    the same narrow one Component 6 makes: identical predictions for a fixed input, a
    fixed row order, a fixed library set and a single thread -- not across library
    versions. See ADR 0016.
    """

    component: str = "boosted_models"
    code_version: str
    boosting_definition_version: str
    built_at: str

    features_path: str
    features_sha256: str
    feature_definition_version: str

    xgboost_version: str
    lightgbm_version: str
    numpy_version: str
    sklearn_version: str
    blas_threads: str

    score_direction: str
    trained_through_semantics: str
    probability_semantics: str
    preprocessing: str
    matrix_columns: list[str]

    fold_sets: dict[str, int]
    folds: int
    models: list[str]
    model_feature_counts: dict[str, int]
    model_estimators: dict[str, str]
    seeds: dict[str, int]
    model_params: dict[str, str]
    tuned_params_provenance: str

    feature_rows: int
    fits: int
    prediction_rows: int
    importance_rows: int
    fit_seconds_total: float

    blocked: list[str]
    artifacts: list[ArtifactRecord]
    checks: list[dict[str, str]]


class TuningManifest(BaseModel):
    """Provenance for one hyperparameter search.

    ``region_start`` / ``region_end`` are the whole point of this record: they are the
    proof that the search could not have seen a test window. A reader who wants to check
    the claim compares ``region_end`` against the fold set's first ``test_start``, both
    of which are recorded here.
    """

    component: str = "boosting_tuning"
    code_version: str
    boosting_definition_version: str
    built_at: str

    features_path: str
    features_sha256: str
    feature_definition_version: str

    xgboost_version: str
    lightgbm_version: str
    optuna_version: str
    numpy_version: str
    blas_threads: str

    objective: str
    sampler: str
    sampler_seed: int
    trials_requested: int
    early_stopping_rounds: int
    max_boosting_rounds: int

    studies: list[dict[str, str]]
    tuning_regions: dict[str, str]
    first_test_start: dict[str, str]
    inner_folds: dict[str, list[str]]
    best_params: dict[str, str]
    best_scores: dict[str, float]

    feature_rows: int
    trial_rows: int
    failed_trials: int
    seconds_total: float

    blocked: list[str]
    artifacts: list[ArtifactRecord]
    checks: list[dict[str, str]]


__all__ = [
    "ArtifactRecord",
    "BoostedModelManifest",
    "BoostingStats",
    "FittedBooster",
    "InnerFoldScore",
    "StudyResult",
    "TrialResult",
    "TuningManifest",
    "ValidationCheck",
]
