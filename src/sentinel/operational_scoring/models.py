"""Data structures for Component 18."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from sentinel.features.models import ValidationCheck  # reused, see candidates.models

__all__ = [
    "ArtifactRecord",
    "HyperparameterProvenance",
    "OperationalScoringManifest",
    "ProductionModelChoice",
    "ValidationCheck",
]


@dataclass(frozen=True, slots=True)
class HyperparameterProvenance:
    """Exactly where one operational fit's hyperparameters came from.

    Introduced so "which hyperparameter configuration was used, and where did it come
    from" is answerable from the manifest alone, rather than only from
    ``operational_scoring.fit``'s module docstring. ``fold_set`` is ``None`` for a
    family with no tuning stage at all (logistic -- Component 6 was never tuned); it is
    Component 18's borrowed ``TUNING_FOLD_SET`` for a family that reuses a frozen study
    (boosted, neural).
    """

    fold_set: str | None
    source: str
    values: dict[str, str]


@dataclass(frozen=True, slots=True)
class ProductionModelChoice:
    """Which model this run scored with, and the exact evidence for the choice.

    Assembled once by ``selection.py`` and carried through the whole run so that every
    downstream module (fitting, calibration, the manifest) reads the same answer to
    "which model" rather than each re-deriving it.
    """

    composite_model_name: str  # e.g. "xgboost_platt" -- Component 13's selection output
    base_model_name: str  # e.g. "xgboost" -- what Components 6/7/8 and Component 9 key on
    method: str  # "platt" | "isotonic"
    calibration_fold_set: str
    calibration_fold_id: str
    decided_on_axis: str
    n_tied_on_nde: int


class ArtifactRecord(BaseModel):
    """Provenance for one written file."""

    path: str
    bytes: int
    sha256: str
    row_count: int
    schema_: dict[str, str] = Field(alias="schema")

    model_config = {"populate_by_name": True}


class OperationalScoringManifest(BaseModel):
    """Self-contained provenance and QA record for one operational scoring run.

    Answers, from the artifact alone: which candidate set was scored, which exact model
    produced the scores, on what training window, with which calibrator, and whether the
    result is reproducible from what is pinned here.
    """

    component: str = "operational_scoring"
    code_version: str
    operational_scoring_definition_version: str
    built_at: str

    planning_date: str

    candidates_path: str
    candidates_sha256: str
    candidate_definition_version: str
    feature_definition_version: str

    historical_features_path: str
    historical_features_sha256: str

    simulation_path: str
    simulation_sha256: str
    metrics_path: str
    metrics_sha256: str
    sensitivity_path: str
    sensitivity_sha256: str
    calibrated_predictions_path: str
    calibrated_predictions_sha256: str
    calibrator_parameters_path: str
    calibrator_parameters_sha256: str
    calibrator_isotonic_breakpoints_path: str
    calibrator_isotonic_breakpoints_sha256: str

    model_family: str
    composite_model_name: str
    base_model_name: str
    model_selection_source: str
    model_selection_decided_on_axis: str
    model_selection_n_tied_on_nde: int

    hyperparameter_fold_set: str | None
    hyperparameter_source: str
    hyperparameter_values: dict[str, str]

    calibration_method: str
    calibration_source: str
    calibration_fold_set: str
    calibration_fold_id: str
    calibrator_fit_start: str
    calibrator_fit_end: str

    operational_fold_set: str
    operational_fold_id: str
    training_window_rule: str
    train_start: str
    train_end: str
    train_rows: int
    train_positive_rate: float | None

    score_direction: str
    tie_break_column: str

    candidate_count: int
    scored_count: int
    excluded_count: int
    coverage_eligible_count: int

    warnings: list[str]
    artifacts: list[ArtifactRecord]
    checks: list[dict[str, str]]
