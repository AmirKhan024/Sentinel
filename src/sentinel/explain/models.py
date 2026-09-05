"""Data structures for Component 11.

The central type is :class:`FoldAttribution`: one model's attributions over one fold's
explanation sample, carrying the values, the base value, the model output being decomposed,
the column names, the raw feature values behind them, and the provenance that says which
model and which information horizon produced it.

It carries the raw feature values as well as the transformed ones deliberately. A local
explanation that reads "``prior_canvass_priority_rate`` contributed +0.31" is only
actionable if the reader can also see that the rate *was* 0.82 -- and the transformed value
the estimator saw, ``+1.74`` standard deviations, is not a number anyone outside this
repository can interpret.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from sentinel.explain.definitions import ExplanationMethod, ExplanationSpec, OutputSpace


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One post-attribution assertion about an explanation or the artifact holding it."""

    name: str
    passed: bool
    severity: str
    detail: str
    offenders: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RefitModel:
    """A re-executed fit, plus everything needed to attribute and to prove its identity.

    ADR 0026 established that no fitted model object is persisted anywhere in this
    repository, and ADR 0029 extends its licence to this component: the fit runs again, at
    the same spec, seed and canonical row order, and is then *proved* to be the committed
    model by scoring the test window and comparing it to the committed artifact with ``==``.

    ``estimator`` is typed ``Any`` for the reason every fitted-model facade in this project
    is: it is a scikit-learn, xgboost, lightgbm or torch object, and annotating it more
    precisely would be a false claim rather than a stronger one.
    """

    spec: ExplanationSpec
    fold_set: str
    fold_id: str
    estimator: Any
    #: Column names in the order this model's matrix presents them. Recovered by the
    #: function ``spec.name_source`` names, never inferred.
    matrix_columns: tuple[str, ...]
    #: The transformed matrix over the explained rows -- what the estimator actually sees.
    matrix: NDArray[np.float64]
    #: The raw pre-transform values, aligned to ``matrix_columns``. NaN where the source
    #: was NULL, which is a real observation rather than a gap for the tree models.
    raw_matrix: NDArray[np.float64]
    #: ``target_inspection_id`` for each row of ``matrix``, in matrix order.
    row_ids: tuple[str, ...]
    #: The model's own log-odds output for each explained row.
    output: NDArray[np.float64]
    #: The probability each explained row was committed with, for the identity gate.
    probability: NDArray[np.float64]
    trained_through: date
    train_start: date
    train_end: date
    #: Reference rows for a method that needs one, already proved to sit at or before
    #: ``train_end``. Empty for the tree models, which need no background at all.
    background: NDArray[np.float64]
    background_max_date: date | None
    fit_seconds: float


@dataclass(frozen=True, slots=True)
class FoldAttribution:
    """One model's attributions for one fold.

    ``values`` is ``(rows, features)`` in the same column order as
    :attr:`RefitModel.matrix_columns`, and in the same row order as
    :attr:`RefitModel.row_ids`. Both alignments are load-bearing and both are re-asserted
    by ``validate`` rather than assumed: a transposed or mis-sorted block would produce an
    artifact that passes every additivity check and attributes every value to the wrong
    feature or the wrong establishment.
    """

    model_name: str
    fold_set: str
    fold_id: str
    method: ExplanationMethod
    output_space: OutputSpace
    is_exact: bool
    row_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: NDArray[np.float64]
    #: One expected value for the whole fold, not one per row. A per-row base value would
    #: mean the decomposition had no common reference point to be compared against.
    base_value: float
    output: NDArray[np.float64]
    seconds: float

    @property
    def reconstruction(self) -> NDArray[np.float64]:
        """``base + sum(phi)`` per row: what the attributions claim the model output was."""
        return self.base_value + self.values.sum(axis=1)

    @property
    def residual(self) -> NDArray[np.float64]:
        """How far that claim is from the model's actual output."""
        return np.abs(self.reconstruction - self.output)


@dataclass(frozen=True, slots=True)
class ExplanationSample:
    """The rows one fold's explanations are computed over, and how they were chosen.

    Carried rather than recomputed at each use so that "which rows were explained" has one
    answer per run, recorded in the artifact beside every value it produced.
    """

    fold_set: str
    fold_id: str
    ids: tuple[str, ...]
    population_rows: int
    strategy: str
    seed: int
    population: str


@dataclass(frozen=True, slots=True)
class ImportanceRow:
    """One model's importance for one feature, on one fold or aggregated over a fold set.

    ``fold_id`` is ``None`` on an aggregate row. ``scope`` says which, so a consumer filters
    on a declared column instead of testing a null.
    """

    model_name: str
    fold_set: str
    fold_id: str | None
    scope: str
    feature_name: str
    original_feature_name: str
    mean_abs_shap: float
    mean_shap: float
    rank: int
    #: Populated on aggregate rows only: the spread of the per-fold statistics being
    #: summarised. A mean importance without its variability invites "feature X is the most
    #: important feature", which is the claim this component most needs to avoid making
    #: when the ranks move.
    sd_abs_shap: float | None
    mean_rank: float | None
    sd_rank: float | None
    best_rank: int | None
    worst_rank: int | None
    folds: int
    rows: int


@dataclass(frozen=True, slots=True)
class StabilityRow:
    """How much one model's importance ranking moved between two folds.

    Rank correlation and top-k overlap answer different questions and disagree usefully:
    a model can reorder its tail while keeping the same top ten, or swap two dominant
    features while every other rank holds.
    """

    model_name: str
    fold_set: str
    comparison: str
    from_fold_id: str
    to_fold_id: str
    spearman_rho: float
    top_k: int
    top_k_jaccard: float
    features: int


@dataclass(frozen=True, slots=True)
class DriftRow:
    """How far one feature's importance rank travelled across a fold set."""

    model_name: str
    fold_set: str
    feature_name: str
    original_feature_name: str
    first_fold_id: str
    last_fold_id: str
    first_rank: int
    last_rank: int
    best_rank: int
    worst_rank: int
    rank_range: int
    mean_abs_shap: float
    sd_abs_shap: float
    coefficient_of_variation: float | None
    #: Measured against ``RANK_DRIFT_THRESHOLD``, which was declared before the ranks were
    #: computed. A "materially changed" flag chosen after seeing the ranks would be a
    #: conclusion dressed as a criterion.
    materially_changed: bool


@dataclass(frozen=True, slots=True)
class RepresentativeCase:
    """One local explanation selected for the report, by predicted score alone."""

    model_name: str
    fold_set: str
    fold_id: str
    tier: str
    quantile: float
    target_inspection_id: str
    base_value: float
    prediction_value: float
    base_score: float
    calibrated_probability: float | None
    calibration_method: str | None
    output_space: str
    method: str
    is_exact: bool


@dataclass(frozen=True, slots=True)
class ReproductionOutcome:
    """The bit-identity gate's verdict for one (model, fold).

    ADR 0026's gate, reused rather than reimplemented: ``calibration.basescores`` owns the
    comparison, and this records what it returned.
    """

    model_name: str
    fold_id: str
    rows: int
    mismatches: int
    offenders: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.mismatches == 0


@dataclass(frozen=True, slots=True)
class ExplainStats:
    """Counters a run reports and the manifest records."""

    folds: int
    fold_sets: dict[str, int]
    feature_rows: int
    models_supported: int
    models_unsupported: int
    refits: int
    explained_rows: int
    attribution_values: int
    reproduction_rows: int
    reproduction_mismatches: int
    refit_seconds: float
    attribute_seconds: float


class ArtifactRecord(BaseModel):
    """Provenance for one written file."""

    path: str
    bytes: int
    sha256: str
    row_count: int
    schema_: dict[str, str] = Field(alias="schema")

    model_config = {"populate_by_name": True}


class ExplainManifest(BaseModel):
    """Self-contained provenance and QA record for one explanation run.

    Pins every input by checksum, including the four prediction artifacts this component
    reads and never rewrites. Their sha256 is recorded *after* the run as well as before,
    so "Component 11 changed no prediction" is a checkable claim rather than an assurance.

    The determinism claim is the same narrow one Components 6-9 make: identical output for a
    fixed input, a fixed row order, a fixed library set, one thread and CPU -- not across
    library versions. A version bump makes the bit-identity gate fail, and that is correct.
    """

    component: str = "explainability"
    code_version: str
    explain_definition_version: str
    built_at: str

    features_path: str
    features_sha256: str
    feature_definition_version: str
    # Optional because a run that explains only Component 7's boosters neither reads nor
    # checksums Components 6 and 8's artifacts. ``None`` states that; an empty string would
    # be indistinguishable from a path that failed to record.
    baseline_predictions_path: str | None
    baseline_predictions_sha256: str | None
    boosted_predictions_path: str | None
    boosted_predictions_sha256: str | None
    neural_predictions_path: str | None
    neural_predictions_sha256: str | None
    calibrated_predictions_path: str | None
    calibrated_predictions_sha256: str | None

    #: The same four checksums re-read after every artifact was written. Equal to the
    #: values above on any correct run.
    prediction_artifacts_unchanged: bool
    prediction_sha256_after: dict[str, str]

    # The integrity claim ADR 0029 rests on.
    reproduction_rows: dict[str, int]
    reproduction_mismatches: dict[str, int]
    reproduction_passed: bool

    # The support matrix, restated so a consumer never has to import this package.
    supported_models: list[str]
    unsupported_models: dict[str, str]
    explanation_methods: dict[str, str]
    output_spaces: dict[str, str]
    exactness: dict[str, bool]
    name_sources: dict[str, str]

    # The sampling and background budget (ADR 0030).
    sample_strategy: str
    sample_size: int
    sampling_seed: int
    sampling_population: str
    background_strategy: str
    background_size: int
    background_seed: int
    permutation_rounds: int
    additivity_tolerance: dict[str, float]
    max_additivity_residual: dict[str, float]

    # Analysis parameters, declared before the ranks were computed.
    top_k: int
    rank_drift_threshold: int
    representative_quantiles: dict[str, float]
    stability_metrics: list[str]
    covid_reported_separately: bool

    # Semantics, as prose, so a consumer never infers them.
    attribution_semantics: str
    calibration_boundary: str
    causality_disclaimer: str

    # Measured.
    fold_sets: dict[str, int]
    folds: int
    feature_rows: int
    refits: int
    explained_rows: int
    attribution_values: int
    refit_seconds_total: float
    attribute_seconds_total: float

    # Environment. Bit-identity depends on every line of it.
    sklearn_version: str
    numpy_version: str
    xgboost_version: str
    lightgbm_version: str
    torch_version: str
    torch_threads: int
    blas_threads: str
    device: str
    determinism_caveat: str

    blocked: list[str]
    artifacts: list[ArtifactRecord]
    checks: list[dict[str, str]]


__all__ = [
    "ArtifactRecord",
    "DriftRow",
    "ExplainManifest",
    "ExplainStats",
    "ExplanationSample",
    "FoldAttribution",
    "ImportanceRow",
    "RefitModel",
    "RepresentativeCase",
    "ReproductionOutcome",
    "StabilityRow",
    "ValidationCheck",
]
