"""Data structures for probability calibration.

``FittedCalibrator`` is the typed facade over an otherwise opaque scikit-learn estimator,
following the pattern ``modeling.models.FittedModel`` established (ADR 0015): scikit-learn
ships no ``py.typed`` marker, so everything the rest of the component needs is extracted
once, converted explicitly, and carried here as ordinary typed Python.

It carries more than it strictly needs to *apply* the mapping, because a calibrator that
cannot be reproduced from the artifact is a black box. For Platt that is two floats; for
isotonic it is the breakpoint arrays plus the clip bounds, which with ``np.interp``
reproduce the map exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from sentinel.calibration.definitions import CandidateSpec, Method


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One post-calibration assertion about a calibrator or its output."""

    name: str
    passed: bool
    severity: str
    detail: str
    offenders: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BaseScores:
    """One base model's regenerated scores for one fold, over both windows.

    The calibration window is what Component 9 wanted. The test window is the **control**:
    it is compared against the committed Component 6/7/8 artifact bit for bit, and if it
    does not match, nothing downstream is trustworthy (ADR 0026).
    """

    model_name: str
    fold_set: str
    fold_id: str
    calibration_ids: tuple[str, ...]
    calibration_scores: tuple[float, ...]
    calibration_margins: tuple[float, ...]
    calibration_labels: tuple[int, ...]
    calibration_dates: tuple[date, ...]
    test_ids: tuple[str, ...]
    test_scores: tuple[float, ...]
    test_margins: tuple[float, ...]
    test_labels: tuple[int, ...]
    base_model_trained_through: date
    fit_seconds: float


@dataclass(frozen=True, slots=True)
class InnerSplit:
    """One calibration window cut chronologically for method selection (ADR 0025).

    ``cut`` is the first date belonging to the select portion, so the two sides never share
    a day -- two inspections of the same establishment days apart share almost all of their
    as-of history, and a mid-day cut would split rows that are not independent.
    """

    cut: date
    fit_index: tuple[int, ...]
    select_index: tuple[int, ...]

    @property
    def fit_rows(self) -> int:
        return len(self.fit_index)

    @property
    def select_rows(self) -> int:
        return len(self.select_index)


@dataclass(frozen=True, slots=True)
class FittedCalibrator:
    """One calibrator fitted to one fold's window.

    ``estimator`` is deliberately typed ``Any`` -- it is a scikit-learn object and treating
    it as opaque is the honest annotation. Everything needed to reproduce the mapping is
    already extracted into the typed fields below.
    """

    model_name: str
    fold_set: str
    fold_id: str
    method: Method
    estimator: Any
    input_transform: str
    fit_rows: int
    fit_positive_rate: float | None
    fit_start: date
    fit_end: date
    #: Platt only: the slope on the logit scale, and the intercept.
    coefficient: float | None = None
    intercept: float | None = None
    #: Isotonic only: the fitted step function and the range it may be applied over.
    x_thresholds: tuple[float, ...] = ()
    y_thresholds: tuple[float, ...] = ()
    x_min: float | None = None
    x_max: float | None = None

    @property
    def breakpoint_count(self) -> int:
        return len(self.x_thresholds)


@dataclass(frozen=True, slots=True)
class MethodTrial:
    """What one method scored on one fold's inner-select window.

    Every number here was measured on a window carved out of the calibration period. None
    of them is a result, and ``inner_select_ece`` in particular decides nothing -- the
    selection metric is the log-loss (ADR 0025).
    """

    model_name: str
    fold_set: str
    fold_id: str
    fold_index: int
    method: Method
    inner_fit_rows: int
    inner_select_rows: int
    inner_split_date: date
    inner_fit_positive_rate: float | None
    inner_select_positive_rate: float | None
    inner_select_log_loss: float
    inner_select_brier: float
    inner_select_ece: float
    inner_select_mce: float
    seconds: float


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    """Which method was frozen for one (model, fold), and on what evidence.

    ``per_fold_winner`` is recorded beside ``method`` so that method instability is a
    measured result rather than an argument: the production rule uses the expanding prefix,
    and the per-fold alternative is auditable without a re-run.
    """

    model_name: str
    fold_set: str
    fold_id: str
    fold_index: int
    method: Method
    per_fold_winner: Method
    prefix_mean_log_loss: dict[Method, float]
    gap: float
    declared_tie: bool
    reason: str


@dataclass(frozen=True, slots=True)
class BrierDecomposition:
    """Murphy's three-term decomposition over 15 equal-mass bins.

    ``within_bin_variance`` is the residual ``brier - (reliability - resolution +
    uncertainty)``. It is reported rather than hidden: the identity is exact only for a
    forecast that is constant within each bin, and reporting ``REL - RES + UNC`` as "the
    Brier score" would be a fabrication.
    """

    reliability: float
    resolution: float
    uncertainty: float
    recomposed: float
    within_bin_variance: float
    n_bins: int


@dataclass(frozen=True, slots=True)
class SlopeIntercept:
    """The Cox recalibration regression of the label on ``logit(p)``.

    Perfect calibration is slope 1.0 and intercept 0.0. Slope below 1 is overconfidence.
    ``None`` on a single-class window, matching ``metrics.roc_auc``'s posture rather than
    inventing a number.
    """

    slope: float | None
    intercept: float | None


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A within-fold percentile interval for one metric under one resampling scheme."""

    metric: str
    scheme: str
    point_estimate: float | None
    replications: int
    seed: int
    mean: float | None
    sd: float | None
    lower: float | None
    upper: float | None
    level: float
    degenerate: int


@dataclass(frozen=True, slots=True)
class RankingPreservation:
    """Whether the calibrator reordered anything, and what it tied together.

    A monotone calibrator cannot reorder. Isotonic can, however, *tie* -- and
    ``evaluation.metrics.top_k_indices`` breaks ties by ``target_inspection_id`` ascending,
    so a plateau can move top-k membership without the map being non-monotone. That is not
    a ranking inversion and must not be reported as one.
    """

    spearman_rho: float | None
    kendall_tau_b: float | None
    distinct_before: int
    distinct_after: int
    new_ties_created: int
    inversions: int
    is_strictly_monotone: bool
    top_k: int
    top_k_membership_changed: int
    precision_at_k_before: float | None
    precision_at_k_after: float | None
    roc_auc_before: float | None
    roc_auc_after: float | None


@dataclass
class CalibrationStats:
    """Counts accumulated during a run, surfaced in the manifest and the CLI."""

    feature_rows: int = 0
    folds: int = 0
    fold_sets: dict[str, int] = field(default_factory=dict)
    candidates: list[str] = field(default_factory=list)
    base_fits: int = 0
    calibrator_fits: int = 0
    calibration_rows: int = 0
    calibrated_prediction_rows: int = 0
    selection_rows: int = 0
    method_counts: dict[str, int] = field(default_factory=dict)
    method_switches: int = 0
    logit_clamped_rows: int = 0
    refit_seconds_total: float = 0.0
    calibrate_seconds_total: float = 0.0


class ArtifactRecord(BaseModel):
    """Provenance for one written file."""

    path: str
    bytes: int
    sha256: str
    row_count: int
    schema_: dict[str, str] = Field(alias="schema")

    model_config = {"populate_by_name": True}


class CalibrationManifest(BaseModel):
    """Self-contained provenance and QA record for one calibration run.

    Pins every input by checksum, including the three Component 6/7/8 prediction artifacts
    -- which this component reads but never rewrites, and whose bit-identity with the
    regenerated scores is the precondition for everything else (ADR 0026).

    The determinism claim is the same narrow one Components 6-8 make: identical output for
    a fixed input, a fixed row order, a fixed library set and one thread -- not across
    library versions. A version bump makes the bit-identity gate fail, and that is correct.
    """

    component: str = "probability_calibration"
    code_version: str
    calibration_definition_version: str
    built_at: str

    features_path: str
    features_sha256: str
    feature_definition_version: str
    categoricals_path: str | None
    categoricals_sha256: str | None
    baseline_predictions_path: str
    baseline_predictions_sha256: str
    boosted_predictions_path: str
    boosted_predictions_sha256: str
    neural_predictions_path: str
    neural_predictions_sha256: str

    # The integrity claim ADR 0026 rests on.
    base_score_reproduction: dict[str, str]
    base_score_reproduction_passed: bool
    logit_recovery_max_error: dict[str, float]
    logit_clamped_rows: int
    margin_tolerance: float

    # Provenance semantics, as prose, so a consumer never infers them.
    trained_through_semantics: str
    available_from_semantics: str
    probability_semantics: str
    score_direction: str

    # The protocol, pre-registered (ADR 0025).
    candidates: list[str]
    experimental_candidates: list[str]
    candidate_rationale: dict[str, str]
    excluded_models: dict[str, str]
    selection_metric: str
    selection_granularity: str
    selection_rule: str
    inner_select_fraction: float
    min_inner_fit_rows: int
    min_inner_select_rows: int
    tie_threshold: float
    tie_preference: str
    method_override: str | None
    selected_methods: dict[str, str]
    method_counts: dict[str, int]
    method_switches: int
    platt_params: str
    isotonic_params: str
    calibration_bins: int
    binning: str

    # Uncertainty.
    bootstrap_replications: int
    bootstrap_seed: int
    bootstrap_schemes: list[str]
    bootstrap_caveat: str

    # Measured.
    fold_sets: dict[str, int]
    folds: int
    feature_rows: int
    base_fits: int
    calibrator_fits: int
    calibration_rows: int
    calibrated_prediction_rows: int
    refit_seconds_total: float
    calibrate_seconds_total: float

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
    "BaseScores",
    "BootstrapInterval",
    "BrierDecomposition",
    "CalibrationManifest",
    "CalibrationStats",
    "CandidateSpec",
    "FittedCalibrator",
    "InnerSplit",
    "MethodTrial",
    "RankingPreservation",
    "SelectionOutcome",
    "SlopeIntercept",
    "ValidationCheck",
]
