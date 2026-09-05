"""Data structures for Component 8.

``FittedNetwork`` is the typed facade over an otherwise opaque fit, for the same reason
``modeling.models.FittedModel`` and ``boosting.models.FittedBooster`` are. The difference
is what it carries that neither of those needs: an epoch history, a vocabulary record and
the embedding tables. All three are fitted artifacts, all three are per fold, and all
three are things a leakage test has to be able to read without reconstructing a fit.

``trained_through`` is the fold's **training** end, as in Components 6 and 7. Component 8
early-stops, which those did not -- but it early-stops against a window carved from the
*end of the training data*, never the fold's calibration window. So the claim
``trained_through = fold.train_end`` is as literally true here as it is there, and
``inner_validation_start`` is recorded so a reader can check rather than trust. See ADR
0021.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, Field

from sentinel.neural.definitions import NeuralSpec
from sentinel.neural.encode import FoldEncoding


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One post-training assertion about a network or its predictions."""

    name: str
    passed: bool
    severity: str
    detail: str
    offenders: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """One training epoch: both losses and the learning rate in force.

    ``validation_loss`` is measured on rows strictly inside the training window. It is
    the early-stopping signal and the ``ReduceLROnPlateau`` signal, and it is **not** a
    result -- it is an in-sample number in exactly the sense Component 7's inner-fold
    PR-AUC is, and reporting it as performance would be the same error.
    """

    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float


@dataclass(frozen=True, slots=True)
class EmbeddingTable:
    """One family's learned vectors, with the categories they belong to."""

    column: str
    categories: tuple[str, ...]
    vectors: tuple[tuple[float, ...], ...]

    @property
    def dim(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


@dataclass(frozen=True, slots=True)
class FittedNetwork:
    """One network fitted to one fold's training window.

    ``stop_reason`` is recorded rather than inferred. "Stopped at epoch 43" is ambiguous
    between early stopping and hitting the cap, and the two mean opposite things about
    whether the budget was adequate.
    """

    spec: NeuralSpec
    fold_set: str
    fold_id: str
    encoding: FoldEncoding
    matrix_columns: tuple[str, ...]
    dense_width: int
    embedding_width: int
    parameter_count: int
    learning_rate: float
    pos_weight: float | None
    epochs: tuple[EpochRecord, ...]
    best_epoch: int
    final_epoch: int
    stop_reason: str
    learning_rate_changes: int
    embeddings: tuple[EmbeddingTable, ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    imputed_values: dict[str, float]
    train_rows: int
    inner_train_rows: int
    inner_validation_rows: int
    inner_validation_start: date
    train_nan_cells: int
    train_positive_rate: float | None
    train_start: date
    train_end: date
    trained_through: date
    calibration_end_unused: date
    seed: int

    def embedding_for(self, column: str) -> EmbeddingTable:
        for table in self.embeddings:
            if table.column == column:
                return table
        raise KeyError(f"{self.spec.name}: no embedding table for {column!r}")


@dataclass(frozen=True, slots=True)
class FittedEmbeddingBooster:
    """One XGBoost fitted on the tree matrix widened by learned chain vectors.

    Deliberately *not* a ``FittedNetwork``: it has no epochs, no scaler and no
    early-stopping history, and forcing it into that shape would mean carrying half a
    dozen fields that mean nothing for it. What it does carry is ``donor_model`` and
    ``donor_fold_id`` -- the network whose embedding table it consumed -- because the
    entire temporal argument for this experiment is that the two are the same fold.

    ``params`` are Component 7's frozen XGBoost parameters, unchanged. Re-tuning for the
    wider matrix would confound "the embeddings helped" with "a second search helped",
    and the experiment asks the first question.
    """

    spec: NeuralSpec
    fold_set: str
    fold_id: str
    matrix_columns: tuple[str, ...]
    params: dict[str, object]
    n_estimators: int
    trees_built: int
    importances: tuple[float, ...]
    embedding_columns: tuple[str, ...]
    donor_model: str
    donor_fold_id: str
    train_rows: int
    train_positive_rate: float | None
    train_nan_cells: int
    train_start: date
    train_end: date
    trained_through: date
    calibration_end_unused: date
    seed: int


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """One learning rate's score on one inner validation window."""

    fold_id: str
    learning_rate: float
    train_rows: int
    validation_rows: int
    pr_auc: float
    best_epoch: int


@dataclass(frozen=True, slots=True)
class SweepResult:
    """One completed learning-rate search over one fold set.

    ``mean_pr_auc`` per rate is a *validation* number over windows that are training data
    for every outer fold the selected rate is then used on. It is never a result and
    never comparable with a Component 5 metric -- which is why the trials land in
    ``data/processed/tuning/`` and not beside the predictions. See ADR 0018.
    """

    study: str
    model_name: str
    fold_set: str
    region_start: date
    region_end: date
    inner_folds: tuple[str, ...]
    points: tuple[SweepPoint, ...]
    scores: tuple[tuple[float, float], ...]
    best_learning_rate: float
    selection_reason: str
    seed: int
    seconds: float


@dataclass(frozen=True, slots=True)
class SeedRun:
    """One seed's outcome for the reproducibility experiment."""

    seed: int
    fold_set: str
    fold_id: str
    pr_auc: float
    roc_auc: float


@dataclass
class NeuralStats:
    """Counts accumulated during a run, surfaced in the manifest and the CLI."""

    feature_rows: int = 0
    folds: int = 0
    fold_sets: dict[str, int] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)
    fits: int = 0
    epochs_total: int = 0
    prediction_rows: int = 0
    epoch_log_rows: int = 0
    embedding_rows: int = 0
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


class NeuralCategoricalsManifest(BaseModel):
    """Provenance for Component 8's experimental categorical join.

    ``as_of_rule`` and ``feature_definition_version_unchanged`` are recorded in every copy
    because they are the two claims this artifact most needs to travel with: the values
    are carried from strictly earlier inspections, and Component 4's contract was not
    modified to produce them. See ADR 0022.
    """

    component: str = "neural_categoricals"
    code_version: str
    neural_definition_version: str
    built_at: str

    features_path: str
    features_sha256: str
    feature_definition_version: str
    raw_path: str
    raw_sha256: str
    assignments_path: str
    assignments_sha256: str

    polars_version: str

    as_of_rule: str
    experimental_status: str
    families: list[str]
    cardinality: dict[str, int]
    coverage: dict[str, float]
    rows: int
    rows_without_prior_inspection: int

    blocked: list[str]
    artifacts: list[ArtifactRecord]
    checks: list[dict[str, str]]


class NeuralModelManifest(BaseModel):
    """Self-contained provenance and QA record for one neural training run.

    Records the device, the thread count and the torch version alongside the usual
    checksums, because a network's numbers depend on all three. The determinism claim is
    the same narrow one Components 6 and 7 make -- identical predictions for a fixed
    input, a fixed row order, a fixed library set and a single thread -- and Component 8
    additionally *measures* the residual through a multi-seed run rather than asserting
    it away.
    """

    component: str = "neural_models"
    code_version: str
    neural_definition_version: str
    built_at: str

    features_path: str
    features_sha256: str
    feature_definition_version: str
    categoricals_path: str
    categoricals_sha256: str

    torch_version: str
    numpy_version: str
    sklearn_version: str
    xgboost_version: str
    device: str
    torch_threads: int
    blas_threads: str
    deterministic_algorithms: bool
    determinism_caveat: str

    score_direction: str
    trained_through_semantics: str
    probability_semantics: str
    preprocessing: str
    missingness_semantics: str
    unknown_category_semantics: str
    matrix_columns: list[str]

    architecture: str
    embedding_dims: dict[str, int]
    hidden_sizes: list[int]
    dropout: float
    batch_size: int
    optimizer: str
    scheduler: str
    loss: str
    max_epochs: int
    early_stopping_patience: int
    gradient_clip_norm: float
    weight_decay: float
    learning_rates: dict[str, float]
    tuned_hyperparams_provenance: str

    fold_sets: dict[str, int]
    folds: int
    models: list[str]
    model_experiments: dict[str, str]
    model_feature_counts: dict[str, int]
    model_entity_columns: dict[str, list[str]]
    model_encodings: dict[str, str]
    model_parameter_counts: dict[str, int]
    seeds: dict[str, int]
    vocabulary_sizes: dict[str, int]
    best_epochs: dict[str, int]

    feature_rows: int
    fits: int
    epochs_total: int
    prediction_rows: int
    epoch_log_rows: int
    embedding_rows: int
    fit_seconds_total: float

    blocked: list[str]
    artifacts: list[ArtifactRecord]
    checks: list[dict[str, str]]


class NeuralTuningManifest(BaseModel):
    """Provenance for one learning-rate search.

    ``region_start`` / ``region_end`` are the point of this record: they are the proof
    that the search could not have seen a test window. A reader checking the claim
    compares ``region_end`` against the fold set's first ``test_start``, both recorded
    here.
    """

    component: str = "neural_tuning"
    code_version: str
    neural_definition_version: str
    built_at: str

    features_path: str
    features_sha256: str
    feature_definition_version: str
    categoricals_path: str
    categoricals_sha256: str

    torch_version: str
    numpy_version: str
    device: str
    torch_threads: int
    blas_threads: str

    objective: str
    search: str
    seed: int
    grid: list[float]
    max_epochs: int
    early_stopping_patience: int

    studies: list[dict[str, str]]
    tuning_regions: dict[str, str]
    first_test_start: dict[str, str]
    inner_folds: dict[str, list[str]]
    best_learning_rate: dict[str, float]
    selection_reasons: dict[str, str]
    mean_pr_auc: dict[str, float]

    feature_rows: int
    trial_rows: int
    seconds_total: float

    blocked: list[str]
    artifacts: list[ArtifactRecord]
    checks: list[dict[str, str]]


__all__ = [
    "ArtifactRecord",
    "EmbeddingTable",
    "EpochRecord",
    "FittedNetwork",
    "NeuralCategoricalsManifest",
    "NeuralModelManifest",
    "NeuralStats",
    "NeuralTuningManifest",
    "SeedRun",
    "SweepPoint",
    "SweepResult",
    "ValidationCheck",
]
