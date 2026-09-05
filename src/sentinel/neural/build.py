"""Orchestration: a feature table in, categoricals, sweep trials or predictions out.

The only module in the package that touches the filesystem, so everything it calls stays a
pure function of its inputs.

Three commands live here and they are deliberately separate.

``build_neural_categoricals`` produces Component 8's experimental categorical layer. It is
its own command, and its own artifact under its own directory, because it is the one part
of this component that reaches outside Component 4's contract. Making it a step a human
runs and can inspect -- rather than a silent join inside training -- is the point. ADR 0022.

``tune_neural`` runs the learning-rate sweep and writes a trials table. It changes no
source file and freezes nothing by itself: it prints the block a human pastes into
``definitions.TUNED_HYPERPARAMS``. That manual step is the design, not an omission -- a
value loaded from disk at training time could change without a diff, and the entire value
of freezing is that it cannot.

``train_neural`` fits one network per model per fold and writes the prediction artifact.
Like Components 6 and 7 it produces **no metrics**: Component 5 evaluates, Component 8
predicts. Computing a second set of numbers here would create two answers to every
question and put the test window within reach of a component allowed to fit things. The
predictions are handed over by artifact: ``sentinel evaluate --predictions <path>``.

**One model per fold, refitted from scratch.** Fold N's network is a fresh fit with freshly
initialised weights and a freshly fitted vocabulary, not fold N-1's network with more data.
That is what a health department retraining quarterly actually does, and for an embedding
model it is also the only construction that keeps the vocabulary honest.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import sklearn
import torch

from sentinel import __version__
from sentinel.config import Settings
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation import metrics
from sentinel.evaluation.models import SCORE_DIRECTION, FoldSpec
from sentinel.manifest import compute_sha256, manifest_path_for, write_manifest
from sentinel.modeling.train import training_frame
from sentinel.neural import (
    categoricals as categoricals_module,
)
from sentinel.neural import (
    embed,
    figures,
    net,
    predict,
    train,
    tuning,
    validate,
    writer,
)
from sentinel.neural.definitions import (
    BATCH_SIZE,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EMBEDDING_DIMS,
    EMBEDDING_DONOR,
    GRADIENT_CLIP_NORM,
    HIDDEN_SIZES,
    LEARNING_RATE_GRID,
    LOSS,
    MAX_EPOCHS,
    NEURAL_DEFINITION_VERSION,
    NEURAL_REGISTRY,
    OPTIMIZER,
    REPRESENTATIVE_MODEL,
    SCHEDULER,
    SEED_SWEEP,
    TUNED_HYPERPARAMS_PROVENANCE,
    TUNING_SEED,
    WEIGHT_DECAY,
    CategoricalEncoding,
    Learner,
    NeuralSpec,
    learning_rate_for,
    spec_for,
)
from sentinel.neural.models import (
    ArtifactRecord,
    FittedEmbeddingBooster,
    FittedNetwork,
    NeuralCategoricalsManifest,
    NeuralModelManifest,
    NeuralStats,
    NeuralTuningManifest,
    SweepResult,
    ValidationCheck,
)

logger = logging.getLogger(__name__)

PREDICTIONS_SLUG = "neural_predictions"
CATEGORICALS_SLUG = "neural_categoricals"
SWEEP_SLUG = "neural_sweep_trials"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

#: Where the figures land. Documentation artifacts rather than data, so they sit beside
#: the findings document that reads them.
FIGURES_DIR = Path("docs/analysis/figures")

#: Stated in the manifest so a consumer never infers it from the code.
TRAINED_THROUGH_SEMANTICS = (
    "fold.train_end -- the last reference date the fit was allowed to learn from. "
    "Component 8 early-stops, which Components 6 and 7 did not, but it stops against a "
    "window carved from the END OF THE TRAINING DATA (the last ~15% of training days), "
    "never the fold's calibration or test window. So no date later than train_end "
    "influenced the weights, the scaler, the vocabularies or the stopping epoch. The "
    "learning rate came from a search confined to a region strictly earlier than any test "
    "window. ADR 0014, ADR 0017, ADR 0021."
)

PROBABILITY_SEMANTICS = (
    "RAW sigmoid of the network's output logit, NOT a calibrated probability. A network "
    "trained under BCE is typically overconfident, and a sigmoid over an unbounded logit "
    "saturates more readily than a penalised GLM's link. Component 8 measures that through "
    "Component 5's ECE and MCE and corrects none of it. Component 9 owns calibration."
)

PREPROCESSING = (
    "median (numeric) or constant-0 (boolean) imputation then standardisation, both "
    "fitted on the inner training rows only, plus the four null-rule family indicators. "
    "Identical rules to Component 6, because the justification for them is about the data "
    "rather than about logistic regression. This is the one place Component 8 differs "
    "materially from Component 7, which imputes nothing and routes NaN to a learned split "
    "direction: a dense layer has no such option, so the network imputes and the family "
    "indicator is how the fact of missingness survives."
)

MISSINGNESS_SEMANTICS = (
    "A NULL in Component 4's table means 'this could not be known as of the decision "
    "point' -- e.g. prior_canvass_fail_rate is NULL because there was no prior canvass. "
    "For the network that fact is carried by the null-rule family indicator, which is a "
    "declared column present in every matrix, while the imputed value keeps the column "
    "numerically usable. The semantics are not changed from Component 6; what changes is "
    "that a network can learn an interaction between an indicator and its imputed column, "
    "which is the open question Component 6's findings recorded and could not answer."
)

UNKNOWN_CATEGORY_SEMANTICS = (
    "Index 0 of every embedding table is __UNKNOWN__ and its vector is LEARNED, not "
    "masked. Genuine unknowns exist in training -- 401 rows have no prior inspection of "
    "any type to carry a categorical forward from -- so index 0 receives gradient and "
    "learns the 'never seen before' offset. A category appearing only after train_end maps "
    "to that same row, which is the honest default: the model has no basis to say anything "
    "else about it. Vocabularies are refitted per fold on training rows only."
)

DETERMINISM_CAVEAT = (
    "Bit-identical re-runs are claimed ONLY for: this feature table, this row order, this "
    "library set, one torch thread, and CPU. torch.use_deterministic_algorithms(True) is "
    "set and torch.set_num_threads(1) is pinned. A CUDA device is present on the build "
    "machine and deliberately unused -- GPU reductions are not bit-reproducible, and this "
    "project's standard for 'did not move' is bit-identity. A library version bump may "
    "move every number. Residual run-to-run variation is MEASURED by the multi-seed "
    "experiment rather than asserted away. See ADR 0020."
)

REQUIRED_COLUMNS = (
    "establishment_id",
    "inspection_date",
    "target_inspection_id",
    "target",
    "feature_definition_version",
)

#: Experiments the current data cannot support, or that belong to a later component.
#: Reported, never faked.
BLOCKED_EXPERIMENTS: tuple[str, ...] = (
    "an establishment_id embedding: refused, not blocked by the data. It is the obvious "
    "thing for this component to learn and it is the largest leakage surface in the "
    "project -- a per-establishment parameter carries whatever the network learned about "
    "that establishment from every row it appeared in. Component 4 excludes identity by "
    "design and modeling.definitions.FORBIDDEN_COLUMNS enforces it. `chain` is the "
    "deliberate lower-cardinality substitute, derived per fold from training rows only. "
    "See ADR 0021",
    "inspector-effect modelling: the Chicago food inspections dataset publishes 22 "
    "columns and none identifies an inspector, so an inspector embedding -- the thing an "
    "entity-embedding component would most naturally add -- is undefined. Unchanged from "
    "Component 7. See ADR 0019",
    "probability calibration: neural probabilities are emitted raw and uncorrected. "
    "Temperature scaling, Platt scaling, isotonic regression and any operating threshold "
    "belong to Component 9",
    "demographic features: no race, income or ACS variable is used anywhere. Community "
    "area is included ONLY as an explicitly labelled experimental embedding with a "
    "matched ablation, and a better score with it is not grounds for retaining it. The "
    "fairness audit is Component 12's. See ADR 0023",
    "SHAP or any attribution over the learned representation: Component 11's. The "
    "embedding tables emitted here are a representation, not an explanation",
    "ensembling the network with Components 6 and 7: comes after the individual model "
    "stages, so that it is clear which member carries a result",
)


class NeuralBuildError(RuntimeError):
    """Raised when a Component 8 build cannot proceed."""


@dataclass
class CategoricalsResult:
    """What ``build_neural_categoricals`` produced."""

    table: pl.DataFrame
    checks: list[ValidationCheck]
    manifest: NeuralCategoricalsManifest
    categoricals_path: Path | None = None
    manifest_path: Path | None = None


@dataclass
class NeuralTuningResult:
    """What ``tune_neural`` produced."""

    tables: dict[str, pl.DataFrame]
    checks: list[ValidationCheck]
    manifest: NeuralTuningManifest
    results: list[SweepResult] = field(default_factory=list)
    trials_path: Path | None = None
    manifest_path: Path | None = None


@dataclass
class NeuralResult:
    """What ``train_neural`` produced."""

    tables: dict[str, pl.DataFrame]
    checks: list[ValidationCheck]
    manifest: NeuralModelManifest
    folds: list[FoldSpec] = field(default_factory=list)
    fitted: list[FittedNetwork] = field(default_factory=list)
    boosted: list[FittedEmbeddingBooster] = field(default_factory=list)
    figure_paths: list[Path] = field(default_factory=list)
    predictions_path: Path | None = None
    manifest_path: Path | None = None


# --- 1. the experimental categorical layer -----------------------------------


def build_neural_categoricals(
    settings: Settings,
    *,
    features_path: Path,
    raw_path: Path,
    assignments_path: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> CategoricalsResult:
    """Carry chain, facility type, community area and zip forward, strictly as-of."""
    if not features_path.exists():
        raise FileNotFoundError(f"Component 4 features not found: {features_path}")

    started = datetime.now(UTC)
    features = pl.read_parquet(features_path)
    history = categoricals_module.load_history(raw_path, assignments_path)
    table = categoricals_module.build_categoricals(features, history)
    table = writer.finalize(table.to_dicts(), CATEGORICALS_SLUG)

    checks = validate.validate_categoricals(features, table)
    without_prior = int(table.filter(pl.col("source_inspection_id").is_null()).height)

    destination = output_dir or settings.neural_processed_dir
    stamp = started.strftime(TIMESTAMP_FORMAT)
    artifacts: list[ArtifactRecord] = []
    categoricals_path: Path | None = None
    if not dry_run:
        categoricals_path = destination / f"{CATEGORICALS_SLUG}_{stamp}.parquet"
        writer.write_table(table, categoricals_path)
        artifacts.append(_artifact_record(categoricals_path, table))

    manifest = NeuralCategoricalsManifest(
        code_version=__version__,
        neural_definition_version=NEURAL_DEFINITION_VERSION,
        built_at=started.isoformat(),
        features_path=features_path.name,
        features_sha256=compute_sha256(features_path),
        feature_definition_version=str(features["feature_definition_version"][0]),
        raw_path=raw_path.name,
        raw_sha256=compute_sha256(raw_path),
        assignments_path=assignments_path.name,
        assignments_sha256=compute_sha256(assignments_path),
        polars_version=pl.__version__,
        as_of_rule=(
            "each categorical is the value recorded at the establishment's most recent "
            "inspection of ANY type STRICTLY BEFORE the row's own inspection_date. The "
            "target row never supplies its own attributes. Rows with no prior inspection "
            "get __UNKNOWN__, which is a real category with a learned embedding row."
        ),
        experimental_status=(
            "EXPERIMENTAL, Component 8 only. This is NOT a Component 4 feature table and "
            "no column here is a feature. feature_definition_version is unchanged at v1 "
            "and Component 4's contract was not modified to produce this. Nothing here "
            "may be joined onto a feature table for any other component. See ADR 0022."
        ),
        families=list(categoricals_module.EMITTED_CATEGORICALS),
        cardinality=categoricals_module.cardinality(table),
        coverage={k: round(v, 6) for k, v in categoricals_module.coverage(table).items()},
        rows=table.height,
        rows_without_prior_inspection=without_prior,
        blocked=list(BLOCKED_EXPERIMENTS),
        artifacts=artifacts,
        checks=_render_checks(checks),
    )

    manifest_path: Path | None = None
    if not dry_run and categoricals_path is not None:
        manifest_path = manifest_path_for(categoricals_path)
        write_manifest(manifest, manifest_path)

    logger.info(
        "Built %d categorical rows in %.1fs (%d without any prior inspection)",
        table.height,
        (datetime.now(UTC) - started).total_seconds(),
        without_prior,
    )
    return CategoricalsResult(
        table=table,
        checks=checks,
        manifest=manifest,
        categoricals_path=categoricals_path,
        manifest_path=manifest_path,
    )


# --- 2. the learning-rate sweep ----------------------------------------------


def tune_neural(
    settings: Settings,
    *,
    features_path: Path,
    categoricals_path: Path,
    output_dir: Path | None = None,
    fold_sets: Sequence[str] | None = None,
    grid: Sequence[float] = LEARNING_RATE_GRID,
    seed: int = TUNING_SEED,
    dry_run: bool = False,
) -> NeuralTuningResult:
    """Search the learning rate for each fold set, strictly before any test window."""
    started = datetime.now(UTC)
    frame = _load(features_path)
    categoricals = _load_categoricals(categoricals_path, frame)
    all_folds = _build_folds(frame)
    chosen = _resolve_fold_sets(fold_sets, all_folds)
    spec = spec_for(REPRESENTATIVE_MODEL)

    results: list[SweepResult] = []
    for fold_set in chosen:
        results.append(
            tuning.sweep_fold_set(
                spec,
                frame,
                all_folds,
                fold_set=fold_set,
                categoricals=categoricals,
                grid=grid,
                seed=seed,
            )
        )

    trial_rows = [row for result in results for row in _sweep_rows(result)]
    tables = {SWEEP_SLUG: writer.finalize(trial_rows, SWEEP_SLUG)}
    checks = validate.validate_sweep(results, all_folds)

    destination = output_dir or settings.tuning_processed_dir
    stamp = started.strftime(TIMESTAMP_FORMAT)
    artifacts: list[ArtifactRecord] = []
    trials_path: Path | None = None
    if not dry_run:
        trials_path = destination / f"{SWEEP_SLUG}_{stamp}.parquet"
        writer.write_table(tables[SWEEP_SLUG], trials_path)
        artifacts.append(_artifact_record(trials_path, tables[SWEEP_SLUG]))

    elapsed = (datetime.now(UTC) - started).total_seconds()
    manifest = NeuralTuningManifest(
        code_version=__version__,
        neural_definition_version=NEURAL_DEFINITION_VERSION,
        built_at=started.isoformat(),
        features_path=features_path.name,
        features_sha256=compute_sha256(features_path),
        feature_definition_version=str(frame["feature_definition_version"][0]),
        categoricals_path=categoricals_path.name,
        categoricals_sha256=compute_sha256(categoricals_path),
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        device=net.device_name(),
        torch_threads=torch.get_num_threads(),
        blas_threads=_blas_threads(),
        objective=tuning.OBJECTIVE,
        search=tuning.SEARCH,
        seed=seed,
        grid=[float(v) for v in grid],
        max_epochs=MAX_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        studies=[
            {
                "study": r.study,
                "fold_set": r.fold_set,
                "region": f"{r.region_start}..{r.region_end}",
                "inner_folds": str(len(r.inner_folds)),
                "best_learning_rate": f"{r.best_learning_rate:g}",
            }
            for r in results
        ],
        tuning_regions={r.fold_set: f"{r.region_start}..{r.region_end}" for r in results},
        first_test_start={
            fs: str(min(f.test_start for f in all_folds if f.fold_set == fs)) for fs in chosen
        },
        inner_folds={r.fold_set: list(r.inner_folds) for r in results},
        best_learning_rate={r.fold_set: r.best_learning_rate for r in results},
        selection_reasons={r.fold_set: r.selection_reason for r in results},
        mean_pr_auc={
            f"{r.fold_set}/lr={rate:g}": round(value, 6)
            for r in results
            for rate, value in r.scores
        },
        feature_rows=frame.height,
        trial_rows=tables[SWEEP_SLUG].height,
        seconds_total=round(elapsed, 3),
        blocked=list(BLOCKED_EXPERIMENTS),
        artifacts=artifacts,
        checks=_render_checks(checks),
    )

    manifest_path: Path | None = None
    if not dry_run and trials_path is not None:
        manifest_path = manifest_path_for(trials_path)
        write_manifest(manifest, manifest_path)

    logger.info("Swept %d fold set(s) in %.1fs", len(results), elapsed)
    return NeuralTuningResult(
        tables=tables,
        checks=checks,
        manifest=manifest,
        results=results,
        trials_path=trials_path,
        manifest_path=manifest_path,
    )


# --- 3. training -------------------------------------------------------------


def train_neural(
    settings: Settings,
    *,
    features_path: Path,
    categoricals_path: Path,
    output_dir: Path | None = None,
    models: Sequence[str] | None = None,
    seed_sweep: bool = True,
    render_figures: bool = True,
    figures_dir: Path | None = None,
    dry_run: bool = False,
) -> NeuralResult:
    """Fit every requested model on every fold and write the prediction artifacts."""
    if not features_path.exists():
        raise FileNotFoundError(f"Component 4 features not found: {features_path}")

    started = datetime.now(UTC)
    specs = _resolve_specs(models)
    frame = _load(features_path)
    categoricals = _load_categoricals(categoricals_path, frame)
    all_folds = _build_folds(frame)

    stats = NeuralStats(feature_rows=frame.height, folds=len(all_folds))
    stats.models = [s.name for s in specs]
    for fold in all_folds:
        stats.fold_sets[fold.fold_set] = stats.fold_sets.get(fold.fold_set, 0) + 1

    networks = [s for s in specs if s.learner is Learner.MLP]
    boosters = [s for s in specs if s.learner is Learner.XGBOOST_EMBEDDING]
    _check_donors(boosters, networks)

    fitted: list[FittedNetwork] = []
    boosted: list[FittedEmbeddingBooster] = []
    prediction_rows: list[dict[str, object]] = []
    epoch_rows: list[dict[str, object]] = []
    embedding_rows: list[dict[str, object]] = []
    log_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []

    for fold in all_folds:
        training = training_frame(frame, fold)
        test = folds_module.window_frame(frame, fold)
        if test.height == 0:
            logger.warning("%s: test window is empty, skipping", fold.fold_id)
            continue

        by_name: dict[str, FittedNetwork] = {}
        for spec in networks:
            fit_started = datetime.now(UTC)
            model = train.fit_fold(spec, training, fold, categoricals=categoricals)
            stats.fit_seconds_total += (datetime.now(UTC) - fit_started).total_seconds()
            stats.fits += 1
            stats.epochs_total += model.final_epoch
            fitted.append(model)
            by_name[spec.name] = model

            ids, scores = predict.score_window(model, test, categoricals=categoricals)
            prediction_rows.extend(
                _prediction_rows(
                    spec, model.fold_set, model.fold_id, model.trained_through, ids, scores
                )
            )
            epoch_rows.extend(_epoch_rows(model))
            log_rows.append(_log_row(model, fold, test_rows=test.height, scores=scores))
            if spec.name == REPRESENTATIVE_MODEL:
                embedding_rows.extend(
                    {**row, "neural_definition_version": NEURAL_DEFINITION_VERSION}
                    for row in embed.embedding_rows(model)
                )

        for spec in boosters:
            donor = by_name[EMBEDDING_DONOR[spec.name]]
            fit_started = datetime.now(UTC)
            booster = embed.fit_fold(spec, training, fold, donor=donor, categoricals=categoricals)
            stats.fit_seconds_total += (datetime.now(UTC) - fit_started).total_seconds()
            stats.fits += 1
            boosted.append(booster)
            ids, scores = embed.score_window(booster, test, donor=donor, categoricals=categoricals)
            prediction_rows.extend(
                _prediction_rows(
                    spec, booster.fold_set, booster.fold_id, booster.trained_through, ids, scores
                )
            )
            log_rows.append(_booster_log_row(booster, fold, test_rows=test.height, scores=scores))

        if seed_sweep and REPRESENTATIVE_MODEL in by_name:
            seed_rows.extend(
                _seed_rows(
                    by_name[REPRESENTATIVE_MODEL],
                    spec_for(REPRESENTATIVE_MODEL),
                    training,
                    test,
                    fold,
                    categoricals,
                    stats,
                )
            )

    if not prediction_rows:
        raise NeuralBuildError(
            "no predictions were produced. Every fold had an empty test window, which "
            "means the fold set and the feature table disagree about the data range."
        )

    tables = {
        "neural_predictions": writer.finalize(prediction_rows, "neural_predictions"),
        "neural_training_log": writer.finalize(log_rows, "neural_training_log"),
        "neural_epoch_log": writer.finalize(epoch_rows, "neural_epoch_log"),
        "neural_embeddings": writer.finalize(embedding_rows, "neural_embeddings"),
        "neural_seed_variation": writer.finalize(seed_rows, "neural_seed_variation"),
    }
    stats.prediction_rows = tables["neural_predictions"].height
    stats.epoch_log_rows = tables["neural_epoch_log"].height
    stats.embedding_rows = tables["neural_embeddings"].height
    stats.training_log_rows = tables["neural_training_log"].height

    checks = validate.validate_neural(
        frame,
        all_folds,
        fitted,
        boosted,
        tables["neural_predictions"],
        categoricals,
        expected_models=[s.name for s in specs],
    )

    figure_paths: list[Path] = []
    if render_figures and not dry_run:
        figure_paths = _render_figures(fitted, categoricals, figures_dir or FIGURES_DIR)

    destination = output_dir or settings.predictions_processed_dir
    stamp = started.strftime(TIMESTAMP_FORMAT)
    artifacts: list[ArtifactRecord] = []
    predictions_path: Path | None = None
    if not dry_run:
        for name, table in tables.items():
            path = destination / f"{name}_{stamp}.parquet"
            writer.write_table(table, path)
            if name == PREDICTIONS_SLUG:
                predictions_path = path
            artifacts.append(_artifact_record(path, table))

    manifest = _model_manifest(
        started=started,
        features_path=features_path,
        categoricals_path=categoricals_path,
        frame=frame,
        specs=specs,
        fitted=fitted,
        stats=stats,
        all_folds=all_folds,
        artifacts=artifacts,
        checks=checks,
    )

    manifest_path: Path | None = None
    if not dry_run and predictions_path is not None:
        manifest_path = manifest_path_for(predictions_path)
        write_manifest(manifest, manifest_path)

    logger.info(
        "Trained %d fit(s) over %d folds in %.1fs (%.1fs fitting, %d epochs)",
        stats.fits,
        len(all_folds),
        (datetime.now(UTC) - started).total_seconds(),
        stats.fit_seconds_total,
        stats.epochs_total,
    )
    return NeuralResult(
        tables=tables,
        checks=checks,
        manifest=manifest,
        folds=all_folds,
        fitted=fitted,
        boosted=boosted,
        figure_paths=figure_paths,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
    )


# --- 4. inputs ---------------------------------------------------------------


def _resolve_specs(models: Sequence[str] | None) -> list[NeuralSpec]:
    """Look up the requested models, or every registered one.

    Unknown names fail here rather than producing a partial run, so a typo cannot quietly
    halve the portfolio.
    """
    if models is None:
        return list(NEURAL_REGISTRY)
    if not models:
        raise NeuralBuildError("no models requested")
    resolved: list[NeuralSpec] = []
    for name in models:
        try:
            resolved.append(spec_for(name))
        except KeyError as exc:
            raise NeuralBuildError(str(exc)) from exc
    return resolved


def _check_donors(boosters: Sequence[NeuralSpec], networks: Sequence[NeuralSpec]) -> None:
    """An embedding-fed booster cannot run without the network that supplies its vectors."""
    available = {s.name for s in networks}
    for spec in boosters:
        donor = EMBEDDING_DONOR[spec.name]
        if donor not in available:
            raise NeuralBuildError(
                f"{spec.name} consumes embeddings from {donor}, which was not requested. "
                f"Add --models {donor}, or drop {spec.name}. The vectors are never taken "
                "from a different fold or a cached fit."
            )


def _resolve_fold_sets(requested: Sequence[str] | None, folds: Sequence[FoldSpec]) -> list[str]:
    present = sorted({f.fold_set for f in folds})
    if requested is None:
        return present
    if not requested:
        raise NeuralBuildError("no fold sets requested")
    unknown = sorted(set(requested) - set(present))
    if unknown:
        raise NeuralBuildError(
            f"unknown fold set(s): {', '.join(unknown)}. Present: {', '.join(present)}"
        )
    return [name for name in present if name in set(requested)]


def _load(features_path: Path) -> pl.DataFrame:
    """Read the feature table and parse its reference date once."""
    frame = pl.read_parquet(features_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise NeuralBuildError(f"Feature table is missing required columns: {', '.join(missing)}")
    if frame.height == 0:
        raise NeuralBuildError("feature table is empty")
    duplicated = frame.height - frame["target_inspection_id"].n_unique()
    if duplicated:
        raise NeuralBuildError(
            f"feature table has {duplicated} duplicated target_inspection_id value(s); the "
            "prediction contract requires one score per id and could not be met"
        )
    return frame.with_columns(pl.col("inspection_date").str.to_date().alias("rd"))


def _load_categoricals(path: Path, frame: pl.DataFrame) -> pl.DataFrame:
    """Read the experimental categorical layer and check it covers the feature table."""
    if not path.exists():
        raise FileNotFoundError(
            f"Component 8 categoricals not found: {path}. Run "
            "`sentinel build-neural-categoricals` first."
        )
    table = pl.read_parquet(path)
    expected = {str(v) for v in frame["target_inspection_id"].to_list()}
    got = {str(v) for v in table["target_inspection_id"].to_list()}
    if not expected.issubset(got):
        raise NeuralBuildError(
            f"the categorical table covers {len(got)} rows but the feature table has "
            f"{len(expected)}; {len(expected - got)} row(s) would have no categoricals. "
            "Rebuild it against this feature table."
        )
    return table


def _build_folds(frame: pl.DataFrame) -> list[FoldSpec]:
    """Component 5's fold set, rebuilt from the data. Never invented here.

    Includes ``covid_shift`` for the same reason Components 6 and 7 do: it is a separate
    fold set precisely so it cannot be averaged into the headline, and all three earlier
    components measured the model ordering reversing on it.
    """
    data_start = folds_module.min_date(frame, "rd")
    data_end = folds_module.max_date(frame, "rd")
    if data_start is None or data_end is None:
        raise NeuralBuildError("feature table has no usable reference dates")
    quarterly = folds_module.quarterly_folds(data_start=data_start, data_end=data_end)
    if not quarterly:
        raise NeuralBuildError(
            f"data spans {data_start}..{data_end}, which is too short to build a single "
            "quarterly fold. Folds are never fabricated."
        )
    return [*quarterly, *folds_module.covid_shift_fold(data_end=data_end)]


# --- 5. row builders ---------------------------------------------------------


def _prediction_rows(
    spec: NeuralSpec,
    fold_set: str,
    fold_id: str,
    trained_through: object,
    ids: Sequence[str],
    scores: Sequence[float],
) -> list[dict[str, object]]:
    return [
        {
            "target_inspection_id": row_id,
            "score": score,
            "model_name": spec.name,
            "model_version": spec.version,
            "fold_set": fold_set,
            "fold_id": fold_id,
            "trained_through": trained_through,
            "is_probability": spec.is_probability,
            "neural_definition_version": NEURAL_DEFINITION_VERSION,
        }
        for row_id, score in zip(ids, scores, strict=True)
    ]


def _epoch_rows(model: FittedNetwork) -> list[dict[str, object]]:
    return [
        {
            "model_name": model.spec.name,
            "fold_set": model.fold_set,
            "fold_id": model.fold_id,
            "seed": model.seed,
            "epoch": record.epoch,
            "train_loss": record.train_loss,
            "validation_loss": record.validation_loss,
            "learning_rate": record.learning_rate,
            "is_best_epoch": record.epoch == model.best_epoch,
            "neural_definition_version": NEURAL_DEFINITION_VERSION,
        }
        for record in model.epochs
    ]


def _log_row(
    model: FittedNetwork, fold: FoldSpec, *, test_rows: int, scores: Sequence[float]
) -> dict[str, object]:
    return {
        "model_name": model.spec.name,
        "model_version": model.spec.version,
        "learner": model.spec.learner.value,
        "encoding": model.spec.encoding.value,
        "experiment": model.spec.experiment,
        "fold_set": model.fold_set,
        "fold_id": model.fold_id,
        "train_start": model.train_start,
        "train_end": model.train_end,
        "trained_through": model.trained_through,
        "calibration_end_unused": model.calibration_end_unused,
        "inner_validation_start": model.inner_validation_start,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "train_rows": model.train_rows,
        "inner_train_rows": model.inner_train_rows,
        "inner_validation_rows": model.inner_validation_rows,
        "test_rows": test_rows,
        "feature_count": len(model.spec.feature_columns),
        "entity_count": len(model.spec.entity_columns),
        "dense_width": model.dense_width,
        "embedding_width": model.embedding_width,
        "parameter_count": model.parameter_count,
        "vocabulary_total": sum(model.encoding.sizes.values()),
        "train_nan_cells": model.train_nan_cells,
        "train_positive_rate": model.train_positive_rate,
        "seed": model.seed,
        "learning_rate": model.learning_rate,
        "pos_weight": model.pos_weight,
        "best_epoch": model.best_epoch,
        "final_epoch": model.final_epoch,
        "learning_rate_changes": model.learning_rate_changes,
        "stop_reason": model.stop_reason,
        "saturated_scores": predict.saturated_count(list(scores)),
        "neural_definition_version": NEURAL_DEFINITION_VERSION,
    }


def _booster_log_row(
    booster: FittedEmbeddingBooster,
    fold: FoldSpec,
    *,
    test_rows: int,
    scores: Sequence[float],
) -> dict[str, object]:
    """The embedding-fed booster's log row, in the same schema as a network's.

    Fields that have no meaning for a booster are null rather than zero: a zero best epoch
    would read as "peaked immediately" and a zero parameter count as "no model".
    """
    return {
        "model_name": booster.spec.name,
        "model_version": booster.spec.version,
        "learner": booster.spec.learner.value,
        "encoding": booster.spec.encoding.value,
        "experiment": booster.spec.experiment,
        "fold_set": booster.fold_set,
        "fold_id": booster.fold_id,
        "train_start": booster.train_start,
        "train_end": booster.train_end,
        "trained_through": booster.trained_through,
        "calibration_end_unused": booster.calibration_end_unused,
        "inner_validation_start": None,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "train_rows": booster.train_rows,
        "inner_train_rows": None,
        "inner_validation_rows": None,
        "test_rows": test_rows,
        "feature_count": len(booster.spec.feature_columns),
        "entity_count": len(booster.spec.entity_columns),
        "dense_width": len(booster.matrix_columns),
        "embedding_width": len(booster.embedding_columns),
        "parameter_count": None,
        "vocabulary_total": None,
        "train_nan_cells": booster.train_nan_cells,
        "train_positive_rate": booster.train_positive_rate,
        "seed": booster.seed,
        "learning_rate": None,
        "pos_weight": None,
        "best_epoch": None,
        "final_epoch": None,
        "learning_rate_changes": None,
        "stop_reason": f"n/a -- XGBoost, {booster.n_estimators} frozen rounds",
        "saturated_scores": predict.saturated_count(list(scores)),
        "neural_definition_version": NEURAL_DEFINITION_VERSION,
    }


def _seed_rows(
    reference: FittedNetwork,
    spec: NeuralSpec,
    training: pl.DataFrame,
    test: pl.DataFrame,
    fold: FoldSpec,
    categoricals: pl.DataFrame,
    stats: NeuralStats,
) -> list[dict[str, object]]:
    """Refit the representative model under every seed and record the spread.

    Seed 42's fit is reused rather than repeated -- it is the fit that produced this
    fold's predictions, so repeating it would measure the same computation twice and
    understate nothing but waste minutes.
    """
    labels = [int(v) for v in test["target"].to_list()]
    rows: list[dict[str, object]] = []

    def record(model: FittedNetwork, scores: Sequence[float]) -> None:
        pr = metrics.pr_auc(labels, list(scores))
        roc = metrics.roc_auc(labels, list(scores))
        rows.append(
            {
                "model_name": spec.name,
                "fold_set": fold.fold_set,
                "fold_id": fold.fold_id,
                "seed": model.seed,
                "pr_auc": pr,
                "roc_auc": roc,
                "best_epoch": model.best_epoch,
                "neural_definition_version": NEURAL_DEFINITION_VERSION,
            }
        )

    _, reference_scores = predict.score_window(reference, test, categoricals=categoricals)
    record(reference, reference_scores)

    for seed in SEED_SWEEP:
        if seed == reference.seed:
            continue
        model = train.fit_fold(spec, training, fold, categoricals=categoricals, seed=seed)
        stats.fits += 1
        stats.epochs_total += model.final_epoch
        _, scores = predict.score_window(model, test, categoricals=categoricals)
        record(model, scores)
    return rows


def _sweep_rows(result: SweepResult) -> list[dict[str, object]]:
    return [
        {
            "study": result.study,
            "model_name": result.model_name,
            "fold_set": result.fold_set,
            "inner_fold_id": point.fold_id,
            "learning_rate": point.learning_rate,
            "pr_auc": point.pr_auc,
            "best_epoch": point.best_epoch,
            "train_rows": point.train_rows,
            "validation_rows": point.validation_rows,
            "selected": point.learning_rate == result.best_learning_rate,
            "seed": result.seed,
            "region_start": result.region_start,
            "region_end": result.region_end,
            "neural_definition_version": NEURAL_DEFINITION_VERSION,
        }
        for point in result.points
    ]


# --- 6. figures --------------------------------------------------------------


def _render_figures(
    fitted: Sequence[FittedNetwork], categoricals: pl.DataFrame, directory: Path
) -> list[Path]:
    """Learning curves and the chain projection for the representative fold.

    The representative fold is the **last quarterly** one: it has the widest training
    window and the largest vocabulary, so it is the most informative single picture. Which
    fold was used is written into the filename so a reader is never guessing.
    """
    candidates = [
        m
        for m in fitted
        if m.spec.name == REPRESENTATIVE_MODEL and m.fold_set == folds_module.QUARTERLY
    ]
    if not candidates:
        logger.warning("no %s fit on a quarterly fold; skipping figures", REPRESENTATIVE_MODEL)
        return []
    model = max(candidates, key=lambda m: m.train_end)
    paths: list[Path] = []

    curve = directory / f"neural_learning_curve_{model.fold_id}.png"
    paths.append(figures.learning_curve_figure(model, curve))

    sizes = _chain_sizes(categoricals, model)
    projection = directory / f"neural_chain_embedding_tsne_{model.fold_id}.png"
    rendered = figures.embedding_figure(model, projection, family="chain", sizes=sizes)
    if rendered is not None:
        paths.append(rendered)
    else:
        logger.warning("chain embedding projection skipped: too few categories")
    return paths


def _chain_sizes(categoricals: pl.DataFrame, model: FittedNetwork) -> dict[str, int]:
    """Establishments per chain, counted in the fitting window only.

    Used for point size in the projection. Counted on the same rows the vocabulary was
    fitted on, because a count over the whole snapshot would be a fact about the future.
    """
    window = categoricals.filter(
        pl.col("inspection_date").str.to_date() < model.inner_validation_start
    )
    grouped = window.group_by("chain_key").agg(pl.col("establishment_id").n_unique().alias("n"))
    return {str(r["chain_key"]): int(r["n"]) for r in grouped.iter_rows(named=True)}


# --- 7. manifest helpers -----------------------------------------------------


def _model_manifest(
    *,
    started: datetime,
    features_path: Path,
    categoricals_path: Path,
    frame: pl.DataFrame,
    specs: Sequence[NeuralSpec],
    fitted: Sequence[FittedNetwork],
    stats: NeuralStats,
    all_folds: Sequence[FoldSpec],
    artifacts: Sequence[ArtifactRecord],
    checks: Sequence[ValidationCheck],
) -> NeuralModelManifest:
    networks = [m for m in fitted if m.spec.learner is Learner.MLP]
    reference = networks[0] if networks else None
    return NeuralModelManifest(
        code_version=__version__,
        neural_definition_version=NEURAL_DEFINITION_VERSION,
        built_at=started.isoformat(),
        features_path=features_path.name,
        features_sha256=compute_sha256(features_path),
        feature_definition_version=str(frame["feature_definition_version"][0]),
        categoricals_path=categoricals_path.name,
        categoricals_sha256=compute_sha256(categoricals_path),
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        sklearn_version=sklearn.__version__,
        xgboost_version=_library_version("xgboost"),
        device=net.device_name(),
        torch_threads=torch.get_num_threads(),
        blas_threads=_blas_threads(),
        deterministic_algorithms=True,
        determinism_caveat=DETERMINISM_CAVEAT,
        score_direction=SCORE_DIRECTION,
        trained_through_semantics=TRAINED_THROUGH_SEMANTICS,
        probability_semantics=PROBABILITY_SEMANTICS,
        preprocessing=PREPROCESSING,
        missingness_semantics=MISSINGNESS_SEMANTICS,
        unknown_category_semantics=UNKNOWN_CATEGORY_SEMANTICS,
        matrix_columns=list(reference.matrix_columns) if reference else [],
        architecture=(
            "embeddings || standardised numerics -> "
            + " -> ".join(
                f"Linear({h}) -> BatchNorm1d -> ReLU -> Dropout({DROPOUT})" for h in HIDDEN_SIZES
            )
            + " -> Linear(1) logit"
        ),
        embedding_dims={family.value: dim for family, dim in EMBEDDING_DIMS.items()},
        hidden_sizes=list(HIDDEN_SIZES),
        dropout=DROPOUT,
        batch_size=BATCH_SIZE,
        optimizer=OPTIMIZER,
        scheduler=SCHEDULER,
        loss=LOSS,
        max_epochs=MAX_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        gradient_clip_norm=GRADIENT_CLIP_NORM,
        weight_decay=WEIGHT_DECAY,
        learning_rates={
            fold_set: learning_rate_for(fold_set)
            for fold_set in sorted({f.fold_set for f in all_folds})
        },
        tuned_hyperparams_provenance=TUNED_HYPERPARAMS_PROVENANCE,
        fold_sets=stats.fold_sets,
        folds=len(all_folds),
        models=[s.name for s in specs],
        model_experiments={s.name: s.experiment for s in specs},
        model_feature_counts={s.name: len(s.feature_columns) for s in specs},
        model_entity_columns={s.name: list(s.entity_columns) for s in specs},
        model_encodings={s.name: s.encoding.value for s in specs},
        model_parameter_counts={m.spec.name: m.parameter_count for m in networks},
        seeds={s.name: s.seed for s in specs},
        vocabulary_sizes=_vocabulary_sizes(networks),
        best_epochs={f"{m.spec.name}/{m.fold_id}": m.best_epoch for m in networks},
        feature_rows=frame.height,
        fits=stats.fits,
        epochs_total=stats.epochs_total,
        prediction_rows=stats.prediction_rows,
        epoch_log_rows=stats.epoch_log_rows,
        embedding_rows=stats.embedding_rows,
        fit_seconds_total=round(stats.fit_seconds_total, 3),
        blocked=list(BLOCKED_EXPERIMENTS),
        artifacts=list(artifacts),
        checks=_render_checks(checks),
    )


def _vocabulary_sizes(networks: Sequence[FittedNetwork]) -> dict[str, int]:
    """Largest vocabulary observed per family, across every fold.

    The maximum rather than the mean, because it is the number that bounds the embedding
    table and therefore the parameter count.
    """
    out: dict[str, int] = {}
    for model in networks:
        if model.spec.encoding is CategoricalEncoding.NONE:
            continue
        for column, size in model.encoding.sizes.items():
            out[column] = max(out.get(column, 0), size)
    return out


def _artifact_record(path: Path, table: pl.DataFrame) -> ArtifactRecord:
    return ArtifactRecord(
        path=path.name,
        bytes=path.stat().st_size,
        sha256=compute_sha256(path),
        row_count=table.height,
        schema=writer.schema_of(table),
    )


def _render_checks(checks: Sequence[ValidationCheck]) -> list[dict[str, str]]:
    return [
        {
            "name": c.name,
            "severity": c.severity,
            "passed": str(c.passed),
            "detail": c.detail,
        }
        for c in checks
    ]


def _library_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except ImportError:  # pragma: no cover - both are hard dependencies
        return "not installed"
    return str(getattr(module, "__version__", "unknown"))


def _blas_threads() -> str:
    """What was actually in effect, not what we would prefer."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(name)
        if value:
            return f"{name}={value}"
    return "unset (library default)"


# --- 8. summaries ------------------------------------------------------------


def summarize(result: NeuralResult) -> str:
    """A short human-readable summary for the CLI."""
    manifest = result.manifest
    lines = [
        f"Component 8 neural models: {manifest.fits} fit(s) over {manifest.folds} fold(s)",
        f"  models        : {', '.join(manifest.models)}",
        f"  device        : {manifest.device} ({manifest.torch_threads} thread), "
        f"torch {manifest.torch_version}",
        f"  epochs        : {manifest.epochs_total} total",
        f"  predictions   : {manifest.prediction_rows} row(s)",
        f"  embeddings    : {manifest.embedding_rows} row(s)",
        f"  fitting time  : {manifest.fit_seconds_total:.1f}s",
    ]
    if manifest.vocabulary_sizes:
        sizes = ", ".join(f"{k}={v}" for k, v in sorted(manifest.vocabulary_sizes.items()))
        lines.append(f"  vocab (max)   : {sizes}")
    for path in result.figure_paths:
        lines.append(f"  figure        : {path}")
    if result.predictions_path:
        lines.append(f"  wrote         : {result.predictions_path}")
        lines.append(f"  next          : sentinel evaluate --predictions {result.predictions_path}")
    return "\n".join(lines)


def summarize_categoricals(result: CategoricalsResult) -> str:
    manifest = result.manifest
    lines = [
        f"Component 8 experimental categoricals: {manifest.rows} row(s)",
        f"  families      : {', '.join(manifest.families)}",
        "  cardinality   : "
        + ", ".join(f"{k}={v}" for k, v in sorted(manifest.cardinality.items())),
        "  coverage      : "
        + ", ".join(f"{k}={v:.4f}" for k, v in sorted(manifest.coverage.items())),
        f"  no prior insp.: {manifest.rows_without_prior_inspection} row(s) -> __UNKNOWN__",
    ]
    if result.categoricals_path:
        lines.append(f"  wrote         : {result.categoricals_path}")
    return "\n".join(lines)


def summarize_tuning(result: NeuralTuningResult) -> str:
    manifest = result.manifest
    lines = [
        f"Component 8 learning-rate sweep: {manifest.trial_rows} trial row(s) in "
        f"{manifest.seconds_total:.1f}s",
        f"  grid          : {', '.join(f'{v:g}' for v in manifest.grid)}",
    ]
    for study in manifest.studies:
        lines.append(
            f"  {study['fold_set']:<12}: lr={study['best_learning_rate']} over "
            f"{study['inner_folds']} inner fold(s), region {study['region']}"
        )
    for result_item in result.results:
        scores = ", ".join(f"{r:g}:{v:.4f}" for r, v in result_item.scores)
        lines.append(f"    {result_item.fold_set} mean inner PR-AUC -> {scores}")
    if result.trials_path:
        lines.append(f"  wrote         : {result.trials_path}")
    lines.append("")
    lines.append("Paste into src/sentinel/neural/definitions.py:")
    lines.append(tuning.frozen_block(result.results))
    return "\n".join(lines)


__all__ = [
    "BLOCKED_EXPERIMENTS",
    "CATEGORICALS_SLUG",
    "PREDICTIONS_SLUG",
    "SWEEP_SLUG",
    "CategoricalsResult",
    "NeuralBuildError",
    "NeuralResult",
    "NeuralTuningResult",
    "build_neural_categoricals",
    "summarize",
    "summarize_categoricals",
    "summarize_tuning",
    "train_neural",
    "tune_neural",
]
