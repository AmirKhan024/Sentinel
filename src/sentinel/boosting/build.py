"""Orchestration: a feature table in, tuning trials or prediction artifacts out.

The only module in the package that touches the filesystem, so everything it calls stays
a pure function of its inputs.

Two commands live here and they are deliberately separate.

``tune_boosting`` runs the searches and writes a trials table. It changes no source file
and freezes nothing by itself -- it prints the parameter block a human pastes into
``definitions.TUNED_PARAMS``. That manual step is the design, not an omission: a
parameter set loaded from disk at training time could change without a diff, and the
entire value of freezing is that it cannot.

``train_boosting`` fits one model per fold from the frozen parameters and writes the
prediction artifact. Like Component 6, it produces **no metrics**: Component 5 evaluates,
Component 7 predicts. Computing a second set of numbers here would create two answers to
every question and would put the test window within reach of a component that is allowed
to fit things. The predictions are handed over by artifact:
``sentinel evaluate --predictions <path>``.

**One model per fold, refitted from scratch.** Fold N's model is a fresh fit over the
expanded training window, not fold N-1's model with more data bolted on. That is what a
health department retraining quarterly actually does.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import sklearn

from sentinel import __version__
from sentinel.boosting import predict, train, tuning, validate, writer
from sentinel.boosting.definitions import (
    BOOSTING_DEFINITION_VERSION,
    BOOSTING_REGISTRY,
    EARLY_STOPPING_ROUNDS,
    MAX_BOOSTING_ROUNDS,
    TUNABLE_MODELS,
    TUNED_PARAMS_PROVENANCE,
    TUNING_SEED,
    BoostingSpec,
    spec_for,
)
from sentinel.boosting.models import (
    ArtifactRecord,
    BoostedModelManifest,
    BoostingStats,
    FittedBooster,
    StudyResult,
    TuningManifest,
    ValidationCheck,
)
from sentinel.config import Settings
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import SCORE_DIRECTION, FoldSpec
from sentinel.manifest import compute_sha256, manifest_path_for, write_manifest
from sentinel.modeling.train import training_frame

logger = logging.getLogger(__name__)

PREDICTIONS_SLUG = "boosted_predictions"
TUNING_SLUG = "tuning_trials"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

#: Stated in the manifest so a consumer never infers it from the code.
TRAINED_THROUGH_SEMANTICS = (
    "fold.train_end -- the last reference date the fit was allowed to learn from. "
    "Component 7 fits no calibrator and, unlike a naive boosted implementation, does no "
    "early stopping in the final fit either, so the calibration window is untouched by "
    "model fitting. The boosting-round count came from a search confined to a region "
    "strictly earlier than any test window. ADR 0014, ADR 0017."
)

#: The single most misreadable thing this component emits, so it is stated in every
#: manifest rather than left to a document nobody opens beside the parquet.
PROBABILITY_SEMANTICS = (
    "RAW predicted probability from predict_proba, NOT a calibrated probability. A "
    "boosted ensemble's probabilities are typically worse calibrated than a penalised "
    "GLM's; Component 7 measures that through Component 5's ECE and MCE and corrects "
    "none of it. Component 9 owns calibration."
)

PREPROCESSING = (
    "none. NULLs reach the estimator as NaN and are routed by a learned default "
    "direction at each split; no imputation and no scaling is fitted. The four "
    "null-rule family indicators are retained so the boosted and baseline matrices are "
    "identical and the C6/C7 comparison is unambiguous."
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
    "inspector-effect modelling and inspector marginalisation: the Chicago food "
    "inspections dataset publishes 22 columns and none identifies an inspector. A "
    "mixed-effects model with inspector as a random intercept, and any marginalisation "
    "over an inspector effect, are undefined without that field. The plausible "
    "substitutes are refused rather than used: violation-text verbosity is "
    "unattributable and already rejected as a target by ADR 0008, ward and community "
    "area are route proxies fully confounded with establishment composition, and "
    "day-of-week and month are already confounded with weather, holidays and staffing. "
    "None identifies a person, so this is not approximated by a proxy. See ADR 0019",
    "probability calibration: boosted probabilities are emitted raw and uncorrected. "
    "Platt scaling, isotonic regression and any operating threshold belong to Component "
    "9, which is why the calibration window is left untouched by every fit here",
    "SHAP attribution: Component 7 emits native split-gain importances as a diagnostic "
    "only. Attribution is Component 11's, and a condition number of 71.8 with one "
    "feature pair correlated at 0.9888 makes tree importances a poor substitute for it",
    "ensembling xgboost with lightgbm or with the Component 6 logistic model: stacking "
    "and rank averaging come after the individual model stages, and blending before "
    "each member is understood would hide which one is carrying the result",
)


class BoostingBuildError(RuntimeError):
    """Raised when a boosted run cannot be completed at all."""


@dataclass
class BoostingResult:
    """Everything a caller needs after a training run, written or not."""

    tables: dict[str, pl.DataFrame]
    checks: list[ValidationCheck]
    manifest: BoostedModelManifest
    folds: list[FoldSpec]
    fitted: list[FittedBooster] = field(default_factory=list)
    predictions_path: Path | None = None
    manifest_path: Path | None = None


@dataclass
class TuningResult:
    """Everything a caller needs after a search, written or not."""

    tables: dict[str, pl.DataFrame]
    checks: list[ValidationCheck]
    manifest: TuningManifest
    studies: list[StudyResult]
    trials_path: Path | None = None
    manifest_path: Path | None = None


# --- 1. tuning ---------------------------------------------------------------


def tune_boosting(
    settings: Settings,
    *,
    features_path: Path,
    output_dir: Path | None = None,
    models: Sequence[str] | None = None,
    fold_sets: Sequence[str] | None = None,
    trials: int = 100,
    seed: int = TUNING_SEED,
    dry_run: bool = False,
) -> TuningResult:
    """Search each requested model's hyperparameters on each requested fold set."""
    if not features_path.exists():
        raise FileNotFoundError(f"Component 4 features not found: {features_path}")

    started = datetime.now(UTC)
    specs = _resolve_specs(models, tunable_only=True)
    frame = _load(features_path)
    outer_folds = _build_folds(frame)
    wanted = _resolve_fold_sets(fold_sets, outer_folds)

    logger.info(
        "Tuning %d model(s) over %d fold set(s) at %d trial(s) each: %s",
        len(specs),
        len(wanted),
        trials,
        ", ".join(s.name for s in specs),
    )

    studies: list[StudyResult] = []
    for spec in specs:
        for fold_set in wanted:
            studies.append(
                tuning.run_study(
                    spec, frame, outer_folds, fold_set=fold_set, trials=trials, seed=seed
                )
            )

    rows = [row for study in studies for row in _trial_rows(study)]
    tables = {TUNING_SLUG: writer.finalize(rows, TUNING_SLUG)}
    checks = validate.validate_tuning(studies, outer_folds)

    destination = output_dir or settings.tuning_processed_dir
    stamp = started.strftime(TIMESTAMP_FORMAT)
    artifacts: list[ArtifactRecord] = []
    trials_path: Path | None = None

    if not dry_run:
        for name, table in tables.items():
            path = destination / f"{name}_{stamp}.parquet"
            writer.write_table(table, path)
            if name == TUNING_SLUG:
                trials_path = path
            artifacts.append(_artifact_record(path, table))

    failed = sum(1 for study in studies for trial in study.trials if trial.failed)
    manifest = TuningManifest(
        code_version=__version__,
        boosting_definition_version=BOOSTING_DEFINITION_VERSION,
        built_at=started.isoformat(),
        features_path=features_path.name,
        features_sha256=compute_sha256(features_path),
        feature_definition_version=str(frame["feature_definition_version"][0]),
        xgboost_version=_library_version("xgboost"),
        lightgbm_version=_library_version("lightgbm"),
        optuna_version=_library_version("optuna"),
        numpy_version=np.__version__,
        blas_threads=_blas_threads(),
        objective=tuning.OBJECTIVE,
        sampler=tuning.SAMPLER,
        sampler_seed=seed,
        trials_requested=trials,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        max_boosting_rounds=MAX_BOOSTING_ROUNDS,
        studies=[
            {
                "study": s.study,
                "model": s.model_name,
                "fold_set": s.fold_set,
                "trials": str(len(s.trials)),
                "seconds": str(s.seconds),
            }
            for s in studies
        ],
        tuning_regions={s.study: f"{s.region_start}..{s.region_end}" for s in studies},
        first_test_start={
            fold_set: str(tuning.first_test_start(fold_set, outer_folds)) for fold_set in wanted
        },
        inner_folds={s.study: list(s.inner_folds) for s in studies},
        best_params={s.study: writer.render_params(tuning.frozen_params(s)) for s in studies},
        best_scores={s.study: round(s.best.mean_pr_auc, 6) for s in studies},
        feature_rows=frame.height,
        trial_rows=tables[TUNING_SLUG].height,
        failed_trials=failed,
        seconds_total=round((datetime.now(UTC) - started).total_seconds(), 3),
        blocked=list(BLOCKED_EXPERIMENTS),
        artifacts=artifacts,
        checks=_render_checks(checks),
    )

    manifest_path: Path | None = None
    if not dry_run and trials_path is not None:
        manifest_path = manifest_path_for(trials_path)
        write_manifest(manifest, manifest_path)

    return TuningResult(
        tables=tables,
        checks=checks,
        manifest=manifest,
        studies=studies,
        trials_path=trials_path,
        manifest_path=manifest_path,
    )


# --- 2. training -------------------------------------------------------------


def train_boosting(
    settings: Settings,
    *,
    features_path: Path,
    output_dir: Path | None = None,
    models: Sequence[str] | None = None,
    dry_run: bool = False,
) -> BoostingResult:
    """Fit every requested booster on every fold and write the prediction artifacts."""
    if not features_path.exists():
        raise FileNotFoundError(f"Component 4 features not found: {features_path}")

    started = datetime.now(UTC)
    specs = _resolve_specs(models, tunable_only=False)
    logger.info(
        "Training %d booster(s) from %s: %s",
        len(specs),
        features_path.name,
        ", ".join(s.name for s in specs),
    )

    frame = _load(features_path)
    all_folds = _build_folds(frame)

    stats = BoostingStats(feature_rows=frame.height, folds=len(all_folds))
    stats.models = [s.name for s in specs]
    for fold in all_folds:
        stats.fold_sets[fold.fold_set] = stats.fold_sets.get(fold.fold_set, 0) + 1

    fitted: list[FittedBooster] = []
    prediction_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    log_rows: list[dict[str, object]] = []

    for fold in all_folds:
        training = training_frame(frame, fold)
        test = folds_module.window_frame(frame, fold)
        if test.height == 0:
            logger.warning("%s: test window is empty, skipping", fold.fold_id)
            continue
        for spec in specs:
            fit_started = datetime.now(UTC)
            model = train.fit_fold(spec, training, fold)
            elapsed = (datetime.now(UTC) - fit_started).total_seconds()
            stats.fit_seconds_total += elapsed
            stats.fits += 1
            fitted.append(model)

            ids, scores = predict.score_window(model, test)
            prediction_rows.extend(_prediction_rows(model, ids, scores))
            importance_rows.extend(_importance_rows(model))
            log_rows.append(_log_row(model, fold, test_rows=test.height, scores=scores))

    if not prediction_rows:
        raise BoostingBuildError(
            "no predictions were produced. Every fold had an empty test window, which "
            "means the fold set and the feature table disagree about the data range."
        )

    tables = {
        "boosted_predictions": writer.finalize(prediction_rows, "boosted_predictions"),
        "boosted_importances": writer.finalize(importance_rows, "boosted_importances"),
        "boosted_training_log": writer.finalize(log_rows, "boosted_training_log"),
    }
    stats.prediction_rows = tables["boosted_predictions"].height
    stats.importance_rows = tables["boosted_importances"].height
    stats.training_log_rows = tables["boosted_training_log"].height

    checks = validate.validate_boosting(
        frame,
        all_folds,
        fitted,
        tables["boosted_predictions"],
        expected_models=[s.name for s in specs],
    )

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

    manifest = BoostedModelManifest(
        code_version=__version__,
        boosting_definition_version=BOOSTING_DEFINITION_VERSION,
        built_at=started.isoformat(),
        features_path=features_path.name,
        features_sha256=compute_sha256(features_path),
        feature_definition_version=str(frame["feature_definition_version"][0]),
        xgboost_version=_library_version("xgboost"),
        lightgbm_version=_library_version("lightgbm"),
        numpy_version=np.__version__,
        sklearn_version=sklearn.__version__,
        blas_threads=_blas_threads(),
        score_direction=SCORE_DIRECTION,
        trained_through_semantics=TRAINED_THROUGH_SEMANTICS,
        probability_semantics=PROBABILITY_SEMANTICS,
        preprocessing=PREPROCESSING,
        matrix_columns=list(fitted[0].matrix_columns),
        fold_sets=stats.fold_sets,
        folds=len(all_folds),
        models=[s.name for s in specs],
        model_feature_counts={s.name: len(s.feature_columns) for s in specs},
        model_estimators={s.name: s.estimator.value for s in specs},
        seeds={s.name: s.seed for s in specs},
        model_params={
            f"{model.spec.name}/{model.fold_set}": writer.render_params(model.params)
            for model in _one_per_model_and_fold_set(fitted)
        },
        tuned_params_provenance=TUNED_PARAMS_PROVENANCE,
        feature_rows=frame.height,
        fits=stats.fits,
        prediction_rows=stats.prediction_rows,
        importance_rows=stats.importance_rows,
        fit_seconds_total=round(stats.fit_seconds_total, 3),
        blocked=list(BLOCKED_EXPERIMENTS),
        artifacts=artifacts,
        checks=_render_checks(checks),
    )

    manifest_path: Path | None = None
    if not dry_run and predictions_path is not None:
        manifest_path = manifest_path_for(predictions_path)
        write_manifest(manifest, manifest_path)

    logger.info(
        "Trained %d fit(s) over %d folds in %.1fs (%.1fs fitting)",
        stats.fits,
        len(all_folds),
        (datetime.now(UTC) - started).total_seconds(),
        stats.fit_seconds_total,
    )
    return BoostingResult(
        tables=tables,
        checks=checks,
        manifest=manifest,
        folds=all_folds,
        fitted=fitted,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
    )


# --- 3. inputs ---------------------------------------------------------------


def _resolve_specs(models: Sequence[str] | None, *, tunable_only: bool) -> list[BoostingSpec]:
    """Look up the requested models, or every registered one.

    Unknown names fail here rather than producing a partial run, so a typo cannot
    quietly halve the portfolio. Tuning excludes the ablation: it borrows its donor's
    parameters by design, and searching for it separately would vary two things at once.
    """
    available = list(BOOSTING_REGISTRY)
    if tunable_only:
        available = [s for s in available if s.name in TUNABLE_MODELS]
    if models is None:
        return available
    if not models:
        raise BoostingBuildError("no models requested")
    allowed = {s.name for s in available}
    resolved: list[BoostingSpec] = []
    for name in models:
        try:
            spec = spec_for(name)
        except KeyError as exc:
            raise BoostingBuildError(str(exc)) from exc
        if spec.name not in allowed:
            raise BoostingBuildError(
                f"{spec.name} is not tunable: it borrows its parameters from another "
                f"model. Tunable models: {', '.join(sorted(allowed))}"
            )
        resolved.append(spec)
    return resolved


def _resolve_fold_sets(requested: Sequence[str] | None, folds: Sequence[FoldSpec]) -> list[str]:
    """The fold sets to tune for, in a stable order."""
    present = sorted({f.fold_set for f in folds})
    if requested is None:
        return present
    if not requested:
        raise BoostingBuildError("no fold sets requested")
    unknown = sorted(set(requested) - set(present))
    if unknown:
        raise BoostingBuildError(
            f"unknown fold set(s): {', '.join(unknown)}. Present: {', '.join(present)}"
        )
    return [name for name in present if name in set(requested)]


def _load(features_path: Path) -> pl.DataFrame:
    """Read the feature table and parse its reference date once."""
    frame = pl.read_parquet(features_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise BoostingBuildError(f"Feature table is missing required columns: {', '.join(missing)}")
    if frame.height == 0:
        raise BoostingBuildError("feature table is empty")
    duplicated = frame.height - frame["target_inspection_id"].n_unique()
    if duplicated:
        raise BoostingBuildError(
            f"feature table has {duplicated} duplicated target_inspection_id value(s); "
            "the prediction contract requires one score per id and could not be met"
        )
    return frame.with_columns(pl.col("inspection_date").str.to_date().alias("rd"))


def _build_folds(frame: pl.DataFrame) -> list[FoldSpec]:
    """Component 5's fold set, rebuilt from the data. Never invented here.

    Includes ``covid_shift`` for the same reason Component 6 does: it is a separate fold
    set precisely so it cannot be averaged into the headline, and both earlier components
    measured the model ordering reversing on it. Omitting it would hide the one result
    most worth seeing.
    """
    data_start = folds_module.min_date(frame, "rd")
    data_end = folds_module.max_date(frame, "rd")
    if data_start is None or data_end is None:
        raise BoostingBuildError("feature table has no usable reference dates")
    quarterly = folds_module.quarterly_folds(data_start=data_start, data_end=data_end)
    if not quarterly:
        raise BoostingBuildError(
            f"data spans {data_start}..{data_end}, which is too short to build a single "
            "quarterly fold. Folds are never fabricated."
        )
    return [*quarterly, *folds_module.covid_shift_fold(data_end=data_end)]


# --- 4. row builders ---------------------------------------------------------


def _prediction_rows(
    model: FittedBooster, ids: Sequence[str], scores: Sequence[float]
) -> list[dict[str, object]]:
    return [
        {
            "target_inspection_id": row_id,
            "score": score,
            "model_name": model.spec.name,
            "model_version": model.spec.version,
            "fold_set": model.fold_set,
            "fold_id": model.fold_id,
            "trained_through": model.trained_through,
            "is_probability": model.spec.is_probability,
            "boosting_definition_version": BOOSTING_DEFINITION_VERSION,
        }
        for row_id, score in zip(ids, scores, strict=True)
    ]


def _importance_rows(model: FittedBooster) -> list[dict[str, object]]:
    """One row per matrix column. A diagnostic, not an attribution -- see the schema."""
    kind = "gain" if model.spec.estimator.value == "xgboost" else "split"
    return [
        {
            "model_name": model.spec.name,
            "model_version": model.spec.version,
            "fold_set": model.fold_set,
            "fold_id": model.fold_id,
            "term": term,
            "importance": model.importances[index],
            "importance_kind": kind,
            "boosting_definition_version": BOOSTING_DEFINITION_VERSION,
        }
        for index, term in enumerate(model.matrix_columns)
    ]


def _log_row(
    model: FittedBooster, fold: FoldSpec, *, test_rows: int, scores: Sequence[float]
) -> dict[str, object]:
    return {
        "model_name": model.spec.name,
        "model_version": model.spec.version,
        "estimator": model.spec.estimator.value,
        "fold_set": model.fold_set,
        "fold_id": model.fold_id,
        "train_start": model.train_start,
        "train_end": model.train_end,
        "trained_through": model.trained_through,
        "calibration_end_unused": model.calibration_end_unused,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "train_rows": model.train_rows,
        "test_rows": test_rows,
        "feature_count": len(model.spec.feature_columns),
        "matrix_column_count": len(model.matrix_columns),
        "train_nan_cells": model.train_nan_cells,
        "train_positive_rate": model.train_positive_rate,
        "seed": model.spec.seed,
        "n_estimators": model.n_estimators,
        "trees_built": model.trees_built,
        "scale_pos_weight": model.scale_pos_weight,
        "class_weighted": model.spec.class_weighted,
        # Always False, and recorded rather than omitted: the claim that a final fit read
        # no validation window is worth carrying in the artifact rather than in prose.
        "early_stopped": False,
        "saturated_scores": predict.saturated_count(list(scores)),
        "boosting_definition_version": BOOSTING_DEFINITION_VERSION,
    }


def _trial_rows(study: StudyResult) -> list[dict[str, object]]:
    return [
        {
            "study": study.study,
            "model_name": trial.model_name,
            "fold_set": trial.fold_set,
            "trial": trial.number,
            "params": writer.render_params(trial.params),
            "mean_pr_auc": trial.mean_pr_auc,
            "inner_fold_pr_aucs": json.dumps(
                {s.fold_id: round(s.pr_auc, 6) for s in trial.inner_scores}, sort_keys=True
            ),
            "inner_folds": len(trial.inner_scores),
            "n_estimators": trial.n_estimators,
            "seconds": trial.seconds,
            "failed": trial.failed,
            "failure": trial.failure,
            "sampler_seed": study.sampler_seed,
            "region_start": study.region_start,
            "region_end": study.region_end,
            "boosting_definition_version": BOOSTING_DEFINITION_VERSION,
        }
        for trial in study.trials
    ]


# --- 5. manifest helpers -----------------------------------------------------


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
        {"name": c.name, "severity": c.severity, "passed": str(c.passed), "detail": c.detail}
        for c in checks
    ]


def _one_per_model_and_fold_set(fitted: Sequence[FittedBooster]) -> list[FittedBooster]:
    """One representative fit per (model, fold set), for the manifest's parameter record.

    Parameters are constant within a fold set by construction, so recording all 54 fits
    would be 54 copies of two facts.
    """
    seen: set[tuple[str, str]] = set()
    out: list[FittedBooster] = []
    for model in fitted:
        key = (model.spec.name, model.fold_set)
        if key in seen:
            continue
        seen.add(key)
        out.append(model)
    return out


def _library_version(module: str) -> str:
    """A library's reported version, or a stated absence. Never a silent empty string."""
    from importlib import import_module

    try:
        return str(getattr(import_module(module), "__version__", "unknown"))
    except ImportError:  # pragma: no cover - the dependency is declared
        return "not installed"


def _blas_threads() -> str:
    """The thread count the fits ran under. Reported as the environment states it.

    Both boosters are additionally pinned to one thread through ``FIXED_PARAMS``, which
    is the setting that actually makes them deterministic; this records the surrounding
    environment because numpy's own reductions still answer to it.
    """
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(variable)
        if value:
            return f"{variable}={value}"
    return "unset (library default; estimators pinned to n_jobs=1)"


# --- 6. summaries ------------------------------------------------------------


def summarize(result: BoostingResult) -> str:
    """One-screen summary of a training run, printed by the CLI."""
    m = result.manifest
    lines = [
        f"feature rows:        {m.feature_rows}",
        f"folds:               {m.folds} ({', '.join(f'{k}={v}' for k, v in m.fold_sets.items())})",
        f"models:              {len(m.models)} ({', '.join(m.models)})",
        f"estimators:          {', '.join(f'{k}={v}' for k, v in m.model_estimators.items())}",
        f"fits:                {m.fits}",
        f"prediction rows:     {m.prediction_rows}",
        f"importance rows:     {m.importance_rows}",
        f"matrix columns:      {len(m.matrix_columns)}",
        "preprocessing:       none (NaN routed natively, no imputation, no scaling)",
        "trained_through:     fold.train_end (no calibrator, no early stopping)",
        f"score direction:     {m.score_direction}",
        "probabilities:       RAW, uncalibrated -- Component 9 owns calibration",
        f"definition:          {m.boosting_definition_version}",
        f"xgboost / lightgbm:  {m.xgboost_version} / {m.lightgbm_version}",
        f"numpy:               {m.numpy_version}",
        f"BLAS threads:        {m.blas_threads}",
        f"fit seconds:         {m.fit_seconds_total}",
        f"params provenance:   {m.tuned_params_provenance[:60]}",
    ]
    for item in m.blocked:
        lines.append(f"  BLOCKED:           {item.split(':')[0]}")
    lines.append("evaluation:          none here -- run `sentinel evaluate --predictions`")
    if result.predictions_path is not None:
        lines.append(f"predictions:         {result.predictions_path}")
        lines.append(f"manifest:            {result.manifest_path}")
    return "\n".join(lines)


def summarize_tuning(result: TuningResult) -> str:
    """One-screen summary of a search, plus the block to paste into ``definitions``."""
    m = result.manifest
    lines = [
        f"feature rows:        {m.feature_rows}",
        f"studies:             {len(m.studies)}",
        f"trials requested:    {m.trials_requested} per study",
        f"trial rows:          {m.trial_rows} ({m.failed_trials} failed/pruned)",
        f"objective:           {m.objective}",
        f"sampler:             {m.sampler} (seed {m.sampler_seed})",
        f"early stopping:      {m.early_stopping_rounds} rounds, inner validation only",
        f"max rounds:          {m.max_boosting_rounds}",
        f"xgboost / lightgbm:  {m.xgboost_version} / {m.lightgbm_version}",
        f"optuna:              {m.optuna_version}",
        f"seconds:             {m.seconds_total}",
        "",
        "tuning regions (each must end before its fold set's first test start):",
    ]
    for study in sorted(m.tuning_regions):
        fold_set = study.rsplit("-", 1)[-1]
        horizon = m.first_test_start.get(fold_set, "?")
        lines.append(f"  {study}: {m.tuning_regions[study]}  <  first test {horizon}")
    lines.append("")
    lines.append("best validation PR-AUC (NOT a result -- these windows are training data):")
    for study in sorted(m.best_scores):
        lines.append(f"  {study}: {m.best_scores[study]:.4f}")
    lines.append("")
    lines.append("freeze these into boosting.definitions.TUNED_PARAMS:")
    for study in sorted(m.best_params):
        lines.append(f"  {study}: {m.best_params[study]}")
    if result.trials_path is not None:
        lines.append("")
        lines.append(f"trials:              {result.trials_path}")
        lines.append(f"manifest:            {result.manifest_path}")
    return "\n".join(lines)


__all__ = [
    "BLOCKED_EXPERIMENTS",
    "PREDICTIONS_SLUG",
    "TUNING_SLUG",
    "BoostingBuildError",
    "BoostingResult",
    "TuningResult",
    "summarize",
    "summarize_tuning",
    "train_boosting",
    "tune_boosting",
]
