"""Orchestration: a feature table and a fold set in, prediction artifacts out.

The only module in the package that touches the filesystem, so everything it calls stays
a pure function of its inputs.

**One model per fold, refitted from scratch.** Fold N's model is not fold N-1's model
with more data bolted on; it is a fresh fit over the expanded training window. That is
what a health department retraining quarterly on everything accumulated so far actually
does, and it is the only version of the exercise whose numbers mean anything.

This module deliberately produces **no metrics**. Component 5 evaluates; Component 6
predicts. Computing a second set of numbers here would create two answers to every
question and no way to tell which was authoritative -- and it would put the test window
in reach of a component that is allowed to fit things. The predictions are handed over
by artifact: ``sentinel evaluate --predictions <path>``.

Determinism rests on four properties, asserted by the test suite:

1. Training rows are sorted canonically before fitting (7.049e-09 rides on this).
2. Preprocessing statistics come only from the training window.
3. Every output table is sorted by a declared key before it is written.
4. No table carries a timestamp or a duration.
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

from sentinel import __version__
from sentinel.config import Settings
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import SCORE_DIRECTION, FoldSpec
from sentinel.manifest import compute_sha256, manifest_path_for, write_manifest
from sentinel.modeling import predict, preprocess, train, validate, writer
from sentinel.modeling.definitions import (
    MODEL_DEFINITION_VERSION,
    MODEL_REGISTRY,
    MissingStrategy,
    ModelSpec,
    indicator_columns,
    max_iter_of,
    nullable_columns,
    spec_for,
)
from sentinel.modeling.models import (
    ArtifactRecord,
    BaselineModelManifest,
    BaselineStats,
    FittedModel,
    ValidationCheck,
)

logger = logging.getLogger(__name__)

DATASET_SLUG = "baseline_predictions"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

#: Stated in the manifest so a consumer never infers it from the code. Component 6
#: declares the training end, not the calibration end -- see ADR 0014.
TRAINED_THROUGH_SEMANTICS = (
    "fold.train_end -- the last reference date the fit was allowed to learn from. "
    "Component 6 fits no calibrator and never reads the calibration window, so it "
    "declares a horizon strictly earlier than the contract's ceiling of calibration_end."
)

REQUIRED_COLUMNS = (
    "establishment_id",
    "inspection_date",
    "target_inspection_id",
    "target",
    "feature_definition_version",
)

#: Experiments the current data cannot support. Reported, never faked.
BLOCKED_EXPERIMENTS: tuple[str, ...] = (
    "faithful CDPH 2015 replication: 7 of that model's 10 input families are not "
    "reachable from the current data contract (inspector identity, 311, crime, licence, "
    "weather, facility type, risk category), so an approximation ships instead and is "
    "labelled as one everywhere it appears",
    "statutory 'days overdue' feature: needs the CDPH risk category to know what the "
    "deadline was, which is in the raw snapshot but not in Component 4's feature table; "
    "it belongs in Component 4 behind a bumped feature_definition_version",
    "hyperparameter tuning: deliberately not done. A baseline exists to be trustworthy "
    "rather than optimal, and a tuning protocol belongs with the model that benefits "
    "from it",
)


class BaselineTrainingError(RuntimeError):
    """Raised when the baselines cannot be trained at all."""


@dataclass
class BaselineResult:
    """Everything a caller needs after a run, written or not."""

    tables: dict[str, pl.DataFrame]
    checks: list[ValidationCheck]
    manifest: BaselineModelManifest
    folds: list[FoldSpec]
    fitted: list[FittedModel] = field(default_factory=list)
    predictions_path: Path | None = None
    manifest_path: Path | None = None


def train_baselines(
    settings: Settings,
    *,
    features_path: Path,
    output_dir: Path | None = None,
    models: Sequence[str] | None = None,
    dry_run: bool = False,
) -> BaselineResult:
    """Fit every requested baseline on every fold and write the prediction artifacts."""
    if not features_path.exists():
        raise FileNotFoundError(f"Component 4 features not found: {features_path}")

    started = datetime.now(UTC)
    specs = _resolve_specs(models)
    logger.info(
        "Training %d baseline(s) from %s: %s",
        len(specs),
        features_path.name,
        ", ".join(s.name for s in specs),
    )

    frame = _load(features_path)
    all_folds = _build_folds(frame)

    stats = BaselineStats(feature_rows=frame.height, folds=len(all_folds))
    stats.models = [s.name for s in specs]
    for fold in all_folds:
        stats.fold_sets[fold.fold_set] = stats.fold_sets.get(fold.fold_set, 0) + 1

    fitted: list[FittedModel] = []
    prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    log_rows: list[dict[str, object]] = []

    for fold in all_folds:
        training = train.training_frame(frame, fold)
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
            coefficient_rows.extend(_coefficient_rows(model))
            log_rows.append(_log_row(model, fold, test_rows=test.height, scores=scores))

    if not prediction_rows:
        raise BaselineTrainingError(
            "no predictions were produced. Every fold had an empty test window, which "
            "means the fold set and the feature table disagree about the data range."
        )

    tables = {
        "baseline_predictions": writer.finalize(prediction_rows, "baseline_predictions"),
        "baseline_coefficients": writer.finalize(coefficient_rows, "baseline_coefficients"),
        "baseline_training_log": writer.finalize(log_rows, "baseline_training_log"),
    }
    stats.prediction_rows = tables["baseline_predictions"].height
    stats.coefficient_rows = tables["baseline_coefficients"].height
    stats.training_log_rows = tables["baseline_training_log"].height

    checks = validate.validate_baselines(
        frame,
        all_folds,
        fitted,
        tables["baseline_predictions"],
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
            if name == DATASET_SLUG:
                predictions_path = path
            artifacts.append(
                ArtifactRecord(
                    path=path.name,
                    bytes=path.stat().st_size,
                    sha256=compute_sha256(path),
                    row_count=table.height,
                    schema=writer.schema_of(table),
                )
            )

    manifest = BaselineModelManifest(
        code_version=__version__,
        model_definition_version=MODEL_DEFINITION_VERSION,
        built_at=started.isoformat(),
        features_path=features_path.name,
        features_sha256=compute_sha256(features_path),
        feature_definition_version=str(frame["feature_definition_version"][0]),
        sklearn_version=sklearn.__version__,
        numpy_version=np.__version__,
        blas_threads=_blas_threads(),
        score_direction=SCORE_DIRECTION,
        trained_through_semantics=TRAINED_THROUGH_SEMANTICS,
        missing_value_rules=_missing_value_rules(),
        indicator_columns=list(indicator_columns()),
        fold_sets=stats.fold_sets,
        folds=len(all_folds),
        models=[s.name for s in specs],
        model_feature_counts={s.name: len(s.feature_columns) for s in specs},
        approximation_notes={
            s.name: s.approximation_note for s in specs if s.approximation_note is not None
        },
        seeds={s.name: s.seed for s in specs},
        model_params={s.name: _render_params(s) for s in specs},
        feature_rows=frame.height,
        fits=stats.fits,
        prediction_rows=stats.prediction_rows,
        coefficient_rows=stats.coefficient_rows,
        fit_seconds_total=round(stats.fit_seconds_total, 3),
        blocked=list(BLOCKED_EXPERIMENTS),
        artifacts=artifacts,
        checks=[
            {
                "name": c.name,
                "severity": c.severity,
                "passed": str(c.passed),
                "detail": c.detail,
            }
            for c in checks
        ],
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
    return BaselineResult(
        tables=tables,
        checks=checks,
        manifest=manifest,
        folds=all_folds,
        fitted=fitted,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
    )


# --- 1. inputs ---------------------------------------------------------------


def _resolve_specs(models: Sequence[str] | None) -> list[ModelSpec]:
    """Look up the requested models, or every registered one.

    Unknown names fail here rather than producing a partial run, so a typo in
    ``--models`` cannot quietly halve the portfolio.
    """
    if models is None:
        return list(MODEL_REGISTRY)
    if not models:
        raise BaselineTrainingError("no models requested")
    resolved: list[ModelSpec] = []
    for name in models:
        try:
            resolved.append(spec_for(name))
        except KeyError as exc:
            raise BaselineTrainingError(str(exc)) from exc
    return resolved


def _load(features_path: Path) -> pl.DataFrame:
    """Read the feature table and parse its reference date once.

    Every column is read, unlike Component 5 which reads only what its heuristics need:
    a model consumes the features, so narrowing here would mean narrowing the model.
    """
    frame = pl.read_parquet(features_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise BaselineTrainingError(
            f"Feature table is missing required columns: {', '.join(missing)}"
        )
    if frame.height == 0:
        raise BaselineTrainingError("feature table is empty")
    duplicated = frame.height - frame["target_inspection_id"].n_unique()
    if duplicated:
        raise BaselineTrainingError(
            f"feature table has {duplicated} duplicated target_inspection_id value(s); "
            "the prediction contract requires one score per id and could not be met"
        )
    return frame.with_columns(pl.col("inspection_date").str.to_date().alias("rd"))


def _build_folds(frame: pl.DataFrame) -> list[FoldSpec]:
    """Component 5's fold set, rebuilt from the data. Never invented here.

    Includes ``covid_shift``: it is a separate fold set precisely so it cannot be
    averaged into the headline, and the Component 5 findings record that the baseline
    ordering reverses on it. Omitting it would hide the one result most worth seeing.
    """
    data_start = folds_module.min_date(frame, "rd")
    data_end = folds_module.max_date(frame, "rd")
    if data_start is None or data_end is None:
        raise BaselineTrainingError("feature table has no usable reference dates")
    quarterly = folds_module.quarterly_folds(data_start=data_start, data_end=data_end)
    if not quarterly:
        raise BaselineTrainingError(
            f"data spans {data_start}..{data_end}, which is too short to build a single "
            "quarterly fold. Folds are never fabricated."
        )
    return [*quarterly, *folds_module.covid_shift_fold(data_end=data_end)]


# --- 2. row builders ---------------------------------------------------------


def _prediction_rows(
    model: FittedModel, ids: Sequence[str], scores: Sequence[float]
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
            "model_definition_version": MODEL_DEFINITION_VERSION,
        }
        for row_id, score in zip(ids, scores, strict=True)
    ]


def _coefficient_rows(model: FittedModel) -> list[dict[str, object]]:
    """One row per term, plus the intercept as ``__intercept__``.

    The scaler statistics ride alongside each coefficient so a reader can recover the
    raw-feature-scale value. Without them the numbers are on a standardised scale that
    is not stated anywhere in the row.
    """
    rows: list[dict[str, object]] = []
    base = {
        "model_name": model.spec.name,
        "model_version": model.spec.version,
        "fold_set": model.fold_set,
        "fold_id": model.fold_id,
        "model_definition_version": MODEL_DEFINITION_VERSION,
    }
    for index, term in enumerate(model.matrix_columns):
        rows.append(
            {
                **base,
                "term": term,
                "standardized_coefficient": model.coefficients[index],
                "scaler_mean": model.scaler_mean[index],
                "scaler_scale": model.scaler_scale[index],
                "imputed_fill_value": model.imputed_values.get(term),
            }
        )
    rows.append(
        {
            **base,
            "term": "__intercept__",
            "standardized_coefficient": model.intercept,
            "scaler_mean": None,
            "scaler_scale": None,
            "imputed_fill_value": None,
        }
    )
    return rows


def _log_row(
    model: FittedModel, fold: FoldSpec, *, test_rows: int, scores: Sequence[float]
) -> dict[str, object]:
    return {
        "model_name": model.spec.name,
        "model_version": model.spec.version,
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
        "train_positive_rate": model.train_positive_rate,
        "seed": model.spec.seed,
        "n_iter": model.n_iter,
        "max_iter": max_iter_of(model.spec),
        "converged": model.converged,
        "saturated_scores": predict.saturated_count(list(scores)),
        "is_approximation": model.spec.approximation_note is not None,
        "model_definition_version": MODEL_DEFINITION_VERSION,
    }


# --- 3. manifest helpers -----------------------------------------------------


def _missing_value_rules() -> dict[str, str]:
    """The fill rule per nullable column, so the contract is in the manifest."""
    return {
        column: preprocess.strategy_for(column).value
        for column in nullable_columns()
        if preprocess.strategy_for(column) is not MissingStrategy.PASSTHROUGH
    }


def _render_params(spec: ModelSpec) -> str:
    """Hyperparameters as a stable string, so two manifests compare textually."""
    items = ", ".join(f"{k}={spec.params[k]!r}" for k in sorted(spec.params))
    return f"LogisticRegression({items}, random_state={spec.seed}, class_weight=None)"


def _blas_threads() -> str:
    """The BLAS thread count, which the coefficients depend on. ADR 0015.

    Reported as whatever the environment says, including "unset", because the honest
    answer is what was in effect and not what we would prefer.
    """
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(variable)
        if value:
            return f"{variable}={value}"
    return "unset (library default)"


def summarize(result: BaselineResult) -> str:
    """One-screen summary of a run, printed by the CLI."""
    m = result.manifest
    lines = [
        f"feature rows:        {m.feature_rows}",
        f"folds:               {m.folds} ({', '.join(f'{k}={v}' for k, v in m.fold_sets.items())})",
        f"models:              {len(m.models)} ({', '.join(m.models)})",
        f"fits:                {m.fits}",
        f"prediction rows:     {m.prediction_rows}",
        f"coefficient rows:    {m.coefficient_rows}",
        f"features per model:  {', '.join(f'{k}={v}' for k, v in m.model_feature_counts.items())}",
        f"indicators:          {len(m.indicator_columns)} ({', '.join(m.indicator_columns)})",
        "trained_through:     fold.train_end (never the calibration window)",
        f"score direction:     {m.score_direction}",
        f"definition:          {m.model_definition_version}",
        f"sklearn / numpy:     {m.sklearn_version} / {m.numpy_version}",
        f"BLAS threads:        {m.blas_threads}",
        f"fit seconds:         {m.fit_seconds_total}",
    ]
    for name in sorted(m.approximation_notes):
        lines.append(f"  APPROXIMATION:     {name} -- see manifest for unreachable inputs")
    for item in m.blocked:
        lines.append(f"  BLOCKED:           {item.split(':')[0]}")
    lines.append("evaluation:          none here -- run `sentinel evaluate --predictions`")
    if result.predictions_path is not None:
        lines.append(f"predictions:         {result.predictions_path}")
        lines.append(f"manifest:            {result.manifest_path}")
    return "\n".join(lines)


__all__ = [
    "BLOCKED_EXPERIMENTS",
    "DATASET_SLUG",
    "BaselineResult",
    "BaselineTrainingError",
    "summarize",
    "train_baselines",
]
