"""Orchestration: a Component 17 candidate set in, a scored and ranked priority set out.

The only module in the package that touches the filesystem or the clock. Every other
module is a pure function of its arguments, matching every other component's shape.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from sentinel import __version__
from sentinel.config import Settings
from sentinel.features.models import ValidationCheck
from sentinel.manifest import compute_sha256, manifest_path_for, write_manifest
from sentinel.operational_scoring import calibrator as calibrator_module
from sentinel.operational_scoring import score as score_module
from sentinel.operational_scoring import selection, writer
from sentinel.operational_scoring.calibrator import CalibratorLoadError
from sentinel.operational_scoring.definitions import (
    CALIBRATION_SOURCE,
    MODEL_SELECTION_SOURCE,
    OPERATIONAL_SCORING_DEFINITION_VERSION,
)
from sentinel.operational_scoring.models import ArtifactRecord, OperationalScoringManifest
from sentinel.operational_scoring.score import OperationalScoringError
from sentinel.operational_scoring.selection import ModelSelectionError
from sentinel.operational_scoring.window import OperationalWindowError

logger = logging.getLogger(__name__)

DATASET_SLUG = "operational_priority"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

REQUIRED_CANDIDATE_COLUMNS = ("target_inspection_id", "establishment_id", "planning_date")


class OperationalPriorityBuildError(RuntimeError):
    """Raised when an operational priority set cannot be built at all."""


@dataclass
class OperationalPriorityResult:
    """Everything a caller needs after a build, written or not."""

    priorities: pl.DataFrame
    checks: list[ValidationCheck]
    manifest: OperationalScoringManifest
    priorities_path: Path | None = None
    manifest_path: Path | None = None


def build_operational_priorities(
    settings: Settings,
    *,
    candidates_path: Path,
    historical_features_path: Path,
    simulation_path: Path,
    metrics_path: Path,
    sensitivity_path: Path,
    calibrated_predictions_path: Path,
    calibrator_parameters_path: Path,
    calibrator_isotonic_breakpoints_path: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> OperationalPriorityResult:
    """Score, calibrate and rank one Component 17 candidate set."""
    for label, path in (
        ("Component 17 candidates", candidates_path),
        ("Component 4 historical features", historical_features_path),
        ("Component 5 simulation summary", simulation_path),
        ("Component 5 evaluation metrics", metrics_path),
        ("Component 5 NDE sensitivity", sensitivity_path),
        ("Component 9 calibrated predictions", calibrated_predictions_path),
        ("Component 9 calibrator parameters", calibrator_parameters_path),
        ("Component 9 calibrator isotonic breakpoints", calibrator_isotonic_breakpoints_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    started = datetime.now(UTC)

    candidates = pl.read_parquet(candidates_path)
    missing = [c for c in REQUIRED_CANDIDATE_COLUMNS if c not in candidates.columns]
    if missing:
        raise OperationalPriorityBuildError(
            f"{candidates_path.name}: not a Component 17 candidate table, missing "
            f"{', '.join(missing)}"
        )
    if candidates.is_empty():
        raise OperationalPriorityBuildError(
            f"{candidates_path.name}: zero candidates. Component 17 already refuses an "
            "unsupportable planning date; an empty candidate table here means the "
            "planning date genuinely has no eligible establishments, and there is "
            "nothing for this component to score"
        )
    planning_date_str = str(candidates["planning_date"][0])
    try:
        planning_date = date.fromisoformat(planning_date_str)
    except ValueError as exc:
        raise OperationalPriorityBuildError(
            f"{candidates_path.name}: planning_date {planning_date_str!r} is not ISO-8601"
        ) from exc
    candidate_definition_version = str(candidates["candidate_definition_version"][0])
    feature_definition_version = str(candidates["feature_definition_version"][0])

    historical_features = pl.read_parquet(historical_features_path).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )

    simulation = pl.read_parquet(simulation_path)
    metrics = pl.read_parquet(metrics_path)
    sensitivity = pl.read_parquet(sensitivity_path)
    calibrated_predictions = pl.read_parquet(calibrated_predictions_path)

    try:
        choice = selection.resolve_production_model(
            simulation=simulation,
            metrics=metrics,
            sensitivity=sensitivity,
            calibrated_predictions=calibrated_predictions,
        )
    except ModelSelectionError as exc:
        raise OperationalPriorityBuildError(f"model selection failed: {exc}") from exc

    try:
        frozen_calibrator = calibrator_module.load_frozen_calibrator(
            base_model_name=choice.base_model_name,
            method=choice.method,
            fold_set=choice.calibration_fold_set,
            fold_id=choice.calibration_fold_id,
            parameters_path=calibrator_parameters_path,
            breakpoints_path=calibrator_isotonic_breakpoints_path,
        )
    except CalibratorLoadError as exc:
        raise OperationalPriorityBuildError(f"calibrator could not be loaded: {exc}") from exc

    try:
        result = score_module.score_candidates(
            candidates=candidates,
            historical_features=historical_features,
            planning_date=planning_date,
            choice=choice,
            calibrator=frozen_calibrator,
        )
    except (OperationalScoringError, OperationalWindowError) as exc:
        raise OperationalPriorityBuildError(str(exc)) from exc

    finalized = writer.finalize(
        result.frame,
        composite_model_name=choice.composite_model_name,
        base_model_name=choice.base_model_name,
        calibration_method=choice.method,
        operational_scoring_definition_version=OPERATIONAL_SCORING_DEFINITION_VERSION,
    )

    destination = output_dir or settings.operational_scoring_processed_dir
    stamp = started.strftime(TIMESTAMP_FORMAT)

    artifacts: list[ArtifactRecord] = []
    priorities_path: Path | None = None
    if not dry_run:
        priorities_path = (
            destination / f"{DATASET_SLUG}_{planning_date.isoformat()}_{stamp}.parquet"
        )
        writer.write_table(finalized, priorities_path)
        artifacts.append(
            ArtifactRecord(
                path=priorities_path.name,
                bytes=priorities_path.stat().st_size,
                sha256=compute_sha256(priorities_path),
                row_count=finalized.height,
                schema=writer.schema_of(finalized),
            )
        )

    hp = result.hyperparameter_provenance
    warnings: list[str] = []
    if hp.fold_set is not None:
        warnings.append(
            f"architectural_provenance: operational inference reuses the frozen "
            f"{hp.fold_set!r} tuning configuration because no separately tuned "
            "operational fold set exists (see hyperparameter_source). This is an "
            "intentional architectural choice, not a data-quality anomaly."
        )

    manifest = OperationalScoringManifest(
        code_version=__version__,
        operational_scoring_definition_version=OPERATIONAL_SCORING_DEFINITION_VERSION,
        built_at=started.isoformat(),
        planning_date=planning_date.isoformat(),
        candidates_path=candidates_path.name,
        candidates_sha256=compute_sha256(candidates_path),
        candidate_definition_version=candidate_definition_version,
        feature_definition_version=feature_definition_version,
        historical_features_path=historical_features_path.name,
        historical_features_sha256=compute_sha256(historical_features_path),
        simulation_path=simulation_path.name,
        simulation_sha256=compute_sha256(simulation_path),
        metrics_path=metrics_path.name,
        metrics_sha256=compute_sha256(metrics_path),
        sensitivity_path=sensitivity_path.name,
        sensitivity_sha256=compute_sha256(sensitivity_path),
        calibrated_predictions_path=calibrated_predictions_path.name,
        calibrated_predictions_sha256=compute_sha256(calibrated_predictions_path),
        calibrator_parameters_path=calibrator_parameters_path.name,
        calibrator_parameters_sha256=compute_sha256(calibrator_parameters_path),
        calibrator_isotonic_breakpoints_path=calibrator_isotonic_breakpoints_path.name,
        calibrator_isotonic_breakpoints_sha256=compute_sha256(calibrator_isotonic_breakpoints_path),
        model_family=result.model_family.value,
        composite_model_name=choice.composite_model_name,
        base_model_name=choice.base_model_name,
        model_selection_source=MODEL_SELECTION_SOURCE,
        model_selection_decided_on_axis=choice.decided_on_axis,
        model_selection_n_tied_on_nde=choice.n_tied_on_nde,
        hyperparameter_fold_set=hp.fold_set,
        hyperparameter_source=hp.source,
        hyperparameter_values=hp.values,
        calibration_method=choice.method,
        calibration_source=CALIBRATION_SOURCE,
        calibration_fold_set=choice.calibration_fold_set,
        calibration_fold_id=choice.calibration_fold_id,
        calibrator_fit_start=frozen_calibrator.fit_start.isoformat(),
        calibrator_fit_end=frozen_calibrator.fit_end.isoformat(),
        operational_fold_set=result.fold.fold_set,
        operational_fold_id=result.fold.fold_id,
        training_window_rule="see operational_scoring.definitions.OPERATIONAL_TRAINING_WINDOW_RULE",
        train_start=result.fold.train_start.isoformat(),
        train_end=result.fold.train_end.isoformat(),
        train_rows=result.train_rows,
        train_positive_rate=result.train_positive_rate,
        score_direction="descending: higher score = higher predicted violation risk "
        "(sentinel.evaluation.models.SCORE_DIRECTION)",
        tie_break_column="target_inspection_id",
        candidate_count=candidates.height,
        scored_count=result.scored_count,
        excluded_count=result.excluded_count,
        coverage_eligible_count=result.coverage_eligible_count,
        warnings=warnings,
        artifacts=artifacts,
        checks=[
            {
                "name": c.name,
                "severity": c.severity,
                "passed": str(c.passed),
                "detail": c.detail,
            }
            for c in result.checks
        ],
    )

    manifest_path: Path | None = None
    if not dry_run and priorities_path is not None:
        manifest_path = manifest_path_for(priorities_path)
        write_manifest(manifest, manifest_path)

    logger.info(
        "Scored %d/%d operational candidates with %s (%d excluded)",
        result.scored_count,
        candidates.height,
        choice.composite_model_name,
        result.excluded_count,
    )
    return OperationalPriorityResult(
        priorities=finalized,
        checks=result.checks,
        manifest=manifest,
        priorities_path=priorities_path,
        manifest_path=manifest_path,
    )


def summarize(result: OperationalPriorityResult) -> str:
    """One-screen summary of a build, printed by the CLI."""
    m = result.manifest
    lines = [
        f"planning date:        {m.planning_date}",
        f"model:                {m.composite_model_name} (base {m.base_model_name}, "
        f"family {m.model_family}, calibration {m.calibration_method})",
        f"training window:      {m.train_start} .. {m.train_end} ({m.train_rows} rows, "
        f"positive rate {m.train_positive_rate})",
        f"hyperparameters:      fold set {m.hyperparameter_fold_set!r}",
        f"candidates:           {m.candidate_count}",
        f"  scored:             {m.scored_count}",
        f"  excluded:           {m.excluded_count}",
        f"  coverage-eligible:  {m.coverage_eligible_count}",
        f"selection axis:       {m.model_selection_decided_on_axis} "
        f"({m.model_selection_n_tied_on_nde} tied on NDE)",
    ]
    if m.warnings:
        lines.append("warnings:")
        lines.extend(f"  - {w}" for w in m.warnings)
    if result.priorities_path is not None:
        lines.append(f"priorities:           {result.priorities_path}")
        lines.append(f"manifest:             {result.manifest_path}")
    return "\n".join(lines)


__all__ = [
    "OperationalPriorityBuildError",
    "OperationalPriorityResult",
    "build_operational_priorities",
    "summarize",
]
