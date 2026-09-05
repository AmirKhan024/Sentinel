"""Orchestration for Component 11: read, re-execute, attribute, validate, write.

The order of operations is the component's safety argument, and it is deliberate:

1. checksum every prediction artifact **before** anything runs;
2. draw one explanation sample per fold, shared by every model, from the test window;
3. re-execute each supported model's frozen fit and **prove** it is the committed model by
   comparing its test scores bit for bit;
4. **raise** if that gate fails, before a single attribution is computed;
5. attribute, aggregate, validate;
6. write, then checksum the prediction artifacts again and record both.

Step 4 is why ``run_explanations`` raises rather than merely reporting: a validation report
would let the artifact be written and then complain about it, and an attribution computed on
a model no committed artifact contains is not a slightly-wrong explanation -- it is an
explanation of a different model, presented as an explanation of the deployed one. Component
9's ``build.py`` takes the same position for the same reason (ADR 0026, ADR 0029).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from sentinel import __version__
from sentinel.config import Settings
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.explain import aggregate, attribute, refit, validate, writer
from sentinel.explain import sample as sample_module
from sentinel.explain.background import select_background
from sentinel.explain.definitions import (
    BACKGROUND_SEED,
    BACKGROUND_SIZE,
    BACKGROUND_STRATEGY,
    BLOCKED_EXPERIMENTS,
    EXPLAIN_DEFINITION_VERSION,
    EXPLAIN_REGISTRY,
    PERMUTATION_ROUNDS,
    RANK_DRIFT_THRESHOLD,
    REPRESENTATIVE_FOLD_SET,
    REPRESENTATIVE_QUANTILES,
    SAMPLE_SIZE,
    SAMPLE_STRATEGY,
    SAMPLING_POPULATION,
    SAMPLING_SEED,
    SUPPORTED_MODELS,
    TOP_K,
    ExplanationStatus,
    kind_of,
    origin_of,
    spec_for,
    tolerance_for,
)
from sentinel.explain.models import (
    ArtifactRecord,
    ExplainManifest,
    ExplainStats,
    ExplanationSample,
    FoldAttribution,
    RefitModel,
    ReproductionOutcome,
    ValidationCheck,
)
from sentinel.features.definitions import FEATURE_DEFINITION_VERSION
from sentinel.manifest import compute_sha256, manifest_path_for, write_manifest

logger = logging.getLogger(__name__)

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

#: Figures are documentation rather than data, so they sit beside the findings document
#: that reads them. Overridable per run; the default is the project's standing location.
FIGURES_DIR = Path("docs/analysis/figures")

#: Which committed artifact each model's scores come from. Restated here rather than
#: inferred so a run names every file it reads.
SOURCE_PREFIX: Mapping[str, str] = {
    "baseline_predictions": "baseline_predictions_",
    "boosted_predictions": "boosted_predictions_",
    "neural_predictions": "neural_predictions_",
}

#: Prose recorded in the manifest, so a consumer never has to infer the semantics.
ATTRIBUTION_SEMANTICS = (
    "a shap_value is the contribution of one feature to one prediction's log-odds, "
    "relative to base_value, under the method named in explanation_method. It describes "
    "how the model used the feature. It is not a measurement of the feature's effect on "
    "food safety and not an estimate of what would happen if the feature changed."
)
CALIBRATION_BOUNDARY = (
    "attributions decompose the BASE model's log-odds output, not Component 9's calibrated "
    "probability. Platt is a monotone two-parameter map applied afterwards; it changes the "
    "number a user is shown and changes no ranking. base_score and calibrated_probability "
    "are carried side by side on explanation_cases so the two can be connected without "
    "either being mistaken for the other. ADR 0030."
)
CAUSALITY_DISCLAIMER = (
    "predictive importance is not causality. A feature can carry a large attribution "
    "because it proxies something the model cannot see, because it encodes a scheduling "
    "policy, or because it is correlated with a feature that matters. Component 7 measured "
    "a condition number of 71.8 with one feature pair correlated at 0.9888, so attributions "
    "are shared between correlated features in a way no SHAP value discloses. Nothing in "
    "this artifact supports an intervention claim."
)
DETERMINISM_CAVEAT = (
    "identical output for a fixed feature table, a fixed row order, this library set, one "
    "torch thread and CPU. Not across library versions: a bump makes the bit-identity gate "
    "fail, which is the correct behaviour and an explicit re-baseline. Run without an "
    "OMP_NUM_THREADS override -- a different BLAS thread count is a different float "
    "summation order, and ADR 0026 records it moving logistic_regression scores by 1e-13."
)


class ExplainBuildError(RuntimeError):
    """Raised when an explanation run cannot proceed."""


@dataclass(slots=True)
class ExplainResult:
    """Everything a caller needs to report on one run."""

    checks: list[ValidationCheck]
    stats: ExplainStats
    tables: dict[str, pl.DataFrame]
    values_path: Path | None = None
    manifest_path: Path | None = None
    written: list[Path] = field(default_factory=list)
    figure_paths: list[Path] = field(default_factory=list)
    dry_run: bool = False


# --- inputs ------------------------------------------------------------------


def _load_features(path: Path) -> pl.DataFrame:
    """Component 4's table with a parsed reference date, as every component reads it."""
    return pl.read_parquet(path).with_columns(pl.col("inspection_date").str.to_date().alias("rd"))


def _build_folds(frame: pl.DataFrame) -> list[FoldSpec]:
    """The same 18 folds Component 5 defines. Derived from the data, never hard-coded."""
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    if start is None or end is None:
        raise ExplainBuildError("feature table has no usable reference dates")
    return [
        *folds_module.quarterly_folds(data_start=start, data_end=end),
        *folds_module.covid_shift_fold(data_end=end),
    ]


def required_sources(models: Sequence[str]) -> set[str]:
    """Which committed artifacts a set of models is scored in.

    Derived from each model's registry entry rather than fixed, so a run that explains only
    Component 7's boosters neither reads nor checksums Component 8's file. Recording a
    dependency a run did not have would make the manifest describe a different run.
    """
    return {spec_for(name).source_slug for name in models}


def _committed_predictions(paths: Mapping[str, Path], models: Sequence[str]) -> pl.DataFrame:
    """The committed scores for every requested model, in one frame.

    Read straight from the Parquet rather than through ``evaluation.contract``: the
    comparison downstream is on raw stored floats, and a round trip through a
    ``PredictionSet`` would add a select and a filter between the file and the assertion for
    no benefit. Component 9's ``committed_test_scores`` reads them the same way and says so.
    """
    columns = ["model_name", "fold_set", "fold_id", "target_inspection_id", "score"]
    wanted = set(models)
    needed = required_sources(models)
    absent = sorted(needed - set(paths))
    if absent:
        raise ExplainBuildError(
            f"explaining {', '.join(sorted(wanted))} needs the committed artifact(s) "
            f"{', '.join(absent)}, which were not supplied"
        )
    parts: list[pl.DataFrame] = []
    for slug in sorted(needed):
        frame = pl.read_parquet(paths[slug]).select(columns)
        parts.append(frame.filter(pl.col("model_name").is_in(list(wanted))))
    if not parts:
        raise ExplainBuildError("no committed prediction artifacts were supplied")
    combined = pl.concat(parts)
    missing = sorted(wanted - set(str(v) for v in combined["model_name"].unique().to_list()))
    if missing:
        raise ExplainBuildError(
            f"committed predictions are missing for {', '.join(missing)}. Component 11 "
            "explains models that were already scored; it does not create a prediction."
        )
    return combined


def _calibrated_lookup(path: Path | None) -> dict[tuple[str, str, str], tuple[float, str]]:
    """``(base model, fold, inspection) -> (calibrated probability, method)``.

    Keyed on ``base_model_name``, not ``model_name``: a calibrated row's ``model_name`` is
    ``"<base>_<method>"`` precisely so it can never be mistaken for its uncalibrated
    ancestor, which means joining on it here would find nothing.
    """
    if path is None:
        return {}
    frame = pl.read_parquet(path).select(
        ["base_model_name", "fold_id", "target_inspection_id", "score", "method"]
    )
    return {
        (str(r["base_model_name"]), str(r["fold_id"]), str(r["target_inspection_id"])): (
            float(r["score"]),
            str(r["method"]),
        )
        for r in frame.to_dicts()
    }


def _horizons(path: Path | None) -> dict[tuple[str, str], tuple[object, object]]:
    """``(base model, fold) -> (calibrator_fitted_through, available_from)``.

    Read from Component 9's artifact rather than recomputed. The two dates are properties of
    a calibrator this component did not fit, and re-deriving them from the fold would be
    asserting what Component 9 recorded instead of reading it.
    """
    if path is None:
        return {}
    frame = (
        pl.read_parquet(path)
        .select(
            [
                "base_model_name",
                "fold_id",
                "calibrator_fitted_through",
                "calibrated_prediction_available_from",
            ]
        )
        .unique()
    )
    return {
        (str(r["base_model_name"]), str(r["fold_id"])): (
            r["calibrator_fitted_through"],
            r["calibrated_prediction_available_from"],
        )
        for r in frame.to_dicts()
    }


# --- row construction --------------------------------------------------------


def _value_rows(
    attribution: FoldAttribution,
    model: RefitModel,
    positions: Sequence[int],
) -> list[dict[str, object]]:
    """The long grain: one row per (model, fold, inspection, feature)."""
    spec = model.spec
    raw = model.raw_matrix[list(positions)]
    transformed = model.matrix[list(positions)]
    rows: list[dict[str, object]] = []
    for row_index, row_id in enumerate(attribution.row_ids):
        prediction = float(attribution.output[row_index])
        for column_index, name in enumerate(attribution.feature_names):
            original, derived = origin_of(name)
            raw_value = float(raw[row_index, column_index])
            rows.append(
                {
                    "model_name": spec.name,
                    "model_version": spec.version,
                    "family": spec.family.value,
                    "fold_set": attribution.fold_set,
                    "fold_id": attribution.fold_id,
                    "target_inspection_id": row_id,
                    "feature_name": name,
                    "original_feature_name": original,
                    "derived_from": derived,
                    "feature_kind": kind_of(name).value,
                    # NaN means the source was NULL. Carried as a null rather than as NaN so
                    # a consumer meets a missing value in the shape its query language
                    # already understands.
                    "feature_value": None if np.isnan(raw_value) else raw_value,
                    "transformed_value": float(transformed[row_index, column_index]),
                    "shap_value": float(attribution.values[row_index, column_index]),
                    "output_space": attribution.output_space.value,
                    "explanation_method": attribution.method.value,
                    "is_exact": attribution.is_exact,
                    "base_value": attribution.base_value,
                    "prediction_value": prediction,
                    "trained_through": model.trained_through,
                    "explain_definition_version": EXPLAIN_DEFINITION_VERSION,
                }
            )
    return rows


def _case_rows(
    attribution: FoldAttribution,
    model: RefitModel,
    positions: Sequence[int],
    sample: ExplanationSample,
    reproduction: ReproductionOutcome,
    calibrated: Mapping[tuple[str, str, str], tuple[float, str]],
    horizons: Mapping[tuple[str, str], tuple[object, object]],
) -> list[dict[str, object]]:
    """One row per explained prediction: additivity, provenance and the calibration link."""
    spec = model.spec
    tolerance = tolerance_for(attribution.method)
    residual = attribution.residual
    reconstruction = attribution.reconstruction
    probability = model.probability[list(positions)]
    positive = np.clip(attribution.values, 0.0, None).sum(axis=1)
    negative = np.clip(attribution.values, None, 0.0).sum(axis=1)
    calibrator_end, available_from = horizons.get((spec.name, attribution.fold_id), (None, None))
    is_permutation = attribution.method is attribute.ExplanationMethod.PERMUTATION_SHAP

    rows: list[dict[str, object]] = []
    for index, row_id in enumerate(attribution.row_ids):
        calibration = calibrated.get((spec.name, attribution.fold_id, row_id))
        rows.append(
            {
                "model_name": spec.name,
                "model_version": spec.version,
                "family": spec.family.value,
                "fold_set": attribution.fold_set,
                "fold_id": attribution.fold_id,
                "target_inspection_id": row_id,
                "output_space": attribution.output_space.value,
                "explanation_method": attribution.method.value,
                "is_exact": attribution.is_exact,
                "base_value": attribution.base_value,
                "prediction_value": float(attribution.output[index]),
                "reconstruction_value": float(reconstruction[index]),
                "reconstruction_residual": float(residual[index]),
                "additivity_tolerance": tolerance,
                "additivity_holds": bool(residual[index] <= tolerance),
                "n_features": len(attribution.feature_names),
                "positive_contribution_sum": float(positive[index]),
                "negative_contribution_sum": float(negative[index]),
                "base_score": float(probability[index]),
                "base_score_reproduced": reproduction.passed,
                "calibrated_probability": calibration[0] if calibration else None,
                "calibration_method": calibration[1] if calibration else None,
                "base_model_trained_through": model.trained_through,
                "calibrator_fitted_through": calibrator_end,
                "prediction_available_from": available_from,
                "sample_strategy": sample.strategy,
                "sample_size": len(sample.ids),
                "sampling_seed": sample.seed,
                "sampling_population": sample.population,
                "population_rows": sample.population_rows,
                # Recorded per row because it differs per method: TreeSHAP needs no
                # reference set at all, and writing a size for it would imply one was used.
                "background_strategy": BACKGROUND_STRATEGY if model.background.shape[0] else "",
                "background_size": int(model.background.shape[0]),
                "background_seed": BACKGROUND_SEED if model.background.shape[0] else 0,
                "background_max_date": model.background_max_date,
                "permutation_rounds": PERMUTATION_ROUNDS if is_permutation else 0,
                "explain_definition_version": EXPLAIN_DEFINITION_VERSION,
            }
        )
    return rows


def _support_rows(values: pl.DataFrame, cases: pl.DataFrame) -> list[dict[str, object]]:
    """The machine-readable support matrix, including what was not explained and why."""
    rows: list[dict[str, object]] = []
    for spec in EXPLAIN_REGISTRY:
        supported = spec.status is ExplanationStatus.SUPPORTED
        rows.append(
            {
                "model_name": spec.name,
                "model_version": spec.version,
                "family": spec.family.value,
                "component": spec.component,
                "source_slug": spec.source_slug,
                "explanation_status": spec.status.value,
                "explanation_method": spec.method.value if spec.method else None,
                "output_space": spec.output_space.value if spec.output_space else None,
                "is_exact": spec.is_exact,
                "is_experimental": spec.is_experimental,
                "name_source": spec.name_source,
                "rationale": spec.rationale,
                "unsupported_reason": spec.unsupported_reason or None,
                "explained_rows": (
                    cases.filter(pl.col("model_name") == spec.name).height if supported else 0
                ),
                "attribution_values": (
                    values.filter(pl.col("model_name") == spec.name).height if supported else 0
                ),
            }
        )
    return rows


def _dataclass_rows(records: Sequence[object]) -> list[dict[str, object]]:
    """Frozen dataclasses to writer rows, without restating any field name."""
    import dataclasses

    return [dataclasses.asdict(record) for record in records]  # type: ignore[call-overload]


# --- the run -----------------------------------------------------------------


def run_explanations(
    settings: Settings,
    *,
    features_path: Path,
    prediction_paths: Mapping[str, Path],
    calibrated_path: Path | None = None,
    output_dir: Path | None = None,
    models: Sequence[str] | None = None,
    sample_size: int = SAMPLE_SIZE,
    figures_dir: Path | None = None,
    write_figures: bool = True,
    dry_run: bool = False,
) -> ExplainResult:
    """Explain every supported model on every fold, and write the artifact."""
    started = datetime.now(UTC)
    stamp = started.strftime(TIMESTAMP_FORMAT)
    requested = list(models) if models else list(SUPPORTED_MODELS)

    unsupported = [
        name for name in requested if spec_for(name).status is not ExplanationStatus.SUPPORTED
    ]
    if unsupported:
        raise ExplainBuildError(
            f"cannot explain {', '.join(unsupported)}: "
            + spec_for(unsupported[0]).unsupported_reason
        )

    # Only the artifacts this run actually reads. A checksum recorded for a file nobody
    # opened would be provenance about a run that did not happen.
    sources = required_sources(requested)
    read_paths = {slug: path for slug, path in prediction_paths.items() if slug in sources}
    sha_before = {path.name: compute_sha256(path) for path in read_paths.values()}
    if calibrated_path is not None:
        sha_before[calibrated_path.name] = compute_sha256(calibrated_path)

    frame = _load_features(features_path)
    folds = _build_folds(frame)
    committed = _committed_predictions(prediction_paths, requested)
    calibrated = _calibrated_lookup(calibrated_path)
    horizons = _horizons(calibrated_path)
    logger.info(
        "Explaining %d model(s) over %d folds; %d feature rows",
        len(requested),
        len(folds),
        frame.height,
    )

    samples: list[ExplanationSample] = []
    backgrounds: dict[str, pl.DataFrame] = {}
    for fold in folds:
        samples.append(
            sample_module.select_sample(frame, fold, size=sample_size, seed=SAMPLING_SEED)
        )
        backgrounds[fold.fold_id] = select_background(
            frame, fold, size=BACKGROUND_SIZE, seed=BACKGROUND_SEED
        )
    samples_by_fold = {s.fold_id: s for s in samples}
    report_fold = folds_for_report(folds)

    refit_seconds = 0.0
    attribute_seconds = 0.0
    fitted: list[RefitModel] = []
    attributions: list[FoldAttribution] = []
    reproductions: list[ReproductionOutcome] = []
    value_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    importance_rows = []
    representative = []

    for name in requested:
        spec = spec_for(name)
        for fold in folds:
            sample = samples_by_fold[fold.fold_id]
            model = refit.regenerate_fold(
                spec,
                frame,
                fold,
                sample.ids,
                background_size=BACKGROUND_SIZE,
                background_seed=BACKGROUND_SEED,
            )
            refit_seconds += model.fit_seconds
            reproduction = refit.check_reproduction(model, committed)
            reproductions.append(reproduction)
            if not reproduction.passed:
                # Before a single attribution is computed. A validation report alone would
                # let the artifact be written and merely complain about it.
                raise ExplainBuildError(
                    f"{name}/{fold.fold_id}: {reproduction.mismatches} of "
                    f"{reproduction.rows} re-executed test scores differ from the committed "
                    "artifact. The model being explained is not the model that produced the "
                    "committed prediction, so no attribution from it would mean anything. "
                    f"First: {reproduction.offenders[0] if reproduction.offenders else 'n/a'}"
                )

            positions = sample_module.row_positions(sample, model.row_ids)
            attribution = attribute.attribute_fold(
                model, positions, rounds=PERMUTATION_ROUNDS, seed=SAMPLING_SEED
            )
            attribute_seconds += attribution.seconds
            fitted.append(model)
            attributions.append(attribution)

            value_rows.extend(_value_rows(attribution, model, positions))
            case_rows.extend(
                _case_rows(
                    attribution,
                    model,
                    positions,
                    sample,
                    reproduction,
                    calibrated,
                    horizons,
                )
            )
            importance_rows.extend(aggregate.fold_importance(attribution))

            if fold.fold_id == report_fold:
                base_scores = {
                    row_id: float(model.probability[position])
                    for row_id, position in zip(attribution.row_ids, positions, strict=True)
                }
                representative.extend(
                    aggregate.representative_cases(
                        attribution,
                        base_scores=base_scores,
                        calibrated={
                            row_id: calibrated[(name, fold.fold_id, row_id)]
                            for row_id in attribution.row_ids
                            if (name, fold.fold_id, row_id) in calibrated
                        },
                    )
                )

    importance_rows.extend(aggregate.aggregate_importance(importance_rows))
    stability_rows = aggregate.stability(importance_rows)
    drift_rows = aggregate.drift(importance_rows)

    tables = {
        "explanation_values": writer.finalize(value_rows, "explanation_values"),
        "explanation_cases": writer.finalize(case_rows, "explanation_cases"),
        "explanation_importance": writer.finalize(
            _dataclass_rows(importance_rows), "explanation_importance"
        ),
        "explanation_stability": writer.finalize(
            _dataclass_rows(stability_rows), "explanation_stability"
        ),
        "explanation_drift": writer.finalize(_dataclass_rows(drift_rows), "explanation_drift"),
        "explanation_representative_cases": writer.finalize(
            _dataclass_rows(representative), "explanation_representative_cases"
        ),
    }
    tables["explanation_support"] = writer.finalize(
        _support_rows(tables["explanation_values"], tables["explanation_cases"]),
        "explanation_support",
    )

    destination = output_dir or settings.explanations_processed_dir
    written: list[Path] = []
    values_path: Path | None = None
    if not dry_run:
        for table, frame_out in sorted(tables.items()):
            path = destination / f"{table}_{stamp}.parquet"
            writer.write_table(frame_out, path)
            written.append(path)
            if table == writer.DATASET_SLUG:
                values_path = path

    sha_after = {path.name: compute_sha256(path) for path in read_paths.values()}
    if calibrated_path is not None:
        sha_after[calibrated_path.name] = compute_sha256(calibrated_path)

    checks = validate.validate_explanations(
        frame,
        folds,
        fitted,
        attributions,
        samples,
        reproductions,
        backgrounds,
        tables,
        committed,
        sha_before,
        sha_after,
        expected_models=requested,
    )

    fold_sets: dict[str, int] = {}
    for fold in folds:
        fold_sets[fold.fold_set] = fold_sets.get(fold.fold_set, 0) + 1
    stats = ExplainStats(
        folds=len(folds),
        fold_sets=fold_sets,
        feature_rows=frame.height,
        models_supported=len(requested),
        # Genuinely unsupported, not merely unrequested. A model left out by --models is
        # skipped; a model in this count could not be explained at all, and conflating the
        # two would overstate what the boundary costs.
        models_unsupported=sum(
            1 for x in EXPLAIN_REGISTRY if x.status is ExplanationStatus.UNSUPPORTED
        ),
        refits=len(fitted),
        explained_rows=tables["explanation_cases"].height,
        attribution_values=tables["explanation_values"].height,
        reproduction_rows=sum(r.rows for r in reproductions),
        reproduction_mismatches=sum(r.mismatches for r in reproductions),
        refit_seconds=refit_seconds,
        attribute_seconds=attribute_seconds,
    )

    figure_paths: list[Path] = []
    if write_figures and not dry_run:
        from sentinel.explain.figures import render

        figure_paths = render(tables, destination=figures_dir or FIGURES_DIR)

    manifest_path: Path | None = None
    if not dry_run and values_path is not None:
        manifest_path = _write_manifest(
            tables,
            written,
            checks,
            stats,
            started=started,
            features_path=features_path,
            prediction_paths=read_paths,
            calibrated_path=calibrated_path,
            sha_before=sha_before,
            sha_after=sha_after,
            reproductions=reproductions,
            attributions=attributions,
            requested=requested,
            sample_size=sample_size,
            values_path=values_path,
        )

    return ExplainResult(
        checks=checks,
        stats=stats,
        tables=tables,
        values_path=values_path,
        manifest_path=manifest_path,
        written=written,
        figure_paths=figure_paths,
        dry_run=dry_run,
    )


def folds_for_report(folds: Sequence[FoldSpec]) -> str:
    """The fold representative local cases and figures are drawn on.

    The last quarterly fold, which is the project's standing choice since Component 8 --
    the most recent window, and never ``covid_shift``, whose behaviour every component so
    far has found to diverge.
    """
    quarterly = [f.fold_id for f in folds if f.fold_set == REPRESENTATIVE_FOLD_SET]
    if not quarterly:
        raise ExplainBuildError("no quarterly fold to draw representative cases from")
    return sorted(quarterly)[-1]


def _name_of(paths: Mapping[str, Path], slug: str) -> str | None:
    """The file name for a source this run read, or ``None`` if it read none.

    ``None`` rather than an empty string: a run that explained only Component 7's boosters
    genuinely has no Component 8 input, and spelling that as "" would be indistinguishable
    from a path that failed to record.
    """
    path = paths.get(slug)
    return path.name if path is not None else None


def _sha_of(paths: Mapping[str, Path], checksums: Mapping[str, str], slug: str) -> str | None:
    path = paths.get(slug)
    return checksums.get(path.name) if path is not None else None


def _blas_threads() -> str:
    """The BLAS thread count in force, as ADR 0026 requires it be recorded rather than assumed."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(name)
        if value:
            return f"{name}={value}"
    return "unset (library default)"


def _write_manifest(
    tables: Mapping[str, pl.DataFrame],
    written: Sequence[Path],
    checks: Sequence[ValidationCheck],
    stats: ExplainStats,
    *,
    started: datetime,
    features_path: Path,
    prediction_paths: Mapping[str, Path],
    calibrated_path: Path | None,
    sha_before: Mapping[str, str],
    sha_after: Mapping[str, str],
    reproductions: Sequence[ReproductionOutcome],
    attributions: Sequence[FoldAttribution],
    requested: Sequence[str],
    sample_size: int,
    values_path: Path,
) -> Path:
    """Provenance sufficient to answer 'what was this model allowed to know' unaided."""
    import lightgbm
    import numpy
    import sklearn
    import torch
    import xgboost

    by_name = {path.name: path for path in written}
    artifacts = [
        ArtifactRecord(
            path=path.name,
            bytes=path.stat().st_size,
            sha256=compute_sha256(path),
            row_count=tables[path.name.rsplit("_", 1)[0]].height,
            schema=writer.schema_of(tables[path.name.rsplit("_", 1)[0]]),
        )
        for path in sorted(by_name.values())
    ]

    rows_by_model: dict[str, int] = {}
    mismatches_by_model: dict[str, int] = {}
    for outcome in reproductions:
        rows_by_model[outcome.model_name] = rows_by_model.get(outcome.model_name, 0) + outcome.rows
        mismatches_by_model[outcome.model_name] = (
            mismatches_by_model.get(outcome.model_name, 0) + outcome.mismatches
        )

    worst: dict[str, float] = {}
    for attribution in attributions:
        residual = attribution.residual
        value = float(residual.max()) if len(residual) else 0.0
        key = attribution.method.value
        worst[key] = max(worst.get(key, 0.0), value)

    manifest = ExplainManifest(
        code_version=__version__,
        explain_definition_version=EXPLAIN_DEFINITION_VERSION,
        built_at=started.isoformat(),
        features_path=features_path.name,
        features_sha256=compute_sha256(features_path),
        feature_definition_version=FEATURE_DEFINITION_VERSION,
        baseline_predictions_path=_name_of(prediction_paths, "baseline_predictions"),
        baseline_predictions_sha256=_sha_of(prediction_paths, sha_before, "baseline_predictions"),
        boosted_predictions_path=_name_of(prediction_paths, "boosted_predictions"),
        boosted_predictions_sha256=_sha_of(prediction_paths, sha_before, "boosted_predictions"),
        neural_predictions_path=_name_of(prediction_paths, "neural_predictions"),
        neural_predictions_sha256=_sha_of(prediction_paths, sha_before, "neural_predictions"),
        calibrated_predictions_path=calibrated_path.name if calibrated_path else None,
        calibrated_predictions_sha256=(
            sha_before[calibrated_path.name] if calibrated_path else None
        ),
        prediction_artifacts_unchanged=dict(sha_before) == dict(sha_after),
        prediction_sha256_after=dict(sha_after),
        reproduction_rows=rows_by_model,
        reproduction_mismatches=mismatches_by_model,
        reproduction_passed=all(o.passed for o in reproductions),
        supported_models=list(requested),
        unsupported_models={
            s.name: s.unsupported_reason
            for s in EXPLAIN_REGISTRY
            if s.status is ExplanationStatus.UNSUPPORTED
        },
        explanation_methods={
            s.name: s.method.value for s in EXPLAIN_REGISTRY if s.method is not None
        },
        output_spaces={
            s.name: s.output_space.value for s in EXPLAIN_REGISTRY if s.output_space is not None
        },
        exactness={s.name: s.is_exact for s in EXPLAIN_REGISTRY},
        name_sources={s.name: s.name_source for s in EXPLAIN_REGISTRY},
        sample_strategy=SAMPLE_STRATEGY,
        sample_size=sample_size,
        sampling_seed=SAMPLING_SEED,
        sampling_population=SAMPLING_POPULATION,
        background_strategy=BACKGROUND_STRATEGY,
        background_size=BACKGROUND_SIZE,
        background_seed=BACKGROUND_SEED,
        permutation_rounds=PERMUTATION_ROUNDS,
        additivity_tolerance={
            method.value: tolerance_for(method) for method in attribute.ExplanationMethod
        },
        max_additivity_residual=worst,
        top_k=TOP_K,
        rank_drift_threshold=RANK_DRIFT_THRESHOLD,
        representative_quantiles=dict(REPRESENTATIVE_QUANTILES),
        stability_metrics=list(aggregate.STABILITY_METRICS),
        covid_reported_separately=True,
        attribution_semantics=ATTRIBUTION_SEMANTICS,
        calibration_boundary=CALIBRATION_BOUNDARY,
        causality_disclaimer=CAUSALITY_DISCLAIMER,
        fold_sets=stats.fold_sets,
        folds=stats.folds,
        feature_rows=stats.feature_rows,
        refits=stats.refits,
        explained_rows=stats.explained_rows,
        attribution_values=stats.attribution_values,
        refit_seconds_total=round(stats.refit_seconds, 2),
        attribute_seconds_total=round(stats.attribute_seconds, 2),
        sklearn_version=sklearn.__version__,
        numpy_version=numpy.__version__,
        xgboost_version=xgboost.__version__,
        lightgbm_version=lightgbm.__version__,
        torch_version=torch.__version__,
        torch_threads=torch.get_num_threads(),
        blas_threads=_blas_threads(),
        device="cpu",
        determinism_caveat=DETERMINISM_CAVEAT,
        blocked=[f"{name}: {reason}" for name, reason in sorted(BLOCKED_EXPERIMENTS.items())],
        artifacts=artifacts,
        checks=[
            {
                "name": c.name,
                "passed": str(c.passed),
                "severity": c.severity,
                "detail": c.detail,
            }
            for c in checks
        ],
    )
    path = manifest_path_for(values_path)
    write_manifest(manifest, path)
    return path


def summarize(result: ExplainResult) -> str:
    """A short human-readable summary of one run."""
    stats = result.stats
    lines = [
        f"Explained {stats.models_supported} model(s) over {stats.folds} folds "
        f"({', '.join(f'{k}={v}' for k, v in sorted(stats.fold_sets.items()))})",
        f"  {stats.refits} re-executed fits in {stats.refit_seconds:.1f}s; "
        f"attribution {stats.attribute_seconds:.1f}s",
        f"  bit-identity gate: {stats.reproduction_rows - stats.reproduction_mismatches}/"
        f"{stats.reproduction_rows} test scores identical to the committed artifacts",
        f"  {stats.explained_rows:,} explained predictions, "
        f"{stats.attribution_values:,} attribution values",
        f"  {stats.models_unsupported} model(s) reported unsupported, with a stated reason",
    ]
    if result.dry_run:
        lines.append("  dry run: nothing written")
    else:
        for path in result.written:
            lines.append(f"  wrote {path}")
        if result.manifest_path:
            lines.append(f"  wrote {result.manifest_path}")
        for path in result.figure_paths:
            lines.append(f"  wrote {path}")
    return "\n".join(lines)


__all__ = [
    "FIGURES_DIR",
    "SOURCE_PREFIX",
    "ExplainBuildError",
    "ExplainResult",
    "folds_for_report",
    "run_explanations",
    "summarize",
]
