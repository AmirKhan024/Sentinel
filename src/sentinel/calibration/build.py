"""Orchestration for Component 9. The only module here that touches the filesystem or the clock.

The order of operations is the component's argument, so it is worth stating before the code:

    for each candidate, for each fold in calibration order:
        1. re-execute Components 6/7/8's unchanged fit          (ADR 0026)
        2. GATE: regenerated test scores == committed, bit for bit -- or stop
        3. cut the calibration window chronologically            (ADR 0025)
        4. fit both methods on the inner-fit portion
        5. score both on the inner-select portion
        6. choose on the expanding prefix of folds 1..k
        7. refit both on the FULL calibration window
        8. freeze, and apply to the test window
        9. measure -- drift, decomposition, ranking, bootstrap

Step 2 is a gate rather than a check. If the re-executed model is not the committed model
then every calibrator below it is a correction to nothing, so ``run_calibration`` raises
before fitting a single one; a validation report alone would let the artifact be written and
merely complain about it.

Step 6 is the one that is easy to get wrong. The prefix is folds 1..k of the *same fold set*,
never a pool over all folds, because **fold N's calibration window is fold N-1's test
window** -- pooling would choose fold 1's method using fold 1's test period.

Nothing between steps 1 and 8 reads a test label. Test labels enter only at step 9, once the
calibrator is frozen, and only to measure.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from sentinel import __version__
from sentinel.calibration import basescores, figures, predict, train, validate, writer
from sentinel.calibration import metrics as cal_metrics
from sentinel.calibration.definitions import (
    AVAILABLE_FROM_SEMANTICS,
    BLOCKED_EXPERIMENTS,
    BOOTSTRAP_CAVEAT,
    BOOTSTRAP_REPLICATIONS,
    BOOTSTRAP_SCHEMES,
    BOOTSTRAP_SEED,
    CALIBRATION_DEFINITION_VERSION,
    CANDIDATE_REGISTRY,
    EXCLUDED_MODELS,
    INNER_SELECT_FRACTION,
    ISOTONIC_PARAMS,
    MARGIN_TOLERANCE,
    MIN_INNER_FIT_ROWS,
    MIN_INNER_SELECT_ROWS,
    PLATT_PARAMS,
    PROBABILITY_SEMANTICS,
    SCORE_DIRECTION,
    SELECTION_GRANULARITY,
    SELECTION_METRIC,
    SELECTION_RULE,
    STAGE_SELECTED,
    STAGE_UNCALIBRATED,
    TIE_PREFERENCE,
    TIE_THRESHOLD,
    TRAINED_THROUGH_SEMANTICS,
    CandidateSpec,
    Method,
    candidate_index,
    spec_for,
)
from sentinel.calibration.models import (
    ArtifactRecord,
    BaseScores,
    CalibrationManifest,
    CalibrationStats,
    FittedCalibrator,
    InnerSplit,
    MethodTrial,
    SelectionOutcome,
    ValidationCheck,
)
from sentinel.calibration.preprocess import (
    calibration_frame,
    clamped_count,
    logit,
    split_calibration_window,
)
from sentinel.config import Settings
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.metrics import (
    DEFAULT_CALIBRATION_BINS,
    brier,
    calibration_bins,
    ece,
    log_loss,
    mce,
)
from sentinel.evaluation.models import FoldSpec
from sentinel.manifest import compute_sha256, manifest_path_for, write_manifest

logger = logging.getLogger(__name__)

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

#: Figures are documentation rather than data, so they sit beside the findings document that
#: reads them -- the same literal ``neural/build.py`` uses.
FIGURES_DIR = Path("docs/analysis/figures")

DETERMINISM_CAVEAT = (
    "Identical output for a fixed feature table, a fixed row order, a fixed library set, one "
    "torch thread and CPU -- NOT across library versions, and not across BLAS thread counts. "
    "The bit-identity gate is sensitive to both: an OMP_NUM_THREADS override alone moves "
    "logistic_regression's scores by up to 5e-10, which this gate correctly rejects. Run "
    "without a thread override, matching the committed manifests' 'unset (library default)'."
)


class CalibrationBuildError(RuntimeError):
    """Raised when a calibration run cannot produce a trustworthy artifact."""


@dataclass
class CalibrationResult:
    """Everything one run produced, whether or not it was written."""

    tables: dict[str, pl.DataFrame]
    checks: list[ValidationCheck]
    manifest: CalibrationManifest
    stats: CalibrationStats
    folds: list[FoldSpec]
    calibrators: list[FittedCalibrator]
    selections: list[SelectionOutcome]
    predictions_path: Path | None = None
    manifest_path: Path | None = None
    figure_paths: list[Path] | None = None


def _blas_threads() -> str:
    import os

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(name)
        if value:
            return f"{name}={value}"
    return "unset (library default)"


def _build_folds(frame: pl.DataFrame) -> list[FoldSpec]:
    """The same 18 folds every component builds, rebuilt rather than read back.

    Component 5 is the definition; re-deriving it here means a fold table on disk cannot
    drift from the fold a calibrator was fitted for.
    """
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    if start is None or end is None:
        raise CalibrationBuildError("feature table has no usable reference dates")
    quarterly = folds_module.quarterly_folds(data_start=start, data_end=end)
    covid = folds_module.covid_shift_fold(data_end=end)
    if not quarterly:
        raise CalibrationBuildError("no complete quarterly fold in this snapshot")
    return [*quarterly, *covid]


def _subset(values: Sequence[float], index: Sequence[int]) -> list[float]:
    return [values[i] for i in index]


def _labels_at(labels: Sequence[int], index: Sequence[int]) -> list[int]:
    return [labels[i] for i in index]


def run_calibration(
    settings: Settings,
    *,
    features_path: Path,
    categoricals_path: Path | None = None,
    prediction_paths: Mapping[str, Path] | None = None,
    output_dir: Path | None = None,
    models: Sequence[str] | None = None,
    method_override: Method | None = None,
    bootstrap_replications: int = BOOTSTRAP_REPLICATIONS,
    figures_dir: Path | None = None,
    write_figures: bool = True,
    dry_run: bool = False,
) -> CalibrationResult:
    """Calibrate every requested candidate on every fold, and measure what changed."""
    started = datetime.now(UTC)
    frame = pl.read_parquet(features_path).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )
    folds = _build_folds(frame)
    fold_by_id = {fold.fold_id: fold for fold in folds}
    # Position within a FOLD SET, which is what the expanding prefix indexes.
    set_ordering: dict[str, int] = {}
    for fold_set in {f.fold_set for f in folds}:
        for index, fold in enumerate(f for f in folds if f.fold_set == fold_set):
            set_ordering[fold.fold_id] = index

    candidates = [spec_for(name) for name in models] if models else list(CANDIDATE_REGISTRY)
    needs_categoricals = any(c.family.value == "neural_embedding_booster" for c in candidates)
    categoricals = (
        pl.read_parquet(categoricals_path)
        if categoricals_path is not None and needs_categoricals
        else None
    )
    if needs_categoricals and categoricals is None:
        raise CalibrationBuildError(
            "xgboost_chain_embeddings needs Component 8's categorical table; pass "
            "--categoricals or drop it with --models"
        )

    paths = dict(prediction_paths or {})
    committed = {slug: pl.read_parquet(path) for slug, path in paths.items()}

    stats = CalibrationStats(
        feature_rows=frame.height,
        folds=len(folds),
        fold_sets={
            fs: sum(1 for f in folds if f.fold_set == fs) for fs in {f.fold_set for f in folds}
        },
        candidates=[c.name for c in candidates],
    )

    rows: dict[str, list[dict[str, object]]] = {name: [] for name in writer.SCHEMAS}
    calibrators: list[FittedCalibrator] = []
    # Both methods are persisted for every fold, so both are checkable and both must be
    # monotone -- a consumer can apply either from the artifact. `calibrators` holds only
    # the frozen one, because "one production calibrator per fold" is its own invariant.
    all_calibrators: list[FittedCalibrator] = []
    selections: list[SelectionOutcome] = []
    fit_ids: dict[tuple[str, str], list[str]] = {}
    inner_splits: dict[tuple[str, str], InnerSplit] = {}
    inner_dates: dict[tuple[str, str], list[object]] = {}
    mismatches: dict[str, int] = {}
    gate_offenders: list[str] = []
    margin_error: dict[str, float] = {}
    margin_unavailable: list[str] = []
    rows_compared = 0
    drift_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []

    for candidate in candidates:
        history: list[dict[Method, MethodTrial]] = []
        mismatches.setdefault(candidate.name, 0)

        for fold in sorted(
            (f for f in folds), key=lambda f: (f.fold_set, f.calibration_end)
        ):
            base = basescores.regenerate_fold(
                candidate, frame, fold, categoricals=categoricals
            )
            stats.base_fits += 1
            stats.refit_seconds_total += base.fit_seconds

            # --- the gate (ADR 0026) --------------------------------------
            reference = committed.get(candidate.source_slug)
            if reference is not None:
                ref = basescores.committed_test_scores(reference, candidate.name, fold.fold_id)
                count, offenders = basescores.reproduction_mismatches(base, ref)
                mismatches[candidate.name] += count
                gate_offenders.extend(offenders)
                rows_compared += len(base.test_ids)

            finite_margins = [m for m in base.calibration_margins if np.isfinite(m)]
            if finite_margins:
                worst = max(
                    abs(logit(p) - m)
                    for p, m in zip(base.calibration_scores, base.calibration_margins, strict=True)
                    if np.isfinite(m)
                )
                margin_error[candidate.name] = max(margin_error.get(candidate.name, 0.0), worst)
            elif candidate.name not in margin_unavailable:
                margin_unavailable.append(candidate.name)

            stats.logit_clamped_rows += clamped_count(base.calibration_scores)
            stats.logit_clamped_rows += clamped_count(base.test_scores)
            stats.calibration_rows += len(base.calibration_ids)

            window = calibration_frame(frame, fold)
            split = split_calibration_window(window, fold, fraction=INNER_SELECT_FRACTION)
            key = (candidate.name, fold.fold_id)
            inner_splits[key] = split
            inner_dates[key] = list(base.calibration_dates)
            fit_ids[key] = list(base.calibration_ids)

            trials = {
                method: train.trial(
                    method,
                    model_name=candidate.name,
                    fold=fold,
                    fold_index=set_ordering[fold.fold_id],
                    inner_split_date=split.cut,
                    fit_labels=_labels_at(base.calibration_labels, split.fit_index),
                    fit_probabilities=_subset(base.calibration_scores, split.fit_index),
                    select_labels=_labels_at(base.calibration_labels, split.select_index),
                    select_probabilities=_subset(base.calibration_scores, split.select_index),
                )
                for method in Method
            }
            history.append(trials)
            outcome = train.select_method(history, override=method_override)
            selections.append(outcome)
            stats.method_counts[outcome.method.value] = (
                stats.method_counts.get(outcome.method.value, 0) + 1
            )

            # Both methods are refitted on the FULL calibration window and both are
            # persisted, even though only one is applied. The counterfactual has to stay
            # answerable from the artifact rather than by re-running with a flag.
            fitted = {
                method: train.fit_method(
                    method,
                    list(base.calibration_labels),
                    list(base.calibration_scores),
                    model_name=candidate.name,
                    fold=fold,
                    fit_start=fold.calibration_start,
                    fit_end=fold.calibration_end,
                )
                for method in Method
            }
            calibrators.append(fitted[outcome.method])
            all_calibrators.extend(fitted.values())
            stats.calibrator_fits += len(fitted)

            _emit_selection(rows, trials, outcome)
            _emit_parameters(rows, fitted, outcome)

            applied = {
                STAGE_UNCALIBRATED: list(base.test_scores),
                **{m.value: predict.apply(fitted[m], base.test_scores) for m in Method},
            }
            applied[STAGE_SELECTED] = applied[outcome.method.value]

            _emit_predictions(rows, candidate, fold, base, applied[outcome.method.value], outcome)
            _emit_base_scores(rows, candidate, base, split, ref if reference is not None else None)

            fold_drift, fold_ranking = _measure(
                rows,
                candidate,
                fold,
                base,
                applied,
                window,
                frame,
                set_ordering[fold.fold_id],
                bootstrap_replications,
            )
            drift_rows.extend(fold_drift)
            ranking_rows.extend(fold_ranking)

    stats.calibrated_prediction_rows = len(rows["calibrated_predictions"])
    stats.selection_rows = len(rows["calibrator_selection"])
    stats.method_switches = _count_switches(selections)

    # --- the gate is a gate, not a check --------------------------------------
    if sum(mismatches.values()) and not dry_run:
        raise CalibrationBuildError(
            f"{sum(mismatches.values())} regenerated test-window score(s) differ from the "
            "committed Component 6/7/8 artifacts, so the re-executed models are not the "
            "published ones and every calibrator below them would be a correction to "
            f"nothing. First: {gate_offenders[0] if gate_offenders else 'n/a'}. See ADR 0026 "
            "-- a tolerance is not the remedy."
        )

    tables = {name: writer.finalize(rows[name], name) for name in writer.SCHEMAS}
    checks = _validate(
        tables,
        frame,
        fold_by_id,
        candidates,
        calibrators,
        all_calibrators,
        selections,
        fit_ids,
        inner_splits,
        inner_dates,
        mismatches,
        gate_offenders,
        rows_compared,
        margin_error,
        margin_unavailable,
        stats,
        set_ordering,
        method_override,
        drift_rows,
        ranking_rows,
    )

    figure_paths: list[Path] = []
    if write_figures and not dry_run:
        figure_paths = figures.render(
            tables,
            destination=figures_dir or FIGURES_DIR,
            candidates=candidates,
        )

    manifest, predictions_path, manifest_path = _write(
        settings,
        tables,
        checks,
        stats,
        started,
        features_path=features_path,
        categoricals_path=categoricals_path if categoricals is not None else None,
        prediction_paths=paths,
        output_dir=output_dir,
        candidates=candidates,
        selections=selections,
        method_override=method_override,
        bootstrap_replications=bootstrap_replications,
        margin_error=margin_error,
        mismatches=mismatches,
        dry_run=dry_run,
    )

    return CalibrationResult(
        tables=tables,
        checks=checks,
        manifest=manifest,
        stats=stats,
        folds=folds,
        calibrators=calibrators,
        selections=selections,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        figure_paths=figure_paths,
    )


def _count_switches(selections: Sequence[SelectionOutcome]) -> int:
    """How many times the frozen method changed from one fold to the next.

    Reported because a series whose method changes underneath it makes the drift plot
    ambiguous: an ECE step could be drift or could be the switch.
    """
    switches = 0
    previous: dict[tuple[str, str], Method] = {}
    for outcome in sorted(selections, key=lambda o: (o.model_name, o.fold_set, o.fold_index)):
        key = (outcome.model_name, outcome.fold_set)
        if key in previous and previous[key] is not outcome.method:
            switches += 1
        previous[key] = outcome.method
    return switches


def _emit_selection(
    rows: dict[str, list[dict[str, object]]],
    trials: Mapping[Method, MethodTrial],
    outcome: SelectionOutcome,
) -> None:
    for method, entry in trials.items():
        other = Method.ISOTONIC if method is Method.PLATT else Method.PLATT
        rows["calibrator_selection"].append(
            {
                "model_name": entry.model_name,
                "fold_set": entry.fold_set,
                "fold_id": entry.fold_id,
                "fold_index": entry.fold_index,
                "method": method.value,
                "inner_fit_rows": entry.inner_fit_rows,
                "inner_select_rows": entry.inner_select_rows,
                "inner_split_date": entry.inner_split_date,
                "inner_fit_positive_rate": entry.inner_fit_positive_rate,
                "inner_select_positive_rate": entry.inner_select_positive_rate,
                "inner_select_log_loss": entry.inner_select_log_loss,
                "inner_select_brier": entry.inner_select_brier,
                "inner_select_ece": entry.inner_select_ece,
                "inner_select_mce": entry.inner_select_mce,
                "prefix_mean_log_loss": outcome.prefix_mean_log_loss[method],
                "per_fold_winner": outcome.per_fold_winner is method,
                "prefix_winner": outcome.method is method,
                "gap_to_other": (
                    outcome.prefix_mean_log_loss[method] - outcome.prefix_mean_log_loss[other]
                ),
                "tie_threshold": TIE_THRESHOLD,
                "declared_tie": outcome.declared_tie,
                "selection_reason": outcome.reason,
                "seconds": entry.seconds,
                "calibration_definition_version": CALIBRATION_DEFINITION_VERSION,
            }
        )


def _emit_parameters(
    rows: dict[str, list[dict[str, object]]],
    fitted: Mapping[Method, FittedCalibrator],
    outcome: SelectionOutcome,
) -> None:
    for method, calibrator in fitted.items():
        selected = outcome.method is method
        common = {
            "model_name": calibrator.model_name,
            "fold_set": calibrator.fold_set,
            "fold_id": calibrator.fold_id,
            "method": method.value,
            "fit_rows": calibrator.fit_rows,
            "fit_positive_rate": calibrator.fit_positive_rate,
            "fit_start": calibrator.fit_start,
            "fit_end": calibrator.fit_end,
            "input_transform": calibrator.input_transform,
            "was_selected": selected,
            "calibration_definition_version": CALIBRATION_DEFINITION_VERSION,
        }
        if method is Method.PLATT:
            terms = (("coef", calibrator.coefficient), ("intercept", calibrator.intercept))
            for term, value in terms:
                rows["calibrator_parameters"].append({**common, "term": term, "value": value})
        else:
            rows["calibrator_parameters"].append(
                {**common, "term": "breakpoint_count", "value": float(calibrator.breakpoint_count)}
            )
            for index, (x, y) in enumerate(
                zip(calibrator.x_thresholds, calibrator.y_thresholds, strict=True)
            ):
                rows["calibrator_isotonic_breakpoints"].append(
                    {
                        "model_name": calibrator.model_name,
                        "fold_set": calibrator.fold_set,
                        "fold_id": calibrator.fold_id,
                        "breakpoint_index": index,
                        "x_threshold": x,
                        "y_threshold": y,
                        "x_min": calibrator.x_min,
                        "x_max": calibrator.x_max,
                        "breakpoint_count": calibrator.breakpoint_count,
                        "was_selected": selected,
                        "calibration_definition_version": CALIBRATION_DEFINITION_VERSION,
                    }
                )


def _emit_predictions(
    rows: dict[str, list[dict[str, object]]],
    candidate: CandidateSpec,
    fold: FoldSpec,
    base: BaseScores,
    calibrated: Sequence[float],
    outcome: SelectionOutcome,
) -> None:
    from datetime import timedelta

    available_from = fold.calibration_end + timedelta(days=1)
    if available_from != fold.test_start:  # pragma: no cover - FoldSpec forbids a gap
        raise CalibrationBuildError(
            f"{fold.fold_id}: calibration_end + 1 day is {available_from}, not test_start "
            f"{fold.test_start}"
        )
    for row_id, base_score, score in zip(base.test_ids, base.test_scores, calibrated, strict=True):
        rows["calibrated_predictions"].append(
            {
                "target_inspection_id": row_id,
                "score": score,
                "model_name": candidate.calibrated_name(outcome.method),
                "model_version": candidate.version,
                "fold_set": fold.fold_set,
                "fold_id": fold.fold_id,
                "trained_through": fold.calibration_end,
                "is_probability": True,
                "base_model_name": candidate.name,
                "base_model_version": candidate.version,
                "base_score": base_score,
                "base_model_trained_through": base.base_model_trained_through,
                "calibrator_fitted_through": fold.calibration_end,
                "calibrated_prediction_available_from": fold.test_start,
                "method": outcome.method.value,
                "is_experimental": candidate.is_experimental,
                "calibration_definition_version": CALIBRATION_DEFINITION_VERSION,
            }
        )


def _emit_base_scores(
    rows: dict[str, list[dict[str, object]]],
    candidate: CandidateSpec,
    base: BaseScores,
    split: InnerSplit,
    committed: Mapping[str, float] | None,
) -> None:
    portion = {i: "inner_fit" for i in split.fit_index}
    portion.update({i: "inner_select" for i in split.select_index})

    for index, row_id in enumerate(base.calibration_ids):
        rows["calibration_base_scores"].append(
            {
                "model_name": candidate.name,
                "fold_set": base.fold_set,
                "fold_id": base.fold_id,
                "split": "calibration",
                "inner_portion": portion.get(index, ""),
                "target_inspection_id": row_id,
                "rd": base.calibration_dates[index],
                "base_score": base.calibration_scores[index],
                "base_logit": logit(base.calibration_scores[index]),
                "native_margin": (
                    base.calibration_margins[index]
                    if np.isfinite(base.calibration_margins[index])
                    else None
                ),
                "target": base.calibration_labels[index],
                # Null on purpose: there is nothing committed to compare a calibration row
                # against, which is the whole reason this component exists.
                "reproduces_committed_artifact": None,
                "calibration_definition_version": CALIBRATION_DEFINITION_VERSION,
            }
        )
    for index, row_id in enumerate(base.test_ids):
        rows["calibration_base_scores"].append(
            {
                "model_name": candidate.name,
                "fold_set": base.fold_set,
                "fold_id": base.fold_id,
                "split": "test",
                "inner_portion": "",
                "target_inspection_id": row_id,
                "rd": None,
                "base_score": base.test_scores[index],
                "base_logit": logit(base.test_scores[index]),
                "native_margin": (
                    base.test_margins[index] if np.isfinite(base.test_margins[index]) else None
                ),
                "target": base.test_labels[index],
                "reproduces_committed_artifact": (
                    None if committed is None else base.test_scores[index] == committed[row_id]
                ),
                "calibration_definition_version": CALIBRATION_DEFINITION_VERSION,
            }
        )


def _measure(
    rows: dict[str, list[dict[str, object]]],
    candidate: CandidateSpec,
    fold: FoldSpec,
    base: BaseScores,
    applied: Mapping[str, Sequence[float]],
    window: pl.DataFrame,
    frame: pl.DataFrame,
    fold_index: int,
    replications: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Every test-window measurement, for all four stages.

    This is the only place a test label is read, and it happens after the calibrator is
    frozen. All four stages are measured -- including the method that lost -- so the
    counterfactual is in the artifact rather than one flag away.
    """
    labels = list(base.test_labels)
    stats = folds_module.fold_stats(frame, fold)
    k = int(stats.test_median_daily_capacity or 1)

    test_frame = folds_module.window_frame(frame, fold)
    establishment = dict(
        zip(
            (str(v) for v in test_frame["target_inspection_id"].to_list()),
            (str(v) for v in test_frame["establishment_id"].to_list()),
            strict=True,
        )
    )
    groups = [establishment[row_id] for row_id in base.test_ids]
    tie_break = list(base.test_ids)

    drift_out: list[dict[str, object]] = []
    ranking_out: list[dict[str, object]] = []

    for stage_index, (stage, scores) in enumerate(applied.items()):
        bins = calibration_bins(labels, list(scores))
        total = sum(count for count, _, _ in bins)
        slope = cal_metrics.calibration_slope_intercept(labels, list(scores))
        drift: dict[str, object] = {
            "model_name": candidate.name,
            "fold_set": fold.fold_set,
            "fold_id": fold.fold_id,
            "fold_index": fold_index,
            "stage": stage,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "test_rows": len(labels),
            "test_positive_rate": stats.test_positive_rate,
            "calibration_positive_rate": stats.calibration_positive_rate,
            "prior_shift": (
                None
                if stats.test_positive_rate is None or stats.calibration_positive_rate is None
                else stats.test_positive_rate - stats.calibration_positive_rate
            ),
            "ece": ece(labels, list(scores)),
            "mce": mce(labels, list(scores)),
            "brier": brier(labels, list(scores)),
            "log_loss": log_loss(labels, list(scores)),
            "calibration_slope": slope.slope,
            "calibration_intercept": slope.intercept,
            "mean_predicted": sum(c * p for c, p, _ in bins) / total,
            "observed_rate": sum(c * o for c, _, o in bins) / total,
            "n_bins": len(bins),
            "calibration_definition_version": CALIBRATION_DEFINITION_VERSION,
        }
        rows["calibration_drift"].append(drift)
        drift_out.append(drift)

        decomposition = cal_metrics.brier_decomposition(labels, list(scores))
        rows["calibration_brier_decomposition"].append(
            {
                "model_name": candidate.name,
                "fold_set": fold.fold_set,
                "fold_id": fold.fold_id,
                "stage": stage,
                "n_bins": decomposition.n_bins,
                "binning": "equal_mass",
                "brier": brier(labels, list(scores)),
                "reliability": decomposition.reliability,
                "resolution": decomposition.resolution,
                "uncertainty": decomposition.uncertainty,
                "recomposed": decomposition.recomposed,
                "within_bin_variance": decomposition.within_bin_variance,
                "calibration_definition_version": CALIBRATION_DEFINITION_VERSION,
            }
        )

        if stage != STAGE_UNCALIBRATED:
            preservation = cal_metrics.ranking_preservation(
                list(base.test_scores), list(scores), labels, tie_break, k
            )
            ranking: dict[str, object] = {
                "model_name": candidate.name,
                "fold_set": fold.fold_set,
                "fold_id": fold.fold_id,
                "stage": stage,
                "spearman_rho": preservation.spearman_rho,
                "kendall_tau_b": preservation.kendall_tau_b,
                "inversions": preservation.inversions,
                "distinct_scores_before": preservation.distinct_before,
                "distinct_scores_after": preservation.distinct_after,
                "new_ties_created": preservation.new_ties_created,
                "is_strictly_monotone": preservation.is_strictly_monotone,
                "top_k": preservation.top_k,
                "top_k_name": "k_1_day",
                "top_k_membership_changed": preservation.top_k_membership_changed,
                "precision_at_k_before": preservation.precision_at_k_before,
                "precision_at_k_after": preservation.precision_at_k_after,
                "roc_auc_before": preservation.roc_auc_before,
                "roc_auc_after": preservation.roc_auc_after,
                "calibration_definition_version": CALIBRATION_DEFINITION_VERSION,
            }
            rows["calibration_ranking_preservation"].append(ranking)
            ranking_out.append(ranking)

        # Bootstrapped for the two stages that answer the question -- what the base model
        # claimed, and what the frozen calibrator claims -- rather than for all four. The
        # losing method's point estimates are still written to every table; what is not
        # computed is an interval around a counterfactual, which would double the cost of
        # the run to widen a number nobody acts on.
        #
        # The calibration slope is deliberately NOT bootstrapped: it refits a logistic
        # regression per replication, which is ~10^6 sklearn fits across the run. Its point
        # estimate is in the drift table for every (model, fold, stage).
        if stage not in (STAGE_UNCALIBRATED, STAGE_SELECTED):
            continue

        for metric_index, (metric_name, fn) in enumerate(
            (
                ("ece", lambda y, p: ece(y, list(p))),
                ("brier", lambda y, p: brier(y, list(p))),
                ("log_loss", lambda y, p: log_loss(y, list(p))),
            )
        ):
            for scheme_index, scheme in enumerate(BOOTSTRAP_SCHEMES):
                interval = cal_metrics.bootstrap(
                    labels,
                    list(scores),
                    metric=fn,
                    metric_name=metric_name,
                    scheme=scheme,
                    groups=groups,
                    seed_key=[
                        BOOTSTRAP_SEED,
                        candidate_index(candidate.name),
                        fold_index,
                        stage_index,
                        metric_index * 2 + scheme_index,
                    ],
                    replications=replications,
                )
                rows["calibration_bootstrap"].append(
                    {
                        "model_name": candidate.name,
                        "fold_set": fold.fold_set,
                        "fold_id": fold.fold_id,
                        "stage": stage,
                        "metric": metric_name,
                        "scheme": scheme,
                        "point_estimate": interval.point_estimate,
                        "replications": interval.replications,
                        "seed": interval.seed,
                        "bootstrap_mean": interval.mean,
                        "bootstrap_sd": interval.sd,
                        "ci_lower": interval.lower,
                        "ci_upper": interval.upper,
                        "ci_level": interval.level,
                        "degenerate_replications": interval.degenerate,
                        "calibration_definition_version": CALIBRATION_DEFINITION_VERSION,
                    }
                )

    return drift_out, ranking_out


def _validate(
    tables: Mapping[str, pl.DataFrame],
    frame: pl.DataFrame,
    fold_by_id: Mapping[str, FoldSpec],
    candidates: Sequence[CandidateSpec],
    calibrators: Sequence[FittedCalibrator],
    all_calibrators: Sequence[FittedCalibrator],
    selections: Sequence[SelectionOutcome],
    fit_ids: Mapping[tuple[str, str], Sequence[str]],
    inner_splits: Mapping[tuple[str, str], InnerSplit],
    inner_dates: Mapping[tuple[str, str], Sequence[object]],
    mismatches: Mapping[str, int],
    gate_offenders: Sequence[str],
    rows_compared: int,
    margin_error: Mapping[str, float],
    margin_unavailable: Sequence[str],
    stats: CalibrationStats,
    set_ordering: Mapping[str, int],
    method_override: Method | None,
    drift_rows: Sequence[Mapping[str, object]],
    ranking_rows: Sequence[Mapping[str, object]],
) -> list[ValidationCheck]:
    predictions = tables["calibrated_predictions"]
    checks = [
        validate.base_scores_reproduce_the_committed_artifact(
            mismatches, gate_offenders, rows_compared
        ),
        validate.recovered_logit_matches_the_native_margin(margin_error, list(margin_unavailable)),
        validate.no_probability_was_clamped(stats.logit_clamped_rows),
        validate.no_test_row_enters_any_calibrator_fit(fit_ids, fold_by_id, frame),
        validate.calibrator_fit_rows_lie_in_the_calibration_window(fit_ids, fold_by_id, frame),
        validate.inner_select_is_strictly_later_than_inner_fit(inner_splits, inner_dates),
        validate.method_selection_reads_no_future_fold(selections, fold_by_id, set_ordering),
        validate.folds_never_share_a_calibrator(calibrators),
        *validate.horizons_are_declared_correctly(predictions, fold_by_id),
        validate.calibrated_predictions_cover_every_fold_exactly(
            predictions, fold_by_id, frame, candidates
        ),
        validate.calibrated_scores_are_probabilities(predictions),
        validate.the_calibrator_is_monotone(all_calibrators),
        validate.platt_does_not_change_the_ranking(ranking_rows),
        validate.isotonic_ties_are_counted_not_hidden(ranking_rows, all_calibrators),
        validate.the_persisted_calibrator_reproduces_the_mapping(all_calibrators),
        validate.the_selection_rule_matches_the_frozen_literals(
            {
                "selection_metric": SELECTION_METRIC,
                "tie_threshold": TIE_THRESHOLD,
                "tie_preference": TIE_PREFERENCE.value,
            }
        ),
        validate.calibration_is_reported_honestly(drift_rows),
    ]
    if method_override is not None:
        checks.append(
            ValidationCheck(
                name="selection_was_overridden_on_the_command_line",
                passed=True,
                severity=validate.SEVERITY_WARN,
                detail=(
                    f"--method {method_override.value} was forced, so the pre-registered "
                    "selection rule did not decide anything in this run. Diagnostic only; "
                    "the production artifact is produced without it."
                ),
            )
        )
    return checks


def _write(
    settings: Settings,
    tables: Mapping[str, pl.DataFrame],
    checks: Sequence[ValidationCheck],
    stats: CalibrationStats,
    started: datetime,
    *,
    features_path: Path,
    categoricals_path: Path | None,
    prediction_paths: Mapping[str, Path],
    output_dir: Path | None,
    candidates: Sequence[CandidateSpec],
    selections: Sequence[SelectionOutcome],
    method_override: Method | None,
    bootstrap_replications: int,
    margin_error: Mapping[str, float],
    mismatches: Mapping[str, int],
    dry_run: bool,
) -> tuple[CalibrationManifest, Path | None, Path | None]:
    import lightgbm
    import numpy
    import sklearn
    import torch
    import xgboost

    stamp = started.strftime(TIMESTAMP_FORMAT)
    destinations = {
        "predictions": output_dir or settings.predictions_processed_dir,
        "tuning": output_dir or settings.tuning_processed_dir,
        "calibration": output_dir or settings.calibration_processed_dir,
    }

    artifacts: list[ArtifactRecord] = []
    predictions_path: Path | None = None
    if not dry_run:
        for name, table in tables.items():
            path = destinations[writer.LAYERS[name]] / f"{name}_{stamp}.parquet"
            writer.write_table(table, path)
            if name == writer.DATASET_SLUG:
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

    def _sha(path: Path | None) -> str | None:
        return compute_sha256(path) if path is not None and path.exists() else None

    manifest = CalibrationManifest(
        code_version=__version__,
        calibration_definition_version=CALIBRATION_DEFINITION_VERSION,
        built_at=started.isoformat(),
        features_path=features_path.name,
        features_sha256=_sha(features_path) or "",
        feature_definition_version="v1",
        categoricals_path=categoricals_path.name if categoricals_path else None,
        categoricals_sha256=_sha(categoricals_path),
        baseline_predictions_path=prediction_paths["baseline_predictions"].name,
        baseline_predictions_sha256=_sha(prediction_paths["baseline_predictions"]) or "",
        boosted_predictions_path=prediction_paths["boosted_predictions"].name,
        boosted_predictions_sha256=_sha(prediction_paths["boosted_predictions"]) or "",
        neural_predictions_path=prediction_paths["neural_predictions"].name,
        neural_predictions_sha256=_sha(prediction_paths["neural_predictions"]) or "",
        base_score_reproduction={
            name: ("bit-identical" if count == 0 else f"{count} mismatch(es)")
            for name, count in sorted(mismatches.items())
        },
        base_score_reproduction_passed=sum(mismatches.values()) == 0,
        logit_recovery_max_error=dict(sorted(margin_error.items())),
        logit_clamped_rows=stats.logit_clamped_rows,
        margin_tolerance=MARGIN_TOLERANCE,
        trained_through_semantics=TRAINED_THROUGH_SEMANTICS,
        available_from_semantics=AVAILABLE_FROM_SEMANTICS,
        probability_semantics=PROBABILITY_SEMANTICS,
        score_direction=SCORE_DIRECTION,
        candidates=[c.name for c in candidates],
        experimental_candidates=[c.name for c in candidates if c.is_experimental],
        candidate_rationale={c.name: c.rationale for c in candidates},
        excluded_models=dict(EXCLUDED_MODELS),
        selection_metric=SELECTION_METRIC,
        selection_granularity=SELECTION_GRANULARITY,
        selection_rule=SELECTION_RULE,
        inner_select_fraction=INNER_SELECT_FRACTION,
        min_inner_fit_rows=MIN_INNER_FIT_ROWS,
        min_inner_select_rows=MIN_INNER_SELECT_ROWS,
        tie_threshold=TIE_THRESHOLD,
        tie_preference=TIE_PREFERENCE.value,
        method_override=method_override.value if method_override else None,
        selected_methods={
            f"{o.model_name}/{o.fold_id}": o.method.value
            for o in sorted(selections, key=lambda s: (s.model_name, s.fold_id))
        },
        method_counts=dict(sorted(stats.method_counts.items())),
        method_switches=stats.method_switches,
        platt_params=", ".join(f"{k}={v}" for k, v in PLATT_PARAMS.items()),
        isotonic_params=", ".join(f"{k}={v}" for k, v in ISOTONIC_PARAMS.items()),
        calibration_bins=DEFAULT_CALIBRATION_BINS,
        binning="equal_mass",
        bootstrap_replications=bootstrap_replications,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_schemes=list(BOOTSTRAP_SCHEMES),
        bootstrap_caveat=BOOTSTRAP_CAVEAT,
        fold_sets=stats.fold_sets,
        folds=stats.folds,
        feature_rows=stats.feature_rows,
        base_fits=stats.base_fits,
        calibrator_fits=stats.calibrator_fits,
        calibration_rows=stats.calibration_rows,
        calibrated_prediction_rows=stats.calibrated_prediction_rows,
        refit_seconds_total=round(stats.refit_seconds_total, 3),
        calibrate_seconds_total=round(stats.calibrate_seconds_total, 3),
        sklearn_version=sklearn.__version__,
        numpy_version=numpy.__version__,
        xgboost_version=xgboost.__version__,
        lightgbm_version=lightgbm.__version__,
        torch_version=torch.__version__,
        torch_threads=torch.get_num_threads(),
        blas_threads=_blas_threads(),
        device="cpu",
        determinism_caveat=DETERMINISM_CAVEAT,
        blocked=list(BLOCKED_EXPERIMENTS),
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

    manifest_path: Path | None = None
    if not dry_run and predictions_path is not None:
        manifest_path = manifest_path_for(predictions_path)
        write_manifest(manifest, manifest_path)
    return manifest, predictions_path, manifest_path


def summarize(result: CalibrationResult) -> str:
    """A fixed-width block for the CLI, ending with what was blocked and what was written."""
    stats = result.stats
    sets = ", ".join(f"{k} {v}" for k, v in sorted(stats.fold_sets.items()))
    chosen = ", ".join(f"{k} {v}" for k, v in sorted(stats.method_counts.items()))
    gate = "BIT-IDENTICAL" if result.manifest.base_score_reproduction_passed else "MISMATCH"
    lines = [
        "",
        "Probability calibration",
        "-----------------------",
        f"  feature rows:            {stats.feature_rows}",
        f"  folds:                   {stats.folds} ({sets})",
        f"  candidates:              {', '.join(stats.candidates)}",
        f"  base fits re-executed:   {stats.base_fits} in {stats.refit_seconds_total:.1f}s",
        f"  base scores reproduce:   {gate}",
        f"  calibration rows:        {stats.calibration_rows}",
        f"  calibrators fitted:      {stats.calibrator_fits}",
        f"  method chosen:           {chosen}",
        f"  method switches:         {stats.method_switches}",
        f"  calibrated predictions:  {stats.calibrated_prediction_rows}",
        f"  logit clamped rows:      {stats.logit_clamped_rows}",
    ]
    lines.append("")
    lines.append("  BLOCKED:")
    lines.extend(f"    - {item}" for item in BLOCKED_EXPERIMENTS)
    if result.predictions_path:
        lines.append("")
        lines.append(f"  wrote {result.predictions_path}")
    if result.manifest_path:
        lines.append(f"  wrote {result.manifest_path}")
    for path in result.figure_paths or []:
        lines.append(f"  wrote {path}")
    return "\n".join(lines)


__all__ = [
    "DETERMINISM_CAVEAT",
    "FIGURES_DIR",
    "CalibrationBuildError",
    "CalibrationResult",
    "run_calibration",
    "summarize",
]
