"""Orchestration: one feature table in, a complete temporal evaluation out.

The only module in the package that touches the filesystem, so everything it
calls stays a pure function of its inputs.

Determinism rests on four properties, asserted by the test suite:

1. Fold boundaries are calendar arithmetic over the data's own date range.
2. Every ordering is fully specified -- score descending, then
   ``target_inspection_id`` ascending -- so no result depends on row order.
3. The only randomness is explicitly seeded, and every seed is written into the
   manifest.
4. Each output table is sorted by a declared key before it is written.

Read the estimand before reading any number this produces. Component 5 measures
the **re-ordering** of establishments that were actually inspected. It cannot
evaluate a schedule that would have inspected somebody nobody visited, because
no label exists for that establishment on that day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from sentinel import __version__
from sentinel.config import Settings
from sentinel.evaluation import contract, rankers, sensitivity, simulate, writer
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation import metrics as metrics_module
from sentinel.evaluation import validate as validate_module
from sentinel.evaluation.models import (
    EVALUATION_DEFINITION_VERSION,
    SCORE_DIRECTION,
    TIE_BREAK_COLUMN,
    ArtifactRecord,
    EvaluationManifest,
    FoldSpec,
    FoldStats,
    PredictionSet,
    ScheduleName,
    ValidationCheck,
)
from sentinel.manifest import compute_sha256, manifest_path_for, write_manifest

logger = logging.getLogger(__name__)

DATASET_SLUG = "evaluation_folds"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

#: Stated in the manifest so a consumer never has to infer it from the code, and
#: so the limitation travels attached to the numbers rather than living only in
#: a findings document.
ESTIMAND = (
    "Re-ordering of canvass inspections that actually occurred in the test window. "
    "Capacity, the establishment set and the labels are all held at their observed "
    "values; only the order changes. This is not a causal estimate and says nothing "
    "about establishments that were never inspected."
)

REQUIRED_COLUMNS = (
    "establishment_id",
    "inspection_date",
    "target_inspection_id",
    "target",
    "code_era_phase",
    "feature_definition_version",
)

#: Score producers evaluated by default. All deterministic, none fitted --
#: Component 6 owns the first trained model.
DEFAULT_MODELS: tuple[str, ...] = (
    "business_as_usual",
    "random",
    "days_since_last_canvass",
    "priority_at_last_canvass",
    "prior_canvass_priority_rate",
    "constant",
)

#: Rankers excluded from the time-invariance sensitivity run: one is noise and
#: the other is a tie-breaking diagnostic, so a band around either says nothing.
SENSITIVITY_EXCLUDED: frozenset[str] = frozenset({"random", "constant"})

#: Experiments the current data cannot support. Reported, never faked.
BLOCKED_EXPERIMENTS: tuple[str, ...] = (
    "temperature covariate in the time-invariance model: no weather data is "
    "ingested (needs NOAA GHCN USW00094846, a Component 1 extension)",
    "faithful CDPH 2015 replication: 7 of that model's 10 input families (inspector "
    "identity, 311, crime, licence, weather, facility type, risk category) are not in "
    "the data contract at all, so it is not reachable. Component 6 ships an explicitly "
    "labelled approximation instead; see docs/data_contracts/baseline_predictions.md",
    "duplicate-record sensitivity: Component 1 owns raw treatment and Component 3 "
    "already collapses same-day canvasses, so there is no Component 5 knob to vary",
)


class EvaluationError(RuntimeError):
    """Raised when the evaluation cannot be run at all."""


@dataclass
class EvaluationResult:
    """Everything a caller needs after a run, written or not."""

    tables: dict[str, pl.DataFrame]
    checks: list[ValidationCheck]
    manifest: EvaluationManifest
    folds: list[FoldSpec]
    stats: list[FoldStats]
    folds_path: Path | None = None
    manifest_path: Path | None = None


def run_evaluation(
    settings: Settings,
    *,
    features_path: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
    folds_only: bool = False,
    random_seeds: int = simulate.DEFAULT_RANDOM_REPLICATIONS,
    sensitivity_replications: int = sensitivity.DEFAULT_REPLICATIONS,
    predictions_path: Path | None = None,
) -> EvaluationResult:
    """Run the rolling-origin backtest and the re-ordering simulation.

    ``predictions_path`` is optional and strictly additive: omitted, this evaluates
    exactly the six built-in heuristics it always has. Supplied, it also reads the
    prediction sets a modelling component wrote and puts them through the same
    validation, the same metrics and the same simulation. The evaluator stays
    model-agnostic -- it never learns what produced a score.
    """
    if not features_path.exists():
        raise FileNotFoundError(f"Component 4 features not found: {features_path}")
    if predictions_path is not None and not predictions_path.exists():
        raise FileNotFoundError(f"Prediction artifact not found: {predictions_path}")

    started = datetime.now(UTC)
    logger.info("Evaluating from %s", features_path.name)

    frame = _load(features_path)
    data_start = folds_module.min_date(frame, "rd")
    data_end = folds_module.max_date(frame, "rd")
    if data_start is None or data_end is None:
        raise EvaluationError("feature table has no usable reference dates")

    quarterly = folds_module.quarterly_folds(data_start=data_start, data_end=data_end)
    if not quarterly:
        raise EvaluationError(
            f"data spans {data_start}..{data_end}, which is too short to build a single "
            "quarterly fold. Folds are never fabricated."
        )
    covid = folds_module.covid_shift_fold(data_end=data_end)
    all_folds = [*quarterly, *covid]

    stats = [folds_module.fold_stats(frame, fold) for fold in all_folds]
    excluded = folds_module.excluded_partial_windows(data_end=data_end, folds=quarterly)

    seeds = list(range(simulate.DEFAULT_RANDOM_SEED, simulate.DEFAULT_RANDOM_SEED + random_seeds))
    observations = validate_module.Observations(
        models=list(DEFAULT_MODELS),
        blocked=list(BLOCKED_EXPERIMENTS),
    )

    tables: dict[str, pl.DataFrame] = {
        "evaluation_folds": _folds_table(all_folds, stats),
    }

    if folds_only:
        for name in (
            "evaluation_metrics",
            "discovery_curves",
            "simulation_summary",
            "seasonality",
            "sensitivity",
        ):
            tables[name] = writer.empty(name)
        observations.models = []
    else:
        seasonality_rows, effects_by_fold = _seasonality(frame, all_folds)
        tables["seasonality"] = writer.finalize(seasonality_rows, "seasonality")

        metric_rows: list[dict[str, object]] = []
        curve_rows: list[dict[str, object]] = []
        simulation_rows: list[dict[str, object]] = []
        sensitivity_rows: list[dict[str, object]] = []

        for fold, fold_stats in zip(all_folds, stats, strict=True):
            external = _external_predictions(predictions_path, fold, fold_stats)
            _evaluate_fold(
                frame=frame,
                fold=fold,
                stats=fold_stats,
                seeds=seeds,
                effects=effects_by_fold.get(fold.fold_id, []),
                sensitivity_replications=sensitivity_replications,
                observations=observations,
                metric_rows=metric_rows,
                curve_rows=curve_rows,
                simulation_rows=simulation_rows,
                sensitivity_rows=sensitivity_rows,
                external=external,
            )

        tables["evaluation_metrics"] = writer.finalize(metric_rows, "evaluation_metrics")
        tables["discovery_curves"] = writer.finalize(curve_rows, "discovery_curves")
        tables["simulation_summary"] = writer.finalize(simulation_rows, "simulation_summary")
        tables["sensitivity"] = writer.finalize(sensitivity_rows, "sensitivity")

    checks = validate_module.validate_evaluation(frame, all_folds, stats, observations)

    destination = output_dir or settings.evaluation_processed_dir
    stamp = started.strftime(TIMESTAMP_FORMAT)
    artifacts: list[ArtifactRecord] = []
    folds_path: Path | None = None

    if not dry_run:
        for name, table in tables.items():
            path = destination / f"{name}_{stamp}.parquet"
            writer.write_table(table, path)
            if name == DATASET_SLUG:
                folds_path = path
            artifacts.append(
                ArtifactRecord(
                    path=path.name,
                    bytes=path.stat().st_size,
                    sha256=compute_sha256(path),
                    row_count=table.height,
                    schema=writer.schema_of(table),
                )
            )

    fold_counts: dict[str, int] = {}
    for fold in all_folds:
        fold_counts[fold.fold_set] = fold_counts.get(fold.fold_set, 0) + 1

    manifest = EvaluationManifest(
        code_version=__version__,
        evaluation_definition_version=EVALUATION_DEFINITION_VERSION,
        built_at=started.isoformat(),
        features_path=features_path.name,
        features_sha256=compute_sha256(features_path),
        feature_definition_version=str(frame["feature_definition_version"][0]),
        predictions_path=predictions_path.name if predictions_path is not None else None,
        predictions_sha256=(
            compute_sha256(predictions_path) if predictions_path is not None else None
        ),
        estimand=ESTIMAND,
        score_direction=SCORE_DIRECTION,
        tie_break_column=TIE_BREAK_COLUMN,
        capacity_semantics=simulate.CAPACITY_SEMANTICS,
        fold_cadence=folds_module.QUARTERLY,
        fold_sets=fold_counts,
        folds=len(all_folds),
        first_test_start=str(quarterly[0].test_start),
        last_test_end=str(quarterly[-1].test_end),
        excluded_partial_windows=excluded,
        models=observations.models,
        random_seeds=seeds,
        sensitivity_replications=0 if folds_only else sensitivity_replications,
        sensitivity_seed=sensitivity.DEFAULT_SENSITIVITY_SEED,
        feature_rows=frame.height,
        metric_rows=tables["evaluation_metrics"].height,
        curve_rows=tables["discovery_curves"].height,
        simulation_rows=tables["simulation_summary"].height,
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
    if not dry_run and folds_path is not None:
        manifest_path = manifest_path_for(folds_path)
        write_manifest(manifest, manifest_path)

    logger.info(
        "Evaluated %d folds over %d feature rows in %.1fs",
        len(all_folds),
        frame.height,
        (datetime.now(UTC) - started).total_seconds(),
    )
    return EvaluationResult(
        tables=tables,
        checks=checks,
        manifest=manifest,
        folds=all_folds,
        stats=stats,
        folds_path=folds_path,
        manifest_path=manifest_path,
    )


# --- 1. loading -------------------------------------------------------------


def _load(features_path: Path) -> pl.DataFrame:
    """Read the feature table and parse its reference date once.

    Only the columns the harness genuinely needs are read: the keys, the label,
    provenance, and whatever the deterministic rankers consume. Component 5
    does not look at the other 23 features -- selecting among them by test
    performance is exactly the leakage this component exists to prevent.
    """
    frame = pl.read_parquet(features_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise EvaluationError(f"Feature table is missing required columns: {', '.join(missing)}")
    needed = [*REQUIRED_COLUMNS, *rankers.required_columns(list(rankers.RANKERS_BY_NAME))]
    absent = [c for c in needed if c not in frame.columns]
    if absent:
        raise EvaluationError(
            f"Feature table is missing columns the baselines require: {', '.join(absent)}"
        )
    return frame.select(list(dict.fromkeys(needed))).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )


# --- 2. seasonality ---------------------------------------------------------


def _external_predictions(
    predictions_path: Path | None, fold: FoldSpec, stats: FoldStats
) -> list[PredictionSet]:
    """Prediction sets a later component wrote for this fold.

    ``read_predictions`` returns an empty list for a fold it finds nothing for, without
    raising -- which is right for a file that legitimately covers only some folds, but
    means a silently missing fold would simply not appear in the results. So the empty
    case is logged when the fold *has* test rows, and ``covid_shift`` is the fold most
    likely to be the one missing.
    """
    if predictions_path is None:
        return []
    sets = contract.read_predictions(predictions_path, fold_id=fold.fold_id)
    if not sets and stats.test_rows:
        logger.warning(
            "%s: %s carries no prediction set for this fold, but it has %d test rows",
            fold.fold_id,
            predictions_path.name,
            stats.test_rows,
        )
    return sets


def _seasonality(
    frame: pl.DataFrame, all_folds: list[FoldSpec]
) -> tuple[list[dict[str, object]], dict[str, list[sensitivity.MonthEffect]]]:
    """Fit the de-trended seasonal model once descriptively, once per fold.

    The descriptive fit uses every row and is a finding in its own right. The
    per-fold fits use **only that fold's training window**, because a seasonal
    model fitted on the test period would be the same class of error this
    component exists to prevent -- and the sensitivity band is computed from
    those, not from the descriptive one.
    """
    rows: list[dict[str, object]] = []
    for effect in sensitivity.month_effects(frame):
        rows.append({"fitted_on": "all_rows", **_effect_row(effect)})

    by_fold: dict[str, list[sensitivity.MonthEffect]] = {}
    for fold in all_folds:
        train = frame.filter(pl.col("rd").is_between(fold.train_start, fold.train_end))
        effects = sensitivity.month_effects(train)
        by_fold[fold.fold_id] = effects
        for effect in effects:
            rows.append({"fitted_on": fold.fold_id, **_effect_row(effect)})
    return rows, by_fold


def _effect_row(effect: sensitivity.MonthEffect) -> dict[str, object]:
    return {
        "month": effect.month,
        "rows": effect.rows,
        "raw_rate": effect.raw_rate,
        "log_odds_effect": effect.log_odds_effect,
        "pct_point_shift_at_base": effect.pct_point_shift_at_base,
    }


# --- 3. one fold ------------------------------------------------------------


def _business_as_usual_scores(window: simulate.Window) -> list[float]:
    """Express the historical order as a score, so it can be ranked against.

    Earlier date means higher priority, so the score is the negated ordinal of
    the date. Rows sharing a date **tie**, which is faithful: the source carries
    no time component, so business as usual genuinely has no intra-day
    resolution. The tie-break then reproduces the canonical order exactly.
    """
    return [-float(day.toordinal()) for day in window.dates]


def _evaluate_fold(
    *,
    frame: pl.DataFrame,
    fold: FoldSpec,
    stats: FoldStats,
    seeds: list[int],
    effects: list[sensitivity.MonthEffect],
    sensitivity_replications: int,
    observations: validate_module.Observations,
    metric_rows: list[dict[str, object]],
    curve_rows: list[dict[str, object]],
    simulation_rows: list[dict[str, object]],
    sensitivity_rows: list[dict[str, object]],
    external: list[PredictionSet] | None = None,
) -> None:
    """Score every model and simulate every schedule for one test window."""
    test = folds_module.window_frame(frame, fold)
    if test.height == 0:
        logger.warning("%s: test window is empty, skipping", fold.fold_id)
        return

    window = simulate.build_window(
        ids=test["target_inspection_id"].to_list(),
        labels=test["target"].to_list(),
        dates=test["rd"].to_list(),
    )
    ordered = test.sort(["rd", "target_inspection_id"])
    median_daily = int(stats.test_median_daily_capacity or 1)
    k_values = simulate.capacity_k_values(window, median_daily=max(1, median_daily))
    observations.k_values.update(k_values)

    labels = list(window.labels)
    tie_break = list(window.ids)

    # -- the two oracle schedules and business as usual ---------------------
    for schedule, order in (
        (ScheduleName.OPTIMAL, simulate.optimal_order(window)),
        (ScheduleName.WORST, simulate.worst_order(window)),
        (ScheduleName.BUSINESS_AS_USUAL, simulate.business_as_usual_order(window)),
    ):
        result = simulate.evaluate_schedule(window, order, schedule=schedule, label=schedule.value)
        _append_simulation(simulation_rows, fold, stats, window, result, model_name=schedule.value)
        _append_curve(curve_rows, fold, window, result, model_name=schedule.value)

    # -- the random reference, replicated over many seeds -------------------
    for seed in seeds:
        result = simulate.evaluate_schedule(
            window,
            simulate.random_order(window, seed=seed),
            schedule=ScheduleName.RANDOM,
            label="random",
            seed=seed,
        )
        _append_simulation(
            simulation_rows, fold, stats, window, result, model_name="random", seed=seed
        )
        if seed == simulate.DEFAULT_RANDOM_SEED:
            _append_curve(curve_rows, fold, window, result, model_name="random", seed=seed)

    # -- the built-in deterministic producers ---------------------------------
    for model_name in DEFAULT_MODELS:
        scores = (
            _business_as_usual_scores(window)
            if model_name == "business_as_usual"
            else rankers.score(model_name, ordered, seed=simulate.DEFAULT_RANDOM_SEED)
        )

        predictions = PredictionSet(
            model_name=model_name,
            model_version=rankers.RANKER_VERSION,
            fold_id=fold.fold_id,
            frame=contract.prediction_frame(window.ids, scores),
            is_probability=False,
            trained_through=fold.calibration_end,
        )
        _score_prediction_set(
            predictions,
            scores,
            fold=fold,
            stats=stats,
            window=window,
            labels=labels,
            tie_break=tie_break,
            k_values=k_values,
            effects=effects,
            sensitivity_replications=sensitivity_replications,
            observations=observations,
            metric_rows=metric_rows,
            curve_rows=curve_rows,
            simulation_rows=simulation_rows,
            sensitivity_rows=sensitivity_rows,
            # Already simulated above as a reference schedule; simulating it again as a
            # model would double-count it in the summary.
            simulate_as_model=model_name != "business_as_usual",
        )

    # -- prediction sets produced by a later component ------------------------
    if not external:
        return
    for predictions in external:
        try:
            contract.validate_predictions(predictions, fold, window.ids)
        except contract.PredictionHorizonError as exc:
            # Discriminated first: a model that saw past its horizon is a different
            # defect from one that miscounted its rows, and the validator reports the
            # two as different checks.
            observations.horizon_rejections.append(str(exc))
            continue
        except contract.PredictionContractError as exc:
            observations.contract_rejections.append(str(exc))
            continue

        scores = _aligned_scores(predictions, window)
        if predictions.model_name not in observations.models:
            observations.models.append(predictions.model_name)
        _score_prediction_set(
            predictions,
            scores,
            fold=fold,
            stats=stats,
            window=window,
            labels=labels,
            tie_break=tie_break,
            k_values=k_values,
            effects=effects,
            sensitivity_replications=sensitivity_replications,
            observations=observations,
            metric_rows=metric_rows,
            curve_rows=curve_rows,
            simulation_rows=simulation_rows,
            sensitivity_rows=sensitivity_rows,
            simulate_as_model=True,
            already_validated=True,
        )


def _aligned_scores(predictions: PredictionSet, window: simulate.Window) -> list[float]:
    """Reorder a prediction set's scores into the window's canonical order.

    A persisted artifact carries whatever row order its producer wrote, which is not the
    window's ``(date, id)`` order. Zipping the two positionally would attach each score
    to the wrong establishment while producing a perfectly plausible number, so the
    scores are looked up by id.

    Safe to call only after ``validate_predictions`` has confirmed exact coverage, which
    is what guarantees every window id is present exactly once.
    """
    frame = predictions.frame
    lookup = dict(
        zip(
            frame["target_inspection_id"].to_list(),
            frame["score"].to_list(),
            strict=True,
        )
    )
    return [float(lookup[row_id]) for row_id in window.ids]


def _score_prediction_set(
    predictions: PredictionSet,
    scores: list[float],
    *,
    fold: FoldSpec,
    stats: FoldStats,
    window: simulate.Window,
    labels: list[int],
    tie_break: list[str],
    k_values: dict[str, int],
    effects: list[sensitivity.MonthEffect],
    sensitivity_replications: int,
    observations: validate_module.Observations,
    metric_rows: list[dict[str, object]],
    curve_rows: list[dict[str, object]],
    simulation_rows: list[dict[str, object]],
    sensitivity_rows: list[dict[str, object]],
    simulate_as_model: bool,
    already_validated: bool = False,
) -> None:
    """Validate one prediction set, then measure it.

    Every producer goes through here -- the six built-in heuristics and anything a
    modelling component hands over -- so a fitted model is measured by exactly the same
    code path as a ranker. That is the property that makes the comparison meaningful,
    and it is why this is one function rather than two loops.
    """
    model_name = predictions.model_name
    if not already_validated:
        try:
            contract.validate_predictions(predictions, fold, window.ids)
        except contract.PredictionHorizonError as exc:
            observations.horizon_rejections.append(str(exc))
            return
        except contract.PredictionContractError as exc:
            observations.contract_rejections.append(str(exc))
            return

    _append_metrics(
        metric_rows,
        fold,
        stats,
        window,
        model_name,
        scores,
        labels,
        tie_break,
        k_values,
        model_version=predictions.model_version,
        is_probability=predictions.is_probability,
    )

    if not simulate_as_model:
        return

    order = simulate.model_order(window, scores)
    result = simulate.evaluate_schedule(
        window, order, schedule=ScheduleName.MODEL, label=model_name
    )
    _append_simulation(simulation_rows, fold, stats, window, result, model_name=model_name)
    _append_curve(curve_rows, fold, window, result, model_name=model_name)

    if model_name not in SENSITIVITY_EXCLUDED and effects and sensitivity_replications:
        band = sensitivity.redraw_sensitivity(
            window,
            order,
            effects,
            replications=sensitivity_replications,
            observed_nde=result.normalized_discovery_efficiency,
        )
        sensitivity_rows.append(
            {
                "fold_set": fold.fold_set,
                "fold_id": fold.fold_id,
                "model_name": model_name,
                "metric": band.metric,
                "replications": band.replications,
                "observed": band.observed,
                "mean": band.mean,
                "std": band.std,
                "p05": band.p05,
                "p50": band.p50,
                "p95": band.p95,
                "label_flip_rate": band.label_flip_rate,
            }
        )


# --- 4. row builders --------------------------------------------------------


def _append_metrics(
    rows: list[dict[str, object]],
    fold: FoldSpec,
    stats: FoldStats,
    window: simulate.Window,
    model_name: str,
    scores: list[float],
    labels: list[int],
    tie_break: list[str],
    k_values: dict[str, int],
    *,
    model_version: str,
    is_probability: bool = False,
) -> None:
    """Emit one long-format row per metric.

    Probability metrics are emitted **only** when the producer declares
    ``is_probability=True``. Inventing a Brier score for a ranking is a
    fabrication, so for the un-fitted baselines the rows simply do not exist
    rather than being filled with a placeholder.

    ``precision``, ``recall`` and ``f1`` are never emitted, even for a
    probability model. They need a threshold, this schema has no threshold
    column, and the only place to record 0.5 would be ``k_name`` -- which would
    misdescribe what ``k`` means. A threshold is genuinely needed only by the
    Component 16 deferral gate, which can record its own.
    """
    base = {
        "fold_set": fold.fold_set,
        "fold_id": fold.fold_id,
        "model_name": model_name,
        "model_version": model_version,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "n_test": window.n,
        "positive_rate": stats.test_positive_rate,
        "evaluation_definition_version": EVALUATION_DEFINITION_VERSION,
    }
    for metric, value in (
        ("roc_auc", metrics_module.roc_auc(labels, scores)),
        ("pr_auc", metrics_module.pr_auc(labels, scores)),
    ):
        rows.append(
            {
                **base,
                "metric": metric,
                "metric_kind": "ranking",
                "k_name": "",
                "k": None,
                "value": value,
            }
        )

    for k_name, k in sorted(k_values.items()):
        rows.append(
            {
                **base,
                "metric": "precision_at_k",
                "metric_kind": "ranking",
                "k_name": k_name,
                "k": k,
                "value": metrics_module.precision_at_k(labels, scores, tie_break, k),
            }
        )
        rows.append(
            {
                **base,
                "metric": "recall_at_k",
                "metric_kind": "ranking",
                "k_name": k_name,
                "k": k,
                "value": metrics_module.recall_at_k(labels, scores, tie_break, k),
            }
        )
        rows.append(
            {
                **base,
                "metric": "lift_at_k",
                "metric_kind": "ranking",
                "k_name": k_name,
                "k": k,
                "value": metrics_module.lift_at_k(labels, scores, tie_break, k),
            }
        )

    if not is_probability:
        return

    # A fitted model's raw predict_proba output. These are diagnostics of an
    # *uncalibrated* model: Component 9 owns calibration, and this is the "before"
    # number its existence is justified against.
    for metric, value in (
        ("brier", metrics_module.brier(labels, scores)),
        ("log_loss", metrics_module.log_loss(labels, scores)),
        ("ece", metrics_module.ece(labels, scores)),
        ("mce", metrics_module.mce(labels, scores)),
    ):
        rows.append(
            {
                **base,
                "metric": metric,
                "metric_kind": "probability",
                "k_name": "",
                "k": None,
                "value": value,
            }
        )


def _append_simulation(
    rows: list[dict[str, object]],
    fold: FoldSpec,
    stats: FoldStats,
    window: simulate.Window,
    result: simulate.ScheduleResult,
    *,
    model_name: str,
    seed: int | None = None,
) -> None:
    positives = result.days_earlier_positives
    all_rows = result.days_earlier_all
    rows.append(
        {
            "fold_set": fold.fold_set,
            "fold_id": fold.fold_id,
            "schedule_name": result.schedule.value,
            "model_name": model_name,
            "seed": seed,
            "n_slots": window.n,
            "n_positives": window.positives,
            "median_daily_capacity": stats.test_median_daily_capacity,
            "discovery_area": result.area,
            "normalized_discovery_efficiency": result.normalized_discovery_efficiency,
            "first_half_discovery": result.first_half_discovery,
            "days_earlier_count": positives.count,
            "mean_days_earlier": positives.mean,
            "median_days_earlier": positives.median,
            "std_days_earlier": positives.std,
            "p25_days_earlier": positives.p25,
            "p75_days_earlier": positives.p75,
            "min_days_earlier": positives.minimum,
            "max_days_earlier": positives.maximum,
            "fraction_improved": positives.fraction_improved,
            "fraction_unchanged": positives.fraction_unchanged,
            "fraction_worse": positives.fraction_worse,
            "mean_days_earlier_all_rows": all_rows.mean,
            "fraction_improved_all_rows": all_rows.fraction_improved,
        }
    )


def _append_curve(
    rows: list[dict[str, object]],
    fold: FoldSpec,
    window: simulate.Window,
    result: simulate.ScheduleResult,
    *,
    model_name: str,
    seed: int | None = None,
) -> None:
    """Store the curve at full resolution. The curve is the result."""
    positives = window.positives or 1
    for slot, cumulative in enumerate(result.cumulative):
        rows.append(
            {
                "fold_set": fold.fold_set,
                "fold_id": fold.fold_id,
                "schedule_name": result.schedule.value,
                "model_name": model_name,
                "seed": seed,
                "slot_index": slot,
                "slot_fraction": slot / window.n if window.n else 0.0,
                "cumulative_positive": cumulative,
                "cumulative_positive_fraction": cumulative / positives,
            }
        )


def _folds_table(all_folds: list[FoldSpec], stats: list[FoldStats]) -> pl.DataFrame:
    rows: list[dict[str, object]] = [
        {
            "fold_set": fold.fold_set,
            "fold_id": fold.fold_id,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "calibration_start": fold.calibration_start,
            "calibration_end": fold.calibration_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "train_rows": stat.train_rows,
            "calibration_rows": stat.calibration_rows,
            "test_rows": stat.test_rows,
            "train_positive_rate": stat.train_positive_rate,
            "calibration_positive_rate": stat.calibration_positive_rate,
            "test_positive_rate": stat.test_positive_rate,
            "test_establishments": stat.test_establishments,
            "test_days": stat.test_days,
            "test_median_daily_capacity": stat.test_median_daily_capacity,
            "evaluation_definition_version": EVALUATION_DEFINITION_VERSION,
        }
        for fold, stat in zip(all_folds, stats, strict=True)
    ]
    return writer.finalize(rows, "evaluation_folds")


def summarize(result: EvaluationResult) -> str:
    """One-screen summary of a run, printed by the CLI."""
    m = result.manifest
    lines = [
        f"feature rows:        {m.feature_rows}",
        f"folds:               {m.folds} ({', '.join(f'{k}={v}' for k, v in m.fold_sets.items())})",
        f"test range:          {m.first_test_start} .. {m.last_test_end}",
        f"models scored:       {len(m.models)}",
        f"metric rows:         {m.metric_rows}",
        f"curve rows:          {m.curve_rows}",
        f"simulation rows:     {m.simulation_rows}",
        f"random seeds:        {len(m.random_seeds)}",
        f"sensitivity reps:    {m.sensitivity_replications}",
        f"definition:          {m.evaluation_definition_version}",
        "estimand:            re-ordering only, of inspections that actually occurred",
    ]
    if m.predictions_path is not None:
        lines.append(f"predictions read:    {m.predictions_path}")
    if m.excluded_partial_windows:
        for window in m.excluded_partial_windows:
            lines.append(f"  excluded:          {window}")
    for item in m.blocked:
        lines.append(f"  BLOCKED:           {item.split(':')[0]}")
    if result.folds_path is not None:
        lines.append(f"folds:               {result.folds_path}")
        lines.append(f"manifest:            {result.manifest_path}")
    return "\n".join(lines)
