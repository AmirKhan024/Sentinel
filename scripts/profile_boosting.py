"""Profile the Component 4 feature table for Component 7's design decisions.

Analysis tooling, not library code: it answers one-off questions about a snapshot,
nothing imports it, and it should not ship in the wheel. Output is markdown on stdout,
pasted into ``docs/analysis/boosting_models_findings.md``.

⚠ **This script never reports a test-window number, and that is a hard rule, not a style
preference.** Component 5 protects evaluation time, but it cannot protect against a human
reading a test metric here, changing a hyperparameter or a search range, and re-running.
That loop is leakage, it leaves no trace in any artifact, and no check in the repository
can detect it. So every fit below is confined to a fold's *training* window, and the only
out-of-sample surface this script may touch is the **calibration** window -- which exists
precisely so design choices can be frozen before the test period is opened.

Component 7 adds a second boundary on top of Component 6's. A hyperparameter is a design
choice, so the ranges and the protocol have to be fixed from what is visible here. The
``tuning_regions`` profile exists to show that the region a study may read is strictly
earlier than the first test window, computed from the fold definitions rather than
asserted.

Questions this script answers
-----------------------------
1. What are the tuning regions, and do they end before their fold sets' first test windows?
2. How large and how balanced are the inner folds each region yields?
3. How much of the matrix is NaN, so is native NULL routing actually exercised?
4. How long does one fit take, and therefore what does a 100-trial search cost?
5. Does row order change a fitted booster, as it changed Component 6's coefficients?
6. Would LightGBM's ``num_leaves`` range exceed XGBoost's implied capacity without a cap?
7. On the calibration window, are boosted probabilities worse calibrated than the GLM's?
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.boosting import preprocess as tree_preprocess  # noqa: E402
from sentinel.boosting import tuning  # noqa: E402
from sentinel.boosting.definitions import (  # noqa: E402
    BOOSTING_REGISTRY,
    SEARCH_SPACE,
    BoostingSpec,
    Estimator,
    estimator_params,
    spec_for,
)
from sentinel.boosting.train import build_estimator, fold_labels  # noqa: E402
from sentinel.config import load_settings  # noqa: E402
from sentinel.evaluation import folds as folds_module  # noqa: E402
from sentinel.evaluation import metrics  # noqa: E402
from sentinel.evaluation.models import FoldSpec  # noqa: E402
from sentinel.modeling.train import training_frame  # noqa: E402
from sentinel.query import duckdb_queries  # noqa: E402

PRIMARY = spec_for("xgboost")
SECONDARY = spec_for("lightgbm")


# --- shared helpers ----------------------------------------------------------


def load_frame(path: Path) -> pl.DataFrame:
    """The feature table with its reference date parsed once."""
    frame = pl.read_parquet(path)
    return frame.with_columns(pl.col("inspection_date").str.to_date().alias("rd"))


def outer_folds(frame: pl.DataFrame) -> list[FoldSpec]:
    """Component 5's folds, rebuilt from the data. Never invented here."""
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    if start is None or end is None:
        raise SystemExit("feature table has no usable reference dates")
    quarterly = folds_module.quarterly_folds(data_start=start, data_end=end)
    return [*quarterly, *folds_module.covid_shift_fold(data_end=end)]


def calibration_frame(frame: pl.DataFrame, fold: FoldSpec) -> pl.DataFrame:
    """The fold's calibration rows -- the only out-of-sample surface allowed here."""
    assigned = folds_module.assign_split(frame, fold)
    return assigned.filter(pl.col("split") == "calibration").sort(
        ["inspection_date", "target_inspection_id"]
    )


def _rate(frame: pl.DataFrame) -> float:
    """A window's positive rate, narrowed out of polars' wide aggregate union."""
    if frame.height == 0:
        return 0.0
    mean = frame["target"].mean()
    return float(mean) if isinstance(mean, (int, float)) else 0.0


def largest_training_fold(folds: list[FoldSpec]) -> FoldSpec:
    """The quarterly fold with the widest training window: the worst-case fit cost."""
    quarterly = [f for f in folds if f.fold_set == folds_module.QUARTERLY]
    return max(quarterly, key=lambda f: f.train_end)


# --- profiles ----------------------------------------------------------------


def profile_tuning_regions(frame: pl.DataFrame) -> str:
    """The protocol, shown rather than asserted: region end versus first test start."""
    folds = outer_folds(frame)
    lines = [
        "| fold set | outer folds | tuning region | region end | first test start | safe |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for fold_set in sorted({f.fold_set for f in folds}):
        start, end = tuning.tuning_region(fold_set, folds)
        horizon = tuning.first_test_start(fold_set, folds)
        count = sum(1 for f in folds if f.fold_set == fold_set)
        safe = "yes" if end < horizon else "**NO**"
        lines.append(f"| {fold_set} | {count} | {start}..{end} | {end} | {horizon} | {safe} |")
    lines.append("")
    lines.append(
        "The quarterly region is the first quarterly fold's `train_start..calibration_end`. "
        "Because quarterly folds expand from a fixed anchor, that span is a subset of every "
        "later fold's own train-plus-calibration, so no fold is scored by parameters chosen "
        "on its own test data."
    )
    lines.append("")
    lines.append(
        "The covid_shift test window sits **inside** the quarterly region, which is why the "
        "two fold sets get separate studies rather than one shared study."
    )
    return "\n".join(lines)


def profile_inner_folds(frame: pl.DataFrame) -> str:
    """Row counts and prevalence for every inner fold a study will actually use."""
    folds = outer_folds(frame)
    lines = [
        "| fold set | inner fold | train | train rate | validation | validation rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for fold_set in sorted({f.fold_set for f in folds}):
        inner = tuning.build_inner_folds(fold_set, folds)
        for fold in inner:
            train = training_frame(frame, fold)
            validation = folds_module.window_frame(frame, fold)
            train_rate = _rate(train)
            validation_rate = _rate(validation)
            lines.append(
                f"| {fold_set} | {fold.fold_id.rsplit('-', 1)[-1]} | {train.height} | "
                f"{train_rate:.4f} | {validation.height} | {validation_rate:.4f} |"
            )
    lines.append("")
    lines.append(
        "Each inner fold keeps the outer structure's unused calibration quarter between "
        "training and validation. Tuning without that gap would select parameters for a "
        "zero-gap regime and apply them to a one-gap regime."
    )
    return "\n".join(lines)


def profile_nan_density(frame: pl.DataFrame) -> str:
    """How much of the matrix arrives as NaN, per column, on the widest training window."""
    folds = outer_folds(frame)
    fold = largest_training_fold(folds)
    train = training_frame(frame, fold)
    matrix = tree_preprocess.tree_matrix(train, PRIMARY)
    columns = tree_preprocess.matrix_columns(PRIMARY)
    nan_by_column = np.isnan(matrix).sum(axis=0)

    lines = [
        f"Training window `{fold.fold_id}`: {train.height} rows x {len(columns)} matrix columns.",
        "",
        "| column | NaN rows | share |",
        "| --- | --- | --- |",
    ]
    for index, name in enumerate(columns):
        count = int(nan_by_column[index])
        if count == 0:
            continue
        lines.append(f"| {name} | {count} | {count / train.height:.4f} |")
    total = int(nan_by_column.sum())
    lines.append("")
    lines.append(
        f"{total} NaN cells of {matrix.size} ({total / matrix.size:.4f}). "
        f"{int((nan_by_column > 0).sum())} of {len(columns)} columns carry any NaN."
    )
    lines.append("")
    lines.append(
        "Every one of these reaches the estimator as NaN and is routed by a learned "
        "default direction at each split. Component 6 had to fill them with a "
        "training-window median, which replaces a fact about the establishment -- there "
        "was no prior canvass -- with the average of a population it is not in."
    )
    return "\n".join(lines)


def profile_fit_runtime(frame: pl.DataFrame) -> str:
    """One fit's wall clock, and the search budget it implies."""
    folds = outer_folds(frame)
    fold = largest_training_fold(folds)
    train = training_frame(frame, fold)
    labels = fold_labels(train, "profile", fold.fold_id)

    lines = [
        f"Widest quarterly training window `{fold.fold_id}`: {train.height} rows.",
        "",
        "| model | rounds | seconds | seconds/round |",
        "| --- | --- | --- | --- |",
    ]
    timings: dict[str, float] = {}
    for spec in (PRIMARY, SECONDARY):
        params = estimator_params(spec, fold.fold_set)
        matrix = tree_preprocess.tree_matrix(train, spec)
        started = time.perf_counter()
        estimator = build_estimator(spec, params)
        estimator.fit(matrix, labels)
        elapsed = time.perf_counter() - started
        declared = params.get("n_estimators", 0)
        rounds = declared if isinstance(declared, int) else 0
        timings[spec.name] = elapsed
        lines.append(f"| {spec.name} | {rounds} | {elapsed:.2f} | {elapsed / max(rounds, 1):.4f} |")

    inner_counts = {
        fold_set: len(tuning.build_inner_folds(fold_set, folds))
        for fold_set in sorted({f.fold_set for f in folds})
    }
    lines.append("")
    lines.append(f"Inner folds per study: {inner_counts}.")
    lines.append("")
    lines.append(
        "A trial fits once per inner fold, so a 100-trial study costs roughly "
        "100 x (inner folds) x (one fit). These fits are single-threaded by design "
        "(`n_jobs=1`), which is what makes them bit-reproducible; the cost of that choice "
        "is visible in the column above."
    )
    for name, seconds in timings.items():
        quarterly = inner_counts.get(folds_module.QUARTERLY, 0)
        shift = inner_counts.get(folds_module.COVID_SHIFT, 0)
        estimate = 100 * (quarterly + shift) * seconds
        lines.append(
            f"- {name}: both studies at 100 trials ~ {estimate / 60:.1f} minutes at this "
            "window size, before early stopping shortens the average fit."
        )
    return "\n".join(lines)


def _fit_and_score(
    spec: BoostingSpec,
    params: dict[str, object],
    rows: pl.DataFrame,
    score_matrix: NDArray[np.float64],
    fold: FoldSpec,
) -> NDArray[np.float64]:
    """Fit on ``rows`` and score a fixed matrix, so only the training order varies."""
    estimator = build_estimator(spec, params)
    estimator.fit(
        tree_preprocess.tree_matrix(rows, spec),
        fold_labels(rows, "profile", fold.fold_id),
    )
    return np.asarray(estimator.predict_proba(score_matrix)[:, 1], dtype=np.float64)


def profile_row_order_sensitivity(frame: pl.DataFrame) -> str:
    """Does re-ordering the same training rows move a fitted booster?

    Component 6 measured its coefficients shifting by up to 7.049e-09 under a
    re-ordering, because float summation is not associative. The equivalent question for
    a booster is whether its *predictions* move, since it has no coefficients to compare.
    """
    folds = outer_folds(frame)
    fold = largest_training_fold(folds)
    train = training_frame(frame, fold)
    calibration = calibration_frame(frame, fold)
    shuffled = train.sample(fraction=1.0, shuffle=True, seed=7)

    lines = [
        f"Fold `{fold.fold_id}`: {train.height} training rows, scored on its "
        f"{calibration.height}-row calibration window (never the test window).",
        "",
        "| model | max abs difference, shuffled | max abs difference, shuffled then re-sorted |",
        "| --- | --- | --- |",
    ]
    for spec in (PRIMARY, SECONDARY):
        params = estimator_params(spec, fold.fold_set)
        score_matrix = tree_preprocess.tree_matrix(calibration, spec)

        reference = _fit_and_score(spec, params, train, score_matrix, fold)
        moved = _fit_and_score(spec, params, shuffled, score_matrix, fold)
        restored = _fit_and_score(
            spec,
            params,
            shuffled.sort(["inspection_date", "target_inspection_id"]),
            score_matrix,
            fold,
        )
        lines.append(
            f"| {spec.name} | {np.max(np.abs(reference - moved)):.6e} | "
            f"{np.max(np.abs(reference - restored)):.6e} |"
        )

    lines.append("")
    lines.append(
        "The second column is the one that matters: re-sorting a shuffled frame must "
        "restore the original fit exactly, because that is what `train.fit_fold`'s "
        "canonical sort relies on. A non-zero value there would mean the sort is not "
        "sufficient for reproducibility and something else in the pipeline carries state."
    )
    return "\n".join(lines)


def profile_capacity_cap(frame: pl.DataFrame) -> str:
    """How often LightGBM's declared ``num_leaves`` range exceeds XGBoost's implied one."""
    del frame  # this profile is about the declared space, not the data
    xgb_depth = next(d for d in SEARCH_SPACE[Estimator.XGBOOST] if d.name == "max_depth")
    lgb_leaves = next(d for d in SEARCH_SPACE[Estimator.LIGHTGBM] if d.name == "num_leaves")

    lines = [
        "| max_depth | XGBoost implied leaves (2^depth) | LightGBM range | capped at |",
        "| --- | --- | --- | --- |",
    ]
    uncapped = 0
    total = 0
    for depth in range(int(xgb_depth.low), int(xgb_depth.high) + 1):
        implied = 2**depth
        cap = min(int(lgb_leaves.high), implied)
        exceeds = int(lgb_leaves.high) > implied
        uncapped += 1 if exceeds else 0
        total += 1
        lines.append(
            f"| {depth} | {implied} | {int(lgb_leaves.low)}..{int(lgb_leaves.high)} | {cap} |"
        )
    lines.append("")
    lines.append(
        f"At {uncapped} of {total} depths the declared LightGBM range would allow more "
        "leaves than a depth-wise XGBoost tree could ever build. Leaf-wise growth means "
        "`max_depth` alone bounds nothing, so without the cap the two searches would "
        "explore different capacity ranges and any measured difference between the "
        "libraries would be a fact about the search space rather than the algorithm."
    )
    return "\n".join(lines)


def profile_calibration_probe(frame: pl.DataFrame) -> str:
    """Boosted versus logistic probability quality, on the calibration window only.

    Pre-registers an expectation rather than reporting a result. If the boosted ECE is
    worse here, then a worse ECE at evaluation time is the predicted behaviour and not a
    defect -- and Component 9, not Component 7, is where it gets corrected.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from sentinel.modeling import preprocess as glm_preprocess
    from sentinel.modeling.definitions import LOGISTIC_PARAMS, MODELS_BY_NAME

    folds = outer_folds(frame)
    fold = largest_training_fold(folds)
    train = training_frame(frame, fold)
    calibration = calibration_frame(frame, fold)
    labels = fold_labels(train, "profile", fold.fold_id).tolist()
    calibration_labels = fold_labels(calibration, "profile", fold.fold_id).tolist()

    lines = [
        f"Fold `{fold.fold_id}`, scored on its calibration window ({calibration.height} rows).",
        "",
        "| model | PR-AUC | ROC-AUC | Brier | ECE | MCE | saturated |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    glm_spec = MODELS_BY_NAME["logistic_regression"]
    pipeline = Pipeline(
        [
            ("preprocess", glm_preprocess.build_preprocessor(glm_spec)),
            ("model", LogisticRegression(**dict(LOGISTIC_PARAMS), random_state=glm_spec.seed)),
        ]
    )
    pipeline.fit(glm_preprocess.to_matrix(train, glm_spec), np.asarray(labels))
    glm_scores = [
        float(v)
        for v in pipeline.predict_proba(glm_preprocess.to_matrix(calibration, glm_spec))[:, 1]
    ]
    candidates: list[tuple[str, list[float]]] = [("logistic_regression", glm_scores)]

    for spec in (PRIMARY, SECONDARY):
        estimator = build_estimator(spec, estimator_params(spec, fold.fold_set))
        estimator.fit(tree_preprocess.tree_matrix(train, spec), np.asarray(labels))
        scores = [
            float(v)
            for v in estimator.predict_proba(tree_preprocess.tree_matrix(calibration, spec))[:, 1]
        ]
        candidates.append((spec.name, scores))

    for name, scores in candidates:
        saturated = sum(1 for s in scores if s in (0.0, 1.0))
        lines.append(
            f"| {name} "
            f"| {metrics.pr_auc(calibration_labels, scores) or float('nan'):.4f} "
            f"| {metrics.roc_auc(calibration_labels, scores) or float('nan'):.4f} "
            f"| {metrics.brier(calibration_labels, scores):.4f} "
            f"| {metrics.ece(calibration_labels, scores):.4f} "
            f"| {metrics.mce(calibration_labels, scores):.4f} "
            f"| {saturated} |"
        )
    lines.append("")
    lines.append(
        "These are calibration-window numbers on one fold, fitted with **provisional** "
        "parameters. They are a design probe, not a result: no test window was read, and "
        "nothing here may be quoted as Component 7's performance."
    )
    return "\n".join(lines)


PROFILES: dict[str, Callable[[pl.DataFrame], str]] = {
    "tuning_regions": profile_tuning_regions,
    "inner_folds": profile_inner_folds,
    "nan_density": profile_nan_density,
    "fit_runtime": profile_fit_runtime,
    "row_order_sensitivity": profile_row_order_sensitivity,
    "capacity_cap": profile_capacity_cap,
    "calibration_probe": profile_calibration_probe,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="profile_boosting",
        description="Profile Component 4 features for Component 7's design decisions.",
    )
    parser.add_argument(
        "--features", type=Path, metavar="PATH", help="Feature table. Defaults to the most recent."
    )
    parser.add_argument(
        "--only", action="append", metavar="NAME", help="Profile to run; repeatable."
    )
    parser.add_argument("--list", action="store_true", dest="list_profiles")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_profiles:
        for name in PROFILES:
            print(name)
        return 0

    requested = args.only or list(PROFILES)
    unknown = [name for name in requested if name not in PROFILES]
    if unknown:
        print(f"unknown profile(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    path = args.features
    if path is None:
        settings = load_settings()
        path = duckdb_queries.latest_parquet(
            settings.features_processed_dir, prefix="as_of_features_"
        )
    frame = load_frame(path)

    print(f"<!-- generated by scripts/profile_boosting.py from {path.name} -->")
    print(f"<!-- {frame.height} rows; models: {', '.join(s.name for s in BOOSTING_REGISTRY)} -->")
    for name in requested:
        print()
        print(f"### {name}")
        print()
        print(PROFILES[name](frame))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
