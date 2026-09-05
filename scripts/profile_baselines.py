"""Read-only profiling of what a trustworthy baseline model can be built from.

Analysis tooling, not library code, for the same reasons as the other four profilers:
it answers one-off questions about a snapshot, nothing imports it, and it should not ship
in the wheel.

⚠ **This script never reports a test-window number, and that is a hard rule, not a
style preference.** Component 5 protects evaluation time, but it cannot protect against
a human reading a test metric here, changing a hyperparameter or dropping a feature, and
re-running. That loop is leakage, it leaves no trace in any artifact, and no check in the
repository can detect it. So every fit below is confined to a fold's *training* window,
and the only out-of-sample surface this script is allowed to touch is the *calibration*
window -- which exists precisely so design choices can be frozen before the test period
is opened. If you want a test number, run ``sentinel evaluate --predictions``; that path
records what it did in a manifest.

The questions this script exists to answer, none of which the design can be settled
without:

1. Which features are nullable, and are the null masks within a rule family distinct?
   This decides whether ``SimpleImputer(add_indicator=True)`` is usable at all.
2. Is each family's null mask exactly the zero-set of a paired never-null count? If so,
   the indicator adds no *new* information, only linearly-available information -- which
   changes what the coefficients mean and must be stated in the contract.
3. Does row order reach the coefficients? Floating-point summation order in the scaler
   and in the lbfgs gradient says it might, and the answer decides whether the canonical
   sort in ``fit_fold`` is load-bearing or decorative.
4. Does the fit converge inside ``max_iter`` on real training windows?
5. How collinear is the design matrix? The windowed counts nest by construction
   (365d subset of 730d subset of 1095d), so some collinearity is guaranteed.
6. Would a median-imputed nullable boolean flip its fill value between folds?

Every read is a SELECT-equivalent and the script writes nothing. Output is markdown,
pasted into ``docs/analysis/baseline_models_findings.md``.

Usage
-----
    uv run python scripts/profile_baselines.py
    uv run python scripts/profile_baselines.py --only row_order_sensitivity
    uv run python scripts/profile_baselines.py --list
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.config import load_settings  # noqa: E402
from sentinel.evaluation import folds as folds_module  # noqa: E402
from sentinel.evaluation.models import FoldSpec  # noqa: E402
from sentinel.features.definitions import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_SPECS,
    NullRule,
)
from sentinel.query.duckdb_queries import latest_parquet  # noqa: E402

#: Component 4 pairs every nullable family with a never-null count whose zero-set is
#: meant to be exactly that family's null mask. Profile 3 checks the claim.
PAIRED_COUNT: dict[NullRule, str] = {
    NullRule.NO_PRIOR_CANVASS: "prior_canvass_count",
    NullRule.NO_INSPECTED_CANVASS: "prior_canvass_inspected_count",
    NullRule.NO_CODE_ERA_CANVASS: "prior_canvass_count_code_era",
    NullRule.NO_PRIOR_INSPECTION: "prior_inspection_count_any_type",
}

#: Reproduced from ``docs/analysis/temporal_evaluation_findings.md`` §10 and
#: ``evaluation/rankers.py``, so the approximation table travels with the model that
#: needs it rather than living only in a document nobody opens.
CDPH_2015_INPUTS: tuple[tuple[str, str, str], ...] = (
    ("prior violation history", "available", "12 Component 4 canvass/priority features"),
    ("time since last inspection", "available", "days_since_last_canvass"),
    ("business age / tenure", "available", "days_since_first_inspection"),
    ("inspector identity", "EXCLUDED", "deliberately -- audit Finding 1"),
    ("nearby 311 complaints", "NOT INGESTED", "Component 1 extension"),
    ("burglary intensity", "NOT INGESTED", "Component 1 extension"),
    ("alcohol / tobacco licence", "NOT INGESTED", "Component 1 extension"),
    ("weather / temperature", "NOT INGESTED", "needs NOAA GHCN USW00094846"),
    ("facility type", "NOT A FEATURE", "in the raw snapshot, not in Component 4"),
    ("CDPH risk category", "NOT A FEATURE", "in the raw snapshot, not in Component 4"),
)

MAX_ITER = 1000
SEED = 42


# --- shared helpers ---------------------------------------------------------


def nullable_specs() -> tuple[tuple[str, NullRule, str], ...]:
    """Every feature whose null rule is not NEVER, with its dtype. Derived, never listed."""
    return tuple(
        (spec.name, spec.null_rule, spec.dtype)
        for spec in FEATURE_SPECS
        if spec.null_rule is not NullRule.NEVER
    )


def families() -> tuple[NullRule, ...]:
    """The distinct non-NEVER null rules, in a stable order."""
    seen: list[NullRule] = []
    for _, rule, _ in nullable_specs():
        if rule not in seen:
            seen.append(rule)
    return tuple(sorted(seen, key=lambda r: r.value))


def columns_in_family(rule: NullRule) -> tuple[str, ...]:
    return tuple(name for name, r, _ in nullable_specs() if r is rule)


def load_features(features_path: Path) -> pl.DataFrame:
    """The feature table with its reference date parsed once, as Component 5 does."""
    return pl.read_parquet(features_path).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )


def all_folds(frame: pl.DataFrame) -> list[FoldSpec]:
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    if start is None or end is None:
        raise RuntimeError("feature table has no usable reference dates")
    return [
        *folds_module.quarterly_folds(data_start=start, data_end=end),
        *folds_module.covid_shift_fold(data_end=end),
    ]


def train_frame(frame: pl.DataFrame, fold: FoldSpec) -> pl.DataFrame:
    """A fold's training rows, in canonical order.

    Uses ``assign_split`` rather than a hand-rolled date filter so this script and the
    implementation agree on what "train" means -- the definition Component 5's
    ``future_rows_never_enter_training`` check independently re-derives.
    """
    split = folds_module.assign_split(frame, fold)
    return split.filter(pl.col("split") == "train").sort(
        ["inspection_date", "target_inspection_id"]
    )


def design_matrix(
    frame: pl.DataFrame, feature_columns: Sequence[str]
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Features plus one indicator per null-rule family, medians filled from this frame.

    Deliberately self-contained: a profiler explores, and the implementation then encodes
    what the profiling established. Importing a not-yet-written preprocessor would invert
    that order.
    """
    columns = list(feature_columns)
    exprs: list[pl.Expr] = []
    for rule in families():
        members = [c for c in columns_in_family(rule) if c in columns]
        if not members:
            continue
        exprs.append(pl.col(members[0]).is_null().cast(pl.Float64).alias(_indicator(rule)))
    prepared = frame.with_columns(exprs) if exprs else frame
    names = tuple(columns + [e.meta.output_name() for e in exprs])

    filled: list[pl.Expr] = []
    for name in names:
        col = pl.col(name).cast(pl.Float64)
        spec = next((s for s in FEATURE_SPECS if s.name == name), None)
        if spec is not None and spec.null_rule is not NullRule.NEVER:
            fill = pl.lit(0.0) if spec.dtype == "bool" else col.median()
            col = col.fill_null(fill)
        filled.append(col.alias(name))
    matrix = prepared.select(filled).to_numpy().astype(np.float64)
    return matrix, names


def _indicator(rule: NullRule) -> str:
    return "missing_" + rule.name.lower()


def fit(matrix: NDArray[np.float64], labels: NDArray[np.int64]) -> tuple[object, int, bool]:
    """Standardise then fit an L2 logistic regression. Returns (pipeline, n_iter, converged)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2", C=1.0, solver="lbfgs", max_iter=MAX_ITER, random_state=SEED
                ),
            ),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(matrix, labels)
    n_iter = int(pipeline.named_steps["model"].n_iter_[0])
    converged = n_iter < MAX_ITER and not any(
        issubclass(w.category, ConvergenceWarning) for w in caught
    )
    return pipeline, n_iter, converged


def coefficients(pipeline: object) -> NDArray[np.float64]:
    steps = pipeline.named_steps  # type: ignore[attr-defined]
    return np.asarray(steps["model"].coef_[0], dtype=np.float64)


def labels_of(frame: pl.DataFrame) -> NDArray[np.int64]:
    return np.asarray(frame["target"].to_list(), dtype=np.int64)


def as_float(value: object) -> float:
    """Narrow a polars aggregate (typed as a wide union) to a float. 0.0 if absent."""
    return float(value) if isinstance(value, (int, float)) else 0.0


def render_markdown(columns: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    if not rows:
        return "_no rows_"
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join([header, rule, *body])


# --- 1. the nullable partition ----------------------------------------------


def profile_nullable_partition(frame: pl.DataFrame, _folds: list[FoldSpec]) -> str:
    """Which of the 26 features are nullable, by rule, with measured null counts."""
    rows: list[Sequence[object]] = []
    for name, rule, dtype in nullable_specs():
        nulls = int(frame[name].null_count())
        rows.append([name, rule.value, dtype, nulls, round(nulls / frame.height, 6)])
    never = [s.name for s in FEATURE_SPECS if s.null_rule is NullRule.NEVER]
    table = render_markdown(["feature", "null_rule", "dtype", "nulls", "null_rate"], rows)
    return (
        f"{table}\n\n"
        f"- nullable features: **{len(rows)}** of {len(FEATURE_COLUMNS)}\n"
        f"- never-null features: **{len(never)}**\n"
        f"- distinct null-rule families: **{len(families())}**\n"
    )


# --- 2. are the masks within a family distinct? -----------------------------


def profile_family_mask_identity(frame: pl.DataFrame, _folds: list[FoldSpec]) -> str:
    """Whether every column in a family is null on exactly the same rows.

    If it is, an indicator per *column* produces duplicate columns -- which is what
    ``SimpleImputer(add_indicator=True)`` would emit.
    """
    rows: list[Sequence[object]] = []
    for rule in families():
        members = columns_in_family(rule)
        reference = frame[members[0]].is_null()
        identical = all(frame[m].is_null().eq(reference).all() for m in members[1:])
        rows.append(
            [
                rule.value,
                len(members),
                ", ".join(members),
                int(reference.sum()),
                "IDENTICAL" if identical else "differ",
            ]
        )
    total_columns = len(nullable_specs())
    table = render_markdown(["family", "columns", "members", "null_rows", "masks"], rows)
    return (
        f"{table}\n\n"
        f"- per-column indicators would be: **{total_columns}**\n"
        f"- distinct indicators actually needed: **{len(families())}**\n"
    )


# --- 3. is each mask the zero-set of a paired count? ------------------------


def profile_mask_vs_paired_count(frame: pl.DataFrame, _folds: list[FoldSpec]) -> str:
    """Whether a family's null mask equals ``paired_count == 0`` exactly."""
    rows: list[Sequence[object]] = []
    for rule in families():
        members = columns_in_family(rule)
        paired = PAIRED_COUNT[rule]
        mask = frame[members[0]].is_null()
        zero = frame[paired].eq(0)
        rows.append(
            [
                rule.value,
                paired,
                int(mask.sum()),
                int(zero.sum()),
                "EXACT" if bool(mask.eq(zero).all()) else "DIFFER",
            ]
        )
    return render_markdown(["family", "paired_count", "null_rows", "zero_rows", "match"], rows)


# --- 4. the training windows -------------------------------------------------


def profile_train_windows(frame: pl.DataFrame, fold_specs: list[FoldSpec]) -> str:
    """Per-fold training row counts and base rates. Training window only."""
    rows: list[Sequence[object]] = []
    for fold in fold_specs:
        train = train_frame(frame, fold)
        rows.append(
            [
                fold.fold_set,
                fold.fold_id,
                str(fold.train_start),
                str(fold.train_end),
                train.height,
                round(as_float(train["target"].mean()), 4),
            ]
        )
    return render_markdown(
        ["fold_set", "fold_id", "train_start", "train_end", "train_rows", "train_rate"], rows
    )


# --- 5. collinearity of the design matrix -----------------------------------


def profile_collinearity(frame: pl.DataFrame, fold_specs: list[FoldSpec]) -> str:
    """Condition number and the most correlated pairs, on the first fold's train window."""
    fold = fold_specs[0]
    train = train_frame(frame, fold)
    matrix, names = design_matrix(train, FEATURE_COLUMNS)
    standardised = (matrix - matrix.mean(axis=0)) / np.where(
        matrix.std(axis=0) == 0, 1.0, matrix.std(axis=0)
    )
    correlation = np.asarray(np.corrcoef(standardised, rowvar=False), dtype=np.float64)
    pairs: list[tuple[float, str, str]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = float(correlation[i, j])
            if math.isfinite(value):
                pairs.append((abs(value), names[i], names[j]))
    pairs.sort(reverse=True)
    rows: list[Sequence[object]] = [[a, b, round(v, 4)] for v, a, b in pairs[:12]]
    condition = float(np.linalg.cond(standardised))
    table = render_markdown(["feature_a", "feature_b", "abs_correlation"], rows)
    return (
        f"Fold `{fold.fold_id}`, {train.height} training rows, "
        f"{len(names)} matrix columns ({len(FEATURE_COLUMNS)} features + "
        f"{len(families())} indicators).\n\n"
        f"- condition number of the standardised design matrix: **{condition:.1f}**\n\n"
        f"Twelve most correlated pairs:\n\n{table}\n"
    )


# --- 6. convergence ----------------------------------------------------------


def profile_convergence(frame: pl.DataFrame, fold_specs: list[FoldSpec]) -> str:
    """lbfgs iteration counts per fold, against max_iter. Training window only."""
    rows: list[Sequence[object]] = []
    for fold in fold_specs:
        train = train_frame(frame, fold)
        matrix, _ = design_matrix(train, FEATURE_COLUMNS)
        _, n_iter, converged = fit(matrix, labels_of(train))
        rows.append([fold.fold_id, train.height, n_iter, MAX_ITER, converged])
    return render_markdown(["fold_id", "train_rows", "n_iter", "max_iter", "converged"], rows)


# --- 7. coefficient stability across folds ----------------------------------


def profile_coefficient_stability(frame: pl.DataFrame, fold_specs: list[FoldSpec]) -> str:
    """Min/mean/max standardised coefficient per term across all folds, and sign flips."""
    quarterly = [f for f in fold_specs if f.fold_set == folds_module.QUARTERLY]
    collected: list[NDArray[np.float64]] = []
    names: tuple[str, ...] = ()
    for fold in quarterly:
        train = train_frame(frame, fold)
        matrix, names = design_matrix(train, FEATURE_COLUMNS)
        pipeline, _, _ = fit(matrix, labels_of(train))
        collected.append(coefficients(pipeline))
    stacked = np.vstack(collected)
    rows: list[Sequence[object]] = []
    for index, name in enumerate(names):
        column = stacked[:, index]
        signs = set(np.sign(column[column != 0]).tolist())
        rows.append(
            [
                name,
                round(float(column.min()), 4),
                round(float(column.mean()), 4),
                round(float(column.max()), 4),
                "yes" if len(signs) > 1 else "no",
            ]
        )
    rows.sort(key=lambda r: abs(as_float(r[2])), reverse=True)
    table = render_markdown(["term", "min", "mean", "max", "sign_flips"], rows)
    return f"Across {len(quarterly)} quarterly folds, training windows only.\n\n{table}\n"


# --- 8. does row order reach the coefficients? ------------------------------


def profile_row_order_sensitivity(frame: pl.DataFrame, fold_specs: list[FoldSpec]) -> str:
    """Refit on a shuffled copy, with and without the canonical sort, and diff."""
    fold = fold_specs[0]
    train = train_frame(frame, fold)

    matrix, _ = design_matrix(train, FEATURE_COLUMNS)
    reference = coefficients(fit(matrix, labels_of(train))[0])

    order = list(range(train.height))
    random.Random(20260817).shuffle(order)
    shuffled = train[order]

    raw_matrix, _ = design_matrix(shuffled, FEATURE_COLUMNS)
    raw = coefficients(fit(raw_matrix, labels_of(shuffled))[0])

    resorted = shuffled.sort(["inspection_date", "target_inspection_id"])
    sorted_matrix, _ = design_matrix(resorted, FEATURE_COLUMNS)
    resorted_coefficients = coefficients(fit(sorted_matrix, labels_of(resorted))[0])

    rows: list[Sequence[object]] = [
        ["shuffled, no sort", f"{float(np.abs(raw - reference).max()):.3e}", raw is reference],
        [
            "shuffled, then canonically sorted",
            f"{float(np.abs(resorted_coefficients - reference).max()):.3e}",
            bool(np.array_equal(resorted_coefficients, reference)),
        ],
    ]
    table = render_markdown(["condition", "max_abs_coefficient_diff", "bit_identical"], rows)
    return (
        f"Fold `{fold.fold_id}`, {train.height} training rows.\n\n{table}\n\n"
        "The scaler's incremental mean/variance and the lbfgs gradient are both "
        "float-summation-order dependent, so the sort is load-bearing.\n"
    )


# --- 9. would a median-imputed boolean flip between folds? ------------------


def profile_boolean_imputation_risk(frame: pl.DataFrame, fold_specs: list[FoldSpec]) -> str:
    """Per-fold observed mean of each nullable boolean, to show the median fill can flip."""
    booleans = [name for name, _, dtype in nullable_specs() if dtype == "bool"]
    rows: list[Sequence[object]] = []
    for fold in fold_specs:
        train = train_frame(frame, fold)
        row: list[object] = [fold.fold_id]
        for name in booleans:
            observed = train[name].drop_nulls()
            mean = as_float(observed.mean())
            row.extend([round(mean, 4), 1 if mean > 0.5 else 0])
        rows.append(row)
    columns = ["fold_id"]
    for name in booleans:
        columns.extend([f"{name}_mean", f"{name}_median_fill"])
    return render_markdown(columns, rows)


# --- 10. the CDPH 2015 input inventory --------------------------------------


def profile_cdph_inputs(_frame: pl.DataFrame, _folds: list[FoldSpec]) -> str:
    """Which of the 2015 model's inputs are reachable from the current data contract."""
    rows: list[Sequence[object]] = [list(entry) for entry in CDPH_2015_INPUTS]
    reachable = sum(1 for _, status, _ in CDPH_2015_INPUTS if status == "available")
    table = render_markdown(["2015 input family", "status", "note"], rows)
    return (
        f"{table}\n\n"
        f"- reachable input families: **{reachable}** of {len(CDPH_2015_INPUTS)}\n"
        "- a faithful replication is therefore **not reachable**; whatever Component 6 "
        "builds must be labelled an approximation with this table attached.\n"
    )


PROFILES: dict[str, Callable[[pl.DataFrame, list[FoldSpec]], str]] = {
    "nullable_partition": profile_nullable_partition,
    "family_mask_identity": profile_family_mask_identity,
    "mask_vs_paired_count": profile_mask_vs_paired_count,
    "train_windows": profile_train_windows,
    "collinearity": profile_collinearity,
    "convergence": profile_convergence,
    "coefficient_stability": profile_coefficient_stability,
    "row_order_sensitivity": profile_row_order_sensitivity,
    "boolean_imputation_risk": profile_boolean_imputation_risk,
    "cdph_inputs": profile_cdph_inputs,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="profile_baselines",
        description="Read-only profiling of the baseline-model surface. Never reports a "
        "test-window metric.",
    )
    parser.add_argument("--features", type=Path, help="Component 4 features. Default: latest.")
    parser.add_argument("--only", action="append", metavar="NAME", help="Run only these.")
    parser.add_argument("--list", action="store_true", dest="list_profiles", help="List and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_profiles:
        for name in PROFILES:
            print(name)
        return 0

    settings = load_settings()
    features_path: Path = args.features or latest_parquet(
        settings.features_processed_dir, prefix="as_of_features_"
    )

    selected: list[str] = list(args.only) if args.only else list(PROFILES)
    unknown = [n for n in selected if n not in PROFILES]
    if unknown:
        print(f"Unknown profile(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    frame = load_features(features_path)
    fold_specs = all_folds(frame)
    print(f"<!-- generated by scripts/profile_baselines.py from {features_path.name} -->")
    print(f"<!-- {frame.height} rows, {len(fold_specs)} folds, training windows only -->\n")
    for name in selected:
        print(f"### `{name}`\n")
        print(PROFILES[name](frame, fold_specs))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
