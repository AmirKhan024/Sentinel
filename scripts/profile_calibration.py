"""Read-only profiling of the calibration windows, before any calibrator is fitted.

Analysis tooling, not library code: it answers one-off questions about a snapshot, nothing
imports it, and it should not ship in the wheel. Output is markdown on stdout, pasted into
``docs/analysis/calibration_findings.md``.

⚠ **No profile in this script may report a test-window outcome.** Component 9's own window
is the fold's *calibration* window, which sits after ``train_end`` and strictly before
``test_start``, and which no component has read until now -- every Component 6, 7 and 8
training log carries a column literally named ``calibration_end_unused``. Scoring it here
is the intended use, not an exception. Two profiles below touch the test window and only
these two: ``base_rate_drift`` reads its *prevalence*, which Component 5 already published
in ``evaluation_folds_*.parquet`` and which is a property of the data rather than of any
model, and ``establishment_recurrence`` counts repeated establishments, which carries no
outcome. No model's test-window score or metric appears anywhere in this file.

The reason for the discipline: Component 5 protects evaluation time, but it cannot protect
against a human reading a test metric, changing a threshold and re-running. That loop is
leakage, it leaves no trace in any artifact, and no check in this repository can detect it.

⚠ **This script is run before ``TIE_THRESHOLD`` is frozen, and is what fixes it.** Profile
6 measures how noisy the selection metric is on an inner-select-sized window. The threshold
is then set from that noise, not from which calibrator happened to win -- the per-fold
winner is printed too, but the rule is declared first and written into ADR 0025 with a date.

Questions this script answers
-----------------------------
1. ``calibration_window_size``  -- how many rows, days and positives does each fold's
                                   calibration window actually hold?
2. ``inner_split_placement``    -- where does a 70/30 whole-day chronological cut land,
                                   and is either side below its declared minimum?
3. ``score_distribution``       -- what do the candidates' calibration-window scores look
                                   like: range, distinct values, saturation?
4. ``logit_round_trip``         -- how far is ``logit(p)`` from the model's own native
                                   decision margin? (justifies calibrating the recovered
                                   logit rather than persisting a margin)
5. ``bin_occupancy``            -- how many rows land in each of 15 equal-mass bins on a
                                   full calibration window and on an inner-select window?
6. ``selection_metric_noise``   -- the bootstrap SD of inner-select log-loss, and of the
                                   paired isotonic-minus-Platt gap. **Fixes TIE_THRESHOLD.**
7. ``base_rate_drift``          -- how far does the calibration window's prevalence sit
                                   from the test window's? (bounds how much residual ECE
                                   is prior shift rather than a failed calibrator)
8. ``establishment_recurrence`` -- how often does one establishment appear more than once
                                   in a window? (sizes the row vs block bootstrap gap)
9. ``isotonic_tie_budget``      -- how many distinct base scores exist per calibration
                                   window, i.e. the ceiling on isotonic's breakpoints?
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.boosting import predict as boosting_predict  # noqa: E402
from sentinel.boosting import preprocess as boosting_preprocess  # noqa: E402
from sentinel.boosting import train as boosting_train  # noqa: E402
from sentinel.boosting.definitions import Estimator  # noqa: E402
from sentinel.boosting.definitions import spec_for as boosting_spec_for  # noqa: E402
from sentinel.config import load_settings  # noqa: E402
from sentinel.evaluation import folds as folds_module  # noqa: E402
from sentinel.evaluation.metrics import (  # noqa: E402
    DEFAULT_CALIBRATION_BINS,
    log_loss,
)
from sentinel.evaluation.models import FoldSpec  # noqa: E402
from sentinel.modeling import predict as modeling_predict  # noqa: E402
from sentinel.modeling import preprocess as modeling_preprocess  # noqa: E402
from sentinel.modeling import train as modeling_train  # noqa: E402
from sentinel.modeling.definitions import spec_for as modeling_spec_for  # noqa: E402
from sentinel.neural import predict as neural_predict  # noqa: E402
from sentinel.neural import preprocess as neural_preprocess  # noqa: E402
from sentinel.neural import train as neural_train  # noqa: E402
from sentinel.neural.definitions import spec_for as neural_spec_for  # noqa: E402
from sentinel.neural.train import inner_split_date  # noqa: E402
from sentinel.query import duckdb_queries  # noqa: E402

#: The candidate set this script profiles. ``xgboost_chain_embeddings`` is deliberately
#: absent: reproducing it needs the ``neural_embeddings`` donor fitted first, which
#: doubles the neural cost for a candidate the profiling questions do not depend on. It
#: is calibrated in the component proper, labelled experimental.
CANDIDATES: tuple[str, ...] = (
    "logistic_regression",
    "xgboost",
    "lightgbm",
    "neural_numeric_only",
)

#: The proportion of each calibration window held back to choose between Platt and
#: isotonic. Larger than Component 8's 0.15 because a calibration window is an order of
#: magnitude smaller than a training window; profile 2 is the check on that reasoning.
INNER_SELECT_FRACTION = 0.30

#: Replications for profile 6. Matches the sensitivity study Component 5 already runs.
NOISE_REPLICATIONS = 1000
NOISE_SEED = 20260819

#: Below these the fold is refused rather than calibrated on a window too small to mean
#: anything. Profile 2 exists to confirm no fold trips them.
MIN_INNER_FIT_ROWS = 400
MIN_INNER_SELECT_ROWS = 250


# --- shared helpers ----------------------------------------------------------


def _folds(frame: pl.DataFrame) -> list[FoldSpec]:
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    if start is None or end is None:
        raise SystemExit("feature table has no usable reference dates")
    quarterly = folds_module.quarterly_folds(data_start=start, data_end=end)
    return [*quarterly, *folds_module.covid_shift_fold(data_end=end)]


def _calibration_frame(frame: pl.DataFrame, fold: FoldSpec) -> pl.DataFrame:
    """The fold's calibration rows, in the project's canonical order.

    ``assign_split`` remains the single definition of every window -- a hand-rolled
    ``rd.is_between`` here would be a second one, and the two would drift.
    """
    assigned = folds_module.assign_split(frame, fold)
    return assigned.filter(pl.col("split") == "calibration").sort(["rd", "target_inspection_id"])


def _logit(p: float) -> float:
    """``log(p / (1 - p))``, computed so neither tail loses precision.

    ``log(p) - log1p(-p)`` rather than ``log(p / (1 - p))``: for p near 1 the subtraction
    ``1 - p`` cancels catastrophically, while ``log1p`` is accurate there by construction.
    """
    return math.log(p) - math.log1p(-p)


def _number(value: object, default: float = 0.0) -> float:
    """Narrow a polars aggregate to a float.

    Polars types a column aggregate as a wide union, so every arithmetic use of one would
    otherwise need an ignore comment. ``evaluation.folds`` narrows once for the same reason.
    """
    return float(value) if isinstance(value, int | float) else default


def _fmt(value: float | None, places: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


# --- base-score regeneration -------------------------------------------------
#
# No component persisted a fitted model, and no component scored a calibration window, so
# the scores this script profiles have to be produced by re-executing Components 6, 7 and
# 8's own fit functions -- unchanged, imported rather than copied, at their own seed. The
# component proper additionally proves the re-execution is faithful by re-deriving the
# test window and comparing it to the committed artifact bit for bit; that gate does not
# belong in a profiling script, which reports no test-window number at all.


def _fit_and_score(
    model: str, frame: pl.DataFrame, fold: FoldSpec
) -> tuple[list[str], list[float], list[float]]:
    """Fit one candidate on one fold's training window; score its calibration window.

    Returns ids, probabilities and the model's own native decision margin, aligned.
    """
    training = modeling_train.training_frame(frame, fold)
    window = _calibration_frame(frame, fold)

    if model == "logistic_regression":
        spec = modeling_spec_for(model)
        fitted = modeling_train.fit_fold(spec, training, fold)
        ids, scores = modeling_predict.score_window(fitted, window)
        matrix = modeling_preprocess.to_matrix(window, spec)
        margins = [float(v) for v in fitted.pipeline.decision_function(matrix)]
        return ids, scores, margins

    if model in ("xgboost", "lightgbm"):
        bspec = boosting_spec_for(model)
        bfitted = boosting_train.fit_fold(bspec, training, fold)
        ids, scores = boosting_predict.score_window(bfitted, window)
        bmatrix = boosting_preprocess.tree_matrix(window, bspec)
        raw: Any = (
            bfitted.estimator.predict(bmatrix, output_margin=True)
            if bspec.estimator is Estimator.XGBOOST
            else bfitted.estimator.predict_proba(bmatrix, raw_score=True)
        )
        return ids, scores, [float(v) for v in np.asarray(raw).ravel()]

    if model == "neural_numeric_only":
        import torch

        nspec = neural_spec_for(model)
        nfitted = neural_train.fit_fold(nspec, training, fold)
        ids, scores = neural_predict.score_window(nfitted, window)
        network, preprocessor = neural_train.scorer_for(nfitted)
        dense = neural_preprocess.dense_matrix(window, nspec, preprocessor, nfitted.encoding)
        codes = neural_preprocess.code_matrix(window, nspec, nfitted.encoding)
        network.eval()
        with torch.no_grad():
            logits = network(
                torch.from_numpy(np.ascontiguousarray(dense, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(codes, dtype=np.int64)),
            )
        return ids, scores, [float(v) for v in logits.numpy().ravel()]

    raise SystemExit(f"unknown candidate {model!r}")


def base_scores(frame: pl.DataFrame, models: Sequence[str]) -> pl.DataFrame:
    """Calibration-window scores for every requested candidate on every fold.

    The expensive part of this script by a wide margin, so it is computed once and shared
    by profiles 3 through 6 and 9.
    """
    rows: list[dict[str, object]] = []
    for fold in _folds(frame):
        window = _calibration_frame(frame, fold)
        labels = dict(
            zip(
                (str(v) for v in window["target_inspection_id"].to_list()),
                (int(v) for v in window["target"].to_list()),
                strict=True,
            )
        )
        dates = dict(
            zip(
                (str(v) for v in window["target_inspection_id"].to_list()),
                window["rd"].to_list(),
                strict=True,
            )
        )
        for model in models:
            started = time.perf_counter()
            ids, scores, margins = _fit_and_score(model, frame, fold)
            elapsed = time.perf_counter() - started
            print(
                f"<!-- {model} / {fold.fold_id}: {len(ids)} calibration rows in "
                f"{elapsed:.1f}s -->",
                file=sys.stderr,
            )
            for row_id, score, margin in zip(ids, scores, margins, strict=True):
                rows.append(
                    {
                        "model_name": model,
                        "fold_set": fold.fold_set,
                        "fold_id": fold.fold_id,
                        "target_inspection_id": row_id,
                        "rd": dates[row_id],
                        "target": labels[row_id],
                        "base_score": score,
                        "native_margin": margin,
                    }
                )
    return pl.DataFrame(rows).sort(["model_name", "fold_set", "fold_id", "target_inspection_id"])


def _inner_split(window: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, object]:
    """Cut one calibration window chronologically on a whole-day boundary."""
    cut = inner_split_date(window, INNER_SELECT_FRACTION)
    return window.filter(pl.col("rd") < cut), window.filter(pl.col("rd") >= cut), cut


# --- 1. how big is a calibration window? -------------------------------------


def calibration_window_size(frame: pl.DataFrame, scores: pl.DataFrame | None) -> str:
    del scores
    rows = []
    for fold in _folds(frame):
        window = _calibration_frame(frame, fold)
        positives = int(window["target"].sum())
        rows.append(
            [
                fold.fold_id,
                f"{fold.calibration_start} .. {fold.calibration_end}",
                str(window.height),
                str(window["rd"].n_unique()),
                str(positives),
                _fmt(positives / window.height if window.height else None),
            ]
        )
    return _table(
        ["fold_id", "calibration window", "rows", "days", "positives", "base rate"], rows
    )


# --- 2. where does the inner split land? -------------------------------------


def inner_split_placement(frame: pl.DataFrame, scores: pl.DataFrame | None) -> str:
    del scores
    rows = []
    for fold in _folds(frame):
        window = _calibration_frame(frame, fold)
        fit, select, cut = _inner_split(window)
        flags = []
        if fit.height < MIN_INNER_FIT_ROWS:
            flags.append(f"fit<{MIN_INNER_FIT_ROWS}")
        if select.height < MIN_INNER_SELECT_ROWS:
            flags.append(f"select<{MIN_INNER_SELECT_ROWS}")
        rows.append(
            [
                fold.fold_id,
                str(cut),
                str(fit.height),
                _fmt(_number(fit["target"].mean())),
                str(select.height),
                _fmt(_number(select["target"].mean())),
                _fmt(select.height / window.height if window.height else None, 3),
                ", ".join(flags) if flags else "ok",
            ]
        )
    return _table(
        [
            "fold_id",
            "cut date",
            "inner-fit rows",
            "fit rate",
            "inner-select rows",
            "select rate",
            "select share",
            "minimums",
        ],
        rows,
    )


# --- 3. what do the calibration-window scores look like? ---------------------


def score_distribution(frame: pl.DataFrame, scores: pl.DataFrame | None) -> str:
    del frame
    if scores is None:
        return "_(requires the base-score regeneration; run without ``--cheap``)_"
    rows = []
    for (model,), group in scores.group_by(["model_name"], maintain_order=True):
        values = group["base_score"].to_list()
        low, high = min(values), max(values)
        rows.append(
            [
                str(model),
                str(len(values)),
                f"{low:.6f}",
                f"{high:.6f}",
                _fmt(_logit(low), 3),
                _fmt(_logit(high), 3),
                str(group["base_score"].n_unique()),
                str(sum(1 for v in values if v in (0.0, 1.0))),
            ]
        )
    return _table(
        ["model", "rows", "min p", "max p", "min logit", "max logit", "distinct", "saturated"],
        rows,
    )


# --- 4. is the recovered logit the model's own margin? -----------------------


def logit_round_trip(frame: pl.DataFrame, scores: pl.DataFrame | None) -> str:
    del frame
    if scores is None:
        return "_(requires the base-score regeneration; run without ``--cheap``)_"
    rows = []
    for (model,), group in scores.group_by(["model_name"], maintain_order=True):
        errors = [
            abs(_logit(p) - m)
            for p, m in zip(
                group["base_score"].to_list(), group["native_margin"].to_list(), strict=True
            )
        ]
        rows.append(
            [
                str(model),
                f"{max(errors):.3e}",
                f"{sum(errors) / len(errors):.3e}",
                str(sum(1 for e in errors if e > 1e-9)),
            ]
        )
    return _table(["model", "max |logit(p) - margin|", "mean", "rows over 1e-9"], rows)


# --- 5. how full is an equal-mass bin? ---------------------------------------


def bin_occupancy(frame: pl.DataFrame, scores: pl.DataFrame | None) -> str:
    if scores is None:
        return "_(requires the base-score regeneration; run without ``--cheap``)_"
    rows = []
    for fold in _folds(frame):
        window = _calibration_frame(frame, fold)
        _, select, _ = _inner_split(window)
        rows.append(
            [
                fold.fold_id,
                str(window.height),
                _fmt(window.height / DEFAULT_CALIBRATION_BINS, 1),
                str(select.height),
                _fmt(select.height / DEFAULT_CALIBRATION_BINS, 1),
                str(int(round(select.height / DEFAULT_CALIBRATION_BINS * 0.42))),
            ]
        )
    return _table(
        [
            "fold_id",
            "cal rows",
            "rows/bin (full)",
            "select rows",
            "rows/bin (select)",
            "~positives/bin",
        ],
        rows,
    )


# --- 6. how noisy is the selection metric? -----------------------------------


def _fit_platt(x: Sequence[float], y: Sequence[int]) -> Callable[[Sequence[float]], list[float]]:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000, fit_intercept=True)
    model.fit(np.asarray(x, dtype=np.float64).reshape(-1, 1), np.asarray(y, dtype=np.int64))

    def apply(values: Sequence[float]) -> list[float]:
        proba = model.predict_proba(np.asarray(values, dtype=np.float64).reshape(-1, 1))
        return [float(v) for v in proba[:, 1]]

    return apply


def _fit_isotonic(x: Sequence[float], y: Sequence[int]) -> Callable[[Sequence[float]], list[float]]:
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    model.fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64))

    def apply(values: Sequence[float]) -> list[float]:
        return [float(v) for v in model.predict(np.asarray(values, dtype=np.float64))]

    return apply


def selection_metric_noise(frame: pl.DataFrame, scores: pl.DataFrame | None) -> str:
    """The number that fixes ``TIE_THRESHOLD``.

    For each (model, fold): fit both calibrators on the inner-fit portion, score the
    inner-select portion, then resample the inner-select rows 1,000 times to see how far
    the comparison moves under nothing but sampling noise. The *paired* gap SD is the one
    that matters -- both methods are scored on the same resample, so their shared
    variation cancels and what is left is the resolution of the comparison itself.
    """
    if scores is None:
        return "_(requires the base-score regeneration; run without ``--cheap``)_"

    rows = []
    gap_sds: list[float] = []
    for fold in _folds(frame):
        window = _calibration_frame(frame, fold)
        fit_rows, select_rows, _ = _inner_split(window)
        fit_ids = set(str(v) for v in fit_rows["target_inspection_id"].to_list())
        select_ids = set(str(v) for v in select_rows["target_inspection_id"].to_list())

        for (model,), group in scores.filter(
            pl.col("fold_id") == fold.fold_id
        ).group_by(["model_name"], maintain_order=True):
            fit_part = group.filter(pl.col("target_inspection_id").is_in(fit_ids))
            select_part = group.filter(pl.col("target_inspection_id").is_in(select_ids))
            if fit_part.height < 2 or select_part.height < 2:
                continue

            fit_p = fit_part["base_score"].to_list()
            fit_y = [int(v) for v in fit_part["target"].to_list()]
            sel_p = select_part["base_score"].to_list()
            sel_y = [int(v) for v in select_part["target"].to_list()]
            if len(set(fit_y)) < 2:
                continue

            platt = _fit_platt([_logit(v) for v in fit_p], fit_y)
            isotonic = _fit_isotonic(fit_p, fit_y)
            platt_out = platt([_logit(v) for v in sel_p])
            iso_out = isotonic(sel_p)

            platt_ll = log_loss(sel_y, platt_out)
            iso_ll = log_loss(sel_y, iso_out)

            rng = np.random.default_rng(np.random.SeedSequence([NOISE_SEED, hash(model) % 9973]))
            n = len(sel_y)
            platt_draws: list[float] = []
            iso_draws: list[float] = []
            gap_draws: list[float] = []
            for _ in range(NOISE_REPLICATIONS):
                idx = rng.integers(0, n, size=n)
                y_draw = [sel_y[i] for i in idx]
                if len(set(y_draw)) < 2:
                    continue
                a = log_loss(y_draw, [platt_out[i] for i in idx])
                b = log_loss(y_draw, [iso_out[i] for i in idx])
                platt_draws.append(a)
                iso_draws.append(b)
                gap_draws.append(b - a)

            gap_sd = float(np.std(gap_draws, ddof=1)) if len(gap_draws) > 1 else float("nan")
            gap_sds.append(gap_sd)
            rows.append(
                [
                    str(model),
                    fold.fold_id,
                    str(n),
                    _fmt(platt_ll),
                    _fmt(iso_ll),
                    _fmt(iso_ll - platt_ll),
                    _fmt(float(np.std(platt_draws, ddof=1)) if len(platt_draws) > 1 else None),
                    _fmt(gap_sd),
                    "platt" if platt_ll <= iso_ll else "isotonic",
                ]
            )

    table = _table(
        [
            "model",
            "fold_id",
            "select rows",
            "platt log-loss",
            "isotonic log-loss",
            "gap",
            "SD(log-loss)",
            "SD(paired gap)",
            "lower",
        ],
        rows,
    )
    if gap_sds:
        finite = [v for v in gap_sds if math.isfinite(v)]
        summary = (
            f"\n\nPaired-gap SD across all (model, fold) cells: "
            f"min {min(finite):.4f}, median {float(np.median(finite)):.4f}, "
            f"max {max(finite):.4f} over {len(finite)} cells."
        )
        return table + summary
    return table


# --- 7. does the prior move between calibration and test? --------------------


def base_rate_drift(frame: pl.DataFrame, scores: pl.DataFrame | None) -> str:
    """Prevalence only -- a property of the data, not of any model. See the banner."""
    del scores
    rows = []
    for fold in _folds(frame):
        stats = folds_module.fold_stats(frame, fold)
        cal, test = stats.calibration_positive_rate, stats.test_positive_rate
        gap = None if cal is None or test is None else test - cal
        rows.append(
            [
                fold.fold_id,
                str(stats.calibration_rows),
                _fmt(cal),
                str(stats.test_rows),
                _fmt(test),
                _fmt(gap),
                _fmt(abs(gap) if gap is not None else None),
            ]
        )
    return _table(
        ["fold_id", "cal rows", "cal rate", "test rows", "test rate", "test - cal", "abs gap"],
        rows,
    )


# --- 8. do establishments repeat inside a window? ----------------------------


def establishment_recurrence(frame: pl.DataFrame, scores: pl.DataFrame | None) -> str:
    """Counts of repeated establishments -- no outcome is read on either window."""
    del scores
    rows = []
    for fold in _folds(frame):
        for label, window in (
            ("calibration", _calibration_frame(frame, fold)),
            ("test", folds_module.window_frame(frame, fold)),
        ):
            if window.height == 0:
                continue
            counts = window.group_by("establishment_id").len()
            repeated = counts.filter(pl.col("len") > 1)
            rows.append(
                [
                    fold.fold_id,
                    label,
                    str(window.height),
                    str(counts.height),
                    str(repeated.height),
                    _fmt(window.height / counts.height, 3),
                    str(int(_number(counts["len"].max()))),
                ]
            )
    return _table(
        [
            "fold_id",
            "window",
            "rows",
            "establishments",
            "repeated",
            "rows/establishment",
            "max repeats",
        ],
        rows,
    )


# --- 9. how many breakpoints can isotonic have? ------------------------------


def isotonic_tie_budget(frame: pl.DataFrame, scores: pl.DataFrame | None) -> str:
    del frame
    if scores is None:
        return "_(requires the base-score regeneration; run without ``--cheap``)_"
    rows = []
    for (model, fold_id), group in scores.group_by(
        ["model_name", "fold_id"], maintain_order=True
    ):
        distinct = group["base_score"].n_unique()
        rows.append(
            [
                str(model),
                str(fold_id),
                str(group.height),
                str(distinct),
                _fmt(distinct / group.height, 4),
            ]
        )
    return _table(["model", "fold_id", "cal rows", "distinct scores", "share distinct"], rows)


PROFILES: dict[str, Callable[[pl.DataFrame, pl.DataFrame | None], str]] = {
    "calibration_window_size": calibration_window_size,
    "inner_split_placement": inner_split_placement,
    "score_distribution": score_distribution,
    "logit_round_trip": logit_round_trip,
    "bin_occupancy": bin_occupancy,
    "selection_metric_noise": selection_metric_noise,
    "base_rate_drift": base_rate_drift,
    "establishment_recurrence": establishment_recurrence,
    "isotonic_tie_budget": isotonic_tie_budget,
}

#: Profiles that do not need a single model fitted. Useful while iterating.
CHEAP_PROFILES: tuple[str, ...] = (
    "calibration_window_size",
    "inner_split_placement",
    "base_rate_drift",
    "establishment_recurrence",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, help="Component 4 feature table.")
    parser.add_argument("--only", action="append", help="Profile to run; repeatable.")
    parser.add_argument(
        "--models",
        action="append",
        help=f"Candidate to regenerate; repeatable. Defaults to {', '.join(CANDIDATES)}.",
    )
    parser.add_argument(
        "--cheap",
        action="store_true",
        help="Run only the profiles that need no model fitted.",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    features_path = args.features or duckdb_queries.latest_parquet(
        settings.features_processed_dir, prefix="as_of_features_"
    )
    frame = pl.read_parquet(features_path).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )

    requested = list(CHEAP_PROFILES) if args.cheap else (args.only or list(PROFILES))
    unknown = [name for name in requested if name not in PROFILES]
    if unknown:
        raise SystemExit(f"unknown profile(s): {', '.join(unknown)}")

    models = args.models or list(CANDIDATES)
    needs_scores = any(name not in CHEAP_PROFILES for name in requested)
    scores = base_scores(frame, models) if needs_scores else None

    print(f"<!-- generated by scripts/profile_calibration.py from {features_path.name} -->")
    print(f"<!-- {frame.height} feature rows; candidates: {', '.join(models)} -->")
    if scores is not None:
        print(f"<!-- {scores.height} regenerated calibration-window scores -->")
    for name in requested:
        print()
        print(f"### {name}")
        print()
        print(PROFILES[name](frame, scores))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
