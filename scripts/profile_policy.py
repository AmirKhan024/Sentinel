"""Read-only profiling of the decision surface, before any policy code is written.

Analysis tooling, not library code: it answers one-off questions about a snapshot, nothing
imports it, and it should not ship in the wheel. Output is markdown on stdout, pasted into
``docs/analysis/policy_findings.md``.

⚠ **This script fits nothing, scores nothing, ranks nothing new and changes nothing.** Every
model is frozen, every prediction and every metric already exists on disk. Component 13 is a
policy layer over artifacts, so the only computation here is arithmetic over columns that
Components 4, 5, 9 and 12 already wrote.

⚠ **This script is run before the eligibility contract, the reserve grid and the
model-selection rule are frozen, and is what fixes them.** Profiles 1, 2 and 3 decide which
column defines coverage eligibility and what reserve shares are non-degenerate; profile 5
supplies the three axes of the pre-registered selection rule. A policy constant chosen from
expectation rather than measurement is a guess wearing a decimal point -- Component 9 set
three thresholds that way and had to correct all three.

⚠ **It reads labels, and it reads Component 5's metric artifact.** Both are legitimate and
neither is model selection by hand. The labels are needed because "what does a coverage
reserve cost" is a question about positives, and refusing to measure the cost would be the
dishonest choice. The metric artifact is read to *apply* a rule that is written down before
the numbers are looked at, not to browse for a winner: profile 5 prints the three axes in the
frozen order and the rule is stated in the findings document above the table.

⚠ **It reads no Component 12 number into the selection rule.** HANDOFF forbids it, and
profile 7 -- the only profile that touches the group frame -- is descriptive, is reported
after the rule, and feeds a governance advisory rather than a score.

Questions this script answers
-----------------------------
1.  ``eligibility_definition``   -- which missing-history rule should define coverage
                                    eligibility, how large is each candidate population, and
                                    what is its outcome rate? **Fixes ELIGIBILITY_COLUMN.**
2.  ``eligibility_by_fold``      -- how many eligible rows does each fold's test window hold,
                                    and what share of the window is that? **Fixes the reserve
                                    share anchor: the measured population share.**
3.  ``capacity_and_reserve``     -- what k does each fold actually have at each capacity
                                    level, and what integer reserve does each candidate share
                                    buy? **Fixes POLICY_GRID, and shows where a configuration
                                    is degenerate at the smallest cutoff.**
4.  ``risk_ranking_of_eligible`` -- under pure risk ranking, how many eligible rows reach the
                                    top k at all, and what share of their positives is found?
                                    **The measured problem statement.**
5.  ``model_selection_axes``     -- the three operational axes of the pre-registered
                                    selection rule, per candidate model, from Components 5
                                    and 9's own artifacts. **Fixes PRODUCTION_MODEL.**
6.  ``opportunity_cost_bound``   -- what is in the marginal band a reserve would displace,
                                    and therefore what can a reserve cost?
7.  ``coverage_trend_and_floor_binding``
                                 -- does the queue's coverage of no-history establishments
                                    hold up across folds, and where would a coverage floor
                                    actually bind? **Decides whether the floor mechanism is a
                                    no-op, and fixes the reserve semantics.**
8.  ``unknown_group_overlap``    -- how does eligibility relate to Component 12's
                                    ``__UNKNOWN__`` group? **Descriptive, and it informs a
                                    governance advisory, never a score.**
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sentinel.evaluation.folds as folds_module  # noqa: E402
import sentinel.evaluation.simulate as simulate  # noqa: E402
from sentinel.config import load_settings  # noqa: E402
from sentinel.evaluation.metrics import top_k_indices  # noqa: E402
from sentinel.evaluation.models import FoldSpec  # noqa: E402
from sentinel.query.duckdb_queries import latest_parquet  # noqa: E402

#: Candidate coverage-eligibility rules, as ``(name, column, description)``. Each is a count
#: column from Component 4 that is zero exactly when a named ``NullRule`` family fires, so
#: eligibility is the *cause* of the missing features rather than a proxy for it. Profile 1
#: sweeps all four and one is frozen.
ELIGIBILITY_CANDIDATES: tuple[tuple[str, str, str], ...] = (
    (
        "no_code_era_canvass",
        "prior_canvass_count_code_era",
        "no canvass on or after 2018-07-01; the four priority features are NULL",
    ),
    (
        "no_prior_canvass",
        "prior_canvass_count",
        "never canvassed; days_since_last_canvass and fail_at_last_canvass are NULL",
    ),
    (
        "no_inspected_canvass",
        "prior_canvass_inspected_count",
        "no canvass that reached a Pass/Fail result; prior_canvass_fail_rate is NULL",
    ),
    (
        "no_prior_inspection",
        "prior_inspection_count_any_type",
        "no inspection of any type; days_since_any_inspection is NULL",
    ),
)

#: The reserve shares swept in profile 3. Anchored on the eligible population share that
#: profile 2 measures: nothing, half of it, all of it, twice it. The realised anchor is
#: printed beside the grid so the reader can check that 0.10 is the measured share rather
#: than a round number that happens to look like one.
RESERVE_GRID: tuple[float, ...] = (0.00, 0.05, 0.10, 0.20)

#: The capacity cutoffs. The same five Component 12 froze as ``K_LEVELS``, for the same
#: reason, so a policy number and an audit number describe the same operating point.
#: ``capacity_k_values`` derives every one of them from the window's own measured median
#: daily rate.
K_LEVELS: tuple[str, ...] = ("k_pct_01", "k_pct_05", "k_pct_10", "k_1_day", "k_1_week")

#: The models a policy may carry. ``xgboost_chain_embeddings_platt`` is excluded before any
#: number is read: ADR 0022 makes it experimental and ADR 0031 reports it unsupported by
#: Component 11, so it cannot be explained to an inspector and is not a deployment candidate.
CANDIDATE_MODELS: tuple[str, ...] = (
    "lightgbm_platt",
    "logistic_regression_platt",
    "neural_numeric_only_platt",
    "xgboost_platt",
)

#: The experimental model, named so its exclusion is visible rather than silent.
EXCLUDED_MODEL = "xgboost_chain_embeddings_platt"

#: The band this plan started with, kept only so the discarded alternative stays visible.
#: Component 8 measured a five-seed ROC-AUC spread of 0.0058 for the network. Thresholding
#: an *NDE* difference with a *ROC-AUC* spread is a unit error, and profile 5 replaces it
#: with Component 5's own NDE sensitivity interval. The number stays here because profile 5
#: reports which model each rule would have chosen, and a discarded rule that leaves no
#: trace is indistinguishable from one that was never considered.
DISCARDED_SEED_BAND = 0.0058

#: One model carries every structural question about eligibility and capacity: all five
#: calibrated models score an identical id set, so the eligible population in a window is a
#: property of the fold, not of the estimator. Profile 4 uses every candidate instead.
PROBE_MODEL = "xgboost_platt"

#: Component 12's absence token. A real group value, not a null.
UNKNOWN = "__UNKNOWN__"

#: Rows shown per table where a full listing would be noise.
DISPLAY_ROWS = 18

BREAK = "\n\n"


# --- shared helpers ----------------------------------------------------------


def _folds(frame: pl.DataFrame) -> list[FoldSpec]:
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    if start is None or end is None:
        raise SystemExit("feature table has no usable reference dates")
    quarterly = folds_module.quarterly_folds(data_start=start, data_end=end)
    return [*quarterly, *folds_module.covid_shift_fold(data_end=end)]


def _fmt(value: float, places: int = 4) -> str:
    return f"{value:.{places}f}"


def _as_float(value: object) -> float:
    """Coerce a polars aggregate to a float, narrowly enough for strict mypy."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"expected a numeric aggregate, got {type(value).__name__}")


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _eligible_expr(column: str) -> pl.Expr:
    """The eligibility predicate: a zero count, and a null is never eligible.

    ``fill_null(-1)`` rather than treating a null as zero. Every candidate column carries
    ``NullRule.NEVER`` in Component 4 -- a count of zero is a true observation and the column
    is never null -- so this branch should be unreachable, and profile 1 asserts that it is.
    Silently mapping a null to "eligible" would admit rows about which nothing at all is
    known, which is the exact defect this component exists to make impossible.
    """
    return pl.col(column).fill_null(-1) == 0


def _windows(frame: pl.DataFrame) -> list[tuple[FoldSpec, pl.DataFrame, int]]:
    """Each fold's test window, in canonical order, with its measured daily capacity."""
    out: list[tuple[FoldSpec, pl.DataFrame, int]] = []
    for fold in _folds(frame):
        stats = folds_module.fold_stats(frame, fold)
        test = folds_module.window_frame(frame, fold)
        if test.is_empty():
            continue
        out.append((fold, test, max(1, int(stats.test_median_daily_capacity or 1))))
    return out


def _k_values(test: pl.DataFrame, median_daily: int) -> dict[str, int]:
    window = simulate.build_window(
        ids=test["target_inspection_id"].to_list(),
        labels=test["target"].to_list(),
        dates=test["rd"].to_list(),
    )
    values: dict[str, int] = simulate.capacity_k_values(window, median_daily=median_daily)
    return values


def _scored(sources: dict[str, pl.DataFrame], model: str) -> pl.DataFrame:
    """One model's calibrated scores, joined to the label and the eligibility columns."""
    predictions = sources["predictions"].filter(pl.col("model_name") == model)
    if predictions.is_empty():
        raise SystemExit(f"model {model} absent from the calibrated artifact")
    columns = ["target_inspection_id", "target", "rd", *(c for _, c, _ in ELIGIBILITY_CANDIDATES)]
    return predictions.select("target_inspection_id", "score", "fold_set", "fold_id").join(
        sources["features"].select(columns), on="target_inspection_id", how="left"
    )


# --- profiles ----------------------------------------------------------------


def eligibility_definition(sources: dict[str, pl.DataFrame]) -> str:
    """Which missing-history rule should define coverage eligibility?

    The reserve exists because the model cannot rank establishments whose predictive history
    is absent. So the eligibility column should be the one whose absence *is* that history,
    and it should carve a population large enough to allocate against and small enough that
    allocating to it is a decision rather than a redistribution of the whole queue.
    """
    frame = sources["features"]
    total = frame.height

    rows: list[list[str]] = []
    for name, column, description in ELIGIBILITY_CANDIDATES:
        eligible = frame.filter(_eligible_expr(column))
        nulls = int(frame[column].null_count())
        n = eligible.height
        positives = int(_as_float(eligible["target"].sum())) if n else 0
        rows.append(
            [
                f"`{name}`",
                f"`{column}`",
                f"{n:,}",
                _fmt(n / total),
                f"{positives:,}",
                _fmt(positives / n) if n else "n/a",
                f"{nulls}",
                description,
            ]
        )

    overall_rate = _as_float(frame["target"].mean())
    body = _table(
        [
            "rule",
            "column",
            "eligible rows",
            "share",
            "positives",
            "base rate",
            "nulls",
            "what is missing",
        ],
        rows,
    )

    reading = (
        f"City-wide outcome rate over all {total:,} rows: **{_fmt(overall_rate)}**."
        + BREAK
        + "**`no_code_era_canvass` is frozen as the eligibility contract.** It is the exact "
        "condition under which `prior_canvass_priority_count`, "
        "`prior_canvass_priority_foundation_count`, `prior_canvass_priority_rate` and "
        "`priority_at_last_canvass` are NULL -- the four features that encode the outcome the "
        "model is predicting. It is therefore the *cause* of the ranking difficulty, not a "
        "correlate of it, and Component 11 measured that two of four models rank the "
        "corresponding missingness indicator second or third in importance."
        + BREAK
        + "`no_prior_inspection` is the strictest and the most intuitive rule, and it is "
        "rejected as the reserve gate on size: a reserve scaled to a population this small is "
        "zero or one slot at every capacity level below a week, which is an allocation "
        "mechanism that cannot be measured. It is carried on the artifact as a secondary "
        "reporting flag instead, because it names a genuinely distinct population."
        + BREAK
        + "The `nulls` column is the assertion that matters: each candidate carries "
        "`NullRule.NEVER` in Component 4, so a zero is a real observation of no history rather "
        "than an absent measurement. If any count were ever non-zero the eligibility predicate "
        "would be admitting rows about which nothing at all is known, and `_eligible_expr` "
        "refuses a null rather than treating it as a zero."
    )
    return body + BREAK + reading


def eligibility_by_fold(sources: dict[str, pl.DataFrame]) -> str:
    """How many eligible rows does each test window hold, and what share is that?

    This is what anchors the reserve grid. A reserve share is defensible when it is read
    against the share of the population it serves: at the measured share, the reserve is
    proportional representation; below it, under-provision; above it, deliberate
    over-provision. None of the three is right by construction, and that is the point.
    """
    frame = sources["features"]
    column = ELIGIBILITY_CANDIDATES[0][1]

    rows: list[list[str]] = []
    quarterly_eligible = 0
    quarterly_rows = 0
    for fold, test, _ in _windows(frame):
        eligible = test.filter(_eligible_expr(column))
        n_elig = eligible.height
        pos_elig = int(_as_float(eligible["target"].sum())) if n_elig else 0
        total_pos = int(_as_float(test["target"].sum()))
        if fold.fold_set == folds_module.QUARTERLY:
            quarterly_eligible += n_elig
            quarterly_rows += test.height
        rows.append(
            [
                fold.fold_id,
                f"{test.height:,}",
                f"{n_elig:,}",
                _fmt(n_elig / test.height),
                f"{pos_elig:,}",
                _fmt(pos_elig / n_elig) if n_elig else "n/a",
                _fmt(_as_float(test["target"].mean())),
                _fmt(pos_elig / total_pos) if total_pos else "n/a",
            ]
        )

    share = quarterly_eligible / quarterly_rows if quarterly_rows else 0.0
    body = _table(
        [
            "fold",
            "test rows",
            "eligible",
            "eligible share",
            "eligible positives",
            "eligible base rate",
            "window base rate",
            "share of positives",
        ],
        rows,
    )
    reading = (
        f"**Pooled across the quarterly test windows: {quarterly_eligible:,} of "
        f"{quarterly_rows:,} rows are coverage-eligible, a share of {_fmt(share)}.** That "
        "number is the anchor for the reserve grid, and it is the only reason a value near "
        "`0.10` appears in it. The grid is then no reserve, half the share, the share itself, "
        "and twice the share -- four points around a measurement rather than four round "
        "numbers."
        + BREAK
        + "Read the last two columns together. Where the eligible base rate sits below the "
        "window base rate, a reserve is spending capacity on a population that is genuinely "
        "less likely to be cited, and the opportunity cost is real rather than notional. "
        "Where it sits above, the pure-risk ranking is leaving findable positives on the "
        "table -- which would be a much stronger argument for a reserve. The profile reports "
        "which of the two is actually true rather than assuming."
    )
    return body + BREAK + reading


def capacity_and_reserve(sources: dict[str, pl.DataFrame]) -> str:
    """What integer reserve does each candidate share buy at each real capacity level?

    A share is a fiction until it is floored into a slot count. ``floor`` is used rather than
    ``round`` so a reserve can never overspend its declared allocation, and the consequence is
    that small shares vanish at small cutoffs. Whether that is acceptable is a policy
    question; whether it happens is a measurement, and this is it.
    """
    frame = sources["features"]
    column = ELIGIBILITY_CANDIDATES[0][1]

    rows: list[list[str]] = []
    degenerate: list[str] = []
    for fold, test, median_daily in _windows(frame):
        if fold.fold_set != folds_module.QUARTERLY:
            continue
        k_values = _k_values(test, median_daily)
        n_eligible = test.filter(_eligible_expr(column)).height
        for k_name in K_LEVELS:
            k = k_values[k_name]
            reserves = [min(int(share * k), n_eligible) for share in RESERVE_GRID]
            if any(r == 0 for r, s in zip(reserves, RESERVE_GRID, strict=True) if s > 0.0):
                degenerate.append(f"{fold.fold_id}/{k_name} (k={k})")
            rows.append(
                [
                    fold.fold_id,
                    k_name,
                    f"{k:,}",
                    f"{n_eligible:,}",
                    *[f"{r}" for r in reserves],
                ]
            )

    shown = rows[:DISPLAY_ROWS] if len(rows) > DISPLAY_ROWS else rows
    body = _table(
        [
            "fold",
            "k level",
            "k",
            "eligible available",
            *[f"reserve @{s:.2f}" for s in RESERVE_GRID],
        ],
        shown,
    )
    note = (
        f"Showing the first {len(shown)} of {len(rows)} (fold, k level) cells."
        if len(shown) < len(rows)
        else f"{len(rows)} (fold, k level) cells, all shown."
    )
    if degenerate:
        verdict = (
            f"**{len(degenerate)} of {len(rows)} cell(s) floor a non-zero share to a zero "
            f"reserve.** First few: {', '.join(degenerate[:6])}. At the smallest cutoff a "
            "quarterly window offers only a handful of slots, so a small share is zero slots. "
            "This is recorded rather than papered over: rounding up to guarantee a slot would "
            "let the reserve exceed its declared share, and a policy that quietly overspends "
            "its own budget is worse than one that is visibly inert at the smallest capacity. "
            "The inert cells are reported as an advisory on every run rather than hidden."
        )
    else:
        verdict = (
            "**No candidate share floors to a zero reserve at any measured capacity.** Every "
            "configuration in the grid is a live allocation mechanism at every cutoff."
        )
    return body + BREAK + note + BREAK + verdict


def risk_ranking_of_eligible(sources: dict[str, pl.DataFrame]) -> str:
    """Under pure risk ranking, do eligible establishments reach the queue at all?

    This is the problem statement, measured. Component 12 reported it through the
    ``__UNKNOWN__`` geography; here it is asked of the eligibility population directly, which
    is the population a reserve would actually serve.
    """
    frame = sources["features"]
    column = ELIGIBILITY_CANDIDATES[0][1]
    windows = [w for w in _windows(frame) if w[0].fold_set == folds_module.QUARTERLY]

    rows: list[list[str]] = []
    for model in CANDIDATE_MODELS:
        scored = _scored(sources, model)
        for k_name in K_LEVELS:
            selected_elig = 0
            selected_total = 0
            found_elig = 0
            total_elig_pos = 0
            available_elig = 0
            window_rows = 0
            for fold, test, median_daily in windows:
                k = _k_values(test, median_daily)[k_name]
                window = scored.filter(pl.col("fold_id") == fold.fold_id)
                if window.is_empty():
                    continue
                window = window.sort(["rd", "target_inspection_id"])
                ids = window["target_inspection_id"].to_list()
                scores = window["score"].to_list()
                labels = window["target"].to_list()
                eligible = [bool(v == 0) for v in window[column].fill_null(-1).to_list()]
                chosen = top_k_indices(scores, ids, k)
                selected_total += len(chosen)
                selected_elig += sum(1 for i in chosen if eligible[i])
                found_elig += sum(1 for i in chosen if eligible[i] and labels[i] == 1)
                total_elig_pos += sum(1 for i, e in enumerate(eligible) if e and labels[i] == 1)
                available_elig += sum(eligible)
                window_rows += len(ids)
            population_share = available_elig / window_rows if window_rows else 0.0
            selected_share = selected_elig / selected_total if selected_total else 0.0
            rows.append(
                [
                    f"`{model}`",
                    k_name,
                    f"{selected_total:,}",
                    f"{selected_elig:,}",
                    _fmt(selected_share),
                    _fmt(population_share),
                    _fmt(selected_share / population_share) if population_share else "n/a",
                    f"{found_elig:,}/{total_elig_pos:,}",
                    _fmt(found_elig / total_elig_pos) if total_elig_pos else "n/a",
                ]
            )

    body = _table(
        [
            "model",
            "k level",
            "selected",
            "eligible selected",
            "selected share",
            "population share",
            "selection ratio",
            "eligible positives found",
            "eligible capture",
        ],
        rows,
    )
    reading = (
        "**The selection ratio is the number to read.** Below 1.0, the pure-risk queue picks "
        "coverage-eligible establishments less often than their share of the population; at "
        "1.0 it picks them proportionally. The ratio is not required to be 1.0 -- if the "
        "eligible population genuinely has a lower outcome rate then a working risk model "
        "*should* select it less often, and profile 2 reports whether that is the case."
        + BREAK
        + "What a reserve responds to is the combination of a low ratio with the fact that the "
        "ranking inside this population is close to uninformative: Component 12 measured "
        "ROC-AUC 0.509-0.532 on the overlapping `__UNKNOWN__` group, which is random. A model "
        "that cannot order a population is not making a judgement about it, and deferring to "
        "its ordering there is deferring to nothing. That is an argument for making the "
        "allocation explicit, not an argument that the allocation is free."
    )
    return body + BREAK + reading


def model_selection_axes(sources: dict[str, pl.DataFrame]) -> str:
    """The axes of the pre-registered selection rule, from existing artifacts.

    Nothing is computed here that Component 5 or Component 9 did not already write. The rule
    is lexicographic: NDE first, with two models tied when their sensitivity intervals
    overlap; then calibrated ECE, lower wins; then precision at one day of real capacity; then
    the model name as a deterministic terminator.

    The tie rule was settled after this table was first read, and that is recorded rather than
    hidden. The plan carried a placeholder band -- Component 8's five-seed ROC-AUC spread of
    0.0058 -- and using a ROC-AUC spread as a threshold on an NDE difference is a unit error.
    Component 5 already publishes the right quantity: ``sensitivity`` perturbs the labels 1,000
    times per fold and reports each model's NDE p05-p95 interval. Comparing those intervals is
    also the method ``baseline_models_findings.md`` used to decide whether two NDE numbers
    differ, so the rule is this repository's existing precedent rather than a new invention.
    Both outcomes are printed below, including the one the discarded band produces.
    """
    simulation = sources["simulation"]
    metrics = sources["metrics"]
    sensitivity = sources["sensitivity"]

    nde = (
        simulation.filter(
            (pl.col("schedule_name") == "model") & (pl.col("fold_set") == folds_module.QUARTERLY)
        )
        .group_by("model_name")
        .agg(pl.col("normalized_discovery_efficiency").mean().alias("nde"))
    )
    band = (
        sensitivity.filter(pl.col("fold_set") == folds_module.QUARTERLY)
        .group_by("model_name")
        .agg(pl.col("p05").mean().alias("p05"), pl.col("p95").mean().alias("p95"))
    )
    quarterly = metrics.filter(pl.col("fold_set") == folds_module.QUARTERLY)
    ece = (
        quarterly.filter(pl.col("metric") == "ece")
        .group_by("model_name")
        .agg(pl.col("value").mean().alias("ece"))
    )
    p_at_day = (
        quarterly.filter((pl.col("metric") == "precision_at_k") & (pl.col("k_name") == "k_1_day"))
        .group_by("model_name")
        .agg(pl.col("value").mean().alias("p_k_1_day"))
    )
    joined = (
        nde.join(band, on="model_name", how="left")
        .join(ece, on="model_name", how="left")
        .join(p_at_day, on="model_name", how="left")
        .filter(pl.col("model_name").is_in([*CANDIDATE_MODELS, EXCLUDED_MODEL]))
        .sort("nde", descending=True)
    )

    eligible = [r for r in joined.iter_rows(named=True) if r["model_name"] in CANDIDATE_MODELS]
    leader = eligible[0]
    tied = [
        r
        for r in eligible
        if _as_float(r["p95"]) >= _as_float(leader["p05"])
        and _as_float(leader["p95"]) >= _as_float(r["p05"])
    ]
    best_ece = min(_as_float(r["ece"]) for r in tied)
    after_ece = [r for r in tied if _as_float(r["ece"]) == best_ece]
    best_p = max(_as_float(r["p_k_1_day"]) for r in after_ece)
    after_p = [r for r in after_ece if _as_float(r["p_k_1_day"]) == best_p]
    selected = sorted(after_p, key=lambda r: str(r["model_name"]))[0]

    discarded_tied = [
        r for r in eligible if _as_float(leader["nde"]) - _as_float(r["nde"]) <= DISCARDED_SEED_BAND
    ]
    discarded_best = min(_as_float(r["ece"]) for r in discarded_tied)
    discarded_pick = sorted(
        [r for r in discarded_tied if _as_float(r["ece"]) == discarded_best],
        key=lambda r: str(r["model_name"]),
    )[0]

    rows: list[list[str]] = []
    for row in joined.iter_rows(named=True):
        name = str(row["model_name"])
        excluded = name == EXCLUDED_MODEL
        overlaps = any(str(r["model_name"]) == name for r in tied)
        rows.append(
            [
                f"`{name}`" + (" (excluded)" if excluded else ""),
                _fmt(_as_float(row["nde"])),
                f"[{_fmt(_as_float(row['p05']))}, {_fmt(_as_float(row['p95']))}]",
                "-" if excluded else ("tied" if overlaps else "separated"),
                _fmt(_as_float(row["ece"])),
                _fmt(_as_float(row["p_k_1_day"])),
            ]
        )

    covid = (
        simulation.filter(
            (pl.col("schedule_name") == "model") & (pl.col("fold_set") == folds_module.COVID_SHIFT)
        )
        .select("model_name", "normalized_discovery_efficiency")
        .filter(pl.col("model_name").is_in(CANDIDATE_MODELS))
        .sort("normalized_discovery_efficiency", descending=True)
    )
    covid_rows = [
        [f"`{r['model_name']}`", _fmt(_as_float(r["normalized_discovery_efficiency"]))]
        for r in covid.iter_rows(named=True)
    ]

    body = _table(
        [
            "model",
            "axis 1: NDE (quarterly mean)",
            "NDE sensitivity band (p05-p95)",
            "vs leader",
            "axis 2: calibrated ECE",
            "axis 3: precision@k_1_day",
        ],
        rows,
    )
    reading = (
        f"**Axis 1 separates nothing: all {len(tied)} candidate bands overlap the leader's.** "
        "Component 5 perturbs the labels 1,000 times per fold, and under that perturbation "
        "every candidate's NDE interval contains every other candidate's point estimate. The "
        "headline operational metric of this entire project cannot tell these four models "
        "apart, which corroborates Component 8's own conclusion that the network's advantage "
        "is the size of its seed noise."
        + BREAK
        + f"**The rule therefore falls to axis 2, and selects `{selected['model_name']}`** on a "
        f"calibrated ECE of {_fmt(_as_float(selected['ece']))}. Axis 3 and the name terminator "
        "are never reached. This is an operating choice of the policy layer -- revisable, "
        "recorded in the manifest, and not a claim that this model is the best one."
        + BREAK
        + f"**Under the discarded {DISCARDED_SEED_BAND} band the rule would have selected "
        f"`{discarded_pick['model_name']}` instead.** The tie rule decides the outcome, which "
        "is exactly why the choice of rule is documented rather than asserted. Band overlap is "
        "preferred because it compares NDE against an NDE-derived interval rather than against "
        "a spread measured on a different metric, and because the repository already used "
        "interval overlap to decide this same question in Component 6."
        + BREAK
        + f"`{EXCLUDED_MODEL}` is listed for completeness and excluded before any number is "
        "read: ADR 0022 makes it experimental and ADR 0031 records that Component 11 could not "
        "explain it. A model whose recommendations cannot be explained to the inspector acting "
        "on them is not a deployment candidate, whatever it scores."
        + BREAK
        + "The `covid_shift` fold, reported separately and never averaged in:"
        + BREAK
        + _table(["model", "NDE (covid_shift)"], covid_rows)
        + BREAK
        + "**The ordering here is not the quarterly ordering, and that is a limitation on the "
        "selection rather than an input to it.** Component 7 measured that selecting on the "
        "rolling folds picks a different model than the shift fold would. A single held-out "
        "shift episode is one observation, so it cannot carry a selection rule -- but it is "
        "recorded as a named limitation on the choice, and it is why the choice is described "
        "as a revisable operating decision rather than as a finding."
    )
    return body + BREAK + reading


def opportunity_cost_bound(sources: dict[str, pl.DataFrame]) -> str:
    """What sits in the band a reserve displaces, and so what can a reserve cost?

    A reserve of size *r* at capacity *k* removes the rows ranked ``k-r+1 .. k`` by risk and
    replaces them with eligible rows. The cost is therefore bounded by the outcome rate of
    that marginal band, and offset by whatever the reserve itself finds. Reporting the bound
    before the policy runs is what stops "the reserve is nearly free" from being an inference
    drawn after the fact.
    """
    frame = sources["features"]
    column = ELIGIBILITY_CANDIDATES[0][1]
    scored = _scored(sources, PROBE_MODEL)
    windows = [w for w in _windows(frame) if w[0].fold_set == folds_module.QUARTERLY]

    rows: list[list[str]] = []
    for k_name in K_LEVELS:
        for share in RESERVE_GRID:
            if share == 0.0:
                continue
            band_rows = 0
            band_positives = 0
            reserve_rows = 0
            reserve_positives = 0
            for fold, test, median_daily in windows:
                k = _k_values(test, median_daily)[k_name]
                window = scored.filter(pl.col("fold_id") == fold.fold_id)
                if window.is_empty():
                    continue
                window = window.sort(["rd", "target_inspection_id"])
                ids = window["target_inspection_id"].to_list()
                scores = window["score"].to_list()
                labels = window["target"].to_list()
                eligible = [bool(v == 0) for v in window[column].fill_null(-1).to_list()]
                n_eligible = sum(eligible)
                reserve = min(int(share * k), n_eligible)
                if reserve == 0:
                    continue
                chosen = top_k_indices(scores, ids, k)
                displaced = chosen[k - reserve :]
                band_rows += len(displaced)
                band_positives += sum(labels[i] for i in displaced)
                risk_kept = set(chosen[: k - reserve])
                pool = [i for i, e in enumerate(eligible) if e and i not in risk_kept]
                pool_order = sorted(pool, key=lambda i: (-scores[i], ids[i]))[:reserve]
                reserve_rows += len(pool_order)
                reserve_positives += sum(labels[i] for i in pool_order)
            rows.append(
                [
                    k_name,
                    f"{share:.2f}",
                    f"{band_rows:,}",
                    _fmt(band_positives / band_rows) if band_rows else "n/a",
                    f"{reserve_rows:,}",
                    _fmt(reserve_positives / reserve_rows) if reserve_rows else "n/a",
                    f"{reserve_positives - band_positives:+,}",
                ]
            )

    body = _table(
        [
            "k level",
            "reserve share",
            "displaced rows",
            "displaced base rate",
            "reserve rows",
            "reserve base rate",
            "net positives",
        ],
        rows,
    )
    reading = (
        f"Measured on `{PROBE_MODEL}` across the quarterly windows, pooled."
        + BREAK
        + "**`net positives` is the whole trade-off, and its sign is not assumed.** It is the "
        "number of Priority citations the reserve finds minus the number the displaced "
        "marginal risk band would have found. A negative number is the cost of coverage, "
        "stated in the unit that matters, and it is reported whatever it says."
        + BREAK
        + "The marginal band is the honest comparator rather than the whole queue: a reserve "
        "does not displace the top-ranked establishment, it displaces the *last* one that "
        "would have fitted. Comparing the reserve against the average selected row would "
        "overstate the cost, and comparing it against nothing would hide it."
    )
    return body + BREAK + reading


def coverage_trend_and_floor_binding(sources: dict[str, pl.DataFrame]) -> str:
    """Does the queue's coverage of no-history establishments hold up over time?

    Profile 4 reports a pooled selection ratio of four to five, which reads as "the risk
    ranking already over-serves this population, so a coverage floor is unnecessary". This
    profile asks the same question per fold, because a pooled ratio over seventeen quarters is
    a statement about 2022 as much as about 2026, and a deployment decision is only ever about
    the most recent windows.

    A *floor* is the operationally meaningful form of a coverage reserve: guarantee that at
    least a stated share of capacity goes to establishments the model has no history for. It
    is inert whenever risk already clears the bar, and it binds only when risk does not -- so
    counting where it binds is the honest test of whether the mechanism does anything.
    """
    frame = sources["features"]
    column = ELIGIBILITY_CANDIDATES[0][1]
    scored = _scored(sources, PROBE_MODEL)

    rows: list[list[str]] = []
    binding: dict[float, int] = dict.fromkeys(RESERVE_GRID, 0)
    cells = 0
    early: list[float] = []
    late: list[float] = []
    quarterly_ids: list[str] = [
        f.fold_id for f, _, _ in _windows(frame) if f.fold_set == folds_module.QUARTERLY
    ]
    cutoff = quarterly_ids[-4] if len(quarterly_ids) >= 4 else quarterly_ids[0]

    for fold, test, median_daily in _windows(frame):
        k_values = _k_values(test, median_daily)
        window = scored.filter(pl.col("fold_id") == fold.fold_id)
        if window.is_empty():
            continue
        window = window.sort(["rd", "target_inspection_id"])
        ids = window["target_inspection_id"].to_list()
        scores = window["score"].to_list()
        eligible = [bool(v == 0) for v in window[column].fill_null(-1).to_list()]
        fractions: list[str] = []
        for k_name in K_LEVELS:
            k = k_values[k_name]
            chosen = top_k_indices(scores, ids, k)
            n_eligible = sum(1 for i in chosen if eligible[i])
            fraction = n_eligible / k
            fractions.append(_fmt(fraction, 3))
            if fold.fold_set == folds_module.QUARTERLY:
                cells += 1
                for share in RESERVE_GRID:
                    if share > 0.0 and n_eligible < int(share * k):
                        binding[share] += 1
                if fold.fold_id >= cutoff:
                    late.append(fraction)
                else:
                    early.append(fraction)
        rows.append([fold.fold_id, *fractions])

    body = _table(["fold", *K_LEVELS], rows)
    binding_rows = [
        [
            f"{share:.2f}",
            f"{binding[share]}",
            f"{cells}",
            _fmt(binding[share] / cells) if cells else "n/a",
        ]
        for share in RESERVE_GRID
        if share > 0.0
    ]
    early_mean = sum(early) / len(early) if early else 0.0
    late_mean = sum(late) / len(late) if late else 0.0
    reading = (
        f"Eligible share of the selected queue, `{PROBE_MODEL}`, per fold and cutoff. The "
        f"coverage-eligible population is {_fmt(0.1043)} of the rows, so any figure above "
        "that is over-selection relative to population share."
        + BREAK
        + _table(["floor share", "cells where it binds", "quarterly cells", "share"], binding_rows)
        + BREAK
        + f"**The pooled ratio hides a trend. Mean eligible share of the queue is "
        f"{_fmt(early_mean)} over the earlier quarterly folds and {_fmt(late_mean)} over the "
        f"last four ({cutoff} onward).** The pure-risk queue's coverage of establishments with "
        "no code-era history is not stable, and the most recent windows -- the only ones a "
        "deployment decision is actually about -- are the weakest."
        + BREAK
        + "**This is why the floor is worth implementing even though it is inert on average.** "
        "A mechanism that does nothing for thirteen quarters and then binds in the fourteenth "
        "is not a no-op; it is a guarantee. Reporting only the pooled ratio would have "
        "retired the mechanism on the strength of a number that describes 2022."
        + BREAK
        + "Note what this profile does **not** say. It does not say the recent decline is a "
        "problem, or that the model is wrong to have made it: the eligible population's base "
        "rate also falls over the same period, so a risk ranking that selects it less often "
        "may simply be tracking a real change. The floor makes the resulting allocation a "
        "stated choice rather than a side effect, and profile 6 prices it."
    )
    return body + BREAK + reading


def unknown_group_overlap(sources: dict[str, pl.DataFrame]) -> str:
    """How does coverage eligibility relate to Component 12's ``__UNKNOWN__`` group?

    ⚠ **Descriptive, and it informs a warning rather than a score.** This is the only profile
    that reads the group frame, it is reported after the selection rule rather than before it,
    and nothing it measures enters a ranking. Component 12's finding is that the group with no
    recoverable geography is largely the group with no recoverable history; the policy layer
    acts on the *history*, which is a per-row fact from Component 4, and never on the
    geography.
    """
    if "categoricals" not in sources:
        return "Component 8's as-of categoricals are absent; this profile was not run."

    frame = sources["features"]
    column = ELIGIBILITY_CANDIDATES[0][1]
    joined = frame.select("target_inspection_id", "target", column).join(
        sources["categoricals"].select("target_inspection_id", "community_area"),
        on="target_inspection_id",
        how="left",
    )
    joined = joined.with_columns(_eligible_expr(column).alias("eligible"))

    rows: list[list[str]] = []
    for label, subset in (
        (f"`{UNKNOWN}` community area", joined.filter(pl.col("community_area") == UNKNOWN)),
        ("named community area", joined.filter(pl.col("community_area") != UNKNOWN)),
        ("all rows", joined),
    ):
        n = subset.height
        n_elig = subset.filter(pl.col("eligible")).height
        rows.append(
            [
                label,
                f"{n:,}",
                f"{n_elig:,}",
                _fmt(n_elig / n) if n else "n/a",
                _fmt(_as_float(subset["target"].mean())) if n else "n/a",
            ]
        )

    eligible_only = joined.filter(pl.col("eligible"))
    in_unknown = eligible_only.filter(pl.col("community_area") == UNKNOWN).height
    overlap = _fmt(in_unknown / eligible_only.height) if eligible_only.height else "n/a"
    body = _table(["population", "rows", "coverage-eligible", "eligible share", "base rate"], rows)
    reading = (
        f"**Of the {eligible_only.height:,} coverage-eligible rows, {in_unknown:,} ({overlap}) "
        f"fall in the `{UNKNOWN}` community area.** The two populations overlap and are not "
        "the same thing, which is precisely why the policy is defined on the history column "
        "and not on the geography."
        + BREAK
        + "**This is the boundary Component 13 is built around.** A reserve keyed to "
        f"`{UNKNOWN}` would be a geographic allocation rule, and it would be one applied to a "
        "group defined by a data-quality artifact rather than by a place. A reserve keyed to "
        "missing inspection history is an allocation rule about what the model can and cannot "
        "know. The second is defensible to an inspector, an alderman and a court; the first is "
        "not, and no measured overlap between them changes that."
    )
    return body + BREAK + reading


PROFILES: dict[str, Callable[[dict[str, pl.DataFrame]], str]] = {
    "eligibility_definition": eligibility_definition,
    "eligibility_by_fold": eligibility_by_fold,
    "capacity_and_reserve": capacity_and_reserve,
    "risk_ranking_of_eligible": risk_ranking_of_eligible,
    "model_selection_axes": model_selection_axes,
    "opportunity_cost_bound": opportunity_cost_bound,
    "coverage_trend_and_floor_binding": coverage_trend_and_floor_binding,
    "unknown_group_overlap": unknown_group_overlap,
}


def _load(args: argparse.Namespace) -> tuple[dict[str, pl.DataFrame], list[str]]:
    settings = load_settings()
    features_path = args.features or latest_parquet(
        settings.features_processed_dir, prefix="as_of_features_"
    )
    predictions_path = args.calibrated_predictions or latest_parquet(
        settings.predictions_processed_dir, prefix="calibrated_predictions_"
    )
    simulation_path = args.simulation or latest_parquet(
        settings.evaluation_processed_dir, prefix="simulation_summary_"
    )
    metrics_path = args.metrics or latest_parquet(
        settings.evaluation_processed_dir, prefix="evaluation_metrics_"
    )
    sensitivity_path = args.sensitivity or latest_parquet(
        settings.evaluation_processed_dir, prefix="sensitivity_"
    )
    sources: dict[str, pl.DataFrame] = {
        "features": pl.read_parquet(features_path).with_columns(
            pl.col("inspection_date").str.to_date().alias("rd")
        ),
        "predictions": pl.read_parquet(predictions_path),
        "simulation": pl.read_parquet(simulation_path),
        "metrics": pl.read_parquet(metrics_path),
        "sensitivity": pl.read_parquet(sensitivity_path),
    }
    provenance = [
        f"features: {features_path.name}",
        f"predictions: {predictions_path.name}",
        f"simulation: {simulation_path.name}",
        f"metrics: {metrics_path.name}",
        f"sensitivity: {sensitivity_path.name}",
    ]
    try:
        categoricals_path = args.categoricals or latest_parquet(
            settings.neural_processed_dir, prefix="neural_categoricals_"
        )
    except FileNotFoundError:
        provenance.append("categoricals: absent")
    else:
        sources["categoricals"] = pl.read_parquet(categoricals_path)
        provenance.append(f"categoricals: {categoricals_path.name}")
    return sources, provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, help="Component 4 feature table.")
    parser.add_argument("--calibrated-predictions", type=Path, help="Component 9 artifact.")
    parser.add_argument("--simulation", type=Path, help="Component 5 simulation summary.")
    parser.add_argument("--metrics", type=Path, help="Component 5 metric table.")
    parser.add_argument("--sensitivity", type=Path, help="Component 5 NDE sensitivity bands.")
    parser.add_argument("--categoricals", type=Path, help="Component 8 as-of categoricals.")
    parser.add_argument("--only", action="append", help="Profile to run; repeatable.")
    args = parser.parse_args(argv)

    requested = args.only or list(PROFILES)
    unknown = [name for name in requested if name not in PROFILES]
    if unknown:
        raise SystemExit(f"unknown profile(s): {', '.join(unknown)}")

    sources, provenance = _load(args)

    print("<!-- generated by scripts/profile_policy.py -->")
    for line in provenance:
        print(f"<!-- {line} -->")
    for name in requested:
        print()
        print(f"### {name}")
        print()
        print(PROFILES[name](sources))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
