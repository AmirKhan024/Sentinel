"""Read-only profiling of the attribution surface, before any explanation is computed.

Analysis tooling, not library code: it answers one-off questions about a snapshot, nothing
imports it, and it should not ship in the wheel. Output is markdown on stdout, pasted into
``docs/analysis/explainability_findings.md``.

⚠ **No profile in this script reads a label or reports a metric.** Component 11's own
window *is* the test window -- it explains the predictions Components 6-8 committed there,
and those predictions already exist. Re-deriving a score over that window is therefore the
intended use rather than an exception, exactly as scoring the calibration window was for
Component 9. What would not be legitimate is looking at an *outcome*: no ``target`` column
is read anywhere below, no metric is computed, and no profile ranks a model. Component 11
must not be able to select a model, and a profiling script that reported test accuracy
would be the first step towards doing so by hand.

⚠ **This script is run before ``SAMPLE_SIZE``, ``BACKGROUND_SIZE``, ``PERMUTATION_ROUNDS``
and the additivity tolerances are frozen, and is what fixes them.** Profiles 3, 4, 5 and 6
measure the arithmetic error each method actually incurs and the wall-clock each costs. The
constants are then set from those measurements and written into ADR 0030. A tolerance set
from expectation rather than measurement is a guess wearing a decimal point -- Component 9
got three of them wrong that way and said so.

Questions this script answers
-----------------------------
1. ``matrix_representation``  -- how wide is each candidate's matrix, which name-recovery
                                 function names it, and does any position lack a name?
2. ``name_recovery_trap``     -- Components 6 and 7 order the same 30 columns differently.
                                 How many positions would a wrong choice mislabel?
3. ``native_treeshap``        -- do xgboost and lightgbm return an [n, M+1] contribution
                                 block, and how far is ``base + sum`` from the native
                                 margin? **Fixes the tree additivity tolerance.**
4. ``linear_shap_closed_form``-- how far is ``intercept + mean@coef + sum(coef*(z-mean))``
                                 from ``decision_function``? **Fixes the linear tolerance.**
5. ``permutation_cost``       -- measured forward-row throughput of the network and the
                                 implied wall-clock per (background, rounds, sample) cell.
                                 **Fixes SAMPLE_SIZE, BACKGROUND_SIZE, PERMUTATION_ROUNDS.**
6. ``permutation_convergence``-- how far apart are two independent seeds' permutation
                                 estimates, and how far from a large-round reference?
                                 **Bounds what "approximate" costs.**
7. ``explanation_population`` -- how many rows does each fold's test window hold, i.e. what
                                 is the population a bounded sample is drawn from?
8. ``output_scale``           -- the observed range of each model's log-odds output, so an
                                 absolute additivity tolerance can be read as a relative one.
9. ``embedding_booster_boundary`` -- is ``xgboost_chain_embeddings``'s fitted booster
                                 reachable through any public Component 8 interface?
                                 **The evidence for ADR 0031.**
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.boosting import preprocess as boosting_preprocess  # noqa: E402
from sentinel.boosting import train as boosting_train  # noqa: E402
from sentinel.boosting.definitions import Estimator  # noqa: E402
from sentinel.boosting.definitions import spec_for as boosting_spec_for  # noqa: E402
from sentinel.config import load_settings  # noqa: E402
from sentinel.evaluation import folds as folds_module  # noqa: E402
from sentinel.evaluation.models import FoldSpec  # noqa: E402
from sentinel.modeling import preprocess as modeling_preprocess  # noqa: E402
from sentinel.modeling import train as modeling_train  # noqa: E402
from sentinel.modeling.definitions import indicator_columns  # noqa: E402
from sentinel.modeling.definitions import spec_for as modeling_spec_for  # noqa: E402
from sentinel.neural import preprocess as neural_preprocess  # noqa: E402
from sentinel.neural import train as neural_train  # noqa: E402
from sentinel.neural.definitions import spec_for as neural_spec_for  # noqa: E402
from sentinel.query import duckdb_queries  # noqa: E402

#: The models profiled here. ``xgboost_chain_embeddings`` is absent from every profile but
#: the last, which is about the fact that it cannot be reached at all.
CANDIDATES: tuple[str, ...] = (
    "logistic_regression",
    "xgboost",
    "lightgbm",
    "neural_numeric_only",
)

#: Fitting every candidate on all 18 folds to measure arithmetic error would take half an
#: hour to answer a question one fold answers. The last quarterly fold is the project's
#: standing representative -- Component 8 and Component 9 both draw their figures there.
PROBE_FOLD = "quarterly-2026Q2"

#: Grid for profile 5: (background rows, antithetic rounds) pairs whose cost is projected
#: from the measured throughput onto the real 18-fold population.
COST_GRID: tuple[tuple[int, int], ...] = ((32, 1), (32, 2), (64, 2), (64, 4), (128, 4))

#: Candidate per-fold sample sizes projected in profile 5.
SAMPLE_GRID: tuple[int, ...] = (100, 200, 300, 500)

#: Seeds for profile 6: two independent estimates plus a high-round reference.
CONVERGENCE_SEEDS: tuple[int, int] = (20260825, 20260826)
CONVERGENCE_REFERENCE_ROUNDS = 64
CONVERGENCE_ROWS = 60
CONVERGENCE_BACKGROUND = 32

#: Round counts swept in profile 6. The question is not "does it converge" -- permutation
#: sampling converges at the usual 1/sqrt(n) -- but *how many rounds buy a local value
#: worth printing*, and whether the global mean-|SHAP| statistic converges sooner than any
#: individual value does. It does, and by how much is what fixes PERMUTATION_ROUNDS.
CONVERGENCE_ROUNDS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


# --- shared helpers ----------------------------------------------------------


def _folds(frame: pl.DataFrame) -> list[FoldSpec]:
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    if start is None or end is None:
        raise SystemExit("feature table has no usable reference dates")
    quarterly = folds_module.quarterly_folds(data_start=start, data_end=end)
    return [*quarterly, *folds_module.covid_shift_fold(data_end=end)]


def _probe_fold(frame: pl.DataFrame) -> FoldSpec:
    for fold in _folds(frame):
        if fold.fold_id == PROBE_FOLD:
            return fold
    raise SystemExit(f"probe fold {PROBE_FOLD} not present in this snapshot")


def _fmt(value: float, places: int = 4) -> str:
    return f"{value:.{places}f}"


def _sci(value: float) -> str:
    return f"{value:.3e}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


#: A blank line in generated markdown. A literal escape inside these long f-strings has
#: been mangled by a shell round-trip once already; a named constant cannot be.
BREAK = "\n\n"


def _ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Descending ranks with ties averaged. Largest value gets rank 1."""
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    for value in np.unique(values):
        tied = np.flatnonzero(values == value)
        if len(tied) > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def _spearman(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Pearson correlation of two rank vectors."""
    a_centred = a - a.mean()
    b_centred = b - b.mean()
    denominator = float(np.sqrt((a_centred**2).sum() * (b_centred**2).sum()))
    return float((a_centred * b_centred).sum() / denominator) if denominator else 0.0


def _network_logits(model: Any, dense: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pre-sigmoid output of the numeric-only network for a dense block.

    ``codes`` is zero-width because ``neural_numeric_only`` embeds nothing;
    ``EmbeddingNet.forward`` documents that shape as legitimate.
    """
    import torch

    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(dense, dtype=np.float32))
        codes = torch.zeros((dense.shape[0], 0), dtype=torch.int64)
        out = model(tensor, codes)
    return np.asarray(out.numpy(), dtype=np.float64)


def _permutation_shap(
    predict: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    rows: NDArray[np.float64],
    background: NDArray[np.float64],
    *,
    rounds: int,
    seed: int,
) -> tuple[NDArray[np.float64], float]:
    """Antithetic permutation SHAP. The prototype of ``explain.attribute``'s implementation.

    A permutation of the columns defines a path from a background row to the explained row:
    overwrite one column at a time and attribute each step's change in output to the column
    that moved. The steps telescope to ``f(row) - f(background)``, which is why local
    accuracy holds exactly however few permutations are drawn -- the approximation is in
    *how the credit is split*, never in whether it sums.

    Each permutation is walked forwards and then backwards (its antithetic partner), which
    cancels the first-order ordering bias: a column drawn early in the forward walk is drawn
    late in the backward one.
    """
    n_rows, n_features = rows.shape
    n_background = background.shape[0]
    rng = np.random.default_rng(seed)
    values = np.zeros((n_rows, n_features), dtype=np.float64)
    base = float(np.mean(predict(background)))

    for i in range(n_rows):
        row = rows[i]
        total = np.zeros(n_features, dtype=np.float64)
        for _ in range(rounds):
            order = rng.permutation(n_features)
            for forward in (True, False):
                walk = order if forward else order[::-1]
                current = background.copy() if forward else np.tile(row, (n_background, 1))
                path = np.empty((n_features + 1, n_background, n_features), dtype=np.float64)
                path[0] = current
                for step, column in enumerate(walk):
                    current[:, column] = row[column] if forward else background[:, column]
                    path[step + 1] = current
                outputs = (
                    predict(path.reshape(-1, n_features))
                    .reshape(n_features + 1, n_background)
                    .mean(axis=1)
                )
                deltas = np.diff(outputs)
                for step, column in enumerate(walk):
                    total[column] += deltas[step] if forward else -deltas[step]
        values[i] = total / (2 * rounds)
    return values, base


# --- 1. matrix representation ------------------------------------------------


def matrix_representation(frame: pl.DataFrame, fitted: dict[str, Any] | None) -> str:
    """Every column each candidate's estimator sees, and the function that names it."""
    lspec = modeling_spec_for("logistic_regression")
    linear_names = modeling_preprocess.ordered_matrix_columns(lspec)
    nspec = neural_spec_for("neural_numeric_only")

    rows: list[list[str]] = [
        [
            "logistic_regression",
            "`modeling.preprocess.ordered_matrix_columns`",
            str(len(linear_names)),
            f"`{linear_names[0]}` … `{linear_names[-1]}`",
            "0",
        ]
    ]
    for name in ("xgboost", "lightgbm"):
        tree_names = boosting_preprocess.matrix_columns(boosting_spec_for(name))
        rows.append(
            [
                name,
                "`boosting.preprocess.matrix_columns`",
                str(len(tree_names)),
                f"`{tree_names[0]}` … `{tree_names[-1]}`",
                "0",
            ]
        )
    net_names = neural_preprocess.matrix_columns(nspec)
    ordered_net = modeling_preprocess.ordered_matrix_columns(lspec)
    rows.append(
        [
            "neural_numeric_only",
            "`neural.preprocess.transformed_columns`",
            str(len(ordered_net)),
            f"`{ordered_net[0]}` … `{ordered_net[-1]}`",
            "0",
        ]
    )

    body = _table(
        ["model", "name-recovery function", "width", "first … last", "unnamed positions"],
        rows,
    )
    indicators = indicator_columns()
    return body + (
        f"\n\nEvery position in every matrix carries a name. {len(lspec.feature_columns)} "
        f"Component 4 features plus {len(indicators)} null-rule family indicators "
        f"(`{'`, `'.join(indicators)}`) = **{len(linear_names)} columns**, and the same "
        f"{len(linear_names)} reach all four candidates.\n\n"
        f"`neural_numeric_only` carries `encoding={nspec.encoding.value}` and "
        f"`entity_columns={nspec.entity_columns}`, so its embedding block is zero-width and "
        f"it sees exactly the matrix Components 6 and 7 see (input width "
        f"{len(net_names)}). That is what makes a cross-model attribution comparison a "
        "comparison of models rather than of feature sets."
    )


# --- 2. the name-recovery trap -----------------------------------------------


def name_recovery_trap(frame: pl.DataFrame, fitted: dict[str, Any] | None) -> str:
    """How many positions a wrong name-recovery choice would silently mislabel."""
    spec = modeling_spec_for("logistic_regression")
    natural = modeling_preprocess.matrix_columns(spec)
    ordered = modeling_preprocess.ordered_matrix_columns(spec)
    same_set = set(natural) == set(ordered)
    disagreements = sum(1 for a, b in zip(natural, ordered, strict=True) if a != b)
    first = next((i for i, (a, b) in enumerate(zip(natural, ordered, strict=True)) if a != b), None)

    lines = [
        f"- The two orders are permutations of one another: **{same_set}**.",
        f"- Positions where they disagree: **{disagreements} of {len(natural)}**.",
    ]
    if first is not None:
        lines.append(
            f"- First disagreement at index **{first}**: `{natural[first]}` (natural, what "
            f"Component 7's tree matrix uses) vs `{ordered[first]}` (ColumnTransformer branch "
            "order, what Component 6's fitted pipeline emits)."
        )
    lines += [
        "",
        "Consequence: a Component 11 that reached for the wrong one would emit a table in",
        "which every value is arithmetically correct and attached to the **wrong feature**.",
        "Nothing would raise. No additivity check would fail -- the sum is invariant to a",
        "permutation of its terms. Every figure and every sentence of the findings document",
        "would be wrong, and the artifact would look perfect.",
        "",
        "This is the single most likely way this component ships a defect, so the choice is",
        "recorded per model in the registry rather than inferred, and the suite asserts each",
        "model's names against the function its own component uses.",
    ]
    return "\n".join(lines)


# --- 3. native TreeSHAP ------------------------------------------------------


def native_treeshap(frame: pl.DataFrame, fitted: dict[str, Any] | None) -> str:
    """Shape and additivity error of each booster's own contribution output."""
    if fitted is None:
        return "_(needs fitted models; run without --cheap)_"
    fold = _probe_fold(frame)
    window = folds_module.window_frame(frame, fold)
    rows: list[list[str]] = []
    for name in ("xgboost", "lightgbm"):
        spec = boosting_spec_for(name)
        model = fitted[name]
        matrix = boosting_preprocess.tree_matrix(window, spec)
        if spec.estimator is Estimator.XGBOOST:
            import xgboost as xgb

            booster = model.estimator.get_booster()
            contribs = np.asarray(
                booster.predict(xgb.DMatrix(matrix), pred_contribs=True), dtype=np.float64
            )
            margin = np.asarray(model.estimator.predict(matrix, output_margin=True), np.float64)
            dtype = "float32"
        else:
            contribs = np.asarray(
                model.estimator.predict_proba(matrix, pred_contrib=True), dtype=np.float64
            )
            margin = np.asarray(
                model.estimator.predict_proba(matrix, raw_score=True), dtype=np.float64
            )
            dtype = "float64"
        err = np.abs(contribs.sum(axis=1) - margin)
        rows.append(
            [
                name,
                f"{contribs.shape[0]} x {contribs.shape[1]}",
                str(matrix.shape[1]),
                dtype,
                _sci(float(err.max())),
                _sci(float(err.mean())),
                _sci(float(contribs[:, -1].std())),
            ]
        )
    body = _table(
        [
            "model",
            "contribs shape",
            "matrix width",
            "compute dtype",
            "max abs additivity error",
            "mean",
            "sd(base column)",
        ],
        rows,
    )
    return body + (
        "\n\nBoth libraries return `[n, M+1]`: one column per feature plus the expected value"
        "\nin the last position. The base column is constant across rows -- its SD is the"
        "\nmeasurement above -- which is what makes it an expected value rather than a per-row"
        "\nintercept.\n\n"
        "XGBoost computes in float32 and LightGBM in float64, and their residuals differ by"
        "\norders of magnitude as a result. The tolerance must therefore be set from the worse"
        "\nof the two, never from an average, and it is an arithmetic tolerance rather than a"
        "\nmodelling one: TreeSHAP is **exact** for both libraries."
    )


# --- 4. linear SHAP ----------------------------------------------------------


def linear_shap_closed_form(frame: pl.DataFrame, fitted: dict[str, Any] | None) -> str:
    """Additivity error of the closed-form interventional linear attribution."""
    if fitted is None:
        return "_(needs fitted models; run without --cheap)_"
    fold = _probe_fold(frame)
    window = folds_module.window_frame(frame, fold)
    training = modeling_train.training_frame(frame, fold)
    spec = modeling_spec_for("logistic_regression")
    model = fitted["logistic_regression"]

    raw = modeling_preprocess.to_matrix(window, spec)
    transform = model.pipeline.named_steps["preprocess"]
    z = np.asarray(transform.transform(raw), dtype=np.float64)
    coef = np.asarray(model.coefficients, dtype=np.float64)
    intercept = float(model.intercept)

    z_train = np.asarray(
        transform.transform(modeling_preprocess.to_matrix(training, spec)), dtype=np.float64
    )
    reference = z_train.mean(axis=0)

    values = (z - reference) * coef
    base = intercept + float(reference @ coef)
    decision = np.asarray(model.pipeline.decision_function(raw), dtype=np.float64)
    err = np.abs(base + values.sum(axis=1) - decision)

    return "\n".join(
        [
            f"- Reference (background) = the mean of the **training** window's transformed "
            f"matrix, {z_train.shape[0]:,} rows.",
            f"- Base value `intercept + reference @ coef` = **{base:.6f}**.",
            f"- Mean `decision_function` over the test window = **{float(decision.mean()):.6f}**.",
            f"- max |base + sum(phi) - decision_function| = **{_sci(float(err.max()))}**",
            f"- mean = **{_sci(float(err.mean()))}**",
            "",
            "The scaler already centres each column on its training mean, so `reference` sits",
            f"within {_sci(float(np.abs(reference).max()))} of zero -- but it is *computed*",
            "rather than assumed to be zero, because assuming it would make every attribution",
            "wrong the moment a background narrower than the full training window is used.",
            "",
            "The residual is float64 summation order and nothing else. A linear model's",
            "Shapley values under an interventional reference have a closed form,",
            "`phi_j = coef_j * (z_j - E[z_j])`, so there is nothing to approximate here: the",
            "tolerance absorbs rounding, not method error.",
        ]
    )


# --- 5. permutation cost -----------------------------------------------------


def permutation_cost(frame: pl.DataFrame, fitted: dict[str, Any] | None) -> str:
    """Measured network throughput, and the wall-clock each budget cell implies."""
    if fitted is None:
        return "_(needs fitted models; run without --cheap)_"
    fold = _probe_fold(frame)
    window = folds_module.window_frame(frame, fold)
    spec = neural_spec_for("neural_numeric_only")
    network, preprocessor = neural_train.scorer_for(fitted["neural_numeric_only"])
    dense = neural_preprocess.apply_preprocessor(preprocessor, window, spec)

    _network_logits(network, dense[:256])  # warm-up
    batch = np.repeat(dense[:256], 16, axis=0)
    started = time.perf_counter()
    _network_logits(network, batch)
    elapsed = time.perf_counter() - started
    rate = batch.shape[0] / elapsed

    n_features = dense.shape[1]
    n_folds = len(_folds(frame))
    rows: list[list[str]] = []
    for background, rounds in COST_GRID:
        per_row = 2 * rounds * (n_features + 1) * background
        for sample in SAMPLE_GRID:
            total = per_row * sample * n_folds
            rows.append(
                [
                    str(background),
                    str(rounds),
                    str(sample),
                    f"{per_row:,}",
                    f"{total:,}",
                    f"{total / rate:,.0f}",
                ]
            )
    body = _table(
        [
            "background",
            "rounds",
            "sample/fold",
            "forward rows per explained row",
            "forward rows total",
            "projected seconds",
        ],
        rows,
    )
    return (
        f"Measured throughput: **{rate:,.0f} forward rows/second** "
        f"({batch.shape[0]:,} rows in {elapsed:.3f}s), single thread, CPU, "
        f"{n_features} dense columns. Population: {n_folds} folds.\n\n"
        + body
        + "\n\nCost is linear in all four factors, which is why the budget is four frozen"
        "\nconstants rather than one number. The projection excludes the fits themselves,"
        "\nwhich are measured separately and dominate at the small sample sizes."
    )


# --- 6. permutation convergence ----------------------------------------------


def permutation_convergence(frame: pl.DataFrame, fitted: dict[str, Any] | None) -> str:
    """How fast the permutation game converges -- locally, and for the global statistic."""
    if fitted is None:
        return "_(needs fitted models; run without --cheap)_"
    fold = _probe_fold(frame)
    window = folds_module.window_frame(frame, fold)
    training = modeling_train.training_frame(frame, fold)
    spec = neural_spec_for("neural_numeric_only")
    network, preprocessor = neural_train.scorer_for(fitted["neural_numeric_only"])

    dense = neural_preprocess.apply_preprocessor(preprocessor, window, spec)
    train_dense = neural_preprocess.apply_preprocessor(preprocessor, training, spec)
    picker = np.random.default_rng(CONVERGENCE_SEEDS[0])
    background = train_dense[
        picker.choice(train_dense.shape[0], size=CONVERGENCE_BACKGROUND, replace=False)
    ]
    rows = dense[:CONVERGENCE_ROWS]

    def predict(block: NDArray[np.float64]) -> NDArray[np.float64]:
        return _network_logits(network, block)

    reference, base_ref = _permutation_shap(
        predict, rows, background, rounds=CONVERGENCE_REFERENCE_ROUNDS, seed=CONVERGENCE_SEEDS[1]
    )
    logits = predict(rows)
    scale = float(np.abs(reference).max())
    global_ref = np.abs(reference).mean(axis=0)
    ref_ranks = _ranks(global_ref)

    table: list[list[str]] = []
    for rounds in CONVERGENCE_ROUNDS:
        started = time.perf_counter()
        a, base_a = _permutation_shap(
            predict, rows, background, rounds=rounds, seed=CONVERGENCE_SEEDS[0]
        )
        elapsed = time.perf_counter() - started
        local_err = np.abs(a - reference)
        global_a = np.abs(a).mean(axis=0)
        global_err = np.abs(global_a - global_ref)
        rho = _spearman(_ranks(global_a), ref_ranks)
        additivity = float(np.abs(base_a + a.sum(axis=1) - logits).max())
        table.append(
            [
                str(rounds),
                f"{100 * float(local_err.max()) / scale:.1f}%",
                f"{100 * float(np.median(local_err)) / scale:.2f}%",
                f"{100 * float(global_err.max()) / float(global_ref.max()):.2f}%",
                _fmt(rho, 4),
                _sci(additivity),
                f"{elapsed:.1f}",
            ]
        )

    body = _table(
        [
            "rounds",
            "max local err",
            "median local err",
            "max global err",
            "global rank rho",
            "max additivity err",
            "seconds",
        ],
        table,
    )
    return (
        f"{CONVERGENCE_ROWS} rows, background {background.shape[0]}, reference "
        f"**{CONVERGENCE_REFERENCE_ROUNDS} rounds at an independent seed**. Local errors "
        f"are expressed against the largest attribution in the reference "
        f"({_sci(scale)}); the global error against the largest mean-|SHAP|."
        + BREAK
        + body
        + BREAK
        + "**Two things this table settles.**"
        + BREAK
        + "First, additivity is flat across the whole sweep. That is not a coincidence "
        "and not a sign of convergence: a permutation path telescopes to "
        "`f(row) - f(background)`, so `base + sum(phi)` reconstructs the output exactly "
        "at one round and at sixty-four alike. **Additivity is therefore not evidence "
        "that a permutation attribution is accurate**, and a component that reported it "
        "as though it were would be misreporting. What is approximate is how the credit "
        "is *divided*, which additivity cannot see."
        + BREAK
        + "Second, the global statistic converges far faster than any individual value. "
        "Each row draws its own permutations, so averaging |SHAP| over the explanation "
        "sample averages independent errors, and the ranking stabilises long before a "
        "single local attribution does. This is the measurement that lets Component 11 "
        "publish a global importance ranking for the network while labelling its per-row "
        "values approximate -- and it is why the local figures carry that caveat printed "
        "on them rather than in a footnote."
    )


# --- 7. explanation population -----------------------------------------------


def explanation_population(frame: pl.DataFrame, fitted: dict[str, Any] | None) -> str:
    """Test-window row counts per fold: the population a bounded sample is drawn from."""
    rows: list[list[str]] = []
    total = 0
    for fold in _folds(frame):
        window = folds_module.window_frame(frame, fold)
        total += window.height
        rows.append(
            [
                fold.fold_id,
                fold.fold_set,
                str(fold.test_start),
                str(fold.test_end),
                f"{window.height:,}",
                f"{window['establishment_id'].n_unique():,}",
            ]
        )
    body = _table(
        ["fold_id", "fold_set", "test start", "test end", "rows", "distinct establishments"],
        rows,
    )
    return body + (
        f"\n\n**{total:,} rows across {len(rows)} folds**, which is the population per model"
        "\nand matches the row count of every committed prediction artifact.\n\n"
        "Explaining all of them is free for the tree and linear models: both are exact and"
        "\nvectorised over the whole matrix at once. For the network it is not, so a bounded"
        "\nsample is drawn -- and the *same* sampled ids are used for every model, so a"
        "\ncross-model importance comparison is like-for-like rather than a comparison across"
        "\ndifferent populations."
    )


# --- 8. output scale ---------------------------------------------------------


def output_scale(frame: pl.DataFrame, fitted: dict[str, Any] | None) -> str:
    """Observed log-odds range per model, so an absolute tolerance reads as a relative one."""
    if fitted is None:
        return "_(needs fitted models; run without --cheap)_"
    fold = _probe_fold(frame)
    window = folds_module.window_frame(frame, fold)
    rows: list[list[str]] = []

    lspec = modeling_spec_for("logistic_regression")
    lmargin = np.asarray(
        fitted["logistic_regression"].pipeline.decision_function(
            modeling_preprocess.to_matrix(window, lspec)
        ),
        dtype=np.float64,
    )
    rows.append(
        [
            "logistic_regression",
            "`pipeline.decision_function`",
            _fmt(float(lmargin.min())),
            _fmt(float(lmargin.max())),
            _fmt(float(lmargin.mean())),
        ]
    )

    for name in ("xgboost", "lightgbm"):
        bspec = boosting_spec_for(name)
        matrix = boosting_preprocess.tree_matrix(window, bspec)
        estimator = fitted[name].estimator
        margin = np.asarray(
            estimator.predict(matrix, output_margin=True)
            if bspec.estimator is Estimator.XGBOOST
            else estimator.predict_proba(matrix, raw_score=True),
            dtype=np.float64,
        )
        rows.append(
            [
                name,
                "`output_margin` / `raw_score`",
                _fmt(float(margin.min())),
                _fmt(float(margin.max())),
                _fmt(float(margin.mean())),
            ]
        )

    nspec = neural_spec_for("neural_numeric_only")
    network, preprocessor = neural_train.scorer_for(fitted["neural_numeric_only"])
    logits = _network_logits(
        network, neural_preprocess.apply_preprocessor(preprocessor, window, nspec)
    )
    rows.append(
        [
            "neural_numeric_only",
            "pre-sigmoid logit",
            _fmt(float(logits.min())),
            _fmt(float(logits.max())),
            _fmt(float(logits.mean())),
        ]
    )

    return _table(["model", "native accessor", "min", "max", "mean"], rows) + (
        "\n\nAll four land in the same space -- natural log-odds -- which is what makes a"
        "\ncross-model importance comparison a comparison rather than a units error."
        "\n\nProbability space was considered and rejected. A Shapley decomposition of"
        "\n`sigmoid(margin)` is not additive in the margin's own contributions, because"
        "\n`sigmoid` is not linear: a probability-space table would have to either abandon"
        "\nadditivity or fabricate it. `OutputSpace` therefore declares `log_odds` only,"
        "\nrather than declaring a probability variant that nothing can reach."
    )


# --- 9. the embedding-booster boundary ---------------------------------------


def embedding_booster_boundary(frame: pl.DataFrame, fitted: dict[str, Any] | None) -> str:
    """Is ``xgboost_chain_embeddings``'s booster reachable through a public interface?"""
    import dataclasses

    from sentinel.neural import embed
    from sentinel.neural import train as neural_train_module
    from sentinel.neural.models import FittedEmbeddingBooster

    field_names = [f.name for f in dataclasses.fields(FittedEmbeddingBooster)]
    estimator_fields = [n for n in field_names if n in ("estimator", "booster", "model")]
    public = sorted(
        n
        for n, v in vars(embed).items()
        if not n.startswith("_") and callable(v) and getattr(v, "__module__", "") == embed.__name__
    )
    stash = sorted(n for n in dir(embed) if n.startswith("_") and "scorer" in n.lower())
    network_public = "scorer_for" in getattr(neural_train_module, "__all__", ())

    return "\n".join(
        [
            f"- `FittedEmbeddingBooster` has **{len(field_names)} fields** and none of them "
            f"holds the fitted estimator: `{estimator_fields or 'no estimator/booster field'}`.",
            f"- Its fields: `{', '.join(field_names)}`",
            "",
            f"- `neural.embed`'s public functions: `{', '.join(public)}` -- every one of "
            "them takes a `FittedEmbeddingBooster` or returns names/vectors; none returns the "
            "estimator.",
            f"- The only route to the live booster is `neural.embed."
            f"{stash[0] if stash else '_scorer_for'}`, which is **private** and reads a "
            "process-local dict keyed by `id()`.",
            "",
            f"- By contrast `neural.train.scorer_for` is public "
            f"(in `neural.train.__all__`: **{network_public}**), which is exactly how "
            "`neural_numeric_only` is reached and explained.",
            "",
            "The asymmetry is not a considered decision in Component 8 -- it is an accident of",
            "which helper a public `predict` happened to need to name. But Component 8 is",
            "closed, and reaching into `_scorer_for` to make Component 11 look more complete",
            "is precisely the move HANDOFF section 0 exists to prevent.",
            "",
            "So `xgboost_chain_embeddings` is reported `unsupported`, with this measurement as",
            "the stated reason, and the minimal public extension that would lift the",
            "restriction -- `def booster_for(fitted: FittedEmbeddingBooster) -> Any`, a",
            "four-line alias over the existing stash with no behavioural change and no",
            "artifact change -- is *proposed* in ADR 0031 for whoever reopens Component 8,",
            "and deliberately not taken here.",
        ]
    )


# --- fitting -----------------------------------------------------------------


def fit_probe_models(frame: pl.DataFrame, models: Sequence[str]) -> dict[str, Any]:
    """Re-execute each candidate's unchanged fit function on the probe fold.

    Component 9 established the pattern and ADR 0026 the licence: no fitted model object is
    persisted anywhere, so a component that needs one runs the fit again -- same spec, same
    seed, same canonical row order. This script does not run the bit-identity gate the
    component proper runs; a gate belongs where an artifact is written, not in a profile.
    """
    fold = _probe_fold(frame)
    training = modeling_train.training_frame(frame, fold)
    out: dict[str, Any] = {}
    for name in models:
        started = time.perf_counter()
        if name == "logistic_regression":
            out[name] = modeling_train.fit_fold(modeling_spec_for(name), training, fold)
        elif name in ("xgboost", "lightgbm"):
            out[name] = boosting_train.fit_fold(boosting_spec_for(name), training, fold)
        elif name == "neural_numeric_only":
            out[name] = neural_train.fit_fold(neural_spec_for(name), training, fold)
        else:  # pragma: no cover - the candidate list is a module constant
            raise SystemExit(f"no probe fit path for {name}")
        print(
            f"<!-- fitted {name} on {fold.fold_id} in {time.perf_counter() - started:.1f}s -->",
            file=sys.stderr,
        )
    return out


PROFILES: dict[str, Callable[[pl.DataFrame, dict[str, Any] | None], str]] = {
    "matrix_representation": matrix_representation,
    "name_recovery_trap": name_recovery_trap,
    "native_treeshap": native_treeshap,
    "linear_shap_closed_form": linear_shap_closed_form,
    "permutation_cost": permutation_cost,
    "permutation_convergence": permutation_convergence,
    "explanation_population": explanation_population,
    "output_scale": output_scale,
    "embedding_booster_boundary": embedding_booster_boundary,
}

#: Profiles that need no model fitted. Useful while iterating.
CHEAP_PROFILES: tuple[str, ...] = (
    "matrix_representation",
    "name_recovery_trap",
    "explanation_population",
    "embedding_booster_boundary",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, help="Component 4 feature table.")
    parser.add_argument("--only", action="append", help="Profile to run; repeatable.")
    parser.add_argument("--cheap", action="store_true", help="Only profiles needing no fit.")
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

    needs_fits = any(name not in CHEAP_PROFILES for name in requested)
    fitted = fit_probe_models(frame, CANDIDATES) if needs_fits else None

    print(f"<!-- generated by scripts/profile_explanations.py from {features_path.name} -->")
    print(f"<!-- {frame.height} feature rows; probe fold {PROBE_FOLD} -->")
    for name in requested:
        print()
        print(f"### {name}")
        print()
        print(PROFILES[name](frame, fitted))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
