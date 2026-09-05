"""Figures for Component 11. Every one answers a stated question; none is decorative.

Six figures, drawn only from the persisted tables -- never from an in-memory object -- so a
figure can be regenerated from the artifact alone and cannot silently disagree with it.

A figure that cannot be drawn honestly returns ``None`` and logs, rather than raising or
drawing something misleading. A stability plot over one fold, for instance, would be a flat
line at 1.0 that reads as evidence of stability when it is evidence of nothing.

Every figure carries a caption stating its caveat, because a figure travels further than the
document that contains it. The two caveats that matter most: the neural model's per-row
values are approximate, and none of this is causal.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from sentinel.explain.definitions import TOP_K

logger = logging.getLogger(__name__)

#: Below this a beeswarm is a scatter of a dozen dots pretending to be a distribution.
MIN_BEESWARM_ROWS = 100

#: Below this a stability panel has too few folds to show a trend.
MIN_STABILITY_FOLDS = 3

#: How many features the importance and drift panels show. The full 30 is unreadable at
#: figure size, and the tail is where the attributions are indistinguishable from zero.
DISPLAY_FEATURES = 15

CAUSALITY_CAPTION = (
    "SHAP shows how the model used a feature, not that changing the feature would change "
    "the outcome. Predictive importance is not causality."
)
APPROXIMATE_CAPTION = (
    "neural_numeric_only values are permutation estimates: the global ranking is stable "
    "(rank rho 0.996 against a 64-round reference) but individual values are approximate."
)


def _matplotlib() -> tuple[Any, Any]:
    """Import matplotlib with a headless backend, chosen before pyplot loads.

    Without it matplotlib picks a backend from the environment, which on a headless runner
    is a different one than on this machine, and a figure that renders differently per
    environment is not an artifact.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


def _save(figure: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    _, plt = _matplotlib()
    plt.close(figure)
    logger.info("Wrote %s", path)
    return path


def _caption(figure: Any, text: str) -> None:
    figure.text(0.5, -0.04, text, ha="center", va="top", fontsize=7, wrap=True, color="#444444")


# --- 1. global importance ----------------------------------------------------


def global_importance(
    importance: pl.DataFrame, model: str, fold_set: str, destination: Path
) -> Path | None:
    """Which features each model leaned on, with the fold-to-fold spread beside each mean.

    The error bar is the point of the figure. A bar chart of mean importance alone invites
    "this is the most important feature"; the spread says whether that claim survives the
    seventeen folds it was averaged over.
    """
    _, plt = _matplotlib()
    rows = importance.filter(
        (pl.col("model_name") == model)
        & (pl.col("fold_set") == fold_set)
        & (pl.col("scope") == "fold_set")
    ).sort("mean_abs_shap", descending=True)
    if rows.height == 0:
        logger.info("No aggregate importance for %s/%s; skipping figure", model, fold_set)
        return None

    top = rows.head(DISPLAY_FEATURES).reverse()
    names = top["feature_name"].to_list()
    means = top["mean_abs_shap"].to_list()
    spreads = [v or 0.0 for v in top["sd_abs_shap"].to_list()]
    # Narrowed rather than cast: a polars aggregate is a wide union under strict mode.
    folds_value = rows["folds"].max()
    folds = folds_value if isinstance(folds_value, int) else 0

    figure, axes = plt.subplots(figsize=(8, 6))
    axes.barh(range(len(names)), means, xerr=spreads, color="#3b6ea5", ecolor="#a33", capsize=2)
    axes.set_yticks(range(len(names)))
    axes.set_yticklabels(names, fontsize=8)
    axes.set_xlabel("mean |SHAP| (log-odds)")
    axes.set_title(
        f"{model} - top {len(names)} features by mean |SHAP|\n"
        f"{fold_set} fold set, {folds} fold(s); bars are the mean, whiskers the "
        "fold-to-fold SD",
        fontsize=10,
    )
    _caption(
        figure,
        f"{CAUSALITY_CAPTION} Attributions are in log-odds of the base model's output, "
        "before Component 9's calibration.",
    )
    return _save(figure, destination / f"explain_global_importance_{model}_{fold_set}.png")


# --- 2. rank stability -------------------------------------------------------


def rank_stability(
    importance: pl.DataFrame, model: str, fold_set: str, destination: Path
) -> Path | None:
    """Does the model rely on the same signals over time, or does its reasoning drift?

    One line per top feature, its importance rank on each fold. A flat set of lines means
    the model's reasoning is stable; crossing lines mean it is not, and a ROC-AUC that held
    steady across the same folds would have said nothing about it either way.
    """
    _, plt = _matplotlib()
    aggregate = importance.filter(
        (pl.col("model_name") == model)
        & (pl.col("fold_set") == fold_set)
        & (pl.col("scope") == "fold_set")
    ).sort("mean_abs_shap", descending=True)
    per_fold = importance.filter(
        (pl.col("model_name") == model)
        & (pl.col("fold_set") == fold_set)
        & (pl.col("scope") == "fold")
    )
    fold_ids = sorted(str(v) for v in per_fold["fold_id"].unique().to_list())
    if len(fold_ids) < MIN_STABILITY_FOLDS:
        logger.info(
            "%s/%s has %d fold(s); a stability panel would be a flat line",
            model,
            fold_set,
            len(fold_ids),
        )
        return None

    figure, axes = plt.subplots(figsize=(10, 6))
    for name in aggregate.head(TOP_K)["feature_name"].to_list():
        series = per_fold.filter(pl.col("feature_name") == name).sort("fold_id")
        axes.plot(
            [str(v) for v in series["fold_id"].to_list()],
            series["rank"].to_list(),
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=str(name),
        )
    axes.invert_yaxis()  # rank 1 at the top, where a reader expects the most important
    axes.set_ylabel("importance rank (1 = most used)")
    axes.set_xlabel("fold")
    axes.tick_params(axis="x", labelrotation=90, labelsize=7)
    axes.grid(axis="y", alpha=0.25)
    axes.legend(fontsize=6, ncol=2, loc="lower left")
    axes.set_title(
        f"{model} - rank of the top {TOP_K} features across {len(fold_ids)} {fold_set} folds",
        fontsize=10,
    )
    _caption(
        figure,
        "Crossing lines are explanation drift: the model changed which signals it leaned "
        "on. That is a different phenomenon from Component 9's calibration drift, and a "
        "model can hold its ROC-AUC steady while this figure moves.",
    )
    return _save(figure, destination / f"explain_rank_stability_{model}_{fold_set}.png")


# --- 3. beeswarm -------------------------------------------------------------


def beeswarm(values: pl.DataFrame, model: str, fold_id: str, destination: Path) -> Path | None:
    """The distribution of each feature's attribution, not just its average.

    A mean absolute importance hides direction and shape. This shows whether a feature
    consistently raises risk, consistently lowers it, or does both depending on its value --
    which is the difference between a signal and a switch.
    """
    _, plt = _matplotlib()
    rows = values.filter((pl.col("model_name") == model) & (pl.col("fold_id") == fold_id))
    if rows.height < MIN_BEESWARM_ROWS:
        logger.info("%s/%s has %d rows; too few for a beeswarm", model, fold_id, rows.height)
        return None

    order = (
        rows.group_by("feature_name")
        .agg(pl.col("shap_value").abs().mean().alias("importance"))
        .sort("importance", descending=True)
        .head(DISPLAY_FEATURES)["feature_name"]
        .to_list()
    )

    figure, axes = plt.subplots(figsize=(9, 6))
    for position, name in enumerate(reversed(order)):
        subset = rows.filter(pl.col("feature_name") == name)
        shap = subset["shap_value"].to_list()
        # Vertical jitter is deterministic: derived from each point's own index rather than
        # drawn, so the figure is byte-identical across runs.
        offsets = [0.34 * ((index % 21) / 10.0 - 1.0) for index in range(len(shap))]
        feature = subset["transformed_value"].to_list()
        axes.scatter(
            shap,
            [position + offset for offset in offsets],
            c=feature,
            cmap="coolwarm",
            s=3,
            alpha=0.55,
            linewidths=0,
        )
    axes.set_yticks(range(len(order)))
    axes.set_yticklabels(list(reversed(order)), fontsize=8)
    axes.axvline(0.0, color="#666666", linewidth=0.8)
    axes.set_xlabel("SHAP value (log-odds)")
    axes.set_title(
        f"{model} - attribution distribution on {fold_id}\n"
        "colour is the standardised feature value (blue low, red high)",
        fontsize=10,
    )
    _caption(figure, CAUSALITY_CAPTION)
    return _save(figure, destination / f"explain_beeswarm_{model}_{fold_id}.png")


# --- 4. explanation drift ----------------------------------------------------


def stability_over_time(stability: pl.DataFrame, fold_set: str, destination: Path) -> Path | None:
    """Fold-to-fold rank agreement per model, over time.

    Two series per model would be misleading on one axis, so this plots the Spearman
    correlation and leaves the top-k Jaccard to the table -- they measure different things
    and a reader who sees them on one axis will average them by eye.
    """
    _, plt = _matplotlib()
    rows = stability.filter(
        (pl.col("fold_set") == fold_set) & (pl.col("comparison") == "consecutive")
    )
    if rows.height < MIN_STABILITY_FOLDS:
        logger.info("Too few consecutive comparisons in %s for a drift figure", fold_set)
        return None

    figure, axes = plt.subplots(figsize=(10, 5))
    for model in sorted(str(v) for v in rows["model_name"].unique().to_list()):
        series = rows.filter(pl.col("model_name") == model).sort("to_fold_id")
        axes.plot(
            [str(v) for v in series["to_fold_id"].to_list()],
            series["spearman_rho"].to_list(),
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=model,
        )
    axes.set_ylabel("Spearman rho of importance ranks vs the previous fold")
    axes.set_xlabel("fold")
    axes.set_ylim(0.0, 1.02)
    axes.grid(axis="y", alpha=0.25)
    axes.tick_params(axis="x", labelrotation=90, labelsize=7)
    axes.legend(fontsize=7)
    axes.set_title(f"Explanation drift - consecutive-fold rank agreement, {fold_set}", fontsize=10)
    _caption(
        figure,
        "1.0 means the model ranked its features identically to the previous quarter. "
        f"{APPROXIMATE_CAPTION}",
    )
    return _save(figure, destination / f"explain_drift_spearman_{fold_set}.png")


# --- 5. local cases ----------------------------------------------------------


def local_case(values: pl.DataFrame, case: dict[str, Any], destination: Path) -> Path | None:
    """One prediction, decomposed: the factors that raised and lowered its risk.

    The figure a non-ML reader actually needs. Bars are the contributions; the annotation
    carries the base value, the model's score before calibration and the calibrated
    probability, so the two numbers are never conflated.
    """
    _, plt = _matplotlib()
    model = str(case["model_name"])
    fold_id = str(case["fold_id"])
    row_id = str(case["target_inspection_id"])
    rows = values.filter(
        (pl.col("model_name") == model)
        & (pl.col("fold_id") == fold_id)
        & (pl.col("target_inspection_id") == row_id)
    )
    if rows.height == 0:
        logger.info("No attributions for case %s/%s/%s", model, fold_id, row_id)
        return None

    top = (
        rows.with_columns(pl.col("shap_value").abs().alias("_magnitude"))
        .sort("_magnitude", descending=True)
        .head(DISPLAY_FEATURES)
        .sort("shap_value")
    )
    names = [str(v) for v in top["feature_name"].to_list()]
    contributions = top["shap_value"].to_list()
    colours = ["#c0392b" if v > 0 else "#2874a6" for v in contributions]

    figure, axes = plt.subplots(figsize=(8, 6))
    axes.barh(range(len(names)), contributions, color=colours)
    axes.set_yticks(range(len(names)))
    axes.set_yticklabels(names, fontsize=8)
    axes.axvline(0.0, color="#333333", linewidth=0.8)
    axes.set_xlabel("contribution to log-odds (red raises risk, blue lowers it)")

    calibrated = case.get("calibrated_probability")
    calibrated_text = (
        f"calibrated probability {float(calibrated):.3f} ({case.get('calibration_method')})"
        if calibrated is not None
        else "no calibrated probability supplied"
    )
    axes.set_title(
        f"{model} - {case['tier']}-risk example on {fold_id}\n"
        f"inspection {row_id}: base {float(case['base_value']):+.3f} -> "
        f"log-odds {float(case['prediction_value']):+.3f}\n"
        f"model score before calibration {float(case['base_score']):.3f}; {calibrated_text}",
        fontsize=9,
    )
    _caption(
        figure,
        "Selected by predicted-risk quantile, never by whether the model was right. "
        f"{CAUSALITY_CAPTION}",
    )
    return _save(figure, destination / f"explain_local_{model}_{case['tier']}_{fold_id}.png")


# --- 6. COVID vs quarterly ---------------------------------------------------


def covid_comparison(importance: pl.DataFrame, model: str, destination: Path) -> Path | None:
    """What the model leaned on under regime shift, beside what it leaned on ordinarily.

    Side by side and never averaged. Component 6 measured the model *ordering* inverting on
    this fold; if the reasoning also changed, this is where it shows, and pooling the two
    would have hidden exactly that.
    """
    _, plt = _matplotlib()
    quarterly = importance.filter(
        (pl.col("model_name") == model)
        & (pl.col("fold_set") == "quarterly")
        & (pl.col("scope") == "fold_set")
    ).sort("mean_abs_shap", descending=True)
    covid = importance.filter(
        (pl.col("model_name") == model)
        & (pl.col("fold_set") == "covid_shift")
        & (pl.col("scope") == "fold_set")
    )
    if quarterly.height == 0 or covid.height == 0:
        logger.info("Missing a fold set for %s; skipping the COVID comparison", model)
        return None

    order = quarterly.head(DISPLAY_FEATURES)["feature_name"].to_list()
    covid_by_name = {str(r["feature_name"]): float(r["mean_abs_shap"]) for r in covid.to_dicts()}
    quarterly_by_name = {
        str(r["feature_name"]): float(r["mean_abs_shap"]) for r in quarterly.to_dicts()
    }
    positions = range(len(order))

    figure, axes = plt.subplots(figsize=(9, 6))
    axes.barh(
        [p + 0.2 for p in positions],
        [quarterly_by_name.get(str(n), 0.0) for n in reversed(order)],
        height=0.4,
        color="#3b6ea5",
        label="quarterly (17 folds, mean)",
    )
    axes.barh(
        [p - 0.2 for p in positions],
        [covid_by_name.get(str(n), 0.0) for n in reversed(order)],
        height=0.4,
        color="#b7791f",
        label="covid_shift (1 fold)",
    )
    axes.set_yticks(list(positions))
    axes.set_yticklabels([str(n) for n in reversed(order)], fontsize=8)
    axes.set_xlabel("mean |SHAP| (log-odds)")
    axes.legend(fontsize=8)
    axes.set_title(
        f"{model} - what the model leaned on, ordinary quarters vs the COVID regime shift",
        fontsize=10,
    )
    _caption(
        figure,
        "The two fold sets are never averaged together. covid_shift is a single fold with "
        "no variance estimate, over a period when the scheduling policy itself broke.",
    )
    return _save(figure, destination / f"explain_covid_comparison_{model}.png")


# --- entry point -------------------------------------------------------------


def render(tables: dict[str, pl.DataFrame], *, destination: Path) -> list[Path]:
    """Every figure this component can honestly draw from the persisted tables."""
    importance = tables["explanation_importance"]
    values = tables["explanation_values"]
    stability = tables["explanation_stability"]
    cases = tables["explanation_representative_cases"]

    paths: list[Path] = []
    models = sorted(str(v) for v in importance["model_name"].unique().to_list())
    for model in models:
        for maybe in (
            global_importance(importance, model, "quarterly", destination),
            rank_stability(importance, model, "quarterly", destination),
            covid_comparison(importance, model, destination),
        ):
            if maybe is not None:
                paths.append(maybe)

    report_folds = sorted(
        str(v)
        for v in values.filter(pl.col("fold_set") == "quarterly")["fold_id"].unique().to_list()
    )
    if report_folds:
        for model in models:
            drawn = beeswarm(values, model, report_folds[-1], destination)
            if drawn is not None:
                paths.append(drawn)

    for maybe in (stability_over_time(stability, "quarterly", destination),):
        if maybe is not None:
            paths.append(maybe)

    for case in cases.to_dicts():
        drawn = local_case(values, case, destination)
        if drawn is not None:
            paths.append(drawn)

    return sorted(paths)


__all__ = [
    "beeswarm",
    "covid_comparison",
    "global_importance",
    "local_case",
    "rank_stability",
    "render",
    "stability_over_time",
]
