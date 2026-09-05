"""Figures for Component 13. Every one answers a stated question; none is decorative.

Drawn only from the persisted tables -- never from an in-memory object -- so a figure can be
regenerated from the artifact alone and cannot silently disagree with it.

Three rules this component needs.

**No point is marked optimal.** The frontier figure shows citations against coverage and
leaves the reader to choose, because choosing requires an exchange rate between a missed
Priority citation and an uninspected establishment with no history, and nothing in this
project measures one. A star on one point would be this component making the governance
decision it spends four modules refusing to make.

**A cost is drawn as a cost.** Where a policy gives up citations the axis says so in citations,
not in a normalised index. "-34" is a number a department can argue about; "-0.7% relative
utility" is one nobody can.

**No figure gets a title that reads as a verdict.** The titles name what was measured.

A figure that cannot be drawn honestly returns ``None`` and logs, rather than raising or
drawing something misleading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from sentinel.policy.definitions import BASELINE_POLICY_ID, PRIMARY_K_LEVEL

logger = logging.getLogger(__name__)

#: Below this there is no shape to plot, only a handful of points pretending to be a curve.
MIN_POINTS = 2

FRONTIER_CAPTION = (
    "Each point is one policy, pooled over the quarterly test windows. No point is marked "
    "optimal: choosing between them needs an exchange rate between a missed Priority citation "
    "and an uninspected establishment with no history, and this project measures neither."
)
COST_CAPTION = (
    "Measured against pure risk prioritisation at the identical model, fold and capacity. "
    "Negative means the policy gave up citations. A reserve is described as free only where "
    "this number is zero."
)
COVERAGE_CAPTION = (
    "The coverage-eligible population -- no canvass since 2018-07-01 -- is 10.4% of the "
    "quarterly test rows. A queue share above that line is over-selection relative to "
    "population share, which is what the risk ranking already delivers."
)
MECHANISM_CAPTION = (
    "How each policy divided one day of real inspection capacity. Risk-priority slots are the "
    "model's ranking; reserve slots are the policy's allocation. The split is the answer to "
    "'did the model decide this, or did the policy?'"
)


def _matplotlib() -> tuple[Any, Any]:
    """Import matplotlib with a headless backend, chosen before pyplot loads.

    Without it matplotlib picks a backend from the environment, which on a headless runner is
    a different one than on this machine, and a figure that renders differently per
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


def _caption(figure: Any, text: str, *, offset: float = -0.06) -> None:
    """Place the caption below the axes.

    ``offset`` exists because a figure with long rotated tick labels needs the caption pushed
    further down. A caption printed on top of the axis labels makes both unreadable, which is
    worse than no caption at all.
    """
    figure.text(0.5, offset, text, ha="center", va="top", fontsize=7, wrap=True, color="#444444")


def _short(policy_id: str) -> str:
    """A policy id short enough for an axis tick, without losing the mechanism."""
    return (
        policy_id.replace("coverage_", "")
        .replace("_share", "")
        .replace("population", "pop")
        .replace("half", "0.5x")
        .replace("double", "2x")
    )


# --- 1. the trade-off frontier -------------------------------------------------


def frontier(
    table: pl.DataFrame, *, model: str, fold_set: str, k_name: str, destination: Path
) -> Path | None:
    """Citations discovered against coverage-eligible establishments served.

    The component's central figure. Both axes are counts of real things, and neither is
    normalised into an index, because the whole point is that they are not commensurable.
    """
    subset = table.filter(
        (pl.col("model_name") == model)
        & (pl.col("fold_set") == fold_set)
        & (pl.col("k_name") == k_name)
    )
    if subset.height < MIN_POINTS:
        logger.info("Frontier: %d point(s) for %s/%s; not drawn", subset.height, model, k_name)
        return None

    # Policies that produced identical outcomes land on the same point, which happens often
    # here: an inert floor *is* the baseline. Their labels are combined rather than
    # overprinted, because two illegible names stacked on one marker hide exactly the fact
    # worth seeing -- that the two policies did the same thing.
    coincident: dict[tuple[float, float], list[str]] = {}
    for row in subset.sort("policy_id").iter_rows(named=True):
        point = (float(row["eligible_selected"]), float(row["positives_selected"]))
        coincident.setdefault(point, []).append(str(row["policy_id"]))

    _, plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    for row in subset.sort("policy_id").iter_rows(named=True):
        dominated = bool(row["is_dominated"])
        baseline = row["policy_id"] == BASELINE_POLICY_ID
        axis.scatter(
            row["eligible_selected"],
            row["positives_selected"],
            s=110 if baseline else 70,
            marker="s" if baseline else "o",
            facecolor="#ffffff" if dominated else ("#1f4e79" if baseline else "#4a90c2"),
            edgecolor="#1f4e79",
            zorder=3,
            label=None,
        )
    for (x_value, y_value), policies in sorted(coincident.items()):
        axis.annotate(
            " = ".join(_short(policy) for policy in policies),
            (x_value, y_value),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=7,
        )
    axis.set_xlabel("coverage-eligible establishments selected (pooled)")
    axis.set_ylabel("Priority citations discovered (pooled)")
    axis.set_title(f"Policy trade-off at {k_name} -- {model}, {fold_set}", fontsize=10)
    axis.grid(alpha=0.25, zorder=0)
    # Hollow markers are dominated policies; the square is the baseline. Stated in the legend
    # rather than in a colour key that implies a ranking among the survivors.
    axis.plot([], [], "s", color="#1f4e79", label="pure risk (baseline)")
    axis.plot([], [], "o", color="#4a90c2", label="on the frontier")
    axis.plot([], [], "o", markerfacecolor="#ffffff", markeredgecolor="#1f4e79", label="dominated")
    axis.legend(fontsize=7, loc="best")
    _caption(figure, FRONTIER_CAPTION)
    return _save(figure, destination / f"policy_frontier_{model}_{fold_set}_{k_name}.png")


# --- 2. what coverage costs ----------------------------------------------------


def opportunity_cost(
    comparison: pl.DataFrame, *, model: str, fold_set: str, destination: Path
) -> Path | None:
    """Citations given up by each policy, at each capacity, pooled.

    Drawn as a bar chart of a signed count. A policy whose bar is at zero cost nothing; a
    policy whose bar is below zero cost exactly that many inspections that would have found a
    Priority violation.
    """
    subset = comparison.filter(
        (pl.col("model_name") == model)
        & (pl.col("fold_set") == fold_set)
        & (pl.col("policy_id") != BASELINE_POLICY_ID)
    )
    if subset.is_empty():
        logger.info("Opportunity cost: no non-baseline rows for %s/%s; not drawn", model, fold_set)
        return None
    pooled = (
        subset.group_by("policy_id", "k_name")
        .agg(pl.col("delta_positives").sum().alias("delta"))
        .sort("policy_id", "k_name")
    )
    policies = sorted(pooled["policy_id"].unique().to_list())
    k_names = sorted(pooled["k_name"].unique().to_list())
    if not policies or not k_names:
        return None

    _, plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(max(7.0, 1.3 * len(policies)), 4.6))
    width = 0.8 / len(k_names)
    for offset, k_name in enumerate(k_names):
        values = [
            float(
                pooled.filter((pl.col("policy_id") == policy) & (pl.col("k_name") == k_name))[
                    "delta"
                ].sum()
            )
            for policy in policies
        ]
        positions = [i + offset * width - 0.4 + width / 2 for i in range(len(policies))]
        axis.bar(positions, values, width=width, label=k_name)
    axis.axhline(0.0, color="#333333", linewidth=0.9)
    axis.set_xticks(range(len(policies)))
    axis.set_xticklabels([_short(p) for p in policies], rotation=20, ha="right", fontsize=8)
    axis.set_ylabel("Priority citations gained (+) or given up (-)")
    axis.set_title(f"What each policy cost against pure risk -- {model}, {fold_set}", fontsize=10)
    axis.grid(alpha=0.25, axis="y")
    axis.legend(fontsize=7, title="capacity", title_fontsize=7)
    _caption(figure, COST_CAPTION, offset=-0.14)
    return _save(figure, destination / f"policy_opportunity_cost_{model}_{fold_set}.png")


# --- 3. the no-history population ----------------------------------------------


def coverage_trend(
    comparison: pl.DataFrame, *, model: str, k_name: str, destination: Path
) -> Path | None:
    """The coverage-eligible share of the queue, per fold, under each policy.

    The figure that carries this component's most surprising measurement: the pure-risk queue
    already over-selects establishments with no code-era history by four to five times their
    population share, and that over-selection is falling fold by fold.
    """
    subset = comparison.filter(
        (pl.col("model_name") == model)
        & (pl.col("fold_set") == "quarterly")
        & (pl.col("k_name") == k_name)
    )
    if subset.is_empty():
        logger.info("Coverage trend: no quarterly rows for %s/%s; not drawn", model, k_name)
        return None
    folds = sorted(subset["fold_id"].unique().to_list())
    if len(folds) < MIN_POINTS:
        return None

    # The fold set is already in the title, so the repeated "quarterly-" prefix on seventeen
    # rotated tick labels is pure noise -- and it is what pushed the labels into the caption.
    labels = [fold.removeprefix("quarterly-") for fold in folds]

    _, plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(max(8.0, 0.5 * len(folds)), 4.8))
    for policy in sorted(subset["policy_id"].unique().to_list()):
        series = subset.filter(pl.col("policy_id") == policy).sort("fold_id")
        axis.plot(
            [str(fold).removeprefix("quarterly-") for fold in series["fold_id"].to_list()],
            series["eligible_selected_share"].to_list(),
            marker="o",
            markersize=3,
            linewidth=2.2 if policy == BASELINE_POLICY_ID else 1.0,
            label=_short(str(policy)),
        )
    axis.axhline(
        0.1043, color="#b03a2e", linestyle="--", linewidth=1.0, label="population share (0.1043)"
    )
    axis.set_ylabel("coverage-eligible share of the queue")
    axis.set_title(
        f"Who the queue serves, fold by fold, at {k_name} -- {model}, quarterly", fontsize=10
    )
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    # Outside the axes: the series run the full height of the plot, so any in-axes legend
    # covers data in some fold.
    axis.legend(fontsize=6, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _caption(figure, COVERAGE_CAPTION, offset=-0.22)
    return _save(figure, destination / f"policy_coverage_trend_{model}_{k_name}.png")


# --- 4. how the queue was filled -----------------------------------------------


def mechanism_composition(
    allocation: pl.DataFrame, *, model: str, fold_set: str, k_name: str, destination: Path
) -> Path | None:
    """Slots filled by risk priority against slots filled by the coverage reserve.

    The figure that makes "did the model decide this, or did the policy?" visible in one
    glance, and the one that shows how rarely the reserve is the answer.
    """
    subset = allocation.filter(
        (pl.col("model_name") == model)
        & (pl.col("fold_set") == fold_set)
        & (pl.col("k_name") == k_name)
    )
    if subset.is_empty():
        logger.info("Mechanism composition: no rows for %s/%s; not drawn", model, k_name)
        return None
    pooled = (
        subset.group_by("policy_id")
        .agg(pl.col("n_risk").sum().alias("risk"), pl.col("n_reserve").sum().alias("reserve"))
        .sort("policy_id")
    )
    policies = pooled["policy_id"].to_list()

    _, plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(max(7.0, 1.2 * len(policies)), 4.4))
    risk = [float(v) for v in pooled["risk"].to_list()]
    reserve = [float(v) for v in pooled["reserve"].to_list()]
    positions = list(range(len(policies)))
    axis.bar(positions, risk, label="risk_priority", color="#1f4e79")
    axis.bar(positions, reserve, bottom=risk, label="coverage_reserve", color="#e08a1e")
    for index, value in enumerate(reserve):
        if value:
            axis.text(
                index,
                risk[index] + value,
                f"{int(value)}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#7a4a05",
            )
    # Every bar is the same height -- capacity is fixed, which is the point -- so without
    # headroom the reserve counts printed on top of them collide with the title.
    axis.set_ylim(0.0, max((r + v) for r, v in zip(risk, reserve, strict=True)) * 1.12)
    axis.set_xticks(positions)
    axis.set_xticklabels([_short(str(p)) for p in policies], rotation=20, ha="right", fontsize=8)
    axis.set_ylabel(f"slots at {k_name}, pooled over {fold_set} folds")
    axis.set_title(f"How each policy filled capacity -- {model}, {fold_set}", fontsize=10)
    axis.grid(alpha=0.25, axis="y")
    axis.legend(fontsize=7)
    _caption(figure, MECHANISM_CAPTION, offset=-0.14)
    return _save(
        figure, destination / f"policy_mechanism_composition_{model}_{fold_set}_{k_name}.png"
    )


def render(tables: dict[str, pl.DataFrame], *, destination: Path) -> list[Path]:
    """Draw every figure the tables can support, and skip the rest.

    ``None`` from a figure function is a normal outcome, not an error: a run restricted to one
    policy genuinely has no frontier to draw, and drawing one anyway is how a figure ends up
    saying more than the data does.
    """
    comparison = tables.get("policy_comparison", pl.DataFrame())
    frontier_table = tables.get("policy_frontier", pl.DataFrame())
    allocation = tables.get("policy_selection_allocation", pl.DataFrame())
    selection = tables.get("policy_model_selection", pl.DataFrame())

    if comparison.is_empty() or selection.is_empty():
        logger.info("No comparison or selection table; drawing no figures")
        return []
    chosen = selection.filter(pl.col("is_selected"))
    if chosen.is_empty():
        logger.info("No selected model; drawing no figures")
        return []
    model = str(chosen["model_name"][0])
    fold_set = "quarterly"

    paths: list[Path] = []
    for figure_path in (
        frontier(
            frontier_table,
            model=model,
            fold_set=fold_set,
            k_name=PRIMARY_K_LEVEL,
            destination=destination,
        ),
        opportunity_cost(comparison, model=model, fold_set=fold_set, destination=destination),
        coverage_trend(comparison, model=model, k_name=PRIMARY_K_LEVEL, destination=destination),
        mechanism_composition(
            allocation,
            model=model,
            fold_set=fold_set,
            k_name=PRIMARY_K_LEVEL,
            destination=destination,
        ),
    ):
        if figure_path is not None:
            paths.append(figure_path)
    return paths


__all__ = [
    "MIN_POINTS",
    "coverage_trend",
    "frontier",
    "mechanism_composition",
    "opportunity_cost",
    "render",
]
