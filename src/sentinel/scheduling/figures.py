"""Figures for Component 14. Four, and each answers a question a table answers worse.

**No figure here implies an optimisation.** There is no frontier, no efficient set, no "before
and after" that suggests the scheduler improved something. The component re-orders an approved
queue in time; it does not make the queue better, and a chart shaped like an optimisation
result would say otherwise regardless of its caption.

**Every figure that shows a scenario says so on the figure.** The ``flat_median`` mode fits its
queue by construction, so a chart that plotted the two modes side by side without labelling
would look like a scheduler improvement and would be a tautology.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from sentinel.scheduling.definitions import CapacityMode

logger = logging.getLogger(__name__)

OBSERVED = str(CapacityMode.OBSERVED_CALENDAR)

HORIZON_CAPTION = (
    "Approved queue against observed capacity. The bars are the inspections Chicago actually "
    "performed on each date; the green line is the running total of slots the horizon has "
    "supplied by that day. Where it ends below the dashed line, the horizon could not hold the "
    "approved queue -- and the gap between them is the backlog."
)

UTILIZATION_CAPTION = (
    "Slots supplied against slots used, by capacity mode. flat_median is a labelled scenario: "
    "it assigns every day the window's median rate, so at k_1_day and k_1_week it holds exactly "
    "k slots and is saturated by construction. The gap between the two modes is the measurement."
)

FLOW_CAPTION = (
    "Where the approved queue ended up. Backlog is not a failure of the scheduler and not a "
    "withdrawal of the recommendation -- those establishments keep their rank, their mechanism "
    "and their reason code, and the horizon simply ran out of days."
)

RESERVE_CAPTION = (
    "Coverage-reserve slots offered by Component 13 against those a strict-priority schedule "
    "reaches. The reserve sits at the tail of the rank order, so a horizon that falls short "
    "takes it first. Measured and reported here; correcting it would be re-ranking."
)


def _matplotlib() -> tuple[Any, Any]:
    """Import matplotlib with a headless backend, chosen before pyplot loads.

    Without it matplotlib picks a backend from the environment, which on a headless runner is a
    different one than on this machine, and a figure that renders differently per environment
    is not an artifact.
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
    figure.text(0.5, offset, text, ha="center", va="top", fontsize=7, wrap=True, color="#444444")


def _short(policy_id: str) -> str:
    return (
        policy_id.replace("coverage_", "")
        .replace("_share", "")
        .replace("population", "pop")
        .replace("half", "0.5x")
        .replace("double", "2x")
    )


def horizon_fill(
    slots: pl.DataFrame,
    summary: pl.DataFrame,
    *,
    fold_id: str,
    k_name: str,
    policy_id: str,
    destination: Path,
) -> Path | None:
    """One horizon, day by day: what the calendar supplied and what the queue needed."""
    _, plt = _matplotlib()
    cell = slots.filter(
        (pl.col("schedule_config_id").str.ends_with(OBSERVED))
        & (pl.col("fold_id") == fold_id)
        & (pl.col("k_name") == k_name)
    ).sort("day_index")
    if cell.is_empty():
        return None
    row = summary.filter(
        (pl.col("schedule_config_id").str.ends_with(OBSERVED))
        & (pl.col("fold_id") == fold_id)
        & (pl.col("k_name") == k_name)
        & (pl.col("policy_id") == policy_id)
    )
    if row.is_empty():
        return None
    k = int(row["k"][0])

    days = cell["day_index"].to_list()
    volumes = cell["n_slots"].to_list()
    cumulative = cell["cumulative_slots"].to_list()

    figure, axes = plt.subplots(figsize=(7.2, 3.6))
    axes.bar(days, volumes, color="#4C72B0", alpha=0.85, label="observed capacity that day")
    axes.plot(
        days,
        cumulative,
        color="#55A868",
        marker="o",
        markersize=3,
        label="cumulative slots supplied",
    )
    axes.axhline(k, color="#C44E52", linestyle="--", linewidth=1.2, label=f"approved queue (k={k})")
    axes.set_xlabel("operating day of the horizon")
    axes.set_ylabel("inspections")
    axes.set_title(f"{fold_id} / {k_name} / {_short(policy_id)} - observed calendar", fontsize=9)
    axes.set_xticks(days)
    axes.legend(fontsize=7, loc="upper left")
    axes.spines[["top", "right"]].set_visible(False)
    _caption(figure, HORIZON_CAPTION, offset=-0.12)
    return _save(figure, destination / f"schedule_horizon_{fold_id}_{k_name}.png")


def utilization_by_mode(summary: pl.DataFrame, *, k_name: str, destination: Path) -> Path | None:
    """Slots supplied and used per fold, both modes, with the scenario labelled."""
    _, plt = _matplotlib()
    cell = summary.filter((pl.col("k_name") == k_name) & (pl.col("policy_id") == "pure_risk")).sort(
        "fold_id"
    )
    if cell.is_empty():
        return None
    observed = cell.filter(~pl.col("is_scenario"))
    scenario = cell.filter(pl.col("is_scenario"))
    if observed.is_empty():
        return None

    folds = [f.replace("quarterly-", "") for f in observed["fold_id"].to_list()]
    positions = range(len(folds))
    figure, axes = plt.subplots(figsize=(7.6, 3.4))
    axes.plot(
        list(positions),
        observed["capacity_utilization"].to_list(),
        color="#4C72B0",
        marker="o",
        markersize=3.5,
        label="observed calendar (measured)",
    )
    if not scenario.is_empty():
        axes.plot(
            list(positions),
            scenario["capacity_utilization"].to_list(),
            color="#CCB974",
            marker="s",
            markersize=3.5,
            linestyle="--",
            label="flat median (SCENARIO)",
        )
    axes.set_ylim(0.0, 1.05)
    axes.set_xticks(list(positions))
    axes.set_xticklabels(folds, rotation=60, ha="right", fontsize=7)
    axes.set_ylabel("capacity utilisation")
    axes.set_title(f"Capacity utilisation at {k_name}, pure_risk", fontsize=9)
    axes.legend(fontsize=7, loc="lower left")
    axes.spines[["top", "right"]].set_visible(False)
    _caption(figure, UTILIZATION_CAPTION, offset=-0.30)
    return _save(figure, destination / f"schedule_utilization_{k_name}.png")


def queue_flow(summary: pl.DataFrame, *, k_name: str, destination: Path) -> Path | None:
    """Where the approved queue ended up, per policy, on the observed calendar."""
    _, plt = _matplotlib()
    cell = (
        summary.filter(
            (pl.col("k_name") == k_name)
            & (~pl.col("is_scenario"))
            & (pl.col("fold_set") == "quarterly")
        )
        .group_by("policy_id")
        .agg(
            pl.col("n_scheduled").sum().alias("scheduled"),
            pl.col("n_backlog").sum().alias("backlog"),
        )
        .sort("policy_id")
    )
    if cell.is_empty():
        return None
    labels = [_short(p) for p in cell["policy_id"].to_list()]
    scheduled = cell["scheduled"].to_list()
    backlog = cell["backlog"].to_list()
    positions = range(len(labels))

    figure, axes = plt.subplots(figsize=(7.2, 3.4))
    axes.bar(list(positions), scheduled, color="#55A868", label="scheduled in horizon")
    axes.bar(
        list(positions),
        backlog,
        bottom=scheduled,
        color="#C44E52",
        alpha=0.85,
        label="backlog (still recommended)",
    )
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    axes.set_ylabel("approved recommendations")
    axes.set_title(f"Recommendation to schedule at {k_name}, quarterly folds", fontsize=9)
    axes.legend(fontsize=7)
    axes.spines[["top", "right"]].set_visible(False)
    _caption(figure, FLOW_CAPTION, offset=-0.26)
    return _save(figure, destination / f"schedule_flow_{k_name}.png")


def reserve_survival(preservation: pl.DataFrame, *, destination: Path) -> Path | None:
    """The headline: coverage-reserve slots offered against those the schedule reaches."""
    _, plt = _matplotlib()
    cell = (
        preservation.filter(~pl.col("is_scenario") & (pl.col("n_reserve_recommended") > 0))
        .group_by("policy_id")
        .agg(
            pl.col("n_reserve_recommended").sum().alias("offered"),
            pl.col("n_reserve_scheduled").sum().alias("reached"),
        )
        .sort("policy_id")
    )
    if cell.is_empty():
        return None
    labels = [_short(p) for p in cell["policy_id"].to_list()]
    offered = cell["offered"].to_list()
    reached = cell["reached"].to_list()
    positions = list(range(len(labels)))
    width = 0.38

    figure, axes = plt.subplots(figsize=(7.2, 3.4))
    axes.bar(
        [p - width / 2 for p in positions],
        offered,
        width,
        color="#4C72B0",
        label="reserve slots allocated by Component 13",
    )
    axes.bar(
        [p + width / 2 for p in positions],
        reached,
        width,
        color="#C44E52",
        label="reserve slots the schedule reaches",
    )
    for position, (a, b) in enumerate(zip(offered, reached, strict=True)):
        if a:
            axes.text(
                position,
                max(a, b) * 1.02,
                f"-{(a - b) / a:.0%}",
                ha="center",
                fontsize=7,
                color="#C44E52",
            )
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    axes.set_ylabel("coverage-reserve slots")
    axes.set_title("What the calendar costs the coverage reserve", fontsize=9)
    axes.legend(fontsize=7)
    axes.spines[["top", "right"]].set_visible(False)
    _caption(figure, RESERVE_CAPTION, offset=-0.26)
    return _save(figure, destination / "schedule_reserve_survival.png")


def render(tables: dict[str, pl.DataFrame], *, destination: Path) -> list[Path]:
    """Every figure this component publishes."""
    written: list[Path] = []
    for path in (
        horizon_fill(
            tables["schedule_slots"],
            tables["schedule_summary"],
            fold_id="quarterly-2026Q2",
            k_name="k_1_week",
            policy_id="pure_risk",
            destination=destination,
        ),
        utilization_by_mode(tables["schedule_summary"], k_name="k_1_week", destination=destination),
        queue_flow(tables["schedule_summary"], k_name="k_1_week", destination=destination),
        reserve_survival(tables["priority_preservation"], destination=destination),
    ):
        if path is not None:
            written.append(path)
    return written


__all__ = [
    "horizon_fill",
    "queue_flow",
    "render",
    "reserve_survival",
    "utilization_by_mode",
]
