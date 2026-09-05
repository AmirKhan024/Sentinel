"""Figures for Component 16. One figure; it answers a stated question.

Drawn only from the persisted queue table, never from an in-memory object, so a figure can be
regenerated from the artifact alone and cannot silently disagree with it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

logger = logging.getLogger(__name__)

CASES_BY_TRIGGER_CAPTION = (
    "Cases flagged this run, by deterministic trigger. A row hit by both triggers is counted "
    "in both bars, so the two bars need not sum to the number of queue rows."
)


def render(tables: dict[str, pl.DataFrame], *, destination: Path) -> list[Path]:
    """Render every figure this component defines, or log and skip one that cannot be drawn."""
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    path = _cases_by_trigger(tables.get("human_review_queue"), destination)
    if path is not None:
        written.append(path)
    return written


def _cases_by_trigger(queue: pl.DataFrame | None, destination: Path) -> Path | None:
    if queue is None or queue.is_empty():
        logger.info("No flagged cases; skipping cases-by-trigger figure")
        return None
    counts: dict[str, int] = {}
    for reason in ("policy_warning_present", "no_execution_record_on_scheduled_row"):
        counts[reason] = queue.filter(
            pl.col("trigger_reasons").str.contains(reason, literal=True)
        ).height
    if not any(counts.values()):
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(counts.keys()), list(counts.values()))
    ax.set_ylabel("cases flagged")
    ax.set_title("Component 16 -- cases flagged by trigger")
    fig.text(0.01, -0.05, CASES_BY_TRIGGER_CAPTION, wrap=True, fontsize=8)
    path = destination / "review_cases_by_trigger.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


__all__ = ["render"]
