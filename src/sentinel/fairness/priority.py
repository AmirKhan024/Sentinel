"""Who reaches the top of the ranking, and what that is worth to them. Pure -- no I/O.

Sentinel is a prioritisation system, so probability metrics alone do not describe what it
does to anyone. This module answers the operational question: **if only the top k
establishments can be inspected first, who is in that k, and how much of each group's actual
risk did it capture?**

Two quantities live here and they are deliberately not combined.

```text
selection rate   n_selected / n_rows          representation: was this group prioritised?
capture rate     positives_selected / n_pos   effectiveness:  was that prioritisation useful?
```

A group can be over-represented in the top k while the ranking finds a smaller share of its
violations than average -- selected often and selected badly. Reporting a single "fairness at
k" number would average those two into something with no interpretation at all.

**The selection is city-wide and competitive.** The cutoff is taken over every audited row in
the fold, then groups are counted inside it. That is what makes capture different from
``recall_at_k``, which selects its top k within whatever rows it is handed: a group's capture
rate is what a competition against every other group left it, and a within-group recall would
hide exactly the effect this audit exists to find.

**Neither number is a target.** Outcome rates differ from 0.220 to 0.566 across supported
community areas, so a working risk model is expected to select at different rates; parity
would require ignoring a measured difference in outcomes. The point is to make the trade-off
visible, and Component 13 owns what to do about it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from sentinel.evaluation import metrics as canonical
from sentinel.evaluation import simulate
from sentinel.evaluation.models import FoldSpec
from sentinel.fairness.definitions import (
    K_LEVELS,
    Grain,
    GroupDefinitionSpec,
    GroupStatus,
    Stage,
)
from sentinel.fairness.metrics import capture_rate, selection_rate_ratio
from sentinel.fairness.models import GroupSupport, PriorityRow

#: The tie-break column, and Component 5's. Ties are settled on ``target_inspection_id``
#: ascending as a string, never on frame order -- two runs over the same rows shuffled must
#: select the same establishments, and Parquet row order is not a contract.
TIE_BREAK = "target_inspection_id"


class PriorityError(ValueError):
    """A top-k audit could not be computed over the rows it was handed."""


def k_values_for(frame: pl.DataFrame, fold: FoldSpec, median_daily: int) -> dict[str, int]:
    """The cutoffs this fold is audited at, from Component 5's own derivation.

    ``capacity_k_values`` is called rather than reimplemented, so the priority audit inherits
    the project's answer to "how many inspections fit in a day" instead of inventing a second
    one. It returns eight cutoffs; ``K_LEVELS`` selects the five this component reports, and
    the selection is frozen in ``definitions`` rather than made here.
    """
    if frame.is_empty():
        return {}
    window = simulate.build_window(
        ids=frame.get_column(TIE_BREAK).to_list(),
        labels=frame.get_column("target").to_list(),
        dates=frame.get_column("rd").to_list(),
    )
    available = simulate.capacity_k_values(window, median_daily=max(1, median_daily))
    return {name: available[name] for name in K_LEVELS if name in available}


def select_top_k(frame: pl.DataFrame, score_column: str, k: int) -> pl.DataFrame:
    """The ``k`` highest-scoring rows of ``frame``, ties settled canonically.

    Uses ``evaluation.metrics.top_k_indices``, which sorts by descending score then ascending
    tie-break key. Reimplementing the sort here would be a second tie-breaking rule in the
    project, and two rules that agree today are two rules that can disagree later -- on a
    ranking whose scores are heavily tied, which Component 9 measured isotonic producing.
    """
    if k < 1:
        raise PriorityError(f"k must be at least 1, got {k}")
    if frame.is_empty():
        return frame
    chosen = canonical.top_k_indices(
        frame.get_column(score_column).to_list(),
        frame.get_column(TIE_BREAK).to_list(),
        k,
    )
    return frame[sorted(chosen)]


def audit(
    frame: pl.DataFrame,
    spec: GroupDefinitionSpec,
    support: Mapping[str, GroupSupport],
    *,
    model_name: str,
    stage: Stage,
    score_column: str,
    grain: Grain,
    fold_set: str,
    fold_id: str,
    k_name: str,
    k: int,
) -> list[PriorityRow]:
    """One (model, stage, group definition, cutoff) audit over one grain's rows.

    Emits a row for **every observed group**, including groups that placed nobody in the top
    k and groups below the support floor. A group absent from the selected set is the most
    interesting row in the table and would vanish under an inner join against the selection.

    Support gates the *reading*, not the arithmetic: counts are always real, and
    ``group_status`` says whether the rates derived from them should be quoted.
    """
    if frame.is_empty():
        return []
    column = spec.source_column
    total_rows = frame.height
    selected = select_top_k(frame, score_column, k)
    overall_selected = selected.height
    overall_positives = int(frame.get_column("target").sum())
    overall_captured = int(selected.get_column("target").sum())
    overall_capture = capture_rate(overall_positives, overall_captured)

    population = (
        frame.group_by(column)
        .agg(pl.len().alias("n_rows"), pl.col("target").sum().alias("n_positive"))
        .sort(column)
    )
    chosen = (
        selected.group_by(column)
        .agg(pl.len().alias("n_selected"), pl.col("target").sum().alias("positives_selected"))
        .sort(column)
    )
    merged = population.join(chosen, on=column, how="left").with_columns(
        pl.col("n_selected").fill_null(0), pl.col("positives_selected").fill_null(0)
    )

    rows: list[PriorityRow] = []
    for record in merged.to_dicts():
        value = str(record[column])
        n_rows = int(record["n_rows"])
        n_positive = int(record["n_positive"])
        n_selected = int(record["n_selected"])
        positives_selected = int(record["positives_selected"])
        entry = support.get(value)
        status = entry.ranking_status if entry else GroupStatus.INSUFFICIENT_SUPPORT
        reason = entry.insufficient_reason if entry else "group absent from the support table"

        rows.append(
            PriorityRow(
                model_name=model_name,
                stage=stage,
                group_definition=spec.name,
                group_value=value,
                grain=grain.value,
                fold_set=fold_set,
                fold_id=fold_id,
                k_name=k_name,
                k=k,
                n_rows=n_rows,
                n_positive=n_positive,
                population_share=n_rows / total_rows,
                n_selected=n_selected,
                selected_share=(n_selected / overall_selected) if overall_selected else 0.0,
                selection_rate=(n_selected / n_rows) if n_rows else None,
                selection_rate_ratio=selection_rate_ratio(
                    n_selected, n_rows, overall_selected, total_rows
                ),
                positives_selected=positives_selected,
                # None rather than 0.0 when nothing was selected: "none of the zero rows we
                # picked were positive" is not a precision of zero.
                precision_in_selected=((positives_selected / n_selected) if n_selected else None),
                capture_rate=capture_rate(n_positive, positives_selected),
                overall_capture_rate=overall_capture,
                group_status=status,
                insufficient_reason=reason,
            )
        )
    return rows


def supported_capture(rows: Sequence[PriorityRow]) -> list[PriorityRow]:
    """The rows whose capture rate is quotable: supported, and with positives to capture."""
    return [
        row
        for row in rows
        if row.group_status is GroupStatus.SUPPORTED and row.capture_rate is not None
    ]


__all__ = [
    "TIE_BREAK",
    "PriorityError",
    "audit",
    "k_values_for",
    "select_top_k",
    "supported_capture",
]
