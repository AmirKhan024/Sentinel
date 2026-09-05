"""Summarising how far apart the groups are, in several ways rather than one. Pure -- no I/O.

**There is deliberately no fairness score here.** A single number would be a weighting of
mutually incompatible criteria, chosen by whoever wrote it and invisible to whoever read it.
Calibration parity and selection-rate parity cannot both hold when base rates differ, and they
differ here from 0.220 to 0.566 across supported community areas -- so any scalar would be
silently answering a question about which criterion matters more, which is a policy question
this component is not delegated.

Four measures are reported side by side because they disagree usefully:

```text
spread        max - min        absolute, in the metric's own units
ratio         max / min        relative, and undefined at a zero denominator
max_deviation max |g - ref|    distance from the pooled population, not from another group
weighted_sd   rows-weighted    how uneven the whole distribution is, not just its ends
```

The first two describe the extremes and are dominated by the two groups at the ends; the last
describes the body. A city where one neighbourhood is an outlier and a city where every
neighbourhood differs are different problems, and only reporting both tells them apart.

**Every measure is computed over supported groups only, and every row says how many were
excluded.** A spread over 51 of 78 community areas is a different claim from one over all 78,
and the two are indistinguishable without ``n_groups_unsupported``.

This module works on frames rather than on dataclasses, and there is exactly one
implementation of each measure. The metrics table and the priority table's capture column are
both reduced to :data:`COMPARABLE_COLUMNS` by the caller, so one code path serves both -- two
that agreed today are two that could disagree later, inside the same comparison.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from sentinel.fairness.definitions import (
    FAIRNESS_DEFINITION_VERSION,
    DisparityMeasure,
    GroupStatus,
)
from sentinel.fairness.metrics import max_deviation, ratio, spread, weighted_sd

#: The columns a frame must carry to be summarised.
COMPARABLE_COLUMNS: tuple[str, ...] = (
    "model_name",
    "stage",
    "group_definition",
    "group_value",
    "grain",
    "fold_set",
    "fold_id",
    "metric",
    "k_name",
    "value",
    "n_rows",
    "group_status",
)

#: What identifies one comparable cell. Comparing an ROC-AUC against an ECE, or one fold's
#: spread against another's, would be arithmetic over incommensurable numbers.
CELL_KEYS: tuple[str, ...] = (
    "model_name",
    "stage",
    "group_definition",
    "grain",
    "fold_set",
    "fold_id",
    "metric",
    "k_name",
)


class DisparityError(ValueError):
    """A disparity summary could not be computed over the rows it was handed."""


def extremes(
    values: Sequence[float], group_values: Sequence[str], weights: Sequence[int]
) -> tuple[float | None, str, int, float | None, str, int]:
    """The highest and lowest group, each with its identity and its row count.

    The row counts travel with the extremes for one reason: a dramatic ratio from a group of
    twelve rows must never be quotable without its support visible in the same record.
    """
    if not values:
        return None, "", 0, None, "", 0
    high = max(range(len(values)), key=lambda i: values[i])
    low = min(range(len(values)), key=lambda i: values[i])
    return (
        values[high],
        group_values[high],
        weights[high],
        values[low],
        group_values[low],
        weights[low],
    )


def undefined_reason(
    measure: str, value: float | None, values: Sequence[float], reference: float | None
) -> str:
    """Why a measure is null, stated rather than left for a reader to infer.

    A null with no reason is indistinguishable from a bug, and in a fairness table it is
    indistinguishable from a group that was quietly dropped. Three causes are separated
    because they mean different things: too few groups cleared the floor, a denominator
    vanished, or there was nothing to compare against.
    """
    if value is not None:
        return ""
    if len(values) < 2:
        return f"{len(values)} supported group(s); a disparity needs at least two"
    if measure == DisparityMeasure.RATIO.value and min(values) <= 0.0:
        return (
            f"minimum value is {min(values)}, so max/min is undefined. Null rather than "
            "infinity: a vanished denominator is not an infinite disparity."
        )
    if measure == DisparityMeasure.MAX_DEVIATION.value and reference is None:
        return "no pooled reference value over these rows"
    return "undefined on these values"


def measures(
    values: Sequence[float],
    weights: Sequence[int],
    reference: float | None,
) -> dict[str, float | None]:
    """The four disparity measures over one comparable cell's supported values."""
    return {
        DisparityMeasure.SPREAD.value: spread(values),
        DisparityMeasure.RATIO.value: ratio(values),
        DisparityMeasure.MAX_DEVIATION.value: max_deviation(values, reference),
        DisparityMeasure.WEIGHTED_SD.value: weighted_sd(values, weights),
    }


def summarise(
    comparable: pl.DataFrame,
    references: Mapping[tuple[str, str, str, str, str], float | None],
) -> list[dict[str, object]]:
    """Disparity rows for every comparable cell in ``comparable``.

    ``references`` maps ``(model, stage, fold_set, fold_id, metric)`` to the pooled population
    value over the same rows. It is passed in rather than derived here because only the caller
    holds the un-grouped rows, and re-deriving it from the group values would produce a *mean
    of group means* -- a different quantity, weighted by group count rather than by rows, and
    wrong by more the more uneven the groups are, which is exactly the situation being
    measured.
    """
    missing = [c for c in COMPARABLE_COLUMNS if c not in comparable.columns]
    if missing:
        raise DisparityError(f"comparable frame is missing {', '.join(missing)}")
    if comparable.is_empty():
        return []

    out: list[dict[str, object]] = []
    cells = comparable.select(CELL_KEYS).unique().sort(list(CELL_KEYS))
    for cell in cells.to_dicts():
        subset = comparable
        for key in CELL_KEYS:
            subset = subset.filter(pl.col(key) == cell[key])
        usable = subset.filter(
            (pl.col("group_status") == GroupStatus.SUPPORTED.value) & pl.col("value").is_not_null()
        ).sort("group_value")

        values = [float(v) for v in usable.get_column("value").to_list()]
        weights = [int(v) for v in usable.get_column("n_rows").to_list()]
        group_values = [str(v) for v in usable.get_column("group_value").to_list()]
        reference = references.get(
            (
                str(cell["model_name"]),
                str(cell["stage"]),
                str(cell["fold_set"]),
                str(cell["fold_id"]),
                str(cell["metric"]),
            )
        )

        high, high_group, high_rows, low, low_group, low_rows = extremes(
            values, group_values, weights
        )
        for measure, value in measures(values, weights, reference).items():
            out.append(
                {
                    "model_name": str(cell["model_name"]),
                    "stage": str(cell["stage"]),
                    "group_definition": str(cell["group_definition"]),
                    "grain": str(cell["grain"]),
                    "fold_set": str(cell["fold_set"]),
                    "fold_id": str(cell["fold_id"]),
                    "metric": str(cell["metric"]),
                    "k_name": str(cell["k_name"]),
                    "measure": measure,
                    "value": value,
                    "reference_value": reference,
                    "max_value": high,
                    "max_group": high_group,
                    "max_group_rows": high_rows,
                    "min_value": low,
                    "min_group": low_group,
                    "min_group_rows": low_rows,
                    "n_groups_supported": usable.height,
                    "n_groups_unsupported": subset.height - usable.height,
                    "undefined_reason": undefined_reason(measure, value, values, reference),
                    "fairness_definition_version": FAIRNESS_DEFINITION_VERSION,
                }
            )
    return out


__all__ = [
    "CELL_KEYS",
    "COMPARABLE_COLUMNS",
    "DisparityError",
    "extremes",
    "measures",
    "summarise",
    "undefined_reason",
]
