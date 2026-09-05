"""Is data availability distributed evenly across groups? Pure -- no filesystem, no clock.

Component 11 measured something this component has to follow up. The missingness indicator
``missing_no_code_era_canvass`` ranks **third** in importance for the logistic model and
**second** for the network -- the *absence* of a record is among the most informative signals
either model has.

That makes data availability a fairness surface in its own right, and the chain is worth
stating because each link is measured rather than assumed:

```text
group  ->  data availability  ->  feature missingness  ->  model reliance  ->  behaviour
```

This module measures the second and third links: how often each null-rule family is missing
inside each group, and whether the rows carrying that absence are over- or under-represented
in the priority set. Component 11's artifact supplies the fourth, in ``attribution.py``.

**Three things are deliberately not done here.**

Missingness features are not removed and removing them is not recommended. "We have never
inspected this place" is a true and relevant fact, and deleting the feature would not undo
the inequality in inspection history that produced it -- it would only stop the model seeing
it.

Missingness is not called unfair by definition. It is measured, and its distribution is
reported.

No causal direction is claimed. A neighbourhood with sparse inspection history and a
neighbourhood the model treats differently are the same rows; which produced which is not
answerable here, and ADR 0019's missing inspector field means it is not answerable anywhere in
this project.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from sentinel.fairness.definitions import (
    Grain,
    GroupDefinitionSpec,
    GroupStatus,
)
from sentinel.fairness.models import GroupSupport, MissingnessRow
from sentinel.modeling.definitions import (
    family_indicator_name,
    indicator_source_column,
    null_families,
)


class MissingnessError(ValueError):
    """A missingness measurement could not be computed over the rows it was handed."""


def _rate(value: object) -> float:
    """Coerce a polars aggregate to a float.

    ``Series.mean()`` is typed as a union of every dtype polars can hold, so a bare
    ``float(...)`` does not type-check under strict mode. Narrowing here keeps the conversion
    in one place rather than scattering ignores through the module.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise MissingnessError(f"expected a numeric aggregate, got {type(value).__name__}")


def indicators(available: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """The four null-rule family indicators and the column whose null mask defines each.

    Derived from ``modeling.definitions`` rather than hardcoded, so the set stays the four
    families Component 6 declared. Component 6 measured the masks *within* a family being
    byte-identical on all 57,727 rows, which is why one source column per family is a
    complete description rather than a sample of one.
    """
    out: list[tuple[str, str]] = []
    for rule in null_families():
        source = indicator_source_column(rule)
        if source in available:
            out.append((family_indicator_name(rule), source))
    if not out:
        raise MissingnessError(
            "no null-rule source column is present in the frame; the missingness audit needs "
            "Component 4's nullable columns to derive the indicators from"
        )
    return tuple(out)


def measure(
    frame: pl.DataFrame,
    spec: GroupDefinitionSpec,
    support: Mapping[str, GroupSupport],
    selected_ids: Sequence[str],
    *,
    grain: Grain,
    fold_set: str,
    fold_id: str,
    k_name: str,
) -> list[MissingnessRow]:
    """Missingness rates per (group, indicator), and the same rate inside the top k.

    ``selected_ids`` is the city-wide priority set at one cutoff. The second rate answers a
    question the first cannot: rows whose history is absent may be *more* likely to be
    prioritised, because the models lean on that absence -- so a group with sparse records
    could be systematically pushed up or down the ranking by the same mechanism.

    A row is emitted for every observed group and every indicator, unsupported groups
    included, with counts always real and ``group_status`` gating the reading.
    """
    column = spec.source_column
    if column not in frame.columns:
        raise MissingnessError(f"frame has no column {column!r}")

    pairs = indicators(frame.columns)
    chosen = set(selected_ids)
    selected_frame = frame.filter(pl.col("target_inspection_id").is_in(list(chosen)))

    rows: list[MissingnessRow] = []
    for indicator, source in pairs:
        overall_rate = _rate(frame.get_column(source).is_null().mean())
        grouped = (
            frame.group_by(column)
            .agg(
                pl.len().alias("n_rows"),
                pl.col(source).is_null().sum().alias("n_missing"),
            )
            .sort(column)
        )
        in_top_k = (
            selected_frame.group_by(column)
            .agg(
                pl.len().alias("k_rows"),
                pl.col(source).is_null().sum().alias("k_missing"),
            )
            .sort(column)
        )
        merged = grouped.join(in_top_k, on=column, how="left")

        for record in merged.to_dicts():
            value = str(record[column])
            n_rows = int(record["n_rows"])
            n_missing = int(record["n_missing"])
            k_rows = int(record["k_rows"] or 0)
            k_missing = int(record["k_missing"] or 0)
            entry = support.get(value)
            rate = n_missing / n_rows if n_rows else 0.0
            rows.append(
                MissingnessRow(
                    group_definition=spec.name,
                    group_value=value,
                    grain=grain.value,
                    fold_set=fold_set,
                    fold_id=fold_id,
                    indicator=indicator,
                    source_column=source,
                    n_rows=n_rows,
                    n_missing=n_missing,
                    missing_rate=rate,
                    overall_missing_rate=overall_rate,
                    deviation=rate - overall_rate,
                    # None rather than 0.0 when the group placed nobody in the top k: "none
                    # of the zero rows we selected were missing history" is not a rate of
                    # zero, and 0.0 would read as this group's prioritised rows being
                    # unusually well documented.
                    missing_rate_in_top_k=(k_missing / k_rows) if k_rows else None,
                    k_name=k_name,
                    group_status=(
                        entry.ranking_status if entry else GroupStatus.INSUFFICIENT_SUPPORT
                    ),
                )
            )
    return rows


def spread_by_indicator(rows: Sequence[MissingnessRow]) -> dict[str, tuple[float, str, str]]:
    """Per indicator, the supported groups' missingness spread and the two extremes.

    Supported groups only, because a 100% missingness rate over eleven rows is a fact about
    eleven rows. The extremes are named so a reader can go and look at them rather than
    taking the spread on trust.
    """
    by_indicator: dict[str, list[MissingnessRow]] = {}
    for row in rows:
        if row.group_status is GroupStatus.SUPPORTED:
            by_indicator.setdefault(row.indicator, []).append(row)

    out: dict[str, tuple[float, str, str]] = {}
    for indicator, records in sorted(by_indicator.items()):
        if len(records) < 2:
            continue
        high = max(records, key=lambda r: r.missing_rate)
        low = min(records, key=lambda r: r.missing_rate)
        out[indicator] = (
            high.missing_rate - low.missing_rate,
            high.group_value,
            low.group_value,
        )
    return out


__all__ = [
    "MissingnessError",
    "indicators",
    "measure",
    "spread_by_indicator",
]
