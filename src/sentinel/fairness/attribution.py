"""Do the models reason differently about different groups? Pure -- no filesystem, no clock.

Component 11 found that four models scoring within 0.0156 NDE of one another rank their
features at a Spearman rho as low as 0.4351. This module asks the same question one level
down: within *one* model, does the feature profile differ across neighbourhoods?

**The existing artifact is grouped, never regenerated.** Component 11 explained 300 rows per
(model, fold) under a frozen protocol with no ``--seed`` flag, and HANDOFF is explicit that
re-running ``sentinel explain`` with a different ``--sample-size`` to check a finding is
forbidden -- it would change the rows every published Component 11 number rests on. So the
question here is what the sample that exists can answer.

Measured: pooled over the quarterly folds, the median (model, community area) cell holds
**40** explained rows and 56 of 312 cells reach 100. That supports a comparison of *global
profiles* -- a 30-feature mean-|SHAP| ranking, whose global statistic Component 11 measured
converging far faster than any individual value (rank rho 0.9964 at 8 rounds against a
64-round reference). It supports nothing per-row, and nothing per fold.

**Three limits, and they are not boilerplate.**

An attribution is not a quality measure (ADR 0030). A model can lean hard on a feature that
is misleading it, which Component 6 measured happening under distribution shift.

A difference between two groups' profiles is a difference in model *reliance*. It is not
evidence of discrimination and it is not causal.

The network's per-row values are approximate (``is_exact = false``), which is carried through
onto every row here rather than dropped in aggregation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import polars as pl

from sentinel.calibration.metrics import spearman
from sentinel.fairness.definitions import (
    ATTRIBUTION_MIN_ROWS,
    GroupDefinitionSpec,
    GroupStatus,
)
from sentinel.fairness.models import AttributionProfileRow

#: Columns read from ``explanation_values_<stamp>.parquet``. Read by name from the artifact
#: rather than by importing ``sentinel.explain``, which is what ADR 0028 designed the
#: denormalised long grain to allow.
EXPLANATION_COLUMNS: tuple[str, ...] = (
    "model_name",
    "fold_set",
    "fold_id",
    "target_inspection_id",
    "feature_name",
    "shap_value",
    "is_exact",
)


#: The order rows are put into before any mean is taken over them.
#:
#: Necessary and, on its own, **not sufficient** -- see :func:`_profile_means`. Polars
#: aggregates a group in parallel, so two runs produced ``mean_abs_shap`` values differing at
#: 1.8e-15 even over a sorted frame. Sorting fixes which rows are in which group; it does not
#: fix the order they are added in.
AGGREGATION_SORT: tuple[str, ...] = (
    "model_name",
    "target_inspection_id",
    "feature_name",
)


class AttributionError(ValueError):
    """A group attribution profile could not be assembled from the rows it was handed."""


def _profile_means(frame: pl.DataFrame) -> pl.DataFrame:
    """Mean and mean-absolute SHAP per feature, summed in a fixed order.

    ``math.fsum`` rather than ``pl.col(...).mean()``, and the reason is measured. Polars sums a
    group in parallel and in whatever order the rows reach a thread, so two runs over identical
    inputs produced means differing at **1.8e-15**. Every rank, every Spearman correlation and
    every count was identical, so nothing a reader would act on moved -- but this project's
    standard for "unchanged" is bit-identical, and **a table that is only nearly reproducible
    is a table whose two-run checksum comparison has stopped being a detector.**

    ``fsum`` is exactly-rounded, so the result does not depend on summation order at all. That
    is a stronger guarantee than sorting: sorting fixes the order on this machine, and ``fsum``
    makes the order irrelevant on every machine.

    It costs a Python loop over the explained rows -- roughly a second on the production
    artifact, against a run that takes 145. Components 6 and 7 each paid a much larger price
    for bit-reproducibility (single-threaded fits throughout) for the same reason.
    """
    totals: dict[str, list[float]] = {}
    absolutes: dict[str, list[float]] = {}
    for feature, value in zip(
        frame.get_column("feature_name").to_list(),
        frame.get_column("shap_value").to_list(),
        strict=True,
    ):
        name = str(feature)
        number = float(value)
        totals.setdefault(name, []).append(number)
        absolutes.setdefault(name, []).append(abs(number))

    names = sorted(totals)
    return pl.DataFrame(
        {
            "feature_name": names,
            "mean_abs_shap": [math.fsum(absolutes[n]) / len(absolutes[n]) for n in names],
            "mean_shap": [math.fsum(totals[n]) / len(totals[n]) for n in names],
        }
    )


def _ranked(profile: pl.DataFrame) -> pl.DataFrame:
    """Add a descending rank on ``mean_abs_shap``, ties broken by feature name.

    Ties broken deterministically rather than left to frame order: two runs over the same
    values must produce the same ranking, and a rank that depends on row order would make the
    rank-delta column meaningless.
    """
    return profile.sort(["mean_abs_shap", "feature_name"], descending=[True, False]).with_columns(
        pl.int_range(1, pl.len() + 1).alias("rank")
    )


def profiles(
    values: pl.DataFrame,
    group_lookup: pl.DataFrame,
    spec: GroupDefinitionSpec,
    *,
    fold_set: str,
    min_rows: int = ATTRIBUTION_MIN_ROWS,
) -> list[AttributionProfileRow]:
    """Mean absolute attribution per feature, per group, for every model, pooled by fold set.

    ``group_lookup`` maps ``target_inspection_id`` to the group column. The join is the same
    one every other module uses, on the same key, so an explained row and an audited row are
    the same row by construction rather than by a fuzzy match.

    Emits rows only for groups clearing ``min_rows``, and returns the count of those that did
    not through the caller's support table rather than by inventing null profiles: a
    30-feature table of nulls per unsupported group would be 30 rows of noise each, and the
    support table already records which groups were excluded and why.
    """
    missing = [column for column in EXPLANATION_COLUMNS if column not in values.columns]
    if missing:
        raise AttributionError(
            f"explanation artifact is missing {', '.join(missing)}. Component 12 reads "
            "Component 11's long grain by column name and does not import sentinel.explain."
        )
    column = spec.source_column
    if column not in group_lookup.columns:
        raise AttributionError(f"group lookup has no column {column!r}")

    scoped = values.filter(pl.col("fold_set") == fold_set)
    if scoped.is_empty():
        return []
    joined = scoped.join(
        group_lookup.select("target_inspection_id", column),
        on="target_inspection_id",
        how="inner",
    ).sort(AGGREGATION_SORT)
    if joined.is_empty():
        return []

    rows: list[AttributionProfileRow] = []
    for model_name in sorted(joined.get_column("model_name").unique().to_list()):
        per_model = joined.filter(pl.col("model_name") == model_name)
        is_exact = bool(per_model.get_column("is_exact").min())

        overall = _ranked(_profile_means(per_model))
        overall_rank: Mapping[str, int] = {
            str(r["feature_name"]): int(r["rank"]) for r in overall.to_dicts()
        }

        counts = per_model.group_by(column).agg(
            pl.col("target_inspection_id").n_unique().alias("n_rows")
        )
        eligible = counts.filter(pl.col("n_rows") >= min_rows).sort(column)

        for record in eligible.to_dicts():
            value = str(record[column])
            n_rows = int(record["n_rows"])
            subset = per_model.filter(pl.col(column) == value)
            profile = _ranked(_profile_means(subset))
            group_ranks = [
                float(overall_rank.get(str(r["feature_name"]), 0)) for r in profile.to_dicts()
            ]
            own_ranks = [float(r["rank"]) for r in profile.to_dicts()]
            rho = spearman(own_ranks, group_ranks) if len(own_ranks) > 1 else None

            for r in profile.to_dicts():
                feature = str(r["feature_name"])
                rank = int(r["rank"])
                overall_position = overall_rank.get(feature, 0)
                rows.append(
                    AttributionProfileRow(
                        model_name=model_name,
                        group_definition=spec.name,
                        group_value=value,
                        fold_set=fold_set,
                        feature_name=feature,
                        mean_abs_shap=float(r["mean_abs_shap"]),
                        mean_shap=float(r["mean_shap"]),
                        rank=rank,
                        overall_rank=overall_position,
                        rank_delta=overall_position - rank if overall_position else 0,
                        n_rows=n_rows,
                        profile_spearman=rho,
                        is_exact=is_exact,
                        group_status=GroupStatus.SUPPORTED,
                    )
                )
    return rows


def divergent_groups(
    rows: Sequence[AttributionProfileRow],
    *,
    limit: int = 10,
) -> list[tuple[str, str, float, int]]:
    """The (model, group) pairs whose profile agrees least with the model's overall profile.

    Sorted ascending by Spearman rho, so the least-agreeing come first. This is descriptive
    and is the whole of what section 13 of the brief can support here: it says the model's
    reliance on features differs most for these populations, and it says nothing about why,
    about whether that is wrong, or about what would change if it stopped.
    """
    seen: dict[tuple[str, str], tuple[float, int]] = {}
    for row in rows:
        if row.profile_spearman is None:
            continue
        seen.setdefault((row.model_name, row.group_value), (row.profile_spearman, row.n_rows))
    ordered = sorted(seen.items(), key=lambda item: (item[1][0], item[0]))
    return [(model, group, rho, n_rows) for (model, group), (rho, n_rows) in ordered[:limit]]


__all__ = [
    "AGGREGATION_SORT",
    "EXPLANATION_COLUMNS",
    "AttributionError",
    "divergent_groups",
    "profiles",
]
