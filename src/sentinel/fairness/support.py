"""Counting what each group has, and deciding whether it is enough. Pure -- no I/O.

This module runs before any metric, and that order is the component's central discipline: the
question "how do the groups compare?" is only answerable after "is there enough data to
compare them?", and answering them the other way round is how a fairness audit ends up
reporting a dramatic ratio from a group of twelve rows.

Two things it does that a simpler version would not.

**It emits a record for every observed group, not for the ones that qualified.** A group
below the floor gets a row with its true counts, a status of ``insufficient_support`` and a
reason naming the floor it missed. That is what makes "we measured 51 of 78 community areas"
sayable, and it is what stops "equal performance across groups" from resting silently on the
groups that were dropped.

**It decides ranking and calibration support separately.** They have different floors for an
arithmetic reason -- a binned calibration statistic spends its rows across 15 bins -- so a
group can legitimately support an ROC-AUC and not an ECE. Collapsing them into one boolean
would either suppress metrics that were computable or publish ones that were not.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from sentinel.fairness.definitions import (
    CALIBRATION_MIN_ROWS,
    GROUP_CALIBRATION_BINS,
    SUPPORT_MIN_NEGATIVE,
    SUPPORT_MIN_POSITIVE,
    SUPPORT_MIN_ROWS,
    Grain,
    GroupDefinitionSpec,
    GroupStatus,
    MetricKind,
)
from sentinel.fairness.models import GroupSupport


def _reason(n_rows: int, n_positive: int, n_negative: int) -> str:
    """Why a group missed a floor, naming the floor and the count that missed it.

    Every clause is included, not just the first. A group that is both too small and
    single-class has two problems, and reporting one would send a reader looking for more
    rows when what is missing is a second class.
    """
    clauses: list[str] = []
    if n_rows < SUPPORT_MIN_ROWS:
        clauses.append(f"{n_rows} rows < {SUPPORT_MIN_ROWS}")
    if n_positive < SUPPORT_MIN_POSITIVE:
        clauses.append(f"{n_positive} positives < {SUPPORT_MIN_POSITIVE}")
    if n_negative < SUPPORT_MIN_NEGATIVE:
        clauses.append(f"{n_negative} negatives < {SUPPORT_MIN_NEGATIVE}")
    if n_rows < CALIBRATION_MIN_ROWS:
        clauses.append(
            f"{n_rows} rows < {CALIBRATION_MIN_ROWS} for a {GROUP_CALIBRATION_BINS}-bin "
            "calibration statistic"
        )
    return "; ".join(clauses)


def classify(n_rows: int, n_positive: int, n_negative: int) -> tuple[GroupStatus, GroupStatus, str]:
    """Ranking status, calibration status, and the reason a floor was missed.

    Calibration support implies ranking support: ``CALIBRATION_MIN_ROWS`` is at least
    ``SUPPORT_MIN_ROWS`` and the registry guard enforces it, because a statistic that spends
    its rows across bins cannot need fewer of them than one that does not.
    """
    class_ok = n_positive >= SUPPORT_MIN_POSITIVE and n_negative >= SUPPORT_MIN_NEGATIVE
    ranking = (
        GroupStatus.SUPPORTED
        if n_rows >= SUPPORT_MIN_ROWS and class_ok
        else GroupStatus.INSUFFICIENT_SUPPORT
    )
    calibration = (
        GroupStatus.SUPPORTED
        if n_rows >= CALIBRATION_MIN_ROWS and class_ok
        else GroupStatus.INSUFFICIENT_SUPPORT
    )
    reason = ""
    if (
        ranking is GroupStatus.INSUFFICIENT_SUPPORT
        or calibration is GroupStatus.INSUFFICIENT_SUPPORT
    ):
        reason = _reason(n_rows, n_positive, n_negative)
    return ranking, calibration, reason


def status_for(support: GroupSupport, kind: MetricKind) -> GroupStatus:
    """Which of a support record's two statuses gates one metric family."""
    if kind is MetricKind.PROBABILITY:
        return support.calibration_status
    return support.ranking_status


def measure(
    frame: pl.DataFrame,
    spec: GroupDefinitionSpec,
    *,
    grain: Grain,
    fold_set: str,
    fold_id: str = "",
) -> list[GroupSupport]:
    """Support records for every group observed in ``frame``, sorted by group value.

    ``frame`` must already be restricted to the rows this grain covers and to a single
    model -- support is model-independent, but the caller holds the frame that proves every
    model scored the same rows, and re-deriving that here would be a second implementation of
    a fact Component 12 already checks.

    Sorted by group value rather than by size, so two runs over the same data emit the rows
    in the same order and the artifact is byte-comparable.
    """
    column = spec.source_column
    if column not in frame.columns:
        raise KeyError(f"frame has no column {column!r}")

    total = frame.height
    grouped = (
        frame.group_by(column)
        .agg(pl.len().alias("n_rows"), pl.col("target").sum().alias("n_positive"))
        .sort(column)
    )

    records: list[GroupSupport] = []
    for row in grouped.to_dicts():
        n_rows = int(row["n_rows"])
        n_positive = int(row["n_positive"])
        n_negative = n_rows - n_positive
        ranking, calibration, reason = classify(n_rows, n_positive, n_negative)
        records.append(
            GroupSupport(
                group_definition=spec.name,
                group_value=str(row[column]),
                grain=grain.value,
                fold_set=fold_set,
                fold_id=fold_id,
                n_rows=n_rows,
                n_positive=n_positive,
                n_negative=n_negative,
                # None rather than 0.0 on an empty group: 0.0 is a legitimate base rate and
                # a group of zero rows has none at all.
                base_rate=(n_positive / n_rows) if n_rows else None,
                representation_share=(n_rows / total) if total else 0.0,
                ranking_status=ranking,
                calibration_status=calibration,
                insufficient_reason=reason,
            )
        )
    return records


def index(records: Sequence[GroupSupport]) -> dict[tuple[str, str, str, str], GroupSupport]:
    """Support records keyed by ``(definition, value, grain, fold_id)`` for fast lookup."""
    return {(r.group_definition, r.group_value, r.grain, r.fold_id): r for r in records}


def supported_values(
    records: Sequence[GroupSupport],
    *,
    kind: MetricKind,
) -> tuple[str, ...]:
    """The group values that clear the floor for one metric family, in sorted order."""
    return tuple(
        sorted(r.group_value for r in records if status_for(r, kind) is GroupStatus.SUPPORTED)
    )


def summarise(records: Sequence[GroupSupport]) -> dict[str, int]:
    """Observed, supported and insufficient counts, for the manifest and the report.

    ``supported`` uses the ranking floor, which is the looser of the two, so this is the
    optimistic count. The calibration count is lower and the artifact carries both per row;
    a summary that reported only the stricter number would understate what was measured, and
    only the looser one would overstate it. The manifest carries this dictionary and the
    findings document quotes both.
    """
    observed = len(records)
    supported = sum(1 for r in records if r.ranking_status is GroupStatus.SUPPORTED)
    calibration = sum(1 for r in records if r.calibration_status is GroupStatus.SUPPORTED)
    return {
        "observed": observed,
        "supported_ranking": supported,
        "supported_calibration": calibration,
        "insufficient": observed - supported,
    }


__all__ = [
    "classify",
    "index",
    "measure",
    "status_for",
    "summarise",
    "supported_values",
]
