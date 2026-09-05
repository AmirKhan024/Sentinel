"""Does the group behaviour itself move over time? Pure -- no filesystem, no clock.

Component 5 measured the ranking drifting, Component 9 measured calibration drifting and
Component 11 measured the models' reasoning drifting. The question here is one level up:
**does the gap between groups change, even when the aggregate is stable?**

Two constraints shape what can honestly be said.

**The per-fold grain is thin.** The support policy means a per-fold group metric usually does
not exist -- the median (fold, community area) cell holds 16 rows, and 4 of 1,288 reach the
200-row floor. So a drift series is assembled from the folds where the disparity was actually
computable, and ``folds_measured`` reports how many that was against ``folds_total``. A series
shorter than ``DRIFT_MIN_FOLDS`` is labelled ``insufficient_folds`` rather than fitted: two
points are a line through any two numbers.

**The evaluated population moves too.** The profiler measured group shares travelling by up to
10.7 percentage points across the 17 quarterly folds, and 11 of 78 community areas being
absent from at least one fold entirely. A group whose share halved and whose measured ECE
moved has two candidate explanations, and this component can separate neither. The
representation series is therefore computed alongside and reported beside every drift claim
rather than in a footnote.

``covid_shift`` is never pooled into a quarterly series. Its single fold cannot support a
trend and is reported as a separate stress-test observation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import polars as pl

from sentinel.fairness.definitions import (
    DRIFT_MATERIAL_CHANGE,
    DRIFT_MIN_FOLDS,
    FAIRNESS_DEFINITION_VERSION,
    Grain,
)
from sentinel.fairness.models import GroupSupport

#: The trend labels. Deliberately four, with the fourth naming the common case rather than
#: hiding it inside "stable" -- a series that could not be measured is not a series that did
#: not move.
TREND_STABLE = "stable"
TREND_WIDENING = "widening"
TREND_NARROWING = "narrowing"
TREND_INSUFFICIENT = "insufficient_folds"

#: What identifies one drift series.
SERIES_KEYS: tuple[str, ...] = (
    "model_name",
    "stage",
    "group_definition",
    "fold_set",
    "metric",
    "k_name",
    "measure",
)


class DriftError(ValueError):
    """A drift series could not be assembled from the rows it was handed."""


def sample_sd(values: Sequence[float]) -> float | None:
    """Sample standard deviation across folds.

    **Not a confidence interval, and the artifact says so.** The folds are temporally
    overlapping and share establishments -- the same premises appears in many test windows on
    a 358-day median canvass cycle -- so this is a fold-to-fold spread, not the standard error
    of independent samples. Component 5 recorded the same caveat about its own SD and it
    applies with more force here, where each fold's value is already a summary over groups.
    """
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def trend(first: float | None, last: float | None, folds: int) -> tuple[str, float | None]:
    """The trend label and the relative change behind it.

    ``DRIFT_MATERIAL_CHANGE`` was frozen before any disparity series existed, for the reason
    ADR 0030 froze ``RANK_DRIFT_THRESHOLD`` in advance: a criterion chosen after seeing the
    numbers is a conclusion with a criterion bolted on.
    """
    if folds < DRIFT_MIN_FOLDS or first is None or last is None:
        return TREND_INSUFFICIENT, None
    if first == 0.0:
        # A spread that started at exactly zero has no relative change to report. Labelled
        # rather than divided by zero, and reported with a null change rather than infinity.
        return (TREND_STABLE if last == 0.0 else TREND_WIDENING), None
    change = (last - first) / abs(first)
    if change > DRIFT_MATERIAL_CHANGE:
        return TREND_WIDENING, change
    if change < -DRIFT_MATERIAL_CHANGE:
        return TREND_NARROWING, change
    return TREND_STABLE, change


def series(disparity: pl.DataFrame) -> list[dict[str, object]]:
    """One drift row per comparable series, from the per-fold disparity rows.

    Only ``grain = fold`` rows contribute: a pooled row is one number and has no series. Folds
    whose value is null -- because too few groups cleared the support floor that quarter --
    are excluded from the statistics and still counted in ``folds_total``, so each record
    states both what was measured and what was available.
    """
    if disparity.is_empty():
        return []
    missing = [c for c in (*SERIES_KEYS, "grain", "fold_id", "value") if c not in disparity.columns]
    if missing:
        raise DriftError(f"disparity frame is missing {', '.join(missing)}")

    per_fold = disparity.filter(pl.col("grain") == Grain.FOLD.value)
    if per_fold.is_empty():
        return []

    out: list[dict[str, object]] = []
    keys = per_fold.select(SERIES_KEYS).unique().sort(list(SERIES_KEYS))
    for key in keys.to_dicts():
        subset = per_fold
        for name in SERIES_KEYS:
            subset = subset.filter(pl.col(name) == key[name])
        subset = subset.sort("fold_id")
        measured = subset.drop_nulls("value")
        values = [float(v) for v in measured.get_column("value").to_list()]
        fold_ids = [str(v) for v in measured.get_column("fold_id").to_list()]
        first = values[0] if values else None
        last = values[-1] if values else None
        label, change = trend(first, last, len(values))
        out.append(
            {
                **{name: str(key[name]) for name in SERIES_KEYS},
                "folds_measured": len(values),
                "folds_total": subset.height,
                "mean_spread": (sum(values) / len(values)) if values else None,
                "sd_spread": sample_sd(values),
                "min_spread": min(values) if values else None,
                "max_spread": max(values) if values else None,
                "first_fold_id": fold_ids[0] if fold_ids else "",
                "first_spread": first,
                "last_fold_id": fold_ids[-1] if fold_ids else "",
                "last_spread": last,
                "relative_change": change,
                "trend": label,
                "fairness_definition_version": FAIRNESS_DEFINITION_VERSION,
            }
        )
    return out


def representation_travel(
    records: Sequence[GroupSupport],
) -> dict[str, tuple[float, float, float, int]]:
    """Per group: minimum share, maximum share, travel, and folds present.

    The context every drift claim has to be read against. A group's metric moving while its
    share of the evaluated population also moved is two effects this component cannot
    separate, and reporting the first without the second would attribute one to the other for
    free.
    """
    by_group: dict[str, list[float]] = {}
    for record in records:
        if record.grain != Grain.FOLD.value:
            continue
        by_group.setdefault(record.group_value, []).append(record.representation_share)
    return {
        value: (min(shares), max(shares), max(shares) - min(shares), len(shares))
        for value, shares in sorted(by_group.items())
    }


def advisory_lines(
    drift_rows: Sequence[dict[str, object]],
    travel: dict[str, tuple[float, float, float, int]],
    *,
    representation_threshold: float,
    limit: int = 10,
) -> list[str]:
    """Notes on disparities that moved, and on populations that moved underneath them.

    Advisories, never failures. A disparity that widened is evidence for Component 13, not a
    defect in this code, and a build that went red on it would put pressure on the measurement
    rather than on the inequality.
    """
    notes: list[str] = []
    for row in drift_rows:
        label = row.get("trend")
        change = row.get("relative_change")
        if label not in (TREND_WIDENING, TREND_NARROWING) or not isinstance(change, float):
            continue
        notes.append(
            f"{row['model_name']} [{row['stage']}] {row['group_definition']} "
            f"{row['metric']} {row['measure']}: {label} {change:+.1%} from "
            f"{row['first_fold_id']} to {row['last_fold_id']} over "
            f"{row['folds_measured']} of {row['folds_total']} folds"
        )
    movers = [
        (value, stats) for value, stats in travel.items() if stats[2] > representation_threshold
    ]
    for value, stats in sorted(movers, key=lambda item: -item[1][2])[:limit]:
        notes.append(
            f"representation: group {value} share travelled {stats[2]:.4f} "
            f"({stats[0]:.4f} to {stats[1]:.4f}) across {stats[3]} fold(s) -- a metric moving "
            "for this group has two candidate explanations"
        )
    return notes


def covid_is_separate(fold_set: str) -> bool:
    """Whether a fold set is the distribution-shift fold, which is never pooled.

    A named predicate rather than a string comparison at each call site, because "do not
    average covid_shift into the quarterly mean" has been an invariant since Component 5 and
    has been violated by accident in exactly the way a bare literal invites.
    """
    return fold_set == "covid_shift"


__all__ = [
    "SERIES_KEYS",
    "TREND_INSUFFICIENT",
    "TREND_NARROWING",
    "TREND_STABLE",
    "TREND_WIDENING",
    "DriftError",
    "advisory_lines",
    "covid_is_separate",
    "representation_travel",
    "sample_sd",
    "series",
    "trend",
]
