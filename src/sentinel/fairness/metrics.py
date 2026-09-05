"""Group-conditional metrics. Pure -- no filesystem, no clock.

**Almost nothing here is a new metric implementation, and that is the point.** ROC-AUC,
PR-AUC, Brier, log loss, ECE, MCE and precision/recall/lift@k live in
``evaluation.metrics``; NDE lives in ``evaluation.simulate``; calibration slope and the
bootstrap live in ``calibration.metrics``. This module imports every one of them and applies
them to a subset of rows.

Component 5 is documented as the only producer of the headline metrics, and a fairness
component that re-derived an ROC-AUC would create a second implementation that could drift
from the first -- so that a group's ROC-AUC and the fold's ROC-AUC would eventually disagree
for reasons no reader could diagnose. Two authoritative answers is the failure ADR 0024 and
ADR 0028 each created a layer to avoid; here it would be worse, because the two numbers would
sit in the same comparison.

What *is* new is the small set of statistics that only exist once a group is involved:
positive-outcome capture at a global cutoff, the selection-rate ratio, and the disparity
measures. Each is a few lines of arithmetic, each is verified against a hand-computed example
in the test suite, and each is defined here so there is one place to read the definition.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

from sentinel.calibration.metrics import calibration_slope_intercept
from sentinel.evaluation import metrics as canonical
from sentinel.evaluation import simulate
from sentinel.fairness.definitions import GROUP_CALIBRATION_BINS, MetricKind

#: Ranking metrics computed per group. Every one is base-rate dependent except ``roc_auc``
#: and ``nde``, so the artifact carries each group's base rate on its support row and the
#: findings document reports them together. MEMORY invariant 44.
RANKING_METRICS: tuple[str, ...] = ("roc_auc", "pr_auc", "nde")

#: Probability metrics computed per group, gated by the stricter calibration floor.
PROBABILITY_METRICS: tuple[str, ...] = (
    "brier",
    "log_loss",
    "ece",
    "mce",
    "calibration_slope",
    "calibration_intercept",
)

#: Metrics at a top-k cutoff. ``precision_at_k`` and ``recall_at_k`` come from Component 5;
#: the rest are the confusion-matrix view of the same cutoff, reported only because section 7
#: of the brief asks for error behaviour and refused at any probability threshold.
THRESHOLD_METRICS: tuple[str, ...] = (
    "precision_at_k",
    "recall_at_k",
    "lift_at_k",
    "true_positive_rate",
    "false_positive_rate",
    "false_discovery_rate",
)

#: Whether a lower value of a metric is better, for the calibration before/after comparison.
#: ``calibration_slope`` is in neither direction -- 1.0 is perfect and both sides are worse --
#: so it is handled separately rather than forced into a boolean.
LOWER_IS_BETTER: dict[str, bool] = {
    "brier": True,
    "log_loss": True,
    "ece": True,
    "mce": True,
}


class GroupMetricError(ValueError):
    """A group metric could not be computed and could not honestly return a number."""


def kind_of(metric: str) -> MetricKind:
    """Which family a metric belongs to, and therefore which support floor gates it."""
    if metric in PROBABILITY_METRICS:
        return MetricKind.PROBABILITY
    if metric in THRESHOLD_METRICS:
        return MetricKind.THRESHOLD_AUDIT
    if metric in RANKING_METRICS:
        return MetricKind.RANKING
    raise GroupMetricError(f"unknown metric {metric!r}")


def ranking_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    dates: Sequence[date],
    ids: Sequence[str],
) -> dict[str, float | None]:
    """ROC-AUC, PR-AUC and NDE over one group's rows.

    NDE is built from Component 5's own pieces rather than reimplemented: a window over this
    group's rows, the model's ordering of them, the discovery curve, and the analytic
    normalisation. It answers "if only this group's inspections were reordered, how
    efficiently would its positives surface" -- a within-group efficiency, which is the
    honest reading of a within-group ranking metric and is deliberately *not* the same
    question as the group's share of a city-wide top-k. That second question is
    ``priority.py``'s, and keeping the two apart is why a group can rank well and still be
    prioritised late.

    ``None`` propagates from the canonical implementations wherever a metric is undefined --
    a single-class group has no ROC-AUC and no NDE, and returning 0.5 or 0.0 to fill a column
    would be a fabrication.
    """
    if not (len(labels) == len(scores) == len(dates) == len(ids)):
        raise GroupMetricError("labels, scores, dates and ids must be the same length")
    if not labels:
        return dict.fromkeys(RANKING_METRICS)

    window = simulate.build_window(ids=ids, labels=labels, dates=dates)
    order = simulate.model_order(window, _reorder(scores, ids, window.ids))
    cumulative = simulate.discovery_curve(window, order)
    area = simulate.normalized_area(cumulative, n=window.n, positives=window.positives)
    return {
        "roc_auc": canonical.roc_auc(labels, scores),
        "pr_auc": canonical.pr_auc(labels, scores),
        "nde": simulate.normalized_discovery_efficiency(
            area, n=window.n, positives=window.positives
        ),
    }


def _reorder(scores: Sequence[float], ids: Sequence[str], window_ids: Sequence[str]) -> list[float]:
    """Scores rearranged into the window's canonical order.

    ``build_window`` sorts by ``(date, id)``, so the caller's score order is no longer the
    window's. Handing the unsorted list to ``model_order`` would attach every score to the
    wrong row -- and produce a perfectly plausible NDE, because a permutation of scores is
    still a valid ranking. This is the group-level analogue of the name-recovery trap
    Component 11 measured, and it is why the reordering is explicit rather than assumed.
    """
    by_id = dict(zip(ids, scores, strict=True))
    return [by_id[i] for i in window_ids]


def probability_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> dict[str, float | None]:
    """Brier, log loss, ECE, MCE and the calibration slope over one group's rows.

    The bin count is Component 5's, imported rather than restated. Reducing it would let more
    groups clear the floor and would make every resulting ECE incomparable with Component 9's
    global figure -- which is the exact comparison section 18 of the brief asks for, so the
    looser threshold would destroy the thing it was loosened to measure.
    """
    if len(labels) != len(probabilities):
        raise GroupMetricError("labels and probabilities must be the same length")
    if not labels:
        return dict.fromkeys(PROBABILITY_METRICS)

    slope = calibration_slope_intercept(labels, probabilities)
    return {
        "brier": canonical.brier(labels, probabilities),
        "log_loss": canonical.log_loss(labels, probabilities),
        "ece": canonical.ece(labels, probabilities, n_bins=GROUP_CALIBRATION_BINS),
        "mce": canonical.mce(labels, probabilities, n_bins=GROUP_CALIBRATION_BINS),
        "calibration_slope": slope.slope,
        "calibration_intercept": slope.intercept,
    }


def threshold_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    ids: Sequence[str],
    *,
    k: int,
) -> dict[str, float | None]:
    """Error behaviour at one descriptive top-k cutoff, within a group.

    **This is a descriptive threshold audit and not a deployment policy.** Component 13 owns
    decision policy. The cutoff is a rank position derived from real inspection capacity, not
    a probability threshold -- a cutoff at p = 0.5 would be a number this project has never
    derived from anything, and per-group error rates reported at one would read as a
    recommendation.

    Precision, recall and lift come from Component 5. The three rates below are the
    confusion-matrix view of the same selection: they are new arithmetic rather than a new
    metric, and each is verified against a hand-computed example.
    """
    if not labels or k < 1:
        return dict.fromkeys(THRESHOLD_METRICS)

    chosen = set(canonical.top_k_indices(scores, ids, k))
    true_positive = sum(1 for i, y in enumerate(labels) if i in chosen and y == 1)
    false_positive = sum(1 for i, y in enumerate(labels) if i in chosen and y == 0)
    positives = sum(labels)
    negatives = len(labels) - positives
    selected = len(chosen)

    return {
        "precision_at_k": canonical.precision_at_k(labels, scores, ids, k),
        "recall_at_k": canonical.recall_at_k(labels, scores, ids, k),
        "lift_at_k": canonical.lift_at_k(labels, scores, ids, k),
        # None rather than 0.0 on a zero denominator throughout: a group with no positives
        # has no true-positive rate, and 0.0 would read as "it caught none of them".
        "true_positive_rate": (true_positive / positives) if positives else None,
        "false_positive_rate": (false_positive / negatives) if negatives else None,
        "false_discovery_rate": (false_positive / selected) if selected else None,
    }


def capture_rate(group_positives: int, captured_positives: int) -> float | None:
    """Share of a group's actual positive outcomes that a selected set contained.

    The opportunity metric, and the one that separates *being prioritised* from *being
    prioritised usefully*. A group can be over-represented in the top k while the ranking
    finds a smaller share of its violations than average, and only reporting both makes that
    visible.

    Note what the denominator is: the group's own positives, not the window's. So this is not
    ``recall_at_k`` restricted to a group -- ``recall_at_k`` selects its top k *within* the
    rows it is handed, whereas capture is measured against a cutoff taken over every audited
    row. That difference is the whole point: the selection is city-wide and competitive, and
    a group's capture rate is what that competition left it.

    ``None`` when the group has no positives -- never 0.0, which would read as a real
    measurement of total failure rather than as the absence of anything to capture.
    """
    if group_positives < 0 or captured_positives < 0:
        raise GroupMetricError("positive counts cannot be negative")
    if captured_positives > group_positives:
        raise GroupMetricError(
            f"captured {captured_positives} positives from a group holding {group_positives}"
        )
    if group_positives == 0:
        return None
    return captured_positives / group_positives


def selection_rate_ratio(
    group_selected: int,
    group_rows: int,
    overall_selected: int,
    overall_rows: int,
) -> float | None:
    """A group's selection rate divided by the overall selection rate.

    1.0 means selected exactly in proportion to its presence in the population. Above 1.0
    means over-represented in the priority set.

    **This is not a target and parity is not automatically desirable.** Outcome rates differ
    from 0.220 to 0.566 across supported community areas, so a working risk model is expected
    to select at different rates -- equal selection would require ignoring a measured
    difference in outcomes. The ratio is reported to make the trade-off visible, not to be
    driven to one.

    ``None`` rather than infinity on a zero denominator, so a reader can distinguish "no rows
    to select from" from "selected at an enormous rate".
    """
    if group_rows <= 0 or overall_rows <= 0:
        return None
    overall_rate = overall_selected / overall_rows
    if overall_rate == 0.0:
        return None
    return (group_selected / group_rows) / overall_rate


def spread(values: Sequence[float]) -> float | None:
    """``max - min``. ``None`` on fewer than two values, where a spread is not a comparison."""
    if len(values) < 2:
        return None
    return max(values) - min(values)


def ratio(values: Sequence[float]) -> float | None:
    """``max / min``.

    ``None`` when the minimum is zero or negative rather than ``inf`` or a large number. A
    zero-denominator ratio is undefined, and a table showing ``inf`` invites a reader to
    treat the group as infinitely worse off when what happened is that a denominator vanished.
    """
    if len(values) < 2:
        return None
    low = min(values)
    if low <= 0.0:
        return None
    return max(values) / low


def max_deviation(values: Sequence[float], reference: float | None) -> float | None:
    """The largest absolute distance from the pooled population value.

    The reference is the pooled value over the same rows, never a nominated group. Choosing a
    reference group after seeing the results would be a conclusion wearing a criterion's
    clothes; choosing one beforehand would still be choosing which neighbourhood counts as
    normal.
    """
    if not values or reference is None:
        return None
    return max(abs(value - reference) for value in values)


def weighted_sd(values: Sequence[float], weights: Sequence[int]) -> float | None:
    """Rows-weighted standard deviation across supported groups.

    Weighted, because an unweighted SD over 51 community areas gives a 200-row group and a
    2,600-row group equal say in how uneven the city is. Both views are defensible and this
    one is chosen and stated rather than defaulted into.
    """
    if len(values) != len(weights):
        raise GroupMetricError("values and weights must be the same length")
    if len(values) < 2:
        return None
    total = sum(weights)
    if total <= 0:
        return None
    mean = sum(v * w for v, w in zip(values, weights, strict=True)) / total
    variance = sum(w * (v - mean) ** 2 for v, w in zip(values, weights, strict=True)) / total
    return math.sqrt(variance)


def improved(metric: str, base: float | None, calibrated: float | None) -> bool | None:
    """Did calibration move this group's metric in the better direction?

    ``None`` when either side is missing -- never ``False``. "We could not tell" and "it got
    worse" are different answers, and a fairness audit that reported the first as the second
    would manufacture a finding.

    ``calibration_slope`` is measured by distance from 1.0 rather than by direction, because
    a slope of 0.6 and a slope of 1.4 are both miscalibrated and no inequality between the
    raw numbers says which is better.
    """
    if base is None or calibrated is None:
        return None
    if metric == "calibration_slope":
        return abs(calibrated - 1.0) < abs(base - 1.0)
    if metric not in LOWER_IS_BETTER:
        return None
    return calibrated < base if LOWER_IS_BETTER[metric] else calibrated > base


__all__ = [
    "LOWER_IS_BETTER",
    "PROBABILITY_METRICS",
    "RANKING_METRICS",
    "THRESHOLD_METRICS",
    "GroupMetricError",
    "capture_rate",
    "improved",
    "kind_of",
    "max_deviation",
    "probability_metrics",
    "ranking_metrics",
    "ratio",
    "selection_rate_ratio",
    "spread",
    "threshold_metrics",
    "weighted_sd",
]
