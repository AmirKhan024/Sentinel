"""Statistical and ranking metrics, implemented rather than imported.

No new runtime dependency is added for this. The project introduces a
technology only when a component genuinely requires it, and Component 6 is
where scikit-learn belongs. Every function here is instead cross-checked
against scikit-learn **in the test suite**, which is added to the dev group
only -- so the maths is independently verified without the wheel growing.

The split that matters is at the bottom of the file:

``RANKING_METRICS``      need only an *order*. Every baseline can be scored.
``PROBABILITY_METRICS``  need a calibrated *number*. Only a model that claims
                         to emit probabilities is scored on them.

That separation exists so a rank-only baseline is never handed invented
probabilities in order to fit an API. A random shuffle has no meaningful Brier
score, and reporting one would be a fabrication.

Ties are handled analytically in ``roc_auc`` and ``pr_auc`` -- both are defined
over the whole score distribution, so a tie-break would change the answer rather
than merely settle an ordering. The top-k metrics do need an explicit
tie-break, and take one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

#: Probabilities are clamped into this open interval before ``log_loss`` takes a
#: logarithm, so a confident-and-wrong prediction costs a large finite number
#: rather than infinity.
LOG_LOSS_EPSILON = 1e-15

#: The project spec asks for 15 equal-mass bins. Equal-mass rather than
#: equal-width because with a skewed score distribution most equal-width bins
#: end up empty, and an empty bin contributes nothing while looking calibrated.
DEFAULT_CALIBRATION_BINS = 15


class MetricError(ValueError):
    """Raised when a metric is asked for something it cannot mean."""


def _check_pairs(labels: Sequence[int], scores: Sequence[float]) -> None:
    if len(labels) != len(scores):
        raise MetricError(f"labels ({len(labels)}) and scores ({len(scores)}) differ in length")
    if any(v is None for v in scores):
        raise MetricError("scores contain a null; the evaluator never imputes a missing score")
    if any(label not in (0, 1) for label in labels):
        raise MetricError("labels must be 0 or 1")


def _positives(labels: Sequence[int]) -> int:
    return sum(1 for label in labels if label == 1)


# --- 1. ranking quality ----------------------------------------------------


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Probability a random positive outscores a random negative.

    Computed from average ranks (the Mann-Whitney U identity) so tied scores
    contribute exactly one half, matching ``sklearn.metrics.roc_auc_score``.

    Returns ``None`` when only one class is present: with no negatives there is
    no pair to rank, and returning 0.5 would invent an answer.

    Base-rate invariant, which is why it is one of the two metrics safe to
    average across folds whose prevalence drifts from 0.88 to 0.39.
    """
    _check_pairs(labels, scores)
    n_pos = _positives(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1

    rank_sum = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pr_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Average precision -- the step-wise estimator of area under precision-recall.

    Equivalent to ``sklearn.metrics.average_precision_score``, not to a
    trapezoid through the PR curve, which is optimistically biased.

    .. math:: AP = \\sum_k (R_k - R_{k-1}) \\cdot P_k

    where *k* indexes distinct score thresholds in decreasing order.

    The project spec names PR-AUC the primary classification metric because the
    positive class is what matters operationally. Note the measured base rate is
    0.525, not the 0.15-0.25 the spec anticipated, so a PR-AUC of 0.53 here is
    the *no-skill* floor -- always read it beside the fold's prevalence.

    Returns ``None`` when there are no positives: recall is undefined.
    """
    _check_pairs(labels, scores)
    n_pos = _positives(labels)
    if n_pos == 0:
        return None

    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    average_precision = 0.0
    previous_recall = 0.0
    true_positives = 0
    seen = 0

    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            true_positives += labels[order[k]]
            seen += 1
        recall = true_positives / n_pos
        precision = true_positives / seen
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        i = j + 1

    return average_precision


# --- 2. probability quality ------------------------------------------------


def brier(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Mean squared error of a probability. Lower is better; 0.25 is a coin flip.

    Decomposes into reliability, resolution and uncertainty, which is why it is
    reported alongside ECE rather than instead of it: ECE says whether 0.3 means
    30%, Brier says whether the model separates anything at all.
    """
    _check_pairs(labels, probabilities)
    if not labels:
        raise MetricError("brier is undefined on an empty set")
    return sum((p - y) ** 2 for y, p in zip(labels, probabilities, strict=True)) / len(labels)


def log_loss(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Negative log likelihood. Punishes confident errors far harder than Brier."""
    _check_pairs(labels, probabilities)
    if not labels:
        raise MetricError("log_loss is undefined on an empty set")
    total = 0.0
    for y, raw in zip(labels, probabilities, strict=True):
        p = min(max(raw, LOG_LOSS_EPSILON), 1.0 - LOG_LOSS_EPSILON)
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return total / len(labels)


def calibration_bins(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> list[tuple[int, float, float]]:
    """Equal-mass bins as ``(count, mean_predicted, observed_rate)``.

    Rows are sorted by predicted probability and cut into ``n_bins`` contiguous
    chunks of near-equal size. Chunking by position rather than by value means
    no bin is ever empty, so every bin carries real weight.
    """
    _check_pairs(labels, probabilities)
    if n_bins < 1:
        raise MetricError("n_bins must be at least 1")
    n = len(labels)
    if n == 0:
        return []

    order = sorted(range(n), key=lambda i: probabilities[i])
    bins: list[tuple[int, float, float]] = []
    for b in range(min(n_bins, n)):
        lo = b * n // min(n_bins, n)
        hi = (b + 1) * n // min(n_bins, n)
        if hi <= lo:
            continue
        idx = order[lo:hi]
        count = len(idx)
        mean_predicted = sum(probabilities[i] for i in idx) / count
        observed = sum(labels[i] for i in idx) / count
        bins.append((count, mean_predicted, observed))
    return bins


def ece(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> float:
    """Expected calibration error: mass-weighted mean gap between claim and truth.

    0 means a predicted 0.30 happens 30% of the time. This is the number that
    licenses the downstream cost-based threshold, so Component 9 treats its
    drift over time as the retraining trigger.
    """
    bins = calibration_bins(labels, probabilities, n_bins=n_bins)
    if not bins:
        raise MetricError("ece is undefined on an empty set")
    total = sum(count for count, _, _ in bins)
    return sum(count * abs(observed - predicted) for count, predicted, observed in bins) / total


def mce(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> float:
    """Maximum calibration error: the worst single bin, unweighted by its mass."""
    bins = calibration_bins(labels, probabilities, n_bins=n_bins)
    if not bins:
        raise MetricError("mce is undefined on an empty set")
    return max(abs(observed - predicted) for _, predicted, observed in bins)


def precision_recall_f1(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Point-estimate classification metrics at an explicit threshold.

    Reported for completeness only. Sentinel is a top-k selection problem, not a
    classification problem -- every establishment is inspected eventually, so
    there is nothing to accept or reject and 0.5 has no operational meaning.
    A threshold is genuinely needed only by the Component 16 deferral gate.
    """
    _check_pairs(labels, probabilities)
    predicted = [1 if p >= threshold else 0 for p in probabilities]
    tp = sum(1 for y, q in zip(labels, predicted, strict=True) if y == 1 and q == 1)
    fp = sum(1 for y, q in zip(labels, predicted, strict=True) if y == 0 and q == 1)
    fn = sum(1 for y, q in zip(labels, predicted, strict=True) if y == 1 and q == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# --- 3. top-k under capacity ----------------------------------------------


def top_k_indices(
    scores: Sequence[float],
    tie_break: Sequence[str],
    k: int,
) -> list[int]:
    """Indices of the ``k`` highest scores, ties settled deterministically.

    Sorted by descending score then ascending tie-break key. Never by frame
    order -- two runs over the same data with the rows shuffled must select the
    same k establishments, and Parquet row order is not a contract.
    """
    if k < 1:
        raise MetricError(f"k must be at least 1, got {k}")
    if len(scores) != len(tie_break):
        raise MetricError("scores and tie_break differ in length")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], tie_break[i]))
    return order[: min(k, len(order))]


def precision_at_k(
    labels: Sequence[int],
    scores: Sequence[float],
    tie_break: Sequence[str],
    k: int,
) -> float:
    """Share of the top ``k`` that were genuine Priority citations.

    The operational metric: if only ``k`` inspections fit in the window, how
    many of them landed on an establishment that was in fact cited?

    Divided by ``min(k, n)`` so that asking for more capacity than the window
    contains does not silently deflate the number.
    """
    _check_pairs(labels, scores)
    chosen = top_k_indices(scores, tie_break, k)
    if not chosen:
        raise MetricError("precision_at_k is undefined on an empty set")
    return sum(labels[i] for i in chosen) / len(chosen)


def recall_at_k(
    labels: Sequence[int],
    scores: Sequence[float],
    tie_break: Sequence[str],
    k: int,
) -> float | None:
    """Share of all the window's positives that the top ``k`` captured.

    Precision@k alone is misleading once ``k`` exceeds the number of positives:
    its ceiling drops below 1 and keeps dropping as capacity grows. Recall@k
    moves the opposite way, so the pair is legible where either alone is not.
    """
    _check_pairs(labels, scores)
    n_pos = _positives(labels)
    if n_pos == 0:
        return None
    chosen = top_k_indices(scores, tie_break, k)
    return sum(labels[i] for i in chosen) / n_pos


def lift_at_k(
    labels: Sequence[int],
    scores: Sequence[float],
    tie_break: Sequence[str],
    k: int,
) -> float | None:
    """Precision@k divided by the window's base rate.

    The drift-robust form of precision@k. Prevalence falls from 0.88 to 0.39
    across this dataset, so a raw precision@k of 0.55 is excellent in 2026 and
    poor in 2019. Lift states how many times better than blind selection the
    ranking did, which is comparable across folds. 1.0 means no better.
    """
    _check_pairs(labels, scores)
    if not labels:
        return None
    base_rate = _positives(labels) / len(labels)
    if base_rate == 0.0:
        return None
    return precision_at_k(labels, scores, tie_break, k) / base_rate


#: Metrics defined on an ordering alone. Every baseline is scored on these.
RANKING_METRICS: tuple[str, ...] = (
    "roc_auc",
    "pr_auc",
    "precision_at_k",
    "recall_at_k",
    "lift_at_k",
)

#: Metrics that require the score to be a calibrated probability. Scored only
#: when the producer sets ``PredictionSet.is_probability``.
PROBABILITY_METRICS: tuple[str, ...] = (
    "brier",
    "log_loss",
    "ece",
    "mce",
    "precision",
    "recall",
    "f1",
)


__all__ = [
    "DEFAULT_CALIBRATION_BINS",
    "LOG_LOSS_EPSILON",
    "PROBABILITY_METRICS",
    "RANKING_METRICS",
    "MetricError",
    "brier",
    "calibration_bins",
    "ece",
    "lift_at_k",
    "log_loss",
    "mce",
    "pr_auc",
    "precision_at_k",
    "precision_recall_f1",
    "recall_at_k",
    "roc_auc",
    "top_k_indices",
]
