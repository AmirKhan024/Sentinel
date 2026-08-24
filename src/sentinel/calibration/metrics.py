"""The metrics Component 5 does not have. Pure -- no filesystem, no clock.

**``evaluation/metrics.py`` is not modified.** Component 5 is closed, and its
``RANKING_METRICS`` / ``PROBABILITY_METRICS`` tuples are part of its contract -- appending
to them would change what every earlier component's artifacts are read as. So ECE, MCE,
Brier, log-loss, ROC-AUC, PR-AUC and precision@k are **imported** from there and used
unchanged, and only the four things Component 9 genuinely adds live here:

1. Murphy's Brier decomposition, over the same 15 equal-mass bins Component 5's ECE uses,
   so a Component 9 reliability number and a Component 5 ECE can never disagree about
   binning.
2. The Cox recalibration slope and intercept.
3. Bootstrap intervals, under two resampling schemes.
4. Ranking preservation -- whether the calibrator reordered anything, and what it tied.

Everything here is hand-implemented against numpy rather than scipy, matching the precedent
``evaluation/metrics.py`` set: a formula can be hand-rolled and verified against a reference
to floating-point tolerance, and the test suite does exactly that with brute-force oracles
on small fixtures. No runtime dependency is added.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from sentinel.calibration.definitions import (
    BOOTSTRAP_REPLICATIONS,
    BOOTSTRAP_SCHEME_BLOCK,
    BOOTSTRAP_SCHEME_ROW,
    CI_LEVEL,
    MAX_DEGENERATE_SHARE,
    PLATT_PARAMS,
)
from sentinel.calibration.models import (
    BootstrapInterval,
    BrierDecomposition,
    RankingPreservation,
    SlopeIntercept,
)
from sentinel.calibration.preprocess import logit
from sentinel.evaluation.metrics import (
    DEFAULT_CALIBRATION_BINS,
    LOG_LOSS_EPSILON,
    MetricError,
    calibration_bins,
    precision_at_k,
    roc_auc,
    top_k_indices,
)

logger = logging.getLogger(__name__)

#: A metric of (labels, probabilities). ``None`` where the metric is undefined -- a
#: single-class resample has no ROC-AUC, and inventing 0.5 would bias the interval.
Metric = Callable[[Sequence[int], Sequence[float]], float | None]


# --- 1. Brier decomposition --------------------------------------------------


def brier_decomposition(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> BrierDecomposition:
    r"""Murphy's (1973) three-term decomposition of the Brier score.

    With bins :math:`k` of size :math:`n_k`, mean forecast :math:`\bar p_k`, observed rate
    :math:`\bar o_k`, and overall base rate :math:`\bar o`:

    .. math::

        \mathrm{REL} &= \frac{1}{N} \sum_k n_k (\bar p_k - \bar o_k)^2 \\
        \mathrm{RES} &= \frac{1}{N} \sum_k n_k (\bar o_k - \bar o)^2 \\
        \mathrm{UNC} &= \bar o (1 - \bar o)

    **Reliability** is what calibration fixes: how far a bin's claim sits from what
    happened. Lower is better, and 0 means every claim was borne out.

    **Resolution** is what the base model provides: how far the bins' outcomes sit from the
    overall base rate, i.e. whether the model separates anything at all. Higher is better.
    A monotone calibrator cannot manufacture it, and if it appears to, something is wrong.

    **Uncertainty** is a property of the data, not of any model: the Brier score of always
    predicting the base rate. Nothing can change it.

    ``BS = REL - RES + UNC`` holds exactly only for a forecast that is *constant within
    each bin*. With continuous probabilities cut into 15 equal-mass groups it does not, and
    the gap is the within-bin variance :math:`\frac{1}{N}\sum_k\sum_{i \in k}(p_i-\bar
    p_k)^2`. **That residual is returned, not hidden.** Reporting ``REL - RES + UNC`` as
    "the Brier score" would be a fabrication of exactly the kind this repository refuses,
    and the test suite asserts the full identity to 1e-12.
    """
    bins = calibration_bins(labels, probabilities, n_bins=n_bins)
    if not bins:
        raise MetricError("brier_decomposition is undefined on an empty set")

    total = sum(count for count, _, _ in bins)
    base_rate = sum(count * observed for count, _, observed in bins) / total

    reliability = sum(count * (predicted - observed) ** 2 for count, predicted, observed in bins)
    resolution = sum(count * (observed - base_rate) ** 2 for count, _, observed in bins)
    reliability /= total
    resolution /= total
    uncertainty = base_rate * (1.0 - base_rate)
    recomposed = reliability - resolution + uncertainty

    observed_brier = sum(
        (p - y) ** 2 for y, p in zip(labels, probabilities, strict=True)
    ) / len(labels)

    return BrierDecomposition(
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        recomposed=recomposed,
        within_bin_variance=observed_brier - recomposed,
        n_bins=len(bins),
    )


# --- 2. calibration slope and intercept --------------------------------------


def calibration_slope_intercept(
    labels: Sequence[int], probabilities: Sequence[float]
) -> SlopeIntercept:
    """The Cox recalibration regression: logistic of the label on ``logit(p)``.

    Perfect calibration is **slope 1.0, intercept 0.0** -- the model's own logit needs no
    adjustment. Slope below 1 means overconfidence (the scores are too extreme for what
    happens); above 1 means underconfidence. The intercept is a shift in the base rate.

    This is deliberately the same estimator Platt fits. On a Platt-calibrated window the
    slope must therefore come back to 1.0 by construction, and if it does not, the
    calibrator was misapplied -- a free self-check, asserted at warn severity.

    Returns ``None`` on a single-class window rather than inventing a number, matching
    ``evaluation.metrics.roc_auc``'s posture.
    """
    if len(labels) != len(probabilities):
        raise MetricError(
            f"labels ({len(labels)}) and probabilities ({len(probabilities)}) differ in length"
        )
    if not labels:
        raise MetricError("calibration_slope_intercept is undefined on an empty set")
    if len(set(labels)) < 2:
        return SlopeIntercept(slope=None, intercept=None)

    from sklearn.linear_model import LogisticRegression

    x = np.asarray([logit(p) for p in probabilities], dtype=np.float64).reshape(-1, 1)
    y = np.asarray(labels, dtype=np.int64)
    model = LogisticRegression(**dict(PLATT_PARAMS))
    model.fit(x, y)
    return SlopeIntercept(
        slope=float(model.coef_[0][0]),
        intercept=float(model.intercept_[0]),
    )


# --- 3. bootstrap ------------------------------------------------------------


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


#: Metrics the vectorised resampler can compute. Anything else falls back to the scalar
#: path, which calls Component 5's functions one resample at a time.
VECTORISED_METRICS: tuple[str, ...] = ("ece", "brier", "log_loss")


def _batch_metrics(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    index: NDArray[np.int64],
    *,
    n_bins: int,
    epsilon: float,
) -> dict[str, NDArray[np.float64]]:
    """ECE, Brier and log-loss for many resamples at once.

    A vectorised twin of ``evaluation.metrics.ece`` / ``brier`` / ``log_loss``, not a second
    definition of them. The binning, ordering and tie handling are identical, and the test
    suite asserts agreement on real-shaped tied data.

    It is **not bit-identical**: numpy sums pairwise where Python's ``sum`` goes left to
    right, so a mean over ~900 terms differs in the last bit or two (measured at ~1 ULP).
    That is acceptable *here and only here*, because this path is used solely for resamples
    feeding a percentile. Component 5's functions remain the only ones used for every
    reported point estimate.

    It exists because the scalar path costs ~90 seconds per fold at 1,000 replications --
    roughly two and a half hours for the production run -- and almost all of that is Python
    loop overhead rather than arithmetic.

    ``index`` is ``(replications, n)`` of row positions. The equal-mass binning matches
    Component 5's exactly: rows are stably sorted by probability and cut into contiguous
    chunks by *position*, ``lo = b * n // n_bins``, so no bin is ever empty.
    """
    drawn_y = labels[index].astype(np.float64)
    drawn_p = probabilities[index]
    reps, n = drawn_p.shape

    brier_values = np.mean((drawn_p - drawn_y) ** 2, axis=1)
    clipped = np.clip(drawn_p, epsilon, 1.0 - epsilon)
    log_loss_values = -np.mean(
        drawn_y * np.log(clipped) + (1.0 - drawn_y) * np.log1p(-clipped), axis=1
    )

    # "stable" so ties resolve by original position, which is what Python's sorted() does
    # in evaluation.metrics.calibration_bins.
    order = np.argsort(drawn_p, axis=1, kind="stable")
    sorted_p = np.take_along_axis(drawn_p, order, axis=1)
    sorted_y = np.take_along_axis(drawn_y, order, axis=1)

    bins = min(n_bins, n)
    edges = np.array([b * n // bins for b in range(bins)], dtype=np.int64)
    counts = np.diff(np.append(edges, n)).astype(np.float64)

    summed_p = np.add.reduceat(sorted_p, edges, axis=1)
    summed_y = np.add.reduceat(sorted_y, edges, axis=1)
    mean_p = summed_p / counts
    observed = summed_y / counts
    ece_values = np.sum(counts * np.abs(observed - mean_p), axis=1) / float(n)

    return {"ece": ece_values, "brier": brier_values, "log_loss": log_loss_values}


def bootstrap(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    metric: Metric,
    metric_name: str,
    scheme: str,
    seed_key: Sequence[int],
    groups: Sequence[str] | None = None,
    replications: int = BOOTSTRAP_REPLICATIONS,
    level: float = CI_LEVEL,
) -> BootstrapInterval:
    """A within-fold percentile interval for one metric.

    **What is resampled.** ``scheme="row"`` draws ``n`` row indices with replacement,
    i.i.d. ``scheme="establishment_block"`` draws *establishments* with replacement and
    takes all of their rows -- because establishments recur within a window and their rows
    share an as-of history, so an i.i.d. row bootstrap understates the standard error.
    Both are run for every reported interval; running both settles the objection with a
    measurement instead of a caveat.

    **Seeding.** ``seed_key`` is a tuple of structured integers -- seed, model, fold,
    stage, scheme -- fed to a ``SeedSequence``. Every cell therefore gets a statistically
    independent stream, the whole bootstrap is reproducible, and no two cells share a
    stream. Never a global RNG.

    **Degenerate resamples.** A draw containing a single class makes ROC-AUC and the
    calibration slope undefined. Those are counted in ``degenerate`` and **excluded from
    the percentile**, never replaced by 0.5. Above ``MAX_DEGENERATE_SHARE`` the interval is
    written as null rather than computed from whatever survived.

    This is a *within-fold* interval. Across folds the component reports ``mean ± SD``,
    which is a dispersion and not a confidence interval -- see ``BOOTSTRAP_CAVEAT``.
    """
    if scheme not in (BOOTSTRAP_SCHEME_ROW, BOOTSTRAP_SCHEME_BLOCK):
        raise MetricError(f"unknown bootstrap scheme {scheme!r}")
    if replications < 1:
        raise MetricError("replications must be at least 1")
    n = len(labels)
    if n == 0:
        raise MetricError("bootstrap is undefined on an empty set")

    point = metric(labels, probabilities)
    rng = np.random.default_rng(np.random.SeedSequence(list(seed_key)))

    if scheme == BOOTSTRAP_SCHEME_BLOCK:
        if groups is None:
            raise MetricError("the establishment_block scheme needs group labels")
        if len(groups) != n:
            raise MetricError("groups and labels differ in length")
        blocks: dict[str, list[int]] = {}
        for position, key in enumerate(groups):
            blocks.setdefault(key, []).append(position)
        block_keys = sorted(blocks)
        block_index = [blocks[key] for key in block_keys]

    indices: list[list[int]] = []
    for _ in range(replications):
        index: list[int]
        if scheme == BOOTSTRAP_SCHEME_ROW:
            index = [int(i) for i in rng.integers(0, n, size=n)]
        else:
            chosen = rng.integers(0, len(block_index), size=len(block_index))
            index = [i for b in chosen for i in block_index[int(b)]]
        indices.append(index)

    draws: list[float] = []
    degenerate = 0
    usable = [ix for ix in indices if len({labels[i] for i in ix}) >= 2]
    degenerate += len(indices) - len(usable)

    if usable and metric_name in VECTORISED_METRICS:
        # The row scheme gives every resample the same length; the establishment-block
        # scheme does not, because drawing establishments with replacement draws a varying
        # number of rows. Grouping by length keeps the batched path available for both --
        # the alternative, falling back to the scalar loop for the block scheme, costs
        # roughly forty times as much and was measured doing so.
        by_length: dict[int, list[list[int]]] = {}
        for candidate_index in usable:
            by_length.setdefault(len(candidate_index), []).append(candidate_index)

        label_array = np.asarray(labels, dtype=np.int64)
        probability_array = np.asarray(probabilities, dtype=np.float64)
        for group in by_length.values():
            batch = _batch_metrics(
                label_array,
                probability_array,
                np.asarray(group, dtype=np.int64),
                n_bins=DEFAULT_CALIBRATION_BINS,
                epsilon=LOG_LOSS_EPSILON,
            )[metric_name]
            for value in batch:
                if math.isfinite(float(value)):
                    draws.append(float(value))
                else:
                    degenerate += 1
    else:
        for index in usable:
            value = metric([labels[i] for i in index], [probabilities[i] for i in index])
            if value is None or not math.isfinite(value):
                degenerate += 1
                continue
            draws.append(value)

    if not draws or degenerate > replications * MAX_DEGENERATE_SHARE:
        logger.warning(
            "%s/%s: %d of %d resamples degenerate; interval written as null",
            metric_name,
            scheme,
            degenerate,
            replications,
        )
        return BootstrapInterval(
            metric=metric_name,
            scheme=scheme,
            point_estimate=point,
            replications=replications,
            seed=int(seed_key[0]),
            mean=None,
            sd=None,
            lower=None,
            upper=None,
            level=level,
            degenerate=degenerate,
        )

    tail = (1.0 - level) / 2.0 * 100.0
    return BootstrapInterval(
        metric=metric_name,
        scheme=scheme,
        point_estimate=point,
        replications=replications,
        seed=int(seed_key[0]),
        mean=float(np.mean(draws)),
        sd=float(np.std(draws, ddof=1)) if len(draws) > 1 else None,
        lower=_percentile(draws, tail),
        upper=_percentile(draws, 100.0 - tail),
        level=level,
        degenerate=degenerate,
    )


# --- 4. ranking preservation -------------------------------------------------


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks with ties averaged, as Spearman's definition requires."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson correlation of the average ranks. ``None`` if either side is constant."""
    if len(x) != len(y):
        raise MetricError("spearman inputs differ in length")
    if len(x) < 2:
        return None
    rx = np.asarray(_average_ranks(x), dtype=np.float64)
    ry = np.asarray(_average_ranks(y), dtype=np.float64)
    sx, sy = float(rx.std()), float(ry.std())
    if sx == 0.0 or sy == 0.0:
        return None
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def _sort_and_count_inversions(values: list[float]) -> int:
    """Discordant pairs, by merge sort, in O(n log n).

    The naive double loop is O(n^2); a test window holds up to 8,840 rows and the project
    has 18 of them per model, so the naive form is not merely slow but unusable.
    """
    if len(values) < 2:
        return 0

    def sort(seq: list[float]) -> tuple[list[float], int]:
        if len(seq) < 2:
            return seq, 0
        middle = len(seq) // 2
        left, a = sort(seq[:middle])
        right, b = sort(seq[middle:])
        merged: list[float] = []
        swaps = a + b
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
                swaps += len(left) - i
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, swaps

    return sort(values)[1]


def _tie_pairs(values: Sequence[object]) -> int:
    """Number of tied pairs: the sum of t(t-1)/2 over groups of equal values.

    Takes ``object`` rather than ``float`` so the joint-tie count can group on the ``(x, y)``
    pair itself. Grouping on the pair directly rather than on ``hash((x, y))`` matters: a
    hash collision would silently merge two distinct pairs and undercount the ties, which
    would bias tau-b.
    """
    counts: dict[object, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sum(t * (t - 1) // 2 for t in counts.values())


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Kendall's tau-b, by Knight's O(n log n) method.

    tau-b rather than tau-a because both sides carry ties: the base scores are already not
    all distinct (xgboost has 39,677 distinct values over 41,536 rows), and isotonic adds
    more. tau-a would report those ties as disagreement.

    .. math::

        \\tau_b = \\frac{n_0 - n_1 - n_2 + n_3 - 2S}{\\sqrt{(n_0-n_1)(n_0-n_2)}}

    where :math:`n_0 = \\binom{n}{2}`, :math:`n_1` and :math:`n_2` count tied pairs in x
    and y, :math:`n_3` counts jointly tied pairs, and :math:`S` is the discordant-pair
    count from the merge sort. Verified in the test suite against a brute-force
    :math:`O(n^2)` oracle on small fixtures.
    """
    if len(x) != len(y):
        raise MetricError("kendall_tau_b inputs differ in length")
    n = len(x)
    if n < 2:
        return None

    order = sorted(range(n), key=lambda i: (x[i], y[i]))
    y_sorted = [y[i] for i in order]

    n0 = n * (n - 1) // 2
    n1 = _tie_pairs(x)
    n2 = _tie_pairs(y)
    joint = _tie_pairs([(x[i], y[i]) for i in range(n)])
    swaps = _sort_and_count_inversions(y_sorted)

    denominator = math.sqrt(float(n0 - n1) * float(n0 - n2))
    if denominator <= 0.0:
        return None
    return (n0 - n1 - n2 + joint - 2 * swaps) / denominator


def ranking_preservation(
    base_scores: Sequence[float],
    calibrated_scores: Sequence[float],
    labels: Sequence[int],
    tie_break: Sequence[str],
    k: int,
) -> RankingPreservation:
    """Did the calibrator reorder anything, and what did it tie together?

    A monotone map cannot reorder, so ``inversions`` should be 0 and Spearman's rho exactly
    1.0 for a strictly monotone calibrator such as Platt.

    Isotonic is only **weakly** monotone: pool-adjacent-violators produces plateaus, so two
    rows with different base scores can leave with the same calibrated score.
    ``evaluation.metrics.top_k_indices`` then settles them by ``target_inspection_id``
    ascending, which can move top-k membership. **That is a tie, not a ranking inversion,
    and must not be reported as one** -- which is why ``new_ties_created`` and
    ``top_k_membership_changed`` are separate columns from ``inversions``.

    ``new_ties_created`` is an *increment* over the ties the base scores already carried,
    not a count of ties in the output.
    """
    n = len(base_scores)
    if not (n == len(calibrated_scores) == len(labels) == len(tie_break)):
        raise MetricError("ranking_preservation inputs differ in length")
    if n == 0:
        raise MetricError("ranking_preservation is undefined on an empty set")

    distinct_before = len(set(base_scores))
    distinct_after = len(set(calibrated_scores))

    # Discordant pairs between the two orderings: sort by the base score, then count
    # inversions in the calibrated sequence. Ties on the base score are ordered by the
    # calibrated value so that a pair tied before calibration is never counted as an
    # inversion after it.
    order = sorted(range(n), key=lambda i: (base_scores[i], calibrated_scores[i]))
    inversions = _sort_and_count_inversions([calibrated_scores[i] for i in order])

    before_k = top_k_indices(list(base_scores), list(tie_break), k)
    after_k = top_k_indices(list(calibrated_scores), list(tie_break), k)

    return RankingPreservation(
        spearman_rho=spearman(base_scores, calibrated_scores),
        kendall_tau_b=kendall_tau_b(base_scores, calibrated_scores),
        distinct_before=distinct_before,
        distinct_after=distinct_after,
        new_ties_created=max(0, distinct_before - distinct_after),
        inversions=inversions,
        is_strictly_monotone=inversions == 0 and distinct_after == distinct_before,
        top_k=k,
        top_k_membership_changed=len(set(before_k) - set(after_k)),
        precision_at_k_before=precision_at_k(labels, list(base_scores), list(tie_break), k),
        precision_at_k_after=precision_at_k(labels, list(calibrated_scores), list(tie_break), k),
        roc_auc_before=roc_auc(labels, base_scores),
        roc_auc_after=roc_auc(labels, calibrated_scores),
    )


__all__ = [
    "Metric",
    "VECTORISED_METRICS",
    "bootstrap",
    "brier_decomposition",
    "calibration_slope_intercept",
    "kendall_tau_b",
    "ranking_preservation",
    "spearman",
]
