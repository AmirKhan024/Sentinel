"""Metrics, cross-checked against an independent implementation.

Component 5 implements its own PR-AUC, ROC-AUC, Brier and log loss so that no
runtime dependency is added for them -- the project introduces a technology only
when a component requires it. scikit-learn arrived as a runtime dependency with
Component 6 (ADR 0015), and remains the oracle for these hand-rolled metrics.

Hand-rolled maths needs a reason to be believed. So scikit-learn is a **dev**
dependency, imported here and nowhere in ``src/sentinel``, and every metric is
compared against it on random and adversarial inputs. "How do you know your
PR-AUC is right?" has a better answer than "I read the formula carefully".

ECE and MCE have no scikit-learn equivalent, so those are checked against
hand-computed values on inputs small enough to verify by eye.
"""

from __future__ import annotations

import math
import random

import pytest
from sklearn import metrics as sk

from sentinel.evaluation.metrics import (
    DEFAULT_CALIBRATION_BINS,
    PROBABILITY_METRICS,
    RANKING_METRICS,
    MetricError,
    brier,
    calibration_bins,
    ece,
    lift_at_k,
    log_loss,
    mce,
    pr_auc,
    precision_at_k,
    precision_recall_f1,
    recall_at_k,
    roc_auc,
    top_k_indices,
)

TOLERANCE = 1e-12


def _dataset(
    seed: int, n: int = 400, *, tie_buckets: int | None = None
) -> tuple[list[int], list[float]]:
    """Random labels and scores. ``tie_buckets`` forces heavy tie structure."""
    rng = random.Random(seed)
    labels = [rng.randint(0, 1) for _ in range(n)]
    if tie_buckets:
        scores = [float(rng.randrange(tie_buckets)) for _ in range(n)]
    else:
        scores = [rng.random() for _ in range(n)]
    if len(set(labels)) == 1:  # keep both classes present
        labels[0] = 1 - labels[0]
    return labels, scores


# --- 1. ranking metrics against scikit-learn -------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8, 13, 21])
def test_roc_auc_matches_scikit_learn(seed: int) -> None:
    labels, scores = _dataset(seed)
    assert roc_auc(labels, scores) == pytest.approx(sk.roc_auc_score(labels, scores), abs=TOLERANCE)


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8])
def test_roc_auc_matches_scikit_learn_when_scores_are_heavily_tied(seed: int) -> None:
    """Ties must contribute exactly one half, which is where naive code goes wrong."""
    labels, scores = _dataset(seed, tie_buckets=4)
    assert roc_auc(labels, scores) == pytest.approx(sk.roc_auc_score(labels, scores), abs=TOLERANCE)


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8, 13, 21])
def test_pr_auc_matches_scikit_learn_average_precision(seed: int) -> None:
    labels, scores = _dataset(seed)
    assert pr_auc(labels, scores) == pytest.approx(
        sk.average_precision_score(labels, scores), abs=TOLERANCE
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8])
def test_pr_auc_matches_scikit_learn_when_scores_are_heavily_tied(seed: int) -> None:
    labels, scores = _dataset(seed, tie_buckets=3)
    assert pr_auc(labels, scores) == pytest.approx(
        sk.average_precision_score(labels, scores), abs=TOLERANCE
    )


def test_pr_auc_is_average_precision_not_a_trapezoid_through_the_curve() -> None:
    """The two differ, and the trapezoid form is optimistically biased.

    Stated as a test so a future refactor toward ``auc(recall, precision)``
    fails loudly rather than shifting every reported number upward.
    """
    labels = [0, 1, 1, 0, 1, 0, 0, 1]
    scores = [0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    precision, recall, _ = sk.precision_recall_curve(labels, scores)
    trapezoid = sk.auc(recall, precision)
    assert pr_auc(labels, scores) == pytest.approx(
        sk.average_precision_score(labels, scores), abs=TOLERANCE
    )
    assert pr_auc(labels, scores) != pytest.approx(trapezoid, abs=1e-6)


# --- 2. probability metrics against scikit-learn ---------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8])
def test_brier_matches_scikit_learn(seed: int) -> None:
    labels, scores = _dataset(seed)
    assert brier(labels, scores) == pytest.approx(
        sk.brier_score_loss(labels, scores), abs=TOLERANCE
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8])
def test_log_loss_matches_scikit_learn(seed: int) -> None:
    """Probabilities are kept away from 0 and 1 so the clamp is a no-op here."""
    labels, scores = _dataset(seed)
    scores = [0.01 + 0.98 * s for s in scores]
    assert log_loss(labels, scores) == pytest.approx(sk.log_loss(labels, scores), abs=1e-10)


def test_log_loss_clamps_rather_than_returning_infinity() -> None:
    """A confident-and-wrong prediction must cost a large finite number."""
    value = log_loss([1, 0], [0.0, 1.0])
    assert math.isfinite(value)
    assert value > 30


def test_brier_of_a_perfect_forecast_is_zero() -> None:
    assert brier([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0]) == 0.0


def test_brier_of_a_coin_flip_is_a_quarter() -> None:
    assert brier([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == 0.25


# --- 3. calibration, checked by hand ---------------------------------------


def test_calibration_bins_are_equal_mass_and_never_empty() -> None:
    labels, scores = _dataset(7, n=100)
    bins = calibration_bins(labels, scores, n_bins=10)
    assert len(bins) == 10
    assert all(count == 10 for count, _, _ in bins)
    assert sum(count for count, _, _ in bins) == 100


def test_calibration_bins_handle_fewer_rows_than_bins() -> None:
    bins = calibration_bins([1, 0, 1], [0.9, 0.1, 0.8], n_bins=DEFAULT_CALIBRATION_BINS)
    assert len(bins) == 3
    assert sum(count for count, _, _ in bins) == 3


def test_ece_is_zero_when_every_bin_is_perfectly_calibrated() -> None:
    """Half the rows claim 0.0 and are all negative; half claim 1.0 and are all positive."""
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    probabilities = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert ece(labels, probabilities, n_bins=2) == pytest.approx(0.0)


def test_ece_equals_the_hand_computed_gap() -> None:
    """Two bins of four. Lower bin claims 0.25 and observes 0.0; upper claims
    0.75 and observes 1.0. Both gaps are 0.25, so the mass-weighted mean is 0.25.
    """
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    probabilities = [0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75]
    assert ece(labels, probabilities, n_bins=2) == pytest.approx(0.25)


def test_mce_reports_the_worst_bin_not_the_average() -> None:
    """One bin is perfect, the other is off by 0.5. ECE halves it; MCE does not."""
    labels = [0, 0, 1, 1]
    probabilities = [0.0, 0.0, 0.5, 0.5]
    assert mce(labels, probabilities, n_bins=2) == pytest.approx(0.5)
    assert ece(labels, probabilities, n_bins=2) == pytest.approx(0.25)


def test_ece_on_an_empty_set_is_an_error_not_a_zero() -> None:
    with pytest.raises(MetricError):
        ece([], [])


# --- 4. top-k under capacity -----------------------------------------------


def test_precision_at_k_counts_only_the_top_k() -> None:
    labels = [1, 1, 0, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    ids = ["a", "b", "c", "d", "e"]
    assert precision_at_k(labels, scores, ids, 2) == 1.0
    assert precision_at_k(labels, scores, ids, 4) == 0.5
    assert precision_at_k(labels, scores, ids, 5) == 0.4


def test_recall_at_k_reaches_one_once_every_positive_is_covered() -> None:
    labels = [1, 1, 0, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    ids = ["a", "b", "c", "d", "e"]
    assert recall_at_k(labels, scores, ids, 2) == 1.0
    assert recall_at_k(labels, scores, ids, 1) == 0.5


def test_lift_at_k_is_one_when_the_ranking_carries_no_information() -> None:
    """Base rate 0.5, and a ranking that picks one of each: lift is exactly 1."""
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    ids = ["a", "b", "c", "d"]
    assert lift_at_k(labels, scores, ids, 2) == pytest.approx(1.0)


def test_lift_at_k_exceeds_one_when_positives_are_ranked_first() -> None:
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    ids = ["a", "b", "c", "d"]
    assert lift_at_k(labels, scores, ids, 2) == pytest.approx(2.0)


def test_k_larger_than_the_window_is_clamped_not_silently_deflated() -> None:
    labels = [1, 1]
    scores = [0.9, 0.8]
    ids = ["a", "b"]
    assert precision_at_k(labels, scores, ids, 100) == 1.0


def test_k_below_one_is_rejected() -> None:
    with pytest.raises(MetricError, match="at least 1"):
        top_k_indices([0.5], ["a"], 0)


def test_top_k_breaks_ties_on_the_secondary_key_not_on_input_order() -> None:
    """Every score identical, so the id decides -- and reversing the input must
    not change which rows are selected."""
    scores = [0.5] * 4
    ids = ["d", "b", "a", "c"]
    chosen = [ids[i] for i in top_k_indices(scores, ids, 2)]
    reversed_choice = [
        list(reversed(ids))[i] for i in top_k_indices(scores, list(reversed(ids)), 2)
    ]
    assert chosen == ["a", "b"]
    assert reversed_choice == ["a", "b"]


# --- 5. thresholded classification ------------------------------------------


def test_precision_recall_f1_match_scikit_learn_at_the_same_threshold() -> None:
    labels, scores = _dataset(11, n=200)
    precision, recall, f1 = precision_recall_f1(labels, scores, threshold=0.5)
    predicted = [1 if s >= 0.5 else 0 for s in scores]
    assert precision == pytest.approx(sk.precision_score(labels, predicted), abs=TOLERANCE)
    assert recall == pytest.approx(sk.recall_score(labels, predicted), abs=TOLERANCE)
    assert f1 == pytest.approx(sk.f1_score(labels, predicted), abs=TOLERANCE)


# --- 6. degenerate inputs ---------------------------------------------------


def test_roc_auc_is_none_when_only_one_class_is_present() -> None:
    """Returning 0.5 would invent an answer where no pair exists to rank."""
    assert roc_auc([1, 1, 1], [0.1, 0.5, 0.9]) is None
    assert roc_auc([0, 0, 0], [0.1, 0.5, 0.9]) is None


def test_pr_auc_is_none_when_there_are_no_positives() -> None:
    assert pr_auc([0, 0, 0], [0.1, 0.5, 0.9]) is None


def test_recall_and_lift_are_none_when_there_are_no_positives() -> None:
    assert recall_at_k([0, 0], [0.9, 0.1], ["a", "b"], 1) is None
    assert lift_at_k([0, 0], [0.9, 0.1], ["a", "b"], 1) is None


def test_a_single_row_is_handled_without_error() -> None:
    assert precision_at_k([1], [0.5], ["a"], 1) == 1.0
    assert brier([1], [0.5]) == pytest.approx(0.25)


def test_a_perfect_ranking_scores_one_on_both_areas() -> None:
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    assert roc_auc(labels, scores) == 1.0
    assert pr_auc(labels, scores) == 1.0


def test_an_inverted_ranking_scores_zero_on_roc_auc() -> None:
    labels = [1, 1, 0, 0]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc(labels, scores) == 0.0


def test_a_constant_ranking_scores_exactly_one_half_on_roc_auc() -> None:
    """The no-information result, and a useful sanity signal on real data."""
    assert roc_auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == 0.5


# --- 7. input validation ----------------------------------------------------


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(MetricError, match="differ in length"):
        roc_auc([1, 0], [0.5])


def test_a_null_score_is_rejected_rather_than_imputed() -> None:
    with pytest.raises(MetricError, match="never imputes"):
        roc_auc([1, 0], [0.5, None])  # type: ignore[list-item]


def test_a_non_binary_label_is_rejected() -> None:
    with pytest.raises(MetricError, match="0 or 1"):
        roc_auc([1, 2], [0.5, 0.6])


# --- 8. the ranking / probability separation -------------------------------


def test_the_two_metric_families_do_not_overlap() -> None:
    """A rank-only baseline must never be scored on a probability metric.

    The separation is what stops a random shuffle being handed a Brier score to
    make it fit an API.
    """
    assert set(RANKING_METRICS).isdisjoint(PROBABILITY_METRICS)


def test_every_declared_ranking_metric_is_importable() -> None:
    import sentinel.evaluation.metrics as module

    for name in RANKING_METRICS:
        assert hasattr(module, name), name
