"""Group-conditional metrics: hand-computed examples, and agreement with the canonical ones.

Two kinds of test here and the split is deliberate.

**Every metric this component did NOT write is cross-checked against the implementation that
owns it.** A group ROC-AUC must be Component 5's ROC-AUC applied to a subset, not a second
implementation that happens to agree today. Two implementations that agree now are two that
can disagree later, inside the same comparison, for reasons no reader could diagnose.

**Every metric this component DID write is checked against arithmetic done by hand.** Capture
rate, selection-rate ratio and the four disparity measures are new here, so each has a worked
example small enough to verify by reading it -- and each has its zero-denominator case pinned,
because null-versus-zero is where a fairness table misleads most easily.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from sentinel.evaluation import metrics as canonical
from sentinel.evaluation import simulate
from sentinel.fairness import metrics as m
from sentinel.fairness.definitions import GROUP_CALIBRATION_BINS, MetricKind

# --- 1. the canonical metrics are reused, not reimplemented -------------------


def test_group_roc_auc_is_component_fives_roc_auc_on_the_subset() -> None:
    labels = [0, 1, 0, 1, 1, 0, 1, 0]
    scores = [0.1, 0.9, 0.2, 0.8, 0.6, 0.3, 0.7, 0.05]
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(8)]
    ids = [f"T{i:03d}" for i in range(8)]

    values = m.ranking_metrics(labels, scores, dates, ids)
    assert values["roc_auc"] == canonical.roc_auc(labels, scores)
    assert values["pr_auc"] == canonical.pr_auc(labels, scores)


def test_group_probability_metrics_are_component_fives() -> None:
    labels = [0, 1, 1, 0, 1, 0, 0, 1]
    probabilities = [0.2, 0.7, 0.9, 0.3, 0.6, 0.1, 0.4, 0.8]

    values = m.probability_metrics(labels, probabilities)
    assert values["brier"] == canonical.brier(labels, probabilities)
    assert values["log_loss"] == canonical.log_loss(labels, probabilities)
    assert values["ece"] == canonical.ece(labels, probabilities, n_bins=GROUP_CALIBRATION_BINS)
    assert values["mce"] == canonical.mce(labels, probabilities, n_bins=GROUP_CALIBRATION_BINS)


def test_group_nde_is_built_from_component_fives_simulation_pieces() -> None:
    labels = [1, 0, 1, 0, 1, 0]
    scores = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3]
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(6)]
    ids = [f"T{i:03d}" for i in range(6)]

    window = simulate.build_window(ids=ids, labels=labels, dates=dates)
    order = simulate.model_order(window, scores)
    cumulative = simulate.discovery_curve(window, order)
    area = simulate.normalized_area(cumulative, n=window.n, positives=window.positives)
    expected = simulate.normalized_discovery_efficiency(
        area, n=window.n, positives=window.positives
    )

    assert m.ranking_metrics(labels, scores, dates, ids)["nde"] == expected


def test_scores_are_reordered_into_the_windows_canonical_order() -> None:
    """`build_window` sorts by (date, id); the caller's order is no longer the window's.

    Handing the unsorted scores to `model_order` attaches every score to the wrong row and
    produces a perfectly plausible NDE, because a permutation of scores is still a valid
    ranking. This is the group-level analogue of Component 11's name-recovery trap.
    """
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.1, 0.8, 0.2]
    ids = ["T003", "T002", "T001", "T000"]
    # Dates descending, so the window's canonical order reverses the caller's.
    dates = [date(2024, 1, 4), date(2024, 1, 3), date(2024, 1, 2), date(2024, 1, 1)]

    perfect = m.ranking_metrics(labels, scores, dates, ids)["nde"]
    # The same rows with the two positives scored lowest must do strictly worse.
    inverted = m.ranking_metrics(labels, [0.1, 0.9, 0.2, 0.8], dates, ids)["nde"]
    assert perfect is not None and inverted is not None
    assert perfect > inverted


def test_a_single_class_group_gets_none_rather_than_a_substitute() -> None:
    """0.5 would read as a measured coin flip. There is nothing to separate."""
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(4)]
    ids = [f"T{i:03d}" for i in range(4)]
    values = m.ranking_metrics([1, 1, 1, 1], [0.4, 0.5, 0.6, 0.7], dates, ids)
    assert values["roc_auc"] is None
    assert values["nde"] is None


def test_threshold_metrics_reuse_component_fives_top_k_and_tie_break() -> None:
    labels = [1, 0, 1, 0, 1, 0]
    scores = [0.9, 0.9, 0.8, 0.8, 0.1, 0.1]
    ids = ["T005", "T004", "T003", "T002", "T001", "T000"]

    values = m.threshold_metrics(labels, scores, ids, k=2)
    assert values["precision_at_k"] == canonical.precision_at_k(labels, scores, ids, 2)
    assert values["recall_at_k"] == canonical.recall_at_k(labels, scores, ids, 2)
    assert values["lift_at_k"] == canonical.lift_at_k(labels, scores, ids, 2)


def test_confusion_rates_at_k_match_arithmetic_done_by_hand() -> None:
    """Six rows, three positive. The top 3 by score are T005(1), T004(0), T003(1).

    true positives   2 of 3 positives   -> TPR 2/3
    false positives  1 of 3 negatives   -> FPR 1/3
    false discovery  1 of 3 selected    -> FDR 1/3
    """
    labels = [1, 0, 1, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    ids = ["T005", "T004", "T003", "T002", "T001", "T000"]

    values = m.threshold_metrics(labels, scores, ids, k=3)
    assert values["true_positive_rate"] == pytest.approx(2 / 3)
    assert values["false_positive_rate"] == pytest.approx(1 / 3)
    assert values["false_discovery_rate"] == pytest.approx(1 / 3)


# --- 2. capture rate: the metric that is not recall_at_k ----------------------


def test_capture_rate_is_captured_over_the_groups_own_positives() -> None:
    assert m.capture_rate(10, 3) == pytest.approx(0.3)
    assert m.capture_rate(4, 4) == 1.0


def test_a_group_with_no_positives_has_no_capture_rate() -> None:
    """None, never 0.0. 0.0 reads as total failure rather than nothing to capture."""
    assert m.capture_rate(0, 0) is None


def test_capturing_more_than_a_group_holds_is_rejected() -> None:
    with pytest.raises(m.GroupMetricError, match="captured 5 positives"):
        m.capture_rate(3, 5)


def test_capture_differs_from_recall_at_k_because_the_cutoff_is_city_wide() -> None:
    """recall_at_k selects its top k WITHIN the rows handed to it; capture does not.

    Here the group's three rows would all be selected by a within-group top-3 -- recall 1.0.
    Against a city-wide cutoff that took only one of them, capture is 1/2. Conflating the two
    would hide exactly the effect this audit exists to find.
    """
    group_labels = [1, 1, 0]
    group_scores = [0.30, 0.20, 0.10]
    group_ids = ["T002", "T001", "T000"]

    within_group = canonical.recall_at_k(group_labels, group_scores, group_ids, 3)
    assert within_group == 1.0

    # City-wide, only the group's best row cleared the cutoff.
    assert m.capture_rate(2, 1) == pytest.approx(0.5)


# --- 3. selection-rate ratio ---------------------------------------------------


def test_selection_rate_ratio_is_one_when_selected_in_proportion() -> None:
    # 10 of the group's 100 rows selected; 100 of the population's 1,000. Both 10%.
    assert m.selection_rate_ratio(10, 100, 100, 1000) == pytest.approx(1.0)


def test_selection_rate_ratio_above_one_means_over_represented() -> None:
    # 20 of 100 in the group (20%) against 100 of 1,000 overall (10%).
    assert m.selection_rate_ratio(20, 100, 100, 1000) == pytest.approx(2.0)


def test_selection_rate_ratio_is_none_rather_than_infinite_on_a_zero_denominator() -> None:
    """ "Nothing to select from" and "selected at an enormous rate" must stay distinct."""
    assert m.selection_rate_ratio(0, 0, 100, 1000) is None
    assert m.selection_rate_ratio(5, 100, 0, 1000) is None
    assert m.selection_rate_ratio(5, 100, 100, 0) is None


# --- 4. the disparity measures --------------------------------------------------


def test_spread_is_max_minus_min() -> None:
    assert m.spread([0.2, 0.5, 0.9]) == pytest.approx(0.7)


def test_ratio_is_max_over_min() -> None:
    assert m.ratio([0.2, 0.5, 0.8]) == pytest.approx(4.0)


def test_ratio_is_none_rather_than_infinite_when_the_minimum_is_zero() -> None:
    """A vanished denominator is not an infinite disparity."""
    assert m.ratio([0.0, 0.5]) is None
    assert m.ratio([-0.1, 0.5]) is None


def test_a_disparity_over_fewer_than_two_groups_is_undefined() -> None:
    assert m.spread([0.4]) is None
    assert m.ratio([0.4]) is None
    assert m.weighted_sd([0.4], [100]) is None


def test_max_deviation_is_measured_from_the_pooled_reference() -> None:
    # Reference 0.50; the furthest group is 0.90, so 0.40 -- not 0.90 - 0.20.
    assert m.max_deviation([0.2, 0.5, 0.9], 0.5) == pytest.approx(0.4)


def test_max_deviation_is_none_without_a_reference() -> None:
    assert m.max_deviation([0.2, 0.9], None) is None


def test_weighted_sd_weights_by_rows_and_matches_hand_arithmetic() -> None:
    """Two groups: 0.2 on 300 rows and 0.6 on 100 rows.

    mean     = (0.2*300 + 0.6*100) / 400 = 0.30
    variance = (300*0.01 + 100*0.09) / 400 = (3 + 9) / 400 = 0.03
    sd       = sqrt(0.03)
    """
    assert m.weighted_sd([0.2, 0.6], [300, 100]) == pytest.approx(0.03**0.5)


def test_weighted_sd_differs_from_the_unweighted_one() -> None:
    """Otherwise a 200-row group and a 2,600-row group would have equal say."""
    weighted = m.weighted_sd([0.2, 0.6], [300, 100])
    balanced = m.weighted_sd([0.2, 0.6], [200, 200])
    assert weighted is not None and balanced is not None
    assert weighted != pytest.approx(balanced)


def test_mismatched_values_and_weights_are_rejected() -> None:
    with pytest.raises(m.GroupMetricError, match="same length"):
        m.weighted_sd([0.2, 0.6], [300])


# --- 5. did calibration improve this group? -------------------------------------


def test_improved_is_direction_aware_per_metric() -> None:
    assert m.improved("ece", 0.08, 0.05) is True
    assert m.improved("ece", 0.05, 0.08) is False
    assert m.improved("brier", 0.24, 0.23) is True


def test_slope_improvement_is_measured_by_distance_from_one() -> None:
    """0.6 and 1.4 are both miscalibrated; no inequality between them says which is better."""
    assert m.improved("calibration_slope", 0.60, 0.95) is True
    assert m.improved("calibration_slope", 0.95, 0.60) is False
    # Overshooting past 1.0 by more than it started is not an improvement.
    assert m.improved("calibration_slope", 0.90, 1.30) is False


def test_improved_is_none_rather_than_false_when_either_side_is_missing() -> None:
    """ "We could not tell" and "it got worse" are different answers."""
    assert m.improved("ece", None, 0.05) is None
    assert m.improved("ece", 0.05, None) is None


# --- 6. metric families ----------------------------------------------------------


def test_each_metric_is_assigned_the_family_that_gates_it() -> None:
    assert m.kind_of("roc_auc") is MetricKind.RANKING
    assert m.kind_of("ece") is MetricKind.PROBABILITY
    assert m.kind_of("calibration_slope") is MetricKind.PROBABILITY
    assert m.kind_of("false_positive_rate") is MetricKind.THRESHOLD_AUDIT


def test_an_unknown_metric_is_rejected_rather_than_defaulted() -> None:
    with pytest.raises(m.GroupMetricError, match="unknown metric"):
        m.kind_of("fairness_score")
