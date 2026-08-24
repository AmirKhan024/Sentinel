"""The metrics Component 9 adds, checked against independent references.

``evaluation/metrics.py`` set the precedent: a hand-rolled formula is verified against a
reference implementation rather than trusted. Here the references are a brute-force
:math:`O(n^2)` Kendall oracle, scikit-learn's own isotonic and logistic estimators, and --
for the Brier decomposition -- the algebraic identity it is supposed to satisfy.

The decomposition test is the important one. ``BS = REL - RES + UNC`` is exact only for a
forecast that is constant within each bin, and Component 9 bins continuous probabilities, so
the identity does *not* hold without the residual term. Asserting the full four-term identity
is what stops the component from quietly reporting a recomposition as the Brier score.
"""

from __future__ import annotations

import math
import random

import pytest

from sentinel.calibration import metrics
from sentinel.calibration.preprocess import expit, logit
from sentinel.evaluation.metrics import DEFAULT_CALIBRATION_BINS, MetricError, brier


def _population(n: int = 1200, slope: float = 0.5, seed: int = 3) -> tuple[list[int], list[float]]:
    """An overconfident forecaster: claims ``expit(z)``, truth is ``expit(slope * z)``."""
    rng = random.Random(seed)
    labels: list[int] = []
    probabilities: list[float] = []
    for _ in range(n):
        z = rng.gauss(0.0, 2.0)
        labels.append(1 if rng.random() < expit(slope * z) else 0)
        probabilities.append(expit(z))
    return labels, probabilities


# --- Brier decomposition -----------------------------------------------------


def test_the_decomposition_identity_holds_exactly_including_the_residual() -> None:
    labels, probabilities = _population()
    d = metrics.brier_decomposition(labels, probabilities)
    recomposed = d.reliability - d.resolution + d.uncertainty + d.within_bin_variance
    assert abs(brier(labels, probabilities) - recomposed) < 1e-12


def test_the_residual_is_not_zero_for_a_continuous_forecast() -> None:
    """The reason the residual is a reported column rather than an assumed zero."""
    labels, probabilities = _population()
    d = metrics.brier_decomposition(labels, probabilities)
    assert d.within_bin_variance > 1e-6
    assert abs(brier(labels, probabilities) - d.recomposed) > 1e-6


def test_a_perfectly_calibrated_forecast_has_near_zero_reliability() -> None:
    """Reliability is what calibration fixes, so a calibrated forecast should have little."""
    labels, probabilities = _population(n=4000, slope=1.0, seed=17)
    d = metrics.brier_decomposition(labels, probabilities)
    assert d.reliability < 0.005


def test_a_constant_forecast_that_separates_nothing_has_near_zero_resolution() -> None:
    """Resolution measures separation, so a forecast that separates nothing scores ~0.

    The labels are interleaved deliberately. With a *constant* forecast every score ties,
    so equal-mass bins are cut by position rather than by value -- and a fixture whose
    labels are sorted would put all the positives in the first bins and report a large
    resolution that is an artefact of row order, not of the forecast. Real base scores are
    almost all distinct (xgboost: 39,677 of 41,536), so this does not arise in production,
    and the canonical row sort makes it deterministic where it could.
    """
    rng = random.Random(20260824)
    labels = [1] * 300 + [0] * 700
    rng.shuffle(labels)
    d = metrics.brier_decomposition(labels, [0.3] * 1000)
    assert d.resolution == pytest.approx(0.0, abs=0.01)
    assert d.uncertainty == pytest.approx(0.3 * 0.7, abs=1e-12)
    assert d.reliability == pytest.approx(0.0, abs=0.01)


def test_uncertainty_depends_only_on_the_labels() -> None:
    """It is a property of the data. No forecast can change it."""
    labels, probabilities = _population()
    first = metrics.brier_decomposition(labels, probabilities).uncertainty
    second = metrics.brier_decomposition(labels, [0.5] * len(labels)).uncertainty
    assert first == pytest.approx(second, abs=1e-12)


def test_the_decomposition_uses_the_projects_bin_count() -> None:
    labels, probabilities = _population()
    assert metrics.brier_decomposition(labels, probabilities).n_bins == DEFAULT_CALIBRATION_BINS


def test_the_decomposition_refuses_an_empty_set() -> None:
    with pytest.raises(MetricError):
        metrics.brier_decomposition([], [])


# --- calibration slope and intercept -----------------------------------------


def test_the_slope_recovers_a_planted_overconfidence() -> None:
    labels, probabilities = _population(n=6000, slope=0.5, seed=5)
    result = metrics.calibration_slope_intercept(labels, probabilities)
    assert result.slope is not None
    assert result.slope == pytest.approx(0.5, abs=0.06)


def test_a_calibrated_forecast_has_slope_one_and_intercept_zero() -> None:
    labels, probabilities = _population(n=6000, slope=1.0, seed=8)
    result = metrics.calibration_slope_intercept(labels, probabilities)
    assert result.slope == pytest.approx(1.0, abs=0.1)
    assert result.intercept == pytest.approx(0.0, abs=0.1)


def test_the_slope_is_none_on_a_single_class_window() -> None:
    """``None`` rather than an invented number, matching ``roc_auc``'s posture."""
    result = metrics.calibration_slope_intercept([1] * 50, [0.6] * 50)
    assert result.slope is None
    assert result.intercept is None


def test_the_slope_matches_a_hand_fitted_logistic_on_the_logit() -> None:
    """The estimator is the Cox recalibration regression, not something adjacent to it."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    labels, probabilities = _population()
    reference = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000, fit_intercept=True)
    reference.fit(np.asarray([logit(p) for p in probabilities]).reshape(-1, 1), np.asarray(labels))

    result = metrics.calibration_slope_intercept(labels, probabilities)
    assert result.slope == pytest.approx(float(reference.coef_[0][0]), abs=1e-9)
    assert result.intercept == pytest.approx(float(reference.intercept_[0]), abs=1e-9)


# --- rank correlations -------------------------------------------------------


def _tau_brute(x: list[float], y: list[float]) -> float:
    """The textbook O(n^2) definition, as an oracle for the O(n log n) implementation."""
    concordant = discordant = tied_x = tied_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            a = (x[i] > x[j]) - (x[i] < x[j])
            b = (y[i] > y[j]) - (y[i] < y[j])
            if a == 0 and b == 0:
                continue
            if a == 0:
                tied_x += 1
            elif b == 0:
                tied_y += 1
            elif a * b > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tied_x) * (concordant + discordant + tied_y)
    )
    return (concordant - discordant) / denominator


def test_kendall_tau_b_matches_a_brute_force_oracle_on_heavily_tied_data() -> None:
    """Ties are the whole reason tau-b is used, so the fixtures are mostly ties."""
    rng = random.Random(20260824)
    checked = 0
    for _ in range(400):
        n = rng.randint(2, 14)
        x = [rng.choice([0.1, 0.2, 0.3, 0.4]) for _ in range(n)]
        y = [rng.choice([1.0, 2.0, 3.0]) for _ in range(n)]
        got = metrics.kendall_tau_b(x, y)
        try:
            want = _tau_brute(x, y)
        except ZeroDivisionError:
            continue
        if got is None:
            continue
        assert got == pytest.approx(want, abs=1e-12), (x, y)
        checked += 1
    assert checked > 200, f"only {checked} usable fixtures; the oracle comparison would be weak"


def test_kendall_tau_b_is_one_for_a_strictly_increasing_transform() -> None:
    x = [i / 50 for i in range(1, 50)]
    assert metrics.kendall_tau_b(x, [v**3 for v in x]) == pytest.approx(1.0, abs=1e-12)


def test_spearman_is_one_for_a_monotone_transform_and_minus_one_when_reversed() -> None:
    x = [i / 50 for i in range(1, 50)]
    assert metrics.spearman(x, [math.log(v) for v in x]) == pytest.approx(1.0, abs=1e-12)
    assert metrics.spearman(x, [-v for v in x]) == pytest.approx(-1.0, abs=1e-12)


def test_spearman_is_none_when_a_side_is_constant() -> None:
    assert metrics.spearman([1.0, 2.0, 3.0], [0.5, 0.5, 0.5]) is None


# --- ranking preservation ----------------------------------------------------


def test_a_strictly_monotone_map_preserves_everything() -> None:
    labels, probabilities = _population(n=400)
    ids = [f"{i:05d}" for i in range(len(labels))]
    mapped = [expit(0.5 * logit(p) + 0.1) for p in probabilities]

    result = metrics.ranking_preservation(probabilities, mapped, labels, ids, k=25)
    assert result.inversions == 0
    assert result.new_ties_created == 0
    assert result.spearman_rho == 1.0
    assert result.is_strictly_monotone
    assert result.top_k_membership_changed == 0
    assert result.roc_auc_before == result.roc_auc_after
    assert result.precision_at_k_before == result.precision_at_k_after


def test_a_step_map_creates_ties_without_creating_inversions() -> None:
    """The isotonic signature: ties are counted, and they are not inversions."""
    labels, probabilities = _population(n=400)
    ids = [f"{i:05d}" for i in range(len(labels))]
    stepped = [round(p, 1) for p in probabilities]

    result = metrics.ranking_preservation(probabilities, stepped, labels, ids, k=25)
    assert result.inversions == 0
    assert result.new_ties_created > 0
    assert not result.is_strictly_monotone
    assert result.distinct_after < result.distinct_before


def test_a_reversed_map_is_detected_as_inversions() -> None:
    """The detector has teeth: a non-monotone map must not read as preserved."""
    labels, probabilities = _population(n=200)
    ids = [f"{i:05d}" for i in range(len(labels))]
    reversed_scores = [1 - p for p in probabilities]
    result = metrics.ranking_preservation(probabilities, reversed_scores, labels, ids, k=20)
    assert result.inversions > 0
    assert result.spearman_rho == pytest.approx(-1.0, abs=1e-12)
    assert not result.is_strictly_monotone


def test_ranking_preservation_refuses_mismatched_inputs() -> None:
    with pytest.raises(MetricError):
        metrics.ranking_preservation([0.1, 0.2], [0.1], [0, 1], ["a", "b"], k=1)


# --- bootstrap ---------------------------------------------------------------


def test_the_bootstrap_is_reproducible_from_its_seed_key() -> None:
    labels, probabilities = _population(n=500)
    kwargs = {
        "metric": lambda y, p: brier(y, p),
        "metric_name": "brier",
        "scheme": "row",
        "replications": 200,
    }
    first = metrics.bootstrap(labels, probabilities, seed_key=[1, 2, 3], **kwargs)  # type: ignore[arg-type]
    second = metrics.bootstrap(labels, probabilities, seed_key=[1, 2, 3], **kwargs)  # type: ignore[arg-type]
    assert (first.mean, first.sd, first.lower, first.upper) == (
        second.mean,
        second.sd,
        second.lower,
        second.upper,
    )


def test_a_different_seed_key_gives_an_independent_stream() -> None:
    labels, probabilities = _population(n=500)
    kwargs = {
        "metric": lambda y, p: brier(y, p),
        "metric_name": "brier",
        "scheme": "row",
        "replications": 200,
    }
    first = metrics.bootstrap(labels, probabilities, seed_key=[1, 2, 3], **kwargs)  # type: ignore[arg-type]
    other = metrics.bootstrap(labels, probabilities, seed_key=[1, 2, 4], **kwargs)  # type: ignore[arg-type]
    assert first.mean != other.mean


def test_the_interval_brackets_the_point_estimate() -> None:
    labels, probabilities = _population(n=800)
    result = metrics.bootstrap(
        labels,
        probabilities,
        metric=lambda y, p: brier(y, p),
        metric_name="brier",
        scheme="row",
        seed_key=[20260824, 0, 0, 0, 0],
        replications=400,
    )
    assert result.lower is not None and result.upper is not None
    assert result.point_estimate is not None
    assert result.lower < result.point_estimate < result.upper


def test_the_block_scheme_needs_groups_and_uses_them() -> None:
    labels, probabilities = _population(n=600)
    with pytest.raises(MetricError, match="group labels"):
        metrics.bootstrap(
            labels, probabilities, metric=lambda y, p: brier(y, p), metric_name="brier",
            scheme="establishment_block", seed_key=[1], replications=10,
        )

    groups = [f"E{i % 40}" for i in range(len(labels))]
    result = metrics.bootstrap(
        labels, probabilities, metric=lambda y, p: brier(y, p), metric_name="brier",
        scheme="establishment_block", seed_key=[1], groups=groups, replications=200,
    )
    assert result.scheme == "establishment_block"
    assert result.lower is not None


def test_a_single_class_window_is_counted_as_degenerate_not_imputed() -> None:
    """Undefined is reported as undefined; 0.5 is never substituted."""
    result = metrics.bootstrap(
        [1] * 200,
        [0.6] * 200,
        metric=lambda y, p: metrics.calibration_slope_intercept(y, p).slope,
        metric_name="calibration_slope",
        scheme="row",
        seed_key=[7],
        replications=50,
    )
    assert result.degenerate == 50
    assert result.lower is None and result.upper is None and result.mean is None


def test_an_unknown_scheme_is_rejected() -> None:
    with pytest.raises(MetricError, match="unknown bootstrap scheme"):
        metrics.bootstrap(
            [0, 1], [0.4, 0.6], metric=lambda y, p: brier(y, p), metric_name="brier",
            scheme="jackknife", seed_key=[1], replications=5,
        )


def test_the_vectorised_bootstrap_agrees_with_component_5_to_the_last_few_bits() -> None:
    """The vectorised resampler is a twin of Component 5's metrics, not a second definition.

    The bootstrap recomputes ECE, Brier and log-loss ~12,000 times per fold, which in pure
    Python costs about two and a half hours across the production run. The batched numpy
    path exists for that reason alone, so it is only legitimate if it gives the same number.

    It is **not bit-identical**, and the reason is worth stating rather than papering over:
    numpy sums pairwise while Python's ``sum`` goes left to right, so the two differ in the
    last bit or two of a mean over ~900 terms. Measured here at ~1 ULP. The binning,
    ordering and tie handling *are* identical -- which is what this test is really checking,
    because those are the parts a reimplementation gets wrong.

    Component 5's functions remain the only ones used for every **reported point estimate**;
    the batched path is used solely for resamples, where a 1e-16 difference cannot reach a
    percentile.
    """
    import numpy as np

    from sentinel.calibration.metrics import _batch_metrics
    from sentinel.evaluation.metrics import LOG_LOSS_EPSILON, ece, log_loss

    rng = random.Random(4242)
    n = 900
    labels = [1 if rng.random() < 0.42 else 0 for _ in range(n)]
    # Deliberate ties: rounding to 3 places makes duplicate scores common, which is where
    # a binning implementation is most likely to disagree.
    probabilities = [round(rng.random(), 3) for _ in range(n)]

    draw = np.random.default_rng(7).integers(0, n, size=(40, n))
    batch = _batch_metrics(
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        draw,
        n_bins=DEFAULT_CALIBRATION_BINS,
        epsilon=LOG_LOSS_EPSILON,
    )

    worst = 0.0
    for row in range(draw.shape[0]):
        index = [int(i) for i in draw[row]]
        drawn_labels = [labels[i] for i in index]
        drawn_probabilities = [probabilities[i] for i in index]
        for name, reference in (
            ("ece", ece(drawn_labels, drawn_probabilities)),
            ("brier", brier(drawn_labels, drawn_probabilities)),
            ("log_loss", log_loss(drawn_labels, drawn_probabilities)),
        ):
            got = float(batch[name][row])
            worst = max(worst, abs(got - reference))
            assert got == pytest.approx(reference, rel=1e-12, abs=1e-12), name

    # A bound, not a vibe: if a future change made the two implementations genuinely differ,
    # the discrepancy would be orders of magnitude above float noise and this would catch it.
    assert worst < 1e-13, f"vectorised path drifted from Component 5 by {worst:.3e}"
