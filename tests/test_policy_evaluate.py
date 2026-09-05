"""Policy measurement: the metrics, the opportunity cost, the frontier and the missing winner.

The most important test in this file is the first one. Component 13 computes precision, capture
and lift over the queue it built rather than calling Component 5's top-k helpers -- because
those re-derive the top ``k`` from the scores, which is the definition of ``pure_risk``, and
handing a coverage policy's queue to them would silently measure the baseline. That is a
plausible-looking, invisible failure, so the equivalence is proved rather than assumed.
"""

from __future__ import annotations

import pytest

from sentinel.evaluation.metrics import lift_at_k, precision_at_k, recall_at_k
from sentinel.policy.allocation import allocate
from sentinel.policy.definitions import PolicySpec, ReserveMechanism
from sentinel.policy.evaluate import (
    EvaluationError,
    cell_metrics,
    frontier,
    group_audit,
    opportunity_cost,
    schedule_order,
    winner,
)
from tests.conftest import make_policy_window

PURE = PolicySpec("pure_risk", ReserveMechanism.NONE, 0.0, "baseline")
FORCED = PolicySpec("forced", ReserveMechanism.FORCED, 0.20, "spend a fifth")


def _window() -> object:
    return make_policy_window(
        scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
        # Index 4 is a positive and index 6 is not, so displacing the marginal risk row for a
        # reserve row visibly costs a citation. A window where the swap happened to be
        # outcome-neutral would let the two measurement paths agree by accident.
        labels=[1, 0, 1, 1, 1, 1, 0, 0, 1, 0],
        eligible=[False, False, False, False, False, False, True, True, True, True],
    )


# --- 1. the equivalence Component 5 owns --------------------------------------------


@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_pure_risk_metrics_match_component_fives_own_top_k_helpers(k: int) -> None:
    """The check that licenses computing over the queue instead of calling the helpers.

    On ``pure_risk`` the queue *is* the top-k, so the two paths must agree exactly. If they
    ever diverge, the reimplementation has drifted and every coverage policy's number is
    measured against a baseline that is not the baseline.
    """
    window = _window()
    allocation = allocate(window, PURE, k_name="k_1_day", k=k)
    cell = cell_metrics(window, allocation, definition_version="v1")

    labels, scores, ids = list(window.labels), list(window.scores), list(window.ids)
    assert cell["precision_at_k"] == pytest.approx(precision_at_k(labels, scores, ids, k))
    assert cell["capture_rate"] == pytest.approx(recall_at_k(labels, scores, ids, k))
    assert cell["lift_at_k"] == pytest.approx(lift_at_k(labels, scores, ids, k))


def test_a_coverage_queue_is_measured_over_its_own_queue_not_the_top_k() -> None:
    """The failure this design exists to avoid, made visible.

    The forced reserve's queue differs from the top-k, so its precision must differ from
    Component 5's top-k precision. If it did not, the policy's number would be the baseline's.
    """
    window = _window()
    allocation = allocate(window, FORCED, k_name="k_1_day", k=5)
    cell = cell_metrics(window, allocation, definition_version="v1")
    top_k = precision_at_k(list(window.labels), list(window.scores), list(window.ids), 5)
    assert allocation.n_reserve == 1
    assert cell["precision_at_k"] != pytest.approx(top_k)


# --- 2. the schedule NDE integrates over ---------------------------------------------


def test_the_schedule_is_the_queue_then_the_risk_tail() -> None:
    """A policy speaks about the first k rows; the tail keeps the model's ranking.

    Stated as a measurement convention rather than buried, because a different tail convention
    would give the same policy a different NDE and nothing in the policy decides the tail.
    """
    window = _window()
    allocation = allocate(window, FORCED, k_name="k_1_day", k=5)
    order = schedule_order(window, allocation)
    assert len(order) == window.n
    assert sorted(order) == list(range(window.n))
    assert list(order[:5]) == [*allocation.risk_indices, *allocation.reserve_indices]


def test_nde_is_reported_and_finite_on_a_mixed_window() -> None:
    window = _window()
    cell = cell_metrics(
        window, allocate(window, PURE, k_name="k_1_day", k=5), definition_version="v1"
    )
    assert cell["nde"] is not None
    assert -1.0 <= float(cell["nde"]) <= 1.0


def test_a_window_with_no_positives_reports_a_null_nde_rather_than_a_zero() -> None:
    """Zero would mean 'no better than random'. There is nothing to be better than."""
    window = make_policy_window(scores=[0.9, 0.5, 0.1], labels=[0, 0, 0])
    cell = cell_metrics(
        window, allocate(window, PURE, k_name="k_1_day", k=2), definition_version="v1"
    )
    assert cell["nde"] is None
    assert cell["capture_rate"] is None


# --- 3. the opportunity cost ---------------------------------------------------------


def test_the_cost_is_reported_in_citations_and_its_sign_is_not_assumed() -> None:
    window = _window()
    baseline = cell_metrics(
        window, allocate(window, PURE, k_name="k_1_day", k=5), definition_version="v1"
    )
    forced = cell_metrics(
        window, allocate(window, FORCED, k_name="k_1_day", k=5), definition_version="v1"
    )
    delta = opportunity_cost(forced, baseline)
    assert delta["delta_positives"] == pytest.approx(
        float(forced["positives_selected"]) - float(baseline["positives_selected"])
    )
    # The reserve bought coverage: more eligible establishments in the queue.
    assert float(delta["delta_eligible_selected"]) > 0


def test_the_baseline_differenced_against_itself_is_exactly_zero() -> None:
    """A reserve is described as free only where this number is zero, so zero must be exact."""
    window = _window()
    cell = cell_metrics(
        window, allocate(window, PURE, k_name="k_1_day", k=5), definition_version="v1"
    )
    delta = opportunity_cost(cell, cell)
    assert all(value == 0 for value in delta.values() if value is not None)


def test_an_undefined_metric_on_either_side_yields_a_null_delta_not_a_zero() -> None:
    """Zero would claim the policy changed nothing; null says the comparison is undefined."""
    delta = opportunity_cost({"nde": None}, {"nde": 0.2})
    assert delta["delta_nde"] is None


# --- 4. the frontier and the winner ----------------------------------------------------


def _comparison(rows: list[tuple[str, float, float]]) -> list[dict[str, object]]:
    return [
        {
            "policy_id": policy,
            "fold_set": "quarterly",
            "k_name": "k_1_day",
            "positives_selected": positives,
            "eligible_selected": eligible,
        }
        for policy, positives, eligible in rows
    ]


def test_a_policy_beaten_on_every_axis_is_marked_dominated() -> None:
    rows = frontier(
        _comparison([("a", 10.0, 10.0), ("b", 5.0, 5.0)]),
        fold_set="quarterly",
        definition_version="v1",
    )
    by_policy = {row["policy_id"]: row for row in rows}
    assert by_policy["b"]["is_dominated"] is True
    assert by_policy["b"]["dominated_by"] == "a"
    assert by_policy["a"]["is_dominated"] is False


def test_a_genuine_trade_off_leaves_both_policies_on_the_frontier() -> None:
    """More citations but less coverage, against less citations but more coverage.

    Neither dominates, and combining them into one score would require an exchange rate
    between a missed Priority citation and an uninspected establishment with no history.
    """
    rows = frontier(
        _comparison([("a", 10.0, 5.0), ("b", 5.0, 10.0)]),
        fold_set="quarterly",
        definition_version="v1",
    )
    assert all(row["is_dominated"] is False for row in rows)


def test_no_winner_is_named_when_several_policies_survive() -> None:
    """The expected outcome, and an acceptable conclusion rather than a failure to conclude."""
    rows = frontier(
        _comparison([("a", 10.0, 5.0), ("b", 5.0, 10.0)]),
        fold_set="quarterly",
        definition_version="v1",
    )
    assert winner(rows, k_name="k_1_day") is None


def test_a_winner_is_named_only_when_it_is_the_unique_survivor() -> None:
    rows = frontier(
        _comparison([("a", 10.0, 10.0), ("b", 5.0, 5.0), ("c", 1.0, 1.0)]),
        fold_set="quarterly",
        definition_version="v1",
    )
    assert winner(rows, k_name="k_1_day") == "a"


def test_the_frontier_ignores_a_fold_set_it_was_not_asked_about() -> None:
    """The covid fold is never pooled with the quarterly one."""
    rows = _comparison([("a", 10.0, 10.0)])
    rows.append(
        {
            "policy_id": "a",
            "fold_set": "covid_shift",
            "k_name": "k_1_day",
            "positives_selected": 999.0,
            "eligible_selected": 999.0,
        }
    )
    out = frontier(rows, fold_set="quarterly", definition_version="v1")
    assert out[0]["positives_selected"] == pytest.approx(10.0)


# --- 5. the descriptive group audit ------------------------------------------------------


def test_the_group_audit_carries_component_twelves_status_unchanged() -> None:
    """An unsupported group stays unsupported. Nothing here filters on the status."""
    window = _window()
    allocation = allocate(window, PURE, k_name="k_1_day", k=5)
    rows = group_audit(
        window,
        allocation,
        groups=["1"] * 5 + ["__UNKNOWN__"] * 5,
        support={"1": "supported"},
        model_name="model_a",
        definition_version="v1",
    )
    statuses = {row["group_value"]: row["group_status"] for row in rows}
    assert statuses["1"] == "supported"
    assert statuses["__UNKNOWN__"] == "insufficient_support"
    assert len(rows) == 2


def test_the_group_audit_refuses_a_label_vector_of_the_wrong_length() -> None:
    """A misaligned join would attribute one establishment's decision to another's neighbourhood."""
    window = _window()
    allocation = allocate(window, PURE, k_name="k_1_day", k=5)
    with pytest.raises(EvaluationError, match="group labels"):
        group_audit(
            window,
            allocation,
            groups=["1"],
            support={},
            model_name="model_a",
            definition_version="v1",
        )
