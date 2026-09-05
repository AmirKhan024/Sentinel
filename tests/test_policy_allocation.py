"""Queue construction: the two mechanisms, capacity, disjointness and the ranks.

The arithmetic at the centre of Component 13. Every test here works on a hand-built window
small enough to check by eye, because an allocation bug that only shows up on a real fold is
one nobody can reason about.
"""

from __future__ import annotations

import pytest

from sentinel.policy.allocation import (
    AllocationError,
    allocate,
    decide,
    model_ranks,
    reserve_target,
    risk_order,
)
from sentinel.policy.definitions import (
    DecisionMechanism,
    DecisionReason,
    PolicySpec,
    ReserveMechanism,
)
from tests.conftest import make_policy_window

PURE = PolicySpec("pure_risk", ReserveMechanism.NONE, 0.0, "baseline")
FLOOR_20 = PolicySpec("floor", ReserveMechanism.FLOOR, 0.20, "floor at a fifth")
FORCED_20 = PolicySpec("forced", ReserveMechanism.FORCED, 0.20, "spend a fifth")


# --- 1. the canonical order ------------------------------------------------------


def test_the_order_is_descending_score_then_ascending_id() -> None:
    """Component 5's rule, reused. The direction is asserted rather than assumed."""
    window = make_policy_window(
        scores=[0.1, 0.9, 0.5], ids=["T3", "T1", "T2"], eligible=[False, False, False]
    )
    assert [window.ids[i] for i in risk_order(window)] == ["T1", "T2", "T3"]


def test_a_tie_is_broken_on_the_id_not_on_the_row_order() -> None:
    """The property that makes the whole queue independent of Parquet row order."""
    ascending = make_policy_window(scores=[0.5, 0.5, 0.5], ids=["T1", "T2", "T3"])
    descending = make_policy_window(scores=[0.5, 0.5, 0.5], ids=["T3", "T2", "T1"])
    assert [ascending.ids[i] for i in risk_order(ascending)] == ["T1", "T2", "T3"]
    assert [descending.ids[i] for i in risk_order(descending)] == ["T1", "T2", "T3"]


def test_model_ranks_are_one_based_and_cover_every_row() -> None:
    window = make_policy_window(scores=[0.1, 0.9, 0.5], ids=["T3", "T1", "T2"])
    assert sorted(model_ranks(window)) == [1, 2, 3]
    assert model_ranks(window)[1] == 1  # T1 has the highest score


# --- 2. the reserve target -------------------------------------------------------


def test_the_target_truncates_so_a_reserve_never_overspends_its_share() -> None:
    """Rounding up would let a 20% reserve take 3 of 12 slots and call it 20%."""
    assert reserve_target(FORCED_20, 12) == 2
    assert reserve_target(FORCED_20, 5) == 1
    assert reserve_target(FORCED_20, 4) == 0


def test_the_null_policy_asks_for_nothing_at_any_capacity() -> None:
    assert reserve_target(PURE, 1000) == 0


# --- 3. pure risk ----------------------------------------------------------------


def test_pure_risk_selects_exactly_the_top_k() -> None:
    window = make_policy_window(scores=[0.9, 0.8, 0.7, 0.6], eligible=[False] * 4)
    allocation = allocate(window, PURE, k_name="k_1_day", k=2)
    assert allocation.n_selected == 2
    assert allocation.n_reserve == 0
    assert [window.ids[i] for i in allocation.risk_indices] == ["T00000", "T00001"]


def test_capacity_larger_than_the_window_is_clamped_not_exceeded() -> None:
    """A schedule can never work more inspections than the window holds."""
    window = make_policy_window(scores=[0.9, 0.8])
    allocation = allocate(window, PURE, k_name="k_1_week", k=50)
    assert allocation.k == 2
    assert allocation.n_selected == 2


def test_a_capacity_below_one_is_refused() -> None:
    window = make_policy_window(scores=[0.9])
    with pytest.raises(AllocationError, match="at least 1"):
        allocate(window, PURE, k_name="k_1_day", k=0)


# --- 4. the floor: a guarantee, not a spend ---------------------------------------


def test_the_floor_is_inert_when_risk_already_clears_it() -> None:
    """Component 13's headline result, in miniature.

    Four of the top five by risk are coverage-eligible, so a floor asking for one is already
    satisfied and grants nothing. This is why the reserve is inert in 84 of 85 quarterly cells
    at the population share: the risk ranking over-selects this population, it does not
    neglect it.
    """
    window = make_policy_window(
        scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        eligible=[True, True, True, True, False, True],
    )
    allocation = allocate(window, FLOOR_20, k_name="k_1_day", k=5)
    assert allocation.reserve_target == 1
    assert allocation.n_eligible_in_risk_top_k == 4
    assert allocation.n_reserve == 0
    assert allocation.reserve_inert is True
    assert allocation.n_risk == 5


def test_the_floor_binds_when_risk_does_not_clear_it() -> None:
    """The case that justifies keeping the mechanism: no eligible row in the top five."""
    window = make_policy_window(
        scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        eligible=[False, False, False, False, False, True],
    )
    allocation = allocate(window, FLOOR_20, k_name="k_1_day", k=5)
    assert allocation.reserve_target == 1
    assert allocation.n_eligible_in_risk_top_k == 0
    assert allocation.n_reserve == 1
    assert allocation.n_risk == 4
    assert [window.ids[i] for i in allocation.reserve_indices] == ["T00005"]
    assert allocation.n_selected == 5


def test_the_floor_grants_only_the_shortfall() -> None:
    """Two asked for, one already in the risk block, so one granted -- never two."""
    window = make_policy_window(
        scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
        eligible=[True, False, False, False, False, False, False, False, True, True],
    )
    spec = PolicySpec("floor2", ReserveMechanism.FLOOR, 0.20, "floor")
    allocation = allocate(window, spec, k_name="k_1_day", k=10)
    assert allocation.reserve_target == 2
    assert allocation.n_eligible_in_risk_top_k == 3
    assert allocation.n_reserve == 0


# --- 5. the forced reserve: a spend ------------------------------------------------


def test_the_forced_reserve_spends_its_allocation_even_when_risk_already_covers() -> None:
    """The mechanism most people mean by 'reserve some capacity', and the one that costs."""
    window = make_policy_window(
        scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        eligible=[True, True, True, True, False, True],
    )
    allocation = allocate(window, FORCED_20, k_name="k_1_day", k=5)
    assert allocation.n_reserve == 1
    assert allocation.n_risk == 4
    # The displaced row is the last one that would have fitted, and the reserve takes the
    # highest-scored eligible row that the shortened risk block no longer holds.
    assert [window.ids[i] for i in allocation.reserve_indices] == ["T00005"]
    assert window.ids[4] not in [window.ids[i] for i in allocation.risk_indices]


def test_the_reserve_cannot_exceed_the_eligible_rows_that_exist() -> None:
    """Asking for three when one is available grants one, and never fabricates a row."""
    window = make_policy_window(
        scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
        eligible=[False] * 9 + [True],
    )
    spec = PolicySpec("forced3", ReserveMechanism.FORCED, 0.30, "spend a third")
    allocation = allocate(window, spec, k_name="k_1_day", k=10)
    assert allocation.reserve_target == 3
    assert allocation.n_reserve == 1
    assert allocation.n_selected == 10


def test_a_reserve_asking_for_rows_already_all_inside_the_risk_block_grants_none() -> None:
    """The solved-rather-than-assumed case: shrinking the risk block frees nothing."""
    window = make_policy_window(scores=[0.9, 0.8, 0.7], eligible=[True, True, True])
    spec = PolicySpec("forced_all", ReserveMechanism.FORCED, 0.60, "spend most of it")
    allocation = allocate(window, spec, k_name="k_1_day", k=3)
    assert allocation.n_selected == 3
    assert allocation.n_reserve + allocation.n_risk == 3


# --- 6. the invariants every allocation must hold ----------------------------------


@pytest.mark.parametrize("spec", [PURE, FLOOR_20, FORCED_20])
@pytest.mark.parametrize("k", [1, 3, 5, 8])
def test_the_two_mechanisms_never_overlap_and_always_fill_capacity(
    spec: PolicySpec, k: int
) -> None:
    """Disjointness and capacity, over a grid rather than at one point.

    The reserve is filled from rows the risk block did not take, so this holds by
    construction -- which is a claim about code that was correct when it was written.
    """
    window = make_policy_window(
        scores=[0.9, 0.85, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3],
        eligible=[True, False, True, False, False, True, False, True],
    )
    allocation = allocate(window, spec, k_name="k_1_day", k=k)
    assert not set(allocation.risk_indices) & set(allocation.reserve_indices)
    assert allocation.n_selected == min(k, window.n)
    assert allocation.n_risk + allocation.n_reserve == allocation.k
    assert allocation.n_reserve <= allocation.reserve_target


def test_every_reserve_selection_is_eligible() -> None:
    window = make_policy_window(
        scores=[0.9, 0.8, 0.7, 0.6, 0.5], eligible=[False, False, False, True, True]
    )
    allocation = allocate(window, FORCED_20, k_name="k_1_day", k=4)
    assert all(window.eligible[i] for i in allocation.reserve_indices)


# --- 7. the decision codes ---------------------------------------------------------


def test_ranks_are_contiguous_with_the_risk_block_first() -> None:
    """Six rows, five slots, a fifth reserved: four risk ranks then one reserve rank."""
    window = make_policy_window(
        scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        eligible=[False, False, False, False, False, True],
    )
    allocation = allocate(window, FORCED_20, k_name="k_1_day", k=5)
    _mechanisms, _reasons, ranks = decide(window, allocation)
    selected = sorted(r for r in ranks if r is not None)
    assert selected == [1, 2, 3, 4, 5]
    # The reserve row takes the last rank, after the shortened risk block -- and the row it
    # displaced, index 4, is the last one that would have fitted on risk alone.
    assert ranks[5] == 5
    assert ranks[4] is None


def test_an_unselected_row_carries_no_rank() -> None:
    window = make_policy_window(scores=[0.9, 0.8, 0.7])
    allocation = allocate(window, PURE, k_name="k_1_day", k=1)
    _mechanisms, _reasons, ranks = decide(window, allocation)
    assert ranks[1] is None and ranks[2] is None


def test_an_unselected_eligible_row_says_the_reserve_did_not_reach_it() -> None:
    """ "You were outranked" and "the reserve ran out" are different facts about one row.

    Only the second is a statement about this component's policy, and a reviewer asking
    "was the reserve too small?" should be able to answer it from the artifact.
    """
    window = make_policy_window(scores=[0.9, 0.8, 0.7, 0.6], eligible=[False, False, True, False])
    allocation = allocate(window, PURE, k_name="k_1_day", k=2)
    _mechanisms, reasons, _ranks = decide(window, allocation)
    assert reasons[2] == DecisionReason.NOT_SELECTED_RESERVE_EXHAUSTED
    assert reasons[3] == DecisionReason.NOT_SELECTED_CAPACITY_EXHAUSTED


def test_the_mechanism_and_the_reason_always_agree() -> None:
    window = make_policy_window(
        scores=[0.9, 0.8, 0.7, 0.6, 0.5], eligible=[False, False, False, False, True]
    )
    allocation = allocate(window, FORCED_20, k_name="k_1_day", k=4)
    mechanisms, reasons, ranks = decide(window, allocation)
    for index in range(window.n):
        if mechanisms[index] == DecisionMechanism.RISK_PRIORITY:
            assert reasons[index] == DecisionReason.SELECTED_BY_RISK_RANK
            assert ranks[index] is not None
        elif mechanisms[index] == DecisionMechanism.COVERAGE_RESERVE:
            assert reasons[index] == DecisionReason.SELECTED_BY_COVERAGE_RESERVE
            assert ranks[index] is not None
        else:
            assert ranks[index] is None
