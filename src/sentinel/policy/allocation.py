"""Queue construction: risk block, coverage reserve, and the ranks that come out. Pure.

**Everything here is allocation. Nothing here is estimation.** No score is read except to
order rows, and no score is written at all. That is the sentence the component exists to make
literally true, and it is why a reserve moves an establishment up the queue without moving its
probability: the model's belief about an establishment is not changed by a decision about
capacity, and a system that blurred the two would be unable to explain either.

**The reserve is filled after the risk block, from what the risk block did not take.** The two
mechanisms are therefore disjoint by construction rather than by a de-duplication pass, and no
establishment can be selected twice. ``validate.py`` asserts the disjointness anyway.

**Ordering inside the reserve is calibrated risk, among the eligible.** This is a deliberate
choice with a cost, and both halves are stated rather than one. Using risk means the reserve
spends its slots on the eligible establishments the model likes most, which is the best
available use of them -- Component 12 measured the model's ranking *inside* this population at
roughly chance, so "best available" is doing modest work. The alternative, a canonical
date-and-id order, would be a lottery, and a lottery is defensible only if one believes the
ranking carries no information at all, which is a stronger claim than the measurement supports.
The reserve changes **which** establishments are inspected, never **how risky** they are held
to be.

**Floors and forced reserves are different policies.** A floor asks whether the outcome has
already been achieved and does nothing when it has. A forced reserve spends its allocation
regardless. Profile 7 measured that the difference is almost the whole of Component 13's
result: at the population share the floor is inert in 84 of 85 quarterly cells on the
production model, while the forced reserve at the same share moves real capacity -- 274 slots
at a week of capacity, for a measured cost of 15 Priority citations. The forced reserve's sign
is not uniform across cutoffs (+2 at a day, -20 at the 5% cutoff), which is why the price is
reported per operating point rather than as one number.
"""

from __future__ import annotations

from functools import lru_cache

from sentinel.evaluation.metrics import top_k_indices
from sentinel.policy.definitions import (
    DecisionMechanism,
    DecisionReason,
    PolicySpec,
    ReserveMechanism,
)
from sentinel.policy.models import Allocation, PolicyWindow


class AllocationError(ValueError):
    """Raised when a queue cannot be built as described."""


@lru_cache(maxsize=512)
def risk_order(window: PolicyWindow) -> tuple[int, ...]:
    """The whole window in canonical risk order: descending score, ties ascending id.

    ``top_k_indices`` with ``k = n`` rather than a local sort. Component 5 owns this ordering
    and a second implementation of it would be a second thing to keep correct -- and the one
    that silently disagreed would produce a queue that looked right.

    Cached on the window, which is a frozen dataclass of tuples and therefore hashable by
    value. A full run allocates the same window under seven policies at five capacities, so
    without the cache the same sort runs thirty-five times and the ordering that decides the
    whole queue is recomputed from scratch each time. The cache is keyed by the window's
    contents, not its identity, so two equal windows share an answer and no stale one survives
    a change of input.
    """
    if window.n == 0:
        return ()
    return tuple(top_k_indices(window.scores, window.ids, window.n))


def reserve_target(spec: PolicySpec, k: int) -> int:
    """Slots the policy asks the reserve to fill, before availability is considered.

    ``int()`` truncates, which is the point: a reserve may never spend more than the share it
    declared. Rounding up would let a 5% reserve take 1 slot out of 18 and call it 5%, and a
    policy that quietly overspends its own budget is worse than one that is visibly inert at
    small capacities. Profile 7 measured where that inertness happens and it is reported as an
    advisory rather than hidden.
    """
    if spec.mechanism is ReserveMechanism.NONE:
        return 0
    return int(spec.reserve_share * k)


def _eligible_prefix_counts(window: PolicyWindow, order: tuple[int, ...]) -> list[int]:
    """``cum[m]`` = eligible rows among the first ``m`` of the risk order."""
    cumulative = [0]
    running = 0
    for index in order:
        running += 1 if window.eligible[index] else 0
        cumulative.append(running)
    return cumulative


def allocate(window: PolicyWindow, spec: PolicySpec, *, k_name: str, k: int) -> Allocation:
    """Build one policy's queue for one window at one capacity.

    The reserve size is solved rather than assumed. Granting ``r`` slots shrinks the retained
    risk block to ``k - r``, which changes how many eligible establishments the risk block
    already holds, which changes how many are left for the reserve to take. The largest
    feasible ``r`` is therefore found by walking up from zero -- and it can land below the
    target when the risk block already contains most of the eligible population, which is
    exactly the case Component 13 turned out to be about.
    """
    if k < 1:
        raise AllocationError(f"{spec.policy_id}/{window.fold_id}/{k_name}: k must be at least 1")
    capacity = min(k, window.n)
    if capacity < 1:
        raise AllocationError(f"{spec.policy_id}/{window.fold_id}/{k_name}: empty window")

    order = risk_order(window)
    cumulative = _eligible_prefix_counts(window, order)
    eligible_in_top_k = cumulative[capacity]
    target = reserve_target(spec, capacity)

    if spec.mechanism is ReserveMechanism.NONE:
        wanted = 0
    elif spec.mechanism is ReserveMechanism.FLOOR:
        # A guarantee, not a spend: ask only for the shortfall the risk block left behind.
        wanted = max(0, target - eligible_in_top_k)
    elif spec.mechanism is ReserveMechanism.FORCED:
        wanted = target
    else:  # pragma: no cover - the enum is closed and the registry guard checks it
        raise AllocationError(f"unknown reserve mechanism {spec.mechanism!r}")

    granted = 0
    while granted < wanted:
        candidate = granted + 1
        available = window.n_eligible - cumulative[capacity - candidate]
        if candidate > available:
            break
        granted = candidate

    risk_indices = tuple(order[: capacity - granted])
    retained = set(risk_indices)
    reserve_indices = tuple(
        index for index in order if window.eligible[index] and index not in retained
    )[:granted]

    if len(reserve_indices) != granted:  # pragma: no cover - the solver guarantees this
        raise AllocationError(
            f"{spec.policy_id}/{window.fold_id}/{k_name}: reserve solved to {granted} slots "
            f"but only {len(reserve_indices)} eligible rows were available"
        )

    return Allocation(
        policy_id=spec.policy_id,
        fold_set=window.fold_set,
        fold_id=window.fold_id,
        k_name=k_name,
        k=capacity,
        n_universe=window.n,
        reserve_target=target,
        n_eligible_available=window.n_eligible,
        n_eligible_in_risk_top_k=eligible_in_top_k,
        risk_indices=risk_indices,
        reserve_indices=reserve_indices,
    )


def decide(
    window: PolicyWindow, allocation: Allocation
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int | None, ...]]:
    """Per-row mechanism, reason and final policy rank, indexed by window position.

    Ranks run ``1 .. k``: the risk block first in risk order, then the reserve block in its
    own order. Contiguous and unique, which the validator checks, because a queue with a
    duplicate rank cannot be worked in order and a queue with a gap has lost a row somewhere.

    The risk block is placed ahead of the reserve deliberately. Both are inspected within the
    same operating window, so the ordering carries no extra capacity -- but it records which
    mechanism has the stronger claim on a slot that turns out not to exist, and leaving that
    to the accident of an index would be leaving it undecided.
    """
    mechanisms: list[str] = [DecisionMechanism.NOT_SELECTED] * window.n
    reasons: list[str] = [DecisionReason.NOT_SELECTED_CAPACITY_EXHAUSTED] * window.n
    ranks: list[int | None] = [None] * window.n

    for position, index in enumerate(allocation.risk_indices, start=1):
        mechanisms[index] = DecisionMechanism.RISK_PRIORITY
        reasons[index] = DecisionReason.SELECTED_BY_RISK_RANK
        ranks[index] = position

    offset = len(allocation.risk_indices)
    for position, index in enumerate(allocation.reserve_indices, start=offset + 1):
        mechanisms[index] = DecisionMechanism.COVERAGE_RESERVE
        reasons[index] = DecisionReason.SELECTED_BY_COVERAGE_RESERVE
        ranks[index] = position

    # An eligible establishment that missed the cut missed it for a different reason than an
    # ineligible one: the reserve did not reach it. Recording that separately is what lets a
    # reviewer ask "was the reserve too small?" from the artifact rather than from the code.
    for index in range(window.n):
        if mechanisms[index] == DecisionMechanism.NOT_SELECTED and window.eligible[index]:
            reasons[index] = DecisionReason.NOT_SELECTED_RESERVE_EXHAUSTED

    return tuple(mechanisms), tuple(reasons), tuple(ranks)


def model_ranks(window: PolicyWindow) -> tuple[int, ...]:
    """Each row's position in the pure model ordering, 1-based, indexed by window position.

    Carried on every recommendation beside the policy rank. The pair is what makes the
    component's central question answerable per row: where the two ranks agree the model
    decided, and where they differ the policy did.
    """
    ranks = [0] * window.n
    for position, index in enumerate(risk_order(window), start=1):
        ranks[index] = position
    return tuple(ranks)


__all__ = [
    "AllocationError",
    "allocate",
    "decide",
    "model_ranks",
    "reserve_target",
    "risk_order",
]
