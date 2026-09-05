"""Measuring what each policy does, and what it costs. Pure -- no filesystem, no clock.

**The discovery-efficiency chain is Component 5's, imported and called.** ``discovery_curve``,
``normalized_area`` and ``normalized_discovery_efficiency`` all take an explicit ordering, so a
policy queue can be handed to them directly.

**Component 5's top-k helpers deliberately are not used, and the reason matters.**
``precision_at_k(labels, scores, ids, k)`` re-derives the top ``k`` *from the scores* -- which
is the definition of ``pure_risk``. Handing a coverage policy's queue to it would silently
measure the baseline and report it as the policy's number, and the failure would be invisible
because the result is a perfectly plausible precision. So precision, capture and lift are
computed over the queue that was actually built, from the same formulae. The equivalence is not
asserted here, it is tested: ``tests/test_policy_evaluate.py`` runs both paths on ``pure_risk``
and requires exact agreement.

**Opportunity cost is measured, not asserted.** Every coverage policy is differenced against
``pure_risk`` at the same model, fold and capacity, and the difference is reported in
citations -- the unit that means something to a department -- as well as in rates. A reserve is
described as free only where the measured delta is zero, and profile 6 established before any
of this ran that it usually is not.

**A note on NDE.** Discovery efficiency needs an ordering of the *whole* window, but a policy
only speaks about the first *k* rows. The convention here is that the tail keeps its risk
order: the policy is applied to the capacity it governs, and everything below the cutoff falls
back to the model's ranking. That is a measurement convention and it is stated rather than
buried, because a different tail convention would produce a different NDE for the same policy
and nothing in the policy itself decides the tail.

**The group audit is descriptive.** It recomputes selection share and capture per as-of
community area under each policy, using Component 12's support floors and carrying its status
column through unchanged. Nothing here is optimised against, no group is rebalanced, and a
group Component 12 called unsupported stays unsupported with its reason attached.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sentinel.evaluation.simulate import (
    Window,
    discovery_curve,
    normalized_area,
    normalized_discovery_efficiency,
)
from sentinel.policy.allocation import risk_order
from sentinel.policy.definitions import FRONTIER_AXES
from sentinel.policy.models import Allocation, PolicyWindow


class EvaluationError(ValueError):
    """Raised when a policy cannot be scored against the window it was built for."""


def _simulation_window(window: PolicyWindow) -> Window:
    """Hand Component 5's simulator its own type, built from the policy window.

    A conversion rather than a subclass or a duck-typed pass. ``Window`` re-checks its own
    invariants on construction, so routing through it means Component 5 validates the rows
    before it integrates them instead of trusting that this component built them correctly.
    """
    return Window(ids=window.ids, labels=window.labels, dates=window.dates)


def schedule_order(window: PolicyWindow, allocation: Allocation) -> tuple[int, ...]:
    """The whole window ordered as the policy would work it: queue first, then risk order.

    Only the first ``k`` entries are a policy statement. The tail is the model's own ranking,
    kept so that discovery efficiency -- which is defined over a full permutation -- has
    something to integrate. See the module docstring.
    """
    selected = [*allocation.risk_indices, *allocation.reserve_indices]
    chosen = set(selected)
    tail = [index for index in risk_order(window) if index not in chosen]
    order = tuple([*selected, *tail])
    if len(order) != window.n:
        raise EvaluationError(
            f"{allocation.policy_id}/{allocation.fold_id}/{allocation.k_name}: schedule covers "
            f"{len(order)} of {window.n} rows"
        )
    return order


def cell_metrics(
    window: PolicyWindow, allocation: Allocation, *, definition_version: str
) -> dict[str, object]:
    """Effectiveness, coverage and composition for one (policy, fold, capacity) cell.

    Precision and capture are reported together throughout. Precision@k alone is misleading
    once ``k`` passes the number of positives, and capture alone hides how much capacity bought
    it -- Component 5 made that argument and this component inherits it rather than restating
    the metric.
    """
    selected = [*allocation.risk_indices, *allocation.reserve_indices]
    if len(selected) != allocation.k:
        raise EvaluationError(
            f"{allocation.policy_id}/{allocation.fold_id}/{allocation.k_name}: selected "
            f"{len(selected)} rows for a capacity of {allocation.k}"
        )

    labels = list(window.labels)
    positives_selected = sum(labels[i] for i in selected)
    eligible_selected = sum(1 for i in selected if window.eligible[i])
    eligible_positives_selected = sum(labels[i] for i in selected if window.eligible[i])
    eligible_positives_total = sum(labels[i] for i in range(window.n) if window.eligible[i])

    order = schedule_order(window, allocation)
    cumulative = discovery_curve(_simulation_window(window), order)
    area = normalized_area(cumulative, n=window.n, positives=window.positives)

    return {
        "policy_id": allocation.policy_id,
        "model_name": "",
        "fold_set": allocation.fold_set,
        "fold_id": allocation.fold_id,
        "k_name": allocation.k_name,
        "k": allocation.k,
        "n_universe": allocation.n_universe,
        "n_selected": len(selected),
        "n_risk": allocation.n_risk,
        "n_reserve": allocation.n_reserve,
        "reserve_target": allocation.reserve_target,
        "reserve_inert": allocation.reserve_inert,
        "n_positive": window.positives,
        "positives_selected": positives_selected,
        "precision_at_k": (positives_selected / len(selected)) if selected else None,
        "capture_rate": (positives_selected / window.positives) if window.positives else None,
        "lift_at_k": _lift(positives_selected, len(selected), window),
        "nde": normalized_discovery_efficiency(area, n=window.n, positives=window.positives),
        "n_eligible_available": allocation.n_eligible_available,
        "n_eligible_in_risk_top_k": allocation.n_eligible_in_risk_top_k,
        "eligible_selected": eligible_selected,
        "eligible_selected_share": (eligible_selected / len(selected)) if selected else None,
        "eligible_positives_selected": eligible_positives_selected,
        "eligible_capture_rate": (
            eligible_positives_selected / eligible_positives_total
            if eligible_positives_total
            else None
        ),
        "policy_definition_version": definition_version,
    }


def _lift(positives_selected: int, n_selected: int, window: PolicyWindow) -> float | None:
    """Precision over the window's base rate: the drift-robust form of precision@k.

    Component 5's argument for reporting lift rather than raw precision is inherited whole --
    prevalence falls from 0.88 to 0.39 across this dataset, so a precision of 0.55 is excellent
    in 2026 and poor in 2019. The arithmetic is Component 5's; only the queue it is measured
    over is this component's.
    """
    if not n_selected or not window.n:
        return None
    base_rate = window.positives / window.n
    if base_rate == 0.0:
        return None
    return (positives_selected / n_selected) / base_rate


def opportunity_cost(
    cell: Mapping[str, object], baseline: Mapping[str, object]
) -> dict[str, object]:
    """What one policy gave up, or gained, against ``pure_risk`` at the same operating point.

    Differenced cell by cell rather than on pooled means. A pooled difference of means would
    let a large fold's behaviour stand in for a small one's, and the small recent folds are
    precisely where profile 7 found the coverage question actually bites.
    """
    return {
        "delta_positives": _delta(cell, baseline, "positives_selected"),
        "delta_precision": _delta(cell, baseline, "precision_at_k"),
        "delta_capture": _delta(cell, baseline, "capture_rate"),
        "delta_nde": _delta(cell, baseline, "nde"),
        "delta_eligible_selected": _delta(cell, baseline, "eligible_selected"),
        "delta_eligible_capture": _delta(cell, baseline, "eligible_capture_rate"),
    }


def _delta(cell: Mapping[str, object], baseline: Mapping[str, object], key: str) -> float | None:
    left, right = cell.get(key), baseline.get(key)
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return float(left) - float(right)


def frontier(
    comparison: Sequence[Mapping[str, object]],
    *,
    fold_set: str,
    definition_version: str,
) -> list[dict[str, object]]:
    """Pool each policy over one fold set and mark the dominated ones.

    Domination is the honest summary of a trade-off: policy A dominates B when it is at least
    as good on every declared axis and strictly better on one, which needs no weighting of
    citations against coverage. Anything that survives is on the frontier, and choosing among
    the survivors is a value judgement -- so this function marks them and stops.

    ``FRONTIER_AXES`` is deliberately two axes and deliberately not summed. A single "policy
    score" would embed an exchange rate between a missed Priority citation and an uninspected
    establishment with no history, and nothing in this project measures that rate.
    """
    axes = [name for name, _ in FRONTIER_AXES]
    pooled: dict[tuple[str, str], dict[str, float]] = {}
    for row in comparison:
        if row.get("fold_set") != fold_set:
            continue
        key = (str(row["policy_id"]), str(row["k_name"]))
        bucket = pooled.setdefault(key, dict.fromkeys(axes, 0.0))
        for axis in axes:
            value = row.get(axis)
            if isinstance(value, (int, float)):
                bucket[axis] += float(value)

    out: list[dict[str, object]] = []
    for (policy_id, k_name), values in pooled.items():
        dominators = [
            other_policy
            for (other_policy, other_k), other in pooled.items()
            if other_k == k_name
            and other_policy != policy_id
            and all(other[axis] >= values[axis] for axis in axes)
            and any(other[axis] > values[axis] for axis in axes)
        ]
        out.append(
            {
                "policy_id": policy_id,
                "fold_set": fold_set,
                "k_name": k_name,
                "positives_selected": values["positives_selected"],
                "eligible_selected": values["eligible_selected"],
                "is_dominated": bool(dominators),
                "dominated_by": ", ".join(sorted(dominators)),
                "policy_definition_version": definition_version,
            }
        )
    return out


def winner(frontier_rows: Sequence[Mapping[str, object]], *, k_name: str) -> str | None:
    """The unique non-dominated policy at one capacity, or ``None``.

    ``None`` is the expected answer and it is not a failure. A grid whose points genuinely
    trade citations against coverage has no mathematically best member, and inventing a
    tolerance to force one would be smuggling in the exchange rate that ``frontier`` refuses
    to assume.
    """
    survivors = sorted(
        {
            str(row["policy_id"])
            for row in frontier_rows
            if row.get("k_name") == k_name and not row.get("is_dominated")
        }
    )
    return survivors[0] if len(survivors) == 1 else None


def group_audit(
    window: PolicyWindow,
    allocation: Allocation,
    *,
    groups: Sequence[str],
    support: Mapping[str, str],
    model_name: str,
    definition_version: str,
) -> list[dict[str, object]]:
    """Selection share and capture per as-of geography, under one policy. Descriptive only.

    ⚠ This is the only place a Component 12 artifact touches a per-row computation, and it
    touches it in one direction: the group label and its support status are *read onto* the
    output rows so a reader can see who the queue served. No group value influences a score, a
    rank or an allocation, and ``validate.py`` proves that by rebuilding the queue with the
    group columns absent and comparing.
    """
    if len(groups) != window.n:
        raise EvaluationError(
            f"{allocation.fold_id}: {len(groups)} group labels for {window.n} window rows"
        )
    selected = set(allocation.risk_indices) | set(allocation.reserve_indices)
    rows: list[dict[str, object]] = []
    for value in sorted(set(groups)):
        members = [i for i in range(window.n) if groups[i] == value]
        chosen = [i for i in members if i in selected]
        positives = sum(window.labels[i] for i in members)
        found = sum(window.labels[i] for i in chosen)
        rows.append(
            {
                "policy_id": allocation.policy_id,
                "model_name": model_name,
                "group_definition": "community_area",
                "group_value": value,
                "fold_set": allocation.fold_set,
                "fold_id": allocation.fold_id,
                "k_name": allocation.k_name,
                "n_rows": len(members),
                "n_positive": positives,
                "population_share": len(members) / window.n if window.n else None,
                "n_selected": len(chosen),
                "selected_share": len(chosen) / len(selected) if selected else None,
                "selection_rate": len(chosen) / len(members) if members else None,
                "positives_selected": found,
                "capture_rate": (found / positives) if positives else None,
                "group_status": support.get(value, "insufficient_support"),
                "policy_definition_version": definition_version,
            }
        )
    return rows


__all__ = [
    "EvaluationError",
    "cell_metrics",
    "frontier",
    "group_audit",
    "opportunity_cost",
    "schedule_order",
    "winner",
]
