"""The pure orchestration: a priority set + a capacity request -> a bounded selection.

No filesystem access, no clock. ``allocate()`` and ``decide()`` are Component 13's own,
imported unmodified; nothing here re-implements a reserve, a cutoff, or a tie-break.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from sentinel.operational_selection.definitions import (
    DEFAULT_POLICY_ID,
    EXCLUDED_UNSCORABLE_REASON,
)
from sentinel.operational_selection.models import OperationalCapacityRequest
from sentinel.operational_selection.window import SelectionWindowError, build_selection_window
from sentinel.policy.allocation import AllocationError, allocate, decide
from sentinel.policy.definitions import (
    PolicyDefinitionError,
    PolicySpec,
    policy_for,
)

#: The ``k_name`` label handed to ``allocate()``/``decide()``. Component 13 uses this
#: label to distinguish capacity cutoffs like ``"k_1_day"``; Component 19 has exactly
#: one capacity per run, so one constant label is enough and never invented per-request.
K_NAME = "maximum_inspections"

#: The columns Component 19 preserves from Component 18's output, without repeating
#: the raw Component 4 feature values -- they were the model's input, not part of the
#: selection record a later component needs.
CARRIED_COLUMNS = (
    "planning_date",
    "candidate_definition_version",
    "feature_definition_version",
    "operational_scoring_definition_version",
    "composite_model_name",
    "base_model_name",
    "calibration_method",
    "establishment_id",
    "target_inspection_id",
    "canonical_name",
    "canonical_address",
    "canonical_zip",
    # Real, preserved location metadata (Component 20's future input) -- carried
    # through for display only. ``window.REQUIRED_COLUMNS`` (what actually feeds
    # ``allocate()``/``decide()``) excludes every one of these; see
    # ``validate.check_selection_never_reads_location``.
    "as_of_dba_name",
    "as_of_address",
    "as_of_zip",
    "as_of_latitude",
    "as_of_longitude",
    "has_location",
    "n_prior_records",
    "scoring_status",
    "base_score",
    "calibrated_score",
    "rank",
    "coverage_eligible",
    "secondary_no_history",
)


class SelectionError(RuntimeError):
    """Raised when a capacity-constrained selection cannot be produced at all."""


@dataclass
class SelectionResult:
    """Everything ``build.py`` needs to write the artifact and the manifest."""

    frame: pl.DataFrame
    policy: PolicySpec
    requested_capacity: int
    ranked_candidate_count: int
    selectable_candidate_count: int
    unscorable_count: int
    selected_count: int
    risk_selected_count: int
    reserve_selected_count: int
    unfilled_capacity: int
    coverage_eligible_selected_count: int


def resolve_policy(policy_id: str) -> PolicySpec:
    """The named policy, or the grid's baseline (plain top-k, no reserve) if unset."""
    resolved = policy_id or DEFAULT_POLICY_ID
    try:
        return policy_for(resolved)
    except PolicyDefinitionError as exc:
        raise SelectionError(str(exc)) from exc


def select_candidates(
    *,
    priority_frame: pl.DataFrame,
    capacity: OperationalCapacityRequest,
    operational_fold_set: str,
    operational_fold_id: str,
) -> SelectionResult:
    """Select up to ``capacity.maximum_inspections`` establishments from a priority set.

    ``priority_frame`` is Component 18's output, unmodified -- every row, selected or
    not, is present in the returned frame. Rows Component 18 could not score
    (``scoring_status != "scored"``) never enter allocation; they are carried through
    with their own status, never silently dropped.
    """
    if capacity.maximum_inspections < 0:
        raise SelectionError(
            f"maximum_inspections must be non-negative, got {capacity.maximum_inspections}"
        )
    if priority_frame.is_empty():
        raise SelectionError("priority set is empty -- there is nothing to select from")

    priority_dates = set(priority_frame["planning_date"].to_list())
    if priority_dates != {capacity.planning_date}:
        raise SelectionError(
            f"planning_date mismatch: the capacity request names {capacity.planning_date!r} "
            f"but the priority set carries {sorted(priority_dates)!r}. Selecting against a "
            "priority set built for a different planning date would silently misdate the plan"
        )

    policy = resolve_policy(capacity.policy_id)

    ranked_candidate_count = priority_frame.height
    unscorable = priority_frame.filter(pl.col("scoring_status") != "scored")
    unscorable_count = unscorable.height

    display = priority_frame.select(list(CARRIED_COLUMNS))

    if capacity.maximum_inspections == 0:
        selectable_count = ranked_candidate_count - unscorable_count
        out = display.with_columns(
            pl.lit("not_selected").alias("selection_mechanism"),
            pl.lit("not_selected_capacity_exhausted").alias("selection_reason"),
            pl.lit(None, dtype=pl.Int64).alias("policy_rank"),
            pl.lit(False).alias("is_selected"),
        )
        out = _mark_unscorable(out)
        return SelectionResult(
            frame=out,
            policy=policy,
            requested_capacity=0,
            ranked_candidate_count=ranked_candidate_count,
            selectable_candidate_count=selectable_count,
            unscorable_count=unscorable_count,
            selected_count=0,
            risk_selected_count=0,
            reserve_selected_count=0,
            unfilled_capacity=0,
            coverage_eligible_selected_count=0,
        )

    try:
        window = build_selection_window(
            priority_frame, fold_set=operational_fold_set, fold_id=operational_fold_id
        )
    except SelectionWindowError as exc:
        raise SelectionError(str(exc)) from exc

    try:
        allocation = allocate(window, policy, k_name=K_NAME, k=capacity.maximum_inspections)
    except AllocationError as exc:
        raise SelectionError(str(exc)) from exc
    mechanisms, reasons, ranks = decide(window, allocation)

    decided = pl.DataFrame(
        {
            "target_inspection_id": list(window.ids),
            # ``mechanisms``/``reasons`` are StrEnum members; cast explicitly to plain
            # Utf8 rather than let polars infer an Enum dtype from them, which would
            # reject the (equally real) "excluded_unscorable" reason on the join below.
            "selection_mechanism": pl.Series([str(m) for m in mechanisms], dtype=pl.Utf8),
            "selection_reason": pl.Series([str(r) for r in reasons], dtype=pl.Utf8),
            "policy_rank": list(ranks),
        }
    )
    out = display.join(decided, on="target_inspection_id", how="left").with_columns(
        pl.col("selection_mechanism").fill_null("not_selected"),
        pl.col("selection_reason").fill_null(EXCLUDED_UNSCORABLE_REASON),
        pl.col("policy_rank").cast(pl.Int64),
    )
    out = _mark_unscorable(out)
    out = out.with_columns((pl.col("selection_mechanism") != "not_selected").alias("is_selected"))

    selected = out.filter(pl.col("is_selected"))
    coverage_eligible_selected_count = int(selected.filter(pl.col("coverage_eligible")).height)
    unfilled_capacity = max(0, capacity.maximum_inspections - allocation.n_selected)

    out = out.sort([pl.col("policy_rank").is_null(), "policy_rank", "target_inspection_id"])

    return SelectionResult(
        frame=out,
        policy=policy,
        requested_capacity=capacity.maximum_inspections,
        ranked_candidate_count=ranked_candidate_count,
        selectable_candidate_count=window.n,
        unscorable_count=unscorable_count,
        selected_count=allocation.n_selected,
        risk_selected_count=allocation.n_risk,
        reserve_selected_count=allocation.n_reserve,
        unfilled_capacity=unfilled_capacity,
        coverage_eligible_selected_count=coverage_eligible_selected_count,
    )


def _mark_unscorable(frame: pl.DataFrame) -> pl.DataFrame:
    """Overwrite the reason for rows Component 18 never scored -- never allocated against."""
    return frame.with_columns(
        pl.when(pl.col("scoring_status") != "scored")
        .then(pl.lit(EXCLUDED_UNSCORABLE_REASON))
        .otherwise(pl.col("selection_reason"))
        .alias("selection_reason"),
        pl.when(pl.col("scoring_status") != "scored")
        .then(pl.lit("not_selected"))
        .otherwise(pl.col("selection_mechanism"))
        .alias("selection_mechanism"),
    )


__all__ = [
    "CARRIED_COLUMNS",
    "K_NAME",
    "SelectionError",
    "SelectionResult",
    "resolve_policy",
    "select_candidates",
]
