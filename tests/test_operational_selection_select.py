"""Component 19: capacity-constrained selection over a Component 18 priority set.

Builds synthetic Component 18-shaped priority frames directly (from
``operational_scoring.writer.OUTPUT_SCHEMA``) rather than running Components 17-18,
matching how the Component 18 test suite itself builds synthetic Component 4-shaped
frames rather than running Components 1-3.
"""

from __future__ import annotations

import polars as pl
import pytest

from sentinel.operational_scoring.writer import OUTPUT_SCHEMA
from sentinel.operational_selection import validate
from sentinel.operational_selection.models import OperationalCapacityRequest
from sentinel.operational_selection.select import SelectionError, select_candidates
from sentinel.policy.allocation import allocate, decide
from sentinel.policy.definitions import policy_for

PLANNING_DATE = "2026-08-28"
FOLD_SET = "operational"
FOLD_ID = "operational-2026-08-28"


def _row(i: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "planning_date": PLANNING_DATE,
        "candidate_definition_version": "v1",
        "feature_definition_version": "v1",
        "operational_scoring_definition_version": "v1",
        "composite_model_name": "xgboost_platt",
        "base_model_name": "xgboost",
        "calibration_method": "platt",
        "establishment_id": f"EST-{i:06d}",
        "target_inspection_id": f"CANDIDATE::{PLANNING_DATE}::EST-{i:06d}",
        "canonical_name": f"NAME-{i}",
        "canonical_address": f"ADDR-{i}",
        "canonical_zip": "60601",
        "as_of_dba_name": f"NAME-{i}",
        "as_of_address": f"ADDR-{i}",
        "as_of_zip": "60601",
        "as_of_latitude": 41.8 + i * 0.001,
        "as_of_longitude": -87.6 - i * 0.001,
        "has_location": True,
        "n_prior_records": 5,
        "scoring_status": "scored",
        "base_score": round(1.0 - i * 0.01, 6),
        "calibrated_score": round(1.0 - i * 0.01, 6),
        "rank": i + 1,
        "coverage_eligible": False,
        "secondary_no_history": False,
    }
    row.update(overrides)
    return row


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    ordered = [{name: row.get(name) for name in OUTPUT_SCHEMA} for row in rows]
    return pl.DataFrame(ordered, schema=OUTPUT_SCHEMA)


def _request(capacity: int, *, policy_id: str = "") -> OperationalCapacityRequest:
    return OperationalCapacityRequest(
        planning_date=PLANNING_DATE, maximum_inspections=capacity, policy_id=policy_id
    )


def _select(rows: list[dict[str, object]], capacity: int, *, policy_id: str = ""):
    return select_candidates(
        priority_frame=_frame(rows),
        capacity=_request(capacity, policy_id=policy_id),
        operational_fold_set=FOLD_SET,
        operational_fold_id=FOLD_ID,
    )


# --- 1. exact capacity -------------------------------------------------------


def test_exact_capacity_selects_exactly_that_many() -> None:
    rows = [_row(i) for i in range(50)]
    result = _select(rows, 30)
    assert result.selected_count == 30
    assert int(result.frame.filter(pl.col("is_selected")).height) == 30
    # The full queue survives: 50 rows in, 50 rows out.
    assert result.frame.height == 50


# --- 2. zero capacity --------------------------------------------------------


def test_zero_capacity_selects_nothing_safely() -> None:
    rows = [_row(i) for i in range(10)]
    result = _select(rows, 0)
    assert result.selected_count == 0
    assert result.unfilled_capacity == 0
    assert not result.frame["is_selected"].any()
    assert result.frame.height == 10


# --- 3. capacity exceeds candidates ------------------------------------------


def test_capacity_exceeding_candidates_selects_all_and_reports_shortfall() -> None:
    rows = [_row(i) for i in range(10)]
    result = _select(rows, 30)
    assert result.selected_count == 10
    assert result.unfilled_capacity == 20
    # No fabricated rows: still exactly 10 in the output.
    assert result.frame.height == 10
    assert int(result.frame.filter(pl.col("is_selected")).height) == 10


# --- 4. determinism -----------------------------------------------------------


def test_same_inputs_produce_identical_selection() -> None:
    rows = [_row(i) for i in range(25)]
    first = _select(rows, 12)
    second = _select(rows, 12)
    assert first.frame.equals(second.frame)
    assert first.selected_count == second.selected_count == 12


# --- 5. rank integrity: Component 18's rank/score are never altered ----------


def test_component_18_rank_and_score_are_preserved() -> None:
    rows = [_row(i) for i in range(15)]
    priority = _frame(rows)
    result = select_candidates(
        priority_frame=priority,
        capacity=_request(8),
        operational_fold_set=FOLD_SET,
        operational_fold_id=FOLD_ID,
    )
    check = validate.check_component_18_rank_unchanged(priority, result.frame)
    assert check.passed


# --- 6. policy/allocation reuse: matches calling allocate()/decide() directly -


def test_selection_matches_calling_policy_allocation_directly() -> None:
    """The exact rows Component 13's own allocate()/decide() would pick, for a reserve policy."""
    rows = [_row(i, coverage_eligible=(i % 5 == 0)) for i in range(40)]
    policy_id = "coverage_forced_population_share"
    result = _select(rows, 20, policy_id=policy_id)

    from sentinel.operational_selection.window import build_selection_window

    window = build_selection_window(_frame(rows), fold_set=FOLD_SET, fold_id=FOLD_ID)
    spec = policy_for(policy_id)
    allocation = allocate(window, spec, k_name="maximum_inspections", k=20)
    mechanisms, _reasons, ranks = decide(window, allocation)

    expected_selected_ids = {
        window.ids[i] for i, m in enumerate(mechanisms) if m != "not_selected"
    }
    actual_selected_ids = set(
        result.frame.filter(pl.col("is_selected"))["target_inspection_id"].to_list()
    )
    assert actual_selected_ids == expected_selected_ids
    assert result.reserve_selected_count == allocation.n_reserve
    assert result.risk_selected_count == allocation.n_risk


# --- 7. selection reasons are from the real vocabulary ------------------------


def test_selection_reasons_come_from_the_real_decision_vocabulary() -> None:
    rows = [_row(i, coverage_eligible=(i < 5)) for i in range(20)]
    result = _select(rows, 10, policy_id="coverage_forced_population_share")

    known_selected_reasons = {"selected_by_risk_rank", "selected_by_coverage_reserve"}
    known_not_selected_reasons = {
        "not_selected_capacity_exhausted",
        "not_selected_reserve_exhausted",
    }
    selected_reasons = set(
        result.frame.filter(pl.col("is_selected"))["selection_reason"].to_list()
    )
    not_selected_reasons = set(
        result.frame.filter(~pl.col("is_selected"))["selection_reason"].to_list()
    )
    assert selected_reasons <= known_selected_reasons
    assert not_selected_reasons <= known_not_selected_reasons


# --- 8. duplicate safety -------------------------------------------------------


def test_duplicate_establishment_id_is_caught_by_the_check() -> None:
    rows = [_row(i) for i in range(5)]
    # Two distinct candidate opportunities naming the same establishment -- should not
    # happen upstream, but the check must catch it rather than silently passing.
    rows.append(_row(5, establishment_id="EST-000000"))
    result = _select(rows, 10)
    check = validate.check_no_duplicate_selected_establishments(result.frame)
    assert not check.passed
    assert validate.has_failures([check])


def test_no_duplicates_in_the_ordinary_case() -> None:
    rows = [_row(i) for i in range(20)]
    result = _select(rows, 10)
    check = validate.check_no_duplicate_selected_establishments(result.frame)
    assert check.passed


# --- 9. location independence --------------------------------------------------


def test_changing_only_location_never_changes_the_selected_id_set() -> None:
    rows_a = [_row(i) for i in range(20)]
    rows_b = [
        _row(i, as_of_latitude=None, as_of_longitude=None, has_location=False)
        for i in range(20)
    ]
    result_a = _select(rows_a, 10)
    result_b = _select(rows_b, 10)
    ids_a = set(result_a.frame.filter(pl.col("is_selected"))["target_inspection_id"].to_list())
    ids_b = set(result_b.frame.filter(pl.col("is_selected"))["target_inspection_id"].to_list())
    assert ids_a == ids_b


def test_allocation_input_contract_excludes_location_columns() -> None:
    check = validate.check_selection_never_reads_location()
    assert check.passed


# --- 10. full priority queue preserved -----------------------------------------


def test_full_priority_queue_is_preserved_not_replaced() -> None:
    rows = [_row(i) for i in range(30)]
    priority = _frame(rows)
    result = select_candidates(
        priority_frame=priority,
        capacity=_request(5),
        operational_fold_set=FOLD_SET,
        operational_fold_id=FOLD_ID,
    )
    check = validate.check_full_priority_queue_preserved(priority, result.frame)
    assert check.passed
    assert result.frame.height == priority.height == 30


def test_unscorable_candidates_are_carried_through_never_allocated() -> None:
    rows = [_row(i) for i in range(10)]
    rows[3]["scoring_status"] = "excluded_feature_contract_violation"
    rows[3]["base_score"] = None
    rows[3]["calibrated_score"] = None
    rows[3]["rank"] = None
    result = _select(rows, 5)
    assert result.unscorable_count == 1
    assert result.ranked_candidate_count == 10
    assert result.selectable_candidate_count == 9
    row = result.frame.filter(pl.col("establishment_id") == "EST-000003").row(0, named=True)
    assert row["is_selected"] is False
    assert row["selection_reason"] == "excluded_unscorable_by_component_18"


# --- validation / error paths --------------------------------------------------


def test_negative_capacity_is_refused() -> None:
    with pytest.raises(SelectionError, match="non-negative"):
        _select([_row(0)], -1)


def test_empty_priority_set_is_refused() -> None:
    empty = _frame([])
    with pytest.raises(SelectionError, match="empty"):
        select_candidates(
            priority_frame=empty,
            capacity=_request(10),
            operational_fold_set=FOLD_SET,
            operational_fold_id=FOLD_ID,
        )


def test_planning_date_mismatch_is_refused() -> None:
    rows = [_row(0)]
    mismatched = OperationalCapacityRequest(
        planning_date="2020-01-01", maximum_inspections=10, policy_id=""
    )
    with pytest.raises(SelectionError, match="mismatch"):
        select_candidates(
            priority_frame=_frame(rows),
            capacity=mismatched,
            operational_fold_set=FOLD_SET,
            operational_fold_id=FOLD_ID,
        )


def test_unknown_policy_id_is_refused() -> None:
    with pytest.raises(SelectionError):
        _select([_row(0)], 10, policy_id="not_a_real_policy")


def test_all_candidates_unscorable_is_refused_clearly() -> None:
    rows = [_row(i) for i in range(3)]
    for row in rows:
        row["scoring_status"] = "excluded_feature_contract_violation"
        row["calibrated_score"] = None
    with pytest.raises(SelectionError, match="no scored candidates"):
        _select(rows, 10)
