"""The two deterministic triggers. No score, no threshold -- only booleans an upstream
component already wrote."""

from __future__ import annotations

from datetime import date

import polars as pl

from sentinel.review.trigger import (
    build_review_cases,
    execution_gap_rows,
    trigger_column,
    warning_triggered_rows,
)


def _recommendation_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "policy_id": "pure_risk",
        "model_name": "lightgbm_platt",
        "fold_set": "quarterly",
        "fold_id": "2026Q1",
        "k_name": "k_1_day",
        "target_inspection_id": "t1",
        "establishment_id": "e1",
        "is_selected": True,
        "warnings": "limited_history",
        "final_policy_rank": 1,
        "decision_mechanism": "risk_priority",
        "decision_reason": "selected_by_risk_rank",
    }
    base.update(overrides)
    return base


def _empty_execution_log() -> pl.DataFrame:
    return pl.DataFrame({"target_inspection_id": []}, schema={"target_inspection_id": pl.Utf8})


def _schedule_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schedule_config_id": "strict_priority__observed_calendar",
        "policy_id": "pure_risk",
        "model_name": "lightgbm_platt",
        "fold_set": "quarterly",
        "fold_id": "2026Q1",
        "k_name": "k_1_day",
        "target_inspection_id": "t1",
        "schedule_status": "scheduled",
        "replan_index": 0,
        "planning_run_id": "run1",
        "scheduled_date": date(2026, 1, 5),
    }
    base.update(overrides)
    return base


def test_warning_trigger_requires_selection() -> None:
    recs = pl.DataFrame(
        [
            _recommendation_row(
                target_inspection_id="t1", is_selected=True, warnings="limited_history"
            ),
            _recommendation_row(
                target_inspection_id="t2", is_selected=False, warnings="limited_history"
            ),
            _recommendation_row(target_inspection_id="t3", is_selected=True, warnings="none"),
        ]
    )
    flagged = warning_triggered_rows(recs)
    assert flagged["target_inspection_id"].to_list() == ["t1"]


def test_empty_recommendations_yield_no_warning_rows() -> None:
    recs = pl.DataFrame([_recommendation_row()]).head(0)
    assert warning_triggered_rows(recs).is_empty()


def test_execution_gap_uses_latest_replan_index_only() -> None:
    schedule = pl.DataFrame(
        [
            _schedule_row(target_inspection_id="t1", replan_index=0),
            _schedule_row(target_inspection_id="t1", replan_index=1),
        ]
    )
    execution_log = _empty_execution_log()
    gapped = execution_gap_rows(schedule, execution_log)
    assert gapped.height == 1
    assert gapped["replan_index"].to_list() == [1]


def test_execution_gap_excludes_rows_with_a_matching_execution_event() -> None:
    schedule = pl.DataFrame(
        [
            _schedule_row(target_inspection_id="t1"),
            _schedule_row(target_inspection_id="t2"),
        ]
    )
    execution_log = pl.DataFrame(
        {
            "schedule_config_id": ["strict_priority__observed_calendar"],
            "policy_id": ["pure_risk"],
            "fold_id": ["2026Q1"],
            "k_name": ["k_1_day"],
            "target_inspection_id": ["t1"],
        }
    )
    gapped = execution_gap_rows(schedule, execution_log)
    assert gapped["target_inspection_id"].to_list() == ["t2"]


def test_execution_gap_excludes_backlog_and_cancelled_rows() -> None:
    schedule = pl.DataFrame(
        [
            _schedule_row(target_inspection_id="t1", schedule_status="backlog"),
            _schedule_row(target_inspection_id="t2", schedule_status="cancelled"),
            _schedule_row(target_inspection_id="t3", schedule_status="deferred"),
        ]
    )
    execution_log = _empty_execution_log()
    gapped = execution_gap_rows(schedule, execution_log)
    assert gapped["target_inspection_id"].to_list() == ["t3"]


def test_a_row_hit_by_both_triggers_carries_both_reasons_sorted() -> None:
    recs = pl.DataFrame([_recommendation_row(target_inspection_id="t1")])
    schedule = pl.DataFrame([_schedule_row(target_inspection_id="t1")])
    execution_log = _empty_execution_log()
    cases = build_review_cases(recs, schedule, execution_log)
    assert len(cases) == 1
    assert trigger_column(cases[0]) == (
        "no_execution_record_on_scheduled_row|policy_warning_present"
    )


def test_without_a_schedule_only_the_warning_trigger_runs() -> None:
    recs = pl.DataFrame(
        [
            _recommendation_row(target_inspection_id="t1", warnings="limited_history"),
            _recommendation_row(target_inspection_id="t2", warnings="none"),
        ]
    )
    cases = build_review_cases(recs, None, None)
    assert [c.target_inspection_id for c in cases] == ["t1"]


def test_a_row_matching_no_trigger_is_absent_from_the_case_list() -> None:
    recs = pl.DataFrame([_recommendation_row(target_inspection_id="t1", warnings="none")])
    cases = build_review_cases(recs, None, None)
    assert cases == []


def test_a_target_id_shared_across_policies_is_not_confused() -> None:
    """The execution-gap anti-join keys on schedule_config_id/policy_id/fold_id/k_name, not
    only target_inspection_id -- the same establishment recurs across every policy's queue."""
    schedule = pl.DataFrame(
        [
            _schedule_row(target_inspection_id="t1", policy_id="pure_risk"),
            _schedule_row(target_inspection_id="t1", policy_id="coverage_forced_population_share"),
        ]
    )
    # An execution event for pure_risk only.
    execution_log = pl.DataFrame(
        {
            "schedule_config_id": ["strict_priority__observed_calendar"],
            "policy_id": ["pure_risk"],
            "fold_id": ["2026Q1"],
            "k_name": ["k_1_day"],
            "target_inspection_id": ["t1"],
        }
    )
    gapped = execution_gap_rows(schedule, execution_log)
    assert gapped["policy_id"].to_list() == ["coverage_forced_population_share"]
