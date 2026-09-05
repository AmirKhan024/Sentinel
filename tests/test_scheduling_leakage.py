"""Deliberate failure injection: seventeen operationally dangerous defects, one test each.

Every case here is a plausible-looking schedule that is wrong in a way a reader would not spot
by eye -- a day one slot over capacity, a row that quietly vanished, an execution outcome that
edited a recommendation. Each is injected into an otherwise-valid artifact and the corresponding
check must go red.

The suite also asserts the other half of the contract, which is just as easy to break: the
advisory findings must **never** fail a run. Backlog, idle capacity and a lost coverage reserve
are measurements of the city's calendar, and a build that went red on them would be a build that
goes green only when the scheduler lies.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from sentinel.scheduling import validate as v
from sentinel.scheduling.definitions import (
    SCHEDULE_DEFINITION_VERSION,
    InversionReason,
    ScheduleReason,
    ScheduleStatus,
)
from sentinel.scheduling.models import SEVERITY_ERROR, SEVERITY_WARN
from sentinel.scheduling.writer import empty, finalize

CONFIG = "strict_priority__observed_calendar"
CELL = {
    "schedule_config_id": CONFIG,
    "policy_id": "pure_risk",
    "model_name": "xgboost_platt",
    "fold_set": "quarterly",
    "fold_id": "quarterly-2026Q2",
    "k_name": "k_1_week",
}


def _schedule_row(index: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        **CELL,
        "k": 4,
        "target_inspection_id": f"T{index:05d}",
        "establishment_id": f"EST-{index:05d}",
        "recommendation_date": date(2026, 4, 1),
        "base_score": 0.5,
        "score": 0.6,
        "model_rank": index + 1,
        "final_policy_rank": index + 1,
        "decision_mechanism": "risk_priority",
        "decision_reason": "selected_by_risk_rank",
        "coverage_eligible": False,
        "warnings": "none",
        "recommendation_override_id": "",
        "policy_definition_version": "v1",
        "planning_run_id": "PR-test",
        "replan_index": 0,
        "schedule_status": ScheduleStatus.SCHEDULED,
        "schedule_reason": ScheduleReason.PLACED_IN_PRIORITY_ORDER,
        "inversion_reason": InversionReason.NONE,
        "scheduled_date": date(2026, 4, 1),
        "day_index": 1,
        "slot_index": index + 1,
        "schedule_rank": index + 1,
        "wait_operating_days": 0,
        "original_scheduled_date": date(2026, 4, 1),
        "original_schedule_rank": index + 1,
        "adjustment_id": "",
        "is_scenario": False,
        "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
    }
    row.update(overrides)
    return row


def _recommendation_row(index: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "policy_id": "pure_risk",
        "model_name": "xgboost_platt",
        "fold_set": "quarterly",
        "fold_id": "quarterly-2026Q2",
        "k_name": "k_1_week",
        "k": 4,
        "target_inspection_id": f"T{index:05d}",
        "establishment_id": f"EST-{index:05d}",
        "inspection_date": date(2026, 4, 1),
        "model_rank": index + 1,
        "final_policy_rank": index + 1,
        "is_selected": True,
        "decision_mechanism": "risk_priority",
        "decision_reason": "selected_by_risk_rank",
        "coverage_eligible": False,
        "score": 0.6,
        "base_score": 0.5,
        "warnings": "none",
        "policy_definition_version": "v1",
    }
    row.update(overrides)
    return row


def _slot_row(day: int, n_slots: int) -> dict[str, object]:
    return {
        "schedule_config_id": CONFIG,
        "fold_set": "quarterly",
        "fold_id": "quarterly-2026Q2",
        "k_name": "k_1_week",
        "k": 4,
        "median_daily_capacity": 4,
        "horizon_days": 1,
        "day_index": day,
        "slot_date": date(2026, 4, day),
        "n_slots": n_slots,
        "capacity_source": "observed_inspection_count",
        "cumulative_slots": n_slots,
        "is_scenario": False,
        "horizon_was_clamped": False,
        "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
    }


@pytest.fixture
def schedule() -> pl.DataFrame:
    return finalize([_schedule_row(i) for i in range(4)], "inspection_schedule")


@pytest.fixture
def recommendations() -> pl.DataFrame:
    return pl.DataFrame([_recommendation_row(i) for i in range(4)])


@pytest.fixture
def slots() -> pl.DataFrame:
    return finalize([_slot_row(1, 4)], "schedule_slots")


def _failed(check: v.ValidationCheck) -> bool:
    return not check.passed and check.severity == SEVERITY_ERROR


class TestFailureInjection:
    """The seventeen cases, in the order the component's specification lists them."""

    def test_01_a_day_over_capacity_turns_the_check_red(
        self, schedule: pl.DataFrame, slots: pl.DataFrame
    ) -> None:
        thin = finalize([_slot_row(1, 3)], "schedule_slots")
        utilization = empty("capacity_utilization")
        assert _failed(v.no_day_exceeds_its_capacity(schedule, thin, utilization))

    def test_02_one_inspection_in_two_slots_turns_the_check_red(self) -> None:
        rows = [_schedule_row(0), _schedule_row(0, slot_index=2, schedule_rank=2)]
        frame = finalize(rows, "inspection_schedule")
        assert _failed(v.no_inspection_occupies_two_slots(frame))

    def test_02b_two_inspections_in_one_slot_turns_the_check_red(self) -> None:
        rows = [_schedule_row(0), _schedule_row(1, slot_index=1, schedule_rank=2)]
        frame = finalize(rows, "inspection_schedule")
        assert _failed(v.no_slot_is_double_booked(frame))

    def test_04_a_lower_priority_row_placed_first_turns_the_check_red(self) -> None:
        """The silent inversion. Ranks 1 and 2 swapped in slot order, with no reason code."""
        rows = [
            _schedule_row(0, slot_index=2, schedule_rank=2),
            _schedule_row(1, slot_index=1, schedule_rank=1),
        ]
        frame = finalize(rows, "inspection_schedule")
        assert _failed(v.schedule_order_follows_policy_rank(frame, empty("priority_preservation")))

    def test_05_an_inversion_without_a_reason_code_turns_the_check_red(self) -> None:
        rows = [
            _schedule_row(0, slot_index=2, schedule_rank=2),
            _schedule_row(1, slot_index=1, schedule_rank=1),
        ]
        frame = finalize(rows, "inspection_schedule")
        assert _failed(v.no_inversion_without_a_reason_code(frame))

    def test_05b_the_same_inversion_with_a_reason_code_is_accepted(self) -> None:
        """The inversion is still real; what makes it acceptable is that it is explained."""
        rows = [
            _schedule_row(
                0,
                slot_index=2,
                schedule_rank=2,
                inversion_reason=InversionReason.DEFERRED_BY_ADJUSTMENT,
            ),
            _schedule_row(1, slot_index=1, schedule_rank=1),
        ]
        frame = finalize(rows, "inspection_schedule")
        assert v.no_inversion_without_a_reason_code(frame).passed

    def test_06_an_unknown_establishment_turns_the_check_red(
        self, recommendations: pl.DataFrame
    ) -> None:
        rows = [_schedule_row(i) for i in range(4)] + [_schedule_row(99)]
        frame = finalize(rows, "inspection_schedule")
        assert _failed(
            v.schedule_rows_originate_in_the_recommendation_universe(frame, recommendations)
        )

    def test_07_a_recommended_row_that_disappears_turns_the_check_red(
        self, recommendations: pl.DataFrame
    ) -> None:
        frame = finalize([_schedule_row(i) for i in range(3)], "inspection_schedule")
        assert _failed(v.every_selected_recommendation_is_accounted_for(frame, recommendations, 1))

    def test_08_a_completed_row_scheduled_again_turns_the_check_red(self) -> None:
        rows = [
            _schedule_row(0),
            _schedule_row(0, replan_index=1, scheduled_date=date(2026, 4, 2), day_index=2),
        ]
        frame = finalize(rows, "inspection_schedule")
        log = finalize(
            [
                {
                    "execution_id": "EX-1",
                    "schedule_config_id": CONFIG,
                    "policy_id": "pure_risk",
                    "fold_id": "quarterly-2026Q2",
                    "k_name": "k_1_week",
                    "target_inspection_id": "T00000",
                    "scheduled_date": date(2026, 4, 1),
                    "plan_scheduled_date": date(2026, 4, 1),
                    "execution_status": "completed",
                    "reason_code": "routine",
                    "actor": "field",
                    "observed_at": "2026-04-01T12:00:00Z",
                    "outcome": "recorded",
                    "triggers_replan": False,
                    "applied_at_replan_index": 0,
                    "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
                }
            ],
            "execution_log",
        )
        assert _failed(v.completed_rows_are_never_rescheduled(frame, log))

    def test_09_a_backlogged_row_missing_from_the_backlog_table_turns_the_check_red(self) -> None:
        rows = [
            _schedule_row(0),
            _schedule_row(
                1,
                schedule_status=ScheduleStatus.BACKLOG,
                schedule_reason=ScheduleReason.CAPACITY_EXHAUSTED_IN_HORIZON,
                scheduled_date=None,
                day_index=None,
                slot_index=None,
                schedule_rank=None,
                wait_operating_days=None,
            ),
        ]
        frame = finalize(rows, "inspection_schedule")
        assert _failed(
            v.backlog_is_exactly_the_unscheduled_remainder(frame, empty("schedule_backlog"))
        )

    def test_10_execution_editing_a_recommendation_turns_the_check_red(
        self, recommendations: pl.DataFrame
    ) -> None:
        """A score that differs across planning runs means execution reached a recommendation."""
        rows = [_schedule_row(0), _schedule_row(0, replan_index=1, score=0.99)]
        frame = finalize(rows, "inspection_schedule")
        assert _failed(v.execution_never_alters_a_recommendation(frame, recommendations))

    def test_10b_an_execution_status_column_on_the_plan_turns_the_check_red(
        self, schedule: pl.DataFrame, recommendations: pl.DataFrame
    ) -> None:
        """The structural form: giving execution a column to write into is itself the defect."""
        contaminated = schedule.with_columns(pl.lit("completed").alias("execution_status"))
        assert _failed(v.execution_never_alters_a_recommendation(contaminated, recommendations))

    def test_11_a_replan_touching_an_earlier_day_turns_the_check_red(self) -> None:
        rows = [
            _schedule_row(0),
            _schedule_row(0, replan_index=1, scheduled_date=date(2026, 4, 5), day_index=5),
        ]
        frame = finalize(rows, "inspection_schedule")
        runs = finalize(
            [
                _run_row(0, None, None, "original_plan"),
                _run_row(1, 0, date(2026, 4, 3), "execution_not_performed"),
            ],
            "replanning_runs",
        )
        assert _failed(
            v.no_execution_event_changes_an_earlier_schedule(runs, frame, empty("execution_log"))
        )

    def test_12_an_invalid_slot_count_turns_the_check_red(
        self, recommendations: pl.DataFrame
    ) -> None:
        wrong = finalize([_slot_row(1, 99)], "schedule_slots")
        assert _failed(
            v.capacity_matches_the_declared_mode(wrong, recommendations, {"quarterly-2026Q2": 4})
        )

    def test_13_an_out_of_order_horizon_turns_the_check_red(
        self, recommendations: pl.DataFrame
    ) -> None:
        rows = [_slot_row(1, 4), _slot_row(2, 4)]
        rows[1]["day_index"] = 5
        frame = finalize(rows, "schedule_slots")
        assert _failed(
            v.horizon_is_ordered_contiguous_and_real(
                frame, recommendations, {"quarterly-2026Q2": 4}
            )
        )

    def test_13b_a_horizon_day_that_is_not_an_operating_day_turns_the_check_red(
        self, recommendations: pl.DataFrame
    ) -> None:
        """A generated calendar would look exactly like this: a plausible date nobody worked."""
        invented = _slot_row(1, 4)
        invented["slot_date"] = date(2026, 4, 4)
        frame = finalize([invented], "schedule_slots")
        assert _failed(
            v.horizon_is_ordered_contiguous_and_real(
                frame, recommendations, {"quarterly-2026Q2": 4}
            )
        )

    def test_14_a_duplicated_planning_run_id_turns_the_check_red(self) -> None:
        runs = finalize(
            [_run_row(0, None, None, "original_plan"), _run_row(1, 0, date(2026, 4, 2), "x")],
            "replanning_runs",
        ).with_columns(pl.lit("PR-same").alias("planning_run_id"))
        assert _failed(v.planning_runs_are_unique_and_chained(runs))

    def test_14b_a_broken_lineage_turns_the_check_red(self) -> None:
        runs = finalize(
            [
                _run_row(0, None, None, "original_plan"),
                _run_row(2, 0, date(2026, 4, 2), "execution_not_performed"),
            ],
            "replanning_runs",
        )
        assert _failed(v.planning_runs_are_unique_and_chained(runs))

    def test_15_an_override_mistaken_for_an_adjustment_turns_the_check_red(self) -> None:
        log = finalize([_adjustment_row("OV-2026-0417")], "schedule_adjustment_log")
        assert _failed(v.adjustments_are_not_overrides(log, ["OV-2026-0417"]))

    def test_15b_an_override_verb_in_the_adjustment_log_turns_the_check_red(self) -> None:
        log = finalize(
            [_adjustment_row("SA-0001", action="force_include")], "schedule_adjustment_log"
        )
        assert _failed(v.adjustments_are_not_overrides(log, []))

    def test_16_an_adjustment_overwriting_the_original_assignment_turns_the_check_red(
        self,
    ) -> None:
        rows = [
            _schedule_row(
                0,
                adjustment_id="SA-0001",
                original_scheduled_date=None,
                original_schedule_rank=None,
            )
        ]
        frame = finalize(rows, "inspection_schedule")
        log = finalize([_adjustment_row("SA-0001")], "schedule_adjustment_log")
        assert _failed(v.adjustments_preserve_the_original_assignment(frame, log))

    def test_17_a_changed_input_checksum_turns_the_check_red(self) -> None:
        assert _failed(v.inputs_were_not_modified({"a": "abc"}, {"a": "def"}))


class TestTheReserveIsNeverDisplaced:
    def test_displacing_a_coverage_reserve_row_turns_the_check_red(self) -> None:
        """Not on the numbered list, and the most consequential of all of them.

        A scheduling adjustment that spent the coverage allocation would be a policy change
        nobody made, arriving through a layer that does not own policy.
        """
        schedule = finalize(
            [_schedule_row(0, decision_mechanism="coverage_reserve")], "inspection_schedule"
        )
        log = finalize(
            [_adjustment_row("SA-0001", displaced_target_inspection_id="T00000")],
            "schedule_adjustment_log",
        )
        assert _failed(v.adjustments_never_displace_a_coverage_reserve_row(log, schedule))


class TestAdvisoriesNeverFailARun:
    """The other half of the contract, and the easier half to break by accident."""

    def _summary(self, **overrides: object) -> pl.DataFrame:
        row: dict[str, object] = {
            **CELL,
            "k": 4,
            "median_daily_capacity": 4,
            "horizon_days": 1,
            "horizon_start_date": date(2026, 4, 1),
            "horizon_end_date": date(2026, 4, 1),
            "horizon_slots": 4,
            "n_recommended": 4,
            "n_scheduled": 4,
            "n_backlog": 0,
            "n_deferred": 0,
            "n_cancelled": 0,
            "idle_slots": 0,
            "capacity_utilization": 1.0,
            "horizon_was_clamped": False,
            "n_adjustments_applied": 0,
            "n_execution_events": 0,
            "planning_run_id": "PR-test",
            "replan_index": 0,
            "is_scenario": False,
            "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
        }
        row.update(overrides)
        return finalize([row], "schedule_summary")

    def test_a_backlog_is_advisory(self) -> None:
        check = v.every_recommendation_was_scheduled(self._summary(n_backlog=2, n_scheduled=2))
        assert not check.passed
        assert check.severity == SEVERITY_WARN

    def test_idle_capacity_is_advisory(self) -> None:
        check = v.capacity_is_fully_utilized(self._summary(idle_slots=3, n_scheduled=1))
        assert not check.passed
        assert check.severity == SEVERITY_WARN

    def test_a_lost_coverage_reserve_is_advisory(self) -> None:
        """The headline. Making this an error would make re-ranking the cheapest way to green."""
        preservation = finalize([_preservation_row()], "priority_preservation")
        check = v.the_coverage_reserve_survived_scheduling(preservation)
        assert not check.passed
        assert check.severity == SEVERITY_WARN

    def test_the_scenario_label_is_advisory(self) -> None:
        frame = finalize([_schedule_row(0, is_scenario=True)], "inspection_schedule")
        check = v.the_scenario_is_not_observed_fact(frame)
        assert not check.passed
        assert check.severity == SEVERITY_WARN

    def test_a_recurring_establishment_is_advisory(self) -> None:
        """1,573 establishment-fold pairs do this on real data. An error would be a red build."""
        rows = [_schedule_row(0), _schedule_row(1, establishment_id="EST-00000", slot_index=2)]
        frame = finalize(rows, "inspection_schedule")
        check = v.an_establishment_recurs_within_a_horizon(frame)
        assert not check.passed
        assert check.severity == SEVERITY_WARN

    def test_a_run_of_only_advisories_does_not_fail(self) -> None:
        checks = [
            v.every_recommendation_was_scheduled(self._summary(n_backlog=2, n_scheduled=2)),
            v.capacity_is_fully_utilized(self._summary(idle_slots=3, n_scheduled=1)),
        ]
        assert not v.has_failures(checks)
        assert len(v.advisory_findings(checks)) == 2

    def test_the_report_leads_with_the_boundary(self) -> None:
        text = v.format_report([v.inputs_were_not_modified({}, {})])
        assert "does not mean the city has enough capacity" in text


def _run_row(
    index: int, parent: int | None, boundary: date | None, trigger: str
) -> dict[str, object]:
    return {
        "schedule_config_id": CONFIG,
        "policy_id": "pure_risk",
        "fold_set": "quarterly",
        "fold_id": "quarterly-2026Q2",
        "k_name": "k_1_week",
        "planning_run_id": f"PR-{index}",
        "replan_index": index,
        "parent_replan_index": parent,
        "replan_from_date": boundary,
        "trigger": trigger,
        "n_preserved_completed": 0,
        "n_preserved_past": 0,
        "n_returned_to_queue": 0,
        "n_cancelled": 0,
        "n_newly_scheduled": 0,
        "n_still_backlog": 0,
        "remaining_slots": 0,
        "execution_log_sha256": "",
        "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
    }


def _adjustment_row(adjustment_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "adjustment_id": adjustment_id,
        "schedule_config_id": CONFIG,
        "policy_id": "pure_risk",
        "fold_id": "quarterly-2026Q2",
        "k_name": "k_1_week",
        "target_inspection_id": "T00001",
        "action": "defer_to_date",
        "target_date": "2026-04-02",
        "reason_code": "establishment_closed",
        "actor": "district.supervisor.4",
        "decided_at": "2026-08-26T09:00:00Z",
        "original_status": "scheduled",
        "original_scheduled_date": date(2026, 4, 1),
        "original_schedule_rank": 2,
        "final_status": "deferred",
        "final_scheduled_date": date(2026, 4, 2),
        "displaced_target_inspection_id": "",
        "displaced_landed_status": "",
        "outcome": "applied",
        "planning_run_id": "PR-test",
        "replan_index": 1,
        "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
    }
    row.update(overrides)
    return row


def _preservation_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        **CELL,
        "k": 4,
        "n_scheduled": 2,
        "n_inversions": 0,
        "max_inversion_depth": 0,
        "rank_spearman": 1.0,
        "strict_priority_preserved": True,
        "n_rows_with_inversion_reason": 0,
        "queue_coverage": 0.5,
        "worst_scheduled_policy_rank": 2,
        "best_backlogged_policy_rank": 3,
        "mean_wait_operating_days": 0.0,
        "median_wait_operating_days": 0.0,
        "max_wait_operating_days": 0,
        "n_risk_recommended": 2,
        "n_reserve_recommended": 2,
        "n_risk_scheduled": 2,
        "n_reserve_scheduled": 0,
        "reserve_share_recommended": 0.5,
        "reserve_share_scheduled": 0.0,
        "reserve_share_delta": -0.5,
        "reserve_slots_lost": 2,
        "n_adjusted": 0,
        "n_displaced_by_adjustment": 0,
        "n_execution_completed": 0,
        "n_execution_not_performed": 0,
        "n_execution_cancelled": 0,
        "n_no_execution_record": 0,
        "is_scenario": False,
        "schedule_definition_version": SCHEDULE_DEFINITION_VERSION,
    }
    row.update(overrides)
    return row


class TestNoOutcomeReachesTheLayer:
    def test_a_label_column_anywhere_turns_the_check_red(self, schedule: pl.DataFrame) -> None:
        """A scheduler that could read the outcome could order by it."""
        contaminated = schedule.with_columns(pl.lit(1).alias("target"))
        assert _failed(
            v.no_outcome_column_reaches_the_schedule({"inspection_schedule": contaminated})
        )

    def test_a_clean_layer_passes(self, schedule: pl.DataFrame, slots: pl.DataFrame) -> None:
        assert v.no_outcome_column_reaches_the_schedule(
            {"inspection_schedule": schedule, "schedule_slots": slots}
        ).passed


class TestProvenanceDrift:
    def test_a_drifted_score_turns_the_check_red(self, recommendations: pl.DataFrame) -> None:
        frame = finalize([_schedule_row(0, score=0.01)], "inspection_schedule")
        assert _failed(v.c13_provenance_is_preserved(frame, recommendations))

    def test_an_untouched_plan_passes(
        self, schedule: pl.DataFrame, recommendations: pl.DataFrame
    ) -> None:
        assert v.c13_provenance_is_preserved(schedule, recommendations).passed
