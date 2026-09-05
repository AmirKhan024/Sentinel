"""The execution contract and rolling re-planning.

The distinction this file guards is the one the component is built around: an execution event
*records* and never *edits*. It cannot touch a score, a rank, a mechanism, an original
assignment, or a plan that was already written. A re-plan produces a new plan beside the old
one, and both are kept.

The one exemption is narrow and deliberate: a row the field reported as ``not_performed`` moves
even though its day has passed. Freezing it would strand exactly the inspection the report was
filed to rescue.
"""

from __future__ import annotations

from datetime import date

import pytest

from sentinel.scheduling.definitions import (
    EXECUTION_REQUIRED_FIELDS,
    NO_EXECUTION_RECORD,
    ScheduleStatus,
)
from sentinel.scheduling.execution import (
    OUTCOME_NO_OP_NOT_SCHEDULED,
    OUTCOME_RECORDED,
    OUTCOME_ROW_NOT_IN_PLAN,
    ExecutionError,
    execution_log_rows,
    execution_summary_row,
    observed_date,
    parse_execution_events,
    record_execution,
    triggers_replan,
)
from sentinel.scheduling.replan import ReplanError, original_run_row, replan, replan_point

from .conftest import make_execution_event, make_plan, make_queue


class TestParsing:
    def test_a_well_formed_file_parses(self) -> None:
        assert parse_execution_events([make_execution_event(1)])[0].execution_id == "EX-0001"

    @pytest.mark.parametrize("field", list(EXECUTION_REQUIRED_FIELDS))
    def test_a_blank_field_refuses_the_whole_file(self, field: str) -> None:
        """A completion rate over an arbitrary subset is worse than no rate at all."""
        with pytest.raises(ExecutionError, match="missing or blank"):
            parse_execution_events([make_execution_event(1, **{field: ""})])

    @pytest.mark.parametrize("field", list(EXECUTION_REQUIRED_FIELDS))
    def test_a_missing_field_refuses_the_whole_file(self, field: str) -> None:
        row = make_execution_event(1)
        del row[field]
        with pytest.raises(ExecutionError):
            parse_execution_events([row])

    def test_an_unknown_status_is_refused(self) -> None:
        with pytest.raises(ExecutionError, match="Known:"):
            parse_execution_events([make_execution_event(1, execution_status="went_fine")])

    def test_the_derived_no_record_category_cannot_be_supplied(self) -> None:
        """ "We do not know" is a summary category, never something a person files."""
        with pytest.raises(ExecutionError, match="derived summary category"):
            parse_execution_events([make_execution_event(1, execution_status=NO_EXECUTION_RECORD)])

    def test_a_duplicate_id_refuses_the_file(self) -> None:
        with pytest.raises(ExecutionError, match="duplicate execution_id"):
            parse_execution_events([make_execution_event(1), make_execution_event(1)])

    def test_parsing_sorts_by_id_regardless_of_file_order(self) -> None:
        parsed = parse_execution_events(
            [make_execution_event(7), make_execution_event(2), make_execution_event(4)]
        )
        assert [e.execution_id for e in parsed] == ["EX-0002", "EX-0004", "EX-0007"]

    def test_an_unknown_field_refuses_the_file(self) -> None:
        with pytest.raises(ExecutionError):
            parse_execution_events([make_execution_event(1, mood="cheerful")])


class TestObservedDate:
    def test_a_zulu_timestamp_resolves_to_its_date(self) -> None:
        event = parse_execution_events([make_execution_event(1)])[0]
        assert observed_date(event) == date(2026, 4, 1)

    def test_a_malformed_timestamp_is_refused(self) -> None:
        event = parse_execution_events([make_execution_event(1, observed_at="last Tuesday")])[0]
        with pytest.raises(ExecutionError, match="not an ISO timestamp"):
            observed_date(event)


class TestTriggering:
    def test_only_a_not_performed_report_returns_a_row_to_the_queue(self) -> None:
        assert triggers_replan("not_performed")
        assert not triggers_replan("completed")
        assert not triggers_replan("cancelled_in_field")


class TestRecording:
    def test_a_scheduled_row_is_recorded(self) -> None:
        plan = make_plan([3], k=3, median=3)
        events = parse_execution_events([make_execution_event(1, target_inspection_id="T00000")])
        outcomes, statuses = record_execution(plan, events)
        assert outcomes[0].outcome == OUTCOME_RECORDED
        assert statuses["T00000"] == "completed"

    def test_a_backlogged_row_is_a_logged_no_op(self) -> None:
        plan = make_plan([1], k=3, median=3)
        events = parse_execution_events([make_execution_event(1, target_inspection_id="T00002")])
        outcomes, statuses = record_execution(plan, events)
        assert outcomes[0].outcome == OUTCOME_NO_OP_NOT_SCHEDULED
        assert statuses == {}

    def test_an_unknown_row_is_logged_rather_than_dropped(self) -> None:
        plan = make_plan([3], k=3, median=3)
        events = parse_execution_events([make_execution_event(1, target_inspection_id="T99999")])
        outcomes, _ = record_execution(plan, events)
        assert outcomes[0].outcome == OUTCOME_ROW_NOT_IN_PLAN

    def test_the_plan_is_not_touched_by_recording(self) -> None:
        plan = make_plan([3], k=3, median=3)
        before = plan.by_id()["T00000"]
        events = parse_execution_events([make_execution_event(1, target_inspection_id="T00000")])
        record_execution(plan, events)
        assert plan.by_id()["T00000"] == before

    def test_the_log_keeps_both_dates_without_merging_them(self) -> None:
        """A field log that disagrees with the plan is a fact, not a value to overwrite."""
        plan = make_plan([3], k=3, median=3)
        events = parse_execution_events(
            [make_execution_event(1, target_inspection_id="T00000", scheduled_date="2026-04-09")]
        )
        outcomes, _ = record_execution(plan, events)
        row = execution_log_rows(outcomes)[0]
        assert row["scheduled_date"] == date(2026, 4, 9)
        assert row["plan_scheduled_date"] == date(2026, 4, 1)


class TestExecutionSummary:
    def test_unreported_rows_are_their_own_category(self) -> None:
        """Silently treating an absent report as a failure would manufacture a rate."""
        plan = make_plan([3], k=3, median=3)
        row = execution_summary_row(plan, {"T00000": "completed"})
        assert row["n_completed"] == 1
        assert row["n_no_execution_record"] == 2

    def test_the_completion_rate_is_over_the_scheduled_rows(self) -> None:
        plan = make_plan([2], k=2, median=2)
        row = execution_summary_row(plan, {"T00000": "completed", "T00001": "completed"})
        assert row["completion_rate"] == pytest.approx(1.0)

    def test_a_cell_with_nothing_scheduled_reports_no_rate(self) -> None:
        plan = make_plan([1], k=1, median=1)
        row = execution_summary_row(plan, {})
        assert row["n_scheduled"] == 1
        assert row["completion_rate"] == pytest.approx(0.0)


class TestReplanPoint:
    def test_the_boundary_is_the_day_after_the_latest_triggering_report(self) -> None:
        plan = make_plan([2, 2, 2], k=6, median=2)
        observed = {"T00000": date(2026, 4, 1), "T00002": date(2026, 4, 2)}
        statuses = {"T00000": "not_performed", "T00002": "not_performed"}
        assert replan_point(plan, statuses, observed) == date(2026, 4, 3)

    def test_a_completed_report_alone_triggers_nothing(self) -> None:
        plan = make_plan([2, 2], k=4, median=2)
        assert replan_point(plan, {"T00000": "completed"}, {"T00000": date(2026, 4, 1)}) is None

    def test_a_report_on_the_last_day_leaves_nothing_to_replan(self) -> None:
        plan = make_plan([2, 2], k=4, median=2)
        point = replan_point(plan, {"T00003": "not_performed"}, {"T00003": date(2026, 4, 2)})
        assert point is None


class TestReplan:
    def _replan(self, plan: object, queue: object, statuses: dict[str, str], boundary: date):
        return replan(
            plan,  # type: ignore[arg-type]
            queue,  # type: ignore[arg-type]
            statuses,
            from_date=boundary,
            planning_run_id="PR-next",
            replan_index=1,
        )

    def test_a_completed_row_keeps_its_slot(self) -> None:
        """Rescheduling something that already happened is not a plan, it is a contradiction."""
        plan = make_plan([2, 2], k=4, median=2)
        queue = make_queue(4)
        new, _ = self._replan(plan, queue, {"T00000": "completed"}, date(2026, 4, 2))
        assert new.by_id()["T00000"].slot_date == date(2026, 4, 1)

    def test_a_past_row_nobody_reported_on_is_frozen(self) -> None:
        plan = make_plan([2, 2], k=4, median=2)
        queue = make_queue(4)
        new, run = self._replan(plan, queue, {"T00003": "not_performed"}, date(2026, 4, 2))
        assert new.by_id()["T00001"].slot_date == date(2026, 4, 1)
        assert run["n_preserved_past"] >= 1

    def test_a_not_performed_row_returns_to_the_queue_even_from_a_past_day(self) -> None:
        """The exemption. Freezing it would strand the inspection the report exists to rescue."""
        plan = make_plan([2, 2], k=4, median=2)
        queue = make_queue(4)
        new, run = self._replan(plan, queue, {"T00000": "not_performed"}, date(2026, 4, 2))
        assert run["n_returned_to_queue"] == 1
        assert new.by_id()["T00000"].slot_date == date(2026, 4, 2)

    def test_a_field_cancellation_does_not_come_back(self) -> None:
        """A cancellation is a removal, and removals are not backfilled."""
        plan = make_plan([2, 2], k=4, median=2)
        queue = make_queue(4)
        new, _ = self._replan(plan, queue, {"T00002": "cancelled_in_field"}, date(2026, 4, 2))
        assert new.by_id()["T00002"].status == ScheduleStatus.CANCELLED

    def test_freed_capacity_is_filled_from_the_backlog_in_rank_order(self) -> None:
        """A day that did not happen is capacity that still exists; stranding it helps nobody."""
        plan = make_plan([2, 2], k=6, median=3)
        queue = make_queue(6)
        assert plan.by_id()["T00004"].status == ScheduleStatus.BACKLOG
        new, _ = self._replan(plan, queue, {"T00002": "cancelled_in_field"}, date(2026, 4, 2))
        assert new.by_id()["T00004"].occupies_a_slot

    def test_the_original_assignment_is_copied_forward_untouched(self) -> None:
        plan = make_plan([2, 2], k=4, median=2)
        queue = make_queue(4)
        new, _ = self._replan(plan, queue, {"T00000": "not_performed"}, date(2026, 4, 2))
        assert new.by_id()["T00000"].original_slot_date == date(2026, 4, 1)

    def test_ordering_is_still_the_policy_rank(self) -> None:
        plan = make_plan([1, 4], k=5, median=3)
        queue = make_queue(5)
        new, _ = self._replan(plan, queue, {"T00000": "not_performed"}, date(2026, 4, 2))
        placed = sorted(
            (p for p in new.placements if p.occupies_a_slot),
            key=lambda p: (p.slot_date, p.slot_index),
        )
        by_rank = {row.target_inspection_id: row.final_policy_rank for row in queue}
        ranks = [by_rank[p.target_inspection_id] for p in placed if p.slot_date >= date(2026, 4, 2)]
        assert ranks == sorted(ranks)

    def test_no_row_disappears(self) -> None:
        plan = make_plan([2, 2], k=6, median=3)
        queue = make_queue(6)
        new, _ = self._replan(plan, queue, {"T00000": "not_performed"}, date(2026, 4, 2))
        assert len(new.placements) == 6

    def test_the_previous_plan_is_untouched(self) -> None:
        plan = make_plan([2, 2], k=4, median=2)
        queue = make_queue(4)
        before = plan.by_id()["T00000"].slot_date
        self._replan(plan, queue, {"T00000": "not_performed"}, date(2026, 4, 2))
        assert plan.by_id()["T00000"].slot_date == before

    def test_a_non_advancing_index_is_refused(self) -> None:
        plan = make_plan([2, 2], k=4, median=2)
        queue = make_queue(4)
        with pytest.raises(ReplanError, match="advance the index"):
            replan(
                plan,
                queue,
                {},
                from_date=date(2026, 4, 2),
                planning_run_id="PR-x",
                replan_index=0,
            )

    def test_two_chained_replans_keep_the_lineage(self) -> None:
        plan = make_plan([2, 2, 2], k=6, median=2)
        queue = make_queue(6)
        first, run_one = self._replan(plan, queue, {"T00000": "not_performed"}, date(2026, 4, 2))
        second, run_two = replan(
            first,
            queue,
            {"T00002": "not_performed"},
            from_date=date(2026, 4, 3),
            planning_run_id="PR-third",
            replan_index=2,
        )
        assert run_one["parent_replan_index"] == 0
        assert run_two["parent_replan_index"] == 1
        assert second.replan_index == 2


class TestTheOriginalRunRow:
    def test_the_first_plan_is_itself_a_planning_run(self) -> None:
        """More honest than an empty table: a reader should see the lineage start at zero."""
        row = original_run_row(make_plan([3], k=3, median=3))
        assert row["replan_index"] == 0
        assert row["trigger"] == "original_plan"
        assert row["parent_replan_index"] is None
        assert row["replan_from_date"] is None
