"""The scheduling adjustment contract, and what applying one costs.

Three properties this file exists to pin down. The parse is **all or nothing**, because a
half-applied adjustment file produces a schedule nobody authorised. Application is in
**id order**, so re-serialising the JSON cannot change the outcome. And a move onto a full day
**never displaces a coverage-reserve row** -- taking the slot from the coverage allocation
would quietly convert a scheduling change into a coverage cut, which is a policy change nobody
made.
"""

from __future__ import annotations

from datetime import date

import pytest

from sentinel.scheduling.adjustments import (
    OUTCOME_APPLIED,
    OUTCOME_NO_OP_ALREADY_ON_DATE,
    OUTCOME_NO_OP_NOT_SCHEDULED,
    OUTCOME_ROW_NOT_IN_PLAN,
    AdjustmentError,
    adjustment_log_rows,
    apply_adjustments,
    parse_adjustments,
)
from sentinel.scheduling.definitions import (
    ADJUSTMENT_REQUIRED_FIELDS,
    InversionReason,
    ScheduleStatus,
)

from .conftest import make_adjustment, make_plan, make_queue


class TestParsing:
    def test_a_well_formed_file_parses(self) -> None:
        parsed = parse_adjustments([make_adjustment(1)])
        assert parsed[0].adjustment_id == "SA-0001"

    @pytest.mark.parametrize("field", [f for f in ADJUSTMENT_REQUIRED_FIELDS if f != "target_date"])
    def test_a_blank_field_refuses_the_whole_file(self, field: str) -> None:
        """Every field, one test each. A file with a hole is a file somebody was mid-edit on."""
        rows = [make_adjustment(1), make_adjustment(2, **{field: "   "})]
        with pytest.raises(AdjustmentError, match="missing or blank"):
            parse_adjustments(rows)

    @pytest.mark.parametrize("field", [f for f in ADJUSTMENT_REQUIRED_FIELDS if f != "target_date"])
    def test_a_missing_field_refuses_the_whole_file(self, field: str) -> None:
        row = make_adjustment(1)
        del row[field]
        with pytest.raises(AdjustmentError):
            parse_adjustments([row])

    def test_an_absent_target_date_is_refused_even_for_a_cancel(self) -> None:
        row = make_adjustment(1, action="cancel")
        del row["target_date"]
        with pytest.raises(AdjustmentError, match="target_date is required"):
            parse_adjustments([row])

    def test_a_cancel_carrying_a_date_is_refused(self) -> None:
        """Ambiguous between striking the row and moving it, so it is neither."""
        with pytest.raises(AdjustmentError, match="ambiguous"):
            parse_adjustments([make_adjustment(1, action="cancel", target_date="2026-04-03")])

    def test_a_move_without_a_date_is_refused(self) -> None:
        with pytest.raises(AdjustmentError, match="requires a target_date"):
            parse_adjustments([make_adjustment(1, target_date="")])

    def test_an_unknown_verb_is_refused_and_the_known_ones_named(self) -> None:
        with pytest.raises(AdjustmentError, match="Known:"):
            parse_adjustments([make_adjustment(1, action="reschedule_everything")])

    def test_an_override_verb_is_not_an_adjustment_verb(self) -> None:
        with pytest.raises(AdjustmentError, match="unknown action"):
            parse_adjustments([make_adjustment(1, action="force_include")])

    def test_a_duplicate_id_refuses_the_file(self) -> None:
        with pytest.raises(AdjustmentError, match="duplicate adjustment_id"):
            parse_adjustments([make_adjustment(1), make_adjustment(1)])

    def test_an_unknown_field_refuses_the_file(self) -> None:
        with pytest.raises(AdjustmentError):
            parse_adjustments([make_adjustment(1, surprise="yes")])

    def test_parsing_sorts_by_id_regardless_of_file_order(self) -> None:
        """Re-serialising the JSON must not change the schedule."""
        parsed = parse_adjustments([make_adjustment(9), make_adjustment(2), make_adjustment(5)])
        assert [a.adjustment_id for a in parsed] == ["SA-0002", "SA-0005", "SA-0009"]

    def test_a_non_object_row_is_refused(self) -> None:
        with pytest.raises(AdjustmentError, match="expected an object"):
            parse_adjustments(["not an object"])  # type: ignore[list-item]


class TestApplication:
    def test_a_defer_moves_the_row_and_marks_it_deferred(self) -> None:
        plan = make_plan([3, 3], k=6, median=3)
        queue = make_queue(6)
        adjustments = parse_adjustments(
            [make_adjustment(1, target_inspection_id="T00000", target_date="2026-04-02")]
        )
        outcomes, adjusted = apply_adjustments(plan, adjustments, queue)
        moved = adjusted.by_id()["T00000"]
        assert outcomes[0].outcome == OUTCOME_APPLIED
        assert moved.slot_date == date(2026, 4, 2)
        assert moved.status == ScheduleStatus.DEFERRED
        assert moved.inversion_reason == InversionReason.DEFERRED_BY_ADJUSTMENT

    def test_an_advance_moves_the_row_earlier(self) -> None:
        plan = make_plan([3, 3], k=6, median=3)
        queue = make_queue(6)
        adjustments = parse_adjustments(
            [
                make_adjustment(
                    1,
                    target_inspection_id="T00005",
                    action="advance_to_date",
                    target_date="2026-04-01",
                )
            ]
        )
        _, adjusted = apply_adjustments(plan, adjustments, queue)
        assert adjusted.by_id()["T00005"].slot_date == date(2026, 4, 1)

    def test_the_original_assignment_survives_the_move(self) -> None:
        plan = make_plan([3, 3], k=6, median=3)
        queue = make_queue(6)
        adjustments = parse_adjustments(
            [make_adjustment(1, target_inspection_id="T00000", target_date="2026-04-02")]
        )
        _, adjusted = apply_adjustments(plan, adjustments, queue)
        assert adjusted.by_id()["T00000"].original_slot_date == date(2026, 4, 1)
        assert adjusted.by_id()["T00000"].original_schedule_rank == 1

    def test_the_input_plan_is_not_mutated(self) -> None:
        plan = make_plan([3, 3], k=6, median=3)
        queue = make_queue(6)
        before = plan.by_id()["T00000"].slot_date
        adjustments = parse_adjustments(
            [make_adjustment(1, target_inspection_id="T00000", target_date="2026-04-02")]
        )
        apply_adjustments(plan, adjustments, queue)
        assert plan.by_id()["T00000"].slot_date == before

    def test_a_cancel_removes_the_row_and_does_not_backfill(self) -> None:
        """The supervisor who struck a row did not ask for a replacement."""
        plan = make_plan([3], k=5, median=5)
        queue = make_queue(5)
        adjustments = parse_adjustments(
            [make_adjustment(1, target_inspection_id="T00000", action="cancel", target_date="")]
        )
        _, adjusted = apply_adjustments(plan, adjustments, queue)
        assert adjusted.by_id()["T00000"].status == ScheduleStatus.CANCELLED
        assert adjusted.n_scheduled == 2
        assert adjusted.by_id()["T00003"].status == ScheduleStatus.BACKLOG

    def test_a_move_onto_a_full_day_displaces_the_lowest_ranked_risk_row(self) -> None:
        plan = make_plan([2, 2], k=4, median=2)
        queue = make_queue(4)
        adjustments = parse_adjustments(
            [make_adjustment(1, target_inspection_id="T00000", target_date="2026-04-02")]
        )
        outcomes, adjusted = apply_adjustments(plan, adjustments, queue)
        assert outcomes[0].displaced_target_inspection_id == "T00003"
        assert adjusted.by_id()["T00000"].slot_date == date(2026, 4, 2)
        assert adjusted.by_id()["T00003"].slot_date == date(2026, 4, 1)

    def test_a_displaced_row_never_disappears(self) -> None:
        plan = make_plan([2, 2], k=4, median=2)
        queue = make_queue(4)
        adjustments = parse_adjustments(
            [make_adjustment(1, target_inspection_id="T00000", target_date="2026-04-02")]
        )
        _, adjusted = apply_adjustments(plan, adjustments, queue)
        assert len(adjusted.placements) == 4
        assert all(p.status != "" for p in adjusted.placements)

    def test_the_coverage_reserve_is_never_displaced(self) -> None:
        """Taking the reserve's slot would convert a scheduling change into a coverage cut."""
        plan = make_plan([2, 2], k=4, median=2, reserve_tail=2)
        queue = make_queue(4, reserve_tail=2)
        adjustments = parse_adjustments(
            [make_adjustment(1, target_inspection_id="T00000", target_date="2026-04-02")]
        )
        with pytest.raises(AdjustmentError, match="coverage reserve"):
            apply_adjustments(plan, adjustments, queue)

    def test_a_target_outside_the_horizon_is_refused(self) -> None:
        """Moving a row past the horizon would extend it -- a capacity increase by another name."""
        plan = make_plan([3, 3], k=6, median=3)
        queue = make_queue(6)
        adjustments = parse_adjustments(
            [make_adjustment(1, target_inspection_id="T00000", target_date="2026-05-01")]
        )
        with pytest.raises(AdjustmentError, match="not an operating day"):
            apply_adjustments(plan, adjustments, queue)

    def test_a_defer_that_moves_a_row_earlier_is_refused(self) -> None:
        """The log must say which the supervisor meant."""
        plan = make_plan([3, 3], k=6, median=3)
        queue = make_queue(6)
        adjustments = parse_adjustments(
            [make_adjustment(1, target_inspection_id="T00005", target_date="2026-04-01")]
        )
        with pytest.raises(AdjustmentError, match="Use advance_to_date"):
            apply_adjustments(plan, adjustments, queue)

    def test_an_advance_that_moves_a_row_later_is_refused(self) -> None:
        plan = make_plan([3, 3], k=6, median=3)
        queue = make_queue(6)
        adjustments = parse_adjustments(
            [
                make_adjustment(
                    1,
                    target_inspection_id="T00000",
                    action="advance_to_date",
                    target_date="2026-04-02",
                )
            ]
        )
        with pytest.raises(AdjustmentError, match="Use defer_to_date"):
            apply_adjustments(plan, adjustments, queue)


class TestOutcomes:
    def test_a_move_to_the_same_day_is_a_logged_no_op(self) -> None:
        """A supervisor asking and it making no difference is an audit fact."""
        plan = make_plan([3, 3], k=6, median=3)
        queue = make_queue(6)
        adjustments = parse_adjustments(
            [make_adjustment(1, target_inspection_id="T00000", target_date="2026-04-01")]
        )
        outcomes, _ = apply_adjustments(plan, adjustments, queue)
        assert outcomes[0].outcome == OUTCOME_NO_OP_ALREADY_ON_DATE

    def test_moving_a_backlogged_row_is_a_logged_no_op(self) -> None:
        plan = make_plan([2], k=5, median=5)
        queue = make_queue(5)
        adjustments = parse_adjustments(
            [make_adjustment(1, target_inspection_id="T00004", target_date="2026-04-01")]
        )
        outcomes, _ = apply_adjustments(plan, adjustments, queue)
        assert outcomes[0].outcome == OUTCOME_NO_OP_NOT_SCHEDULED

    def test_an_unknown_row_is_logged_rather_than_silently_dropped(self) -> None:
        plan = make_plan([3], k=3, median=3)
        queue = make_queue(3)
        adjustments = parse_adjustments(
            [make_adjustment(1, target_inspection_id="T99999", target_date="2026-04-01")]
        )
        outcomes, _ = apply_adjustments(plan, adjustments, queue)
        assert outcomes[0].outcome == OUTCOME_ROW_NOT_IN_PLAN


class TestTheLog:
    def test_every_adjustment_produces_a_row_whatever_the_outcome(self) -> None:
        plan = make_plan([3], k=3, median=3)
        queue = make_queue(3)
        adjustments = parse_adjustments(
            [
                make_adjustment(1, target_inspection_id="T99999", target_date="2026-04-01"),
                make_adjustment(2, target_inspection_id="T00000", target_date="2026-04-01"),
            ]
        )
        outcomes, _ = apply_adjustments(plan, adjustments, queue)
        assert len(adjustment_log_rows(outcomes)) == 2

    def test_the_log_carries_the_attribution_the_supervisor_supplied(self) -> None:
        plan = make_plan([3], k=3, median=3)
        queue = make_queue(3)
        adjustments = parse_adjustments(
            [make_adjustment(1, target_inspection_id="T00000", target_date="2026-04-01")]
        )
        outcomes, _ = apply_adjustments(plan, adjustments, queue)
        row = adjustment_log_rows(outcomes)[0]
        assert row["actor"] == "district.supervisor.4"
        assert row["reason_code"] == "establishment_closed"
        assert row["decided_at"] == "2026-08-26T09:00:00Z"

    def test_the_log_records_the_original_assignment_beside_the_final_one(self) -> None:
        plan = make_plan([3, 3], k=6, median=3)
        queue = make_queue(6)
        adjustments = parse_adjustments(
            [make_adjustment(1, target_inspection_id="T00000", target_date="2026-04-02")]
        )
        outcomes, _ = apply_adjustments(plan, adjustments, queue)
        row = adjustment_log_rows(outcomes)[0]
        assert row["original_scheduled_date"] == date(2026, 4, 1)
        assert row["final_scheduled_date"] == date(2026, 4, 2)

    def test_the_log_carries_no_queue_column(self) -> None:
        """An adjustment changes when, never who. A selection column here would blur the two."""
        plan = make_plan([3], k=3, median=3)
        queue = make_queue(3)
        adjustments = parse_adjustments(
            [make_adjustment(1, target_inspection_id="T00000", target_date="2026-04-01")]
        )
        outcomes, _ = apply_adjustments(plan, adjustments, queue)
        row = adjustment_log_rows(outcomes)[0]
        for banned in ("is_selected", "final_policy_rank", "decision_mechanism", "score"):
            assert banned not in row
