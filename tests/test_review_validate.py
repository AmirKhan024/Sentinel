"""Runtime checks over a completed review run, each driven both ways."""

from __future__ import annotations

import polars as pl

from sentinel.review import validate, writer


def _queue_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "policy_id": "pure_risk",
        "model_name": "lightgbm_platt",
        "fold_set": "quarterly",
        "fold_id": "2026Q1",
        "k_name": "k_1_day",
        "target_inspection_id": "t1",
        "establishment_id": "e1",
        "final_policy_rank": 1,
        "decision_mechanism": "risk_priority",
        "decision_reason": "selected_by_risk_rank",
        "warnings": "limited_history",
        "trigger_reasons": "policy_warning_present",
        "schedule_config_id": "",
        "planning_run_id": "",
        "replan_index": None,
        "scheduled_date": None,
        "review_status": "flagged",
        "review_id": "",
        "resolution_action": "",
        "review_definition_version": "v1",
    }
    base.update(overrides)
    return base


def _recommendation_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "policy_id": "pure_risk",
        "model_name": "lightgbm_platt",
        "fold_set": "quarterly",
        "fold_id": "2026Q1",
        "k_name": "k_1_day",
        "target_inspection_id": "t1",
        "is_selected": True,
        "warnings": "limited_history",
    }
    base.update(overrides)
    return base


def _resolution_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "review_id": "R1",
        "policy_id": "pure_risk",
        "fold_id": "2026Q1",
        "k_name": "k_1_day",
        "target_inspection_id": "t1",
        "resolution_action": "acknowledge",
        "reason_code": "reviewed",
        "actor": "alice",
        "decided_at": "2026-01-06T00:00:00Z",
        "referenced_override_id": "",
        "referenced_adjustment_id": "",
        "escalation_note": "",
        "original_status": "flagged",
        "final_status": "resolved",
        "outcome": "applied",
        "review_definition_version": "v1",
    }
    base.update(overrides)
    return base


def test_every_case_carries_a_trigger_passes_on_a_real_queue() -> None:
    queue = writer.finalize([_queue_row()], "human_review_queue")
    check = validate.every_case_carries_a_trigger(queue)
    assert check.passed


def test_every_case_carries_a_trigger_fails_on_a_blank_column() -> None:
    queue = writer.finalize([_queue_row(trigger_reasons="none")], "human_review_queue")
    check = validate.every_case_carries_a_trigger(queue)
    assert not check.passed


def test_warning_trigger_rows_are_selected_and_warned_passes() -> None:
    queue = writer.finalize([_queue_row()], "human_review_queue")
    recs = pl.DataFrame([_recommendation_row()])
    check = validate.warning_trigger_rows_are_selected_and_warned(queue, recs)
    assert check.passed


def test_warning_trigger_rows_are_selected_and_warned_fails_if_not_selected() -> None:
    queue = writer.finalize([_queue_row()], "human_review_queue")
    recs = pl.DataFrame([_recommendation_row(is_selected=False)])
    check = validate.warning_trigger_rows_are_selected_and_warned(queue, recs)
    assert not check.passed


def test_warning_trigger_check_does_not_confuse_rows_from_a_different_policy() -> None:
    """A cell key mismatch (different policy_id for the same target id) must not mask a real
    defect -- this is the bug the join-on-target-id-only version had."""
    queue = writer.finalize([_queue_row(policy_id="pure_risk")], "human_review_queue")
    recs = pl.DataFrame(
        [
            _recommendation_row(
                policy_id="pure_risk", is_selected=True, warnings="limited_history"
            ),
            _recommendation_row(
                policy_id="coverage_forced_population_share",
                is_selected=False,
                warnings="none",
            ),
        ]
    )
    check = validate.warning_trigger_rows_are_selected_and_warned(queue, recs)
    assert check.passed


def test_no_duplicate_review_id_fails_on_a_duplicate() -> None:
    log = writer.finalize(
        [_resolution_row(review_id="R1"), _resolution_row(review_id="R1")],
        "review_resolution_log",
    )
    check = validate.no_duplicate_review_id(log)
    assert not check.passed


def test_pointer_fields_are_mutually_exclusive_fails_on_a_missing_pointer() -> None:
    log = writer.finalize(
        [_resolution_row(resolution_action="refer_to_override", referenced_override_id="")],
        "review_resolution_log",
    )
    check = validate.pointer_fields_are_mutually_exclusive(log)
    assert not check.passed


def test_pointer_fields_are_mutually_exclusive_passes_with_correct_pointer() -> None:
    log = writer.finalize(
        [_resolution_row(resolution_action="refer_to_override", referenced_override_id="O1")],
        "review_resolution_log",
    )
    check = validate.pointer_fields_are_mutually_exclusive(log)
    assert check.passed


def test_review_status_reflects_one_applied_resolution() -> None:
    queue = writer.finalize([_queue_row(review_status="resolved")], "human_review_queue")
    log = writer.finalize([_resolution_row(outcome="applied")], "review_resolution_log")
    check = validate.review_status_reflects_one_applied_resolution(queue, log)
    assert check.passed


def test_review_status_mismatch_fails() -> None:
    queue = writer.finalize([_queue_row(review_status="flagged")], "human_review_queue")
    log = writer.finalize([_resolution_row(outcome="applied")], "review_resolution_log")
    check = validate.review_status_reflects_one_applied_resolution(queue, log)
    assert not check.passed


def test_resolution_verbs_do_not_collide_passes_on_known_actions() -> None:
    log = writer.finalize([_resolution_row()], "review_resolution_log")
    check = validate.resolution_verbs_do_not_collide(log)
    assert check.passed


def test_inputs_were_not_modified_detects_a_changed_hash() -> None:
    before = {"recommendations": "abc"}
    after = {"recommendations": "xyz"}
    check = validate.inputs_were_not_modified(before, after)
    assert not check.passed


def test_pointer_targets_exist_is_advisory_and_never_fails_a_build() -> None:
    log = writer.finalize(
        [
            _resolution_row(
                resolution_action="refer_to_override", referenced_override_id="O_NOT_YET"
            )
        ],
        "review_resolution_log",
    )
    check = validate.pointer_targets_exist(log, frozenset(), frozenset())
    assert check.severity == "warn"


def test_has_failures_ignores_advisories() -> None:
    from sentinel.review.models import ValidationCheck

    checks = [
        ValidationCheck("a", True, "error", ""),
        ValidationCheck("b", False, "warn", ""),
    ]
    assert not validate.has_failures(checks)


def test_has_failures_true_on_an_error() -> None:
    from sentinel.review.models import ValidationCheck

    checks = [ValidationCheck("a", False, "error", "")]
    assert validate.has_failures(checks)
