"""Warnings and the human override layer.

Two responsibilities seen from either side: what the system tells a reviewer, and what a
reviewer may tell the system back. The tests that matter most here are the ones asserting what
an override may *not* do -- silently edit the recommendation, arrive without an actor, or
quietly raise capacity.
"""

from __future__ import annotations

import pytest

from sentinel.policy.allocation import allocate, decide
from sentinel.policy.definitions import (
    NO_WARNING,
    WARNING_SEPARATOR,
    PolicySpec,
    PolicyWarning,
    ReserveMechanism,
)
from sentinel.policy.governance import (
    OUTCOME_APPLIED,
    OUTCOME_NO_OP_ALREADY_SELECTED,
    OUTCOME_NO_OP_NOT_SELECTED,
    OUTCOME_ROW_NOT_IN_WINDOW,
    GovernanceError,
    apply_overrides,
    parse_overrides,
    warnings_for,
)
from sentinel.policy.models import Override
from tests.conftest import make_override, make_policy_window

PURE = PolicySpec("pure_risk", ReserveMechanism.NONE, 0.0, "baseline")


# --- 1. warnings ------------------------------------------------------------------


def test_a_row_with_nothing_to_flag_carries_the_token_not_an_empty_string() -> None:
    """An empty cell is ambiguous between 'no warning' and 'warnings were not computed'."""
    assert (
        warnings_for(
            eligible=False, secondary_no_history=False, group_value="12", group_status="supported"
        )
        == NO_WARNING
    )


def test_limited_history_is_raised_for_a_coverage_eligible_row() -> None:
    assert (
        warnings_for(
            eligible=True, secondary_no_history=False, group_value="12", group_status="supported"
        )
        == PolicyWarning.LIMITED_HISTORY
    )


def test_several_warnings_are_a_sorted_set_not_a_precedence() -> None:
    """Choosing one to display would choose which fact a reviewer is allowed to see."""
    value = warnings_for(
        eligible=True,
        secondary_no_history=True,
        group_value="__UNKNOWN__",
        group_status="insufficient_support",
    )
    codes = value.split(WARNING_SEPARATOR)
    assert codes == sorted(codes)
    assert set(codes) == {
        PolicyWarning.LIMITED_HISTORY,
        PolicyWarning.NO_PRIOR_INSPECTION,
        PolicyWarning.UNKNOWN_GEOGRAPHY,
        PolicyWarning.INSUFFICIENT_GROUP_AUDIT_SUPPORT,
    }


def test_an_absent_group_frame_raises_no_geography_warning() -> None:
    """Without Component 8's categoricals the queue is identical and the columns go blank."""
    assert (
        warnings_for(
            eligible=False, secondary_no_history=False, group_value=None, group_status=None
        )
        == NO_WARNING
    )


def test_an_unmeasured_neighbourhood_is_flagged_as_unmeasured_not_as_bad() -> None:
    """Component 12 could not say how the model behaves there. That is a different claim."""
    value = warnings_for(
        eligible=False,
        secondary_no_history=False,
        group_value="7",
        group_status="insufficient_support",
    )
    assert value == PolicyWarning.INSUFFICIENT_GROUP_AUDIT_SUPPORT


# --- 2. parsing the override contract ------------------------------------------------


def test_a_well_formed_override_file_parses() -> None:
    overrides = parse_overrides([make_override(1), make_override(2)])
    assert [o.override_id for o in overrides] == ["OV-0001", "OV-0002"]


@pytest.mark.parametrize("field", ["actor", "reason_code", "decided_at", "override_id"])
def test_a_blank_required_field_refuses_the_whole_file(field: str) -> None:
    """An override with no actor is an anonymous change to who gets inspected."""
    with pytest.raises(GovernanceError, match="blank"):
        parse_overrides([make_override(1, **{field: "  "})])


def test_a_missing_field_refuses_the_whole_file() -> None:
    row = make_override(1)
    del row["actor"]
    with pytest.raises(GovernanceError):
        parse_overrides([row])


def test_an_unknown_action_names_the_ones_that_exist() -> None:
    with pytest.raises(GovernanceError, match="force_include"):
        parse_overrides([make_override(1, action="delete_establishment")])


def test_a_duplicate_override_id_refuses_the_whole_file() -> None:
    """The id is how a decision is referred to afterwards, so two cannot share one."""
    with pytest.raises(GovernanceError, match="duplicate override_id"):
        parse_overrides([make_override(1), make_override(1)])


def test_an_unknown_column_is_refused_rather_than_ignored() -> None:
    """A typo'd field name would otherwise be silently dropped and the reviewer never told."""
    with pytest.raises(GovernanceError):
        parse_overrides([make_override(1, reasoncode="typo")])


def test_one_malformed_row_refuses_the_file_rather_than_being_skipped() -> None:
    """A partially applied override file produces a queue nobody authorised."""
    with pytest.raises(GovernanceError):
        parse_overrides([make_override(1), make_override(2, actor="")])


# --- 3. applying overrides --------------------------------------------------------


def _cell() -> tuple[object, object, tuple[str, ...], tuple[str, ...], tuple[int | None, ...]]:
    window = make_policy_window(
        scores=[0.9, 0.8, 0.7, 0.6, 0.5], ids=["T1", "T2", "T3", "T4", "T5"]
    )
    allocation = allocate(window, PURE, k_name="k_1_day", k=3)
    mechanisms, reasons, ranks = decide(window, allocation)
    return window, allocation, mechanisms, reasons, ranks


def _apply(overrides: list[Override]) -> tuple[list[dict[str, object]], dict[str, bool]]:
    window, allocation, mechanisms, reasons, ranks = _cell()
    return apply_overrides(
        window,
        allocation,
        overrides,
        mechanisms=mechanisms,
        reasons=reasons,
        ranks=ranks,
        definition_version="v1",
    )


def test_forcing_a_row_in_displaces_the_lowest_ranked_risk_selection() -> None:
    """Capacity is fixed. An inclusion costs an exclusion, and the displaced row is named."""
    log, final = _apply(parse_overrides([make_override(1, target_inspection_id="T5")]))
    assert log[0]["outcome"] == OUTCOME_APPLIED
    assert log[0]["displaced_target_inspection_id"] == "T3"
    assert final["T5"] is True
    assert final["T3"] is False
    assert sum(final.values()) == 3


def test_forcing_a_row_out_does_not_backfill_the_freed_slot() -> None:
    """Backfilling would be the policy making a second decision on the back of a human one."""
    log, final = _apply(
        parse_overrides([make_override(1, action="force_exclude", target_inspection_id="T1")])
    )
    assert log[0]["outcome"] == OUTCOME_APPLIED
    assert final["T1"] is False
    assert final["T4"] is False
    assert sum(final.values()) == 2


def test_including_a_row_that_is_already_selected_is_a_recorded_no_op() -> None:
    """'The override did nothing' and 'the override was applied' must stay distinguishable."""
    log, final = _apply(parse_overrides([make_override(1, target_inspection_id="T1")]))
    assert log[0]["outcome"] == OUTCOME_NO_OP_ALREADY_SELECTED
    assert final["T1"] is True
    assert sum(final.values()) == 3


def test_excluding_a_row_that_was_not_selected_is_a_recorded_no_op() -> None:
    log, _final = _apply(
        parse_overrides([make_override(1, action="force_exclude", target_inspection_id="T5")])
    )
    assert log[0]["outcome"] == OUTCOME_NO_OP_NOT_SELECTED


def test_an_override_for_a_row_outside_the_window_is_logged_not_silently_dropped() -> None:
    """A reviewer who typed the wrong id must find out, and the audit must show they tried."""
    log, final = _apply(parse_overrides([make_override(1, target_inspection_id="T999")]))
    assert log[0]["outcome"] == OUTCOME_ROW_NOT_IN_WINDOW
    assert sum(final.values()) == 3


def test_overrides_addressed_to_another_cell_are_left_alone() -> None:
    log, final = _apply(
        parse_overrides([make_override(1, fold_id="quarterly-2099Q9", target_inspection_id="T5")])
    )
    assert log == []
    assert final["T5"] is False


def test_the_original_recommendation_is_recorded_beside_the_final_decision() -> None:
    """An audit asks what would have happened as well as what did."""
    log, _final = _apply(parse_overrides([make_override(1, target_inspection_id="T5")]))
    row = log[0]
    assert row["original_is_selected"] is False
    assert row["final_is_selected"] is True
    assert row["original_mechanism"] == "not_selected"
    assert row["actor"] == "inspector.smith"
    assert row["reason_code"] == "outbreak_investigation"
    assert row["decided_at"] == "2026-08-26T09:00:00Z"


def test_the_apply_order_is_the_id_not_the_file_order() -> None:
    """Re-serialising the file must not change the queue.

    Two interacting decisions -- one including, one excluding -- have to resolve the same way
    whichever order the file happened to list them in.
    """
    forward = parse_overrides(
        [
            make_override(1, target_inspection_id="T4"),
            make_override(2, action="force_exclude", target_inspection_id="T4"),
        ]
    )
    backward = parse_overrides(
        [
            make_override(2, action="force_exclude", target_inspection_id="T4"),
            make_override(1, target_inspection_id="T4"),
        ]
    )
    _log_a, final_a = _apply(forward)
    _log_b, final_b = _apply(backward)
    assert final_a == final_b


def test_an_inclusion_with_nothing_left_to_displace_is_refused_rather_than_raising_capacity() -> (
    None
):
    """Adding capacity is the one thing this project's simulation has never been willing to do."""
    window = make_policy_window(scores=[0.9, 0.8], ids=["T1", "T2"])
    allocation = allocate(window, PURE, k_name="k_1_day", k=1)
    mechanisms, reasons, ranks = decide(window, allocation)
    overrides = parse_overrides(
        [
            make_override(1, action="force_exclude", target_inspection_id="T1"),
            make_override(2, target_inspection_id="T2"),
        ]
    )
    with pytest.raises(GovernanceError, match="nothing left to displace"):
        apply_overrides(
            window,
            allocation,
            overrides,
            mechanisms=mechanisms,
            reasons=reasons,
            ranks=ranks,
            definition_version="v1",
        )
