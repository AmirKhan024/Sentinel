"""The frozen review contracts, and the guard that stops them drifting apart.

``definitions.py`` runs ``_guard_registry()`` at import time, so most of these tests work by
constructing a contradictory registry and asserting the guard would have caught it.
"""

from __future__ import annotations

import pytest

from sentinel.policy.definitions import OverrideAction
from sentinel.review import definitions
from sentinel.review.definitions import (
    POINTER_FIELD_FOR_ACTION,
    REVIEW_REQUIRED_FIELDS,
    ReviewCaseStatus,
    ReviewDefinitionError,
    ReviewResolutionAction,
    ReviewTriggerReason,
)
from sentinel.scheduling.definitions import AdjustmentAction, ScheduleStatus


def test_two_triggers_and_only_two() -> None:
    assert len(list(ReviewTriggerReason)) == 2


def test_no_trigger_token_is_not_separable() -> None:
    for trigger in ReviewTriggerReason:
        assert definitions.TRIGGER_SEPARATOR not in trigger
    assert definitions.TRIGGER_SEPARATOR not in definitions.NO_TRIGGER


def test_every_resolution_action_has_a_pointer_declaration() -> None:
    assert set(POINTER_FIELD_FOR_ACTION) == {a.value for a in ReviewResolutionAction}


def test_required_fields_carry_actor_and_reason_code() -> None:
    assert "actor" in REVIEW_REQUIRED_FIELDS
    assert "reason_code" in REVIEW_REQUIRED_FIELDS


def test_resolution_verbs_do_not_collide_with_override_or_adjustment() -> None:
    review_verbs = {a.value for a in ReviewResolutionAction}
    assert not review_verbs & {a.value for a in OverrideAction}
    assert not review_verbs & {a.value for a in AdjustmentAction}


def test_no_component_16_vocabulary_value_reuses_defer() -> None:
    """Component 14's ScheduleStatus.DEFERRED means something structurally different."""
    for value in (*ReviewCaseStatus, *ReviewTriggerReason, *ReviewResolutionAction):
        assert "defer" not in str(value)
    # Sanity: the reserved word really is in use elsewhere, so this test is not vacuous.
    assert "defer" in str(ScheduleStatus.DEFERRED)


def test_no_probability_threshold_is_offered_anywhere() -> None:
    assert "no numeric score" in definitions.NO_THRESHOLD
    assert not hasattr(definitions, "PROBABILITY_THRESHOLD")
    assert not hasattr(definitions, "CONFIDENCE_THRESHOLD")


# --- the registry guard, driven red ---------------------------------------------


def test_a_pointer_field_action_mismatch_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    trimmed = {
        k: v for k, v in POINTER_FIELD_FOR_ACTION.items() if k != ReviewResolutionAction.ESCALATE
    }
    monkeypatch.setattr(definitions, "POINTER_FIELD_FOR_ACTION", trimmed)
    with pytest.raises(ReviewDefinitionError, match="exactly one entry"):
        definitions._guard_registry()


def test_an_empty_required_fields_list_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(definitions, "REVIEW_REQUIRED_FIELDS", ())
    with pytest.raises(ReviewDefinitionError, match="required-fields list is empty"):
        definitions._guard_registry()


def test_a_required_fields_list_missing_actor_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    trimmed = tuple(f for f in REVIEW_REQUIRED_FIELDS if f != "actor")
    monkeypatch.setattr(definitions, "REVIEW_REQUIRED_FIELDS", trimmed)
    with pytest.raises(ReviewDefinitionError, match="no 'actor' field"):
        definitions._guard_registry()


def test_a_verb_that_collides_with_override_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from enum import StrEnum

    class BrokenActions(StrEnum):
        ACKNOWLEDGE = "acknowledge"
        FORCE_INCLUDE = "force_include"  # collides with OverrideAction
        REFER_TO_OVERRIDE = "refer_to_override"
        REFER_TO_ADJUSTMENT = "refer_to_adjustment"
        ESCALATE = "escalate"

    monkeypatch.setattr(definitions, "ReviewResolutionAction", BrokenActions)
    monkeypatch.setattr(
        definitions,
        "POINTER_FIELD_FOR_ACTION",
        {
            BrokenActions.ACKNOWLEDGE: None,
            BrokenActions.FORCE_INCLUDE: None,
            BrokenActions.REFER_TO_OVERRIDE: "referenced_override_id",
            BrokenActions.REFER_TO_ADJUSTMENT: "referenced_adjustment_id",
            BrokenActions.ESCALATE: None,
        },
    )
    with pytest.raises(ReviewDefinitionError, match="collide with OverrideAction"):
        definitions._guard_registry()


def test_a_value_containing_defer_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from enum import StrEnum

    class BrokenTriggers(StrEnum):
        POLICY_WARNING_PRESENT = "policy_warning_present"
        DEFERRED_FOR_REVIEW = "deferred_for_review"

    monkeypatch.setattr(definitions, "ReviewTriggerReason", BrokenTriggers)
    with pytest.raises(ReviewDefinitionError, match="reuses 'defer'"):
        definitions._guard_registry()


def test_an_empty_boundary_list_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(definitions, "DOES_NOT_ESTABLISH", ())
    with pytest.raises(ReviewDefinitionError, match="is empty"):
        definitions._guard_registry()
