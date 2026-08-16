"""Regression tests built from real cases in the Chicago snapshot.

Unit tests prove the rules do what they say; these prove the rules do the right
thing to inspections that actually happened. Several encode decisions that a
simpler target definition would have got wrong.
"""

from __future__ import annotations

import pytest

from sentinel.target.construct import classify_inspection
from tests.fixtures.target_cases import TARGET_CASES, TargetCase


def _classify(case: TargetCase):  # type: ignore[no-untyped-def]
    return classify_inspection(
        inspection_id=case.inspection_id,
        establishment_id="EST-00000000001",
        inspection_date=case.inspection_date,
        inspection_type=case.inspection_type,
        results=case.results,
        violations=case.violations,
    )


@pytest.mark.parametrize("case", TARGET_CASES, ids=lambda c: c.case_id)
def test_real_case_gets_the_expected_status(case: TargetCase) -> None:
    outcome = _classify(case)
    assert outcome.status.value == case.expected_status, (
        f"{case.case_id}: expected status {case.expected_status!r}, "
        f"got {outcome.status.value!r}.\n{case.why}"
    )


@pytest.mark.parametrize("case", TARGET_CASES, ids=lambda c: c.case_id)
def test_real_case_gets_the_expected_label(case: TargetCase) -> None:
    outcome = _classify(case)
    actual = None if outcome.label is None else int(outcome.label)
    assert actual == case.expected_target, (
        f"{case.case_id}: expected target {case.expected_target}, got {actual}.\n{case.why}"
    )


@pytest.mark.parametrize(
    "case", [c for c in TARGET_CASES if c.expected_target == 1], ids=lambda c: c.case_id
)
def test_positive_cases_carry_traceable_evidence(case: TargetCase) -> None:
    """A positive label that cannot be traced back to words is not auditable."""
    outcome = _classify(case)
    assert outcome.evidence, f"{case.case_id} produced no evidence span"
    assert "PRIORITY" in outcome.evidence.upper()


def test_cases_cover_every_target_status() -> None:
    statuses = {c.expected_status for c in TARGET_CASES}
    assert statuses == {
        "eligible",
        "ineligible_era",
        "ineligible_type",
        "ineligible_result",
        "unknown_violations",
    }


def test_cases_cover_both_labels() -> None:
    labels = {c.expected_target for c in TARGET_CASES}
    assert {0, 1, None} <= labels


def test_every_case_carries_an_explanation() -> None:
    for case in TARGET_CASES:
        assert len(case.why) > 60, f"{case.case_id} needs a real explanation"


def test_narrative_exclusion_case_is_present() -> None:
    """The specific false positive naive substring matching produces."""
    case = next(
        c
        for c in TARGET_CASES
        if c.case_id == "grace_period_boilerplate_is_not_a_priority_violation"
    )
    assert case.expected_target == 0
    assert "narrative_exclusion" in case.tags


def test_era_boundary_case_is_on_the_boundary_itself() -> None:
    case = next(c for c in TARGET_CASES if c.case_id == "era_boundary_is_inclusive_of_2018_07_01")
    assert case.inspection_date.startswith("2018-07-01")
    assert case.expected_status == "eligible"
