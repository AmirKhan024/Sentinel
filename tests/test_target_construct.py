"""Eligibility rules, labelling and the same-day collapse.

Each eligibility gate has a test, and each exclusion has a test asserting the
row is excluded *with the right reason* rather than merely absent — a silently
missing row and a countable exclusion are very different things when 81% of the
dataset is ineligible.
"""

from __future__ import annotations

import pytest

from sentinel.target.construct import (
    build_target_rows,
    classify_inspection,
    code_era_phase,
    collapse_same_day,
    in_code_era,
    is_canvass,
    normalize_inspection_type,
)
from sentinel.target.models import CodeEraPhase, TargetStatus


def inspection(
    inspection_id: str = "1001",
    *,
    establishment_id: str = "EST-00000001001",
    date: str = "2022-03-14T00:00:00.000",
    inspection_type: str = "Canvass",
    results: str = "Pass",
    violations: str | None = "55. PHYSICAL FACILITIES - Comments: DIRTY FLOOR.",
):  # type: ignore[no-untyped-def]
    return classify_inspection(
        inspection_id=inspection_id,
        establishment_id=establishment_id,
        inspection_date=date,
        inspection_type=inspection_type,
        results=results,
        violations=violations,
    )


PRIORITY_TEXT = "3. MANAGEMENT - Comments: NO POLICY. PRIORITY FOUNDATION 7-38-010."


# --- inspection type normalization ---------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Canvass", "CANVASS"), ("  canvass  ", "CANVASS"), ("CANVASS", "CANVASS"), (None, "")],
)
def test_inspection_type_normalization(raw: str | None, expected: str) -> None:
    assert normalize_inspection_type(raw) == expected


@pytest.mark.parametrize("raw", ["Canvass", "CANVASS", "canvass", " Canvass "])
def test_canvass_variants_are_recognized(raw: str) -> None:
    assert is_canvass(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "Canvass Re-Inspection",
        "CANVASS/SPECIAL EVENT",
        "CANVASS SCHOOL/SPECIAL EVENT",
        "Complaint",
        "License",
        "Short Form Complaint",
        "CANVAS",
        None,
        "",
    ],
)
def test_non_canvass_types_are_rejected(raw: str | None) -> None:
    """Re-inspections in particular: they exist only because something failed."""
    assert not is_canvass(raw)


# --- era boundary ---------------------------------------------------------


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("2018-06-30T00:00:00.000", False),
        ("2018-07-01T00:00:00.000", True),
        ("2018-07-02T00:00:00.000", True),
        ("2010-01-04T00:00:00.000", False),
        ("2026-08-14T00:00:00.000", True),
        ("", False),
    ],
)
def test_code_era_boundary_is_exactly_2018_07_01(date: str, expected: bool) -> None:
    """Findings §5: the cutover is clean, with no overlap month."""
    assert in_code_era(date) is expected


@pytest.mark.parametrize(
    ("date", "phase"),
    [
        ("2015-01-01T00:00:00.000", CodeEraPhase.PRE_CODE),
        ("2018-07-01T00:00:00.000", CodeEraPhase.ADOPTION),
        ("2018-12-31T00:00:00.000", CodeEraPhase.ADOPTION),
        ("2019-01-01T00:00:00.000", CodeEraPhase.STABLE),
        ("2024-05-05T00:00:00.000", CodeEraPhase.STABLE),
    ],
)
def test_code_era_phase(date: str, phase: CodeEraPhase) -> None:
    assert code_era_phase(date) is phase


# --- eligibility gates ----------------------------------------------------


def test_pre_code_era_inspection_is_ineligible() -> None:
    out = inspection(date="2015-06-01T00:00:00.000", violations=PRIORITY_TEXT)
    assert out.status is TargetStatus.INELIGIBLE_ERA
    assert out.label is None


def test_non_canvass_inspection_is_ineligible() -> None:
    out = inspection(inspection_type="Complaint", violations=PRIORITY_TEXT)
    assert out.status is TargetStatus.INELIGIBLE_TYPE
    assert out.label is None


def test_canvass_reinspection_is_ineligible() -> None:
    out = inspection(inspection_type="Canvass Re-Inspection", violations=PRIORITY_TEXT)
    assert out.status is TargetStatus.INELIGIBLE_TYPE


@pytest.mark.parametrize(
    "results", ["Out of Business", "No Entry", "Not Ready", "Business Not Located"]
)
def test_non_inspection_results_are_ineligible_not_negative(results: str) -> None:
    """Findings §9. Labelling these negative would teach that a closed
    establishment is a clean one, which is exactly backwards."""
    out = inspection(results=results, violations=None)
    assert out.status is TargetStatus.INELIGIBLE_RESULT
    assert out.label is None


@pytest.mark.parametrize("results", ["Pass", "Pass w/ Conditions", "Fail"])
def test_inspected_results_are_eligible(results: str) -> None:
    assert inspection(results=results).status is TargetStatus.ELIGIBLE


def test_gates_are_evaluated_in_order_so_the_reason_is_the_first_failure() -> None:
    """A pre-code complaint is reported as an era exclusion, not a type one."""
    out = inspection(date="2012-01-01T00:00:00.000", inspection_type="Complaint")
    assert out.status is TargetStatus.INELIGIBLE_ERA


# --- labelling ------------------------------------------------------------


def test_priority_violation_makes_the_label_positive() -> None:
    out = inspection(violations=PRIORITY_TEXT)
    assert out.status is TargetStatus.ELIGIBLE
    assert out.label is True
    assert out.n_priority_foundation == 1


def test_core_only_violations_make_the_label_negative() -> None:
    out = inspection(violations="55. PHYSICAL FACILITIES - Comments: DIRTY FLOOR.")
    assert out.label is False


def test_pass_with_no_violations_is_a_true_zero() -> None:
    """Findings §10: 5,185 canvasses; a passing inspection with nothing written
    down is the expected encoding of 'no violations found'."""
    out = inspection(results="Pass", violations=None)
    assert out.status is TargetStatus.ELIGIBLE
    assert out.label is False


@pytest.mark.parametrize("results", ["Fail", "Pass w/ Conditions"])
@pytest.mark.parametrize("violations", [None, "", "   "])
def test_failing_result_with_no_violations_is_unknown(results: str, violations: str | None) -> None:
    """Self-contradictory: the result asserts violations, the text records none.
    79 rows in the snapshot; guessed in neither direction."""
    out = inspection(results=results, violations=violations)
    assert out.status is TargetStatus.UNKNOWN_VIOLATIONS
    assert out.label is None


def test_pass_with_a_genuine_priority_violation_is_positive() -> None:
    """The label comes from the violation text, never from `results`."""
    out = inspection(results="Pass", violations=PRIORITY_TEXT)
    assert out.label is True


def test_pass_with_conditions_is_labelled_by_text_not_by_result() -> None:
    """Almost all are positive in practice, but that is an outcome of the rule."""
    positive = inspection(results="Pass w/ Conditions", violations=PRIORITY_TEXT)
    negative = inspection(results="Pass w/ Conditions", violations="55. A - Comments: DIRTY.")
    assert positive.label is True
    assert negative.label is False


def test_evidence_is_captured_only_for_positives() -> None:
    assert inspection(violations=PRIORITY_TEXT).evidence is not None
    assert inspection(violations="55. A - Comments: DIRTY.").evidence is None


# --- same-day collapse ----------------------------------------------------


def test_single_canvass_collapses_to_itself() -> None:
    row = collapse_same_day([inspection("1001", violations=PRIORITY_TEXT)])
    assert row.target == 1
    assert row.n_contributing_inspections == 1
    assert row.contributing_inspection_ids == "1001"


def test_same_day_collapse_is_an_or() -> None:
    """Findings §11: 160 multi-canvass days disagree, so this rule matters."""
    row = collapse_same_day(
        [
            inspection("1001", violations="55. A - Comments: DIRTY."),
            inspection("1002", violations=PRIORITY_TEXT),
        ]
    )
    assert row.target == 1


def test_same_day_collapse_negative_when_none_positive() -> None:
    row = collapse_same_day(
        [
            inspection("1001", violations="55. A - Comments: DIRTY."),
            inspection("1002", violations="47. B - Comments: BROKEN GASKET."),
        ]
    )
    assert row.target == 0
    assert row.n_contributing_inspections == 2


def test_representative_is_the_positive_inspection() -> None:
    """Provenance must point at the evidence, not at an arbitrary member."""
    row = collapse_same_day(
        [
            inspection("1001", violations="55. A - Comments: DIRTY."),
            inspection("1002", violations=PRIORITY_TEXT),
        ]
    )
    assert row.target_inspection_id == "1002"
    assert row.evidence is not None


def test_representative_is_the_lowest_id_when_all_negative() -> None:
    row = collapse_same_day(
        [
            inspection("1002", violations="55. A - Comments: DIRTY."),
            inspection("1001", violations="55. B - Comments: DIRTY."),
        ]
    )
    assert row.target_inspection_id == "1001"


def test_collapse_orders_ids_numerically_not_lexicographically() -> None:
    row = collapse_same_day(
        [
            inspection("900", violations="55. A - Comments: DIRTY."),
            inspection("1000", violations="55. B - Comments: DIRTY."),
        ]
    )
    assert row.target_inspection_id == "900"
    assert row.contributing_inspection_ids == "900 1000"


def test_collapse_is_independent_of_input_order() -> None:
    a = inspection("1001", violations="55. A - Comments: DIRTY.")
    b = inspection("1002", violations=PRIORITY_TEXT)
    assert collapse_same_day([a, b]) == collapse_same_day([b, a])


def test_collapse_sums_evidence_counts_across_the_day() -> None:
    row = collapse_same_day(
        [
            inspection("1001", violations=PRIORITY_TEXT),
            inspection("1002", violations=PRIORITY_TEXT),
        ]
    )
    assert row.n_priority_foundation_entries == 2
    assert row.n_violation_entries == 2


def test_collapsing_nothing_raises() -> None:
    with pytest.raises(ValueError, match="empty set"):
        collapse_same_day([])


# --- row assembly ---------------------------------------------------------


def test_eligible_inspections_group_by_establishment_and_date() -> None:
    rows = build_target_rows(
        [
            inspection("1001", establishment_id="EST-A", violations=PRIORITY_TEXT),
            inspection("1002", establishment_id="EST-A", violations="55. A - Comments: X."),
            inspection("1003", establishment_id="EST-B"),
        ]
    )
    assert len(rows) == 2
    by_est = {r.establishment_id: r for r in rows}
    assert by_est["EST-A"].n_contributing_inspections == 2
    assert by_est["EST-A"].target == 1


def test_different_dates_are_separate_rows() -> None:
    rows = build_target_rows(
        [
            inspection("1001", establishment_id="EST-A", date="2022-01-01T00:00:00.000"),
            inspection("1002", establishment_id="EST-A", date="2023-01-01T00:00:00.000"),
        ]
    )
    assert len(rows) == 2


def test_excluded_inspections_are_emitted_one_to_one() -> None:
    """Every raw inspection stays accounted for; no exclusion is silent."""
    rows = build_target_rows(
        [
            inspection("1001", results="Out of Business", violations=None),
            inspection("1002", inspection_type="Complaint"),
            inspection("1003", date="2012-01-01T00:00:00.000"),
        ]
    )
    assert len(rows) == 3
    assert {r.target_status for r in rows} == {
        TargetStatus.INELIGIBLE_RESULT,
        TargetStatus.INELIGIBLE_TYPE,
        TargetStatus.INELIGIBLE_ERA,
    }
    assert all(r.target is None for r in rows)


def test_excluded_rows_can_be_omitted() -> None:
    rows = build_target_rows(
        [inspection("1001", results="Out of Business", violations=None)],
        include_excluded=False,
    )
    assert rows == []


def test_rows_are_sorted_deterministically() -> None:
    forward = build_target_rows(
        [
            inspection("1002", establishment_id="EST-B"),
            inspection("1001", establishment_id="EST-A"),
        ]
    )
    backward = build_target_rows(
        [
            inspection("1001", establishment_id="EST-A"),
            inspection("1002", establishment_id="EST-B"),
        ]
    )
    assert [r.establishment_id for r in forward] == ["EST-A", "EST-B"]
    assert forward == backward


def test_an_establishment_with_one_inspection_still_produces_a_row() -> None:
    """The unit is a decision point, not a pair, so no row is dropped for lack
    of a successor (findings §13)."""
    rows = build_target_rows([inspection("1001", establishment_id="EST-A")])
    assert len(rows) == 1
    assert rows[0].target == 0


def test_every_row_carries_the_definition_version() -> None:
    rows = build_target_rows([inspection("1001")])
    assert rows[0].target_definition_version == "v1"
