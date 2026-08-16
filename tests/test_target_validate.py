"""Target validation checks.

Each error-severity check gets a passing case and a failing case: a check that
cannot fail is not a check. The failing cases construct deliberately malformed
rows, since a correct build cannot produce them.
"""

from __future__ import annotations

from dataclasses import replace

from sentinel.target.models import CodeEraPhase, TargetRow, TargetStatus, ValidationCheck
from sentinel.target.validate import (
    SEVERITY_ERROR,
    format_report,
    has_failures,
    validate_targets,
)

EST = "EST-00000001001"
INSP = "1001"


def row(**overrides: object) -> TargetRow:
    base = TargetRow(
        establishment_id=EST,
        inspection_date="2022-03-14T00:00:00.000",
        target_inspection_id=INSP,
        inspection_type="Canvass",
        results="Fail",
        target=1,
        target_status=TargetStatus.ELIGIBLE,
        has_priority=False,
        has_priority_foundation=True,
        n_priority_entries=0,
        n_priority_foundation_entries=1,
        n_violation_entries=1,
        evidence="PRIORITY FOUNDATION 7-38-010",
        n_contributing_inspections=1,
        contributing_inspection_ids=INSP,
        code_era_phase=CodeEraPhase.STABLE,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def run(rows: list[TargetRow], *, source_rows: int | None = None) -> list[ValidationCheck]:
    return validate_targets(
        rows,
        known_establishment_ids=frozenset({EST}),
        known_inspection_ids=frozenset({INSP, "1002", "1003"}),
        source_row_count=source_rows
        if source_rows is not None
        else sum(r.n_contributing_inspections for r in rows),
    )


def named(checks: list[ValidationCheck], name: str) -> ValidationCheck:
    return next(c for c in checks if c.name == name)


# --- the happy path -------------------------------------------------------


def test_a_correct_build_has_no_failures() -> None:
    assert not has_failures(run([row()]))


def test_an_empty_build_has_no_failures() -> None:
    assert not has_failures(run([], source_rows=0))


# --- structural checks ----------------------------------------------------


def test_unknown_establishment_id_is_detected() -> None:
    checks = run([row(establishment_id="EST-99999999999")])
    assert not named(checks, "establishment_id_from_component_2").passed


def test_unknown_inspection_id_is_detected() -> None:
    checks = run([row(target_inspection_id="9999999")])
    assert not named(checks, "target_inspection_exists").passed


def test_duplicate_establishment_date_is_detected() -> None:
    checks = run([row(), row(target_inspection_id="1002")])
    assert not named(checks, "one_eligible_row_per_establishment_date").passed


def test_unaccounted_inspections_are_detected() -> None:
    """Guards the invariant that every raw row is represented exactly once."""
    checks = run([row()], source_rows=99)
    assert not named(checks, "every_inspection_accounted_for").passed


def test_label_without_eligibility_is_detected() -> None:
    checks = run([row(target=1, target_status=TargetStatus.INELIGIBLE_RESULT)])
    assert not named(checks, "label_present_exactly_when_eligible").passed


def test_eligible_row_without_a_label_is_detected() -> None:
    checks = run([row(target=None)])
    assert not named(checks, "label_present_exactly_when_eligible").passed


def test_non_binary_target_is_detected() -> None:
    checks = run([row(target=7)])
    assert not named(checks, "target_is_binary").passed


# --- eligibility invariants ----------------------------------------------


def test_eligible_row_before_the_code_era_is_detected() -> None:
    """The check that would catch a target built on undefined terminology."""
    checks = run([row(inspection_date="2015-06-01T00:00:00.000")])
    assert not named(checks, "eligible_rows_are_in_the_code_era").passed


def test_eligible_non_canvass_row_is_detected() -> None:
    checks = run([row(inspection_type="Complaint")])
    assert not named(checks, "eligible_rows_are_canvasses").passed


def test_eligible_row_with_a_non_inspection_result_is_detected() -> None:
    checks = run([row(results="Out of Business")])
    assert not named(checks, "eligible_rows_describe_a_real_inspection").passed


def test_positive_without_evidence_is_detected() -> None:
    """A positive label that cannot be traced to text is not auditable."""
    checks = run([row(evidence=None)])
    assert not named(checks, "positives_carry_evidence").passed


def test_blank_evidence_counts_as_missing() -> None:
    checks = run([row(evidence="   ")])
    assert not named(checks, "positives_carry_evidence").passed


def test_label_disagreeing_with_its_priority_flags_is_detected() -> None:
    checks = run([row(target=1, has_priority=False, has_priority_foundation=False)])
    assert not named(checks, "label_matches_priority_flags").passed


def test_negative_claiming_priority_flags_is_detected() -> None:
    checks = run([row(target=0, has_priority_foundation=True, evidence=None)])
    assert not named(checks, "label_matches_priority_flags").passed


def test_mixed_definition_versions_are_detected() -> None:
    checks = run([row(), row(target_inspection_id="1002", target_definition_version="v2")])
    assert not named(checks, "single_target_definition_version").passed


# --- distributional checks ------------------------------------------------


def test_distributional_checks_never_fail_the_build() -> None:
    """The base rate ranges 87%-39% across years, so a check that failed on
    drift would fail on correct data."""
    checks = run([row()])
    warnings = [c for c in checks if c.severity != SEVERITY_ERROR]
    assert warnings
    assert all(c.passed for c in warnings)


def test_positive_rate_is_reported() -> None:
    checks = run(
        [
            row(),
            row(
                target_inspection_id="1002",
                target=0,
                evidence=None,
                has_priority_foundation=False,
                inspection_date="2023-01-01",
            ),
        ]
    )
    assert "1 of 2" in named(checks, "positive_rate").detail


def test_status_breakdown_is_reported() -> None:
    checks = run(
        [
            row(),
            row(
                target_inspection_id="1002",
                target=None,
                target_status=TargetStatus.INELIGIBLE_ERA,
                evidence=None,
                has_priority_foundation=False,
                inspection_date="2012-01-01",
            ),
        ]
    )
    detail = named(checks, "status_breakdown").detail
    assert "eligible=1" in detail
    assert "ineligible_era=1" in detail


def test_drift_is_reported_per_year() -> None:
    checks = run([row()])
    assert any("2022" in o for o in named(checks, "positive_rate_by_year").offenders)


def test_collapsed_days_are_reported() -> None:
    checks = run([row(n_contributing_inspections=2, contributing_inspection_ids="1001 1002")])
    assert "1 establishment-dates" in named(checks, "collapsed_multi_canvass_days").detail


def test_code_era_phase_breakdown_is_reported() -> None:
    checks = run([row(code_era_phase=CodeEraPhase.ADOPTION)])
    assert "adoption=1" in named(checks, "code_era_phase_breakdown").detail


# --- report rendering -----------------------------------------------------


def test_report_marks_failures_notes_and_passes_distinctly() -> None:
    checks = run([row(evidence=None)])
    report = format_report(checks)
    assert "[FAIL] positives_carry_evidence" in report
    assert "[PASS] target_is_binary" in report
    assert "[note] positive_rate" in report
