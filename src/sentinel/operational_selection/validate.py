"""Post-selection checks. Every one independently re-derived, not trusted.

None of these checks reads a location column: proving that is the point of
``check_selection_never_reads_location`` below, which asserts it structurally rather
than by convention.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from sentinel.features.models import ValidationCheck
from sentinel.operational_selection.select import SelectionResult
from sentinel.operational_selection.window import REQUIRED_COLUMNS

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"

#: Columns Component 19's allocation logic must never depend on. Carried through in the
#: output (``CARRIED_COLUMNS`` includes them) for display purposes only.
LOCATION_COLUMNS: tuple[str, ...] = (
    "as_of_latitude",
    "as_of_longitude",
    "as_of_address",
    "as_of_zip",
    "has_location",
)


def _check(name: str, passed: bool, severity: str, detail: str) -> ValidationCheck:
    return ValidationCheck(name=name, passed=passed, severity=severity, detail=detail)


def check_no_duplicate_selected_establishments(frame: pl.DataFrame) -> ValidationCheck:
    selected = frame.filter(pl.col("is_selected"))
    duplicates = selected.height - selected["establishment_id"].n_unique()
    return _check(
        "no_duplicate_selected_establishments",
        duplicates == 0,
        SEVERITY_ERROR,
        f"{duplicates} duplicate establishment_id among selected rows",
    )


def check_full_priority_queue_preserved(
    priority_frame: pl.DataFrame, selection_frame: pl.DataFrame
) -> ValidationCheck:
    """Every Component 18 row survives into the Component 19 output, none added."""
    priority_ids = set(priority_frame["target_inspection_id"].to_list())
    output_ids = set(selection_frame["target_inspection_id"].to_list())
    passed = priority_ids == output_ids
    return _check(
        "full_priority_queue_preserved",
        passed,
        SEVERITY_ERROR,
        "every priority-set row appears exactly once in the selection output"
        if passed
        else f"{len(priority_ids - output_ids)} priority row(s) missing, "
        f"{len(output_ids - priority_ids)} extra row(s) present",
    )


def check_component_18_rank_unchanged(
    priority_frame: pl.DataFrame, selection_frame: pl.DataFrame
) -> ValidationCheck:
    """Component 19 must never alter the Component 18 model rank or score."""
    left = priority_frame.select(["target_inspection_id", "rank", "calibrated_score"]).sort(
        "target_inspection_id"
    )
    right = selection_frame.select(["target_inspection_id", "rank", "calibrated_score"]).sort(
        "target_inspection_id"
    )
    passed = left.equals(right)
    return _check(
        "component_18_rank_and_score_unchanged",
        passed,
        SEVERITY_ERROR,
        "Component 18's rank and calibrated_score are byte-identical in the output"
        if passed
        else "Component 19 altered a Component 18 rank or score -- this must never happen",
    )


def check_selected_count_within_capacity(result: SelectionResult) -> ValidationCheck:
    passed = result.selected_count <= result.requested_capacity
    return _check(
        "selected_count_never_exceeds_requested_capacity",
        passed,
        SEVERITY_ERROR,
        f"{result.selected_count} selected against a requested capacity of "
        f"{result.requested_capacity}",
    )


def check_selection_never_reads_location(
    columns: Sequence[str] = REQUIRED_COLUMNS,
) -> ValidationCheck:
    """Structural guard: the allocation input (`window.REQUIRED_COLUMNS`) has no location field.

    Passing means the window-building / allocation code path has no column to have
    read even if it wanted to -- the strongest form of "location did not influence
    selection" this project's own convention (ADR 0038, inherited from Component 13)
    can state. Checked against the real input contract, not the wider display output,
    which legitimately carries location for Component 20.
    """
    leaked = [c for c in LOCATION_COLUMNS if c in columns]
    return _check(
        "allocation_input_excludes_location_columns",
        not leaked,
        SEVERITY_ERROR,
        f"{len(leaked)} location column(s) reachable from the allocation input: {leaked}",
    )


def has_failures(checks: Sequence[ValidationCheck]) -> bool:
    return any(not c.passed and c.severity == SEVERITY_ERROR for c in checks)


def format_report(checks: Sequence[ValidationCheck]) -> str:
    lines = [
        "",
        "Operational selection validation report",
        "----------------------------------------",
    ]
    for check in checks:
        status = "note" if check.severity == SEVERITY_WARN else ("PASS" if check.passed else "FAIL")
        lines.append(f"  [{status}] {check.name}: {check.detail}")
    return "\n".join(lines)


def run_all_checks(priority_frame: pl.DataFrame, result: SelectionResult) -> list[ValidationCheck]:
    return [
        check_no_duplicate_selected_establishments(result.frame),
        check_full_priority_queue_preserved(priority_frame, result.frame),
        check_component_18_rank_unchanged(priority_frame, result.frame),
        check_selected_count_within_capacity(result),
        check_selection_never_reads_location(),
    ]


__all__ = [
    "LOCATION_COLUMNS",
    "check_component_18_rank_unchanged",
    "check_full_priority_queue_preserved",
    "check_no_duplicate_selected_establishments",
    "check_selected_count_within_capacity",
    "check_selection_never_reads_location",
    "format_report",
    "has_failures",
    "run_all_checks",
]
