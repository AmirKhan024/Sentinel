"""Feature validation checks.

Each error check gets a passing case and a failing case. The failing cases inject
a deliberately corrupt feature table, since a correct build cannot produce one —
which is the point: these checks exist to catch a future regression in
`historical.py`, so they must be shown to fire.
"""

from __future__ import annotations

import duckdb
import polars as pl
import pytest

from sentinel.features.validate import (
    SEVERITY_ERROR,
    format_report,
    has_failures,
    validate_features,
)
from sentinel.features.writer import OUTPUT_COLUMNS, finalize

EST = "EST-A"


def _base_feature_row(**overrides: object) -> dict[str, object]:
    """A single well-formed feature row, before casting."""
    row: dict[str, object] = {
        "establishment_id": EST,
        "inspection_date": "2022-06-15",
        "target_inspection_id": "9999",
        "prior_canvass_count": 2,
        "prior_canvass_count_code_era": 2,
        "prior_canvass_inspected_count": 2,
        "days_since_last_canvass": 100,
        "prior_canvass_fail_count": 1,
        "prior_canvass_pass_w_conditions_count": 0,
        "prior_canvass_fail_rate": 0.5,
        "fail_at_last_canvass": True,
        "prior_canvass_priority_count": 1,
        "prior_canvass_priority_foundation_count": 1,
        "prior_canvass_priority_rate": 0.5,
        "priority_at_last_canvass": True,
        "prior_inspection_count_any_type": 3,
        "days_since_any_inspection": 50,
        "prior_complaint_count": 1,
        "prior_reinspection_count": 0,
        "prior_license_inspection_count": 0,
        "name_changed_since_last_canvass": False,
        "prior_canvass_count_current_name": 2,
        "days_since_first_inspection": 900,
        "canvasses_last_365d": 1,
        "canvass_priority_events_last_365d": 1,
        "canvasses_last_730d": 2,
        "canvass_priority_events_last_730d": 1,
        "canvasses_last_1095d": 2,
        "canvass_priority_events_last_1095d": 1,
        "target": 1,
        "target_status": "eligible",
        "code_era_phase": "stable",
    }
    row.update(overrides)
    return row


def run(rows: list[dict[str, object]], *, eligible: int | None = None):  # type: ignore[no-untyped-def]
    """Register a synthetic feature table and validate it."""
    frame = finalize(pl.DataFrame(rows))
    targets = pl.DataFrame(
        {
            "target_inspection_id": [str(r["target_inspection_id"]) for r in rows],
            "establishment_id": [str(r["establishment_id"]) for r in rows],
            "inspection_date": [str(r["inspection_date"]) for r in rows],
        }
    )
    assignments = pl.DataFrame(
        {"establishment_id": [EST], "inspection_id": ["1"]},
        schema={"establishment_id": pl.Utf8, "inspection_id": pl.Utf8},
    )
    history = pl.DataFrame(
        {"establishment_id": [EST], "inspection_date": [pl.Series([None], dtype=pl.Date)[0]]},
        schema={"establishment_id": pl.Utf8, "inspection_date": pl.Date},
    )

    conn = duckdb.connect(database=":memory:")
    conn.register("features_src", frame)
    conn.register("targets_src", targets)
    conn.register("assignments", assignments)
    conn.register("history_src", history)
    conn.execute("CREATE TABLE features AS SELECT * FROM features_src")
    conn.execute(
        "CREATE VIEW targets AS SELECT target_inspection_id, establishment_id, "
        "TRY_CAST(inspection_date AS DATE) AS inspection_date FROM targets_src"
    )
    conn.execute("CREATE VIEW history AS SELECT * FROM history_src WHERE FALSE")
    try:
        return validate_features(
            conn,
            columns=list(frame.columns),
            eligible_target_rows=eligible if eligible is not None else len(rows),
        )
    finally:
        conn.close()


def named(checks, name):  # type: ignore[no-untyped-def]
    return next(c for c in checks if c.name == name)


# --- the happy path -------------------------------------------------------


def test_a_correct_table_has_no_failures() -> None:
    assert not has_failures(run([_base_feature_row()]))


# --- structure ------------------------------------------------------------


def test_row_count_mismatch_is_detected() -> None:
    checks = run([_base_feature_row()], eligible=99)
    assert not named(checks, "one_feature_row_per_eligible_target").passed


def test_duplicate_primary_key_is_detected() -> None:
    checks = run([_base_feature_row(), _base_feature_row()])
    assert not named(checks, "target_inspection_id_unique").passed


def test_unknown_establishment_is_detected() -> None:
    checks = run([_base_feature_row(establishment_id="EST-UNKNOWN")])
    assert not named(checks, "establishment_id_from_component_2").passed


# --- value ranges ---------------------------------------------------------


def test_negative_count_is_detected() -> None:
    checks = run([_base_feature_row(prior_canvass_count=-1)])
    assert not named(checks, "no_negative_counts_or_recency").passed


def test_negative_recency_is_detected() -> None:
    checks = run([_base_feature_row(days_since_last_canvass=-5)])
    assert not named(checks, "no_negative_counts_or_recency").passed


def test_rate_above_one_is_detected() -> None:
    checks = run([_base_feature_row(prior_canvass_fail_rate=1.5)])
    assert not named(checks, "rates_within_zero_and_one").passed


# --- the missing-value rules ---------------------------------------------


def test_recency_zero_instead_of_null_is_detected() -> None:
    """The exact bug the rule exists to prevent: 0 read as 'today'."""
    checks = run(
        [
            _base_feature_row(
                prior_canvass_count=0,
                prior_canvass_count_code_era=0,
                prior_canvass_inspected_count=0,
                days_since_last_canvass=0,
                prior_canvass_fail_rate=None,
                fail_at_last_canvass=None,
                prior_canvass_priority_count=None,
                prior_canvass_priority_foundation_count=None,
                prior_canvass_priority_rate=None,
                priority_at_last_canvass=None,
                name_changed_since_last_canvass=None,
                canvasses_last_365d=0,
                canvass_priority_events_last_365d=0,
                canvasses_last_730d=0,
                canvass_priority_events_last_730d=0,
                canvasses_last_1095d=0,
                canvass_priority_events_last_1095d=0,
            )
        ]
    )
    assert not named(checks, "null_rules_hold_exactly").passed


def test_priority_zero_instead_of_null_is_detected() -> None:
    """Absence of evidence rendered as evidence of absence."""
    checks = run(
        [
            _base_feature_row(
                prior_canvass_count_code_era=0,
                prior_canvass_priority_count=0,
                prior_canvass_priority_foundation_count=0,
                prior_canvass_priority_rate=0.0,
                priority_at_last_canvass=False,
                canvass_priority_events_last_365d=0,
                canvass_priority_events_last_730d=0,
                canvass_priority_events_last_1095d=0,
            )
        ]
    )
    assert not named(checks, "null_rules_hold_exactly").passed


def test_a_count_that_is_null_is_detected() -> None:
    checks = run([_base_feature_row(prior_complaint_count=None)])
    assert not named(checks, "null_rules_hold_exactly").passed


# --- windows --------------------------------------------------------------


def test_window_exceeding_the_unbounded_count_is_detected() -> None:
    checks = run([_base_feature_row(canvasses_last_365d=99)])
    assert not named(checks, "windows_are_nested_and_bounded").passed


def test_unnested_windows_are_detected() -> None:
    checks = run([_base_feature_row(canvasses_last_730d=1, canvasses_last_365d=2)])
    assert not named(checks, "windows_are_nested_and_bounded").passed


def test_priority_events_exceeding_canvasses_is_detected() -> None:
    checks = run([_base_feature_row(canvass_priority_events_last_1095d=99)])
    assert not named(checks, "windows_are_nested_and_bounded").passed


# --- schema separation ----------------------------------------------------


def test_all_declared_columns_present_passes_on_a_full_table() -> None:
    checks = run([_base_feature_row()])
    assert named(checks, "all_declared_columns_present").passed
    assert named(checks, "every_column_is_declared").passed
    assert named(checks, "features_and_labels_are_disjoint").passed
    assert named(checks, "no_outcome_columns_present").passed


def test_output_columns_cover_every_group() -> None:
    checks = run([_base_feature_row()])
    assert named(checks, "all_declared_columns_present").detail.startswith("0 ")
    assert set(OUTPUT_COLUMNS) <= set(_base_feature_row()) | {"feature_definition_version"}


# --- distributional -------------------------------------------------------


def test_distributional_checks_never_fail() -> None:
    checks = run([_base_feature_row()])
    warnings = [c for c in checks if c.severity != SEVERITY_ERROR]
    assert warnings
    assert all(c.passed for c in warnings)


def test_family_summaries_are_reported() -> None:
    checks = run([_base_feature_row()])
    assert "8 features" in named(checks, "family_canvass_history").detail
    assert "6 features" in named(checks, "family_window").detail


def test_cold_start_rows_are_reported() -> None:
    checks = run([_base_feature_row()])
    assert "0 rows have no history at all" in named(checks, "cold_start_rows").detail


# --- report rendering -----------------------------------------------------


def test_report_marks_failures_notes_and_passes_distinctly() -> None:
    checks = run([_base_feature_row(prior_canvass_count=-1)])
    report = format_report(checks)
    assert "[FAIL] no_negative_counts_or_recency" in report
    assert "[PASS] target_inspection_id_unique" in report
    assert "[note] family_window" in report


@pytest.mark.parametrize(
    "check_name",
    [
        "one_feature_row_per_eligible_target",
        "target_inspection_id_unique",
        "temporal_boundary_holds",
        "null_rules_hold_exactly",
        "windows_are_nested_and_bounded",
        "no_outcome_columns_present",
    ],
)
def test_key_checks_exist(check_name: str) -> None:
    """Guards against a check being silently deleted."""
    checks = run([_base_feature_row()])
    assert named(checks, check_name) is not None
