"""Feature values against hand-computed expectations.

Where `test_features_leakage.py` proves features cannot cheat, this file proves
they are *right*: the counts count the right things, the denominators exclude the
right things, the windows are half-open, and every missing-value rule produces
NULL exactly where it should and 0 exactly where it should.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.features.build import build_features
from tests.conftest import assignment_frame, make_inspection_record, target_scenario

PRIORITY_FOUNDATION = "3. MANAGEMENT - Comments: NO POLICY. PRIORITY FOUNDATION 7-38-010."
PRIORITY_ONLY = "22. COLD HOLDING - Comments: TEMP 49F. PRIORITY 7-38-005, CITATION ISSUED"
CORE = "55. PHYSICAL FACILITIES - Comments: DIRTY FLOOR."

EST = "EST-A"
REFERENCE = "2022-06-15T00:00:00.000"


def build(
    settings: Settings,
    tmp_path: Path,
    history: list[dict[str, object]],
    *,
    reference_date: str = REFERENCE,
    reference_violations: str | None = CORE,
    reference_name: str | None = None,
) -> dict[str, object]:
    """Build features for one reference row preceded by the given history."""
    reference = make_inspection_record(
        99,
        inspection_id="9999",
        inspection_date=reference_date,
        inspection_type="Canvass",
        results="Pass",
        violations=reference_violations,
        **({"dba_name": reference_name} if reference_name else {}),
    )
    rows = [*history, reference]
    pairs = [(str(r["inspection_id"]), EST) for r in rows]

    raw = tmp_path / "raw.parquet"
    asg = tmp_path / "asg.parquet"
    tgt = tmp_path / "tgt.parquet"
    target_scenario(rows).write_parquet(raw)
    assignment_frame(pairs).write_parquet(asg)
    pl.DataFrame(
        {
            "establishment_id": [EST],
            "inspection_date": [reference_date],
            "target_inspection_id": ["9999"],
            "target": [0],
            "target_status": ["eligible"],
            "code_era_phase": ["stable"],
        },
        schema={
            "establishment_id": pl.Utf8,
            "inspection_date": pl.Utf8,
            "target_inspection_id": pl.Utf8,
            "target": pl.Int8,
            "target_status": pl.Utf8,
            "code_era_phase": pl.Utf8,
        },
    ).write_parquet(tgt)

    frame = build_features(
        settings, parquet_path=raw, assignments_path=asg, targets_path=tgt, dry_run=True
    ).features
    assert frame.height == 1
    return frame.row(0, named=True)


def canvass(
    index: int,
    date: str,
    *,
    results: str = "Pass",
    violations: str | None = CORE,
    name: str | None = None,
) -> dict[str, object]:
    extra = {"dba_name": name} if name else {}
    return make_inspection_record(
        index,
        inspection_id=str(1000 + index),
        inspection_date=f"{date}T00:00:00.000",
        inspection_type="Canvass",
        results=results,
        violations=violations,
        **extra,
    )


def other(index: int, date: str, inspection_type: str) -> dict[str, object]:
    return make_inspection_record(
        index,
        inspection_id=str(1000 + index),
        inspection_date=f"{date}T00:00:00.000",
        inspection_type=inspection_type,
        results="Pass",
        violations=CORE,
    )


# --- counts ---------------------------------------------------------------


def test_prior_canvass_count(settings: Settings, tmp_path: Path) -> None:
    row = build(settings, tmp_path, [canvass(1, "2019-01-01"), canvass(2, "2020-01-01")])
    assert row["prior_canvass_count"] == 2


def test_prior_canvass_count_excludes_other_types(settings: Settings, tmp_path: Path) -> None:
    row = build(
        settings,
        tmp_path,
        [
            canvass(1, "2019-01-01"),
            other(2, "2019-06-01", "Complaint"),
            other(3, "2019-07-01", "Canvass Re-Inspection"),
        ],
    )
    assert row["prior_canvass_count"] == 1
    assert row["prior_inspection_count_any_type"] == 3


def test_no_history_gives_zero_counts_not_null(settings: Settings, tmp_path: Path) -> None:
    """Rule 1: a count of 0 is a true observation."""
    row = build(settings, tmp_path, [])
    assert row["prior_canvass_count"] == 0
    assert row["prior_inspection_count_any_type"] == 0
    assert row["prior_complaint_count"] == 0


def test_context_counts_by_type(settings: Settings, tmp_path: Path) -> None:
    row = build(
        settings,
        tmp_path,
        [
            other(1, "2019-01-01", "Complaint"),
            other(2, "2019-02-01", "Short Form Complaint"),
            other(3, "2019-03-01", "Canvass Re-Inspection"),
            other(4, "2019-04-01", "License"),
            other(5, "2019-05-01", "License Re-Inspection"),
        ],
    )
    assert row["prior_complaint_count"] == 2
    assert row["prior_reinspection_count"] == 2  # canvass + licence re-inspections
    assert row["prior_license_inspection_count"] == 2


# --- the inspected-only denominator ---------------------------------------


def test_non_inspection_canvasses_are_excluded_from_the_denominator(
    settings: Settings, tmp_path: Path
) -> None:
    """Findings §6: 16,517 prior canvasses are Out of Business; a locked door is
    not a clean result."""
    row = build(
        settings,
        tmp_path,
        [
            canvass(1, "2019-01-01", results="Pass"),
            canvass(2, "2019-06-01", results="Out of Business", violations=None),
            canvass(3, "2020-01-01", results="No Entry", violations=None),
            canvass(4, "2020-06-01", results="Fail"),
        ],
    )
    assert row["prior_canvass_count"] == 4
    assert row["prior_canvass_inspected_count"] == 2
    assert row["prior_canvass_fail_count"] == 1
    assert row["prior_canvass_fail_rate"] == pytest.approx(0.5)


def test_fail_rate_is_null_when_nothing_was_inspected(settings: Settings, tmp_path: Path) -> None:
    """Rule 3: 0/0 is not 0."""
    row = build(
        settings,
        tmp_path,
        [canvass(1, "2019-01-01", results="Out of Business", violations=None)],
    )
    assert row["prior_canvass_count"] == 1
    assert row["prior_canvass_inspected_count"] == 0
    assert row["prior_canvass_fail_rate"] is None


def test_pass_with_conditions_is_counted_separately(settings: Settings, tmp_path: Path) -> None:
    row = build(
        settings,
        tmp_path,
        [
            canvass(1, "2019-01-01", results="Pass w/ Conditions"),
            canvass(2, "2020-01-01", results="Fail"),
        ],
    )
    assert row["prior_canvass_pass_w_conditions_count"] == 1
    assert row["prior_canvass_fail_count"] == 1
    assert row["prior_canvass_inspected_count"] == 2


# --- recency --------------------------------------------------------------


def test_days_since_last_canvass(settings: Settings, tmp_path: Path) -> None:
    row = build(settings, tmp_path, [canvass(1, "2022-06-05")])
    assert row["days_since_last_canvass"] == 10


def test_days_since_last_canvass_uses_the_most_recent(settings: Settings, tmp_path: Path) -> None:
    row = build(settings, tmp_path, [canvass(1, "2019-01-01"), canvass(2, "2022-06-05")])
    assert row["days_since_last_canvass"] == 10


def test_recency_is_null_without_history_not_zero(settings: Settings, tmp_path: Path) -> None:
    """Rule 2: the trap the component brief calls out explicitly."""
    row = build(settings, tmp_path, [])
    assert row["days_since_last_canvass"] is None
    assert row["days_since_any_inspection"] is None
    assert row["days_since_first_inspection"] is None


def test_canvass_recency_is_null_when_only_other_types_exist(
    settings: Settings, tmp_path: Path
) -> None:
    row = build(settings, tmp_path, [other(1, "2022-06-01", "Complaint")])
    assert row["days_since_last_canvass"] is None
    assert row["days_since_any_inspection"] == 14


def test_days_since_first_inspection_uses_the_earliest(settings: Settings, tmp_path: Path) -> None:
    row = build(settings, tmp_path, [canvass(1, "2020-06-15"), canvass(2, "2021-06-15")])
    assert row["days_since_first_inspection"] == 730


# --- at-last flags --------------------------------------------------------


def test_fail_at_last_canvass(settings: Settings, tmp_path: Path) -> None:
    row = build(
        settings,
        tmp_path,
        [canvass(1, "2019-01-01", results="Fail"), canvass(2, "2020-01-01", results="Pass")],
    )
    assert row["fail_at_last_canvass"] is False

    row = build(
        settings,
        tmp_path,
        [canvass(1, "2019-01-01", results="Pass"), canvass(2, "2020-01-01", results="Fail")],
    )
    assert row["fail_at_last_canvass"] is True


def test_at_last_flags_are_null_without_history(settings: Settings, tmp_path: Path) -> None:
    """Rule 4: no last inspection is not a clean last inspection."""
    row = build(settings, tmp_path, [])
    assert row["fail_at_last_canvass"] is None
    assert row["priority_at_last_canvass"] is None


# --- priority history -----------------------------------------------------


def test_priority_counts(settings: Settings, tmp_path: Path) -> None:
    row = build(
        settings,
        tmp_path,
        [
            canvass(1, "2019-01-01", violations=PRIORITY_FOUNDATION, results="Fail"),
            canvass(2, "2020-01-01", violations=PRIORITY_ONLY, results="Fail"),
            canvass(3, "2021-01-01", violations=CORE),
        ],
    )
    assert row["prior_canvass_count_code_era"] == 3
    assert row["prior_canvass_priority_count"] == 2
    assert row["prior_canvass_priority_foundation_count"] == 1
    assert row["prior_canvass_priority_rate"] == pytest.approx(2 / 3)


def test_priority_features_are_null_without_code_era_history(
    settings: Settings, tmp_path: Path
) -> None:
    """The sharpest zero-versus-null case: 24.5% of real rows land here."""
    row = build(settings, tmp_path, [canvass(1, "2015-01-01", results="Fail")])
    assert row["prior_canvass_count"] == 1
    assert row["prior_canvass_count_code_era"] == 0
    assert row["prior_canvass_priority_count"] is None
    assert row["prior_canvass_priority_rate"] is None
    assert row["priority_at_last_canvass"] is None


def test_priority_features_are_zero_when_code_era_history_is_clean(
    settings: Settings, tmp_path: Path
) -> None:
    """Evidence of absence, as distinct from absence of evidence."""
    row = build(settings, tmp_path, [canvass(1, "2019-01-01", violations=CORE)])
    assert row["prior_canvass_count_code_era"] == 1
    assert row["prior_canvass_priority_count"] == 0
    assert row["prior_canvass_priority_rate"] == pytest.approx(0.0)
    assert row["priority_at_last_canvass"] is False


def test_pre_code_priority_text_is_ignored(settings: Settings, tmp_path: Path) -> None:
    """Priority did not exist before 2018-07-01, so a pre-code row cannot supply it."""
    row = build(
        settings,
        tmp_path,
        [
            canvass(1, "2016-01-01", violations=PRIORITY_FOUNDATION, results="Fail"),
            canvass(2, "2019-01-01", violations=CORE),
        ],
    )
    assert row["prior_canvass_count"] == 2
    assert row["prior_canvass_count_code_era"] == 1
    assert row["prior_canvass_priority_count"] == 0


def test_code_era_boundary_is_inclusive_of_2018_07_01(settings: Settings, tmp_path: Path) -> None:
    row = build(
        settings,
        tmp_path,
        [canvass(1, "2018-07-01", violations=PRIORITY_FOUNDATION, results="Fail")],
    )
    assert row["prior_canvass_count_code_era"] == 1
    assert row["prior_canvass_priority_count"] == 1


def test_pre_code_history_still_counts_for_generic_features(
    settings: Settings, tmp_path: Path
) -> None:
    """The era boundary constrains classification, not counting."""
    row = build(settings, tmp_path, [canvass(1, "2014-01-01", results="Fail")])
    assert row["prior_canvass_count"] == 1
    assert row["prior_canvass_fail_count"] == 1
    assert row["prior_canvass_fail_rate"] == pytest.approx(1.0)


# --- windows: the boundary cases ------------------------------------------


@pytest.mark.parametrize("days", [365, 730, 1095])
def test_window_includes_a_record_exactly_at_the_boundary(
    settings: Settings, tmp_path: Path, days: int
) -> None:
    """The window is half-open [d - N, d), so exactly N days before is inside."""
    from datetime import date, timedelta

    boundary = date(2022, 6, 15) - timedelta(days=days)
    row = build(settings, tmp_path, [canvass(1, boundary.isoformat())])
    assert row[f"canvasses_last_{days}d"] == 1


@pytest.mark.parametrize("days", [365, 730, 1095])
def test_window_excludes_a_record_one_day_outside(
    settings: Settings, tmp_path: Path, days: int
) -> None:
    from datetime import date, timedelta

    outside = date(2022, 6, 15) - timedelta(days=days + 1)
    row = build(settings, tmp_path, [canvass(1, outside.isoformat())])
    assert row[f"canvasses_last_{days}d"] == 0
    assert row["prior_canvass_count"] == 1  # still counted unbounded


@pytest.mark.parametrize("days", [365, 730, 1095])
def test_window_includes_a_record_one_day_inside(
    settings: Settings, tmp_path: Path, days: int
) -> None:
    from datetime import date, timedelta

    inside = date(2022, 6, 15) - timedelta(days=days - 1)
    row = build(settings, tmp_path, [canvass(1, inside.isoformat())])
    assert row[f"canvasses_last_{days}d"] == 1


def test_windows_are_nested(settings: Settings, tmp_path: Path) -> None:
    row = build(
        settings,
        tmp_path,
        [canvass(1, "2022-01-01"), canvass(2, "2021-01-01"), canvass(3, "2020-01-01")],
    )
    assert row["canvasses_last_365d"] == 1
    assert row["canvasses_last_730d"] == 2
    assert row["canvasses_last_1095d"] == 3


def test_window_priority_events_are_paired_with_their_counts(
    settings: Settings, tmp_path: Path
) -> None:
    """An empty window must be legible: 0 events beside 0 canvasses.

    2020-01-01 is 896 days before the reference, so it falls inside the 1095-day
    window and outside the 365- and 730-day ones.
    """
    row = build(
        settings, tmp_path, [canvass(1, "2020-01-01", violations=PRIORITY_ONLY, results="Fail")]
    )
    assert row["canvasses_last_365d"] == 0
    assert row["canvass_priority_events_last_365d"] == 0
    assert row["canvasses_last_730d"] == 0
    assert row["canvasses_last_1095d"] == 1
    assert row["canvass_priority_events_last_1095d"] == 1


# --- tenant change --------------------------------------------------------


def test_name_change_is_detected(settings: Settings, tmp_path: Path) -> None:
    row = build(
        settings,
        tmp_path,
        [canvass(1, "2020-01-01", name="OLD DINER")],
        reference_name="NEW DINER",
    )
    assert row["name_changed_since_last_canvass"] is True
    assert row["prior_canvass_count"] == 1
    assert row["prior_canvass_count_current_name"] == 0


def test_no_name_change_when_the_name_is_stable(settings: Settings, tmp_path: Path) -> None:
    row = build(
        settings,
        tmp_path,
        [canvass(1, "2020-01-01", name="SAME DINER")],
        reference_name="SAME DINER",
    )
    assert row["name_changed_since_last_canvass"] is False
    assert row["prior_canvass_count_current_name"] == 1


def test_name_change_flag_is_null_without_a_prior_canvass(
    settings: Settings, tmp_path: Path
) -> None:
    row = build(settings, tmp_path, [], reference_name="NEW DINER")
    assert row["name_changed_since_last_canvass"] is None
    assert row["prior_canvass_count_current_name"] == 0


def test_current_name_count_partitions_the_history(settings: Settings, tmp_path: Path) -> None:
    """History spans the tenant change; the split says how much belongs to now."""
    row = build(
        settings,
        tmp_path,
        [
            canvass(1, "2018-01-01", name="OLD DINER"),
            canvass(2, "2019-01-01", name="OLD DINER"),
            canvass(3, "2021-01-01", name="NEW DINER"),
        ],
        reference_name="NEW DINER",
    )
    assert row["prior_canvass_count"] == 3
    assert row["prior_canvass_count_current_name"] == 1
    assert row["name_changed_since_last_canvass"] is False
