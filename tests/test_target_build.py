"""End-to-end target construction, the output contract, and determinism.

The leakage tests are as load-bearing as the determinism ones. Component 4 joins
onto this table, so a stray historical column here would become a feature by
default rather than by decision.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.manifest import read_manifest_as
from sentinel.target.build import TargetConstructionError, build_targets, summarize
from sentinel.target.models import TargetManifest
from sentinel.target.writer import TARGET_EVENT_COLUMNS, TARGETS_SCHEMA
from tests.conftest import assignment_frame, make_inspection_record, target_scenario

PRIORITY = "3. MANAGEMENT - Comments: NO POLICY. PRIORITY FOUNDATION 7-38-010."
CORE = "55. PHYSICAL FACILITIES - Comments: DIRTY FLOOR."

# A scenario with a known correct answer:
#   1001 EST-A  canvass, priority                  -> eligible, target 1
#   1002 EST-A  canvass same day, core only        -> collapses into 1001 (OR)
#   1003 EST-A  canvass later date, core only      -> eligible, target 0
#   1004 EST-B  canvass, Pass, no violations       -> eligible, target 0
#   1005 EST-B  Out of Business                    -> ineligible_result
#   1006 EST-B  Complaint with a priority          -> ineligible_type
#   1007 EST-C  pre-code canvass with a priority   -> ineligible_era
#   1008 EST-C  canvass, Fail, no violations       -> unknown_violations
SCENARIO = [
    make_inspection_record(
        1,
        inspection_id="1001",
        violations=PRIORITY,
        results="Pass w/ Conditions",
        inspection_date="2022-03-14T00:00:00.000",
    ),
    make_inspection_record(
        2,
        inspection_id="1002",
        violations=CORE,
        results="Pass",
        inspection_date="2022-03-14T00:00:00.000",
    ),
    make_inspection_record(
        3,
        inspection_id="1003",
        violations=CORE,
        results="Pass",
        inspection_date="2023-05-02T00:00:00.000",
    ),
    make_inspection_record(4, inspection_id="1004", violations=None, results="Pass"),
    make_inspection_record(5, inspection_id="1005", violations=None, results="Out of Business"),
    make_inspection_record(
        6, inspection_id="1006", violations=PRIORITY, inspection_type="Complaint", results="Fail"
    ),
    make_inspection_record(
        7,
        inspection_id="1007",
        violations=PRIORITY,
        results="Fail",
        inspection_date="2015-06-01T00:00:00.000",
    ),
    make_inspection_record(8, inspection_id="1008", violations=None, results="Fail"),
]

PAIRS = [
    ("1001", "EST-A"),
    ("1002", "EST-A"),
    ("1003", "EST-A"),
    ("1004", "EST-B"),
    ("1005", "EST-B"),
    ("1006", "EST-B"),
    ("1007", "EST-C"),
    ("1008", "EST-C"),
]

# Three eligible ROWS, not four eligible inspections: 1001 and 1002 are the same
# establishment on the same date and collapse into a single decision point.
EXPECTED_ELIGIBLE = 3
EXPECTED_POSITIVE = 1


@pytest.fixture
def raw_path(tmp_path: Path) -> Path:
    path = tmp_path / "food_inspections_20260816T000000Z.parquet"
    target_scenario(SCENARIO).write_parquet(path, compression="zstd")
    return path


@pytest.fixture
def assignments_path(tmp_path: Path) -> Path:
    path = tmp_path / "establishment_assignments_20260816T000000Z.parquet"
    assignment_frame(PAIRS).write_parquet(path)
    return path


def build(settings: Settings, raw: Path, asg: Path, **kwargs: object):  # type: ignore[no-untyped-def]
    return build_targets(settings, parquet_path=raw, assignments_path=asg, **kwargs)  # type: ignore[arg-type]


# --- the scenario resolves as designed ------------------------------------


def test_scenario_produces_the_expected_labels(
    settings: Settings, raw_path: Path, assignments_path: Path
) -> None:
    frame = build(settings, raw_path, assignments_path, dry_run=True).targets
    eligible = frame.filter(pl.col("target_status") == "eligible")
    assert eligible.height == EXPECTED_ELIGIBLE
    assert eligible["target"].sum() == EXPECTED_POSITIVE

    by_id = {r["target_inspection_id"]: r for r in eligible.iter_rows(named=True)}
    assert by_id["1001"]["target"] == 1
    assert by_id["1001"]["n_contributing_inspections"] == 2  # same-day collapse
    assert by_id["1003"]["target"] == 0
    assert by_id["1004"]["target"] == 0  # Pass with no violations is a true zero


def test_every_exclusion_has_the_right_reason(
    settings: Settings, raw_path: Path, assignments_path: Path
) -> None:
    frame = build(settings, raw_path, assignments_path, dry_run=True).targets
    status = {r["target_inspection_id"]: r["target_status"] for r in frame.iter_rows(named=True)}
    assert status["1005"] == "ineligible_result"
    assert status["1006"] == "ineligible_type"
    assert status["1007"] == "ineligible_era"
    assert status["1008"] == "unknown_violations"


def test_every_inspection_is_accounted_for(
    settings: Settings, raw_path: Path, assignments_path: Path
) -> None:
    frame = build(settings, raw_path, assignments_path, dry_run=True).targets
    assert frame["n_contributing_inspections"].sum() == len(SCENARIO)


def test_excluded_rows_carry_no_label(
    settings: Settings, raw_path: Path, assignments_path: Path
) -> None:
    frame = build(settings, raw_path, assignments_path, dry_run=True).targets
    excluded = frame.filter(pl.col("target_status") != "eligible")
    assert excluded["target"].null_count() == excluded.height


def test_validation_passes_on_a_clean_scenario(
    settings: Settings, raw_path: Path, assignments_path: Path
) -> None:
    result = build(settings, raw_path, assignments_path, dry_run=True)
    assert [c for c in result.checks if not c.passed and c.severity == "error"] == []


# --- leakage boundary -----------------------------------------------------


def test_output_contains_no_historical_aggregate_columns(
    settings: Settings, raw_path: Path, assignments_path: Path
) -> None:
    """Component 4's job. A stray column here becomes a feature by default."""
    frame = build(settings, raw_path, assignments_path, dry_run=True).targets
    forbidden_prefixes = ("prev_", "n_prior_", "days_since_", "rolling_", "hist_", "prior_")
    offenders = [c for c in frame.columns if c.startswith(forbidden_prefixes)]
    assert offenders == []


def test_target_event_columns_are_enumerated_for_the_contract() -> None:
    """The contract forbids these as features; the set is asserted in code too."""
    assert "target" in TARGET_EVENT_COLUMNS
    assert "results" in TARGET_EVENT_COLUMNS
    assert "evidence" in TARGET_EVENT_COLUMNS
    assert set(TARGETS_SCHEMA) >= TARGET_EVENT_COLUMNS


def test_inspection_date_is_present_as_the_as_of_boundary(
    settings: Settings, raw_path: Path, assignments_path: Path
) -> None:
    frame = build(settings, raw_path, assignments_path, dry_run=True).targets
    assert "inspection_date" in frame.columns
    assert frame["inspection_date"].null_count() == 0


# --- determinism ----------------------------------------------------------


def test_two_builds_produce_identical_tables(
    settings: Settings, raw_path: Path, assignments_path: Path
) -> None:
    first = build(settings, raw_path, assignments_path, dry_run=True).targets
    second = build(settings, raw_path, assignments_path, dry_run=True).targets
    assert first.equals(second)


def test_shuffled_input_produces_identical_labels(
    settings: Settings, tmp_path: Path, assignments_path: Path
) -> None:
    """Row order must not reach the labels."""
    ordered = tmp_path / "ordered.parquet"
    shuffled = tmp_path / "shuffled.parquet"
    target_scenario(SCENARIO).write_parquet(ordered)

    rows = SCENARIO[:]
    random.Random(20260816).shuffle(rows)
    target_scenario(rows).write_parquet(shuffled)

    a = build(settings, ordered, assignments_path, dry_run=True).targets
    b = build(settings, shuffled, assignments_path, dry_run=True).targets
    assert a.equals(b)


# --- output contract ------------------------------------------------------


def test_schema_matches_the_contract(
    settings: Settings, raw_path: Path, assignments_path: Path
) -> None:
    frame = build(settings, raw_path, assignments_path, dry_run=True).targets
    assert list(frame.columns) == list(TARGETS_SCHEMA)
    assert [str(d) for d in frame.dtypes] == [str(d) for d in TARGETS_SCHEMA.values()]


def test_identifiers_stay_utf8(settings: Settings, raw_path: Path, assignments_path: Path) -> None:
    frame = build(settings, raw_path, assignments_path, dry_run=True).targets
    for column in ("establishment_id", "target_inspection_id", "inspection_date"):
        assert frame.schema[column] == pl.Utf8


def test_target_is_nullable_int_not_boolean(
    settings: Settings, raw_path: Path, assignments_path: Path
) -> None:
    """Null is a meaningful third state; a nullable boolean invites coercion."""
    frame = build(settings, raw_path, assignments_path, dry_run=True).targets
    assert frame.schema["target"] == pl.Int8
    assert frame["target"].null_count() > 0


# --- writing and provenance -----------------------------------------------


def test_writes_a_table_and_a_manifest(
    settings: Settings, raw_path: Path, assignments_path: Path, tmp_path: Path
) -> None:
    result = build(settings, raw_path, assignments_path, output_dir=tmp_path / "out")
    assert result.targets_path is not None and result.targets_path.exists()
    assert result.manifest_path is not None and result.manifest_path.exists()


def test_manifest_pins_both_inputs(
    settings: Settings, raw_path: Path, assignments_path: Path, tmp_path: Path
) -> None:
    """Labels are a function of the snapshot AND the assignments together."""
    result = build(settings, raw_path, assignments_path, output_dir=tmp_path / "out")
    assert result.manifest_path is not None
    loaded = read_manifest_as(TargetManifest, result.manifest_path)
    assert loaded.source_path == raw_path.name
    assert loaded.assignments_path == assignments_path.name
    assert len(loaded.source_sha256) == 64
    assert len(loaded.assignments_sha256) == 64
    assert loaded.eligible_rows == EXPECTED_ELIGIBLE
    assert loaded.target_definition_version == "v1"


def test_manifest_records_the_definition_parameters(
    settings: Settings, raw_path: Path, assignments_path: Path, tmp_path: Path
) -> None:
    result = build(settings, raw_path, assignments_path, output_dir=tmp_path / "out")
    assert result.manifest.code_era_start == "2018-07-01"
    assert result.manifest.canvass_type == "CANVASS"
    assert result.manifest.inspected_results == ["Fail", "Pass", "Pass w/ Conditions"]


def test_manifest_is_readable_json(
    settings: Settings, raw_path: Path, assignments_path: Path, tmp_path: Path
) -> None:
    result = build(settings, raw_path, assignments_path, output_dir=tmp_path / "out")
    assert result.manifest_path is not None
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["component"] == "target_construction"


def test_written_table_reloads_identically(
    settings: Settings, raw_path: Path, assignments_path: Path, tmp_path: Path
) -> None:
    result = build(settings, raw_path, assignments_path, output_dir=tmp_path / "out")
    assert result.targets_path is not None
    assert pl.read_parquet(result.targets_path).equals(result.targets)


def test_dry_run_writes_nothing(
    settings: Settings, raw_path: Path, assignments_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    result = build(settings, raw_path, assignments_path, output_dir=out, dry_run=True)
    assert result.targets_path is None
    assert not out.exists()


# --- failure modes --------------------------------------------------------


def test_missing_raw_file_raises(
    settings: Settings, tmp_path: Path, assignments_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        build(settings, tmp_path / "absent.parquet", assignments_path, dry_run=True)


def test_missing_assignments_raises(settings: Settings, raw_path: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build(settings, raw_path, tmp_path / "absent.parquet", dry_run=True)


def test_raw_file_without_required_columns_raises(
    settings: Settings, tmp_path: Path, assignments_path: Path
) -> None:
    path = tmp_path / "bad.parquet"
    target_scenario(SCENARIO).drop("violations").write_parquet(path)
    with pytest.raises(TargetConstructionError, match="missing required columns"):
        build(settings, path, assignments_path, dry_run=True)


def test_empty_input_produces_an_empty_table(
    settings: Settings, tmp_path: Path, assignments_path: Path
) -> None:
    path = tmp_path / "empty.parquet"
    target_scenario([]).write_parquet(path)
    result = build(settings, path, assignments_path, dry_run=True)
    assert result.targets.height == 0
    assert list(result.targets.columns) == list(TARGETS_SCHEMA)


def test_unassigned_inspections_are_dropped_not_guessed(settings: Settings, tmp_path: Path) -> None:
    """Identity is Component 2's contract; an unassigned row has no establishment."""
    raw = tmp_path / "raw.parquet"
    asg = tmp_path / "asg.parquet"
    target_scenario(SCENARIO).write_parquet(raw)
    assignment_frame(PAIRS[:4]).write_parquet(asg)
    result = build(settings, raw, asg, dry_run=True)
    assert result.targets["n_contributing_inspections"].sum() == 4


def test_summarize_reports_the_headline_counts(
    settings: Settings, raw_path: Path, assignments_path: Path
) -> None:
    text = summarize(build(settings, raw_path, assignments_path, dry_run=True))
    assert f"eligible:       {EXPECTED_ELIGIBLE}" in text
    assert "definition:       v1" in text
