"""End-to-end feature construction, the output contract, and provenance."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.features.build import FeatureConstructionError, build_features, summarize
from sentinel.features.definitions import (
    FEATURE_COLUMNS,
    KEY_COLUMNS,
    LABEL_COLUMNS,
    PROVENANCE_COLUMNS,
)
from sentinel.features.models import FeatureManifest
from sentinel.features.writer import OUTPUT_COLUMNS, output_schema
from sentinel.manifest import read_manifest_as
from sentinel.target.models import TARGET_DEFINITION_VERSION
from tests.conftest import assignment_frame, make_inspection_record, target_scenario

PRIORITY = "3. MANAGEMENT - Comments: NO POLICY. PRIORITY FOUNDATION 7-38-010."
CORE = "55. PHYSICAL FACILITIES - Comments: DIRTY FLOOR."

#   EST-A  2019 canvass (priority)  -> history for the 2021 row
#   EST-A  2021 canvass             -> target row, one prior canvass
#   EST-B  2022 canvass             -> target row, no history at all
ROWS = [
    make_inspection_record(
        1,
        inspection_id="1001",
        inspection_date="2019-05-01T00:00:00.000",
        violations=PRIORITY,
        results="Fail",
    ),
    make_inspection_record(
        2,
        inspection_id="1002",
        inspection_date="2021-05-01T00:00:00.000",
        violations=CORE,
        results="Pass",
    ),
    make_inspection_record(
        3,
        inspection_id="1003",
        inspection_date="2022-05-01T00:00:00.000",
        violations=CORE,
        results="Pass",
    ),
]
PAIRS = [("1001", "EST-A"), ("1002", "EST-A"), ("1003", "EST-B")]

TARGETS = pl.DataFrame(
    {
        "establishment_id": ["EST-A", "EST-B", "EST-A"],
        "inspection_date": [
            "2021-05-01T00:00:00.000",
            "2022-05-01T00:00:00.000",
            "2019-05-01T00:00:00.000",
        ],
        "target_inspection_id": ["1002", "1003", "1001"],
        "target": [0, 0, 1],
        "target_status": ["eligible", "eligible", "ineligible_era"],
        "code_era_phase": ["stable", "stable", "pre_code"],
    },
    schema={
        "establishment_id": pl.Utf8,
        "inspection_date": pl.Utf8,
        "target_inspection_id": pl.Utf8,
        "target": pl.Int8,
        "target_status": pl.Utf8,
        "code_era_phase": pl.Utf8,
    },
)

EXPECTED_ROWS = 2  # only the eligible target rows are featurised


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw = tmp_path / "food_inspections_20260816T000000Z.parquet"
    asg = tmp_path / "establishment_assignments_20260816T000000Z.parquet"
    tgt = tmp_path / "inspection_targets_20260816T000000Z.parquet"
    target_scenario(ROWS).write_parquet(raw)
    assignment_frame(PAIRS).write_parquet(asg)
    TARGETS.write_parquet(tgt)
    return raw, asg, tgt


def build(settings: Settings, inputs: tuple[Path, Path, Path], **kwargs: object):  # type: ignore[no-untyped-def]
    raw, asg, tgt = inputs
    return build_features(
        settings,
        parquet_path=raw,
        assignments_path=asg,
        targets_path=tgt,
        **kwargs,  # type: ignore[arg-type]
    )


# --- grain and content ----------------------------------------------------


def test_one_row_per_eligible_target(settings: Settings, inputs: tuple[Path, Path, Path]) -> None:
    frame = build(settings, inputs, dry_run=True).features
    assert frame.height == EXPECTED_ROWS
    assert set(frame["target_inspection_id"]) == {"1002", "1003"}


def test_ineligible_target_rows_are_not_featurised(
    settings: Settings, inputs: tuple[Path, Path, Path]
) -> None:
    """An ineligible row has no prediction to make, so it has no feature row."""
    frame = build(settings, inputs, dry_run=True).features
    assert "1001" not in set(frame["target_inspection_id"])


def test_expected_feature_values(settings: Settings, inputs: tuple[Path, Path, Path]) -> None:
    frame = build(settings, inputs, dry_run=True).features
    by_id = {r["target_inspection_id"]: r for r in frame.iter_rows(named=True)}

    with_history = by_id["1002"]
    assert with_history["prior_canvass_count"] == 1
    assert with_history["prior_canvass_priority_count"] == 1
    assert with_history["days_since_last_canvass"] == 731

    cold = by_id["1003"]
    assert cold["prior_canvass_count"] == 0
    assert cold["days_since_last_canvass"] is None
    assert cold["prior_canvass_priority_count"] is None


def test_primary_key_is_unique(settings: Settings, inputs: tuple[Path, Path, Path]) -> None:
    frame = build(settings, inputs, dry_run=True).features
    assert frame["target_inspection_id"].n_unique() == frame.height


def test_validation_passes(settings: Settings, inputs: tuple[Path, Path, Path]) -> None:
    result = build(settings, inputs, dry_run=True)
    assert [c for c in result.checks if not c.passed and c.severity == "error"] == []


# --- output contract ------------------------------------------------------


def test_schema_matches_the_contract(settings: Settings, inputs: tuple[Path, Path, Path]) -> None:
    frame = build(settings, inputs, dry_run=True).features
    assert list(frame.columns) == list(OUTPUT_COLUMNS)
    expected = output_schema()
    assert [str(d) for d in frame.dtypes] == [str(expected[c]) for c in frame.columns]


def test_column_groups_are_all_present(settings: Settings, inputs: tuple[Path, Path, Path]) -> None:
    columns = set(build(settings, inputs, dry_run=True).features.columns)
    for group in (KEY_COLUMNS, FEATURE_COLUMNS, LABEL_COLUMNS, PROVENANCE_COLUMNS):
        assert set(group) <= columns


def test_no_component_3_outcome_columns_leak_through(
    settings: Settings, inputs: tuple[Path, Path, Path]
) -> None:
    """results / evidence / n_*_entries describe the outcome, not the past."""
    columns = set(build(settings, inputs, dry_run=True).features.columns)
    forbidden = {
        "results",
        "evidence",
        "n_priority_entries",
        "n_violation_entries",
        "has_priority",
        "inspection_type",
    }
    assert not columns & forbidden


def test_target_is_carried_but_is_not_a_feature(
    settings: Settings, inputs: tuple[Path, Path, Path]
) -> None:
    frame = build(settings, inputs, dry_run=True).features
    assert "target" in frame.columns
    assert "target" not in FEATURE_COLUMNS


def test_feature_count_is_reported(settings: Settings, inputs: tuple[Path, Path, Path]) -> None:
    result = build(settings, inputs, dry_run=True)
    assert result.manifest.feature_count == len(FEATURE_COLUMNS)


# --- determinism ----------------------------------------------------------


def test_two_builds_are_identical(settings: Settings, inputs: tuple[Path, Path, Path]) -> None:
    first = build(settings, inputs, dry_run=True).features
    second = build(settings, inputs, dry_run=True).features
    assert first.equals(second)


def test_rows_are_canonically_sorted(settings: Settings, inputs: tuple[Path, Path, Path]) -> None:
    frame = build(settings, inputs, dry_run=True).features
    assert frame.equals(frame.sort(["establishment_id", "inspection_date", "target_inspection_id"]))


# --- writing and provenance -----------------------------------------------


def test_writes_a_table_and_a_manifest(
    settings: Settings, inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    result = build(settings, inputs, output_dir=tmp_path / "out")
    assert result.features_path is not None and result.features_path.exists()
    assert result.manifest_path is not None and result.manifest_path.exists()


def test_manifest_pins_all_three_inputs(
    settings: Settings, inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """Features are a function of raw + identities + targets together."""
    result = build(settings, inputs, output_dir=tmp_path / "out")
    assert result.manifest_path is not None
    loaded = read_manifest_as(FeatureManifest, result.manifest_path)
    for digest in (loaded.source_sha256, loaded.assignments_sha256, loaded.targets_sha256):
        assert len(digest) == 64
    assert len({loaded.source_sha256, loaded.assignments_sha256, loaded.targets_sha256}) == 3


def test_manifest_records_the_temporal_boundary(
    settings: Settings, inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """A consumer should never have to infer the boundary from the code."""
    result = build(settings, inputs, output_dir=tmp_path / "out")
    assert "strictly before" in result.manifest.temporal_boundary
    assert result.manifest.window_days == [365, 730, 1095]
    assert result.manifest.code_era_start == "2018-07-01"


def test_manifest_is_readable_json(
    settings: Settings, inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    result = build(settings, inputs, output_dir=tmp_path / "out")
    assert result.manifest_path is not None
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["component"] == "as_of_features"
    assert payload["feature_definition_version"] == "v1"


def test_written_table_reloads_identically(
    settings: Settings, inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    result = build(settings, inputs, output_dir=tmp_path / "out")
    assert result.features_path is not None
    assert pl.read_parquet(result.features_path).equals(result.features)


def test_dry_run_writes_nothing(
    settings: Settings, inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    result = build(settings, inputs, output_dir=out, dry_run=True)
    assert result.features_path is None
    assert not out.exists()


# --- failure modes --------------------------------------------------------


@pytest.mark.parametrize("index", [0, 1, 2])
def test_missing_any_input_raises(
    settings: Settings, inputs: tuple[Path, Path, Path], tmp_path: Path, index: int
) -> None:
    paths = list(inputs)
    paths[index] = tmp_path / "absent.parquet"
    with pytest.raises(FileNotFoundError):
        build_features(
            settings,
            parquet_path=paths[0],
            assignments_path=paths[1],
            targets_path=paths[2],
            dry_run=True,
        )


def test_raw_without_required_columns_raises(
    settings: Settings, inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    _, asg, tgt = inputs
    bad = tmp_path / "bad.parquet"
    target_scenario(ROWS).drop("dba_name").write_parquet(bad)
    with pytest.raises(FeatureConstructionError, match="missing required columns"):
        build_features(
            settings, parquet_path=bad, assignments_path=asg, targets_path=tgt, dry_run=True
        )


def test_no_eligible_targets_produces_an_empty_table(
    settings: Settings, inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    raw, asg, _ = inputs
    tgt = tmp_path / "none.parquet"
    TARGETS.with_columns(pl.lit("ineligible_era").alias("target_status")).write_parquet(tgt)
    result = build_features(
        settings, parquet_path=raw, assignments_path=asg, targets_path=tgt, dry_run=True
    )
    assert result.features.height == 0
    assert list(result.features.columns) == list(OUTPUT_COLUMNS)


def test_summarize_reports_the_headline_counts(
    settings: Settings, inputs: tuple[Path, Path, Path]
) -> None:
    text = summarize(build(settings, inputs, dry_run=True))
    assert f"feature rows:           {EXPECTED_ROWS}" in text
    assert "strictly before" in text


def test_component_3_definition_version_is_unchanged() -> None:
    """Component 4 must not silently alter the target definition."""
    assert TARGET_DEFINITION_VERSION == "v1"
