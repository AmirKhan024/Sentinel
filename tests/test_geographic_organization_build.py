"""Component 20 v2, build-level: threshold resolution, organization mode, and the
honesty note, exercised through ``build_geographic_plan`` against a real Component 19
artifact on disk -- not just the pure functions in isolation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.geographic_organization.build import (
    GeographicOrganizationBuildError,
    build_geographic_plan,
)
from sentinel.geographic_organization.definitions import (
    GeographicOrganizationDefinitionError,
    OrganizationMode,
)
from sentinel.manifest import compute_sha256, manifest_path_for, write_manifest
from sentinel.operational_selection.models import ArtifactRecord, OperationalSelectionManifest
from sentinel.operational_selection.writer import OUTPUT_SCHEMA

PLANNING_DATE = "2026-08-28"
CITY_HALL = (41.8838, -87.6319)


def _row(i: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "planning_date": PLANNING_DATE,
        "operational_selection_definition_version": "v1",
        "requested_capacity": 30,
        "policy_id": "pure_risk",
        "candidate_definition_version": "v1",
        "feature_definition_version": "v1",
        "operational_scoring_definition_version": "v1",
        "composite_model_name": "xgboost_platt",
        "base_model_name": "xgboost",
        "calibration_method": "platt",
        "establishment_id": f"EST-{i:06d}",
        "target_inspection_id": f"CANDIDATE::{PLANNING_DATE}::EST-{i:06d}",
        "canonical_name": f"NAME-{i}",
        "canonical_address": f"ADDR-{i}",
        "canonical_zip": "60601",
        "as_of_dba_name": f"NAME-{i}",
        "as_of_address": f"ADDR-{i}",
        "as_of_zip": "60601",
        "as_of_latitude": CITY_HALL[0] + i * 0.05,
        "as_of_longitude": CITY_HALL[1] - i * 0.05,
        "has_location": True,
        "n_prior_records": 5,
        "scoring_status": "scored",
        "base_score": round(1.0 - i * 0.01, 6),
        "calibrated_score": round(1.0 - i * 0.01, 6),
        "rank": i + 1,
        "coverage_eligible": False,
        "secondary_no_history": False,
        "selection_mechanism": "selected_by_risk_rank",
        "selection_reason": "selected_by_risk_rank",
        "policy_rank": i + 1,
        "is_selected": True,
    }
    row.update(overrides)
    return row


def _write_selection_fixture(tmp_path: Path, n: int = 10) -> Path:
    rows = [_row(i) for i in range(n)]
    ordered = [{name: row.get(name) for name in OUTPUT_SCHEMA} for row in rows]
    frame = pl.DataFrame(ordered, schema=OUTPUT_SCHEMA)

    selection_dir = tmp_path / "operational_selection"
    selection_dir.mkdir(parents=True)
    selection_path = selection_dir / f"operational_selection_{PLANNING_DATE}_cap{n}_fixture.parquet"
    frame.write_parquet(selection_path)

    manifest = OperationalSelectionManifest(
        code_version="test",
        operational_selection_definition_version="v1",
        built_at=datetime.now(UTC).isoformat(),
        planning_date=PLANNING_DATE,
        priority_set_path="fixture.parquet",
        priority_set_sha256="0" * 64,
        operational_scoring_definition_version="v1",
        composite_model_name="xgboost_platt",
        requested_capacity=n,
        policy_id="pure_risk",
        policy_mechanism="risk_block",
        policy_reserve_share=0.0,
        allocation_source="policy.allocation",
        ranked_candidate_count=n,
        selectable_candidate_count=n,
        unscorable_count=0,
        selected_count=n,
        reserve_selected_count=0,
        risk_selected_count=n,
        unfilled_capacity=0,
        capacity_utilization=1.0,
        coverage_eligible_selected_count=0,
        warnings=[],
        artifacts=[
            ArtifactRecord(
                path=selection_path.name,
                bytes=selection_path.stat().st_size,
                sha256=compute_sha256(selection_path),
                row_count=frame.height,
                schema={},
            )
        ],
        checks=[],
    )
    write_manifest(manifest, manifest_path_for(selection_path))
    return selection_path


@pytest.fixture
def selection_path(tmp_path: Path) -> Path:
    return _write_selection_fixture(tmp_path)


@pytest.fixture
def build_settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


def test_work_block_id_aliases_geographic_group_id(
    build_settings: Settings, selection_path: Path
) -> None:
    result = build_geographic_plan(
        build_settings, selection_path=selection_path, dry_run=True
    )
    frame = result.plan_frame
    assert frame["work_block_id"].to_list() == frame["geographic_group_id"].to_list()
    assert frame["work_block_label"].to_list() == frame["geographic_group_label"].to_list()


def test_default_mode_is_risk_first_and_notes_include_rationale(
    build_settings: Settings, selection_path: Path
) -> None:
    result = build_geographic_plan(
        build_settings, selection_path=selection_path, dry_run=True
    )
    assert result.manifest.organization_mode == "risk_first"
    assert any("policy_rank ascending" in n for n in result.manifest.notes)


def test_singleton_note_appears_when_establishments_are_dispersed(
    build_settings: Settings, selection_path: Path
) -> None:
    # Fixture spaces establishments 0.05 deg (~5+ km) apart -- well beyond the tight
    # default threshold, so every block should be a singleton.
    result = build_geographic_plan(
        build_settings,
        selection_path=selection_path,
        threshold_preset="tight",
        dry_run=True,
    )
    assert any("contain a single establishment" in n for n in result.manifest.notes)


def test_singleton_note_absent_when_a_broad_threshold_merges_blocks(
    build_settings: Settings, tmp_path: Path
) -> None:
    # Tightly clustered establishments (~0.5 km apart) so a broad (5 km) threshold
    # merges all of them into one work block.
    tight_rows = [
        _row(i, as_of_latitude=CITY_HALL[0] + i * 0.004, as_of_longitude=CITY_HALL[1])
        for i in range(10)
    ]
    ordered = [{name: r.get(name) for name in OUTPUT_SCHEMA} for r in tight_rows]
    frame = pl.DataFrame(ordered, schema=OUTPUT_SCHEMA)
    tight_selection_path = _write_selection_fixture(tmp_path, n=10)
    frame.write_parquet(tight_selection_path)

    result = build_geographic_plan(
        build_settings,
        selection_path=tight_selection_path,
        threshold_preset="broad",
        dry_run=True,
    )
    assert result.manifest.geographic_group_count == 1
    assert not any("contain a single establishment" in n for n in result.manifest.notes)


def test_geography_assisted_mode_is_recorded_and_still_passes_all_checks(
    build_settings: Settings, selection_path: Path
) -> None:
    from sentinel.geographic_organization import validate as geo_validate

    result = build_geographic_plan(
        build_settings,
        selection_path=selection_path,
        organization_mode=OrganizationMode.GEOGRAPHY_ASSISTED,
        threshold_preset="broad",
        dry_run=True,
    )
    assert result.manifest.organization_mode == "geography_assisted"
    assert not geo_validate.has_failures(result.checks)


def test_threshold_km_and_preset_together_is_refused(
    build_settings: Settings, selection_path: Path
) -> None:
    with pytest.raises(GeographicOrganizationDefinitionError):
        build_geographic_plan(
            build_settings,
            selection_path=selection_path,
            threshold_km=2.0,
            threshold_preset="broad",
            dry_run=True,
        )


def test_all_invariant_checks_pass_for_a_real_build(
    build_settings: Settings, selection_path: Path
) -> None:
    from sentinel.geographic_organization import validate as geo_validate

    result = build_geographic_plan(
        build_settings, selection_path=selection_path, dry_run=True
    )
    assert not geo_validate.has_failures(result.checks)
    assert result.plan_frame.height == 10


def test_unknown_capacity_error_type_unaffected(build_settings: Settings, tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.parquet"
    with pytest.raises((FileNotFoundError, GeographicOrganizationBuildError)):
        build_geographic_plan(build_settings, selection_path=missing, dry_run=True)
