"""Component 21, build-level: a real Component 20 artifact in, a plan review out."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.geographic_organization.definitions import GEOGRAPHIC_ORGANIZATION_DEFINITION_VERSION
from sentinel.geographic_organization.models import ArtifactRecord as GeoArtifactRecord
from sentinel.geographic_organization.models import GeographicOrganizationManifest
from sentinel.geographic_organization.writer import OUTPUT_SCHEMA as GEO_OUTPUT_SCHEMA
from sentinel.manifest import compute_sha256, manifest_path_for, write_manifest
from sentinel.plan_review.build import PlanReviewBuildError, build_plan_review

PLANNING_DATE = "2026-08-28"


def _plan_row(i: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "planning_date": PLANNING_DATE,
        "geographic_organization_definition_version": GEOGRAPHIC_ORGANIZATION_DEFINITION_VERSION,
        "geographic_algorithm": "distance_threshold_connected_components",
        "threshold_km": 1.5,
        "operational_selection_definition_version": "v1",
        "requested_capacity": 5,
        "policy_id": "pure_risk",
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
        "as_of_latitude": 41.88 + i * 0.001,
        "as_of_longitude": -87.63 - i * 0.001,
        "n_prior_records": 5,
        "base_score": round(1.0 - i * 0.01, 6),
        "calibrated_score": round(1.0 - i * 0.01, 6),
        "rank": i + 1,
        "policy_rank": i + 1,
        "coverage_eligible": False,
        "secondary_no_history": False,
        "selection_mechanism": "selected_by_risk_rank",
        "selection_reason": "selected_by_risk_rank",
        "is_selected": True,
        "location_status": "location_available",
        "geographic_group_id": f"area_{i + 1}",
        "geographic_group_label": f"Area {i + 1}",
        "work_block_id": f"area_{i + 1}",
        "work_block_label": f"Area {i + 1}",
        "suggested_order_in_block": 1,
        "organization_mode": "risk_first",
        "highest_sentinel_rank_in_block": i + 1,
    }
    row.update(overrides)
    return row


def _write_plan_fixture(tmp_path: Path, n: int = 5) -> Path:
    rows = [_plan_row(i) for i in range(n)]
    ordered = [{name: r.get(name) for name in GEO_OUTPUT_SCHEMA} for r in rows]
    frame = pl.DataFrame(ordered, schema=GEO_OUTPUT_SCHEMA)

    geo_dir = tmp_path / "geographic_organization"
    geo_dir.mkdir(parents=True)
    plan_path = geo_dir / f"geographic_inspection_plan_{PLANNING_DATE}_fixture.parquet"
    frame.write_parquet(plan_path)

    manifest = GeographicOrganizationManifest(
        code_version="test",
        geographic_organization_definition_version=GEOGRAPHIC_ORGANIZATION_DEFINITION_VERSION,
        built_at=datetime.now(UTC).isoformat(),
        planning_date=PLANNING_DATE,
        selection_artifact_path="fixture.parquet",
        selection_artifact_sha256="0" * 64,
        operational_selection_definition_version="v1",
        composite_model_name="xgboost_platt",
        geographic_algorithm="distance_threshold_connected_components",
        threshold_km=1.5,
        selected_count=n,
        location_available_count=n,
        location_unavailable_count=0,
        location_coverage_pct=100.0,
        geographic_group_count=n,
        group_metrics=[],
        organization_mode="risk_first",
        threshold_preset=None,
        work_blocks=[],
        notes=[],
        warnings=[],
        artifacts=[
            GeoArtifactRecord(
                path=plan_path.name,
                bytes=plan_path.stat().st_size,
                sha256=compute_sha256(plan_path),
                row_count=frame.height,
                schema={},
            )
        ],
        checks=[],
    )
    write_manifest(manifest, manifest_path_for(plan_path))
    return plan_path


@pytest.fixture
def plan_path(tmp_path: Path) -> Path:
    return _write_plan_fixture(tmp_path)


@pytest.fixture
def review_settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


def _write_decisions(tmp_path: Path, decisions: list[dict[str, object]]) -> Path:
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(decisions), encoding="utf-8")
    return path


# --- core invariants --------------------------------------------------------------------


def test_plan_review_preserves_the_full_establishment_set(
    review_settings: Settings, plan_path: Path
) -> None:
    result = build_plan_review(review_settings, plan_path=plan_path, dry_run=True)
    assert result.plan_frame.height == 5


def test_no_decisions_file_leaves_every_decision_column_null(
    review_settings: Settings, plan_path: Path
) -> None:
    result = build_plan_review(review_settings, plan_path=plan_path, dry_run=True)
    assert result.plan_frame["supervisor_decision_action"].null_count() == 5
    assert result.summary.decisions_recorded == 0
    assert result.summary.approval_status == "draft"


def test_a_decision_is_joined_beside_not_instead_of_the_recommendation(
    review_settings: Settings, plan_path: Path, tmp_path: Path
) -> None:
    tid = f"CANDIDATE::{PLANNING_DATE}::EST-000000"
    decisions_path = _write_decisions(
        tmp_path,
        [
            {
                "decision_id": "DEC-0001",
                "planning_date": PLANNING_DATE,
                "target_inspection_id": tid,
                "decision_action": "do_not_proceed_as_planned",
                "reason_code": "duplicate_inspection",
                "actor": "supervisor.jsmith",
                "decided_at": "2026-09-02T15:00:00Z",
            }
        ],
    )
    result = build_plan_review(
        review_settings, plan_path=plan_path, decisions_path=decisions_path, dry_run=True
    )
    row = result.plan_frame.filter(pl.col("target_inspection_id") == tid).row(0, named=True)
    assert row["supervisor_decision_action"] == "do_not_proceed_as_planned"
    # Sentinel's own recommendation is untouched, right beside the decision.
    assert row["policy_rank"] == 1
    assert row["selection_reason"] == "selected_by_risk_rank"
    assert result.summary.decisions_recorded == 1
    assert result.summary.approval_status == "under_supervisor_review"


def test_deciding_every_establishment_yields_adjusted_status(
    review_settings: Settings, plan_path: Path, tmp_path: Path
) -> None:
    decisions = [
        {
            "decision_id": f"DEC-{i:04d}",
            "planning_date": PLANNING_DATE,
            "target_inspection_id": f"CANDIDATE::{PLANNING_DATE}::EST-{i:06d}",
            "decision_action": "keep_selected",
            "reason_code": "no_concern",
            "actor": "supervisor.jsmith",
            "decided_at": "2026-09-02T15:00:00Z",
        }
        for i in range(5)
    ]
    decisions_path = _write_decisions(tmp_path, decisions)
    result = build_plan_review(
        review_settings, plan_path=plan_path, decisions_path=decisions_path, dry_run=True
    )
    assert result.summary.approval_status == "adjusted"
    assert result.summary.decisions_recorded == 5


def test_a_decision_for_an_establishment_not_in_the_plan_does_not_crash_and_is_logged_rejected(
    review_settings: Settings, plan_path: Path, tmp_path: Path
) -> None:
    decisions_path = _write_decisions(
        tmp_path,
        [
            {
                "decision_id": "DEC-0001",
                "planning_date": PLANNING_DATE,
                "target_inspection_id": "CANDIDATE::2026-08-28::EST-999999",
                "decision_action": "keep_selected",
                "reason_code": "no_concern",
                "actor": "supervisor.jsmith",
                "decided_at": "2026-09-02T15:00:00Z",
            }
        ],
    )
    result = build_plan_review(
        review_settings, plan_path=plan_path, decisions_path=decisions_path, dry_run=True
    )
    assert result.summary.decisions_recorded == 0
    assert any("did not apply" in w for w in result.manifest.warnings)


def test_malformed_decisions_file_refuses_the_whole_run(
    review_settings: Settings, plan_path: Path, tmp_path: Path
) -> None:
    decisions_path = _write_decisions(
        tmp_path,
        [
            {
                "decision_id": "DEC-0001",
                "planning_date": PLANNING_DATE,
                "target_inspection_id": f"CANDIDATE::{PLANNING_DATE}::EST-000000",
                "decision_action": "move_to_later_workday",  # missing revised_planned_date
                "reason_code": "no_concern",
                "actor": "supervisor.jsmith",
                "decided_at": "2026-09-02T15:00:00Z",
            }
        ],
    )
    with pytest.raises(PlanReviewBuildError):
        build_plan_review(
            review_settings, plan_path=plan_path, decisions_path=decisions_path, dry_run=True
        )


def test_all_invariant_checks_pass_for_a_real_build(
    review_settings: Settings, plan_path: Path
) -> None:
    from sentinel.plan_review import validate as plan_review_validate

    result = build_plan_review(review_settings, plan_path=plan_path, dry_run=True)
    assert not plan_review_validate.has_failures(result.checks)


def test_adjust_operational_priority_never_changes_the_machine_rank(
    review_settings: Settings, plan_path: Path, tmp_path: Path
) -> None:
    tid = f"CANDIDATE::{PLANNING_DATE}::EST-000002"
    decisions_path = _write_decisions(
        tmp_path,
        [
            {
                "decision_id": "DEC-0001",
                "planning_date": PLANNING_DATE,
                "target_inspection_id": tid,
                "decision_action": "adjust_operational_priority",
                "reason_code": "coordination_required",
                "actor": "supervisor.jsmith",
                "decided_at": "2026-09-02T15:00:00Z",
                "revised_operational_priority": 1,
            }
        ],
    )
    result = build_plan_review(
        review_settings, plan_path=plan_path, decisions_path=decisions_path, dry_run=True
    )
    row = result.plan_frame.filter(pl.col("target_inspection_id") == tid).row(0, named=True)
    # Sentinel's own risk rank/policy_rank is untouched (3rd establishment -> rank/policy_rank 3).
    assert row["rank"] == 3
    assert row["policy_rank"] == 3
    # The display-only operational_priority column reflects the supervisor's override.
    assert row["operational_priority"] == 1
    assert row["supervisor_revised_operational_priority"] == 1

    other_row = result.plan_frame.filter(
        pl.col("target_inspection_id") != tid
    ).row(0, named=True)
    # An establishment with no override still has operational_priority == policy_rank.
    assert other_row["operational_priority"] == other_row["policy_rank"]


def test_approval_is_refused_when_a_recorded_decision_has_no_reason_code(
    review_settings: Settings, plan_path: Path, tmp_path: Path
) -> None:
    from sentinel.plan_review.approval import check_readiness, has_blocking_failures

    result = build_plan_review(review_settings, plan_path=plan_path, dry_run=True)
    review_frame = result.plan_frame.with_columns(
        pl.when(pl.col("target_inspection_id") == f"CANDIDATE::{PLANNING_DATE}::EST-000000")
        .then(pl.lit("keep_selected"))
        .otherwise(pl.col("supervisor_decision_action"))
        .alias("supervisor_decision_action")
    )
    checks = check_readiness(review_frame)
    assert has_blocking_failures(checks)
    failing_names = {c.name for c in checks if not c.passed}
    assert "every_recorded_decision_has_a_reason" in failing_names


def test_approval_readiness_passes_for_a_real_build_with_no_decisions(
    review_settings: Settings, plan_path: Path
) -> None:
    from sentinel.plan_review.approval import check_readiness, has_blocking_failures

    result = build_plan_review(review_settings, plan_path=plan_path, dry_run=True)
    checks = check_readiness(result.plan_frame)
    assert not has_blocking_failures(checks)


def test_approve_plan_end_to_end_writes_an_immutable_artifact_and_preserves_priority_override(
    review_settings: Settings, plan_path: Path, tmp_path: Path
) -> None:
    from sentinel.plan_review.build import build_approved_plan
    from sentinel.plan_review.models import PlanApprovalRequest

    tid = f"CANDIDATE::{PLANNING_DATE}::EST-000001"
    decisions_path = _write_decisions(
        tmp_path,
        [
            {
                "decision_id": "DEC-0001",
                "planning_date": PLANNING_DATE,
                "target_inspection_id": tid,
                "decision_action": "adjust_operational_priority",
                "reason_code": "coordination_required",
                "actor": "supervisor.jsmith",
                "decided_at": "2026-09-02T15:00:00Z",
                "revised_operational_priority": 1,
            }
        ],
    )
    review_result = build_plan_review(
        review_settings,
        plan_path=plan_path,
        decisions_path=decisions_path,
        output_dir=tmp_path / "review_out",
        dry_run=False,
    )
    assert review_result.plan_review_path is not None

    approval_request = PlanApprovalRequest(
        approval_id="APPR-0001",
        planning_date=PLANNING_DATE,
        approved_by="supervisor.jsmith",
        approved_at="2026-09-02T16:00:00Z",
    )
    approval_result = build_approved_plan(
        review_settings,
        review_path=review_result.plan_review_path,
        approval_request=approval_request,
        output_dir=tmp_path / "approved_out",
        dry_run=False,
    )
    assert approval_result.approved_path is not None
    assert approval_result.manifest.final_selected_count == 5
    row = approval_result.approved_frame.filter(
        pl.col("target_inspection_id") == tid
    ).row(0, named=True)
    assert row["operational_priority"] == 1
    assert row["policy_rank"] == 2  # EST-000001 is the 2nd row -> rank/policy_rank 2, unchanged

    # The approved artifact is content-addressed by its source review checksum, not overwritten
    # in place -- re-approving the same source (a no-op here, since nothing changed) still
    # names the identical source checksum.
    assert approval_result.manifest.source_plan_review_sha256


def test_approve_plan_refuses_when_planning_date_does_not_match(
    review_settings: Settings, plan_path: Path, tmp_path: Path
) -> None:
    from sentinel.plan_review.build import PlanApprovalBuildError, build_approved_plan
    from sentinel.plan_review.models import PlanApprovalRequest

    review_result = build_plan_review(
        review_settings, plan_path=plan_path, output_dir=tmp_path / "review_out2", dry_run=False
    )
    assert review_result.plan_review_path is not None

    approval_request = PlanApprovalRequest(
        approval_id="APPR-0002",
        planning_date="2099-01-01",
        approved_by="supervisor.jsmith",
        approved_at="2026-09-02T16:00:00Z",
    )
    with pytest.raises(PlanApprovalBuildError):
        build_approved_plan(
            review_settings,
            review_path=review_result.plan_review_path,
            approval_request=approval_request,
            dry_run=True,
        )


def test_a_v1_geographic_plan_missing_work_block_id_is_refused(
    review_settings: Settings, tmp_path: Path
) -> None:
    rows = [_plan_row(0)]
    ordered = [{name: r.get(name) for name in GEO_OUTPUT_SCHEMA} for r in rows]
    frame = pl.DataFrame(ordered, schema=GEO_OUTPUT_SCHEMA).drop("work_block_id")
    geo_dir = tmp_path / "geographic_organization"
    geo_dir.mkdir(parents=True)
    v1_plan_path = geo_dir / f"geographic_inspection_plan_{PLANNING_DATE}_v1fixture.parquet"
    frame.write_parquet(v1_plan_path)
    manifest_src = _write_plan_fixture(tmp_path.parent / "other", n=1)
    # Reuse a valid manifest shape but pointed at the v1-shaped (missing-column) table.
    import shutil

    shutil.copy(manifest_path_for(manifest_src), manifest_path_for(v1_plan_path))
    with pytest.raises(PlanReviewBuildError):
        build_plan_review(review_settings, plan_path=v1_plan_path, dry_run=True)
