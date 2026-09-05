"""End-to-end orchestration: run_review over synthetic Component 13/14 fixtures."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.review.build import run_review
from sentinel.review.resolution import ReviewGovernanceError


def _recommendation_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "policy_id": "pure_risk",
        "model_name": "lightgbm_platt",
        "fold_set": "quarterly",
        "fold_id": "2026Q1",
        "k_name": "k_1_day",
        "k": 2,
        "target_inspection_id": "t1",
        "establishment_id": "e1",
        "inspection_date": date(2026, 1, 5),
        "base_score": 0.5,
        "score": 0.6,
        "model_rank": 1,
        "final_policy_rank": 1,
        "is_selected": True,
        "decision_mechanism": "risk_priority",
        "decision_reason": "selected_by_risk_rank",
        "coverage_eligible": False,
        "secondary_no_history": False,
        "warnings": "limited_history",
        "group_value": "",
        "group_status": "",
        "policy_definition_version": "v1",
    }
    base.update(overrides)
    return base


def _schedule_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schedule_config_id": "strict_priority__observed_calendar",
        "policy_id": "pure_risk",
        "model_name": "lightgbm_platt",
        "fold_set": "quarterly",
        "fold_id": "2026Q1",
        "k_name": "k_1_day",
        "k": 2,
        "target_inspection_id": "t1",
        "establishment_id": "e1",
        "recommendation_date": date(2026, 1, 5),
        "base_score": 0.5,
        "score": 0.6,
        "model_rank": 1,
        "final_policy_rank": 1,
        "decision_mechanism": "risk_priority",
        "decision_reason": "selected_by_risk_rank",
        "coverage_eligible": False,
        "warnings": "limited_history",
        "recommendation_override_id": "",
        "policy_definition_version": "v1",
        "planning_run_id": "P1",
        "replan_index": 0,
        "schedule_status": "scheduled",
        "schedule_reason": "placed_in_priority_order",
        "inversion_reason": "none",
        "scheduled_date": date(2026, 1, 5),
        "day_index": 1,
        "slot_index": 1,
        "schedule_rank": 1,
        "wait_operating_days": 0,
        "original_scheduled_date": date(2026, 1, 5),
        "original_schedule_rank": 1,
        "adjustment_id": "",
        "is_scenario": False,
        "schedule_definition_version": "v1",
    }
    base.update(overrides)
    return base


@pytest.fixture
def recommendations_path(tmp_path: Path) -> Path:
    path = tmp_path / "inspection_recommendations_20260101T000000Z.parquet"
    pl.DataFrame([_recommendation_row()]).write_parquet(path)
    return path


@pytest.fixture
def schedule_path(tmp_path: Path) -> Path:
    path = tmp_path / "inspection_schedule_20260101T000000Z.parquet"
    pl.DataFrame([_schedule_row()]).write_parquet(path)
    return path


@pytest.fixture
def execution_log_path(tmp_path: Path) -> Path:
    path = tmp_path / "execution_log_20260101T000000Z.parquet"
    pl.DataFrame(
        {
            "schedule_config_id": ["strict_priority__observed_calendar"],
            "policy_id": ["pure_risk"],
            "fold_id": ["2026Q1"],
            "k_name": ["k_1_day"],
            "target_inspection_id": ["t1"],
        }
    ).write_parquet(path)
    return path


def test_empty_recommendations_produce_an_empty_queue(settings: Settings, tmp_path: Path) -> None:
    empty_path = tmp_path / "inspection_recommendations_20260101T000000Z.parquet"
    pl.DataFrame([_recommendation_row()]).head(0).write_parquet(empty_path)
    result = run_review(
        settings, recommendations_path=empty_path, write_figures=False, dry_run=True
    )
    assert result.tables["human_review_queue"].is_empty()
    assert result.stats.cases_flagged == 0


def test_no_schedule_flags_only_via_the_warning_trigger(
    settings: Settings, recommendations_path: Path
) -> None:
    result = run_review(
        settings, recommendations_path=recommendations_path, write_figures=False, dry_run=True
    )
    queue = result.tables["human_review_queue"]
    assert queue.height == 1
    assert queue["trigger_reasons"].to_list() == ["policy_warning_present"]


def test_one_case_flagged_by_both_triggers(
    settings: Settings, recommendations_path: Path, schedule_path: Path
) -> None:
    result = run_review(
        settings,
        recommendations_path=recommendations_path,
        schedule_path=schedule_path,
        write_figures=False,
        dry_run=True,
    )
    queue = result.tables["human_review_queue"]
    assert queue.height == 1
    assert queue["trigger_reasons"].to_list() == [
        "no_execution_record_on_scheduled_row|policy_warning_present"
    ]


def test_an_execution_record_closes_the_gap(
    settings: Settings, recommendations_path: Path, schedule_path: Path, execution_log_path: Path
) -> None:
    result = run_review(
        settings,
        recommendations_path=recommendations_path,
        schedule_path=schedule_path,
        execution_log_path=execution_log_path,
        write_figures=False,
        dry_run=True,
    )
    queue = result.tables["human_review_queue"]
    assert queue["trigger_reasons"].to_list() == ["policy_warning_present"]


def test_a_resolution_moves_a_case_to_resolved(
    settings: Settings, recommendations_path: Path, tmp_path: Path
) -> None:
    resolutions_path = tmp_path / "resolutions.json"
    resolutions_path.write_text(
        json.dumps(
            [
                {
                    "review_id": "R1",
                    "policy_id": "pure_risk",
                    "fold_id": "2026Q1",
                    "k_name": "k_1_day",
                    "target_inspection_id": "t1",
                    "resolution_action": "acknowledge",
                    "reason_code": "reviewed",
                    "actor": "alice",
                    "decided_at": "2026-01-06T00:00:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )
    result = run_review(
        settings,
        recommendations_path=recommendations_path,
        resolutions_path=resolutions_path,
        write_figures=False,
        dry_run=True,
    )
    queue = result.tables["human_review_queue"]
    assert queue["review_status"].to_list() == ["resolved"]
    assert result.tables["review_resolution_log"].height == 1
    assert result.stats.resolutions_applied == 1


def test_an_invalid_resolution_file_raises_governance_error(
    settings: Settings, recommendations_path: Path, tmp_path: Path
) -> None:
    resolutions_path = tmp_path / "bad_resolutions.json"
    resolutions_path.write_text(
        json.dumps([{"review_id": "R1", "resolution_action": "not_real"}]), encoding="utf-8"
    )
    with pytest.raises(ReviewGovernanceError):
        run_review(
            settings,
            recommendations_path=recommendations_path,
            resolutions_path=resolutions_path,
            write_figures=False,
            dry_run=True,
        )


def test_a_missing_recommendations_file_raises(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_review(settings, recommendations_path=tmp_path / "nope.parquet", write_figures=False)


def test_dry_run_writes_nothing(settings: Settings, recommendations_path: Path) -> None:
    result = run_review(
        settings, recommendations_path=recommendations_path, write_figures=False, dry_run=True
    )
    assert result.written == []
    assert result.queue_path is None
    assert result.manifest_path is None
    assert not any(settings.review_processed_dir.glob("*"))


def test_a_real_run_writes_the_queue_and_a_manifest(
    settings: Settings, recommendations_path: Path
) -> None:
    result = run_review(
        settings, recommendations_path=recommendations_path, write_figures=False, dry_run=False
    )
    assert result.queue_path is not None
    assert result.queue_path.exists()
    assert result.manifest_path is not None
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["cases_flagged"] == 1
    assert manifest["no_threshold"]
    assert manifest["inputs_unchanged"] is True


def test_upstream_artifacts_are_never_mutated(
    settings: Settings, recommendations_path: Path
) -> None:
    from sentinel.manifest import compute_sha256

    before = compute_sha256(recommendations_path)
    run_review(settings, recommendations_path=recommendations_path, write_figures=False)
    after = compute_sha256(recommendations_path)
    assert before == after


def test_policies_filter_narrows_the_universe(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "inspection_recommendations_20260101T000000Z.parquet"
    pl.DataFrame(
        [
            _recommendation_row(target_inspection_id="t1", policy_id="pure_risk"),
            _recommendation_row(
                target_inspection_id="t2", policy_id="coverage_forced_population_share"
            ),
        ]
    ).write_parquet(path)
    result = run_review(
        settings,
        recommendations_path=path,
        policies=["pure_risk"],
        write_figures=False,
        dry_run=True,
    )
    assert result.tables["human_review_queue"]["target_inspection_id"].to_list() == ["t1"]
