"""Fixtures for the Sentinel API test suite.

Seeds minimal, correctly-typed Parquet artifacts directly through each component's own
``writer.finalize``/``write_table`` -- never by running the pipeline -- exactly as the
project's other component tests build fixtures. A JSON manifest sidecar is written by hand
(a plain dict, not the full ``PolicyManifest``/``ScheduleManifest`` model) because the API only
ever reads ``built_at`` out of it; constructing the real manifest models would mean restating
every one of their many required fields for no behaviour under test.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from fastapi.testclient import TestClient

from sentinel.api.app import create_app
from sentinel.api.deps import get_settings
from sentinel.config import Settings
from sentinel.explain import writer as explain_writer
from sentinel.plan_review import writer as plan_review_writer
from sentinel.policy import writer as policy_writer
from sentinel.review import writer as review_writer
from sentinel.scheduling import writer as schedule_writer

TS = "20260101T000000Z"


def _write(directory: Path, writer_module: Any, table: str, rows: list[dict[str, object]]) -> Path:
    frame = writer_module.finalize(rows, table)
    path = directory / f"{table}_{TS}.parquet"
    writer_module.write_table(frame, path)
    manifest_path = path.with_name(f"manifest_{path.stem}.json")
    manifest_path.write_text(
        json.dumps({"built_at": "2026-01-01T00:00:00+00:00"}), encoding="utf-8"
    )
    return path


@pytest.fixture
def api_settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def client(api_settings: Settings) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: api_settings
    return TestClient(app)


# --- Row builders -------------------------------------------------------------


def recommendation_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "policy_id": "pure_risk",
        "model_name": "lightgbm_platt",
        "fold_set": "quarterly",
        "fold_id": "quarterly-2026Q1",
        "k_name": "k_1_day",
        "k": 10,
        "target_inspection_id": "T1",
        "establishment_id": "E1",
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
        "warnings": "none",
        "group_value": "__UNKNOWN__",
        "group_status": "unsupported",
        "policy_definition_version": "v1",
    }
    base.update(overrides)
    return base


def allocation_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "policy_id": "pure_risk",
        "model_name": "lightgbm_platt",
        "fold_set": "quarterly",
        "fold_id": "quarterly-2026Q1",
        "k_name": "k_1_day",
        "k": 10,
        "n_universe": 100,
        "reserve_mechanism": "none",
        "reserve_share": 0.0,
        "reserve_target": 0,
        "n_eligible_available": 5,
        "n_eligible_in_risk_top_k": 1,
        "n_risk": 10,
        "n_reserve": 0,
        "n_selected": 10,
        "reserve_inert": False,
        "policy_definition_version": "v1",
    }
    base.update(overrides)
    return base


def override_log_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "override_id": "O1",
        "policy_id": "pure_risk",
        "fold_set": "quarterly",
        "fold_id": "quarterly-2026Q1",
        "k_name": "k_1_day",
        "target_inspection_id": "T1",
        "action": "force_include",
        "reason_code": "supervisor_request",
        "actor": "jsmith",
        "decided_at": "2026-01-02T00:00:00Z",
        "original_is_selected": False,
        "original_mechanism": "not_selected",
        "original_reason": "not_selected_capacity_exhausted",
        "original_policy_rank": 12,
        "final_is_selected": True,
        "displaced_target_inspection_id": "T2",
        "outcome": "applied",
        "policy_definition_version": "v1",
    }
    base.update(overrides)
    return base


def schedule_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schedule_config_id": "strict_priority__observed_calendar",
        "policy_id": "pure_risk",
        "model_name": "lightgbm_platt",
        "fold_set": "quarterly",
        "fold_id": "quarterly-2026Q1",
        "k_name": "k_1_day",
        "k": 10,
        "target_inspection_id": "T1",
        "establishment_id": "E1",
        "recommendation_date": date(2026, 1, 5),
        "base_score": 0.5,
        "score": 0.6,
        "model_rank": 1,
        "final_policy_rank": 1,
        "decision_mechanism": "risk_priority",
        "decision_reason": "selected_by_risk_rank",
        "coverage_eligible": False,
        "warnings": "none",
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


def backlog_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schedule_config_id": "strict_priority__observed_calendar",
        "policy_id": "pure_risk",
        "model_name": "lightgbm_platt",
        "fold_set": "quarterly",
        "fold_id": "quarterly-2026Q1",
        "k_name": "k_1_day",
        "k": 10,
        "target_inspection_id": "T9",
        "establishment_id": "E9",
        "final_policy_rank": 20,
        "decision_mechanism": "risk_priority",
        "decision_reason": "selected_by_risk_rank",
        "coverage_eligible": False,
        "backlog_position": 1,
        "backlog_reason": "capacity_exhausted_in_horizon",
        "horizon_slots": 10,
        "slots_short": 1,
        "would_fit_on_day_index": None,
        "first_available_date": None,
        "planning_run_id": "P1",
        "replan_index": 0,
        "is_scenario": False,
        "schedule_definition_version": "v1",
    }
    base.update(overrides)
    return base


def adjustment_log_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "adjustment_id": "A1",
        "schedule_config_id": "strict_priority__observed_calendar",
        "policy_id": "pure_risk",
        "fold_id": "quarterly-2026Q1",
        "k_name": "k_1_day",
        "target_inspection_id": "T1",
        "action": "defer_to_date",
        "target_date": "2026-01-06",
        "reason_code": "supervisor_request",
        "actor": "jsmith",
        "decided_at": "2026-01-02T00:00:00Z",
        "original_status": "scheduled",
        "original_scheduled_date": date(2026, 1, 5),
        "original_schedule_rank": 1,
        "final_status": "scheduled",
        "final_scheduled_date": date(2026, 1, 6),
        "displaced_target_inspection_id": "",
        "displaced_landed_status": "",
        "outcome": "applied",
        "planning_run_id": "P1",
        "replan_index": 1,
        "schedule_definition_version": "v1",
    }
    base.update(overrides)
    return base


def execution_log_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "execution_id": "X1",
        "schedule_config_id": "strict_priority__observed_calendar",
        "policy_id": "pure_risk",
        "fold_id": "quarterly-2026Q1",
        "k_name": "k_1_day",
        "target_inspection_id": "T1",
        "scheduled_date": date(2026, 1, 5),
        "plan_scheduled_date": date(2026, 1, 5),
        "execution_status": "completed",
        "reason_code": "field_report",
        "actor": "inspector1",
        "observed_at": "2026-01-05T18:00:00Z",
        "outcome": "recorded",
        "triggers_replan": False,
        "applied_at_replan_index": 0,
        "schedule_definition_version": "v1",
    }
    base.update(overrides)
    return base


def execution_summary_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schedule_config_id": "strict_priority__observed_calendar",
        "policy_id": "pure_risk",
        "model_name": "lightgbm_platt",
        "fold_set": "quarterly",
        "fold_id": "quarterly-2026Q1",
        "k_name": "k_1_day",
        "n_scheduled": 10,
        "n_completed": 8,
        "n_not_performed": 1,
        "n_cancelled_in_field": 0,
        "n_no_execution_record": 1,
        "completion_rate": 0.8,
        "final_replan_index": 0,
        "execution_log_sha256": "",
        "schedule_definition_version": "v1",
    }
    base.update(overrides)
    return base


def replanning_run_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schedule_config_id": "strict_priority__observed_calendar",
        "policy_id": "pure_risk",
        "fold_set": "quarterly",
        "fold_id": "quarterly-2026Q1",
        "k_name": "k_1_day",
        "planning_run_id": "P1",
        "replan_index": 0,
        "parent_replan_index": None,
        "replan_from_date": None,
        "trigger": "original_plan",
        "n_preserved_completed": 0,
        "n_preserved_past": 0,
        "n_returned_to_queue": 0,
        "n_cancelled": 0,
        "n_newly_scheduled": 10,
        "n_still_backlog": 1,
        "remaining_slots": 0,
        "execution_log_sha256": "",
        "schedule_definition_version": "v1",
    }
    base.update(overrides)
    return base


def execution_contract_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "contract_name": "execution_event",
        "field_name": "execution_status",
        "required": True,
        "dtype": "str",
        "allowed_values": "completed|not_performed|cancelled_in_field",
        "meaning": "what the field reported",
        "schedule_definition_version": "v1",
    }
    base.update(overrides)
    return base


def review_queue_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "policy_id": "pure_risk",
        "model_name": "lightgbm_platt",
        "fold_set": "quarterly",
        "fold_id": "quarterly-2026Q1",
        "k_name": "k_1_day",
        "target_inspection_id": "T1",
        "establishment_id": "E1",
        "final_policy_rank": 1,
        "decision_mechanism": "risk_priority",
        "decision_reason": "selected_by_risk_rank",
        "warnings": "limited_history",
        "trigger_reasons": "policy_warning_present",
        "schedule_config_id": "",
        "planning_run_id": "",
        "replan_index": None,
        "scheduled_date": None,
        "review_status": "flagged",
        "review_id": "",
        "resolution_action": "",
        "review_definition_version": "v1",
    }
    base.update(overrides)
    return base


def review_resolution_log_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "review_id": "R1",
        "policy_id": "pure_risk",
        "fold_id": "quarterly-2026Q1",
        "k_name": "k_1_day",
        "target_inspection_id": "T1",
        "resolution_action": "acknowledge",
        "reason_code": "reviewed",
        "actor": "jsmith",
        "decided_at": "2026-01-02T00:00:00Z",
        "referenced_override_id": "",
        "referenced_adjustment_id": "",
        "escalation_note": "",
        "original_status": "flagged",
        "final_status": "resolved",
        "outcome": "applied",
        "review_definition_version": "v1",
    }
    base.update(overrides)
    return base


def plan_review_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "planning_date": "2026-08-28",
        "geographic_organization_definition_version": "v2",
        "geographic_algorithm": "distance_threshold_connected_components",
        "threshold_km": 1.5,
        "operational_selection_definition_version": "v1",
        "requested_capacity": 30,
        "policy_id": "pure_risk",
        "composite_model_name": "xgboost_platt",
        "base_model_name": "xgboost",
        "calibration_method": "platt",
        "establishment_id": "E1",
        "target_inspection_id": "T1",
        "canonical_name": "NAME-1",
        "canonical_address": "ADDR-1",
        "canonical_zip": "60601",
        "as_of_dba_name": "NAME-1",
        "as_of_address": "ADDR-1",
        "as_of_zip": "60601",
        "as_of_latitude": 41.88,
        "as_of_longitude": -87.63,
        "n_prior_records": 5,
        "base_score": 0.9,
        "calibrated_score": 0.9,
        "rank": 1,
        "policy_rank": 1,
        "coverage_eligible": False,
        "secondary_no_history": False,
        "selection_mechanism": "selected_by_risk_rank",
        "selection_reason": "selected_by_risk_rank",
        "is_selected": True,
        "location_status": "location_available",
        "geographic_group_id": "area_1",
        "geographic_group_label": "Area 1",
        "work_block_id": "area_1",
        "work_block_label": "Area 1",
        "suggested_order_in_block": 1,
        "organization_mode": "risk_first",
        "highest_sentinel_rank_in_block": 1,
        "plan_review_definition_version": "v1",
        "supervisor_decision_id": None,
        "supervisor_decision_action": None,
        "supervisor_decision_reason_code": None,
        "supervisor_decision_actor": None,
        "supervisor_decision_decided_at": None,
        "supervisor_revised_planned_date": None,
        "supervisor_revised_work_block_id": None,
    }
    base.update(overrides)
    return base


def plan_decision_log_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "decision_id": "DEC-0001",
        "planning_date": "2026-08-28",
        "target_inspection_id": "T1",
        "decision_action": "keep_selected",
        "reason_code": "no_concern",
        "actor": "jsmith",
        "decided_at": "2026-01-02T00:00:00Z",
        "revised_planned_date": None,
        "revised_work_block_id": None,
        "outcome": "applied",
        "plan_review_definition_version": "v1",
    }
    base.update(overrides)
    return base


def explanation_support_row(**overrides: object) -> dict[str, object]:
    # Component 11's own tables carry the *base* model name, never Component 9's calibrated
    # name (docs/data_contracts/explanations.md 0a) -- e.g. "lightgbm", never "lightgbm_platt".
    # `recommendation_row()`'s "lightgbm_platt" is correct for Component 13/14; this is not.
    base: dict[str, object] = {
        "model_name": "lightgbm",
        "model_version": "v1",
        "family": "boosted_tree",
        "component": 11,
        "source_slug": "boosted_predictions",
        "explanation_status": "supported",
        "explanation_method": "tree_shap",
        "output_space": "probability",
        "is_exact": True,
        "is_experimental": False,
        "name_source": "definitions",
        "rationale": "exact TreeSHAP",
        "unsupported_reason": "",
        "explained_rows": 10,
        "attribution_values": 300,
    }
    base.update(overrides)
    return base


def explanation_case_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "model_name": "lightgbm",
        "model_version": "v1",
        "family": "boosted_tree",
        "fold_set": "quarterly",
        "fold_id": "quarterly-2026Q1",
        "target_inspection_id": "T1",
        "output_space": "probability",
        "explanation_method": "tree_shap",
        "is_exact": True,
        "base_value": 0.4,
        "prediction_value": 0.6,
        "reconstruction_value": 0.6,
        "reconstruction_residual": 0.0,
        "additivity_tolerance": 1e-6,
        "additivity_holds": True,
        "n_features": 2,
        "positive_contribution_sum": 0.3,
        "negative_contribution_sum": -0.1,
        "base_score": 0.6,
        "base_score_reproduced": True,
        "calibrated_probability": 0.6,
        "calibration_method": "platt",
        "base_model_trained_through": date(2025, 12, 31),
        "calibrator_fitted_through": date(2025, 12, 31),
        "prediction_available_from": date(2026, 1, 1),
        "sample_strategy": "stratified",
        "sample_size": 10,
        "sampling_seed": 0,
        "sampling_population": "test_window",
        "population_rows": 100,
        "background_strategy": "median",
        "background_size": 1,
        "background_seed": 0,
        "background_max_date": date(2025, 12, 31),
        "permutation_rounds": 1,
        "explain_definition_version": "v1",
    }
    base.update(overrides)
    return base


def explanation_value_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "model_name": "lightgbm",
        "model_version": "v1",
        "family": "boosted_tree",
        "fold_set": "quarterly",
        "fold_id": "quarterly-2026Q1",
        "target_inspection_id": "T1",
        "feature_name": "prior_canvass_count_code_era",
        "original_feature_name": "prior_canvass_count_code_era",
        "derived_from": "prior_canvass_count_code_era",
        "feature_kind": "numeric",
        "feature_value": 2.0,
        "transformed_value": 2.0,
        "shap_value": 0.1,
        "output_space": "probability",
        "explanation_method": "tree_shap",
        "is_exact": True,
        "base_value": 0.4,
        "prediction_value": 0.6,
        "trained_through": date(2025, 12, 31),
        "explain_definition_version": "v1",
    }
    base.update(overrides)
    return base


def feature_row(**overrides: object) -> dict[str, object]:
    """A curated slice of a real Component 4 as-of-feature row -- values match the establishment
    traced in the product reality check (EST-00002282595 / target 2637537 in production data)."""
    base: dict[str, object] = {
        "target_inspection_id": "T1",
        "prior_canvass_count_code_era": 7,
        "prior_canvass_priority_count": 6,
        "prior_canvass_priority_rate": 0.857143,
        "prior_canvass_fail_rate": 0.333333,
        "fail_at_last_canvass": False,
        "priority_at_last_canvass": True,
        "days_since_last_canvass": 345,
        "days_since_any_inspection": 331,
        "prior_inspection_count_any_type": 26,
        "name_changed_since_last_canvass": True,
    }
    base.update(overrides)
    return base


# --- Seeders --------------------------------------------------------------


def seed_recommendations(settings: Settings, rows: list[dict[str, object]]) -> Path:
    return _write(settings.policy_processed_dir, policy_writer, "inspection_recommendations", rows)


def seed_allocation(settings: Settings, rows: list[dict[str, object]]) -> Path:
    return _write(settings.policy_processed_dir, policy_writer, "policy_selection_allocation", rows)


def seed_override_log(settings: Settings, rows: list[dict[str, object]]) -> Path:
    return _write(settings.policy_processed_dir, policy_writer, "policy_override_log", rows)


def seed_schedule(settings: Settings, rows: list[dict[str, object]]) -> Path:
    return _write(settings.scheduling_processed_dir, schedule_writer, "inspection_schedule", rows)


def seed_backlog(settings: Settings, rows: list[dict[str, object]]) -> Path:
    return _write(settings.scheduling_processed_dir, schedule_writer, "schedule_backlog", rows)


def seed_adjustment_log(settings: Settings, rows: list[dict[str, object]]) -> Path:
    return _write(
        settings.scheduling_processed_dir, schedule_writer, "schedule_adjustment_log", rows
    )


def seed_execution_log(settings: Settings, rows: list[dict[str, object]]) -> Path:
    return _write(settings.scheduling_processed_dir, schedule_writer, "execution_log", rows)


def seed_execution_summary(settings: Settings, rows: list[dict[str, object]]) -> Path:
    return _write(settings.scheduling_processed_dir, schedule_writer, "execution_summary", rows)


def seed_execution_contract(settings: Settings, rows: list[dict[str, object]]) -> Path:
    return _write(settings.scheduling_processed_dir, schedule_writer, "execution_contract", rows)


def seed_replanning_runs(settings: Settings, rows: list[dict[str, object]]) -> Path:
    return _write(settings.scheduling_processed_dir, schedule_writer, "replanning_runs", rows)


def seed_review_queue(settings: Settings, rows: list[dict[str, object]]) -> Path:
    return _write(settings.review_processed_dir, review_writer, "human_review_queue", rows)


def seed_review_resolution_log(settings: Settings, rows: list[dict[str, object]]) -> Path:
    return _write(settings.review_processed_dir, review_writer, "review_resolution_log", rows)


def seed_operational_selection(
    settings: Settings, *, planning_date: str = "2026-08-28", **manifest_overrides: object
) -> Path:
    """Writes a minimal ``operational_selection`` artifact + a bare manifest sidecar --
    ``meta_service.get_manifest`` reads only the JSON sidecar, never the parquet's schema, so
    unlike ``seed_approved_plan`` this needs no real Pydantic model, just realistic field names."""
    directory = settings.operational_selection_processed_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"operational_selection_{planning_date}_cap30_{TS}.parquet"
    pl.DataFrame({"establishment_id": ["E1"]}).write_parquet(path)
    manifest_path = path.with_name(f"manifest_{path.stem}.json")
    fields: dict[str, object] = {
        "built_at": "2026-01-01T00:00:00+00:00",
        "ranked_candidate_count": 100,
        "selectable_candidate_count": 95,
        "selected_count": 30,
        "risk_selected_count": 30,
        "reserve_selected_count": 0,
        "coverage_eligible_selected_count": 30,
    }
    fields.update(manifest_overrides)
    manifest_path.write_text(json.dumps(fields), encoding="utf-8")
    return path


def seed_plan_review(settings: Settings, rows: list[dict[str, object]]) -> Path:
    """Writes a ``supervisor_plan_review`` table directly with a full row schema (Component
    20's plan columns plus Component 21's decision columns) -- not through
    ``plan_review.writer.finalize``, which joins decisions onto an already-built Component 20
    frame rather than accepting raw rows, unlike the other components' ``finalize(rows, table)``
    this module's ``_write`` helper assumes."""
    directory = settings.plan_review_processed_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"supervisor_plan_review_2026-08-28_{TS}.parquet"
    pl.DataFrame(rows).write_parquet(path)
    manifest_path = path.with_name(f"manifest_{path.stem}.json")
    manifest_path.write_text(
        json.dumps({"built_at": "2026-01-01T00:00:00+00:00"}), encoding="utf-8"
    )
    return path


def seed_approved_plan(
    settings: Settings, *, planning_date: str = "2026-08-28", **manifest_overrides: object
) -> Path:
    """Writes a committed ``approved_operational_plan`` artifact + its real manifest, exactly
    the shape `plan_review_service.get_plan_approval` reads back through
    ``read_manifest_as(ApprovedPlanManifest, ...)`` -- unlike the other ``seed_*`` helpers, this
    one cannot get away with a bare ``{"built_at": ...}`` sidecar, because that reader validates
    the full model."""
    from sentinel.manifest import manifest_path_for, write_manifest
    from sentinel.plan_review.models import ApprovedPlanManifest

    directory = settings.plan_review_processed_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"approved_operational_plan_{planning_date}_{TS}.parquet"
    pl.DataFrame([plan_review_row(planning_date=planning_date)]).write_parquet(path)

    fields: dict[str, object] = {
        "code_version": "test",
        "plan_review_definition_version": "v1",
        "built_at": "2026-01-01T00:00:00+00:00",
        "approval_id": "APPR-SEED-0001",
        "planning_date": planning_date,
        "approved_by": "supervisor.demo",
        "approved_at": "2026-01-01T00:00:00+00:00",
        "note": None,
        "source_plan_review_path": "supervisor_plan_review_seed.parquet",
        "source_plan_review_sha256": "0" * 64,
        "source_decision_log_path": None,
        "source_decision_log_sha256": None,
        "final_selected_count": 1,
        "final_active_count": 1,
        "final_deferred_count": 0,
        "final_not_proceeding_count": 0,
        "final_undecided_count": 1,
        "plan_review_cannot": "",
        "does_not_establish": [],
        "inherited_limitations": [],
        "artifacts": [],
        "readiness_checks": [],
    }
    fields.update(manifest_overrides)
    manifest = ApprovedPlanManifest.model_validate(fields)
    write_manifest(manifest, manifest_path_for(path))
    return path


def seed_plan_decision_log(settings: Settings, rows: list[dict[str, object]]) -> Path:
    directory = settings.plan_review_processed_dir
    directory.mkdir(parents=True, exist_ok=True)
    frame = plan_review_writer.finalize_decision_log(rows)
    path = directory / f"plan_decision_log_2026-08-28_{TS}.parquet"
    plan_review_writer.write_table(frame, path)
    manifest_path = path.with_name(f"manifest_{path.stem}.json")
    manifest_path.write_text(
        json.dumps({"built_at": "2026-01-01T00:00:00+00:00"}), encoding="utf-8"
    )
    return path


def seed_explanation_support(settings: Settings, rows: list[dict[str, object]]) -> Path:
    return _write(settings.explanations_processed_dir, explain_writer, "explanation_support", rows)


def seed_explanation_cases(settings: Settings, rows: list[dict[str, object]]) -> Path:
    return _write(settings.explanations_processed_dir, explain_writer, "explanation_cases", rows)


def seed_explanation_values(settings: Settings, rows: list[dict[str, object]]) -> Path:
    return _write(settings.explanations_processed_dir, explain_writer, "explanation_values", rows)


def seed_features(settings: Settings, rows: list[dict[str, object]]) -> Path:
    """Writes a minimal feature table directly, not through ``features.writer.finalize`` --

    the API only ever reads a small curated subset of columns (see
    ``establishment_service._HISTORY_FACTOR_FIELDS``), so a test fixture only needs those columns
    plus the key, unlike Component 4's own full 33-column contract.
    """
    directory = settings.features_processed_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"as_of_features_{TS}.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


DEFAULT_SCOPE: dict[str, str] = {
    "policy_id": "pure_risk",
    "fold_set": "quarterly",
    "fold_id": "quarterly-2026Q1",
    "k_name": "k_1_day",
}

DEFAULT_SCHEDULE_SCOPE: dict[str, str] = {
    **DEFAULT_SCOPE,
    "schedule_config_id": "strict_priority__observed_calendar",
}
