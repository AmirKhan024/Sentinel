from __future__ import annotations

from datetime import date

from fastapi.testclient import TestClient

from sentinel.config import Settings
from tests.api.conftest import (
    DEFAULT_SCHEDULE_SCOPE,
    backlog_row,
    replanning_run_row,
    schedule_row,
    seed_backlog,
    seed_replanning_runs,
    seed_schedule,
)


def test_schedule_requires_full_cell_scope(client: TestClient, api_settings: Settings) -> None:
    seed_schedule(api_settings, [schedule_row()])
    response = client.get("/v1/schedule", params={"policy_id": "pure_risk"})
    assert response.status_code == 422
    assert response.json()["error"] == "ambiguous_scope"


def test_schedule_defaults_to_latest_replan_index(
    client: TestClient, api_settings: Settings
) -> None:
    rows = [
        schedule_row(replan_index=0, scheduled_date=None, schedule_status="backlog"),
        schedule_row(replan_index=1, scheduled_date=date(2026, 1, 6)),
    ]
    seed_schedule(api_settings, rows)
    response = client.get("/v1/schedule", params=DEFAULT_SCHEDULE_SCOPE)
    body = response.json()
    assert len(body["data"]) == 1
    assert body["data"][0]["replan_index"] == 1


def test_schedule_pinned_to_explicit_replan_index(
    client: TestClient, api_settings: Settings
) -> None:
    rows = [
        schedule_row(replan_index=0, schedule_status="scheduled"),
        schedule_row(replan_index=1, schedule_status="scheduled"),
    ]
    seed_schedule(api_settings, rows)
    response = client.get("/v1/schedule", params={**DEFAULT_SCHEDULE_SCOPE, "replan_index": 0})
    body = response.json()
    assert len(body["data"]) == 1
    assert body["data"][0]["replan_index"] == 0


def test_backlog_list(client: TestClient, api_settings: Settings) -> None:
    seed_backlog(api_settings, [backlog_row()])
    response = client.get("/v1/schedule/backlog", params=DEFAULT_SCHEDULE_SCOPE)
    assert response.status_code == 200
    assert response.json()["data"][0]["backlog_reason"] == "capacity_exhausted_in_horizon"


def test_replanning_runs_ordered_by_replan_index(
    client: TestClient, api_settings: Settings
) -> None:
    rows = [
        replanning_run_row(replan_index=1, trigger="execution_not_performed"),
        replanning_run_row(replan_index=0, trigger="original_plan"),
    ]
    seed_replanning_runs(api_settings, rows)
    response = client.get("/v1/schedule/replanning-runs", params=DEFAULT_SCHEDULE_SCOPE)
    body = response.json()
    assert [row["replan_index"] for row in body] == [0, 1]


def test_schedule_missing_artifact_is_404(client: TestClient) -> None:
    response = client.get("/v1/schedule", params=DEFAULT_SCHEDULE_SCOPE)
    assert response.status_code == 404
