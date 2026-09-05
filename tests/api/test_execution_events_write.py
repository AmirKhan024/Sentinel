from __future__ import annotations

from fastapi.testclient import TestClient

from sentinel.config import Settings
from tests.api.conftest import (
    DEFAULT_SCHEDULE_SCOPE,
    execution_contract_row,
    execution_log_row,
    seed_execution_contract,
    seed_execution_log,
)

VALID_EVENT = {
    "execution_id": "X1",
    "schedule_config_id": "strict_priority__observed_calendar",
    "policy_id": "pure_risk",
    "fold_id": "quarterly-2026Q1",
    "k_name": "k_1_day",
    "target_inspection_id": "T1",
    "scheduled_date": "2026-01-05",
    "execution_status": "completed",
    "reason_code": "field_report",
    "actor": "inspector1",
    "observed_at": "2026-01-05T18:00:00Z",
}


def test_valid_execution_event_is_staged_pending(client: TestClient) -> None:
    response = client.post("/v1/execution/events", json=VALID_EVENT)
    assert response.status_code == 201
    assert response.json()["status"] == "pending"


def test_derived_status_cannot_be_submitted(client: TestClient) -> None:
    """`no_execution_record` is a derived summary category, never a status a person can supply."""
    payload = {**VALID_EVENT, "execution_status": "no_execution_record"}
    response = client.post("/v1/execution/events", json=payload)
    assert response.status_code == 422
    assert "derived summary category" in response.json()["detail"]


def test_unknown_status_is_refused(client: TestClient) -> None:
    payload = {**VALID_EVENT, "execution_status": "vanished"}
    response = client.post("/v1/execution/events", json=payload)
    assert response.status_code == 422


def test_execution_contract_is_readable_without_scope(
    client: TestClient, api_settings: Settings
) -> None:
    seed_execution_contract(api_settings, [execution_contract_row()])
    response = client.get("/v1/execution/contract")
    assert response.status_code == 200
    assert response.json()[0]["contract_name"] == "execution_event"


def test_target_inspection_id_filter_keeps_only_that_establishments_events(
    client: TestClient, api_settings: Settings
) -> None:
    seed_execution_log(
        api_settings,
        [
            execution_log_row(execution_id="X1", target_inspection_id="T1"),
            execution_log_row(execution_id="X2", target_inspection_id="T2"),
        ],
    )
    response = client.get(
        "/v1/execution/events", params={**DEFAULT_SCHEDULE_SCOPE, "target_inspection_id": "T1"}
    )
    assert response.status_code == 200
    rows = response.json()["data"]
    assert [row["execution_id"] for row in rows] == ["X1"]
