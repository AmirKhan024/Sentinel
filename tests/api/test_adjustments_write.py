from __future__ import annotations

from fastapi.testclient import TestClient

from sentinel.config import Settings
from tests.api.conftest import DEFAULT_SCHEDULE_SCOPE, adjustment_log_row, seed_adjustment_log

VALID_ADJUSTMENT = {
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
}


def test_valid_adjustment_is_staged_pending(client: TestClient) -> None:
    response = client.post("/v1/schedule/adjustments", json=VALID_ADJUSTMENT)
    assert response.status_code == 201
    assert response.json()["status"] == "pending"


def test_cancel_with_a_target_date_is_refused(client: TestClient) -> None:
    """A cancel carries no target_date; one that does is ambiguous (adjustments.py's own rule)."""
    payload = {**VALID_ADJUSTMENT, "action": "cancel", "target_date": "2026-01-06"}
    response = client.post("/v1/schedule/adjustments", json=payload)
    assert response.status_code == 422
    assert "ambiguous" in response.json()["detail"]


def test_unknown_action_is_refused(client: TestClient) -> None:
    payload = {**VALID_ADJUSTMENT, "action": "teleport"}
    response = client.post("/v1/schedule/adjustments", json=payload)
    assert response.status_code == 422
    assert "unknown action" in response.json()["detail"]


def test_duplicate_adjustment_id_different_payload_is_refused(client: TestClient) -> None:
    client.post("/v1/schedule/adjustments", json=VALID_ADJUSTMENT)
    changed = {**VALID_ADJUSTMENT, "reason_code": "other"}
    response = client.post("/v1/schedule/adjustments", json=changed)
    assert response.status_code == 409


def test_target_inspection_id_filter_keeps_only_that_establishments_adjustments(
    client: TestClient, api_settings: Settings
) -> None:
    seed_adjustment_log(
        api_settings,
        [
            adjustment_log_row(adjustment_id="A1", target_inspection_id="T1"),
            adjustment_log_row(adjustment_id="A2", target_inspection_id="T2"),
        ],
    )
    response = client.get(
        "/v1/schedule/adjustments", params={**DEFAULT_SCHEDULE_SCOPE, "target_inspection_id": "T2"}
    )
    assert response.status_code == 200
    rows = response.json()["data"]
    assert [row["adjustment_id"] for row in rows] == ["A2"]
