from __future__ import annotations

from fastapi.testclient import TestClient

from sentinel.config import Settings
from tests.api.conftest import (
    override_log_row,
    review_resolution_log_row,
    seed_override_log,
    seed_review_resolution_log,
)

VALID_OVERRIDE = {
    "override_id": "O1",
    "policy_id": "pure_risk",
    "fold_id": "quarterly-2026Q1",
    "k_name": "k_1_day",
    "target_inspection_id": "T1",
    "action": "force_include",
    "reason_code": "supervisor_request",
    "actor": "jsmith",
    "decided_at": "2026-01-02T00:00:00Z",
}


def test_staged_request_starts_pending(client: TestClient) -> None:
    client.post("/v1/policy/overrides", json=VALID_OVERRIDE)
    response = client.get("/v1/staged-requests")
    assert response.status_code == 200
    row = response.json()[0]
    assert row["status"] == "pending"
    assert row["natural_id"] == "O1"


def test_staged_request_reconciles_to_applied_once_committed(
    client: TestClient, api_settings: Settings
) -> None:
    client.post("/v1/policy/overrides", json=VALID_OVERRIDE)
    # Simulate an operator running `sentinel decide` against the staged file: the id now
    # appears in the committed log the batch CLI writes.
    seed_override_log(api_settings, [override_log_row(override_id="O1")])

    response = client.get("/v1/staged-requests", params={"kind": "override"})
    row = response.json()[0]
    assert row["status"] == "applied"


VALID_RESOLUTION = {
    "review_id": "R1",
    "policy_id": "pure_risk",
    "fold_id": "quarterly-2026Q1",
    "k_name": "k_1_day",
    "target_inspection_id": "T1",
    "resolution_action": "acknowledge",
    "reason_code": "reviewed",
    "actor": "jsmith",
    "decided_at": "2026-01-02T00:00:00Z",
}


def test_fourth_kind_reconciles_to_applied_once_committed(
    client: TestClient, api_settings: Settings
) -> None:
    client.post("/v1/review/resolutions", json=VALID_RESOLUTION)
    seed_review_resolution_log(api_settings, [review_resolution_log_row(review_id="R1")])

    response = client.get("/v1/staged-requests", params={"kind": "review_resolution"})
    row = response.json()[0]
    assert row["status"] == "applied"


def test_filter_by_status(client: TestClient, api_settings: Settings) -> None:
    client.post("/v1/policy/overrides", json=VALID_OVERRIDE)
    seed_override_log(api_settings, [override_log_row(override_id="O1")])
    pending = client.get("/v1/staged-requests", params={"status": "pending"}).json()
    applied = client.get("/v1/staged-requests", params={"status": "applied"}).json()
    assert pending == []
    assert len(applied) == 1
