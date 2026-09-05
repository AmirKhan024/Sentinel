from __future__ import annotations

from fastapi.testclient import TestClient

from sentinel.config import Settings
from tests.api.conftest import (
    DEFAULT_SCOPE,
    review_resolution_log_row,
    seed_review_resolution_log,
)

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


def test_valid_resolution_is_staged_pending(client: TestClient) -> None:
    response = client.post("/v1/review/resolutions", json=VALID_RESOLUTION)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "pending"
    assert body["natural_id"] == "R1"
    assert body["kind"] == "review_resolution"

    staged = client.get("/v1/staged-requests", params={"kind": "review_resolution"})
    assert staged.status_code == 200
    assert staged.json()[0]["natural_id"] == "R1"


def test_corrupted_action_is_refused_with_parser_message(client: TestClient) -> None:
    payload = {**VALID_RESOLUTION, "resolution_action": "delete_forever"}
    response = client.post("/v1/review/resolutions", json=payload)
    assert response.status_code == 422
    assert response.json()["error"] == "validation_refused"
    assert "unknown resolution_action" in response.json()["detail"]


def test_refer_to_override_without_pointer_is_refused(client: TestClient) -> None:
    payload = {**VALID_RESOLUTION, "resolution_action": "refer_to_override"}
    response = client.post("/v1/review/resolutions", json=payload)
    assert response.status_code == 422
    assert response.json()["error"] == "validation_refused"
    assert "referenced_override_id" in response.json()["detail"]


def test_refer_to_override_with_pointer_is_staged(client: TestClient) -> None:
    payload = {
        **VALID_RESOLUTION,
        "resolution_action": "refer_to_override",
        "referenced_override_id": "O1",
    }
    response = client.post("/v1/review/resolutions", json=payload)
    assert response.status_code == 201


def test_acknowledge_with_a_pointer_is_refused(client: TestClient) -> None:
    payload = {**VALID_RESOLUTION, "referenced_override_id": "O1"}
    response = client.post("/v1/review/resolutions", json=payload)
    assert response.status_code == 422
    assert "must not carry" in response.json()["detail"]


def test_missing_required_field_is_refused(client: TestClient) -> None:
    payload = {k: v for k, v in VALID_RESOLUTION.items() if k != "actor"}
    response = client.post("/v1/review/resolutions", json=payload)
    assert response.status_code == 422


def test_duplicate_id_with_identical_payload_is_idempotent(client: TestClient) -> None:
    first = client.post("/v1/review/resolutions", json=VALID_RESOLUTION)
    second = client.post("/v1/review/resolutions", json=VALID_RESOLUTION)
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["request_id"] == second.json()["request_id"]


def test_duplicate_id_with_different_payload_is_refused(client: TestClient) -> None:
    client.post("/v1/review/resolutions", json=VALID_RESOLUTION)
    changed = {**VALID_RESOLUTION, "reason_code": "different_reason"}
    response = client.post("/v1/review/resolutions", json=changed)
    assert response.status_code == 409
    assert response.json()["error"] == "duplicate_key"


def test_id_already_committed_is_refused(client: TestClient, api_settings: Settings) -> None:
    seed_review_resolution_log(api_settings, [review_resolution_log_row(review_id="R1")])
    response = client.post("/v1/review/resolutions", json=VALID_RESOLUTION)
    assert response.status_code == 409


def test_resolution_log_read_reports_committed_status(
    client: TestClient, api_settings: Settings
) -> None:
    seed_review_resolution_log(api_settings, [review_resolution_log_row(review_id="R1")])
    response = client.get("/v1/review/resolutions", params=DEFAULT_SCOPE)
    assert response.status_code == 200
    row = response.json()["data"][0]
    assert row["status"] == "committed"
    assert row["review_id"] == "R1"


def test_original_provenance_unchanged_after_staging(
    client: TestClient, api_settings: Settings
) -> None:
    """Staging a new resolution never touches the committed log for a different id."""
    seed_review_resolution_log(api_settings, [review_resolution_log_row(review_id="R_EXISTING")])
    before = client.get("/v1/review/resolutions", params=DEFAULT_SCOPE).json()["data"]

    client.post("/v1/review/resolutions", json=VALID_RESOLUTION)

    after = client.get("/v1/review/resolutions", params=DEFAULT_SCOPE).json()["data"]
    assert before == after


def test_target_inspection_id_filter_keeps_only_that_establishments_resolutions(
    client: TestClient, api_settings: Settings
) -> None:
    seed_review_resolution_log(
        api_settings,
        [
            review_resolution_log_row(review_id="R1", target_inspection_id="T1"),
            review_resolution_log_row(review_id="R2", target_inspection_id="T2"),
        ],
    )
    response = client.get(
        "/v1/review/resolutions", params={**DEFAULT_SCOPE, "target_inspection_id": "T2"}
    )
    assert response.status_code == 200
    rows = response.json()["data"]
    assert [row["review_id"] for row in rows] == ["R2"]
