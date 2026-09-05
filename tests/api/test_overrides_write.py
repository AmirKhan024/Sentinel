from __future__ import annotations

from fastapi.testclient import TestClient

from sentinel.config import Settings
from tests.api.conftest import DEFAULT_SCOPE, override_log_row, seed_override_log

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


def test_valid_override_is_staged_pending(client: TestClient) -> None:
    response = client.post("/v1/policy/overrides", json=VALID_OVERRIDE)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "pending"
    assert body["natural_id"] == "O1"
    assert body["kind"] == "override"

    staged = client.get("/v1/staged-requests", params={"kind": "override"})
    assert staged.status_code == 200
    assert staged.json()[0]["natural_id"] == "O1"


def test_corrupted_action_is_refused_with_parser_message(client: TestClient) -> None:
    payload = {**VALID_OVERRIDE, "action": "delete_forever"}
    response = client.post("/v1/policy/overrides", json=payload)
    assert response.status_code == 422
    assert response.json()["error"] == "validation_refused"
    assert "unknown action" in response.json()["detail"]


def test_missing_required_field_is_refused(client: TestClient) -> None:
    payload = {k: v for k, v in VALID_OVERRIDE.items() if k != "actor"}
    response = client.post("/v1/policy/overrides", json=payload)
    # Pydantic's own field validation refuses this before it ever reaches the parser.
    assert response.status_code == 422


def test_duplicate_id_with_identical_payload_is_idempotent(client: TestClient) -> None:
    first = client.post("/v1/policy/overrides", json=VALID_OVERRIDE)
    second = client.post("/v1/policy/overrides", json=VALID_OVERRIDE)
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["request_id"] == second.json()["request_id"]


def test_duplicate_id_with_different_payload_is_refused(client: TestClient) -> None:
    client.post("/v1/policy/overrides", json=VALID_OVERRIDE)
    changed = {**VALID_OVERRIDE, "reason_code": "different_reason"}
    response = client.post("/v1/policy/overrides", json=changed)
    assert response.status_code == 409
    assert response.json()["error"] == "duplicate_key"


def test_id_already_committed_is_refused(client: TestClient, api_settings: Settings) -> None:
    seed_override_log(api_settings, [override_log_row(override_id="O1")])
    response = client.post("/v1/policy/overrides", json=VALID_OVERRIDE)
    assert response.status_code == 409


def test_override_log_read_reports_committed_status(
    client: TestClient, api_settings: Settings
) -> None:
    seed_override_log(api_settings, [override_log_row(override_id="O1")])
    response = client.get("/v1/policy/overrides", params=DEFAULT_SCOPE)
    assert response.status_code == 200
    row = response.json()["data"][0]
    assert row["status"] == "committed"
    assert row["override_id"] == "O1"


def test_original_provenance_unchanged_after_staging(
    client: TestClient, api_settings: Settings
) -> None:
    """Staging a new override never touches the committed log for a different id."""
    seed_override_log(api_settings, [override_log_row(override_id="O_EXISTING")])
    before = client.get("/v1/policy/overrides", params=DEFAULT_SCOPE).json()["data"]

    client.post("/v1/policy/overrides", json=VALID_OVERRIDE)

    after = client.get("/v1/policy/overrides", params=DEFAULT_SCOPE).json()["data"]
    assert before == after


def test_target_inspection_id_filter_keeps_only_that_establishments_overrides(
    client: TestClient, api_settings: Settings
) -> None:
    """The filter a per-establishment decision-history view needs -- proves it narrows the log
    rather than just being accepted and ignored."""
    seed_override_log(
        api_settings,
        [
            override_log_row(override_id="O1", target_inspection_id="T1"),
            override_log_row(override_id="O2", target_inspection_id="T2"),
        ],
    )
    response = client.get(
        "/v1/policy/overrides", params={**DEFAULT_SCOPE, "target_inspection_id": "T1"}
    )
    assert response.status_code == 200
    rows = response.json()["data"]
    assert [row["override_id"] for row in rows] == ["O1"]
