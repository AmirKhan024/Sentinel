from __future__ import annotations

from fastapi.testclient import TestClient

from sentinel.config import Settings


def _payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "decision_id": "DEC-0001",
        "planning_date": "2026-08-28",
        "target_inspection_id": "T1",
        "decision_action": "keep_selected",
        "reason_code": "no_concern",
        "actor": "jsmith",
        "decided_at": "2026-01-02T00:00:00Z",
    }
    base.update(overrides)
    return base


def test_a_decision_is_staged_not_applied(client: TestClient, api_settings: Settings) -> None:
    response = client.post("/v1/plan-review/decisions", json=_payload())
    assert response.status_code == 201
    body = response.json()
    assert body["kind"] == "plan_decision"
    assert body["natural_id"] == "DEC-0001"
    assert body["status"] == "pending"
    # Never applied immediately: no supervisor_plan_review artifact was written.
    assert not api_settings.plan_review_processed_dir.exists() or not list(
        api_settings.plan_review_processed_dir.glob("supervisor_plan_review_*.parquet")
    )


def test_reposting_the_identical_payload_is_idempotent(
    client: TestClient, api_settings: Settings
) -> None:
    first = client.post("/v1/plan-review/decisions", json=_payload())
    second = client.post("/v1/plan-review/decisions", json=_payload())
    assert first.json()["request_id"] == second.json()["request_id"]


def test_reposting_a_different_payload_with_the_same_id_is_refused(
    client: TestClient, api_settings: Settings
) -> None:
    client.post("/v1/plan-review/decisions", json=_payload())
    conflicting = client.post(
        "/v1/plan-review/decisions", json=_payload(reason_code="different_reason")
    )
    assert conflicting.status_code == 409
    assert conflicting.json()["error"] == "duplicate_key"


def test_a_malformed_decision_is_refused_with_422(
    client: TestClient, api_settings: Settings
) -> None:
    response = client.post(
        "/v1/plan-review/decisions", json=_payload(decision_action="not_a_real_action")
    )
    assert response.status_code == 422


def test_move_to_later_workday_without_revised_date_is_refused(
    client: TestClient, api_settings: Settings
) -> None:
    response = client.post(
        "/v1/plan-review/decisions", json=_payload(decision_action="move_to_later_workday")
    )
    assert response.status_code == 422


def test_move_to_later_workday_with_revised_date_is_accepted(
    client: TestClient, api_settings: Settings
) -> None:
    response = client.post(
        "/v1/plan-review/decisions",
        json=_payload(decision_action="move_to_later_workday", revised_planned_date="2026-09-04"),
    )
    assert response.status_code == 201


def test_extra_field_is_refused_by_the_schema(client: TestClient, api_settings: Settings) -> None:
    response = client.post("/v1/plan-review/decisions", json=_payload(unexpected_field="x"))
    assert response.status_code == 422
