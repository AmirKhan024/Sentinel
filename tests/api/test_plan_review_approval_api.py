from __future__ import annotations

from fastapi.testclient import TestClient

from sentinel.config import Settings
from tests.api.conftest import plan_review_row, seed_approved_plan, seed_plan_review


def _decision_payload(**overrides: object) -> dict[str, object]:
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


def _approval_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "approval_id": "APPR-0001",
        "planning_date": "2026-08-28",
        "approved_by": "jsmith",
        "approved_at": "2026-01-02T00:00:00Z",
    }
    base.update(overrides)
    return base


# --- new PlanDecisionAction: adjust_operational_priority -------------------------------


def test_adjust_operational_priority_without_revised_value_is_refused(
    client: TestClient, api_settings: Settings
) -> None:
    response = client.post(
        "/v1/plan-review/decisions",
        json=_decision_payload(decision_action="adjust_operational_priority"),
    )
    assert response.status_code == 422


def test_adjust_operational_priority_with_revised_value_is_accepted(
    client: TestClient, api_settings: Settings
) -> None:
    response = client.post(
        "/v1/plan-review/decisions",
        json=_decision_payload(
            decision_action="adjust_operational_priority", revised_operational_priority=1
        ),
    )
    assert response.status_code == 201


# --- GET /v1/plan-review/approval -------------------------------------------------------


def test_no_approval_yet_is_404(client: TestClient, api_settings: Settings) -> None:
    seed_plan_review(api_settings, [plan_review_row(target_inspection_id="T1")])
    response = client.get("/v1/plan-review/approval", params={"planning_date": "2026-08-28"})
    assert response.status_code == 404
    assert response.json()["error"] == "artifact_not_found"


def test_committed_approval_is_returned(client: TestClient, api_settings: Settings) -> None:
    seed_approved_plan(
        api_settings,
        approval_id="APPR-XYZ",
        approved_by="supervisor.demo",
        final_active_count=29,
        final_deferred_count=1,
    )
    response = client.get("/v1/plan-review/approval", params={"planning_date": "2026-08-28"})
    assert response.status_code == 200
    body = response.json()
    assert body["approval_id"] == "APPR-XYZ"
    assert body["approved_by"] == "supervisor.demo"
    assert body["final_active_count"] == 29
    assert body["final_deferred_count"] == 1


def test_plan_summary_reflects_a_committed_approval(
    client: TestClient, api_settings: Settings
) -> None:
    seed_plan_review(api_settings, [plan_review_row(target_inspection_id="T1")])
    seed_approved_plan(api_settings)
    response = client.get("/v1/plan-review/summary", params={"planning_date": "2026-08-28"})
    assert response.status_code == 200
    assert response.json()["approval_status"] == "approved"


# --- POST /v1/plan-review/approve -------------------------------------------------------


def test_an_approval_is_staged_not_applied(client: TestClient, api_settings: Settings) -> None:
    response = client.post("/v1/plan-review/approve", json=_approval_payload())
    assert response.status_code == 201
    body = response.json()
    assert body["kind"] == "plan_approval"
    assert body["natural_id"] == "APPR-0001"
    assert body["status"] == "pending"
    # Never applied immediately: no approved_operational_plan artifact was written.
    assert not api_settings.plan_review_processed_dir.exists() or not list(
        api_settings.plan_review_processed_dir.glob("approved_operational_plan_*.parquet")
    )


def test_reposting_the_identical_approval_payload_is_idempotent(
    client: TestClient, api_settings: Settings
) -> None:
    first = client.post("/v1/plan-review/approve", json=_approval_payload())
    second = client.post("/v1/plan-review/approve", json=_approval_payload())
    assert first.json()["request_id"] == second.json()["request_id"]


def test_an_approval_missing_a_required_field_is_refused(
    client: TestClient, api_settings: Settings
) -> None:
    response = client.post("/v1/plan-review/approve", json=_approval_payload(approved_by=""))
    assert response.status_code == 422


def test_an_approval_with_an_extra_field_is_refused_by_the_schema(
    client: TestClient, api_settings: Settings
) -> None:
    response = client.post(
        "/v1/plan-review/approve", json=_approval_payload(unexpected_field="x")
    )
    assert response.status_code == 422
