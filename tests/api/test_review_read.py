from __future__ import annotations

from fastapi.testclient import TestClient

from sentinel.config import Settings
from tests.api.conftest import DEFAULT_SCOPE, review_queue_row, seed_review_queue


def test_scope_is_required(client: TestClient) -> None:
    response = client.get("/v1/review/queue")
    assert response.status_code == 422
    assert response.json()["error"] == "ambiguous_scope"


def test_list_review_queue(client: TestClient, api_settings: Settings) -> None:
    seed_review_queue(api_settings, [review_queue_row(target_inspection_id="T1")])
    response = client.get("/v1/review/queue", params=DEFAULT_SCOPE)
    assert response.status_code == 200
    body = response.json()
    assert body["data"][0]["target_inspection_id"] == "T1"
    assert body["data"][0]["status"] == "committed"


def test_get_one_review_case(client: TestClient, api_settings: Settings) -> None:
    seed_review_queue(api_settings, [review_queue_row(target_inspection_id="T1")])
    response = client.get("/v1/review/queue/T1", params=DEFAULT_SCOPE)
    assert response.status_code == 200
    assert response.json()["target_inspection_id"] == "T1"


def test_missing_case_is_404(client: TestClient, api_settings: Settings) -> None:
    seed_review_queue(api_settings, [review_queue_row(target_inspection_id="T1")])
    response = client.get("/v1/review/queue/DOES_NOT_EXIST", params=DEFAULT_SCOPE)
    assert response.status_code == 404


def test_no_artifact_yet_is_404(client: TestClient) -> None:
    response = client.get("/v1/review/queue", params=DEFAULT_SCOPE)
    assert response.status_code == 404
    assert response.json()["error"] == "artifact_not_found"


def test_trigger_filter_keeps_only_matching_cases(
    client: TestClient, api_settings: Settings
) -> None:
    """A literal substring match on the existing trigger_reasons column, not a new
    classification -- proves a case with only the execution-gap trigger is excluded when asked
    for policy_warning_present, and included when asked for its own trigger."""
    seed_review_queue(
        api_settings,
        [
            review_queue_row(target_inspection_id="T1", trigger_reasons="policy_warning_present"),
            review_queue_row(
                target_inspection_id="T2", trigger_reasons="no_execution_record_on_scheduled_row"
            ),
            review_queue_row(
                target_inspection_id="T3",
                trigger_reasons="no_execution_record_on_scheduled_row|policy_warning_present",
            ),
        ],
    )
    response = client.get(
        "/v1/review/queue", params={**DEFAULT_SCOPE, "trigger": "policy_warning_present"}
    )
    assert response.status_code == 200
    ids = {row["target_inspection_id"] for row in response.json()["data"]}
    assert ids == {"T1", "T3"}

    response = client.get(
        "/v1/review/queue",
        params={**DEFAULT_SCOPE, "trigger": "no_execution_record_on_scheduled_row"},
    )
    ids = {row["target_inspection_id"] for row in response.json()["data"]}
    assert ids == {"T2", "T3"}


def test_no_trigger_filter_returns_every_case(client: TestClient, api_settings: Settings) -> None:
    seed_review_queue(
        api_settings,
        [
            review_queue_row(target_inspection_id="T1", trigger_reasons="policy_warning_present"),
            review_queue_row(
                target_inspection_id="T2", trigger_reasons="no_execution_record_on_scheduled_row"
            ),
        ],
    )
    response = client.get("/v1/review/queue", params=DEFAULT_SCOPE)
    assert response.json()["page"]["total"] == 2
