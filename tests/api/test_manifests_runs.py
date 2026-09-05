from __future__ import annotations

from fastapi.testclient import TestClient

from sentinel.config import Settings
from tests.api.conftest import (
    recommendation_row,
    review_queue_row,
    seed_operational_selection,
    seed_recommendations,
    seed_review_queue,
)


def test_healthz_never_touches_artifacts(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_component_manifest_is_404(client: TestClient) -> None:
    response = client.get("/v1/manifests/routing")
    assert response.status_code == 404
    assert response.json()["error"] == "unknown_component"


def test_manifest_missing_run_is_404(client: TestClient) -> None:
    response = client.get("/v1/manifests/policy")
    assert response.status_code == 404
    assert response.json()["error"] == "artifact_not_found"


def test_manifest_reads_built_at(client: TestClient, api_settings: Settings) -> None:
    seed_recommendations(api_settings, [recommendation_row()])
    response = client.get("/v1/manifests/policy")
    assert response.status_code == 200
    assert response.json()["built_at"] == "2026-01-01T00:00:00+00:00"


def test_list_runs_reports_the_seeded_run(client: TestClient, api_settings: Settings) -> None:
    seed_recommendations(api_settings, [recommendation_row()])
    response = client.get("/v1/runs", params={"component": "policy"})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["component"] == "policy"


def test_list_runs_unknown_component_is_404(client: TestClient) -> None:
    response = client.get("/v1/runs", params={"component": "routing"})
    assert response.status_code == 404


def test_review_manifest_reads_built_at(client: TestClient, api_settings: Settings) -> None:
    seed_review_queue(api_settings, [review_queue_row()])
    response = client.get("/v1/manifests/review")
    assert response.status_code == 200
    assert response.json()["built_at"] == "2026-01-01T00:00:00+00:00"


def test_list_runs_reports_the_seeded_review_run(
    client: TestClient, api_settings: Settings
) -> None:
    seed_review_queue(api_settings, [review_queue_row()])
    response = client.get("/v1/runs", params={"component": "review"})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["component"] == "review"


def test_operational_selection_manifest_reads_the_real_coverage_counts(
    client: TestClient, api_settings: Settings
) -> None:
    seed_operational_selection(
        api_settings,
        ranked_candidate_count=35859,
        selectable_candidate_count=35859,
        selected_count=30,
    )
    response = client.get("/v1/manifests/operational_selection")
    assert response.status_code == 200
    body = response.json()
    assert body["ranked_candidate_count"] == 35859
    assert body["selectable_candidate_count"] == 35859
    assert body["selected_count"] == 30


def test_list_runs_reports_the_seeded_operational_selection_run(
    client: TestClient, api_settings: Settings
) -> None:
    seed_operational_selection(api_settings)
    response = client.get("/v1/runs", params={"component": "operational_selection"})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["component"] == "operational_selection"
