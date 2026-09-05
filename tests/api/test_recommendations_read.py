from __future__ import annotations

from fastapi.testclient import TestClient

from sentinel.config import Settings
from tests.api.conftest import DEFAULT_SCOPE, recommendation_row, seed_recommendations


def test_missing_artifact_returns_404_not_500(client: TestClient) -> None:
    response = client.get("/v1/recommendations", params=DEFAULT_SCOPE)
    assert response.status_code == 404
    assert response.json()["error"] == "artifact_not_found"


def test_ambiguous_scope_returns_422_with_missing_fields(
    client: TestClient, api_settings: Settings
) -> None:
    seed_recommendations(api_settings, [recommendation_row()])
    response = client.get("/v1/recommendations", params={"policy_id": "pure_risk"})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "ambiguous_scope"
    assert set(body["missing_scope_fields"]) == {"fold_set", "fold_id", "k_name"}


def test_list_recommendations_deterministic_order(
    client: TestClient, api_settings: Settings
) -> None:
    rows = [
        recommendation_row(target_inspection_id="T2", final_policy_rank=2),
        recommendation_row(target_inspection_id="T1", final_policy_rank=1),
    ]
    seed_recommendations(api_settings, rows)
    response = client.get("/v1/recommendations", params=DEFAULT_SCOPE)
    assert response.status_code == 200
    body = response.json()
    assert [row["target_inspection_id"] for row in body["data"]] == ["T1", "T2"]
    assert body["page"] == {"offset": 0, "limit": 50, "total": 2}

    # A repeated, identical request returns byte-identical JSON.
    again = client.get("/v1/recommendations", params=DEFAULT_SCOPE)
    assert again.json()["data"] == body["data"]


def test_pagination_bounds_and_caps_limit(client: TestClient, api_settings: Settings) -> None:
    rows = [
        recommendation_row(target_inspection_id=f"T{i}", final_policy_rank=i) for i in range(1, 4)
    ]
    seed_recommendations(api_settings, rows)
    response = client.get("/v1/recommendations", params={**DEFAULT_SCOPE, "offset": 1, "limit": 1})
    body = response.json()
    assert body["page"] == {"offset": 1, "limit": 1, "total": 3}
    assert body["data"][0]["target_inspection_id"] == "T2"


def test_filter_by_establishment_and_selected(client: TestClient, api_settings: Settings) -> None:
    rows = [
        recommendation_row(target_inspection_id="T1", establishment_id="E1", is_selected=True),
        recommendation_row(target_inspection_id="T2", establishment_id="E2", is_selected=False),
    ]
    seed_recommendations(api_settings, rows)
    response = client.get("/v1/recommendations", params={**DEFAULT_SCOPE, "is_selected": "true"})
    body = response.json()
    assert [row["target_inspection_id"] for row in body["data"]] == ["T1"]


def test_not_selected_row_with_null_final_policy_rank_serializes(
    client: TestClient, api_settings: Settings
) -> None:
    """final_policy_rank is null for the vast majority of real rows: every non-selected one."""
    seed_recommendations(
        api_settings,
        [
            recommendation_row(
                final_policy_rank=None,
                is_selected=False,
                decision_mechanism="not_selected",
                decision_reason="not_selected_capacity_exhausted",
            )
        ],
    )
    response = client.get("/v1/recommendations", params=DEFAULT_SCOPE)
    assert response.status_code == 200
    assert response.json()["data"][0]["final_policy_rank"] is None


def test_get_single_recommendation_unknown_id_is_404(
    client: TestClient, api_settings: Settings
) -> None:
    seed_recommendations(api_settings, [recommendation_row()])
    response = client.get("/v1/recommendations/does-not-exist", params=DEFAULT_SCOPE)
    assert response.status_code == 404
    assert response.json()["error"] == "row_not_found"


def test_get_single_recommendation_found(client: TestClient, api_settings: Settings) -> None:
    seed_recommendations(api_settings, [recommendation_row()])
    response = client.get("/v1/recommendations/T1", params=DEFAULT_SCOPE)
    assert response.status_code == 200
    assert response.json()["establishment_id"] == "E1"


def test_immutable_write_verbs_are_structurally_absent(client: TestClient) -> None:
    """No PATCH/PUT route exists on any resource: immutability is structural, not policed."""
    assert client.patch("/v1/schedule").status_code == 405
    assert client.put("/v1/recommendations/T1").status_code == 405
