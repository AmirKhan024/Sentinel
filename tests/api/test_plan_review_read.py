from __future__ import annotations

from fastapi.testclient import TestClient

from sentinel.config import Settings
from tests.api.conftest import (
    plan_decision_log_row,
    plan_review_row,
    seed_plan_decision_log,
    seed_plan_review,
)


def test_no_artifact_yet_is_404(client: TestClient) -> None:
    response = client.get("/v1/plan-review/summary")
    assert response.status_code == 404
    assert response.json()["error"] == "artifact_not_found"


def test_plan_summary(client: TestClient, api_settings: Settings) -> None:
    seed_plan_review(
        api_settings,
        [
            plan_review_row(target_inspection_id="T1", supervisor_decision_action="keep_selected"),
            plan_review_row(target_inspection_id="T2", establishment_id="E2"),
        ],
    )
    response = client.get("/v1/plan-review/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["selected_inspection_workload"] == 2
    assert body["decisions_recorded"] == 1
    assert body["approval_status"] == "under_supervisor_review"


def test_plan_summary_all_decided_is_adjusted(client: TestClient, api_settings: Settings) -> None:
    seed_plan_review(
        api_settings,
        [
            plan_review_row(target_inspection_id="T1", supervisor_decision_action="keep_selected"),
        ],
    )
    response = client.get("/v1/plan-review/summary")
    assert response.json()["approval_status"] == "adjusted"


def test_list_plan_rows(client: TestClient, api_settings: Settings) -> None:
    seed_plan_review(
        api_settings,
        [
            plan_review_row(target_inspection_id="T1", suggested_order_in_block=2),
            plan_review_row(target_inspection_id="T2", establishment_id="E2", suggested_order_in_block=1),
        ],
    )
    response = client.get("/v1/plan-review/rows")
    assert response.status_code == 200
    body = response.json()
    assert body["page"]["total"] == 2
    # Default sort is suggested_order_in_block ascending.
    assert [r["target_inspection_id"] for r in body["data"]] == ["T2", "T1"]


def test_get_one_plan_row(client: TestClient, api_settings: Settings) -> None:
    seed_plan_review(api_settings, [plan_review_row(target_inspection_id="T1")])
    response = client.get("/v1/plan-review/rows/T1")
    assert response.status_code == 200
    assert response.json()["target_inspection_id"] == "T1"
    # Sentinel's own recommendation is exposed alongside the (empty) decision.
    assert response.json()["policy_rank"] == 1
    assert response.json()["supervisor_decision_action"] is None


def test_missing_row_is_404(client: TestClient, api_settings: Settings) -> None:
    seed_plan_review(api_settings, [plan_review_row(target_inspection_id="T1")])
    response = client.get("/v1/plan-review/rows/DOES_NOT_EXIST")
    assert response.status_code == 404


def test_list_work_blocks_aggregates_by_block(client: TestClient, api_settings: Settings) -> None:
    seed_plan_review(
        api_settings,
        [
            plan_review_row(target_inspection_id="T1", work_block_id="area_1", policy_rank=1),
            plan_review_row(
                target_inspection_id="T2",
                establishment_id="E2",
                work_block_id="area_1",
                policy_rank=5,
            ),
            plan_review_row(
                target_inspection_id="T3",
                establishment_id="E3",
                work_block_id="unmapped",
                work_block_label="Unmapped / Location unavailable",
                location_status="location_unavailable",
                policy_rank=9,
            ),
        ],
    )
    response = client.get("/v1/plan-review/work-blocks")
    assert response.status_code == 200
    blocks = response.json()
    area_1 = next(b for b in blocks if b["work_block_id"] == "area_1")
    assert area_1["size"] == 2
    assert area_1["highest_sentinel_rank"] == 1
    assert area_1["rank_range"] == [1, 5]
    unmapped = next(b for b in blocks if b["work_block_id"] == "unmapped")
    assert unmapped["is_unmapped"] is True
    assert unmapped["size"] == 1


def test_list_decisions(client: TestClient, api_settings: Settings) -> None:
    seed_plan_decision_log(
        api_settings, [plan_decision_log_row(decision_id="DEC-0001"), plan_decision_log_row(decision_id="DEC-0002")]
    )
    response = client.get("/v1/plan-review/decisions")
    assert response.status_code == 200
    body = response.json()
    assert body["page"]["total"] == 2
    assert body["data"][0]["status"] == "committed"


def test_planning_date_filters_to_matching_artifact_only(
    client: TestClient, api_settings: Settings
) -> None:
    seed_plan_review(api_settings, [plan_review_row(target_inspection_id="T1")])
    response = client.get("/v1/plan-review/summary", params={"planning_date": "2026-08-28"})
    assert response.status_code == 200
    response_missing = client.get(
        "/v1/plan-review/summary", params={"planning_date": "2099-01-01"}
    )
    assert response_missing.status_code == 404
