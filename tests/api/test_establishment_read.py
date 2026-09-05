from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sentinel.config import Settings
from tests.api.conftest import (
    DEFAULT_SCOPE,
    explanation_case_row,
    explanation_support_row,
    explanation_value_row,
    feature_row,
    recommendation_row,
    schedule_row,
    seed_explanation_cases,
    seed_explanation_support,
    seed_explanation_values,
    seed_features,
    seed_recommendations,
    seed_schedule,
)


def test_unknown_establishment_is_404(client: TestClient, api_settings: Settings) -> None:
    seed_recommendations(api_settings, [recommendation_row()])
    response = client.get("/v1/establishments/does-not-exist", params=DEFAULT_SCOPE)
    assert response.status_code == 404


def test_ambiguous_establishment_history_requires_more_scope(
    client: TestClient, api_settings: Settings
) -> None:
    """Two inspection events for the same establishment in the same cell -> 422, not a guess."""
    rows = [
        recommendation_row(target_inspection_id="T1", establishment_id="E1"),
        recommendation_row(target_inspection_id="T2", establishment_id="E1"),
    ]
    seed_recommendations(api_settings, rows)
    response = client.get("/v1/establishments/E1", params=DEFAULT_SCOPE)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "ambiguous_scope"
    assert set(body["candidate_values"]) == {"T1", "T2"}


def test_establishment_bundle_composes_recommendation_and_schedule(
    client: TestClient, api_settings: Settings
) -> None:
    seed_recommendations(api_settings, [recommendation_row()])
    seed_schedule(api_settings, [schedule_row()])
    scope = {**DEFAULT_SCOPE, "schedule_config_id": "strict_priority__observed_calendar"}
    response = client.get("/v1/establishments/E1", params=scope)
    assert response.status_code == 200
    body = response.json()
    assert body["recommendation"]["target_inspection_id"] == "T1"
    assert body["schedule"]["schedule_status"] == "scheduled"
    # The recommendation reason and the schedule reason are never merged into one field.
    assert body["recommendation"]["decision_reason"] == "selected_by_risk_rank"
    assert body["schedule"]["schedule_reason"] == "placed_in_priority_order"


def test_establishment_bundle_without_schedule_scope_omits_schedule(
    client: TestClient, api_settings: Settings
) -> None:
    seed_recommendations(api_settings, [recommendation_row()])
    response = client.get("/v1/establishments/E1", params=DEFAULT_SCOPE)
    assert response.status_code == 200
    assert response.json()["schedule"] is None


def test_establishment_bundle_includes_explanation_when_sampled(
    client: TestClient, api_settings: Settings
) -> None:
    seed_recommendations(api_settings, [recommendation_row()])
    seed_explanation_support(api_settings, [explanation_support_row()])
    seed_explanation_cases(api_settings, [explanation_case_row()])
    seed_explanation_values(api_settings, [explanation_value_row()])
    response = client.get("/v1/establishments/E1", params=DEFAULT_SCOPE)
    body = response.json()
    assert body["explanation"] is not None
    assert body["explanation"]["values"][0]["feature_name"] == "prior_canvass_count_code_era"
    assert body["explanation_unavailable_reason"] is None


def test_establishment_bundle_reports_why_explanation_is_absent(
    client: TestClient, api_settings: Settings
) -> None:
    seed_recommendations(api_settings, [recommendation_row()])
    seed_explanation_support(api_settings, [explanation_support_row()])
    # No explanation_cases seeded at all -> falls through to ArtifactNotFound, surfaced as a
    # reason rather than a silently empty explanation field.
    response = client.get("/v1/establishments/E1", params=DEFAULT_SCOPE)
    body = response.json()
    assert body["explanation"] is None
    assert body["explanation_unavailable_reason"] is not None


def test_establishment_bundle_includes_history_factors_when_a_feature_table_exists(
    client: TestClient, api_settings: Settings
) -> None:
    """The concrete, human-legible reasons behind a score -- not a raw feature dump, and not
    invented: every value here is a column that already exists in Component 4's own table."""
    seed_recommendations(api_settings, [recommendation_row()])
    seed_features(api_settings, [feature_row(target_inspection_id="T1")])
    response = client.get("/v1/establishments/E1", params=DEFAULT_SCOPE)
    assert response.status_code == 200
    body = response.json()
    assert body["history_factors"] is not None
    assert body["history_factors"]["prior_canvass_count_code_era"] == 7
    assert body["history_factors"]["prior_canvass_priority_count"] == 6
    assert body["history_factors"]["prior_canvass_priority_rate"] == pytest.approx(0.857143)
    assert body["history_factors"]["days_since_last_canvass"] == 345
    assert body["history_factors_unavailable_reason"] is None


def test_establishment_bundle_reports_why_history_factors_are_absent(
    client: TestClient, api_settings: Settings
) -> None:
    """No feature table built at all -> a stated reason, never a silently empty section."""
    seed_recommendations(api_settings, [recommendation_row()])
    response = client.get("/v1/establishments/E1", params=DEFAULT_SCOPE)
    assert response.status_code == 200
    body = response.json()
    assert body["history_factors"] is None
    assert body["history_factors_unavailable_reason"] is not None


def test_establishment_bundle_reports_why_history_factors_are_absent_for_this_row(
    client: TestClient, api_settings: Settings
) -> None:
    """A feature table exists, but not for this establishment's target_inspection_id."""
    seed_recommendations(api_settings, [recommendation_row()])
    seed_features(api_settings, [feature_row(target_inspection_id="T999")])
    response = client.get("/v1/establishments/E1", params=DEFAULT_SCOPE)
    assert response.status_code == 200
    body = response.json()
    assert body["history_factors"] is None
    assert body["history_factors_unavailable_reason"] is not None


def test_establishment_bundle_resolves_the_calibrated_model_name_to_the_base_name(
    client: TestClient, api_settings: Settings
) -> None:
    """Regression: found by running the API against real production artifacts.

    Component 13/14 carry Component 9's *calibrated* model name (e.g. "lightgbm_platt");
    Component 11's tables carry the *base* name ("lightgbm") and never a calibrated one
    (docs/data_contracts/explanations.md 0a). Passing the calibrated name straight through to
    the explanation lookup silently reported every real establishment as "not explainable,"
    even for models Component 11 genuinely supports. `recommendation_row()` uses the real
    Component 13 convention ("lightgbm_platt"); this test proves the bundle still finds the
    explanation despite the naming mismatch.
    """
    seed_recommendations(api_settings, [recommendation_row(model_name="lightgbm_platt")])
    seed_explanation_support(api_settings, [explanation_support_row(model_name="lightgbm")])
    seed_explanation_cases(api_settings, [explanation_case_row(model_name="lightgbm")])
    seed_explanation_values(api_settings, [explanation_value_row(model_name="lightgbm")])
    response = client.get("/v1/establishments/E1", params=DEFAULT_SCOPE)
    assert response.status_code == 200
    body = response.json()
    assert body["explanation"] is not None
    assert body["explanation"]["model_name"] == "lightgbm"
    assert body["explanation_unavailable_reason"] is None
