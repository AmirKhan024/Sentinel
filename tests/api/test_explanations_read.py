from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sentinel.api.services.explain_service import base_model_name_of
from sentinel.config import Settings
from tests.api.conftest import (
    explanation_case_row,
    explanation_support_row,
    explanation_value_row,
    seed_explanation_cases,
    seed_explanation_support,
    seed_explanation_values,
)

SCOPE = {"model_name": "lightgbm", "fold_set": "quarterly", "fold_id": "quarterly-2026Q1"}


@pytest.mark.parametrize(
    ("calibrated", "expected_base"),
    [
        ("lightgbm_platt", "lightgbm"),
        ("xgboost_platt", "xgboost"),
        ("neural_numeric_only_platt", "neural_numeric_only"),
        ("logistic_regression_isotonic", "logistic_regression"),
        # Already a base name: nothing to strip, returned unchanged.
        ("lightgbm", "lightgbm"),
        # A base name that happens to end in a real calibration-method word is not touched,
        # because no known suffix matches when there is no separating underscore.
        ("platt", "platt"),
    ],
)
def test_base_model_name_of_strips_the_calibration_suffix(
    calibrated: str, expected_base: str
) -> None:
    assert base_model_name_of(calibrated) == expected_base


def test_support_lists_every_model(client: TestClient, api_settings: Settings) -> None:
    seed_explanation_support(
        api_settings,
        [
            explanation_support_row(model_name="lightgbm"),
            explanation_support_row(
                model_name="xgboost_chain_embeddings",
                explanation_status="unsupported",
                unsupported_reason="experimental",
            ),
        ],
    )
    response = client.get("/v1/explanations/support")
    assert response.status_code == 200
    names = {row["model_name"] for row in response.json()}
    assert names == {"lightgbm", "xgboost_chain_embeddings"}


def test_supported_model_with_null_unsupported_reason_serializes(
    client: TestClient, api_settings: Settings
) -> None:
    """In real production data, unsupported_reason and explanation_method/output_space are
    genuinely null for a supported model -- not an empty string. Regression for a bug found by
    running the API against the real artifacts."""
    seed_explanation_support(
        api_settings,
        [
            explanation_support_row(
                unsupported_reason=None, explanation_method=None, output_space=None
            )
        ],
    )
    response = client.get("/v1/explanations/support")
    assert response.status_code == 200
    row = response.json()[0]
    assert row["unsupported_reason"] is None
    assert row["explanation_method"] is None


def test_unsupported_model_returns_404_with_reason(
    client: TestClient, api_settings: Settings
) -> None:
    seed_explanation_support(
        api_settings,
        [
            explanation_support_row(
                model_name="xgboost_chain_embeddings",
                explanation_status="unsupported",
                unsupported_reason="experimental and unsupported",
            )
        ],
    )
    scope = {**SCOPE, "model_name": "xgboost_chain_embeddings"}
    response = client.get("/v1/explanations/T1", params=scope)
    assert response.status_code == 404
    assert "not explainable" in response.json()["detail"]


def test_not_in_sampled_subset_is_distinguished_from_unsupported(
    client: TestClient, api_settings: Settings
) -> None:
    seed_explanation_support(api_settings, [explanation_support_row()])
    seed_explanation_cases(api_settings, [explanation_case_row(target_inspection_id="T1")])
    seed_explanation_values(api_settings, [explanation_value_row(target_inspection_id="T1")])
    response = client.get("/v1/explanations/T999", params=SCOPE)
    assert response.status_code == 404
    assert "sampled subset" in response.json()["detail"]


def test_explanation_found_carries_feature_values(
    client: TestClient, api_settings: Settings
) -> None:
    seed_explanation_support(api_settings, [explanation_support_row()])
    seed_explanation_cases(api_settings, [explanation_case_row()])
    seed_explanation_values(api_settings, [explanation_value_row()])
    response = client.get("/v1/explanations/T1", params=SCOPE)
    assert response.status_code == 200
    body = response.json()
    assert body["target_inspection_id"] == "T1"
    assert body["values"][0]["shap_value"] == 0.1
