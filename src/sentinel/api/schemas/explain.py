"""Response shapes for Component 11's artifacts."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class ExplanationValueOut(BaseModel):
    feature_name: str
    original_feature_name: str
    derived_from: str
    feature_kind: str
    feature_value: float | None
    transformed_value: float | None
    shap_value: float
    output_space: str
    is_exact: bool


class ExplanationCaseOut(BaseModel):
    model_name: str
    model_version: str
    fold_set: str
    fold_id: str
    target_inspection_id: str
    output_space: str
    explanation_method: str
    is_exact: bool
    base_value: float
    prediction_value: float
    reconstruction_value: float
    additivity_holds: bool
    n_features: int
    base_score: float
    calibrated_probability: float | None
    base_model_trained_through: date | None = None
    sample_strategy: str
    values: list[ExplanationValueOut]


class SupportOut(BaseModel):
    model_name: str
    explanation_status: str
    explanation_method: str | None
    output_space: str | None
    is_exact: bool
    is_experimental: bool
    rationale: str
    unsupported_reason: str | None


__all__ = ["ExplanationCaseOut", "ExplanationValueOut", "SupportOut"]
