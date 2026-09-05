"""Read access to Component 11's artifacts. No attribution, no re-fitting, no SHAP computed here."""

from __future__ import annotations

import polars as pl

from sentinel.api.errors import RowNotFound
from sentinel.api.schemas.common import DecisionScope
from sentinel.api.schemas.explain import ExplanationCaseOut, ExplanationValueOut, SupportOut
from sentinel.api.services.artifacts import (
    apply_scope_filter,
    read_table,
    require_scope,
    resolve_latest,
)
from sentinel.calibration.definitions import Method
from sentinel.config import Settings

EXPLAIN_SCOPE = ("model_name", "fold_set", "fold_id")


def base_model_name_of(model_name: str) -> str:
    """Strip Component 9's calibration-method suffix, if the name carries one.

    ``docs/data_contracts/explanations.md`` section 0a documents this name mismatch explicitly:
    Component 9 names a calibrated model ``xgboost_platt``; Component 11's tables name the same
    model ``xgboost`` and never a calibrated name. A caller holding a Component 13/14 row (which
    carries the calibrated name) must resolve it to the base name before looking up explanation
    support -- looking it up under the calibrated name finds nothing, and a missing row looks
    exactly like a model this project cannot explain.
    """
    for method in Method:
        suffix = f"_{method}"
        if model_name.endswith(suffix):
            return model_name.removesuffix(suffix)
    return model_name


def get_support(settings: Settings) -> list[SupportOut]:
    path = resolve_latest(settings.explanations_processed_dir, prefix="explanation_support")
    frame = read_table(path).sort("model_name")
    return [SupportOut.model_validate(row) for row in frame.to_dicts()]


def is_explainable(settings: Settings, model_name: str) -> tuple[bool, str | None]:
    """Whether a model has any explanation support at all, and why not if it does not."""
    for row in get_support(settings):
        if row.model_name == model_name:
            return row.explanation_status == "supported", row.unsupported_reason
    return False, f"{model_name!r} is not a recognised model in the explanation support table."


def get_explanation(
    settings: Settings, target_inspection_id: str, scope: DecisionScope
) -> ExplanationCaseOut:
    require_scope(scope, required=EXPLAIN_SCOPE)
    assert scope.model_name is not None  # narrowed by require_scope

    supported, reason = is_explainable(settings, scope.model_name)
    if not supported:
        raise RowNotFound(f"{scope.model_name} is not explainable: {reason}")

    cases_path = resolve_latest(settings.explanations_processed_dir, prefix="explanation_cases")
    cases = apply_scope_filter(read_table(cases_path), scope)
    cases = cases.filter(pl.col("target_inspection_id") == target_inspection_id)
    if cases.height == 0:
        raise RowNotFound(
            f"{target_inspection_id!r} is not in the sampled subset explained for "
            f"{scope.model_name} on {scope.fold_set}/{scope.fold_id}. Only a sample of "
            "inspections is explained per fold; see explanation_cases.sample_strategy."
        )
    case_row = cases.row(0, named=True)

    values_path = resolve_latest(settings.explanations_processed_dir, prefix="explanation_values")
    values = apply_scope_filter(read_table(values_path), scope)
    values = values.filter(pl.col("target_inspection_id") == target_inspection_id)
    values = values.sort("feature_name")

    case_fields = {k: v for k, v in case_row.items() if k in ExplanationCaseOut.model_fields}
    case_fields["values"] = [ExplanationValueOut.model_validate(row) for row in values.to_dicts()]
    return ExplanationCaseOut.model_validate(case_fields)


__all__ = ["base_model_name_of", "get_explanation", "get_support", "is_explainable"]
