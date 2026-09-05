"""Parquet schemas and deterministic writing for Component 11.

Column order is part of the data contract; changing it is a contract change. See
``docs/data_contracts/explanations.md``.

Six tables in one new processed layer (ADR 0028). The split follows the grain of the
question each answers: ``explanation_values`` is one row per (model, fold, inspection,
feature) and is the only large table; everything else is a summary a reader can open
without a 1.3-million-row scan.

**The artifact must be readable without loading a model.** That rule drives one deliberate
denormalisation: ``base_value``, ``prediction_value`` and ``trained_through`` are repeated
on every value row even though they are constant within a case. A consumer asking "does
this row's decomposition add up, and what horizon produced it?" can then answer from the
one table, rather than needing a join it might get wrong. zstd stores thirty identical
floats for almost nothing, and the alternative -- a reader who joins on the wrong key and
publishes a wrong reconstruction -- costs considerably more.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


#: One row per (model, fold, explained inspection, feature). The component's long grain.
VALUES_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "model_version": pl.Utf8(),
    "family": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "target_inspection_id": pl.Utf8(),
    # The transformed representation the estimator actually indexes...
    "feature_name": pl.Utf8(),
    # ...and the Component 4 column it came from. Equal for a plain feature; for a family
    # indicator the original is the null-rule family and ``derived_from`` lists its members,
    # because an indicator does not belong to any single column and saying it does would be
    # the false aggregation this component exists not to commit.
    "original_feature_name": pl.Utf8(),
    "derived_from": pl.Utf8(),
    "feature_kind": pl.Utf8(),
    # The value a human can read: the raw pre-transform number. Null where the source was
    # NULL -- which for the tree models is a real observation the split routed on, not a gap.
    "feature_value": pl.Float64(),
    # The value the estimator saw. Imputed and standardised for the linear and neural
    # models; identical to ``feature_value`` for the boosters, which fit no preprocessing.
    "transformed_value": pl.Float64(),
    "shap_value": pl.Float64(),
    "output_space": pl.Utf8(),
    "explanation_method": pl.Utf8(),
    "is_exact": pl.Boolean(),
    "base_value": pl.Float64(),
    "prediction_value": pl.Float64(),
    "trained_through": pl.Date(),
    "explain_definition_version": pl.Utf8(),
}

#: One row per explained prediction. Where additivity, provenance and the calibration link
#: live, so the long table stays narrow enough to scan.
CASES_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "model_version": pl.Utf8(),
    "family": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "target_inspection_id": pl.Utf8(),
    "output_space": pl.Utf8(),
    "explanation_method": pl.Utf8(),
    "is_exact": pl.Boolean(),
    "base_value": pl.Float64(),
    "prediction_value": pl.Float64(),
    "reconstruction_value": pl.Float64(),
    "reconstruction_residual": pl.Float64(),
    "additivity_tolerance": pl.Float64(),
    "additivity_holds": pl.Boolean(),
    "n_features": pl.Int64(),
    "positive_contribution_sum": pl.Float64(),
    "negative_contribution_sum": pl.Float64(),
    # The committed Component 6/7/8 probability for this row, carried so the attribution
    # can be tied to the artifact it explains without a join.
    "base_score": pl.Float64(),
    "base_score_reproduced": pl.Boolean(),
    # Component 9's output, where one exists. Null rather than absent when the calibrated
    # artifact was not supplied, so the column's meaning never depends on the run's flags.
    "calibrated_probability": pl.Float64(),
    "calibration_method": pl.Utf8(),
    # Three horizons, never one. ``trained_through`` on this table is the *base model's*,
    # which is what the attribution decomposes; the calibrator read a later window, and the
    # prediction was operationally available later still.
    "base_model_trained_through": pl.Date(),
    "calibrator_fitted_through": pl.Date(),
    "prediction_available_from": pl.Date(),
    "sample_strategy": pl.Utf8(),
    "sample_size": pl.Int64(),
    "sampling_seed": pl.Int64(),
    "sampling_population": pl.Utf8(),
    "population_rows": pl.Int64(),
    "background_strategy": pl.Utf8(),
    "background_size": pl.Int64(),
    "background_seed": pl.Int64(),
    "background_max_date": pl.Date(),
    "permutation_rounds": pl.Int64(),
    "explain_definition_version": pl.Utf8(),
}

#: Per-fold and per-fold-set mean absolute attribution. ``scope`` says which, so a consumer
#: filters on a declared column instead of testing ``fold_id`` for null.
#:
#: **A quarterly aggregate row never includes covid_shift.** The two fold sets are separate
#: values of ``fold_set`` and are aggregated separately, which is the same protection
#: Component 5 built into its metrics table and for the same reason: one regime-shift fold
#: averaged into seventeen ordinary ones would move the headline and leave no trace.
IMPORTANCE_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "scope": pl.Utf8(),
    "feature_name": pl.Utf8(),
    "original_feature_name": pl.Utf8(),
    "mean_abs_shap": pl.Float64(),
    "mean_shap": pl.Float64(),
    "rank": pl.Int64(),
    "sd_abs_shap": pl.Float64(),
    "mean_rank": pl.Float64(),
    "sd_rank": pl.Float64(),
    "best_rank": pl.Int64(),
    "worst_rank": pl.Int64(),
    "folds": pl.Int64(),
    "rows": pl.Int64(),
}

#: How far one model's importance ranking moved between two folds.
STABILITY_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "comparison": pl.Utf8(),
    "from_fold_id": pl.Utf8(),
    "to_fold_id": pl.Utf8(),
    "spearman_rho": pl.Float64(),
    "top_k": pl.Int64(),
    "top_k_jaccard": pl.Float64(),
    "features": pl.Int64(),
}

#: Per-feature rank travel across a fold set: the explanation-drift table.
DRIFT_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "feature_name": pl.Utf8(),
    "original_feature_name": pl.Utf8(),
    "first_fold_id": pl.Utf8(),
    "last_fold_id": pl.Utf8(),
    "first_rank": pl.Int64(),
    "last_rank": pl.Int64(),
    "best_rank": pl.Int64(),
    "worst_rank": pl.Int64(),
    "rank_range": pl.Int64(),
    "mean_abs_shap": pl.Float64(),
    "sd_abs_shap": pl.Float64(),
    "coefficient_of_variation": pl.Float64(),
    "materially_changed": pl.Boolean(),
}

#: The deterministic high/medium/low local cases the report draws.
CASES_SELECTED_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "tier": pl.Utf8(),
    "quantile": pl.Float64(),
    "target_inspection_id": pl.Utf8(),
    "base_value": pl.Float64(),
    "prediction_value": pl.Float64(),
    "base_score": pl.Float64(),
    "calibrated_probability": pl.Float64(),
    "calibration_method": pl.Utf8(),
    "output_space": pl.Utf8(),
    "method": pl.Utf8(),
    "is_exact": pl.Boolean(),
}

#: The support matrix, machine-readable. An unsupported model appears here **and nowhere
#: else**: it has no values, no cases and no importance rows, so a consumer that joins on
#: this table gets nulls rather than zeros, and a zero would have read as "this model used
#: no features".
SUPPORT_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "model_version": pl.Utf8(),
    "family": pl.Utf8(),
    "component": pl.Int64(),
    "source_slug": pl.Utf8(),
    "explanation_status": pl.Utf8(),
    "explanation_method": pl.Utf8(),
    "output_space": pl.Utf8(),
    "is_exact": pl.Boolean(),
    "is_experimental": pl.Boolean(),
    "name_source": pl.Utf8(),
    "rationale": pl.Utf8(),
    "unsupported_reason": pl.Utf8(),
    "explained_rows": pl.Int64(),
    "attribution_values": pl.Int64(),
}

SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "explanation_values": VALUES_SCHEMA,
    "explanation_cases": CASES_SCHEMA,
    "explanation_importance": IMPORTANCE_SCHEMA,
    "explanation_stability": STABILITY_SCHEMA,
    "explanation_drift": DRIFT_SCHEMA,
    "explanation_representative_cases": CASES_SELECTED_SCHEMA,
    "explanation_support": SUPPORT_SCHEMA,
}

#: Every table lives in the new explanations layer. Unlike Component 9, nothing here is a
#: prediction or a tuning trial, so there is no second or third home: an attribution is
#: neither a model output (it explains one) nor an evaluation result (it measures nothing).
#: ADR 0028.
LAYERS: dict[str, str] = dict.fromkeys(SCHEMAS, "explanations")

#: The anchor artifact the manifest is keyed to.
DATASET_SLUG = "explanation_values"

#: Sort keys per table, so two runs over the same data produce byte-identical files. Row
#: order is the last place non-determinism could hide.
#:
#: Every key is a full key: it determines the row uniquely. A partial sort key leaves ties
#: whose order polars is free to choose, and a stable sort over an unstable input is not
#: determinism -- Component 7 learned that the expensive way with its row-order sensitivity.
SORT_KEYS: dict[str, list[str]] = {
    "explanation_values": [
        "model_name",
        "fold_set",
        "fold_id",
        "target_inspection_id",
        "feature_name",
    ],
    "explanation_cases": ["model_name", "fold_set", "fold_id", "target_inspection_id"],
    "explanation_importance": ["model_name", "fold_set", "scope", "fold_id", "feature_name"],
    "explanation_stability": [
        "model_name",
        "fold_set",
        "comparison",
        "from_fold_id",
        "to_fold_id",
    ],
    "explanation_drift": ["model_name", "fold_set", "feature_name"],
    "explanation_representative_cases": [
        "model_name",
        "fold_set",
        "fold_id",
        "tier",
        "target_inspection_id",
    ],
    "explanation_support": ["model_name"],
}


def finalize(rows: list[dict[str, object]], table: str) -> pl.DataFrame:
    """Cast to the contract schema and order deterministically.

    The schema is passed to the ``DataFrame`` constructor rather than cast afterwards:
    ``feature_value`` is null wherever the source was NULL, ``calibrated_probability`` is
    null when no calibrated artifact was supplied, and every aggregate-only column on the
    importance table is null on a per-fold row -- so inference would type those columns from
    whichever kind of value happened to come first.
    """
    if table not in SCHEMAS:
        raise KeyError(f"Unknown table: {table}")
    schema = SCHEMAS[table]
    if not rows:
        return empty(table)
    missing = [c for c in schema if c not in rows[0]]
    if missing:
        raise ValueError(f"{table} rows are missing columns: {', '.join(missing)}")
    extra = [c for c in rows[0] if c not in schema]
    if extra:
        raise ValueError(f"{table} rows carry unknown columns: {', '.join(extra)}")
    return pl.DataFrame(rows, schema=schema).sort(SORT_KEYS[table])


def empty(table: str) -> pl.DataFrame:
    """A correctly typed zero-row frame, so a reader meets the right columns."""
    if table not in SCHEMAS:
        raise KeyError(f"Unknown table: {table}")
    return pl.DataFrame(schema=SCHEMAS[table])


def write_table(frame: pl.DataFrame, path: Path) -> Path:
    """Write a table as zstd Parquet, matching project convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")
    logger.info("Wrote %s (%d rows, %d columns)", path, frame.height, frame.width)
    return path


def schema_of(frame: pl.DataFrame) -> dict[str, str]:
    """Column name to dtype string, for recording in the manifest."""
    return {name: str(dtype) for name, dtype in frame.schema.items()}


__all__ = [
    "DATASET_SLUG",
    "LAYERS",
    "SCHEMAS",
    "SORT_KEYS",
    "empty",
    "finalize",
    "schema_of",
    "write_table",
]
