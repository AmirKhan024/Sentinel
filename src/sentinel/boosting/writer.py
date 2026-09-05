"""Output schemas for the four Component 7 tables.

Column order is part of the data contract; changing it is a contract change.

``boosted_predictions`` is deliberately shaped to be readable by
``evaluation.contract.read_predictions`` without translation, exactly as
``baseline_predictions`` is: the contract's two columns plus the metadata that makes a
file on disk self-describing. The two artifacts are separate files under separate slugs
so Component 6's benchmark stays visible and byte-identical -- "did Component 7 actually
improve on Component 6?" is only answerable if Component 6's answer was not overwritten.

``tuning_trials`` lives in a different directory from the other three, and the separation
is the point. Its ``mean_pr_auc`` is a *validation* number measured on windows that are
training data for every fold the winning parameters are used on. Filed beside the
predictions it would eventually be read as a result. See ADR 0018.

No table carries a timestamp or a duration, with one deliberate exception: ``seconds`` on
the trials table, because how long a search took is the fact that justifies its trial
count, and the trials table makes no determinism claim about its bytes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

#: One row per (model, fold, scored test row). The artifact Component 5 consumes.
PREDICTIONS_SCHEMA: dict[str, pl.DataType] = {
    "target_inspection_id": pl.Utf8(),
    "score": pl.Float64(),
    "model_name": pl.Utf8(),
    "model_version": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    # The fold's training end. Never null -- a null reads as "undeclared" and silently
    # skips the evaluator's horizon check.
    "trained_through": pl.Date(),
    # Never null for the same reason: ``read_predictions`` coerces a null to False,
    # downgrading a probability model to ranking-only and suppressing the probability
    # metrics. True here means "predict_proba output", not "calibrated".
    "is_probability": pl.Boolean(),
    "boosting_definition_version": pl.Utf8(),
}

#: One row per (model, fold, matrix column). Native split importances.
#:
#: ``importance`` is a **diagnostic**, not an attribution. A tree ensemble splits credit
#: between collinear features according to which it happened to reach first, and
#: Component 6 measured a condition number of 71.8 with one pair correlated at 0.9888.
#: The column is named ``importance`` rather than anything suggesting contribution or
#: effect, and the data contract says so in bold. Component 11 owns attribution.
IMPORTANCES_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "model_version": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "term": pl.Utf8(),
    "importance": pl.Float64(),
    "importance_kind": pl.Utf8(),
    "boosting_definition_version": pl.Utf8(),
}

#: One row per (model, fold). What the fit saw and what it did.
TRAINING_LOG_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "model_version": pl.Utf8(),
    "estimator": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "train_start": pl.Date(),
    "train_end": pl.Date(),
    "trained_through": pl.Date(),
    # Recorded to make visible that Component 7 did *not* use it -- not for the model,
    # and not for early stopping either. Component 9 will.
    "calibration_end_unused": pl.Date(),
    "test_start": pl.Date(),
    "test_end": pl.Date(),
    "train_rows": pl.Int32(),
    "test_rows": pl.Int32(),
    "feature_count": pl.Int32(),
    "matrix_column_count": pl.Int32(),
    # Cells that reached the estimator as NaN. The observable proving no imputation
    # happened: a zero here on a window with nullable columns means one did.
    "train_nan_cells": pl.Int32(),
    "train_positive_rate": pl.Float64(),
    "seed": pl.Int32(),
    "n_estimators": pl.Int32(),
    "trees_built": pl.Int32(),
    "scale_pos_weight": pl.Float64(),
    "class_weighted": pl.Boolean(),
    "early_stopped": pl.Boolean(),
    "saturated_scores": pl.Int32(),
    "boosting_definition_version": pl.Utf8(),
}

#: One row per (study, trial). Written to the tuning layer, never beside the predictions.
TUNING_TRIALS_SCHEMA: dict[str, pl.DataType] = {
    "study": pl.Utf8(),
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "trial": pl.Int32(),
    "params": pl.Utf8(),
    "mean_pr_auc": pl.Float64(),
    "inner_fold_pr_aucs": pl.Utf8(),
    "inner_folds": pl.Int32(),
    "n_estimators": pl.Int32(),
    "seconds": pl.Float64(),
    "failed": pl.Boolean(),
    "failure": pl.Utf8(),
    "sampler_seed": pl.Int32(),
    "region_start": pl.Date(),
    "region_end": pl.Date(),
    "boosting_definition_version": pl.Utf8(),
}

SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "boosted_predictions": PREDICTIONS_SCHEMA,
    "boosted_importances": IMPORTANCES_SCHEMA,
    "boosted_training_log": TRAINING_LOG_SCHEMA,
    "tuning_trials": TUNING_TRIALS_SCHEMA,
}

#: Sort keys per table, so two runs over the same data produce byte-identical files.
#: Row order is the last place non-determinism could hide after the training sort.
SORT_KEYS: dict[str, list[str]] = {
    "boosted_predictions": ["model_name", "fold_set", "fold_id", "target_inspection_id"],
    "boosted_importances": ["model_name", "fold_set", "fold_id", "term"],
    "boosted_training_log": ["model_name", "fold_set", "fold_id"],
    "tuning_trials": ["study", "trial"],
}


def finalize(rows: list[dict[str, object]], table: str) -> pl.DataFrame:
    """Cast to the contract schema and order deterministically.

    The schema is passed to the ``DataFrame`` constructor rather than cast afterwards:
    ``train_positive_rate`` is null for an empty window and ``mean_pr_auc`` is NaN for a
    failed trial, so inference would type those columns from whichever kind of value
    happened to come first.
    """
    if table not in SCHEMAS:
        raise KeyError(f"Unknown table: {table}")
    schema = SCHEMAS[table]
    if not rows:
        return empty(table)
    missing = [c for c in schema if c not in rows[0]]
    if missing:
        raise ValueError(f"{table} rows are missing columns: {', '.join(missing)}")
    frame = pl.DataFrame(rows, schema=schema)
    return frame.sort(SORT_KEYS[table])


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


def render_params(params: dict[str, object]) -> str:
    """Hyperparameters as a stable string, so two manifests compare textually."""
    return ", ".join(f"{key}={params[key]!r}" for key in sorted(params))


__all__ = [
    "IMPORTANCES_SCHEMA",
    "PREDICTIONS_SCHEMA",
    "SCHEMAS",
    "SORT_KEYS",
    "TRAINING_LOG_SCHEMA",
    "TUNING_TRIALS_SCHEMA",
    "empty",
    "finalize",
    "render_params",
    "schema_of",
    "write_table",
]
