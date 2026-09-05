"""Output schemas for the seven Component 8 tables.

Column order is part of the data contract; changing it is a contract change.

``neural_predictions`` is deliberately shaped to be readable by
``evaluation.contract.read_predictions`` without translation, exactly as
``baseline_predictions`` and ``boosted_predictions`` are. The three artifacts are separate
files under separate slugs so Components 6's and 7's benchmarks stay visible and
byte-identical -- "did Component 8 actually improve on Component 7?" is only answerable if
Component 7's answer was not overwritten.

``neural_categoricals`` lives in a different directory from the rest, and the separation
is the point. It is Component 8's own experimental input, not a Component 4 feature table,
and filing it under ``features/`` would eventually get it joined onto one. See ADR 0022.

``neural_epoch_log`` is the one table in this project that records a per-iteration
trajectory. Its ``validation_loss`` is measured on rows carved from inside the training
window: it is an in-sample number in exactly the sense Component 7's inner-fold PR-AUC is,
and no value in this table is a result.

No table carries a timestamp or a duration, with one deliberate exception inherited from
Component 7's reasoning: ``seconds`` on the sweep trials table, because how long a search
took is the fact that justifies its size, and that table makes no determinism claim about
its bytes.
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
    # downgrading a probability model to ranking-only and suppressing every probability
    # metric. True here means "sigmoid of the network's logit", NOT "calibrated".
    "is_probability": pl.Boolean(),
    "neural_definition_version": pl.Utf8(),
}

#: The experimental categorical join. Its own layer, its own slug. ADR 0022.
CATEGORICALS_SCHEMA: dict[str, pl.DataType] = {
    "target_inspection_id": pl.Utf8(),
    "establishment_id": pl.Utf8(),
    "inspection_date": pl.Utf8(),
    "chain_key": pl.Utf8(),
    "facility_type": pl.Utf8(),
    "community_area": pl.Utf8(),
    "zip": pl.Utf8(),
    # Which earlier inspection supplied the values, and how much earlier. Emitted so the
    # as-of claim is checkable per row rather than asserted once in a manifest: every
    # source date must be strictly earlier than the row's own inspection_date, and
    # ``validate`` re-derives exactly that.
    "source_inspection_id": pl.Utf8(),
    "source_inspection_date": pl.Date(),
    "days_since_source": pl.Int32(),
}

#: One row per (model, fold). What the fit saw and what it did.
TRAINING_LOG_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "model_version": pl.Utf8(),
    "learner": pl.Utf8(),
    "encoding": pl.Utf8(),
    "experiment": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "train_start": pl.Date(),
    "train_end": pl.Date(),
    "trained_through": pl.Date(),
    # Recorded to make visible that Component 8 did not use it -- not for the model, and
    # not for early stopping either. Component 9 will.
    "calibration_end_unused": pl.Date(),
    # The first date of the early-stopping window, which sits INSIDE the training data.
    # This is the column that makes the trained_through claim checkable.
    "inner_validation_start": pl.Date(),
    "test_start": pl.Date(),
    "test_end": pl.Date(),
    "train_rows": pl.Int32(),
    "inner_train_rows": pl.Int32(),
    "inner_validation_rows": pl.Int32(),
    "test_rows": pl.Int32(),
    "feature_count": pl.Int32(),
    "entity_count": pl.Int32(),
    "dense_width": pl.Int32(),
    "embedding_width": pl.Int32(),
    "parameter_count": pl.Int32(),
    "vocabulary_total": pl.Int32(),
    # Cells that were NULL before imputation. Component 7 records this to prove
    # imputation did not happen; Component 8 records it to say how much did.
    "train_nan_cells": pl.Int32(),
    "train_positive_rate": pl.Float64(),
    "seed": pl.Int32(),
    "learning_rate": pl.Float64(),
    "pos_weight": pl.Float64(),
    "best_epoch": pl.Int32(),
    "final_epoch": pl.Int32(),
    "learning_rate_changes": pl.Int32(),
    "stop_reason": pl.Utf8(),
    "saturated_scores": pl.Int32(),
    "neural_definition_version": pl.Utf8(),
}

#: One row per (model, fold, epoch). The learning curves, as data.
EPOCH_LOG_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "seed": pl.Int32(),
    "epoch": pl.Int32(),
    "train_loss": pl.Float64(),
    # In-sample: measured on rows inside the training window. Not a result.
    "validation_loss": pl.Float64(),
    "learning_rate": pl.Float64(),
    "is_best_epoch": pl.Boolean(),
    "neural_definition_version": pl.Utf8(),
}

#: One row per (model, fold, family, category, dimension). The learned representations.
EMBEDDINGS_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "family": pl.Utf8(),
    "category": pl.Utf8(),
    "category_index": pl.Int32(),
    "dimension": pl.Int32(),
    "value": pl.Float64(),
    "neural_definition_version": pl.Utf8(),
}

#: One row per (model, fold, seed, metric). The reproducibility experiment.
SEED_VARIATION_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "seed": pl.Int32(),
    "pr_auc": pl.Float64(),
    "roc_auc": pl.Float64(),
    "best_epoch": pl.Int32(),
    "neural_definition_version": pl.Utf8(),
}

#: One row per (study, learning rate, inner fold). Written to the tuning layer.
SWEEP_TRIALS_SCHEMA: dict[str, pl.DataType] = {
    "study": pl.Utf8(),
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "inner_fold_id": pl.Utf8(),
    "learning_rate": pl.Float64(),
    "pr_auc": pl.Float64(),
    "best_epoch": pl.Int32(),
    "train_rows": pl.Int32(),
    "validation_rows": pl.Int32(),
    "selected": pl.Boolean(),
    "seed": pl.Int32(),
    "region_start": pl.Date(),
    "region_end": pl.Date(),
    "neural_definition_version": pl.Utf8(),
}

SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "neural_predictions": PREDICTIONS_SCHEMA,
    "neural_categoricals": CATEGORICALS_SCHEMA,
    "neural_training_log": TRAINING_LOG_SCHEMA,
    "neural_epoch_log": EPOCH_LOG_SCHEMA,
    "neural_embeddings": EMBEDDINGS_SCHEMA,
    "neural_seed_variation": SEED_VARIATION_SCHEMA,
    "neural_sweep_trials": SWEEP_TRIALS_SCHEMA,
}

#: Sort keys per table, so two runs over the same data produce byte-identical files. Row
#: order is the last place non-determinism could hide after the training sort.
SORT_KEYS: dict[str, list[str]] = {
    "neural_predictions": ["model_name", "fold_set", "fold_id", "target_inspection_id"],
    "neural_categoricals": ["target_inspection_id"],
    "neural_training_log": ["model_name", "fold_set", "fold_id"],
    "neural_epoch_log": ["model_name", "fold_set", "fold_id", "seed", "epoch"],
    "neural_embeddings": ["model_name", "fold_set", "fold_id", "family", "category", "dimension"],
    "neural_seed_variation": ["model_name", "fold_set", "fold_id", "seed"],
    "neural_sweep_trials": ["study", "learning_rate", "inner_fold_id"],
}


def finalize(rows: list[dict[str, object]], table: str) -> pl.DataFrame:
    """Cast to the contract schema and order deterministically.

    The schema is passed to the ``DataFrame`` constructor rather than cast afterwards:
    ``train_positive_rate`` is null for an empty window and ``pos_weight`` is null for
    every model except one, so inference would type those columns from whichever kind of
    value happened to come first.
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
    "CATEGORICALS_SCHEMA",
    "EMBEDDINGS_SCHEMA",
    "EPOCH_LOG_SCHEMA",
    "PREDICTIONS_SCHEMA",
    "SCHEMAS",
    "SEED_VARIATION_SCHEMA",
    "SORT_KEYS",
    "SWEEP_TRIALS_SCHEMA",
    "TRAINING_LOG_SCHEMA",
    "empty",
    "finalize",
    "render_params",
    "schema_of",
    "write_table",
]
