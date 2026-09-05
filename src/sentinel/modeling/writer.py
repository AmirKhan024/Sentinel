"""Output schemas for the three Component 6 tables.

Column order is part of the data contract; changing it is a contract change.

``baseline_predictions`` is deliberately shaped to be readable by
``evaluation.contract.read_predictions`` without translation: it carries the contract's
two columns plus the metadata that makes a file on disk self-describing. ``fold_set`` is
extra, and harmless -- ``read_predictions`` selects the columns it needs -- and it lets a
reader separate the quarterly folds from ``covid_shift`` without joining the fold table.

No table carries a timestamp or a duration. Wall-clock in a Parquet file would mean two
runs over identical inputs producing different bytes, which contradicts the determinism
this component claims; timings are logged and totalled in the manifest, beside
``built_at``, where non-reproducibility already lives.
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
    # The last reference date the fit was allowed to learn from: the fold's training
    # end. Never null -- a null would read as "undeclared" and silently skip the
    # evaluator's horizon check.
    "trained_through": pl.Date(),
    # Never null for the same reason: ``read_predictions`` coerces a null to False,
    # silently downgrading a probability model to ranking-only and suppressing the
    # probability metrics.
    "is_probability": pl.Boolean(),
    "model_definition_version": pl.Utf8(),
}

#: One row per (model, fold, term). The intercept is a row with term ``__intercept__``,
#: matching the long format Component 5 chose for its metrics rather than widening.
#:
#: ``standardized_coefficient`` is named for what it is: the coefficient on the
#: standardised scale. ``scaler_mean`` and ``scaler_scale`` are emitted beside it so a
#: reader can recover the raw-feature-scale value instead of misreading a standardised
#: one. Without them this table would be numbers nobody can interpret.
COEFFICIENTS_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "model_version": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "term": pl.Utf8(),
    "standardized_coefficient": pl.Float64(),
    "scaler_mean": pl.Float64(),
    "scaler_scale": pl.Float64(),
    "imputed_fill_value": pl.Float64(),
    "model_definition_version": pl.Utf8(),
}

#: One row per (model, fold). What the fit saw and what it did.
TRAINING_LOG_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "model_version": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "train_start": pl.Date(),
    "train_end": pl.Date(),
    "trained_through": pl.Date(),
    # Recorded to make visible that Component 6 did *not* use it. Component 9 will.
    "calibration_end_unused": pl.Date(),
    "test_start": pl.Date(),
    "test_end": pl.Date(),
    "train_rows": pl.Int32(),
    "test_rows": pl.Int32(),
    "feature_count": pl.Int32(),
    "matrix_column_count": pl.Int32(),
    "train_positive_rate": pl.Float64(),
    "seed": pl.Int32(),
    "n_iter": pl.Int32(),
    "max_iter": pl.Int32(),
    "converged": pl.Boolean(),
    "saturated_scores": pl.Int32(),
    "is_approximation": pl.Boolean(),
    "model_definition_version": pl.Utf8(),
}

SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "baseline_predictions": PREDICTIONS_SCHEMA,
    "baseline_coefficients": COEFFICIENTS_SCHEMA,
    "baseline_training_log": TRAINING_LOG_SCHEMA,
}

#: Sort keys per table, so two runs over the same data produce byte-identical files.
#: Row order is the last place non-determinism could hide after the training sort.
SORT_KEYS: dict[str, list[str]] = {
    "baseline_predictions": ["model_name", "fold_set", "fold_id", "target_inspection_id"],
    "baseline_coefficients": ["model_name", "fold_set", "fold_id", "term"],
    "baseline_training_log": ["model_name", "fold_set", "fold_id"],
}


def finalize(rows: list[dict[str, object]], table: str) -> pl.DataFrame:
    """Cast to the contract schema and order deterministically.

    The schema is passed to the ``DataFrame`` constructor rather than cast afterwards:
    ``train_positive_rate`` is null for an empty window and ``imputed_fill_value`` is
    null for a never-null column, so inference would type those columns from whichever
    kind of value happened to come first.
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


__all__ = [
    "COEFFICIENTS_SCHEMA",
    "PREDICTIONS_SCHEMA",
    "SCHEMAS",
    "SORT_KEYS",
    "TRAINING_LOG_SCHEMA",
    "empty",
    "finalize",
    "schema_of",
    "write_table",
]
