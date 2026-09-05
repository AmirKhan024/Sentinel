"""Build and write the Component 5 output tables.

Six tables, each with an explicit schema whose column order is part of the
contract, following the same rule as Components 2-4: a reader scanning a table
should meet the keys first and the answer last.

The metrics table is deliberately **long** rather than wide -- one row per
``(fold, model, metric)`` -- so Component 6 can add a model, and Component 9 a
calibration metric, without either reshaping the table or breaking a query
written against it today.

Curves are stored at full resolution rather than summarized. The discovery
curve *is* the result; a reader who cannot redraw it has to take the headline
number on trust.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from sentinel.evaluation.models import EVALUATION_DEFINITION_VERSION

logger = logging.getLogger(__name__)

FOLDS_SCHEMA: dict[str, pl.DataType] = {
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "train_start": pl.Date(),
    "train_end": pl.Date(),
    "calibration_start": pl.Date(),
    "calibration_end": pl.Date(),
    "test_start": pl.Date(),
    "test_end": pl.Date(),
    "train_rows": pl.Int32(),
    "calibration_rows": pl.Int32(),
    "test_rows": pl.Int32(),
    "train_positive_rate": pl.Float64(),
    "calibration_positive_rate": pl.Float64(),
    "test_positive_rate": pl.Float64(),
    "test_establishments": pl.Int32(),
    "test_days": pl.Int32(),
    "test_median_daily_capacity": pl.Float64(),
    "evaluation_definition_version": pl.Utf8(),
}

METRICS_SCHEMA: dict[str, pl.DataType] = {
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "model_name": pl.Utf8(),
    "model_version": pl.Utf8(),
    "metric": pl.Utf8(),
    "metric_kind": pl.Utf8(),
    "k_name": pl.Utf8(),
    "k": pl.Int32(),
    "value": pl.Float64(),
    "test_start": pl.Date(),
    "test_end": pl.Date(),
    "n_test": pl.Int32(),
    "positive_rate": pl.Float64(),
    "evaluation_definition_version": pl.Utf8(),
}

CURVES_SCHEMA: dict[str, pl.DataType] = {
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "schedule_name": pl.Utf8(),
    "model_name": pl.Utf8(),
    "seed": pl.Int32(),
    "slot_index": pl.Int32(),
    "slot_fraction": pl.Float64(),
    "cumulative_positive": pl.Int32(),
    "cumulative_positive_fraction": pl.Float64(),
}

SIMULATION_SCHEMA: dict[str, pl.DataType] = {
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "schedule_name": pl.Utf8(),
    "model_name": pl.Utf8(),
    "seed": pl.Int32(),
    "n_slots": pl.Int32(),
    "n_positives": pl.Int32(),
    "median_daily_capacity": pl.Float64(),
    "discovery_area": pl.Float64(),
    "normalized_discovery_efficiency": pl.Float64(),
    "first_half_discovery": pl.Float64(),
    "days_earlier_count": pl.Int32(),
    "mean_days_earlier": pl.Float64(),
    "median_days_earlier": pl.Float64(),
    "std_days_earlier": pl.Float64(),
    "p25_days_earlier": pl.Float64(),
    "p75_days_earlier": pl.Float64(),
    "min_days_earlier": pl.Float64(),
    "max_days_earlier": pl.Float64(),
    "fraction_improved": pl.Float64(),
    "fraction_unchanged": pl.Float64(),
    "fraction_worse": pl.Float64(),
    "mean_days_earlier_all_rows": pl.Float64(),
    "fraction_improved_all_rows": pl.Float64(),
}

SEASONALITY_SCHEMA: dict[str, pl.DataType] = {
    "fitted_on": pl.Utf8(),
    "month": pl.Int32(),
    "rows": pl.Int32(),
    "raw_rate": pl.Float64(),
    "log_odds_effect": pl.Float64(),
    "pct_point_shift_at_base": pl.Float64(),
}

SENSITIVITY_SCHEMA: dict[str, pl.DataType] = {
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "model_name": pl.Utf8(),
    "metric": pl.Utf8(),
    "replications": pl.Int32(),
    "observed": pl.Float64(),
    "mean": pl.Float64(),
    "std": pl.Float64(),
    "p05": pl.Float64(),
    "p50": pl.Float64(),
    "p95": pl.Float64(),
    "label_flip_rate": pl.Float64(),
}

SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "evaluation_folds": FOLDS_SCHEMA,
    "evaluation_metrics": METRICS_SCHEMA,
    "discovery_curves": CURVES_SCHEMA,
    "simulation_summary": SIMULATION_SCHEMA,
    "seasonality": SEASONALITY_SCHEMA,
    "sensitivity": SENSITIVITY_SCHEMA,
}

#: Sort keys per table, so two runs over the same data produce byte-identical
#: files. Determinism in this component is structural apart from the explicitly
#: seeded random schedules, and this is the last place row order could leak in.
SORT_KEYS: dict[str, list[str]] = {
    "evaluation_folds": ["fold_set", "test_start", "fold_id"],
    "evaluation_metrics": ["fold_set", "fold_id", "model_name", "metric", "k_name"],
    "discovery_curves": [
        "fold_set",
        "fold_id",
        "schedule_name",
        "model_name",
        "seed",
        "slot_index",
    ],
    "simulation_summary": ["fold_set", "fold_id", "schedule_name", "model_name", "seed"],
    "seasonality": ["fitted_on", "month"],
    "sensitivity": ["fold_set", "fold_id", "model_name", "metric"],
}


def finalize(rows: list[dict[str, object]], table: str) -> pl.DataFrame:
    """Build a table from records against its contract schema, then sort it.

    The schema is supplied to the constructor rather than applied afterwards.
    That matters: several columns are legitimately null for some rows and
    populated for others -- ``seed`` is null for the deterministic schedules and
    an integer for the random ones -- and letting polars infer from the leading
    rows would type the column from whichever kind happened to come first.
    """
    if table not in SCHEMAS:
        raise KeyError(f"unknown evaluation table: {table}")
    schema = dict(SCHEMAS[table])
    if not rows:
        return pl.DataFrame(schema=schema)

    missing = [c for c in schema if c not in rows[0]]
    if missing:
        raise ValueError(f"{table} is missing columns: {', '.join(missing)}")
    return pl.DataFrame(rows, schema=schema).sort(SORT_KEYS[table])


def empty(table: str) -> pl.DataFrame:
    """An empty frame with the table's exact schema.

    Used when a section legitimately produces nothing -- a blocked experiment,
    say -- so downstream readers meet the right columns rather than a crash.
    """
    return pl.DataFrame(schema=dict(SCHEMAS[table]))


def write_table(frame: pl.DataFrame, path: Path) -> Path:
    """Write one evaluation table as zstd Parquet, matching project convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")
    logger.info("Wrote %s (%d rows, %d columns)", path, frame.height, frame.width)
    return path


def schema_of(frame: pl.DataFrame) -> dict[str, str]:
    """Column name to dtype string, for recording in the manifest."""
    return {name: str(dtype) for name, dtype in frame.schema.items()}


__all__ = [
    "CURVES_SCHEMA",
    "EVALUATION_DEFINITION_VERSION",
    "FOLDS_SCHEMA",
    "METRICS_SCHEMA",
    "SCHEMAS",
    "SEASONALITY_SCHEMA",
    "SENSITIVITY_SCHEMA",
    "SIMULATION_SCHEMA",
    "SORT_KEYS",
    "empty",
    "finalize",
    "schema_of",
    "write_table",
]
