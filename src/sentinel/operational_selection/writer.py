"""Build and write the Component 19 output table: `OperationalSelectionSet`."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "planning_date": pl.Utf8(),
    "operational_selection_definition_version": pl.Utf8(),
    "requested_capacity": pl.Int64(),
    "policy_id": pl.Utf8(),
    "candidate_definition_version": pl.Utf8(),
    "feature_definition_version": pl.Utf8(),
    "operational_scoring_definition_version": pl.Utf8(),
    "composite_model_name": pl.Utf8(),
    "base_model_name": pl.Utf8(),
    "calibration_method": pl.Utf8(),
    "establishment_id": pl.Utf8(),
    "target_inspection_id": pl.Utf8(),
    "canonical_name": pl.Utf8(),
    "canonical_address": pl.Utf8(),
    "canonical_zip": pl.Utf8(),
    "as_of_dba_name": pl.Utf8(),
    "as_of_address": pl.Utf8(),
    "as_of_zip": pl.Utf8(),
    "as_of_latitude": pl.Float64(),
    "as_of_longitude": pl.Float64(),
    "has_location": pl.Boolean(),
    "n_prior_records": pl.Int64(),
    "scoring_status": pl.Utf8(),
    "base_score": pl.Float64(),
    "calibrated_score": pl.Float64(),
    "rank": pl.Int64(),
    "coverage_eligible": pl.Boolean(),
    "secondary_no_history": pl.Boolean(),
    "selection_mechanism": pl.Utf8(),
    "selection_reason": pl.Utf8(),
    "policy_rank": pl.Int64(),
    "is_selected": pl.Boolean(),
}


def finalize(
    frame: pl.DataFrame,
    *,
    operational_selection_definition_version: str,
    requested_capacity: int,
    policy_id: str,
) -> pl.DataFrame:
    """Attach run-level provenance and cast to the declared output contract.

    ``planning_date`` is not stamped here -- it is already present, carried straight
    through from Component 18's own output via ``select.CARRIED_COLUMNS``.
    """
    prepared = frame.with_columns(
        pl.lit(operational_selection_definition_version).alias(
            "operational_selection_definition_version"
        ),
        pl.lit(requested_capacity).alias("requested_capacity"),
        pl.lit(policy_id).alias("policy_id"),
    )
    missing = [c for c in OUTPUT_SCHEMA if c not in prepared.columns]
    if missing:
        raise ValueError(f"Operational selection frame is missing columns: {', '.join(missing)}")
    return prepared.select(
        [pl.col(name).cast(dtype) for name, dtype in OUTPUT_SCHEMA.items()]
    ).sort([pl.col("policy_rank").is_null(), "policy_rank", "target_inspection_id"])


def write_table(frame: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")
    logger.info("Wrote %s (%d rows, %d columns)", path, frame.height, frame.width)
    return path


def schema_of(frame: pl.DataFrame) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in frame.schema.items()}


__all__ = ["OUTPUT_SCHEMA", "finalize", "schema_of", "write_table"]
