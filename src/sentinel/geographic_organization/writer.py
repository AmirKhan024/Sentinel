"""Build and write the Component 20 output table: `GeographicInspectionPlan`.

The grain is the SELECTED set only (``is_selected == True`` from Component 19), not
the full ranked/priority queue -- unlike Components 18/19, which preserve their whole
input for audit purposes, Component 20's job is to organize a bounded plan a
supervisor is about to review, and a consumer who needs the wider context can already
join back to the Component 19 artifact by ``target_inspection_id``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "planning_date": pl.Utf8(),
    "geographic_organization_definition_version": pl.Utf8(),
    "geographic_algorithm": pl.Utf8(),
    "threshold_km": pl.Float64(),
    "operational_selection_definition_version": pl.Utf8(),
    "requested_capacity": pl.Int64(),
    "policy_id": pl.Utf8(),
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
    "n_prior_records": pl.Int64(),
    "base_score": pl.Float64(),
    "calibrated_score": pl.Float64(),
    "rank": pl.Int64(),
    "policy_rank": pl.Int64(),
    "coverage_eligible": pl.Boolean(),
    "secondary_no_history": pl.Boolean(),
    "selection_mechanism": pl.Utf8(),
    "selection_reason": pl.Utf8(),
    "is_selected": pl.Boolean(),
    "location_status": pl.Utf8(),
    "geographic_group_id": pl.Utf8(),
    "geographic_group_label": pl.Utf8(),
    # --- v2 additive columns: operational work-block planning fields ---
    # Alias ``geographic_group_id``/``geographic_group_label`` under the operational
    # vocabulary -- same value, so a reader of either column pair sees the same block.
    "work_block_id": pl.Utf8(),
    "work_block_label": pl.Utf8(),
    "suggested_order_in_block": pl.Int64(),
    "organization_mode": pl.Utf8(),
    "highest_sentinel_rank_in_block": pl.Int64(),
}


def finalize(
    frame: pl.DataFrame,
    *,
    geographic_organization_definition_version: str,
    geographic_algorithm: str,
    threshold_km: float,
) -> pl.DataFrame:
    """Attach run-level provenance and cast to the declared output contract."""
    prepared = frame.with_columns(
        pl.lit(geographic_organization_definition_version).alias(
            "geographic_organization_definition_version"
        ),
        pl.lit(geographic_algorithm).alias("geographic_algorithm"),
        pl.lit(threshold_km).alias("threshold_km"),
    )
    missing = [c for c in OUTPUT_SCHEMA if c not in prepared.columns]
    if missing:
        raise ValueError(f"Geographic plan frame is missing columns: {', '.join(missing)}")
    return prepared.select(
        [pl.col(name).cast(dtype) for name, dtype in OUTPUT_SCHEMA.items()]
    ).sort(
        [
            pl.col("work_block_id"),
            pl.col("suggested_order_in_block").is_null(),
            "suggested_order_in_block",
        ]
    )


def write_table(frame: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")
    logger.info("Wrote %s (%d rows, %d columns)", path, frame.height, frame.width)
    return path


def schema_of(frame: pl.DataFrame) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in frame.schema.items()}


__all__ = ["OUTPUT_SCHEMA", "finalize", "schema_of", "write_table"]
