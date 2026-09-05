"""Build and write the Component 17 output table.

Two frames are combined, deliberately kept apart until the last step:

1. The **pure feature frame** -- ``features.writer.finalize()``'s output, cast to
   Component 4's exact schema (``features.writer.output_schema()``) with no
   extra column. This is what gets validated against
   ``features.validate.validate_features``, so that check's
   ``every_column_is_declared`` assertion stays meaningful: it would trivially
   fail if location metadata were mixed in before validation ran.

2. The **candidate metadata frame** -- display and as-of location fields
   Component 4's contract has never carried, plus Component 2's own canonical
   name/address (display fields there too, per ``entity.writer.build_establishments``'s
   own docstring).

They are joined only after the feature frame has passed validation. Column
order mirrors ``features.writer``'s convention: keys, then features, then
labels, then candidate metadata, then provenance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from sentinel.candidates.definitions import CANDIDATE_DEFINITION_VERSION
from sentinel.features.definitions import FEATURE_SPECS
from sentinel.features.writer import output_schema as feature_output_schema

logger = logging.getLogger(__name__)

METADATA_SCHEMA: dict[str, pl.DataType] = {
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
    "first_known_date": pl.Utf8(),
    "last_known_date": pl.Utf8(),
}


def output_schema() -> dict[str, pl.DataType]:
    """The full candidate output schema: Component 4's columns, then candidate metadata."""
    schema = dict(feature_output_schema())
    schema.update(METADATA_SCHEMA)
    schema["planning_date"] = pl.Utf8()
    schema["candidate_definition_version"] = pl.Utf8()
    return schema


def combine(
    pure_features: pl.DataFrame,
    metadata: pl.DataFrame,
    establishments: pl.DataFrame,
    *,
    planning_date: str,
) -> pl.DataFrame:
    """Join the validated feature frame with location and display metadata.

    ``pure_features`` must already be ``features.writer.finalize()``'s output --
    Component 4's exact 33-plus-key-plus-label schema, validated. Nothing here
    changes a single feature value; every operation is a left join on
    ``establishment_id``, so a row present in ``pure_features`` is present in the
    output whether or not metadata for it exists.
    """
    display = establishments.select(
        ["establishment_id", "canonical_name", "canonical_address", "canonical_zip"]
    )
    combined = (
        pure_features.join(metadata, on="establishment_id", how="left")
        .join(display, on="establishment_id", how="left")
        .with_columns(
            pl.lit(planning_date).alias("planning_date"),
            pl.lit(CANDIDATE_DEFINITION_VERSION).alias("candidate_definition_version"),
        )
    )
    schema = output_schema()
    missing = [c for c in schema if c not in combined.columns]
    if missing:
        raise ValueError(f"Candidate frame is missing columns: {', '.join(missing)}")
    return combined.select([pl.col(name).cast(dtype) for name, dtype in schema.items()]).sort(
        ["establishment_id", "target_inspection_id"]
    )


def write_table(frame: pl.DataFrame, path: Path) -> Path:
    """Write the candidate table as zstd Parquet, matching project convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")
    logger.info("Wrote %s (%d rows, %d columns)", path, frame.height, frame.width)
    return path


def schema_of(frame: pl.DataFrame) -> dict[str, str]:
    """Column name to dtype string, for recording in the manifest."""
    return {name: str(dtype) for name, dtype in frame.schema.items()}


def null_rates(frame: pl.DataFrame) -> dict[str, float]:
    """Null fraction per feature, for the manifest and the report."""
    if frame.height == 0:
        return {spec.name: 0.0 for spec in FEATURE_SPECS}
    return {
        spec.name: round(frame[spec.name].null_count() / frame.height, 6) for spec in FEATURE_SPECS
    }


__all__ = ["METADATA_SCHEMA", "combine", "null_rates", "output_schema", "schema_of", "write_table"]
