"""Build and write the Component 4 output table.

The schema is derived from ``FEATURE_SPECS`` rather than written out, so a column
cannot exist without a specification, a description and a missing-value rule.

Column order is deliberate and part of the contract: keys, then features grouped
by family, then labels, then provenance. A reader scanning the table sees the
join keys first and the answer last.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from sentinel.features.definitions import (
    FEATURE_DEFINITION_VERSION,
    FEATURE_SPECS,
    KEY_COLUMNS,
    LABEL_COLUMNS,
    PROVENANCE_COLUMNS,
)

logger = logging.getLogger(__name__)

_DTYPES: dict[str, pl.DataType] = {
    "int32": pl.Int32(),
    "float64": pl.Float64(),
    "bool": pl.Boolean(),
}


def output_schema() -> dict[str, pl.DataType]:
    """The full output schema, in contract order."""
    schema: dict[str, pl.DataType] = {
        "establishment_id": pl.Utf8(),
        "inspection_date": pl.Utf8(),
        "target_inspection_id": pl.Utf8(),
    }
    for spec in FEATURE_SPECS:
        schema[spec.name] = _DTYPES[spec.dtype]
    schema["target"] = pl.Int8()
    schema["target_status"] = pl.Utf8()
    schema["code_era_phase"] = pl.Utf8()
    schema["feature_definition_version"] = pl.Utf8()
    return schema


OUTPUT_COLUMNS: tuple[str, ...] = tuple(output_schema())


def finalize(frame: pl.DataFrame) -> pl.DataFrame:
    """Cast to the contract schema, add provenance, and order deterministically."""
    schema = output_schema()
    prepared = frame.with_columns(
        pl.lit(FEATURE_DEFINITION_VERSION).alias("feature_definition_version")
    )
    missing = [c for c in schema if c not in prepared.columns]
    if missing:
        raise ValueError(f"Feature frame is missing columns: {', '.join(missing)}")

    return prepared.select([pl.col(name).cast(dtype) for name, dtype in schema.items()]).sort(
        ["establishment_id", "inspection_date", "target_inspection_id"]
    )


def write_table(frame: pl.DataFrame, path: Path) -> Path:
    """Write the feature table as zstd Parquet, matching project convention."""
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


__all__ = [
    "FEATURE_DEFINITION_VERSION",
    "KEY_COLUMNS",
    "LABEL_COLUMNS",
    "OUTPUT_COLUMNS",
    "PROVENANCE_COLUMNS",
    "finalize",
    "null_rates",
    "output_schema",
    "schema_of",
    "write_table",
]
