"""Build and write the Component 3 output table.

Interim layer, so real types are used. Identifiers stay ``Utf8`` for the same
reason as Component 2: they are labels, not quantities.

``target`` is ``Int8`` and nullable rather than ``Boolean``, because a null is a
meaningful third state — the row was not labellable — and a nullable boolean
would invite silent coercion to False somewhere downstream.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from sentinel.target.models import TargetRow

logger = logging.getLogger(__name__)

# Column order is part of the data contract; changing it is a contract change.
TARGETS_SCHEMA: dict[str, pl.DataType] = {
    # -- identity and timing ------------------------------------------------
    "establishment_id": pl.Utf8(),
    "inspection_date": pl.Utf8(),
    "target_inspection_id": pl.Utf8(),
    # -- the label ----------------------------------------------------------
    "target": pl.Int8(),
    "target_status": pl.Utf8(),
    # -- audit: describes the target event itself, never a feature ----------
    "inspection_type": pl.Utf8(),
    "results": pl.Utf8(),
    "has_priority": pl.Boolean(),
    "has_priority_foundation": pl.Boolean(),
    "n_priority_entries": pl.Int32(),
    "n_priority_foundation_entries": pl.Int32(),
    "n_violation_entries": pl.Int32(),
    "evidence": pl.Utf8(),
    "n_contributing_inspections": pl.Int32(),
    "contributing_inspection_ids": pl.Utf8(),
    # -- provenance ---------------------------------------------------------
    "code_era_phase": pl.Utf8(),
    "target_definition_version": pl.Utf8(),
}

# Columns that describe the outcome of the target event. Component 4 must not
# use any of these as model inputs; they are here so a label can be audited.
# Enumerated in code as well as in the contract so a test can assert it.
TARGET_EVENT_COLUMNS = frozenset(
    {
        "target",
        "results",
        "has_priority",
        "has_priority_foundation",
        "n_priority_entries",
        "n_priority_foundation_entries",
        "n_violation_entries",
        "evidence",
    }
)


def _empty(schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=dict(schema))


def build_targets(rows: Sequence[TargetRow]) -> pl.DataFrame:
    """Materialise target rows as a typed frame, sorted deterministically."""
    if not rows:
        return _empty(TARGETS_SCHEMA)

    records = [
        {
            "establishment_id": r.establishment_id,
            "inspection_date": r.inspection_date,
            "target_inspection_id": r.target_inspection_id,
            "target": r.target,
            "target_status": r.target_status.value,
            "inspection_type": r.inspection_type,
            "results": r.results,
            "has_priority": r.has_priority,
            "has_priority_foundation": r.has_priority_foundation,
            "n_priority_entries": r.n_priority_entries,
            "n_priority_foundation_entries": r.n_priority_foundation_entries,
            "n_violation_entries": r.n_violation_entries,
            "evidence": r.evidence,
            "n_contributing_inspections": r.n_contributing_inspections,
            "contributing_inspection_ids": r.contributing_inspection_ids,
            "code_era_phase": r.code_era_phase.value,
            "target_definition_version": r.target_definition_version,
        }
        for r in rows
    ]
    return pl.DataFrame(records, schema=dict(TARGETS_SCHEMA)).sort(
        ["establishment_id", "inspection_date", "target_inspection_id"]
    )


def write_table(frame: pl.DataFrame, path: Path) -> Path:
    """Write the target table as zstd Parquet, matching the project convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")
    logger.info("Wrote %s (%d rows)", path, frame.height)
    return path


def schema_of(frame: pl.DataFrame) -> dict[str, str]:
    """Column name to dtype string, for recording in the manifest."""
    return {name: str(dtype) for name, dtype in frame.schema.items()}
