"""Minimal DuckDB access to the raw Parquet layer.

DuckDB reads Parquet directly from disk, so there is no load step, no schema
DDL, and no server. That is the whole point of using it here: we get SQL over
the raw layer for the cost of one import.

Scope discipline: these queries only describe the raw data (how many rows, what
date range, which inspection types). They deliberately contain no Sentinel
business logic -- no filtering to relevant facility types, no outcome
definitions, no deduplication. That is Component 2 onward.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)

# Every query reads from the parameterised relation `src`. Values are bound as
# parameters rather than interpolated, so a path never becomes SQL.
NAMED_QUERIES: dict[str, str] = {
    "row_count": """
        SELECT count(*) AS row_count
        FROM read_parquet(?)
    """,
    "unique_licenses": """
        SELECT
            count(DISTINCT license_)                     AS unique_licenses,
            count(*) FILTER (WHERE license_ IS NULL)     AS null_license_rows
        FROM read_parquet(?)
    """,
    "inspection_date_range": """
        SELECT
            min(inspection_date) AS earliest_inspection_date,
            max(inspection_date) AS latest_inspection_date,
            count(*) FILTER (WHERE inspection_date IS NULL) AS null_date_rows
        FROM read_parquet(?)
    """,
    "inspection_types": """
        SELECT inspection_type, count(*) AS n
        FROM read_parquet(?)
        GROUP BY inspection_type
        ORDER BY n DESC
    """,
    "results_breakdown": """
        SELECT results, count(*) AS n
        FROM read_parquet(?)
        GROUP BY results
        ORDER BY n DESC
    """,
    "facility_types": """
        SELECT facility_type, count(*) AS n
        FROM read_parquet(?)
        GROUP BY facility_type
        ORDER BY n DESC
        LIMIT 20
    """,
    "schema": """
        SELECT column_name, column_type AS duckdb_type
        FROM (DESCRIBE SELECT * FROM read_parquet(?))
    """,
}


def connect() -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection.

    In-memory because nothing here needs to persist: the Parquet files on disk
    are the durable artifact. A persistent database file would be a second
    source of truth we would then have to keep in sync.
    """
    return duckdb.connect(database=":memory:")


def latest_parquet(directory: Path, *, prefix: str = "food_inspections_") -> Path:
    """Return the most recent raw Parquet file in a directory.

    Filenames embed a UTC timestamp in a sortable format, so lexicographic max
    is chronological max. Resolving "latest" is a read-time concern; the
    ingestion layer deliberately does not maintain a `latest.parquet` symlink.
    """
    candidates = sorted(directory.glob(f"{prefix}*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No Parquet files matching '{prefix}*.parquet' in {directory}. "
            "Run `sentinel ingest --dev` first."
        )
    return candidates[-1]


@dataclass
class QueryResult:
    """Column headers plus rows, so callers can render a table."""

    columns: list[str]
    rows: list[tuple[Any, ...]]


def run_named_query(parquet_path: Path, name: str) -> QueryResult:
    """Run one of the named descriptive queries against a Parquet file."""
    if name not in NAMED_QUERIES:
        raise KeyError(f"Unknown query '{name}'. Available: {', '.join(sorted(NAMED_QUERIES))}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    sql = NAMED_QUERIES[name]
    logger.info("Running query '%s' against %s", name, parquet_path)
    with connect() as conn:
        cursor = conn.execute(sql, [str(parquet_path)])
        columns = [d[0] for d in cursor.description or []]
        return QueryResult(columns=columns, rows=cursor.fetchall())


def describe(parquet_path: Path) -> QueryResult:
    """Column names and DuckDB-inferred types for a raw Parquet file."""
    return run_named_query(parquet_path, "schema")
