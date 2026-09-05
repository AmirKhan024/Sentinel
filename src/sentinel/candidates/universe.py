"""The operational candidate universe: which establishments, as of a planning date.

Builds two things from ``raw`` and ``assignments`` (both already registered on the
connection, exactly as ``features.build`` registers them before calling
``historical.compute_features``):

1. A ``targets``-shaped frame -- the same seven columns
   ``historical.TARGETS_SQL`` produces from Component 3's real target table --
   built instead from a planning date and each candidate's most recent prior
   record. This is the one artifact ``candidates.features`` needs to run
   Component 4's feature engine unmodified.

2. A metadata frame carrying what Component 4's contract deliberately excludes:
   the as-of location and display fields Component 20 (geographic planning,
   not part of this component) will eventually read.

Both come from one query, filtered once, by the same rule: only records with
``inspection_date < planning_date`` are ever read. That filter is the entire
leakage boundary for this module -- no later step re-reads ``raw`` or
``assignments`` directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import duckdb
import polars as pl

from sentinel.candidates.definitions import (
    CoverageWarning,
    synthetic_candidate_id,
)
from sentinel.target.construct import code_era_phase

logger = logging.getLogger(__name__)

#: Column order matches ``historical.TARGETS_SQL``'s output exactly, because
#: ``historical.aggregate_sql()`` reads these seven columns from a view named
#: ``targets`` and does not care how that view was built.
TARGETS_COLUMNS: tuple[str, ...] = (
    "establishment_id",
    "inspection_date",
    "target_inspection_id",
    "target",
    "target_status",
    "code_era_phase",
    "reference_dba_name",
)

METADATA_COLUMNS: tuple[str, ...] = (
    "establishment_id",
    "as_of_dba_name",
    "as_of_address",
    "as_of_zip",
    "as_of_latitude",
    "as_of_longitude",
    "has_location",
    "n_prior_records",
    "first_known_date",
    "last_known_date",
)


def _prior_sql(planning_date: str) -> str:
    """The as-of filter, parameterised on a planning date already validated as ISO-8601.

    Interpolated rather than bound: DuckDB does not support prepared parameters inside
    a ``CREATE VIEW`` statement (``Unexpected prepared parameter``). Safe here because
    ``planning_date`` has already passed ``date.fromisoformat`` -- it can only be a
    literal ``YYYY-MM-DD`` string, never arbitrary caller-supplied SQL.
    """
    return f"""
    SELECT
        a.establishment_id,
        TRY_CAST(r.inspection_date[1:10] AS DATE) AS record_date,
        r.dba_name,
        r.address,
        r.zip,
        TRY_CAST(r.latitude AS DOUBLE)  AS latitude,
        TRY_CAST(r.longitude AS DOUBLE) AS longitude
    FROM raw r
    JOIN assignments a USING (inspection_id)
    WHERE TRY_CAST(r.inspection_date[1:10] AS DATE) IS NOT NULL
      AND TRY_CAST(r.inspection_date[1:10] AS DATE) < DATE '{planning_date}'
    """


_BOUNDS_SQL = """
    SELECT
        min(TRY_CAST(inspection_date[1:10] AS DATE)),
        max(TRY_CAST(inspection_date[1:10] AS DATE))
    FROM raw
    WHERE TRY_CAST(inspection_date[1:10] AS DATE) IS NOT NULL
"""

_CANDIDATE_AGG_SQL = """
    CREATE OR REPLACE TABLE candidate_prior_agg AS
    SELECT
        establishment_id,
        count(*)                              AS n_prior_records,
        min(record_date)                      AS first_known_date,
        max(record_date)                      AS last_known_date,
        arg_max(dba_name, record_date)        AS as_of_dba_name,
        arg_max(address, record_date)         AS as_of_address,
        arg_max(zip, record_date)             AS as_of_zip,
        arg_max(latitude, record_date)        AS as_of_latitude,
        arg_max(longitude, record_date)       AS as_of_longitude
    FROM candidate_prior
    GROUP BY establishment_id
"""


class CandidateGenerationError(RuntimeError):
    """Raised when a planning date cannot be supported at all.

    Distinct from a :class:`~sentinel.candidates.definitions.CoverageWarning`: a
    warning describes a supportable-but-imperfect request (stale data). This is
    raised only when generating *any* candidate would require assuming
    information the data cannot supply -- a malformed date, or a date on or
    before the earliest record in the entire raw snapshot, for which the
    candidate pool is provably empty rather than merely small.
    """


@dataclass
class CandidateUniverse:
    """Everything ``candidates.features`` and ``candidates.writer`` need."""

    targets: pl.DataFrame
    metadata: pl.DataFrame
    min_raw_inspection_date: str
    max_raw_inspection_date: str
    warnings: list[str]


def _parse_planning_date(planning_date: str) -> date:
    try:
        return date.fromisoformat(planning_date)
    except ValueError as exc:
        raise CandidateGenerationError(
            f"planning_date {planning_date!r} is not a valid ISO date (YYYY-MM-DD)."
        ) from exc


def build_candidate_universe(
    conn: duckdb.DuckDBPyConnection, *, planning_date: str
) -> CandidateUniverse:
    """The candidate universe for one planning date.

    Expects ``raw`` and ``assignments`` already registered on ``conn``, exactly
    as ``features.build.build_features`` registers them. Reads only records with
    ``inspection_date < planning_date``; the planning date itself and every date
    after it are never referenced.
    """
    parsed = _parse_planning_date(planning_date)
    normalized = parsed.isoformat()

    bounds = conn.execute(_BOUNDS_SQL).fetchone()
    min_raw, max_raw = (bounds[0], bounds[1]) if bounds else (None, None)
    if min_raw is None:
        raise CandidateGenerationError(
            "The raw snapshot has no parseable inspection_date values; no planning "
            "date can be supported."
        )
    min_raw_s, max_raw_s = str(min_raw), str(max_raw)

    if normalized <= min_raw_s:
        raise CandidateGenerationError(
            f"planning_date {normalized} is not supportable: the raw snapshot's "
            f"earliest parseable inspection record is {min_raw_s}, so no establishment "
            f"can have any record strictly before {normalized}. The earliest "
            f"supportable planning date is the day after {min_raw_s}."
        )

    warnings: list[str] = []
    days_beyond = (parsed - date.fromisoformat(max_raw_s)).days
    if days_beyond > 0:
        warnings.append(
            f"{CoverageWarning.PLANNING_DATE_BEYOND_INGESTED_DATA.value}: raw data is "
            f"ingested through {max_raw_s}; planning_date {normalized} is {days_beyond} "
            "day(s) beyond that. Candidates reflect the state of the last ingest, not a "
            "live feed -- run `sentinel ingest` again first for a more current snapshot."
        )

    conn.execute("CREATE OR REPLACE VIEW candidate_prior AS " + _prior_sql(normalized))
    conn.execute(_CANDIDATE_AGG_SQL)
    agg = conn.sql("SELECT * FROM candidate_prior_agg ORDER BY establishment_id").pl()

    if agg.height == 0:
        warnings.append(CoverageWarning.NO_CANDIDATES_FOUND.value)
        logger.warning("No candidates found for planning_date=%s", normalized)
        return CandidateUniverse(
            targets=pl.DataFrame(
                schema={
                    "establishment_id": pl.Utf8,
                    "inspection_date": pl.Date,
                    "target_inspection_id": pl.Utf8,
                    "target": pl.Int8,
                    "target_status": pl.Utf8,
                    "code_era_phase": pl.Utf8,
                    "reference_dba_name": pl.Utf8,
                }
            ),
            metadata=pl.DataFrame(
                schema={
                    "establishment_id": pl.Utf8,
                    "as_of_dba_name": pl.Utf8,
                    "as_of_address": pl.Utf8,
                    "as_of_zip": pl.Utf8,
                    "as_of_latitude": pl.Float64,
                    "as_of_longitude": pl.Float64,
                    "has_location": pl.Boolean,
                    "n_prior_records": pl.Int64,
                    "first_known_date": pl.Utf8,
                    "last_known_date": pl.Utf8,
                }
            ),
            min_raw_inspection_date=min_raw_s,
            max_raw_inspection_date=max_raw_s,
            warnings=warnings,
        )

    n = agg.height
    # Every candidate row shares one planning date, so the code-era phase (a pure
    # function of the reference date alone -- see target.construct.code_era_phase)
    # is identical for all of them. Computed once rather than per row.
    phase = code_era_phase(normalized).value

    targets = pl.DataFrame(
        {
            "establishment_id": agg["establishment_id"],
            "inspection_date": pl.Series([parsed] * n, dtype=pl.Date),
            "target_inspection_id": pl.Series(
                [
                    synthetic_candidate_id(planning_date=normalized, establishment_id=est)
                    for est in agg["establishment_id"].to_list()
                ],
                dtype=pl.Utf8,
            ),
            "target": pl.Series([None] * n, dtype=pl.Int8),
            "target_status": pl.Series(["operational_candidate"] * n, dtype=pl.Utf8),
            "code_era_phase": pl.Series([phase] * n, dtype=pl.Utf8),
            "reference_dba_name": agg["as_of_dba_name"],
        }
    )

    has_location = agg["as_of_latitude"].is_not_null() & agg["as_of_longitude"].is_not_null()
    missing_location = int((~has_location).sum())
    if missing_location:
        warnings.append(
            f"{CoverageWarning.CANDIDATES_MISSING_LOCATION.value}: {missing_location} of "
            f"{n} candidates have no resolvable as-of coordinates."
        )

    metadata = pl.DataFrame(
        {
            "establishment_id": agg["establishment_id"],
            "as_of_dba_name": agg["as_of_dba_name"],
            "as_of_address": agg["as_of_address"],
            "as_of_zip": agg["as_of_zip"],
            "as_of_latitude": agg["as_of_latitude"].cast(pl.Float64),
            "as_of_longitude": agg["as_of_longitude"].cast(pl.Float64),
            "has_location": has_location,
            "n_prior_records": agg["n_prior_records"].cast(pl.Int64),
            "first_known_date": agg["first_known_date"].cast(pl.Utf8),
            "last_known_date": agg["last_known_date"].cast(pl.Utf8),
        }
    )

    logger.info(
        "Built %d operational candidates for planning_date=%s (%d missing location)",
        n,
        normalized,
        missing_location,
    )
    return CandidateUniverse(
        targets=targets,
        metadata=metadata,
        min_raw_inspection_date=min_raw_s,
        max_raw_inspection_date=max_raw_s,
        warnings=warnings,
    )


__all__ = [
    "METADATA_COLUMNS",
    "TARGETS_COLUMNS",
    "CandidateGenerationError",
    "CandidateUniverse",
    "build_candidate_universe",
]
