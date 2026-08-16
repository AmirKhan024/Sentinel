"""As-of historical aggregation.

Every feature is computed from one range join with an explicit temporal
condition:

    FROM targets t
    LEFT JOIN history h
      ON  h.establishment_id = t.establishment_id
      AND h.inspection_date  <  t.inspection_date

That single condition is the whole component. It is written once, in one query,
so there is exactly one place to get it right -- and it is re-derived
independently in ``validate.py`` so a bug here would be caught rather than
shipped.

Three things this deliberately is **not**:

* Not ``groupby(establishment_id)`` followed by a merge. That computes
  whole-history statistics and attaches future information to past rows -- the
  classic all-history bug.
* Not a Python loop over targets x history. That is O(N x M) and, at 57,727
  targets against 314,245 inspections, would be unusable.
* Not ``<=``. Findings §4: ``inspection_date`` has exactly one distinct time
  component across all 314,245 rows, so same-day records cannot be ordered, and
  43 same-day canvass re-inspections at reference dates happened *after* the
  canvass they follow.

``LEFT JOIN`` rather than ``JOIN`` so that the 401 history-less target rows
survive as rows with NULL/0 features instead of vanishing.

The priority flags are computed by **Component 3's own parser**, not by a SQL
approximation. A ``LIKE '%PRIORITY%'`` would silently drift from the target's
definition, which excludes narrative spans such as the 90-day grace-period
boilerplate. Reusing the parser guarantees that "priority" means the same thing
on both sides of the training row.
"""

from __future__ import annotations

import logging

import duckdb
import polars as pl

from sentinel.features.definitions import WINDOW_DAYS
from sentinel.target.models import CODE_ERA_START, INSPECTED_RESULTS, Severity
from sentinel.target.violations import parse_violations

logger = logging.getLogger(__name__)

# Inspection-type predicates, evaluated against the uppercased, whitespace-
# collapsed type so casing variants cannot silently drop a record. Component 3
# measured 111 distinct raw values.
_CANVASS = "h.type_key = 'CANVASS'"
_COMPLAINT = "h.type_key LIKE '%COMPLAINT%'"
_REINSPECTION = "(h.type_key LIKE '%RE-INSPECTION%' OR h.type_key LIKE '%REINSPECTION%')"
_LICENSE = "h.type_key LIKE 'LICENSE%'"

_INSPECTED_SQL = ", ".join(f"'{r}'" for r in sorted(INSPECTED_RESULTS))


def priority_flags(frame: pl.DataFrame) -> pl.DataFrame:
    """Classify inspections' violation text with Component 3's own parser.

    Returns one row per ``inspection_id`` with two booleans. Applied to *all*
    inspection types, not just canvasses: a prior complaint that found a priority
    violation has no target row, but it is still genuine prior evidence about the
    premises.

    Only code-era rows are parsed. Priority and Priority Foundation did not exist
    before 2018-07-01 (ADR 0009), so a pre-code row's flag is ``False`` by
    definition rather than by measurement -- parsing it would burn time to
    rediscover that the words are absent. That halves the work: 141,366 code-era
    rows against 314,245 total.

    Expects ``inspection_id``, ``violations`` and ``inspection_date``.
    """
    ids: list[str] = []
    priority: list[bool] = []
    foundation: list[bool] = []

    for inspection_id, violations, date in zip(
        frame["inspection_id"].to_list(),
        frame["violations"].to_list(),
        frame["inspection_date"].to_list(),
        strict=True,
    ):
        ids.append(str(inspection_id))
        if not date or str(date)[:10] < CODE_ERA_START:
            priority.append(False)
            foundation.append(False)
            continue
        entries = parse_violations(violations)
        priority.append(any(e.is_priority for e in entries))
        foundation.append(any(e.severity is Severity.PRIORITY_FOUNDATION for e in entries))

    return pl.DataFrame(
        {
            "inspection_id": ids,
            "has_priority": priority,
            "has_priority_foundation": foundation,
        },
        schema={
            "inspection_id": pl.Utf8,
            "has_priority": pl.Boolean,
            "has_priority_foundation": pl.Boolean,
        },
    )


HISTORY_SQL = f"""
    CREATE OR REPLACE VIEW history AS
    SELECT
        a.establishment_id,
        TRY_CAST(r.inspection_date[1:10] AS DATE)                     AS inspection_date,
        regexp_replace(upper(trim(r.inspection_type)), '\\s+', ' ', 'g') AS type_key,
        r.results,
        r.dba_name,
        (r.inspection_date[1:10] >= '{CODE_ERA_START}')               AS is_code_era,
        (r.results IN ({_INSPECTED_SQL}))                             AS was_inspected,
        coalesce(f.has_priority, FALSE)                               AS has_priority,
        coalesce(f.has_priority_foundation, FALSE)          AS has_priority_foundation
    FROM raw r
    JOIN assignments a USING (inspection_id)
    LEFT JOIN flags f USING (inspection_id)
    WHERE TRY_CAST(r.inspection_date[1:10] AS DATE) IS NOT NULL
"""

# The reference inspection's own attributes, needed for the tenant-change
# comparison. Read from the raw row the target points at, because Component 3's
# table deliberately carries no name.
TARGETS_SQL = """
    CREATE OR REPLACE VIEW targets AS
    SELECT
        t.establishment_id,
        TRY_CAST(t.inspection_date[1:10] AS DATE) AS inspection_date,
        t.target_inspection_id,
        t.target,
        t.target_status,
        t.code_era_phase,
        r.dba_name AS reference_dba_name
    FROM target_rows t
    JOIN raw r ON r.inspection_id = t.target_inspection_id
    WHERE t.target_status = 'eligible'
"""


def _window_columns() -> str:
    """Per-window aggregate columns, paired so an empty window stays legible."""
    parts: list[str] = []
    for days in WINDOW_DAYS:
        in_window = f"h.inspection_date >= t.inspection_date - INTERVAL {days} DAY"
        parts.append(
            f"count(*) FILTER (WHERE {_CANVASS} AND {in_window}) AS canvasses_last_{days}d"
        )
        parts.append(
            f"count(*) FILTER (WHERE {_CANVASS} AND h.is_code_era AND h.has_priority"
            f" AND {in_window}) AS canvass_priority_events_last_{days}d"
        )
    return ",\n        ".join(parts)


def aggregate_sql() -> str:
    """The single as-of aggregation.

    Every aggregate below is evaluated only over rows the join already restricted
    to ``inspection_date < reference``, so the temporal guarantee is structural
    rather than repeated per feature.

    ``arg_max`` picks the value at the most recent qualifying row, which is how
    the "at last canvass" features are computed without a second pass.

    Materialized as a TABLE rather than a VIEW: ``validate.py`` runs dozens of
    small checks against the result, and against a view each one would re-execute
    the whole range join. Measured at 2m14s as a view against 6s as a table.
    """
    return f"""
    CREATE OR REPLACE TABLE aggregated AS
    SELECT
        t.establishment_id,
        t.inspection_date,
        t.target_inspection_id,
        t.target,
        t.target_status,
        t.code_era_phase,
        t.reference_dba_name,

        count(*) FILTER (WHERE {_CANVASS})                    AS prior_canvass_count,
        count(*) FILTER (WHERE {_CANVASS} AND h.is_code_era)
                                                      AS prior_canvass_count_code_era,
        count(*) FILTER (WHERE {_CANVASS} AND h.was_inspected)
                                                      AS prior_canvass_inspected_count,
        count(h.establishment_id)                     AS prior_inspection_count_any_type,
        count(*) FILTER (WHERE {_COMPLAINT})                  AS prior_complaint_count,
        count(*) FILTER (WHERE {_REINSPECTION})            AS prior_reinspection_count,
        count(*) FILTER (WHERE {_LICENSE})         AS prior_license_inspection_count,
        count(*) FILTER (WHERE {_CANVASS}
                           AND h.dba_name IS NOT DISTINCT FROM t.reference_dba_name)
                                                    AS prior_canvass_count_current_name,

        count(*) FILTER (WHERE {_CANVASS} AND h.was_inspected AND h.results = 'Fail')
                                                             AS prior_canvass_fail_count,
        count(*) FILTER (WHERE {_CANVASS} AND h.was_inspected
                           AND h.results = 'Pass w/ Conditions')
                                              AS prior_canvass_pass_w_conditions_count,

        count(*) FILTER (WHERE {_CANVASS} AND h.is_code_era AND h.has_priority)
                                                         AS prior_canvass_priority_count,
        count(*) FILTER (WHERE {_CANVASS} AND h.is_code_era AND h.has_priority_foundation)
                                              AS prior_canvass_priority_foundation_count,

        {_window_columns()},

        max(h.inspection_date) FILTER (WHERE {_CANVASS})      AS last_canvass_date,
        max(h.inspection_date)                                AS last_any_date,
        min(h.inspection_date)                                AS first_any_date,

        arg_max(h.results, h.inspection_date) FILTER (WHERE {_CANVASS})
                                                              AS last_canvass_results,
        arg_max(h.dba_name, h.inspection_date) FILTER (WHERE {_CANVASS})
                                                              AS last_canvass_dba_name,
        arg_max(h.has_priority, h.inspection_date)
            FILTER (WHERE {_CANVASS} AND h.is_code_era)        AS last_code_era_priority
    FROM targets t
    LEFT JOIN history h
      ON  h.establishment_id = t.establishment_id
      AND h.inspection_date  <  t.inspection_date   -- THE temporal boundary
    GROUP BY
        t.establishment_id, t.inspection_date, t.target_inspection_id,
        t.target, t.target_status, t.code_era_phase, t.reference_dba_name
    """


def features_sql() -> str:
    """Derive the final feature columns from the aggregates.

    Kept separate from the aggregation so the missing-value rules are visible in
    one place. Each ``CASE`` implements one rule from findings §7: recency is
    NULL when the event never happened; rates are NULL when the denominator is
    0; at-last flags are NULL when there is no prior event; counts pass through
    unchanged, because 0 is a true observation.
    """
    window_cols = ",\n        ".join(
        f"canvasses_last_{d}d,\n        canvass_priority_events_last_{d}d" for d in WINDOW_DAYS
    )
    return f"""
    CREATE OR REPLACE TABLE features AS
    SELECT
        establishment_id,
        strftime(inspection_date, '%Y-%m-%d')                        AS inspection_date,
        target_inspection_id,
        target,
        target_status,
        code_era_phase,

        prior_canvass_count::INTEGER                                 AS prior_canvass_count,
        prior_canvass_count_code_era::INTEGER              AS prior_canvass_count_code_era,
        prior_canvass_inspected_count::INTEGER            AS prior_canvass_inspected_count,

        CASE WHEN last_canvass_date IS NULL THEN NULL
             ELSE date_diff('day', last_canvass_date, inspection_date)::INTEGER
        END AS days_since_last_canvass,
        CASE WHEN last_any_date IS NULL THEN NULL
             ELSE date_diff('day', last_any_date, inspection_date)::INTEGER
        END AS days_since_any_inspection,
        CASE WHEN first_any_date IS NULL THEN NULL
             ELSE date_diff('day', first_any_date, inspection_date)::INTEGER
        END AS days_since_first_inspection,

        prior_canvass_fail_count::INTEGER                       AS prior_canvass_fail_count,
        prior_canvass_pass_w_conditions_count::INTEGER
                                              AS prior_canvass_pass_w_conditions_count,

        CASE WHEN prior_canvass_inspected_count = 0 THEN NULL
             ELSE prior_canvass_fail_count::DOUBLE / prior_canvass_inspected_count
        END AS prior_canvass_fail_rate,

        CASE WHEN last_canvass_results IS NULL THEN NULL
             ELSE last_canvass_results = 'Fail'
        END AS fail_at_last_canvass,

        CASE WHEN prior_canvass_count_code_era = 0 THEN NULL
             ELSE prior_canvass_priority_count::INTEGER
        END AS prior_canvass_priority_count,
        CASE WHEN prior_canvass_count_code_era = 0 THEN NULL
             ELSE prior_canvass_priority_foundation_count::INTEGER
        END AS prior_canvass_priority_foundation_count,
        CASE WHEN prior_canvass_count_code_era = 0 THEN NULL
             ELSE prior_canvass_priority_count::DOUBLE / prior_canvass_count_code_era
        END AS prior_canvass_priority_rate,
        CASE WHEN prior_canvass_count_code_era = 0 THEN NULL
             ELSE coalesce(last_code_era_priority, FALSE)
        END AS priority_at_last_canvass,

        prior_inspection_count_any_type::INTEGER        AS prior_inspection_count_any_type,
        prior_complaint_count::INTEGER                           AS prior_complaint_count,
        prior_reinspection_count::INTEGER                     AS prior_reinspection_count,
        prior_license_inspection_count::INTEGER         AS prior_license_inspection_count,

        CASE WHEN last_canvass_dba_name IS NULL THEN NULL
             ELSE last_canvass_dba_name IS DISTINCT FROM reference_dba_name
        END AS name_changed_since_last_canvass,
        prior_canvass_count_current_name::INTEGER      AS prior_canvass_count_current_name,

        {window_cols}
    FROM aggregated
    """


def compute_features(conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyRelation:
    """Run the as-of pipeline and return the feature relation, canonically sorted.

    Expects ``raw``, ``assignments``, ``target_rows`` and ``flags`` to already be
    registered on the connection.
    """
    conn.execute(HISTORY_SQL)
    conn.execute(TARGETS_SQL)
    conn.execute(aggregate_sql())
    conn.execute(features_sql())
    logger.info("Computed as-of features (boundary: inspection_date < reference date)")
    return conn.sql(
        "SELECT * FROM features ORDER BY establishment_id, inspection_date, target_inspection_id"
    )
