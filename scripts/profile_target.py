"""Read-only profiling of inspection outcomes, for target construction.

Analysis tooling, not library code, for the same reasons as
``scripts/profile_entities.py``: it answers one-off questions about a snapshot,
nothing imports it, and it should not ship in the wheel.

Unlike the entity profiler, this one joins Component 2's assignments, because
several questions ("does this establishment get inspected again after going out
of business?", "how often are there two canvasses on one day?") are only
answerable against resolved identities. It never re-derives identity.

Usage
-----
    uv run python scripts/profile_target.py
    uv run python scripts/profile_target.py --only results_value_set
    uv run python scripts/profile_target.py --list

Every query is a SELECT; the script writes nothing. Output is markdown, pasted
into ``docs/analysis/target_construction_findings.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.config import load_settings  # noqa: E402
from sentinel.query.duckdb_queries import latest_parquet  # noqa: E402

# Chicago adopted the FDA-style Priority / Priority Foundation / Core scheme on
# 2018-07-01. Before that the terminology was Critical / Serious. Measured, not
# assumed: see the era_* profiles below.
CODE_ERA_START = "2018-07-01"

# The four routine-inspection results. The other three observed values mean no
# inspection actually took place.
INSPECTED_RESULTS = "('Pass', 'Pass w/ Conditions', 'Fail')"

PROFILES: dict[str, str] = {
    # -- results ------------------------------------------------------------
    "results_value_set": """
        SELECT coalesce(results, '<NULL>') AS results,
               count(*) AS n,
               round(100.0 * count(*) / sum(count(*)) OVER (), 3) AS pct
        FROM j
        GROUP BY 1
        ORDER BY n DESC
    """,
    "results_anomalies": """
        SELECT
            count(*)                                                   AS total_rows,
            count(*) FILTER (WHERE results IS NULL)                    AS n_null,
            count(*) FILTER (WHERE results IS NOT NULL
                               AND trim(results) = '')                 AS n_blank,
            count(*) FILTER (WHERE results <> trim(results))           AS n_untrimmed,
            count(DISTINCT results)                                    AS n_distinct_raw,
            count(DISTINCT upper(trim(results)))                       AS n_distinct_norm
        FROM j
    """,
    # -- inspection type ----------------------------------------------------
    "inspection_type_value_set": """
        SELECT coalesce(nullif(trim(inspection_type), ''), '<BLANK>') AS inspection_type,
               count(*) AS n,
               round(100.0 * count(*) / sum(count(*)) OVER (), 3) AS pct
        FROM j
        GROUP BY 1
        ORDER BY n DESC
        LIMIT 30
    """,
    "inspection_type_cardinality": """
        SELECT
            count(DISTINCT inspection_type)                    AS n_distinct_raw,
            count(DISTINCT upper(trim(inspection_type)))       AS n_distinct_upper,
            count(*) FILTER (WHERE inspection_type IS NULL
                               OR trim(inspection_type) = '')  AS n_null_or_blank
        FROM j
    """,
    "canvass_family": """
        SELECT inspection_type, count(*) AS n,
               min(inspection_date)[1:10] AS first_seen,
               max(inspection_date)[1:10] AS last_seen
        FROM j
        WHERE upper(inspection_type) LIKE '%CANVAS%'
        GROUP BY 1
        ORDER BY n DESC
    """,
    "canvass_recognition_rule": f"""
        SELECT
            count(*) FILTER (WHERE inspection_type = 'Canvass')            AS exact_canvass,
            count(*) FILTER (WHERE upper(trim(inspection_type)) = 'CANVASS')
                                                                           AS norm_canvass,
            count(*) FILTER (WHERE inspection_type = 'Canvass Re-Inspection')
                                                                           AS canvass_reinsp,
            count(*) FILTER (WHERE upper(inspection_type) LIKE '%CANVAS%'
                               AND upper(trim(inspection_type)) <> 'CANVASS'
                               AND inspection_type <> 'Canvass Re-Inspection')
                                                                           AS other_variants
        FROM j
        WHERE inspection_date >= '{CODE_ERA_START}'
    """,
    "canvass_variant_tail": f"""
        SELECT inspection_type, count(*) AS n
        FROM j
        WHERE upper(inspection_type) LIKE '%CANVAS%'
          AND inspection_type NOT IN ('Canvass', 'Canvass Re-Inspection')
          AND inspection_date >= '{CODE_ERA_START}'
        GROUP BY 1
        ORDER BY n DESC
    """,
    # -- era comparability --------------------------------------------------
    "era_terminology_by_year": """
        SELECT inspection_date[1:4] AS yr,
               count(*) AS n,
               count(*) FILTER (WHERE violations IS NULL) AS violations_null,
               count(*) FILTER (WHERE upper(violations) LIKE '%PRIORITY%') AS has_priority_word,
               count(*) FILTER (WHERE upper(violations) LIKE '%SERIOUS VIOLATION%'
                                   OR upper(violations) LIKE '%CRITICAL VIOLATION%')
                                                                            AS has_old_terms
        FROM j
        GROUP BY 1
        ORDER BY 1
    """,
    "era_cutover_by_month": """
        SELECT inspection_date[1:7] AS month,
               count(*) AS n,
               count(*) FILTER (WHERE upper(violations) LIKE '%PRIORITY%') AS new_terminology,
               count(*) FILTER (WHERE upper(violations) LIKE '%SERIOUS VIOLATION%'
                                   OR upper(violations) LIKE '%CRITICAL VIOLATION%')
                                                                            AS old_terminology
        FROM j
        WHERE inspection_date >= '2018-01' AND inspection_date < '2019-01'
        GROUP BY 1
        ORDER BY 1
    """,
    # -- violations format --------------------------------------------------
    "violations_missingness": """
        SELECT
            count(*)                                                  AS total_rows,
            count(*) FILTER (WHERE violations IS NULL)                AS n_null,
            count(*) FILTER (WHERE violations IS NOT NULL
                               AND trim(violations) = '')             AS n_blank,
            round(100.0 * count(*) FILTER (WHERE violations IS NULL) / count(*), 2) AS pct_null,
            round(avg(length(violations)), 0)                         AS avg_length,
            max(length(violations))                                   AS max_length
        FROM j
    """,
    "violations_missing_by_result": f"""
        SELECT results,
               count(*) AS n,
               count(*) FILTER (WHERE violations IS NULL) AS n_null,
               round(100.0 * count(*) FILTER (WHERE violations IS NULL) / count(*), 1) AS pct_null
        FROM j
        WHERE inspection_date >= '{CODE_ERA_START}' AND inspection_type = 'Canvass'
        GROUP BY 1
        ORDER BY n DESC
    """,
    "violations_entry_counts": f"""
        WITH e AS (
            SELECT inspection_id, len(string_split(violations, '|')) AS n_entries
            FROM j
            WHERE violations IS NOT NULL AND inspection_date >= '{CODE_ERA_START}'
        )
        SELECT
            count(*)                          AS inspections,
            min(n_entries)                    AS min_entries,
            round(avg(n_entries), 2)          AS avg_entries,
            quantile_cont(n_entries, 0.5)     AS p50,
            quantile_cont(n_entries, 0.95)    AS p95,
            max(n_entries)                    AS max_entries
        FROM e
    """,
    "violations_entry_structure": f"""
        SELECT
            count(*)                                                        AS entries,
            count(*) FILTER (WHERE regexp_matches(trim(e), '^[0-9]+[.]'))   AS numbered,
            count(*) FILTER (WHERE u LIKE '%- COMMENTS:%')                  AS has_comments,
            count(*) FILTER (WHERE NOT regexp_matches(trim(e), '^[0-9]+[.]')
                               AND trim(e) <> '')                           AS unnumbered,
            count(*) FILTER (WHERE trim(e) = '')                            AS empty_entries
        FROM entries
        WHERE inspection_date >= '{CODE_ERA_START}'
    """,
    # -- priority classification --------------------------------------------
    "priority_marker_phrasings": f"""
        SELECT regexp_extract(u, '(PRIORITY[ A-Z]{{0,12}})', 1) AS phrase, count(*) AS n
        FROM entries
        WHERE inspection_date >= '{CODE_ERA_START}' AND u LIKE '%PRIORITY%'
        GROUP BY 1
        ORDER BY n DESC
        LIMIT 15
    """,
    "priority_by_violation_number": f"""
        SELECT num,
               count(*) AS entries,
               round(100.0 * count(*) FILTER (WHERE u LIKE '%PRIORITY FOUNDATION%')
                     / count(*), 1) AS pct_pf,
               round(100.0 * count(*) FILTER (WHERE u LIKE '%PRIORITY%'
                                          AND u NOT LIKE '%PRIORITY FOUNDATION%')
                     / count(*), 1) AS pct_priority,
               round(100.0 * count(*) FILTER (WHERE u NOT LIKE '%PRIORITY%')
                     / count(*), 1) AS pct_unlabelled
        FROM entries
        WHERE inspection_date >= '{CODE_ERA_START}' AND num IS NOT NULL
        GROUP BY 1
        ORDER BY entries DESC
        LIMIT 25
    """,
    "priority_code_anchoring": f"""
        SELECT
            count(*)                                                     AS entries_with_priority,
            count(*) FILTER (WHERE u LIKE '%7-38%')                      AS with_municipal_code,
            count(*) FILTER (WHERE u NOT LIKE '%7-38%')                  AS without_code,
            count(*) FILTER (WHERE u LIKE '%CITATION%')                  AS mentions_citation,
            count(*) FILTER (WHERE u LIKE '%NO CITATION%')               AS explicitly_no_citation
        FROM entries
        WHERE inspection_date >= '{CODE_ERA_START}' AND u LIKE '%PRIORITY%'
    """,
    "priority_narrative_candidates": f"""
        SELECT
            count(*) FILTER (WHERE u LIKE '%GRACE PERIOD%')       AS grace_period,
            count(*) FILTER (WHERE u LIKE '%WILL BE ISSUED%')     AS will_be_issued,
            count(*) FILTER (WHERE u LIKE '%MAY BE ISSUED%')      AS may_be_issued,
            count(*) FILTER (WHERE u LIKE '%IF NOT CORRECTED%')   AS if_not_corrected,
            count(*) FILTER (WHERE u LIKE '%COULD RESULT%')       AS could_result,
            count(*) FILTER (WHERE u LIKE '%NO PRIORITY%')        AS no_priority,
            count(*)                                              AS total_with_priority
        FROM entries
        WHERE inspection_date >= '{CODE_ERA_START}' AND u LIKE '%PRIORITY%'
    """,
    "core_marker_presence": f"""
        SELECT
            count(*)                                                       AS entries,
            count(*) FILTER (WHERE u LIKE '%PRIORITY FOUNDATION%')         AS priority_foundation,
            count(*) FILTER (WHERE u LIKE '%PRIORITY%'
                               AND u NOT LIKE '%PRIORITY FOUNDATION%')     AS priority_only,
            count(*) FILTER (WHERE u LIKE '%CORE%')                        AS core_word,
            count(*) FILTER (WHERE u NOT LIKE '%PRIORITY%'
                               AND u NOT LIKE '%CORE%')                    AS unlabelled
        FROM entries
        WHERE inspection_date >= '{CODE_ERA_START}'
    """,
    # -- results vs the candidate target ------------------------------------
    "results_vs_priority": f"""
        SELECT results,
               count(*) AS n,
               count(*) FILTER (WHERE violations IS NULL) AS violations_null,
               count(*) FILTER (WHERE upper(violations) LIKE '%PRIORITY%') AS with_priority,
               round(100.0 * count(*) FILTER (WHERE upper(violations) LIKE '%PRIORITY%')
                     / count(*), 1) AS pct_with_priority
        FROM j
        WHERE inspection_date >= '{CODE_ERA_START}'
        GROUP BY 1
        ORDER BY n DESC
    """,
    "canvass_results_vs_priority": f"""
        SELECT results,
               count(*) AS n,
               count(*) FILTER (WHERE violations IS NULL) AS violations_null,
               count(*) FILTER (WHERE upper(violations) LIKE '%PRIORITY%') AS with_priority,
               round(100.0 * count(*) FILTER (WHERE upper(violations) LIKE '%PRIORITY%')
                     / count(*), 1) AS pct_with_priority
        FROM j
        WHERE inspection_date >= '{CODE_ERA_START}' AND inspection_type = 'Canvass'
        GROUP BY 1
        ORDER BY n DESC
    """,
    "pass_with_priority_examples": f"""
        SELECT inspection_id,
               regexp_extract(upper(violations), '(.{{45}}PRIORITY.{{35}})', 1) AS context
        FROM j
        WHERE inspection_date >= '{CODE_ERA_START}'
          AND inspection_type = 'Canvass'
          AND results = 'Pass'
          AND upper(violations) LIKE '%PRIORITY%'
        ORDER BY inspection_id
        LIMIT 15
    """,
    # -- out of business ----------------------------------------------------
    "out_of_business_followup": """
        WITH oob AS (
            SELECT establishment_id, inspection_date
            FROM j WHERE results = 'Out of Business'
        ), nxt AS (
            SELECT o.establishment_id, o.inspection_date AS oob_date,
                   (SELECT min(k.inspection_date) FROM j k
                     WHERE k.establishment_id = o.establishment_id
                       AND k.inspection_date > o.inspection_date) AS next_date
            FROM oob o
        )
        SELECT count(*) AS oob_records,
               count(*) FILTER (WHERE next_date IS NOT NULL) AS followed_by_inspection,
               round(100.0 * count(*) FILTER (WHERE next_date IS NOT NULL) / count(*), 1) AS pct,
               round(median(TRY_CAST(next_date[1:10] AS DATE)
                          - TRY_CAST(oob_date[1:10] AS DATE)), 0) AS median_days_gap
        FROM nxt
    """,
    "out_of_business_violations": """
        SELECT results,
               count(*) AS n,
               count(*) FILTER (WHERE violations IS NULL) AS violations_null,
               round(100.0 * count(*) FILTER (WHERE violations IS NULL) / count(*), 1) AS pct_null
        FROM j
        WHERE results IN ('Out of Business', 'No Entry', 'Not Ready', 'Business Not Located')
        GROUP BY 1
        ORDER BY n DESC
    """,
    # -- same-day multiples -------------------------------------------------
    "same_day_all_types": f"""
        SELECT n_inspections, count(*) AS n_establishment_dates
        FROM (
            SELECT establishment_id, inspection_date, count(*) AS n_inspections
            FROM j WHERE inspection_date >= '{CODE_ERA_START}'
            GROUP BY 1, 2
        )
        GROUP BY 1
        ORDER BY 1
    """,
    "same_day_eligible_canvass": f"""
        SELECT n_inspections, count(*) AS n_establishment_dates
        FROM (
            SELECT establishment_id, inspection_date, count(*) AS n_inspections
            FROM j
            WHERE inspection_date >= '{CODE_ERA_START}'
              AND inspection_type = 'Canvass'
              AND results IN {INSPECTED_RESULTS}
            GROUP BY 1, 2
        )
        GROUP BY 1
        ORDER BY 1
    """,
    "same_day_disagreement": f"""
        WITH d AS (
            SELECT establishment_id, inspection_date,
                   count(*) AS n,
                   count(*) FILTER (WHERE upper(violations) LIKE '%PRIORITY%') AS n_pos
            FROM j
            WHERE inspection_date >= '{CODE_ERA_START}'
              AND inspection_type = 'Canvass'
              AND results IN {INSPECTED_RESULTS}
            GROUP BY 1, 2
            HAVING count(*) > 1
        )
        SELECT count(*) AS multi_canvass_days,
               count(*) FILTER (WHERE n_pos = 0)      AS all_negative,
               count(*) FILTER (WHERE n_pos = n)      AS all_positive,
               count(*) FILTER (WHERE n_pos > 0 AND n_pos < n) AS disagreeing
        FROM d
    """,
    # -- eligibility funnel and drift ---------------------------------------
    "eligibility_funnel": f"""
        SELECT
            count(*)                                                          AS all_rows,
            count(*) FILTER (WHERE inspection_date >= '{CODE_ERA_START}')     AS in_code_era,
            count(*) FILTER (WHERE inspection_date >= '{CODE_ERA_START}'
                               AND inspection_type = 'Canvass')               AS plus_canvass,
            count(*) FILTER (WHERE inspection_date >= '{CODE_ERA_START}'
                               AND inspection_type = 'Canvass'
                               AND results IN {INSPECTED_RESULTS})            AS plus_inspected,
            count(*) FILTER (WHERE inspection_date >= '{CODE_ERA_START}'
                               AND inspection_type = 'Canvass'
                               AND results IN {INSPECTED_RESULTS}
                               AND NOT (violations IS NULL
                                        AND results <> 'Pass'))               AS plus_parseable
        FROM j
    """,
    "positive_rate_by_year": f"""
        SELECT inspection_date[1:4] AS yr,
               count(*) AS eligible,
               count(*) FILTER (WHERE upper(violations) LIKE '%PRIORITY%') AS positive,
               round(100.0 * count(*) FILTER (WHERE upper(violations) LIKE '%PRIORITY%')
                     / count(*), 1) AS pct_positive
        FROM j
        WHERE inspection_date >= '{CODE_ERA_START}'
          AND inspection_type = 'Canvass'
          AND results IN {INSPECTED_RESULTS}
        GROUP BY 1
        ORDER BY 1
    """,
    "establishment_canvass_counts": f"""
        SELECT n_canvass, count(*) AS n_establishments
        FROM (
            SELECT establishment_id, count(*) AS n_canvass
            FROM j
            WHERE inspection_date >= '{CODE_ERA_START}'
              AND inspection_type = 'Canvass'
              AND results IN {INSPECTED_RESULTS}
            GROUP BY 1
        )
        GROUP BY 1
        ORDER BY 1
        LIMIT 20
    """,
    "inter_canvass_gap": f"""
        WITH c AS (
            SELECT establishment_id,
                   TRY_CAST(inspection_date[1:10] AS DATE) AS d,
                   lag(TRY_CAST(inspection_date[1:10] AS DATE))
                       OVER (PARTITION BY establishment_id ORDER BY inspection_date) AS prev_d
            FROM j
            WHERE inspection_date >= '{CODE_ERA_START}'
              AND inspection_type = 'Canvass'
              AND results IN {INSPECTED_RESULTS}
        )
        SELECT count(*) AS consecutive_pairs,
               quantile_cont(d - prev_d, 0.25) AS p25_days,
               quantile_cont(d - prev_d, 0.50) AS p50_days,
               quantile_cont(d - prev_d, 0.75) AS p75_days,
               max(d - prev_d)                 AS max_days
        FROM c WHERE prev_d IS NOT NULL
    """,
    "join_coverage": """
        SELECT
            (SELECT count(*) FROM src)  AS raw_rows,
            (SELECT count(*) FROM asg)  AS assignment_rows,
            (SELECT count(*) FROM j)    AS joined_rows,
            (SELECT count(*) FROM src) - (SELECT count(*) FROM j) AS unjoined
    """,
}


def render_markdown(columns: list[str], rows: list[tuple[Any, ...]]) -> str:
    """Render a result set as a GitHub-flavoured markdown table."""
    if not columns:
        return "_(no columns)_"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    if not rows:
        return "\n".join([header, divider, "_(no rows)_"])
    body = [
        "| " + " | ".join("" if v is None else str(v).replace("|", "\\|") for v in row) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def run_profile(conn: duckdb.DuckDBPyConnection, name: str) -> str:
    """Execute one profile and return it as a markdown section."""
    cursor = conn.execute(PROFILES[name])
    columns = [d[0] for d in cursor.description or []]
    rows = cursor.fetchall()
    return f"### `{name}`\n\n{render_markdown(columns, rows)}\n"


def prepare_views(conn: duckdb.DuckDBPyConnection, raw_path: Path, assignments_path: Path) -> None:
    """Register the raw snapshot, the Component 2 assignments, and helper views.

    ``j`` is the inspection-level join used by most profiles. ``entries`` is the
    exploded violation text, one row per ``|``-separated entry, which several
    profiles need. Paths go through the relation API because CREATE VIEW cannot
    take a bound parameter.
    """
    conn.read_parquet(str(raw_path)).create_view("src")
    conn.read_parquet(str(assignments_path)).create_view("asg")
    conn.execute("""
        CREATE VIEW j AS
        SELECT a.establishment_id, s.inspection_id, s.inspection_date, s.inspection_type,
               s.results, s.violations, s.facility_type
        FROM src s
        JOIN asg a USING (inspection_id)
    """)
    conn.execute("""
        CREATE VIEW entries AS
        SELECT j.inspection_id, j.inspection_date, trim(e) AS e, upper(trim(e)) AS u,
               TRY_CAST(regexp_extract(trim(e), '^([0-9]+)[.]', 1) AS INTEGER) AS num
        FROM j, unnest(string_split(j.violations, '|')) AS t(e)
        WHERE j.violations IS NOT NULL
    """)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="profile_target",
        description="Read-only profiling of inspection outcomes for target construction.",
    )
    parser.add_argument("--parquet", type=Path, help="Raw Parquet. Default: latest raw.")
    parser.add_argument(
        "--assignments", type=Path, help="Component 2 assignments. Default: latest."
    )
    parser.add_argument("--only", action="append", metavar="NAME", help="Run only these profiles.")
    parser.add_argument("--list", action="store_true", dest="list_profiles", help="List and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_profiles:
        for name in PROFILES:
            print(name)
        return 0

    settings = load_settings()
    raw_path: Path = args.parquet or latest_parquet(settings.food_inspections_raw_dir)
    assignments_path: Path = args.assignments or latest_parquet(
        settings.entity_resolution_interim_dir, prefix="establishment_assignments_"
    )

    selected: list[str] = list(args.only) if args.only else list(PROFILES)
    unknown = [n for n in selected if n not in PROFILES]
    if unknown:
        print(f"Unknown profile(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    conn = duckdb.connect(database=":memory:")
    try:
        prepare_views(conn, raw_path, assignments_path)
        print(
            f"<!-- generated by scripts/profile_target.py from {raw_path.name} "
            f"+ {assignments_path.name} -->\n"
        )
        for name in selected:
            print(run_profile(conn, name))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
