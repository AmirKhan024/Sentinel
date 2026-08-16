"""Read-only profiling of the historical information available at prediction time.

Analysis tooling, not library code, for the same reasons as the other two profilers: it
answers one-off questions about a snapshot, nothing imports it, and it should not ship in
the wheel.

This one joins all three upstream artifacts — the raw snapshot, Component 2's assignments
and Component 3's targets — because the question it answers is "for each prediction
opportunity, what history existed before it?" That question is meaningless without all
three.

Every query is a SELECT and the script writes nothing. Output is markdown, pasted into
``docs/analysis/as_of_feature_engineering_findings.md``.

Usage
-----
    uv run python scripts/profile_features.py
    uv run python scripts/profile_features.py --only history_depth_any_type
    uv run python scripts/profile_features.py --list

Note on the temporal condition: every profile below that looks backwards uses
``h.d < t.rd`` — strictly less than. That is the boundary Component 4 adopts, and using it
here means the profile numbers describe exactly the history the features will see.
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

# Chicago adopted the Priority / Priority Foundation / Core scheme on this date (ADR 0009).
# Priority history is only defined at or after it.
CODE_ERA_START = "2018-07-01"

PROFILES: dict[str, str] = {
    # -- what history exists at all ----------------------------------------
    "history_depth_any_type": """
        WITH x AS (
            SELECT t.tid,
                   (SELECT count(*) FROM hist h
                     WHERE h.est = t.est AND h.d < t.rd) AS n
            FROM t
        )
        SELECT CASE WHEN n = 0 THEN 'a. none'
                    WHEN n <= 2 THEN 'b. 1-2'
                    WHEN n <= 5 THEN 'c. 3-5'
                    WHEN n <= 10 THEN 'd. 6-10'
                    WHEN n <= 20 THEN 'e. 11-20'
                    ELSE 'f. 21+' END AS prior_inspections,
               count(*) AS target_rows,
               round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
        FROM x GROUP BY 1 ORDER BY 1
    """,
    "history_depth_canvass": """
        WITH x AS (
            SELECT t.tid,
                   (SELECT count(*) FROM hist h
                     WHERE h.est = t.est AND h.d < t.rd AND h.typ = 'Canvass') AS n
            FROM t
        )
        SELECT CASE WHEN n = 0 THEN 'a. none'
                    WHEN n <= 2 THEN 'b. 1-2'
                    WHEN n <= 5 THEN 'c. 3-5'
                    WHEN n <= 10 THEN 'd. 6-10'
                    ELSE 'e. 11+' END AS prior_canvasses,
               count(*) AS target_rows,
               round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
        FROM x GROUP BY 1 ORDER BY 1
    """,
    "history_availability_summary": f"""
        SELECT count(*) AS target_rows,
               count(*) FILTER (WHERE n_any = 0)      AS no_history_at_all,
               count(*) FILTER (WHERE n_canv = 0)     AS no_prior_canvass,
               count(*) FILTER (WHERE n_canv_era = 0) AS no_prior_code_era_canvass,
               count(*) FILTER (WHERE n_pre > 0)      AS has_pre_code_history,
               round(avg(n_any), 2)  AS avg_prior_any,
               round(avg(n_canv), 2) AS avg_prior_canvass
        FROM (
            SELECT t.tid,
                   count(h.est)                                              AS n_any,
                   count(*) FILTER (WHERE h.typ = 'Canvass')                 AS n_canv,
                   count(*) FILTER (WHERE h.typ = 'Canvass'
                                      AND h.d >= DATE '{CODE_ERA_START}')    AS n_canv_era,
                   count(*) FILTER (WHERE h.d < DATE '{CODE_ERA_START}')     AS n_pre
            FROM t LEFT JOIN hist h ON h.est = t.est AND h.d < t.rd
            GROUP BY t.tid
        )
    """,
    # -- the boundary question ---------------------------------------------
    "intraday_time_component": """
        SELECT count(DISTINCT inspection_date[11:]) AS distinct_time_parts,
               any_value(inspection_date[11:])      AS the_only_value,
               count(*)                             AS rows
        FROM src
    """,
    "same_day_composition": """
        SELECT h.typ AS inspection_type_on_reference_date,
               count(*) AS occurrences
        FROM t JOIN hist h ON h.est = t.est AND h.d = t.rd
        GROUP BY 1 ORDER BY occurrences DESC
    """,
    "same_day_row_counts": """
        SELECT n_same_day, count(*) AS target_rows
        FROM (
            SELECT t.tid, count(*) AS n_same_day
            FROM t JOIN hist h ON h.est = t.est AND h.d = t.rd
            GROUP BY t.tid
        )
        GROUP BY 1 ORDER BY 1
    """,
    # -- cadence, which sizes the windows ----------------------------------
    "inter_canvass_interval": """
        WITH c AS (
            SELECT est, d,
                   lag(d) OVER (PARTITION BY est ORDER BY d) AS prev_d
            FROM hist WHERE typ = 'Canvass'
        )
        SELECT count(*) AS consecutive_pairs,
               quantile_cont(d - prev_d, 0.25) AS p25_days,
               quantile_cont(d - prev_d, 0.50) AS p50_days,
               quantile_cont(d - prev_d, 0.75) AS p75_days,
               quantile_cont(d - prev_d, 0.90) AS p90_days,
               max(d - prev_d)                 AS max_days
        FROM c WHERE prev_d IS NOT NULL
    """,
    "inter_any_inspection_interval": """
        WITH c AS (
            SELECT est, d,
                   lag(d) OVER (PARTITION BY est ORDER BY d) AS prev_d
            FROM hist
        )
        SELECT count(*) AS consecutive_pairs,
               quantile_cont(d - prev_d, 0.25) AS p25_days,
               quantile_cont(d - prev_d, 0.50) AS p50_days,
               quantile_cont(d - prev_d, 0.75) AS p75_days,
               quantile_cont(d - prev_d, 0.90) AS p90_days
        FROM c WHERE prev_d IS NOT NULL
    """,
    "window_occupancy": """
        SELECT window_days,
               count(*) AS target_rows,
               count(*) FILTER (WHERE n = 0) AS rows_with_empty_window,
               round(100.0 * count(*) FILTER (WHERE n = 0) / count(*), 1) AS pct_empty,
               round(avg(n), 2) AS avg_canvasses_in_window
        FROM (
            SELECT 365 AS window_days, t.tid,
                   count(*) FILTER (WHERE h.typ = 'Canvass'
                                      AND h.d >= t.rd - INTERVAL 365 DAY) AS n
            FROM t LEFT JOIN hist h ON h.est = t.est AND h.d < t.rd GROUP BY t.tid
            UNION ALL
            SELECT 730, t.tid,
                   count(*) FILTER (WHERE h.typ = 'Canvass'
                                      AND h.d >= t.rd - INTERVAL 730 DAY)
            FROM t LEFT JOIN hist h ON h.est = t.est AND h.d < t.rd GROUP BY t.tid
            UNION ALL
            SELECT 1095, t.tid,
                   count(*) FILTER (WHERE h.typ = 'Canvass'
                                      AND h.d >= t.rd - INTERVAL 1095 DAY)
            FROM t LEFT JOIN hist h ON h.est = t.est AND h.d < t.rd GROUP BY t.tid
        )
        GROUP BY window_days ORDER BY window_days
    """,
    # -- what the history contains -----------------------------------------
    "history_type_mix": """
        SELECT h.typ AS inspection_type, count(*) AS prior_records
        FROM t JOIN hist h ON h.est = t.est AND h.d < t.rd
        GROUP BY 1 ORDER BY prior_records DESC LIMIT 15
    """,
    "history_result_mix": """
        SELECT h.res AS results, count(*) AS prior_records
        FROM t JOIN hist h ON h.est = t.est AND h.d < t.rd
        GROUP BY 1 ORDER BY prior_records DESC
    """,
    "history_canvass_result_mix": """
        SELECT h.res AS results, count(*) AS prior_canvasses
        FROM t JOIN hist h ON h.est = t.est AND h.d < t.rd AND h.typ = 'Canvass'
        GROUP BY 1 ORDER BY prior_canvasses DESC
    """,
    "priority_history_availability": f"""
        SELECT count(*) AS target_rows,
               count(*) FILTER (WHERE n_era = 0) AS priority_features_null,
               round(100.0 * count(*) FILTER (WHERE n_era = 0) / count(*), 1) AS pct_null,
               count(*) FILTER (WHERE n_era > 0 AND n_pri = 0) AS genuine_zero_priority,
               count(*) FILTER (WHERE n_pri > 0) AS has_prior_priority
        FROM (
            SELECT t.tid,
                   count(*) FILTER (WHERE h.typ = 'Canvass'
                                      AND h.d >= DATE '{CODE_ERA_START}') AS n_era,
                   count(*) FILTER (WHERE h.typ = 'Canvass'
                                      AND h.d >= DATE '{CODE_ERA_START}'
                                      AND h.priority) AS n_pri
            FROM t LEFT JOIN hist h ON h.est = t.est AND h.d < t.rd
            GROUP BY t.tid
        )
    """,
    # -- tenant change (the spec vs Component 2 divergence) ----------------
    "tenant_change_scale": """
        SELECT CASE WHEN e.n_names = 1 THEN 'a. one name'
                    WHEN e.n_names = 2 THEN 'b. two names'
                    ELSE 'c. three or more' END AS premises_names,
               count(*) AS establishments,
               sum(e.n_inspections) AS inspections
        FROM est e GROUP BY 1 ORDER BY 1
    """,
    "tenant_change_target_rows": """
        SELECT CASE WHEN e.n_names > 1 THEN 'multi-name premises'
                    ELSE 'single-name premises' END AS kind,
               count(*) AS target_rows,
               round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
        FROM t JOIN est e ON e.establishment_id = t.est
        GROUP BY 1 ORDER BY target_rows DESC
    """,
    "name_change_between_consecutive_canvasses": """
        WITH c AS (
            SELECT est, d, nm,
                   lag(nm) OVER (PARTITION BY est ORDER BY d) AS prev_nm
            FROM hist WHERE typ = 'Canvass'
        )
        SELECT count(*) AS consecutive_canvass_pairs,
               count(*) FILTER (WHERE nm <> prev_nm) AS with_name_change,
               round(100.0 * count(*) FILTER (WHERE nm <> prev_nm) / count(*), 2) AS pct
        FROM c WHERE prev_nm IS NOT NULL
    """,
    # -- data quality ------------------------------------------------------
    "date_quality": """
        SELECT count(*) AS rows,
               count(*) FILTER (WHERE inspection_date IS NULL)            AS null_dates,
               count(*) FILTER (WHERE TRY_CAST(inspection_date[1:10] AS DATE) IS NULL)
                                                                          AS unparseable_dates,
               min(inspection_date[1:10])                                 AS earliest,
               max(inspection_date[1:10])                                 AS latest
        FROM src
    """,
    "duplicate_inspection_ids": """
        SELECT count(*) AS rows,
               count(DISTINCT inspection_id) AS distinct_ids,
               count(*) - count(DISTINCT inspection_id) AS duplicates
        FROM src
    """,
    "duplicate_establishment_dates_in_history": """
        SELECT n_on_date, count(*) AS establishment_dates
        FROM (SELECT est, d, count(*) AS n_on_date FROM hist GROUP BY 1, 2)
        GROUP BY 1 ORDER BY 1 LIMIT 12
    """,
    "join_coverage": """
        SELECT (SELECT count(*) FROM src)  AS raw_rows,
               (SELECT count(*) FROM hist) AS history_rows,
               (SELECT count(*) FROM tgt)  AS target_rows_all,
               (SELECT count(*) FROM t)    AS target_rows_eligible
    """,
    "observation_window": """
        SELECT count(*) AS target_rows,
               count(*) FILTER (WHERE days IS NULL) AS no_prior_record,
               round(quantile_cont(days, 0.25), 0) AS p25_days,
               round(quantile_cont(days, 0.50), 0) AS p50_days,
               round(quantile_cont(days, 0.75), 0) AS p75_days,
               max(days) AS max_days
        FROM (
            SELECT t.tid, date_diff('day', min(h.d), t.rd) AS days
            FROM t LEFT JOIN hist h ON h.est = t.est AND h.d < t.rd
            GROUP BY t.tid, t.rd
        )
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


def prepare_views(
    conn: duckdb.DuckDBPyConnection,
    raw_path: Path,
    assignments_path: Path,
    establishments_path: Path,
    targets_path: Path,
) -> None:
    """Register the three upstream artifacts plus the two helper views.

    ``hist`` is every inspection with a parsed date, its establishment, and a
    pre-computed priority flag. ``t`` is the eligible target rows reduced to the
    three things a backward-looking query needs: establishment, reference date,
    and the row's id.

    Paths go through the relation API because CREATE VIEW cannot bind a parameter.
    """
    conn.read_parquet(str(raw_path)).create_view("src")
    conn.read_parquet(str(assignments_path)).create_view("asg")
    conn.read_parquet(str(establishments_path)).create_view("est")
    conn.read_parquet(str(targets_path)).create_view("tgt")
    conn.execute(f"""
        CREATE VIEW hist AS
        SELECT a.establishment_id AS est,
               TRY_CAST(s.inspection_date[1:10] AS DATE) AS d,
               s.inspection_type AS typ,
               s.results AS res,
               s.dba_name AS nm,
               s.license_ AS lic,
               (s.inspection_date[1:10] >= '{CODE_ERA_START}'
                AND upper(s.violations) LIKE '%PRIORITY%') AS priority
        FROM src s
        JOIN asg a USING (inspection_id)
        WHERE TRY_CAST(s.inspection_date[1:10] AS DATE) IS NOT NULL
    """)
    conn.execute("""
        CREATE VIEW t AS
        SELECT establishment_id AS est,
               TRY_CAST(inspection_date[1:10] AS DATE) AS rd,
               target_inspection_id AS tid
        FROM tgt
        WHERE target_status = 'eligible'
    """)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="profile_features",
        description="Read-only profiling of history available at prediction time.",
    )
    parser.add_argument("--parquet", type=Path, help="Raw Parquet. Default: latest raw.")
    parser.add_argument("--assignments", type=Path, help="Component 2 assignments.")
    parser.add_argument("--establishments", type=Path, help="Component 2 establishments.")
    parser.add_argument("--targets", type=Path, help="Component 3 targets.")
    parser.add_argument("--only", action="append", metavar="NAME", help="Run only these.")
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
    establishments_path: Path = args.establishments or latest_parquet(
        settings.entity_resolution_interim_dir, prefix="establishments_"
    )
    targets_path: Path = args.targets or latest_parquet(
        settings.target_interim_dir, prefix="inspection_targets_"
    )

    selected: list[str] = list(args.only) if args.only else list(PROFILES)
    unknown = [n for n in selected if n not in PROFILES]
    if unknown:
        print(f"Unknown profile(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    conn = duckdb.connect(database=":memory:")
    try:
        prepare_views(conn, raw_path, assignments_path, establishments_path, targets_path)
        print(
            f"<!-- generated by scripts/profile_features.py from {raw_path.name} "
            f"+ {assignments_path.name} + {targets_path.name} -->\n"
        )
        for name in selected:
            print(run_profile(conn, name))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
