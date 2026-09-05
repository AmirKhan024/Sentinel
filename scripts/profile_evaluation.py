"""Read-only profiling of what an honest temporal evaluation can be built from.

Analysis tooling, not library code, for the same reasons as the other three profilers:
it answers one-off questions about a snapshot, nothing imports it, and it should not ship
in the wheel.

This one reads a single artifact -- Component 4's as-of feature table -- because every
question Component 5 must answer before it can be designed is a question about that
table: how many rows fall in each candidate fold, what the base rate does over time, how
many inspections a day Chicago actually performs, and whether citation rates move with
the season once the secular drift is removed.

Component 5 does not recompute identity, labels or features. It reads them.

Every query is a SELECT and the script writes nothing. Output is markdown, pasted into
``docs/analysis/temporal_evaluation_findings.md``.

Usage
-----
    uv run python scripts/profile_evaluation.py
    uv run python scripts/profile_evaluation.py --only capacity_per_day
    uv run python scripts/profile_evaluation.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.config import load_settings  # noqa: E402
from sentinel.query.duckdb_queries import latest_parquet  # noqa: E402

PROFILES: dict[str, str] = {
    # -- 1. what the table actually contains --------------------------------
    "coverage": """
        SELECT count(*) AS rows,
               count(DISTINCT establishment_id) AS establishments,
               count(DISTINCT rd) AS distinct_dates,
               min(rd) AS first_date,
               max(rd) AS last_date,
               round(avg(target), 4) AS positive_rate
        FROM f
    """,
    "positive_rate_by_year": """
        SELECT year(rd) AS yr,
               count(*) AS rows,
               sum(target) AS positives,
               round(avg(target), 4) AS positive_rate
        FROM f GROUP BY 1 ORDER BY 1
    """,
    "positive_rate_by_quarter": """
        SELECT quarter_key AS quarter,
               count(*) AS rows,
               count(DISTINCT rd) AS days,
               sum(target) AS positives,
               round(avg(target), 4) AS positive_rate
        FROM f GROUP BY 1 ORDER BY 1
    """,
    # -- 2. capacity: precision@k needs a defensible k ----------------------
    "capacity_per_day": """
        WITH d AS (SELECT rd, count(*) AS n FROM f GROUP BY 1)
        SELECT count(*) AS inspection_days,
               round(avg(n), 2) AS mean_per_day,
               median(n) AS median_per_day,
               quantile_cont(n, 0.25) AS p25,
               quantile_cont(n, 0.75) AS p75,
               quantile_cont(n, 0.90) AS p90,
               min(n) AS min_per_day,
               max(n) AS max_per_day
        FROM d
    """,
    "capacity_per_day_by_year": """
        WITH d AS (SELECT rd, count(*) AS n FROM f GROUP BY 1)
        SELECT year(rd) AS yr,
               count(*) AS days,
               round(avg(n), 2) AS mean_per_day,
               median(n) AS median_per_day,
               min(n) AS min_per_day,
               max(n) AS max_per_day
        FROM d GROUP BY 1 ORDER BY 1
    """,
    "capacity_by_weekday": """
        SELECT dayname(rd) AS weekday, count(*) AS rows
        FROM f GROUP BY 1 ORDER BY 2 DESC
    """,
    # -- 3. candidate quarterly folds ---------------------------------------
    "candidate_test_windows": """
        SELECT quarter_key AS test_quarter,
               count(*) AS test_rows,
               count(DISTINCT rd) AS test_days,
               count(DISTINCT establishment_id) AS test_establishments,
               sum(target) AS test_positives,
               round(avg(target), 4) AS test_positive_rate,
               median(daily_n) AS median_daily_capacity
        FROM (SELECT f.*, count(*) OVER (PARTITION BY rd) AS daily_n FROM f)
        GROUP BY 1 ORDER BY 1
    """,
    "expanding_train_sizes": """
        WITH q AS (SELECT DISTINCT quarter_key AS qk, quarter_start AS qs FROM f)
        SELECT q.qk AS calibration_quarter,
               (SELECT count(*) FROM f WHERE f.rd < q.qs) AS train_rows,
               (SELECT round(avg(target), 4) FROM f WHERE f.rd < q.qs) AS train_positive_rate
        FROM q ORDER BY 1
    """,
    # -- 4. does an establishment reappear inside a short window? -----------
    "establishment_recurrence_within_quarter": """
        WITH x AS (
            SELECT quarter_key, establishment_id, count(*) AS n
            FROM f GROUP BY 1, 2
        )
        SELECT n AS rows_for_one_establishment_in_one_quarter,
               count(*) AS occurrences
        FROM x GROUP BY 1 ORDER BY 1
    """,
    # -- 5. seasonality, de-trended -----------------------------------------
    # The raw month rate is contaminated by the secular drift, so the useful
    # quantity is each month's deviation from its own year's rate.
    "citation_rate_by_month_raw": """
        SELECT month(rd) AS mo,
               count(*) AS rows,
               round(avg(target), 4) AS positive_rate
        FROM f GROUP BY 1 ORDER BY 1
    """,
    "citation_rate_by_month_detrended": """
        WITH yearly AS (
            SELECT year(rd) AS yr, avg(target) AS year_rate FROM f GROUP BY 1
        ),
        monthly AS (
            SELECT year(rd) AS yr, month(rd) AS mo,
                   count(*) AS rows, avg(target) AS month_rate
            FROM f GROUP BY 1, 2
        )
        SELECT m.mo,
               sum(m.rows) AS rows,
               round(avg(m.month_rate - y.year_rate), 4) AS mean_deviation_from_year,
               round(stddev_samp(m.month_rate - y.year_rate), 4) AS sd_of_deviation
        FROM monthly m JOIN yearly y USING (yr)
        GROUP BY 1 ORDER BY 1
    """,
    "citation_rate_by_month_and_year": """
        SELECT year(rd) AS yr, month(rd) AS mo,
               count(*) AS rows, round(avg(target), 4) AS positive_rate
        FROM f GROUP BY 1, 2 ORDER BY 1, 2
    """,
    # -- 6. what the deterministic rankers have to work with ----------------
    "ranker_column_availability": """
        SELECT 'days_since_last_canvass' AS column_name,
               count(*) - count(days_since_last_canvass) AS nulls,
               round(100.0 * (count(*) - count(days_since_last_canvass)) / count(*), 2) AS null_pct
        FROM f
        UNION ALL
        SELECT 'priority_at_last_canvass',
               count(*) - count(priority_at_last_canvass),
               round(100.0 * (count(*) - count(priority_at_last_canvass)) / count(*), 2)
        FROM f
        UNION ALL
        SELECT 'fail_at_last_canvass',
               count(*) - count(fail_at_last_canvass),
               round(100.0 * (count(*) - count(fail_at_last_canvass)) / count(*), 2)
        FROM f
        UNION ALL
        SELECT 'prior_canvass_priority_rate',
               count(*) - count(prior_canvass_priority_rate),
               round(100.0 * (count(*) - count(prior_canvass_priority_rate)) / count(*), 2)
        FROM f
        ORDER BY 1
    """,
    "signal_in_priority_at_last_canvass": """
        SELECT coalesce(CAST(priority_at_last_canvass AS VARCHAR), 'NULL') AS prior_priority,
               count(*) AS rows,
               round(avg(target), 4) AS positive_rate
        FROM f GROUP BY 1 ORDER BY 1
    """,
    "signal_in_days_since_last_canvass": """
        SELECT CASE WHEN days_since_last_canvass IS NULL THEN 'z. NULL (no prior canvass)'
                    WHEN days_since_last_canvass <= 180 THEN 'a. <=180'
                    WHEN days_since_last_canvass <= 365 THEN 'b. 181-365'
                    WHEN days_since_last_canvass <= 547 THEN 'c. 366-547'
                    WHEN days_since_last_canvass <= 730 THEN 'd. 548-730'
                    ELSE 'e. >730' END AS bucket,
               count(*) AS rows,
               round(avg(target), 4) AS positive_rate
        FROM f GROUP BY 1 ORDER BY 1
    """,
    # -- 7. the distribution-shift experiment -------------------------------
    "covid_shift_windows": """
        SELECT CASE WHEN rd < DATE '2020-03-01' THEN 'a. train  2018-07-01..2020-02-29'
                    WHEN rd < DATE '2020-06-01' THEN 'b. cal    2020-03-01..2020-05-31'
                    WHEN rd < DATE '2022-01-01' THEN 'c. test   2020-06-01..2021-12-31'
                    ELSE 'd. after the experiment' END AS window,
               count(*) AS rows,
               count(DISTINCT rd) AS days,
               round(avg(target), 4) AS positive_rate
        FROM f GROUP BY 1 ORDER BY 1
    """,
    "adoption_phase_rows": """
        SELECT code_era_phase, count(*) AS rows,
               min(rd) AS first_date, max(rd) AS last_date,
               round(avg(target), 4) AS positive_rate
        FROM f GROUP BY 1 ORDER BY 1
    """,
}


def render_markdown(columns: list[str], rows: list[tuple[object, ...]]) -> str:
    """Render a result set as a markdown table."""
    header = [str(c) for c in columns]
    body = [["" if v is None else str(v) for v in row] for row in rows]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(lines)


def run_profile(conn: duckdb.DuckDBPyConnection, name: str) -> str:
    """Execute one profile and return it as a markdown section."""
    cursor = conn.execute(PROFILES[name])
    columns = [d[0] for d in cursor.description or []]
    rows = cursor.fetchall()
    return f"### `{name}`\n\n{render_markdown(columns, rows)}\n"


def prepare_views(conn: duckdb.DuckDBPyConnection, features_path: Path) -> None:
    """Register the feature table with its reference date parsed once.

    ``rd`` is the reference date as a DATE, and ``quarter_key`` / ``quarter_start``
    are the fold-window keys every candidate-fold profile groups by. Parsing here
    means no profile repeats the cast, and every profile agrees what a quarter is.

    Paths go through the relation API because CREATE VIEW cannot bind a parameter.
    """
    conn.read_parquet(str(features_path)).create_view("src")
    conn.execute("""
        CREATE VIEW f AS
        SELECT establishment_id,
               target_inspection_id,
               target,
               code_era_phase,
               days_since_last_canvass,
               priority_at_last_canvass,
               fail_at_last_canvass,
               prior_canvass_priority_rate,
               CAST(inspection_date AS DATE) AS rd,
               printf('%dQ%d', year(CAST(inspection_date AS DATE)),
                               quarter(CAST(inspection_date AS DATE))) AS quarter_key,
               date_trunc('quarter', CAST(inspection_date AS DATE)) AS quarter_start
        FROM src
    """)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="profile_evaluation",
        description="Read-only profiling of the temporal evaluation surface.",
    )
    parser.add_argument("--features", type=Path, help="Component 4 features. Default: latest.")
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
    features_path: Path = args.features or latest_parquet(
        settings.features_processed_dir, prefix="as_of_features_"
    )

    selected: list[str] = list(args.only) if args.only else list(PROFILES)
    unknown = [n for n in selected if n not in PROFILES]
    if unknown:
        print(f"Unknown profile(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    conn = duckdb.connect(database=":memory:")
    try:
        prepare_views(conn, features_path)
        print(f"<!-- generated by scripts/profile_evaluation.py from {features_path.name} -->\n")
        for name in selected:
            print(run_profile(conn, name))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
