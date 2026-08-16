"""Read-only profiling of the raw food inspections layer for entity resolution.

This is analysis tooling, not library code. It lives outside ``src/sentinel``
deliberately: it answers one-off questions about a snapshot, it is not imported
by anything, and it should not ship in the wheel. It is also not added to
``sentinel.query.duckdb_queries.NAMED_QUERIES``, whose docstring reserves that
module for plain description of the raw layer.

Every query here is a SELECT. The script writes nothing. Its output is pasted
into ``docs/analysis/entity_resolution_findings.md``, which is why results are
rendered as markdown tables rather than the fixed-width terminal tables the CLI
uses.

Usage
-----
    uv run python scripts/profile_entities.py                  # every profile
    uv run python scripts/profile_entities.py --only geo_quality
    uv run python scripts/profile_entities.py --list
    uv run python scripts/profile_entities.py --parquet PATH

The input Parquet is all-VARCHAR by contract (ADR 0002), so every numeric or
temporal comparison below casts explicitly with TRY_CAST and treats a failed
cast as a measurable outcome rather than a silent null.
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

# A licence value that is present, non-blank and not the '0' sentinel. Repeated
# verbatim in several queries; kept as one constant so the definition of
# "usable licence" cannot drift between measurements.
USABLE_LICENSE = "license_ IS NOT NULL AND trim(license_) <> '' AND license_ <> '0'"

# Chicago bounding box, used to detect coordinates that cannot be real.
LAT_MIN, LAT_MAX = 41.60, 42.10
LON_MIN, LON_MAX = -87.95, -87.50

# Metres per degree at Chicago's latitude (~41.85 N). Used for cheap bounding
# box spans; exact haversine is the resolver's job, not the profiler's.
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = 82_860.0

PROFILES: dict[str, str] = {
    # -- 5.1 licence completeness -----------------------------------------
    "license_completeness": """
        SELECT
            count(*)                                                        AS total_rows,
            count(*) FILTER (WHERE license_ IS NULL)                        AS null_license,
            count(*) FILTER (WHERE license_ = '')                           AS empty_license,
            count(*) FILTER (WHERE license_ IS NOT NULL
                               AND license_ <> ''
                               AND trim(license_) = '')                     AS whitespace_license,
            count(*) FILTER (WHERE license_ = '0')                          AS zero_license,
            count(*) FILTER (WHERE license_ IS NOT NULL
                               AND NOT regexp_matches(license_, '^[0-9]+$')) AS non_numeric,
            count(*) FILTER (WHERE regexp_matches(coalesce(license_, ''), '^0[0-9]+$'))
                                                                            AS leading_zero,
            count(DISTINCT license_)                                        AS distinct_license
        FROM src
    """,
    "license_length_distribution": """
        SELECT length(license_) AS license_length, count(*) AS rows
        FROM src
        WHERE license_ IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """,
    "license_sentinel_scale": f"""
        SELECT
            coalesce(license_, '<NULL>')  AS license_value,
            count(*)                      AS rows,
            count(DISTINCT dba_name)      AS distinct_names,
            count(DISTINCT address)       AS distinct_addresses,
            count(DISTINCT zip)           AS distinct_zips
        FROM src
        WHERE NOT ({USABLE_LICENSE})
        GROUP BY 1
        ORDER BY rows DESC
    """,
    # -- 5.2 licence -> attribute fan-out ---------------------------------
    "license_fanout_summary": f"""
        WITH per_license AS (
            SELECT
                license_,
                count(DISTINCT dba_name)      AS n_names,
                count(DISTINCT address)       AS n_addresses,
                count(DISTINCT zip)           AS n_zips,
                count(DISTINCT facility_type) AS n_facility_types
            FROM src
            WHERE {USABLE_LICENSE}
            GROUP BY license_
        )
        SELECT 'dba_name' AS attribute, count(*) AS n_licenses,
               round(100.0 * count(*) FILTER (WHERE n_names = 1) / count(*), 2) AS pct_single,
               quantile_cont(n_names, 0.90) AS p90,
               quantile_cont(n_names, 0.99) AS p99,
               quantile_cont(n_names, 0.995) AS p995,
               max(n_names) AS max_value
        FROM per_license
        UNION ALL
        SELECT 'address', count(*),
               round(100.0 * count(*) FILTER (WHERE n_addresses = 1) / count(*), 2),
               quantile_cont(n_addresses, 0.90), quantile_cont(n_addresses, 0.99),
               quantile_cont(n_addresses, 0.995), max(n_addresses)
        FROM per_license
        UNION ALL
        SELECT 'zip', count(*),
               round(100.0 * count(*) FILTER (WHERE n_zips = 1) / count(*), 2),
               quantile_cont(n_zips, 0.90), quantile_cont(n_zips, 0.99),
               quantile_cont(n_zips, 0.995), max(n_zips)
        FROM per_license
        UNION ALL
        SELECT 'facility_type', count(*),
               round(100.0 * count(*) FILTER (WHERE n_facility_types = 1) / count(*), 2),
               quantile_cont(n_facility_types, 0.90), quantile_cont(n_facility_types, 0.99),
               quantile_cont(n_facility_types, 0.995), max(n_facility_types)
        FROM per_license
    """,
    "license_fanout_top_names": f"""
        SELECT
            license_,
            count(DISTINCT dba_name)     AS n_names,
            count(DISTINCT address)      AS n_addresses,
            count(*)                     AS n_rows,
            min(inspection_date)[1:10]   AS first_seen,
            max(inspection_date)[1:10]   AS last_seen,
            string_agg(DISTINCT dba_name, ' | ' ORDER BY dba_name)[1:170] AS names
        FROM src
        WHERE {USABLE_LICENSE}
        GROUP BY license_
        ORDER BY n_names DESC, n_rows DESC
        LIMIT 20
    """,
    "license_fanout_top_addresses": f"""
        SELECT
            license_,
            count(DISTINCT address)      AS n_addresses,
            count(DISTINCT zip)          AS n_zips,
            count(*)                     AS n_rows,
            any_value(dba_name)          AS a_name,
            string_agg(DISTINCT address, ' | ' ORDER BY address)[1:170] AS addresses
        FROM src
        WHERE {USABLE_LICENSE}
        GROUP BY license_
        ORDER BY n_addresses DESC
        LIMIT 20
    """,
    "license_fanout_geo_span": f"""
        WITH per_license AS (
            SELECT
                license_,
                min(TRY_CAST(latitude  AS DOUBLE)) AS min_lat,
                max(TRY_CAST(latitude  AS DOUBLE)) AS max_lat,
                min(TRY_CAST(longitude AS DOUBLE)) AS min_lon,
                max(TRY_CAST(longitude AS DOUBLE)) AS max_lon
            FROM src
            WHERE {USABLE_LICENSE}
              AND TRY_CAST(latitude  AS DOUBLE) BETWEEN {LAT_MIN} AND {LAT_MAX}
              AND TRY_CAST(longitude AS DOUBLE) BETWEEN {LON_MIN} AND {LON_MAX}
            GROUP BY license_
        ), spans AS (
            SELECT
                sqrt(
                    pow((max_lat - min_lat) * {M_PER_DEG_LAT}, 2)
                  + pow((max_lon - min_lon) * {M_PER_DEG_LON}, 2)
                ) AS span_m
            FROM per_license
        )
        SELECT
            CASE
                WHEN span_m = 0        THEN 'a. exactly 0 m'
                WHEN span_m <= 25      THEN 'b. 0-25 m'
                WHEN span_m <= 50      THEN 'c. 25-50 m'
                WHEN span_m <= 250     THEN 'd. 50-250 m'
                WHEN span_m <= 1000    THEN 'e. 250 m - 1 km'
                WHEN span_m <= 10000   THEN 'f. 1-10 km'
                ELSE                        'g. over 10 km'
            END AS bucket,
            count(*) AS n_licenses,
            round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
        FROM spans
        GROUP BY 1
        ORDER BY 1
    """,
    # -- 5.3 (name, address) -> licence fan-in ----------------------------
    "name_address_license_fanin": f"""
        WITH grouped AS (
            SELECT
                upper(trim(dba_name)) AS nm,
                upper(trim(address))  AS ad,
                count(DISTINCT license_) AS n_licenses
            FROM src
            WHERE {USABLE_LICENSE}
              AND dba_name IS NOT NULL AND trim(dba_name) <> ''
              AND address  IS NOT NULL AND trim(address)  <> ''
            GROUP BY 1, 2
        )
        SELECT
            count(*)                                           AS name_address_pairs,
            count(*) FILTER (WHERE n_licenses = 1)             AS with_1_license,
            count(*) FILTER (WHERE n_licenses = 2)             AS with_2_licenses,
            count(*) FILTER (WHERE n_licenses BETWEEN 3 AND 5) AS with_3_to_5,
            count(*) FILTER (WHERE n_licenses > 5)             AS with_over_5,
            round(100.0 * count(*) FILTER (WHERE n_licenses > 1) / count(*), 2) AS pct_multi,
            max(n_licenses)                                    AS max_licenses
        FROM grouped
    """,
    "name_address_license_fanin_examples": f"""
        SELECT
            upper(trim(dba_name))    AS nm,
            upper(trim(address))     AS ad,
            count(DISTINCT license_) AS n_licenses,
            count(*)                 AS n_rows,
            string_agg(DISTINCT license_, ', ' ORDER BY license_)[1:90] AS licenses,
            min(inspection_date)[1:10] AS first_seen,
            max(inspection_date)[1:10] AS last_seen
        FROM src
        WHERE {USABLE_LICENSE}
          AND dba_name IS NOT NULL AND trim(dba_name) <> ''
          AND address  IS NOT NULL AND trim(address)  <> ''
        GROUP BY 1, 2
        HAVING count(DISTINCT license_) > 1
        ORDER BY n_licenses DESC, n_rows DESC
        LIMIT 25
    """,
    # -- 5.4 name variation ------------------------------------------------
    "name_completeness": """
        SELECT
            count(*)                                                   AS total_rows,
            count(*) FILTER (WHERE dba_name IS NULL)                   AS null_dba,
            count(*) FILTER (WHERE trim(coalesce(dba_name, '')) = '')  AS blank_dba,
            count(DISTINCT dba_name)                                   AS distinct_dba,
            count(DISTINCT upper(trim(dba_name)))                      AS distinct_upper,
            count(DISTINCT regexp_replace(upper(dba_name), '[^A-Z0-9]', '', 'g'))
                                                                       AS distinct_alnum
        FROM src
    """,
    "name_trailing_tokens": """
        WITH names AS (
            SELECT DISTINCT
                trim(regexp_replace(upper(dba_name), '[^A-Z0-9]+', ' ', 'g')) AS nm
            FROM src
            WHERE dba_name IS NOT NULL AND trim(dba_name) <> ''
        )
        SELECT
            regexp_extract(nm, '([A-Z0-9]+)$', 1) AS trailing_token,
            count(*)                              AS distinct_names
        FROM names
        WHERE nm <> ''
        GROUP BY 1
        ORDER BY distinct_names DESC
        LIMIT 40
    """,
    "name_token_frequency": """
        WITH names AS (
            SELECT DISTINCT
                trim(regexp_replace(upper(dba_name), '[^A-Z0-9]+', ' ', 'g')) AS nm
            FROM src
            WHERE dba_name IS NOT NULL AND trim(dba_name) <> ''
        ), toks AS (
            SELECT unnest(string_split(nm, ' ')) AS tok FROM names WHERE nm <> ''
        )
        SELECT tok AS token, count(*) AS distinct_names
        FROM toks
        WHERE tok <> ''
        GROUP BY 1
        ORDER BY distinct_names DESC
        LIMIT 40
    """,
    "generic_name_spread": """
        SELECT
            upper(trim(dba_name))                AS nm,
            count(DISTINCT upper(trim(address))) AS n_addresses,
            count(DISTINCT license_)             AS n_licenses,
            count(*)                             AS n_rows
        FROM src
        WHERE dba_name IS NOT NULL AND trim(dba_name) <> ''
        GROUP BY 1
        ORDER BY n_addresses DESC
        LIMIT 30
    """,
    "name_variation_within_license": f"""
        WITH per_license AS (
            SELECT license_
            FROM src
            WHERE {USABLE_LICENSE}
            GROUP BY license_
            HAVING count(DISTINCT dba_name) > 1
        )
        SELECT
            s.license_,
            count(DISTINCT s.dba_name) AS n_names,
            count(DISTINCT upper(regexp_replace(s.dba_name, '[^A-Za-z0-9]', '', 'g')))
                                       AS n_names_alnum_upper,
            string_agg(DISTINCT s.dba_name, ' | ' ORDER BY s.dba_name)[1:150] AS names
        FROM src AS s
        JOIN per_license AS p USING (license_)
        GROUP BY s.license_
        ORDER BY n_names DESC
        LIMIT 25
    """,
    "name_variation_collapse_rate": f"""
        WITH per_license AS (
            SELECT
                license_,
                count(DISTINCT dba_name) AS raw_names,
                count(DISTINCT upper(trim(dba_name))) AS upper_names,
                count(DISTINCT regexp_replace(upper(dba_name), '[^A-Z0-9]', '', 'g'))
                    AS alnum_names,
                count(DISTINCT regexp_replace(
                    regexp_replace(upper(dba_name), '[^A-Z0-9]', '', 'g'),
                    '(LLC|INC|CORP|LTD)$', '', 'g')) AS suffixless_names
            FROM src
            WHERE {USABLE_LICENSE}
            GROUP BY license_
        )
        SELECT
            count(*)                                     AS licenses,
            count(*) FILTER (WHERE raw_names = 1)        AS single_raw,
            count(*) FILTER (WHERE upper_names = 1)      AS single_after_upper,
            count(*) FILTER (WHERE alnum_names = 1)      AS single_after_alnum,
            count(*) FILTER (WHERE suffixless_names = 1) AS single_after_suffix_strip
        FROM per_license
    """,
    # -- 5.5 aka_name ------------------------------------------------------
    "aka_name_behaviour": """
        SELECT
            count(*)                                                  AS total_rows,
            count(*) FILTER (WHERE aka_name IS NULL)                  AS null_aka,
            count(*) FILTER (WHERE trim(coalesce(aka_name, '')) = '') AS blank_aka,
            count(*) FILTER (WHERE upper(trim(aka_name)) = upper(trim(dba_name)))
                                                                      AS equal_to_dba,
            count(*) FILTER (WHERE aka_name IS NOT NULL
                               AND trim(aka_name) <> ''
                               AND upper(trim(aka_name)) <> upper(trim(dba_name)))
                                                                      AS differs_from_dba,
            count(*) FILTER (WHERE
                       regexp_replace(upper(coalesce(aka_name, '')), '[^A-Z0-9]', '', 'g')
                     = regexp_replace(upper(coalesce(dba_name, '')), '[^A-Z0-9]', '', 'g')
                   AND upper(trim(aka_name)) <> upper(trim(dba_name)))
                                                                      AS equal_only_after_norm
        FROM src
    """,
    "aka_name_examples": """
        SELECT DISTINCT dba_name, aka_name
        FROM src
        WHERE aka_name IS NOT NULL AND trim(aka_name) <> ''
          AND upper(trim(aka_name)) <> upper(trim(dba_name))
          AND regexp_replace(upper(aka_name), '[^A-Z0-9]', '', 'g')
           <> regexp_replace(upper(dba_name), '[^A-Z0-9]', '', 'g')
        ORDER BY dba_name
        LIMIT 30
    """,
    # -- 5.6 address variation ---------------------------------------------
    "address_completeness": """
        SELECT
            count(*)                                                 AS total_rows,
            count(*) FILTER (WHERE address IS NULL)                  AS null_address,
            count(*) FILTER (WHERE trim(coalesce(address, '')) = '') AS blank_address,
            count(DISTINCT address)                                  AS distinct_address,
            count(DISTINCT upper(trim(address)))                     AS distinct_upper,
            count(*) FILTER (WHERE NOT regexp_matches(coalesce(address, ''), '^[0-9]'))
                                                                     AS no_house_number
        FROM src
    """,
    "address_pattern_census": """
        WITH addrs AS (
            SELECT DISTINCT upper(trim(address)) AS addr
            FROM src
            WHERE address IS NOT NULL AND trim(address) <> ''
        )
        SELECT
            count(*)                                                          AS distinct_addresses,
            count(*) FILTER (WHERE regexp_matches(addr, '^[0-9]+ +(N|S|E|W) '))
                                                                              AS short_directional,
            count(*) FILTER (WHERE regexp_matches(addr, '^[0-9]+ +(NORTH|SOUTH|EAST|WEST) '))
                                                                              AS long_directional,
            count(*) FILTER (WHERE regexp_matches(
                addr, '\\b(STE|SUITE|APT|UNIT|RM|ROOM|FL|FLOOR|BLDG|SPACE|REAR|BSMT|LL)\\b'))
                                                                              AS unit_word,
            count(*) FILTER (WHERE addr LIKE '%#%')                           AS hash_marker,
            count(*) FILTER (WHERE regexp_matches(addr, '^[0-9]+-[0-9]+ '))   AS ranged_house,
            count(*) FILTER (WHERE regexp_matches(addr, '[0-9] 1/2'))         AS fractional,
            count(*) FILTER (WHERE addr LIKE '%&%')                           AS ampersand,
            count(*) FILTER (WHERE addr LIKE '%.%')                           AS has_period,
            count(*) FILTER (WHERE regexp_matches(addr, '  '))                AS double_space,
            count(*) FILTER (WHERE NOT regexp_matches(addr, '^[ -~]*$'))      AS non_ascii
        FROM addrs
    """,
    "address_suffix_census": """
        WITH addrs AS (
            SELECT DISTINCT upper(trim(address)) AS addr
            FROM src
            WHERE address IS NOT NULL AND trim(address) <> ''
        )
        SELECT
            regexp_extract(regexp_replace(addr, '[^A-Z0-9 ]', ' ', 'g'), '([A-Z0-9]+) *$', 1)
                AS final_token,
            count(*) AS distinct_addresses
        FROM addrs
        GROUP BY 1
        ORDER BY distinct_addresses DESC
        LIMIT 40
    """,
    "address_near_duplicates": """
        WITH parts AS (
            SELECT DISTINCT
                zip,
                regexp_extract(upper(trim(address)), '^([0-9]+)', 1) AS house,
                upper(trim(address))                                 AS addr
            FROM src
            WHERE address IS NOT NULL AND regexp_matches(trim(address), '^[0-9]')
              AND zip IS NOT NULL AND trim(zip) <> ''
        )
        SELECT
            zip, house,
            count(*) AS n_distinct_strings,
            string_agg(addr, ' | ' ORDER BY addr)[1:170] AS variants
        FROM parts
        GROUP BY zip, house
        HAVING count(*) > 1
        ORDER BY n_distinct_strings DESC
        LIMIT 30
    """,
    # -- 5.7 geographic quality --------------------------------------------
    "geo_quality": f"""
        SELECT
            count(*)                                                        AS total_rows,
            count(*) FILTER (WHERE latitude IS NULL OR trim(latitude) = '') AS null_latitude,
            count(*) FILTER (WHERE latitude IS NOT NULL AND trim(latitude) <> ''
                               AND TRY_CAST(latitude AS DOUBLE) IS NULL)    AS unparseable,
            count(*) FILTER (WHERE TRY_CAST(latitude AS DOUBLE) IS NOT NULL
                       AND (TRY_CAST(latitude  AS DOUBLE) NOT BETWEEN {LAT_MIN} AND {LAT_MAX}
                         OR TRY_CAST(longitude AS DOUBLE) NOT BETWEEN {LON_MIN} AND {LON_MAX}))
                                                                            AS outside_bbox,
            count(DISTINCT latitude || ',' || longitude)                    AS distinct_coords
        FROM src
    """,
    "geo_precision": """
        SELECT
            length(split_part(latitude, '.', 2)) AS latitude_decimal_places,
            count(*)                             AS rows
        FROM src
        WHERE latitude IS NOT NULL AND trim(latitude) <> '' AND latitude LIKE '%.%'
        GROUP BY 1
        ORDER BY 1
    """,
    "geo_shared_coordinates": """
        SELECT
            latitude, longitude,
            count(DISTINCT upper(trim(address))) AS n_addresses,
            count(*)                             AS n_rows,
            any_value(dba_name)                  AS a_name
        FROM src
        WHERE latitude IS NOT NULL AND trim(latitude) <> ''
        GROUP BY latitude, longitude
        HAVING count(DISTINCT upper(trim(address))) > 3
        ORDER BY n_addresses DESC
        LIMIT 25
    """,
    "geo_spread_within_address": """
        WITH per_address AS (
            SELECT
                upper(trim(address)) AS addr,
                min(TRY_CAST(latitude  AS DOUBLE)) AS min_lat,
                max(TRY_CAST(latitude  AS DOUBLE)) AS max_lat,
                min(TRY_CAST(longitude AS DOUBLE)) AS min_lon,
                max(TRY_CAST(longitude AS DOUBLE)) AS max_lon
            FROM src
            WHERE address IS NOT NULL AND trim(address) <> ''
              AND TRY_CAST(latitude AS DOUBLE) IS NOT NULL
            GROUP BY 1
        ), spans AS (
            SELECT sqrt(
                pow((max_lat - min_lat) * 111320.0, 2)
              + pow((max_lon - min_lon) * 82860.0, 2)
            ) AS span_m
            FROM per_address
        )
        SELECT
            count(*)                               AS addresses,
            round(quantile_cont(span_m, 0.50), 2)  AS p50_m,
            round(quantile_cont(span_m, 0.90), 2)  AS p90_m,
            round(quantile_cont(span_m, 0.99), 2)  AS p99_m,
            round(quantile_cont(span_m, 0.999), 2) AS p999_m,
            round(max(span_m), 2)                  AS max_m
        FROM spans
    """,
    # -- 5.8 temporal identity behaviour -----------------------------------
    "successor_candidates": f"""
        WITH per_place_license AS (
            SELECT
                upper(trim(dba_name)) AS nm,
                upper(trim(address))  AS ad,
                license_,
                min(inspection_date)  AS first_date,
                max(inspection_date)  AS last_date
            FROM src
            WHERE {USABLE_LICENSE}
              AND dba_name IS NOT NULL AND trim(dba_name) <> ''
              AND address  IS NOT NULL AND trim(address)  <> ''
            GROUP BY 1, 2, 3
        ), pairs AS (
            SELECT
                (a.last_date < b.first_date OR b.last_date < a.first_date) AS disjoint
            FROM per_place_license AS a
            JOIN per_place_license AS b
              ON a.nm = b.nm AND a.ad = b.ad AND a.license_ < b.license_
        )
        SELECT
            count(*)                             AS license_pairs_at_same_place,
            count(*) FILTER (WHERE disjoint)     AS temporally_disjoint,
            count(*) FILTER (WHERE NOT disjoint) AS overlapping,
            round(100.0 * count(*) FILTER (WHERE disjoint) / nullif(count(*), 0), 2) AS pct_disjoint
        FROM pairs
    """,
    "same_place_different_name": f"""
        SELECT
            upper(trim(address)) AS ad,
            license_,
            count(DISTINCT upper(trim(dba_name))) AS n_names,
            min(inspection_date)[1:10] AS first_date,
            max(inspection_date)[1:10] AS last_date,
            string_agg(DISTINCT upper(trim(dba_name)), ' -> '
                       ORDER BY upper(trim(dba_name)))[1:130] AS names
        FROM src
        WHERE {USABLE_LICENSE}
          AND address IS NOT NULL AND trim(address) <> ''
        GROUP BY 1, 2
        HAVING count(DISTINCT upper(trim(dba_name))) > 1
        ORDER BY n_names DESC
        LIMIT 25
    """,
    "address_entity_density": f"""
        SELECT
            upper(trim(address))                  AS addr,
            any_value(zip)                        AS zip,
            count(DISTINCT license_)              AS n_licenses,
            count(DISTINCT upper(trim(dba_name))) AS n_names,
            count(*)                              AS n_rows
        FROM src
        WHERE {USABLE_LICENSE}
          AND address IS NOT NULL AND trim(address) <> ''
        GROUP BY 1
        ORDER BY n_licenses DESC
        LIMIT 25
    """,
    # -- 5.9 inspection history continuity ---------------------------------
    "history_length_distribution": f"""
        WITH per_license AS (
            SELECT license_, count(*) AS n_rows
            FROM src
            WHERE {USABLE_LICENSE}
            GROUP BY license_
        )
        SELECT
            count(*)                                        AS licenses,
            count(*) FILTER (WHERE n_rows = 1)              AS with_1_inspection,
            count(*) FILTER (WHERE n_rows BETWEEN 2 AND 5)  AS with_2_to_5,
            count(*) FILTER (WHERE n_rows BETWEEN 6 AND 20) AS with_6_to_20,
            count(*) FILTER (WHERE n_rows > 20)             AS with_over_20,
            quantile_cont(n_rows, 0.90)                     AS p90,
            quantile_cont(n_rows, 0.99)                     AS p99,
            quantile_cont(n_rows, 0.995)                    AS p995,
            max(n_rows)                                     AS max_inspections
        FROM per_license
    """,
    "history_top_licenses": f"""
        SELECT
            license_,
            count(*)                              AS n_inspections,
            count(DISTINCT upper(trim(dba_name))) AS n_names,
            count(DISTINCT upper(trim(address)))  AS n_addresses,
            min(inspection_date)[1:10]            AS first_date,
            max(inspection_date)[1:10]            AS last_date,
            any_value(dba_name)                   AS a_name,
            any_value(address)                    AS an_address
        FROM src
        WHERE {USABLE_LICENSE}
        GROUP BY license_
        ORDER BY n_inspections DESC
        LIMIT 20
    """,
    # -- 5.10 facility type -------------------------------------------------
    "facility_type_stability": f"""
        WITH per_license AS (
            SELECT
                license_,
                count(DISTINCT facility_type) AS n_raw,
                count(DISTINCT upper(trim(facility_type))) AS n_normalized
            FROM src
            WHERE {USABLE_LICENSE}
              AND facility_type IS NOT NULL AND trim(facility_type) <> ''
            GROUP BY license_
        )
        SELECT
            count(*)                                 AS licenses,
            count(*) FILTER (WHERE n_raw = 1)        AS single_raw_type,
            count(*) FILTER (WHERE n_normalized = 1) AS single_normalized_type,
            round(100.0 * count(*) FILTER (WHERE n_normalized = 1) / count(*), 2) AS pct_stable,
            max(n_normalized)                        AS max_types
        FROM per_license
    """,
    "facility_type_values": """
        SELECT
            coalesce(nullif(trim(facility_type), ''), '<BLANK>') AS facility_type,
            count(*) AS rows
        FROM src
        GROUP BY 1
        ORDER BY rows DESC
        LIMIT 25
    """,
    # -- cross-cutting -------------------------------------------------------
    "duplicate_inspection_ids": """
        SELECT
            count(*)                                 AS total_rows,
            count(DISTINCT inspection_id)            AS distinct_inspection_ids,
            count(*) - count(DISTINCT inspection_id) AS duplicate_rows,
            count(*) FILTER (WHERE NOT regexp_matches(coalesce(inspection_id, ''), '^[0-9]+$'))
                                                     AS non_numeric_ids
        FROM src
    """,
    "date_range": """
        SELECT
            min(inspection_date)[1:10] AS earliest,
            max(inspection_date)[1:10] AS latest,
            count(*) FILTER (WHERE inspection_date IS NULL) AS null_dates,
            count(DISTINCT inspection_date[1:4]) AS distinct_years
        FROM src
    """,
    "city_state_values": """
        SELECT
            coalesce(nullif(trim(upper(city)), ''), '<BLANK>')  AS city,
            coalesce(nullif(trim(upper(state)), ''), '<BLANK>') AS state,
            count(*) AS rows
        FROM src
        GROUP BY 1, 2
        ORDER BY rows DESC
        LIMIT 15
    """,
    "zip_quality": """
        SELECT
            count(*)                                              AS total_rows,
            count(*) FILTER (WHERE zip IS NULL OR trim(zip) = '') AS null_or_blank_zip,
            count(*) FILTER (WHERE NOT regexp_matches(coalesce(zip, ''), '^[0-9]{5}$'))
                                                                  AS not_five_digits,
            count(DISTINCT zip)                                   AS distinct_zips
        FROM src
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="profile_entities",
        description="Read-only entity-resolution profiling of a raw food inspections Parquet.",
    )
    parser.add_argument("--parquet", type=Path, help="Parquet to profile. Default: latest raw.")
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
    parquet_path: Path = args.parquet or latest_parquet(settings.food_inspections_raw_dir)

    selected: list[str] = list(args.only) if args.only else list(PROFILES)
    unknown = [n for n in selected if n not in PROFILES]
    if unknown:
        print(f"Unknown profile(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    conn = duckdb.connect(database=":memory:")
    try:
        # One view, so each profile below is parameter-free and readable.
        # CREATE VIEW cannot take a bound parameter, so the path goes through
        # the Python relation API instead of being interpolated into SQL.
        conn.read_parquet(str(parquet_path)).create_view("src")
        print(f"<!-- generated by scripts/profile_entities.py from {parquet_path.name} -->\n")
        for name in selected:
            print(run_profile(conn, name))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
