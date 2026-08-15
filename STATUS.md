# Current Status

**Last updated:** 2026-08-15
**Current component:** 1 of 21 — Project Foundation + Chicago Food Inspection Ingestion
**State:** Component 1 complete and verified.

---

## Completed

### Component 1 — Project foundation + Chicago food inspection ingestion

* **Project foundation.** Python 3.12 + uv, `src/` layout, hatchling build,
  `uv.lock` committed. Ruff, mypy (strict), pytest configured. Minimal GitHub
  Actions CI. Optional pre-commit config.
* **Configuration.** Pydantic Settings, `SENTINEL_` env prefix, `.env` support.
  No dataset ID, endpoint, path or limit is hardcoded at a call site.
* **Socrata API investigation.** Endpoint, dataset, pagination, page-size
  ceiling, ordering behaviour, error shape and value encoding all verified
  against the live API and documented in `docs/api/socrata_findings.md`.
* **Socrata client.** Explicit `$limit`/`$offset`/`$order` pagination as a
  generator; runtime field discovery; bounded exponential-backoff retry on
  429/5xx/timeouts; immediate failure on other 4xx; strict response validation.
* **Raw ingestion.** Records → all-`Utf8` Polars frame → timestamped, zstd
  Parquet under `data/raw/food_inspections/`. Nothing is cast, cleaned,
  filtered or deduplicated.
* **Manifest.** JSON provenance sidecar per raw file: source, dataset, UTC
  timestamp, code version, mode, page/request parameters, row count, columns,
  Socrata-declared types, Parquet schema, size, SHA-256.
* **DuckDB query layer.** Named descriptive queries over raw Parquet via
  `read_parquet()`. In-memory, no schema DDL.
* **CLI.** `sentinel ingest` (`--dev` / `--limit N` / `--full`) and
  `sentinel query`, built on stdlib argparse.
* **Logging.** Python `logging` throughout; no `print()` in `src/sentinel`.
* **Tests.** 77 unit tests (HTTP mocked with respx) + 3 opt-in live tests.
* **Documentation.** README, raw data contract, verified API findings, 5 ADRs,
  plus STATUS / MEMORY / HANDOFF.

---

## In Progress

Nothing. Component 1 is closed.

---

## Not Started

Components 2–21. No code exists for any of them.

| # | Component | State |
|---|---|---|
| 2 | Entity resolution | Not started — **next** |
| 3 | Target construction | Not started |
| 4 | As-of feature engineering | Not started |
| 5 | Temporal evaluation framework | Not started |
| 6 | Baseline models | Not started |
| 7 | XGBoost / LightGBM | Not started |
| 8 | Neural baseline | Not started |
| 9 | Probability calibration | Not started |
| 10 | Inspector-effect modelling | Not started |
| 11 | SHAP explainability | Not started |
| 12 | Fairness auditing | Not started |
| 13 | Deterministic statutory policy engine | Not started |
| 14 | Constrained scheduling | Not started |
| 15 | OR-Tools routing | Not started |
| 16 | Deferral / human-review gate | Not started |
| 17 | LangGraph orchestration | Not started |
| 18 | LLM-generated inspector briefings | Not started |
| 19 | Deterministic briefing verification | Not started |
| 20 | Audit trail | Not started |
| 21 | Frontend demo | Not started |

---

## Current Architecture

```text
src/sentinel/
  __init__.py            __version__, stamped into every manifest
  config.py              Pydantic Settings (env prefix SENTINEL_)
  logging_setup.py       configure_logging()
  cli.py                 argparse: ingest, query
  ingest/
    socrata.py           SocrataClient: build_params, discover_fields,
                         fetch_page, iter_pages, bounded retry
    food_inspections.py  orchestration: pages -> all-Utf8 frame -> Parquet
    manifest.py          IngestionManifest model, sha256, read/write
  query/
    duckdb_queries.py    NAMED_QUERIES, latest_parquet, run_named_query
```

Runtime dependencies: httpx, pydantic, pydantic-settings, polars, pyarrow,
duckdb. Nothing for future components.

---

## Current Data Flow

```text
Socrata SODA 2.1  (data.cityofchicago.org/resource/4ijn-s7e5.json)
   |
   |  1. GET ?$limit=1                     discover field list (22 fields)
   |  2. GET $select=<22> $order=inspection_id $limit=N $offset=M   per page
   |     retry 429/5xx/timeout; raise on other 4xx
   v
Page(records, offset, limit, field_names, field_types)      generator
   |
   |  validate: JSON array of objects, else raise
   |  warn on any divergence from the declared field list
   v
Polars DataFrame, every column Utf8; nested location -> JSON string
   |
   v
data/raw/food_inspections/food_inspections_<UTC>.parquet     (zstd)
data/raw/food_inspections/manifest_food_inspections_<UTC>.json
   |
   v
DuckDB read_parquet()  ->  named SQL queries  ->  CLI table
```

Only `data/raw/` is written. `data/interim/` and `data/processed/` are empty.

---

## Tests

**Command:** `uv run pytest` · **Result: 77 passed, 3 deselected** (2026-08-15)

| Area | Coverage |
|---|---|
| Request construction | `$limit`/`$offset`/`$order`/`$select` params; app-token header; input validation |
| Pagination | multi-page offset walking; short-page and empty-page termination; mid-page truncation at `total_limit`; zero-limit; `$select` forwarded to every page |
| Retry | 500 then success; 429; timeout then success; exponential delays (1, 2, 4); bounded budget then raise |
| Non-retryable | 400 and 404 raise on the first attempt, no retry |
| Malformed responses | non-JSON; JSON object instead of array; array of non-objects; missing schema headers |
| Field discovery | discovery request sends no `$order`; skipped when disabled |
| Raw output | all-`Utf8` schema; nested `location` serialized; missing keys null; extra field kept; declared-but-absent field kept as null column; empty dataset; timestamped filenames; output-dir override |
| Manifest | full provenance; SHA-256 matches file on disk; JSON round-trip; schema reported as all `String` |
| DuckDB | row count; unique licences; grouping; `DESCRIBE` reports `VARCHAR`; latest-file resolution; error paths |
| CLI | scope flags required and mutually exclusive; limit resolution; `--log-level` on either side of the subcommand |

**Live tests:** `uv run pytest -m live` — **3 passed** (2026-08-15). Asserts the
real endpoint returns records and schema headers, that values are still
string-encoded, and that ordered pages do not overlap.

**Quality gates**, all passing as of 2026-08-15:

```text
uv run ruff check .            All checks passed!
uv run ruff format --check .   19 files already formatted
uv run mypy src/sentinel       Success: no issues found in 10 source files
```

---

## Verified Data

Development ingestion executed 2026-08-15
(`retrieved_at` `2026-08-15T14:57:03.089773Z`):

| | |
|---|---|
| Command | `sentinel ingest --limit 5000 --page-size 2000` |
| Rows | 5,000 |
| Pages | 3 (2,000 + 2,000 + 1,000 — final page truncated to the limit) |
| Columns | 22 (17 source + 5 `:@computed_region_*`), all `Utf8` |
| File size | 827,350 bytes |
| SHA-256 | `86573f20dbcfa522c305ae96d0f307998e074711e19196ddabfac759b88b31bf` |
| Date range in extract | 2010-01-04 → 2011-05-09 (oldest rows; ordered by `inspection_id`) |
| Unique `license_` | 3,703 of 5,000 rows, 0 null |
| Results breakdown | Pass 3,498 · Fail 1,133 · Pass w/ Conditions 322 · Out of Business 47 |

Dataset total as of 2026-08-15: **314,245 rows** (`$select=count(*)`).

---

## Known Issues

1. **Full ingestion has never been executed.** `--full` is implemented and unit
   tested, but no 314,245-row run has happened. Behaviour at that scale —
   runtime, memory, throttling — is **NOT VERIFIED**.
   Verify with: `uv run sentinel ingest --full`
2. **All pages are held in memory before writing.** `iter_pages` streams, but
   `ingest_food_inspections` accumulates every page's records and writes one
   Parquet file at the end. Fine for 5,000 rows; unmeasured for 314,245.
   Incremental or row-group writing would fix it if needed.
3. **A dev extract is not a random sample.** Ordering by `inspection_id`
   correlates with time, so `--limit N` returns the oldest N inspections. Do
   not estimate distributions or train on a dev extract.
4. **No incremental ingestion.** Every run is a full pull of whatever scope is
   requested. `X-SODA2-Truth-Last-Modified` and `ETag` are documented as leads
   but unused.
5. **No cross-run deduplication.** Two runs produce two independent files with
   overlapping content, by design (ADR 0005).
6. **Schema divergence warns, it does not fail.** An added or removed upstream
   column is logged as a warning and the data is kept. Whether that should be
   fatal is an open question for a later component.
7. **A same-second re-run can collide.** Filenames have one-second resolution.
   Two ingestions starting within the same second would target the same path
   and the second would overwrite the first. Not observed; would need
   sub-second or a counter to eliminate.
8. **CI has never run.** The workflow is committed but no push has yet
   triggered GitHub Actions. **NOT VERIFIED.**

---

## Next Component

**Component 2 — Entity resolution.**

Resolving inspection records to stable establishment identities. `license_` is
not a reliable key: it is not guaranteed unique, stable or present, and the same
physical establishment can appear under varying `dba_name` / `aka_name` /
`address` spellings.

Not started. No code, no directories, no dependencies added.
