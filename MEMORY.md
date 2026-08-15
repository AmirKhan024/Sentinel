# MEMORY

Compact context for future Claude Code sessions. Not a project description —
see README.md for that and STATUS.md for current state.

**Read this first, then STATUS.md, then HANDOFF.md.**

---

## Working agreement

* **One component at a time.** 21 components are planned; only Component 1
  exists. Never implement ahead. If something belongs to a later component,
  write it down as a TODO or an architectural note instead of building it.
* **No fake completion.** Never claim tests pass, ingestion works, or a schema
  is what it is, without having run the command. Anything unverified must be
  labelled `NOT VERIFIED` with the command that would verify it.
* **Introduce a technology only when the component needing it is being built.**
  This is why there is no pandas, no PostgreSQL, no OR-Tools, no LangGraph.
* **Explain before abstraction.** Simple explicit code beats clever layering.
* Update STATUS.md at every milestone. It must reflect reality.

---

## Hard constraints — do not change without a very good reason

1. **`$order=inspection_id` on every paged request.** Socrata does not
   guarantee row order; without a total order, `$offset` paging can duplicate or
   skip rows. `build_params()` raises if the order column is empty. This is the
   correctness backbone of ingestion.
2. **Raw Parquet is all `Utf8`.** No casting at ingestion, ever. The API returns
   every value as a string; the raw layer preserves that exactly. Typing belongs
   to a later component. (ADR 0002)
3. **`data/raw/` is append-only.** Timestamped filenames, never overwritten. No
   `latest.parquet` pointer — resolving "latest" is a read-time concern handled
   by `latest_parquet()`. (ADR 0005)
4. **Non-retryable 4xx must raise immediately.** Only 429, 5xx, timeouts and
   transport errors are retried. A 400 is our bug; retrying hides it.
5. **No `print()` in `src/sentinel`.** Logging only, except for the CLI's final
   result lines, which are deliberate stdout output.
6. **Live tests stay deselected by default** (`addopts = -m 'not live'`). CI
   must never depend on the Chicago API being reachable.
7. **Never commit raw data.** `.gitignore` excludes Parquet but whitelists
   `manifest_*.json`, so provenance is versioned and bulk data is not.

---

## Key API facts (verified 2026-08-15, live)

* Endpoint `https://data.cityofchicago.org/resource/4ijn-s7e5.json`, no auth.
* **314,245 rows** total at that date.
* **Every value is a JSON string**, including `number` and `calendar_date`
  columns. `location` is the one nested object.
* `$limit=60000` works — there is no 50k cap on this endpoint.
* Pagination ends on a short page or an empty page. There is no cursor and no
  "has more" flag.
* Errors are JSON with an `errorCode` (e.g. `query.soql.no-such-column`) + 4xx.
* **`$order` suppresses the 5 `:@computed_region_*` columns** unless they are
  explicitly named in `$select`. This is why ingestion makes an extra unordered
  `?$limit=1` request to discover the field list, then selects it. Do not
  replace that with a hardcoded column list — a new upstream column would then
  be silently dropped. See `docs/api/socrata_findings.md` §6.
* Response headers `X-SODA2-Fields` / `X-SODA2-Types` carry the declared schema
  on every request. Positionally aligned arrays.
* `X-SODA2-Truth-Last-Modified` and `ETag` exist and are unused — leads for a
  future incremental-ingestion component.

---

## Technology decisions and why

| Choice | Why | ADR |
|---|---|---|
| Python 3.12 + uv | downstream ML ecosystem is Python; uv is fast and gives a committed lockfile | 0001 |
| httpx (not requests/sodapy) | modern client, respx mocks it at transport level; SDKs hide the pagination we most need to see | 0004 |
| Hand-written pagination | highest-risk logic in the component; must be visible and unit-testable | 0004 |
| Parquet, all `Utf8` | columnar + self-describing + compresses well; strings keep the raw layer faithful | 0002 |
| Polars (not pandas) | fast, explicit schema control, `pl.Utf8` enforcement is trivial | 0002 |
| DuckDB in-memory | reads Parquet in place, no load step, no server, real analytical SQL | 0003 |
| argparse (not Typer) | ~4 flags across 2 subcommands; Typer would add 3 deps for help-text polish | — |
| respx | mocks httpx at the transport layer, so real request/status/retry code runs | — |
| JSON manifest sidecar | human-readable, diffable, greppable, zero infrastructure | — |

---

## Important paths

```text
src/sentinel/ingest/socrata.py            the client. Most important file.
src/sentinel/ingest/food_inspections.py   orchestration
src/sentinel/ingest/manifest.py           provenance record
src/sentinel/query/duckdb_queries.py      NAMED_QUERIES live here
src/sentinel/config.py                    every tunable setting
data/raw/food_inspections/                output: parquet + manifest_*.json
docs/api/socrata_findings.md              verified API behaviour — read before
                                          touching the client
docs/data_contracts/food_inspections_raw.md   what a raw file guarantees
docs/decisions/                           5 ADRs
```

---

## Commands

```bash
uv sync                                       # setup
uv run sentinel ingest --dev                  # small pull (SENTINEL_DEV_ROW_LIMIT)
uv run sentinel ingest --limit 5000           # explicit cap
uv run sentinel ingest --full                 # entire dataset (never yet run)
uv run sentinel query --list
uv run sentinel query --name row_count
uv run pytest                                 # 77 unit tests, offline
uv run pytest -m live                         # 3 live tests, hits the real API
uv run ruff check . && uv run ruff format --check .
uv run mypy src/sentinel
```

One of `--dev`, `--limit`, `--full` is required — there is no default scope, so
a bare `sentinel ingest` cannot accidentally pull 314k rows.

---

## Naming conventions

* Raw files: `food_inspections_<YYYYMMDD>T<HHMMSS>Z.parquet` (UTC, start time).
* Manifests: `manifest_<parquet stem>.json`, same directory. The `manifest_`
  prefix is what `.gitignore` whitelists — keep it.
* Env vars: `SENTINEL_` prefix, matching the `Settings` field name.
* Private helpers are `_prefixed`; tests import public names only.

---

## Assumptions being made

* `inspection_id` is unique and totally orderable. Verified consistent with
  observed pagination; not proven across the whole dataset.
* The Chicago dataset stays publicly available without authentication.
* A single machine can hold a full pull in memory. **Unverified at 314k rows.**
* `:@computed_region_*` values are Socrata-derived, not authoritative source
  data, and are re-derivable from latitude/longitude.

---

## Open design questions

1. **Should schema divergence be fatal?** Currently a warning. If Chicago drops
   a column Sentinel depends on, ingestion succeeds with a null column. A
   later component may want a declared required-column set that fails loudly.
2. **Incremental ingestion.** Full pulls will get wasteful. `ETag` /
   `X-SODA2-Truth-Last-Modified` / a `$where` on `inspection_date` are the
   options. Deferred until a component needs freshness.
3. **Memory at full scale.** All pages accumulate before the single Parquet
   write. If `--full` proves heavy, write per-page row groups instead.
4. **Same-second filename collision.** One-second filename resolution means two
   runs starting in the same second collide. Not observed; needs sub-second
   precision or a counter if it ever matters.
5. **Is `license_` ever a usable key?** Component 2's core question. Do not
   assume it is unique, stable, or present.

---

## Lessons learned this session

* **Probe the API before writing the client.** Two behaviours would have been
  wrong if assumed from documentation: the absent 50k page cap, and the
  string-encoding of every value.
* **`$order` changing the returned column set was not documented anywhere.** It
  was found only by comparing the field count between an exploratory request
  and the real paginated one. Compare what you *get* against what you *expected*
  at every step.
* **An empty result still carries schema headers.** The first implementation
  lost the schema on a zero-row pull because `iter_pages` returns before
  yielding an empty page. Fixed by retaining the last-seen schema on the client.
* **DuckDB's `DESCRIBE` columns are `column_name` / `column_type`**, not
  `name` / `type`.
