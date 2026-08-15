# HANDOFF

For the next Claude Code session, which will open this repository with no
conversation history.

**Written:** 2026-08-15 · **Component 1 complete.**

---

## What was completed

**Component 1 — Project foundation + Chicago food inspection ingestion.**

A reproducible Python project and a single, explicit, tested path from the
Chicago Data Portal's Socrata API to a provenance-tracked raw Parquet file that
DuckDB can query.

Concretely:

* Python 3.12 / uv project, `src/` layout, committed `uv.lock`.
* Ruff, mypy (strict), pytest, minimal GitHub Actions CI, optional pre-commit.
* Pydantic-Settings configuration; nothing hardcoded at a call site.
* The Socrata API was **investigated live** before any client code was written;
  findings are in `docs/api/socrata_findings.md`.
* `SocrataClient` — explicit `$limit`/`$offset`/`$order` pagination as a
  generator, runtime field discovery, bounded exponential-backoff retry,
  immediate failure on non-retryable 4xx, strict response validation.
* Raw ingestion to timestamped zstd Parquet, every column `Utf8`.
* A JSON manifest per raw file with full provenance and a SHA-256.
* DuckDB named descriptive queries over the raw Parquet.
* An argparse CLI: `sentinel ingest`, `sentinel query`.
* 77 unit tests (HTTP mocked) + 3 opt-in live tests.
* README, raw data contract, API findings, 5 ADRs, STATUS / MEMORY / HANDOFF.

Component 2 was **not** started.

---

## Current repository state

```text
Sentinel/
  README.md  STATUS.md  MEMORY.md  HANDOFF.md
  pyproject.toml  uv.lock  .gitignore  .env.example  .pre-commit-config.yaml
  .github/workflows/ci.yml
  src/sentinel/
    __init__.py  config.py  logging_setup.py  cli.py
    ingest/  __init__.py  socrata.py  food_inspections.py  manifest.py
    query/   __init__.py  duckdb_queries.py
  tests/
    __init__.py  conftest.py
    test_socrata.py  test_food_inspections.py  test_manifest.py
    test_duckdb_queries.py  test_cli.py  test_smoke_live.py
  data/
    raw/food_inspections/   one Parquet (gitignored) + one manifest (committed)
    interim/  processed/    empty, .gitkeep only
  docs/
    api/socrata_findings.md
    data_contracts/food_inspections_raw.md
    decisions/0001..0005
```

Branch `main`. Remote `origin` → `https://github.com/AmirKhan024/Sentinel.git`.
See the end of this file for the verified push status.

---

## Important files

Read in this order if you are new to the repository:

| File | Why it matters |
|---|---|
| `MEMORY.md` | working agreement, hard constraints, verified API facts |
| `STATUS.md` | what is done, what is not, known issues, verified numbers |
| `docs/api/socrata_findings.md` | **read before touching the client.** Live-verified API behaviour, including two non-obvious traps |
| `docs/data_contracts/food_inspections_raw.md` | what a raw file guarantees; the all-strings contract and its caveats |
| `src/sentinel/ingest/socrata.py` | the most important code. Pagination + retry |
| `src/sentinel/ingest/food_inspections.py` | orchestration, all-`Utf8` frame construction |
| `src/sentinel/config.py` | every tunable setting |
| `docs/decisions/` | why the architecture is what it is |

---

## Commands

```bash
uv sync                                  # setup, installs .venv

uv run sentinel ingest --dev             # small pull (SENTINEL_DEV_ROW_LIMIT, 5000)
uv run sentinel ingest --limit 5000      # explicit row cap
uv run sentinel ingest --full            # entire dataset — NEVER YET RUN
uv run sentinel ingest --limit 5000 --page-size 2000 --log-level DEBUG

uv run sentinel query --list
uv run sentinel query --name row_count
uv run sentinel query --name schema
uv run sentinel query --name results_breakdown

uv run pytest                            # 77 unit tests, fully offline
uv run pytest -m live                    # 3 live tests, hits the real API
uv run ruff check . && uv run ruff format --check .
uv run mypy src/sentinel
```

One of `--dev`, `--limit`, `--full` is required. There is no default scope.

---

## What was tested

**Executed on 2026-08-15, with real output:**

| Check | Result |
|---|---|
| `uv sync` | environment built |
| `uv run pytest` | **77 passed**, 3 deselected |
| `uv run pytest -m live` | **3 passed** against the real Chicago API |
| `uv run ruff check .` | All checks passed |
| `uv run ruff format --check .` | 19 files already formatted |
| `uv run mypy src/sentinel` | Success, no issues in 10 source files |
| `sentinel ingest --limit 5000 --page-size 2000` | 5,000 rows, 3 pages, Parquet + manifest written |
| `sentinel query --name ...` | row_count, schema, unique_licenses, inspection_date_range, inspection_types, results_breakdown, facility_types all returned real results |

**Verified ingestion output** (`retrieved_at` `2026-08-15T14:57:03.089773Z`):

* 5,000 rows, 3 pages (2,000 + 2,000 + 1,000)
* 22 columns, all `Utf8` / DuckDB `VARCHAR`
* 827,350 bytes
* sha256 `86573f20dbcfa522c305ae96d0f307998e074711e19196ddabfac759b88b31bf`
* dates 2010-01-04 → 2011-05-09; 3,703 unique `license_`; 0 null licences
* results: Pass 3,498 · Fail 1,133 · Pass w/ Conditions 322 · Out of Business 47

**NOT VERIFIED:**

* **Full ingestion at 314,245 rows.** `--full` is implemented and unit tested,
  but has never been executed. Runtime, memory and throttling behaviour at that
  scale are unknown. Verify with `uv run sentinel ingest --full`.
* **GitHub Actions CI.** The workflow is committed but has never run.

---

## Known issues

1. `--full` never executed — see above.
2. All pages accumulate in memory before the single Parquet write. Fine at
   5,000 rows; unmeasured at 314,245. Fix by writing row groups per page if it
   becomes a problem.
3. A dev extract is **not a random sample** — ordering by `inspection_id`
   correlates with time, so `--limit N` returns the *oldest* N inspections. Do
   not estimate distributions or train on one.
4. Schema divergence logs a warning rather than failing. Open question.
5. No incremental ingestion; every run is a full pull of its scope.
6. Filenames have one-second resolution, so two runs starting within the same
   second would collide. Not observed.

---

## Decisions made

Full reasoning is in `docs/decisions/`. The load-bearing ones:

1. **Python 3.12 + uv**, dependencies added only when a component needs them.
   Six runtime deps today. (ADR 0001)
2. **Parquet, every column `Utf8`.** The API returns everything as strings;
   casting would create silent nulls and break the raw guarantee. Downstream
   components must cast explicitly. (ADR 0002)
3. **DuckDB in-memory over Parquet**, no load step, no server, no PostgreSQL
   yet. (ADR 0003)
4. **Direct SODA API calls with hand-written pagination**, not `sodapy`, not
   scraping, not a manual CSV. `$order` is mandatory. Retry only transient
   failures. (ADR 0004)
5. **raw / interim / processed separation, raw append-only**, timestamped
   filenames, manifests committed and data gitignored. (ADR 0005)
6. **Runtime field discovery** rather than a hardcoded `$select`, because
   `$order` suppresses the `:@computed_region_*` columns and a stale hardcoded
   list would silently drop any new upstream column.
7. **argparse over Typer** — the CLI has ~4 flags; Typer would add three
   dependencies for help-text ergonomics.

---

## Do not change

* `$order=inspection_id` on every paged request. Correctness depends on it.
* The all-`Utf8` raw contract. No casting in the ingestion layer.
* Append-only `data/raw/` with timestamped filenames.
* Immediate failure on non-retryable 4xx.
* Live tests deselected by default.
* The `manifest_` filename prefix — `.gitignore` whitelists it.
* The runtime field-discovery request — do not replace it with a hardcoded
  column list.

---

## Next task

**Component 2 — Entity resolution.**

Resolve inspection records to stable establishment identities.

The problem: `license_` is not a reliable key. It is not guaranteed unique,
stable, or present, and the same physical establishment appears under varying
`dba_name` / `aka_name` / `address` spellings across years. Everything
downstream — target construction, as-of features, temporal splits — depends on
knowing which rows describe the same establishment.

Nothing for this exists yet: no code, no directories, no dependencies.

---

## Recommended first action next session

Do **not** start writing entity-resolution code first. Start by looking at the
data, because the resolution strategy should follow from what is actually there.

1. Read `MEMORY.md`, then `STATUS.md`, then
   `docs/data_contracts/food_inspections_raw.md`.
2. Get a realistic extract. The committed development extract is the *oldest*
   5,000 rows and is not representative. Either run
   `uv run sentinel ingest --full` (and thereby also close the biggest
   NOT VERIFIED item in STATUS.md), or take a larger dev pull knowing its bias.
3. Use the existing DuckDB layer to characterise the entity problem before
   designing anything. For example:
   * how many distinct `license_` values, and how many rows have `license_`
     blank or `"0"`
   * whether one `license_` ever maps to multiple `dba_name` / `address` values
   * whether one `(dba_name, address)` ever maps to multiple `license_` values
   * how `aka_name` differs from `dba_name`
   * how address strings vary for what is plainly the same location
4. Write the findings into a new `docs/data_contracts/` or `docs/analysis/`
   note **before** implementing. The Component 1 pattern — investigate, document,
   then build — is what made the API traps findable.
5. Only then design the resolution approach, and confirm it before implementing.

Ad-hoc SQL against the raw file, if you need something the named queries do not
cover:

```python
import duckdb
from pathlib import Path
from sentinel.query.duckdb_queries import latest_parquet

path = latest_parquet(Path("data/raw/food_inspections"))
duckdb.sql(f"SELECT license_, count(*) n FROM read_parquet('{path}') "
           "GROUP BY 1 ORDER BY n DESC LIMIT 20").show()
```

**Reminder: one component at a time.** Component 2 is entity resolution only.
No target construction, no features, no models.
