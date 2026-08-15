# Sentinel

Risk-prioritized food inspection scheduling using Chicago open data.

The goal of Sentinel is to help a public health department decide **which food
establishments to inspect next**, by combining a calibrated risk model with a
deterministic statutory policy engine and constrained scheduling. Most of that
system does not exist yet.

This repository is being built **one component at a time**. Only Component 1 is
implemented. See [STATUS.md](STATUS.md) for the authoritative project state.

---

## Current status

**Component 1 — Project foundation + Chicago food inspection ingestion. Implemented.**

What exists today:

* A reproducible Python 3.12 project managed with `uv`.
* An explicit, paginating, retrying client for the Chicago Data Portal's
  Socrata API.
* Raw ingestion of the Food Inspections dataset (`4ijn-s7e5`) into
  timestamped Parquet files.
* A JSON manifest recording provenance for every ingestion run.
* A DuckDB query layer for inspecting the raw Parquet.
* Unit tests with mocked HTTP, plus an opt-in live smoke test.

Nothing else. No entity resolution, no features, no models, no scheduling.

---

## Architecture

The whole of Component 1 is one path from the API to a queryable raw file:

```text
Chicago Data Portal (Socrata SODA 2.1)
  https://data.cityofchicago.org/resource/4ijn-s7e5.json
                │
                │  GET ?$limit=1        field discovery (see note below)
                │
                │  GET  $limit / $offset / $order=inspection_id / $select
                │  bounded retry on 429, 5xx, timeouts
                ▼
        SocrataClient.iter_pages()          src/sentinel/ingest/socrata.py
                │  generator, one page resident at a time
                ▼
        response validation                 JSON array of objects, or raise
                │  X-SODA2-Fields / X-SODA2-Types captured
                ▼
        records -> Polars DataFrame         all columns Utf8, no coercion
                │
                ▼
        data/raw/food_inspections/
          food_inspections_<UTC>.parquet    zstd, timestamped, never overwritten
          manifest_food_inspections_<UTC>.json
                │
                ▼
        DuckDB  read_parquet(...)           src/sentinel/query/duckdb_queries.py
                │
                ▼
              SQL result
```

### Raw stays raw

The Socrata API returns **every value as a JSON string**, including columns it
declares as `number` and `calendar_date`:

```json
{"inspection_id": "2641210", "inspection_date": "2026-08-14T00:00:00.000"}
```

The raw Parquet preserves that exactly: every column is written as `Utf8`. No
casting, no parsing, no cleaning. Two reasons:

1. **Fidelity.** A cast turns any unexpected value into a silent null. The raw
   layer must match the source so that a data quality problem is discoverable
   rather than already destroyed.
2. **Separation of concerns.** Typing and semantic interpretation are modelling
   decisions. They belong in a later component where they can be tested and
   documented, not smuggled into the download step.

The nested `location` object is serialized to its JSON string rather than
flattened or dropped, so nothing is lost.

### Why there is a field-discovery request

Stable pagination requires `$order`. This endpoint, however, **drops its five
`:@computed_region_*` columns** (Socrata-generated ward, community area, census
tract and zip-code spatial joins) from both the response and the schema header
whenever `$order` is present — unless they are named explicitly in `$select`.

Hardcoding a 22-column `$select` would fix that but create a worse problem: a
new upstream column would then be silently excluded. So ingestion instead issues
one unordered `?$limit=1` request to discover the current field list, and
selects exactly those fields. One extra request buys a complete raw layer that
still adapts to upstream schema changes.

Controlled by `SENTINEL_INCLUDE_COMPUTED_REGIONS` (default `true`). Full detail
in [docs/api/socrata_findings.md](docs/api/socrata_findings.md) §6.

---

## Data source

**Chicago Data Portal**, powered by Socrata (SODA 2.1).

| | |
|---|---|
| Dataset | Food Inspections |
| Dataset ID | `4ijn-s7e5` |
| Endpoint | `https://data.cityofchicago.org/resource/4ijn-s7e5.json` |
| Authentication | None required |
| Pagination | `$limit` + `$offset`, ordered by `$order=inspection_id` |
| Portal page | <https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5> |

The API is used directly. No HTML scraping, no manually downloaded CSV, no
third-party SDK wrapping the pagination.

An application token is **not** required. Socrata app tokens only relieve
anonymous rate-limit throttling; they grant no additional data access. The
optional `SENTINEL_SOCRATA_APP_TOKEN` setting exists for that purpose alone.

Detailed API findings, verified against the live service, are in
[docs/api/socrata_findings.md](docs/api/socrata_findings.md). The raw data
contract is in
[docs/data_contracts/food_inspections_raw.md](docs/data_contracts/food_inspections_raw.md).

---

## Running locally

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
# install dependencies into .venv
uv sync

# optional: start from the documented defaults
cp .env.example .env
```

### Development ingestion

Small, fast pull. Use this while working.

```bash
uv run sentinel ingest --dev              # uses SENTINEL_DEV_ROW_LIMIT (default 5000)
uv run sentinel ingest --limit 1000       # explicit row cap
```

### Full ingestion

Pulls the entire dataset (~314k rows at time of writing).

```bash
uv run sentinel ingest --full
```

### Other options

```bash
uv run sentinel ingest --limit 5000 --page-size 1000 --log-level DEBUG
uv run sentinel ingest --limit 5000 --output-dir /tmp/sentinel-raw
```

One of `--dev`, `--limit`, or `--full` is required. There is no default scope,
so a bare `sentinel ingest` cannot accidentally trigger a full download.

### Querying the raw data

```bash
uv run sentinel query --list                       # show available queries
uv run sentinel query --name row_count             # uses the most recent raw file
uv run sentinel query --name inspection_types
uv run sentinel query --name results_breakdown
uv run sentinel query --name inspection_date_range
uv run sentinel query --name schema
uv run sentinel query --name row_count --parquet path/to/file.parquet
```

These queries only describe the raw data. They contain no Sentinel filtering or
business logic.

> Because every raw column is `VARCHAR`, aggregating over dates or numbers
> requires an explicit cast in SQL. That is intentional — see
> [ADR 0002](docs/decisions/0002-parquet-raw-storage.md).

> **A development pull is not a random sample.** Pages are ordered by
> `inspection_id`, which correlates with time, so `--limit 5000` returns the
> *oldest* 5,000 inspections (observed: 2010-01-04 to 2011-05-09). Never
> estimate distributions from a dev extract.

---

## Output

Every ingestion run writes two files into `data/raw/food_inspections/`:

```text
food_inspections_20260815T143000Z.parquet          the data, zstd-compressed
manifest_food_inspections_20260815T143000Z.json    the provenance record
```

Filenames embed the UTC retrieval timestamp, so **a re-run never overwrites
previously downloaded raw data**. The raw layer is append-only by construction.

The manifest records source, dataset ID, retrieval timestamp, code version,
mode, row limit, page size, pages fetched, the exact request parameters for
every page, row count, column names, the Socrata-declared field types, the
resulting Parquet schema, output path, file size, and a SHA-256 checksum of the
Parquet file.

Parquet files are gitignored; manifests are committed, so the repository keeps a
history of what was ingested without carrying the bulk data.

---

## Testing

```bash
uv run pytest                 # unit tests, fully offline
uv run pytest -v
uv run pytest -m live         # opt-in: makes one real call to the Chicago API
```

Unit tests mock HTTP at the transport layer with `respx`, so real request
construction, status handling, pagination and retry logic are all exercised
without touching the network. Live tests are marked and deselected by default,
so neither CI nor the normal test run depends on an external service.

Linting and type checking:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/sentinel
```

---

## Repository layout

```text
src/sentinel/
  config.py                    configuration (env-driven, no hardcoded values)
  logging_setup.py             logging configuration
  cli.py                       argparse CLI: ingest, query
  ingest/
    socrata.py                 paginating, retrying Socrata client
    food_inspections.py        orchestration: pages -> Parquet + manifest
    manifest.py                provenance record model and I/O
  query/
    duckdb_queries.py          DuckDB over the raw Parquet
tests/                         unit tests (mocked HTTP) + live smoke test
data/raw|interim|processed/    data layers; contents gitignored
docs/api/                      verified API behaviour
docs/data_contracts/           raw dataset contract
docs/decisions/                architecture decision records
```

---

## Project roadmap

Component 1 is the only implemented component. Everything below is **planned,
not implemented** — no code for any of it exists in this repository.

| # | Component | Status |
|---|---|---|
| 1 | Project foundation + Chicago data ingestion | **Implemented** |
| 2 | Entity resolution | Not implemented |
| 3 | Target construction | Not implemented |
| 4 | As-of feature engineering | Not implemented |
| 5 | Temporal evaluation framework | Not implemented |
| 6 | Baseline models | Not implemented |
| 7 | XGBoost / LightGBM | Not implemented |
| 8 | Neural baseline | Not implemented |
| 9 | Probability calibration | Not implemented |
| 10 | Inspector-effect modelling | Not implemented |
| 11 | SHAP explainability | Not implemented |
| 12 | Fairness auditing | Not implemented |
| 13 | Deterministic statutory policy engine | Not implemented |
| 14 | Constrained scheduling | Not implemented |
| 15 | OR-Tools routing | Not implemented |
| 16 | Deferral / human-review gate | Not implemented |
| 17 | LangGraph orchestration | Not implemented |
| 18 | LLM-generated inspector briefings | Not implemented |
| 19 | Deterministic briefing verification | Not implemented |
| 20 | Audit trail | Not implemented |
| 21 | Frontend demo | Not implemented |

Technologies for later components (modelling libraries, OR-Tools, LangGraph, a
frontend) are deliberately absent from `pyproject.toml`. Each is introduced only
when the component that needs it is built.

---

## License

MIT
