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

**Components 1-4 complete: ingestion, entity resolution, target construction,
and as-of feature engineering.**

What exists today:

* A reproducible Python 3.12 project managed with `uv`.
* An explicit, paginating, retrying client for the Chicago Data Portal's
  Socrata API.
* Raw ingestion of the Food Inspections dataset (`4ijn-s7e5`) into
  timestamped Parquet files. Verified end-to-end on the full 314,245 rows.
* A JSON manifest recording provenance for every ingestion and resolution run.
* A DuckDB query layer for inspecting the raw Parquet.
* **Entity resolution**: a deterministic, explainable mapping from each
  inspection to a stable `establishment_id` representing a physical premises,
  plus an audit table recording why every merge — and every declined merge —
  was decided the way it was.
* **Target construction**: a precise, leakage-safe prediction target -- for each
  establishment-date on which a routine canvass occurred, did that canvass find
  at least one Priority or Priority Foundation violation?
* **As-of feature engineering**: 26 historical features per prediction
  opportunity, each computed strictly from inspections dated *before* that
  inspection, with the boundary enforced in one place and re-checked on every row.
* 745 unit and integration tests with mocked HTTP, plus an opt-in live smoke
  test.

Nothing else. No models, no evaluation framework, no scheduling.

---

## Architecture

Four components, one path from the API to a model-ready training table:

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


        ---- Component 2: entity resolution ----------------------------

        data/raw/food_inspections/*.parquet
                │  identity columns only; no results/violations/risk/date
                ▼
        normalize names, addresses, coordinates   entity/normalize.py
                │
                ▼
        build nodes                               entity/nodes.py
                │  314,245 rows -> 51,099 distinct identity signatures
                ▼
        block on (zip, house), coordinate, licence entity/blocking.py
                │  every non-licence block requires location agreement
                ▼
        evaluate pairs against named rules         entity/evidence.py
                │  S1-S3 strong · P1-P2 probable · A1-A2 ambiguous · V1-V4 veto
                ▼
        union-find + cluster invariants            entity/cluster.py
                │  deterministic split ladder if an invariant fails
                ▼
        data/interim/entity_resolution/
          establishment_assignments_<UTC>.parquet   inspection_id -> establishment_id
          establishments_<UTC>.parquet              one row per establishment
          entity_resolution_edges_<UTC>.parquet     the audit trail
          manifest_establishment_assignments_<UTC>.json


        ---- Component 3: target construction --------------------------

        raw Parquet  +  establishment_assignments
                │  identity columns are Component 2's contract, never re-derived
                ▼
        eligibility gates                         target/construct.py
                │  era >= 2018-07-01 · type == CANVASS
                │  results in {Pass, Pass w/ Conditions, Fail}
                ▼
        parse and classify violations             target/violations.py
                │  PRIORITY / PRIORITY FOUNDATION markers,
                │  narrative text excluded, violation number NOT used
                ▼
        collapse same establishment-date          target/construct.py
                │  one scheduling decision = one row, target = OR
                ▼
        data/interim/target/
          inspection_targets_<UTC>.parquet         313,624 rows
          manifest_inspection_targets_<UTC>.json


        ---- Component 4: as-of feature engineering --------------------

        raw Parquet + assignments + targets
                │
                ▼
        one range join, one temporal condition    features/historical.py
                │  h.inspection_date < t.inspection_date
                │  strictly before, never same-day
                ▼
        26 features in six families               features/definitions.py
                │  canvass history · priority history · windows
                │  context · tenant change · observation
                ▼
        validate: boundary re-derived per row     features/validate.py
                │
                ▼
        data/processed/features/
          as_of_features_<UTC>.parquet             57,727 rows x 33 columns
          manifest_as_of_features_<UTC>.json
```

The **as-of rule** is the whole of Component 4: a feature for the row at
`inspection_date = d` may use only records dated **strictly before** `d`. The
boundary is exclusive because `inspection_date` carries no time component in this
dataset, so same-day records cannot be ordered — and 43 same-day canvass
re-inspections at reference dates provably happened *after* the canvass they
follow. Observable consequence: `days_since_last_canvass` has a minimum of 1 and
contains no zeros.

Reasoning in
[`docs/analysis/as_of_feature_engineering_findings.md`](docs/analysis/as_of_feature_engineering_findings.md)
and [ADR 0010](docs/decisions/0010-as-of-feature-construction.md).

The **target** is: for each establishment-date on which a routine canvass
occurred, did that canvass find at least one **Priority or Priority Foundation**
violation? Not `results == 'Fail'` -- among canvasses, priority violations appear
in 97.9% of `Pass w/ Conditions` inspections, so a result-based label would
mislabel 16,261 of them. The reasoning is in
[`docs/analysis/target_construction_findings.md`](docs/analysis/target_construction_findings.md).

`inspection_date` is the **as-of boundary**: Component 4 may use only information
dated strictly before it.

An **establishment** is a *physical food-service premises*, not a licence and not
a business name. Successive tenants at one address are the same establishment
with a changing name; a mobile-food commissary holding 47 cart permits is one
establishment. The reasoning is in
[`docs/analysis/entity_resolution_findings.md`](docs/analysis/entity_resolution_findings.md).

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

### Resolving establishment identities

```bash
# resolve the most recent raw snapshot, writing three tables + a manifest
uv run sentinel resolve

# compute and validate without writing anything, printing the full report
uv run sentinel resolve --dry-run --report

# resolve a specific file into a specific directory
uv run sentinel resolve --parquet path/to/raw.parquet --output-dir out/
```

The command exits non-zero if a structural validation check fails, so broken
identities stop the pipeline rather than being handed quietly to the next
component. On the full snapshot it takes about 45 seconds.

To ask why two inspections were or were not treated as the same establishment,
filter the edges table on their `node_id`:

```python
import duckdb

duckdb.sql("""
    SELECT rule_id, tier, same_license, same_addr_key, name_exact, left_name, right_name
    FROM read_parquet('data/interim/entity_resolution/entity_resolution_edges_*.parquet')
    WHERE left_node_id = 'N-...' OR right_node_id = 'N-...'
""").show()
```

---

### Building the prediction target

```bash
# build from the latest raw snapshot and the latest Component 2 assignments
uv run sentinel build-target

# construct and validate without writing anything
uv run sentinel build-target --dry-run --report
```

Exits non-zero if a structural validation check fails. Takes about 25 seconds on
the full snapshot.

To see why a row is labelled the way it is, read its evidence span:

```python
import duckdb

duckdb.sql("""
    SELECT establishment_id, inspection_date, target, results, evidence
    FROM read_parquet('data/interim/target/inspection_targets_*.parquet')
    WHERE target = 1
    LIMIT 5
""").show()
```

---

### Building the as-of feature table

```bash
# build from the latest raw snapshot, assignments and targets
uv run sentinel build-features

# construct and validate without writing anything
uv run sentinel build-features --dry-run --report
```

Exits non-zero if a validation check fails — including the temporal invariant,
which is re-derived independently and checked on every row. Takes about 15
seconds on the full snapshot.

```python
import duckdb

duckdb.sql("""
    SELECT establishment_id, inspection_date,
           prior_canvass_count, days_since_last_canvass,
           prior_canvass_priority_rate, target
    FROM read_parquet('data/processed/features/as_of_features_*.parquet')
    LIMIT 5
""").show()
```

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

Entity resolution writes three tables plus its own manifest under
`data/interim/entity_resolution/`, target construction writes one table plus a
manifest under `data/interim/target/`, and feature engineering writes the
model-ready table under `data/processed/features/` — all following the same
rules. [ADR 0011](docs/decisions/0011-processed-layer-for-model-ready-tables.md)
records what makes a table model-ready and therefore eligible for the processed
layer. The full schemas,
identifier semantics and stability guarantees are in
[`docs/data_contracts/establishment_assignments.md`](docs/data_contracts/establishment_assignments.md).
The short version: `establishment_assignments` maps every `inspection_id` to an
`establishment_id` and deliberately carries no dates, counts or outcomes, so a
downstream join cannot pull whole-history information into a training row.

---

## Testing

```bash
uv run pytest                 # unit tests, fully offline
uv run pytest -v
uv run pytest -m live         # opt-in: makes one real call to the Chicago API
```

745 tests pass and 3 live tests are deselected. Unit tests mock HTTP at the
transport layer with `respx`, so real request
construction, status handling, pagination and retry logic are all exercised
without touching the network. Live tests are marked and deselected by default,
so neither CI nor the normal test run depends on an external service.

Linting and type checking:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/sentinel scripts
```

---

## Repository layout

```text
src/sentinel/
  config.py                    configuration (env-driven, no hardcoded values)
  logging_setup.py             logging configuration
  cli.py                       argparse CLI: ingest, query, resolve
  manifest.py                  generic manifest helpers (hash, read, write)
  ingest/
    socrata.py                 paginating, retrying Socrata client
    food_inspections.py        orchestration: pages -> Parquet + manifest
    manifest.py                ingestion provenance model
  query/
    duckdb_queries.py          DuckDB over the raw Parquet
  features/                    Component 4: as-of feature engineering
    definitions.py             FeatureSpec list; the single source of truth
    historical.py              the range join and the temporal boundary
    validate.py                checks, incl. the full-table temporal invariant
    writer.py                  schema derived from the specs
    build.py                   orchestration (the only module doing I/O)
  target/                      Component 3: target construction
    models.py                  frozen structures + the definition constants
    violations.py              deterministic violation parsing/classification
    construct.py               eligibility gates, labelling, same-day collapse
    validate.py                post-construction checks
    writer.py                  the output table schema
    build.py                   orchestration (the only module doing I/O)
  entity/                      Component 2: entity resolution
    models.py                  frozen data structures + DEFAULT_THRESHOLDS
    normalize.py               name / address / geo normalization
    nodes.py                   rows -> distinct identity signatures
    blocking.py                candidate pair generation
    evidence.py                signals, vetoes, named match rules
    unionfind.py               disjoint-set union (no networkx)
    cluster.py                 components, invariants, split ladder
    validate.py                post-resolution checks
    writer.py                  the three output tables
    resolve.py                 orchestration (the only module doing I/O)
scripts/profile_entities.py    read-only entity profiling (analysis, not library)
scripts/profile_target.py      read-only target profiling
scripts/profile_features.py    read-only history-availability profiling
tests/                         unit + integration tests; tests/fixtures/ holds
                               real regression cases as literal Python
data/raw|interim|processed/    data layers; contents gitignored, manifests kept
docs/analysis/                 empirical findings
docs/api/                      verified API behaviour
docs/data_contracts/           raw dataset + entity resolution output contracts
docs/decisions/                architecture decision records
```

---

## Project roadmap

Components 1-4 are implemented. Everything below them is **planned, not
implemented** — no code for any of it exists in this repository.

| # | Component | Status |
|---|---|---|
| 1 | Project foundation + Chicago data ingestion | **Implemented** |
| 2 | Entity resolution | **Implemented** |
| 3 | Target construction | **Implemented** |
| 4 | As-of feature engineering | **Implemented** |
| 5 | Temporal evaluation framework | Next |
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
