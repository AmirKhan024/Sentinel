# Current Status

**Last updated:** 2026-08-16
**Current component:** 2 of 21 — Entity Resolution
**State:** Component 2 complete and verified against the full 314,245-row snapshot.

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

### Component 2 — Entity resolution

Maps every inspection to a stable `establishment_id` representing a **physical
food-service premises**. Deterministic, rule-based and fully auditable: no
fuzzy scores, no clustering library, no LLM, and no new dependency.

* `scripts/profile_entities.py` — 36 read-only DuckDB profiles that
  characterised the problem before any resolver code was written. Runs in 8 s
  over 314,245 rows.
* `docs/analysis/entity_resolution_findings.md` — the measurements, and the
  design decisions each one forced.
* `src/sentinel/entity/` — normalization, node construction, blocking, named
  evidence rules with vetoes, union-find clustering with invariants and a
  deterministic split ladder, validation, and the three output tables.
* `sentinel resolve` — CLI, with `--dry-run` and `--report`.
* Output under `data/interim/entity_resolution/`, contract in
  `docs/data_contracts/establishment_assignments.md`.
* ADR 0006 (why rules rather than probabilistic linkage) and ADR 0007 (the
  identifier scheme and its stability limits).

**Verified on the full snapshot:** 314,245 rows → 51,099 nodes → **35,859
establishments** in 43 s. All nine structural validation checks pass. Resolving
a seeded random permutation of every input row produces a byte-identical
mapping.

---

## In Progress

Nothing. Components 1 and 2 are closed.

---

## Not Started

Components 3–21. No code exists for any of them.

| # | Component | State |
|---|---|---|
| 3 | Target construction | Not started — **next** |
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
  cli.py                 argparse: ingest, query, resolve
  manifest.py            generic sha256 / read / write helpers
  ingest/
    socrata.py           SocrataClient: build_params, discover_fields,
                         fetch_page, iter_pages, bounded retry
    food_inspections.py  orchestration: pages -> all-Utf8 frame -> Parquet
    manifest.py          IngestionManifest model (helpers re-exported)
  query/
    duckdb_queries.py    NAMED_QUERIES, latest_parquet, run_named_query
  entity/                Component 2
    models.py            frozen structures, MatchTier, DEFAULT_THRESHOLDS
    normalize.py         normalize_name / _address / _geo / _license / _zip
    nodes.py             build_nodes, IDENTITY_COLUMNS, blacklisted_coordinates
    blocking.py          spatial / coordinate / licence blocks, candidate_pairs
    evidence.py          compute_signals, evaluate_pair, vetoes, haversine_m
    unionfind.py         UnionFind (find / union / components)
    cluster.py           build_clusters, check_invariants, establishment_id_for
    validate.py          validate_output, format_report, has_failures
    writer.py            the three table schemas and builders
    resolve.py           resolve_establishments (the only I/O)
scripts/
  profile_entities.py    36 read-only profiles; analysis tooling, not library
```

Runtime dependencies: httpx, pydantic, pydantic-settings, polars, pyarrow,
duckdb. **Component 2 added none.** Union-find and haversine are written out
rather than importing networkx or a geo library; string similarity is token-set
equality over frozensets rather than rapidfuzz. Nothing for future components.

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


Component 2
-----------
data/raw/food_inspections/*.parquet
   |
   |  read 9 identity columns only; results/violations/risk/inspection_date
   |  are never read, which is the leakage boundary
   v
normalize -> 51,099 distinct identity nodes
   |
   |  block on (zip, house number), exact coordinate, licence
   |  every non-licence block requires location agreement
   v
evaluate 335,393 candidate pairs against named rules
   |
   |  vetoes first (V1-V4), then strong (S1-S3), probable (P1-P2),
   |  ambiguous (A1-A2, recorded but never merged), no-match (N0-N2)
   v
union-find -> cluster invariants -> deterministic split ladder
   |
   v
data/interim/entity_resolution/
   establishment_assignments_<UTC>.parquet    314,245 rows
   establishments_<UTC>.parquet                35,859 rows
   entity_resolution_edges_<UTC>.parquet       90,643 rows (audit trail)
   manifest_establishment_assignments_<UTC>.json
```

`data/raw/` and `data/interim/` are written. `data/processed/` is still empty.

---

## Tests

**Command:** `uv run pytest` · **Result: 342 passed, 3 deselected** (2026-08-16)

Component 2 added 265 tests. Quality gate, all passing:

```bash
uv run pytest                  # 342 passed, 3 deselected
uv run ruff check .            # All checks passed
uv run ruff format --check .   # 56 files already formatted
uv run mypy src/sentinel scripts   # no issues in 23 source files
```

| Component 2 area | Coverage |
|---|---|
| Normalization (103) | every name and address rule; parametrized unit markers, directionals, suffixes; idempotence properties; cases asserting digits and descriptive words are *not* stripped |
| Nodes and blocking (21) | signature collapse; node ids stable under row order; numeric id comparison; sentinel exclusion; oversized blocks skipped and reported; canonical pair ordering |
| Evidence (35) | one test per rule S1–N2 and per veto V1–V4, each veto pitted against otherwise-strong agreement; symmetry property; haversine against known distances |
| Union-find (10) | components identical under 10 seeded edge shuffles and reversed item order |
| Clustering (22) | id anchoring and numeric comparison; every invariant tripped; all three rungs of the split ladder; content-hash sensitivity |
| Validation (14) | a passing and a failing case for each error check; distributional checks proven non-fatal |
| Integration (25) | a 10-row scenario with a known correct grouping; schema contract tests on names *and* dtypes; manifest round-trip; dry-run writes nothing; empty and all-sentinel-licence inputs |
| Determinism | same input twice, and a seeded row-shuffled input, yield identical mappings — asserted in unit tests and verified separately on all 314,245 real rows |
| Regression (22) | 12 real cases copied verbatim from the snapshot, including the O'Hare over-merge the first run produced |

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

### Full ingestion (2026-08-16) — now verified

| Property | Value |
|---|---|
| File | `food_inspections_20260816T070911Z.parquet` |
| sha256 | `7d3c4069340a68d197204c6cca9fca6399c6565bc3668760f145f43cd377ad38` |
| Rows | 314,245 |
| Pages | 7 (6 × 50,000 + a 14,245-row short page) |
| Size | 48,801,672 bytes |
| Wall time | 69 m 24 s (server-side; one recovered `ReadTimeout`) |
| Peak RSS | ~966 MB |
| Date range | 2010-01-04 → 2026-08-14 |

### Entity resolution (2026-08-16)

| Property | Value |
|---|---|
| Nodes | 51,099 |
| **Establishments** | **35,859** |
| Distinct usable licences (comparison) | 48,963 |
| Reduction ratio | 0.73 |
| Candidate pairs | 335,393 (29,280 strong, 2,915 probable, 747 ambiguous) |
| Runtime | 43 s |
| Single-inspection establishments | 6,084 — down 51% from 12,356 under naive licence grouping |
| Rows with unusable licence | 850 (0.27%) |
| Oversized blocks / blacklisted coordinates | 0 / 0 |

Key data facts driving the design: **18.47% of (name, address) pairs hold more
than one licence** (max 47); the `'0'` licence sentinel covers 323 distinct
names; case and whitespace alone collapse 33,261 address strings to 20,313;
coordinate spread within an address is exactly 0 m; 75.5% of same-place licence
pairs overlap in time rather than succeeding one another.

---

## Known Issues

1. **Same-name outlets at a dense address can still merge.** Two outlets of one
   chain at one address with no store number and no distinguishing `aka_name`
   are indistinguishable from the data. `MCDONALD'S` at O'Hare (22 nodes, 20
   licences, 5 names) is the known example. Bounded to mega-addresses and
   surfaced by the cluster-size and address-density checks.
2. **747 ambiguous pairs have never been manually adjudicated.** They are the
   intended review queue, recorded in the edges table.
3. **`establishment_id` is not stable across snapshots.** It is a deterministic
   function of one input file. A later snapshot can merge or split clusters,
   retiring ids. A crosswalk is future work (ADR 0007).
4. **Stadiums and arenas resolve to one establishment.** The United Center is
   one establishment holding 16 licences. Whether an arena is one premises or
   many is definitional, not a defect.
5. **No `--as-of` resolution mode.** Identity is reconstructed from the whole
   snapshot. This is argued to be legitimate rather than leakage, but a strict
   as-of mode is named as future work.
6. **All pages are held in memory before writing.** `ingest_food_inspections`
   accumulates every page's records and writes one Parquet at the end. Measured
   at ~966 MB peak for the full pull, so it was left alone; it would need
   revisiting if the dataset grew several-fold.
7. **A dev extract is not a random sample.** Ordering by `inspection_id`
   correlates with time, so `--limit N` returns the oldest N inspections. Do
   not estimate distributions or train on a dev extract.
8. **No incremental ingestion.** Every run is a full pull of whatever scope is
   requested. `X-SODA2-Truth-Last-Modified` and `ETag` are documented as leads
   but unused.
9. **No cross-run deduplication.** Two runs produce two independent files with
   overlapping content, by design (ADR 0005).
10. **Schema divergence warns, it does not fail.** An added or removed upstream
   column is logged as a warning and the data is kept. Whether that should be
   fatal is an open question for a later component.
11. **A same-second re-run can collide.** Filenames have one-second resolution.
   Two ingestions starting within the same second would target the same path
   and the second would overwrite the first. Not observed; would need
   sub-second or a counter to eliminate.
12. **CI has never run.** The workflow is committed but no push has yet
   triggered GitHub Actions. **NOT VERIFIED.**

---

## Next Component

**Component 3 — Target construction.**

Defining what the model predicts, from the `results` column, over the
establishment identities Component 2 produced.

Not started. No code, no directories, no dependencies added.

Before writing code, investigate the actual data as Component 2 did:

* What exactly is in `results`? The four documented values are `Pass`, `Fail`,
  `Pass w/ Conditions` and `Out of Business`, but the full snapshot has never
  been profiled for the complete value set, casing variants or blanks.
* How does `inspection_type` interact with it? A `License` inspection is not
  the same event as a `Canvass` or a `Complaint`, and pooling them may be wrong.
* Is `Pass w/ Conditions` a pass or a failure for Sentinel's purposes? This is
  a definitional decision that needs an ADR, not a default.
* What does `Out of Business` mean for an establishment's timeline, given that
  Component 2 treats successive tenants at one premises as one establishment?
* Are there multiple inspections of one establishment on one date, and what
  should the target be then?
* `violations` is unstructured free text with a ` | ` separator and averages
  ~1.55 KB per row. Whether it feeds the target or waits for a later component
  is a scoping decision.

Component 2 gives you `inspection_id -> establishment_id`. Join on it; do not
re-derive identity.
