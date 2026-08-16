# HANDOFF

For a fresh Claude Code session picking this repository up. Read `MEMORY.md`
first, then `STATUS.md`, then this file.

**Last session completed Component 2 — Entity Resolution.**
**Next task: Component 3 — Target Construction.**

---

## 1. What was completed

Component 2 maps every inspection to a stable `establishment_id` representing a
**physical food-service premises**, together with an audit trail explaining
every merge and every declined merge.

The work was done in the order the project's engineering philosophy requires:
investigate → document → design → implement → test.

1. **Full ingestion executed and verified** — the largest open `NOT VERIFIED`
   item from Component 1. 314,245 rows, 7 pages, 46.5 MB, 69 m 24 s wall time,
   one recovered `ReadTimeout`, ~966 MB peak RSS. The manifest is committed.
2. **`scripts/profile_entities.py`** — 36 read-only DuckDB profiles that
   characterised the entity problem across the whole snapshot in 8 s.
3. **`docs/analysis/entity_resolution_findings.md`** — the measurements and the
   design decision each one forced. Written *before* any resolver code.
4. **`src/sentinel/entity/`** — ten modules: normalization, node construction,
   blocking, evidence rules, union-find, clustering, validation, writer,
   orchestration, models.
5. **`sentinel resolve`** CLI with `--dry-run` and `--report`.
6. **265 new tests** (77 → 342), including 12 regression cases copied verbatim
   from the real data.
7. **Contract and ADRs** — `docs/data_contracts/establishment_assignments.md`,
   ADR 0006 (why rules, not probabilistic linkage), ADR 0007 (the identifier
   scheme and its stability limits).

**No new dependencies were added.** Union-find and haversine are written out
rather than importing networkx or a geo library; string comparison is token-set
equality over frozensets rather than rapidfuzz.

---

## 2. Current repository state

```text
src/sentinel/
  cli.py                     ingest | query | resolve
  config.py                  + entity_resolution_interim_dir
  manifest.py                NEW: generic sha256 / read / write helpers
  ingest/                    Component 1, unchanged except manifest re-exports
  query/duckdb_queries.py    unchanged
  entity/                    NEW: Component 2
    models.py                frozen structures, MatchTier, DEFAULT_THRESHOLDS
    normalize.py             name / address / geo / licence / zip
    nodes.py                 build_nodes, IDENTITY_COLUMNS
    blocking.py              spatial / coordinate / licence blocks
    evidence.py              signals, vetoes, named rules
    unionfind.py             disjoint-set union
    cluster.py               components, invariants, split ladder
    validate.py              post-resolution checks
    writer.py                the three table schemas
    resolve.py               orchestration; the only module doing I/O
scripts/profile_entities.py  NEW: analysis tooling (in mypy's `files`)
tests/                       342 passing, 3 live deselected
  fixtures/real_cases.py     NEW: 12 real regression cases as literal Python
docs/analysis/               NEW
docs/data_contracts/         + establishment_assignments.md
docs/decisions/              7 ADRs (0006, 0007 new)
data/raw/food_inspections/         full snapshot + 2 manifests (committed)
data/interim/entity_resolution/    3 parquet + manifest (manifest committed)
```

Branch `main`, working tree clean. `.gitignore` gained an interim manifest
whitelist; without it the resolution manifest would have been silently
uncommittable.

---

## 3. The entity-resolution design in brief

### Identity model

```text
inspection_id ──► node ──► establishment_id
                  (a distinct identity signature)
```

A **node** is one distinct tuple of (licence, names, address, unit, zip,
coordinate). 314,245 rows collapse to 51,099 nodes. Matching runs on nodes, so
an audit row is a statement about two ways of recording a place rather than
about two individual visits.

An **establishment** is a *physical premises*. Successive tenants at one address
are the same establishment with a changing name; a commissary holding 47 cart
permits is one establishment. Findings §11.1 argues the tradeoff.

`establishment_id = "EST-" + zero-pad(min(inspection_id), 11)`.

### Matching

```text
normalize → nodes → block → evaluate pairs → union-find → invariants → split ladder
```

Blocks: `(zip, house number)`, exact coordinate, and licence. **No name block.**

Rules, first match wins, id recorded on every edge:

| tier | rules |
|---|---|
| strong | `S1` licence+address · `S2` address+name · `S3` licence+name+near-address |
| probable | `P1` licence+near-address · `P2` name containment at one address |
| ambiguous (never merges) | `A1` one licence at two places · `A2` containment with facility conflict |
| no match | `N0` `N1` `N2` |
| veto (beats everything) | `V1` directional · `V2` unit · `V3` store number · `V4` trade name |

Four properties do the heavy lifting:

1. **Address equivalence is required for every non-licence merge** — this is what
   makes 247 Subways safe without a chain-name list, and what bounds chaining.
2. **Licence inequality is never evidence against a match** — 18.47% of
   establishments hold more than one licence.
3. **Name matching is exact after normalization**, over both `dba_name` and
   `aka_name`. There is no fuzzy tier.
4. **Vetoes outrank agreement.**

Address equivalence has two routes: an equal normalized address key, or an equal
geocoded coordinate within one zip. The second exists because coordinate spread
within an address is exactly 0 m, so the city's geocoder resolves variants no
string rule can.

---

## 4. Verified data findings

Snapshot `7d3c4069340a68d197204c6cca9fca6399c6565bc3668760f145f43cd377ad38`.

| Measurement | Value |
|---|---|
| Rows / nodes / establishments | 314,245 / 51,099 / **35,859** |
| Distinct usable licences | 48,963 (reduction ratio 0.73) |
| Candidate pairs | 335,393 — 29,280 strong, 2,915 probable, 747 ambiguous |
| Runtime | 43 s |
| Establishments holding >1 licence | 8,931 (24.9%), max 62 |
| Single-inspection establishments | 6,084, **down 51%** from 12,356 under licence grouping |
| `(name, address)` pairs with >1 licence | 18.47%, max 47 |
| `'0'` licence sentinel | 323 names, 364 addresses |
| Address strings → normalized | 33,261 → 20,313 (−39% from case/whitespace alone) |
| Long-form street suffixes in the data | **0** (and one long directional) |
| Coordinate spread within an address | **0.00 m** |
| Same-place licence pairs overlapping in time | 75.5% |

---

## 5. Tests

```bash
uv run pytest                       # 342 passed, 3 deselected
uv run pytest -m live               # 3 live tests, hits the real API
uv run ruff check .                 # All checks passed
uv run ruff format --check .        # 56 files already formatted
uv run mypy src/sentinel scripts    # no issues in 23 source files
```

Determinism is asserted in unit tests and verified separately on the full
snapshot: resolving a seeded random permutation of all 314,245 input rows
produces a byte-identical `inspection_id → establishment_id` mapping.

---

## 6. Known limitations — be honest about these

1. **Same-name outlets at a dense address can still merge.** Two outlets of one
   chain at one address with no store number and no distinguishing `aka_name`
   are indistinguishable from the data. `MCDONALD'S` at O'Hare is 22 nodes / 20
   licences / 5 names. This is the residual false-merge risk.
2. **747 ambiguous pairs have never been manually adjudicated.**
3. **`establishment_id` is not stable across snapshots** (ADR 0007). No
   crosswalk exists.
4. **Stadiums and arenas resolve to one establishment.** Definitional, not a bug.
5. **No `--as-of` resolution mode.**
6. **Exact-only name matching** means a genuine typo splits an establishment
   unless licence, address or coordinate carries it.
7. **CI has still never run.** **NOT VERIFIED.**
8. **Ranged house numbers key on the low endpoint**, so a record filed only under
   the high endpoint would not join.

---

## 7. Important decisions

* Output goes to **`data/interim/`**, not `processed/`: ADR 0005 reserves
  processed for model-ready tables, and a crosswalk is a mid-pipeline mapping.
* **No fuzzy name matching.** Measured yield was 0.21% of licences against
  serious risk from 247 Subways and 219 names at one address. The plan's
  documented fallback (`T1 = 1.0`) was reached by measurement, not omission.
* **No temporal logic in matching**, because 75.5% of same-place licence pairs
  overlap rather than succeed. This also keeps `inspection_date` out of the
  matcher entirely, which is a leakage benefit.
* **The street suffix is excluded from the address key** rather than
  canonicalized.
* **Ambiguity is preserved, not resolved.** A false merge contaminates years of
  history; a false split only costs statistical power.

---

## 8. What must NOT be changed

**Component 1 invariants** (unchanged and still binding): `$order=inspection_id`
on every paged request; raw Parquet is all `Utf8` with no casting; `data/raw/` is
append-only and only an ingestion component may write there; non-retryable 4xx
raise immediately; live tests stay deselected by default; raw data is never
committed.

**Component 2 invariants:**

1. **Only identity columns reach the matcher.** `IDENTITY_COLUMNS` in
   `entity/nodes.py` is the leakage boundary. `results`, `violations`, `risk`
   and `inspection_date` are excluded and a test asserts it.
2. **The assignments table carries no dates, counts or outcomes.** That absence
   is the anti-leakage guarantee. Do not add them for convenience.
3. **Every non-licence merge requires address equivalence.** Do not add a
   name-only block or merge rule.
4. **Licence inequality is never evidence against a match.**
5. **Resolution stays deterministic**: content-hashed node ids, canonically
   ordered pairs, components labelled by minimum member, ids from the earliest
   inspection.
6. **The `n_*` columns on the establishments table are audit-only.** They
   summarise the whole history and are not model features.
7. Do not change a normalization or matching rule without re-reading
   `docs/analysis/entity_resolution_findings.md` and updating it. Every rule
   exists because of a specific measurement recorded there, and changing one
   changes `establishment_id` values.

---

## 9. Next task: Component 3 — Target Construction

Define what the model predicts. Scope is target construction **only** — no
features, no models, no calibration, no scheduling.

### Investigate before coding — the Component 2 pattern worked, repeat it

Component 2's design was reversed in six places by measurement. Expect the same.
Add profiles to `scripts/` (or a sibling script) and write a findings document
before designing anything.

Questions the data must answer:

1. **What is actually in `results`?** The contract documents four values
   (`Pass`, `Fail`, `Pass w/ Conditions`, `Out of Business`) but the full
   snapshot has never been profiled for the complete value set, casing variants,
   blanks or nulls.
2. **Is `Pass w/ Conditions` a pass or a failure?** This is the central
   definitional decision and it needs an ADR, not a default. Look at what
   distinguishes those inspections before choosing.
3. **How does `inspection_type` interact with the target?** A `License`
   inspection is a different event from a `Canvass` or a `Complaint`. Pooling
   them may be wrong. Profile the joint distribution of type and result.
4. **What does `Out of Business` mean for a timeline**, given that Component 2
   treats successive tenants at one premises as one establishment? An
   establishment can plausibly go out of business and reopen under a new name.
5. **Are there multiple inspections of one establishment on one date?** If so,
   what is the target for that date? Component 1 does no deduplication.
6. **Does `violations` belong in the target?** It is unstructured free text with
   a ` | ` separator, averaging ~1.55 KB per row. Deciding it is out of scope is
   a legitimate answer, but decide it explicitly.
7. **What is the base rate, and how does it drift over 2010–2026?** A target
   whose prevalence shifts materially over time changes what Component 5's
   temporal evaluation has to do.

### How to join to Component 2

```python
import duckdb

duckdb.sql("""
    SELECT a.establishment_id, r.inspection_date, r.results, r.inspection_type
    FROM read_parquet('data/raw/food_inspections/food_inspections_*.parquet') r
    JOIN read_parquet(
        'data/interim/entity_resolution/establishment_assignments_*.parquet'
    ) a USING (inspection_id)
""").show()
```

Join on `inspection_id`. **Do not re-derive identity** — that is Component 2's
job and its output is the contract.

### Leakage warning specific to Component 3

Component 2 deliberately emits no dates, counts or outcomes so that a join
cannot pull whole-history information into a training row. Component 3 will
necessarily read `results` and `inspection_date`. The moment it does, every
aggregate it computes must be *as-of* a reference date. Keep target definition
(a property of a single inspection) strictly separate from any historical
summary (a property of an establishment up to a date) — the latter belongs to
Component 4.

### Reminder

**One component at a time.** Component 3 is target construction only. No
feature engineering, no models, no XGBoost, no calibration, no OR-Tools, no
LangGraph, no frontend.
