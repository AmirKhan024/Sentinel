# Data contract: establishment assignments (Component 2 output)

**Produced by:** `sentinel resolve` (`src/sentinel/entity/`)
**Layer:** `data/interim/entity_resolution/`
**Consumed by:** Component 3 onward
**Design rationale:** `docs/analysis/entity_resolution_findings.md`, ADR 0006, ADR 0007

---

## 1. Purpose

Component 2 answers one question: **which inspection records describe the same
physical establishment?** It produces a mapping from `inspection_id` to a stable
`establishment_id`, plus the evidence that justifies every merge and every
declined merge.

It does not compute features, targets, aggregates or risk scores.

---

## 2. Input

One raw Parquet file from Component 1 (`data/raw/food_inspections/`), all 22
columns `Utf8` per ADR 0002.

Only these nine columns are read:

```
inspection_id  dba_name  aka_name  license_  facility_type
address  zip  latitude  longitude
```

The list is `IDENTITY_COLUMNS` in `sentinel.entity.nodes`, and a test asserts it
excludes `results`, `violations`, `risk` and `inspection_date`. **No outcome
field reaches the matcher.** See §8.

Raw data is never modified. ADR 0005 reserves writes to `data/raw/` for
ingestion components.

---

## 3. Layer choice

`data/interim/`, not `data/processed/`. ADR 0005 defines the processed layer as
analysis- and model-ready output. An establishment crosswalk is a mid-pipeline
key mapping that Component 3 consumes to *build* such a table; putting it in
`processed` would erode the distinction that ADR exists to protect.

Files are timestamped `<slug>_<YYYYMMDD>T<HHMMSS>Z.parquet` and never
overwritten, mirroring Component 1. Parquet is gitignored; the manifest sidecar
is committed.

---

## 4. Output: three tables

### 4.1 `establishment_assignments_<stamp>.parquet` — the primary deliverable

One row per inspection. `inspection_id` is unique.

| column | type | null? | meaning |
|---|---|---|---|
| `inspection_id` | Utf8 | never | Raw value, unchanged. Primary key. |
| `establishment_id` | Utf8 | never | `EST-<11 digits>`. See §5. |
| `node_id` | Utf8 | never | `N-<16 hex>`. The distinct identity signature this row carried. |
| `resolution_tier` | Utf8 | never | `singleton` \| `high` \| `medium` \| `reduced`. See §6. |
| `has_ambiguous_link` | Boolean | never | This row's node had at least one candidate pair declined as ambiguous. |
| `license_key` | Utf8 | **yes** | Normalized licence, or null when unusable (§7). |
| `name_key` | Utf8 | yes | Normalized `dba_name`. |
| `addr_key` | Utf8 | yes | `house\|directional street\|zip`. Null when no house number. |
| `unit` | Utf8 | yes | Suite/floor designator, when recorded. |
| `zip_key` | Utf8 | yes | Five-digit zip. |
| `geo_usable` | Boolean | never | Whether the coordinate passed parsing and range checks. |

**This table deliberately carries no dates, no counts and no outcomes.** That
absence is the anti-leakage guarantee: a downstream join against it cannot pull
a full-history aggregate into a training row, because there is none to pull.

### 4.2 `establishments_<stamp>.parquet` — one row per establishment

| column | type | null? | meaning |
|---|---|---|---|
| `establishment_id` | Utf8 | never | Primary key. |
| `anchor_inspection_id` | Utf8 | never | The earliest inspection; the id is derived from it. |
| `cluster_content_sha256` | Utf8 | never | Change detector, **not** an identifier. See §5.3. |
| `resolution_confidence` | Utf8 | never | `high` \| `medium` \| `reduced`. |
| `split_reason` | Utf8 | yes | Non-null when a cluster invariant forced a split. |
| `n_nodes` | Int64 | never | Distinct identity signatures merged. |
| `n_inspections` | Int64 | never | Inspections attributed. **Audit only — not a feature.** |
| `n_licenses` | Int64 | never | Distinct usable licences. **Audit only.** |
| `n_addresses` | Int64 | never | Distinct address keys. **Audit only.** |
| `n_names` | Int64 | never | Distinct normalized names. **Audit only.** |
| `canonical_name` | Utf8 | never | Anchor's raw `dba_name`. **Display only — not a join key.** |
| `canonical_address` | Utf8 | never | Anchor's raw `address`. **Display only.** |
| `canonical_zip` | Utf8 | yes | Anchor's zip. |

The `n_*` columns are whole-history counts. They are legitimate for auditing and
illegitimate as model features, because they summarise information from after
any given inspection. Component 3 must compute its own as-of counts (§8).

### 4.3 `entity_resolution_edges_<stamp>.parquet` — the audit table

One row per candidate pair worth explaining: every merged pair, every ambiguous
pair, and every vetoed pair that shared a licence or an address. Ordinary
no-matches between unrelated neighbours are omitted — there are hundreds of
thousands and they answer no question anyone asks.

Columns: `left_node_id`, `right_node_id`, `left_establishment_id`,
`right_establishment_id`, `tier`, `rule_id`, the eleven signal columns
(`same_license`, `same_addr_key`, `same_coord`, `near_addr`, `same_zip`,
`name_exact`, `name_containment`, `digit_conflict`, `unit_conflict`,
`dir_conflict`, `aka_conflict`, `facility_agree`), and the raw names and
addresses of both sides for readability.

`facility_agree` is nullable and three-valued: true, false, or null meaning
"unknown" because at least one side had a blank facility type.

**How to audit a decision.** "Why was inspection X assigned to establishment Y?"
— look up X's `node_id` in the assignments table, then filter the edges table on
that node id. Each matching row names the rule and shows every signal that fed
it. "Why were these two *not* merged?" is the same query; a `V*` rule id means a
veto fired and a `no_match`/`ambiguous` tier means no rule reached the merge
threshold.

### 4.4 `manifest_establishment_assignments_<stamp>.json`

Provenance and QA in one file: `source_path` and **`source_sha256`** (what pins
reproducibility), `normalization_version`, the full threshold set, node and
establishment counts, edge counts by tier and by rule, split counts by reason,
and every validation check with its result. Per-artifact `sha256`, byte size,
row count and schema for all three tables.

---

## 5. `establishment_id` semantics

### 5.1 What it means

An establishment is a **physical food-service premises**, identified by
location. This is a deliberate definition with consequences:

- Successive businesses at one premises are the **same** establishment with a
  changing name. When 1208 N Wells St houses four bars over fifteen years, that
  is one establishment.
- Concurrent permits at one premises are the **same** establishment with
  multiple licences. A mobile-food commissary holding 47 cart permits is one
  establishment.

The rationale is that Sentinel decides where to send an inspector, and the unit
of inspection is a premises. A new tenant does not make the kitchen, the grease
trap or the walk-in cooler new. Findings §11.1 states the tradeoff in full.

**Consequence for Component 3:** an establishment's history can span a change of
owner, cuisine and name. Do not assume behavioural continuity across such a
transition. `n_names` and `n_licenses` are exposed so you can detect it.

### 5.2 Format and derivation

```
establishment_id = "EST-" + zero-pad(min(inspection_id over the cluster), 11)
```

e.g. `EST-00000067435`. Inspection ids are compared numerically; a non-numeric
id raises rather than silently sorting as a string.

Chosen over a content hash because inspection ids are assigned monotonically, so
appending a later snapshot cannot change which row is a cluster's earliest, and
because the id points at exactly one raw row whose name and address a human can
read. ADR 0007 records the alternatives.

### 5.3 Stability guarantees — read this before joining on it

**Guaranteed:** resolving the same input snapshot with the same code version
always produces identical `establishment_id` values. This is verified directly
on the full 314,245-row snapshot: a seeded random permutation of every input row
produces a byte-identical mapping.

**Not guaranteed:** stability across *different* snapshots. A later snapshot can
supply evidence that merges two clusters (the higher id is retired) or trips an
invariant that splits one (the non-anchor half gets a new id).

> `establishment_id` is a deterministic function of one raw snapshot, identified
> by its sha256. It is **not a durable primary key across snapshots** and must
> not be used as one without a crosswalk.

`cluster_content_sha256` exists to make this tractable rather than merely
disclaimed. Diff two runs on `(establishment_id, cluster_content_sha256)`:

- same id, same hash → unchanged
- same id, different hash → membership moved
- id absent → retired by a merge

A cross-snapshot crosswalk is **future work**, not built.

Changing a normalization rule also changes ids. `normalization_version` in the
manifest exists so a diff between two outputs can be attributed to code rather
than to data.

---

## 6. Tier and confidence vocabulary

There is no numeric confidence score. A float would invite false precision and
could not be explained; categorical tiers backed by named rules can.

| `resolution_tier` (per inspection) | meaning |
|---|---|
| `singleton` | Its establishment holds exactly one node. No merge occurred. |
| `high` | Merged using strong edges only. |
| `medium` | At least one probable edge contributed. |
| `reduced` | The cluster failed an invariant and was rebuilt more conservatively. |

| tier on an edge | meaning |
|---|---|
| `strong` | Two independent identity signals agree. Merges. |
| `probable` | One signal agrees, nothing contradicts. Merges, flagged. |
| `ambiguous` | Real evidence, not enough of it. **Does not merge**, recorded for review. |
| `no_match` | A veto fired, or no rule was satisfied. |

Ambiguity is preserved rather than resolved. A false merge contaminates years of
inspection history for every establishment involved; a false split costs
statistical power. The asymmetry is deliberate.

---

## 7. Null and edge-case behaviour

| situation | behaviour |
|---|---|
| `license_` is null, blank, or `'0'` | `license_key` is null; the row carries no licence evidence and resolves on name and address. 850 rows in the snapshot. |
| `address` has no house number | `addr_key` is null; the row can only match via licence. 14 rows. |
| `latitude`/`longitude` missing, unparseable, or outside Chicago | `geo_usable` is false; matching falls back to the address key. 1,042 rows. |
| `facility_type` is blank | Treated as **unknown**, never as a mismatch. |
| `aka_name` missing | No trade-name evidence; no conflict is inferred. |
| Nothing matches | The node becomes a singleton establishment. Every inspection always receives an id. |
| A block exceeds `max_block_size` | Not pair-expanded. Members stay unmerged and the block is recorded in the manifest, so incomplete coverage is visible rather than silent. |

**Every inspection always receives an `establishment_id`.** There is no null and
no "unresolved" bucket; failure to find evidence produces a singleton.

---

## 8. Temporal semantics and leakage

Three distinctions Components 3–5 depend on.

**1. Retrospective identity reconstruction is legitimate.** Resolving that a
2012 inspection happened at the premises now called *Tavern on Wells* uses the
whole snapshot, including later rows. That is not leakage: identity is a property
of the place, not a prediction about it. The building was the same building in
2012 whether or not we learned it in 2025.

**2. As-of feature availability is a different thing.** Any per-establishment
*quantity* — inspection counts, prior failure rates, days since last inspection
— is only meaningful relative to a reference date. Component 2 computes none,
and the assignments table carries none.

**3. The residual channel is bounded.** Cluster membership does encode
information distributed across the whole history. Two enforced bounds:

- Matching reads identity fields only (§2). `results`, `violations`, `risk` and
  `inspection_date` are never inputs.
- The `n_*` columns on the establishments table are marked audit-only above.

A strict `--as-of DATE` resolution mode is named as future work; it would cost
one resolution run per evaluation fold.

---

## 9. Guarantees a consumer may rely on

Asserted by error-severity validation checks that fail the command:

1. Exactly one assignment row per input row; `inspection_id` unique.
2. Every `establishment_id` matches `^EST-\d{11}$`.
3. Every `establishment_id` is used by exactly one cluster.
4. Every node belongs to exactly one establishment.
5. Every establishment is anchored on its own earliest inspection.
6. No establishment spans more than one zip code.
7. No establishment spans more than 4 address keys.
8. No establishment contains conflicting unit designators.
9. Re-running on the same input reproduces the same ids.

Reported but **not** enforced, because they are legitimate on real data:
cluster-size distribution, address density, singleton rate, ambiguous-pair count,
split count.

---

## 10. Known limitations

- **Same-name outlets at one dense address may merge.** Two outlets of one chain
  at one address with no store number and no distinguishing trade name are
  indistinguishable from the data. Bounded to mega-addresses; `MCDONALD'S` at
  O'Hare is the known example.
- **Stadiums and arenas resolve to one establishment.** Whether an arena is one
  premises or many is definitional, not a defect.
- **747 ambiguous pairs** have never been manually adjudicated. They are the
  intended review queue.
- **Cross-snapshot identity** is not solved (§5.3).
- **Ranged house numbers key on the low endpoint.** `4749-4753` and `4749-51`
  unify correctly, but a record filed only under `4753` would not join them.
