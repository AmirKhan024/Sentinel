# HANDOFF

For a fresh Claude Code session picking this repository up. Read `MEMORY.md`
first, then `STATUS.md`, then this file.

**Last session completed Component 3 — Target Construction.**
**Next task: Component 4 — As-of Feature Engineering.**

---

## 1. What was completed

Component 3 defines what Sentinel predicts. It turns inspection history into a
precise, leakage-safe, auditable label, following the same discipline as
Components 1 and 2: investigate → document → design → implement → test.

1. **`scripts/profile_target.py`** — 31 read-only DuckDB profiles joined to
   Component 2's assignments, run in 10.5 s before any target code was written.
2. **`docs/analysis/target_construction_findings.md`** — the measurements and the
   design decision each one forced, written *before* the implementation.
3. **`src/sentinel/target/`** — six modules: violation parsing and severity
   classification, eligibility gates and labelling, validation, output writing,
   models, orchestration.
4. **`sentinel build-target`** CLI with `--dry-run` and `--report`.
5. **201 new tests** (342 → 543), including 12 regression cases copied verbatim
   from real inspections.
6. **Contract and ADRs** — `docs/data_contracts/inspection_targets.md`,
   ADR 0008 (the target definition), ADR 0009 (the code-era boundary).
7. **Corrected `docs/data_contracts/food_inspections_raw.md`**, which documented
   four `results` values where the data has seven.

**No new dependencies.** Parsing is regular expressions over stdlib; no LLM, no
embeddings, no fuzzy matching.

---

## 2. Current repository state

```text
src/sentinel/
  cli.py                     ingest | query | resolve | build-target
  config.py                  + target_interim_dir
  manifest.py                generic sha256 / read / write helpers
  ingest/                    Component 1
  query/                     DuckDB over raw Parquet
  entity/                    Component 2
  target/                    NEW: Component 3
    models.py                Severity, TargetStatus, CodeEraPhase,
                             CODE_ERA_START, TARGET_DEFINITION_VERSION
    violations.py            split / parse / classify + narrative exclusions
    construct.py             eligibility gates, labelling, same-day collapse
    validate.py              12 error checks + 6 distributional notes
    writer.py                TARGETS_SCHEMA, TARGET_EVENT_COLUMNS
    build.py                 orchestration; the only module doing I/O
scripts/profile_target.py    NEW: 31 read-only profiles
tests/                       543 passing, 3 live deselected
  fixtures/target_cases.py   NEW: 12 real regression cases as literal Python
docs/analysis/               + target_construction_findings.md
docs/data_contracts/         + inspection_targets.md (raw contract corrected)
docs/decisions/              9 ADRs (0008, 0009 new)
data/interim/target/         inspection_targets_*.parquet + manifest (committed)
```

Branch `main`, working tree clean.

---

## 3. The target definition

**Plain English.** For each establishment-date on which a routine canvass
happened, the target is 1 if that day's canvassing found at least one **Priority
or Priority Foundation** violation, and 0 if it found none.

**Logical form.**

```
target(e, d) = 1  ⟺  ∃ i ∈ Canvasses(e, d) : ∃ v ∈ Violations(i) :
                          severity(v) ∈ {PRIORITY, PRIORITY_FOUNDATION}
```

**Example.** Establishment `EST-00000068356` was canvassed on 2019-03-08. The
inspector recorded `3. MANAGEMENT … - Comments: NO EMPLOYEE HEALTH POLICY
ON-PREMISES. PRIORITY FOUNDATION 7-38-010.` → `target = 1`, with `evidence` set
to the matched span. The `results` column said `Pass w/ Conditions`.

### Prediction unit

**One target row = one (establishment, date) on which at least one eligible
canvass occurred.** Not one row per inspection — 530 establishment-dates carry
more than one eligible canvass and 160 of the underlying days disagreed, so they
collapse with OR. "Inspect E on date D" is a single scheduling decision.

### Reference event and target event

They are the **same inspection**. The canvass *is* the prediction event, and
`inspection_date` is the as-of boundary — the instant a scheduler would decide
whether to send an inspector.

A "predict the next canvass" formulation was rejected: it discards every
establishment's most recent canvass (15,148 rows) and attaches labels to events a
median of 377 days later (IQR 306–511). See ADR 0008.

### Positive / negative / excluded

| class | definition | rows |
|---|---|---|
| positive (`1`) | ≥1 Priority or Priority Foundation violation found | 30,316 |
| negative (`0`) | eligible, text interpretable, no priority violation | 27,411 |
| excluded (`NULL`) | an eligibility gate failed | 255,818 |
| unknown (`NULL`) | result and text contradict each other | 79 |

Negatives are not empty rows: 22,437 recorded Core-only violations, 4,974 were
clean passes with no text.

### Eligibility, in evaluation order

| gate | rule | excluded |
|---|---|---|
| era | `inspection_date >= 2018-07-01` | 172,879 |
| type | `upper(trim(inspection_type)) == 'CANVASS'` | 70,848 |
| result | `results ∈ {Pass, Pass w/ Conditions, Fail}` | 12,091 |
| text | violation text interpretable | 79 |

### Temporal semantics

There is no lookahead. The target event is the canvass on `inspection_date`, and
the label is derived from that inspection's own violation text. Chronology is
used only to group same-day canvasses — never to find a later event, and never
to compute a historical quantity.

---

## 4. Verified data findings

Raw sha256 `7d3c4069340a68d197204c6cca9fca6399c6565bc3668760f145f43cd377ad38`.

| measurement | value |
|---|---|
| inspections → target rows → eligible | 314,245 → 313,624 → **57,727** |
| positive / negative | 30,316 / 27,411 (**52.52%**) |
| establishments with ≥1 eligible row | 15,144 |
| runtime | 25 s |
| `results` distinct values | **7**, not the 4 documented |
| code-era cutover | **2018-07-01**, clean (June: 0 new / 415 old; July: 761 / 0) |
| priority presence: Fail / PwC / Pass | 99.4% / **97.9%** / 0.45% |
| violation number → severity | **not predictive** (item 10: 42/11/47) |
| priority markers with no citation code | 21,281 (all genuine) |
| entries with no severity label | 72% |
| narrative exclusion effect | 74 entries, **10 inspection labels** |
| OOB followed by a later inspection | 24.9%, median 273 days |
| positive rate drift | 87.6% (2018 H2) → 39.1% (2026) |

---

## 5. Tests

```bash
uv run pytest                       # 543 passed, 3 deselected
uv run pytest -m live               # 3 live tests, hits the real API
uv run ruff check .                 # All checks passed
uv run ruff format --check .        # 74 files already formatted
uv run mypy src/sentinel scripts    # no issues in 31 source files
```

Determinism is asserted in unit tests and verified separately on the full
snapshot: rebuilding reproduces the committed table exactly, and building from a
seeded random permutation of all 314,245 input rows produces identical labels.

---

## 6. Known limitations — be honest about these

1. **52% of the dataset cannot be labelled** (172,879 rows predate 2018-07-01).
   They remain usable as *features*.
2. **The base rate drifts from 87.6% to 39.1%.** Flagged via `code_era_phase`,
   not corrected.
3. **The narrative-exclusion list is judgement** — four patterns, 74 entries, 10
   labels. Enumerated in the contract so it can be argued with.
4. **8 `Pass w/ Conditions` rows are negative** where the result implies
   otherwise, because the parser stays independent of `results`.
5. **Inspector write-up variation is unmeasurable.** A priority violation found
   but not labelled is a false negative and the open data has no ground truth.
   **NOT VERIFIED.**
6. **Severity within positive is not represented.** One priority violation and
   twelve give the same label.
7. **`Canvass Re-Inspection` (16,998 code-era rows) and complaint inspections are
   excluded** — a deliberate scope restriction.
8. **CI has still never run. NOT VERIFIED.**

---

## 7. Important decisions

* **ADR 0008** — the target definition. Why not `results == 'Fail'` (would
  mislabel 16,261), why not any-violation, why not a count, why not the
  next-canvass formulation, why re-inspections and complaints are excluded.
* **ADR 0009** — the 2018-07-01 boundary, why the adoption period is flagged
  rather than dropped, and that pre-2018 rows remain usable as features.
* Output goes to `data/interim/target/`, not `processed/`: ADR 0005 reserves
  processed for model-ready tables and this is labels only.
* `target` is a nullable `Int8`, not a `Boolean`, because null is a meaningful
  third state.

---

## 8. What must NOT be changed

**Component 1** (still binding): `$order=inspection_id` on every paged request;
raw Parquet is all `Utf8`; `data/raw/` is append-only; non-retryable 4xx raise;
live tests deselected by default; raw data never committed.

**Component 2** (still binding): only identity columns reach the matcher; the
assignments table carries no dates, counts or outcomes; every non-licence merge
requires address equivalence; licence inequality is never evidence against;
resolution stays deterministic; `establishment_id` is snapshot-scoped.

**Component 3:**

1. **The label is read from the violation text, never from `results`.** Using the
   result would make the target circular.
2. **`Out of Business`, `No Entry`, `Not Ready`, `Business Not Located` are
   ineligible, not negative.** No inspection happened.
3. **Eligibility starts 2018-07-01.** Before it the target is undefined.
4. **One row per (establishment, date), target = OR.**
5. **Every positive carries an `evidence` span** — enforced by a check that fails
   the build.
6. **The violation number is never used to classify severity.**
7. **`UNCLASSIFIED` does not mean "Core".** 72% of entries carry no label.
8. Do not change a rule without re-reading
   `docs/analysis/target_construction_findings.md` and bumping
   `TARGET_DEFINITION_VERSION`.

---

## 9. Next task: Component 4 — As-of Feature Engineering

Build the information a scheduler actually had **before** each inspection.
Scope is features only — no models, no calibration, no evaluation framework, no
scheduling.

### ⚠ The leakage rule — read this twice

Component 3 emits one row per (establishment, date). **`inspection_date` is the
as-of boundary. A feature for that row may use only information dated strictly
before that date.**

**Never use as features** — these describe the outcome:

```
target                          results
has_priority                    evidence
has_priority_foundation         n_priority_entries
n_violation_entries             n_priority_foundation_entries
```

The set is enumerated in `sentinel.target.writer.TARGET_EVENT_COLUMNS` and
asserted by a test, so you can check it programmatically.

**Also never use:** any inspection dated on or after `inspection_date` for that
row, any aggregate computed over an establishment's whole history, and
`code_era_phase` as a risk predictor (it is a stratification variable describing
the regulatory regime, not a property of the establishment).

**Why Component 3 was allowed to look at the target event and Component 4 is
not:** Component 3 was *constructing* the label, so the outcome is what it is
defined from. Component 4 constructs what was knowable *before* the outcome
existed. Same row, two different time positions.

**A useful non-obvious permission:** pre-2018 inspections are perfectly usable as
features. The era boundary constrains what can be *labelled*, not what can be
*known*. A 2014 inspection is legitimate history for a 2022 target row.

### Investigate before coding — the pattern has worked twice

Components 2 and 3 each had multiple planned decisions reversed by measurement.
Expect the same. Add profiles under `scripts/` and write a findings document
before designing the feature set.

Questions the data should answer:

1. **How much history does a target row actually have?** Distribution of prior
   inspections at each as-of date; how many rows have none at all. This bounds
   what any history feature can do.
2. **How should the pre-2018 era contribute?** Its violation vocabulary differs,
   so "prior priority violations" is undefined there while "prior failures" is
   not. Decide explicitly rather than letting nulls decide.
3. **Does days-since-last-inspection encode risk or scheduling policy?**
   Chicago already inspects riskier establishments more often (median canvass gap
   377 days, IQR 306–511). A recency feature may be learning the city's existing
   policy rather than the establishment's state — worth measuring and documenting
   before including it.
4. **Do features span a tenant change?** Component 2 defines an establishment as
   a physical premises, so history can cross a change of owner and cuisine.
   `n_names` and `n_licenses` on the establishments table let you detect it.
5. **How stable are establishment attributes over time** (`facility_type`,
   `risk`, zip, geography)? Component 2 measured `facility_type` at 99.5% stable
   *per licence*, which is not the same as per establishment.
6. **Is `risk` usable?** It is a city-assigned risk category and may itself be
   derived from inspection history, which would make it a partial leak of the
   very thing being predicted. Profile it before trusting it.

### How to join

```python
import duckdb

duckdb.sql("""
    SELECT t.establishment_id, t.inspection_date, t.target
    FROM read_parquet('data/interim/target/inspection_targets_*.parquet') t
    WHERE t.target_status = 'eligible'
""").show()
```

Then, for each row, gather prior inspections from the raw snapshot joined to
Component 2's assignments, filtered to `inspection_date < <the row's date>`.

**Do not re-derive identity or labels.** Component 2 owns `establishment_id`;
Component 3 owns `target`.

### Reminder

**One component at a time.** Component 4 is as-of feature engineering only. No
models, no XGBoost, no calibration, no temporal evaluation framework, no
OR-Tools, no LangGraph, no frontend.
