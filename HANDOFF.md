# HANDOFF

For a fresh Claude Code session picking this repository up. Read `MEMORY.md`
first, then `STATUS.md`, then this file.

**Last session completed Component 4 — As-of Feature Engineering.**
**Next task: Component 5 — Temporal Evaluation.**

---

## 0. The one warning that matters most

```
TARGET   = what the inspection found          (Component 3, future information)
FEATURES = what was knowable before it        (Component 4, strictly earlier)
```

Component 4 guarantees that **no feature contains information from on or after
its reference date**. That guarantee is necessary and **not sufficient**.

An evaluation that splits rows **randomly** trains on the future even when every
feature is correct: a 2019 row and a 2024 row land in the same fold, and the
model is scored on establishments it has already learned from. **Component 5 must
split chronologically.** See §9.

---

## 1. What was completed

Component 4 builds 26 historical features for each of Component 3's 57,727
prediction opportunities, following the same discipline as Components 1–3:
investigate → document → design → implement → test.

1. **`scripts/profile_features.py`** — 21 read-only DuckDB profiles joining all
   three upstream artifacts, run in 3 s before any feature code.
2. **`docs/analysis/as_of_feature_engineering_findings.md`** — the measurements
   and the decision each one forced, written *before* the implementation.
3. **`src/sentinel/features/`** — five modules: declared feature specifications,
   the range join carrying the temporal boundary, validation, writer,
   orchestration.
4. **`sentinel build-features`** CLI with `--dry-run` and `--report`.
5. **202 new tests** (543 → 745), including a 12-test leakage suite in its own
   file.
6. **Contract and ADRs** — `docs/data_contracts/as_of_features.md`, ADR 0010 (the
   boundary and construction), ADR 0011 (the processed layer).
7. **First use of `data/processed/`**, with a written criterion for what belongs
   there.

**No new dependencies.** DuckDB for the range join, Polars for the frame, both
already present.

---

## 2. Current repository state

```text
src/sentinel/
  cli.py                 ingest | query | resolve | build-target | build-features
  config.py              + features_processed_dir
  manifest.py            generic sha256 / read / write helpers
  ingest/ query/ entity/ target/          Components 1-3
  features/              NEW: Component 4
    definitions.py       FEATURE_SPECS, WINDOW_DAYS, NullRule, Family
    historical.py        priority_flags + the range join (THE boundary)
    validate.py          15 error checks incl. temporal_boundary_holds
    writer.py            output_schema derived from the specs
    build.py             build_features; the only module doing I/O
    models.py            FeatureManifest, ValidationCheck
scripts/profile_features.py   NEW: 21 read-only profiles
tests/                        745 passing, 3 live deselected
  test_features_leakage.py    NEW: the safety wall
docs/analysis/                + as_of_feature_engineering_findings.md
docs/data_contracts/          + as_of_features.md
docs/decisions/               11 ADRs (0010, 0011 new)
data/processed/features/      as_of_features_*.parquet + manifest (committed)
```

Branch `main`, working tree clean.

---

## 3. The feature table

| | |
|---|---|
| path | `data/processed/features/as_of_features_<stamp>.parquet` |
| grain | one row per Component 3 **eligible** target row |
| primary key | `target_inspection_id` |
| natural key | `(establishment_id, inspection_date)` |
| rows | **57,727** |
| columns | **33** = 3 keys + 26 features + 2 labels + 2 provenance |

```text
keys        establishment_id · inspection_date · target_inspection_id
features    the 26 below (FEATURE_COLUMNS)
labels      target · target_status
provenance  code_era_phase · feature_definition_version
```

Component 3's outcome columns (`results`, `evidence`, `has_priority`,
`n_*_entries`, `inspection_type`) are **not present at all** — a check fails the
build if any appears.

---

## 4. Temporal semantics

> **A feature for the row at `inspection_date = d` may use only records dated
> STRICTLY BEFORE `d`.**

**Why `<` and not `<=`, settled by measurement:** `inspection_date` has exactly
**one** distinct time component across all 314,245 rows (`T00:00:00.000`). It is
a date, not a timestamp, so same-day records **cannot be ordered**. At reference
dates there are 1,075 same-day `License`, **43 same-day `Canvass
Re-Inspection`** and 42 same-day `Complaint` records — and a re-inspection exists
*because* an inspection failed, so it provably follows the canvass being
predicted.

An inspection dated on the reference date is therefore **never** history,
including the target's own contributing inspections.

**Cost, accepted:** up to 1,075 genuinely-prior licence inspections are
discarded. Component 4 exists to prevent leakage, so it fails toward exclusion.

**Observable proof:** `days_since_last_canvass` has a minimum of **1** and
contains **no zeros**. A zero-day recency is unconstructable.

**Implementation:** one range join, one condition, one place.

```sql
LEFT JOIN history h
  ON  h.establishment_id = t.establishment_id
  AND h.inspection_date  <  t.inspection_date
```

`LEFT JOIN` so the 401 history-less rows survive rather than vanish.

---

## 5. The 26 features

Full definitions in `docs/data_contracts/as_of_features.md` §5. Summary:

**Canvass history (8)** — the routine series, comparable to the target:
`prior_canvass_count`, `prior_canvass_count_code_era`,
`prior_canvass_inspected_count`, `days_since_last_canvass`,
`prior_canvass_fail_count`, `prior_canvass_pass_w_conditions_count`,
`prior_canvass_fail_rate`, `fail_at_last_canvass`.

**Priority history (4)** — code-era canvasses only, because Priority did not
exist before 2018-07-01: `prior_canvass_priority_count`,
`prior_canvass_priority_foundation_count`, `prior_canvass_priority_rate`,
`priority_at_last_canvass`.

**Windows (6)** — half-open `[d − N, d)` for N ∈ {365, 730, 1095}:
`canvasses_last_{N}d`, `canvass_priority_events_last_{N}d`.

**All-type context (5)**: `prior_inspection_count_any_type`,
`days_since_any_inspection`, `prior_complaint_count`, `prior_reinspection_count`,
`prior_license_inspection_count`.

**Tenant change (2)**: `name_changed_since_last_canvass`,
`prior_canvass_count_current_name`.

**Observation (1)**: `days_since_first_inspection`.

### Missing values — four rules

| kind | rule |
|---|---|
| counts | never NULL; `0` = "none observed" |
| recency (`days_since_*`) | **NULL** when the event never happened |
| rates | **NULL** when the denominator is 0 |
| at-last flags | **NULL** when there is no prior event |

Every event count is emitted **beside its inspection count**, so `0` is legible:
`canvass_priority_events_last_365d = 0` next to `canvasses_last_365d = 0` means
"not inspected", next to `= 2` means "inspected twice and clean".

---

## 6. Leakage protections

1. **One boundary, one place** — every feature is an aggregate over the same
   restricted row set.
2. **Independently re-derived** — `temporal_boundary_holds` recomputes the latest
   contributing date per row from a *separate* query. A check that reuses the
   aggregation only proves it agrees with itself.
3. **Whole table, not a sample** — that check runs on all 57,727 rows. The
   project spec suggests 500 random rows in CI; this dominates it at no cost.
4. **No all-history aggregate exists** — the pipeline has no establishment-level
   intermediate that could be merged onto every row.
5. **Four dedicated regression tests** in `tests/test_features_leakage.py`:
   future insertion, future mutation, target self-exclusion, same-day exclusion —
   plus a paired test proving a record one day earlier *is* counted, so the
   boundary is exclusive rather than absent.
6. **`null_rules_hold_exactly`** — asserts each feature is NULL *exactly* when
   its declared rule says so, not merely that nulls exist.
7. **No feature selected by accuracy** — there is no model; features are
   justified by domain reasoning only.

---

## 7. Verified full-data results

| measurement | value |
|---|---|
| eligible target rows in → feature rows out | 57,727 → **57,727**, 0 unmatched |
| features / columns | 26 / 33 |
| runtime | **15.6 s** |
| no history at all | 401 (0.69%) |
| no prior canvass | 5,615 (9.73%) |
| no prior code-era canvass | 14,162 (24.53%) |
| after a business-name change | 1,962 (3.40%) |
| `prior_canvass_count` | mean 7.86, p50 7, max 247 |
| `days_since_last_canvass` | **min 1**, mean 485, p50 386, max 5,612 |
| `prior_canvass_fail_rate` | mean 0.218, p50 0.167 |
| `prior_canvass_priority_rate` | mean 0.575, p50 0.500 |
| `canvasses_last_365d` = 0 | 35,781 (62.0%) |
| target balance | 27,411 / 30,316 — **identical to Component 3** |

All 15 error checks pass. Rebuilding and a seeded row shuffle both reproduce
identical values.

---

## 8. Tests

```bash
uv run pytest                       # 745 passed, 3 deselected
uv run pytest -m live               # 3 live tests, hits the real API
uv run ruff check .                 # All checks passed
uv run ruff format --check .        # 91 files already formatted
uv run mypy src/sentinel scripts    # no issues in 39 source files
uv run sentinel build-features --dry-run --report
```

---

## 9. Next task: Component 5 — Temporal Evaluation

Build an honest way to measure whether a ranking is good, **before** any model
exists. Scope is the evaluation harness only — no models, no calibration, no
SHAP, no scheduling.

The project spec is emphatic that this comes before modelling and must never be
cut: "the evaluation harness, the inspector decomposition, calibration, and the
fairness audit — those four are the project."

### Split chronologically, never randomly

Rolling-origin backtest: train on a period, calibrate on the next, test on the
one after, roll forward. Report mean ± SD across folds, not one number from one
split. A single train/test split invites "how do you know that isn't luck?"

### What the data already tells you

* **The base rate drifts hard**: 87.6% positive in 2018 H2 → 39.1% in 2026. Any
  evaluation pooling across time measures the drift rather than the model.
  `code_era_phase` marks the 2,829 adoption-period rows for optional holdout.
* **The canvass cycle is a 358-day median**, so a test window shorter than a year
  contains mostly establishments that will not reappear.
* **57,727 labelled rows** spanning 2018-07 to 2026-08. Folds are not free.
* **`days_since_any_inspection` partly encodes scheduling policy** (p25 of 9 days
  is the re-inspection pattern), so build the with/without ablation into the
  harness rather than bolting it on later.

### Investigate before coding

* **State the estimand first.** Only establishments *actually inspected* in a
  window can be re-ordered, so this measures re-ordering, not counterfactual
  coverage. Say it before building, not after being asked.
* What is the real inspection capacity in the data? `precision@k` needs a
  defensible `k`.
* How are the five reference schedules built from this table — optimal, model,
  business-as-usual, random, worst?
* Does time-invariance hold? The spec's Finding 2 says it does not for
  temperature-related violations. That is a measurement, not an assumption.

### What must NOT be changed

**Components 1–3 invariants** (all still binding): `$order=inspection_id` on
every paged request; raw is all-`Utf8` and append-only; only identity columns
reach the entity matcher; licence inequality is never evidence against a match;
the target label is read from violation text, never from `results`; eligibility
starts 2018-07-01; `Out of Business` is ineligible, not negative.

**Component 4 invariants:**

1. **A feature may use only records dated strictly before its reference date.**
2. **The boundary is `<`, not `<=`** — dates carry no time component.
3. **One range join carries the condition**, and validation re-derives it
   independently. Never `groupby(establishment_id)` then merge.
4. **The four missing-value rules**, and the pairing convention that makes a `0`
   legible.
5. **Priority features use code-era canvasses only** and are NULL for the 24.5%
   without one.
6. **`FEATURE_COLUMNS` is the complete set of model inputs.** `target`,
   `target_status`, `inspection_date` and `code_era_phase` are not features —
   `inspection_date` is a legitimate *split* key and `code_era_phase` a
   legitimate *stratification* variable, but neither is a predictor.
7. Do not change a feature without re-reading the findings document and bumping
   `feature_definition_version`.

**Do not add features in Component 5.** If one is missing it belongs in
Component 4, behind a bumped definition version. And do not select features by
test-set performance — that is leakage by another route.

### How to join

```python
import duckdb

duckdb.sql("""
    SELECT * FROM read_parquet('data/processed/features/as_of_features_*.parquet')
""").show()
```

Everything needed to train and evaluate is on that one table. Identity is
Component 2's, labels are Component 3's, features are Component 4's — join, do
not recompute.

### Reminder

**One component at a time.** Component 5 is the evaluation harness only. No
XGBoost, no calibration, no SHAP, no OR-Tools, no LangGraph, no frontend.
