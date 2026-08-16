# Data contract: as-of features (Component 4 output)

**Produced by:** `sentinel build-features` (`src/sentinel/features/`)
**Layer:** `data/processed/features/`
**Consumed by:** Component 5 (temporal evaluation) and Component 6+ (models)
**Design rationale:** `docs/analysis/as_of_feature_engineering_findings.md`, ADR 0010, ADR 0011

---

## 1. Grain and primary key

> **One row = one Component 3 eligible target row = one (establishment, date) on
> which at least one routine canvass occurred.**

| | |
|---|---|
| grain | prediction opportunity |
| primary key | `target_inspection_id` |
| natural key | `(establishment_id, inspection_date)` |
| rows | **57,727** |
| columns | 33 (3 keys + 26 features + 2 labels + 2 provenance) |

**Not** one row per establishment. **Not** one row per inspection. The grain is
inherited from Component 3 and asserted: a check fails the build if the row count
does not equal Component 3's eligible count.

Ineligible target rows are **not** featurised. An ineligible row has no
prediction to make, so it has no features. This is not a silent drop — the count
reconciles exactly against Component 3, and the exclusions remain fully
enumerated in that table.

---

## 2. The reference event and the cutoff

The reference event **is** the canvass at `inspection_date`. Component 3
constructs the label from what that inspection found; Component 4 constructs what
was knowable before it happened.

```
        PAST                    reference date d              FUTURE
  ────────────────────────────────────┼────────────────────────────────►
   every record dated < d             │   every record dated >= d
   is a legitimate FEATURE            │   is forbidden
                                      │
                              the canvass on d
                              is the TARGET
```

### The boundary is strictly `<`

**Settled by measurement, not preference.** `inspection_date` has exactly **one**
distinct time component across all 314,245 rows (`T00:00:00.000`). It is a date,
not a timestamp, so two inspections on the same establishment-date **cannot be
ordered** by any means this dataset provides.

At reference dates specifically, the other same-day records are 1,075 `License`
inspections, **43 `Canvass Re-Inspection`** and 42 `Complaint` records. A
re-inspection exists *because* an inspection failed, so a same-day re-inspection
almost certainly happened *after* the canvass being predicted. Admitting it would
feed the outcome back in as an input, and no amount of care could unwind it.

> **An inspection dated on the reference date is never history.** That includes
> the target's own contributing inspections and the same-day licence inspections
> that may genuinely have preceded the canvass.

The cost is real and accepted: up to 1,075 genuinely-prior records are discarded.
Component 4 exists to prevent leakage, so the trade runs that way.

**Observable consequence:** `days_since_last_canvass` has a minimum of **1** and
contains **no zeros** in the full build. A zero-day recency is unconstructable.

---

## 3. Historical eligibility

Every inspection in the snapshot is a candidate for history — all types, all
eras — subject to the boundary above and joined to Component 2's identities.

Two restrictions apply to particular feature families:

| restriction | applies to | why |
|---|---|---|
| result must be `Pass`, `Pass w/ Conditions` or `Fail` | outcome counts and `prior_canvass_fail_rate` | 16,517 prior canvasses are `Out of Business` and 13,077 are `No Entry`; a locked door is not a clean result |
| date must be on or after **2018-07-01** | all four priority features and the windowed priority events | Priority and Priority Foundation did not exist before the food-code change (ADR 0009) |

**Pre-2018 history is still used** for every count, outcome and recency feature.
The era boundary constrains what can be *classified*, not what can be *counted*.
80% of target rows have pre-2018 history available.

---

## 4. Missing-value semantics — four rules

These are the part most easily got wrong, and the failure is silent: a table
where a NULL was rendered as `0` looks perfectly healthy and teaches a model that
*no evidence* means *no problem*.

| kind | rule | rationale |
|---|---|---|
| **counts** | never NULL; `0` = "none observed" | 0 is a true observation, and every subtype count is emitted beside its total so 0 stays interpretable |
| **recency** (`days_since_*`) | **NULL** when the event never happened | `0` would mean "it happened today", the opposite of "it never happened" |
| **rates** | **NULL** when the denominator is 0 | `0/0` is not `0` |
| **at-last flags** | **NULL** when there is no prior event | the absence of a last inspection is not a clean last inspection |

Each rule is declared per feature in `FeatureSpec.null_rule` and enforced by the
`null_rules_hold_exactly` check, which asserts a feature is NULL **exactly** when
its rule says so — not merely that nulls exist.

### The pairing convention

Every event count is emitted beside its inspection count for the same scope:

* `canvass_priority_events_last_365d` beside `canvasses_last_365d`
* `prior_canvass_priority_count` beside `prior_canvass_count_code_era`
* `prior_canvass_fail_count` beside `prior_canvass_inspected_count`

This is what makes a `0` legible. `canvass_priority_events_last_365d = 0` alone is
ambiguous; read next to `canvasses_last_365d = 0` it plainly means "not inspected
in that window", and next to `canvasses_last_365d = 2` it means "inspected twice
and clean".

---

## 5. The features

26 features in six families. Every one is declared in
`sentinel.features.definitions.FEATURE_SPECS` with a description, sources and a
null rule; a validation check fails on any column without a spec.

Window notation: `[d − N, d)` is half-open — a record exactly `N` days before the
reference date is **inside** the window; a record on the reference date is not.

### Canvass history (8) — the routine series, comparable to the target

| feature | type | NULL when | definition |
|---|---|---|---|
| `prior_canvass_count` | int32 | never | canvasses with `date < d` |
| `prior_canvass_count_code_era` | int32 | never | of those, with `date >= 2018-07-01` |
| `prior_canvass_inspected_count` | int32 | never | of those, with an inspected result |
| `days_since_last_canvass` | int32 | no prior canvass | `d − max(canvass date)` |
| `prior_canvass_fail_count` | int32 | never | inspected prior canvasses with result `Fail` |
| `prior_canvass_pass_w_conditions_count` | int32 | never | ditto with `Pass w/ Conditions` |
| `prior_canvass_fail_rate` | float64 | no inspected prior canvass | `fail_count / inspected_count` |
| `fail_at_last_canvass` | bool | no prior canvass | result at `max(canvass date)` is `Fail` |

### Priority history (4) — code-era canvasses only

| feature | type | NULL when | definition |
|---|---|---|---|
| `prior_canvass_priority_count` | int32 | no prior code-era canvass | code-era prior canvasses with ≥1 Priority or Priority Foundation violation |
| `prior_canvass_priority_foundation_count` | int32 | same | ditto, Priority Foundation specifically |
| `prior_canvass_priority_rate` | float64 | same | `priority_count / count_code_era` |
| `priority_at_last_canvass` | bool | same | priority found at the most recent code-era canvass |

Priority is classified by **Component 3's own parser**
(`sentinel.target.violations`), not a substring match. A SQL approximation would
drift from the target's definition — which excludes narrative spans such as the
grace-period boilerplate — so "priority" would mean one thing in the label and
another in the feature.

### Windows (6) — `[d − N, d)` for N ∈ {365, 730, 1095}

| feature | type | NULL when | definition |
|---|---|---|---|
| `canvasses_last_{N}d` | int32 | never | canvasses in the window |
| `canvass_priority_events_last_{N}d` | int32 | never | code-era canvasses in the window with a priority violation |

Sizes match the project spec's 1y/2y/3y and are justified by the measured 358-day
median canvass cycle. **A 365-day window is empty for 62% of rows** — a
consequence of that cycle, not a defect. A non-zero value means the establishment
was canvassed *off-cycle*, which is itself signal.

### All-type context (5) — a different series, labelled as such

| feature | type | NULL when | definition |
|---|---|---|---|
| `prior_inspection_count_any_type` | int32 | never | every prior inspection |
| `days_since_any_inspection` | int32 | no prior inspection | `d − max(any date)` |
| `prior_complaint_count` | int32 | never | prior complaint-driven inspections, all variants |
| `prior_reinspection_count` | int32 | never | prior re-inspections, all variants |
| `prior_license_inspection_count` | int32 | never | prior licence-related inspections |

⚠ **`days_since_any_inspection` partly encodes scheduling policy, not risk.** The
any-type interval has a p25 of **9 days** — the re-inspection pattern. It reads
"9 days" precisely when the previous inspection was a failure. This is not
leakage (the re-inspection genuinely preceded the reference date), but a model
may learn Chicago's existing scheduling behaviour rather than the
establishment's state. It is emitted beside the canvass-only recency so
Component 5 can ablate it deliberately.

### Tenant change (2)

| feature | type | NULL when | definition |
|---|---|---|---|
| `name_changed_since_last_canvass` | bool | no prior canvass | reference `dba_name` differs from the name at the most recent prior canvass |
| `prior_canvass_count_current_name` | int32 | never | prior canvasses carrying the reference name |

Component 2 tracks *physical premises*, so history can span a change of tenant
(15.9% of target rows sit in a premises that changed name at some point; 1,962
rows immediately follow a change). The project spec §3.3 would treat an ownership
change as a new establishment; this contract deliberately does not — see §11.

### Observation (1)

| feature | type | NULL when | definition |
|---|---|---|---|
| `days_since_first_inspection` | int32 | no prior inspection | `d − min(any date)` |

Named for what it is: time since the first **record**, not the age of the
business, which this dataset cannot observe. Provides the exposure denominator
for reading raw counts.

---

## 6. Output schema

Column order is part of the contract: keys, features by family, labels,
provenance. A reader scanning the table sees the join keys first and the answer
last.

| group | columns | type |
|---|---|---|
| keys | `establishment_id`, `inspection_date`, `target_inspection_id` | Utf8 |
| features | the 26 above | int32 / float64 / bool |
| labels | `target` | Int8, nullable |
| labels | `target_status` | Utf8 (always `eligible` here) |
| provenance | `code_era_phase` | Utf8 |
| provenance | `feature_definition_version` | Utf8 (`v1`) |

---

## 7. ⚠ What Component 5 and Component 6 may not use

**`FEATURE_COLUMNS` is the complete set of model inputs.** Nothing else in this
table is a feature.

| column | why not |
|---|---|
| `target` | the label |
| `target_status` | derived from the label's eligibility |
| `inspection_date` | the as-of boundary; a legitimate *split* key, not a predictor |
| `code_era_phase` | describes the regulatory regime, not the establishment; a legitimate *stratification* variable, not a predictor |
| `establishment_id`, `target_inspection_id` | identifiers |

Component 3's outcome columns (`results`, `evidence`, `has_priority`,
`n_*_entries`, `inspection_type`) are **not present at all** — a check fails the
build if any appears.

---

## 8. Leakage protections

| protection | mechanism |
|---|---|
| one boundary, one place | every feature comes from a single range join carrying `h.inspection_date < t.inspection_date` |
| independent re-derivation | `temporal_boundary_holds` recomputes the latest contributing date per row from a *separate* query and asserts it is earlier; a bug in the aggregation is caught, not shipped |
| whole table, not a sample | that check runs on all 57,727 rows. The project spec suggests 500 random rows in CI; this dominates it at no cost |
| no all-history aggregate | the aggregation is per target row over a range join; there is no establishment-level intermediate that could be merged onto every row |
| four dedicated regression tests | future insertion, future mutation, target self-exclusion, same-day exclusion — in their own file, `tests/test_features_leakage.py` |
| no feature selection by accuracy | there is no model yet; features are justified by domain reasoning and availability, never by downstream performance |

---

## 9. Determinism

**Guaranteed:** the same three inputs and the same `feature_definition_version`
always produce identical feature values.

Three properties, each asserted by tests:

1. The priority classifier is a pure function of the violation text.
2. The aggregation is a `GROUP BY` over a range join, which is order-independent.
3. Output is sorted by `(establishment_id, inspection_date, target_inspection_id)`.

Verified by rebuilding and by a seeded row shuffle.

`feature_definition_version` is stamped on every row. Bump it whenever a
formula, the boundary, a window size or a null rule changes.

The manifest pins **three** input checksums — raw, assignments, targets — because
the features are a function of all three together. Note that `establishment_id`
is itself snapshot-scoped (ADR 0007), so re-resolving identities can change which
history attaches to which row.

---

## 10. Guarantees a consumer may rely on

Asserted by error-severity checks that fail the command:

1. Exactly one row per Component 3 eligible target row.
2. `target_inspection_id` unique, and every one exists in the target table.
3. Every `establishment_id` exists in Component 2's assignments.
4. **Every row's features used only inspections dated strictly before its
   reference date** — checked on every row, independently re-derived.
5. No same-day record entered any row's history.
6. No negative count or recency; every rate in [0, 1].
7. Every feature is NULL exactly when its declared rule says so.
8. Windows are nested and bounded by their unbounded counterparts.
9. No Component 3 outcome column is present.
10. Features and labels are disjoint, and every column has a `FeatureSpec`.

Reported but not enforced: per-feature null rates, family summaries, cold-start
counts, tenant-change counts.

---

## 11. Known limitations

1. **The boundary discards some genuinely prior information** — up to 1,075
   same-day licence inspections. Deliberate (§2).
2. **Priority features are NULL for 24.5% of rows** (14,162). Correct, but a
   quarter of the table carries no priority history.
3. **A 365-day window is empty for 62% of rows.**
4. **`days_since_any_inspection` partly encodes scheduling policy** (§5).
5. **History can span a tenant change** for 15.9% of rows. Exposed via two
   features rather than resolved. The claim "a rename means a new business" is
   **NOT VERIFIED** in this data, which is why history is not reset; a
   reset-history variant remains available as an ablation.
6. **Pre-2018 outcome history is a different regulatory regime.** A `Fail` in
   2014 and a `Fail` in 2024 are not identical events. Counts pool them; the
   era-restricted priority features do not.
7. **No text, spatial, weather or licence features.** The project spec lists
   these (its Finding-5 answer); their datasets are not ingested, so adding them
   is a Component 1 extension rather than something to fabricate here.
8. **Feature usefulness is unmeasured.** No model exists, and selecting features
   by downstream accuracy would itself be a form of leakage. **NOT VERIFIED**
   that any feature predicts the target.
