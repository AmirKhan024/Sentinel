# Data contract: inspection targets (Component 3 output)

**Produced by:** `sentinel build-target` (`src/sentinel/target/`)
**Layer:** `data/interim/target/`
**Consumed by:** Component 4 (as-of features) onward
**Design rationale:** `docs/analysis/target_construction_findings.md`, ADR 0008, ADR 0009

---

## 1. Prediction unit

> **One target row = one (establishment, date) on which at least one eligible
> routine canvass occurred.**

Not one row per inspection: 530 establishment-dates carry more than one eligible
canvass and 160 of those disagree on the outcome (findings §11). "Inspect
establishment E on date D" is a single scheduling decision, so it gets a single
row.

Not one row per establishment: an establishment is inspected repeatedly, and each
visit is a separate decision with a separate outcome. 15,144 establishments
produce 57,727 eligible rows.

---

## 2. Reference event and target event

They are the **same inspection**, viewed from two sides:

```
            reference time                target event
                  │                            │
   ───────────────┼────────────────────────────┼──────────────►
   information    │  the canvass on            │
   available      │  inspection_date           │
   before this    │                            │
   point is a     │  what it found is          │
   FEATURE        │  the TARGET                │
```

The canvass *is* the prediction event. `inspection_date` is therefore the **as-of
boundary**: Component 4 may use only information dated strictly before it.

This matches the real decision. A scheduler standing on the morning of
`inspection_date` asks "if I send an inspector to E today, will they find a
priority violation?" — and the answer is recorded by the inspection that
followed.

A "predict the *next* canvass" formulation was considered and rejected (findings
§13, ADR 0008): it discards every establishment's most recent canvass (15,148
rows), and attaches the label to an event a median of 377 days later with an
interquartile range of over six months.

---

## 3. Target definition

**Plain English.** For each establishment-date on which a routine canvass
happened, the target is 1 if that day's canvassing found at least one Priority or
Priority Foundation violation, and 0 if it found none.

**Logical form.**

```
target(e, d) = 1  ⟺  ∃ i ∈ Canvasses(e, d) : ∃ v ∈ Violations(i) :
                          severity(v) ∈ {PRIORITY, PRIORITY_FOUNDATION}
```

where `Canvasses(e, d)` is the set of eligible canvasses for establishment `e` on
date `d`, and `severity` is the deterministic classifier in §6.

**Not** `results == 'Fail'`. Among eligible canvasses, priority violations are
present in 99.4% of `Fail` rows and **97.9% of `Pass w/ Conditions`** rows. A
result-based target would mislabel 16,261 inspections (§8).

---

## 4. Eligibility

All four gates must hold. They are evaluated in this order, so a row's exclusion
reason is the *first* thing that disqualified it.

| # | gate | rule | rows failing |
|---|---|---|---|
| 1 | era | `inspection_date >= 2018-07-01` | 172,879 |
| 2 | type | `upper(trim(inspection_type)) == 'CANVASS'` | 70,848 |
| 3 | result | `results ∈ {Pass, Pass w/ Conditions, Fail}` | 12,091 |
| 4 | text | violation text is interpretable (§7) | 79 |

**Era.** Chicago replaced Critical/Serious with Priority / Priority Foundation /
Core on 2018-07-01, cleanly (findings §5). Before that date the target is not
sparse — it is *undefined*. ADR 0009.

**Type.** Only routine canvasses. `Canvass Re-Inspection` (16,998 code-era rows)
is excluded because it exists only *because* an earlier inspection failed;
including it would condition the target on past failure. `Complaint`, `License`,
`Short Form Complaint` and the rest are excluded because they are triggered
events, not routine visits.

**Result.** The other four values mean no inspection took place (§5).

---

## 5. Class definitions

| class | definition | rows |
|---|---|---|
| **positive** (`target = 1`) | ≥1 Priority or Priority Foundation violation found | 30,316 |
| **negative** (`target = 0`) | eligible, violation text interpretable, no priority violation found | 27,411 |
| **excluded** (`target = NULL`) | any eligibility gate failed | 255,818 |
| **unknown** (`target = NULL`) | `unknown_violations`: result and text contradict | 79 |

Negatives include both "violations found but all Core" (22,437) and "no
violations recorded on a passing inspection" (4,974).

### `target_status`

Every source inspection receives exactly one status, so **no exclusion is
silent** and every category is countable:

| value | meaning | rows |
|---|---|---|
| `eligible` | labelled | 57,727 |
| `ineligible_era` | before 2018-07-01 | 172,879 |
| `ineligible_type` | not a routine canvass | 70,848 |
| `ineligible_result` | no inspection took place | 12,091 |
| `unknown_violations` | result asserts violations, text records none | 79 |

---

## 6. Priority / Priority Foundation detection

The classification lives in the **comment text**, not in the violation number.

**The number is not used.** Item 10 is 42.2% Priority Foundation, 11.3%
Priority and 46.5% unlabelled, because the same numbered item covers "no hot
water at the hand sink" and "no hand-washing sign" (findings §7.1).

**A municipal code is not required.** 21,281 code-era entries carry a marker with
no `7-38-xxx` code and are genuine — `PRIORITY FOUNDATION VIOLATION. NO CITATION
ISSUED.` Requiring one would create ~21,281 false negatives (findings §7.3).

### The rule

For each `|`-separated entry:

1. Split the text into chunks on `.` and `;`.
2. Discard chunks matching a **narrative pattern** (below).
3. If any surviving chunk contains `PRIORITY FOUNDATION` → `PRIORITY_FOUNDATION`.
4. Else if any contains `PRIORITY` → `PRIORITY`.
5. Else → `UNCLASSIFIED`.

`UNCLASSIFIED` deliberately does **not** mean "Core". 72% of entries carry no
label and `CORE` is written only 7,943 times, so an unlabelled entry is absence
of evidence, not evidence of Core.

### Narrative patterns, enumerated

These are the only places the parser decides that text containing "PRIORITY" is
not a priority violation. Each is justified by a real example in findings §7.4.

| name | pattern | why |
|---|---|---|
| `grace_period_days` | `\d+\s*-?\s*DAY\s+GRACE\s+PERIOD` | "A 90 day grace period was given for all new priority and priority foundation violations" — boilerplate notice |
| `grace_period` | `GRACE\s+PERIOD` | same, without a day count |
| `future_citation` | `WILL\s+BE\s+ISSUED` | "OR CITATION PRIORITY FOUNDATION WILL BE ISSUED" — a citation that has not happened |
| `negation` | `\bNO\s+PRIORITY` | "NO PRIORITY FOUNDATION VIOLATION 7-38-030(c)" — explicit negation |

Exclusion is **span-based, not entry-based**: a genuine citation can co-occur
with a warning in the same entry, so only the offending chunk is dropped.

**Measured effect** (challenge this number if you disagree with a pattern):

| level | naive matching | with exclusions | difference |
|---|---|---|---|
| entries | 137,598 | 137,524 | 74 (0.054%) |
| inspection labels | 30,498 | 30,488 | 10 (0.017%) |

---

## 7. Special-case handling

### `Pass w/ Conditions`

**Eligible, labelled exactly like every other result.** It is not a pass in risk
terms: priority violations are present in 97.9% of them against 0.5% of plain
`Pass` (findings §8.1). In practice almost all are positive, but that is an
outcome of the rule rather than a special case in it.

### `Out of Business`

**Ineligible, not negative.** 99.9% have null `violations` — no inspection took
place. Labelling them negative would teach a model that a closed establishment is
a clean one, and since closure correlates with prior poor performance that is
exactly backwards.

It is also **not terminal**. 24.9% of `Out of Business` records are followed by
another inspection at the same establishment, median 273 days later, because
Component 2 tracks physical premises and a new tenant moves in. An establishment
with an OOB record remains eligible for later canvasses.

`No Entry`, `Not Ready` and `Business Not Located` are handled identically.

### Multiple canvasses on one date

Collapsed to one row with `target = OR` over the day. 530 establishment-dates in
the output; 160 of the underlying multi-canvass days disagreed.

`target_inspection_id` points at the **positive** contributing inspection when
there is one (else the lowest id), so provenance points at the evidence.
`n_contributing_inspections` and `contributing_inspection_ids` preserve the rest.

### Missing violation text

Interpreted **by result**, not uniformly (findings §10):

| situation | rows | treatment |
|---|---|---|
| `Pass` + null | 4,974 | **negative** — a true zero; the expected encoding of "nothing found" |
| `Pass w/ Conditions` + null | 49 | **`unknown_violations`** — self-contradictory |
| `Fail` + null | 30 | **`unknown_violations`** — self-contradictory |
| non-inspection result + null | 12,091 | already `ineligible_result` |

This is the **only** place construction reads `results`, and it is used to decide
whether a row is *labellable*, never to decide the label. Using the result for
the label would make the target partly circular, since the result is itself a
summary of what was found.

---

## 8. Output schema

`data/interim/target/inspection_targets_<stamp>.parquet`, one row per decision
point. 313,624 rows in the reference build.

### Identity and timing

| column | type | null? | meaning |
|---|---|---|---|
| `establishment_id` | Utf8 | never | From Component 2. **Never re-derived.** |
| `inspection_date` | Utf8 | never | ISO-8601. **The as-of boundary.** |
| `target_inspection_id` | Utf8 | never | The inspection carrying the label |

### The label

| column | type | null? | meaning |
|---|---|---|---|
| `target` | Int8 | **yes** | 1 / 0, null when not `eligible` |
| `target_status` | Utf8 | never | See §5 |

`target` is a nullable `Int8` rather than `Boolean` because null is a meaningful
third state and a nullable boolean invites silent coercion to False.

### Audit — describes the target event, **not** history

| column | type | meaning |
|---|---|---|
| `inspection_type` | Utf8 | raw value |
| `results` | Utf8 | raw value |
| `has_priority` | Boolean | a plain Priority violation was found |
| `has_priority_foundation` | Boolean | a Priority Foundation violation was found |
| `n_priority_entries` | Int32 | count across the day |
| `n_priority_foundation_entries` | Int32 | count across the day |
| `n_violation_entries` | Int32 | total parsed entries |
| `evidence` | Utf8 | the exact matched span that produced a positive |
| `n_contributing_inspections` | Int32 | same-day collapse |
| `contributing_inspection_ids` | Utf8 | space-separated |

### Provenance

| column | type | meaning |
|---|---|---|
| `code_era_phase` | Utf8 | `pre_code` / `adoption` / `stable` |
| `target_definition_version` | Utf8 | `v1` |

---

## 9. ⚠ Leakage rules for Component 4

**These columns describe what the inspection found. They are the outcome. Using
any of them as a model input is leakage:**

```
target                          results
has_priority                    evidence
has_priority_foundation         n_priority_entries
n_violation_entries             n_priority_foundation_entries
```

The set is also enumerated in code as `sentinel.target.writer.TARGET_EVENT_COLUMNS`
and asserted by a test, so it can be checked programmatically.

**`inspection_date` is the boundary.** A feature for row *(e, d)* may use only
information dated **strictly before** *d*. In particular:

- the establishment's prior inspections — yes, those before *d*
- this inspection's own violations, results or type — **no**
- any later inspection — **no**
- an aggregate computed over the whole history — **no**, it must be as-of *d*

**Why Component 3 may look at the target event while Component 4 may not.**
Component 3 is *constructing* the label; the outcome is what it is defined from.
Component 4 is constructing the information a decision-maker had *before* the
outcome existed. Same row, two different time positions.

Component 3 computes **no** historical quantity — no counts, no gaps, no prior
rates. A test asserts the schema contains no `prev_*`, `n_prior_*`,
`days_since_*`, `rolling_*`, `hist_*` or `prior_*` column.

---

## 10. Join to Component 2

```sql
SELECT t.*, a.establishment_id
FROM read_parquet('data/interim/target/inspection_targets_*.parquet') t
JOIN read_parquet(
    'data/interim/entity_resolution/establishment_assignments_*.parquet'
) a USING (establishment_id)
```

Identity comes from Component 2 and is never re-derived from `license_`,
`dba_name` or `address`. Inspections without an assignment are dropped, with a
warning, because there is no establishment to attribute the outcome to. In the
reference build this dropped **0** rows.

---

## 11. Determinism and versioning

**Guaranteed:** the same raw snapshot plus the same assignments plus the same
`target_definition_version` always produce identical labels. Verified on the full
314,245 rows: rebuilding reproduces the committed table exactly, and building
from a seeded random permutation of every input row produces an identical label
set.

This rests on three properties: classification is a pure function of the text;
same-day collapse orders contributing inspections numerically; output is sorted
by `(establishment_id, inspection_date, target_inspection_id)`.

**`target_definition_version`** is stamped on every row and recorded in the
manifest. Bump it whenever eligibility, the positive class, the parser's rules or
the collapse rule change. A model run can then identify exactly which definition
produced its labels.

The manifest pins **both** input checksums. The labels are a function of the raw
snapshot and the Component 2 assignments *together*, so recording only one would
leave the output unreproducible. Note that `establishment_id` is itself
snapshot-scoped (ADR 0007), so re-resolving may change which rows group together.

---

## 12. Guarantees a consumer may rely on

Asserted by error-severity checks that fail the command:

1. Every `establishment_id` exists in the Component 2 assignments.
2. Every `target_inspection_id` exists in the raw snapshot.
3. At most one eligible row per `(establishment_id, inspection_date)`.
4. Every source inspection is represented exactly once across all rows.
5. `target` is non-null **iff** `target_status == 'eligible'`.
6. `target ∈ {0, 1}` when present.
7. No eligible row is dated before 2018-07-01.
8. Every eligible row is a routine canvass.
9. Every eligible row has a result meaning an inspection occurred.
10. Every positive carries a non-empty `evidence` span.
11. `target == 1` **iff** `has_priority or has_priority_foundation`.
12. A single `target_definition_version` across the table.

Reported but **not** enforced, because they are legitimate: positive rate, status
breakdown, per-year drift, same-day collapse count, phase breakdown.

---

## 13. Known limitations

1. **52% of the dataset is `ineligible_era`.** The 2010 – 2018 H1 period cannot
   support this target. A separate Critical/Serious target could be defined for
   it; not attempted.
2. **The base rate drifts from 87.6% to 39.1%.** Component 5 must account for
   this; any evaluation that shuffles rows across time measures the drift rather
   than the model.
3. **The narrative exclusion list is judgement.** Four patterns, 74 entries, 10
   inspection labels. Enumerated in §6 so a reviewer can disagree with a specific
   rule.
4. **8 `Pass w/ Conditions` rows are negative** where the result implies
   otherwise, because the parser stays independent of `results`.
5. **Inspector write-up variation is unmeasurable here.** A priority violation
   found but not labelled is a false negative, and the open data contains no
   ground truth to check against. **NOT VERIFIED.**
6. **Severity within positive is not represented.** One priority violation and
   twelve produce the same label.
7. **`Canvass Re-Inspection` is excluded** (16,998 code-era rows) — a deliberate
   scope restriction, not a claim they are uninformative.
8. **The adoption phase (2,829 rows) is anomalous** at 87.6% and flagged rather
   than corrected.
