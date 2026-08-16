# ADR 0010 — As-of feature construction and the temporal boundary

**Status:** Accepted · **Date:** 2026-08-16

## Context

Component 3 defines what Sentinel predicts: at an eligible routine canvass, did
that canvass find a Priority or Priority Foundation violation? Component 4 must
supply the other half of a training row — what a scheduler knew *before* that
inspection happened.

This is the first place in the project where a plausible-looking implementation
can produce entirely fake results. A model trained on features that quietly
include the outcome will report excellent metrics and rank real establishments
badly. The failure is silent: nothing errors, the table looks healthy, and the
only symptom is performance that is too good.

Three specific ways it happens, all of which this component must exclude by
construction rather than by care:

1. `groupby(establishment_id)` over the whole history, merged onto every row —
   a 2019 row receives statistics computed from 2024 data.
2. A self-join with no date predicate, or with the predicate on only one branch.
3. `<=` instead of `<`, admitting records dated on the reference date itself.

The data settles the third question. Measured over the full snapshot:

- `inspection_date` has exactly **one** distinct time component
  (`T00:00:00.000`) across all 314,245 rows. It is a date, not a timestamp.
- 2,103 target rows have at least one other inspection on their reference date:
  **1,075 `License`**, **43 `Canvass Re-Inspection`**, 42 `Complaint`, and a
  handful of others.
- 793,200 (target, prior inspection) pairs exist under a strictly-before join,
  which aggregates in about 0.1 s.

## Decision

**Every feature is computed from one range join carrying one explicit temporal
condition, and the boundary is strictly `<` on the date.**

```sql
FROM targets t
LEFT JOIN history h
  ON  h.establishment_id = t.establishment_id
  AND h.inspection_date  <  t.inspection_date
GROUP BY t.target_inspection_id
```

Five supporting decisions:

**1. The boundary is exclusive.** Since there is no intra-day ordering, same-day
records cannot be placed before or after the canvass. A same-day
`Canvass Re-Inspection` exists *because* an inspection failed, so it almost
certainly followed the canvass being predicted. Admitting it would feed the
outcome back in, and nothing in the data could unwind it. An inspection dated on
the reference date is therefore never history — including the target's own.

**2. One join, not one per feature.** All 26 features are aggregates over the
same restricted row set, so the temporal guarantee is structural. There is one
predicate to review rather than 26.

**3. `LEFT JOIN`, not `JOIN`.** The 401 history-less target rows survive as rows
with NULL/0 features. An inner join would drop them silently, which is exactly
the "do not drop data silently" failure.

**4. Priority is classified by Component 3's own parser.** A SQL
`LIKE '%PRIORITY%'` would drift from the target's definition, which excludes
narrative spans such as the 90-day grace-period boilerplate. Reusing
`sentinel.target.violations` guarantees "priority" means the same thing on both
sides of the training row.

**5. The invariant is re-derived independently in validation.** A separate query
recomputes the latest contributing date per row and asserts it is earlier. It
runs on all 57,727 rows, not a sample.

### Missing values: four rules

| kind | rule |
|---|---|
| counts | never NULL; `0` = "none observed" |
| recency | NULL when the event never happened |
| rates | NULL when the denominator is 0 |
| at-last flags | NULL when there is no prior event |

Every event count is emitted beside its inspection count, so a `0` is legible.

## Alternatives rejected

**`groupby(establishment_id)` then merge.** The canonical leakage bug. It
computes whole-history statistics and attaches them to every row of that
establishment, so a 2019 row learns from 2024. Rejected outright; the component
contains no establishment-level intermediate that could be merged.

**`<=` on the date.** Would admit 43 same-day canvass re-inspections that
provably follow their canvass, plus 42 same-day complaints of unknown order.
Rejected because the data offers no way to order same-day records — the choice is
not between precision and convenience, it is between exclusion and guessing.

The cost is real: up to 1,075 same-day licence inspections are discarded although
they may genuinely have preceded the canvass. Accepted, because a component whose
purpose is preventing leakage should fail toward exclusion.

**A Python loop over targets × history.** Straightforward to reason about and
O(N × M) — 57,727 targets against 314,245 inspections. Rejected on cost, though
notably the range join is *also* easier to audit: the predicate appears once,
in SQL, instead of being reimplemented per feature.

**Window functions over a unioned event stream.** Elegant for recency, awkward
for the counts and rates, and it would have required a second pass for the
"at last canvass" values. `arg_max` inside the same aggregation does that in one.

**Materialising features per establishment and joining by date range at read
time.** Defers the temporal logic to every consumer, which is precisely the
mistake — the guarantee must be baked into the artifact, not left to whoever
reads it.

**Selecting features by downstream model performance.** There is no model, and
choosing features by test-set accuracy is itself a form of leakage. Features are
justified by domain reasoning, availability and data quality only.

## Consequences

- **A zero-day recency is unconstructable.** `days_since_last_canvass` has a
  minimum of 1 and no zeros in the full build — the cheapest available proof the
  boundary works.
- **The invariant is testable, and tested.** Four dedicated leakage regression
  tests live in their own file: future insertion, future mutation, target
  self-exclusion, same-day exclusion.
- **Reproducibility is exact.** Rebuilding and a seeded row shuffle both produce
  identical values.
- **The build is fast enough to iterate on**: 15.6 s over the full snapshot,
  dominated by the Python violation parser rather than the temporal logic.
- **Some information is deliberately forgone**: same-day licence inspections, and
  priority history for the 24.5% of rows without a prior code-era canvass.
- **Changing any rule changes the features**, so `feature_definition_version` is
  stamped on every row and recorded in the manifest.
- **The boundary must be restated for Component 5.** The same discipline applies
  one level up: an evaluation that splits randomly trains on the future even when
  every feature is correct.
