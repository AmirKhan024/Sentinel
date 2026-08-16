# As-of feature engineering: empirical findings

Component 4 investigation. Every number was measured against one snapshot with
`scripts/profile_features.py`, joining the raw data, Component 2's assignments
and Component 3's targets.

**Read this before changing `src/sentinel/features/`.** The temporal boundary in
particular was settled by a measurement, not a preference, and the missing-value
rules exist because the alternative teaches a model that "no history" means
"clean history".

---

## 1. The snapshot and the inputs

| Input | File | Rows |
|---|---|---|
| Raw inspections | `food_inspections_20260816T070911Z.parquet` | 314,245 |
| Component 2 assignments | `establishment_assignments_20260816T085729Z.parquet` | 314,245 |
| Component 3 targets | `inspection_targets_20260816T122821Z.parquet` | 313,624 |
| **Eligible target rows** | (the prediction opportunities) | **57,727** |

Raw sha256 `7d3c4069340a68d197204c6cca9fca6399c6565bc3668760f145f43cd377ad38`.
All 314,245 raw rows join to an establishment, so the history pool is complete.

---

## 2. Data quality of the history pool

```
rows                          314,245
null inspection_date                0
unparseable inspection_date         0
duplicate inspection_id             0
date range          2010-01-04 .. 2026-08-14
```

This is unusually clean, and it removes three of the questions the component
brief anticipated: there are no missing dates to handle, no dates to fabricate,
and no duplicate `inspection_id` to deduplicate.

Multiple inspections *do* occur on one establishment-date — 12,135
establishment-dates have 2, 1,784 have 3, up to 12 — but those are distinct
inspections with distinct ids, not duplicate records. §4 explains why that
matters more than it looks.

---

## 3. How much history exists at prediction time

Counting records strictly before each row's reference date:

| prior inspections (any type) | target rows | % |
|---|---|---|
| none | 401 | 0.69 |
| 1–2 | 4,370 | 7.57 |
| 3–5 | 7,134 | 12.36 |
| 6–10 | 11,301 | 19.58 |
| 11–20 | 24,765 | 42.90 |
| 21+ | 9,756 | 16.90 |

| prior **canvasses** | target rows | % |
|---|---|---|
| none | 5,615 | 9.73 |
| 1–2 | 8,168 | 14.15 |
| 3–5 | 9,659 | 16.73 |
| 6–10 | 17,369 | 30.09 |
| 11+ | 16,916 | 29.30 |

Summary:

| quantity | value |
|---|---|
| target rows | 57,727 |
| with no history at all | **401 (0.69%)** |
| with no prior canvass | **5,615 (9.73%)** |
| with no prior **code-era** canvass | **14,162 (24.53%)** |
| with pre-2018 history available | 46,156 (79.96%) |
| mean prior inspections (any type) | 13.74 |
| mean prior canvasses | 7.86 |

**History is abundant.** Only 0.69% of prediction opportunities are genuinely
cold-start. That is a good position to be in, and it means the no-history path
must be *correct* rather than merely tolerable — it is rare enough that a bug in
it would not show up in an aggregate.

The observation window is long: median 3,277 days (about nine years) between an
establishment's first recorded inspection and a target row's reference date
(p25 1,788, p75 4,419, max 6,060).

---

## 4. The temporal boundary — settled by measurement

### 4.1 There is no intra-day ordering

```
distinct time components in inspection_date : 1
the only value                              : T00:00:00.000
rows                                        : 314,245
```

Every one of the 314,245 timestamps has a zero time component. `inspection_date`
is a **date**, not a timestamp. Two inspections on the same establishment-date
therefore **cannot be ordered**, by any means available in this dataset.

### 4.2 What sits on the reference date itself

| inspection type on the reference date | occurrences |
|---|---|
| Canvass | 59,049 |
| **License** | **1,075** |
| **Canvass Re-Inspection** | **43** |
| **Complaint** | **42** |
| Non-Inspection | 20 |
| License Re-Inspection | 10 |
| Complaint Re-Inspection | 4 |
| Recent Inspection | 2 |
| Not Ready | 2 |
| Short Form Complaint | 1 |

2,103 target rows have at least one other inspection on their reference date
(1,803 have one extra, 230 have two, and one has twelve).

The 43 same-day `Canvass Re-Inspection` records are decisive. A re-inspection
exists *because* an inspection failed, so a same-day re-inspection almost
certainly happened **after** the canvass being predicted. Including it as
"history" would feed the outcome back in as an input. And because §4.1 shows
there is no time component, no amount of care could separate the ones that came
before from the ones that came after.

> **Decision. The boundary is strictly `<` on the date.** An inspection dated on
> the reference date is **never** history — including the target's own
> contributing inspections, and including the 1,075 same-day licence inspections
> that might well have preceded the canvass.
>
> This is deliberately conservative. It discards a small amount of genuinely
> prior information (some of those 1,075 licence inspections) in exchange for a
> guarantee that no outcome-contaminated record can reach a feature. Given that
> Component 4's entire purpose is preventing leakage, the trade is the right way
> round.

---

## 5. Inspection cadence — this sizes the windows

Consecutive **canvasses** within an establishment (131,556 pairs):

| p25 | p50 | p75 | p90 | max |
|---|---|---|---|---|
| 251 d | **358 d** | 482 d | 752 d | 5,612 d |

Consecutive inspections of **any type** (278,386 pairs):

| p25 | p50 | p75 | p90 |
|---|---|---|---|
| **9 d** | 206 d | 366 d | 524 d |

Two things follow.

**The canvass cycle is roughly annual** — a median of 358 days, consistent with
the statutory risk-based frequencies the policy engine will later enforce.

**The any-type series has a heavy short tail.** A p25 of **9 days** is the
re-inspection pattern: an establishment fails, and someone returns within days.
This is exactly the circularity the project spec warns about — a feature like
"days since last inspection of any type" will read 9 days precisely when the
previous inspection was a failure. It is not leakage (the re-inspection genuinely
happened before the reference date), but it risks a model learning Chicago's
*scheduling policy* rather than an establishment's *risk*. §8 records how this is
handled.

### 5.1 Window occupancy — the sparsity is real

| window | target rows | rows with an empty window | % empty | mean canvasses in window |
|---|---|---|---|---|
| 365 d | 57,727 | 35,781 | **62.0** | 0.44 |
| 730 d | 57,727 | 12,716 | 22.0 | 1.25 |
| 1095 d | 57,727 | 8,229 | 14.3 | 2.00 |

**A one-year window is empty for 62% of rows.** That is not a defect in the
window; it is the direct consequence of a 358-day median cycle — the previous
canvass usually falls just *outside* a 365-day lookback.

> **Decision.** Keep 365 / 730 / 1095 days, matching the spec's 1y/2y/3y, and
> publish the occupancy. The 365-day window is sparse but not useless: a non-zero
> value means the establishment was canvassed *off-cycle*, which is itself a
> signal. What matters is that a reader can tell an empty window from a clean one,
> which is why every event count is emitted beside its inspection count (§7).

---

## 6. What the history contains

Prior records reaching target rows, by type:

| type | prior records |
|---|---|
| Canvass | 453,796 |
| Canvass Re-Inspection | 100,550 |
| License | 78,476 |
| Complaint | 70,239 |
| Complaint Re-Inspection | 25,953 |
| Short Form Complaint | 21,268 |
| License Re-Inspection | 20,518 |
| Non-Inspection | 9,393 |
| Suspected Food Poisoning | 3,719 |

By result, restricted to prior **canvasses**:

| results | prior canvasses |
|---|---|
| Pass | 254,820 |
| Fail | 84,698 |
| Pass w/ Conditions | 84,450 |
| Out of Business | 16,517 |
| No Entry | 13,077 |
| Not Ready | 193 |
| Business Not Located | 41 |

Note the last four. A prior canvass can be an `Out of Business` or `No Entry`
record where **no inspection happened**. Those must not count as a "pass" in a
historical outcome rate, for the same reason Component 3 declared them ineligible
as targets: nobody inspected anything.

> **Decision.** Historical outcome counts and rates are computed over prior
> canvasses whose result is one of `Pass`, `Pass w/ Conditions`, `Fail` — the same
> `INSPECTED_RESULTS` set Component 3 uses. Non-inspection results are counted
> separately, not folded into the denominator.

### 6.1 Priority history availability

| | target rows |
|---|---|
| total | 57,727 |
| **priority features NULL** (no prior code-era canvass) | **14,162 (24.5%)** |
| genuine zero (code-era history exists, no priority found) | 6,442 |
| has at least one prior priority violation | 37,123 |

This is the sharpest instance of the zero-versus-null problem in the whole
component. 14,162 rows have *no evidence*, and 6,442 rows have *evidence of
absence*. Collapsing them to `0` would tell a model that a quarter of
establishments have a clean priority record when in fact nothing is known about
them.

Priority history is restricted to code-era canvasses because Priority and
Priority Foundation did not exist before 2018-07-01 (ADR 0009). Pre-2018
inspections still contribute to the generic outcome and count features — the era
boundary constrains what can be *classified*, not what can be *counted*.

---

## 7. Missing-value semantics — four rules

| kind | rule | why |
|---|---|---|
| **counts** | never NULL; `0` means "none observed" | 0 is a true observation. Every subtype count is emitted beside its total, so a reader can distinguish "none of these" from "none at all" |
| **recency** (`days_since_*`) | **NULL** when no such prior event | `0` would mean "it happened today", which is the opposite of "it never happened" |
| **rates** | **NULL** when the denominator is 0 | `0/0` is not `0` |
| **at-last flags** | **NULL** when there is no prior event | the absence of a last inspection is not a clean last inspection |

The pairing convention is the safety net: `canvass_priority_events_last_365d = 0`
is ambiguous on its own, but read next to `canvasses_last_365d = 0` it plainly
means "not inspected in that window" rather than "inspected and clean".

---

## 8. Tenant changes — a measured divergence from the project spec

The project spec (§3.3) says an ownership transition should be treated as a
**new establishment**, on the reasoning that a change of owner resets the risk
profile. Component 2 deliberately does the opposite: an establishment is a
*physical premises*, and successive tenants at one address are one establishment
(ADR 0006).

Scale:

| premises | establishments | inspections |
|---|---|---|
| one name | 32,788 | 258,424 |
| two names | 2,572 | 43,030 |
| three or more | 499 | 12,791 |

| | target rows | % |
|---|---|---|
| single-name premises | 48,551 | 84.1 |
| **multi-name premises** | **9,176** | **15.9** |

At the transition level: of 131,556 consecutive canvass pairs, **6,523 (4.96%)**
show a name change between them.

> **Decision.** Keep Component 2's contract and **expose the transition as
> features** rather than resetting history.
>
> Reasons: ADR 0006 and 0007 are settled and reopening them from a downstream
> component would be the wrong direction of travel; the claim "a rename means a
> new business" is **NOT VERIFIED** in this data (a rename can be a rebrand under
> the same owner); and exposing the signal is strictly more informative than
> acting on it, because a later component can learn to discount history across a
> transition, whereas discarded history cannot be recovered.
>
> A reset-history variant remains available as future work and would be an
> ablation, not a redesign.

---

## 9. Candidate features and what was rejected

### Accepted — 26 features in six families

**Canvass history** (8) — the routine series, directly comparable to the target:
`prior_canvass_count`, `prior_canvass_count_code_era`, `days_since_last_canvass`,
`prior_canvass_inspected_count`, `prior_canvass_fail_count`,
`prior_canvass_pass_w_conditions_count`, `prior_canvass_fail_rate`,
`fail_at_last_canvass`.

**Priority history** (4) — code-era canvasses only:
`prior_canvass_priority_count`, `prior_canvass_priority_foundation_count`,
`prior_canvass_priority_rate`, `priority_at_last_canvass`.

**Windows** (6) — half-open `[d − N, d)`, N ∈ {365, 730, 1095}:
`canvasses_last_{365,730,1095}d`,
`canvass_priority_events_last_{365,730,1095}d`.

**All-type context** (5) — a different series, labelled as such:
`prior_inspection_count_any_type`, `days_since_any_inspection`,
`prior_complaint_count`, `prior_reinspection_count`,
`prior_license_inspection_count`.

**Tenant change** (2): `name_changed_since_last_canvass`,
`prior_canvass_count_current_name`.

**Observation** (1): `days_since_first_inspection`.

### Rejected, with reasons

| candidate | why not |
|---|---|
| any model score, embedding or predicted probability | Component 6+; Component 4 is deterministic history only |
| global normalisation, standardisation, percentile ranks | computed over the whole dataset they use future statistics; scaling belongs inside a temporally-bounded training fold |
| demographic / ACS variables | spec §8: audit-only, never model inputs |
| 311, weather, crime, business-licence features | their datasets are not ingested. Adding them is a Component 1 extension, not something to fabricate here |
| `code_era_phase` as a predictor | a stratification variable describing the regulatory regime, not a property of the establishment |
| more than three windows | the three are already correlated (a 365-day count is a subset of the 730-day count); more would add collinearity without information |
| `days_since_last_inspection` as the *primary* recency | the p25 of 9 days is the re-inspection pattern; retained as clearly-labelled context alongside the canvass-only recency so a later component can ablate it |
| historical violation *text* features | the spec lists these (its Finding-5 answer) and they are a legitimate future addition, but they need a text-feature pipeline that is out of scope for a component whose job is the temporal boundary |

---

## 10. Leakage risks and how each is closed

| risk | how it would happen | closure |
|---|---|---|
| The target's own inspection enters its features | a join on `establishment_id` with no date condition | the boundary is `<`, in the SQL, and re-derived independently in validation |
| A same-day re-inspection enters | `<=` instead of `<` | §4: strictly `<`, justified by the absent time component |
| Whole-history aggregates | `groupby(establishment_id)` then merge onto every row | the aggregation is per target row, over a range join; there is no establishment-level intermediate to accidentally merge |
| Future inspections | forgetting the condition on one branch of a multi-part query | a single range join feeds every feature, so there is one place to get it right, plus four dedicated leakage tests |
| Priority features silently using pre-2018 records | applying the priority filter without the era filter | priority aggregates carry both conditions and a validation check asserts the NULL pattern matches `prior_canvass_count_code_era == 0` |
| Feature selection by downstream accuracy | trying features against a model | there is no model; features are justified by domain reasoning and availability only |

---

## 11. Full-data build results

Deferred to §14, written after the build.

---

## 12. Determinism

The aggregation is order-independent (a `GROUP BY` over a range join), and the
output is sorted by `(establishment_id, inspection_date, target_inspection_id)`.
Verified on the full snapshot in §14.

---

## 13. Limitations

1. **The boundary discards some genuinely prior same-day information** — up to
   1,075 licence inspections that may have preceded their canvass. Accepted
   deliberately (§4).
2. **Priority features are NULL for 24.5% of rows.** Real, and correct, but it
   means a quarter of rows carry no priority history at all.
3. **A 365-day window is empty for 62% of rows** (§5.1).
4. **History can span a tenant change for 15.9% of rows** (§8). Exposed, not
   resolved.
5. **`days_since_any_inspection` partly encodes scheduling policy**, not risk
   (§5). Labelled and separable, not removed.
6. **Feature usefulness is unmeasured and unmeasurable here.** No model exists,
   and selecting features by downstream accuracy would itself be a form of
   leakage. **NOT VERIFIED** that any of these features predicts the target.
7. **No text, spatial, weather or licence features**, so the spec's Finding-5
   answer is not yet delivered. Out of scope for this component.
8. **The pre-2018 outcome history is a different regulatory regime.** A `Fail` in
   2014 and a `Fail` in 2024 are not identical events. The counts pool them; the
   era-restricted priority features do not.

---

<!-- §14 is appended after the feature table has been built. -->
