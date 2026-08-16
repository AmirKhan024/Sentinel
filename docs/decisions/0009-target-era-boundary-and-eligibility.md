# ADR 0009 — The 2018-07-01 code-era boundary

**Status:** Accepted · **Date:** 2026-08-16

## Context

The Sentinel target is defined in terms of Priority and Priority Foundation
violations (ADR 0008). Those categories come from the FDA-style food code that
Chicago adopted partway through the dataset.

Measured over all 314,245 rows, the changeover is unambiguous:

| month (2018) | rows | "Priority" terminology | "Critical/Serious" terminology |
|---|---|---|---|
| 2018-01 | 1,582 | 0 | 450 |
| … | … | 0 | 342–443 |
| 2018-06 | 1,589 | **0** | **415** |
| **2018-07** | 1,207 | **761** | **0** |
| 2018-08 | 1,423 | 1,030 | 0 |
| 2018-12 | 1,018 | 636 | 0 |

Zero rows use the new terminology before July 2018; essentially zero use the old
terminology after. There is no overlap month, no gradual migration, and no
mixed-vocabulary period.

The consequence is not that pre-2018 data is *sparse* for this target. It is that
the target **does not exist** there. Asking "did this 2014 inspection find a
Priority Foundation violation?" is not a hard question — it is a question about a
classification scheme that had not been introduced.

A second measurement complicates the boundary. The positive rate immediately
after adoption is far above its later level:

| period | eligible rows | positive rate |
|---|---|---|
| 2018-07-01 → 2018-12-31 | 2,829 | **87.6%** |
| 2019 | 8,230 | 77.4% |
| 2021 | 6,154 | 50.3% |
| 2025 | 8,269 | 39.2% |
| 2026 (partial) | 3,883 | 39.1% |

The violation text itself explains part of this: entries from that period carry
boilerplate reading *"A 90 day grace period was given for all new priority and
priority foundation violations."* Inspectors were newly applying an unfamiliar
classification under a transitional enforcement policy.

## Decision

**Target eligibility begins on 2018-07-01, inclusive.**

Rows before that date are emitted with `target_status = 'ineligible_era'` and a
null target. They are **not deleted**: 172,879 rows (55% of the dataset) remain
visible and countable in the output, so the exclusion can be audited rather than
inferred from a missing row count.

**The adoption period is included but flagged.** Every row carries
`code_era_phase`:

| value | range | eligible rows |
|---|---|---|
| `pre_code` | before 2018-07-01 | — (never eligible) |
| `adoption` | 2018-07-01 → 2018-12-31 | 2,829 |
| `stable` | 2019-01-01 onward | 54,898 |

Component 3 does not decide whether to use the adoption rows. Component 5 does.

## Alternatives rejected

**Start eligibility at 2019-01-01.** Cleaner and more homogeneous, and the
adoption phase is only 4.9% of eligible rows. Rejected because it silently
discards six months of genuine inspections on the basis of a distributional
judgement that belongs to the evaluation component, not to target construction.
The flag gives Component 5 the same option without Component 3 pre-empting it.

**Include the adoption period with no marker.** Simplest schema, but the 87.6% →
39.1% drift would become an undocumented trap. Anyone training on pooled data and
evaluating on a recent holdout would see a large, unexplained calibration gap.

**Back-fill a Priority target for 2010–2018 H1** by mapping the old
Critical/Serious categories onto the new scheme. Rejected for this component:
the mapping is not one-to-one, it is not published in the dataset, and asserting
one would fabricate labels for 52% of the data. Defining a *separate*
Critical/Serious target for that era is legitimate future work, and would
produce a different target with its own definition version.

**Treat the 90-day grace period as a reason to exclude adoption-era positives.**
The violations were real and were classified as Priority; the grace period
affected whether a citation was issued, not whether the violation existed. The
narrative-exclusion rule already removes the boilerplate sentence itself from
classification (ADR 0008), which is the correct scope.

## Consequences

- **The usable history is 2018-07-01 onward**: 57,727 eligible rows across 15,144
  establishments. Component 4 and Component 5 should plan around roughly eight
  years of labelled data, not sixteen.
- **Pre-2018 inspections remain useful as features.** Nothing prevents Component 4
  from computing a 2015 inspection into an establishment's history for a 2022
  target row. The era boundary constrains what can be *labelled*, not what can be
  *known*. This is worth stating because it is the opposite of the usual
  assumption about excluded data.
- **A validation check enforces the boundary** (`eligible_rows_are_in_the_code_era`)
  and fails the build if any eligible row predates it, so an off-by-one cannot
  silently admit a month of undefined labels. A regression test pins the boundary
  at 2018-07-01 itself.
- **`code_era_phase` is a legitimate stratification variable and an illegitimate
  feature.** It describes the regulatory regime a row was labelled under. Using
  it to hold out or reweight is correct; feeding it to a model as a predictor of
  establishment risk is not.
- **If Chicago changes its code again**, the same measurement should be repeated
  and a new boundary and definition version introduced rather than silently
  extending this one.
