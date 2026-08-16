# ADR 0008 — Predict Priority violations at a routine canvass

**Status:** Accepted · **Date:** 2026-08-16

## Context

Sentinel prioritises food inspections. To do that it needs a prediction target:
a precise statement of what a model is asked to forecast. Everything downstream
— features, evaluation, calibration, policy, scheduling — inherits whatever this
decision gets right or wrong, and a target that is subtly mis-specified cannot be
rescued by a better model.

The obvious candidate is `results == 'Fail'`. An empirical investigation of all
314,245 rows (`docs/analysis/target_construction_findings.md`) shows it is wrong,
and several other assumptions with it:

- **`results` has seven values, not the four documented.** `No Entry` (14,045),
  `Not Ready` (4,557) and `Business Not Located` (95) were undocumented.
- **`Pass w/ Conditions` is not a pass.** Among routine canvasses, priority
  violations are present in 99.4% of `Fail`, **97.9% of `Pass w/ Conditions`**
  and 0.5% of `Pass`. It behaves like a failure.
- **`Out of Business` is not an outcome.** 99.9% have no violation text, because
  no inspection took place. It is also not terminal: 24.9% are followed by
  another inspection at the same premises, median 273 days later.
- **The violation number does not encode severity.** Item 10 is 42.2% Priority
  Foundation, 11.3% Priority and 46.5% unlabelled — the same numbered item covers
  a hand sink with no hot water and a missing hand-washing sign.
- **Requiring a municipal citation code would be wrong.** 21,281 entries carry a
  Priority marker with no `7-38-xxx` code and are genuine.

## Decision

> **One target row = one (establishment, date) on which at least one eligible
> routine canvass occurred. The target is 1 if that day's canvassing found at
> least one Priority or Priority Foundation violation.**

```
target(e, d) = 1  ⟺  ∃ i ∈ Canvasses(e, d) : ∃ v ∈ Violations(i) :
                          severity(v) ∈ {PRIORITY, PRIORITY_FOUNDATION}
```

Eligibility: date on or after 2018-07-01 (ADR 0009), inspection type normalizing
to `CANVASS`, and a result in `{Pass, Pass w/ Conditions, Fail}`.

Five supporting decisions, each measured rather than assumed:

**1. The label is read from the violation text, never from `results`.** Using the
result would make the target partly circular, since the result is itself a
summary of what the inspector found. `results` is consulted only to decide
whether a row is *labellable* — a `Fail` with no violation text is
self-contradictory and becomes `unknown_violations` rather than being guessed.

**2. `Pass w/ Conditions` is eligible and labelled like anything else.** In
practice 97.9% are positive, but that falls out of the rule instead of being
written into it.

**3. Non-inspection results are ineligible, not negative.** Labelling `Out of
Business` negative would teach a model that a closed establishment is a clean
one; since closure correlates with prior poor performance, that is exactly
backwards.

**4. Severity comes from marker text with narrative exclusions, not from the
violation number and not from a citation code.** Four enumerated narrative
patterns are excluded, span-based rather than entry-based. Measured effect: 74
entries, 10 inspection labels.

**5. Same-day canvasses collapse with OR.** "Inspect E on date D" is one
decision; 160 multi-canvass days disagreed, so the rule is load-bearing.

## Alternatives rejected

**`results == 'Fail'`.** Would label all 16,617 eligible `Pass w/ Conditions`
canvasses negative, 16,261 of which contain a Priority or Priority Foundation
violation. A systematic mislabelling of 28% of eligible rows, in the direction
that matters most.

**`results != 'Pass'`.** Sweeps in `Out of Business`, `No Entry` and `Not Ready`
— 12,091 canvasses where nobody inspected anything.

**Any violation at all.** 72% of violation entries carry no severity label and
are overwhelmingly minor; item 55 alone accounts for 78,983 code-era entries,
none of them priority. This predicts write-up volume, not food-safety risk.

**Count of priority violations (regression).** The count reflects inspector
write-up verbosity at least as much as establishment state. A binary presence
indicator is the robust reading. Revisitable as a `v2` definition.

**Predict the *next* canvass** (reference event → future target event). Rejected
on three grounds: it discards the most recent canvass of every establishment
(15,148 rows); the reference-to-outcome gap is a median of 377 days with an IQR
of 306–511, so features would be badly stale by the time the outcome occurred;
and it predicts an event on a date nobody chose. The chosen formulation is not
weaker temporally — the target event is at `inspection_date` and every feature
must come from strictly before it.

**Including `Canvass Re-Inspection`** (16,998 code-era rows). A re-inspection
exists only because an earlier inspection failed, so its outcome is conditioned
on a prior failure. Including it would inflate the base rate and change what the
model is learning. A deliberate scope restriction, not a claim they are
uninformative.

**Including `Complaint` inspections.** A complaint inspection is triggered by an
external signal that is not available at scheduling time and that a scheduler
does not choose. Mixing triggered and routine visits would blur what the model
is being asked.

**Using an LLM or fuzzy matching to classify violations.** Non-deterministic,
unauditable and impossible to regression-test. Every classification decision here
is a regular expression a reviewer can argue with.

## Consequences

- **The target is auditable end to end.** Every positive carries the exact text
  span that produced it, enforced by a check that fails the build otherwise.
- **18.6% of the dataset is labelled** (57,727 of 314,245). The rest is
  explicitly categorised — `ineligible_era` 172,879, `ineligible_type` 70,848,
  `ineligible_result` 12,091, `unknown_violations` 79 — so nothing is silently
  dropped.
- **The positive rate is 52.5%**, close to balanced overall but ranging from
  87.6% to 39.1% by year. No resampling, reweighting or threshold tuning was
  applied, and the definition was not adjusted to change the balance. Imbalance
  is a modelling concern for later components.
- **Component 4 inherits a hard boundary.** `inspection_date` is the as-of line;
  the outcome columns are enumerated in the contract and in
  `TARGET_EVENT_COLUMNS` as forbidden inputs.
- **Excluding re-inspections and complaints means the model is trained only on
  routine visits.** If it is ever scored on a complaint-driven visit, that is
  out-of-distribution use and should be treated as such.
- **A false negative is possible when an inspector finds a priority violation and
  does not label it.** There is no ground truth in the open data to measure this
  against. **NOT VERIFIED.**
