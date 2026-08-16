# Target construction: empirical findings

Component 3 investigation. Every number was measured against one snapshot with
`scripts/profile_target.py` joined to Component 2's assignments. Nothing here is
assumed or carried over from documentation — and in two places the existing
documentation turned out to be wrong.

**Read this before changing `src/sentinel/target/`.** The target definition is a
regulatory statement, not a convenience, and each rule below exists because of a
specific measurement.

---

## 1. The snapshot these measurements describe

| Property | Value |
|---|---|
| Raw file | `food_inspections_20260816T070911Z.parquet` |
| Raw sha256 | `7d3c4069340a68d197204c6cca9fca6399c6565bc3668760f145f43cd377ad38` |
| Rows | 314,245 |
| Assignments | `establishment_assignments_20260816T085729Z.parquet` (Component 2) |
| Establishments | 35,859 |
| Date range | 2010-01-04 → 2026-08-14 |
| Profiles | 31, running in 10.5 s |

Join coverage is exact: 314,245 raw rows, 314,245 assignment rows, **314,245
joined, 0 unjoined**. Component 2's contract holds, and identity is never
re-derived here.

---

## 2. `results` — the documented value set is wrong

```
Pass                  162,607   51.745%
Fail                   60,513   19.257%
Pass w/ Conditions     46,661   14.849%
Out of Business        25,767    8.200%
No Entry               14,045    4.469%
Not Ready               4,557    1.450%
Business Not Located       95    0.030%
```

**Seven values, not four.** `docs/data_contracts/food_inspections_raw.md`
documents only the first four; `No Entry`, `Not Ready` and `Business Not Located`
— 18,697 rows, 5.9% of the dataset — are undocumented. That contract is
corrected as part of this component.

The column is otherwise clean: 0 nulls, 0 blanks, 0 untrimmed values, and 7
distinct values whether or not you uppercase and trim. No normalization is
needed, which is worth stating because it is unusual in this dataset.

The last four values share a property that decides how they are treated: they
describe an inspection that **did not happen**. §9 measures it.

---

## 3. `inspection_type` — messy, with a clean core

111 distinct raw values, 105 after uppercasing and trimming, 1 blank. The head:

| type | n | pct |
|---|---|---|
| Canvass | 162,397 | 51.678 |
| License | 42,138 | 13.409 |
| Canvass Re-Inspection | 35,022 | 11.145 |
| Complaint | 29,756 | 9.469 |
| License Re-Inspection | 12,774 | 4.065 |
| Complaint Re-Inspection | 12,396 | 3.945 |
| Short Form Complaint | 9,252 | 2.944 |
| Non-Inspection | 5,617 | 1.787 |
| Suspected Food Poisoning | 1,014 | 0.323 |
| Consultation | 679 | 0.216 |

The tail is free text: `CANVASS`, `CANVAS`, `TWO PEOPLE ATE AND GOT SICK.`,
`OWNER SUSPENDED OPERATION/LICENSE`, `Duplicated`, `task force liquor
inspection 1474`. Most occur once.

These are genuinely different events. A `Complaint` inspection happens *because*
somebody complained; a `License` inspection happens because somebody applied for
a licence; a `Canvass Re-Inspection` happens because a previous inspection
failed. Only `Canvass` is a routine, unconditional visit — which is what makes
it the right basis for a scheduling target (§13).

---

## 4. What a canvass actually looks like

In the code era (§5), the canvass family is exactly two values:

| type | n |
|---|---|
| Canvass | 70,518 |
| Canvass Re-Inspection | 16,998 |

Checking the recognition rule directly:

| rule | n |
|---|---|
| `inspection_type = 'Canvass'` | 70,518 |
| `upper(trim(inspection_type)) = 'CANVASS'` | 70,518 |
| `= 'Canvass Re-Inspection'` | 16,998 |
| other `%CANVAS%` variants | 0 |

> **Decision.** Canvass recognition is uppercase + whitespace-collapse, then
> exact match on `CANVASS`. In this snapshot that is identical to the literal
> `inspection_type == 'Canvass'`, but normalizing first costs nothing and
> protects against the casing variants that demonstrably exist elsewhere in the
> column (`CANVASS`, `CANVAS` both appear, though outside the code era).

The variants `CANVASS/SPECIAL EVENT`, `CANVASS SCHOOL/SPECIAL EVENT` and
`CANVASS SPECIAL EVENTS` exist in the dataset but **not** in the code era, so
they are excluded by the rule without needing a special case.

---

## 5. The regulatory cutover — the single most important finding

Chicago replaced its food-code violation scheme partway through the data. The
words "Priority" and "Priority Foundation" do not appear anywhere before it; the
words "Critical violation" and "Serious violation" do not appear after.

| month (2018) | rows | new terminology | old terminology |
|---|---|---|---|
| 2018-01 | 1,582 | 0 | 450 |
| 2018-02 | 1,369 | 0 | 342 |
| 2018-03 | 1,561 | 0 | 400 |
| 2018-04 | 1,690 | 0 | 443 |
| 2018-05 | 1,604 | 0 | 431 |
| 2018-06 | 1,589 | 0 | 415 |
| **2018-07** | **1,207** | **761** | **0** |
| 2018-08 | 1,423 | 1,030 | 0 |
| 2018-09 | 1,361 | 957 | 0 |
| 2018-10 | 1,493 | 968 | 3 |
| 2018-11 | 1,295 | 817 | 0 |
| 2018-12 | 1,018 | 636 | 0 |

By year, the same break:

| year | rows | "Priority" | "Priority Foundation" | old terms |
|---|---|---|---|---|
| 2010–2017 | 163,484 | **0** | **0** | 762–4,153/yr |
| 2018 | 17,192 | 5,169 | 5,074 | 1,919 |
| 2019 | 19,052 | 11,163 | 10,853 | 1 |
| 2020–2026 | 114,517 | ~5–7k/yr | ~5–6k/yr | 0–3 |

**The cutover is 2018-07-01, and it is clean** — a single month with no overlap.

> **Decision.** A Priority/Priority Foundation target is **undefined before
> 2018-07-01**. Eligibility starts there. This is not a modelling convenience: the
> concept being predicted did not exist in the data before that date.
>
> The 2010 – 2018 H1 period (163,484 rows, 52% of the dataset) cannot support
> this target. It is **not deleted** — those rows appear in the output with
> `target_status = 'ineligible_era'`, so the exclusion is visible and countable
> rather than silent. A different target could be defined for that era (the old
> Critical/Serious scheme); doing so is out of scope and is recorded as future
> work.

---

## 6. The `violations` format

```
rows                 314,245
violations NULL       88,481  (28.16%)
violations blank           0
average length         1,554 chars
maximum length         9,599 chars
```

Structure, measured over 497,208 entries in the code era:

| property | count |
|---|---|
| entries | 497,208 |
| start with `NN.` | **497,208 (100%)** |
| contain `- Comments:` | 496,110 (99.78%) |
| unnumbered | 0 |
| empty after trim | 0 |

Entries per inspection: mean 5.01, p50 4, p95 13, max 40, min 1.

So the format is far more regular than "free text" suggests:

```
NN. UPPERCASE VIOLATION TITLE - Comments: free text observations   |   NN. ...
```

Splitting on `|` is safe: every resulting entry is numbered and non-empty. The
1,098 entries without `- Comments:` are titles with no observation attached.

---

## 7. How Priority and Priority Foundation are actually encoded

### 7.1 The violation number does *not* encode the class

The obvious hypothesis — that the item number determines severity, as in a
standard code — is **false**. Association between number and marker, code era:

| item | entries | % Priority Foundation | % Priority | % unlabelled |
|---|---|---|---|---|
| 3 | 13,910 | 97.5 | 0.5 | 2.0 |
| 5 | 14,804 | 96.7 | 0.7 | 2.6 |
| **10** | 26,670 | **42.2** | **11.3** | **46.5** |
| **16** | 8,652 | **23.0** | **29.9** | **47.1** |
| 22 | 6,169 | 1.4 | 96.8 | 1.8 |
| **38** | 21,895 | **45.5** | **2.2** | **52.3** |
| 47 | 36,648 | 0.1 | 0.1 | 99.9 |
| 55 | 78,983 | 0.0 | 0.0 | 100.0 |

Some items are near-pure, but several of the highest-volume items are genuinely
mixed. Reading them explains why — the same item covers observations of
different severity:

- Item 10, **Priority**: *"OBSERVED NO HOT RUNNING WATER AT GRILL AREA
  HANDWASHING SINK. TEMPERATURE OF WATER WAS 72.5F…"*
- Item 10, **unlabelled**: *"OBSERVED NO HAND WASHING SIGN AT REAR BAR PREP
  AREA MUST PROVIDE AND MAINTAIN."*

A missing hand-washing sign and a hand sink with no hot water are the same
numbered item and obviously not the same severity.

> **Decision.** The violation number is **not** used to classify severity. It is
> parsed and retained for audit, nothing more.

### 7.2 The classification is written into the comment text

Marker phrasings in the code era, by frequency:

| phrase | n |
|---|---|
| `PRIORITY FOUNDATION ` | 86,300 |
| `PRIORITY VIOLATION ` | 12,004 |
| `PRIORITY FOUNDATION` | 11,566 |
| `PRIORITY VIOLATION` | 5,665 |
| `PRIORITY ` | 3,787 |
| `PRIORITY` | 626 |
| `PRIORITY VIOLATION I` | 314 |
| `PRIORITY CITATION IS` | 219 |
| `PRIORITY VIOLATIONI ` | 53 |

Distribution across all 497,208 code-era entries:

| class | entries |
|---|---|
| contains `PRIORITY FOUNDATION` | 112,245 |
| contains `PRIORITY` but not `PRIORITY FOUNDATION` | 25,353 |
| contains `CORE` | 7,943 |
| contains neither | 358,171 (72.0%) |

**72% of entries carry no severity label at all**, and `CORE` is written only
7,943 times. So absence of a marker is *not* positive evidence of a Core
classification — it is absence of evidence.

> **Decision.** The parser emits `PRIORITY_FOUNDATION`, `PRIORITY`, or
> `UNCLASSIFIED`. It deliberately does **not** emit "Core". Calling 358,171
> unlabelled entries "Core" would assert something the data does not say.
> For the target this does not matter — the label asks whether a Priority or
> Priority Foundation violation is *present* — but the naming keeps the
> uncertainty visible.

### 7.3 A municipal code must **not** be required

Requiring the marker to sit next to a `7-38-xxx` municipal code looks like a way
to reject narrative text. It is wrong:

| entries containing `PRIORITY` | 137,598 |
|---|---|
| with a `7-38` code | 116,317 |
| **without any code** | **21,281** |
| mentioning a citation | 100,569 |
| explicitly saying `NO CITATION` | 43,093 |

Reading the 21,281 shows they are genuine:

> *"1. PERSON IN CHARGE PRESENT… — Comments: NO PERSON IN CHARGE PRESENT AT THE
> TIME OF INSPECTION. **PRIORITY FOUNDATION VIOLATION. NO CITATION ISSUED.**"*

A violation that was found but not cited is still a violation. Requiring a code
would create **~21,281 false negatives to avoid a few hundred false positives** —
a bad trade by two orders of magnitude.

### 7.4 Narrative text that must be excluded

Naive substring matching does produce real false positives, in three shapes:

**Boilerplate policy text** — the violation is real but the only "priority"
mention is a standard notice:

> *"58. ALLERGEN TRAINING AS REQUIRED — Comments: Observed food allergen
> requirements not met. Instructed manager to provided. **A 90 day grace period
> was given for all new priority and priority foundation violations.** Citations
> will be issued on the next inspection…"*

**Forward-looking warnings** — a citation that has not happened:

> *"…INSTRUCTED MANAGER HE MUST PROVIDE OR **CITATION PRIORITY FOUNDATION WILL
> BE ISSUED** #7-38…"*

**Explicit negation** — the inspector saying it is *not* priority:

> *"…INSTRUCTED TO REPAIR OR REPLACE AND MAINTAIN. **NO PRIORITY FOUNDATION
> VIOLATION** 7-38-030(c)"*

Counts among the 137,598 entries containing `PRIORITY`:

| pattern | entries |
|---|---|
| `GRACE PERIOD` | 208 |
| `WILL BE ISSUED` | 146 |
| `IF NOT CORRECTED` | 3 |
| `NO PRIORITY` | 2 |
| `MAY BE ISSUED` / `COULD RESULT` | 0 |

**Exclusion must be span-based, not entry-based.** This entry is a genuine
Priority Foundation violation that also contains `IF NOT CORRECTED`:

> *"…**PRIORITY FOUNDATION VIOLATION#: 7-38-030(C). NO CITATION ISSUED.** WITH…
> A CITATION WILL BE ISSUED IF NOT CORRECTED…"*

Dropping the whole entry would lose a real positive. Dropping only the
offending clause leaves the citation intact.

> **Decision.** Split each entry's text into sentence-like chunks on `.` and `;`,
> discard chunks matching the narrative patterns
> (`\d+\s*-?\s*DAY GRACE PERIOD`, `GRACE PERIOD`, `WILL BE ISSUED`,
> `\bNO PRIORITY`), then look for markers in what remains. `PRIORITY FOUNDATION`
> is matched before bare `PRIORITY`, since it contains it.

**Measured effect of the exclusion** (this is the number a reviewer should
challenge):

| level | naive | narrative-excluded | difference |
|---|---|---|---|
| entries with a marker | 137,598 | 137,524 | **74 (0.054%)** |
| eligible canvass inspections labelled positive | 30,498 | 30,488 | **10 (0.017%)** |

The 10 flipped inspections are 8 `Pass w/ Conditions`, 1 `Pass`, 1 `Fail`. The
8 `Pass w/ Conditions` flips are the uncomfortable ones: the result suggests
conditions were imposed, yet the only priority mention was boilerplate. They are
left as negative because the parser deliberately does not consult `results` —
using the result to infer the violation class would make the target partly
circular (§8). Recorded as a known limitation (§18).

---

## 8. `results` is correlated with the target but is not the target

Code-era inspections, all types:

| results | n | violations null | with Priority | % |
|---|---|---|---|---|
| Pass | 60,373 | 17,685 | 327 | **0.5** |
| Pass w/ Conditions | 26,034 | 678 | 24,722 | **95.0** |
| Fail | 25,601 | 654 | 24,657 | **96.3** |
| Out of Business | 9,913 | 9,895 | 8 | 0.1 |
| No Entry | 8,531 | 8,171 | 272 | 3.2 |
| Not Ready | 3,088 | 3,044 | 29 | 0.9 |
| Business Not Located | 29 | 29 | 0 | 0.0 |

Restricted to plain canvasses, which is what the target uses:

| results | n | violations null | with Priority | % |
|---|---|---|---|---|
| Pass | 27,539 | 5,185 | 124 | 0.5 |
| Pass w/ Conditions | 16,790 | 49 | 16,387 | 97.6 |
| Fail | 14,098 | 30 | 13,987 | 99.2 |
| Out of Business | 10,389 | 10,378 | 5 | 0.0 |
| No Entry | 1,653 | 1,634 | 5 | 0.3 |
| Not Ready | 31 | 31 | 0 | 0.0 |
| Business Not Located | 18 | 18 | 0 | 0.0 |

> **This table is the argument for the whole component.** A
> `results == 'Fail'` target would label all 16,790 `Pass w/ Conditions`
> canvasses negative, and 16,387 of them (97.6%) contain a Priority or Priority
> Foundation violation. That is not a rounding error — it is a systematic
> mislabelling of 28% of the eligible rows in the direction that matters most.

### 8.1 `Pass w/ Conditions` — the central decision, measured

`Pass w/ Conditions` behaves like `Fail`, not like `Pass`: 97.6% vs 99.2% vs
0.5% priority presence among canvasses. Operationally it means priority
violations were found and either corrected on site or made subject to
conditions. The establishment was not clean.

> **Decision.** `Pass w/ Conditions` is **eligible**, and its label is decided by
> the violation text exactly like every other eligible result. In practice
> almost all of these rows are positive, but that is an outcome of the rule, not
> a special case in it. The target never reads `results`.

### 8.2 The 124 `Pass` canvasses with a priority marker

These are exactly the cases §7.4 is about. Inspecting them:

- *"…A 90 DAY GRACE PERIOD WAS GIVEN FOR ALL NEW PRIORITY AND PRIORITY
  FOUNDATION VIOLA…"* → boilerplate, excluded
- *"…INSTRUCTED MANAGER HE MUST PROVIDE OR CITATION PRIORITY FOUNDATION WILL BE
  ISSUED #7-…"* → forward-looking, excluded
- *"…COLD RUNNING WATER SOAP AND PAPER TOWELS. PRIORITY FOUNDATION 7-38-030 (C)
  NO CI…"* → **genuine**, kept positive

A `Pass` result with a real uncited Priority Foundation violation is a true
positive under this target, and it stays positive. The target measures what the
inspector found, not what the result summary says.

---

## 9. `Out of Business` and the other non-inspections

Two measurements decide this.

**They are not inspections.** Violation text is essentially always absent:

| results | n | violations null | % null |
|---|---|---|---|
| Out of Business | 25,767 | 25,733 | 99.9 |
| No Entry | 14,045 | 13,690 | 97.5 |
| Not Ready | 4,557 | 4,528 | 99.4 |
| Business Not Located | 95 | 95 | 100.0 |

**Out of business is not terminal for the premises.** Following each of the
25,767 OOB records forward within its Component 2 establishment:

```
OOB records                       25,767
followed by a later inspection     6,425  (24.9%)
median gap                           273 days
```

A quarter of "out of business" records are followed by another inspection at the
same physical premises — precisely what Component 2's physical-premises
definition predicts when a new tenant moves in (ADR 0006, findings §11.1). OOB
is a statement about a *business*, and Component 2 deliberately tracks *places*.

> **Decision.** `Out of Business`, `No Entry`, `Not Ready` and
> `Business Not Located` are **ineligible**, not negative. No inspection
> occurred, so there is no outcome to label.
>
> Labelling them negative would be actively harmful: it would teach a model that
> a closed or inaccessible establishment is a *clean* establishment, and since
> closure correlates with prior poor performance, that is exactly backwards.
>
> They are **not** treated as a terminal state either. An establishment with an
> OOB record continues to be eligible for later canvasses, because 24.9% of the
> time it genuinely is inspected again.

---

## 10. Missing violation text does not mean the same thing everywhere

Among code-era plain canvasses:

| results | n | violations null | % null |
|---|---|---|---|
| Pass | 27,539 | 5,185 | **18.8** |
| Pass w/ Conditions | 16,790 | 49 | 0.3 |
| Fail | 14,098 | 30 | 0.2 |
| Out of Business | 10,389 | 10,378 | 99.9 |

Three different meanings, and the result column distinguishes them:

1. **`Pass` + null (5,185)** — an inspection happened and nothing was written
   down. For a passing inspection that is the expected encoding of "no
   violations found". → **true zero, negative.**
2. **`Pass w/ Conditions` + null (49) and `Fail` + null (30)** — self-
   contradictory. The result asserts violations were found; the text records
   none. Nothing can be concluded. → **unknown, excluded from the label.**
3. **Non-inspection results + null** — already ineligible (§9).

> **Decision.** Missingness is interpreted by result, not uniformly. The 79
> contradictory rows get `target_status = 'unknown_violations'` and a null
> target rather than being guessed in either direction. That is 0.14% of
> otherwise-eligible rows.
>
> Note this is the one place the target construction *does* read `results` — not
> to decide the label, but to decide whether the row is labellable at all.

---

## 11. Multiple inspections on the same establishment and date

Code-era, all inspection types:

| inspections on one establishment-date | establishment-dates |
|---|---|
| 1 | 127,238 |
| 2 | 5,118 |
| 3 | 860 |
| 4 | 199 |
| 5–16 | 85 |

Restricted to eligible canvasses:

| eligible canvasses on one establishment-date | establishment-dates |
|---|---|
| 1 | 67,733 |
| 2 | 1,180 |
| 3 | 96 |
| 4 | 26 |
| 5 | 5 |
| 8 | 1 |

582 of these multi-canvass days are in the eligible set, and they do **not**
always agree:

| | count |
|---|---|
| multi-canvass eligible days | 582 |
| all contributing canvasses negative | 289 |
| all positive | 133 |
| **disagreeing** | **160** |

160 days where the choice of rule actually changes the label.

> **Decision.** One target row per **(establishment_id, inspection_date)**, with
> `target = OR` over that day's eligible canvasses.
>
> The reasoning is the decision problem: "inspect establishment E on date D" is a
> single scheduling decision. Emitting two rows would give them an identical
> as-of boundary and near-identical features with possibly opposite labels —
> irreducible noise injected by construction. And the risk question, "would an
> inspector visiting that day find a priority violation?", is answered yes if any
> of the day's canvasses found one.
>
> Provenance is preserved: `target_inspection_id` points at the positive
> contributing inspection when there is one (else the lowest id), and
> `n_contributing_inspections` plus `contributing_inspection_ids` record the
> rest.

---

## 12. Inspection sequences

Among eligible canvasses, consecutive pairs within an establishment:

| statistic | days |
|---|---|
| pairs | 43,279 |
| p25 | 306 |
| p50 | **377** |
| p75 | 511 |
| max | 2,697 |

The median gap is roughly annual, consistent with a risk-based canvass schedule.

This analysis exists **only** to inform the target-event definition, and it
produced a design consequence: a "predict the *next* canvass" formulation would
attach a label to an event a median of 377 days after the reference point, with
an interquartile range of more than six months. Features computed at the
reference date would be badly stale by the time the outcome occurred. §13
records the resulting choice.

**No feature was derived from these sequences.** Inter-inspection gaps, counts
and prior outcomes are Component 4's work.

---

## 13. The prediction unit

Two formulations were considered.

**A — one row per eligible canvass.** The canvass *is* the prediction event.
Reference time is the instant before it, which is precisely when a scheduler
would decide whether to send an inspector.

**B — reference canvass → next eligible canvass.** Each canvass is a reference
point and the label comes from the establishment's *next* canvass.

| | A | B |
|---|---|---|
| rows | 58,427 | ~43,279 |
| establishments with no target row | 0 | 15,148 (their last canvass) |
| gap between reference and outcome | none | median 377 d, IQR 306–511 |
| maps to a real decision | yes — "inspect E now?" | no — the outcome date is arbitrary |
| auditable to one inspection | yes | requires two |

> **Decision. One target row = one (establishment, date) on which at least one
> eligible canvass occurred.**
>
> B was rejected because it discards the most recent canvass of every
> establishment, because the reference-to-outcome gap is long and highly
> variable, and because it predicts an event on a date nobody chose. A is the
> standard formulation for inspection prioritisation and matches Sentinel's
> actual decision.
>
> **This does not weaken the temporal guarantee.** The target event is at time
> `inspection_date`; every feature must be built from information strictly
> before it. Component 3 publishes that boundary explicitly and Component 4 is
> contractually bound to it (§16 of the data contract).

---

## 14. Eligibility funnel

| stage | rows | retained |
|---|---|---|
| all rows | 314,245 | — |
| `inspection_date >= 2018-07-01` | 141,366 | 45.0% |
| ∧ `inspection_type` normalizes to `CANVASS` | 70,518 | 22.4% |
| ∧ `results ∈ {Pass, Pass w/ Conditions, Fail}` | 58,427 | 18.6% |
| ∧ violation text interpretable (§10) | 58,348 | 18.6% |

18.6% of the dataset supports this target. That is not a defect — it is the
honest consequence of a regulatory scheme that changed in 2018 and of the fact
that most inspection records are not routine canvasses.

---

## 15. Temporal drift

Positive rate among eligible canvasses (naive marker matching, before the
narrative exclusion):

| year | eligible | positive | % |
|---|---|---|---|
| 2018 (H2 only) | 2,867 | 2,506 | **87.4** |
| 2019 | 8,353 | 6,419 | 76.8 |
| 2020 | 6,168 | 3,648 | 59.1 |
| 2021 | 6,239 | 3,112 | 49.9 |
| 2022 | 6,644 | 3,072 | 46.2 |
| 2023 | 7,280 | 3,332 | 45.8 |
| 2024 | 8,592 | 3,627 | 42.2 |
| 2025 | 8,367 | 3,260 | 39.0 |
| 2026 (partial) | 3,917 | 1,522 | 38.9 |

**The base rate more than halves over the period**, from 87.4% to 38.9%.

Two plausible drivers, neither of which this component can separate:
enforcement ramp-up after the July 2018 code adoption (inspectors newly applying
an unfamiliar classification, plus the 90-day grace period visible in the
boilerplate of §7.4), and pandemic-era disruption from 2020.

The 2018 H2 figure is the clear outlier and covers only six months.

> **Decision.** Eligibility still starts 2018-07-01 — nothing is silently
> discarded — but each row carries `code_era_phase`:
> `adoption` for 2018-07-01 → 2018-12-31 (2,867 rows) and `stable` for 2019-01-01
> onward (55,560 rows). Component 5 can hold the adoption period out; Component 3
> does not make that call.
>
> This drift is the most important thing Component 5 inherits. Any evaluation
> that shuffles rows across time will be measuring the drift rather than the
> model.

---

## 16. Class balance

The positive rate over the full eligible set is **52.2%** (30,498 of 58,427,
naive; 30,488 narrative-excluded). The target is close to balanced overall,
though it ranges from 87% to 39% by year.

No resampling, reweighting, threshold tuning or redefinition was applied, and
none should be. The definition follows the regulatory question; class balance is
a modelling and evaluation concern for later components.

---

## 17. Candidate target definitions considered

| candidate | verdict |
|---|---|
| `results == 'Fail'` | **Rejected.** Mislabels 16,387 canvasses where priority violations were found but the result was `Pass w/ Conditions` (§8). |
| `results != 'Pass'` | **Rejected.** Sweeps in `Out of Business`, `No Entry` and `Not Ready`, which are not inspections at all (§9). |
| any violation at all | **Rejected.** 72% of entries are unlabelled and mostly minor; item 55 alone has 78,983 entries, none priority. This predicts paperwork, not risk. |
| count of priority violations (regression) | **Rejected for v1.** The count depends on inspector write-up verbosity more than on establishment state. A binary presence indicator is the robust reading. Recorded as possible future work. |
| next-canvass formulation | **Rejected.** §13. |
| **≥1 Priority or Priority Foundation violation at an eligible canvass** | **Chosen.** |

---

## 18. Remaining limitations

1. **The pre-2018 era cannot support this target.** 163,484 rows (52%) are
   `ineligible_era`. A separate Critical/Serious target could be defined for
   them; not attempted.
2. **The narrative exclusion list is judgement.** Four patterns, derived from
   reading examples, affecting 74 entries and flipping 10 inspection labels. Every
   pattern is enumerated in the data contract so a reviewer can disagree with a
   specific rule rather than with a black box.
3. **8 `Pass w/ Conditions` rows are labelled negative** where the result implies
   otherwise (§7.4). Accepted so that the parser stays independent of `results`.
4. **Inspector write-up variation is unmeasurable here.** If an inspector finds a
   priority violation and does not write the label, the row is a false negative.
   43,093 entries say `NO CITATION` while still labelling the class, which
   suggests labelling discipline is good, but there is no ground truth to verify
   against. **NOT VERIFIED** — it would need an independent audit of inspection
   records, which the open data does not contain.
5. **Severity within positive is not represented.** One priority violation and
   twelve produce the same label.
6. **`Canvass Re-Inspection` is excluded** (16,998 code-era rows). Those exist
   only because an earlier inspection failed, so including them would condition
   the target on past failure and inflate the base rate. This is a deliberate
   restriction of scope, not a claim that re-inspections are uninformative.
7. **The 2018 H2 base rate is anomalous** (87.4%) and flagged rather than fixed.

---

<!-- §19 is appended after the target has been built against this snapshot. -->
