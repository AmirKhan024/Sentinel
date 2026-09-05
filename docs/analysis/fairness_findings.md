# Component 12 — fairness and geographic equity findings

Measured on the 2026-08-25 production run: `sentinel audit-fairness --report`, calibrated
predictions `calibrated_predictions_20260824T163608Z.parquet` (207,680 rows, 5 models × 18
folds), feature table `as_of_features_20260816T150313Z.parquet` (57,727 rows), categoricals
`neural_categoricals_20260818T125631Z.parquet`, attributions
`explanation_values_20260825T145444Z.parquet`.

**Every number here is measured.** Where a claim could not be measured it is not made.

**A green validation run means the audit is internally sound. It does not mean Sentinel is
fair.** See §12 and ADR 0035 for what this component does not establish.

---

## 0. The run

| | |
| --- | --- |
| command | `uv run sentinel audit-fairness --report` |
| runtime | **162.2 s** |
| models audited | 5 (4 primary + 1 experimental) |
| group definitions | 2 audited, 5 refused |
| folds | 18 (17 quarterly + 1 `covid_shift`) |
| audited rows | **207,680** |
| groups observed | 286 |
| groups supported (ranking floor) | 132 |
| insufficient support | 154, each recorded with counts and a reason |
| metric rows | 122,850 (64,260 null for insufficient support) |
| priority rows | 136,850 |
| error-severity checks | **13, all pass** |
| advisory findings | 13 |
| inputs unchanged (sha256 before vs after) | **true** |
| refits, re-executions, bit-identity gates | **none — this component re-executes nothing** |
| determinism | **10 of 10 tables byte-identical** across two full production runs |

---

## 1. What group data exists (profile `group_source_inventory`)

The raw snapshot publishes ten geographic columns. **None of them is a model feature.**

| raw column | concept | distinct | nulls | a model feature? |
| --- | --- | ---: | ---: | --- |
| `zip` | postal code as typed | 140 | 42 | **no** |
| `:@computed_region_vrxf_vc4k` | community area | 78 | 1,365 | **no** |
| `:@computed_region_6mkv_f3dw` | ZIP-code region | 66 | 1,042 | **no** |
| `:@computed_region_43wa_7qmu` | ward, current | 51 | 1,365 | **no** |
| `:@computed_region_awaf_s7ux` | ward, 2003–2015 | 51 | 1,365 | **no** |
| `:@computed_region_bdys_3d7i` | census tract | 797 | 1,089 | **no** |
| `latitude` / `longitude` | point geography | 18,931 | 1,042 | **no** |
| `address` | street address | 33,261 | 0 | **no** |
| `city` | municipality as typed | 95 | 181 | **no** |
| `state` | state as typed | 8 | 69 | **no** |

Component 4's feature table has 33 columns, of which 26 are numeric inspection history, and
not one is geographic.

### What this says

**This is safety by absence, not by design, and it proves nothing about fairness.** A model
with no geographic input can still behave differently across geography, because the 26
features it does see are themselves distributed unevenly across the city — §7 measures exactly
that. Fairness through unawareness is not fairness, and it is the reason this component
measures behaviour rather than inspecting a feature list.

**No demographic variable exists anywhere in this project.** No race, income, ACS, census or
deprivation field is ingested. That is what makes this a geographic group audit rather than a
protected-class fairness certification, and no result below supports a statement about a
protected class.

---

## 2. Which geographies are admissible (profile `group_temporal_stability`)

The audited value comes from the establishment's most recent inspection of any type strictly
before the row's own date. The alternative was the value recorded on the row itself. Measured
before choosing:

| group definition | rows where both values exist | disagreements | rate |
| --- | ---: | ---: | ---: |
| `community_area` | 57,041 | **0** | 0.000000 |
| `zip` | 57,326 | **0** | 0.000000 |

### What this says

**The temporally safe choice costs nothing.** A community area is an attribute of a fixed
premises, so carrying the last observed value forward reproduces the contemporaneous one
exactly. Taking it means this component needs no exception to ADR 0010, introduces no new join
against raw, and inherits a frame Component 8 already validates strictly as-of per row
(minimum observed lag **1 day**, median 357).

### Ward fails the same test, and the dataset proves it

The two published ward layers assign different region ids to **56,451 of 57,403** rows —
**98.3%**. That the publisher ships a 2003–2015 vintage alongside a current one is the point: a
ward identifier is a property of a boundary version, not of a place, so attaching the current
ward to a 2019 row assigns it to a district that did not exist when it was inspected.

Chicago's 77 community areas have been fixed since the 1920s, which is exactly why the city
publishes statistics against them. **Ward is refused, and the refusal is a row in
`fairness_group_definitions` rather than a sentence in a document.** So are census tract (797
groups over 32,696 rows — nothing would clear a support floor), point geography, city/state
(312,957 of 314,245 rows say `CHICAGO`, and the 95 distinct values include `CCHICAGO`), and
facility type (available, and not geography).

---

## 3. Support: what could be measured at all (profile `group_support_population`)

**This is the profile that decided the shape of the whole component.**

| grain | cells | min | p25 | median | p75 | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| per (fold, community area) | 1,288 | 1 | 8 | **16** | 30 | 56 | 325 |
| per (fold, ZIP) | 1,024 | 1 | 13 | **27** | 45 | 66 | 184 |

| row floor | per-fold community-area cells | pooled community areas | pooled ZIPs |
| ---: | ---: | ---: | ---: |
| 100 | 57 / 1,288 | 69 / 78 | 58 / 69 |
| **200** | **4 / 1,288** | **51 / 78** | **56 / 69** |
| 300 | 1 / 1,288 | 33 / 78 | 41 / 69 |

Frozen floors (ADR 0034): `SUPPORT_MIN_ROWS = 200`, `SUPPORT_MIN_POSITIVE = 20`,
`SUPPORT_MIN_NEGATIVE = 20`, `CALIBRATION_MIN_ROWS = 300`.

Production run, pooled over the 17 quarterly test windows:

| definition | observed | ranking-supported | calibration-supported |
| --- | ---: | ---: | ---: |
| `community_area` | 78 | **51** | 33 |
| `zip` | 69 | **56** | 41 |

### What this says

**The per-fold grain cannot support a group metric, and the pooled grain can.** Four of 1,288
(fold, community area) cells clear the 200-row floor. An ROC-AUC over sixteen rows is noise
wearing four decimal places.

Pooling is legitimate and is *not* the leak ADR 0025 forbids: every pooled row is strictly
held out for its own fold. What it costs is stated on every pooled row — the 17 windows were
scored by 17 differently-fitted models, so **a pooled number describes the system as operated
over 2022Q2–2026Q2, not one estimator.**

**The calibration floor is arithmetic, not taste.** `evaluation.metrics.ece` uses 15
equal-mass bins; 20 rows per bin needs 300 rows. Ten bins would have let 18 more community
areas through and made every resulting ECE incomparable with Component 9's global figure —
which is the exact comparison §8 exists to make.

**27 of 78 community areas and 13 of 69 ZIPs are excluded from every comparison below.** They
are recorded with real counts and a stated reason, never dropped.

---

## 4. What the data looks like before any model (profile `group_outcome_rates`)

Pooled over the quarterly test windows, groups at or above 200 rows:

| definition | supported groups | outcome rate range | overall | spread |
| --- | ---: | --- | ---: | ---: |
| `community_area` | 51 | **0.2200 → 0.5658** | 0.4283 | 0.3457 |
| `zip` | 56 | **0.2350 → 0.6099** | 0.4283 | 0.3749 |

### What this says

**A thirty-point disparity exists in the outcomes themselves, before Sentinel is involved.**
Two consequences, and both govern how every later number is read.

First, **a difference in selection rate across these groups is the expected behaviour of a
working risk model, not evidence of a defect.** Equal selection would require ignoring a
measured difference in outcomes. This is why §9 reports selection and capture separately and
refuses to collapse either into a verdict.

Second — inherited from ADR 0019 rather than discovered here — **the outcome is that a
violation was *cited*, not that an establishment was unsafe.** The dataset publishes no
inspector identifier and Chicago assigns inspectors by district, so a neighbourhood-level
difference in citation rate cannot be decomposed into establishment risk versus differential
inspection practice. **That limitation bounds every number in this document.**

---

## 5. Representation drift (profile `representation_drift`)

Across the 17 quarterly folds, group shares travel by up to **10.7 percentage points**
(community area 37: 0.0461 → 0.1530), and **11 of 78 community areas** and **16 of 69 ZIPs**
are absent from at least one fold entirely.

### What this says

The evaluated population is not stationary. A group whose share halved and whose measured ECE
moved has two candidate explanations, and **this component can separate neither.** Every drift
claim in §11 is reported beside this table rather than on its own.

---

## 6. Ranking performance by group

ROC-AUC per community area, calibrated stage, pooled quarterly, 51 supported groups:

| model | min | max | spread |
| --- | --- | --- | ---: |
| `xgboost_chain_embeddings_platt` ⚠ | 0.5112 (`__UNKNOWN__`, 405 rows) | 0.7094 (ca 53, 457 rows) | **0.1982** |
| `xgboost_platt` | 0.5092 (`__UNKNOWN__`, 405) | 0.6971 (ca 53) | 0.1879 |
| `neural_numeric_only_platt` | 0.5322 (`__UNKNOWN__`, 405) | 0.7095 (ca 53) | 0.1773 |
| `lightgbm_platt` | 0.5178 (`__UNKNOWN__`, 405) | 0.6911 (ca 53) | 0.1733 |
| `logistic_regression_platt` | 0.5241 (ca 45, 225) | 0.6880 (ca 53) | 0.1640 |

⚠ experimental Component 8 derivative (ADR 0022) — must not become a headline.

### What this says

**The ranking works far better in some neighbourhoods than others, and the models agree about
which.** Community area 53 is the best-ranked group for all five; the same group is at the top
of every column.

**The worst-ranked group is `__UNKNOWN__` for four of five models, at ROC-AUC 0.509–0.532 —
statistically indistinguishable from random.** Component 5 measured business-as-usual at
ROC-AUC 0.5040 and called it indistinguishable from random within a quarter. On the 405 rows
with no recoverable geography, four of Sentinel's five models are at the same place. §10
explains why, and the explanation is not about geography.

Note also what the spread is measured against. Component 8's headline quarterly ROC-AUC is
0.6241 for `neural_numeric_only`; the **within-city spread of 0.177** is larger than the entire
difference between the best and worst *model* in this project.

---

## 7. Calibration: did the global improvement reach the groups?

Component 9 measured global quarterly ECE falling for every model — `xgboost` 0.0621 →
**0.0474**, a 24% cut, with the ranking bit-for-bit unchanged. **That is not what happened
inside the groups.**

Supported community areas (33) and ZIPs (41), pooled quarterly:

| model | community areas improved | mean ECE base → calibrated | ZIPs improved |
| --- | ---: | --- | ---: |
| `lightgbm_platt` | **25 / 33** | 0.0934 → 0.0828 | 33 / 41 |
| `xgboost_platt` | **25 / 33** | 0.0948 → 0.0844 | 27 / 41 |
| `logistic_regression_platt` | 23 / 33 | 0.0966 → 0.0854 | 27 / 41 |
| `xgboost_chain_embeddings_platt` ⚠ | 23 / 33 | 0.0929 → 0.0823 | 29 / 41 |
| `neural_numeric_only_platt` | **17 / 33** | 0.0884 → 0.0862 | 23 / 41 |

### What this says

**A global calibration improvement is not a group-level calibration improvement, and this is
the measurement that shows it.** For the best case, `xgboost` and `lightgbm`, calibration made
**8 of 33 community areas worse**. For `neural_numeric_only` it improved barely half — **16 of
33 got worse**, and the mean moved by 0.0022.

The neural result is coherent with Component 9's own finding rather than at odds with it.
Component 9 recorded that `neural_numeric_only` had the best *uncalibrated* ECE and therefore
the least to gain, and that its ECE ordering inverted after calibration. This is the same
effect resolved by neighbourhood: a two-parameter monotone map fitted on the whole calibration
window moves every group's probabilities in one direction, and for a model that was already
close, that direction is wrong for about half of them.

**Nothing was done about it.** Fitting a calibrator per community area would change Component
9, which is closed, and it is a substantive fairness decision disguised as a fix — it trades
overall calibration for group calibration, needs a per-group calibration window most groups do
not have the rows for, and makes the probability a function of the neighbourhood, which is
the thing ADR 0023 declined to let the *model* do. ADR 0034 records the refusal.

Group ECE levels are also roughly **double** the global figure — 0.082–0.094 against Component
9's 0.0474 for `xgboost`. A pooled ECE averages over neighbourhoods whose miscalibration
partly cancels.

---

## 8. Priority allocation and capture — the most important section

If Sentinel can prioritise only the top 5%, who is in it, and how much of each group's actual
risk does it find? `xgboost_platt`, calibrated, community area, pooled quarterly:

| model | overall capture | worst group | best group | spread |
| --- | ---: | --- | --- | ---: |
| `lightgbm_platt` | 0.0713 | **0.0060** (`__UNKNOWN__`) | 0.1551 (ca 35) | 0.1491 |
| `xgboost_platt` | 0.0701 | **0.0060** (`__UNKNOWN__`) | 0.1510 (ca 35) | 0.1450 |
| `xgboost_chain_embeddings_platt` ⚠ | 0.0698 | 0.0060 (`__UNKNOWN__`) | 0.1351 (ca 40) | 0.1291 |
| `neural_numeric_only_platt` | 0.0689 | 0.0227 (ca 75) | 0.1510 (ca 35) | 0.1283 |
| `logistic_regression_platt` | 0.0680 | 0.0123 (ca 69) | 0.1466 (ca 68) | 0.1342 |

`xgboost_platt`, the five worst-captured and three best-captured supported groups:

| community area | rows | positives | selection-rate ratio | capture at top 5% |
| --- | ---: | ---: | ---: | ---: |
| `__UNKNOWN__` | 405 | 166 | **0.20** | **0.0060** |
| 69 | 282 | 81 | 0.57 | 0.0123 |
| 16 | 506 | 177 | 0.55 | 0.0226 |
| 22 | 413 | 149 | 0.82 | 0.0268 |
| 39 | 292 | 103 | 0.62 | 0.0291 |
| … | | | | |
| 6 | 639 | 267 | 1.41 | 0.1236 |
| 40 | 259 | 111 | 1.85 | 0.1351 |
| 35 | 477 | 245 | **2.27** | **0.1510** |

### What this says

**Representation in the top k and effectiveness of the top k are different questions, and this
table is why they are never combined.** Community area 35 is selected at 2.27× the overall
rate *and* has the highest capture; `__UNKNOWN__` is selected at 0.20× *and* has the lowest.
Here the two move together, but nothing in the arithmetic requires that, which is why both are
reported.

**The `__UNKNOWN__` group is captured at 0.006 against an overall 0.070 — roughly one twelfth
of the average**, on 405 rows holding 166 real violations. Of those 166, the top 5% found one.

**Read this against §4 before reading it as unfairness.** Outcome rates differ from 0.220 to
0.566 across these groups, so a working risk model is *expected* to select at different rates.
What the capture column adds is that unequal selection here is not buying proportionally
unequal discovery: the groups the model deprioritises are also the groups whose violations it
finds least well when it does look.

**Every figure in this section is a descriptive threshold audit, not a deployment policy.** The
cutoffs are rank positions derived from real inspection capacity via
`simulate.capacity_k_values`. No probability threshold is introduced, and there is no flag to
add one. Component 13 owns decision policy.

---

## 9. Missingness, and the chain that closes on one row set

Component 11 measured `missing_no_code_era_canvass` ranking **third** in importance for the
logistic model and **second** for the network — the absence of a record is among the most
informative signals either has. Distribution across supported community areas, pooled
quarterly:

| indicator | overall | min group | max group | spread |
| --- | ---: | ---: | ---: | ---: |
| `missing_no_prior_inspection` | 0.0074 | 0.0000 | **0.5951** (`__UNKNOWN__`) | 0.5951 |
| `missing_no_code_era_canvass` | 0.1043 | 0.0388 | **0.6173** (`__UNKNOWN__`) | 0.5785 |
| `missing_no_inspected_canvass` | 0.0997 | 0.0388 | 0.6148 (`__UNKNOWN__`) | 0.5761 |
| `missing_no_prior_canvass` | 0.0957 | 0.0388 | 0.6148 (`__UNKNOWN__`) | 0.5761 |

### What this says

**The chain the brief asks about closes, and every link is measured:**

```text
group            __UNKNOWN__ community area, 405 quarterly test rows
   |
data availability   59.5% have no prior inspection of any type, against 0.74% overall -- 80x
   |
feature missingness 61.7% missing code-era canvass history, against 10.4% overall -- 6x
   |
model reliance      Component 11: that indicator ranks 2nd/3rd in importance for two models
   |
group behaviour     ROC-AUC 0.509-0.532 (random), selection ratio 0.20, capture 0.006 vs 0.070
```

The group with no recoverable geography **is** the group with no inspection history. That is
not a coincidence to be explained away: Component 8's as-of join can only carry a community
area forward from an *earlier* inspection, so an establishment with no earlier inspection has
no geography here for the same reason it has no history.

**And this is a measurement, not an accusation.** Three things it does not say:

* It does not say the models are wrong to lean on missingness. "We have never inspected this
  place" is a true and relevant fact, and removing the feature would not undo the inequality
  in inspection history behind it.
* It does not establish a causal direction. Whether sparse records make these establishments
  hard to rank, or whether whatever makes them hard to rank also makes them rarely inspected,
  is not answerable here.
* It does not generalise to the named neighbourhoods. `__UNKNOWN__` is not a place; it is the
  absence of one.

**The one operationally uncomfortable number**: of the `__UNKNOWN__` rows that *did* reach the
top 5%, **100%** were missing code-era canvass history. The model both deprioritises this
group overall and, within it, prioritises exactly the rows it knows least about.

---

## 10. Model comparison

The five models do **not** rank groups equally well, and the ordering is not the accuracy
ordering. `xgboost_chain_embeddings_platt` has the **widest** ROC-AUC spread across community
areas (0.1982) and `logistic_regression_platt` the **narrowest** (0.1640) — while Component 8
measured the two within 0.0118 NDE of each other overall. `neural_numeric_only_platt`, which
Component 8 measured as the best model in the project on NDE, is the model whose calibration
reached the **fewest** groups (17 of 33).

### What this says

**A model that is marginally more accurate overall is not automatically preferable**, and this
component was built to make that visible rather than to act on it. It is recorded as blocked
from model selection in every manifest it emits; MEMORY open question 13 is a policy
component's to settle, and this is one more input to it rather than an answer.

`xgboost_chain_embeddings_platt` is an experimental Component 8 derivative (ADR 0022) which
lost on NDE and which Component 11 could not explain at all (ADR 0031). **Its numbers here must
not become a headline in either direction.**

---

## 11. Temporal disparity drift

**Almost nothing can be said, and that is the measurement.** Per (model, group definition), the
number of quarterly folds in which a calibrated ROC-AUC or ECE disparity was computable at all
is **exactly one**. The support policy predicted it: 4 of 1,288 (fold, community area) cells
clear the 200-row floor.

Every such series is therefore labelled `insufficient_folds` rather than fitted.
`DRIFT_MIN_FOLDS = 3` was frozen before any series existed, and a trend through one point is
not a trend.

What *is* measurable is the population moving underneath: §5's share travel of up to 10.7
percentage points, reported as advisories on every run.

### What this says

The brief's §12 asks whether equity performance drifts over time. **This data cannot answer
it at the fold grain**, and reporting a line through one point would have turned a shortage of
data into a finding. A future component wanting this answer needs either a coarser geography
or a wider fold.

---

## 12. `covid_shift`, reported separately

| definition | fold set | observed | ranking-supported | calibration-supported |
| --- | --- | ---: | ---: | ---: |
| `community_area` | quarterly | 78 | 51 | 33 |
| `community_area` | **covid_shift** | 78 | **11** | **5** |
| `zip` | quarterly | 69 | 56 | 41 |
| `zip` | **covid_shift** | 61 | **14** | **3** |

### What this says

**The covid window holds more rows than any single quarterly fold (8,840) and supports far
fewer groups.** The inspection programme was suspended and restarted, and the establishments
inspected during it were not a cross-section of the city — 8 community areas disappear from
ZIP coverage entirely, and only 5 community areas clear the calibration floor.

It is reported as **a separate stress-test observation**. No trend is claimed from it, it is
never averaged into a quarterly mean, and a group disparity appearing only there is an
observation about one abnormal nineteen-month period. Five components have now measured this
fold diverging from the quarterly answer; a sixth divergence would be the expectation.

---

## 12a. The question ADR 0023 handed over, answered

Component 8 embedded community area as an explicitly audited experimental input, declared in
advance that a better score would **not** be grounds for keeping it, and handed Component 12
one question with three candidate answers:

> whether the improvement is (a) genuine geographic risk signal, (b) a proxy for the
> demographics of a neighbourhood, or (c) a proxy for *inspection practice* in a neighbourhood
> — the third being indistinguishable from the first two in this data, per ADR 0019.

**The answer is that the question is still not separable, and now it is separable-in-principle
rather than merely unasked.**

What can be said:

* Component 8 measured the embedding buying **nothing**: `neural_no_community_area` (NDE
  0.2258) beat the full embedding model (0.2215). So there is no improvement to attribute to
  (a), (b) or (c) in the first place. The non-retention rule cost nothing and still stands.
* Component 12 measured that geography is nonetheless a real surface on which the **system**
  behaves unevenly — a 0.164–0.198 ROC-AUC spread, a calibration improvement that reaches two
  thirds of neighbourhoods, and a 25× capture range — **without any model having a geographic
  input at all.** So the unevenness is carried by the 26 history features, not by a group label.
* **Distinguishing (a) from (c) remains impossible**, for exactly the reason ADR 0019 gave
  before either component existed: the target is that a violation was *cited*, and Chicago
  assigns inspectors by district. A neighbourhood where more violations are recorded may have
  riskier establishments or stricter inspection, and this project observes only the record.
* **(b) is not testable at all here**, because no demographic variable is ingested.

**Community area is therefore not promoted**, and the recommendation ADR 0023 carried forward
is unchanged. Promoting it would require a Component 4 release behind a bumped
`feature_definition_version` *and* a finding this component cannot produce.

---

## 12b. What went wrong, and what the measurements corrected

Three defects, all found by tests rather than by reading code, and all of the same kind: the
numbers were right and the *reproducibility* was not.

**1. Shuffling the prediction rows changed every disparity.** Not the group metrics — those
were stable — but the pooled *reference value* each disparity is measured against. The cause is
that `evaluation.metrics.ece` uses **equal-mass bins**, so rows tied at a bin boundary are
assigned to a bin by the order they arrive in. A shuffled input therefore produced a slightly
different reference, and every `max_deviation` moved with it. Fixed by `groups.CANONICAL_SORT`,
which puts the audited frame into Component 5's canonical order — `(inspection_date,
target_inspection_id)`, extended by model and fold — before any metric touches it.

This is the third time this project has measured row order being load-bearing: Component 6
found it moving coefficients by 7.049e-09, Component 7 found it moving a *prediction* by
1.12e-01, and Component 12 found it moving a binning decision.

**2. Sorting was necessary and not sufficient.** Two full production runs still produced
`mean_abs_shap` values differing at **1.8e-15**, because polars aggregates a group in parallel
and adds the rows in whatever order they reach a thread — sorting fixes *which* rows are in a
group, not the order they are added in. Fixed by summing with `math.fsum`, which is exactly
rounded and therefore order-independent on every machine rather than on this one.

**The difference was far below anything a reader would act on**: every rank, every Spearman
correlation and every count was identical. It was still worth fixing, because **a table that is
only nearly reproducible is a table whose two-run checksum comparison has stopped being a
detector** — and that comparison is how this project has found real defects three times.

**3. A figure was silently missing, and a missing figure looks exactly like a figure the data
could not support.** Component 9 names a calibrated model `xgboost_platt`; Component 11 names
the same model `xgboost`. The attribution figure looked profiles up under the calibrated name,
found nothing, and returned `None` — which is a *legitimate* outcome for a figure that lacks
support, so nothing logged an error. Fixed by `figures.base_model_name`, which is now the one
place that translation lives.

The general lesson, which the drift figure proves is not paranoia: **this component has a
legitimate "cannot draw this" path, and that path is indistinguishable from a bug unless the
name resolution is correct.** The disparity-drift figure is genuinely absent for want of data;
the attribution figure was absent for want of a suffix.

---

## 13. What this component does not establish

Restated here because it belongs beside the numbers rather than only in ADR 0035:

* **not causality** — every number is observational
* **not discrimination** — no model here has a geographic input; a difference arises through
  correlated features. And the converse: the absence of a group feature does not prove the
  absence of a disparity
* **not the absence of bias** — 27 of 78 community areas were excluded from every comparison
* **not legal or regulatory compliance** — no protected characteristic is observed anywhere in
  this project, and a correlate is not the attribute
* **not ethical acceptability, not equal treatment, not an optimal fairness policy**

And the inherited gap, stated in ADR 0019 *before* this component existed: the target is that
a violation was **cited**, not that an establishment was unsafe, and geography is close to the
strongest available proxy for who inspected. **Nothing here separates establishment risk from
differential inspection practice.**

---

## Limitations

1. **A geographic group is not a protected class.** Community areas correlate strongly with
   race and income by construction — that is what the city publishes them for — but the audit
   observes the correlate, never the attribute.
2. **`community_area` is a Socrata computed region id**, not necessarily the official
   community-area number. No boundary file is ingested, so no neighbourhood is named anywhere.
   Every finding above is attached to an opaque stable id.
3. **27 of 78 community areas and 13 of 69 ZIPs are below the support floor** and appear in no
   comparison. A system can be even across the groups it could measure and fail badly for one
   it could not.
4. **The folds are not independent samples.** The same premises appears in many test windows on
   a 358-day median canvass cycle, so a fold-to-fold SD is a dispersion and not a confidence
   interval.
5. **The pooled grain mixes 17 differently-fitted models.** A pooled number is a statement
   about the system as operated, not about one estimator.
6. **The drift question is unanswerable at the fold grain** on this data — see §11.
7. **`covid_shift` is one fold** with no variance estimate, and it supports 11 of 78 groups.
8. **No fairness intervention was implemented or tested.** No reweighting, no threshold
   adjustment, no per-group calibrator, no constraint. "Optimal" is undefined until someone
   chooses which criterion to prefer, and that is a policy question this component is not
   delegated.
9. **Component 11's sample bounds the attribution analysis.** 300 rows per (model, fold), so
   the median (model, community area) cell holds 40 explained rows. Profiles are compared for
   groups clearing 100; nothing per-row is claimed. Re-running `sentinel explain` at a larger
   sample to check a finding is forbidden — it would change the rows every Component 11 number
   rests on.
10. **Fairness criteria are mutually incompatible when base rates differ**, and they differ by
    thirty points here. That is why there is no single fairness score, and why the artifact
    reports four disparity measures side by side instead.
11. **Nothing was fixed.** Where an uncomfortable result was found it was reported. The
    calibration regression in §7 was measured and left alone, deliberately, with the reason
    recorded in ADR 0034.

---

## Figures

| figure | question it answers |
| --- | --- |
| `fairness_representation_<def>_<fold_set>.png` | how many rows does each group contribute, and where does the support floor cut? |
| `fairness_base_rates_<def>_<fold_set>.png` | what does the outcome rate look like before any model? |
| `fairness_ranking_<model>_<def>_quarterly.png` | does the ranking work equally well inside every group? |
| `fairness_calibration_<model>_<def>_quarterly.png` | did the global calibration improvement reach every group? |
| `fairness_slope_<model>_<def>_quarterly.png` | how far from 1.0 is each group's calibration slope? |
| `fairness_topk_<model>_<def>_k_pct_05_quarterly.png` | who appears in the priority set, relative to their share? |
| `fairness_capture_<model>_<def>_k_pct_05_quarterly.png` | how much of each group's risk did that priority set find? |
| `fairness_missingness_<def>_<fold_set>.png` | is the data itself distributed evenly? |
| `fairness_attribution_<model>_<def>_quarterly.png` | does the model rely on different features for different groups? |
| `fairness_covid_support_<def>.png` | how much less is measurable under the distribution shift? |

**72 figures**, of which 8 are attribution profiles — one per *explainable* model per
geography. `xgboost_chain_embeddings` correctly has none: Component 11 could not explain it
(ADR 0031), so there is no profile to group.

**Display policy**, stated on every figure: the best-supported `DISPLAY_TOP_N = 20` groups are
drawn, and the full table is the source of truth. Model-level figures are drawn for the
quarterly fold set only — a per-model covid panel would present eleven groups at the same
visual weight as fifty, and no trend may be claimed from that fold anyway.

The disparity-drift figure is **absent by measurement**: it needs four folds and there is one.

---

## Reproducing

```bash
uv run python scripts/profile_fairness.py            # read-only; fixes the frozen constants
uv run sentinel audit-fairness --dry-run --report    # writes nothing
uv run sentinel audit-fairness --report              # ~162 s
```

No thread override is needed and none should be used — unlike `calibrate` and `explain`, this
command fits nothing and re-executes nothing, so it has no bit-identity gate to be sensitive
to BLAS summation order.
