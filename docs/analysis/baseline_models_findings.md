# Baseline risk models: empirical findings

**Read this before changing `src/sentinel/modeling/`.**

Every number here was measured on the full snapshot before the implementation was
written. Nothing is estimated, illustrative or carried over from the project
specification. Where the data contradicts an expectation, the data wins and the
divergence is recorded.

**Source:** `data/processed/features/as_of_features_20260816T150313Z.parquet`
(sha256 `b7db5b2d…25e2f`)
**Profiling:** `uv run python scripts/profile_baselines.py`
**Libraries:** scikit-learn 1.9.0, numpy 2.5.2, polars 1.43.2

⚠ **Every fit reported in this document was confined to a fold's training window.**
No test-window number appears anywhere in the profiling script or in this file. That
is a hard rule: Component 5 protects evaluation time, but it cannot detect a human
reading a test metric, adjusting a hyperparameter, and re-running. The test numbers
for Component 6 come from `sentinel evaluate --predictions`, which records what it
did in a manifest.

---

## 1. What Component 6 is for

Components 1–5 established that ranking, not classification, is the operational
question, and measured how well the un-fitted heuristics answer it. The strongest —
`prior_canvass_priority_rate` — reaches NDE 0.1845, and business-as-usual is
indistinguishable from random within a quarter (NDE 0.0066, ROC-AUC 0.504).

Component 6 asks one question: **does fitting a model to the 26 as-of features add
anything beyond those heuristics?** It is the first component that fits anything, so
it is also the component that has to establish how fitting is done safely — what
"train" means, where preprocessing statistics come from, and what a model is allowed
to declare about itself.

It is deliberately a *baseline*: interpretable, fixed hyperparameters, no tuning. A
baseline whose numbers cannot be trusted is worse than no baseline, because
Components 7 and 8 will be measured against it.

---

## 2. The nullable partition — exactly 10 of 26, in 4 families

Derived from `FEATURE_SPECS[*].null_rule`, never hand-listed, so a Component 4 change
surfaces as a test failure rather than as silently wrong preprocessing.

| feature | null rule | dtype | nulls | null rate |
|---|---|---|---|---|
| `days_since_last_canvass` | no prior canvass | int32 | 5,615 | 0.097268 |
| `fail_at_last_canvass` | no prior canvass | bool | 5,615 | 0.097268 |
| `name_changed_since_last_canvass` | no prior canvass | bool | 5,615 | 0.097268 |
| `prior_canvass_fail_rate` | no inspected prior canvass | float64 | 5,961 | 0.103262 |
| `prior_canvass_priority_count` | no prior code-era canvass | int32 | 14,162 | 0.245327 |
| `prior_canvass_priority_foundation_count` | no prior code-era canvass | int32 | 14,162 | 0.245327 |
| `prior_canvass_priority_rate` | no prior code-era canvass | float64 | 14,162 | 0.245327 |
| `priority_at_last_canvass` | no prior code-era canvass | bool | 14,162 | 0.245327 |
| `days_since_any_inspection` | no prior inspection | int32 | 401 | 0.006946 |
| `days_since_first_inspection` | no prior inspection | int32 | 401 | 0.006946 |

**16 features are never null.** For those, `0` is a real observation and must pass
through untouched — Component 4's whole missing-value design exists to keep `0`
distinguishable from unknown.

### 2.1 The masks within a family are byte-identical

| family | columns | null rows | masks |
|---|---|---|---|
| no prior canvass | 3 | 5,615 | **IDENTICAL** |
| no prior code-era canvass | 4 | 14,162 | **IDENTICAL** |
| no inspected prior canvass | 1 | 5,961 | (single column) |
| no prior inspection | 2 | 401 | **IDENTICAL** |

This single measurement decides the preprocessing design.

### 2.2 Each mask is exactly a paired count's zero-set

| family | paired never-null count | null rows | count == 0 rows | match |
|---|---|---|---|---|
| no prior canvass | `prior_canvass_count` | 5,615 | 5,615 | **EXACT** |
| no prior code-era canvass | `prior_canvass_count_code_era` | 14,162 | 14,162 | **EXACT** |
| no inspected prior canvass | `prior_canvass_inspected_count` | 5,961 | 5,961 | **EXACT** |
| no prior inspection | `prior_inspection_count_any_type` | 401 | 401 | **EXACT** |

Component 4's paired-count design works exactly as its docstring claims. The
consequence for interpretation matters: **a missingness indicator adds no new
information to this table, only linearly-available information.** The fact "this
establishment has no prior canvass" is already present as `prior_canvass_count == 0`.
The indicator exists so a *linear* model can use it as a level shift instead of
having to express it as a kink in a slope.

---

## 3. Missing-value encoding: the decision and the alternative

### 3.1 Why `SimpleImputer(add_indicator=True)` is unusable here

It would emit **10** indicator columns, of which only **4** are distinct — 3 exact
duplicates in one family, 4 in another, 2 in a third. Under L2 the fit still
converges, but the coefficient mass splits arbitrarily across identical columns, and
`baseline_coefficients` becomes a table of numbers nobody can read. That defeats the
entire purpose of shipping an interpretable baseline.

A second, subtler reason: `add_indicator` uses `MissingIndicator(features="missing-only")`,
so the indicator set is chosen **by observation on the training window**, not by
declaration. On the full data every nullable column has nulls in every training
window, so this would never be noticed — but on a small synthetic fixture the column
count changes per fold, coefficient term names change with it, and nulls in an
untracked column get imputed with no indicator and no error. That is schema drift
that only bites in tests and on future re-ingests.

**Decision: build 4 family indicators explicitly, always present, in a declared
order.** Never `add_indicator`.

### 3.2 Median + indicator, not a sentinel value

`rankers.py` encodes a missing `days_since_last_canvass` as `max + 1`, and that is
correct *for a ranker* — a sentinel only has to sort. A coefficient has to mean
something, and the argument is specific to a linear model:

- Logistic regression is linear in each feature. A sentinel places "unknown" at a
  point on the **same slope** as the observed values, so one coefficient must
  simultaneously fit the real dose-response over observed rows and the arbitrary
  position of the unknown group. With 24.5% of rows in one such group, the sentinel
  dominates the estimate.
- Median + indicator is the classic missing-indicator parameterisation: algebraically
  it gives the missing group its own intercept while pinning its slope contribution
  at a constant. The slope is then estimated on observed rows only and the indicator
  absorbs the level shift. Nothing is asserted about the unknown group beyond "it
  differs by a constant".
- Median specifically, not mean: the median is partition-based and therefore exactly
  order-invariant, which matters for §5.

**The honest cost, stated:** the indicator cannot represent an *interaction* — a
differing slope in the missing group. That is a Component 7/8 question, not a
baseline one, and it is recorded as a limitation rather than worked around.

### 3.3 Nullable booleans fill with constant 0, not median

The three nullable booleans would, under `strategy="median"`, be filled with
whichever class is more common **in that fold's training window**. Measured, per fold:

| fold | `fail_at_last_canvass` mean | median fill | `priority_at_last_canvass` mean | median fill |
|---|---|---|---|---|
| quarterly-2022Q2 | 0.2049 | 0 | **0.6310** | 1 |
| quarterly-2023Q2 | 0.2090 | 0 | 0.5907 | 1 |
| quarterly-2024Q2 | 0.2164 | 0 | 0.5550 | 1 |
| quarterly-2025Q2 | 0.2230 | 0 | 0.5284 | 1 |
| quarterly-2026Q1 | 0.2235 | 0 | 0.5106 | 1 |
| quarterly-2026Q2 | 0.2223 | 0 | **0.5056** | 1 |

`priority_at_last_canvass` drifts monotonically from 0.6310 to 0.5056 across the 17
quarterly folds — it currently sits **0.0056 above the median-fill boundary**. The
fill has not flipped yet. It is roughly one year of additional data from flipping
mid-fold-sequence, and when it does, the encoding of "unknown" moves from the false
side of the linear predictor to the true side and the family indicator's coefficient
sign flips to compensate — a reversal with no substantive cause whatsoever.

**Decision: `strategy="constant", fill_value=0.0` for the three nullable booleans.**
"Unknown" then means "false, plus the indicator", and the indicator's coefficient is
exactly the log-odds offset of the unknown group. This is the interpretable
parameterisation, and it is immune to the drift above.

This is a case where the hazard was measured rather than assumed. A median rule would
have passed every test today and silently changed meaning in 2027.

### 3.4 The rule, in full

| group | count | rule |
|---|---|---|
| never-null numerics | 16 | passthrough — `0` is a true observation |
| nullable numerics | 7 | `SimpleImputer(strategy="median")`, fitted on train only |
| nullable booleans | 3 | `SimpleImputer(strategy="constant", fill_value=0.0)` |
| family indicators | 4 | passthrough — computed, always present |

Then `StandardScaler()` over all 30 columns, also fitted on train only.

---

## 4. Collinearity: measured, and it shapes how coefficients may be read

Fold `quarterly-2022Q2`, 23,346 training rows, 30 matrix columns.

**Condition number of the standardised design matrix: 71.8.** Moderate — L2
regularisation handles it and every fold converges — but high enough that individual
coefficients are not feature importances.

| feature A | feature B | \|r\| |
|---|---|---|
| `prior_canvass_count` | `prior_canvass_inspected_count` | 0.9888 |
| `prior_canvass_priority_count` | `prior_canvass_priority_foundation_count` | 0.9826 |
| `canvasses_last_730d` | `canvasses_last_1095d` | 0.9550 |
| `missing_no_prior_canvass` | `missing_no_inspected_canvass` | 0.9528 |
| `prior_canvass_count` | `prior_inspection_count_any_type` | 0.9215 |

Some of this is guaranteed by construction: the windowed counts nest
(`365d ⊆ 730d ⊆ 1095d`), and Component 4 deliberately emits each subtype count beside
its total so `0` stays interpretable. The correlation is a consequence of a good
feature design, not a defect in it.

The consequence shows up directly in §5: `prior_canvass_count` carries mean
coefficient **+1.99** while `prior_canvass_inspected_count`, its 0.9888-correlated
partner, carries **−1.47**. That pair is jointly estimating one effect, and reading
either number alone is meaningless. **Component 11 owns attribution; the coefficients
table is provenance, not an explanation.** This is stated in the data contract so no
reader mistakes a large coefficient for an important feature.

---

## 5. Coefficient stability across folds

All 17 quarterly folds, training windows only. Ten largest by mean magnitude:

| term | min | mean | max | sign flips |
|---|---|---|---|---|
| `prior_canvass_count` | 1.7191 | **1.9867** | 2.3719 | no |
| `prior_canvass_inspected_count` | −1.9069 | **−1.4659** | −1.2833 | no |
| `prior_inspection_count_any_type` | −0.9994 | −0.8346 | −0.7015 | no |
| `missing_no_code_era_canvass` | 0.6127 | 0.6528 | 0.7249 | no |
| `missing_no_prior_canvass` | −0.5160 | −0.4260 | −0.3190 | no |
| `prior_canvass_count_code_era` | −0.4544 | −0.4134 | −0.3724 | no |
| `prior_canvass_priority_foundation_count` | 0.2104 | 0.2997 | 0.5420 | no |
| `prior_reinspection_count` | 0.2535 | 0.2808 | 0.3267 | no |
| `prior_canvass_pass_w_conditions_count` | 0.2352 | 0.2591 | 0.2834 | no |
| `days_since_any_inspection` | 0.2095 | **0.2429** | 0.2811 | no |

Six terms change sign across folds: `canvass_priority_events_last_1095d`,
`prior_canvass_fail_count`, `prior_canvass_count_current_name`, `canvasses_last_730d`,
`prior_canvass_priority_rate`, `canvass_priority_events_last_730d`, and
`missing_no_prior_inspection`. **Every one has mean magnitude below 0.13** — they are
the nested window features and the 401-row indicator, i.e. exactly the terms §4
predicts will be unstable under collinearity. No large-magnitude term flips.

**A finding worth recording against an expectation:** `days_since_any_inspection`
carries a stable positive coefficient (+0.2095 … +0.2811, mean +0.2429, no sign flip)
and ranks tenth of thirty. It was speculated during design that this feature might be
near-worthless because it partly encodes scheduling policy rather than risk. **The
training-window evidence does not support that.** Whether it helps *test* ranking is a
question only `sentinel evaluate` may answer, and the ablation model exists so that it
is answered by the harness rather than by argument.

---

## 6. Determinism: row order reaches the coefficients

Fold `quarterly-2022Q2`, 23,346 training rows, same seed, same data, order changed.

| condition | max abs coefficient diff | bit-identical |
|---|---|---|
| shuffled, no sort | **7.049e-09** | no |
| shuffled, then canonically sorted | **0.000e+00** | **yes** |

`StandardScaler` accumulates mean and variance through `_incremental_mean_and_var`,
and the lbfgs gradient is a BLAS reduction; both are float-summation-order dependent.
7e-09 is numerically negligible and operationally irrelevant to any ranking — but it
means "identical inputs produce identical outputs" is **false** without an explicit
order, and that is a property this project asserts in its manifests.

**Decision: `fit_fold` sorts training rows by `(inspection_date, target_inspection_id)`
before fitting.** The canonical sort is load-bearing, not decorative, and the test that
shuffles input and asserts bit-identical coefficients is testing the sort.

Two honest limits on the reproducibility claim:

- Bit-identity holds for **a fixed library set and BLAS thread count**, not across
  them. So `sklearn_version`, `numpy_version` and `blas_threads` go into the manifest;
  nothing else in the repository pins them and the coefficients depend on them.
- `random_state=42` has **no effect** on `solver="lbfgs"` — there is no randomness in
  the fit. It is recorded because it documents intent and because a future switch to
  `saga` would make it load-bearing, but it must not be described as the thing that
  makes the model deterministic. The sort is.

---

## 7. Convergence

All 18 folds, `max_iter=1000`:

| | |
|---|---|
| iterations, min | 68 (quarterly-2022Q3) |
| iterations, max | 83 (quarterly-2022Q2) |
| covid_shift | 81 |
| folds hitting `max_iter` | **0** |
| `ConvergenceWarning` raised | **0** |

Every fit converges with two orders of magnitude of headroom. **Non-convergence is
therefore an error-severity check, not a log field** — a model that stopped at the
iteration cap has coefficients that depend on the cap and are not comparable across
folds. Installing the check while it passes is the point; discovering it later would
mean discovering it after publishing a number.

---

## 8. The model set

Three models, all L2 logistic regression, fixed hyperparameters, no tuning:
`penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42, class_weight=None`.

| name | features | why it exists |
|---|---|---|
| `logistic_regression` | 26 + 4 indicators | the primary baseline; spec §6.1 |
| `logistic_regression_no_scheduling` | 25 + 4 (− `days_since_any_inspection`) | the ablation, built as a *second model* so the harness compares them |
| `cdph_2015_approximation` | reconstructible 2015 inputs only | the historical comparison, labelled an approximation |

**No class weighting and no resampling.** Measured prevalence is 52.52% overall and
0.379–0.492 per test window. There is no imbalance problem, the project spec's
anticipated 15–25% does not match the data, and resampling would corrupt the
probability scale that Component 9 depends on.

**Why the ablation is two models rather than a post-hoc adjustment.** Refitting
without the feature is the only way to see what the model does when it cannot lean on
it; dropping a column at scoring time measures something else entirely (a
mis-specified model). Component 5 accepts any scored prediction set, so two models is
the natural expression.

---

## 9. The CDPH 2015 baseline: an approximation, and why

Component 5 deferred this deliberately — it is a fitted logistic regression, so it
crossed the component boundary. The deferral came with a table, reproduced here from
`temporal_evaluation_findings.md` §10 and `rankers.py`:

| 2015 input family | status | note |
|---|---|---|
| prior violation history | available | 12 Component 4 canvass/priority features |
| time since last inspection | available | `days_since_last_canvass` |
| business age / tenure | available | `days_since_first_inspection` |
| inspector identity | **excluded** | deliberately — audit Finding 1 |
| nearby 311 complaints | **not ingested** | Component 1 extension |
| burglary intensity | **not ingested** | Component 1 extension |
| alcohol / tobacco licence | **not ingested** | Component 1 extension |
| weather / temperature | **not ingested** | needs NOAA GHCN USW00094846 |
| facility type | **not a feature** | in the raw snapshot, not in Component 4's table |
| CDPH risk category | **not a feature** | in the raw snapshot, not in Component 4's table |

**3 of 10 input families are reachable. A faithful replication is not possible from
the current data contract.**

So `cdph_2015_approximation` is exactly that — an approximation. It carries an
`approximation_note` naming every unreachable input, the note is written into the
manifest and the data contract, and no artifact or document describes it as the CDPH
model. **No coefficient, input or historical behaviour is fabricated.** Two further
reasons the 2015 published numbers are not a benchmark this project can be measured
against: a different food code, a different target definition, and a 14.1% base rate
against this data's 52.52%.

Related gap, carried forward: `days_since_last_canvass` is **not** the specification's
"days overdue". A statutory overdue figure needs the CDPH risk category to know what
the deadline was (Risk 1 twice yearly, Risk 2 annually, Risk 3 every two years), and
that column is in the raw snapshot but not in Component 4's feature table.

---

## 10. What was blocked, and why

| item | status | reason |
|---|---|---|
| faithful CDPH 2015 replication | **NOT POSSIBLE** | 7 of 10 input families unreachable (§9); an approximation ships instead, labelled |
| statutory "days overdue" feature | **BLOCKED** | needs CDPH risk category; belongs in Component 4 behind a bumped `feature_definition_version` |
| temperature covariate | **BLOCKED** | no weather ingested; NOAA GHCN USW00094846 is a Component 1 extension |
| hyperparameter tuning | **DELIBERATELY NOT DONE** | a baseline exists to be trustworthy, not optimal; tuning needs a train/calibration protocol and belongs with the model that benefits from it |
| probability calibration | **Component 9** | the raw `predict_proba` values are reported *uncalibrated*, and that is the "before" number C9 is justified against |
| feature attribution | **Component 11** | §4 shows why coefficients are not importances here |
| missing-slope interactions | **Component 7/8** | the indicator captures a level shift, not a differing slope (§3.2) |

---

## 11. Limitations

1. **Coefficients are not feature importances.** Condition number 71.8 and a
   0.9888-correlated pair that splits into +1.99 / −1.47 (§4). The coefficients table
   is provenance and a sanity check, not an explanation.
2. **Bit-reproducibility is conditional** on library versions and BLAS thread count
   (§6), which is why all three are in the manifest.
3. **Uncalibrated probabilities.** `predict_proba` is reported as-is. Brier, log-loss,
   ECE and MCE are emitted as *diagnostics of an uncalibrated model*.
4. **The estimand is unchanged and still not causal.** Component 6 produces a
   re-ordering of inspections that actually occurred. It says nothing about
   establishments nobody visited — they have no labels — and nothing about illness
   prevented.
5. **The ablation answers one question, not the general one.** Removing
   `days_since_any_inspection` measures that feature's contribution to *this* model
   class with *this* preprocessing. It is not a general statement about the feature.
6. **`NOT VERIFIED` until the full run:** every test-window number. Verify with
   `uv run sentinel train-baselines` then
   `uv run sentinel evaluate --predictions <path> --report`.

---

## 12. Full-data results

Measured, not estimated. Produced by `sentinel train-baselines` then
`sentinel evaluate --predictions`, and read back out of the Component 5 artifacts.

**Run:** `baseline_predictions_20260817T142118Z.parquet` →
`evaluation_metrics_20260817T142157Z.parquet`
**Population:** 57,727 feature rows, 18 folds (17 quarterly + 1 `covid_shift`), 3 models,
**54 fits**, 124,608 prediction rows, 1,530 coefficient rows.
**Runtime:** training 29.7 s (22.3 s of it fitting); evaluation 237.8 s.
**Artifacts:** predictions 1,210,344 B; coefficients 25,957 B; training log 8,903 B;
manifest 10,433 B.

### 12.1 Ranking, mean over the 17 quarterly folds

The three trained models are separated from the six existing reference producers by a
line. Nothing is averaged across fold sets.

| model | ROC-AUC | PR-AUC | NDE | mean days earlier | precision@k_1_day | lift@k_1_day |
|---|---|---|---|---|---|---|
| **logistic_regression** | **0.6163** | **0.5321** | **0.2326** | **+5.70** | **0.6576** | **1.5254** |
| logistic_regression_no_scheduling | 0.6119 | 0.5245 | 0.2238 | +5.48 | 0.6173 | 1.4364 |
| cdph_2015_approximation *(approximation)* | 0.6059 | 0.5118 | 0.2119 | +5.18 | 0.5618 | 1.3122 |
| — | | | | | | |
| prior_canvass_priority_rate | 0.5915 | 0.5012 | 0.1845 | +4.47 | 0.5551 | 1.3000 |
| priority_at_last_canvass | 0.5747 | 0.4746 | 0.1522 | +3.68 | 0.5382 | 1.2630 |
| days_since_last_canvass | 0.5381 | 0.4601 | 0.0765 | +1.76 | 0.4672 | 1.0902 |
| random | 0.5051 | 0.4376 | 0.0101 | +0.09 | 0.4484 | 1.0393 |
| business_as_usual | 0.5040 | 0.4347 | 0.0066 | 0.00 | 0.4323 | 1.0081 |
| constant | 0.5000 | 0.4307 | 0.0065 | −0.01 | 0.4283 | 0.9989 |

Mean test prevalence — and therefore the PR-AUC no-skill floor — is **0.4307**.

### 12.2 What this does and does not establish

**The trained model beats the strongest heuristic, on every metric and on every fold.**
NDE 0.1845 → **0.2326** (+26% relative), mean days earlier +4.47 → **+5.70**,
precision@k_1_day 0.5551 → **0.6576**. It wins on **17 of 17** quarterly folds, by
+0.0019 to +0.0542 ROC-AUC. Per fold, generated from the artifact rather
than transcribed, so the claim is auditable:

| fold | logistic_regression | prior_canvass_priority_rate | delta |
|---|---|---|---|
| 2022Q2 | 0.6003 | 0.5714 | +0.0289 |
| 2022Q3 | 0.6035 | 0.5878 | +0.0157 |
| 2022Q4 | 0.6022 | 0.5747 | +0.0275 |
| 2023Q1 | 0.5849 | 0.5645 | +0.0204 |
| 2023Q2 | 0.6136 | 0.5793 | +0.0343 |
| 2023Q3 | 0.6322 | 0.6298 | **+0.0024** |
| 2023Q4 | 0.6152 | 0.5610 | +0.0542 |
| 2024Q1 | 0.6006 | 0.5987 | **+0.0020** |
| 2024Q2 | 0.5993 | 0.5851 | +0.0142 |
| 2024Q3 | 0.6190 | 0.5946 | +0.0244 |
| 2024Q4 | 0.6134 | 0.5796 | +0.0338 |
| 2025Q1 | 0.6463 | 0.6202 | +0.0261 |
| 2025Q2 | 0.6611 | 0.6197 | +0.0414 |
| 2025Q3 | 0.6289 | 0.6068 | +0.0221 |
| 2025Q4 | 0.6357 | 0.6037 | +0.0320 |
| 2026Q1 | 0.6024 | 0.6006 | **+0.0019** |
| 2026Q2 | 0.6182 | 0.5775 | +0.0407 |

Widest 2023Q4, narrowest 2026Q1. **But read the bolded rows before quoting
"17 of 17":** 3 folds (2023Q3, 2024Q1, 2026Q1) are won by under
0.0025 ROC-AUC, which is not a meaningful margin. The honest statement is
that the model is never *worse* than the best heuristic and is clearly better on most
folds -- not that it is reliably better on all of them.

**The improvement survives the time-invariance perturbation.** Component 5's label
re-draw band (1,000 replications) puts `logistic_regression` at NDE 0.2326
[p05 0.2160, p95 0.2374] against the heuristic's 0.1845 [0.1720, 0.1922]. **The bands do
not overlap**, so the gap is not an artifact of the seasonal drift Component 5 measured.

**But the honest reading is that this is a modest model of a hard problem:**

1. **PR-AUC 0.5321 against a 0.4307 floor is +0.10, not a good absolute result.** The
   spec named PR-AUC the primary classification metric while anticipating a 15–25% base
   rate; at 43% prevalence the floor is high and the headroom small.
2. **43.24% of violations are still found *later* than business-as-usual.** That is
   marginally *worse* than the best heuristic's 42.88%. Re-ordering under a fixed
   capacity is zero-sum: moving one establishment earlier moves another later, and the
   model's gain is a net effect, not a free one. Anyone quoting "+5.7 days earlier"
   without this number is quoting half the result.
3. **ROC-AUC 0.6163 is a weak classifier by any general standard.** It is reported
   because Component 5 reports it, not because it is the operative quantity.
4. The bar was low. Business-as-usual is indistinguishable from random within a quarter
   (NDE 0.0066, ROC-AUC 0.5040), so beating it is not evidence of a good model — only
   beating `prior_canvass_priority_rate` is, and that margin is real but small.

### 12.3 The `days_since_any_inspection` ablation, and the distribution-shift inversion

On the quarterly folds, keeping the feature helps slightly: NDE 0.2326 vs 0.2238, ROC-AUC
0.6163 vs 0.6119. So the design-time speculation that it might be near-worthless is **not
supported** — it is a small positive, consistent with its stable +0.24 training
coefficient (§5).

On `covid_shift`, the ordering **reverses**:

| model | ROC-AUC | NDE | mean days earlier |
|---|---|---|---|
| **logistic_regression_no_scheduling** | **0.6286** | **0.2571** | **+25.60** |
| logistic_regression | 0.6256 | 0.2512 | +25.03 |
| cdph_2015_approximation | 0.6221 | 0.2442 | +23.69 |
| days_since_last_canvass | 0.5842 | 0.1704 | +14.18 |
| prior_canvass_priority_rate | 0.5614 | 0.1357 | +8.82 |

This is exactly the inversion the Component 5 handoff predicted, now observed with fitted
models: **the ablation that loses on the rolling folds wins under distribution shift.**
The reading is that `days_since_any_inspection` partly encodes scheduling policy, and when
the scheduling policy itself breaks — which is what 2020 was — a model leaning on it is
the more fragile one.

**Model selection on the quarterly folds would have picked the wrong model for
`covid_shift`.** That is the single most important methodological result of this
component, and it is why the two fold sets are never averaged together. Note also that
the covid_shift days-earlier figures (+25.6) are large only because its test window is 19
months rather than a quarter; they are not comparable to the quarterly numbers.

### 12.4 Probability metrics (uncalibrated)

Emitted for the first time in the project, because these are the first producers to
declare `is_probability=True`.

| model | Brier | log-loss | ECE | MCE |
|---|---|---|---|---|
| logistic_regression | 0.2382 | 0.6723 | 0.0635 | 0.1664 |
| logistic_regression_no_scheduling | 0.2391 | 0.6742 | 0.0657 | 0.1680 |
| cdph_2015_approximation | 0.2401 | 0.6764 | 0.0633 | 0.1817 |

⚠ **These are diagnostics of an uncalibrated model.** ECE ≈ 0.064 and MCE ≈ 0.17 mean the
predicted probabilities are meaningfully off in places — which is the measured "before"
number that justifies Component 9's existence, and must never be reported as though
calibration had been done. `precision`, `recall` and `f1` are deliberately not emitted:
they need a threshold, `METRICS_SCHEMA` has no column to record one in, and the only
place to put 0.5 would be `k_name`.

### 12.5 Coefficients from the full run

Identical to the training-window profiling in §5, which is the expected result and a
check that the implementation encodes what the profiling established. Largest terms:

| term | mean | min | max |
|---|---|---|---|
| `prior_canvass_count` | +1.9867 | +1.7191 | +2.3719 |
| `prior_canvass_inspected_count` | −1.4659 | −1.9069 | −1.2833 |
| `prior_inspection_count_any_type` | −0.8346 | −0.9994 | −0.7015 |
| `missing_no_code_era_canvass` | +0.6528 | +0.6127 | +0.7249 |
| `__intercept__` | +0.4492 | +0.1743 | +0.8018 |
| `missing_no_prior_canvass` | −0.4260 | −0.5160 | −0.3190 |

Seven of thirty terms change sign across folds, and **every one has mean magnitude below
0.118** — the nested window features, `prior_canvass_priority_rate`,
`prior_canvass_fail_count`, `prior_canvass_count_current_name` and the 401-row
`missing_no_prior_inspection` indicator. No large-magnitude term flips. Read §4 before
reading any of these as importances: the top two are a 0.9888-correlated pair jointly
estimating one effect.

### 12.6 Validation

All **15** error-severity Component 6 checks pass on the full data, including
`preprocessing_comes_from_train` (342 imputation medians re-derived from the training
window and matched) and `trained_through_is_the_training_end` (54 fits).

All **14** error-severity Component 5 checks pass with the predictions attached,
including `scores_respect_the_decision_point` — the check that was unreachable before
this component existed, and which a test now proves can fail.

Scores span 2.44e-06 … 0.999603 with **zero** saturated at exactly 0 or 1.
