# Component 7 — Gradient-boosted risk models: findings

**Source:** `data/processed/features/as_of_features_20260816T150313Z.parquet` (57,727 rows,
`feature_definition_version = v1`)
**Profiling command:** `uv run python scripts/profile_boosting.py`
**Search:** `uv run sentinel tune-boosting --trials 100 --report`
**Training:** `uv run sentinel train-boosting --report`
**Evaluation:** `uv run sentinel evaluate --predictions <boosted_predictions>.parquet --report`
**Libraries:** xgboost 3.4.1, lightgbm 4.7.0, optuna 4.9.0, numpy 2.5.2, scikit-learn 1.9.0

⚠ **Sections 1–7 contain no test-window number, and that was a hard rule rather than a style
preference.** Every fit reported before §8 is confined to a fold's *training* window, with one
exception stated where it occurs: §7 scores a **calibration** window, which exists precisely so
design choices can be frozen before the test period is opened. Component 5 protects evaluation
time, but it cannot protect against a human reading a test metric, changing a search range, and
re-running. That loop is leakage, it leaves no trace in any artifact, and no check in this
repository can detect it. §8 onwards reports the evaluation, which was run once, after the
hyperparameters were frozen and committed.

---

## 1. What this component is for

Component 6 established that Component 4's 26-feature representation carries usable linear signal:
`logistic_regression` beat the strongest heuristic on all 17 quarterly folds. It also left three
things open. The margins on several folds were thin. The model ordering **reversed** on the
`covid_shift` fold. And 43.24% of violations were still surfaced *later* under the logistic
ranking than under business-as-usual.

Component 7 asks one question: **can a nonlinear model learn a better risk ranking from the same
features, under the same Component 5 evaluation?**

It does not ask whether a better feature representation exists (Component 4 owns that), whether the
probabilities can be corrected (Component 9), or what drives an individual score (Component 11).

## 2. Why boosting, and why two of them

Gradient-boosted trees are the natural next estimator for three reasons specific to this data.

**The features interact and the GLM cannot see it.** Component 6 measured a condition number of
71.8 and one feature pair correlated at 0.9888, which is a representation a linear model handles
by splitting weight between collinear terms. A tree can condition on one and then split on the
other. MEMORY.md's open question 14 — whether the missing-indicator encoding needs an interaction
term — is answered here for free: a tree gets those interactions without any being declared.

**Ten of 26 features are nullable, and the NULLs mean something.** Component 6 had to impute
`prior_canvass_fail_rate` with a training-window median when it was NULL. But that NULL means
"there was no prior canvass" — a fact about the establishment, replaced by the average of a
population it is not in. A booster routes NaN to a learned default direction at each split, so the
fact survives.

**Both libraries were used because the comparison is the question.** XGBoost grows depth-wise and
LightGBM leaf-wise; "a tree of depth 4" is not the same object in each. A single result would be a
fact about one library's inductive bias. See ADR 0016.

## 3. Tuning regions (profile `tuning_regions`)

| fold set | outer folds | tuning region | region end | first test start | safe |
| --- | --- | --- | --- | --- | --- |
| covid_shift | 1 | 2018-07-01..2020-05-31 | 2020-05-31 | 2020-06-01 | yes |
| quarterly | 17 | 2018-07-01..2022-03-31 | 2022-03-31 | 2022-04-01 | yes |

The quarterly region is the first quarterly fold's `train_start..calibration_end`. Because
quarterly folds expand from a fixed anchor, that span is a subset of every later fold's own
train-plus-calibration — so no fold is scored by parameters chosen on its own test data, or on any
other quarterly fold's.

**The `covid_shift` test window (2020-06-01..2021-12-31) sits inside the quarterly region.** That
single fact is why there are two studies rather than one. A shared study would have selected
parameters using the shift fold's own test labels, biasing the one number this project most needs
to keep clean. See ADR 0017.

## 4. Inner folds (profile `inner_folds`)

| fold set | inner fold | train rows | train rate | validation rows | validation rate |
| --- | --- | --- | --- | --- | --- |
| covid_shift | 2019Q4 | 7,148 | 0.8271 | 2,160 | 0.7213 |
| covid_shift | 2020Q1 | 8,899 | 0.8186 | 2,316 | 0.6015 |
| quarterly | 2020Q4 | 15,009 | 0.7608 | 1,061 | 0.4779 |
| quarterly | 2021Q1 | 16,131 | 0.7425 | 1,534 | 0.5091 |
| quarterly | 2021Q2 | 17,192 | 0.7262 | 2,023 | 0.5220 |
| quarterly | 2021Q3 | 18,726 | 0.7084 | 1,328 | 0.4947 |
| quarterly | 2021Q4 | 20,749 | 0.6902 | 1,269 | 0.4736 |
| quarterly | 2022Q1 | 22,077 | 0.6784 | 1,357 | 0.4311 |

Each inner fold keeps the outer structure's unused calibration quarter between training and
validation. Every outer fold has that gap, so tuning without it would select parameters for a
zero-gap regime and apply them to a one-gap regime.

**`covid_shift` yields only two inner folds**, because its region is eight quarters long. That is
thin for an eight-dimensional search and is reported as thin — its parameters are less well
determined than the quarterly ones, and its results are a robustness observation rather than a
selection criterion. A study yielding fewer than two inner folds is refused rather than run.

Note the base-rate drift visible in the training column: 0.83 → 0.68 across the quarterly inner
folds, continuing into test-window rates of 0.38–0.51. Any base-rate-dependent metric has to be
read beside its own window's prevalence.

## 5. NaN density (profile `nan_density`)

Widest quarterly training window `quarterly-2026Q2`, 53,844 rows × 30 matrix columns:

| column | NaN rows | share |
| --- | --- | --- |
| prior_canvass_priority_count | 13,857 | 0.2574 |
| prior_canvass_priority_foundation_count | 13,857 | 0.2574 |
| prior_canvass_priority_rate | 13,857 | 0.2574 |
| priority_at_last_canvass | 13,857 | 0.2574 |
| prior_canvass_fail_rate | 5,651 | 0.1050 |
| days_since_last_canvass | 5,315 | 0.0987 |
| fail_at_last_canvass | 5,315 | 0.0987 |
| name_changed_since_last_canvass | 5,315 | 0.0987 |
| days_since_any_inspection | 382 | 0.0071 |
| days_since_first_inspection | 382 | 0.0071 |

77,788 NaN cells of 1,615,320 (4.82%); 10 of 30 columns carry any NaN. Over the whole production
run, **3,404,772 NaN cells reached the estimators** across 54 fits.

A quarter of rows have no code-era canvass history at all. Component 6 filled those with a
training-window median; Component 7 does not fill them. The four null-rule family indicators are
kept anyway even though a NaN-native learner does not need them — dropping them would mean the
boosted and baseline matrices differ, and every C6-versus-C7 comparison would become ambiguous
between "the estimator is better" and "the matrix is different".

## 6. Row order is load-bearing, far more than in Component 6 (profile `row_order_sensitivity`)

Fold `quarterly-2026Q2`, 53,844 training rows, scored on its 1,917-row calibration window:

| model | max abs difference, shuffled | max abs difference, shuffled then re-sorted |
| --- | --- | --- |
| xgboost | **1.124396e-01** | 0.000000e+00 |
| lightgbm | **1.230255e-01** | 0.000000e+00 |

This is the sharpest single finding in the profiling pass. Component 6 measured its coefficients
moving by 7.049e-09 under a re-ordering of the same rows — a float-summation artifact. A booster
moves a **prediction by 0.11**, seven orders of magnitude larger, because it draws row and column
subsamples in row order rather than merely summing over them. Two people fitting "the same model"
on the same rows in a different order would get materially different rankings.

The second column is what makes the component reproducible: re-sorting a shuffled frame restores
the original fit **exactly**. `train.fit_fold` re-sorts unconditionally rather than trusting the
caller, and `tests/test_boosting_train.py` and `tests/test_boosting_build.py` assert bit-identity
through the whole command.

## 7. Capacity cap and the calibration probe

**Capacity cap** (profile `capacity_cap`). LightGBM grows leaf-wise, so `max_depth` alone bounds
nothing; `num_leaves` is the real capacity knob. At 5 of the 8 searchable depths the declared
LightGBM range (8–256) would allow more leaves than a depth-wise XGBoost tree could ever build
(2^depth). `tuning.suggest_params` caps `num_leaves` at 2^`max_depth`, so both searches explore the
same capacity range and a measured difference is about the algorithm rather than the space. The cap
bound in practice: the frozen quarterly LightGBM parameters are `max_depth=4, num_leaves=16`, i.e.
exactly at the ceiling.

**Calibration probe** (profile `calibration_probe`), fold `quarterly-2026Q2`, 1,917 calibration
rows, **provisional** parameters:

| model | PR-AUC | ROC-AUC | Brier | ECE | MCE | saturated |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.4806 | 0.6028 | 0.2347 | 0.0503 | 0.1219 | 0 |
| xgboost | 0.4816 | 0.6183 | 0.2307 | 0.0450 | 0.1530 | 0 |
| lightgbm | 0.4742 | 0.6135 | 0.2319 | 0.0378 | 0.1932 | 0 |

This probe was run to pre-register an expectation: HANDOFF.md warned that "a booster's raw
probabilities are usually *worse* than a logistic model's", and if that held, a worse ECE at
evaluation would be predicted behaviour rather than a defect. **The probe did not support the
warning** — boosted ECE was slightly *better*, with worse MCE. Recorded here as a pre-registration
so §9's evaluation result can be read against a prediction made before the test window was opened.

## 8. The search (production run)

400 trials over 4 studies, **0 failed**, 563.8s total. `TPESampler(seed=20260817)`; early stopping
at 50 rounds against inner validation only; round cap 2000.

| study | trials | best trial | mean inner-validation PR-AUC | frozen rounds | seconds |
| --- | --- | --- | --- | --- | --- |
| xgboost-quarterly | 100 | 36 | 0.6054 | 103 | 314.1 |
| lightgbm-quarterly | 100 | 28 | 0.6033 | 63 | 152.4 |
| xgboost-covid_shift | 100 | 63 | 0.7786 | 192 | 72.6 |
| lightgbm-covid_shift | 100 | 74 | 0.7777 | 54 | 24.7 |

⚠ **None of those PR-AUCs is a result.** They are measured on inner validation quarters that are
training data for every outer fold the parameters are then used on. The `covid_shift` figures are
higher only because that region sits in the pre-2020 era when the base rate was around 0.82.
Quoted without their region they would look like a breakthrough. See ADR 0018.

Frozen parameters (`boosting/definitions.py`, provenance sha256
`a77687b7…adec8b14`):

| | xgboost / quarterly | xgboost / covid_shift | lightgbm / quarterly | lightgbm / covid_shift |
| --- | --- | --- | --- | --- |
| max_depth | 4 | 3 | 4 | 8 |
| num_leaves | — | — | 16 | 10 |
| learning_rate | 0.1926 | 0.0560 | 0.2992 | 0.0578 |
| leaf-size regularizer | min_child_weight 12.61 | min_child_weight 2.65 | min_child_samples 54 | min_child_samples 44 |
| row subsample | 0.7878 | 0.7621 | 0.7781 | 0.6321 |
| column subsample | 0.6690 | 0.6173 | 0.6047 | 0.8935 |
| L1 | 1.09e-08 | 1.99e-07 | 2.19e-04 | 1.42e-05 |
| L2 | 5.89e-03 | 2.98e-02 | 7.10e-07 | 2.92e-01 |
| n_estimators | 103 | 192 | 63 | 54 |

The two XGBoost studies landed in genuinely different places — the shift study chose shallower
trees, a quarter of the learning rate and nearly twice the rounds. That is the pre-COVID regime's
own answer, and it is a concrete argument for the two-study design rather than one shared set.

Every search chose shallow trees. `max_depth` 3–4 for three of the four studies, against a
searchable range of 3–10. The data does not support deep interactions.

## 9. Results

Training run: 54 fits (3 models × 18 folds), 21.4s, 124,608 prediction rows — the same row count
Component 6 produces, because both score every test row of every fold exactly once. All 17
Component 7 validation checks passed and all 14 Component 5 checks passed, including
`scores_respect_the_decision_point`.

### 9.1 Quarterly (17 folds) — the headline

| model | ROC-AUC | PR-AUC | NDE | days earlier | median | SD | P@k_1day | lift@k_1day | first-half | found later |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **xgboost_class_weighted** | 0.6195 | 0.5355 | **0.2390** | +5.85 | 5.62 | 35.46 | 0.6629 | 1.5436 | 0.6023 | 0.4283 |
| **xgboost** | 0.6188 | 0.5343 | 0.2376 | +5.83 | 5.35 | 35.48 | 0.6308 | 1.4640 | 0.6020 | 0.4289 |
| **lightgbm** | 0.6177 | 0.5342 | 0.2355 | +5.75 | 5.71 | 35.50 | 0.6598 | 1.5333 | 0.5997 | 0.4285 |
| logistic_regression (C6) | 0.6163 | 0.5321 | 0.2326 | +5.70 | 5.26 | 35.30 | 0.6576 | 1.5254 | 0.6005 | 0.4324 |
| logistic_regression_no_scheduling | 0.6119 | 0.5245 | 0.2238 | +5.48 | 4.74 | 35.13 | 0.6173 | 1.4364 | 0.5975 | 0.4337 |
| cdph_2015_approximation | 0.6059 | 0.5118 | 0.2119 | +5.18 | 4.59 | 35.01 | 0.5618 | 1.3122 | 0.5951 | 0.4412 |
| prior_canvass_priority_rate | 0.5915 | 0.5012 | 0.1845 | +4.47 | 4.88 | 32.60 | 0.5551 | 1.3000 | 0.5764 | 0.4288 |
| priority_at_last_canvass | 0.5747 | 0.4746 | 0.1522 | +3.68 | 3.09 | 27.36 | 0.5382 | 1.2630 | 0.5802 | 0.4613 |
| days_since_last_canvass | 0.5381 | 0.4601 | 0.0765 | +1.76 | 2.18 | 36.86 | 0.4672 | 1.0902 | 0.5365 | 0.4623 |
| random | 0.5051 | 0.4376 | 0.0101 | +0.09 | 0.65 | 36.25 | 0.4484 | 1.0393 | 0.5046 | 0.4791 |
| business_as_usual | 0.5040 | 0.4347 | — | — | — | — | 0.4323 | 1.0081 | — | — |
| constant | 0.5000 | 0.4307 | 0.0065 | −0.01 | 0.00 | 0.62 | 0.4283 | 0.9989 | 0.5037 | 0.0095 |

**The tree models beat the logistic baseline, by very little.** XGBoost improves NDE by 0.0050
(0.2326 → 0.2376), which is **+2.1% relative**; ROC-AUC by 0.0025; PR-AUC by 0.0022; days earlier
by 0.13 days. For comparison, Component 6's improvement over the best heuristic was NDE +0.0481,
roughly ten times larger.

Two of these numbers deserve reading against a floor: PR-AUC's floor here is **0.4307**, not 0.5,
because that is what a constant score achieves at this prevalence. And `business_as_usual` scores
ROC-AUC 0.5040 — indistinguishable from random within a quarter, which is the Component 5 finding
that makes any reordering worth studying at all.

### 9.2 The mean hides the fold-level picture

| fold | logistic | xgboost | lightgbm | xgb − logistic | winner |
| --- | ---: | ---: | ---: | ---: | --- |
| 2022Q2 | 0.2007 | 0.2102 | 0.2035 | +0.0095 | xgboost |
| 2022Q3 | 0.2069 | 0.2083 | 0.2110 | +0.0014 | lightgbm |
| 2022Q4 | 0.2044 | 0.1936 | 0.1953 | −0.0108 | **logistic** |
| 2023Q1 | 0.1699 | 0.1747 | 0.1931 | +0.0048 | lightgbm |
| 2023Q2 | 0.2272 | 0.2222 | 0.2205 | −0.0051 | **logistic** |
| 2023Q3 | 0.2644 | 0.2486 | 0.2518 | −0.0158 | **logistic** |
| 2023Q4 | 0.2303 | 0.2477 | 0.2540 | +0.0174 | lightgbm |
| 2024Q1 | 0.2012 | 0.2290 | 0.2246 | +0.0277 | xgboost |
| 2024Q2 | 0.1987 | 0.2460 | 0.2442 | +0.0473 | xgboost |
| 2024Q3 | 0.2379 | 0.2280 | 0.2266 | −0.0099 | **logistic** |
| 2024Q4 | 0.2268 | 0.2174 | 0.2121 | −0.0094 | **logistic** |
| 2025Q1 | 0.2926 | 0.3257 | 0.3052 | +0.0330 | xgboost |
| 2025Q2 | 0.3222 | 0.3198 | 0.2946 | −0.0024 | **logistic** |
| 2025Q3 | 0.2578 | 0.2604 | 0.2724 | +0.0026 | lightgbm |
| 2025Q4 | 0.2714 | 0.2545 | 0.2336 | −0.0169 | **logistic** |
| 2026Q1 | 0.2048 | 0.2182 | 0.2127 | +0.0134 | xgboost |
| 2026Q2 | 0.2364 | 0.2342 | 0.2477 | −0.0022 | lightgbm |

**Fold wins: logistic 7, xgboost 5, lightgbm 5.** The logistic baseline wins the plurality of
folds while losing the mean. The tree models win by more when they win (+0.047 at 2024Q2) than
they lose by when they lose (−0.017 at 2025Q4), which is what produces the positive average.

This is the single most important table in the document. Reporting "XGBoost improves NDE by 2.1%"
without it would be true and misleading in the same way Component 6's findings warned about the
2015 result: a mean that conceals how often the ordering reverses.

### 9.3 The improvement is inside the seasonality band

Component 5's time-invariance sensitivity re-draws labels under a de-trended monthly effect
(1,000 replications, seed 20260816):

| model | observed NDE | p05 | p95 | label flip rate |
| --- | ---: | ---: | ---: | ---: |
| xgboost_class_weighted | 0.2390 | 0.2239 | 0.2459 | 0.0192 |
| xgboost | 0.2376 | 0.2224 | 0.2444 | 0.0191 |
| lightgbm | 0.2355 | 0.2201 | 0.2419 | 0.0189 |
| logistic_regression | 0.2326 | 0.2160 | 0.2374 | 0.0185 |
| cdph_2015_approximation | 0.2119 | 0.1965 | 0.2178 | 0.0185 |
| prior_canvass_priority_rate | 0.1845 | 0.1720 | 0.1922 | 0.0171 |

The logistic model's observed 0.2326 sits comfortably inside XGBoost's [0.2224, 0.2444] redraw
interval. **The Component 6 → Component 7 gap is smaller than the variation induced by
seasonality alone.** The heuristic-to-logistic gap is not — 0.1845 sits well below 0.2160.

That is the honest characterisation: Component 6's improvement over heuristics survives this
sensitivity; Component 7's improvement over Component 6 does not clearly survive it.

### 9.4 COVID shift (1 fold) — the ordering depends on the metric

| model | ROC-AUC | PR-AUC | NDE | days earlier | median | P@k_1day | found later |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lightgbm | **0.6292** | 0.6321 | **0.2585** | +26.24 | 21.00 | 0.8182 | 0.4508 |
| xgboost_class_weighted | 0.6291 | 0.6309 | 0.2583 | +26.06 | 14.00 | 0.8182 | 0.4632 |
| xgboost | 0.6286 | 0.6287 | 0.2572 | +25.86 | 14.00 | 0.7727 | 0.4640 |
| logistic_regression_no_scheduling | 0.6286 | 0.6200 | 0.2571 | +25.60 | 13.00 | 0.7273 | 0.4702 |
| logistic_regression | 0.6256 | **0.6328** | 0.2512 | +25.03 | 14.00 | **0.9545** | 0.4665 |
| cdph_2015_approximation | 0.6221 | 0.6113 | 0.2442 | +23.69 | 9.00 | 0.6818 | 0.4737 |
| days_since_last_canvass | 0.5842 | 0.5866 | 0.1704 | +14.18 | 12.00 | 0.9091 | 0.4676 |

**Under distribution shift the ranking of models depends on which metric you read.** LightGBM
takes NDE and ROC-AUC; the logistic model takes PR-AUC (0.6328, the highest of any model) and
precision@k_1day (0.9545, by a wide margin). Component 6 measured an ordering inversion on this
fold; Component 7 measures a metric-dependent ordering, which is the same lesson in a different
shape.

Caveats that must travel with this table: it is **one fold**, `k_1_day` is 22 slots so P@k is
extremely noisy, and days-earlier has SD ≈ 208 against a mean of 26. The fold exists to answer
"does the ranking survive a regime change", not "which model is best".

### 9.5 Probability quality — raw, and left uncorrected

| fold set | model | Brier | ECE | MCE |
| --- | --- | ---: | ---: | ---: |
| quarterly | xgboost | 0.2379 | **0.0621** | 0.1741 |
| quarterly | logistic_regression | 0.2382 | 0.0635 | **0.1664** |
| quarterly | lightgbm | 0.2383 | 0.0644 | 0.1755 |
| quarterly | xgboost_class_weighted | 0.2417 | 0.0836 | 0.1845 |
| covid_shift | logistic_regression | 0.2522 | **0.1124** | 0.1827 |
| covid_shift | xgboost | 0.2552 | 0.1253 | 0.2147 |
| covid_shift | xgboost_class_weighted | 0.2551 | 0.1241 | 0.1923 |
| covid_shift | lightgbm | 0.2606 | 0.1518 | 0.1926 |

On the quarterly folds the boosted probabilities are **not** worse calibrated — XGBoost's ECE is
marginally better than the logistic model's, matching §7's pre-registered probe and contradicting
the expectation carried in HANDOFF.md. Under shift the expectation holds: LightGBM's ECE is 0.1518
against the logistic model's 0.1124.

**Nothing here corrects any of it.** These are raw `predict_proba` outputs; Component 9 owns
calibration, which is why every fit leaves the calibration window untouched.

### 9.6 The class-weighting ablation

`xgboost_class_weighted` reuses XGBoost's frozen parameters with
`scale_pos_weight = (1−p)/p` from each fold's own training prevalence. It has the **best quarterly
NDE of any model** (0.2390) and the best P@k_1day (0.6629).

It is still not the default, and this is the tradeoff §17 of the brief asked to be made explicit:
weighting buys NDE +0.0014 over unweighted XGBoost and costs **ECE 0.0836 versus 0.0621**, a 35%
degradation in the probability scale Component 9 has to calibrate. Measured prevalence is 52.52%,
so there is no imbalance being corrected — the weight is distorting a well-balanced problem to buy
a ranking margin smaller than the seasonality band. Adopting it would be tuning until something
wins.

SMOTE was never tested. At 52.52% prevalence there is nothing to resample, and synthetic minority
rows would corrupt both the probability scale and the temporal structure the whole project rests on.

## 10. What did not work, and what was blocked

**Blocked — inspector-effect modelling.** The dataset publishes 22 columns and none identifies an
inspector, so a mixed-effects model with inspector as a random intercept and any marginalisation
over an inspector effect are undefined. Proxies (violation-text verbosity, ward, day-of-week) were
considered and refused. See ADR 0019 and `tests/test_boosting_inspector_blocked.py`.

**Deferred — SHAP.** Component 11 owns attribution. Component 7 emits native split-gain
importances as a diagnostic only, and the data contract says so: with a condition number of 71.8
and one feature pair at 0.9888 correlation, a tree distributes credit between collinear features
according to which it happened to split on first.

**Deferred — calibration (Component 9), fairness (Component 12), neural baseline (Component 8),
ensembling.** Stacking and rank averaging come after the individual model stages; blending before
each member is understood would hide which one carries the result.

**Did not work as expected:** the pre-registered expectation that boosted probabilities would be
worse calibrated failed on the quarterly folds (§7, §9.5). Recorded rather than quietly dropped.

**Not attempted:** deeper trees. Every study chose `max_depth` 3–4 from a range of 3–10, so the
search itself reports that the data does not support deep interactions.

## 11. Runtime

| stage | wall clock |
| --- | --- |
| profiling (`profile_boosting.py`, 7 profiles) | ~35s |
| tuning, 400 trials over 4 studies | 563.8s |
| training, 54 fits over 18 folds | 21.4s (fitting 21.4s) |
| evaluation, 18 folds, 9 models, 1000 sensitivity reps | 68.4s |
| Component 7 test suite | ~55s |

Single-threaded by design (`n_jobs=1`), which is what makes the fits bit-reproducible. On the
widest 53,844-row window one 200-round XGBoost fit takes 3.57s and LightGBM 0.66s.

## 12. Limitations

1. **The improvement over Component 6 is small and not clearly outside the seasonality band.**
   NDE +0.0050 against a redraw interval that spans 0.022.
2. **The logistic model wins 7 of 17 folds.** A mean improvement is not a per-quarter improvement.
3. **42.89% of violations are still surfaced later** under the XGBoost ranking, against 43.24%
   under Component 6. Effectively unchanged.
4. **Days-earlier has SD 35.5 against a mean of 5.8** on quarterly folds, and SD 208 against a
   mean of 26 on the shift fold. The mean alone would be misleading in exactly the way the 2015
   published result was.
5. **`covid_shift` is one fold**, tuned on two inner folds. Its numbers are a robustness
   observation, not a measurement.
6. **The two fold sets use different hyperparameters**, so a quarterly-versus-shift comparison
   confounds the regime with the parameter set. They are reported separately and never averaged.
7. **Inspector variation is unquantifiable here**, so the gap between "a violation was cited" and
   "the establishment was unsafe" cannot be characterised.
8. **Only establishments that were actually inspected are evaluated.** The estimand is a
   re-ordering of inspections that occurred.
9. **The simulation is retrodictive.** Labels are held fixed; nothing here shows a violation would
   have been found had the inspector arrived earlier.
10. **Probabilities are uncalibrated** and must not be read as risk levels until Component 9.
11. **Determinism holds within a fixed library set only.** A version bump may move every number.
12. **Single-threaded fitting will not scale** to a snapshot an order of magnitude larger.

## 13. Reproducing

```bash
uv run python scripts/profile_boosting.py                    # §3-§7
uv run sentinel tune-boosting --trials 100 --report          # §8; then freeze into definitions.py
uv run sentinel train-boosting --report                      # §9
uv run sentinel evaluate --predictions data/processed/predictions/boosted_predictions_<stamp>.parquet --report
```

`train-boosting` does not require re-running the search: the parameters are frozen literals in
`boosting/definitions.py`, and `TUNED_PARAMS_PROVENANCE` names the study artifact that produced
them.

## 14. What Component 8 should do next

Component 8 is the neural baseline. Three things from this component bear directly on it.

**The ceiling is probably the representation, not the estimator.** XGBoost and LightGBM finish
0.0021 apart in NDE after 200 tuned trials, and both finish 0.005 above a penalised GLM. Two very
different nonlinear learners agreeing that closely with a linear one is evidence that the 26
features carry roughly this much signal and no more. A neural network is a third nonlinear learner
on the same 26 features; the prior should be that it lands in the same place.

**Entity embeddings are the one genuinely new capability.** The thing Component 8 can do that
neither of these could is learn a representation of `establishment_id` — which is exactly what
Component 7 cannot do and Component 4 deliberately excludes. That is where a real difference would
come from, and also where the leakage risk is highest: an establishment embedding fitted across
folds would carry future information about that establishment backwards.

**Reuse the tuning protocol, do not reinvent it.** `boosting.tuning` derives its region from the
fold definitions and refuses one that reaches a test window; the same machinery applies to a
learning-rate sweep. And keep the row sort: a mini-batched optimiser has the same order dependence
a booster does, only worse.
