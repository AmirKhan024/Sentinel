# Component 11 — explainability findings

Measured on the 2026-08-25 production run: `sentinel explain --report`, feature table
`as_of_features_20260816T150313Z.parquet` (57,727 rows), 18 folds, 4 supported models.

**Every number here is measured.** Where a claim could not be measured it is not made.

---

## 0. The run

| | |
|---|---|
| models supported / registered | **4 of 5** |
| re-executed fits | 72 (4 models × 18 folds), 438.6 s |
| attribution | 647.3 s |
| **bit-identity gate** | **166,144 / 166,144 test scores identical to the committed artifacts, 0 mismatches** |
| explained predictions | 21,600 (300 per model per fold) |
| attribution values | 648,000 |
| prediction artifacts changed | **none** — sha256 identical before and after |
| checks | 19 error-severity + 1 advisory, **all pass** |
| environment | `blas_threads = unset (library default)`, `torch_threads = 1`, CPU |

Artifact sizes: `explanation_values` 648,000 rows / 4.6 MB; everything else under 1 MB.

### Additivity, measured

| method | max residual | mean | frozen tolerance |
|---|---|---|---|
| `linear_shap` | 1.421e-14 | 2.679e-16 | 1e-10 |
| `tree_shap` | 1.662e-06 | 7.143e-08 | 1e-5 |
| `permutation_shap` | **0.0** | 0.0 | 1e-6 |

The permutation residual is exactly zero, which is the telescoping-path property doing what
ADR 0030 says it does — and is exactly why a green additivity check is **not** evidence that
a permutation attribution is accurate. The tree residual came in at 1.66e-06 against a
measured probe-fold maximum of 8.92e-07, which is why the tolerance was frozen at 1e-5 rather
than at 1e-6: the probe fold was not the worst fold.

---

## 1. Global importance — what each model leaned on

Mean `|SHAP|` in log-odds over the 17 quarterly folds, with the fold-to-fold SD and the rank
range beside every mean. Top 8 shown; the artifact carries all 30.

### `xgboost`

| feature | mean abs | SD | mean signed | rank | mean rank | best–worst |
|---|---:|---:|---:|---:|---:|---|
| `prior_canvass_count_code_era` | 0.2810 | 0.0392 | −0.2339 | 1 | 1.24 | 1–2 |
| `days_since_first_inspection` | 0.2042 | 0.0578 | −0.0943 | 2 | 3.06 | 1–6 |
| `days_since_any_inspection` | 0.1499 | 0.0204 | −0.0303 | 3 | 4.24 | 3–7 |
| `prior_reinspection_count` | 0.1420 | 0.0116 | +0.0260 | 4 | 4.29 | 3–7 |
| `prior_canvass_priority_foundation_count` | 0.1341 | 0.0688 | −0.0755 | 5 | 7.59 | 2–14 |
| `prior_canvass_priority_count` | 0.1238 | 0.0257 | −0.0879 | 6 | 6.53 | 2–10 |
| `prior_canvass_inspected_count` | 0.1231 | 0.0204 | −0.0325 | 7 | 6.65 | 3–11 |
| `prior_canvass_pass_w_conditions_count` | 0.1153 | 0.0121 | +0.0143 | 8 | 7.47 | 5–10 |

### `lightgbm`

| feature | mean abs | SD | rank | best–worst |
|---|---:|---:|---:|---|
| `prior_canvass_count_code_era` | 0.3605 | 0.1186 | 1 | 1–3 |
| `days_since_first_inspection` | 0.1830 | 0.0540 | 2 | 1–8 |
| `days_since_any_inspection` | 0.1535 | 0.0268 | 3 | 2–9 |
| `prior_reinspection_count` | 0.1443 | 0.0131 | 4 | 3–7 |
| `prior_canvass_inspected_count` | 0.1424 | 0.0326 | 5 | 2–10 |

### `logistic_regression`

| feature | mean abs | SD | mean signed | rank | best–worst |
|---|---:|---:|---:|---:|---|
| `prior_canvass_count` | 1.2198 | 0.2124 | +0.3256 | 1 | 1–1 |
| `prior_canvass_inspected_count` | 0.8974 | 0.1654 | −0.2483 | 2 | 2–2 |
| **`missing_no_code_era_canvass`** | 0.5943 | 0.0645 | −0.4270 | 3 | 3–5 |
| `prior_inspection_count_any_type` | 0.5585 | 0.0970 | −0.2111 | 4 | 3–5 |
| `prior_canvass_count_code_era` | 0.4584 | 0.0825 | −0.3990 | 5 | 4–6 |

### `neural_numeric_only`

| feature | mean abs | SD | rank | best–worst |
|---|---:|---:|---:|---|
| `prior_canvass_count_code_era` | 0.4292 | 0.0460 | 1 | 1–1 |
| **`missing_no_code_era_canvass`** | 0.2557 | 0.0343 | 2 | 2–5 |
| `prior_canvass_inspected_count` | 0.2147 | 0.0697 | 3 | 2–8 |
| `prior_canvass_priority_foundation_count` | 0.1823 | 0.0718 | 4 | 3–14 |
| `days_since_first_inspection` | 0.1792 | 0.0304 | 5 | 2–7 |

### What this says

**Prior inspection history dominates, in every model.** The top signal is a count of
previous code-era canvasses for three of four models, and a count of previous canvasses for
the fourth. That is reassuring rather than surprising: a model whose top signal was
semantically unrelated to food safety would have been a red flag.

**A missingness indicator is a top-three signal for two models.**
`missing_no_code_era_canvass` — "this establishment has no post-2018 canvass history at all"
— ranks 3rd for the logistic model and 2nd for the network, with a strongly negative mean
signed contribution (−0.427 and −0.196). The absence of a record is one of the most
informative things these models have. That is worth stating plainly because it is invisible
to any importance measure computed on features alone, and it is a direct consequence of
Component 6's decision to build four family indicators rather than dropping the null pattern.

**The tree models never used those indicators, and the linear and neural models leaned on
them heavily.** On a tree-model-only run the advisory check reported three indicators with
zero attribution everywhere; on the full run all 30 representations were used by at least one
model. The boosters route NULLs by a learned default direction instead, which is the same
information reaching the model through a different door.

**Direction matters and is reported separately.** `prior_canvass_count_code_era` has a large
mean absolute contribution (0.281 for xgboost) and a *negative* mean signed contribution
(−0.234): on average, having a longer code-era canvass record **lowers** predicted risk. A
table reporting only `mean_abs_shap` would have hidden that.

---

## 2. The headline finding: near-identical accuracy, substantially different reasoning

Spearman rank correlation of the 30-feature importance rankings, quarterly aggregate,
between every pair of models:

| pair | rank rho | top-10 Jaccard |
|---|---:|---:|
| `lightgbm` vs `xgboost` | **+0.9871** | 0.818 |
| `logistic_regression` vs `neural_numeric_only` | +0.8033 | 0.667 |
| `neural_numeric_only` vs `xgboost` | +0.6822 | 0.667 |
| `lightgbm` vs `neural_numeric_only` | +0.6675 | 0.538 |
| **`logistic_regression` vs `xgboost`** | **+0.4623** | 0.429 |
| **`lightgbm` vs `logistic_regression`** | **+0.4351** | 0.333 |

Component 8 measured these four models landing **within 0.0156 NDE of each other**, and
Component 7 concluded that "the ceiling is the 26-feature representation, not the estimator".

Component 11 shows those four near-identical scores are produced by **materially different
reasoning**. The two boosters agree with each other almost perfectly (ρ = 0.987 — they are
the same algorithm family on the same matrix). The logistic model and the boosters agree at
ρ ≈ 0.44–0.46 and share only 3 to 4 of their top ten features.

This is the component's most interesting result, and it cuts both ways:

- It **strengthens** Component 7's conclusion. If four models reasoning this differently all
  land within 0.016 NDE, the limit really is the information in the 26 features rather than
  the estimator's ability to exploit it.
- It **weakens** any argument from explanation to model choice. There is no "the models agree
  that X matters" story to tell. Two of them agree; the other two do not, and picking the one
  whose attribution table reads most sensibly would be selecting a model on legibility —
  which this component is forbidden to do (ADR 0030's blocked list).

---

## 3. Stability over time

Consecutive-fold rank agreement across the 17 quarterly folds, 16 comparisons per model:

| model | mean rho | min | max | mean top-10 Jaccard | min |
|---|---:|---:|---:|---:|---:|
| `logistic_regression` | 0.9753 | 0.9377 | 0.9933 | 0.776 | 0.538 |
| `xgboost` | 0.9705 | 0.9523 | 0.9884 | 0.886 | 0.818 |
| `lightgbm` | 0.9606 | 0.9392 | 0.9840 | 0.867 | 0.667 |
| `neural_numeric_only` | **0.8914** | **0.7909** | 0.9706 | 0.795 | 0.667 |

First fold to last (`quarterly-2022Q2` → `quarterly-2026Q2`, four years apart):

| model | rank rho | top-10 Jaccard |
|---|---:|---:|
| `lightgbm` | 0.9197 | 0.667 |
| `xgboost` | 0.8883 | 0.818 |
| `logistic_regression` | 0.8839 | 0.667 |
| `neural_numeric_only` | **0.7495** | **0.538** |

**Quarter to quarter the reasoning is stable; over four years it measurably drifts.** Every
model's first-to-last correlation is lower than its mean consecutive correlation, and the
top-ten overlap falls to between 0.54 and 0.82 — so between a third and a half of each
model's top ten features changed over the study period.

**The network is the least stable of the four, and part of that is its own noise.** Its
consecutive correlation (0.891) is meaningfully below the other three (0.96–0.98), and its
per-row attributions are the only approximate ones in the artifact. ADR 0030 measured the
global ranking converging to ρ = 0.9964 against a 64-round reference at the frozen budget,
so the *measurement* noise is around 0.004 — an order of magnitude smaller than the 0.109 gap
between the network and the logistic model. The instability is therefore mostly real, but the
network is also the model whose stability number should be quoted with the most caution.

### Explanation drift, per feature

Features whose importance rank moved by at least 5 positions (`RANK_DRIFT_THRESHOLD`,
declared before the ranks were computed) at some point across the 17 quarterly folds:

| model | features that materially changed | of 30 | largest rank range |
|---|---:|---:|---:|
| `neural_numeric_only` | **27** | 30 | 20 |
| `lightgbm` | 21 | 30 | 19 |
| `xgboost` | 21 | 30 | 13 |
| `logistic_regression` | 18 | 30 | 12 |

The picture is consistent across models: **the top of the ranking holds and the tail churns.**
`prior_canvass_count_code_era` never leaves the top 3 for any tree or neural model;
`prior_canvass_count` is rank 1 on all 17 folds for the logistic model. Meanwhile features in
the 0.02–0.09 importance band routinely move 8 to 19 positions, which is what a rank
comparison of near-tied values does and is not by itself evidence of instability.

Two movements are large enough to be worth naming:

- `xgboost` / `prior_canvass_priority_rate`: rank 16 on the first fold, rank 3 on the last,
  range 13.
- `xgboost` / `prior_canvass_priority_foundation_count`: rank 2 on the first fold, rank 14 on
  the last, range 12.

Those two are near-substitutes — both count serious violations — so this is most likely credit
moving between correlated features rather than a change in what the model attends to.
**That is a hypothesis, not a measurement**: SHAP splits credit between correlated features
and discloses nothing about how, and Component 7 measured a condition number of 71.8 with one
feature pair correlated at 0.9888.

---

## 4. COVID: the models leaned much harder on scheduling

`covid_shift` is reported separately and never averaged into the quarterly figures.

Rank agreement between the quarterly aggregate and the COVID fold:

| model | rank rho | top-10 Jaccard |
|---|---:|---:|
| `xgboost` | 0.9170 | 0.667 |
| `lightgbm` | 0.8795 | 0.667 |
| `neural_numeric_only` | 0.7393 | 0.538 |
| `logistic_regression` | 0.7037 | 0.538 |

The specific change, and it is the same change in three of four models:

| model | `days_since_any_inspection` — quarterly | — COVID | ratio |
|---|---:|---:|---:|
| `xgboost` | 0.1499 (rank 3) | **0.3728 (rank 1)** | 2.5× |
| `neural_numeric_only` | 0.1232 (rank 8) | **0.3546 (rank 1)** | 2.9× |
| `lightgbm` | 0.1535 (rank 3) | 0.3006 (rank 2) | 2.0× |

**Under the regime shift, the scheduling-gap feature became the models' dominant signal.**

This lines up exactly with what Component 6 measured and could not explain. On `covid_shift`
the model *ordering inverted*: `logistic_regression_no_scheduling` — the ablation that drops
`days_since_any_inspection` and loses on the quarterly folds — **won** under shift (NDE 0.2571
vs 0.2512). Component 6's reading was that the feature partly encodes scheduling policy, and
that a model leaning on it is the more fragile one when the policy breaks.

Component 11 supplies the missing half of that argument: the models did not merely *have* that
feature during COVID, they **leaned on it two to three times harder than usual**. The
mechanism Component 6 hypothesised from an ablation is now visible directly in the
attributions.

The logistic model is the exception, and instructively so: its `days_since_any_inspection`
importance barely moves, and its top two features (`prior_canvass_count`,
`prior_canvass_inspected_count`) *grow* — 1.22 → 1.89 and 0.90 → 1.49. A linear model cannot
reallocate its reliance the way a tree ensemble can; its coefficients are fixed and only the
feature distribution moved.

**Caveat, restated: this is one fold, with no variance estimate**, over a period when the
scheduling policy itself broke. Nothing here supports a claim about how the models would
behave in a different shift.

---

## 5. A local explanation, end to end

Two real rows from the artifact — `xgboost`, fold `quarterly-2026Q2`, selected by predicted-score
quantile and by nothing else.

### The high-risk case: inspection 2636591

```text
base value (expected log-odds)   +0.1453
this prediction   (log-odds)     +0.5314
base + sum(shap)                 +0.5314     residual 8.15e-09

model score before calibration    0.6298
calibrated risk probability       0.5339  (platt)

base model trained through   2025-12-31
calibrator fitted through    2026-03-31
available from               2026-04-01

increasing risk                                          feature value
  +0.1802  days_since_first_inspection                    1032
  +0.1669  prior_canvass_priority_count                   NULL
  +0.1668  prior_canvass_priority_rate                    NULL
  +0.1395  prior_canvass_count_current_name               0
  +0.1327  prior_canvass_inspected_count                  0

reducing risk
  -0.2623  days_since_last_canvass                        NULL
  -0.1475  prior_canvass_count                            0
  -0.1175  prior_canvass_pass_w_conditions_count          0
  -0.0824  prior_reinspection_count                       1
```

**In plain language: this establishment is high risk because Sentinel knows almost nothing
about it.** It has never been canvassed — `prior_canvass_count = 0`, and the priority-history
columns are NULL rather than zero — and it has existed for 1,032 days. Three of its five
risk-increasing factors are *absences of evidence*. The model has learned that an
establishment with a long life and no inspection record is a poor bet, which is a sensible
thing to have learned and is not a claim that inspecting it would make it safer.

Note the honesty the artifact preserves: `feature_value = NULL`, not 0. For a tree model a
NULL is a real observation routed by a learned default direction, and recording it as zero
would have made "never canvassed" indistinguishable from "canvassed zero times with a
recorded rate of 0.0".

### The low-risk case: inspection 2638553

```text
base value                       +0.1453
this prediction                  -1.2584     residual 4.93e-07
model score before calibration    0.2213
calibrated risk probability       0.2769  (platt)

reducing risk                                            feature value
  -0.3901  prior_canvass_count_code_era                   6
  -0.2855  days_since_first_inspection                    5858
  -0.2166  prior_canvass_priority_rate                    0.1667
  -0.1905  prior_canvass_fail_rate                        0
  -0.1723  days_since_any_inspection                      237

increasing risk
  +0.2167  prior_canvass_pass_w_conditions_count          4
  +0.1788  prior_inspection_count_any_type                32
  +0.0608  prior_complaint_count                          18
```

The mirror image: a sixteen-year-old establishment with six code-era canvasses, a 16.7%
priority-citation rate, a **zero** fail rate, and a visit only 237 days ago. Long record, good
record, recently seen. The factors pushing the other way are its sheer volume of history — 32
inspections, 18 complaints, four passes-with-conditions — which the model reads as mild
counter-evidence.

**Both cases show the calibration boundary working.** The high-risk case's model score is
0.630 and its calibrated probability is 0.534; the low-risk case's are 0.221 and 0.277. The
attributions decompose the first number. Component 9's Platt map produces the second, and it
moves both toward the base rate — which is what a correction for an underconfident model does.
The ranking is untouched.

---

## 6. Model support

| model | status | method | exact |
|---|---|---|---|
| `logistic_regression` | supported | `linear_shap`, closed form | yes |
| `xgboost` | supported | `tree_shap`, native `pred_contribs` | yes |
| `lightgbm` | supported | `tree_shap`, native `pred_contrib` | yes |
| `neural_numeric_only` | supported | `permutation_shap`, 8 antithetic rounds | **no** |
| `xgboost_chain_embeddings` | **unsupported** | — | — |

`xgboost_chain_embeddings` carries **no values, no cases and no importance rows** — nulls, not
zeros. Its fitted booster is reachable only through `neural.embed._scorer_for`, a private
process-local stash, and Component 8 is closed. ADR 0031 records the measurement and the
four-line public accessor that would lift the restriction, proposed and not taken.

It is also the model with the best PR-AUC in the project, so **its reasoning is the one piece
of evidence Component 11 cannot supply.** That is a real gap, not a formality.

---

## 7. Limitations

1. **Not causality.** A SHAP value says how the model used a feature. Nothing here says that
   changing a feature would change an outcome.
2. **Correlated features share credit invisibly.** Condition number 71.8, one pair at 0.9888
   (Component 7). The two large rank movements in §3 are most likely this, and SHAP cannot
   tell us.
3. **The network's per-row values are approximate.** Global ranking stable to ρ ≈ 0.996;
   median per-value error ≈ 1% of the largest attribution. Its stability number in §3 is the
   one to quote most cautiously.
4. **Additivity is not accuracy for the permutation method** — its residual is exactly 0.0 at
   any round count.
5. **300 of 41,536 rows per model per fold.** Enough for stable global rankings; not a census,
   and a rare local pattern could be absent entirely.
6. **One of five models is unexplained**, and it is the best-PR-AUC one.
7. **`covid_shift` is one fold with no variance estimate.**
8. **17 folds are not independent samples** — the same establishments recur — so every SD here
   is a fold-to-fold spread, not a confidence interval. Component 5 makes the same warning.
9. **The explanation inherits its subject's limitations.** Sentinel observes violations
   *cited*, not committed; the dataset has no inspector field (ADR 0019), so the gap between
   "was cited" and "was unsafe" cannot be characterised. These are attributions of a model of
   citations.
10. **Component 11 selects no model**, and nothing here should be read as evidence for one.
    §2 in particular is a reason to be *more* cautious about model choice, not less.

---

## 8. Figures

Twenty-nine figures in `docs/analysis/figures/`, all `explain_*`, all regenerable from the
persisted tables -- four models x (importance, rank stability, beeswarm, COVID comparison,
three local cases) plus one cross-model drift panel:

| figure | question |
|---|---|
| `explain_global_importance_<model>_quarterly.png` | what did this model lean on, with the fold-to-fold spread? |
| `explain_rank_stability_<model>_quarterly.png` | did its top-10 ranks hold across 17 quarters? |
| `explain_beeswarm_<model>_quarterly-2026Q2.png` | direction and shape, not just average magnitude |
| `explain_drift_spearman_quarterly.png` | consecutive-fold rank agreement, all models, over time |
| `explain_covid_comparison_<model>.png` | ordinary quarters vs the regime shift, side by side |
| `explain_local_<model>_<tier>_quarterly-2026Q2.png` | one prediction decomposed, high / medium / low |

Every figure carries a caption stating its caveat. The local-case captions state that the case
was selected by predicted risk and never by whether the model was right; the drift caption
states that the neural series is a permutation estimate.

---

## 9. Reproducing

```bash
uv run python scripts/profile_explanations.py     # read-only, fixes the frozen constants
uv run sentinel explain --report                  # ~18 min; do NOT set OMP_NUM_THREADS
```

The thread-count warning is load-bearing: ADR 0026 records a BLAS thread override moving
`logistic_regression` scores by 1e-13 and correctly failing the bit-identity gate.
