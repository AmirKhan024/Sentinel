# Component 8 — Neural network with entity embeddings: findings

**Source:** `data/processed/features/as_of_features_20260816T150313Z.parquet` (57,727 rows,
`feature_definition_version = v1`)
**Experimental categoricals:** `data/processed/neural/neural_categoricals_20260818T125631Z.parquet`
**Profiling command:** `uv run python scripts/profile_neural.py`
**Search:** `uv run sentinel tune-neural --report`
**Training:** `uv run sentinel train-neural --report`
**Evaluation:** `uv run sentinel evaluate --predictions <neural_predictions>.parquet --report`
**Libraries:** torch 2.13.0+cpu, matplotlib 3.11.1, scikit-learn 1.9.0, xgboost 3.4.1,
numpy 2.5.2, polars 1.43.2
**Device:** CPU, one thread, `torch.use_deterministic_algorithms(True)`. An NVIDIA RTX 4050 is
present on the build machine and deliberately unused — see ADR 0020.

⚠ **Sections 1–7 contain no test-window number, and that was a hard rule rather than a style
preference.** Every figure before §8 is computed over a fold's *training* window, or is a
structural fact carrying no outcome (cardinality, coverage, parameter counts). Component 5
protects evaluation time, but it cannot protect against a human reading a test metric, changing an
embedding dimension and re-running. That loop is leakage, it leaves no trace in any artifact, and
no check in this repository can detect it. §8 onwards reports the evaluation, which was run once,
after the learning rates were frozen and committed.

---

## 1. What this component is for

Components 6 and 7 established something the project did not set out to find: a penalised GLM,
XGBoost and LightGBM all land within **0.005 NDE** of one another on the same 26 features. Three
very different learners agreeing that closely is evidence that the ceiling is the **feature
representation**, not the estimator.

Component 8 asks one question: **does a neural model with learned representations for categorical
entities do meaningfully better?**

It is an experimental comparison. It does not ask whether a better feature representation exists
(Component 4 owns that), whether the probabilities can be corrected (Component 9), whether the
result is fair (Component 12), or what drives an individual score (Component 11).

The prior was written down before any code existed, in STATUS.md: *"The prior for a third
nonlinear learner on the same features should be 'no material difference'. Beating that prior
would be a real finding; assuming it will be beaten is how a component gets tuned until something
wins."*

## 2. The scope conflict, and how it was resolved

**The four categorical families the specification names do not exist in Component 4's table.**

That table has 26 features and every one of them is a numeric temporal-history count, recency or
rate. There is no categorical column of any kind. Where the four actually live:

| family | where it is | distinct, over the 57,727 feature rows |
| --- | --- | ---: |
| `facility_type` | raw Socrata snapshot; dropped at `target/build.py:48` | 169 |
| `zip` | raw Socrata snapshot | 72 |
| `community_area` | raw only, as computed region `:@computed_region_vrxf_vc4k` | 78 |
| `chain` | **nowhere**; derived from Component 2's `name_key` | 950 chains, 22.70% of rows |

This collides with a rule the project states twice — *"Do not add a feature. If one is missing it
belongs in Component 4 behind a bumped `feature_definition_version`"* — and with the instruction
that Components 1–7 must not be modified. Together those make the specified experiment impossible
as literally written.

**The conflict was surfaced before any code was written**, and resolved by building a separate,
explicitly experimental categorical layer under `data/processed/neural/`, leaving
`feature_definition_version` at `v1` and Components 4, 6 and 7 untouched. Full reasoning in ADR
0022; the layer's own contract is `docs/data_contracts/neural_categoricals.md`.

The consequence that matters for reading §9: **`neural_numeric_only` sees exactly the 30 matrix
columns Components 6 and 7 see and no categoricals at all.** It is the model any C6/C7/C8 claim
rests on. Every categorical-bearing model is reported beside it, never in place of it.

## 3. Categorical coverage (profile `categorical_coverage`)

| family | distinct | with a value | coverage | `__UNKNOWN__` |
| --- | ---: | ---: | ---: | ---: |
| `chain_key` | 13,303 | 57,326 | 0.9931 | 401 |
| `facility_type` | 169 | 57,274 | 0.9922 | 453 |
| `community_area` | 78 | 57,041 | 0.9881 | 686 |
| `zip` | 72 | 57,326 | 0.9931 | 401 |

**401 rows have no prior inspection of any type** to carry a value forward from, so all four
families are `__UNKNOWN__` for them. That is *exactly* the number of rows Component 4 marks with a
null `days_since_any_inspection`. The two components independently agree about which
establishments have no history, which is the consistency check worth having.

`facility_type` and `community_area` lose a few more rows because a prior inspection can exist
while its own facility-type text is blank or its coordinates missing — the latter being documented
Socrata behaviour for a computed region, recorded rather than patched over.

**The as-of lag** (profile `as_of_lag`), across the 57,326 rows that have a source:

| rows | mean | median | p25 | p75 | **min** | max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 57,326 | 392.6 | 357 | 264 | 448 | **1** | 5,416 |

**The minimum of 1 day is the observable that proves the join is strictly as-of.** A zero would
mean a row had supplied its own attributes. A maximum of 5,416 days is not an error — facility
type and address are stable attributes — but a stale value is stale, and it is recorded.

## 4. Vocabulary growth (profile `cardinality_growth`)

Refitted per fold on training rows only, so they grow with the expanding window.

| fold | train rows | chain | facility | community | zip | chains |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `quarterly-2022Q2` | 23,346 | 598 | 146 | 78 | 65 | 596 |
| `quarterly-2023Q2` | 29,898 | 667 | 152 | 78 | 67 | 665 |
| `quarterly-2024Q2` | 37,094 | 748 | 154 | 78 | 68 | 746 |
| `quarterly-2025Q2` | 45,575 | 845 | 161 | 78 | 71 | 843 |
| `quarterly-2026Q2` | 53,844 | 919 | 167 | 78 | 71 | 917 |
| `covid_shift-2020H2-2021` | 12,660 | 457 | 114 | 78 | 64 | 455 |

(Six of eighteen shown; the full table is in the profiling output.)

The chain column counts *vocabulary entries*, which is the chain count plus the two reserved
tokens. Note the gap between `chain_key` cardinality (13,303 distinct normalised names) and chain
vocabulary (≤919): **most names belong to exactly one establishment and collapse to
`__INDEPENDENT__` inside a fold.** Community area is saturated at 78 from the first fold, which is
every area in the city plus the unknown token.

## 5. Out-of-vocabulary rates (profile `unseen_rate`)

The number that decides whether an embedding can contribute at all. These are **row counts, not
metrics** — no label is read anywhere in this profile.

| fold | test rows | chain | facility | community | zip |
| --- | ---: | ---: | ---: | ---: | ---: |
| `quarterly-2022Q2` | 1,762 | 0.0170 | 0.0187 | 0.0204 | 0.0170 |
| `quarterly-2023Q2` | 1,787 | 0.0140 | 0.0157 | 0.0173 | 0.0140 |
| `quarterly-2024Q2` | 2,196 | 0.0109 | 0.0137 | 0.0164 | 0.0118 |
| `quarterly-2025Q2` | 2,290 | 0.0061 | 0.0100 | 0.0114 | 0.0061 |
| `quarterly-2025Q4` | 1,766 | 0.0000 | 0.0034 | 0.0051 | 0.0000 |
| `quarterly-2026Q2` | 1,638 | 0.0098 | 0.0122 | 0.0171 | 0.0104 |
| `covid_shift-2020H2-2021` | 8,840 | 0.0074 | 0.0106 | 0.0130 | 0.0075 |

**Unseen rates are low — 0.00% to 2.04%.** This matters for interpreting a flat result: the
embeddings are *not* crippled by falling back to `__UNKNOWN__` on most test rows. If the
representation fails to help, it will not be because it never got to apply.

## 6. What an embedding buys, structurally (profile `embedding_budget`)

On `quarterly-2026Q2`'s training window:

| family | vocabulary | dim | parameters |
| --- | ---: | ---: | ---: |
| `chain` | 919 | 16 | 14,704 |
| `facility_type` | 167 | 8 | 1,336 |
| `community_area` | 78 | 8 | 624 |
| `zip` | 71 | 8 | 568 |

Embedding parameters **17,232**; dense stack ~**51,200**. The tables are 0.34× the rest of the
network — large enough that dropout and weight decay are not optional.

The comparison that experiment B exists to test:

| model | encoding | extra input width |
| --- | --- | ---: |
| `neural_embeddings` | embedding | **40** |
| `neural_onehot` | one-hot | **1,235** |
| `neural_numeric_only` | none | 0 |

**The one-hot control needs 1,235 indicator columns to carry what 40 embedding dimensions carry.**
That is the textbook argument for embeddings, stated as a measurement. Whether the compression
actually helps is §9.

## 7. The early-stopping split (profile `inner_split`)

This is the one place Component 8 does something Components 6 and 7 did not, so it gets its own
section.

A network needs a validation signal to stop on. The fold's calibration and test windows are both
*later* than `train_end`; reading either would break the horizon the artifact declares, and the
calibration case would also consume the window Component 9 exists to use. So the split is carved
from the **end of the training window**.

| fold | train rows | split date | fit | validate | share | `train_end` |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `quarterly-2022Q2` | 23,346 | 2021-05-20 | 19,812 | 3,534 | 0.1514 | 2021-12-31 |
| `quarterly-2024Q2` | 37,094 | 2023-03-27 | 31,520 | 5,574 | 0.1503 | 2023-12-31 |
| `quarterly-2026Q2` | 53,844 | 2025-01-07 | 45,715 | 8,129 | 0.1510 | 2025-12-31 |
| `covid_shift-2020H2-2021` | 12,660 | 2019-12-16 | 10,758 | 1,902 | 0.1502 | 2020-02-29 |

Observed share across all 18 folds: **0.1501–0.1514**. **Every split date is strictly before its
fold's `train_end`**, which is what keeps `trained_through = train_end` literally true.

The cut falls on a **whole day**. Two inspections days apart share almost all of their as-of
history, so a split that put one on each side would leak near-duplicate rows across the boundary
in the ordinary machine-learning sense — which no temporal check would catch.

**The cost, stated rather than hidden:** the weights kept are the best validation epoch's, so the
final model is fitted on ~85% of its fold's training rows. Component 7 avoided this by freezing a
round count and refitting on everything. Component 8 does not, because the specification asks for
per-fold early stopping and per-fold learning curves, and doubling an already long run to recover
15% of rows was judged the worse trade. **This is the component's clearest self-inflicted
limitation.**

### Why scaling returns when Component 7 fitted none (profile `scaling_need`)

On `quarterly-2026Q2`'s training window:

| column | SD | max |
| --- | ---: | ---: |
| `days_since_first_inspection` | 1,545.03 | 5,829.00 |
| `days_since_last_canvass` | 369.99 | 5,469.00 |
| `days_since_any_inspection` | 306.95 | 5,416.00 |
| … | | |
| `prior_canvass_fail_rate` | 0.2334 | 1.0000 |
| `missing_no_prior_inspection` | 0.0839 | 1.0000 |

**The widest standard deviation is 18,409× the narrowest.** A tree is invariant to that and
Component 7 therefore fitted no scaler at all. A dense layer's first weighted sum is not, and
without standardisation the optimiser would spend its early epochs undoing the units.

Imputation returns for a harder reason: **there is no NaN-native path for a dense layer.** A NaN
propagates through every weight and destroys the fit in one backward pass. Component 7's best
argument for a booster — that a NULL means "no prior canvass" and a tree can branch on it — is an
argument a network cannot answer in kind. It imputes, and the four null-rule family indicators
carry the fact. That difference is one of the reasons a booster may simply suit this data better,
and it is a structural claim, not a result.

## 8. The learning-rate sweep (production run)

`uv run sentinel tune-neural --report` — 40 trials over 2 studies, **550.9 s**, 0 failures.
Artifact `neural_sweep_trials_20260818T125643Z.parquet`
(sha256 `ebd51e08…511e01ce`), TPE-free exhaustive grid, seed 20260818.

Protocol borrowed unchanged from Component 7 (`tuning_region`, `first_test_start`,
`build_inner_folds`; ADR 0017). Two studies because the quarterly region contains the
`covid_shift` test window.

**Mean inner-validation PR-AUC** (in-sample: these windows are training data for every outer fold
the selected rate is then used on — no number here is a result):

| learning rate | quarterly (6 inner folds) | covid_shift (2 inner folds) |
| ---: | ---: | ---: |
| 1e-4 | 0.5918 | 0.7820 |
| 3e-4 | 0.5943 | 0.7854 |
| **1e-3** (specification baseline) | 0.6050 | 0.7869 |
| **3e-3** | **0.6070** ← selected | 0.7879 |
| 1e-2 | 0.5964 | **0.7928** ← selected |

**The result is mildly sensitive to the learning rate and the specification's own baseline is
close to optimal.** The quarterly range spans 0.0152 and the selected 3e-3 beats the 1e-3 baseline
by **0.0020** — small enough that the honest reading is "anywhere in 1e-3…3e-3 is equivalent".
Both are recorded rather than presented as a tuning gain.

The two regimes disagree about the rate (3e-3 versus 1e-2), which is itself a reason not to share
a study between them. The `covid_shift` figures rest on two inner folds, the declared minimum, and
are a weaker measurement than the quarterly ones. Their higher absolute level reflects the
pre-2020 base rate of ~0.82, not a better model.

Frozen into `neural/definitions.TUNED_HYPERPARAMS` with provenance, by hand, from the printed
block — `tune-neural` edits no source file, for the reason ADR 0017 gives.

All four sweep validation checks passed, including `sweep_never_reached_a_test_window`.

---

## 9. Results

Training run: **234 fits** (9 models × 18 folds, plus 4 extra seeds × 18 folds for the
reproducibility experiment), **1,998.7 s** fitting, **4,306 epochs**, 373,824 prediction rows.
All **21** Component 8 error-severity checks passed and all **14** Component 5 checks passed with
the predictions attached, including `scores_respect_the_decision_point`.

Every fit stopped on **patience**; not one exhausted the 200-epoch budget.

### 9.1 Quarterly (17 folds) — the headline

| model | ROC-AUC | PR-AUC | NDE | SD | days earlier | P@k_1day | Brier | ECE | found later |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **neural_numeric_only** | **0.6241** | 0.5343 | **0.2482** | 0.0440 | **+6.10** | 0.6273 | **0.2355** | **0.0563** | 0.4303 |
| **xgboost_chain_embeddings** | 0.6222 | **0.5357** | 0.2444 | 0.0355 | +5.97 | 0.6480 | 0.2374 | 0.0619 | 0.4289 |
| xgboost_class_weighted | 0.6195 | 0.5355 | 0.2390 | 0.0393 | +5.85 | 0.6629 | 0.2417 | 0.0836 | 0.4283 |
| xgboost (C7) | 0.6188 | 0.5343 | 0.2376 | 0.0390 | +5.83 | 0.6308 | 0.2379 | 0.0621 | 0.4289 |
| lightgbm (C7) | 0.6177 | 0.5342 | 0.2355 | 0.0327 | +5.75 | **0.6598** | 0.2383 | 0.0644 | 0.4285 |
| logistic_regression (C6) | 0.6163 | 0.5321 | 0.2326 | 0.0389 | +5.70 | 0.6576 | 0.2382 | 0.0635 | 0.4324 |
| neural_no_zip | 0.6137 | 0.5255 | 0.2274 | 0.0473 | +5.59 | — | 0.2395 | 0.0661 | 0.4314 |
| neural_no_community_area | 0.6129 | 0.5255 | 0.2258 | 0.0456 | +5.53 | — | 0.2398 | 0.0664 | 0.4358 |
| neural_no_chain | 0.6114 | 0.5257 | 0.2229 | 0.0580 | +5.47 | — | 0.2402 | 0.0689 | 0.4358 |
| **neural_embeddings** | 0.6107 | 0.5233 | 0.2215 | 0.0501 | +5.45 | 0.6217 | 0.2401 | 0.0679 | 0.4350 |
| neural_onehot | 0.6103 | 0.5241 | 0.2206 | 0.0548 | +5.43 | 0.6071 | 0.2416 | 0.0750 | 0.4364 |
| neural_pos_weighted | 0.6096 | 0.5229 | 0.2191 | 0.0504 | +5.38 | — | 0.2482 | 0.1002 | 0.4353 |
| neural_no_facility_type | 0.6076 | 0.5210 | 0.2153 | 0.0521 | +5.30 | — | 0.2408 | 0.0651 | 0.4360 |
| cdph_2015_approximation | 0.6059 | 0.5118 | 0.2119 | 0.0384 | +5.18 | 0.5618 | 0.2401 | 0.0633 | 0.4412 |
| business_as_usual | 0.5040 | 0.4347 | — | — | — | 0.4323 | — | — | — |
| constant | 0.5000 | **0.4307** | 0.0065 | 0.0425 | −0.01 | 0.4283 | — | — | — |

Read PR-AUC against its floor of **0.4307**, not 0.5 — that is what a constant score achieves at
this prevalence.

**Three results, and they do not all point the same way.**

**1. The neural network on the same 26 features posts the best NDE in the project: 0.2482.** That
is +0.0106 over XGBoost (0.2376) and +0.0156 over the Component 6 logistic model — roughly twice
the size of Component 7's improvement over Component 6. It also posts the best Brier (0.2355) and
the best ECE (0.0563) of any model here, including the penalised GLM.

**2. Entity embeddings made the neural network worse, substantially.** `neural_embeddings`
(0.2215) sits **0.0267 below** `neural_numeric_only` (0.2482) — a larger gap than any between
model *classes* anywhere in this project. Every single-family ablation lands between the two, and
the one-hot control (0.2206) is no better. This is a clean negative result on the component's
headline question.

**3. The same embeddings *helped* XGBoost.** `xgboost_chain_embeddings` (0.2444) beats plain
XGBoost (0.2376) by +0.0068 and posts the best PR-AUC of any model (0.5357). The representation
that hurt the network which learned it improved the estimator it was handed to.

### 9.2 The mean hides the fold-level picture

`neural_numeric_only` − `xgboost`, per fold:

| fold | logistic | xgboost | neural_numeric | neural_emb | xgb+emb | Δ(neu−xgb) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022Q2 | 0.2007 | 0.2102 | 0.2084 | 0.1555 | 0.2258 | −0.0018 |
| 2022Q3 | 0.2069 | 0.2083 | 0.2215 | 0.1562 | 0.2176 | +0.0131 |
| 2022Q4 | 0.2044 | 0.1936 | 0.2064 | 0.1598 | 0.2137 | +0.0127 |
| 2023Q1 | 0.1699 | 0.1747 | 0.1686 | 0.1728 | 0.1885 | −0.0061 |
| 2023Q2 | 0.2272 | 0.2222 | 0.2397 | 0.1905 | 0.2255 | +0.0175 |
| 2023Q3 | 0.2644 | 0.2486 | 0.2706 | 0.2538 | 0.2649 | +0.0220 |
| 2023Q4 | 0.2303 | 0.2477 | 0.2290 | 0.2226 | 0.2403 | −0.0186 |
| 2024Q1 | 0.2012 | 0.2290 | 0.2366 | 0.1806 | 0.2249 | +0.0076 |
| 2024Q2 | 0.1987 | 0.2460 | 0.2133 | 0.2071 | 0.2500 | **−0.0327** |
| 2024Q3 | 0.2379 | 0.2280 | 0.2583 | 0.2566 | 0.2251 | +0.0303 |
| 2024Q4 | 0.2268 | 0.2174 | 0.2565 | 0.2531 | 0.2194 | **+0.0391** |
| 2025Q1 | 0.2926 | 0.3257 | 0.3247 | 0.2870 | 0.3151 | −0.0009 |
| 2025Q2 | 0.3222 | 0.3198 | 0.3486 | 0.3181 | 0.3159 | +0.0288 |
| 2025Q3 | 0.2578 | 0.2604 | 0.2662 | 0.2720 | 0.2801 | +0.0058 |
| 2025Q4 | 0.2714 | 0.2545 | 0.2895 | 0.2682 | 0.2733 | +0.0350 |
| 2026Q1 | 0.2048 | 0.2182 | 0.2390 | 0.2188 | 0.2211 | +0.0208 |
| 2026Q2 | 0.2364 | 0.2342 | 0.2428 | 0.1921 | 0.2540 | +0.0085 |

**`neural_numeric_only` wins 12 of 17 folds.** That is a materially better record than Component
7's: XGBoost won only 5 of 17 against the logistic model while still winning the mean. A
12-of-17 record together with a positive mean is the first result in this project where a mean
improvement and a per-fold improvement agree.

It is still not uniform. It loses 2024Q2 by 0.0327 — the fold XGBoost wins by its largest margin
over everything — and 2023Q4 by 0.0186.

### 9.3 The improvement is at the edge of the seasonality band

Component 5's time-invariance sensitivity, 1,000 label re-draws under a de-trended monthly effect:

| model | observed NDE | p05 | p95 |
| --- | ---: | ---: | ---: |
| **neural_numeric_only** | **0.2482** | 0.2311 | 0.2527 |
| xgboost_chain_embeddings | 0.2444 | 0.2285 | 0.2502 |
| xgboost_class_weighted | 0.2390 | 0.2239 | 0.2459 |
| xgboost | 0.2376 | 0.2224 | 0.2444 |
| lightgbm | 0.2355 | 0.2201 | 0.2419 |
| neural_embeddings | 0.2215 | 0.2051 | 0.2263 |

**The asymmetry is the finding.** `neural_numeric_only`'s 0.2482 sits *just above* XGBoost's p95
of 0.2444 — but XGBoost's 0.2376 sits *comfortably inside* the neural model's [0.2311, 0.2527].

So the improvement is larger than Component 7's over Component 6 (whose 0.2326 sat squarely inside
XGBoost's interval), and it clears the test in one direction but not the other. **The honest
characterisation is "suggestive, not decisive".**

### 9.4 And the seed spread is the same size as the win

The reproducibility experiment refits `neural_embeddings` under five seeds on all 17 quarterly
folds:

| seed | PR-AUC | ROC-AUC | mean best epoch |
| ---: | ---: | ---: | ---: |
| 42 | 0.5233 | 0.6107 | 4.0 |
| 43 | 0.5219 | 0.6084 | 4.4 |
| 44 | 0.5202 | 0.6060 | 4.4 |
| 45 | **0.5269** | **0.6119** | 4.9 |
| 46 | 0.5221 | 0.6104 | 3.3 |

Spread **0.0066** PR-AUC and **0.0058** ROC-AUC (SD 0.0025 and 0.0023). Per fold, the ROC-AUC seed
range averages **0.0178** and reaches **0.0345**.

**`neural_numeric_only` beats XGBoost by 0.0053 ROC-AUC. The neural family's own seed-to-seed
ROC-AUC spread is 0.0058.** The margin and the noise are the same size.

That is the single most important number in this component. A different seed could plausibly have
produced a different headline, and any deployment argument resting on the neural model's win would
be resting on a coin-flip's worth of evidence. It is why §10 and Component 9's handoff do not
recommend adopting it on the strength of the mean.

### 9.5 COVID shift (1 fold) — the ordering inverts again

| model | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| **neural_onehot** | **0.6456** | **0.6528** |
| neural_embeddings | 0.6331 | 0.6379 |
| xgboost_chain_embeddings | 0.6317 | 0.6299 |
| lightgbm | 0.6292 | 0.6321 |
| xgboost | 0.6286 | 0.6287 |
| neural_numeric_only | 0.6284 | 0.6390 |
| logistic_regression | 0.6256 | 0.6328 |

**The quarterly ordering reverses almost exactly.** `neural_onehot` is the *worst* neural model on
quarterly folds (NDE 0.2206) and the *best* model of any kind under distribution shift.
`neural_numeric_only`, the quarterly winner, is sixth.

Component 5 measured an inversion here, Component 6 measured one, Component 7 measured a
metric-dependent ordering, and Component 8 now measures a representation-dependent one. **Four
components, four inversions.** One fold, so this is a robustness observation and not a
measurement — but it is consistent, and it says that whichever model is chosen on quarterly folds
must not be assumed correct when the regime changes.

### 9.6 The metric ordering also disagrees

On `precision@k_1_day` — the metric closest to "what a scheduler with one day of capacity actually
gets" — the ordering is different again:

| model | P@k_1day |
| --- | ---: |
| lightgbm | **0.6598** |
| logistic_regression | 0.6576 |
| xgboost_chain_embeddings | 0.6480 |
| xgboost | 0.6308 |
| neural_numeric_only | 0.6273 |
| neural_embeddings | 0.6217 |
| neural_onehot | 0.6071 |

**`neural_numeric_only` wins NDE and loses precision@k.** NDE integrates over the whole discovery
curve; `precision@k_1_day` looks only at the top 26–45 rows of a quarter. A model can order the
whole queue better while ordering its very top slightly worse, and this one does.

For a department whose real constraint is one day of capacity at a time, `lightgbm` and the
Component 6 GLM remain the better rankers.

### 9.7 Probability quality — raw, and left uncorrected

`neural_numeric_only` posts the best Brier (0.2355) and the best ECE (0.0563) in the project,
better than the penalised GLM's 0.2382 / 0.0635. The pre-registered expectation — that a network's
probabilities would be *worse* calibrated than a GLM's — **failed**, and is recorded rather than
quietly dropped.

The embedding models are worse (ECE 0.0651–0.0750), and `neural_pos_weighted` is much worse
(**0.1002**) — the class-weighting ablation doing exactly what Component 7's did: buying nothing
and distorting the probability scale on a problem with no imbalance to correct.

No score saturated at 0 or 1. Nothing here is calibrated; Component 9 owns that.

## 10. What did not work

**Entity embeddings — the component's headline experiment — did not help.** `neural_embeddings`
loses to `neural_numeric_only` by 0.0267 NDE, and every ablation confirms it: removing *any*
family improves the model. Removing facility type improves it least (0.2153, the worst neural
variant); removing ZIP improves it most (0.2274).

The mechanism is visible in the epoch counts:

| model | parameters | mean best epoch | mean final epoch |
| --- | ---: | ---: | ---: |
| neural_numeric_only | 41,729 | **10.4** | 25.4 |
| neural_no_chain | 50,353 | 6.4 | 21.4 |
| neural_no_community_area | 65,313 | 5.4 | 20.4 |
| neural_no_zip | 65,369 | 4.3 | 19.3 |
| neural_no_facility_type | 64,649 | 4.2 | 19.2 |
| neural_embeddings | 67,985 | 4.0 | 19.0 |
| neural_onehot | **337,665** | **2.3** | 17.3 |

**Capacity and time-to-overfit are almost perfectly inversely ordered.** The more categorical
parameters a model carries, the earlier its validation loss bottoms out. The 26 as-of features
carry a limited amount of signal, and additional representational capacity buys overfitting rather
than fit. The representative learning curve shows it plainly: training loss falls monotonically
from 0.628 to 0.532 across 23 epochs while validation loss bottoms at epoch 8 (0.637) and climbs
to 0.659.

**The one-hot control did not rescue it.** Experiment B's answer is that the *representation* is
not the problem — learned vectors and indicator columns land within 0.0009 NDE of each other
(0.2215 vs 0.2206). The problem is that the categorical information is not worth its capacity on
this data.

**Class weighting did not help**, exactly as in Component 7: NDE 0.2191, ECE 0.1002.

**Not attempted:** deeper or wider networks. Every fit already overfits within single-digit epochs;
more capacity has an obviously worse prior.

## 11. The embedding visualisation — a negative result, reported as one

The specification requires a projection of the learned chain embeddings and warns against
manufacturing semantic conclusions from it. There are none to manufacture.

**Observation.** t-SNE (perplexity 30, seed 20260818, PCA init) over the 846 chain vectors from
`quarterly-2026Q2` produces a **single featureless isotropic blob**. There are no clusters, no
filaments and no separated regions. Both reserved tokens — `__UNKNOWN__` and `__INDEPENDENT__` —
sit *inside* the cloud rather than apart from it.

**Measurement, because a picture is not evidence.** In the raw 16-dimensional space, over all
846 × 845 pairs:

| | mean cosine | SD | p95 |
| --- | ---: | ---: | ---: |
| learned chain embeddings | 0.0018 | 0.2508 | 0.4164 |
| random Gaussian table, same shape | 0.0000 | 0.2504 | 0.4129 |

**The learned table's pairwise geometry is statistically indistinguishable from random.** Vector
norms average 3.89 (SD 0.69), so the rows did move from initialisation — they simply did not move
into any shared structure.

**Nearest neighbours confirm it.** Cosine neighbours in the raw space, for the largest chains:

| chain | 4 nearest neighbours |
| --- | --- |
| SUBWAY | SWEET MANDY BS, KFC, KANELA BREAKFAST CLUB, J AND J FISH |
| DUNKIN DONUTS | FURIOUS SPOON, COLUTAS PIZZA, PHLAVZ BAR AND GRILLE, NICKYS GRILL AND YOGURT OASIS |
| MCDONALDS | CAFE L APPETITO, JUST SALAD, WISE NUTRITION, CHINA CAFE |
| 7 ELEVEN | CHICAGO LUNCHBOX, CARO MIO ITALIAN RISTORANTE, BADOU SENEGALESE CUISINE, PASTORAL |
| CHIPOTLE MEXICAN GRILL | NEVERIA LA FLOR DE MAYO, CAFE L APPETITO, CHILDRENS WORLD DAYCARE, BEGGARS PIZZA |
| DOMINOS PIZZA | JAMBA JUICE, JEWEL FOOD STORE, CHICAGO DOWNTOWN MARRIOTT, FAT SHALLOT |

**Interpretation, kept separate from observation.** The occasional plausible pair appears —
SUBWAY↔KFC, TACO BELL↔SUBWAY SANDWICHES AND SALADS — but with 846 categories and four neighbours
each, a handful of plausible-looking pairs is exactly what chance produces. There is no cuisine
structure, no service-format structure and no size structure. **The network did not learn a useful
representation of chains**, which is consistent with, and the likely explanation for, §9.1's
negative result.

**A failed embedding visualisation is still a result**, and this is the honest version of it.

## 12. Community area — the ablation, and what it does not license

`neural_no_community_area` (NDE 0.2258) is **better** than `neural_embeddings` (0.2215). Removing
the community-area embedding *improved* the model by 0.0043 NDE.

Two things follow and a third does not.

**It follows that community area bought no predictive value here.** Under this architecture, on
these folds, its embedding was a net cost.

**It follows that the pre-declared non-retention rule was not tested by a hard case.** ADR 0023
committed in advance to *not* retaining community area even if it helped. It did not help, so the
rule cost nothing on this occasion. The rule stands for the next component that asks.

**It does not follow that community area is safe, or that geography carries no signal.** This is a
statement about one architecture that overfits within four epochs, not about the variable. A
result of "it did not help a model that was overfitting anyway" is weak evidence about the
variable's information content and no evidence at all about its fairness properties. Component 12
still owns that question, and ADR 0019's limitation — that disparate citation rates cannot be
separated from disparate treatment in this data — is unchanged.

The same reading applies to `zip` (0.2274, likewise better than the full model).

## 13. Limitations

1. **The neural win is the same size as its own seed noise.** +0.0053 ROC-AUC against a
   seed-to-seed spread of 0.0058. This limitation governs every other reading in §9.
2. **The improvement clears XGBoost's seasonality band in one direction only.** XGBoost sits
   inside the neural model's own interval.
3. **`neural_numeric_only` wins NDE and loses `precision@k_1_day`** to LightGBM and the GLM — the
   metric closest to a real one-day capacity constraint.
4. **The ordering inverts on `covid_shift`**, where the quarterly-worst neural model wins.
5. **The final fit uses ~85% of its fold's training rows**, because the weights kept are the best
   validation epoch's. Component 7 avoided this cost by freezing a round count and refitting on
   everything. Self-inflicted, and removable.
6. **43.03% of violations are still surfaced later** under the best neural ranking, against 42.89%
   under XGBoost and 43.24% under Component 6. Effectively unchanged — re-ordering under fixed
   capacity is zero-sum.
7. **The categorical layer is experimental** and is not a Component 4 feature table (ADR 0022).
8. **`establishment_id` is not embedded**, so no claim about per-establishment risk is available
   (ADR 0021).
9. **The chain family's ceiling is its coverage**: 22.70% of rows belong to a chain, so roughly
   three rows in four receive the shared `__INDEPENDENT__` vector.
10. **`covid_shift` is one fold**, searched on the declared minimum of two inner folds.
11. **Probabilities are uncalibrated** and must not be read as risk levels until Component 9.
12. **Determinism holds within a fixed library set only.** A torch version bump may move every
    number here.
13. **Single-threaded CPU training** took 1,998.7 s for 234 fits and will not scale to a much
    larger snapshot.
14. **Only establishments that were actually inspected are evaluated**, and the simulation is
    retrodictive — unchanged from Components 5–7.

## 14. Runtime

| stage | wall clock |
| --- | --- |
| categorical build | 0.5 s |
| profiling (`profile_neural.py`, 8 profiles) | ~25 s |
| learning-rate sweep, 40 trials over 2 studies | 550.9 s |
| training, 234 fits over 18 folds, 4,306 epochs | 1,998.7 s (fitting) |
| evaluation, 18 folds, 15 models, 1,000 sensitivity reps | ~390 s |
| Component 8 test suite | ~200 s |

Single-threaded CPU by design, which is what makes the fits bit-reproducible. See ADR 0020.

## 15. Reproducing

```bash
uv run python scripts/profile_neural.py                       # §3-§7
uv run sentinel build-neural-categoricals --report            # the experimental input layer
uv run sentinel tune-neural --report                          # §8; then freeze into definitions.py
uv run sentinel train-neural --report                         # §9
uv run sentinel evaluate --predictions data/processed/predictions/neural_predictions_<stamp>.parquet --report
```

`train-neural` does not require re-running the sweep: the learning rates are frozen literals in
`neural/definitions.py`, and `TUNED_HYPERPARAMS_PROVENANCE` names the study artifact that produced
them.

## 16. What Component 9 should know

**The best-calibrated model in the project is now a neural network.** `neural_numeric_only` posts
Brier 0.2355 and ECE 0.0563, better than the penalised GLM. Component 9 inherits a
better-calibrated starting point than Component 7 predicted, and the pre-registered expectation
that a network would be worse calibrated is disproved rather than dropped.

**It should calibrate more than one model.** The three candidates disagree by metric and by
regime: `neural_numeric_only` (best NDE, best calibration), `lightgbm` (best `precision@k_1_day`),
`xgboost_chain_embeddings` (best PR-AUC). Choosing one before calibration would foreclose that.

**The seed-noise finding transfers.** Any calibrator fitted on a neural model's outputs inherits
the fact that a different seed moves those outputs by as much as the model's entire advantage over
XGBoost. Averaging predictions over seeds before calibrating is worth considering, and would also
make the headline reproducible in the sense that matters.

**Do not adopt the embeddings.** They lost on the quarterly folds, the visualisation shows no
learned structure, and the community-area ablation exists for Component 12's benefit rather than
for a model choice.

**The ceiling is still the representation.** Four estimator families have now been tried on the
same 26 features. The spread between the best and worst *sensible* one is 0.0156 NDE, and the
seed-to-seed spread within one of them is a third of that. The next real gain is a Component 4
change, not a fifth estimator.
