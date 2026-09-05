# ADR 0030 — The attribution protocol: one method per family, one output space, and shap as a dev-only oracle

**Status:** Accepted · **Date:** 2026-08-25

## Context

"Use SHAP" is not a decision. SHAP is a family of estimators of the same quantity, and the
choices that actually determine whether an attribution is trustworthy are made underneath
that word: which estimator, in which output space, against which reference distribution, over
how many rows, and how the reader is told which parts are exact.

Component 11 has four supported models from three different architectures, and every one of
those choices differs between them. This ADR fixes all of them, from measurements taken by
`scripts/profile_explanations.py` **before** the implementation existed, in the order the
project's working agreement requires: investigate → document → design → implement → test.

## Decision

### One method per family, chosen because of what the estimator is

| model | method | exact | needs a background |
|---|---|---|---|
| `logistic_regression` | closed form, `phi_j = coef_j * (z_j − E_R[z_j])` | **yes** | yes |
| `xgboost` | the booster's own TreeSHAP (`pred_contribs`) | **yes** | no |
| `lightgbm` | the booster's own TreeSHAP (`pred_contrib`) | **yes** | no |
| `neural_numeric_only` | antithetic permutation sampling | **no** | yes |

Forcing all four through one model-agnostic explainer would have been tidier, and would have
thrown away two exact computations to buy that tidiness. A linear model has no interactions,
so the feature-order average that defines a Shapley value collapses to a closed form. A tree
ensemble admits an exact polynomial-time algorithm that both libraries already ship. Only the
network has no exact answer, and only the network gets an approximation.

The boosters need **no reference dataset**, and this is a substantive point rather than a
convenience: the tree-path-dependent algorithm takes its conditional expectations over the
*cover recorded in the trees at fit time*, so the reference distribution is the training data
the model already saw. That is temporally safe by construction and not something a caller
could override. The artifact records `background_size = 0` for those models rather than a
number, because writing one would imply a reference set they never used.

### `shap` is a dev dependency, not a runtime one

Two of the three methods are already reachable through libraries the project depends on, and
the third is arithmetic rather than an algorithm. So the values are computed in-house and the
test suite cross-checks them against `shap.TreeExplainer` and `shap.LinearExplainer`.

This is ADR 0015's dividing line applied consistently, and it is exactly what Component 5 did
with scikit-learn: the evaluation metrics are hand-implemented and every one is cross-checked
against sklearn in the suite. Measured agreement:

```text
xgboost native pred_contribs  vs shap.TreeExplainer     max abs diff  0.0
lightgbm native pred_contrib  vs shap.TreeExplainer     max abs diff  0.0
closed-form linear SHAP       vs shap.LinearExplainer   max abs diff  0.0
```

Exact agreement, not approximate. Adding `shap` to `[project.dependencies]` would have bought
nothing except its own RNG and threading behaviour to pin down.

The permutation explainer has a stronger oracle available than any library, and uses it:
`test_permutation_shap_converges_on_brute_force_shapley` compares it against the **Shapley
definition itself**, enumerated over all 2^M subsets on a deliberately non-additive model.
That is the definition rather than another implementation of it.

`shap` also needs `numba>=0.67` and `llvmlite>=0.49` pinned in the dev group. Without those
floors the resolver backtracks numba to a release predating numpy 2.x and the build fails.
The floors are what keep numpy at 2.5.2 — the version ADR 0026's bit-identity gate is
baselined on. **Downgrading numpy to satisfy a test oracle would re-baseline every model in
the project**, which is the tail wagging the dog.

### Everything is in log-odds, and probability space is refused rather than declared

`OutputSpace` has one member, `LOG_ODDS`. All four supported models expose a natural log-odds
output — `decision_function`, `output_margin`/`raw_score`, the pre-sigmoid logit — so every
value in the artifact is comparable with every other and a cross-model importance table is a
comparison of models rather than of units. Measured ranges on the representative fold:

| model | min | max | mean |
|---|---|---|---|
| `logistic_regression` | −2.0453 | 1.3399 | −0.4515 |
| `xgboost` | −2.4419 | 1.4923 | −0.5093 |
| `lightgbm` | −2.5085 | 1.6274 | −0.4788 |
| `neural_numeric_only` | −2.4192 | 1.2100 | −0.4570 |

A probability-space variant is **not declared**. A Shapley decomposition of
`sigmoid(margin)` is not additive in the margin's own contributions, because `sigmoid` is not
linear, so such a table would have to either abandon additivity or fabricate it. Declaring an
`OutputSpace.PROBABILITY` that nothing can reach would repeat the defect ADR 0014 records: a
rejection that has never been observed is indistinguishable from one that cannot happen.

### SHAP explains the base score, not the calibrated probability

```text
base model  ──►  base_score  ──►  Platt  ──►  calibrated_probability
     │
     └── this is what the attributions decompose
```

Component 9's calibrator is a separate two-parameter monotone map fitted on a window the base
model never read. Attributing the calibrated number would mean attributing a composition of
two models, one of which has two parameters and no features, and the resulting values would
answer a question nobody asked.

So `explanation_cases` carries `base_score` and `calibrated_probability` **side by side**,
with `calibration_method`, and a consumer can show a user both:

```text
Model score before calibration:  0.706
Calibrated risk probability:     0.664
```

`calibrated_probability` is null when Component 9's artifact was not supplied — null rather
than absent, so the column's meaning never depends on the run's flags, and null rather than
`base_score`, so a reader can never mistake an uncalibrated number for a calibrated one.

### The budget, frozen from measurement

| constant | value | measurement it came from |
|---|---|---|
| `SAMPLE_SIZE` | 300 rows per (model, fold) | 41,536 test rows over 18 folds is the population |
| `SAMPLING_SEED` | 20260825 | an integer literal, never `hash()` of a string |
| `BACKGROUND_SIZE` | 64 rows, from the **training** window | ADR 0028's temporal argument |
| `BACKGROUND_SEED` | 20260826 | as above |
| `PERMUTATION_ROUNDS` | 8 antithetic pairs | the convergence sweep below |

The **same sampled ids are used for every model**. Explaining all rows for the cheap models
and a sample for the network would compare importance across two different populations, and
any difference between two importance tables would then be part real and part sampling
artifact with no way to separate them.

Nothing about the label or the score participates in the sample. The representative *cases*
in the report are chosen by predicted-score quantile, which is a separate, later, reporting
decision that never affects which rows were explained.

### `PERMUTATION_ROUNDS = 8`, and the number that fixed it

Measured against a 64-round reference drawn at an independent seed, 60 rows, background 32:

| rounds | median local err | max global err | global rank rho | seconds |
|---|---|---|---|---|
| 1 | 2.76% | 16.61% | 0.9706 | 0.2 |
| 2 | 1.90% | 8.14% | 0.9902 | 0.4 |
| 4 | 1.29% | 5.63% | 0.9942 | 1.9 |
| **8** | **1.00%** | **2.85%** | **0.9964** | **4.6** |
| 16 | 0.71% | 4.41% | 0.9973 | 9.2 |
| 32 | 0.53% | 1.67% | 0.9991 | 6.5 |

Two things this settles, and the second is the one that matters.

**The global statistic converges far faster than any individual value.** Each row draws its
own permutations, so averaging `|SHAP|` over 300 rows averages independent errors and the
ranking stabilises long before a single local attribution does. That is what licenses
publishing a global importance ranking for the network while labelling its per-row values
approximate — and it is why the local figures carry that caveat printed on them.

**Additivity is flat across the whole sweep, at 6.1e-10, and is therefore not evidence of
accuracy.** A permutation path telescopes to `f(row) − f(background)`, so `base + sum(phi)`
reconstructs the output exactly at one round and at sixty-four alike. A component that
reported a passing additivity check as evidence its permutation values were accurate would be
misreporting. `attribute.py` says so, `validate.py` says so, and
`test_additivity_holds_at_one_round_and_is_therefore_not_evidence_of_accuracy` asserts both
halves so the point cannot quietly be lost.

### Additivity tolerances, per method

| method | measured max | frozen at |
|---|---|---|
| `tree_shap` | 8.920e-07 (xgboost, float32) | 1e-5 |
| `linear_shap` | 1.332e-15 | 1e-10 |
| `permutation_shap` | 6.112e-10 | 1e-6 |

Each is three to five orders above the measured maximum — enough to absorb fold-to-fold
variation in tree count and window size, and no more, because a tolerance generous enough
never to trip is a check that has stopped checking. The tree tolerance is set from the worse
of the two libraries: xgboost computes in float32 and lightgbm in float64, and their
residuals differ by nine orders of magnitude as a result. Averaging them would have produced
a tolerance xgboost could not meet.

Component 9 set three thresholds from expectation and had to correct all three. These were
set from measurement first.

### Stability metrics, declared before the ranks existed

Spearman rank correlation of `mean_abs_shap` ranks with ties averaged, and top-10 Jaccard
overlap. Both, because they disagree usefully: a model can reorder its tail while keeping the
same top ten, or swap two dominant features while every other rank holds. `RANK_DRIFT_THRESHOLD
= 5` — a sixth of the 30-feature ranking — decides what the findings document may call a
material change. A threshold chosen after seeing the ranks would be a conclusion wearing a
criterion's clothes.

## Alternatives rejected

**`shap.KernelExplainer` for everything.** One code path, and it would have replaced two exact
computations with an approximation whose sample size is another parameter to defend. It is
also markedly slower than TreeSHAP on 30 columns.

**`shap.DeepExplainer` (DeepLIFT) for the network.** Faster than permutation sampling and
neural-specific. Rejected because its additivity is only approximate for ReLU-plus-dropout
networks, its behaviour has broken across torch releases, and it would have forced `shap`
into the runtime dependencies. The brief's instruction was explicit: *do not silently use an
invalid approximation.*

**Explain the calibrated probability directly.** Rejected above. A monotone two-parameter map
applied after the fact is not part of the model whose features are being attributed.

**Explain all 41,536 rows per model.** Free for the tree and linear models. Measured at 0.12 s
per row for the network at the frozen budget, so the full population would be about **85
minutes** for that model alone — affordable, and still rejected, because the four importance
tables would then rest on two different populations and any difference between them would be
part real and part sampling artifact. The bounded sample is chosen for comparability, not for
compute.

**Stratify the background to resemble the test window.** Would make the reference point depend
on the rows being explained, and the expected value would stop being a property of the model's
training period.

**Report interaction values.** Both boosters support an `[n, M, M]` tensor. Not computed: 30×
the storage for a question nobody has asked, and a later component can add it without changing
anything here. Recorded in `BLOCKED_EXPERIMENTS`.

## Consequences

- The artifact's `is_exact` column is load-bearing. Three of four models carry `true` and one
  carries `false`, and a consumer that ignores the column will over-trust the network's
  per-row values by roughly one percent of the largest attribution.
- `explanation_method` and `output_space` are on every row rather than in the manifest, so a
  table can never be read without its units.
- Changing `--sample-size` changes the artifact, and the value used is recorded in every case
  row. There is deliberately **no `--seed` flag**: a seed a caller can change is a seed nobody
  can cite.
- The dev group now carries `shap`, `numba`, `llvmlite` and, transitively, `pandas`. That is a
  real cost, paid for a test oracle, and it is why they are dev-only.
