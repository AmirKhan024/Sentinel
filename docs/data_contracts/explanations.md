# Data contract — feature attributions (Component 11)

**Producer:** `sentinel explain` · **Layer:** `data/processed/explanations/` (ADR 0028)
**Definition version:** `explain_definition_version = "v1"`

Seven Parquet tables and one manifest, all timestamped `<slug>_<UTC>.parquet` with the
manifest keyed to `explanation_values`. Column order is part of this contract; changing it is
a contract change.

---

## 0a. Component 12 groups this artifact; it does not regenerate it

`sentinel audit-fairness` reads `explanation_values` by column name -- without importing
`sentinel.explain`, which is what ADR 0028's denormalised long grain was designed to allow --
and groups it by geography to ask whether a model's feature *reliance* differs across
neighbourhoods.

**Re-running `sentinel explain` at a different `--sample-size` to check such a finding is
forbidden**: it would change the rows every published Component 11 number rests on. Component
12 works with the sample that exists, and measured what it supports -- the median (model,
community area) cell holds **40** explained rows, and 56 of 312 clear 100. That supports a
comparison of 30-feature *profiles* and nothing per-row.

Note the name mismatch, because it caused a silent failure once: Component 9 names a
calibrated model `xgboost_platt` and this artifact names the same model `xgboost`. Looking a
profile up under the calibrated name finds nothing and draws no figure -- and **a missing
figure looks exactly like a figure the data could not support.**

## 0. What this artifact is, and what it is not

It answers **"why did this model give this establishment this score?"** for the models
Components 6–8 fitted and Component 9 calibrated.

It is **not**:

- a prediction. Nothing here has a `score` column, and
  `sentinel evaluate --predictions` will reject it.
- a measurement of model quality. A large `mean_abs_shap` says a model *relied* on a feature.
  It does not say the model was right to — Component 6 measured `days_since_any_inspection`
  helping on the quarterly folds and hurting under distribution shift, which is exactly a case
  where high reliance and low value coincide.
- **causal.** A SHAP value states how the model used a feature. It does not state that
  changing the feature would change the outcome. Component 7 measured a condition number of
  71.8 with one feature pair correlated at 0.9888, so attributions are shared between
  correlated features in a way no SHAP value discloses.

**Nothing here may be joined onto a feature table.** `explanation_values` is keyed by
`target_inspection_id`, which is Component 4's key, so the join is one line away — and it
would be a model's own reasoning about a row becoming a feature of that row.

---

## 1. `explanation_values` — the long grain

One row per **(model, fold, explained inspection, feature)**. The only large table.

| column | type | meaning |
|---|---|---|
| `model_name` | Utf8 | the base model, e.g. `xgboost`. Never a calibrated name. |
| `model_version` | Utf8 | from the explain registry |
| `family` | Utf8 | `logistic` / `boosted` / `neural_mlp` |
| `fold_set` | Utf8 | `quarterly` or `covid_shift` |
| `fold_id` | Utf8 | e.g. `quarterly-2026Q2` |
| `target_inspection_id` | Utf8 | the prediction being explained |
| `feature_name` | Utf8 | the **transformed representation** the estimator indexes |
| `original_feature_name` | Utf8 | the Component 4 feature, or the null-rule family name |
| `derived_from` | Utf8 | comma-joined Component 4 columns this representation covers |
| `feature_kind` | Utf8 | `feature` or `family_indicator` |
| `feature_value` | Float64 | the **raw** pre-transform value. Null where the source was NULL. |
| `transformed_value` | Float64 | the value the estimator actually saw |
| `shap_value` | Float64 | the contribution, in `output_space` |
| `output_space` | Utf8 | always `log_odds` |
| `explanation_method` | Utf8 | `tree_shap` / `linear_shap` / `permutation_shap` |
| `is_exact` | Boolean | **false for `permutation_shap` only.** Load-bearing — see §8. |
| `base_value` | Float64 | the expected output, constant within a (model, fold) |
| `prediction_value` | Float64 | the model's log-odds for this row |
| `trained_through` | Date | the **base model's** horizon = `fold.train_end` |
| `explain_definition_version` | Utf8 | `v1` |

**Sort key:** `(model_name, fold_set, fold_id, target_inspection_id, feature_name)` — a full
key, so two runs produce byte-identical files.

### Reading a local explanation

```text
prediction_value  =  base_value  +  sum(shap_value) over all feature_name
```

within `additivity_tolerance` (see `explanation_cases`). Positive `shap_value` raised the
risk; negative lowered it.

### On the two names and the two values

`feature_name` is the representation; `original_feature_name` is where it came from. For a
plain feature they are equal, because the transformations applied — median or constant
imputation, then standardisation — are monotone and one-to-one, so the attribution belongs to
the feature it started as.

A **family indicator** (`missing_no_prior_canvass` and three siblings) does not belong to any
single column. It summarises the null mask that several Component 4 features share, so
`original_feature_name` is the null-rule family and `derived_from` lists every column it
covers. Pretending it belonged to one column would be the false aggregation this contract
exists to avoid.

`feature_value` is the number a human recognises; `transformed_value` is what the estimator
saw. They are identical for the boosters, which fit no preprocessing at all, and differ for
the linear and neural models, which impute and standardise. **`feature_value` is null, never
zero, where the source was NULL** — for a tree model a NULL is a real observation the split
routed on, not a gap.

### No anonymous names

Every `feature_name` is a member of Component 4's 26 features or its 4 null-rule indicators.
`validate.every_feature_maps_to_a_known_representation` rejects both an unknown name and
anything matching `^(f|x|v|col|column|feature)?[_-]?\d+$`. `feature_127` is unrepresentable.

---

## 2. `explanation_cases` — one row per explained prediction

Where additivity, provenance and the calibration link live, so the long table stays scannable.

Beyond the identity columns:

| column | type | meaning |
|---|---|---|
| `base_value`, `prediction_value` | Float64 | as above |
| `reconstruction_value` | Float64 | `base_value + sum(shap_value)` |
| `reconstruction_residual` | Float64 | `abs(reconstruction_value − prediction_value)` |
| `additivity_tolerance` | Float64 | the frozen tolerance for this method (ADR 0030) |
| `additivity_holds` | Boolean | residual within tolerance |
| `n_features` | Int64 | 30 |
| `positive_contribution_sum` | Float64 | sum of the positive contributions |
| `negative_contribution_sum` | Float64 | sum of the negative ones (≤ 0) |
| `base_score` | Float64 | **the committed Component 6/7/8 probability**, bit-identical |
| `base_score_reproduced` | Boolean | this (model, fold) passed the bit-identity gate |
| `calibrated_probability` | Float64 | Component 9's output. **Null** if no C9 artifact was read. |
| `calibration_method` | Utf8 | `platt` / `isotonic`, or null |
| `base_model_trained_through` | Date | the estimator's horizon, `fold.train_end` |
| `calibrator_fitted_through` | Date | the calibrator's, `fold.calibration_end`, or null |
| `prediction_available_from` | Date | `fold.test_start`, or null |
| `sample_strategy`, `sample_size`, `sampling_seed`, `sampling_population`, `population_rows` | | how these rows were chosen |
| `background_strategy`, `background_size`, `background_seed`, `background_max_date` | | the reference set, **0 / null for the boosters** |
| `permutation_rounds` | Int64 | 8 for the network, **0** otherwise |

**Sort key:** `(model_name, fold_set, fold_id, target_inspection_id)`.

### Three horizons, never one

```text
base_model_trained_through   fold.train_end        what the estimator saw
calibrator_fitted_through    fold.calibration_end  what Component 9's map saw
prediction_available_from    fold.test_start       first operationally usable date
```

The attribution decomposes the **estimator's** output, so `trained_through` on
`explanation_values` is the first of these. Do not read it as Component 9's artifact's
`trained_through`, which is the maximum of all three.

### The calibration boundary

SHAP explains `base_score`. Platt then maps it to `calibrated_probability`. They are connected
and not the same thing, and a consumer can show both:

```text
Model score before calibration:  0.706
Calibrated risk probability:     0.664
```

A `calibrated_probability` of null means Component 9's artifact was not supplied to this run —
not that calibration failed, and not that the two numbers are equal.

### `background_size = 0` is a statement, not a gap

TreeSHAP takes its conditional expectations over the cover recorded in the trees at fit time,
so the boosters use **no** reference dataset. Writing a size for them would imply one they
never had.

---

## 3. `explanation_importance` — global importance

Per **(model, fold_set, feature)**, at two scopes.

| `scope` | `fold_id` | what the row is |
|---|---|---|
| `fold` | set | one fold's `mean(abs(shap))` and `mean(shap)`, with its rank |
| `fold_set` | **null** | the aggregate over that fold set, with spread |

Aggregate rows additionally carry `sd_abs_shap`, `mean_rank`, `sd_rank`, `best_rank`,
`worst_rank`, `folds` and `rows`.

**`covid_shift` is never pooled into a `quarterly` aggregate.** They are separate values of
`fold_set` and aggregated separately — structural, not conventional. Component 6 measured the
model *ordering* inverting on that fold.

Both the absolute and the signed mean are emitted. `mean_abs_shap` says how much a feature was
used and is what the ranking is built on; `mean_shap` keeps the direction. A feature with
`mean_abs_shap = 1.0` and `mean_shap = 0.0` is a switch that raises risk for some
establishments and lowers it for others — not a dead feature.

The aggregate `rank` is computed from the mean importance; `mean_rank` averages the per-fold
ranks. **They can disagree, and the disagreement is the finding**: a feature ranked second on
most folds and twenty-fifth on two has a good mean rank and a poor mean importance.

---

## 4. `explanation_stability` — does the reasoning hold over time?

One row per model per fold comparison, within a fold set.

| column | meaning |
|---|---|
| `comparison` | `consecutive` or `first_to_last` |
| `from_fold_id`, `to_fold_id` | the two folds compared |
| `spearman_rho` | rank correlation of `mean_abs_shap` ranks, ties averaged |
| `top_k`, `top_k_jaccard` | overlap of the top 10 features |
| `features` | 30 |

Two metrics because they disagree usefully: a model can reorder its tail while keeping the
same top ten, or swap two dominant features while every other rank holds.

A fold set with one fold (`covid_shift`) produces **no rows** rather than a self-comparison of
1.0, which would read as evidence of stability and is evidence of nothing.

**This is not Component 9's calibration drift.** A model can hold its ECE and its ROC-AUC
steady while this table moves. They are different phenomena.

---

## 5. `explanation_drift` — which features moved

One row per (model, fold_set, feature): `first_rank`, `last_rank`, `best_rank`, `worst_rank`,
`rank_range`, `mean_abs_shap`, `sd_abs_shap`, `coefficient_of_variation`,
`materially_changed`.

`materially_changed` is `rank_range >= RANK_DRIFT_THRESHOLD` (5, a sixth of the ranking),
**declared before any rank was computed**. `coefficient_of_variation` is null rather than
infinite when the mean is zero — a feature the model never split on is a real and informative
case, not a NaN.

---

## 6. `explanation_representative_cases` — the report's examples

One row per (model, fold, tier) for the representative fold. `tier` ∈ `high` / `medium` /
`low`, selected at predicted-score quantiles 0.90 / 0.50 / 0.10, nearest-rank, ties broken by
`target_inspection_id`.

**Selected by the prediction, never by the outcome.** A case chosen because the model was
right about it would be storytelling with a reproducible rule bolted on, and the
reproducibility would make it more persuasive rather than less misleading.
`validate.representative_cases_are_ordered_by_predicted_risk` asserts
`low <= medium <= high` by the committed score.

Carries `base_value`, `prediction_value`, `base_score`, `calibrated_probability`,
`calibration_method`, `output_space`, `method` and `is_exact`.

---

## 7. `explanation_support` — the support matrix

One row per registered model, **including the one that could not be explained**.

| column | meaning |
|---|---|
| `explanation_status` | `supported` / `unsupported` |
| `explanation_method`, `output_space` | **null** when unsupported |
| `is_exact`, `is_experimental` | |
| `name_source` | the function that recovered this model's column names |
| `rationale` | why this method suits this architecture |
| `unsupported_reason` | non-null exactly when unsupported |
| `explained_rows`, `attribution_values` | 0 when unsupported |

An unsupported model appears here **and nowhere else** — no values, no cases, no importance
rows. A consumer joining on this table gets nulls rather than zeros, and a zero would have
read as "this model used no features".

`xgboost_chain_embeddings` is unsupported: its fitted booster is reachable only through
`neural.embed._scorer_for`, a private process-local stash, and Component 8 is closed. ADR
0031, which also records the four-line public extension that would lift the restriction.

---

## 8. `is_exact`, and the additivity trap

| method | exact | what "approximate" means |
|---|---|---|
| `linear_shap` | yes | nothing; the closed form is the answer |
| `tree_shap` | yes | nothing; TreeSHAP is exact |
| `permutation_shap` | **no** | how the credit is split among columns |

**A passing additivity check is not evidence that a permutation attribution is accurate.** The
permutation path telescopes to `f(row) − f(background)`, so `base + sum(phi)` reconstructs the
output exactly at one round and at sixty-four alike; the measured residual is 6.1e-10 at every
round count tested.

What is approximate is the credit split. Measured against a 64-round reference: the median
per-value error at 8 rounds is 1.00% of the largest attribution, while the **global importance
ranking** reaches a rank correlation of 0.9964. So the network's global table is quotable and
its individual values should not be quoted to three decimal places. ADR 0030.

---

## 9. Provenance and integrity

The manifest (`manifest_explanation_values_<UTC>.json`) records:

- every input path and sha256 — features, and only the prediction artifacts this run actually
  read. A run explaining only Component 7's boosters records `null` for Components 6 and 8;
- **`prediction_sha256_after` and `prediction_artifacts_unchanged`** — the same files
  checksummed again after everything was written, so "Component 11 changed no prediction" is
  checkable rather than asserted;
- `reproduction_rows`, `reproduction_mismatches`, `reproduction_passed` — ADR 0029's
  bit-identity gate, per model;
- the full support matrix, methods, output spaces and exactness;
- the sampling and background budget, and `max_additivity_residual` per method;
- `attribution_semantics`, `calibration_boundary` and `causality_disclaimer` as prose;
- library versions, `torch_threads`, `blas_threads` and the determinism caveat;
- `blocked` — including *model selection* and *causal interpretation*;
- every validation check and its outcome.

### Guarantees a consumer may rely on

1. Every `(model_name, fold_id, target_inspection_id)` exists in a committed prediction
   artifact.
2. `base_score` equals that artifact's `score` **bit for bit**, not approximately.
3. Every explained row lies inside its fold's test window.
4. Every background row lies inside its fold's **training** window, on or before
   `train_end` — re-derived from the frame, not read from a field.
5. Every `feature_name` is a known Component 4 representation.
6. `base_value + sum(shap_value)` reconstructs `prediction_value` within
   `additivity_tolerance`.
7. Every table is sorted by its declared key, so two runs are byte-identical.
8. The prediction artifacts are unchanged.

### Reproducing

```bash
uv run sentinel explain --report      # ~19 min; do NOT set OMP_NUM_THREADS
```

The thread-count warning is load-bearing: ADR 0026 records a BLAS thread override moving
`logistic_regression` scores by 1e-13 and correctly failing the identity gate.
