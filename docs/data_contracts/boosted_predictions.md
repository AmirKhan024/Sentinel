# Data contract — boosted predictions

**Produced by:** Component 7 (`sentinel train-boosting`, `sentinel tune-boosting`)
**Layer:** `data/processed/predictions/` (scores) and `data/processed/tuning/` (search trials)
**Consumed by:** Component 5 (`sentinel evaluate --predictions`); Component 9 will consume the
scores for calibration
**Design rationale:** ADR 0014 (predictions are outputs), ADR 0016 (the libraries), ADR 0017 (the
tuning protocol), ADR 0018 (the tuning layer), ADR 0019 (inspector modelling is blocked)

This contract is deliberately parallel to `baseline_predictions.md`. Component 7 replaces the
estimator and adds a tuning protocol; it changes nothing about folds, features, the target, the
metrics or the evaluation. Where a rule is unchanged from Component 6 this document says so rather
than restating it differently, because two slightly different statements of one rule is how a
contract drifts.


> **Component 9 status (2026-08-24):** this artifact was consumed by Component 9 and is
> **unchanged and byte-identical** — Component 9 re-executed the fits behind a bit-identity
> gate rather than rewriting anything here. The calibrated scores are a separate artifact,
> `calibrated_predictions_<stamp>.parquet`; see `calibrated_predictions.md`. The
> probabilities in *this* file remain uncalibrated.
---

## 1. Identity and file naming

| slug | directory | grain |
| --- | --- | --- |
| `boosted_predictions_<stamp>.parquet` | `predictions/` | one row per (model, fold, scored test row) |
| `boosted_importances_<stamp>.parquet` | `predictions/` | one row per (model, fold, matrix column) |
| `boosted_training_log_<stamp>.parquet` | `predictions/` | one row per (model, fold) |
| `tuning_trials_<stamp>.parquet` | `tuning/` | one row per (study, trial) |

`<stamp>` is `%Y%m%dT%H%M%SZ` at run start. All files are zstd Parquet. One manifest per run,
sidecar to the anchor artifact: `manifest_boosted_predictions_<stamp>.json` and
`manifest_tuning_trials_<stamp>.json`.

**The slug differs from Component 6's on purpose.** `baseline_predictions` and `boosted_predictions`
are separate files under separate manifests, so Component 6's benchmark stays visible and
byte-identical. "Did Component 7 improve on Component 6?" is only answerable if Component 6's answer
was not overwritten.

## 2. The models

| model | estimator | features | tuned | notes |
| --- | --- | --- | --- | --- |
| `xgboost` | XGBoost, `hist` | 26 | yes, per fold set | primary |
| `lightgbm` | LightGBM, leaf-wise | 26 | yes, per fold set | primary |
| `xgboost_class_weighted` | XGBoost, `hist` | 26 | **no** — borrows `xgboost`'s | the weighting ablation |

All three consume the identical Component 4 contract: the same 26 features in the same order. The
ablation borrows its donor's parameters rather than being tuned separately, so the only thing that
differs between it and `xgboost` is `scale_pos_weight`. Tuning it independently would vary two
things at once and make the ablation uninterpretable.

## 3. Score semantics

`score` is `predict_proba(...)[:, 1]` — the model's estimate of
`P(a Priority or Priority Foundation violation is cited | information available at the inspection
decision point)`.

**Higher score means higher predicted risk.** Component 5 assumes this direction and probes it live;
inverting it would produce a plausible, confidently wrong result rather than an error.

⚠ **These are RAW probabilities, not calibrated ones.** `is_probability = true` means "this came
from `predict_proba`", which is what enables Component 5's Brier, ECE and MCE. It does not mean the
number can be read as a risk level. Component 7 corrects nothing: measured quarterly ECE is 0.0621
for `xgboost` and 0.0644 for `lightgbm`; under `covid_shift` it is 0.1253 and 0.1518. Component 9
owns calibration, and until it ships, **no document may describe these as calibrated.**

## 4. `trained_through`

`trained_through = fold.train_end`, always. Same as Component 6, and for a stronger reason.

Component 6 declares the training end because it fits no calibrator and never reads the calibration
window. Component 7 has a second way it could have reached later data — **early stopping** — and
does not take it. The boosting-round count is fixed in advance from a search confined to a region
strictly earlier than any test window (ADR 0017), and `train.fit_fold` runs exactly that many rounds
with no `eval_set` at all.

The contract's ceiling is `fold.calibration_end`, so Component 7 sits strictly inside it. Guarantees:

- `validate.trained_through_is_the_training_end` asserts equality, stricter than the contract.
- `validate.final_fits_did_no_early_stopping` asserts no fit carries `early_stopping_rounds` or
  `eval_set`.
- `boosted_training_log.early_stopped` is written `false` on every row — the claim travels in the
  artifact rather than only in prose.
- `tests/test_boosting_leakage.py` deletes the entire calibration window and asserts the predictions
  are bit-identical.

## 5. Preprocessing: none

**No imputation and no scaling is fitted.** The matrix is Component 6's matrix — the same 30
columns, the same order — with nothing done to it.

| | Component 6 | Component 7 |
| --- | --- | --- |
| NULL handling | median / constant-0.0 fill per declared rule | **NaN, routed by a learned default direction at each split** |
| Scaling | `StandardScaler` | none (trees are invariant to monotone transforms) |
| Matrix | 26 features + 4 family indicators = 30 | identical |
| Fitted statistics | medians, means, scales | **none** |

The four null-rule family indicators are retained even though a NaN-native learner does not need
them. Dropping them would mean the two components' matrices differ, and every C6-versus-C7
comparison would become ambiguous between "the estimator is better" and "the matrix is different".

Because there are no fitted preprocessing statistics, Component 6's
`preprocessing_comes_from_train` check has no analogue. It is replaced by a stricter one:
`validate.no_preprocessing_statistics_were_fitted` recomputes the NULL mask from the source frame
and asserts it equals the NaN mask of the matrix the estimator received, cell for cell. A companion
check, `nulls_reached_the_estimator`, asserts the total NaN count is non-zero, so the mask
comparison cannot pass vacuously. Measured on the production run: **3,404,772 NaN cells** across 54
fits.

## 6. The tuning protocol

Fully specified in ADR 0017. The three properties a consumer may rely on:

1. **Every study's region ends strictly before its fold set's first test window.** `quarterly`:
   2018-07-01..2022-03-31 against a first test start of 2022-04-01. `covid_shift`:
   2018-07-01..2020-05-31 against 2020-06-01. Both recorded in the tuning manifest as
   `tuning_regions` and `first_test_start`, and re-derived from the fold definitions by
   `validate.tuning_never_reached_a_test_window` on every run.
2. **Each fold set has its own study.** Parameters are never borrowed across fold sets —
   `tuned_params` raises rather than falling back. The `covid_shift` test window sits inside the
   quarterly region, so one shared study would have contaminated it.
3. **Early stopping happens only inside the objective**, against inner validation quarters that are
   training data for every outer fold.

### 6.1 The XGBoost ↔ LightGBM parameter mapping

Both searches explore the same *concepts*; the parameter names are each library's own, because
copying XGBoost's names into LightGBM would silently tune nothing.

| concept | XGBoost | LightGBM | range | note |
| --- | --- | --- | --- | --- |
| tree depth | `max_depth` | `max_depth` | 3–10 | same name, different force — see leaf count |
| leaf count | — | `num_leaves` | 8–256, **capped at 2^`max_depth`** | LightGBM only |
| learning rate | `learning_rate` | `learning_rate` | 0.01–0.3 (log) | identical |
| leaf-size regularizer | `min_child_weight` | `min_child_samples` | 1–20 / 5–100 (log) | **not interchangeable** |
| row subsample | `subsample` | `bagging_fraction` (+ `bagging_freq=1`) | 0.6–1.0 | identical |
| column subsample | `colsample_bytree` | `feature_fraction` | 0.6–1.0 | per tree in both |
| L1 | `reg_alpha` | `lambda_l1` | 1e-8–10 (log) | identical |
| L2 | `reg_lambda` | `lambda_l2` | 1e-8–10 (log) | identical |
| boosting rounds | `n_estimators` | `n_estimators` | **not searched** | from early stopping, then frozen |

Two entries need reading carefully.

**Leaf count.** XGBoost grows depth-wise, so `max_depth` alone bounds a tree at 2^depth leaves.
LightGBM grows leaf-wise, so `max_depth` bounds nothing on its own. Without the cap the two searches
would explore different capacity ranges and any measured difference between the libraries would be a
fact about the search space rather than the algorithm. At 5 of the 8 searchable depths the declared
range would exceed XGBoost's implied ceiling, so the cap binds in practice — the frozen quarterly
LightGBM parameters are `max_depth=4, num_leaves=16`, exactly at it.

**Leaf-size regularizer.** XGBoost sums the Hessian in a child; LightGBM counts rows. These are
different quantities on different scales, so each is tuned over its own natural range rather than a
shared one. Treating them as equivalent would be the most likely silent error in this table.

`bagging_freq=1` is set unconditionally for LightGBM. `bagging_fraction` without it subsamples
nothing.

## 7. ⚠ Importances are a diagnostic, not an attribution

`boosted_importances` carries native split importance (gain for XGBoost, split count for LightGBM).

**It is not a feature attribution, not an effect size, and not causal.** A tree ensemble distributes
credit between collinear features according to which one it happened to split on first, and
Component 6 measured a condition number of 71.8 with one feature pair correlated at 0.9888. The
column is named `importance` rather than anything suggesting contribution or effect, and
`importance_kind` records which quantity it is because the two libraries do not report the same one.

**Component 11 owns attribution.** SHAP is deliberately not implemented here. Nothing in this
component may be used to answer "why was this establishment ranked highly?".

## 8. Schemas

### `boosted_predictions`

| column | type | notes |
| --- | --- | --- |
| `target_inspection_id` | Utf8 | contract column |
| `score` | Float64 | contract column; raw probability, higher = higher risk |
| `model_name` | Utf8 | |
| `model_version` | Utf8 | |
| `fold_set` | Utf8 | `quarterly` or `covid_shift`; **never averaged together** |
| `fold_id` | Utf8 | |
| `trained_through` | Date | never null — a null silently skips the horizon check |
| `is_probability` | Boolean | never null — a null downgrades to ranking-only and suppresses ECE/MCE |
| `boosting_definition_version` | Utf8 | |

### `boosted_importances`

`model_name`, `model_version`, `fold_set`, `fold_id`, `term`, `importance` (Float64),
`importance_kind` (Utf8: `gain` or `split`), `boosting_definition_version`.

### `boosted_training_log`

25 columns. Beyond the identifiers and the six fold dates: `train_rows`, `test_rows`,
`feature_count`, `matrix_column_count`, **`train_nan_cells`** (the observable proving no imputation
happened), `train_positive_rate`, `seed`, `n_estimators`, **`trees_built`**, `scale_pos_weight`,
`class_weighted`, **`early_stopped`** (always false), `saturated_scores`.

`trees_built` is recorded rather than assumed equal to `n_estimators`: a booster can stop early on
its own when a round finds no usable split, and a silent gap would make two folds incomparable
without saying so.

### `tuning_trials`

`study`, `model_name`, `fold_set`, `trial`, `params`, `mean_pr_auc`, `inner_fold_pr_aucs` (JSON),
`inner_folds`, `n_estimators`, `seconds`, `failed`, `failure`, `sampler_seed`, `region_start`,
`region_end`, `boosting_definition_version`.

⚠ **No number in this table is a result.** `mean_pr_auc` is measured on inner validation windows
that are training data for every fold the winning parameters are used on. See ADR 0018.

Failed trials are kept with `failed = true` and the exception text, rather than dropped — a silently
shorter table would read as "100 trials, all successful".

## 9. Determinism

Two runs over the same feature table, the same library versions and a single thread produce
**byte-identical** output. Four properties carry that:

1. Training rows are canonically sorted before every fit (`inspection_date`,
   `target_inspection_id`).
2. Both estimators are pinned to one thread, with LightGBM's `deterministic=True` and
   `force_row_wise=True`.
3. Every output table is sorted by a declared key before it is written.
4. No prediction, importance or training-log table carries a timestamp or a duration.

**Property 1 matters far more here than in Component 6.** Fitting the same 53,844 rows in a
shuffled order moves a prediction by **1.12e-01** (XGBoost) and **1.23e-01** (LightGBM), against
7.049e-09 for Component 6's coefficients — a booster draws row and column subsamples in row order.
Re-sorting a shuffled frame restores the fit exactly.

The determinism claim is narrow and deliberate: identical output for a fixed input, a fixed row
order, a fixed library set and one thread — **not** across library versions. The manifest records
`xgboost_version`, `lightgbm_version`, `numpy_version`, `sklearn_version` and `blas_threads`.

## 10. Guarantees a consumer may rely on

1. Every test row of every fold receives exactly one score from every model.
2. No duplicate `(model, fold, target_inspection_id)` row.
3. Every score is finite and inside [0, 1].
4. `trained_through` equals the fold's training end on every row, and is never null.
5. No fit read any row dated after its `trained_through`.
6. No fit read the calibration or test window, for training or for early stopping.
7. No model input is an identifier, a label or a provenance column.
8. The matrix reaching each estimator carries the source frame's NULLs unchanged.
9. Only `xgboost_class_weighted` carries a class weight, and it carries a real one.
10. Every importance is labelled with its matrix column.
11. Every hyperparameter was selected from data strictly earlier than the fold set's first test
    window.
12. The artifact is readable by `evaluation.contract.read_predictions` without translation.
13. Component 6's artifacts are untouched — verified: re-running `train-baselines` under the current
    library set reproduces sha256 `a2bb9411…00ff5b44`, matching the committed manifest.

All are re-derived per run by `boosting/validate.py` (17 checks) and recorded in the manifest.

## 11. Known limitations

1. **The improvement over Component 6 is small** — NDE 0.2326 → 0.2376 on the quarterly mean, +2.1%
   relative — and the logistic model's observed NDE sits inside XGBoost's seasonality redraw
   interval.
2. **The logistic model wins 7 of 17 quarterly folds.** The mean is not a per-quarter result.
3. **42.89% of violations are still surfaced later** than under business-as-usual.
4. **The two fold sets use different hyperparameters**, so a quarterly-versus-shift comparison
   confounds the regime with the parameter set. Report them separately.
5. **`covid_shift` is one fold tuned on two inner folds.** A robustness observation, not a
   measurement.
6. **Probabilities are uncalibrated.**
7. **Inspector effects cannot be modelled** — the field does not exist (ADR 0019) — so the gap
   between "cited" and "unsafe" cannot be characterised.
8. **Single-threaded fitting** will not scale to a much larger snapshot.

## 12. Reproducing

```bash
uv run sentinel tune-boosting --trials 100 --report      # then paste the printed block into definitions.py
uv run sentinel train-boosting --report
uv run sentinel evaluate --predictions data/processed/predictions/boosted_predictions_<stamp>.parquet --report
```

`train-boosting` does not require the search: the parameters are frozen literals in
`boosting/definitions.py`, and `TUNED_PARAMS_PROVENANCE` names the study artifact
(`tuning_trials_20260817T155315Z.parquet`, sha256 `a77687b7…adec8b14`) that produced them.
