# Data contract — calibrated predictions and calibrators

**Produced by:** Component 9 (`sentinel calibrate`)
**Layers:** `data/processed/predictions/` (the calibrated scores), `data/processed/tuning/`
(the method-selection log), `data/processed/calibration/` (the calibrators and their
diagnostics — a sixth processed layer)
**Consumed by:** Component 5 (`sentinel evaluate --predictions`); Component 12
(`sentinel audit-fairness`), which reads `score` and `base_score` **from the same row** — that
is what lets a group's calibrated and uncalibrated behaviour be compared with no join, and it
is why the group audit needs this artifact rather than Components 6–8's; Component 13 onwards,
which should start from the calibrated scores rather than from Components 6–8's raw ones
**Design rationale:** ADR 0012 (why a calibration window exists), ADR 0014 (predictions are
outputs), ADR 0024 (where these tables live), ADR 0025 (the selection protocol, pre-registered),
ADR 0026 (why the base models were re-executed), ADR 0027 (what the calibrator is fed)

---

## 1. Identity and file naming

Nine tables across three directories, one timestamp per run
(`<name>_<YYYYMMDDTHHMMSSZ>.parquet`, zstd):

```
data/processed/predictions/
  calibrated_predictions_<stamp>.parquet          207,680 rows   <- the anchor artifact
  manifest_calibrated_predictions_<stamp>.json    the run's manifest

data/processed/tuning/
  calibrator_selection_<stamp>.parquet            180 rows

data/processed/calibration/
  calibration_base_scores_<stamp>.parquet         378,985 rows
  calibrator_parameters_<stamp>.parquet
  calibrator_isotonic_breakpoints_<stamp>.parquet
  calibration_drift_<stamp>.parquet               360 rows
  calibration_ranking_preservation_<stamp>.parquet 180 rows
  calibration_brier_decomposition_<stamp>.parquet  360 rows
  calibration_bootstrap_<stamp>.parquet
```

Three directories because ADR 0014 and ADR 0018 each named a home for their grain in advance,
and ADR 0024 honours both rather than consolidating. The practical consequence is the one that
matters: the calibrated scores sit in `predictions/` with the contract's column set, so
`sentinel evaluate --predictions <file>` reads them **with no change to Component 5**.

Figures are documentation rather than data and live in `docs/analysis/figures/`:
`calibration_reliability_<model>_<fold_id>.png` (one per candidate for the latest quarterly
fold and for `covid_shift`) and `calibration_ece_drift.png`.

## 2. The models

Five base models are calibrated. Each is in the set for a stated reason, because three
different metrics disagreed about which model was best and choosing before calibrating would
have foreclosed the comparison.

| calibrated `model_name` | base model | from | why it is in the set |
|---|---|---|---|
| `logistic_regression_platt` | `logistic_regression` | C6 | the reference simple model; best MCE in the project before calibration |
| `xgboost_platt` | `xgboost` | C7 | the stronger tuned booster on NDE and ROC-AUC |
| `lightgbm_platt` | `lightgbm` | C7 | best precision@k_1_day; worst uncalibrated ECE of the four non-experimental candidates |
| `neural_numeric_only_platt` | `neural_numeric_only` | C8 | best uncalibrated Brier, ECE and NDE in the project |
| `xgboost_chain_embeddings_platt` ⚠ | `xgboost_chain_embeddings` | C8 | best PR-AUC. **Experimental** (ADR 0022) |

Ten other models have prediction artifacts on disk and are **not** calibrated. Every exclusion
is recorded with its reason in `calibration.definitions.EXCLUDED_MODELS` and copied into the
manifest — ablations, the CDPH approximation (it reaches only 3 of 10 input families), the
class-weighting variant, and `neural_embeddings`, which is fitted as a donor for the embedding
booster but never calibrated in its own right.

`model_name` is always `"<base>_<method>"`, never a bare base name, so a calibrated row and its
uncalibrated ancestor can sit in one results table without either being mistaken for the other.

## 3. Score semantics

`score` is a **calibrated** probability of a Priority or Priority Foundation citation on the
target inspection. Higher means higher risk — the same direction Components 5–8 use.

The chain, stated once so no consumer has to infer it:

```
base model  →  probability p           (committed by C6/C7/C8; uncalibrated)
p           →  logit(p)                (Component 9, Platt only; isotonic takes p)
logit(p)    →  score                   (the frozen per-fold calibrator)
```

The sigmoid is never applied twice, and an already-calibrated probability is never calibrated
again. `base_score` carries the uncalibrated probability alongside, so the correction is always
visible without joining back to Components 6–8.

**Measured quality** (quarterly mean over 17 folds): calibration slope 1.00–1.03, ECE
0.047–0.052, MCE 0.115–0.130. A predicted 0.30 happens about 30% of the time, to within roughly
5 percentage points on average. On `covid_shift` the slope reaches only 0.75–0.90 and ECE is
roughly double — see §10.

## 4. The three horizons

This is the part most likely to be misread, so the artifact carries all three fields rather
than one.

| column | value | meaning |
|---|---|---|
| `base_model_trained_through` | `fold.train_end` | the last date the **estimator's weights** learned from. Identical to what C6/C7/C8 declare |
| `calibrator_fitted_through` | `fold.calibration_end` | the last date the **correction** learned from |
| `calibrated_prediction_available_from` | `fold.test_start` | the first date this score could have been produced in operation |
| `trained_through` | `fold.calibration_end` | the contract field, and the **maximum** of the two above |

`trained_through` is the calibration end because that is the only honest single number: the
calibrator really did read the calibration window. Writing `train_end` there would be **false**.

This sits **exactly at the contract's ceiling, not past it** —
`evaluation.contract._training_horizon` returns `fold.calibration_end` and rejects only a
declared horizon *later* than it, which is what the calibration window exists for (ADR 0012,
ADR 0014). Worked example, `quarterly-2022Q2`:

```
train 2018-07-01 .. 2021-12-31   base_model_trained_through = 2021-12-31
cal   2022-01-01 .. 2022-03-31   calibrator_fitted_through  = 2022-03-31
                                 trained_through            = 2022-03-31
test  2022-04-01 .. 2022-06-30   available_from             = 2022-04-01
```

**This artifact must not be described as trained only through `train_end`.**

## 5. Where the scores came from, and why they had to be regenerated

No component persisted a fitted model, and no component ever scored a calibration window — the
34,261 calibration rows per model did not exist on disk. Component 9 therefore **re-executes
Components 6, 7 and 8's unchanged fit functions** (same spec, seed 42, hyperparameters, row
order) to obtain them. Full reasoning in ADR 0026.

The claim that these are the *same* models is proved rather than asserted. Every fit also scores
the test window, and the result is compared to the committed artifact with `==`:

> **207,680 regenerated test-window scores across 5 models × 18 folds, zero mismatches.**

`build.py` raises before fitting a single calibrator if this fails. `calibration_base_scores`
records the outcome per row in `reproduces_committed_artifact`.

**Components 6, 7 and 8's artifacts are unchanged and still byte-identical**, verified:

| artifact | sha256 (first 16) |
|---|---|
| `baseline_predictions_20260817T142118Z.parquet` | `a2bb94113bcc1c93…` |
| `boosted_predictions_20260817T160354Z.parquet` | `c8fcbd098f19650b…` |
| `neural_predictions_20260818T134648Z.parquet` | `a707a650b275da96…` |

⚠ **Bit-identity is environment-scoped**, and narrowly. scikit-learn 1.9.0, numpy 2.5.2,
xgboost 3.4.1, lightgbm 4.7.0, torch 2.13.0+cpu, one torch thread, CPU, **and the BLAS thread
count left at the library default**. An `OMP_NUM_THREADS=1` override alone moves
`logistic_regression`'s scores by up to 5e-10 and the gate correctly rejects it. A version bump
is an explicit re-baseline, never a reason to loosen the comparison.

## 6. The selection protocol

Both Platt and isotonic are fitted for every fold. One is frozen, by the rule pre-registered in
ADR 0025 **before the first production run**:

1. Cut the calibration window chronologically on a whole-day boundary, 70 / 30.
2. Fit both methods on the earlier portion; score both on the later one.
3. Fold *k*'s method is the winner on **mean inner-select log-loss over folds 1…k of the same
   fold set** — an expanding prefix, so every input has `rd ≤ fold_k.calibration_end`.
4. Choose isotonic only if it beats Platt by more than **0.005 nats**; otherwise Platt.
5. Refit the chosen method on the **full** calibration window, freeze, apply to test.

The prefix is not a stylistic choice. **Fold N's calibration window is fold N−1's test window**,
so pooling every fold to pick one method per model would choose fold 1's method using fold 1's
test period. A dedicated test asserts that the rejected design is detectably leaky.

Outcome on this data: **Platt was frozen for all 90 (model, fold) cells; 0 method switches.**
Per fold in isolation, isotonic would have won 16 of 90 — which is what the prefix exists to
smooth.

## 7. Schemas

### `calibrated_predictions`

| column | type | meaning |
|---|---|---|
| `target_inspection_id` | Utf8 | Component 3's key. The join key for everything |
| `score` | Float64 | the **calibrated** probability, in [0, 1], never null |
| `model_name` | Utf8 | `"<base>_<method>"` |
| `model_version` | Utf8 | `v1` |
| `fold_set` | Utf8 | `quarterly` or `covid_shift` |
| `fold_id` | Utf8 | e.g. `quarterly-2026Q2` |
| `trained_through` | Date | `= fold.calibration_end`. See §4 |
| `is_probability` | Boolean | always true |
| `base_model_name` / `base_model_version` | Utf8 | the uncalibrated ancestor |
| `base_score` | Float64 | the uncalibrated probability |
| `base_model_trained_through` | Date | `= fold.train_end` |
| `calibrator_fitted_through` | Date | `= fold.calibration_end` |
| `calibrated_prediction_available_from` | Date | `= fold.test_start` |
| `method` | Utf8 | `platt` or `isotonic` |
| `is_experimental` | Boolean | true only for `xgboost_chain_embeddings` |
| `calibration_definition_version` | Utf8 | `v1` |

Sorted by `(model_name, fold_set, fold_id, target_inspection_id)`.

### `calibrator_parameters` and `calibrator_isotonic_breakpoints`

Long form. Platt contributes two rows per (model, fold): `term ∈ {coef, intercept}`. Isotonic
contributes one `breakpoint_count` row plus one row per breakpoint in the breakpoints table,
carrying `x_threshold`, `y_threshold` and the clip bounds `x_min` / `x_max`.

**These two tables are sufficient to reproduce the mapping.** Platt is
`sigmoid(coef · logit(p) + intercept)`; isotonic is `np.interp(p, x_thresholds, y_thresholds)`
with clipping at the ends. `input_transform` records which (`logit` / `identity`), and
`was_selected` records whether that method was the frozen one. **Both methods are written for
every fold, including the one that lost**, so the counterfactual is answerable from the artifact
rather than by re-running with a different flag — which is how a selection quietly becomes a
test-set selection.

The figures replay both calibrators from these tables, which is a standing demonstration that
they really are sufficient.

### The remaining tables

`calibration_base_scores` — one row per (model, fold, window, scored row), with `base_score`,
`base_logit`, `native_margin` (nullable), `target`, `inner_portion`
(`inner_fit`/`inner_select`/empty) and `reproduces_committed_artifact` (null on calibration rows,
because there is nothing committed to compare them against).

`calibrator_selection` — one row per (model, fold, method): both methods' inner-select log-loss,
Brier, ECE and MCE, the prefix mean, `per_fold_winner`, `prefix_winner`, `gap_to_other`, the
`tie_threshold` in force, `declared_tie`, and the `selection_reason` in prose.

`calibration_drift` — one row per (model, fold, **stage**), stage ∈ {`uncalibrated`, `platt`,
`isotonic`, `selected`}: ECE, MCE, Brier, log-loss, calibration slope and intercept, mean
predicted, observed rate, test base rate and `prior_shift`.

`calibration_ranking_preservation` — Spearman ρ, Kendall τ-b, `inversions`, distinct-score
counts before/after, `new_ties_created`, `is_strictly_monotone`, top-k membership change,
precision@k and ROC-AUC before/after.

`calibration_brier_decomposition` — `reliability`, `resolution`, `uncertainty`, `recomposed` and
`within_bin_variance`, over 15 equal-mass bins.

`calibration_bootstrap` — within-fold percentile intervals for ECE, Brier and log-loss under two
resampling schemes (`row`, `establishment_block`), 1,000 replications, seed 20260824.

## 8. Determinism

Two runs over the same inputs produce byte-identical tables. **Verified**, by running the
component twice into separate directories and comparing sha256 per table: eight of the nine
match exactly.

That check earned its keep. The first comparison failed on
`calibration_bootstrap_*.parquet`, because the bootstrap seed key was derived from
`hash(model_name)` and **Python salts `str` hashing per process** — so every run drew different
resamples. It now uses the candidate's position in the frozen registry, and a regression test
pins that mapping. No table carries a timestamp or a
duration, with one exception: `calibrator_selection.seconds`, under the narrow exception ADR
0018 already granted the tuning layer, which makes no determinism claim about its bytes.

Isotonic is exactly order-invariant. Platt is not, and the measured effect is **one ULP**
(2.2e-16 on an applied probability) — the same lbfgs summation-order sensitivity Component 6
recorded for its own coefficients. It is moot in production because every window comes from
`calibration_frame`, which sorts canonically by `(rd, target_inspection_id)`.

## 9. Guarantees a consumer may rely on

1. Every candidate scored **every test row of every fold exactly once** — 41,536 rows per model,
   no row dropped, no row duplicated, coverage re-derived from `window_frame`.
2. Every `score` is finite and in [0, 1]; nothing is null and nothing was imputed.
3. **No test row entered any calibrator fit**, re-derived by intersecting each fit's row ids
   with the fold's test window.
4. Every calibrator fit row re-derives to `split == "calibration"` via `assign_split`.
5. Every calibrator is **monotone**, probed over 200 points spanning (0, 1).
6. **Platt changed no ranking, exactly**: 0 inversions, 0 new ties, Spearman ρ = 1.0, and
   PR-AUC / ROC-AUC / NDE / precision@k identical to Components 6–8's to the last bit, verified
   independently by re-running `sentinel evaluate`.
7. Each (model, fold) has exactly one frozen calibrator, fitted on its own window. No fold
   borrows another's, and `covid_shift` never borrows a quarterly one.
8. The manifest's selection rule matches the literals in `calibration/definitions.py`, so a run
   cannot report a rule it did not apply.
9. The persisted parameters reproduce the fitted estimator to within 1e-9.

## 10. ⚠ What a consumer may not use

**`data/processed/calibration/` may not be joined onto a feature table, and the
calibration-window scores in particular must never reach a fit.** Those rows sit after
`train_end`; a base model that saw them would have been fitted past its own declared horizon.

**No number in `calibrator_selection_*.parquet` is a result.** Every one was measured on a
window carved out of the calibration period. An inner-select log-loss read as a headline metric
would be an in-sample number reported as out-of-sample.

**Nothing in `data/processed/calibration/` is the project's headline.** The headline PR-AUC,
ROC-AUC, NDE, precision@k and days-earlier come from `sentinel evaluate --predictions`, exactly
as they do for an uncalibrated model. Component 5 remains the only producer of those.

**`xgboost_chain_embeddings_platt` is experimental** (`is_experimental = true`, ADR 0022). It
posts the best calibrated Brier, and that must not make it the headline: it lost on NDE, and
Component 8's embeddings were explicitly not adopted.

**`covid_shift` must not be averaged into the quarterly mean.** Its ECE is roughly double and
its post-calibration slope reaches only 0.75–0.90, because the base rate moved 17 points between
its calibration and test windows. Its probabilities are the least trustworthy in the project.

**The calibrator must not be refitted on a test quarter**, including as a diagnostic. That
reintroduces exactly the leak ADR 0012 built the calibration window to prevent.

**The retraining trigger proposed in `calibration_findings.md` §12.8 is a design proposal**, not
a validated threshold. It was written after seeing the drift series and has not been checked
against any outcome.

## 11. Known limitations

1. **Seed 42 only.** Component 8's neural advantage (0.0053 ROC-AUC) is smaller than its own
   five-seed spread (0.0058). Averaging over seeds was deferred by decision — it would create a
   base model Component 8 never evaluated and break the bit-identity gate — not by oversight.
2. **No fitted model is persisted, still.** Reproducing this component requires re-running the
   fits (~340 s of refitting, ~25 min end to end). A future component that wants cheaper
   reproduction should persist the estimators.
3. **The native decision margin is not available for `xgboost_chain_embeddings`**, because
   reaching it would need a private helper of a closed component. Its logit-recovery check
   reports "not available" rather than passing.
4. **The bootstrap covers ECE, Brier and log-loss at the `uncalibrated` and `selected` stages
   only.** The calibration slope is not bootstrapped — it refits a logistic regression per
   replication — and its point estimate is in `calibration_drift` for every stage.
5. **Prior shift is not corrected and cannot be** by a monotone recalibration fitted on an
   earlier window. On `covid_shift` this is most of the residual error.
6. **The improvement is visible in the 17-fold mean, not in any single quarter**: within-fold
   bootstrap intervals for before and after overlap on one quarter.

## 12. Reproducing

```bash
uv run python scripts/profile_calibration.py          # the pre-registration evidence
uv run sentinel calibrate --report                    # ~25 minutes
uv run sentinel evaluate --predictions data/processed/predictions/calibrated_predictions_<stamp>.parquet
```

Run `calibrate` **without** an `OMP_NUM_THREADS` override — see §5.
