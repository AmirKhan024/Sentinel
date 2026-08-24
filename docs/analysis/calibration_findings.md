# Component 9 — Probability calibration: findings

**Source:** `data/processed/features/as_of_features_20260816T150313Z.parquet` (57,727 rows,
`feature_definition_version = v1`,
sha256 `b7db5b2d3d25bf4ccd251a8614aa08f1167cadcbe86d6c6788d665e745825e2f`)
**Base models re-executed from:** Components 6, 7 and 8, unchanged (ADR 0026)
**Profiling command:** `uv run python scripts/profile_calibration.py`
**Libraries:** scikit-learn 1.9.0, numpy 2.5.2, polars 1.43.2, xgboost 3.4.1, lightgbm 4.7.0,
torch 2.13.0+cpu
**Device:** CPU, one thread, `torch.use_deterministic_algorithms(True)`

⚠ **Sections 1–9 contain no test-window outcome, and that was a hard rule rather than a style
preference.** Every score, calibrator and metric before §10 was computed on a fold's *calibration*
window — the window this component exists to use, which sits strictly after `train_end` and
strictly before `test_start`, and which no earlier component has read. Two profiles touch the test
window and only these two: §7 reads its *prevalence*, which Component 5 already published in
`evaluation_folds_*.parquet` and which is a property of the data rather than of any model, and §8
counts repeated establishments, which carries no outcome.

The reason for the discipline: Component 5 protects evaluation time, but it cannot protect against
a human reading a test metric, changing a threshold and re-running. That loop is leakage, it leaves
no trace in any artifact, and no check in this repository can detect it.

**Everything in §1–§9 was measured before `TIE_THRESHOLD` was frozen, and §6 is what fixed it.**
ADR 0025 records the rule with a date. §10 onwards reports the evaluation, which was run once,
after the rule was committed.

Every number here was measured on the full snapshot. Nothing is estimated, illustrative or carried
over from the project specification. Where the data contradicts an expectation, the data wins and
the divergence is recorded.

---

## 1. What this component is for, and what it is not for

Components 6, 7 and 8 built a **ranking**. The best of them, `neural_numeric_only`, reaches
NDE 0.2482 and finds Priority violations 5.7 days earlier than business-as-usual scheduling.

Component 9 asks a different question, and it is worth being blunt that it is a different one:

> When Sentinel says an establishment has a 0.30 probability of a Priority violation, does that
> happen 30% of the time?

Nothing in Components 6–8 answers this. A ranking is invariant to any strictly increasing
transformation of the score, so a model can rank perfectly and still say 0.30 when it means 0.45.
The downstream components need the number, not only the order: a cost threshold, a deferral gate
and any statutory prioritisation all consume a probability and are meaningless if it is not one.

**Calibration is not expected to improve the ranking, and cannot.** A monotone map leaves every
pairwise comparison intact, so NDE, PR-AUC and precision@k are unchanged by construction. That is
the intended outcome, not a null result. The one exception is discussed in §9 and §14: isotonic
regression is *weakly* monotone, and its plateaus create ties that can move a top-k membership.

### The blocker, stated plainly

The scores this component calibrates **did not exist when it started**. Every prediction artifact
on disk covers exactly the test window (41,536 rows per model over 18 folds); the 34,261
calibration-window rows were never scored, and no fitted model object is persisted anywhere in the
repository. The resolution — re-executing Components 6–8's unchanged fit functions under a
bit-identity gate — is ADR 0026. The 34,261 scores per model profiled below are the first
calibration-window scores this project has ever produced.

---

## 2. The calibration window is real, and it is small (profile `calibration_window_size`)

| fold_id | calibration window | rows | days | positives | base rate |
|---|---|---|---|---|---|
| quarterly-2022Q2 | 2022-01-01 .. 2022-03-31 | 1357 | 59 | 585 | 0.4311 |
| quarterly-2022Q3 | 2022-04-01 .. 2022-06-30 | 1762 | 63 | 821 | 0.4659 |
| quarterly-2022Q4 | 2022-07-01 .. 2022-09-30 | 1733 | 64 | 852 | 0.4916 |
| quarterly-2023Q1 | 2022-10-01 .. 2022-12-31 | 1700 | 61 | 791 | 0.4653 |
| quarterly-2023Q2 | 2023-01-01 .. 2023-03-31 | 1801 | 59 | 802 | 0.4453 |
| quarterly-2023Q3 | 2023-04-01 .. 2023-06-30 | 1787 | 63 | 866 | 0.4846 |
| quarterly-2023Q4 | 2023-07-01 .. 2023-09-30 | 1650 | 63 | 798 | 0.4836 |
| quarterly-2024Q1 | 2023-10-01 .. 2023-12-31 | 1958 | 60 | 848 | 0.4331 |
| quarterly-2024Q2 | 2024-01-01 .. 2024-03-31 | 1913 | 59 | 877 | 0.4584 |
| quarterly-2024Q3 | 2024-04-01 .. 2024-06-30 | 2196 | 59 | 987 | 0.4495 |
| quarterly-2024Q4 | 2024-07-01 .. 2024-09-30 | 2124 | 64 | 896 | 0.4218 |
| quarterly-2025Q1 | 2024-10-01 .. 2024-12-31 | 2248 | 62 | 853 | 0.3794 |
| quarterly-2025Q2 | 2025-01-01 .. 2025-03-31 | 2459 | 58 | 936 | 0.3806 |
| quarterly-2025Q3 | 2025-04-01 .. 2025-06-30 | 2290 | 62 | 893 | 0.3900 |
| quarterly-2025Q4 | 2025-07-01 .. 2025-09-30 | 1754 | 64 | 742 | 0.4230 |
| quarterly-2026Q1 | 2025-10-01 .. 2025-12-31 | 1766 | 61 | 671 | 0.3800 |
| quarterly-2026Q2 | 2026-01-01 .. 2026-03-31 | 1917 | 57 | 750 | 0.3912 |
| covid_shift-2020H2-2021 | 2020-03-01 .. 2020-05-31 | 1846 | 61 | 1260 | 0.6826 |

Three facts decide the rest of the design.

**The windows are small.** 1,357 to 2,459 rows, median around 1,800. A training window at the same
point in the fold sequence holds 23,346 to 53,844. Whatever is fitted here has to be fitted on
roughly a thirtieth of the data the base model saw, which is the argument for a two-parameter
calibrator and against a flexible one.

**The base rate moves.** From 0.4916 (2022Q4) down to 0.3794 (2025Q1) across the quarterly folds —
an 11-point swing — and 0.6826 on `covid_shift`, which is a different regime entirely. A calibrator
fitted on a 0.68 window and applied to a 0.51 window is being asked to correct something it has not
seen.

**Fold *N*'s calibration window is fold *N−1*'s test window.** `quarterly-2022Q3` calibrates on
2022-04-01…06-30, which is exactly `quarterly-2022Q2`'s test period, and the row counts confirm it
(1,762 in both). ADR 0017 already used this to reject tuning across all folds. It is the reason the
selection protocol in §6 cannot pool folds naively, and it is the single most likely way this
component could have shipped a quiet leak.

---

## 3. The inner split lands where it should (profile `inner_split_placement`)

Choosing between Platt and isotonic requires held-out data, and it may not be test data. So each
calibration window is cut chronologically into an **inner-fit** portion and an **inner-select**
portion, on a whole-day boundary — reusing `neural.train.inner_split_date`, which walks distinct
dates backwards rather than taking a row quantile. Component 8's reasoning applies unchanged: two
inspections of the same establishment days apart share almost all of their as-of history, so a cut
that lands mid-day splits rows that are not independent.

The fraction is **0.30**, not Component 8's 0.15, because a calibration window is an order of
magnitude smaller than a training window. At 0.15 the smallest inner-select portion would be about
204 rows.

| fold_id | cut date | inner-fit rows | fit rate | inner-select rows | select rate | select share | minimums |
|---|---|---|---|---|---|---|---|
| quarterly-2022Q2 | 2022-03-10 | 948 | 0.4156 | 409 | 0.4670 | 0.301 | ok |
| quarterly-2022Q3 | 2022-06-06 | 1213 | 0.4493 | 549 | 0.5027 | 0.312 | ok |
| quarterly-2022Q4 | 2022-09-08 | 1198 | 0.5100 | 535 | 0.4505 | 0.309 | ok |
| quarterly-2023Q1 | 2022-11-30 | 1152 | 0.4740 | 548 | 0.4471 | 0.322 | ok |
| quarterly-2023Q2 | 2023-03-10 | 1260 | 0.4381 | 541 | 0.4621 | 0.300 | ok |
| quarterly-2023Q3 | 2023-06-02 | 1237 | 0.4753 | 550 | 0.5055 | 0.308 | ok |
| quarterly-2023Q4 | 2023-09-06 | 1135 | 0.4934 | 515 | 0.4621 | 0.312 | ok |
| quarterly-2024Q1 | 2023-11-30 | 1352 | 0.4438 | 606 | 0.4092 | 0.309 | ok |
| quarterly-2024Q2 | 2024-03-06 | 1318 | 0.4583 | 595 | 0.4588 | 0.311 | ok |
| quarterly-2024Q3 | 2024-05-24 | 1535 | 0.4397 | 661 | 0.4720 | 0.301 | ok |
| quarterly-2024Q4 | 2024-09-09 | 1465 | 0.4341 | 659 | 0.3945 | 0.310 | ok |
| quarterly-2025Q1 | 2024-12-03 | 1555 | 0.3955 | 693 | 0.3434 | 0.308 | ok |
| quarterly-2025Q2 | 2025-03-10 | 1703 | 0.3740 | 756 | 0.3955 | 0.307 | ok |
| quarterly-2025Q3 | 2025-05-29 | 1594 | 0.3739 | 696 | 0.4267 | 0.304 | ok |
| quarterly-2025Q4 | 2025-09-03 | 1208 | 0.4296 | 546 | 0.4084 | 0.311 | ok |
| quarterly-2026Q1 | 2025-12-02 | 1222 | 0.3936 | 544 | 0.3493 | 0.308 | ok |
| quarterly-2026Q2 | 2026-02-25 | 1333 | 0.4014 | 584 | 0.3682 | 0.305 | ok |
| covid_shift-2020H2-2021 | 2020-04-29 | 1269 | 0.6753 | 577 | 0.6984 | 0.313 | ok |

Every fold clears both declared minimums (`MIN_INNER_FIT_ROWS = 400`,
`MIN_INNER_SELECT_ROWS = 250`) with room to spare: the smallest inner-fit portion is 948 rows
(2022Q2) and the smallest inner-select portion is 409. The realised select share is 0.300–0.322 —
the whole-day constraint costs at most 2.2 percentage points of drift from the target.

The minimums are still enforced in code, and a fold that trips them is **refused rather than
calibrated on a window too small to mean anything** — the same posture
`neural.train.split_training_window` takes. On this snapshot none do; if a future snapshot changes
that, it surfaces as a failure rather than as a silent method switch.

---

## 4. What the calibrator is given (profile `score_distribution`)

| model | rows | min p | max p | min logit | max logit | distinct | saturated |
|---|---|---|---|---|---|---|---|
| lightgbm | 34261 | 0.045338 | 0.950973 | -3.047 | 2.965 | 32575 | 0 |
| logistic_regression | 34261 | 0.000028 | 0.999623 | -10.483 | 7.884 | 33937 | 0 |
| neural_numeric_only | 34261 | 0.003384 | 0.997919 | -5.685 | 6.173 | 33880 | 0 |
| xgboost | 34261 | 0.039826 | 0.960094 | -3.183 | 3.181 | 32880 | 0 |

**Nothing saturates.** Not one score of the 34,261 per model sits at exactly 0.0 or 1.0, so
`logit(p)` is finite everywhere and no clamping is needed. This is worth recording because ADR 0020
predicted the opposite for the network — *"a network's overconfidence is the thing that component
will have to correct"* — and the measurement does not support it. `neural_numeric_only`'s scores
span 0.0034 to 0.9979, a **narrower** range than `logistic_regression`'s 0.000028 to 0.9996.

The two tree models occupy a strikingly narrow band: lightgbm 0.045–0.951, xgboost 0.040–0.960, or
about ±3.2 in logit space against the GLM's −10.5 to +7.9. A model that never says less than 0.04
or more than 0.96 has limited room to be *over*confident and considerable room to be
*under*confident, which is the opposite of the usual expectation for boosted trees and is a
consequence of the shallow tuned depths Component 7 selected.

Distinct-score counts (32,575–33,937 of 34,261) matter for §9: the base scores already contain ties
before any calibrator touches them, so ranking preservation must be measured against that baseline
rather than against an assumed all-distinct ordering.

---

## 5. The recovered logit is the right input, and float32 is why it is not exact
   (profile `logit_round_trip`)

The calibrator needs a real-valued input. Platt conventionally fits a logistic regression on the
base model's raw decision score. No component persists one — every artifact stores the probability
after `predict_proba` or `torch.sigmoid`.

Both routes are available. The native margin is reachable through public attributes
(`pipeline.decision_function`, XGBoost's `output_margin=True`, LightGBM's `raw_score=True`,
`neural.train.scorer_for` before the sigmoid) with no change to any closed component. The
alternative is to recover it: `logit(p) = log(p) − log1p(−p)`, written that way rather than as
`log(p / (1 − p))` because `1 − p` cancels catastrophically for p near 1 while `log1p` is accurate
there by construction.

**The recovered logit is what Component 9 calibrates** (ADR 0027). The measurement below is the
check on that choice, and it produced the most interesting surprise in this profiling run.

| model | max |logit(p) - margin| | mean | rows over 1e-9 |
|---|---|---|---|
| lightgbm | 1.776e-15 | 1.367e-16 | 0 |
| logistic_regression | 2.602e-13 | 1.985e-16 | 0 |
| neural_numeric_only | 2.615e-05 | 8.148e-08 | 33898 |
| xgboost | 1.393e-06 | 7.122e-08 | 33922 |

For the two float64 models the recovery is essentially exact: lightgbm to 1.8e-15, logistic
regression to 2.6e-13. For the other two it is not, and the reason is not the arithmetic in
`logit()`:

- **`xgboost` computes its margin in float32.** Max discrepancy 1.4e-6, mean 7.1e-8.
- **`neural_numeric_only` computes in float32 throughout** — the network's forward pass, the logit
  and the sigmoid. Max discrepancy 2.6e-5, mean 8.1e-8.

Float32 has about 1.2e-7 of relative precision. A logit is passed through a sigmoid, rounded to
float32, widened to float64 for storage, and then inverted; the round trip cannot recover more than
float32 carried, and the error is amplified in the tails where the sigmoid is flattest. **The
persisted probability is the float64 image of a float32 sigmoid, so `logit(score)` is the best
possible recovery from what exists on disk.** The discrepancy is a property of the base model's
precision, not of the recovery.

This does not change the decision, and arguably strengthens it. Calibrating the recovered logit
makes the calibrated test artifact a pure function of the *already-committed* `score` column plus
two fitted floats — reproducible from artifacts alone, years later, with no live model. Calibrating
the native margin would make it a function of a value persisted nowhere.

The practical size of the error is negligible for the purpose: 2.6e-5 in logit space moves a
calibrated probability by roughly `2.6e-5 × slope × p(1−p)`, i.e. under 1e-5 — four orders of
magnitude below the ECE differences this component is measuring.

**Consequence for the validation check.** The plan pre-declared a warn-severity assertion that
`max |logit(p) − margin| < 1e-9`. That threshold was set from the float64 expectation and is
**wrong**: it would fire on 33,898 of 34,261 neural rows and 33,922 xgboost rows, every one of them
correct behaviour. The threshold is set from this measurement instead, at **1e-4** — comfortably
above the observed 2.6e-5 maximum and still tight enough to catch the defect it exists for. A
double sigmoid, a sign flip or a mis-join would show up as an O(1) discrepancy, not an O(1e-5) one.

---

## 6. How noisy is the choice between Platt and isotonic? (profile `selection_metric_noise`)

**This is the profile that fixes `TIE_THRESHOLD`, and it was run before the threshold existed.**

### Why the selection metric is log-loss and not ECE

| fold_id | cal rows | rows/bin (full) | select rows | rows/bin (select) | ~positives/bin |
|---|---|---|---|---|---|
| quarterly-2022Q2 | 1357 | 90.5 | 409 | 27.3 | 11 |
| quarterly-2022Q3 | 1762 | 117.5 | 549 | 36.6 | 15 |
| quarterly-2022Q4 | 1733 | 115.5 | 535 | 35.7 | 15 |
| quarterly-2023Q1 | 1700 | 113.3 | 548 | 36.5 | 15 |
| quarterly-2023Q2 | 1801 | 120.1 | 541 | 36.1 | 15 |
| quarterly-2023Q3 | 1787 | 119.1 | 550 | 36.7 | 15 |
| quarterly-2023Q4 | 1650 | 110.0 | 515 | 34.3 | 14 |
| quarterly-2024Q1 | 1958 | 130.5 | 606 | 40.4 | 17 |
| quarterly-2024Q2 | 1913 | 127.5 | 595 | 39.7 | 17 |
| quarterly-2024Q3 | 2196 | 146.4 | 661 | 44.1 | 19 |
| quarterly-2024Q4 | 2124 | 141.6 | 659 | 43.9 | 18 |
| quarterly-2025Q1 | 2248 | 149.9 | 693 | 46.2 | 19 |
| quarterly-2025Q2 | 2459 | 163.9 | 756 | 50.4 | 21 |
| quarterly-2025Q3 | 2290 | 152.7 | 696 | 46.4 | 19 |
| quarterly-2025Q4 | 1754 | 116.9 | 546 | 36.4 | 15 |
| quarterly-2026Q1 | 1766 | 117.7 | 544 | 36.3 | 15 |
| quarterly-2026Q2 | 1917 | 127.8 | 584 | 38.9 | 16 |
| covid_shift-2020H2-2021 | 1846 | 123.1 | 577 | 38.5 | 16 |

The project specification asks for ECE with 15 equal-mass bins, and Component 5 implements exactly
that. On a **full** calibration window that is 90–164 rows per bin, which is usable. On an
**inner-select** window it is 27–50 rows per bin, of which 11–21 are positives.

Three reasons the selection metric is mean inner-select **log-loss** instead:

1. At ~15 positives per bin, the sampling noise in a bin's observed rate is comparable to the
   effect being measured.
2. ECE is not a proper scoring rule. A calibrator that reshuffles probabilities *within* a bin can
   lower ECE without being better, and a degenerate calibrator that predicts the base rate for
   every row scores an ECE near zero while carrying no information at all.
3. Log-loss has no free parameter. A selection made on ECE could be changed by changing the bin
   count, and a tunable selection rule is not a rule.

ECE, MCE and Brier are still recorded on the inner-select window in the selection log, as
diagnostics that decide nothing.

### The measurement

For every (model, fold): fit both calibrators on the inner-fit portion, score the inner-select
portion, then resample the inner-select rows 1,000 times. Both methods are scored on the **same**
resample, so their shared variation cancels and the paired gap's SD is the resolution of the
comparison itself.

| model | fold_id | select rows | platt log-loss | isotonic log-loss | gap | SD(log-loss) | SD(paired gap) | lower |
|---|---|---|---|---|---|---|---|---|
| lightgbm | quarterly-2022Q2 | 409 | 0.6830 | 0.6865 | 0.0035 | 0.0117 | 0.0030 | platt |
| logistic_regression | quarterly-2022Q2 | 409 | 0.6783 | 0.6739 | -0.0044 | 0.0106 | 0.0032 | isotonic |
| neural_numeric_only | quarterly-2022Q2 | 409 | 0.6794 | 0.6791 | -0.0003 | 0.0111 | 0.0044 | isotonic |
| xgboost | quarterly-2022Q2 | 409 | 0.6836 | 0.7668 | 0.0832 | 0.0113 | 0.0796 | platt |
| lightgbm | quarterly-2022Q3 | 549 | 0.6944 | 0.6974 | 0.0031 | 0.0079 | 0.0027 | platt |
| logistic_regression | quarterly-2022Q3 | 549 | 0.6915 | 0.7027 | 0.0111 | 0.0082 | 0.0055 | platt |
| neural_numeric_only | quarterly-2022Q3 | 549 | 0.6887 | 0.7527 | 0.0640 | 0.0085 | 0.0610 | platt |
| xgboost | quarterly-2022Q3 | 549 | 0.6987 | 0.6954 | -0.0033 | 0.0099 | 0.0037 | isotonic |
| lightgbm | quarterly-2022Q4 | 535 | 0.6850 | 0.7502 | 0.0651 | 0.0076 | 0.0629 | platt |
| logistic_regression | quarterly-2022Q4 | 535 | 0.6867 | 0.8717 | 0.1850 | 0.0092 | 0.1078 | platt |
| neural_numeric_only | quarterly-2022Q4 | 535 | 0.6750 | 0.7379 | 0.0629 | 0.0080 | 0.0611 | platt |
| xgboost | quarterly-2022Q4 | 535 | 0.6762 | 0.6812 | 0.0050 | 0.0068 | 0.0032 | platt |
| lightgbm | quarterly-2023Q1 | 548 | 0.6823 | 1.0467 | 0.3645 | 0.0112 | 0.1463 | platt |
| logistic_regression | quarterly-2023Q1 | 548 | 0.6893 | 0.8255 | 0.1362 | 0.0107 | 0.0860 | platt |
| neural_numeric_only | quarterly-2023Q1 | 548 | 0.6850 | 1.1028 | 0.4178 | 0.0094 | 0.1595 | platt |
| xgboost | quarterly-2023Q1 | 548 | 0.6867 | 0.8766 | 0.1899 | 0.0104 | 0.1021 | platt |
| lightgbm | quarterly-2023Q2 | 541 | 0.6784 | 0.6764 | -0.0019 | 0.0083 | 0.0029 | isotonic |
| logistic_regression | quarterly-2023Q2 | 541 | 0.6873 | 0.7441 | 0.0567 | 0.0103 | 0.0540 | platt |
| neural_numeric_only | quarterly-2023Q2 | 541 | 0.6834 | 0.7399 | 0.0565 | 0.0090 | 0.0635 | platt |
| xgboost | quarterly-2023Q2 | 541 | 0.6814 | 0.6828 | 0.0014 | 0.0087 | 0.0029 | platt |
| lightgbm | quarterly-2023Q3 | 550 | 0.6851 | 0.6948 | 0.0098 | 0.0085 | 0.0051 | platt |
| logistic_regression | quarterly-2023Q3 | 550 | 0.6713 | 0.7275 | 0.0562 | 0.0089 | 0.0634 | platt |
| neural_numeric_only | quarterly-2023Q3 | 550 | 0.6722 | 0.6661 | -0.0060 | 0.0083 | 0.0045 | isotonic |
| xgboost | quarterly-2023Q3 | 550 | 0.6800 | 0.6799 | -0.0001 | 0.0083 | 0.0040 | isotonic |
| lightgbm | quarterly-2023Q4 | 515 | 0.6605 | 0.7855 | 0.1250 | 0.0099 | 0.0911 | platt |
| logistic_regression | quarterly-2023Q4 | 515 | 0.6612 | 0.6662 | 0.0050 | 0.0091 | 0.0054 | platt |
| neural_numeric_only | quarterly-2023Q4 | 515 | 0.6575 | 0.7806 | 0.1231 | 0.0096 | 0.0921 | platt |
| xgboost | quarterly-2023Q4 | 515 | 0.6673 | 0.6723 | 0.0050 | 0.0096 | 0.0051 | platt |
| lightgbm | quarterly-2024Q1 | 606 | 0.6582 | 0.8223 | 0.1641 | 0.0099 | 0.0944 | platt |
| logistic_regression | quarterly-2024Q1 | 606 | 0.6617 | 0.9842 | 0.3225 | 0.0117 | 0.1313 | platt |
| neural_numeric_only | quarterly-2024Q1 | 606 | 0.6584 | 0.8359 | 0.1776 | 0.0105 | 0.0922 | platt |
| xgboost | quarterly-2024Q1 | 606 | 0.6577 | 0.6664 | 0.0087 | 0.0099 | 0.0054 | platt |
| lightgbm | quarterly-2024Q2 | 595 | 0.6647 | 0.7317 | 0.0670 | 0.0079 | 0.0580 | platt |
| logistic_regression | quarterly-2024Q2 | 595 | 0.6661 | 0.7281 | 0.0620 | 0.0074 | 0.0555 | platt |
| neural_numeric_only | quarterly-2024Q2 | 595 | 0.6600 | 0.6709 | 0.0108 | 0.0084 | 0.0041 | platt |
| xgboost | quarterly-2024Q2 | 595 | 0.6650 | 0.7205 | 0.0555 | 0.0082 | 0.0545 | platt |
| lightgbm | quarterly-2024Q3 | 661 | 0.6679 | 0.7265 | 0.0586 | 0.0101 | 0.0494 | platt |
| logistic_regression | quarterly-2024Q3 | 661 | 0.6763 | 0.6742 | -0.0020 | 0.0077 | 0.0030 | isotonic |
| neural_numeric_only | quarterly-2024Q3 | 661 | 0.6710 | 0.6650 | -0.0060 | 0.0084 | 0.0028 | isotonic |
| xgboost | quarterly-2024Q3 | 661 | 0.6686 | 0.6711 | 0.0025 | 0.0087 | 0.0024 | platt |
| lightgbm | quarterly-2024Q4 | 659 | 0.6527 | 0.6562 | 0.0035 | 0.0090 | 0.0034 | platt |
| logistic_regression | quarterly-2024Q4 | 659 | 0.6607 | 0.6659 | 0.0051 | 0.0115 | 0.0037 | platt |
| neural_numeric_only | quarterly-2024Q4 | 659 | 0.6561 | 0.6660 | 0.0099 | 0.0109 | 0.0030 | platt |
| xgboost | quarterly-2024Q4 | 659 | 0.6553 | 0.6557 | 0.0004 | 0.0093 | 0.0023 | platt |
| lightgbm | quarterly-2025Q1 | 693 | 0.6371 | 0.6467 | 0.0096 | 0.0105 | 0.0038 | platt |
| logistic_regression | quarterly-2025Q1 | 693 | 0.6311 | 0.6339 | 0.0028 | 0.0103 | 0.0031 | platt |
| neural_numeric_only | quarterly-2025Q1 | 693 | 0.6216 | 0.6185 | -0.0032 | 0.0099 | 0.0023 | isotonic |
| xgboost | quarterly-2025Q1 | 693 | 0.6365 | 0.6419 | 0.0054 | 0.0104 | 0.0025 | platt |
| lightgbm | quarterly-2025Q2 | 756 | 0.6411 | 0.6409 | -0.0001 | 0.0116 | 0.0022 | isotonic |
| logistic_regression | quarterly-2025Q2 | 756 | 0.6378 | 0.6790 | 0.0411 | 0.0116 | 0.0439 | platt |
| neural_numeric_only | quarterly-2025Q2 | 756 | 0.6290 | 0.6262 | -0.0028 | 0.0122 | 0.0029 | isotonic |
| xgboost | quarterly-2025Q2 | 756 | 0.6335 | 0.6773 | 0.0438 | 0.0121 | 0.0429 | platt |
| lightgbm | quarterly-2025Q3 | 696 | 0.6385 | 0.6422 | 0.0037 | 0.0130 | 0.0027 | platt |
| logistic_regression | quarterly-2025Q3 | 696 | 0.6401 | 0.6836 | 0.0435 | 0.0121 | 0.0472 | platt |
| neural_numeric_only | quarterly-2025Q3 | 696 | 0.6317 | 0.7351 | 0.1034 | 0.0136 | 0.0656 | platt |
| xgboost | quarterly-2025Q3 | 696 | 0.6406 | 0.6419 | 0.0013 | 0.0131 | 0.0030 | platt |
| lightgbm | quarterly-2025Q4 | 546 | 0.6391 | 0.8785 | 0.2395 | 0.0117 | 0.1239 | platt |
| logistic_regression | quarterly-2025Q4 | 546 | 0.6609 | 0.7679 | 0.1070 | 0.0185 | 0.0746 | platt |
| neural_numeric_only | quarterly-2025Q4 | 546 | 0.6409 | 0.8174 | 0.1765 | 0.0126 | 0.1012 | platt |
| xgboost | quarterly-2025Q4 | 546 | 0.6389 | 0.7614 | 0.1226 | 0.0123 | 0.0875 | platt |
| lightgbm | quarterly-2026Q1 | 544 | 0.6546 | 0.7777 | 0.1231 | 0.0149 | 0.0859 | platt |
| logistic_regression | quarterly-2026Q1 | 544 | 0.6352 | 0.6367 | 0.0015 | 0.0134 | 0.0036 | platt |
| neural_numeric_only | quarterly-2026Q1 | 544 | 0.6440 | 0.6495 | 0.0055 | 0.0147 | 0.0046 | platt |
| xgboost | quarterly-2026Q1 | 544 | 0.6375 | 0.6512 | 0.0137 | 0.0133 | 0.0070 | platt |
| lightgbm | quarterly-2026Q2 | 584 | 0.6445 | 0.6469 | 0.0024 | 0.0111 | 0.0039 | platt |
| logistic_regression | quarterly-2026Q2 | 584 | 0.6537 | 0.6520 | -0.0016 | 0.0129 | 0.0068 | isotonic |
| neural_numeric_only | quarterly-2026Q2 | 584 | 0.6460 | 0.6420 | -0.0040 | 0.0130 | 0.0052 | isotonic |
| xgboost | quarterly-2026Q2 | 584 | 0.6410 | 0.6441 | 0.0030 | 0.0111 | 0.0030 | platt |
| lightgbm | covid_shift-2020H2-2021 | 577 | 0.5706 | 0.6353 | 0.0647 | 0.0187 | 0.0590 | platt |
| logistic_regression | covid_shift-2020H2-2021 | 577 | 0.5781 | 0.5734 | -0.0047 | 0.0207 | 0.0032 | isotonic |
| neural_numeric_only | covid_shift-2020H2-2021 | 577 | 0.5727 | 0.6264 | 0.0537 | 0.0203 | 0.0506 | platt |
| xgboost | covid_shift-2020H2-2021 | 577 | 0.5641 | 0.6222 | 0.0582 | 0.0185 | 0.0529 | platt |

Paired-gap SD across all (model, fold) cells: min 0.0022, median 0.0054, max 0.1595 over 72 cells.

### What this shows

**Isotonic is not merely worse on average here; it is intermittently catastrophic.** Platt's
inner-select log-loss sits in a tight band of 0.56–0.70 across all 72 cells. Isotonic's ranges from
0.566 to **1.1028**. The worst cases are not marginal:

| cell | Platt | isotonic |
|---|---|---|
| `neural_numeric_only` / `quarterly-2023Q1` | 0.6850 | **1.1028** |
| `lightgbm` / `quarterly-2023Q1` | 0.6823 | **1.0467** |
| `logistic_regression` / `quarterly-2022Q4` | 0.6867 | **0.8717** |
| `xgboost` / `quarterly-2023Q1` | 0.6867 | **0.8766** |

The mechanism is not mysterious. Isotonic's pool-adjacent-violators fit on a ~1,200-row window
produces plateaus that reach exactly 0 and exactly 1. Any inner-select row landing on a wrong
plateau costs `−log(ε)`, which is precisely what log-loss is for and precisely what a
probability consumed by a cost threshold must not do.

**Paired-gap SD: min 0.0022, median 0.0054, max 0.1595 over 72 cells.** The maximum is large for
the same reason — when isotonic destabilises, so does the comparison.

### The rule, declared here and frozen in ADR 0025

```
TIE_THRESHOLD  = 0.005 nats        (≈ the median paired-gap SD, 0.0054)
TIE_PREFERENCE = platt
Rule: choose isotonic only if  mean_ll(isotonic) < mean_ll(platt) − TIE_THRESHOLD.
      Otherwise Platt.
```

The plan proposed 0.002 before this profile existed. **0.002 is below the smallest observed
paired-gap SD (0.0022)**, so it would have declared a winner on differences finer than the noise of
the comparison. The threshold is therefore set to one median paired-gap SD instead: isotonic must
beat Platt by more than roughly one standard deviation of the paired comparison before the project
switches away from the simpler, strictly monotone default.

It is deliberately conservative for later folds. The production rule compares the *expanding-prefix
mean* over folds 1…k, whose SD shrinks roughly as 1/√k, so a fixed 0.005 becomes a stricter bar as
the fold index grows. Being conservative in favour of the simpler method is the declared
preference, not an accident.

Three reasons Platt is the tie preference, all independent of any result:

1. Two parameters against a step function with up to ~1,700 breakpoints (§9), on a ~1,200-row fit.
2. **Platt is strictly monotone; isotonic is only weakly monotone.** Its plateaus create ties, and
   `evaluation.metrics.top_k_indices` breaks ties by `target_inspection_id` ascending, so a
   plateau can move top-k membership without the calibrator being non-monotone. "Do not re-rank" is
   satisfied exactly by Platt and only approximately by isotonic.
3. Isotonic requires `out_of_bounds="clip"` and therefore has a hard floor and ceiling at the
   calibration window's observed extremes. Platt extrapolates smoothly.

⚠ This threshold favours Platt, and Platt does in fact win most cells above. That ordering is a
**calibration-window** finding, measured on the window this component owns, and it is frozen before
any test window is opened. It is not, and must not be reported as, evidence about test performance.

---

## 7. Prior shift bounds what a calibrator can achieve (profile `base_rate_drift`)

| fold_id | cal rows | cal rate | test rows | test rate | test - cal | abs gap |
|---|---|---|---|---|---|---|
| quarterly-2022Q2 | 1357 | 0.4311 | 1762 | 0.4659 | 0.0348 | 0.0348 |
| quarterly-2022Q3 | 1762 | 0.4659 | 1733 | 0.4916 | 0.0257 | 0.0257 |
| quarterly-2022Q4 | 1733 | 0.4916 | 1700 | 0.4653 | -0.0263 | 0.0263 |
| quarterly-2023Q1 | 1700 | 0.4653 | 1801 | 0.4453 | -0.0200 | 0.0200 |
| quarterly-2023Q2 | 1801 | 0.4453 | 1787 | 0.4846 | 0.0393 | 0.0393 |
| quarterly-2023Q3 | 1787 | 0.4846 | 1650 | 0.4836 | -0.0010 | 0.0010 |
| quarterly-2023Q4 | 1650 | 0.4836 | 1958 | 0.4331 | -0.0505 | 0.0505 |
| quarterly-2024Q1 | 1958 | 0.4331 | 1913 | 0.4584 | 0.0253 | 0.0253 |
| quarterly-2024Q2 | 1913 | 0.4584 | 2196 | 0.4495 | -0.0090 | 0.0090 |
| quarterly-2024Q3 | 2196 | 0.4495 | 2124 | 0.4218 | -0.0276 | 0.0276 |
| quarterly-2024Q4 | 2124 | 0.4218 | 2248 | 0.3794 | -0.0424 | 0.0424 |
| quarterly-2025Q1 | 2248 | 0.3794 | 2459 | 0.3806 | 0.0012 | 0.0012 |
| quarterly-2025Q2 | 2459 | 0.3806 | 2290 | 0.3900 | 0.0093 | 0.0093 |
| quarterly-2025Q3 | 2290 | 0.3900 | 1754 | 0.4230 | 0.0331 | 0.0331 |
| quarterly-2025Q4 | 1754 | 0.4230 | 1766 | 0.3800 | -0.0431 | 0.0431 |
| quarterly-2026Q1 | 1766 | 0.3800 | 1917 | 0.3912 | 0.0113 | 0.0113 |
| quarterly-2026Q2 | 1917 | 0.3912 | 1638 | 0.3791 | -0.0121 | 0.0121 |
| covid_shift-2020H2-2021 | 1846 | 0.6826 | 8840 | 0.5127 | -0.1699 | 0.1699 |

A calibrator learns the map from score to probability on the calibration window. If the test
window's base rate differs, part of the resulting miscalibration is **prior shift**, which no
monotone recalibration fitted on the earlier window can remove.

This profile bounds that contribution in advance, so §12's residual ECE can be attributed rather
than argued about afterwards. The quarterly gaps are mostly small — |test − cal| under 0.04 on 11
of 17 folds — with `quarterly-2023Q4` the worst at 0.0505.

`covid_shift` is the outlier and is reported separately everywhere: calibration base rate 0.6826,
test base rate 0.5127, a gap of **0.17**. Its calibration window is 2020-03-01…05-31 — the exact
months Chicago's inspection programme was suspended. A calibrator fitted on 1,846 rows drawn from a
programme shutdown and applied to a 19-month test window at a base rate 17 points lower is the
least trustworthy calibrator in this component, and it is expected to fail. Reporting that failure
is the point of the fold.

---

## 8. Establishments repeat, so the bootstrap needs two schemes
   (profile `establishment_recurrence`)

| fold_id | window | rows | establishments | repeated | rows/establishment | max repeats |
|---|---|---|---|---|---|---|
| quarterly-2022Q2 | calibration | 1357 | 1356 | 1 | 1.001 | 2 |
| quarterly-2022Q2 | test | 1762 | 1754 | 5 | 1.005 | 3 |
| quarterly-2022Q3 | calibration | 1762 | 1754 | 5 | 1.005 | 3 |
| quarterly-2022Q3 | test | 1733 | 1728 | 5 | 1.003 | 2 |
| quarterly-2022Q4 | calibration | 1733 | 1728 | 5 | 1.003 | 2 |
| quarterly-2022Q4 | test | 1700 | 1692 | 7 | 1.005 | 3 |
| quarterly-2023Q1 | calibration | 1700 | 1692 | 7 | 1.005 | 3 |
| quarterly-2023Q1 | test | 1801 | 1791 | 5 | 1.006 | 6 |
| quarterly-2023Q2 | calibration | 1801 | 1791 | 5 | 1.006 | 6 |
| quarterly-2023Q2 | test | 1787 | 1777 | 9 | 1.006 | 3 |
| quarterly-2023Q3 | calibration | 1787 | 1777 | 9 | 1.006 | 3 |
| quarterly-2023Q3 | test | 1650 | 1646 | 4 | 1.002 | 2 |
| quarterly-2023Q4 | calibration | 1650 | 1646 | 4 | 1.002 | 2 |
| quarterly-2023Q4 | test | 1958 | 1953 | 5 | 1.003 | 2 |
| quarterly-2024Q1 | calibration | 1958 | 1953 | 5 | 1.003 | 2 |
| quarterly-2024Q1 | test | 1913 | 1910 | 3 | 1.002 | 2 |
| quarterly-2024Q2 | calibration | 1913 | 1910 | 3 | 1.002 | 2 |
| quarterly-2024Q2 | test | 2196 | 2166 | 12 | 1.014 | 9 |
| quarterly-2024Q3 | calibration | 2196 | 2166 | 12 | 1.014 | 9 |
| quarterly-2024Q3 | test | 2124 | 2119 | 5 | 1.002 | 2 |
| quarterly-2024Q4 | calibration | 2124 | 2119 | 5 | 1.002 | 2 |
| quarterly-2024Q4 | test | 2248 | 2236 | 8 | 1.005 | 4 |
| quarterly-2025Q1 | calibration | 2248 | 2236 | 8 | 1.005 | 4 |
| quarterly-2025Q1 | test | 2459 | 2454 | 5 | 1.002 | 2 |
| quarterly-2025Q2 | calibration | 2459 | 2454 | 5 | 1.002 | 2 |
| quarterly-2025Q2 | test | 2290 | 2278 | 9 | 1.005 | 4 |
| quarterly-2025Q3 | calibration | 2290 | 2278 | 9 | 1.005 | 4 |
| quarterly-2025Q3 | test | 1754 | 1738 | 7 | 1.009 | 8 |
| quarterly-2025Q4 | calibration | 1754 | 1738 | 7 | 1.009 | 8 |
| quarterly-2025Q4 | test | 1766 | 1761 | 3 | 1.003 | 4 |
| quarterly-2026Q1 | calibration | 1766 | 1761 | 3 | 1.003 | 4 |
| quarterly-2026Q1 | test | 1917 | 1906 | 9 | 1.006 | 3 |
| quarterly-2026Q2 | calibration | 1917 | 1906 | 9 | 1.006 | 3 |
| quarterly-2026Q2 | test | 1638 | 1637 | 1 | 1.001 | 2 |
| covid_shift-2020H2-2021 | calibration | 1846 | 1845 | 1 | 1.001 | 2 |
| covid_shift-2020H2-2021 | test | 8840 | 7291 | 1471 | 1.212 | 12 |

Rows are not independent within a window: the same establishment can appear more than once, and its
rows share an as-of history. An i.i.d. row bootstrap therefore understates the standard error.

Both schemes are run for every reported interval — `row` (i.i.d.) and `establishment_block` (draw
establishments with replacement, take all their rows) — and both are written to
`calibration_bootstrap_*.parquet`. Running both is cheap and settles the objection with a
measurement instead of a caveat.

The across-fold summary is `mean ± SD` over the 17 quarterly folds, and that **SD is a dispersion,
not a confidence interval**. The folds share an expanding training window — fold 17's training rows
are a superset of fold 1's — so the per-fold estimates are strongly positively dependent and a
t-interval built from this SD would be anticonservative. The within-fold percentile interval is the
interval; the across-fold SD describes how much the number moves over time. `covid_shift` is never
pooled with the quarterly folds.

---

## 9. Isotonic's breakpoint budget, and the ties it will create
   (profile `isotonic_tie_budget`)

| model | fold_id | cal rows | distinct scores | share distinct |
|---|---|---|---|---|
| lightgbm | covid_shift-2020H2-2021 | 1846 | 1535 | 0.8315 |
| lightgbm | quarterly-2022Q2 | 1357 | 1330 | 0.9801 |
| lightgbm | quarterly-2022Q3 | 1762 | 1682 | 0.9546 |
| lightgbm | quarterly-2022Q4 | 1733 | 1669 | 0.9631 |
| lightgbm | quarterly-2023Q1 | 1700 | 1640 | 0.9647 |
| lightgbm | quarterly-2023Q2 | 1801 | 1724 | 0.9572 |
| lightgbm | quarterly-2023Q3 | 1787 | 1706 | 0.9547 |
| lightgbm | quarterly-2023Q4 | 1650 | 1603 | 0.9715 |
| lightgbm | quarterly-2024Q1 | 1958 | 1905 | 0.9729 |
| lightgbm | quarterly-2024Q2 | 1913 | 1858 | 0.9712 |
| lightgbm | quarterly-2024Q3 | 2196 | 2090 | 0.9517 |
| lightgbm | quarterly-2024Q4 | 2124 | 2038 | 0.9595 |
| lightgbm | quarterly-2025Q1 | 2248 | 2093 | 0.9310 |
| lightgbm | quarterly-2025Q2 | 2459 | 2323 | 0.9447 |
| lightgbm | quarterly-2025Q3 | 2290 | 2156 | 0.9415 |
| lightgbm | quarterly-2025Q4 | 1754 | 1680 | 0.9578 |
| lightgbm | quarterly-2026Q1 | 1766 | 1695 | 0.9598 |
| lightgbm | quarterly-2026Q2 | 1917 | 1848 | 0.9640 |
| logistic_regression | covid_shift-2020H2-2021 | 1846 | 1814 | 0.9827 |
| logistic_regression | quarterly-2022Q2 | 1357 | 1346 | 0.9919 |
| logistic_regression | quarterly-2022Q3 | 1762 | 1729 | 0.9813 |
| logistic_regression | quarterly-2022Q4 | 1733 | 1719 | 0.9919 |
| logistic_regression | quarterly-2023Q1 | 1700 | 1682 | 0.9894 |
| logistic_regression | quarterly-2023Q2 | 1801 | 1771 | 0.9833 |
| logistic_regression | quarterly-2023Q3 | 1787 | 1759 | 0.9843 |
| logistic_regression | quarterly-2023Q4 | 1650 | 1638 | 0.9927 |
| logistic_regression | quarterly-2024Q1 | 1958 | 1943 | 0.9923 |
| logistic_regression | quarterly-2024Q2 | 1913 | 1898 | 0.9922 |
| logistic_regression | quarterly-2024Q3 | 2196 | 2171 | 0.9886 |
| logistic_regression | quarterly-2024Q4 | 2124 | 2112 | 0.9944 |
| logistic_regression | quarterly-2025Q1 | 2248 | 2231 | 0.9924 |
| logistic_regression | quarterly-2025Q2 | 2459 | 2443 | 0.9935 |
| logistic_regression | quarterly-2025Q3 | 2290 | 2266 | 0.9895 |
| logistic_regression | quarterly-2025Q4 | 1754 | 1743 | 0.9937 |
| logistic_regression | quarterly-2026Q1 | 1766 | 1761 | 0.9972 |
| logistic_regression | quarterly-2026Q2 | 1917 | 1911 | 0.9969 |
| neural_numeric_only | covid_shift-2020H2-2021 | 1846 | 1814 | 0.9827 |
| neural_numeric_only | quarterly-2022Q2 | 1357 | 1346 | 0.9919 |
| neural_numeric_only | quarterly-2022Q3 | 1762 | 1729 | 0.9813 |
| neural_numeric_only | quarterly-2022Q4 | 1733 | 1719 | 0.9919 |
| neural_numeric_only | quarterly-2023Q1 | 1700 | 1682 | 0.9894 |
| neural_numeric_only | quarterly-2023Q2 | 1801 | 1770 | 0.9828 |
| neural_numeric_only | quarterly-2023Q3 | 1787 | 1758 | 0.9838 |
| neural_numeric_only | quarterly-2023Q4 | 1650 | 1638 | 0.9927 |
| neural_numeric_only | quarterly-2024Q1 | 1958 | 1943 | 0.9923 |
| neural_numeric_only | quarterly-2024Q2 | 1913 | 1898 | 0.9922 |
| neural_numeric_only | quarterly-2024Q3 | 2196 | 2170 | 0.9882 |
| neural_numeric_only | quarterly-2024Q4 | 2124 | 2111 | 0.9939 |
| neural_numeric_only | quarterly-2025Q1 | 2248 | 2231 | 0.9924 |
| neural_numeric_only | quarterly-2025Q2 | 2459 | 2443 | 0.9935 |
| neural_numeric_only | quarterly-2025Q3 | 2290 | 2266 | 0.9895 |
| neural_numeric_only | quarterly-2025Q4 | 1754 | 1743 | 0.9937 |
| neural_numeric_only | quarterly-2026Q1 | 1766 | 1760 | 0.9966 |
| neural_numeric_only | quarterly-2026Q2 | 1917 | 1911 | 0.9969 |
| xgboost | covid_shift-2020H2-2021 | 1846 | 1666 | 0.9025 |
| xgboost | quarterly-2022Q2 | 1357 | 1322 | 0.9742 |
| xgboost | quarterly-2022Q3 | 1762 | 1695 | 0.9620 |
| xgboost | quarterly-2022Q4 | 1733 | 1682 | 0.9706 |
| xgboost | quarterly-2023Q1 | 1700 | 1631 | 0.9594 |
| xgboost | quarterly-2023Q2 | 1801 | 1731 | 0.9611 |
| xgboost | quarterly-2023Q3 | 1787 | 1717 | 0.9608 |
| xgboost | quarterly-2023Q4 | 1650 | 1602 | 0.9709 |
| xgboost | quarterly-2024Q1 | 1958 | 1897 | 0.9688 |
| xgboost | quarterly-2024Q2 | 1913 | 1869 | 0.9770 |
| xgboost | quarterly-2024Q3 | 2196 | 2116 | 0.9636 |
| xgboost | quarterly-2024Q4 | 2124 | 2051 | 0.9656 |
| xgboost | quarterly-2025Q1 | 2248 | 2120 | 0.9431 |
| xgboost | quarterly-2025Q2 | 2459 | 2350 | 0.9557 |
| xgboost | quarterly-2025Q3 | 2290 | 2196 | 0.9590 |
| xgboost | quarterly-2025Q4 | 1754 | 1703 | 0.9709 |
| xgboost | quarterly-2026Q1 | 1766 | 1717 | 0.9723 |
| xgboost | quarterly-2026Q2 | 1917 | 1864 | 0.9724 |

Isotonic can place at most one breakpoint per distinct input value, so these counts are the ceiling
on the complexity of the fitted map: 1,330–2,300 breakpoints against Platt's two.

The `share distinct` column is also the baseline for §14's ranking-preservation measurement. The
base scores are **not** all distinct before calibration — lightgbm on `covid_shift` has 1,535
distinct values over 1,846 rows (0.83) — so the ties isotonic creates must be counted as an
*increment* over this, not as a count of ties in the output.

---

## 10. What this profiling fixed, before any implementation

| constant | value | fixed by |
|---|---|---|
| `INNER_SELECT_FRACTION` | 0.30 | §3 — 0.15 would leave ~204 select rows |
| `MIN_INNER_FIT_ROWS` | 400 | §3 — smallest observed 948, so the guard is real but not binding |
| `MIN_INNER_SELECT_ROWS` | 250 | §3 — smallest observed 409 |
| `SELECTION_METRIC` | inner-select log-loss | §6 — ECE is 27–50 rows/bin here, and is not proper |
| `TIE_THRESHOLD` | 0.005 nats | §6 — median paired-gap SD is 0.0054; the planned 0.002 was below the noise floor |
| `TIE_PREFERENCE` | Platt | §6 — strictly monotone, two parameters, no clipping |
| calibrator input | recovered logit | §5 — reproducible from the committed artifact alone |
| margin-check tolerance | 1e-4 | §5 — the planned 1e-9 was wrong; float32 base models give 2.6e-5 |
| `DEFAULT_CALIBRATION_BINS` | 15, equal-mass | unchanged, imported from Component 5 |

Two of these corrected the plan. Both corrections are recorded above with the measurement that
forced them, and neither was made after seeing a test-window number.

---

## 11. Reproducing

```bash
uv run python scripts/profile_calibration.py            # §1–§9, ~4 minutes
uv run python scripts/profile_calibration.py --cheap    # §2, §3, §7, §8 only, no model fitted
```

The script refits Components 6–8's models to obtain the calibration-window scores (ADR 0026); it
does not read them from disk, because they are not there.

---

## 12. Evaluation

_To be completed after the production run. This section will report the test-window results:
before/after Brier, ECE, MCE, log-loss and reliability; the Brier decomposition; ranking
preservation; per-quarter calibration drift; and `covid_shift` separately._

## 13. What Component 10 should know

_To be completed after the production run._
