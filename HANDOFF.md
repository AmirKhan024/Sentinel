# HANDOFF

Read in this order: [MEMORY.md](MEMORY.md) → [STATUS.md](STATUS.md) → this file.

---

## 0. The one warning that matters most

Components 1–9 are closed. **Do not silently change an earlier component.** If
you find a defect upstream, document it and stop before modifying it.

Three invariants now sit on top of each other and are easy to confuse:

```text
Component 4:  can my FEATURES see the future?     -> a feature uses only records < d
Component 5:  can my EVALUATION see the future?   -> a model is scored only on a
                                                     window later than everything
                                                     it was allowed to learn from
Component 6:  can my MODEL see the future?        -> the fit AND every preprocessing
                                                     statistic come only from the
                                                     fold's training window
```

The first is necessary and **not sufficient**. A feature table with a perfect
temporal boundary still produces a dishonest result if it is split randomly, if
calibration touches the test period, or if a feature is chosen by looking at a
test score.

The third has a failure mode the other two cannot catch: an imputation median or a
scaler mean computed over the whole table *before* splitting. The fold boundary is
respected by the fit while the transform already knows the future, and nothing about
the split looks wrong. `modeling.validate._preprocessing_comes_from_train` re-derives
every median from the training rows for exactly this reason.

And the failure modes differ in how visible they are. A leaked feature makes one column
wrong. A leaked evaluation, or a leaked preprocessing statistic, makes every number
right and every conclusion wrong.

---

## 1. Working agreement

**investigate → document → design → implement → test.** The findings document is
written *before* the implementation, from the output of a read-only profiling
script. Anything unverified is labelled `NOT VERIFIED` with the command that
would verify it. One component at a time.

---

## 2. What was completed

Sections 2–12 describe **Component 5**, the evaluation harness, and remain accurate:
its contracts are what Component 6 consumes and what Component 7 will consume.
**Component 6 is section 13 onward.**

**Component 5 — temporal evaluation.** A model-agnostic evaluation harness built
*before* any model exists, so the first model is measured by a yardstick it did
not get to shape.

| Artifact | Path |
|---|---|
| profiling | `scripts/profile_evaluation.py` — 17 read-only profiles |
| findings | `docs/analysis/temporal_evaluation_findings.md` |
| contract | `docs/data_contracts/temporal_evaluation.md` |
| decisions | ADR 0012 (rolling origin over random CV), ADR 0013 (results are artifacts) |
| code | `src/sentinel/evaluation/` — 11 modules |
| CLI | `sentinel evaluate` |
| tests | 278 new, in 9 files plus CLI additions |
| output | `data/processed/evaluation/` — six tables, one manifest |

---

## 3. What Component 5 does

It answers one question:

> If the same historically observed inspection opportunities had been ordered as
> Sentinel suggests, how much sooner would the known positive outcomes have
> appeared?

Not "how accurate is the classifier". Sentinel is a **ranking and scheduling**
problem: every establishment is inspected on a statutory cycle, capacity is
fixed, and only the *order* is decidable. So the harness measures statistical
prediction quality and operational ranking quality side by side.

---

## 4. Why a random split is forbidden

The feature table spans 2018-07-03 to 2026-08-14. `train_test_split` over those
rows will happily produce:

```text
train:  2019  2021  2023  2024
test:         2020        2022
```

The model then learns how establishments behaved *after* the period it is judged
on. Sentinel can never operate that way — at a real decision point in 2020, 2021
has not happened.

Two measured properties make it worse here. The **base rate drifts** from 0.876
in 2018 H2 to 0.391 in 2026, so a model that had seen the later low-prevalence
years carries that into an earlier test window. And **establishments recur** on a
358-day median cycle, so a random split also puts the same premises on both sides
of the line at different dates.

`train_test_split`, `KFold` and `StratifiedKFold` are forbidden for the primary
evaluation. `FoldSpec` refuses to construct a fold whose windows are not strictly
ordered, so a leaky split is unrepresentable rather than merely discouraged.

---

## 5. Fold structure — the actual folds

Rolling origin, **expanding** training window, quarterly, anchored at
`2018-07-01`. Calibration sits strictly between train and test.

```text
TRAIN ─────────────────────► | CAL | TEST
     anchor 2018-07-01         3mo   3mo
```

17 quarterly folds. The count is **derived from the data** — a test quarter is
emitted only if the snapshot covers it entirely, so 2026Q3 (328 rows over 32
days) is excluded and named in the manifest.

| fold_id | train end | cal | test | train rows | test rows | train rate | test rate | median capacity |
|---|---|---|---|---|---|---|---|---|
| quarterly-2022Q2 | 2021-12-31 | 2022Q1 | 2022Q2 | 23,346 | 1,762 | 0.667 | 0.466 | 29 |
| quarterly-2022Q3 | 2022-03-31 | 2022Q2 | 2022Q3 | 24,703 | 1,733 | 0.654 | 0.492 | 28 |
| quarterly-2022Q4 | 2022-06-30 | 2022Q3 | 2022Q4 | 26,465 | 1,700 | 0.642 | 0.465 | 30 |
| quarterly-2023Q1 | 2022-09-30 | 2022Q4 | 2023Q1 | 28,198 | 1,801 | 0.633 | 0.445 | 31 |
| quarterly-2023Q2 | 2022-12-31 | 2023Q1 | 2023Q2 | 29,898 | 1,787 | 0.623 | 0.485 | 29 |
| quarterly-2023Q3 | 2023-03-31 | 2023Q2 | 2023Q3 | 31,699 | 1,650 | 0.613 | 0.484 | 26 |
| quarterly-2023Q4 | 2023-06-30 | 2023Q3 | 2023Q4 | 33,486 | 1,958 | 0.606 | 0.433 | 33.5 |
| quarterly-2024Q1 | 2023-09-30 | 2023Q4 | 2024Q1 | 35,136 | 1,913 | 0.600 | 0.458 | 34 |
| quarterly-2024Q2 | 2023-12-31 | 2024Q1 | 2024Q2 | 37,094 | 2,196 | 0.592 | 0.449 | 38 |
| quarterly-2024Q3 | 2024-03-31 | 2024Q2 | 2024Q3 | 39,007 | 2,124 | 0.585 | 0.422 | 33.5 |
| quarterly-2024Q4 | 2024-06-30 | 2024Q3 | 2024Q4 | 41,203 | 2,248 | 0.578 | 0.379 | 40 |
| quarterly-2025Q1 | 2024-09-30 | 2024Q4 | 2025Q1 | 43,327 | 2,459 | 0.570 | 0.381 | 45 |
| quarterly-2025Q2 | 2024-12-31 | 2025Q1 | 2025Q2 | 45,575 | 2,290 | 0.561 | 0.390 | 39 |
| quarterly-2025Q3 | 2025-03-31 | 2025Q2 | 2025Q3 | 48,034 | 1,754 | 0.552 | 0.423 | 28 |
| quarterly-2025Q4 | 2025-06-30 | 2025Q3 | 2025Q4 | 50,324 | 1,766 | 0.544 | 0.380 | 30 |
| quarterly-2026Q1 | 2025-09-30 | 2025Q4 | 2026Q1 | 52,078 | 1,917 | 0.540 | 0.391 | 35 |
| quarterly-2026Q2 | 2025-12-31 | 2026Q1 | 2026Q2 | 53,844 | 1,638 | 0.535 | 0.379 | 28 |

Plus one non-rolling `covid_shift` fold: train 2018-07-01…2020-02-29 (12,660
rows, 0.773), calibrate 2020-03…05 (1,846, 0.683), test 2020-06-01…2021-12-31
(8,840, 0.513). Kept in a separate `fold_set` so it can never be averaged into
the headline by accident.

**Why calibration is a separate, interposed window.** `TRAIN → CAL → TEST`, never
`TRAIN + TEST → calibration` and never `TRAIN → TEST → calibration`. Component 9
will fit Platt or isotonic scaling there; a calibrator fitted on test makes the
reported probabilities self-fulfilling, and one fitted on train inherits the
model's own overfitting. The window exists now, empty, so Component 9 has
nowhere else to put it.

---

## 6. Metric definitions

Implemented in `evaluation/metrics.py`. **No runtime dependency was added** —
each is cross-checked against scikit-learn in the test suite, and scikit-learn is
in the dev group only.

| Metric | Definition | Base-rate invariant? |
|---|---|---|
| `roc_auc` | Mann-Whitney U with average ranks; ties count one half | **yes** |
| `pr_auc` | average precision, `Σ (R_k − R_{k−1})·P_k` — **not** a PR trapezoid | no |
| `brier` | mean squared error of a probability | no |
| `log_loss` | negative log likelihood, clamped at 1e-15 | no |
| `ece` / `mce` | 15 **equal-mass** bins; mass-weighted mean gap / worst bin | no |
| `precision_at_k` | share of the top *k* that were genuine citations | no |
| `recall_at_k` | share of the window's positives the top *k* captured | no |
| `lift_at_k` | `precision_at_k / base_rate`; 1.0 = no better than blind | **yes** |

`None` is returned rather than a substitute when a metric is undefined.

**`RANKING_METRICS` and `PROBABILITY_METRICS` are disjoint and the separation is
load-bearing.** A producer with `is_probability=False` is scored only on ranking
metrics. A random shuffle has no meaningful Brier score, and emitting one so it
fits an API would be a fabrication.

**Headline the drift-robust pair.** ROC-AUC and NDE are rank-based. PR-AUC,
precision@k and first-half discovery all move with the base rate, which ranges
0.379 to 0.513 across test windows — always report them beside prevalence.

---

## 7. Simulation semantics

**The slot model.** A test window's capacity profile is the exact multiset of
`(date → count)` the city actually worked. Sort those dates and you have a slot
calendar. A schedule is a permutation; position *i* takes slot *i*, which assigns
every inspection a simulated date. Capacity is held constant **by construction**:
`n_slots == n_inspections`, and the multiset of simulated dates equals the
observed one.

**The business-as-usual identity.** Ordering by actual date and filling slots in
date order returns every inspection to its own real date, so
`bau_simulated_date == actual inspection_date`. "Days earlier than business as
usual" therefore means "days earlier than what really happened". Asserted on
every fold of every run by `business_as_usual_is_real`.

**Labels never move.** An establishment cited on 14 June is still cited when the
schedule moves it to 2 May. This is a reordering simulation, not a causal
simulator.

**Discovery curve.** x = fraction of slots consumed, y = fraction of positives
found, both in [0,1]. Area by trapezoid over `n+1` points.

**Normalized discovery efficiency.**

```text
NDE = (A_model − 0.5) / ((1 − P/(2N)) − 0.5)
```

Both denominator terms are **analytic**: a uniformly random permutation's
expected curve is the diagonal, so `A_random = 0.5` exactly; optimal is
`1 − P/(2N)`. Scale: 1 perfect, 0 random, −1 worst. `None` when `P == 0` or
`P == N`.

**Days earlier.** `bau_date − model_date`, positives only (all-rows as
secondary). Reported as mean, median, SD, p25, p75, min, max and the fractions
improved / unchanged / worse. **The mean alone is forbidden.**

**Tie-breaking.** `target_inspection_id` ascending, as a string. Intra-day order
is unrecoverable from the source, so within a date the order is
arbitrary-but-deterministic — which cannot bias a date-based metric because those
rows share a date.

**Score direction.** Higher = inspected sooner. Probed on every run.

---

## 8. Baselines

All deterministic; **nothing in Component 5 is fitted.**

| name | rule | null rule |
|---|---|---|
| `business_as_usual` | negated date ordinal, so same-day rows tie | n/a |
| `random` | uniform noise, 20 seeds (42…61) | n/a |
| `days_since_last_canvass` | longest gap first | never canvassed sorts first |
| `priority_at_last_canvass` | cited → 1.0, clean → 0.0 | unknown → 0.5 |
| `prior_canvass_priority_rate` | historical citation rate | unknown → 0.5 |
| `constant` | all equal — a tie-breaking diagnostic | n/a |

Each null rule was chosen on a **semantic** argument and its consequence measured
*afterwards*; both are recorded in `rankers.py`. Choosing a null tier by its
outcome would be fitting to the label under another name.

**`days_since_last_canvass` is not the spec's "days overdue"** — a statutory
overdue figure needs the CDPH risk category, which is in raw but not in the
feature table, and Component 5 may not add features.

---

## 9. Results, measured

17 quarterly folds, mean ± SD.

| schedule / model | NDE | days earlier | SD | worse | ROC-AUC |
|---|---|---|---|---|---|
| optimal | 1.0000 ± 0.0000 | +24.75 | 14.95 | 0.0% | — |
| `prior_canvass_priority_rate` | **0.1845** ± 0.0404 | **+4.47** | 32.60 | **42.9%** | 0.5915 |
| `priority_at_last_canvass` | 0.1522 ± 0.0384 | +3.68 | 27.36 | 46.1% | 0.5747 |
| `days_since_last_canvass` | 0.0765 ± 0.0384 | +1.76 | 36.87 | 46.2% | 0.5381 |
| business_as_usual | 0.0066 ± 0.0422 | 0.00 | 0.00 | 0.0% | 0.5040 |
| random | −0.0016 ± 0.0271 | −0.19 | 36.05 | 49.3% | 0.5051 |
| worst | −1.0000 ± 0.0000 | −25.39 | 14.35 | 98.4% | — |

Three things Component 6 must carry forward:

1. **The SD is 7.3× the mean and 42.9% of positives are found later.** A
   reordering under fixed capacity is zero-sum in slots. The mean is the net
   effect; the fractions say how it was achieved.
2. **Business-as-usual is indistinguishable from random within a quarter.** The
   bar is low, so beating it is not itself evidence of a good model. The number
   to beat is **0.1845**.
3. **Time invariance does not hold** — de-trended seasonal swing of 11.77 pp,
   peaking in August, troughing in December. The sensitivity band nonetheless
   leaves the headline at [0.172, 0.192] over 1,000 label re-draws.

Full detail: `docs/analysis/temporal_evaluation_findings.md`.

---

## 10. Outputs

```text
data/processed/evaluation/
  evaluation_folds_<UTC>.parquet          18 × 18    the split; the primary artifact
  evaluation_metrics_<UTC>.parquet     2,808 × 14    tidy long
  discovery_curves_<UTC>.parquet     373,986 ×  9    full resolution
  simulation_summary_<UTC>.parquet       504 × 24    per schedule
  seasonality_<UTC>.parquet              228 ×  6    month effects
  sensitivity_<UTC>.parquet               54 × 12    uncertainty bands
  manifest_evaluation_folds_<UTC>.json
```

One manifest per run, keyed to `evaluation_folds` — the same convention Component
2 uses for its three tables. It pins the feature table by SHA-256 and records the
estimand, the score direction, the tie-break column, the capacity semantics,
every seed, the excluded partial window, the blocked experiments, and the full
check list. **"Exactly what was the model allowed to know when this score was
produced?" is answerable from the manifest alone.**

---

## 11. Tests

```bash
uv run pytest                                # 1,254 passed, 3 deselected, 227 s
uv run pytest tests/test_evaluation_leakage.py -v   # 19, the evaluation safety wall
uv run pytest tests/test_modeling_leakage.py -v     # 21, the modelling safety wall
uv run pytest -m live                        # 3 live tests, hits the real API
uv run ruff check .                          # All checks passed
uv run ruff format --check .                 # 137 files already formatted
uv run mypy src/sentinel scripts             # no issues in 61 source files
uv run sentinel evaluate --folds-only --report      # < 1 s
uv run sentinel evaluate --dry-run --report         # full run, writes nothing
```

278 new tests across 9 files. The three worth knowing:

- `test_evaluation_leakage.py` (19) — appending two years of future data must not
  change an earlier fold's rows; a declared training horizon past the fold is
  rejected; a prediction artifact carrying a label column is rejected.
- `test_evaluation_metrics.py` (64) — **every metric cross-checked against
  scikit-learn**, including on heavily tied scores, plus an assertion that
  PR-AUC is average precision and *not* the optimistically-biased PR trapezoid.
- `test_evaluation_simulate.py` (41) — the business-as-usual identity, capacity
  conservation, and the analytic bounds landing at exactly ±1.

---

## 12. Limitations to carry forward

1. **Re-ordering only.** No labels exist for establishments nobody inspected, so
   coverage cannot be evaluated. State this before any result.
2. **Not causal**, and **no claim about illness** — Sentinel observes violations
   *cited*, not illnesses prevented.
3. **Time invariance does not hold** (11.77 pp swing). The band bounds it; it
   does not remove the assumption.
4. **Temperature attribution BLOCKED** — no weather ingested, so the seasonal
   effect confounds temperature with daylight, holidays and staffing.
5. **17 folds is not many**, and folds are not independent samples — the same
   premises appears in many test windows, so the SD is a fold-to-fold spread, not
   a confidence interval.
6. **The `covid_shift` fold is one fold**, with no variance estimate.
7. **Calibration windows are unused** until Component 9.
8. **No fitted baseline yet** — the CDPH replication is Component 6's.

---
## 13. What Component 6 did

**Component 6 — baseline risk models.** The first fitted models. Component 5 built the
yardstick first, deliberately, so the model was measured by something it did not shape.

| Artifact | Path |
|---|---|
| profiling | `scripts/profile_baselines.py` — 10 read-only profiles, **train windows only** |
| findings | `docs/analysis/baseline_models_findings.md` |
| contract | `docs/data_contracts/baseline_predictions.md` |
| decisions | ADR 0014 (predictions are a third artifact kind; `trained_through`), ADR 0015 (scikit-learn as a runtime dependency) |
| code | `src/sentinel/modeling/` — 8 modules |
| CLI | `sentinel train-baselines`, and `sentinel evaluate --predictions PATH` |
| tests | 231 across 6 new files, plus `test_evaluation_predictions.py` for the seam |
| output | `data/processed/predictions/` — 3 tables + 1 manifest |

### What it implemented

**Three models**, all L2 logistic regression with identical fixed hyperparameters
(`penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42`,
`class_weight=None`), one fit per fold, 54 fits total:

| model | features | note |
|---|---|---|
| `logistic_regression` | 26 + 4 indicators | the primary baseline |
| `logistic_regression_no_scheduling` | 25 + 4 | the `days_since_any_inspection` ablation, as a separate fit |
| `cdph_2015_approximation` | 19 + 4 | **an approximation**, labelled everywhere |

**Preprocessing**, fitted on the training window only: 16 never-null features pass
through, 7 nullable numerics get a train-fitted median, 3 nullable booleans get a
constant 0.0, and **4 family indicators** are computed. Then `StandardScaler`.

### What it did NOT implement

No XGBoost/LightGBM (C7), no neural model (C8), **no calibration** (C9), no inspector
effects (C10), no SHAP (C11), no fairness (C12), no policy/scheduling/routing (C13+).
No hyperparameter tuning — deliberately, and recorded as such in the manifest's
`blocked` list. No new features. No second evaluator.

### Four decisions worth knowing before you touch it

1. **Four family indicators, not ten per-column ones.** The null masks *within* a
   Component 4 null-rule family are byte-identical (measured on all 57,727 rows), so
   `SimpleImputer(add_indicator=True)` would emit 10 columns of which only 4 are
   distinct — collinear duplicates that make the coefficients artifact unreadable. It
   also picks indicators by observation, so matrix width would vary per fold on a small
   table. Several tests fail if anyone reintroduces it.
2. **Nullable booleans fill with constant 0, not a median.**
   `priority_at_last_canvass` drifts 0.6310 → **0.5056** across the 17 training windows
   — 0.0056 from flipping the median fill, which would reverse the encoding of "unknown"
   mid-sequence and flip the indicator's coefficient sign for no substantive reason.
3. **`trained_through = fold.train_end`, not `calibration_end`.** The contract's ceiling
   is the calibration end, and the six heuristics declare it; Component 6 fits no
   calibrator and never reads that window, so declaring the later date would claim a
   horizon it did not use. This deliberately diverges from the example snippet the
   previous handoff carried. ADR 0014.
4. **The canonical training sort is load-bearing.** Refitting the same 23,346 rows in a
   different order moves coefficients by up to **7.049e-09** — `StandardScaler`'s
   incremental variance and the lbfgs gradient are both float-summation-order dependent.
   The sort, **not `random_state`** (which `lbfgs` ignores entirely), is what makes a
   re-run reproducible.

### The Component 5 change, and why it was allowed

`contract.read_predictions` and `PREDICTION_METADATA_COLUMNS` shipped with Component 5
and were **never called anywhere** — the seam was designed and left unwired. Wiring it
exposed three defects that were latent only because no fitted model existed:

* `_append_metrics` hardcoded `model_version=rankers.RANKER_VERSION`;
* `Observations.horizon_rejections` was declared and never populated, so
  `scores_respect_the_decision_point` **could not fail** — a violation surfaced as a
  coverage failure instead;
* `metrics.PROBABILITY_METRICS` was implemented, tested, and never emitted.

All three are fixed as the smallest compatible change, behind ADR 0014, with a
regression test asserting that `run_evaluation(...)` **without** `--predictions` produces
byte-identical tables. `PredictionHorizonError` is a *subclass* of
`PredictionContractError`, so every existing `except` keeps working. Nothing else in
Component 5 changed — not the cadence, the anchor, the nine rules, the simulation
semantics, the score direction, or `EVALUATION_DEFINITION_VERSION`.

---

## 14. Results, measured

Full run 2026-08-17: 57,727 rows, 18 folds, 54 fits, training 29.7 s, evaluation 237.8 s.
Mean over the **17 quarterly folds** — `covid_shift` is never averaged in.

| model | ROC-AUC | PR-AUC | NDE | days earlier | P@k_1_day | lift |
|---|---|---|---|---|---|---|
| **logistic_regression** | **0.6163** | **0.5321** | **0.2326** | **+5.70** | **0.6576** | **1.53** |
| logistic_regression_no_scheduling | 0.6119 | 0.5245 | 0.2238 | +5.48 | 0.6173 | 1.44 |
| cdph_2015_approximation | 0.6059 | 0.5118 | 0.2119 | +5.18 | 0.5618 | 1.31 |
| prior_canvass_priority_rate *(best heuristic)* | 0.5915 | 0.5012 | 0.1845 | +4.47 | 0.5551 | 1.30 |
| business_as_usual | 0.5040 | 0.4347 | 0.0066 | 0.00 | 0.4323 | 1.01 |

PR-AUC no-skill floor = mean test prevalence = **0.4307**.

**The good news.** The trained model beats the strongest heuristic on every metric and on
**17 of 17** folds (+0.0019 to +0.0542 ROC-AUC) -- though **3 of those wins are by under
0.0025**, so the defensible claim is "never worse, clearly better on most", not "reliably
better on all". The time-invariance bands do not overlap:
0.2326 [0.2160, 0.2374] against 0.1845 [0.1720, 0.1922], over 1,000 label re-draws. So the
gap is not an artifact of the seasonal drift Component 5 measured.

**The honest reading.** PR-AUC +0.10 over the floor is modest. ROC-AUC 0.6163 is a weak
classifier by any general standard. And **43.24% of violations are still found later** than
business-as-usual — *marginally worse* than the heuristic's 42.88%, because re-ordering
under fixed capacity is zero-sum. Quoting "+5.7 days earlier" without that number is
quoting half the result.

**The methodological result, and the most important thing in this file.** On `covid_shift`
the ordering **inverts**:

| model | ROC-AUC | NDE |
|---|---|---|
| **logistic_regression_no_scheduling** | **0.6286** | **0.2571** |
| logistic_regression | 0.6256 | 0.2512 |

The ablation that *loses* on the quarterly folds *wins* under distribution shift.
`days_since_any_inspection` partly encodes scheduling policy, and when the policy itself
breaks — which is what 2020 was — a model leaning on it is the more fragile one.
**Model selection on the rolling folds would have picked the wrong model.** Component 5
predicted this with heuristics; Component 6 confirmed it with fitted models.

**Uncalibrated probabilities:** Brier 0.2382, log-loss 0.6723, ECE 0.0635, MCE 0.1664.
That is the measured "before" number justifying Component 9, not a result.

---

## 15. What Component 7 did

Replaced the estimator and added a tuning protocol. Changed nothing about folds, features,
the target, the metrics or the evaluation.

### What it implemented

`src/sentinel/boosting/` — a sibling of `modeling/` that imports from it rather than
forking it. Ten modules: `definitions.py` (registry, declared search space, frozen tuned
parameters, import-time guard), `preprocess.py`, `tuning.py`, `train.py`, `predict.py`,
`models.py`, `writer.py`, `validate.py`, `build.py`, `__init__.py`. Two CLI subcommands,
`sentinel tune-boosting` and `sentinel train-boosting`. One profiler,
`scripts/profile_boosting.py`. Nine test files, 235 tests.

Three models: `xgboost` and `lightgbm` on all 26 features, tuned separately per fold set,
and `xgboost_class_weighted` — the weighting ablation, which borrows `xgboost`'s frozen
parameters so the only difference between them is the weight.

### What it did NOT implement

* **No calibration.** Component 9's. Raw `predict_proba` output, ECE and MCE reported and
  uncorrected. The calibration window is untouched by every fit, which is what lets
  Component 9 have it.
* **No SHAP.** Component 11's. Native split-gain importances ship as a **diagnostic only**,
  and the data contract says so in bold.
* **No inspector modelling.** BLOCKED — the field does not exist. See below.
* **No ensembling, no neural network, no fairness audit, no policy engine.**

### Five decisions worth knowing before you touch it

**1. A separate registry and a separate artifact slug.** HANDOFF §15 previously suggested
appending a `ModelSpec` to `MODEL_REGISTRY` and reusing `train-baselines`. That was not
done, and the reason is the brief's own requirement that Component 6's benchmark stay
visible: appending would change the default output of `train-baselines` and mix C6 and C7
rows in one file. `boosted_predictions` is its own slug under its own manifest, and
Component 6's artifact was verified to reproduce **byte-identically** (sha256
`a2bb9411…00ff5b44`) under the current library set.

**2. Two tuning studies, not one.** The `covid_shift` test window (2020-06-01..2021-12-31)
sits *inside* the quarterly tuning region (2018-07-01..2022-03-31). One shared study would
have selected hyperparameters using the shift fold's own test labels — biasing the single
number most likely to change a release decision. Each fold set has its own study over its
own region, and `tuned_params` **raises** rather than borrowing across fold sets. ADR 0017.

**3. Early stopping lives only inside the tuning objective.** The winning trial's mean
`best_iteration` is frozen, and `train.fit_fold` runs exactly that many rounds with no
`eval_set`. That is what makes `trained_through = fold.train_end` literally true rather
than nearly true. `validate.final_fits_did_no_early_stopping` re-checks it, and the leakage
suite deletes the entire calibration window and asserts bit-identity.

**4. No preprocessing is fitted at all.** NULLs reach the estimator as NaN. Component 6's
`preprocessing_comes_from_train` check has no analogue, so it was replaced by a stricter
one: recompute the NULL mask from the source frame and compare it cell-for-cell with the
matrix's NaN mask, plus a companion check that the NaN count is non-zero so the comparison
cannot pass vacuously. The four family indicators are kept anyway, so the C6 and C7
matrices are identical and the comparison is unambiguous.

**5. Frozen parameters are source literals.** `tune-boosting` prints the block and edits no
file; a human pastes it into `definitions.TUNED_PARAMS` and commits. A parameter set read
from disk at training time could change without a diff, and freezing is only meaningful if
it cannot.

### Results, measured

**Quarterly, mean over 17 folds.** `xgboost` NDE **0.2376** / ROC-AUC 0.6188 / PR-AUC
0.5343 / +5.83 days. `lightgbm` 0.2355 / 0.6177 / 0.5342 / +5.75. Component 6's
`logistic_regression` 0.2326 / 0.6163 / 0.5321 / +5.70.

**The improvement is +2.1% relative and does not clearly survive scrutiny.** Per fold,
logistic wins **7 of 17** (xgboost 5, lightgbm 5). The seasonality redraw gives xgboost
[0.2224, 0.2444] — which contains logistic's observed 0.2326. Component 6's gain over the
heuristics survived the same test; this one does not.

**The two libraries agree to 0.0021 NDE** after 100 tuned trials each. That agreement is
the component's real finding: **the ceiling is the 26-feature representation, not the
estimator.**

**`covid_shift` has no single winner.** lightgbm takes NDE (0.2585) and ROC-AUC;
`logistic_regression` takes PR-AUC (0.6328, highest of any model) and precision@k_1day
(0.9545). One fold, k=22 slots, days-earlier SD 208.

**42.89% of violations are still found later**, against C6's 43.24%.

**Boosted probabilities are not worse calibrated on the quarterly folds** (ECE 0.0621 vs
0.0635) — contradicting the expectation this document previously carried, and matching a
probe pre-registered on the calibration window before training. Under shift the expectation
holds (0.1518 vs 0.1124).

**`xgboost_class_weighted` posts the best NDE in the project (0.2390) and is not adopted.**
It costs ECE 0.0621 → 0.0836. Prevalence is 52.52%, so it distorts a balanced problem to
buy a margin smaller than the seasonality band.

**Runtime:** 400 trials / 4 studies / 0 failed in 563.8 s; 54 fits in 21.4 s; evaluation
68.4 s.

### The one thing that was blocked

**Inspector-effect modelling.** The specification asks for inspector strictness as a
nuisance effect — a mixed-effects logistic regression with inspector as a random intercept,
marginalised away at inference. **The dataset publishes 22 columns and none identifies an
inspector.** A random intercept over an unobserved grouping has no likelihood to maximise;
a marginalisation over an unestimated effect is arithmetic on a made-up number.

Proxies were considered and refused: violation-text verbosity (unattributable, and already
rejected as a target by ADR 0008), ward or community area (a route proxy fully confounded
with establishment composition), day-of-week and month (already confounded with weather,
holidays and staffing). A model labelled "inspector-adjusted" that adjusted for ward would
be the most misleading artifact this project could ship.

Recorded in ADR 0019, in every Component 7 manifest, and in
`tests/test_boosting_inspector_blocked.py`, which re-derives the absence from the raw
contract and **fails if such a column ever appears**.

The consequence is project-wide, not local: the gap between "a violation was cited" and
"the establishment was unsafe" cannot be characterised anywhere in Sentinel, and Component
12's fairness audit inherits the same limitation.

---

## 16. What Component 8 did

**A PyTorch network with entity embeddings, plus the controls that make the answer
interpretable.** Nine models over the same 18 folds, one artifact, no metrics of its own.

### What it implemented

* `src/sentinel/neural/` — 14 modules. `definitions.py` (registry, architecture constants,
  frozen learning rates, two separate guards), `categoricals.py`, `encode.py`,
  `preprocess.py`, `net.py`, `train.py`, `predict.py`, `tuning.py`, `embed.py`,
  `figures.py`, `writer.py`, `validate.py`, `models.py`, `build.py`.
* Three commands: `build-neural-categoricals`, `tune-neural`, `train-neural`.
* A fifth processed layer, `data/processed/neural/`, holding one experimental table.
* 287 tests. 21 error-severity checks, 3 advisories.
* ADR 0020 (torch + matplotlib + CPU determinism), 0021 (what may be embedded),
  0022 (the experimental categorical layer), 0023 (community area).

### What it did NOT implement

No calibration (Component 9), no SHAP (Component 11), no fairness audit (Component 12), no
ensembling, no OR-Tools, no LangGraph. No `establishment_id` embedding — refused, not
omitted. No demographic variable of any kind.

### Five decisions worth knowing before you touch it

1. **The specified features did not exist, and the conflict was surfaced rather than worked
   around.** Chain, facility type, community area and ZIP are not in Component 4's table.
   Component 8 built its own explicitly experimental as-of layer and left
   `feature_definition_version` at `v1`. ADR 0022. **`neural_numeric_only` is fitted without
   any of it and is the model every C6/C7/C8 comparison rests on.**
2. **Early stopping validates on the last ~15% of the training window**, cut on a whole day.
   Never the calibration window, never the test window. That is what keeps
   `trained_through = train_end` literally true for the first component that early-stops.
   The cost: a final fit uses ~85% of its fold's training rows.
3. **`establishment_id` is refused.** A closed `EntityFamily` allowlist plus
   `FORBIDDEN_COLUMNS`, enforced at import and restated at runtime. ADR 0021.
4. **CPU, one thread, deterministic algorithms.** A CUDA device is present and unused. Seed
   variation is *measured*, not asserted away. ADR 0020.
5. **The learning-rate sweep reuses Component 7's protocol unchanged** — `tuning_region`,
   `first_test_start`, `build_inner_folds` are imported, not reimplemented. ADR 0017.

### Results, measured (2026-08-18)

40 sweep trials in 550.9 s; **234 fits** over 18 folds in 1,998.7 s across 4,306 epochs.
All 21 Component 8 checks and all 14 Component 5 checks pass.

Quarterly means, 17 folds:

| model | NDE | ROC-AUC | PR-AUC | Brier | ECE | P@k_1day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **neural_numeric_only** | **0.2482** | **0.6241** | 0.5343 | **0.2355** | **0.0563** | 0.6273 |
| xgboost_chain_embeddings | 0.2444 | 0.6222 | **0.5357** | 0.2374 | 0.0619 | 0.6480 |
| xgboost (C7) | 0.2376 | 0.6188 | 0.5343 | 0.2379 | 0.0621 | 0.6308 |
| lightgbm (C7) | 0.2355 | 0.6177 | 0.5342 | 0.2383 | 0.0644 | **0.6598** |
| logistic (C6) | 0.2326 | 0.6163 | 0.5321 | 0.2382 | 0.0635 | 0.6576 |
| neural_embeddings | 0.2215 | 0.6107 | 0.5233 | 0.2401 | 0.0679 | 0.6217 |

**Three findings, and they do not agree with each other.**

* The network **on the same 26 features** posts the best NDE and the best calibration in the
  project, and wins **12 of 17 folds** — the first time a mean improvement and a per-fold
  improvement agree here.
* **The entity embeddings made it worse by 0.0267 NDE.** Every ablation beats the full model.
  The one-hot control lands within 0.0009 of it, so the representation is not the problem —
  capacity is. Mean best epoch orders by parameter count: 10.4 (41,729 params) → 4.0 (67,985)
  → 2.3 (337,665).
* **The same embeddings helped XGBoost** (+0.0068 NDE, best PR-AUC of any model).

**And the win is the size of its own noise.** Five-seed ROC-AUC spread **0.0058**; the neural
advantage over XGBoost is **0.0053**. Suggestive, not decisive.

The chain embedding t-SNE is a **featureless blob**; pairwise cosine (mean 0.0018, SD 0.2508)
is indistinguishable from a random Gaussian table (0.0000, 0.2504).

### The three things that went wrong

* **A fixture bug produced seven false leakage failures.** `neural_categoricals_for` assigned
  chains by row *position*, so appending or shuffling a row changed which establishment was in
  which chain. Component 7's advice — *when a leakage test fails, suspect the test first* —
  paid for itself immediately.
* **One of those failures was Component 7's own bug, reproduced.** The corruption test mutated
  rows after `train_end`, which includes the fold's own *test* rows, whose scores are supposed
  to change. It now mutates strictly after `test_end`.
* **A real bug, caught by a test: `id()` reuse.** The live torch module for each
  `FittedNetwork` was kept in a dict keyed by `id()`. CPython reuses an address as soon as the
  object at it is collected, and the multi-seed fits go out of scope — so a later fold's record
  could land on a collected record's slot and be handed **another fold's network**. The
  training run in flight was killed and restarted after the fix.

## 16b. What Component 9 did

**Fitted a probability calibrator per (model, fold) on the window every earlier component
deliberately left untouched, and proved it changed the numbers without changing the ranking.**

### What it implemented

* `src/sentinel/calibration/` — 11 modules. `definitions.py` (the candidate registry and every
  pre-declared constant, with an import-time guard), `models.py`, `preprocess.py`,
  `basescores.py` (the regeneration seam), `train.py`, `predict.py`, `metrics.py`,
  `validate.py`, `writer.py`, `figures.py`, `build.py`.
* One command: `sentinel calibrate`.
* A **sixth** processed layer, `data/processed/calibration/`, plus the two homes ADR 0014 and
  ADR 0018 named in advance.
* 135 new tests (1,776 -> 1,911 in the suite), 21 runtime checks, 11 figures.
* ADR 0024 (where the artifacts live), 0025 (the selection protocol, pre-registered),
  0026 (why the base models were re-executed), 0027 (what the calibrator is fed).

### The blocker it hit, and why it did not just work around it

**The scores it calibrates did not exist.** Every prediction artifact covered exactly the test
window — 41,536 rows per model, which is `sum(test_rows)` to the row. The 34,261 calibration-window
rows were never scored. And **no fitted model object is persisted anywhere in this repository**:
only coefficients, split importances and embedding tables.

This was reported before anything was built, and the resolution approved, because §0's rule is
that an upstream gap is documented rather than silently worked around.

The resolution (ADR 0026): Component 9 **re-executes** Components 6–8's unchanged fit functions —
same spec, seed 42, hyperparameters, canonical row order — and proves they are the same models by
scoring the test window too and comparing it to the committed artifact with `==`, not
`math.isclose`.

> **207,680 rows across 5 models × 18 folds. Zero mismatches.**

`build.py` raises before fitting a single calibrator if that gate fails, because a calibrator
fitted on scores no committed artifact contains is a correction to nothing. Components 6, 7 and
8's artifacts are **unchanged and still byte-identical**, verified by sha256 and recorded in the
data contract.

### The protocol, and the trap it avoids

Both Platt and isotonic are fitted for every fold. The choice is made on an inner chronological
(whole-day) split of the calibration window, 70/30, by mean **log-loss** — not ECE, which at 15
equal-mass bins over ~500 inner-select rows is 27–50 rows per bin, is not a proper scoring rule,
and has a tunable bin count.

The choice is made on an **expanding prefix** of folds 1…k within a fold set, never a pool over
all folds. **Fold N's calibration window is fold N−1's test window**, so a pooled selection would
choose fold 1's method using fold 1's test period. That is the single most likely way this
component could have shipped a quiet leak, and
`test_a_pooled_global_selection_would_read_an_earlier_folds_test_window` asserts the rejected
design is detectably leaky — the rejection is executable, not just written down.

Tie rule: 0.005 nats, prefer Platt. Frozen in ADR 0025 **with a git date, before the first
production run**, from a paired-bootstrap noise measurement.

### Results, measured

Quarterly mean over 17 folds. Every model was **underconfident** — the opposite of ADR 0020's
prediction for the network.

| model | ECE before → after | MCE before → after | Brier before → after | slope before → after |
|---|---|---|---|---|
| `xgboost` | 0.0621 → **0.0474** | 0.1741 → 0.1150 | 0.2379 → 0.2350 | 0.640 → 1.005 |
| `lightgbm` | 0.0644 → **0.0490** | 0.1755 → 0.1260 | 0.2383 → 0.2351 | 0.618 → 1.015 |
| `logistic_regression` | 0.0635 → **0.0518** | 0.1664 → 0.1297 | 0.2382 → 0.2358 | 0.611 → 1.015 |
| `neural_numeric_only` | 0.0563 → **0.0524** | 0.1444 → 0.1201 | 0.2355 → 0.2347 | 0.791 → 1.003 |
| `xgboost_chain_embeddings` ⚠ | 0.0619 → **0.0481** | 0.1767 → 0.1236 | 0.2374 → 0.2346 | 0.651 → 1.029 |

**Ranking, verified by re-running `sentinel evaluate` on the calibrated artifact with no change
to Component 5: every delta exactly 0.00e+00.** PR-AUC, ROC-AUC, NDE and precision@k are
identical floats.

**Brier decomposition** — the entire gain is the term theory says it should be. Reliability falls
16–46%; **resolution is unchanged to five decimal places**; uncertainty is identical (0.24362) for
every model and every stage, because it is a property of the labels.

**Platt won all 90 (model, fold) cells; 0 method switches.** Per fold in isolation isotonic would
have won 16 of 90. Isotonic lost legibly: PAVA on a ~1,200-row window produces plateaus at exactly
0 and 1, so its test log-loss is worse than the *uncalibrated* model's on four of five candidates
and its post-calibration slope collapses to 0.42–0.58.

### What it did NOT implement

* No operating threshold, no cost-sensitive decision, no deferral gate — Component 16's.
* No temperature scaling: it is Platt with the intercept fixed at zero, and the fitted Platt
  intercepts (−0.000 to +0.011) already answer what a temperature would have been.
* No seed averaging of the neural candidate — **deferred by decision, not oversight**. It would
  create a base model Component 8 never evaluated and break the bit-identity gate.
* No ensembling, no new features, no re-tuning, no change to any earlier component's artifacts.

### Traps and findings to carry forward

* **Never seed anything from `hash()` of a string.** Python salts `str` hashing per process,
  so the bootstrap was not byte-reproducible until the seed key was changed to the candidate's
  registry position. A two-run sha256 comparison found it; nothing else would have.
* **Run `calibrate` without an `OMP_NUM_THREADS` override.** A first run under
  `OMP_NUM_THREADS=1` failed the gate on 32,696 of 41,536 `logistic_regression` rows, by 1e-13 to
  5e-10. Nothing was wrong with the model: the committed run used the library default, and a
  different thread count is a different BLAS summation order. The gate is *supposed* to be that
  sensitive.
* **`xgboost` and the network compute in float32.** `logit(p)` therefore differs from their native
  margin by up to 2.6e-5 — correct behaviour, not a defect. No downstream component should treat a
  difference below ~1e-6 in those models' scores as meaningful.
* **Three thresholds pre-declared from expectation were wrong**, and the measurements corrected
  all three (tie 0.002 → 0.005; margin 1e-9 → 1e-4; Platt self-check 1e-6 → 1e-3). A threshold set
  from expectation rather than measurement is a guess wearing a decimal point.
* **The model ordering by ECE inverted.** `neural_numeric_only` had the best uncalibrated ECE and
  now has the second worst. "Best uncalibrated" and "best calibrated" are different questions.
* **Isotonic ties are not ranking inversions.** It produced zero inversions but tied ~40,000 pairs,
  moving top-k membership 226–265 times and precision@k by up to 0.21. Conflating the two would
  misreport a correct calibrator as a broken one.
* **`covid_shift` diverged again** — five components, five divergences. Calibration *helped* there
  (ECE −10 to −23%) but the slope reaches only 0.75–0.90 because the base rate moves 17 points
  between its calibration and test windows. **Prior shift is not miscalibration and a monotone map
  cannot fix it.**

---

## 16c. Next task: Component 10 — and it is still blocked

**ADR 0019 stands.** The dataset has 22 columns and no inspector field, and nothing in Component 9
changes that. The next *implementable* component is 11 (SHAP) or 12 (fairness); Component 12 has
an input waiting for it in Component 8's community-area ablation (ADR 0023).

### Start from

```python
from sentinel.evaluation.contract import read_predictions   # reads the calibrated artifact as-is
# data/processed/predictions/calibrated_predictions_<stamp>.parquet -- 207,680 rows
```

Not from Components 6–8's raw scores. The calibrated artifact carries `base_score` alongside, so
the uncalibrated value is available without a join.

### Do not

* **Recalibrate.** The calibrators are frozen per (model, fold). Refitting one on a test quarter,
  including as a diagnostic, reintroduces the leak ADR 0012 built the calibration window to
  prevent.
* **Re-fit a base model.** Component 9 re-executed them once to recover a missing recording, behind
  a bit-identity gate. That is not licence to revisit them.
* **Read `trained_through` as `train_end`.** On this artifact it is `calibration_end` — the
  calibrator really did read that window. The estimator's horizon is `base_model_trained_through`;
  the first operationally available date is `calibrated_prediction_available_from`.
* **Average `covid_shift` into a quarterly mean**, or trust its probabilities as much as the
  quarterly ones.
* **Promote `xgboost_chain_embeddings`** on its calibrated Brier. Experimental (ADR 0022); it lost
  on NDE.

### Still open, and now sharper

**MEMORY open question 13 — which model should Sentinel carry forward — is still open, and
Component 9 made the disagreement worse rather than better.** `neural_numeric_only` wins NDE
(0.2482) and now has the *second-worst* calibrated ECE; `xgboost` has the best calibrated ECE
(0.0474) and the second-best NDE; `lightgbm` still wins precision@k_1_day. The four families
remain within 0.0156 NDE and the neural advantage remains smaller than its own seed spread.
A policy component will have to settle it.

### Reproducing Component 9

```bash
uv run python scripts/profile_calibration.py                      # the pre-registration evidence
uv run sentinel calibrate --report                                # ~25 min; NO thread override
uv run sentinel evaluate --predictions data/processed/predictions/calibrated_predictions_<stamp>.parquet
```

---

## 17. Reminder

**One component at a time.** Component 9 was probability calibration only, and is closed. No SHAP
(Component 11), no fairness (Component 12), no ensembling, no OR-Tools, no LangGraph, no
frontend.

And the rule that has now paid off five times: **investigate → document → design →
implement → test.** The findings document is written before the implementation, from a
read-only profiling script. In Component 6 that discipline caught the `add_indicator`
collinearity, the boolean median drift and the row-order sensitivity. In Component 7 it
caught the 0.11 row-order sensitivity — three orders of magnitude worse than Component 6's
and invisible from reading code — and it produced the calibration probe that let a
pre-registered expectation be *disproved* rather than quietly dropped.

One addition Component 7 earned: **when a leakage test fails, suspect the test first.**
Both failures in that component were test bugs that looked exactly like leaks.
