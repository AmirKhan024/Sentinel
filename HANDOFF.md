# HANDOFF

Read in this order: [MEMORY.md](MEMORY.md) → [STATUS.md](STATUS.md) → this file.

---

## 0. The one warning that matters most

Components 1–9 and 11–13 are closed. **Do not silently change an earlier component.** If
you find a defect upstream, document it and stop before modifying it.

Component 11 is the worked example of doing that right. It needed a fitted booster that
only a *private* helper in Component 8 could reach; it stopped, measured the boundary,
reported the model `unsupported` with the measurement as its reason, and wrote the
four-line public accessor that would lift the restriction into ADR 0031 without adding it.
"The change is small" is the argument every unauthorised edit to a closed component has
ever been made with.

Three invariants now sit on top of each other and are easy to confuse:

```text
Component 4:  can my FEATURES see the future?     -> a feature uses only records < d
Component 5:  can my EVALUATION see the future?   -> a model is scored only on a
                                                     window later than everything
                                                     it was allowed to learn from
Component 6:  can my MODEL see the future?        -> the fit AND every preprocessing
                                                     statistic come only from the
                                                     fold's training window
Component 11: can my EXPLANATION see the future?  -> the reference distribution a
                                                     SHAP value is measured against
                                                     comes only from the training
                                                     window
Component 12: can my GROUP LABEL see the future? -> the attribute a row is audited
                                                     under comes from an inspection
                                                     strictly earlier than that row
Component 13: can my POLICY see the future?      -> eligibility reads one as-of
                                                     column and no outcome column,
                                                     and no decision reads a
                                                     group-conditional number
```

The fifth is the quietest of all. The other four make a number wrong; a leaked group label
leaves every number finite, additive, plausible and in range, and changes only *which
neighbourhood the number is about*. No metric moves, no check fires, and the audit reports a
real disparity attached to the wrong place. That is why `groups.check_temporal_validity`
re-derives the inequality per row rather than reading Component 8's manifest, and why ward is
refused: its two published boundary vintages disagree on 98.3% of rows, so "the current ward"
attached to a 2019 inspection is exactly this leak wearing a geography's clothes.

The sixth has two halves and the second is the one with no precedent. The first half is
ordinary: a policy that read `target` would allocate inspections using the answer to the
question the inspection is meant to settle, so `eligibility.refuse_forbidden` raises before the
predicate is even built, and no decision table carries a label column at all.

The second half is new to this component. Component 12's per-group numbers are computed from
held-out outcomes, so a policy that ranked on one would be ranking on the future — and the
artifact would look completely normal, because a group label and a support status are perfectly
ordinary strings. So the group columns are read *onto* the recommendation rows and never back
into a rank, and the claim is checked rather than argued: `_queue_signature` rebuilds the entire
queue with the group label and the support status absent, and
`validate.warnings_do_not_change_the_queue` compares the ranks exactly. The end-to-end version
runs the whole component twice, with and without the group artifacts, and requires identical
output.

The fourth is the subtlest, because nothing about a leaked background *looks* wrong: the
values stay finite, additive and plausible, and every additivity check still passes. What
changes is the question the numbers answer.

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
effects (C10), no SHAP (C11 — since built), no fairness (C12), no policy/scheduling/routing
(C13+).
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
* **No SHAP.** Component 11's, and since built: attribution now lives in
  `data/processed/explanations/`. Native split-gain importances remain a **diagnostic only**,
  and the data contract still says so in bold — a consumer wanting attribution should read
  Component 11's artifact, not `boosted_importances_*.parquet`.
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

No calibration (Component 9), no SHAP (Component 11 — since built, and it could explain
`neural_numeric_only` but *not* `xgboost_chain_embeddings`; see ADR 0031), no fairness audit
(Component 12), no
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

## 16d. What Component 11 did

**Attributed every supported model's predictions to features, per fold, and proved the models it
explained are the models the committed artifacts were written by.**

### What it implemented

* `src/sentinel/explain/` — 11 modules. `definitions.py` (the support matrix and every frozen
  constant, with an import-time guard), `models.py`, `refit.py` (the re-execution seam),
  `background.py`, `sample.py`, `attribute.py`, `aggregate.py`, `validate.py`, `writer.py`,
  `figures.py`, `build.py`.
* One command: `sentinel explain`.
* A **seventh** processed layer, `data/processed/explanations/`, holding seven tables.
* 173 new tests (1,911 -> 2,084), 19 error-severity checks + 1 advisory, 29 figures.
* ADR 0028 (the layer), 0029 (re-execution), 0030 (the protocol), 0031 (the unsupported model).

### The blocker it hit, twice, and how each was resolved

**First: no fitted model object is persisted anywhere** — ADR 0026's problem again, for a
different reason. Component 9 needed scores over an unscored window; Component 11 needs the
model itself, because TreeSHAP walks trees and a permutation game calls a network. So the fits
run again, under the *same* gate, calling
`calibration.basescores.{committed_test_scores, reproduction_mismatches}` rather than
reimplementing them — one definition of "the same model" in the project, not two.

> **166,144 test rows across 4 models x 18 folds. Zero mismatches.**

`build.py` raises before computing a single attribution if that fails.

**Second: `xgboost_chain_embeddings` cannot be reached.** Its fitted booster lives only in
`neural.embed._scorer_for`, a private process-local stash; `FittedEmbeddingBooster` has 19
fields and none is the estimator. `neural.train.scorer_for` is public — which is exactly how
the *network* is explained — and the asymmetry is an accident rather than a decision. Component
8 is closed, so the model is reported `unsupported`, with the measurement as the reason and the
four-line fix proposed in ADR 0031 and not taken.

### Five decisions worth knowing before you touch it

1. **One method per family, chosen because of what the estimator is.** Exact closed form for
   the linear model, the boosters' own exact TreeSHAP, an antithetic permutation game for the
   network. Forcing all three through one model-agnostic explainer would have thrown away two
   exact computations to buy uniformity.
2. **`shap` is dev-only.** xgboost and lightgbm already ship exact TreeSHAP; linear SHAP is
   three lines of arithmetic. Only the network needs an approximation. So the values are
   computed in-house and the suite cross-checks them against `shap` — ADR 0015's dividing line,
   and exactly what Component 5 did with scikit-learn. **No runtime dependency was added.**
3. **Everything is in log-odds, and probability space is refused rather than declared.**
   Contributions add up in the margin and not in probability, because sigmoid is not linear.
   `OutputSpace` has one member.
4. **SHAP explains the base score, not the calibrated probability.** Platt is a separate
   two-parameter monotone map; the two numbers are carried side by side on
   `explanation_cases` so neither can be mistaken for the other.
5. **The background is the leakage surface.** Reference rows come from the fold's *training*
   window via `modeling.train.training_frame`, and two checks re-derive that from the frame —
   one on dates, one on split membership, because a date comparison is weaker than the split.

### Results, measured (2026-08-25)

72 refits in 438.6 s; attribution 647.3 s; ~19 min wall clock. 21,600 explained predictions,
648,000 attribution values. All 20 checks pass.

**The headline is a disagreement.** The four models score within **0.0156 NDE** of one another
(Component 8) and their importance rankings correlate at **rho = 0.4351** between
`logistic_regression` and `lightgbm`, sharing 3 of their top 10 features. The two boosters agree
at 0.9871. Near-identical accuracy, materially different reasoning. That strengthens Component
7's "the ceiling is the representation, not the estimator" — and it removes any argument from
explanation to model choice, because there is no consensus to appeal to.

**What they lean on:** prior inspection history, in every model.
`prior_canvass_count_code_era` is rank 1 for three of four. Notably a *missingness indicator*,
`missing_no_code_era_canvass`, is rank 3 for the logistic model and rank 2 for the network — the
absence of a record is among the most informative signals available.

**Stability:** consecutive-fold rank rho 0.9606-0.9753 for tree and linear, **0.8914** for the
network. First fold to last (four years) falls to 0.7495-0.9197. Stable quarter to quarter;
measurably drifting over the study period.

**COVID confirms Component 6's inference.** Three of four models leaned **2-3x harder** on
`days_since_any_inspection` under the shift (xgboost rank 3 -> 1, 0.1499 -> 0.3728). Component 6
found the model ordering inverting there and hypothesised from an ablation that the feature
encodes scheduling policy. Component 11 shows the mechanism directly.

### Traps and findings to carry forward

* **Additivity is not accuracy for the permutation method.** Its path telescopes, so
  `base + sum(phi)` reconstructs the output *exactly at one round* — measured residual 0.0 at
  every round count. A green additivity check says the arithmetic is sound and nothing about
  whether the credit split is right. Three places say so, and a test asserts both halves.
* **The name-recovery trap is real and was measured before it bit.** Components 6 and 7 order
  the same 30 columns differently at **19 of 30 positions**. Picking the wrong list produces a
  table whose every value is arithmetically correct and attached to the wrong feature — no
  exception, no failed check, because a sum is invariant to a permutation of its terms.
  `feature_names_match_the_declared_name_source` re-derives them independently.
* **A first draft of the leakage test failed, and the test was wrong.** It shifted
  `frame.head(500)` by 900 days to simulate "the future"; those rows landed back inside fold 0's
  own training window. Component 7's rule paid for itself a third time: **when a leakage test
  fails, suspect the test first.**
* **The tolerance set from the probe fold was too tight for the real run.** The probe measured a
  tree additivity residual of 8.92e-07; the production run reached 1.66e-06. The frozen 1e-5
  absorbed it. A tolerance set from one fold is a tolerance set from a sample of one.
* **`shap` needs `numba>=0.67` and `llvmlite>=0.49` pinned**, or the resolver backtracks numba
  to a pre-numpy-2 release and the build fails. Those floors are what keep numpy at 2.5.2, which
  is what ADR 0026's gate is baselined on.

### One upstream issue found and deliberately NOT fixed

`uv run ruff format --check .` fails on **11 Component 9 files** — the `calibration/` package,
its four test files and `scripts/profile_calibration.py`. This is **pre-existing and unrelated
to Component 11**: `uv.lock` pins ruff 0.16.3 and Component 11 did not change it, so those
files were already unformatted under the installed ruff before this component existed. The
formatter's line-breaking changed between the release Component 9 was written against and
0.16.3.

Running `ruff format .` fixes it in one command and touches nothing but whitespace — and it
was **reverted** rather than kept, because Component 9 is closed and a cosmetic diff across a
closed component is still a diff across a closed component. Every Component 11 file passes.

Whoever next opens Component 9 should run it then. If it is fixed in isolation, do it as its
own commit with nothing else in it, so the diff is self-evidently formatting.

---

## 16e. What Component 12 did

**Audited whether Sentinel behaves the same way across the geographies this data can define,
across seven surfaces measured independently, and reported what it found without fixing any
of it.**

### What it implemented

* `src/sentinel/fairness/` — 14 modules. `definitions.py` (the group registry, holding the
  **refused** definitions as well as the audited ones, plus every frozen constant behind an
  import-time guard), `models.py`, `groups.py`, `support.py`, `metrics.py`, `priority.py`,
  `missingness.py`, `attribution.py`, `disparity.py`, `drift.py`, `validate.py`, `writer.py`,
  `figures.py`, `build.py`.
* One command: `sentinel audit-fairness`.
* An **eighth** processed layer, `data/processed/fairness/`, holding ten tables.
* 247 new tests (2,084 -> **2,331**), 13 error checks + 3 advisories, 72 figures.
* ADR 0032 (the layer), 0033 (the group frame, and the refused geographies), 0034 (the support
  policy and the advisory boundary), 0035 (what this component does not claim).

### The first component in this project that re-executes nothing

Component 9 regenerated scores that were never recorded. Component 11 regenerated the models
themselves. Both did it behind ADR 0026's bit-identity gate, and both had to defend the
regeneration.

Component 12's inputs all exist on disk, so its integrity claim is the *opposite* one:
**nothing moved.** Every input's sha256 is read before the run and again after the last table
is written, and a difference is an error-severity failure. There is no gate, no refit and no
BLAS thread sensitivity — `audit-fairness` can be run with any thread count.

### Five decisions worth knowing before you touch it

**1. The group label comes from the previous inspection, and that cost nothing.** The row's
own recorded location would have been defensible — it is contemporaneous, not future — but it
would have made Component 12 the only component that reads a field off the row it is
processing. Before choosing, both were measured: **0 disagreements on 57,041 community-area
rows and 0 on 57,326 ZIP rows.** A restaurant does not move. Taking the safe option needed no
exception to ADR 0010 and reused a frame Component 8 already validates per row.

**2. Ward is refused, and the dataset is the evidence.** The snapshot publishes *two* ward
layers — current and 2003–2015 — and they assign different region ids to **56,451 of 57,403**
rows. A ward id is a property of a boundary *version*, not of a place. Census tract (797
groups over 32,696 rows), point geography and city/state are refused too, each with a
measurement. **The refusals are rows in `fairness_group_definitions`**, so they keep travelling
when somebody opens the Parquet instead of the ADR.

**3. Support is decided before any metric, and the per-fold grain does not survive it.** The
median (fold, community area) cell holds **16 rows**; 4 of 1,288 clear the 200-row floor. So
the reporting grain is the pooled fold set — every row still strictly held out, and every
pooled row labelled as *the system as operated over 2022Q2–2026Q2* rather than one estimator.
Floors were frozen from `scripts/profile_fairness.py` before any result: 200/20/20 for ranking,
**300** for calibration, which is 15 equal-mass bins x 20 rows. **The bin count was not
reduced** to let more groups qualify, because that would make every group ECE incomparable with
Component 9's global one — the exact comparison the component exists to make.

**4. A disparity is advisory and can never fail the build. There is no flag to change that.**
13 error checks are all about the audit's own integrity; 3 advisories are about the world. The
reason is not that inequality does not matter — this component exists because it does — but
that a red build is a demand for action, and the actions available to whoever faces a red
fairness check are to change the model, the metric or the threshold. Two of those three are
worse than the disparity. `test_an_enormous_disparity_is_advisory_and_never_an_error` asserts a
0.95 ECE spread leaves every error check green and the exit code zero.

**5. Nothing was fixed, deliberately.** Calibration was measured making a third of
neighbourhoods worse and was left alone: a per-group calibrator would change Component 9, which
is closed, and it is a substantive fairness decision disguised as a repair — it trades overall
calibration for group calibration and makes the probability a function of the neighbourhood,
which is the thing ADR 0023 declined to let the *model* do.

### Results, measured (2026-08-25)

**5 models x 2 geographies x 18 folds, 207,680 audited rows, 145.1 s.** All 13 error checks
pass; 13 advisory findings; inputs byte-identical before and after.

**The disparity exists in the data before any model does**: outcome rate spans **0.2200 →
0.5658** across supported community areas against a city-wide 0.4283. So a working risk model
is *expected* to select at different rates, and "unequal selection" is not by itself a finding.

**Ranking varies more between neighbourhoods than between models.** Within-group ROC-AUC spans
**0.509 → 0.710** across the 51 supported community areas — a spread of 0.164 to 0.198
depending on the model, against roughly 0.008 between the project's best and worst model.
Community area 53 is the best-ranked group for all five models.

**Component 9's global improvement did not reach every group.**

| model | community areas improved | mean ECE base → calibrated |
|---|---:|---|
| `lightgbm_platt` / `xgboost_platt` | 25 / 33 | ~0.094 → ~0.084 |
| `logistic_regression_platt` | 23 / 33 | 0.0966 → 0.0854 |
| `neural_numeric_only_platt` | **17 / 33** | 0.0884 → 0.0862 |

Coherent with Component 9 rather than against it: the network had the best *uncalibrated* ECE
and the least to gain, and Component 9 already recorded its ECE ordering inverting.

**The sharpest finding is a group with no geography, and the chain closes with every link
measured.** 405 quarterly rows carry `community_area = __UNKNOWN__`:

```text
59.5% have no prior inspection of any kind   (0.74% overall)  -- 80x
61.7% have no code-era canvass history       (10.4% overall)  --  6x
      -> ROC-AUC 0.509 (chance)
      -> selected at 0.20x the city rate
      -> 0.6% of its violations captured by the top 5% (city-wide 7.0%)
```

Of its 166 real violations the top 5% found **one**. The group with no recoverable geography
*is* the group with no history — Component 8's as-of join can only carry a location forward
from an earlier inspection. This is the missingness indicator Component 11 ranked 2nd/3rd in
importance, resolved by neighbourhood. **No causal direction is claimed**, and "we have never
inspected this place" remains a true and relevant fact.

**Drift could not be answered.** Exactly **one** quarterly fold per (model, geography) has
enough support to compute a disparity at all, so every series is `insufficient_folds` rather
than a line through one point.

**`covid_shift`, separately.** 8,840 rows — more than any single quarter — supporting **11 of
78** community areas. A stress-test observation; no trend claimed. Six components, six
divergences.

### Traps and findings to carry forward

* **A leaked group label is the quietest leak in the project.** It leaves every number finite,
  additive and in range and changes only which neighbourhood the number is about. Nothing
  raises. Two checks re-derive it from the dates rather than from a manifest.
* **Polars aggregates in parallel, and float summation order is not stable.** Two runs produced
  `mean_abs_shap` values differing at **4.4e-16** — every rank, correlation and count
  identical, so nothing a reader would act on moved. Fixed by sorting before aggregating. The
  same lesson Components 6 and 7 learned about row order at fit time, arriving at the
  arithmetic instead. **A table that is *nearly* reproducible is a table whose two-run checksum
  comparison has stopped being a detector.**
* **`ece` uses equal-mass bins, so rows tied at a bin boundary are assigned by arrival order.**
  Shuffling the prediction rows changed the pooled reference value and therefore every
  disparity, until `groups.CANONICAL_SORT` was added. Found by a test, not by reading code.
* **Component 9 and Component 11 name the same model differently** — `xgboost_platt` against
  `xgboost`. Looking a profile up under the calibrated name found nothing and drew no figure,
  which is the quietest possible failure: **a missing figure looks exactly like a figure the
  data could not support.** `figures.base_model_name` is the one place that translation lives.
* **The support policy is what makes the small-group problem visible rather than solved.** 27
  of 78 community areas are below the floor and appear in no comparison. They are rows with
  real counts and a stated reason, and a check compares the support table against the values
  observed in the data.

### What it must not be read as

`does_not_establish` travels in every manifest and prints on every run: **not causality, not
discrimination, not the absence of bias, not legal compliance, not ethical acceptability, not
equal treatment, not an optimal fairness policy.** ADR 0019's gap is inherited rather than
discovered — the target is that a violation was *cited*, and Chicago assigns inspectors by
district, so **nothing here separates establishment risk from differential inspection
practice.**

---

## 16f. What Component 13 did

**The component that turns a probability into an instruction, and the one whose headline result
is a refutation.**

### The surprise, and why the profiler caught it

Component 12 handed over a closed loop: establishments in the `__UNKNOWN__` community area are
ranked at chance and the top 5% found one of their 166 citations. The obvious response is to
reserve capacity for low-information establishments.

**The profiler measured that reading before any policy code existed, and it is wrong.**
Establishments with no code-era canvass history are 10.4% of the candidates and already take
**40–58% of the top of the queue** — a selection ratio of **3.96 to 5.57** across all four
candidate models — because their citation rate is genuinely higher (0.4883 against 0.4283). The
models are not neglecting that population; they lean on it heavily and are right to.

Component 12's finding is about a **different** population. Only **456 of 14,162** eligible rows
(3.2%) sit in `__UNKNOWN__`. "No history" and "no geography" overlap and are not the same thing,
and conflating them would have produced an intervention aimed at nothing.

### What it built

`src/sentinel/policy/` (12 modules), `sentinel decide`, a **ninth** processed layer with eleven
tables, 228 new tests (2,331 → **2,559**), 18 error checks + 4 advisories, 4 figures, ADRs
0036–0040.

Seven policies: pure risk, plus a coverage **floor** (guarantee an outcome) and a **forced**
reserve (guarantee a spend) at half, exactly, and twice the measured 0.1043 population share.

| at one week of capacity | reserve slots | citations | Δ | eligible served | Δ |
|---|---:|---:|---:|---:|---:|
| `pure_risk` | 0 | 1,657 | — | 1,170 | — |
| `coverage_floor_double_share` | 2 | 1,657 | 0 | 1,172 | +2 |
| `coverage_forced_population_share` | 274 | 1,642 | **−15** | 1,325 | +155 |
| `coverage_forced_double_share` | 556 | 1,623 | **−34** | 1,513 | +343 |

The floor is inert in **338 of 340** quarterly cells at the population share. The forced reserve
buys coverage at a price in citations, and the price grows with capacity.

### Open question 13, closed — and what it revealed

`xgboost_platt` is the production model, chosen by a rule frozen in advance. **Axis 1 separated
nothing:** under Component 5's own 1,000-replication label-flip study, all four candidates' NDE
intervals overlap. The rule fell to calibration.

⚠ **The tie rule decides the deployment and was fixed after its inputs were first read.** The
plan carried a placeholder borrowed from a different metric — Component 8's five-seed *ROC-AUC*
spread of 0.0058 — and under it `neural_numeric_only_platt` would have been selected. Both
outcomes are emitted on every run and ADR 0039 records the sequence.

### What it refused

* No score adjusted by geography, no group-specific threshold or calibrator, no quota, no
  probability threshold. `CAPACITY_SEMANTICS` states that every cutoff is a rank position.
* **Nothing changed for the `__UNKNOWN__` group.** At one day of capacity it gets 2 of 556 slots
  and 1 citation found — identical under all seven policies. Reaching it would need an
  allocation keyed to a failed geocode (ADR 0038). Reported, not worked around.
* **No policy winner declared.** Two policies survive and neither dominates; the run prints *the
  data does not determine the correct policy*.

---

## 16g. Component 14 — operational scheduling. **CLOSED.**

The brief below was written before Component 14 was built. It is kept verbatim, because the
component was measured against it and honoured every prohibition in it. What follows the brief
is what the component found.

Component 10 remains **BLOCKED**. ADR 0019 stands; nothing in Components 11, 12, 13 or 14
changes it — and Component 14 measured the same absence again from its own angle.

### Start from

```python
# data/processed/policy/inspection_recommendations_<stamp>.parquet   1,453,760 rows
# data/processed/policy/policy_selection_allocation_<stamp>.parquet  the arithmetic
# data/processed/policy/policy_comparison_<stamp>.parquet            what each policy costs
```

**Component 13 is the first component whose output is an instruction rather than a
description.** Every other layer says what *is*; `inspection_recommendations` says what to do —
this establishment is the fourth inspection on Tuesday. Component 14 turns that into a schedule
against inspector counts, working days and whatever else the city actually constrains, none of
which Component 13 modelled.

### Do not

* **Re-rank.** Component 13 owns the queue. A scheduler that reordered by risk would be a second
  policy layer with no ADR behind it.
* **Adjust a score.** No component after 9 writes one. ADR 0037 gives the reason: once a score is
  adjusted, nobody can say whether an establishment is in the queue because the model thinks it is
  risky or because a policy promoted it — and that question is the whole point of Component 13.
* **Raise capacity.** Every cutoff descends from the window's own measured median daily rate. A
  schedule that fitted more inspections than the city worked would beat every alternative for
  that reason alone, and this project's simulation has never been willing to change capacity.
* **Introduce a probability threshold.** Refused by Component 12 in prose and by Component 13 in
  `CAPACITY_SEMANTICS`. There is no flag to add one, deliberately.
* **Treat the coverage reserve as slack.** It is an allocation with a measured price. A scheduler
  that raided it to fit a route would silently be changing the policy.
* **Read a one-day policy delta as a result.** They are ±1 to ±3 citations out of 348 across
  seventeen folds — inside the noise. The week-scale numbers are the reliable ones, and the
  findings document says so including where the noise flatters the component.
* **Join anything in `data/processed/policy/` onto a feature table.** It is keyed by *row* and
  holds the system's own past decisions; broadcast back into training it closes the feedback
  loop Component 12 measured and Component 13 was built to keep visible.
* **Assume the production model is settled science.** It is an operating choice from a rule whose
  tie band decides the answer, and two defensible bands pick two different models.

## 16h. What Component 14 measured, and what it left for Component 15

### The finding

**Component 13's coverage reserve is substantially notional once a real calendar is applied.**

`policy/allocation.py` fills the risk block at ranks `1..n_risk` and places the reserve after it,
so the reserve is *always* the tail of the rank order — checked across all 273 reserve-bearing
cells with no exception. A strict-priority schedule fills the horizon from the top, so a horizon
that falls short takes the reserve first, every time, without the scheduler ever looking at a
mechanism.

```text
1,012 of 3,459 coverage-reserve slots lost to the horizon        29.3%
136 of 273 reserve-bearing cells lose some of it
 91 of 273 lose it entirely
```

Neither layer is wrong on its own terms. ADR 0037 priced the reserve in forgone citations and
granted it a slot count; nothing in that decision said the slots had to sit at the *end* of the
queue, and Component 13 had no calendar with which to notice that it would matter. The cost
lands on the mechanism the policy layer went to the most trouble to make explicit.

**Component 14 reports it and does not correct it.** Promoting reserve rows in the schedule is
re-ranking, which §16g forbids in those words, and it would put one coverage decision in two
layers. The advisory fires; the build stays green. This is the second time Sentinel has measured
a prior component's premise instead of inheriting it.

### The measurement that decided the component's shape

Component 13's cutoffs descend from a quarter-wide median, so a schedule built on that same
median is feasible before anything is measured — the horizon is `k / median` days of `median`
slots, so it holds exactly `k`. At `k_1_day` and `k_1_week` that is a **tautology**: backlog
zero, utilisation exactly 1.000, by construction.

Built against the days Chicago actually worked it is not:

```text
44 of 90 (fold, capacity) cells cannot fit their approved queue     48.9%
784 approved inspections do not fit inside their own horizon
 0 backlog under the flat median, in every cell
```

Both modes ship. `observed_calendar` is the default because it describes days that happened;
`flat_median` is retained and labelled a scenario everywhere, because the contrast *is* the
finding.

### Start from

```python
# data/processed/scheduling/inspection_schedule_<stamp>.parquet     136,094 scheduled rows
# data/processed/scheduling/schedule_backlog_<stamp>.parquet        approved and not reached
# data/processed/scheduling/priority_preservation_<stamp>.parquet   what the calendar cost
# data/processed/scheduling/schedule_slots_<stamp>.parquet          the observed capacity grid
# data/processed/scheduling/execution_contract_<stamp>.parquet      both external file formats
```

### Do not

* **Re-rank.** Component 13 owns the queue and Component 14 preserves it exactly — 0 inversions
  across 1,260 cells. A router that reordered by risk would be a third layer with an opinion
  about priority, and no ADR behind it.
* **Raise capacity.** Every horizon descends from the window's own measured median daily rate by
  way of Component 13's `k`. There is no `--capacity`, `--slots-per-day`, `--horizon-days`,
  `--extend-horizon` or `--threshold` flag, and `tests/test_cli_scheduling.py` asserts each
  absence. A schedule that fitted more inspections than the city worked would beat every
  alternative for that reason alone.
* **Treat the coverage reserve as slack.** Component 14 measured what the calendar already costs
  it. A router that spent what remains to shorten a drive would be making a policy change
  invisibly.
* **Let an execution outcome edit a plan or a recommendation.** `inspection_schedule` has no
  `execution_status` column, deliberately: a column that does not exist cannot be written to by
  a future edit that seemed reasonable at the time.
* **Merge the three human layers.** A recommendation override changes *who*, a scheduling
  adjustment changes *when*, an execution deviation records *what happened*. Three id
  namespaces, three disjoint verb vocabularies, enforced at import time.
* **Join anything in `data/processed/scheduling/` onto a feature table.** One layer further out
  than Component 13 already refused to close.
* **Fabricate an inspector.** Two components are now blocked on the absence and both measured it
  independently. A third that invented one would make all three unfalsifiable.
* **Read the observed calendar as a forecast.** It is measured from the window it schedules. It
  says what capacity *existed*, not what a planner could have known on day one.

### Component 15 is blocked

OR-Tools routing needs an inspector, a base location, a duration and a travel time. The snapshot
has none of the four. ADR 0043 records the check rather than the assumption, and
`scripts/profile_scheduling.py` profile 7 is the inventory: twelve operational fields a real
inspection department schedules against, all absent. Latitude and longitude do exist, and ADR
0033 and ADR 0038 already refuse geography-keyed allocation on separate grounds.

**Component 16 — the deferral / human-review gate — is now implemented.** See §16k below. It used
the two pieces Component 14 shipped for it: an external contract shaped like Component 13's
override, and a re-planner that appends a planning run rather than mutating one.

### The rule that has now paid off seven times

**investigate → document → design → implement → test.** The findings document is written before
the implementation, from a read-only profiling script.

In Component 14 it paid off three times in one run. Profile 3 found that the flat-median
capacity mode is a tautology at two of five cutoffs, which is why it is not the default. Profile
4 killed a shortcut before it was written — a synthesised Monday-to-Friday calendar would have
been wrong at the edges and needed an unverifiable holiday list. Profile 5 refuted an invariant
the component was about to assert: "no establishment occupies two slots" is false on this data,
because 1,573 establishment-fold pairs hold more than one scored canvass, and asserting it would
have produced a red build on correct data — the failure mode that makes a suite stop being
believed. And profile 8, added last, found the reserve-loss result that became the headline.

One addition Component 14 earned: **when a validator fails on your own new code, read the
validator before the code.** Five error checks went red the first time the adjustment and
execution paths ran end to end. Four were real defects — a sort key that was no longer a total
order once a re-plan appended a second plan, a displaced row landing back in its own slot, a
check comparing the full backlog table against only planning run 0, and an identity that
double-counted deferred rows. The fifth was the *check* being wrong: the temporal boundary as
first written forbade moving a `not_performed` row whose day had passed, which is precisely the
operation re-planning exists to perform. Component 7 learned to suspect the test first; this is
the same lesson one layer out.

---

## 16i. The Sentinel API — cross-cutting infrastructure, not a numbered component

Built after Component 14, and deliberately **not** called "Component 15" or "Component 16" —
those names keep the meanings above (routing, blocked; the deferral/human-review gate, since
built — see §16k). The Sentinel API (`src/sentinel/api/`, run with `sentinel serve`) is a
validated read/write HTTP boundary over the artifacts Components 1-16 already produce. It
computes nothing: every field a response carries was already written by a batch CLI command.

Three decisions, each with its own ADR:

* **ADR 0048** — it is a sixth *interface* over the five layers ADR 0042 named, never a sixth
  *layer*. No routing endpoint exists, for the same reason Component 14 built none (ADR 0019,
  ADR 0043).
* **ADR 0049** — writes (an override, a scheduling adjustment, an execution event) are validated
  against the exact contracts Components 13 and 14 already define, then **staged** to an
  append-only file this package owns, in the exact shape the batch CLI already reads. The API
  never calls `select`, `run_schedule`, `replan`, `apply_adjustments` or `record_execution`
  itself — turning a staged request into a new artifact stays a manual step through
  `sentinel decide` / `sentinel schedule`, because those commands rebuild a whole cell's worth of
  checksummed artifacts in one batch, and reimplementing that inside a single HTTP request would
  mean a second, unaudited copy of it.
* **ADR 0050** — a request that does not fully specify its decision scope (policy, fold,
  capacity, schedule configuration) is refused with `422`, never silently resolved to "latest" or
  "first."

See `docs/data_contracts/sentinel_api.md` for the endpoint list and `docs/interview/api_layer.md`
for the plain-language walkthrough, including what a real deployment would still need
(authentication, an operational drain process for the staging store) that this component
deliberately does not build.

---

## 16j. The Sentinel Frontend — minimal, read-only, product testing only

A minimal React + TypeScript + Vite frontend (`frontend/`) was added on top of the Sentinel API,
for one purpose: let a human actually browse recommendations, explanations, schedule and backlog
and confirm the product behaves the way `docs/data_contracts/sentinel_api.md` says it does. It is
**not Component 21** ("Frontend demo", still unbuilt in the roadmap table) — that remains a
separate, larger, future deliverable, and this frontend does not occupy or redefine its place.

It is **read-only**: no override/adjustment/execution-event forms in this pass. Its scope UX
mirrors ADR 0050 client-side — no request fires while a required `DecisionScope` field is unset,
and a returned `422 ambiguous_scope` is rendered from its actual body. It duplicates no model,
policy or scheduling logic; every value it shows is read verbatim from an API response.

**The one sanctioned change to `src/sentinel/`:** CORS middleware was added to
`src/sentinel/api/app.py` (allowed origins configurable via the new `Settings.api_cors_origins`,
defaulting to the Vite dev server's `localhost:5173`), because a browser cannot call a
cross-origin API without it and none existed before. Nothing else under `src/sentinel/` — no
router, service, schema, or any closed component's logic — was touched. One new backend test file
(`tests/api/test_cors.py`) covers it.

See `frontend/README.md` for how to run both processes together and the full list of what was
deliberately left out (routing/map, auth, write actions, a sort-column picker).

---

## 16k. Component 16 — the deferral / human-review gate

Two deterministic triggers, no numeric threshold, ever:

* `policy_warning_present` — a *selected* Component 13 recommendation carries at least one
  `PolicyWarning`. This is the escalation ADR 0040 declined to build automatically.
* `no_execution_record_on_scheduled_row` — an occupying Component 14 schedule row (at the cell's
  latest `replan_index`) has no matching row in the accumulated execution log, keyed on the
  execution contract's own fields.

Both are boolean facts an upstream component already wrote or a plain anti-join already computes.
`queue_is_deterministically_rebuildable` proves neither trigger reads `score`, `base_score` or
`final_policy_rank`, by rebuilding the queue from the two triggers and comparing byte for byte.

**The threshold tension, resolved on the record.** Three docstrings elsewhere in this project
(`evaluation/metrics.py`, `calibration/definitions.py`, `evaluation/build.py`) each say "a
threshold is genuinely needed only by the Component 16 deferral gate." ADR 0040 forbids
fabricating a numeric confidence cutoff — this project has never built a predictive interval, a
conformal set or an ensemble spread. ADR 0051 resolves the tension by reading "threshold" as the
flag/no-flag decision boundary the two triggers already draw, not a probability cutoff, and states
so explicitly rather than silently picking a number or silently ignoring the hint.

**A fourth human layer, disjoint from the other three by construction.** Component 14's
`ScheduleStatus.DEFERRED` already means "moved to a later operating day" — a different idea from
"send to a human for review." `sentinel.review.definitions._guard_registry()` checks, at import
time, that `ReviewResolutionAction` (`acknowledge`, `refer_to_override`, `refer_to_adjustment`,
`escalate`) shares no value with `OverrideAction` or `AdjustmentAction`, and — stronger than mere
set disjointness — that no Component 16 vocabulary value contains the literal substring `"defer"`.
A `refer_to_override`/`refer_to_adjustment` resolution records a pointer only; it never creates
the override or adjustment itself, which remains a separate submission through Component 13's or
Component 14's own contract.

**The queue is rebuilt fresh each run; the resolution log is append-only.** `human_review_queue`
reflects current upstream state — a case whose trigger condition later resolves (an execution
record arrives) drops off the next run's queue, mirroring `inspection_recommendations`.
`review_resolution_log` is the permanent record of what a human decided, mirroring
`policy_override_log`.

**Measured against real data (2026-08-27):** of 1,453,760 recommendation rows, 70,791 were
flagged — 39,652 by `policy_warning_present`, 70,791 by `no_execution_record_on_scheduled_row`.
That second number is **not** an operational finding: the production `execution_log` is currently
empty, so every occupying schedule row counts as "missing" a report nobody was ever going to
file. Stated plainly in ADR 0051 rather than reported as a discovery.

CLI: `sentinel review` (mirrors `decide`/`schedule` exactly — flag validation before path
resolution, optional `--schedule`/`--execution` with informational-log-on-missing, `--dry-run`,
`--report`). API: `GET /v1/review/queue`, `GET /v1/review/queue/{id}`, `GET
/v1/review/resolutions`, `POST /v1/review/resolutions` (stage-only, ADR 0049's pattern, a fourth
`_KIND_CONFIG` entry in `staging_service.py`). See ADR 0051,
`docs/data_contracts/human_review.md`, and `docs/interview/component_16.md`.

---

## 16l. End-to-end integration verification (2026-08-27)

With Components 1–14 and 16 individually closed, a separate pass asked a different question:
does the *system* work when actually run through its real interfaces, not just through unit
tests seeded with synthetic fixtures? `sentinel decide`/`schedule`/`review` were run against the
real committed artifact chain; `sentinel serve` was started for real and exercised with live HTTP
requests (including negative cases: ambiguous scope, unknown row, duplicate staged id, invalid
resolution, a schedule fed a truncated recommendations file); the frontend dev server was started
against the real running API and its own test suite run against it.

**Two real integration bugs were found, both invisible to the existing unit-test suite because a
test fixture on one side of each bug encoded the same wrong assumption as the code on the other
side:**

1. `establishment_service.py` passed Component 9's *calibrated* model name
   (`xgboost_platt`) straight through to Component 11's explanation lookup, which only ever
   carries the *base* name (`xgboost`) — the exact mismatch `docs/data_contracts/explanations.md`
   §0a already named as a known footgun, now recurred at the API boundary. Every real
   establishment's explanation reported "not a recognised model," for models Component 11
   genuinely explains. Fixed with `explain_service.base_model_name_of()`, which strips a known
   `sentinel.calibration.definitions.Method` suffix once, at the boundary; the test fixtures
   (`tests/api/conftest.py`'s `explanation_support_row`/`explanation_case_row`/
   `explanation_value_row`) had defaulted to the calibrated name too, which is why the existing
   test for this path passed for the wrong reason.
2. `frontend/.env` (local, untracked) pointed `VITE_SENTINEL_API_BASE_URL` at port 8010, which
   nothing serves; `.env.example`, the MSW test fixtures and `Settings.api_port`'s own default all
   agree on 8000. The frontend's own test suite was failing 12 of 33 tests for exactly this
   reason. Fixed by correcting the stray local value.

Full detail, including the exact commands run and their results, and one real limitation found
but deliberately not fixed in this pass (Component 16 does not cross-validate that a manually
overridden `--recommendations` file matches the run its auto-discovered `--schedule`/`--execution`
artifacts came from), is in `docs/analysis/integration_verification_20260827.md`. Regression after
both fixes: 3,190 tests pass (up from 3,181), ruff/mypy clean, frontend `vitest`/`tsc` clean.

---

## 16m. Frontend product-clarity pass (2026-08-28)

The frontend in §16j was technically correct and fully integration-verified in §16l, but it
assumed a visitor already understood Sentinel's internal vocabulary — a raw scope selector first,
tables whose column headers were literal field names (`decision_mechanism`, `backlog_reason`),
an Overview page that was a manifest dump. A follow-up pass rebuilt the *primary* experience
around plain language, without removing any of the above.

**Progressive disclosure, not deletion.** `src/lib/copy.ts` is the one new module that maps a
technical code to a plain-language string (e.g. `no_execution_record_on_scheduled_row` → "This
scheduled inspection does not currently have a matching record of what happened"). The raw code
is always still shown too, moved into a collapsed "Technical details" section on every page.

**A working scope with no clicks.** `useDefaultScope` fills in a real, verified-non-empty
decision scope automatically from the live policy/scheduling manifests (most recent fold, the
manifest's own `selected_model`/`primary_k_level`, the scheduling manifest's own default
config), so a first visit shows real recommendations immediately. It never overwrites a scope the
visitor already set, and the full manual form remains under "Advanced options."

**A new page for an existing API surface.** Component 16's `/v1/review/queue` and
`/v1/review/resolutions` endpoints (§16k) had no frontend before this pass. `HumanReviewPage`,
`api/review.ts`, and the `ReviewCaseOut`/`ResolutionLogRowOut` TypeScript types are new; the
endpoints themselves are unchanged. Still read-only — no resolution-submission form yet.

**An establishment journey.** `EstablishmentDetailPage` now renders one establishment's path
through all five layers (available information → prioritization → recommendation → schedule →
human review) as a five-step visual sequence in plain language, with the prior field-by-field
technical view preserved underneath.

**A real bug found and fixed, the same way as §16l — by the test suite catching a behavior gap,
not by inspection.** `useDefaultScope`'s first version called `setScopeField` up to six times in
one effect; only the last field ever actually landed in the URL, because react-router's
`setSearchParams` updater reads the currently *committed* search params at call time, not the
result of an unapplied sibling call from earlier in the same tick — repeated rapid calls do not
compose the way `useState`'s functional updater does. Pages got stuck on "Preparing an inspection
plan…" forever. Caught because tests asserting the page *eventually shows real data* (not merely
that a loading state appears) timed out. Fixed by adding a bulk `setScopeFields` to
`useDecisionScope` and using it instead of repeated single-field calls.

No backend file, ML model, Component 13/14/16 semantic, or API contract changed in this pass.
Frontend suite: 47 tests pass (up from 33), `tsc -b --noEmit` and `oxlint` clean. Backend
untouched and reconfirmed: 3,190 tests, ruff/mypy clean. Full detail:
`docs/analysis/frontend_product_clarity_20260828.md`.

---

## 16n. Actionability & operational workflow pass (2026-08-28)

§16m made the frontend readable. It was still read-only, and it still presented `is_selected` as
if it were a judgment about the establishment rather than a fact about this run's capacity cutoff.
A product-reality review named both gaps directly: the UI told a supervisor to "resolve this" or
"confirm what happened" with no control anywhere on the page that could do either.

**Four write forms, exactly the existing contracts, nothing invented.** `OverrideForm`,
`AdjustmentForm`, `ExecutionOutcomeForm` and `ResolutionForm` submit `POST /v1/policy/overrides`,
`/v1/schedule/adjustments`, `/v1/execution/events`, `/v1/review/resolutions` respectively — every
field the real `Override`/`Adjustment`/`ExecutionEvent`/`ReviewResolution` pydantic models require,
no more. `ExecutionOutcomeForm`'s status options are read live from `GET /v1/execution/contract`
rather than hardcoded, so if that table ever changes, the form does not silently disagree with it.
Every submission shows a receipt worded against ADR 0049 — "staged, not applied" — never implying
the visible plan updated.

**A four-endpoint read-filter addition, and a real documentation bug found while making it.** To
build a per-establishment "Decision history" panel, `target_inspection_id` filters were added to
the four log-read endpoints and a `trigger` filter (substring match on `trigger_reasons`) to
`/v1/review/queue` — the same pattern as the existing `establishment_id`/`schedule_status` filters,
not new business logic. While reading the four log services to add these, their `status` field
docstrings turned out to be wrong: `OverrideLogRowOut`, `ResolutionLogRowOut` and `ReviewCaseOut`
all documented a "pending"/`pending_review` presentation status that **was never implemented** —
`get_override_log`/`get_adjustment_log`/`get_execution_events`/`get_resolution_log` read only the
committed artifact and stamp `"committed"` unconditionally. This had been true since those modules
were written and nothing had exercised the gap until a frontend needed the pending view. Fixed by
correcting the docstrings and `docs/data_contracts/sentinel_api.md`, and by having
`DecisionHistory.tsx` do what the docstrings had wrongly claimed the backend already did: call
`GET /v1/staged-requests` separately and merge client-side.

**A second real bug, this one product-relevant rather than cosmetic.** With an action form
mounted, a background refetch — `useDefaultScope` filling in `schedule_config_id` slightly after
the establishment record itself loaded, exactly the same class of race as §16m's — cycled the
page's main query back through `'loading'`, which unmounted the entire journey (every form inside
it) for a refresh nobody asked for, discarding whatever a person had already typed. Found by a new
test clicking through an action form, not by inspection. Fixed with a stale-while-revalidate
pattern local to the two affected queries (the journey renders from the last successful payload,
which only moves forward, updated during render per React's own documented pattern rather than in
an effect) — not a hook-library change, since the bug was in how this one page used the existing
`useApiQuery`, not in the hook itself.

**Terminology, not cosmetics.** `is_selected` is now "Selected for this plan," with an explicit
statement that the same establishment can cross that line purely from a capacity change. The raw
score is no longer a list page's headline number — `relativePriorityLabel` (rank/percentile from
`model_rank`/`n_universe`, fixed *before* any capacity cutoff, per Component 13's own allocation
order) replaced it, and a single "How to use this priority" note states the project's own measured
ROC-AUC (~0.61-0.62, from Components 6-8's own numbers) once per page rather than not at all or on
every row. Human Review's one blended count is now two, using the new `trigger` filter — measured
against real production data, 2 decision-concern cases vs 28 missing-outcome cases for the default
scope, which is exactly the distinction a single "28 need review" number was destroying.

Backend: 6 new tests (`tests/api`: 82 → 88), ruff/mypy --strict clean, full 3,193-test suite green.
Frontend: 47 → 64 tests, `tsc -b --noEmit`/`oxlint` clean. Manually verified against the real
running API and production artifacts, not only mocks — staged and then removed a real override so
no test data was left in `data/staging/`. No routing, inspector assignment, authentication, or
live staffing feed exists or was implied; none of those have a backend contract to build against.

---

## 16o. Component 21 — supervisor plan review, adjustment, and approval

The bridge between Component 20's machine-generated geographic plan and a human operational
decision. Reads only Component 20's output (never Component 18/19 directly, so a decision can
never bypass geographic organization or capacity/policy selection). Lives in
`src/sentinel/plan_review/`; contract in `docs/data_contracts/plan_review.md`.

**Five distinct human-decision layers, kept disjoint by construction.** Component 13's
`OverrideAction` (who's in the queue), Component 14's `AdjustmentAction`/`ExecutionStatus`
(when/what happened), Component 16's `ReviewResolutionAction` (flagged historical case review),
and Component 21's own `PlanDecisionAction` (a supervisor decision about a live proposed
workload) each live on their own key, with an import-time guard checking no verb collides across
layers and that no value contains `"defer"` (reserved by Component 14's
`ScheduleStatus.DEFERRED`).

**Four decision verbs**, none of which can overwrite Sentinel's own recommendation: `keep_selected`,
`move_to_later_workday` (requires `revised_planned_date`), `adjust_operational_priority` (requires
`revised_operational_priority` — sets a display-only `operational_priority` field-work-order
column, computed as `coalesce(supervisor_revised_operational_priority, policy_rank)`; `rank` and
`policy_rank` are never touched, checked by the same byte-identical `IMMUTABLE_FIELDS` invariant
Component 20 established), and `do_not_proceed_as_planned`. "risk_rank = 7,
supervisor_operational_priority = 2" is a valid, intended state — Sentinel's own rank is never
made to say 2.

**Approval is a fifth, separate act, not a side effect of decision completeness.** A fully-decided
plan reaches `adjusted`, not `approved` — approval requires an explicit `PlanApprovalRequest`
(`approval_id`, `planning_date`, `approved_by`, `approved_at`), and it is refused outright, never
partially applied, if `plan_review/approval.py::check_readiness()`'s 5-point checklist (no
duplicate establishments, every row carries the machine recommendation, geographic provenance
present, every recorded decision has a reason, undecided rows default to the machine
recommendation) finds a blocking problem. The written `approved_operational_plan` artifact is
never rewritten in place, true the same way it's true for every other Sentinel artifact — a
supervisor amending the plan afterward produces a new `supervisor_plan_review` snapshot and, if
they choose, a new approval event, named by its own source checksum; the original stays untouched
and becomes a permanent record of exactly what was handed to Component 22.

**Two real integration bugs, both invisible to isolated unit tests, caught only by an end-to-end
smoke test against the running API.** `PLAN_APPROVAL_REQUIRED_FIELDS` (and the `_guard_registry()`
check enforcing it) named `source_plan_review_sha256` as a required field of the supervisor's
*request* — but that field only ever exists on `ApprovedPlanManifest`, computed independently from
the real plan-review file at commit time; `PlanApprovalRequest` never carries it. Every approval
request was refused with "field is blank," unconditionally, regardless of payload, until this was
fixed by narrowing `PLAN_APPROVAL_REQUIRED_FIELDS` to the fields the request actually has.
Separately, `StagingService._KIND_CONFIG` had no `"plan_approval"` entry at all — a request that
passed both schema and governance validation still crashed with a bare `KeyError` inside
`staging.append()` itself. Both were found only by literally POSTing to the running app via
`TestClient` with `create_app()` + `app.dependency_overrides[get_settings]`, not by running the
existing (and passing) unit test suite, which exercised each piece in isolation and never the seam
between them. New regression tests (`tests/api/test_plan_review_approval_api.py`) cover both paths
directly so this class of bug — the contract compiles, each function is separately correct, and
the wired-together request still 500s — cannot silently reappear.

Validated end to end against real data: a fresh 30-establishment plan review, a 3-action demo
decisions file (keep, defer-with-reason, adjust-priority-with-reason), approval (30 total, 29
active, 1 deferred, 27 undecided, all readiness checks READY), the blocked path (mismatched
`planning_date` refused with a clear error, nothing written), and determinism (two approvals of
byte-identical input produce byte-identical content outside the approval identity fields
themselves, which are new by construction each time).

Frontend: `PlanApprovalPanel` (approve-or-show-approved), `PlanDecisionForm` extended for the new
action, `operational_priority` shown beside `policy_rank` whenever they differ with an explicit
"machine rank unchanged" note. New `SupervisorPlanReviewPage.test.tsx` — the first test file this
page ever had.

**What it does not do**, stated the same way every other component states its refusals: no
retraining, no score/rank change, no recomputation of Component 19/20's output, no invented
staffing/routes/travel-time, and execution-gap review findings (Component 16) are never presented
as a plan-review decision concern — they are structurally a different queue.

## 16p. Final completion pass — product coherence audit (2026-09-05)

Not a new component. With C17-21 all built and individually correct, the gap left was product
coherence: two research agents (a full frontend page/copy survey, a backend/API/docs survey) found
`GeographicPlanPage`/`SupervisorPlanReviewPage` — the two newest, most operationally real pages —
structurally orphaned. Neither was linked from `OverviewPage`, `TodayPage`, or
`EstablishmentDetailPage`; neither showed which `planning_date` it was displaying; `WorkflowDiagram`
(shown on Overview) still described only the old 5-step backtest flow. Separately, seven raw
technical identifiers (a bare `establishment_id`, a raw API error code, a `review_id`, a
`work_block_id` fallback, two CLI command names, and one already-adequately-muted reference id)
leaked into primary, non-collapsed UI.

**Fixed, all additive, nothing rewritten:** `NavBar` reordered into landing → live-plan → analysis
groups; `OverviewPage` gained an unconditional "Today's field plan" section; `WorkflowDiagram`
extended 5→8 steps; both orphaned pages gained a plain "for {date}" header from data their own API
responses already return; `EstablishmentDetailPage` gained a 7th journey step linking generically
into both (no establishment-filtered deep link exists on the backend — noted as a real, explicitly
out-of-scope gap rather than worked around). All seven raw-ID leaks fixed via three new
`lib/copy.ts` translators (`apiErrorCodeLabel`, `workBlockDisplayLabel`,
`operationalCoverageNote`) plus small JSX edits — the raw codes are de-emphasized, never deleted,
so support/audit use is preserved.

**The one backend change:** `operational_selection`'s manifest already computed
`ranked_candidate_count`/`selectable_candidate_count`/`selected_count` — never reachable through
any API route (`meta_service._COMPONENTS` whitelisted only 4 of the pipeline's components). Added
one dict entry. Verified against real data: `GET /v1/manifests/operational_selection` returns
`35859 → 35859 → 30` for the 2026-08-28 plan, exactly matching the manifest on disk. No new
computation anywhere.

Also fixed: `frontend/README.md` was genuinely stale (still called Component 21 "unbuilt"),
contradicting the root `README.md`/`HANDOFF.md` — rewritten to match reality, including the two
plan-review write contracts it had never mentioned.

Two pages (`TodayPage`, `GeographicPlanPage`) had zero test coverage before this pass, and their
MSW mock handlers (`/v1/schedule/dates`, `/v1/plan-review/work-blocks`) didn't exist either — both
added. Frontend: 67 → **89 tests passing** (16 files, up from 11). Backend: +2 tests
(`test_manifests_runs.py`), full suite **3,384 passed, 3 deselected**. `ruff`/`mypy`/`tsc
--noEmit`/`oxlint` all clean throughout. Verified end to end against the real running API (not
mocks): Component 20's 22 real work blocks, Component 21's real approved plan, the new coverage
manifest's real counts, and one freshly staged plan decision — then cleaned every staged test
artifact created (plus two leftover from earlier manual testing) out of
`data/staging/plan_review/`, since nothing there was ever applied (ADR 0049).

Deliberately deferred, not silently dropped: `docs/data_contracts/candidates.md` /
`operational_scoring.md` / `operational_selection.md` still don't exist (every other implemented
component has one) — noted honestly in `STATUS.md`'s roadmap table rather than rushed or ignored.

## 16q. The "Today = April 1, 2026" bug (2026-09-05)

A real bug, found from screenshots, root-caused by reading code rather than assumed: the landing
page (`frontend/src/pages/TodayPage.tsx`, route `/`) was entirely wired to Side A (the historical
backtest pipeline, Components 1-14/16, fold-scoped) despite being named "Today." Exact mechanism:
`useDefaultScope` auto-picks the *last* fold in `frontend/src/api/folds.ts`'s hardcoded
`FOLD_TABLE` — `quarterly-2026Q2`, a simulated Apr-Jun 2026 window — and `TodayPage` called
Component 14's fold-scoped `listScheduleDates`, taking the *first* date in that fold's simulated
calendar: April 1, 2026. There was no "current date" concept anywhere in the frontend or backend
before this fix — "today" had simply never been connected to anything real.

**Fix, both parts required, neither alone would have been honest.** `TodayPage.tsx` was rebuilt to
read Side B (Components 17-21's live `planning_date` pipeline) instead — the same no-scope
`getPlanSummary`/`listPlanRows` calls `GeographicPlanPage`/`SupervisorPlanReviewPage` already make,
no new endpoint. New `frontend/src/lib/today.ts::currentOperationalDate()` computes the real
current date live (`new Date()`, never a hardcoded literal, call-time-evaluated so a long-lived tab
stays correct) and `lib/copy.ts::planLabelForToday` compares it against the plan's own
`planning_date`, saying "Today's inspection plan" only when they genuinely match, and honestly
"Plan for {date} — not today's plan yet" when they don't — this function never fabricates
agreement. Separately, since a UI fix alone would just be papering over stale data, the real CLI
pipeline was run once for the real current date (`plan-candidates` → `score-candidates` →
`select-inspections --capacity 30 --policy pure_risk` → `organize-geography` → `review-plan` →
`approve-plan`, all `--planning-date 2026-09-04`), producing genuine new artifacts — 35,859 real
candidates, 30 selected, 21 real geographic work areas, approved — from the same real Chicago data
and frozen production model every other run uses. The prior `2026-08-28` artifacts are untouched
(nothing is ever deleted or overwritten in this project) — "latest" simply now resolves to the new
ones.

The displaced Side-A day-view content wasn't deleted — moved verbatim to a new route
`/schedule/day` (`ScheduleDayPage.tsx`), explicitly labeled "Historical Day View," linked from
`SchedulePage`. Three small, explicitly-requested UX fixes rode along: `Area N` → `Work Area N`
(new `copy.ts::workAreaLabel`, purely a display wrapper — Component 20's own label is untouched);
`SupervisorPlanReviewPage`'s undecided-row text is now approval-state-aware; the ROC-AUC disclosure
on two pages, previously always-visible, is now collapsed behind "How Sentinel prioritizes
locations," matching this app's established progressive-disclosure pattern everywhere else.

**Deliberately not built**: no "generate today's plan" trigger anywhere in the API or frontend —
builds stay a CLI/operator action (ADR 0049); no operational-date picker for Side B — it never had
one, and adding prev/next-day navigation is a materially larger change this bug fix didn't need.
"Today always means the latest Side-B artifact, labeled honestly against the real current date" is
the whole design, and it is now structurally guaranteed, not just tested: `TodayPage.tsx` contains
zero references to `useDefaultScope`, `FOLD_TABLE`, `useDecisionScope`, or `listScheduleDates` —
confirmed by grep, not assumed. Frontend: 89 → **110 tests passing** (18 files). No backend code
changed; verified via the real running API that `/v1/plan-review/summary`, `/work-blocks`, and
`/rows` all agree on `planning_date=2026-09-04` — exactly what Today, Field Plan, and Plan Review
each independently read.

## 16r. Fixed broken establishment navigation from the live plan (2026-09-05)

Found immediately after §16q's date fix, same root cause pattern (Side A/Side B conflation), this
time breaking navigation instead of a date. `TodayPage.tsx` and `GeographicPlanPage.tsx` (Side B,
grep-confirmed the complete list of offenders) both linked into `EstablishmentDetailPage.tsx`'s
route (`/establishments/:establishmentId`), whose backend call
(`get_establishment_history` in `src/sentinel/api/services/establishment_service.py`, read
directly) filters **Component 13's fold-scoped historical recommendation table** — a population
with no necessary relationship to a live-`planning_date` establishment at all. Whichever fold
`useDefaultScope` happened to auto-select, the clicked establishment predictably wasn't in it.

Fixed by giving Side B its own detail page, **not** by forcing establishments through the
Side-A-only lookup: `frontend/src/pages/EstablishmentPlanDetailPage.tsx`
(`/plan/establishments/:targetInspectionId`), built entirely from
`GET /v1/plan-review/rows/{target_inspection_id}` — an endpoint that already existed, was already
tested, and the frontend had simply never called. No new backend code. Worded the priority number
honestly as `#{rank} in today's plan` rather than reusing Side-A's `relativePriorityLabel` (real
data confirmed `rank`/`policy_rank` are identical, 1..30 — the plan's own rank, not the
35,859-candidate universe; the Side-A label's denominator semantics would have silently
misrepresented it). States plainly, rather than hiding, that deeper historical/SHAP detail isn't
available for live operational establishments (Component 18 doesn't run explainability).

Verified against real data across six distinct establishments (first/last/middle-ranked, grouped,
singleton, missing-location) — all 200, confirmed via the real running API, not just the first
record that happened to work. Frontend: 110 → **118 tests passing** (19 files). No backend files
touched; full regression suite re-run to confirm.

## 16s. Final acceptance audit (2026-09-05) — Sentinel declared complete

A strict, no-new-features audit against real data end to end. Found and fixed two more genuine
gaps, both root-caused:

**"Needs Attention" merged two semantically different concerns into one list.**
`HumanReviewPage.tsx` computed policy-warning and missing-outcome cases into one de-duplicated
list with an inline tag distinguishing them — real, but not "visually separate," and a
both-reasons case read as one concatenated line. Split into two headed sections with their own
counts and honesty statements; a dual-reason case now genuinely appears in both, not merged.

**A decision submitted after approval gave no warning it wouldn't retroactively change the
approved plan.** `PlanDecisionForm` gained an optional `planAlreadyApproved` prop, shown on both
`SupervisorPlanReviewPage` and `EstablishmentPlanDetailPage` (which also gained its own plan-level
approval-status read, closing a real cross-page consistency gap it previously had none of). The
backend semantics were already correct and documented; this closes the gap between that intent and
what the UI said.

**A real, previously-uncaught code-quality gap in `geographic_organization/organization.py`**,
found only because this audit ran `ruff`/`mypy` over the whole `src/sentinel/` tree instead of
only touched files: a closure-over-loop-variable pattern (verified behavior-preserving by tracing
the control flow — the closure was consumed synchronously within the same iteration it was
defined in, so this was a static-analysis false positive, not a live bug — and confirmed by
re-running its 20-test suite unchanged) plus two missing-generic-type-argument spots. Fixed with
the standard default-argument closure-binding idiom and precise typing. 20 of 25 formatting-drifted
files across Components 17-21 and two API files were reformatted (the 5 Component-9 calibration
files remain deliberately unformatted, per this file's own pre-existing, documented exception).

**Verified end to end with a real decision + re-approval cycle**, not just reads: committed a real
`adjust_operational_priority` decision via `sentinel review-plan --decisions` against the live
2026-09-04 plan — `rank`/`policy_rank` stayed byte-identical (6/6), `operational_priority` became
1, a full audit record appeared in `plan_decision_log` — then re-ran `sentinel approve-plan` and
confirmed every Side-B API endpoint (and therefore every Side-B page) agrees on the new state.
Re-verified establishment navigation across 7 categories including the new real decision-bearing
establishment — all 200.

Frontend: 118 → **120 passing**, `tsc --noEmit`/`oxlint` clean. Backend: **3,384 passed, 3
deselected**, re-run twice (before and after the `geographic_organization` fixes), identical count
both times; `ruff check`/`ruff format --check`/`mypy` now clean on every file this session touched.
No staged test artifacts left behind — this pass used the real CLI directly, never staging.

**Sentinel is COMPLETE for its current scope.** See the chat transcript for the full 15-section
acceptance audit report (cross-page consistency table, the 30/28/21 number-by-number breakdown,
date/approval-state audits, and the final verdict).

## 17. Reminder

**One component at a time.** Component 14 was the operational schedule only, and is closed. No
OR-Tools routing (Component 15, and blocked), no inspector assignment, no LangGraph — and no
score adjustment and no re-ranking, which ADR 0037 and ADR 0045 forbid and which every manifest
records in its blocked list so the boundary travels with the artifact. **Correction: this
paragraph used to say Component 21 was "still unbuilt."** That was true when §16j's minimal
read-only `frontend/` was first added alongside the Sentinel API; it is not true any longer —
Component 21 (§16o) is implemented, tested, and documented, including its own frontend pages.
Component 22 (outcome logging) remains unbuilt and out of scope for anything done so far.

Component 13 did settle model selection, which Components 11 and 12 were forbidden from doing —
and it is worth being precise about what that means. It settled it as an **operating choice from
a rule fixed in advance**, and it recorded that the four candidates are statistically
indistinguishable on the metric the question was really about. The scientific question is not
answered; it may not be answerable on this data.

And the rule that has now paid off six times: **investigate → document → design →
implement → test.** The findings document is written before the implementation, from a
read-only profiling script. In Component 6 that discipline caught the `add_indicator`
collinearity, the boolean median drift and the row-order sensitivity. In Component 7 it
caught the 0.11 row-order sensitivity — three orders of magnitude worse than Component 6's
and invisible from reading code — and it produced the calibration probe that let a
pre-registered expectation be *disproved* rather than quietly dropped.

One addition Component 7 earned: **when a leakage test fails, suspect the test first.**
Both failures in that component were test bugs that looked exactly like leaks. Component 13 hit
the same thing three times: every early red test was a fixture that had not thought hard enough
about `floor(share * k)`.

And one Component 13 earned: **profile the premise, not only the parameters.** The profiler was
built to fix constants — the eligibility column, the reserve sizes — and what it actually caught
was that the intervention the whole component was scoped around was aimed at a problem that does
not exist in this data. A component that had gone straight to implementation would have shipped
a working, well-tested, thoroughly documented coverage reserve and never learned that the risk
queue already over-serves the population fourfold.
