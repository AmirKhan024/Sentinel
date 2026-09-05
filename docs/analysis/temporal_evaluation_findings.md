# Temporal evaluation: empirical findings

Every number here was measured on the full snapshot. Nothing is estimated,
illustrative or carried over from the project specification. Where the data
contradicts an expectation, the data wins and the divergence is recorded.

**Source:** `data/processed/features/as_of_features_20260816T150313Z.parquet`
(sha256 `b7db5b2d…25e2f`)
**Run:** `evaluation_folds_20260816T164834Z.parquet` and siblings
**Profiling:** `uv run python scripts/profile_evaluation.py`

---

## 1. The estimand, stated before the results

Component 5 measures the **re-ordering** of canvass inspections that actually
occurred. The set of establishments, the number of inspections and the labels are
all held at their observed values; only the order changes.

There are no labels for establishments nobody inspected, so coverage cannot be
evaluated — only ordering. Nothing here is causal, and nothing licenses a claim
about illness: Sentinel observes violations *cited*.

---

## 2. The snapshot and the inputs

| measurement | value |
|---|---|
| feature rows | 57,727 |
| distinct establishments | 15,144 |
| distinct inspection dates | 1,991 |
| date range | 2018-07-03 → 2026-08-14 |
| overall positive rate | 0.5252 |

**Positive rate by year — the drift that shapes every decision below:**

| year | rows | positive rate |
|---|---|---|
| 2018 | 2,829 | 0.8756 |
| 2019 | 8,230 | 0.7735 |
| 2020 | 6,133 | 0.5937 |
| 2021 | 6,154 | 0.5029 |
| 2022 | 6,552 | 0.4654 |
| 2023 | 7,196 | 0.4605 |
| 2024 | 8,481 | 0.4260 |
| 2025 | 8,269 | 0.3921 |
| 2026 | 3,883 | 0.3912 |

A 48-percentage-point fall over eight years. Any evaluation pooling across time
measures this rather than the model, which is why every result below is reported
per fold beside its own prevalence, and why the headline metrics are the two that
are base-rate invariant.

**A note on the base rate itself.** The project specification anticipated 15–25%.
The measured rate is 52.52%. The specification's figure descends from the 2015
model's 14.1%, which counted the *old* code's critical violations 1–14; Priority
plus Priority Foundation under the 2018 code is a much broader class. The data
wins. One consequence matters here: a PR-AUC of 0.50 on this data is roughly the
*no-skill floor*, not a good result.

---

## 3. Inspection capacity — where `k` comes from

Measured, not chosen. 57,727 inspections across 1,991 days:

| statistic | inspections per day |
|---|---|
| mean | 28.99 |
| median | **29** |
| p25 | 21 |
| p75 | 37 |
| p90 | 45 |
| min / max | 1 / 68 |

Effectively weekday-only: 19 Saturday rows and 1 Sunday row in eight years.

Per year the median runs 23 (2020) to 36 (2024), and per test window the measured
median ranges **22 to 45**. Each fold's `k` is derived from its own median rather
than from a global constant.

---

## 4. Fold construction

17 quarterly folds plus 1 distribution-shift fold. The count is derived from the
data: a test quarter is emitted only when the snapshot covers it entirely.

**2026Q3 is excluded** — it exists as 328 rows over 32 days, and comparing a
two-thirds window against full ones would be a silent truncation. The exclusion
is named in the manifest.

| fold_id | train end | calibration | test | train rows | cal rows | test rows | train rate | cal rate | test rate | median capacity |
|---|---|---|---|---|---|---|---|---|---|---|
| quarterly-2022Q2 | 2021-12-31 | 2022Q1 | 2022Q2 | 23,346 | 1,357 | 1,762 | 0.667 | 0.431 | 0.466 | 29 |
| quarterly-2022Q3 | 2022-03-31 | 2022Q2 | 2022Q3 | 24,703 | 1,762 | 1,733 | 0.654 | 0.466 | 0.492 | 28 |
| quarterly-2022Q4 | 2022-06-30 | 2022Q3 | 2022Q4 | 26,465 | 1,733 | 1,700 | 0.642 | 0.492 | 0.465 | 30 |
| quarterly-2023Q1 | 2022-09-30 | 2022Q4 | 2023Q1 | 28,198 | 1,700 | 1,801 | 0.633 | 0.465 | 0.445 | 31 |
| quarterly-2023Q2 | 2022-12-31 | 2023Q1 | 2023Q2 | 29,898 | 1,801 | 1,787 | 0.623 | 0.445 | 0.485 | 29 |
| quarterly-2023Q3 | 2023-03-31 | 2023Q2 | 2023Q3 | 31,699 | 1,787 | 1,650 | 0.613 | 0.485 | 0.484 | 26 |
| quarterly-2023Q4 | 2023-06-30 | 2023Q3 | 2023Q4 | 33,486 | 1,650 | 1,958 | 0.606 | 0.484 | 0.433 | 33.5 |
| quarterly-2024Q1 | 2023-09-30 | 2023Q4 | 2024Q1 | 35,136 | 1,958 | 1,913 | 0.600 | 0.433 | 0.458 | 34 |
| quarterly-2024Q2 | 2023-12-31 | 2024Q1 | 2024Q2 | 37,094 | 1,913 | 2,196 | 0.592 | 0.458 | 0.449 | 38 |
| quarterly-2024Q3 | 2024-03-31 | 2024Q2 | 2024Q3 | 39,007 | 2,196 | 2,124 | 0.585 | 0.449 | 0.422 | 33.5 |
| quarterly-2024Q4 | 2024-06-30 | 2024Q3 | 2024Q4 | 41,203 | 2,124 | 2,248 | 0.578 | 0.422 | 0.379 | 40 |
| quarterly-2025Q1 | 2024-09-30 | 2024Q4 | 2025Q1 | 43,327 | 2,248 | 2,459 | 0.570 | 0.379 | 0.381 | 45 |
| quarterly-2025Q2 | 2024-12-31 | 2025Q1 | 2025Q2 | 45,575 | 2,459 | 2,290 | 0.561 | 0.381 | 0.390 | 39 |
| quarterly-2025Q3 | 2025-03-31 | 2025Q2 | 2025Q3 | 48,034 | 2,290 | 1,754 | 0.552 | 0.390 | 0.423 | 28 |
| quarterly-2025Q4 | 2025-06-30 | 2025Q3 | 2025Q4 | 50,324 | 1,754 | 1,766 | 0.544 | 0.423 | 0.380 | 30 |
| quarterly-2026Q1 | 2025-09-30 | 2025Q4 | 2026Q1 | 52,078 | 1,766 | 1,917 | 0.540 | 0.380 | 0.391 | 35 |
| quarterly-2026Q2 | 2025-12-31 | 2026Q1 | 2026Q2 | 53,844 | 1,917 | 1,638 | 0.535 | 0.391 | 0.379 | 28 |
| **covid_shift** | 2020-02-29 | 2020-03…05 | 2020-06…2021-12 | 12,660 | 1,846 | 8,840 | 0.773 | 0.683 | 0.513 | 22 |

Training grows from 23,346 to 53,844 rows. The training base rate falls
monotonically from 0.667 to 0.535 as the expanding window absorbs later years —
a direct consequence of the expanding-window choice, visible rather than hidden.

**Establishments recur once per quarter.** Of 57,727 establishment-quarter
groups, 57,321 hold a single row. A test window is close to a set of distinct
premises, which is the right unit for a ranking.

---

## 5. Baseline results — the headline table

17 quarterly folds, mean ± SD. Higher NDE is better; 1 is perfect, 0 is random,
−1 is worst-possible.

| schedule | model | NDE | mean days earlier | first-half discovery |
|---|---|---|---|---|
| optimal | — | **1.0000** ± 0.0000 | +24.75 | 1.0000 |
| model | `prior_canvass_priority_rate` | **0.1845** ± 0.0404 | **+4.47** | 0.5764 |
| model | `priority_at_last_canvass` | 0.1522 ± 0.0384 | +3.68 | 0.5802 |
| model | `days_since_last_canvass` | 0.0765 ± 0.0384 | +1.76 | 0.5365 |
| model | `random` | 0.0101 ± 0.0263 | +0.09 | 0.5046 |
| **business_as_usual** | — | **0.0066** ± 0.0422 | 0.00 | 0.5038 |
| model | `constant` | 0.0065 ± 0.0425 | −0.01 | 0.5037 |
| random | — (340 fold-seeds) | −0.0016 ± 0.0271 | −0.19 | 0.4994 |
| worst | — | **−1.0000** ± 0.0000 | −25.39 | 0.0000 |

Three things in this table are worth stating out loud.

**The analytic bounds land exactly.** Optimal is 1.0000 and worst is −1.0000 with
zero variance across every fold, and the 340 random fold-seeds average −0.0016.
That is the strongest available evidence that the area formula and its analytic
denominator are right.

**Business-as-usual sits essentially at random.** NDE 0.0066 ± 0.0422, ROC-AUC
0.504, first-half discovery 0.5038 against a 0.5 null. Within a quarter, the
city's actual inspection order carries almost no risk information. This is a
finding, not a defect: the order inside a quarter is driven by geography, routing
and statutory cycle, none of which tracks which establishment is about to fail.
It also means the bar Component 6 has to clear is *lower than expected* — and
that the honest comparator is nonetheless the one that matters.

**`constant` reproduces business-as-usual almost exactly** (0.0065 vs 0.0066).
With every score tied, the ordering falls entirely to `target_inspection_id`,
which increases roughly with time — so the tie-break reconstructs approximate
date order. A useful accident: it confirms the tie-break path is exercised and
behaves as documented.

### Statistical metrics, same folds

| metric | best baseline | BAU | random | constant |
|---|---|---|---|---|
| ROC-AUC | 0.5915 ± 0.0202 | 0.5040 ± 0.0212 | 0.5051 ± 0.0132 | **0.5000** ± 0.0000 |
| PR-AUC | 0.5012 ± 0.0406 | 0.4347 ± 0.0418 | 0.4376 ± 0.0396 | 0.4307 ± 0.0410 |

`constant` scores ROC-AUC exactly 0.5000 with zero variance, which is the
no-information result and another confirmation the tie handling is correct.

**PR-AUC must be read beside prevalence.** Mean test prevalence is about 0.43, so
the best baseline's 0.5012 is a modest gain over the floor, not the strong result
the number looks like in isolation.

### Lift@k across capacities

`lift@k = precision@k / base rate`; 1.0 means no better than blind selection.

| k | rows | `prior_canvass_priority_rate` | `priority_at_last_canvass` | `days_since_last_canvass` | BAU |
|---|---|---|---|---|---|
| `k_pct_01` | 19 | 1.259 | 1.310 | 1.121 | 1.028 |
| `k_1_day` | 33 | **1.300** | 1.263 | 1.090 | 1.008 |
| `k_pct_05` | 96 | **1.319** | 1.207 | 1.146 | 1.004 |
| `k_1_week` | 164 | 1.298 | 1.191 | 1.137 | 1.005 |
| `k_pct_10` | 192 | 1.297 | 1.178 | 1.129 | 1.005 |
| `k_pct_25` | 481 | 1.254 | 1.183 | 1.094 | 0.998 |
| `k_1_month` | 687 | 1.205 | 1.183 | 1.183 | 1.003 |
| `k_pct_50` | 962 | 1.153 | 1.161 | 1.073 | 1.008 |

Lift decays as *k* grows, exactly as it should — the top of a ranking is where
the information is. At one day of real capacity, ranking by prior Priority rate
would put **30% more genuine citations** in the first 33 inspections than blind
selection, while the historical order delivers 0.8%.

---

## 6. Days-earlier detection — the distribution, not the mean

This is where the honest reporting matters most. Means across 17 folds:

| model | mean | median | SD | p25 | p75 | improved | unchanged | **worse** |
|---|---|---|---|---|---|---|---|---|
| optimal | +24.75 | +23.71 | 14.95 | — | +37.96 | 98.1% | 1.9% | 0.0% |
| `prior_canvass_priority_rate` | **+4.47** | +4.88 | **32.60** | — | +26.82 | 55.4% | 1.7% | **42.9%** |
| `priority_at_last_canvass` | +3.68 | +3.09 | 27.36 | — | +25.75 | 51.9% | 2.0% | 46.1% |
| `days_since_last_canvass` | +1.76 | +2.18 | 36.87 | — | +28.87 | 51.9% | 1.9% | 46.2% |
| business_as_usual | 0.00 | 0.00 | 0.00 | 0 | 0 | 0% | 100% | 0% |
| worst | −25.39 | −26.65 | 14.35 | — | −12.18 | 0.0% | 1.6% | 98.4% |

**The standard deviation is 7.3× the mean, and 42.9% of positives would have been
found *later*.** Reporting "+4.47 days earlier" alone would be misleading in
exactly the way the 2015 Chicago result was: it reported 7.438 days with SD
25.156 and never said how many establishments were found later.

The mechanism is unavoidable. A reordering under fixed capacity is zero-sum in
slots: moving one establishment earlier moves another later. A schedule can only
win by moving *positives* earlier and *negatives* later, and a weak ranking moves
plenty of both in the wrong direction. The mean is the net effect; the fractions
say how it was achieved.

### Per-fold stability of the best baseline

| fold | slots | positives | NDE | mean days | SD | worse | first half |
|---|---|---|---|---|---|---|---|
| 2022Q2 | 1,762 | 821 | 0.1399 | +4.33 | 31.96 | 41.4% | 0.560 |
| 2022Q3 | 1,733 | 852 | 0.1852 | +3.03 | 32.58 | 41.8% | 0.560 |
| 2022Q4 | 1,700 | 791 | 0.1538 | +3.40 | 28.98 | 44.4% | 0.550 |
| 2023Q1 | 1,801 | 802 | 0.1193 | +3.79 | 33.27 | 39.7% | 0.531 |
| 2023Q2 | 1,787 | 866 | 0.1590 | +3.97 | 33.57 | 45.7% | 0.554 |
| 2023Q3 | 1,650 | 798 | **0.2619** | +5.56 | 30.47 | 41.2% | 0.602 |
| 2023Q4 | 1,958 | 848 | **0.1273** | +1.87 | 31.42 | 45.6% | 0.551 |
| 2024Q1 | 1,913 | 877 | 0.1973 | +4.77 | 34.54 | 43.1% | 0.588 |
| 2024Q2 | 2,196 | 987 | 0.1697 | +4.49 | 32.93 | 40.2% | 0.558 |
| 2024Q3 | 2,124 | 896 | 0.1986 | +3.55 | 32.71 | 44.4% | 0.593 |
| 2024Q4 | 2,248 | 853 | 0.1652 | +3.16 | 32.58 | 45.5% | 0.567 |
| 2025Q1 | 2,459 | 936 | 0.2375 | +7.16 | 35.36 | 41.2% | 0.600 |
| 2025Q2 | 2,290 | 893 | 0.2313 | **+8.68** | 34.02 | 38.7% | 0.597 |
| 2025Q3 | 1,754 | 742 | 0.2165 | +5.35 | 34.26 | 41.5% | 0.598 |
| 2025Q4 | 1,766 | 671 | 0.2146 | +3.68 | 31.12 | 43.4% | 0.615 |
| 2026Q1 | 1,917 | 750 | 0.2023 | +4.97 | 32.82 | 43.2% | 0.603 |
| 2026Q2 | 1,638 | 621 | 0.1567 | +4.24 | 31.54 | 47.8% | 0.570 |

NDE spans 0.119 to 0.262 — a factor of 2.2 between the weakest and strongest
quarter. Mean days-earlier spans +1.87 to +8.68. **A single split would have
reported anything in that range and called it the result.** That is the argument
for the rolling backtest in one table.

There is a mild upward trend from 2022 to 2025. It is not evidence the ranking is
improving: prevalence falls over the same period, and more training data is
available to later folds. Both explanations are visible in §4's table, and
Component 5 deliberately does not collapse them.

### First-half discovery

Best baseline 57.6% against business-as-usual 50.4%. For orientation only, the
2015 Chicago programme reported 69% vs 55% — under a different food code, a
different target and a 14.1% base rate. **Those are not a benchmark this project
can be measured against**, and no claim of comparability is made.

---

## 7. Time-invariance — audit Finding 2, measured

De-trended month-of-year effect on citation rate, fitted on all rows, expressed
as a percentage-point shift applied at the overall base rate.

| month | rows | raw rate | log-odds effect | pp shift at base |
|---|---|---|---|---|
| Jan | 5,435 | 0.4984 | −0.0191 | −0.48 |
| Feb | 4,532 | 0.5079 | +0.0275 | +0.69 |
| Mar | 5,374 | 0.5060 | +0.0141 | +0.35 |
| Apr | 5,576 | 0.5122 | +0.0540 | +1.35 |
| May | 5,392 | 0.5187 | +0.0919 | +2.28 |
| Jun | 4,637 | 0.5614 | +0.2184 | +5.40 |
| Jul | 4,189 | 0.5629 | +0.1263 | +3.14 |
| **Aug** | 4,503 | 0.5814 | +0.2581 | **+6.36** |
| Sep | 4,492 | 0.5171 | −0.1461 | −3.65 |
| Oct | 4,856 | 0.5305 | −0.1268 | −3.17 |
| Nov | 4,543 | 0.5188 | −0.1989 | −4.97 |
| **Dec** | 4,198 | 0.4998 | −0.2166 | **−5.41** |

**Peak-to-trough amplitude: 11.77 percentage points**, peaking in August and
troughing in December.

**Time invariance does not hold on this data.** An establishment inspected in
August is materially more likely to be cited than the same establishment
inspected in December, so a schedule that moves inspections across the calendar
is also moving them across a real change in citation probability. The audit's
Finding 2 is confirmed, on eight years of post-2018 data, on a target the 2015
model never used.

The shape is consistent with the temperature hypothesis — the summer peak is
where temperature-related violations would be expected — but **this cannot be
attributed to temperature.** No weather data is ingested, so the measured effect
is a proxy confounding temperature with daylight, holiday scheduling and summer
staffing patterns. Separating them requires NOAA GHCN station USW00094846, which
is a Component 1 extension.

> **BLOCKED — temperature covariate.** Needs a Component 1 ingestion extension.
> The sensitivity interface accepts the covariate the moment it exists.

### The sensitivity band

1,000 label re-draws per fold under the de-trended seasonal model, fitted on each
fold's **training window only**, using a monotone coupling that leaves a row's
label unchanged when the schedule leaves it on its own date.

| model | observed NDE | re-drawn mean | p05 | p95 | label flip rate |
|---|---|---|---|---|---|
| `prior_canvass_priority_rate` | 0.1845 | 0.1823 | 0.1720 | 0.1922 | 1.71% |
| `priority_at_last_canvass` | 0.1522 | 0.1514 | 0.1439 | 0.1586 | 1.84% |
| `days_since_last_canvass` | 0.0765 | 0.0813 | 0.0701 | 0.0919 | 2.11% |

**The headline survives.** Re-drawing labels under the measured seasonality moves
the best baseline's NDE from 0.1845 to a distribution centred on 0.1823 with a
90% interval of [0.172, 0.192] — narrower than the fold-to-fold SD of 0.0404. So
seasonality is real and large in the *base rate*, but the ranking's advantage is
not an artifact of it.

Between 1.7% and 2.1% of labels flip per replication. Business-as-usual's flip
rate is **exactly zero** by construction, which is what keeps the comparison fair.

---

## 8. Distribution shift — the COVID fold

Train 2018-07-01…2020-02-29 (12,660 rows, 0.773 positive), calibrate on
2020 Q1's tail (1,846 rows, 0.683), test 2020-06-01…2021-12-31 (8,840 rows,
4,532 positives, 0.513).

| schedule | model | NDE | mean days earlier | first half |
|---|---|---|---|---|
| optimal | — | 1.0000 | +120.50 | 0.975 |
| model | `days_since_last_canvass` | **0.1704** | +14.18 | 0.560 |
| model | `prior_canvass_priority_rate` | 0.1357 | +8.82 | 0.543 |
| model | `priority_at_last_canvass` | 0.1351 | +8.07 | 0.536 |
| business_as_usual | — | 0.0605 | 0.00 | 0.515 |
| model | `constant` | 0.0602 | −0.05 | 0.514 |
| random | — | 0.0035 | −8.04 | 0.497 |
| worst | — | −1.0000 | −138.68 | 0.025 |

Two observations, both worth carrying into Component 6.

**The ordering of the baselines inverts.** `days_since_last_canvass` is the
*weakest* baseline on the quarterly folds (NDE 0.077) and the *strongest* here
(0.170). During the 2020 suspension and restart, elapsed time since the last
canvass became far more informative than violation history — which is exactly
what one would expect when the schedule itself breaks down. A model selected on
the rolling folds would not have been the best model for this period.

**Business-as-usual does better here** (0.0605 vs 0.0066 on the quarterly folds),
because the test window is 19 months rather than 3, and over that span the city's
statutory cycle does carry ordering information that is invisible inside a single
quarter. The window length, not the schedule quality, is doing that work — which
is a caution against reading NDE across fold sets of different length.

This is **one fold**, so it carries no variance estimate. Read it as an
illustration, not a measurement. A full distribution-shift study needs a fitted
model and belongs to Component 6 onward.

---

## 9. Discovery curves

Stored at full resolution — 373,986 points, 868 KB — rather than summarized. The
curve *is* the result, and a reader who cannot redraw it has to take the headline
on trust.

`discovery_curves_<stamp>.parquet` carries one row per
`(fold, schedule, model, seed, slot_index)` with `slot_fraction` on the x-axis and
`cumulative_positive_fraction` on the y-axis, both normalized to [0,1] so folds of
different size and prevalence overlay directly. Curves are written for the three
deterministic reference schedules, one random seed (42), and each score producer.

---

## 10. What was blocked, and why

| experiment | status | reason |
|---|---|---|
| temperature in the time-invariance model | **BLOCKED** | no weather data ingested; needs NOAA GHCN USW00094846, a Component 1 extension |
| CDPH 2015 replication baseline | **DEFERRED to Component 6** | it is a fitted logistic regression; and several inputs (311, crime, licence, weather, inspector) are not in the data contract at all |
| duplicate-record sensitivity | **NOT APPLICABLE at Component 5** | Component 1 owns raw treatment, Component 3 already collapses same-day canvasses (530 days); there is no Component 5 knob to vary |
| feature-family ablation | **READY, awaiting Component 6** | the harness accepts any scored prediction set; an ablation is a set of models, which Component 6 produces |

Which of the 2015 model's inputs Sentinel could reconstruct today:

| 2015 input family | status here |
|---|---|
| prior violation history | available — 12 Component 4 features |
| time since last inspection | available — `days_since_last_canvass` |
| business age / tenure | available — `days_since_first_inspection` |
| inspector identity | **deliberately excluded** — audit Finding 1 |
| nearby 311 complaints | not ingested |
| burglary intensity | not ingested |
| alcohol / tobacco licence | not ingested |
| weather / temperature | not ingested |
| facility type, CDPH risk category | in raw, not in the feature table |

So a faithful replication is not reachable from the current data contract, and
whatever Component 6 builds must be labelled an approximation with this table
attached.

---

## 11. Full-data run results

| measurement | value |
|---|---|
| feature rows read | 57,727 |
| folds | 18 (17 quarterly + 1 covid_shift) |
| test range | 2022-04-01 → 2026-06-30 |
| excluded partial window | 2026Q3 (snapshot ends 2026-08-14) |
| score producers | 6 |
| metric rows | 2,808 |
| curve rows | 373,986 |
| simulation rows | 504 |
| sensitivity rows | 54 |
| random seeds | 20 (42…61) |
| sensitivity replications | 1,000, seed 20260816 |
| **runtime** | **164.2 s** |
| error-severity checks | **14 / 14 pass** |

Artifacts, 962 KB total:

```text
     7,899  evaluation_folds_20260816T164834Z.parquet
    21,520  evaluation_metrics_20260816T164834Z.parquet
   867,632  discovery_curves_20260816T164834Z.parquet
    41,072  simulation_summary_20260816T164834Z.parquet
     6,374  seasonality_20260816T164834Z.parquet
     7,712  sensitivity_20260816T164834Z.parquet
    10,537  manifest_evaluation_folds_20260816T164834Z.json
```

---

## 12. Validation

All 14 error-severity checks pass on the full run:

```text
fold_boundaries_are_strict          calibration_sits_between
folds_advance_monotonically         training_window_expands
future_rows_never_enter_training    test_is_isolated
no_split_overlap                    test_windows_are_complete
capacity_is_conserved               business_as_usual_is_real
score_direction_is_descending       scores_respect_the_decision_point
predictions_cover_test_exactly      labels_are_read_not_redefined
```

Following Component 4's precedent, the important ones **re-derive** their answer
from the data rather than reading back what the orchestrator reported. A check
that trusts the thing it is checking only proves the code agrees with itself.

Two are worth singling out. `business_as_usual_is_real` rebuilds every fold's
window and asserts the business-as-usual schedule reproduces the observed dates
exactly — without it, "days earlier" would silently stop meaning "earlier than
what really happened". `capacity_is_conserved` asserts every schedule consumes
exactly the observed slots, so no schedule can win by inspecting more.

---

## 13. Determinism

The only randomness is explicitly seeded and recorded in the manifest:
20 schedule seeds and one sensitivity seed. Everything else is structural — fold
boundaries are calendar arithmetic, every ordering is fully specified down to the
tie-break, and each table is sorted by a declared key before writing.

Two runs over the same input produce identical tables, and shuffling the input
rows changes nothing. Both are asserted by tests rather than assumed.

---

## 14. Limitations

1. **Re-ordering only.** No labels exist for establishments nobody inspected, so
   counterfactual coverage cannot be evaluated.
2. **Not causal.** Nothing here shows a different order would have *caused* a
   different outcome. No claim about illness is licensed — Sentinel observes
   violations cited.
3. **Time invariance does not hold** — 11.77 pp de-trended seasonal swing. The
   sensitivity band bounds the impact; it does not remove the assumption.
4. **The temperature attribution is BLOCKED.** The seasonal effect confounds
   temperature with daylight, holidays and staffing.
5. **Intra-day order is unrecoverable**, so business-as-usual has no within-day
   resolution.
6. **`days_since_last_canvass` is not statutory overdue** — the CDPH risk
   category is not in the feature table and Component 5 may not add features.
7. **No fitted baseline yet.** The CDPH replication is Component 6's, and several
   of its original inputs are not in the data contract at all.
8. **17 folds is not many**, and an unusual quarter moves the mean visibly.
9. **The covid_shift fold is a single fold** with no variance estimate.
10. **Folds are not independent samples.** The same premises appears in many test
    windows across eight years, so the reported SD is a fold-to-fold spread, not
    a confidence interval.
11. **Calibration windows are unused** until Component 9 exists.
12. **The baselines are weak by design.** They exist to establish a floor, and
    the fact that the best of them reaches NDE 0.18 says more about how little
    ordering information the historical schedule carries than about how good a
    ranking can get.
