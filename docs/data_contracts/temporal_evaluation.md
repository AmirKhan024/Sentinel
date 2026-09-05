# Data contract: temporal evaluation (Component 5 output)

**Produced by:** `sentinel evaluate` (`src/sentinel/evaluation/`)
**Layer:** `data/processed/evaluation/`
**Consumed by:** Component 6+ (models), Component 9 (calibration), Component 21 (demo)
**Design rationale:** `docs/analysis/temporal_evaluation_findings.md`, ADR 0012, ADR 0013

---

## 1. The estimand — read this first

Component 5 measures the **re-ordering** of canvass inspections that actually
occurred inside a historical test window.

> If the same historically observed inspection opportunities had been ordered as
> Sentinel suggests, how much sooner would the known positive outcomes have
> appeared?

Three things are held at their observed values and never varied: the set of
establishments inspected, the number of inspections available, and the labels.
Only the order changes.

**What this cannot say.** There are no labels for establishments nobody
inspected, so a schedule that would have visited a premises the city skipped
cannot be evaluated. Coverage is not an estimand here; ordering is.

**This is not causal.** Nothing in this component establishes that a different
order would have caused a different outcome, and nothing in it licenses a claim
about foodborne illness. Sentinel observes violations *cited*, not illnesses
prevented, and the literature genuinely disagrees about how tightly the two are
coupled.

The correct form of a result sentence is:

> In historical retrodictive simulation over 17 quarterly folds, ranking by
> `prior_canvass_priority_rate` would have re-ordered the historically observed
> inspection opportunities and changed the timing of observed violation discovery
> by a mean of +4.47 days (SD 32.60), with 42.9% of positives discovered later.

---

## 2. Evaluation unit and fold grain

| Object | Grain |
|---|---|
| evaluation row | one Component 4 feature row = one `(establishment_id, inspection_date)` eligible canvass |
| fold | one `(fold_set, fold_id)` — three date windows |
| metric row | one `(fold, model, metric, k)` |
| curve point | one `(fold, schedule, model, seed, slot_index)` |
| simulation row | one `(fold, schedule, model, seed)` |

Component 5 **reads** identity, labels and features. It never recomputes any of
them. The join key is `target_inspection_id`, and the label column `target` is
Component 3's, unchanged.

---

## 3. Train / calibration / test semantics

Rolling-origin backtest, expanding training window, quarterly cadence.

```text
TRAIN ─────────────────────► | CAL | TEST
     anchor 2018-07-01         3mo   3mo
```

- `train`: `train_start <= rd <= train_end`, anchored at `2018-07-01` for every fold.
- `calibration`: the quarter after training ends. **Strictly between** train and test.
- `test`: the quarter after calibration.
- `outside`: everything else. A legitimate label, not an error.

Enforced invariants, asserted at construction in `FoldSpec.__post_init__`:

```text
train_end < calibration_start <= calibration_end < test_start <= test_end
```

A fold violating these **cannot be constructed**. There is no code path that
produces a leaky fold and reports it afterwards.

**Calibration is never merged into training or test.** `TRAIN → CAL → TEST`, and
never `TRAIN + TEST → calibration` or `TRAIN → TEST → calibration`. Component 9
fits Platt or isotonic scaling on the calibration window; a calibrator fitted on
test makes the reported probabilities self-fulfilling.

The fold count is derived from the data. A test quarter is emitted only if it is
**entirely** covered by the snapshot; the trailing partial quarter is excluded and
named in the manifest.

---

## 4. Chronological ordering

Rows are ordered by `(inspection_date, target_inspection_id)` ascending, which is
also the business-as-usual schedule. Sorting is explicit at every stage; Parquet
row order is never a contract.

**Intra-day order is unrecoverable.** `inspection_date` carries no time component
anywhere in the source, so inspections sharing a date cannot be ordered from the
data. They are settled deterministically on `target_inspection_id` and this
cannot bias any date-based metric, because every row tied that way shares a date.

---

## 5. Prediction contract

A future model hands over a `PredictionSet`:

| Field | Meaning |
|---|---|
| `model_name` | producer identity |
| `model_version` | producer version |
| `fold_id` | which fold these scores are for |
| `frame` | exactly two columns: `target_inspection_id`, `score` |
| `is_probability` | whether `score` is a calibrated probability |
| `trained_through` | last reference date the producer was allowed to learn from |

Persisted artifacts additionally carry `model_name`, `model_version`, `fold_id`
as columns so a file on disk is self-describing, and may carry `trained_through`
and `is_probability`.

**Rejections** (all raise `PredictionContractError`, none warn):

1. any column other than the two — a label cannot ride along
2. a null or non-finite score
3. a duplicate `target_inspection_id`
4. coverage that is not *exactly* the fold's test window
5. `trained_through` later than the fold's `calibration_end`
6. a `fold_id` that does not match the fold offered

Rule 4 exists because a model that quietly drops the establishments it finds hard
would post a better precision@k for a reason unrelated to being better. Rule 5 is
what makes retrospective cheating hard to do by accident.

---

## 6. Score direction

```text
higher score = higher predicted probability of a Priority / Priority Foundation
citation = inspected sooner
```

`0.90` ranks ahead of `0.10`. Asserted on every run by the
`score_direction_is_descending` check and by a unit test, because a convention
inverted during a refactor would silently reverse every conclusion in the project.

---

## 7. Tie-breaking

Ties in score are broken on `target_inspection_id` **ascending, as a string**.

The column is stable, unique per row and independent of frame order, which is the
entire requirement. Whether the ordering is lexicographic or numeric is arbitrary
for a tie-break; being deterministic is not.

ROC-AUC and PR-AUC do **not** use the tie-break — both are defined over the whole
score distribution and handle ties analytically, so imposing an order would change
the answer rather than merely settle it.

---

## 8. Statistical metrics

Implemented in `evaluation/metrics.py`, no runtime dependency added, every one
cross-checked against scikit-learn in the test suite.

| Metric | Meaning | Drift-robust? |
|---|---|---|
| `roc_auc` | P(random positive outranks random negative) | **yes** |
| `pr_auc` | average precision (not a PR trapezoid) | no |
| `brier` | mean squared error of a probability | no |
| `log_loss` | negative log likelihood | no |
| `ece` | expected calibration error, 15 equal-mass bins | no |
| `mce` | worst single bin | no |
| `precision` / `recall` / `f1` | at an explicit threshold | no |

`None` is returned rather than a substitute when a metric is undefined:
`roc_auc` with one class present, `pr_auc` with no positives. Returning 0.5 or 0
would invent an answer.

**PR-AUC is `average_precision_score`, not `auc(recall, precision)`.** The
trapezoid form is optimistically biased; a test asserts the two differ so a
refactor toward the wrong one fails loudly.

---

## 9. Ranking metrics

| Metric | Meaning |
|---|---|
| `precision_at_k` | share of the top *k* that were genuine citations |
| `recall_at_k` | share of the window's positives the top *k* captured |
| `lift_at_k` | `precision_at_k / base_rate` — 1.0 means no better than blind |

**`RANKING_METRICS` and `PROBABILITY_METRICS` are disjoint, and the separation is
load-bearing.** A producer that sets `is_probability=False` is scored only on
ranking metrics. A random shuffle has no meaningful Brier score, and emitting one
so it fits an API would be a fabrication — the rows simply do not exist.

**Read precision@k beside its prevalence.** With base rates from 0.379 to 0.513
and *k* frequently exceeding the number of positives, precision@k has a ceiling
below 1 that moves between folds. `lift_at_k` is the comparable form.

---

## 10. Simulation semantics

**The slot model.** A test window's capacity profile is the exact multiset of
`(date → count)` the city actually worked. Sorting those dates gives a *slot
calendar*: slot *i* has a real date, and there are exactly as many slots as
inspections. A schedule is a permutation; position *i* takes slot *i*, which
assigns every inspection a **simulated date**.

**The business-as-usual identity.** Ordering by actual date and filling slots in
date order returns every inspection to its own real date — the sorted sequence
holds exactly `count(d)` consecutive entries for date *d*, and the slot array
holds exactly `count(d)` consecutive positions for *d*, in the same order. So
`bau_simulated_date == actual inspection_date` always, and "days earlier than
business as usual" means "days earlier than what really happened". Asserted, not
argued: `business_as_usual_is_real` runs on every fold of every run.

**Labels never move.** An establishment cited on 14 June is still cited when the
schedule moves it to 2 May. This is a reordering simulation, not a causal
simulator, and §14 records how questionable that is.

### The five reference schedules

| Schedule | Definition | Role |
|---|---|---|
| `optimal` | every positive first | unreachable upper bound |
| `model` | descending score, ties on id | the thing under test |
| `business_as_usual` | actual historical date order | the comparator that matters |
| `random` | seeded shuffle, replicated over seeds | the floor |
| `worst` | every positive last | lower bound |

The envelope is **measured, not assumed**. On this data business-as-usual sits
almost exactly at random within a quarter, which is a finding rather than a bug.

### Discovery curve

x = fraction of slots consumed ∈ [0,1]. y = fraction of the window's positives
discovered ∈ [0,1]. Stored at full resolution, `n + 1` points from `(0,0)`.

Area by the **trapezoid rule** over the normalized points:

```text
A = (1/n) · Σ_{i=1..n} (y_{i-1} + y_i) / 2
```

### Normalized discovery efficiency

```text
NDE = (A_model − A_random) / (A_optimal − A_random)
```

Both denominator terms are **analytic, not sampled**:

- `A_random = 0.5` exactly. A uniformly random permutation has
  `E[positives after i slots] = P·i/N`, so the expected curve is the diagonal.
- `A_optimal = 1 − P/(2N)`.

So the denominator is `(N − P) / 2N` and the scale is fixed: **1** perfect, **0**
random, **−1** worst-possible. Essentially a CAP/Gini coefficient for the
schedule, and base-rate invariant — which is why it is the headline for
cross-fold aggregation. The empirical multi-seed random band is still computed
and reported, as a check on the analytic claim.

`None` when `P == 0` or `P == N`: with nothing to separate, every schedule is
identical and an efficiency would be a fiction.

### Days-earlier detection

`bau_simulated_date − model_simulated_date`, in days, positive meaning earlier.
Primary population is **positives only** — moving a clean establishment forward
buys nothing. An all-rows variant is reported as secondary.

Reported as a distribution: mean, median, SD, p25, p75, min, max, and the
fractions **improved / unchanged / worse**. The mean alone is forbidden. The 2015
Chicago result was 7.438 days with SD 25.156 and never said how many
establishments were found later; on this data the best baseline's SD is 7× its
mean and 42.9% of positives are found later.

### First-half discovery

Share of the window's positives found in the first `n // 2` slots. Reported for
conceptual comparability with the 2015 result's 69%-vs-55%, which came from a
different food code, a different target and a 14.1% base rate — **not a benchmark
this project can be measured against**.

---

## 11. Capacity semantics

```text
slots = the observed multiset of inspection dates in the window;
every schedule is a permutation over the same slots
```

Capacity is held constant **by construction**, not by assumption:
`n_slots == n_inspections` and the multiset of simulated dates equals the
observed one. Verified per fold by `capacity_is_conserved`.

The system changes the **order**, never the **number**. A simulation that
quietly added inspectors would beat business as usual for a reason that has
nothing to do with the model.

### Where *k* comes from

`k` is derived from each fold's own **measured** median daily capacity — never
chosen for convenience.

| Name | Value |
|---|---|
| `k_1_day` | fold median daily capacity |
| `k_1_week` | × 5 |
| `k_1_month` | × 21 |
| `k_pct_01` … `k_pct_50` | 1%, 5%, 10%, 25%, 50% of the window |

All clamped to the window size. Measured medians range 22 to 45 across test
windows.

---

## 12. Baseline definitions

All deterministic. **Nothing in Component 5 is fitted** — Component 6 owns the
first trained model.

| Name | Rule | Null rule |
|---|---|---|
| `business_as_usual` | negated date ordinal, so same-day rows tie | n/a |
| `random` | uniform noise, seeded, replicated over 20 seeds | n/a |
| `days_since_last_canvass` | longest gap first | never canvassed sorts first |
| `priority_at_last_canvass` | cited last time → 1.0, clean → 0.0 | unknown → 0.5 |
| `prior_canvass_priority_rate` | historical citation rate | unknown → 0.5 |
| `constant` | all equal — a tie-breaking diagnostic | n/a |

Each null rule was chosen on a **semantic** argument and its consequence measured
afterwards; the arguments and the measured costs are in `rankers.py` and the
findings document. Choosing a null tier by its outcome would be fitting to the
label under another name.

**`days_since_last_canvass` is not the spec's "days overdue".** A statutory
overdue figure needs the CDPH risk category to know what the deadline was, and
that column is in the raw snapshot but not in Component 4's feature table.
Component 5 may not add features, so this measures elapsed time, not deficit
against a deadline. Recorded as a limitation, not passed off.

**The CDPH 2015 replication was deferred to Component 6, which has now shipped an
explicitly labelled approximation** -- only 3 of the 2015 model's 10 input families
are reachable from the current data contract, so a faithful replication remains
impossible. See `docs/data_contracts/baseline_predictions.md` §11. The original
reasoning for the deferral follows.

It is a fitted logistic
regression. `rankers.py` carries the table of which of its inputs are
reconstructible from the current data contract; several (311, crime, licence,
weather, inspector) are not ingested at all, so whatever Component 6 builds must
be labelled an approximation with that table attached.

---

## 13. Missing-data behaviour

- **A missing score is a rejection, never an imputation.** Filling a gap would
  silently rank that establishment last and call it a prediction.
- A null feature value is handled by the ranker's declared null rule, and by no
  other mechanism.
- A metric that is undefined returns `None`, and the row carries a null rather
  than a substitute value.
- A window with no positives, or with every row positive, yields `None` for NDE
  and first-half discovery.

---

## 14. Randomness and reproducibility

The only randomness is explicitly seeded, and every seed is written to the
manifest.

| Use | Default | Recorded as |
|---|---|---|
| random reference schedule | seeds 42…61 (20 replicates) | `random_seeds` |
| time-invariance label re-draw | seed 20260816, 1000 replications | `sensitivity_seed`, `sensitivity_replications` |

Everything else is structural: fold boundaries are calendar arithmetic, every
ordering is fully specified, and each table is sorted by a declared key before it
is written. **Two runs over the same input produce identical tables**, and
shuffling the input rows changes nothing — both asserted by tests.

The manifest pins the feature table by SHA-256 and records the definition
version, the fold configuration, the excluded partial window, every seed, and the
full check list.

---

## 15. Time-invariance sensitivity

The simulation assumes an establishment cited on one day would be cited on a
nearby day. The published audit named that as a flaw. This component measures it
rather than assuming it.

**Model.** Month-of-year effect on citation rate, de-trended by fitting each
month's deviation **from its own year** in log-odds, weighted by cell size. The
year term absorbs the secular drift; what remains is seasonal.

**Fitted per fold on that fold's training window only** — a seasonal model fitted
on the test period would be the same error this component exists to prevent. A
descriptive all-rows fit is also emitted, labelled `fitted_on = "all_rows"`.

**Re-draw.** A monotone coupling: one uniform per row, conditioned on the
observed label. A row left on its own date keeps its label **exactly**, so
business-as-usual is untouched and the comparison stays fair; a positive moved
from a high-risk month to a low-risk one can flip, in proportion to the measured
effect and nothing else.

**BLOCKED.** The specification asks for day-of-year *and temperature*. No weather
data is ingested anywhere in this repository and Component 5 does not fabricate a
data source. The measured seasonal effect is therefore a **proxy** that confounds
temperature with daylight, holidays and staffing without separating them.
Ingesting NOAA GHCN station USW00094846 is a Component 1 extension; the interface
accepts the covariate the moment it exists.

---

## 16. Output schema

Six tables, one stamp per run, **one** manifest keyed to `evaluation_folds`.

```text
data/processed/evaluation/
  evaluation_folds_<stamp>.parquet          18 rows × 18 cols   the split
  evaluation_metrics_<stamp>.parquet     2,808 rows × 14 cols   tidy long
  discovery_curves_<stamp>.parquet     373,986 rows ×  9 cols   full resolution
  simulation_summary_<stamp>.parquet       504 rows × 24 cols   per schedule
  seasonality_<stamp>.parquet              228 rows ×  6 cols   month effects
  sensitivity_<stamp>.parquet               54 rows × 12 cols   uncertainty bands
  manifest_evaluation_folds_<stamp>.json
```

Column order is part of the contract in every case; see
`src/sentinel/evaluation/writer.py`, where the schemas are declared once and both
written and validated from the same definition.

---

## 17. ⚠ What downstream may not use

- **Nothing in `evaluation/` may be joined onto a training table.** These are
  measurements about models; a feature derived from a test score is the most
  damaging leakage in the project and the hardest to spot afterwards. ADR 0013.
- **The test period is held out.** Do not try a feature, look at the test score,
  and keep the feature. Design choices are frozen against train and calibration;
  the test result is reported afterwards.
- **Do not average across `fold_set`.** `quarterly` and `covid_shift` answer
  different questions and are kept separate for that reason.
- **Do not compare results computed under different fold configurations.**
  Changing the cadence, anchor or calibration placement needs an ADR.

---

## 18. Guarantees a consumer may rely on

1. `train_end < calibration_start <= calibration_end < test_start <= test_end`
   for every fold, unconstructable otherwise.
2. No `target_inspection_id` appears in two splits of the same fold.
3. Every training row is dated inside its fold's training window; every test row
   is later than every training and calibration row in that fold.
4. Test windows within a fold set are strictly increasing and disjoint.
5. Training is an expanding window anchored at `2018-07-01`; the anchor never moves.
6. Every emitted fold's test window is fully covered by the snapshot.
7. Every schedule consumes exactly the observed inspection slots — capacity is
   identical across schedules.
8. `bau_simulated_date == actual inspection_date`, every row, every fold.
9. Higher score is scheduled first.
10. Row counts reproduce a direct re-count of the feature table — the target is
    read, never redefined.
11. Two runs over the same input produce identical tables.
12. Every scored model declared a training horizon within its fold.

Checks 1–12 run on **every** invocation, are error-severity, and a failure exits
the CLI non-zero.

---

## 19. Known limitations

1. **The estimand is re-ordering only.** No labels exist for establishments
   nobody inspected, so counterfactual coverage cannot be evaluated. §1.
2. **Not causal.** Nothing here shows a different order would have caused a
   different outcome, and no claim about illness is licensed. §1.
3. **Time invariance does not hold.** Measured, not assumed: the de-trended
   seasonal swing is 11.77 percentage points peak-to-trough. The sensitivity band
   bounds the impact but does not eliminate it. §15.
4. **The temperature covariate is BLOCKED** — no weather data is ingested, so the
   seasonal effect is a proxy that cannot separate temperature from daylight,
   holidays and staffing. §15.
5. **Intra-day order is unrecoverable**, so business-as-usual has no within-day
   resolution and the tie-break within a date is arbitrary-but-deterministic. §4.
6. **`days_since_last_canvass` is not statutory overdue** — the CDPH risk
   category is not in the feature table. §12.
7. **~~No fitted baseline yet.~~ Resolved by Component 6**, which added three fitted
   logistic regressions via `sentinel evaluate --predictions`. The best reaches NDE
   0.2326 against the best heuristic's 0.1845. The CDPH replication ships as a
   labelled approximation. Original note: deferred to Component 6,
   and several of its original inputs are not in the data contract at all. §12.
8. **The calibration windows are unused** until Component 9 exists. The structure
   is built early on purpose.
9. **Duplicate-record sensitivity is not a Component 5 experiment.** Component 1
   owns raw treatment and Component 3 already collapses same-day canvasses (530
   days), so there is no knob here to vary. Evaluating it would mean redefining
   an upstream contract.
10. **17 folds is not many.** Fold-to-fold SD is reported for every metric, but
    with quarterly windows a genuinely unusual quarter moves the mean visibly.
11. **The `covid_shift` fold is a single fold**, so it carries no variance
    estimate and should be read as an illustration rather than a measurement.
12. **Establishment-level dependence across folds is not modelled.** The same
    premises appears in many test windows over eight years, so folds are not
    independent samples and the reported SD is a fold-to-fold spread, not a
    confidence interval.
