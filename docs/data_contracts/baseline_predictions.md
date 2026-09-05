# Data contract: baseline model predictions (Component 6 output)

**Produced by:** `sentinel train-baselines` (`src/sentinel/modeling/`)
**Layer:** `data/processed/predictions/`
**Consumed by:** Component 5 (`sentinel evaluate --predictions`), Component 9 (calibration),
Component 11 (attribution), Component 21 (demo)
**Design rationale:** `docs/analysis/baseline_models_findings.md`, ADR 0014, ADR 0015


> **Component 9 status (2026-08-24):** this artifact was consumed by Component 9 and is
> **unchanged and byte-identical** — Component 9 re-executed the fits behind a bit-identity
> gate rather than rewriting anything here. The calibrated scores are a separate artifact,
> `calibrated_predictions_<stamp>.parquet`; see `calibrated_predictions.md`. The
> probabilities in *this* file remain uncalibrated.
---

## 1. What this component produces, and what it deliberately does not

Component 6 trains the first fitted models in the project and emits their scores. It
produces **no metrics**. Component 5 evaluates; Component 6 predicts. Computing a second
set of numbers here would create two answers to every question with no way to tell which
was authoritative, and it would put the test window in reach of a component that is
allowed to fit things.

```text
Component 4 feature table
        ↓
Component 5 fold definitions          (rebuilt from the data, never invented here)
        ↓
Component 6 one model per fold        ← this contract
        ↓
baseline_predictions_<stamp>.parquet
        ↓
Component 5 prediction contract       (validate_predictions, 9 rules)
        ↓
Component 5 metrics + simulation
```

The estimand is unchanged from Component 5 and is still **not causal**: a re-ordering of
canvass inspections that actually occurred, with the establishment set, the capacity and
the labels all held at their observed values.

---

## 2. Grain and keys

| | |
|---|---|
| grain | one row = one (model, fold, scored test row) |
| primary key | `(model_name, model_version, fold_id, target_inspection_id)` |
| joins to | Component 4's feature table on `target_inspection_id` |
| rows (full run) | **124,608** = 3 models × 18 folds × that fold's test rows |

Two sibling tables share the run's timestamp: `baseline_coefficients` (one row per model,
fold, term) and `baseline_training_log` (one row per model, fold).

---

## 3. The models

Three, all L2-penalised logistic regression with **identical, fixed** hyperparameters:
`penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42, class_weight=None`.

| `model_name` | `model_version` | features | why it exists |
|---|---|---|---|
| `logistic_regression` | v1 | 26 + 4 indicators | the primary baseline; the model the spec names |
| `logistic_regression_no_scheduling` | v1 | 25 + 4 (− `days_since_any_inspection`) | the ablation, as a separate fitted model |
| `cdph_2015_approximation` | v1 | 19 + 4 | the historical comparison, **an approximation** |

**No tuning, no class weighting, no resampling.** Prevalence is 52.52% overall and
0.379–0.513 per test window; there is no imbalance to correct, and resampling would
corrupt the probability scale Component 9 depends on.

⚠ `cdph_2015_approximation` **is not the CDPH 2015 model** and must never be presented as
one. See §11.

---

## 4. Score semantics

> `score` is `P(target = 1)` from `predict_proba`. **Higher means higher predicted risk**
> of a Priority or Priority Foundation citation.

Component 5 assumes this direction (`SCORE_DIRECTION`, plus a live two-row probe in its
validator). Inverting it would produce a plausible, confidently wrong result rather than
an error.

Scores are in the **closed** interval [0, 1]. `predict_proba` can return exactly 0.0 or
1.0 by rounding when the linear predictor is extreme; that is legitimate, harmless for
ranking, and reported as a warning-severity `saturated_scores` count rather than rejected.
On the full run the observed range is 2.44e-06 … 0.999603 with zero saturated values.

⚠ **The probabilities are uncalibrated.** Measured ECE ≈ 0.064, MCE ≈ 0.17. Component 9
owns calibration; nothing in this component's output may be described as calibrated.

---

## 5. `trained_through` is the training end

```text
fold quarterly-2022Q2
  train        2018-07-01 .. 2021-12-31   <- fitted here; trained_through = 2021-12-31
  calibration  2022-01-01 .. 2022-03-31   <- UNUSED by Component 6
  test         2022-04-01 .. 2022-06-30
```

Component 5's contract permits `trained_through <= calibration_end`, and the six built-in
heuristics declare `calibration_end`. **Component 6 declares `train_end`**, which is
stricter, because it fits no calibrator and never reads the calibration window — declaring
the later date would claim a horizon the model did not use and would disable Component 6's
own leakage check. The unused value is recorded separately as `calibration_end_unused`.
See ADR 0014.

`trained_through` and `is_probability` are **never null**. A null horizon makes the
evaluator skip its horizon check; a null `is_probability` is coerced to `False` by
`read_predictions`, silently downgrading the model to ranking-only and suppressing the
probability metrics.

---

## 6. Missing-value rules

Component 4 declares four NULL rules across exactly **10** of its 26 features; the other
16 are never null and `0` is a real observation for them. Component 6's preprocessing is
declared per column and fitted **on the training window only**.

| group | count | rule |
|---|---|---|
| never-null numerics | 16 | passthrough |
| nullable numerics | 7 | `SimpleImputer(strategy="median")`, train-fitted |
| nullable booleans | 3 | `SimpleImputer(strategy="constant", fill_value=0.0)` |
| family indicators | 4 | computed, always present |

Then `StandardScaler()` over all 30 columns, also train-fitted.

### 6.1 Four family indicators, not ten per-column ones

The null masks **within** a Component 4 null-rule family are byte-identical, measured on
all 57,727 rows. So `SimpleImputer(add_indicator=True)` would emit 10 indicators of which
only 4 are distinct — perfectly collinear duplicates that split coefficient mass
arbitrarily and make the coefficients artifact unreadable. It also selects indicators by
observation (`features="missing-only"`) rather than by declaration, so on a small table the
matrix width would vary per fold.

The four indicators are always present, in a declared order, for every model regardless of
its feature subset — so matrix width is a property of the rule set, not of the fold:

```text
missing_no_prior_canvass
missing_no_code_era_canvass
missing_no_inspected_canvass
missing_no_prior_inspection
```

Each family's mask is also exactly the zero-set of a paired never-null count
(`prior_canvass_count == 0` and so on), so **an indicator adds no new information to this
table** — only linearly-available information. That distinction matters for reading the
coefficients.

### 6.2 Why booleans get a constant, not a median

`priority_at_last_canvass` drifts monotonically from 0.6310 to **0.5056** across the 17
quarterly training windows — currently 0.0056 above the median-fill boundary. A median rule
would flip that column's fill from 1 to 0 partway through a future fold sequence, moving
"unknown" from the false side of the linear predictor to the true side and flipping the
family indicator's coefficient sign to compensate, for no substantive reason. Constant 0
makes "unknown ≡ false, plus the indicator" and is immune to the drift.

---

## 7. ⚠ What a consumer may not use, and how to read the coefficients

**`baseline_predictions` may never be joined onto a feature table.** It is a model output;
joining a score onto a training table is the most damaging leakage available. ADR 0014.

**Coefficients are not feature importances.** The design matrix has condition number 71.8
and `prior_canvass_count` / `prior_canvass_inspected_count` are correlated at 0.9888 while
carrying mean coefficients of **+1.99** and **−1.47** — one effect, split across two terms.
Component 11 owns attribution. The coefficients table exists for provenance and for
sanity-checking, and `standardized_coefficient` is named for what it is:

- values are on the **standardised** scale;
- `scaler_mean` and `scaler_scale` are emitted beside each term so a reader can recover the
  raw-feature-scale value;
- the intercept is a row with `term = "__intercept__"` and null scaler statistics.

**Do not average across fold sets.** `quarterly` and `covid_shift` are separate, and the
model ordering *reverses* between them (§12.3 of the findings).

---

## 8. Output schemas

Column order is part of the contract; changing it is a contract change. No table carries a
timestamp or a duration — wall clock in a Parquet file would mean two runs over identical
inputs producing different bytes.

### 8.1 `baseline_predictions`

| column | type | note |
|---|---|---|
| `target_inspection_id` | Utf8 | joins to Component 4 |
| `score` | Float64 | P(target = 1); higher = higher risk |
| `model_name` | Utf8 | |
| `model_version` | Utf8 | |
| `fold_set` | Utf8 | `quarterly` or `covid_shift` |
| `fold_id` | Utf8 | |
| `trained_through` | Date | the fold's training end; never null |
| `is_probability` | Boolean | always true here; never null |
| `model_definition_version` | Utf8 | |

`read_predictions` selects the columns it needs, so the extra provenance is harmless to
the evaluator and useful to a human.

### 8.2 `baseline_coefficients`

`model_name`, `model_version`, `fold_set`, `fold_id`, `term`,
`standardized_coefficient`, `scaler_mean`, `scaler_scale`, `imputed_fill_value`,
`model_definition_version`.

### 8.3 `baseline_training_log`

`model_name`, `model_version`, `fold_set`, `fold_id`, `train_start`, `train_end`,
`trained_through`, `calibration_end_unused`, `test_start`, `test_end`, `train_rows`,
`test_rows`, `feature_count`, `matrix_column_count`, `train_positive_rate`, `seed`,
`n_iter`, `max_iter`, `converged`, `saturated_scores`, `is_approximation`,
`model_definition_version`.

---

## 9. Determinism

Two runs over the same input produce identical tables; only the filename stamp differs.
This rests on four properties, each asserted by a test:

1. **Training rows are sorted by `(inspection_date, target_inspection_id)` before
   fitting.** This is load-bearing, not decorative: without it, refitting the same 23,346
   rows in a different order moves coefficients by up to **7.049e-09**, because
   `StandardScaler` accumulates variance incrementally and the lbfgs gradient is a BLAS
   reduction. The sort — not `random_state`, which `lbfgs` ignores entirely — is what makes
   a re-run reproducible.
2. Preprocessing statistics come only from the training window.
3. Every table is sorted by a declared key before writing.
4. No table carries a timestamp or duration.

⚠ **The claim is conditional and the conditions are recorded per run.** Bit-identity holds
for a fixed input, row order, library set and BLAS thread count — **not** across library
versions or thread counts. `sklearn_version`, `numpy_version` and `blas_threads` are in the
manifest for exactly this reason. Verified on scikit-learn 1.9.0 / numpy 2.5.2.

---

## 10. Guarantees a consumer may rely on

Each is an error-severity check; a failure exits non-zero and writes nothing usable.

1. Every model input is a declared Component 4 feature. No identifier, label or provenance
   column is ever a predictor.
2. One `feature_definition_version` across the table, recorded in the manifest.
3. Every fit's training rows fall exactly inside its fold's training window, re-derived
   from the data rather than reported by the orchestrator.
4. No training row id appears in the same fold's calibration or test split.
5. Every imputation median is re-derived from the training window and matched — the
   mechanical proof that preprocessing did not see the future.
6. Every model declares `trained_through == fold.train_end`, strictly before
   `calibration_start`.
7. The null masks within each family are identical, re-asserted per run.
8. All four family indicators are present in every fit.
9. Every fit converged strictly inside `max_iter` (observed 43–83 of 1000).
10. One coefficient per matrix column, with a scaler statistic for each.
11. Predictions cover every fold's test window exactly — no missing row, no extra row.
12. No duplicated `(model, fold, row)`.
13. Every requested model covers every fold, `covid_shift` included.
14. Every score is finite and within [0, 1].
15. No null in the columns that make a prediction file self-describing.

Warning-severity notes report the saturated-score count, the approximation models, that
the probabilities are uncalibrated, and the training base-rate drift (0.5348–0.7731 across
folds, because expanding windows dilute the early years).

---

## 11. CDPH 2015: an approximation, and why

`cdph_2015_approximation` is restricted to the feature families the 2015 model's input list
can actually reach. **Only 3 of its 10 input families are reachable from the current data
contract.**

| 2015 input family | status |
|---|---|
| prior violation history | available |
| time since last inspection | available (`days_since_last_canvass`) |
| business age / tenure | available (`days_since_first_inspection`) |
| inspector identity | **excluded deliberately** — audit Finding 1 |
| nearby 311 complaints | **not ingested** |
| burglary intensity | **not ingested** |
| alcohol / tobacco licence | **not ingested** |
| weather / temperature | **not ingested** (needs NOAA GHCN USW00094846) |
| facility type | **not a feature** (in raw, not in Component 4's table) |
| CDPH risk category | **not a feature** (in raw, not in Component 4's table) |

A faithful replication is therefore **not reachable**. The approximation carries an
`approximation_note` naming every unreachable input; the note is written into the manifest
and flagged as `is_approximation` in the training log. **No coefficient, input or
historical behaviour is fabricated.**

The 2015 published numbers are also not a benchmark this project can be measured against:
a different food code, a different target definition, and a 14.1% base rate against this
data's 52.52%.

Related gap: `days_since_last_canvass` is **not** the specification's "days overdue". A
statutory overdue figure needs the CDPH risk category to know what the deadline was, and
that column is in the raw snapshot but not in Component 4's feature table.

---

## 12. Known limitations

1. **Coefficients are not importances** (§7). Condition number 71.8; a 0.9888-correlated
   pair splits into +1.99 / −1.47.
2. **Bit-reproducibility is conditional** on library versions and BLAS thread count (§9).
3. **Probabilities are uncalibrated** (§4). Component 9 owns calibration.
4. **The missing-indicator encoding captures a level shift, not a differing slope.** An
   interaction in the missing group cannot be represented by this model class; that is a
   Component 7/8 question.
5. **43.24% of violations are still discovered later** than business-as-usual under the
   best model — marginally worse than the best heuristic's 42.88%. Re-ordering under fixed
   capacity is zero-sum; the gain is net, not free.
6. **PR-AUC 0.5321 against a 0.4307 floor is a modest gain**, not a good absolute result.
7. **The ablation answers one question, not the general one.** It measures
   `days_since_any_inspection`'s contribution to *this* model class with *this*
   preprocessing.
8. **Model selection on the quarterly folds would have picked the wrong model for
   `covid_shift`** (findings §12.3). This is a measured warning about the whole
   model-selection exercise, not a defect in the models.
9. **Nothing here is causal**, and nothing licenses a claim about establishments nobody
   inspected — they have no labels — or about illness prevented.

---

## 13. Reproducing

```bash
uv run sentinel train-baselines --dry-run --report      # validate, write nothing
uv run sentinel train-baselines                         # write the artifacts
uv run sentinel evaluate --predictions data/processed/predictions/baseline_predictions_<stamp>.parquet --report
```

Full-data timings: training 29.7 s (22.3 s fitting, 54 fits); evaluation 237.8 s.
Profiling, read-only and train-window-only:
`uv run python scripts/profile_baselines.py`.
