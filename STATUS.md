# Current Status

**Last updated:** 2026-08-17
**Current component:** 9 of 21 — Probability Calibration
**State:** Component 7 complete and verified against the full 57,727-row feature table.
XGBoost and LightGBM, tuned under a protocol that cannot reach a test window, and a PyTorch
network with entity embeddings. 1,776 tests
pass; ruff, ruff format and mypy (strict) are clean.

**The measured answer:** the tree models beat the Component 6 logistic baseline on the
quarterly mean — NDE 0.2326 → 0.2376, about **+2% relative** — but the logistic model wins
**7 of 17 individual folds**, and its observed NDE sits *inside* XGBoost's seasonality
redraw interval. Component 6's improvement over the heuristics survived that test;
Component 7's improvement over Component 6 does not clearly survive it. Two very different
nonlinear learners landing within 0.005 NDE of a penalised GLM is evidence that the ceiling
here is the 26-feature representation rather than the estimator.

---

## Completed

### Component 1 — Project foundation + Chicago food inspection ingestion

* **Project foundation.** Python 3.12 + uv, `src/` layout, hatchling build,
  `uv.lock` committed. Ruff, mypy (strict), pytest configured. Minimal GitHub
  Actions CI. Optional pre-commit config.
* **Configuration.** Pydantic Settings, `SENTINEL_` env prefix, `.env` support.
  No dataset ID, endpoint, path or limit is hardcoded at a call site.
* **Socrata API investigation.** Endpoint, dataset, pagination, page-size
  ceiling, ordering behaviour, error shape and value encoding all verified
  against the live API and documented in `docs/api/socrata_findings.md`.
* **Socrata client.** Explicit `$limit`/`$offset`/`$order` pagination as a
  generator; runtime field discovery; bounded exponential-backoff retry on
  429/5xx/timeouts; immediate failure on other 4xx; strict response validation.
* **Raw ingestion.** Records → all-`Utf8` Polars frame → timestamped, zstd
  Parquet under `data/raw/food_inspections/`. Nothing is cast, cleaned,
  filtered or deduplicated.
* **Manifest.** JSON provenance sidecar per raw file: source, dataset, UTC
  timestamp, code version, mode, page/request parameters, row count, columns,
  Socrata-declared types, Parquet schema, size, SHA-256.
* **DuckDB query layer.** Named descriptive queries over raw Parquet via
  `read_parquet()`. In-memory, no schema DDL.
* **CLI.** `sentinel ingest` (`--dev` / `--limit N` / `--full`) and
  `sentinel query`, built on stdlib argparse.
* **Logging.** Python `logging` throughout; no `print()` in `src/sentinel`.
* **Tests.** 77 unit tests (HTTP mocked with respx) + 3 opt-in live tests.
* **Documentation.** README, raw data contract, verified API findings, 5 ADRs,
  plus STATUS / MEMORY / HANDOFF.

---

### Component 2 — Entity resolution

Maps every inspection to a stable `establishment_id` representing a **physical
food-service premises**. Deterministic, rule-based and fully auditable: no
fuzzy scores, no clustering library, no LLM, and no new dependency.

* `scripts/profile_entities.py` — 36 read-only DuckDB profiles that
  characterised the problem before any resolver code was written. Runs in 8 s
  over 314,245 rows.
* `docs/analysis/entity_resolution_findings.md` — the measurements, and the
  design decisions each one forced.
* `src/sentinel/entity/` — normalization, node construction, blocking, named
  evidence rules with vetoes, union-find clustering with invariants and a
  deterministic split ladder, validation, and the three output tables.
* `sentinel resolve` — CLI, with `--dry-run` and `--report`.
* Output under `data/interim/entity_resolution/`, contract in
  `docs/data_contracts/establishment_assignments.md`.
* ADR 0006 (why rules rather than probabilistic linkage) and ADR 0007 (the
  identifier scheme and its stability limits).

**Verified on the full snapshot:** 314,245 rows → 51,099 nodes → **35,859
establishments** in 43 s. All nine structural validation checks pass. Resolving
a seeded random permutation of every input row produces a byte-identical
mapping.

---

### Component 3 — Target construction

Defines what Sentinel predicts: **for each establishment-date on which a routine
canvass occurred, did that canvass find at least one Priority or Priority
Foundation violation?**

* `scripts/profile_target.py` — 31 read-only profiles run before any target code.
* `docs/analysis/target_construction_findings.md` — the measurements and the
  decision each one forced.
* `src/sentinel/target/` — violation parsing and severity classification,
  eligibility gates, same-day collapse, validation, output table.
* `sentinel build-target` — CLI with `--dry-run` and `--report`.
* Contract in `docs/data_contracts/inspection_targets.md`; ADR 0008 (the target
  definition) and ADR 0009 (the 2018-07-01 code-era boundary).
* Also corrected `docs/data_contracts/food_inspections_raw.md`, which documented
  four `results` values where there are seven.

**Verified on the full snapshot:** 314,245 inspections → 313,624 target rows →
**57,727 eligible, 30,316 positive (52.52%)** in 25 s. All twelve structural
checks pass. Rebuilding reproduces the table exactly, and a seeded permutation of
every input row produces identical labels.

---

### Component 4 — As-of feature engineering

Builds the information a scheduler had **before** each prediction opportunity.
The rule the component exists for: a feature for the row at `inspection_date = d`
may use only records dated **strictly before** `d`.

* `scripts/profile_features.py` — 21 read-only profiles run before any feature
  code, answering how much history each target row actually has.
* `docs/analysis/as_of_feature_engineering_findings.md` — the measurements and
  the decision each one forced.
* `src/sentinel/features/` — declared feature specifications, one range join
  carrying the temporal condition, validation, output.
* `sentinel build-features` — CLI with `--dry-run` and `--report`.
* Contract in `docs/data_contracts/as_of_features.md`; ADR 0010 (the boundary and
  the construction) and ADR 0011 (the processed layer).
* **First use of `data/processed/`.**

**Verified on the full snapshot:** 57,727 eligible target rows in → **57,727
feature rows out, 0 unmatched, 26 features**, in 15.6 s. All 15 error checks pass,
including the temporal invariant re-derived independently on every row.
Rebuilding and a seeded row shuffle both reproduce identical values.

---

### Component 5 — Temporal evaluation

Builds an honest way to measure whether a ranking is any good, **before** any
model exists to measure. Component 4 stops a *feature* from seeing the future;
this component stops the *evaluation* from seeing it.

* `scripts/profile_evaluation.py` — 17 read-only profiles run before any
  evaluation code: capacity, candidate fold windows, base-rate drift, de-trended
  seasonality, and what the deterministic baselines have to work with.
* `docs/analysis/temporal_evaluation_findings.md` — the measurements and the
  decision each one forced.
* `src/sentinel/evaluation/` — fold construction, the model-agnostic prediction
  contract, hand-rolled metrics, deterministic baselines, the re-ordering
  simulation, seasonal sensitivity, validation, output.
* `sentinel evaluate` — CLI with `--folds-only`, `--dry-run`, `--report`,
  `--seeds` and `--sensitivity-replications`.
* Contract in `docs/data_contracts/temporal_evaluation.md`; ADR 0012 (rolling
  origin over random CV) and ADR 0013 (evaluation results are artifacts).
* **First use of `data/processed/evaluation/`.**

**Folds:** 17 quarterly (test windows 2022Q2 … 2026Q2) plus 1 `covid_shift` fold.
Expanding training window anchored at 2018-07-01, calibration strictly between
train and test. The count is derived from the data; the partial 2026Q3 window is
excluded and named in the manifest.

**Metrics implemented:** ROC-AUC, PR-AUC (average precision), Brier, log loss,
ECE, MCE, precision/recall/F1, precision@k, recall@k, lift@k. Every one is
cross-checked against scikit-learn in the test suite — added to the **dev group
only**, so no runtime dependency was introduced.

**Simulation metrics:** discovery curves at full resolution, normalized discovery
efficiency (analytic bounds: optimal 1, random 0, worst −1), days-earlier as a
full distribution, first-half discovery, precision@k at measured capacity.

**Baselines:** business-as-usual, random (20 seeds), days-since-last-canvass,
priority-at-last-canvass, prior-canvass-priority-rate, constant. **Nothing is
fitted** — Component 6 owns the first trained model.

**Verified on the full snapshot:** 57,727 feature rows → 18 folds, 2,808 metric
rows, 373,986 curve points, 504 simulation rows, in **164.2 s**. All 14
error-severity checks pass. Two runs and a seeded row shuffle both reproduce
identical tables.

**Headline result:** the best deterministic baseline reaches NDE 0.1845 ± 0.0404
and a mean of +4.47 days-earlier — but with SD 32.60 and **42.9% of positives
discovered later**. Business-as-usual sits at NDE 0.0066, statistically
indistinguishable from random within a quarter.

**Time invariance does not hold**: the de-trended seasonal swing is 11.77
percentage points, peaking in August and troughing in December. The sensitivity
band shows the ranking's advantage survives it — NDE [0.172, 0.192] over 1,000
label re-draws.

---

### Component 6 — Baseline risk models

The first fitted models in the project. Consumes Component 4's feature table and
Component 5's folds, trains one model per fold, and hands scores to Component 5's
prediction contract. Produces **no metrics of its own** — Component 5 evaluates.

**Code:** `src/sentinel/modeling/` — `definitions.py` (registry + partitions derived
from `FEATURE_SPECS`), `preprocess.py` (matrix + train-only pipeline), `train.py`,
`predict.py`, `writer.py`, `validate.py`, `models.py`, `build.py`.
**CLI:** `sentinel train-baselines`, plus `sentinel evaluate --predictions PATH`.
**Artifacts:** `data/processed/predictions/` — `baseline_predictions_<stamp>.parquet`
(124,608 rows, 1.2 MB), `baseline_coefficients_` (1,530 rows), `baseline_training_log_`
(54 rows), one manifest keyed to the predictions.
**Docs:** `docs/analysis/baseline_models_findings.md`,
`docs/data_contracts/baseline_predictions.md`, ADR 0014, ADR 0015.

**Three models**, all L2 logistic regression with identical fixed hyperparameters
(`C=1.0`, `lbfgs`, `max_iter=1000`, no class weighting — prevalence is 52.52%):
`logistic_regression` (26 features), `logistic_regression_no_scheduling` (the
`days_since_any_inspection` ablation, as a separate fit), and
`cdph_2015_approximation` (19 features, **labelled an approximation** — only 3 of the
2015 model's 10 input families are reachable).

**Verified on the full feature table (2026-08-17):** 57,727 rows, 18 folds, **54 fits**,
training 29.7 s, evaluation 237.8 s. All 15 Component 6 error checks pass, and all 14
Component 5 checks pass with the predictions attached.

**Headline result:** `logistic_regression` reaches **NDE 0.2326** and **+5.70 mean
days-earlier**, against the best heuristic's 0.1845 and +4.47. It wins on **17 of 17**
quarterly folds -- 3 of them by under 0.0025 ROC-AUC, so read that as "never worse,
clearly better on most" -- and the time-invariance bands do not overlap ([0.2160, 0.2374]
vs [0.1720, 0.1922]). Precision@k_1_day 0.6576 vs 0.5551; lift 1.53.

**But read the caveats.** PR-AUC 0.5321 against a 0.4307 no-skill floor is +0.10, a
modest gain. **43.24% of violations are still found later** than business-as-usual —
marginally worse than the heuristic's 42.88%, because re-ordering under fixed capacity is
zero-sum. ROC-AUC 0.6163 is a weak classifier by any general standard.

**The measured methodological result:** on `covid_shift` the ordering **inverts** — the
ablation wins (ROC-AUC 0.6286 vs 0.6256, NDE 0.2571 vs 0.2512). Model selection on the
quarterly folds would have picked the wrong model for the shifted period. This is why the
two fold sets are never averaged.

**Probabilities are uncalibrated**: ECE 0.0635, MCE 0.1664. That is the measured "before"
number Component 9 exists to improve, not a result.

---

### Component 7 — Gradient-boosted risk models

Replaces the estimator and adds a tuning protocol. Changes nothing about folds, features,
the target or the metrics. Produces **no metrics of its own** — Component 5 evaluates.

**Code:** `src/sentinel/boosting/` — `definitions.py` (registry, search space, frozen
tuned parameters), `preprocess.py` (the tree matrix: no imputation, no scaling),
`tuning.py` (temporally valid search), `train.py`, `predict.py`, `writer.py`,
`validate.py`, `models.py`, `build.py`.
**CLI:** `sentinel tune-boosting`, `sentinel train-boosting`, plus the existing
`sentinel evaluate --predictions PATH`.
**Artifacts:** `data/processed/predictions/` — `boosted_predictions_<stamp>.parquet`
(124,608 rows), `boosted_importances_` (1,620 rows), `boosted_training_log_` (54 rows);
and a new fourth processed layer `data/processed/tuning/` — `tuning_trials_<stamp>.parquet`
(400 rows). One manifest per anchor artifact.
**Docs:** `docs/analysis/boosting_models_findings.md`,
`docs/data_contracts/boosted_predictions.md`, `docs/interview/component_7.md`,
ADR 0016–0019.

**Three models.** `xgboost` and `lightgbm` (both on all 26 features, tuned separately per
fold set), and `xgboost_class_weighted` — the class-weighting ablation, which borrows
`xgboost`'s frozen parameters so the only difference between them is the weight.

**No preprocessing is fitted.** NULLs reach the estimator as NaN and are routed by a
learned default direction at each split; there is no imputer and no scaler. The four
null-rule family indicators are kept anyway, so the boosted and baseline matrices are
identical and the C6/C7 comparison is unambiguous. 3,404,772 NaN cells reached the
estimators across 54 fits.

**The tuning protocol (ADR 0017).** Hyperparameters may only be selected from data strictly
earlier than the fold set's first test window. `quarterly`: 2018-07-01..2022-03-31 against
a first test start of 2022-04-01. `covid_shift`: 2018-07-01..2020-05-31 against 2020-06-01.
**Two studies, not one**, because the covid_shift test window sits *inside* the quarterly
region. Early stopping runs only inside the tuning objective; the winning round count is
frozen and the final fit uses no eval set, which is what keeps
`trained_through = fold.train_end` literally true.

**Verified on the full feature table (2026-08-17):** 400 trials over 4 studies (0 failed)
in 563.8 s; 54 fits in 21.4 s; evaluation 68.4 s. All 17 Component 7 error checks pass and
all 14 Component 5 checks pass with the predictions attached.

**Headline result (quarterly, 17 folds).** `xgboost` NDE **0.2376**, ROC-AUC 0.6188, PR-AUC
0.5343, +5.83 days earlier. `lightgbm` 0.2355 / 0.6177 / 0.5342 / +5.75. Against Component
6's `logistic_regression` 0.2326 / 0.6163 / 0.5321 / +5.70. The `xgboost_class_weighted`
ablation posts the best NDE of any model (0.2390) and is **not adopted** — it costs ECE
0.0621 → 0.0836, distorting a well-balanced problem to buy a margin smaller than the
seasonality band.

**But read the caveats.** Per fold, **logistic wins 7 of 17**, xgboost 5, lightgbm 5. The
tree models win by more when they win (+0.047 at 2024Q2) than they lose by (−0.017 at
2025Q4), which is what produces the positive mean. The seasonality redraw gives xgboost
[0.2224, 0.2444], which **contains** the logistic model's 0.2326. **42.89% of violations
are still found later**, against Component 6's 43.24% — effectively unchanged.

**The measured methodological result:** on `covid_shift` the ordering is now
**metric-dependent** rather than simply inverted. LightGBM takes NDE (0.2585 vs 0.2512) and
ROC-AUC; `logistic_regression` takes PR-AUC (0.6328, the highest of any model) and
precision@k_1day (0.9545, by a wide margin). One fold, k=22 slots, days-earlier SD 208 — a
robustness observation, never a selection criterion.

**Probabilities are raw.** Quarterly ECE 0.0621 (xgboost) and 0.0644 (lightgbm) against the
logistic model's 0.0635 — the boosters are **not** worse calibrated here, contradicting the
expectation carried in HANDOFF.md and matching a probe pre-registered before training. Under
shift the expectation holds: lightgbm 0.1518 vs logistic 0.1124. Nothing is corrected;
Component 9 owns calibration.

**Inspector-effect modelling is BLOCKED.** The dataset publishes 22 columns and none
identifies an inspector, so a mixed-effects model with inspector as a random intercept and
any marginalisation over an inspector effect are undefined. Proxies (violation-text
verbosity, ward, day-of-week) were considered and refused. Recorded in every manifest, in
ADR 0019, and in a regression test that fails if such a column ever appears.

**Component 6 is untouched and verified so:** re-running `train-baselines` under the current
library set reproduces sha256 `a2bb9411…00ff5b44`, matching the committed manifest exactly.

---

### Component 8 — Neural network with entity embeddings

Replaces the estimator again, adds a learned categorical representation, and adds the
project's first **experimental input layer**. Changes nothing about folds, the target or the
metrics. Produces **no metrics of its own** — Component 5 evaluates.

**Code:** `src/sentinel/neural/` — `definitions.py` (registry, architecture constants, frozen
learning rates, the two guards), `categoricals.py` (the experimental as-of join),
`encode.py` (per-fold vocabularies and chain membership), `preprocess.py` (Component 6's
pipeline plus a categorical block), `net.py` (the module and the determinism switches),
`train.py` (the inner split and the training loop), `predict.py`, `tuning.py` (the
learning-rate sweep), `embed.py` (embeddings → XGBoost), `figures.py`, `writer.py`,
`validate.py`, `models.py`, `build.py`.
**CLI:** `sentinel build-neural-categoricals`, `sentinel tune-neural`, `sentinel train-neural`,
plus the existing `sentinel evaluate --predictions PATH`.
**Artifacts:** a new fifth processed layer `data/processed/neural/` —
`neural_categoricals_<stamp>.parquet` (57,727 rows); `data/processed/predictions/` —
`neural_predictions_<stamp>.parquet` (373,824 rows), `neural_training_log_` (162 rows),
`neural_epoch_log_` (2,921 rows), `neural_embeddings_` (237,472 rows),
`neural_seed_variation_` (90 rows); `data/processed/tuning/` —
`neural_sweep_trials_<stamp>.parquet` (40 rows). Figures under `docs/analysis/figures/`.
**Docs:** `docs/analysis/neural_models_findings.md`,
`docs/data_contracts/neural_predictions.md`, `docs/data_contracts/neural_categoricals.md`,
`docs/interview/component_8.md`, ADR 0020–0023.

**Nine models.** `neural_embeddings` (the specified network), `neural_numeric_only` (the
fair-comparison control — the same 30 matrix columns Components 6 and 7 see, no categoricals),
`neural_onehot` (indicator columns instead of learned vectors), four single-family ablations,
`neural_pos_weighted`, and `xgboost_chain_embeddings` (Component 7's XGBoost with its frozen
parameters, widened by the 16 chain-embedding dimensions learned on the *same* fold).

**⚠ The specified features did not exist.** Chain, facility type, community area and ZIP are
not in Component 4's table — it is 26 numeric columns and nothing categorical. Facility type
and ZIP are in the raw snapshot, community area only as a Socrata computed region, chain
nowhere. The conflict was surfaced before any code was written and resolved by building a
**separate experimental layer** under `data/processed/neural/` with
`feature_definition_version` unchanged at `v1`. Components 1–7 were not modified. See ADR 0022.

**Early stopping is the one thing C6 and C7 did not do**, and it is where the temporal argument
had to be rebuilt. The validation window is carved from the **last ~15% of the training
window** (0.1501–0.1514 measured across 18 folds), cut on a whole day so no date straddles the
split. `trained_through = fold.train_end` stays literally true, the fold's calibration window
is untouched, and `inner_validation_start` is written into the training log so a reader can
check rather than trust. **Cost:** the weights kept are the best validation epoch's, so a final
fit uses ~85% of its fold's training rows. Self-inflicted and recorded.

**`establishment_id` is refused, not absent.** A closed `EntityFamily` allowlist plus
`FORBIDDEN_COLUMNS` reject it at import, and `validate` restates the refusal at runtime.
`chain` — membership recomputed inside each fold from that fold's training rows — is the
substitute. See ADR 0021.

**Verified on the full feature table (2026-08-18):** 40 sweep trials over 2 studies in 550.9 s;
**234 fits** over 18 folds in 1,998.7 s across 4,306 epochs; evaluation ~390 s. All **21**
Component 8 error checks pass and all **14** Component 5 checks pass with the predictions
attached. Every fit stopped on patience; none exhausted the 200-epoch budget.

**Headline result (quarterly, 17 folds).** `neural_numeric_only` posts the **best NDE in the
project: 0.2482**, ROC-AUC 0.6241, PR-AUC 0.5343, +6.10 days earlier — against `xgboost`
0.2376 / 0.6188 / 0.5343 and `logistic_regression` 0.2326 / 0.6163 / 0.5321. It also posts the
best **Brier (0.2355)** and best **ECE (0.0563)** of any model, better than the penalised GLM,
which **disproves** the pre-registered expectation that a network would be worse calibrated.
It wins **12 of 17 folds** — the first result in this project where a mean improvement and a
per-fold improvement agree.

**But the win is the size of its own noise.** The five-seed experiment gives a ROC-AUC spread
of **0.0058**; `neural_numeric_only` beats XGBoost by **0.0053**. Its 0.2482 sits just above
XGBoost's seasonality p95 of 0.2444, but XGBoost's 0.2376 sits comfortably inside the neural
model's [0.2311, 0.2527]. **Suggestive, not decisive**, and not a deployment recommendation.

**The embeddings — the component's headline experiment — did not help.** `neural_embeddings`
(NDE 0.2215) loses to `neural_numeric_only` (0.2482) by 0.0267, a larger gap than any between
model *classes* in this project. Every single-family ablation is better than the full model.
The one-hot control lands within 0.0009 of the embedding model, so the *representation* is not
the problem: capacity is. Mean best epoch orders almost perfectly by parameter count —
`neural_numeric_only` 10.4 (41,729 params), `neural_embeddings` 4.0 (67,985),
`neural_onehot` 2.3 (337,665). The 26 features carry a fixed amount of signal and extra
capacity buys overfitting.

**The same embeddings *helped* XGBoost.** `xgboost_chain_embeddings` beats plain XGBoost on
NDE (0.2444 vs 0.2376) and posts the **best PR-AUC of any model (0.5357)**.

**The embedding visualisation is a negative result, reported as one.** t-SNE over the 846
learned chain vectors gives a single featureless blob. In the raw 16-d space the pairwise
cosine distribution (mean 0.0018, SD 0.2508) is **statistically indistinguishable from a random
Gaussian table** (0.0000, 0.2504). Nearest neighbours are semantically meaningless — SUBWAY's
are SWEET MANDY BS, KFC, KANELA BREAKFAST CLUB.

**Community area (ADR 0023).** `neural_no_community_area` (0.2258) is **better** than the full
embedding model (0.2215) — it bought no predictive value. The pre-declared non-retention rule
therefore cost nothing on this occasion and stands for the next component that asks. This is
*not* evidence that geography carries no signal, and it is no evidence at all about fairness;
Component 12 still owns that.

**The ordering inverts again on `covid_shift`**, where `neural_onehot` — the *worst* quarterly
neural model — is the best model of any kind (ROC 0.6456, PR 0.6528). Four components, four
inversions. And the metric ordering disagrees too: `neural_numeric_only` wins NDE but loses
`precision@k_1_day` (0.6273) to `lightgbm` (0.6598) and the GLM (0.6576).

## Component 9 — Probability Calibration ✅

**Question:** Components 6–8 built a ranking. When Sentinel says 0.30, does it happen 30% of the
time?

**Answer:** It did not — every candidate was **underconfident** (slope 0.61–0.79, where 1.0 is
perfect). Platt scaling fitted on the previously untouched calibration window pulls the slope to
**1.00–1.03** and cuts ECE by **20–25%**, with the ranking **bit-for-bit unchanged**.

### The blocker, and how it was resolved

The scores this component calibrates **did not exist**. Every prediction artifact covered exactly
the test window (41,536 rows/model); the 34,261 calibration rows were never scored, and **no
fitted model is persisted anywhere in the repository**. Component 9 therefore re-executes
Components 6–8's *unchanged* fit functions and proves they are the same models with a
**bit-identity gate**: the re-derived test window is compared to the committed artifact with `==`.

> **207,680 rows across 5 models × 18 folds. Zero mismatches.** `build.py` raises before fitting
> a single calibrator if that fails. ADR 0026.

Components 6, 7 and 8's artifacts are **unchanged and byte-identical**, verified by sha256.

### Results — quarterly mean over 17 folds

| model | ECE before → after | MCE before → after | Brier before → after | slope before → after |
|---|---|---|---|---|
| `xgboost` | 0.0621 → **0.0474** | 0.1741 → 0.1150 | 0.2379 → 0.2350 | 0.640 → 1.005 |
| `lightgbm` | 0.0644 → **0.0490** | 0.1755 → 0.1260 | 0.2383 → 0.2351 | 0.618 → 1.015 |
| `logistic_regression` | 0.0635 → **0.0518** | 0.1664 → 0.1297 | 0.2382 → 0.2358 | 0.611 → 1.015 |
| `neural_numeric_only` | 0.0563 → **0.0524** | 0.1444 → 0.1201 | 0.2355 → 0.2347 | 0.791 → 1.003 |
| `xgboost_chain_embeddings` ⚠ | 0.0619 → **0.0481** | 0.1767 → 0.1236 | 0.2374 → 0.2346 | 0.651 → 1.029 |

⚠ experimental Component 8 derivative (ADR 0022) — must not become the headline.

**Ranking, verified independently by re-running `sentinel evaluate` on the calibrated artifact:
every delta exactly 0.00e+00.** PR-AUC, ROC-AUC, NDE and precision@k are identical floats.

**Brier decomposition** — the whole gain is reliability, which is the term calibration is
supposed to move:

| | reliability ↓ | resolution ↑ | uncertainty |
|---|---|---|---|
| `xgboost` before → after | 0.00638 → **0.00347** | 0.01190 → 0.01190 | 0.24362 → 0.24362 |

Resolution unchanged to **five decimal places**; uncertainty identical for every model and stage.

### The selection protocol (ADR 0025, pre-registered)

Both methods fitted for every fold; the choice made on an inner chronological split of the
calibration window, by mean log-loss over an **expanding prefix** of folds — never pooled,
because **fold N's calibration window is fold N−1's test window**. Tie rule 0.005 nats, prefer
Platt, frozen with a git date *before* any test window was opened.

**Platt won all 90 (model, fold) cells; 0 method switches.** Per fold in isolation isotonic would
have won **16 of 90** — the instability the prefix exists to smooth. Isotonic lost legibly: its
test log-loss is worse than the *uncalibrated* model's on four of five candidates (0.7064 vs
0.6639 for the network) and its post-calibration slope collapses to 0.42–0.58.

### What went wrong, and what the measurements corrected

* **Three pre-declared thresholds were wrong.** Tie threshold 0.002 → **0.005** (0.002 sat below
  the smallest observed noise SD); logit-recovery tolerance 1e-9 → **1e-4** (xgboost and the
  network compute in **float32**, so a sigmoid round-trip loses ~2.6e-5); Platt self-check 1e-6 →
  **1e-3** (`C=1e10` is large but finite).
* **The bootstrap was not reproducible, and a byte-for-byte check caught it.** The seed key
  used `hash(model_name)`, and **Python salts `str` hashing per process** — so two runs over
  identical inputs drew different resamples. Now seeded from the candidate's registry position,
  with a regression test. Eight of the nine tables are byte-identical across runs;
  `calibrator_selection` carries `seconds` under ADR 0018's exception.
* **The gate failed once, correctly.** A first run under `OMP_NUM_THREADS=1` differed on 32,696
  of 41,536 `logistic_regression` rows by 1e-13 to 5e-10. The committed run used the library
  default thread count; a different count is a different BLAS summation order. Run `calibrate`
  **without** a thread override.
* **ADR 0020's prediction was disproved.** It expected the network to be *over*confident. All
  five models were *under*confident, and the network was the least miscalibrated.
* **The model ordering by ECE inverted.** `neural_numeric_only` had the best uncalibrated ECE and
  now has the second worst — it started closest to calibrated and had least to gain.

### `covid_shift`, reported separately

Calibration **helped** (ECE −10 to −23% on every model) but the level stays roughly double
quarterly and the slope reaches only **0.75–0.90**. The residual is **prior shift**: the base rate
falls 0.683 → 0.513 between its calibration and test windows, and no monotone recalibration
fitted on the earlier window can correct that. Five components, five `covid_shift` divergences.

Isotonic beat Platt on ECE there for `logistic_regression` (0.0861 vs 0.0973) and Platt was
still frozen — which is the evidence the choice was not made on test performance.

### Artifacts

A **sixth processed layer**, `data/processed/calibration/` (ADR 0024), plus the two homes ADR
0014 and ADR 0018 named in advance:

```
data/processed/predictions/  calibrated_predictions_<stamp>.parquet   207,680 rows + manifest
data/processed/tuning/       calibrator_selection_<stamp>.parquet     180 rows
data/processed/calibration/  calibration_base_scores_ (378,985 rows), calibrator_parameters_,
                             calibrator_isotonic_breakpoints_, calibration_drift_ (360),
                             calibration_ranking_preservation_ (180),
                             calibration_brier_decomposition_ (360), calibration_bootstrap_
docs/analysis/figures/       calibration_reliability_<model>_<fold>.png (10),
                             calibration_ece_drift.png
```

The calibrated artifact is readable by `evaluation.contract.read_predictions` **with no change to
Component 5** — which is why it lives under `predictions/`.

### Verification

**21 runtime checks**, all re-derived from production data; every error-severity check passes.
**135 new tests** (leakage 29, metrics 26, train 24, definitions 22, writer 17, CLI 17), full
suite **1,911 passing** (1,776 before), ruff and mypy --strict clean on `src/sentinel`.

Two tests have teeth rather than happy paths: `test_the_bit_identity_detector_itself_works`
perturbs one score by a single ULP and asserts the gate goes red;
`test_the_leakage_detector_itself_works` fits on calibration ∪ test and asserts the checks turn
red. A third,
`test_a_pooled_global_selection_would_read_an_earlier_folds_test_window`, asserts the *rejected*
selection design is detectably leaky — the rejection is executable, not just written down.

### Proposed retraining trigger — a design proposal, not a validated fact

Refit when quarterly ECE exceeds **0.075**, or the calibration slope leaves **[0.80, 1.25]**, in
**two consecutive quarters**. Written *after* seeing the drift series and not validated against
any outcome; stated as a proposal because nothing downstream consumes it yet.

---

## In Progress

Nothing. Components 1 through 9 are closed.

---

## Not Started

Components 10–21. No code exists for any of them.

| # | Component | State |
|---|---|---|
| 9 | Probability calibration | **Implemented** |
| 10 | Inspector-effect modelling | **Blocked** — no inspector field exists (ADR 0019) |
| 11 | SHAP explainability | Not started — C7 emits split-gain importances as a diagnostic only |
| 12 | Fairness auditing | Not started |
| 13 | Deterministic statutory policy engine | Not started |
| 14 | Constrained scheduling | Not started |
| 15 | OR-Tools routing | Not started |
| 16 | Deferral / human-review gate | Not started |
| 17 | LangGraph orchestration | Not started |
| 18 | LLM-generated inspector briefings | Not started |
| 19 | Deterministic briefing verification | Not started |
| 20 | Audit trail | Not started |
| 21 | Frontend demo | Not started |

---

## Current Architecture

```text
src/sentinel/
  __init__.py            __version__, stamped into every manifest
  config.py              Pydantic Settings (env prefix SENTINEL_)
  logging_setup.py       configure_logging()
  cli.py                 argparse: ingest, query, resolve, build-target,
                         build-features, train-baselines, evaluate
  manifest.py            generic sha256 / read / write helpers
  ingest/
    socrata.py           SocrataClient: build_params, discover_fields,
                         fetch_page, iter_pages, bounded retry
    food_inspections.py  orchestration: pages -> all-Utf8 frame -> Parquet
    manifest.py          IngestionManifest model (helpers re-exported)
  query/
    duckdb_queries.py    NAMED_QUERIES, latest_parquet, run_named_query
  evaluation/            Component 5
    models.py            FoldSpec (refuses a leaky fold), PredictionSet, ScheduleName
    folds.py             quarterly_folds, covid_shift_fold, assign_split
    contract.py          validate_predictions, prediction_frame, read_predictions
    metrics.py           roc_auc, pr_auc, brier, log_loss, ece, mce, *_at_k
    rankers.py           deterministic baselines; RANKERS registry
    simulate.py          Window, the five schedules, discovery_curve, NDE
    sensitivity.py       month_effects (de-trended), redraw_sensitivity
    validate.py          validate_evaluation (the seven leakage checks)
    writer.py            six output schemas and their sort keys
    build.py             run_evaluation (the only I/O; --predictions seam)
  boosting/              Component 7
    definitions.py       BOOSTING_REGISTRY, BoostingSpec, SEARCH_SPACE,
                         FIXED_PARAMS, TUNED_PARAMS (frozen), PARAM_MAPPING,
                         the import-time guard
    preprocess.py        tree_matrix (no imputation, no scaling), null_mask,
                         positive_weight
    tuning.py            tuning_region, build_inner_folds, run_study; the
                         protocol that cannot reach a test window
    train.py             fit_fold (canonical sort; no early stopping),
                         build_estimator, fold_labels
    predict.py           score_window (predict_proba[:, 1]), saturated_count
    models.py            FittedBooster (typed facade), StudyResult, TrialResult,
                         the two manifests
    writer.py            four output schemas and their sort keys
    validate.py          validate_boosting (15 error checks + 2 advisories),
                         validate_tuning (5 error checks), all re-derived
    build.py             tune_boosting, train_boosting (the only I/O)
  modeling/              Component 6
    definitions.py       MODEL_REGISTRY, ModelSpec, partitions derived from
                         FEATURE_SPECS, MissingStrategy, the import-time guard
    preprocess.py        to_matrix (null->NaN in one place), build_preprocessor,
                         the four family indicators, ordered_matrix_columns
    train.py             training_frame (via assign_split), fit_fold (canonical
                         sort; raises on non-convergence)
    predict.py           score_window (predict_proba[:, 1]), saturated_count
    models.py            FittedModel (typed facade over sklearn), the manifest
    writer.py            three output schemas and their sort keys
    validate.py          validate_baselines (15 error checks, re-derived)
    build.py             train_baselines (the only I/O)
  features/              Component 4
    definitions.py       FeatureSpec list, WINDOW_DAYS, NullRule
    historical.py        the range join and the temporal boundary
    validate.py          validate_features (incl. temporal_boundary_holds)
    writer.py            output_schema derived from the specs
    build.py             build_features (the only I/O)
  target/                Component 3
    models.py            Severity, TargetStatus, TARGET_DEFINITION_VERSION
    violations.py        split_violations, parse_entry, classify
    construct.py         classify_inspection, collapse_same_day, build_target_rows
    validate.py          validate_targets, format_report, has_failures
    writer.py            TARGETS_SCHEMA, TARGET_EVENT_COLUMNS
    build.py             build_targets (the only I/O)
  entity/                Component 2
    models.py            frozen structures, MatchTier, DEFAULT_THRESHOLDS
    normalize.py         normalize_name / _address / _geo / _license / _zip
    nodes.py             build_nodes, IDENTITY_COLUMNS, blacklisted_coordinates
    blocking.py          spatial / coordinate / licence blocks, candidate_pairs
    evidence.py          compute_signals, evaluate_pair, vetoes, haversine_m
    unionfind.py         UnionFind (find / union / components)
    cluster.py           build_clusters, check_invariants, establishment_id_for
    validate.py          validate_output, format_report, has_failures
    writer.py            the three table schemas and builders
    resolve.py           resolve_establishments (the only I/O)
scripts/
  profile_entities.py    36 read-only profiles; analysis tooling, not library
  profile_target.py      31 read-only profiles over raw + resolved data
  profile_features.py    21 read-only profiles over history availability
  profile_evaluation.py  17 read-only profiles over the evaluation surface
  profile_baselines.py   10 read-only profiles; train windows only, never test
  profile_boosting.py     7 read-only profiles; train + calibration only
```

Runtime dependencies: httpx, pydantic, pydantic-settings, polars, pyarrow,
duckdb, **scikit-learn, numpy, xgboost, lightgbm, optuna**. **Components 2, 3, 4 and 5
added no runtime dependency.** Union-find and haversine are written out rather than importing networkx
or a geo library; string similarity is token-set equality over frozensets rather than
rapidfuzz; every evaluation metric is implemented rather than imported.

**Component 6 added the first two**, and ADR 0015 records why the same reasoning that
kept Component 5's metrics hand-rolled points the other way here: a metric is
arithmetic over two arrays and verifiable against a reference to floating-point
tolerance, whereas an L2 logistic regression is an iterative optimisation whose subtle
defects show up as a slightly worse model rather than a wrong number. scikit-learn
remains a test oracle for Component 5's metrics as well. It ships no `py.typed`, so the
fitted estimator is treated as opaque behind a typed facade (`modeling.models.FittedModel`)
and a mypy override is declared for `sklearn.*`.

**Component 7 added three more** — xgboost, lightgbm and optuna — under ADR 0015's same
rule, and ADR 0016 records the reasoning. Both boosters are taken because the comparison
between them *is* the component's question rather than an implementation detail: XGBoost
grows depth-wise and LightGBM leaf-wise, so a single result would be a fact about one
library's inductive bias. All three are treated as untyped, so a fitted booster sits behind
`boosting.models.FittedBooster` and a mypy override covers `xgboost.*`, `lightgbm.*` and
`optuna.*`. **Every fit in the project is single-threaded** (`n_jobs=1`, plus LightGBM's
`deterministic` and `force_row_wise`), because a multi-threaded histogram reduction is only
approximately reproducible and this project's standard for "unchanged" is bit-identical.

---

## Current Data Flow

```text
Socrata SODA 2.1  (data.cityofchicago.org/resource/4ijn-s7e5.json)
   |
   |  1. GET ?$limit=1                     discover field list (22 fields)
   |  2. GET $select=<22> $order=inspection_id $limit=N $offset=M   per page
   |     retry 429/5xx/timeout; raise on other 4xx
   v
Page(records, offset, limit, field_names, field_types)      generator
   |
   |  validate: JSON array of objects, else raise
   |  warn on any divergence from the declared field list
   v
Polars DataFrame, every column Utf8; nested location -> JSON string
   |
   v
data/raw/food_inspections/food_inspections_<UTC>.parquet     (zstd)
data/raw/food_inspections/manifest_food_inspections_<UTC>.json
   |
   v
DuckDB read_parquet()  ->  named SQL queries  ->  CLI table


Component 2
-----------
data/raw/food_inspections/*.parquet
   |
   |  read 9 identity columns only; results/violations/risk/inspection_date
   |  are never read, which is the leakage boundary
   v
normalize -> 51,099 distinct identity nodes
   |
   |  block on (zip, house number), exact coordinate, licence
   |  every non-licence block requires location agreement
   v
evaluate 335,393 candidate pairs against named rules
   |
   |  vetoes first (V1-V4), then strong (S1-S3), probable (P1-P2),
   |  ambiguous (A1-A2, recorded but never merged), no-match (N0-N2)
   v
union-find -> cluster invariants -> deterministic split ladder
   |
   v
data/interim/entity_resolution/
   establishment_assignments_<UTC>.parquet    314,245 rows
   establishments_<UTC>.parquet                35,859 rows
   entity_resolution_edges_<UTC>.parquet       90,643 rows (audit trail)
   manifest_establishment_assignments_<UTC>.json


Component 3
-----------
raw Parquet + establishment_assignments
   |
   |  read 5 outcome columns; identity comes from Component 2, never re-derived
   v
eligibility gates
   |  era >= 2018-07-01 (172,879 excluded)
   |  type == CANVASS   (70,848 excluded)
   |  results in {Pass, Pass w/ Conditions, Fail} (12,091 excluded)
   v
parse violations -> PRIORITY / PRIORITY FOUNDATION / UNCLASSIFIED
   |  markers in comment text; narrative spans excluded;
   |  the violation NUMBER is deliberately not used
   v
collapse (establishment, date) -> one row, target = OR over the day
   |
   v
data/interim/target/
   inspection_targets_<UTC>.parquet          313,624 rows, 57,727 labelled
   manifest_inspection_targets_<UTC>.json


Component 4
-----------
raw Parquet + assignments + targets
   |
   |  one range join, one temporal condition:
   |     h.inspection_date < t.inspection_date   (strictly before)
   |  same-day records are never history -- dates carry no time component
   v
aggregate per (establishment, reference date)
   |  26 features: canvass history, priority history (code-era only),
   |  windows [d-N, d) for N in {365, 730, 1095}, all-type context,
   |  tenant change, observation window
   v
validate: boundary re-derived independently on all 57,727 rows
   |
   v
data/processed/features/
   as_of_features_<UTC>.parquet          57,727 rows x 33 columns
   manifest_as_of_features_<UTC>.json    pins all three input checksums


Component 6                                    Component 5
-----------                                    -----------
as_of_features_<UTC>.parquet
   |
   |  folds rebuilt from the data (never invented):
   |     17 quarterly + 1 covid_shift = 18
   v
for each fold, for each of 3 models:
   |  train  = assign_split(...) == "train", sorted canonically
   |  fit    = impute (train-only) -> scale (train-only) -> L2 logistic
   |  score  = predict_proba[:, 1] over window_frame(...)
   |  declare trained_through = fold.train_end   (never calibration_end)
   v
54 fits, no metrics computed here
   |
   v
data/processed/predictions/                    <-- ADR 0014, a third kind
   baseline_predictions_<UTC>.parquet     124,608 rows
   baseline_coefficients_<UTC>.parquet    1,530 rows
   baseline_training_log_<UTC>.parquet    54 rows
   manifest_baseline_predictions_<UTC>.json
   |
   |  sentinel evaluate --predictions <path>
   v                                           read_predictions
                                                  |
                                               validate_predictions (9 rules)
                                                  |  scores re-aligned BY ID,
                                                  |  never by row order
                                                  v
                                               same metrics + simulation as the
                                               six built-in heuristics
                                                  v
                                               data/processed/evaluation/
                                                  9 models scored


Component 7                                    Component 5
-----------                                    -----------
as_of_features_<UTC>.parquet
   |
   |  sentinel tune-boosting --trials 100
   |     region per fold set = first fold's train_start..calibration_end
   |        quarterly   2018-07-01..2022-03-31  <  first test 2022-04-01
   |        covid_shift 2018-07-01..2020-05-31  <  first test 2020-06-01
   |     6 / 2 inner rolling-origin folds; early stopping HERE and only here
   |     objective = mean evaluation.metrics.pr_auc over inner validation
   v
data/processed/tuning/                         <-- ADR 0018, a fourth kind
   tuning_trials_<UTC>.parquet            400 rows, 0 failed
   manifest_tuning_trials_<UTC>.json
   |
   |  the winning parameters are FROZEN BY HAND into
   |  boosting/definitions.py::TUNED_PARAMS  (a literal, so it appears in a diff)
   v
   |  sentinel train-boosting
   |  for each of 18 folds, for each of 3 models:
   |     train  = assign_split(...) == "train", sorted canonically
   |     matrix = 26 features + 4 indicators; NULL -> NaN, NOT imputed, NOT scaled
   |     fit    = frozen n_estimators rounds, NO eval_set, NO early stopping
   |     score  = predict_proba[:, 1] over window_frame(...)
   |     declare trained_through = fold.train_end   (literally true, not nearly)
   v
54 fits, no metrics computed here
   |
   v
data/processed/predictions/                    <-- own slug; C6's file untouched
   boosted_predictions_<UTC>.parquet      124,608 rows
   boosted_importances_<UTC>.parquet      1,620 rows (diagnostic, NOT attribution)
   boosted_training_log_<UTC>.parquet     54 rows
   manifest_boosted_predictions_<UTC>.json
   |
   |  sentinel evaluate --predictions <path>
   v                                           the same contract, unchanged
                                               data/processed/evaluation/
                                                  9 models scored
```

All four layers are written, and `data/processed/` now holds four kinds of thing: the
model-ready table (ADR 0011), model outputs (ADR 0014), measurements about models
(ADR 0013) and hyperparameter search trials (ADR 0018). **None of the last three may ever
be joined onto the first**, and no number in the tuning layer is a result — every one is
measured on a window that is training data downstream.

---

## Tests

**Command:** `uv run pytest` · **Result: 1,776 passed, 3 deselected** (2026-08-18)

Component 2 added 265 tests, Component 3 added 201, Component 4 added 202,
Component 5 added 278, Component 6 added 231, Component 7 added 235, Component 8 added
287. Quality gate,
all passing:

```bash
uv run pytest                  # 1,776 passed, 3 deselected, 332 s
uv run ruff check .            # All checks passed
uv run ruff format --check .   # 164 files already formatted
uv run mypy src/sentinel scripts   # no issues in 72 source files
```

| Component 7 area | Coverage |
|---|---|
| `test_boosting_definitions.py` (36) | registry and version stability; the import-time guard shown raising on eight distinct defects, including a search space that would reach a fixed determinism parameter and one that would search `n_estimators`; **the two searches proven concept-comparable** — every shared concept tuned in both libraries over identical ranges, `leaf_count` the only asymmetry |
| `test_boosting_preprocess.py` (18) | the boosted and baseline matrices proven column-identical; every nullable column reaching the estimator as NaN; a null boolean **not** filling to 0.0; the NaN mask equal to the frame's NULL mask cell-for-cell; two frames of different spread mapping a shared value identically (a scaler would not) |
| `test_boosting_train.py` (33) | `trained_through == train_end` for every registered model; **bit-identical scores on shuffled input**; the two libraries proven to differ; no `early_stopping_rounds` or `eval_set` on a final fit; score direction; the ablation differing from its donor by exactly the weight; every refusal |
| `test_boosting_tuning.py` (35) | every inner window proven to end before its fold set's first test start; the quarterly region proven to cover the covid_shift test window (the fact that forces two studies); `num_leaves` capped at 2^depth; a study reproducible at a fixed seed **and** different at a different seed; the frozen round count proven to come from early stopping rather than the cap |
| `test_boosting_leakage.py` (20) | the safety wall — appended future rows of either class, flipped future labels, a corrupted post-test feature, and deletion of the calibration window all leave a fold **bit-identical**; the objective's reachable row ids proven disjoint from every test window; a widened tuning region proven refused; plus a test that plants the label in a feature and proves the detector itself works |
| `test_boosting_contract.py` (18) | a real artifact read by `read_predictions`; validated against every fold; each of the contract's six rejections driven; the declared horizon proven strictly inside the contract's ceiling; the boosted and baseline schemas proven shaped alike |
| `test_boosting_build.py` (38) | end-to-end, three schemas on names *and* dtypes, sort keys, two runs identical, a shuffled input file changing no output, manifest provenance and checksums, dry-run, and **no metric Component 5 owns printed by the summary** |
| `test_boosting_inspector_blocked.py` (9) | the absence of an inspector field **re-derived** from the raw data contract, the ingested manifest, Component 4's features and every model's feature columns — so the block fails loudly if such a column ever appears |
| `test_cli_boosting.py` (28) | both new invocations; `--models` and `--fold-set` repeatability; the trial budget defaulting to the documented 100; exit codes without tracebacks; the untunable ablation refused; and the whole train→evaluate seam |

| Component 6 area | Coverage |
|---|---|
| `test_modeling_definitions.py` | registry and version stability, the 10-column nullable partition and its 4 families derived from `FEATURE_SPECS`, forbidden-column disjointness, the import-time guard shown raising (not asserting) on five distinct defects |
| `test_modeling_preprocess.py` | all four NULL rules exercised, matrix width constant whether or not a frame has nulls (the `add_indicator` regression), booleans filling to 0.0, medians fitted on train only, branch ordering that labels the coefficients |
| `test_modeling_train.py` | `trained_through == train_end`, the training anchor respected, **bit-identical coefficients on shuffled input**, score direction, non-convergence raising |
| `test_modeling_leakage.py` | the safety wall — two years of future rows, flipped future labels, a mutated future feature, and deletion of the calibration or test window all leave an earlier fold bit-identical; plus a test that proves the detector itself works |
| `test_modeling_validate.py` | every one of the 15 error checks shown failing on a deliberately broken input |
| `test_modeling_build.py` | end-to-end on a synthetic table, one-model-per-fold, artifact schemas and sort keys, manifest provenance, determinism, and acceptance by Component 5's contract |
| `test_evaluation_predictions.py` | the Component 5 seam — the flag is additive (byte-identical tables without it), scores aligned by id not row order, the four probability metrics, the horizon check shown failing |
| `test_cli_baselines.py` | both new invocations, `--models` repeatability, exit codes, and no metric printed by `train-baselines` |

| Component 2 area | Coverage |
|---|---|
| Normalization (103) | every name and address rule; parametrized unit markers, directionals, suffixes; idempotence properties; cases asserting digits and descriptive words are *not* stripped |
| Nodes and blocking (21) | signature collapse; node ids stable under row order; numeric id comparison; sentinel exclusion; oversized blocks skipped and reported; canonical pair ordering |
| Evidence (35) | one test per rule S1–N2 and per veto V1–V4, each veto pitted against otherwise-strong agreement; symmetry property; haversine against known distances |
| Union-find (10) | components identical under 10 seeded edge shuffles and reversed item order |
| Clustering (22) | id anchoring and numeric comparison; every invariant tripped; all three rungs of the split ladder; content-hash sensitivity |
| Validation (14) | a passing and a failing case for each error check; distributional checks proven non-fatal |
| Integration (25) | a 10-row scenario with a known correct grouping; schema contract tests on names *and* dtypes; manifest round-trip; dry-run writes nothing; empty and all-sentinel-licence inputs |
| Determinism | same input twice, and a seeded row-shuffled input, yield identical mappings — asserted in unit tests and verified separately on all 314,245 real rows |
| Regression (22) | 12 real cases copied verbatim from the snapshot, including the O'Hare over-merge the first run produced |

| Component 3 area | Coverage |
|---|---|
| Violation parsing (44) | entry splitting and structure; Priority Foundation before Priority; typo tolerance; every narrative exclusion, each with a test that it fires *and* one that it does not fire on a genuine citation; malformed entries kept |
| Construction (67) | every eligibility gate; every `results` value; the era boundary asserted at 2018-07-01 exactly; canvass vs re-inspection vs typo variants; Pass+null as a true zero and Fail+null as unknown; same-day collapse and OR semantics; numeric id ordering |
| Validation (24) | a passing and a failing case for each of the twelve error checks |
| Build (25) | a scenario with a known-correct label set; schema contract on names *and* dtypes; manifest pinning both input hashes; dry-run; empty input; **leakage guard** asserting no historical-aggregate column exists |
| Regression (33) | 12 real inspections copied verbatim, covering every `target_status` and both labels |

| Component 4 area | Coverage |
|---|---|
| **Leakage (12)** | its own file. Future insertion; future mutation; target self-exclusion; same-day exclusion — with a paired test proving a record one day earlier *is* counted, so the boundary is exclusive rather than absent. Plus determinism and a direct restatement of the invariant |
| Historical values (35) | counts, the inspected-only denominator, recency, rates, at-last flags, priority restricted to the code era, tenant change; every window boundary at exactly N, N−1 and N+1 days |
| Definitions (98) | every spec complete and self-explaining; no model-derived or demographic features; `inspection_date` and `code_era_phase` are keys/provenance, not features |
| Validation (25) | a passing and a failing case per error check, from deliberately corrupt tables — including recency rendered as 0 and priority counts rendered as 0 |
| Build (25) + CLI (7) | grain, schema contract on names *and* dtypes, manifest pinning all three input checksums, dry-run, empty input |
| Determinism | same input twice and a seeded row shuffle — asserted in unit tests and verified separately on all 314,245 real rows |

| Component 5 area | Coverage |
|---|---|
| **Leakage (19)** | its own file. Future data appended must not change an earlier fold's rows; future mutation must not change its statistics; calibration strictly between; the three splits partition; a declared training horizon past the fold is rejected; a label column in a prediction artifact is rejected. Plus a teeth test proving the checks can fail |
| Metrics (64) | **every metric cross-checked against scikit-learn** on random and heavily-tied inputs; PR-AUC asserted to be average precision and *not* the PR trapezoid; ECE and MCE against hand-computed values; degenerate inputs return `None` rather than an invented answer; a constant ranking scores exactly 0.5 ROC-AUC |
| Simulation (41) | the business-as-usual identity, including under heavy same-day ties; capacity conservation per schedule; analytic bounds optimal +1 / worst −1 / random ≈ 0 over 500 seeds; labels proven unchanged by reordering; tie-breaking independent of input order; every degenerate window |
| Folds (31) | calendar arithmetic including leap years; the spec's Fold 1 reproduced exactly; fold count derived from the data; partial windows excluded and reported; a leaky `FoldSpec` proven unconstructable |
| Contract (24) | each of the six rejection rules; row order immaterial; persisted artifacts round-trip and split by producer |
| Rankers (19) | each rule's ordering and its declared null rule; seed stability; no ranker claims to emit a probability; scoring independent of row order |
| Sensitivity (17) | a pure secular trend produces ~zero seasonal effect; a pure seasonal pattern is recovered; both together are separated; **business-as-usual labels are never re-drawn** (flip rate exactly 0); seed reproducibility |
| Validation (22) | severities; which failures are fatal; the report's PASS/FAIL/note rendering; offender capping; a tampered row count is caught |
| Build (29) + CLI (12) | six schemas asserted on names *and* dtypes; manifest pins its input and states its estimand; dry-run writes nothing; folds-only; too little data refused rather than fabricated |
| Determinism | two runs identical, and a seeded row shuffle identical — asserted per output table |

| Area | Coverage |
|---|---|
| Request construction | `$limit`/`$offset`/`$order`/`$select` params; app-token header; input validation |
| Pagination | multi-page offset walking; short-page and empty-page termination; mid-page truncation at `total_limit`; zero-limit; `$select` forwarded to every page |
| Retry | 500 then success; 429; timeout then success; exponential delays (1, 2, 4); bounded budget then raise |
| Non-retryable | 400 and 404 raise on the first attempt, no retry |
| Malformed responses | non-JSON; JSON object instead of array; array of non-objects; missing schema headers |
| Field discovery | discovery request sends no `$order`; skipped when disabled |
| Raw output | all-`Utf8` schema; nested `location` serialized; missing keys null; extra field kept; declared-but-absent field kept as null column; empty dataset; timestamped filenames; output-dir override |
| Manifest | full provenance; SHA-256 matches file on disk; JSON round-trip; schema reported as all `String` |
| DuckDB | row count; unique licences; grouping; `DESCRIBE` reports `VARCHAR`; latest-file resolution; error paths |
| CLI | scope flags required and mutually exclusive; limit resolution; `--log-level` on either side of the subcommand |

**Live tests:** `uv run pytest -m live` — **3 passed** (2026-08-15). Asserts the
real endpoint returns records and schema headers, that values are still
string-encoded, and that ordered pages do not overlap.

**Quality gates**, all passing as of 2026-08-15:

```text
uv run ruff check .            All checks passed!
uv run ruff format --check .   19 files already formatted
uv run mypy src/sentinel       Success: no issues found in 10 source files
```

---

## Verified Data

### Neural models (2026-08-18)

* 234 fits over 18 folds in 1,998.7 s, 4,306 epochs, CPU, one torch thread.
* 40 sweep trials over 2 studies in 550.9 s. Selected lr 3e-3 (quarterly), 1e-2
  (covid_shift); the specification's 1e-3 baseline is 0.0020 behind on quarterly.
* Quarterly means, 17 folds: `neural_numeric_only` NDE **0.2482**, ROC-AUC 0.6241, PR-AUC
  0.5343, Brier **0.2355**, ECE **0.0563**, precision@k_1_day 0.6273, +6.10 days earlier;
  wins 12 of 17 folds against `xgboost`.
* `neural_embeddings` NDE 0.2215 — **0.0267 below** the no-categoricals control.
* `xgboost_chain_embeddings` NDE 0.2444, PR-AUC **0.5357** (best of any model).
* Five-seed spread on `neural_embeddings`: PR-AUC 0.5202–0.5269, ROC-AUC 0.6060–0.6119
  (spread 0.0058). Per fold the ROC-AUC seed range averages 0.0178, max 0.0345.
* Seasonality redraw: `neural_numeric_only` observed 0.2482, p05 0.2311, p95 0.2527;
  `xgboost` observed 0.2376, p05 0.2224, p95 0.2444.
* covid_shift (1 fold): `neural_onehot` best at ROC 0.6456 / PR 0.6528.
* Chain embedding geometry: pairwise cosine mean 0.0018, SD 0.2508 — a random Gaussian
  table of the same shape gives 0.0000, 0.2504.
* Experimental categoricals: 57,727 rows, coverage 0.9881–0.9931, 401 rows with no prior
  inspection, minimum as-of lag **1 day**.

### Boosted models (2026-08-17)

Search: `sentinel tune-boosting --trials 100` — **400 trials over 4 studies, 0 failed**,
563.8 s. Training: `sentinel train-boosting` — 54 fits, 21.4 s, 124,608 prediction rows.
Evaluation: 68.4 s. Libraries xgboost 3.4.1, lightgbm 4.7.0, optuna 4.9.0, numpy 2.5.2.

Quarterly, mean over 17 folds:

| model | ROC-AUC | PR-AUC | NDE | days earlier | P@k_1day | found later |
|---|---|---|---|---|---|---|
| xgboost_class_weighted | 0.6195 | 0.5355 | **0.2390** | +5.85 | 0.6629 | 0.4283 |
| xgboost | 0.6188 | 0.5343 | 0.2376 | +5.83 | 0.6308 | 0.4289 |
| lightgbm | 0.6177 | 0.5342 | 0.2355 | +5.75 | 0.6598 | 0.4285 |
| logistic_regression (C6) | 0.6163 | 0.5321 | 0.2326 | +5.70 | 0.6576 | 0.4324 |
| prior_canvass_priority_rate | 0.5915 | 0.5012 | 0.1845 | +4.47 | 0.5551 | 0.4288 |
| business_as_usual | 0.5040 | 0.4347 | — | — | 0.4323 | — |

Per-fold NDE winners: **logistic_regression 7, xgboost 5, lightgbm 5** of 17. Seasonality
redraw (1,000 replications): xgboost [0.2224, 0.2444], logistic [0.2160, 0.2374] — the
intervals overlap and each observed value lies inside the other's range.

`covid_shift` (1 fold): lightgbm NDE 0.2585 and ROC-AUC 0.6292; logistic_regression
PR-AUC 0.6328 and precision@k_1day 0.9545. **No single winner** — the ordering is
metric-dependent.

Probability quality, raw and uncorrected: quarterly ECE 0.0621 (xgboost) / 0.0644
(lightgbm) / 0.0635 (logistic); covid_shift 0.1253 / 0.1518 / 0.1124.

Frozen hyperparameters (from `tuning_trials_20260817T155315Z.parquet`, sha256
`a77687b7…adec8b14`): xgboost/quarterly `max_depth=4, lr=0.193, n_estimators=103`;
xgboost/covid_shift `max_depth=3, lr=0.056, n_estimators=192`; lightgbm/quarterly
`max_depth=4, num_leaves=16, lr=0.299, n_estimators=63`; lightgbm/covid_shift
`max_depth=8, num_leaves=10, lr=0.058, n_estimators=54`. **Every study chose shallow
trees** — depth 3–4 in three of four — from a searchable range of 3–10.

Row-order sensitivity: shuffling the same 53,844 training rows moves a prediction by
**1.12e-01** (xgboost) and **1.23e-01** (lightgbm); re-sorting restores the fit
**exactly**. Component 6's coefficients moved by 7.049e-09 under the same treatment.

NaN routing: 3,404,772 NaN cells reached the estimators across 54 fits; 10 of 30 matrix
columns carry any NaN; 25.74% of rows have no code-era canvass history.

**Component 6 verified untouched:** re-running `train-baselines` under the current library
set reproduces `baseline_predictions` sha256 `a2bb9411…00ff5b44`, matching the committed
manifest byte for byte.


Development ingestion executed 2026-08-15
(`retrieved_at` `2026-08-15T14:57:03.089773Z`):

| | |
|---|---|
| Command | `sentinel ingest --limit 5000 --page-size 2000` |
| Rows | 5,000 |
| Pages | 3 (2,000 + 2,000 + 1,000 — final page truncated to the limit) |
| Columns | 22 (17 source + 5 `:@computed_region_*`), all `Utf8` |
| File size | 827,350 bytes |
| SHA-256 | `86573f20dbcfa522c305ae96d0f307998e074711e19196ddabfac759b88b31bf` |
| Date range in extract | 2010-01-04 → 2011-05-09 (oldest rows; ordered by `inspection_id`) |
| Unique `license_` | 3,703 of 5,000 rows, 0 null |
| Results breakdown | Pass 3,498 · Fail 1,133 · Pass w/ Conditions 322 · Out of Business 47 |

Dataset total as of 2026-08-15: **314,245 rows** (`$select=count(*)`).

---

### Full ingestion (2026-08-16) — now verified

| Property | Value |
|---|---|
| File | `food_inspections_20260816T070911Z.parquet` |
| sha256 | `7d3c4069340a68d197204c6cca9fca6399c6565bc3668760f145f43cd377ad38` |
| Rows | 314,245 |
| Pages | 7 (6 × 50,000 + a 14,245-row short page) |
| Size | 48,801,672 bytes |
| Wall time | 69 m 24 s (server-side; one recovered `ReadTimeout`) |
| Peak RSS | ~966 MB |
| Date range | 2010-01-04 → 2026-08-14 |

### Temporal evaluation (2026-08-16)

Run: `evaluation_folds_20260816T164834Z.parquet` and five sibling tables.

| | |
|---|---|
| feature rows read | 57,727 |
| folds | **18** — 17 quarterly + 1 covid_shift |
| test range | 2022-04-01 → 2026-06-30 |
| excluded partial window | 2026Q3 (snapshot ends 2026-08-14) |
| score producers | 6, none fitted |
| metric rows / curve points / simulation rows | 2,808 / 373,986 / 504 |
| random seeds | 20 (42…61) |
| sensitivity replications | 1,000, seed 20260816 |
| runtime | **164.2 s** |
| error-severity checks | **14 / 14 pass** |
| artifacts | 962 KB total |

**Baselines, 17 quarterly folds, mean ± SD:**

| schedule / model | NDE | mean days earlier | SD | worse | first half | ROC-AUC |
|---|---|---|---|---|---|---|
| optimal | 1.0000 ± 0.0000 | +24.75 | 14.95 | 0.0% | 1.000 | — |
| `prior_canvass_priority_rate` | **0.1845** ± 0.0404 | **+4.47** | 32.60 | **42.9%** | 0.576 | 0.5915 |
| `priority_at_last_canvass` | 0.1522 ± 0.0384 | +3.68 | 27.36 | 46.1% | 0.580 | 0.5747 |
| `days_since_last_canvass` | 0.0765 ± 0.0384 | +1.76 | 36.87 | 46.2% | 0.537 | 0.5381 |
| business_as_usual | 0.0066 ± 0.0422 | 0.00 | 0.00 | 0.0% | 0.504 | 0.5040 |
| random (340 fold-seeds) | −0.0016 ± 0.0271 | −0.19 | 36.05 | 49.3% | 0.499 | 0.5051 |
| worst | −1.0000 ± 0.0000 | −25.39 | 14.35 | 98.4% | 0.000 | — |

The analytic bounds land exactly and random averages −0.0016, which is the
strongest available evidence the area formula and its denominator are right.
`constant` scores ROC-AUC exactly 0.5000 with zero variance.

**Seasonality (de-trended, all rows):** peak August **+6.36 pp**, trough December
**−5.41 pp**, amplitude **11.77 pp**. Time invariance does not hold. The
sensitivity band over 1,000 label re-draws leaves the best baseline's NDE at
[0.172, 0.192] against an observed 0.1845 — the ranking's advantage survives.

**Distribution shift (`covid_shift`, one fold):** the baseline ordering inverts —
`days_since_last_canvass` is strongest here (NDE 0.170) and weakest on the
quarterly folds (0.077).

---

### As-of features (2026-08-16)

| Property | Value |
|---|---|
| Eligible target rows in | 57,727 |
| **Feature rows out** | **57,727** (0 unmatched) |
| Features | 26 in six families |
| Columns | 33 (3 keys + 26 features + 2 labels + 2 provenance) |
| Runtime | 15.6 s |
| Rows with no history at all | 401 (0.69%) |
| Rows with no prior canvass | 5,615 (9.73%) |
| Rows with no prior code-era canvass | 14,162 (24.53%) |
| Rows after a business-name change | 1,962 (3.40%) |
| Output | `data/processed/features/` |

Key facts driving the design: **`inspection_date` has exactly one distinct time
component** across all 314,245 rows, so same-day records cannot be ordered and the
boundary must be strictly `<`; **43 same-day canvass re-inspections** sit at
reference dates and provably follow their canvass; the **canvass cycle has a
358-day median**, so a 365-day window is empty for 62% of rows; the **any-type
interval has a p25 of 9 days**, the re-inspection pattern, which is why
`days_since_any_inspection` is labelled as policy-encoding context.

`days_since_last_canvass` has a minimum of **1** and no zeros — the strict
boundary makes a zero-day recency unconstructable.

### Target construction (2026-08-16)

| Property | Value |
|---|---|
| Target rows | 313,624 |
| **Eligible (labelled)** | **57,727** |
| Positive / negative | 30,316 / 27,411 |
| **Positive rate** | **52.52%** |
| Establishments with ≥1 eligible row | 15,144 |
| Runtime | 25 s |
| `ineligible_era` (pre 2018-07-01) | 172,879 |
| `ineligible_type` (not a canvass) | 70,848 |
| `ineligible_result` (no inspection) | 12,091 |
| `unknown_violations` | 79 |

Positive rate by year: 87.6% (2018 H2) · 77.4% · 59.4% · 50.3% · 46.5% · 46.1% ·
42.6% · 39.2% · 39.1% (2026). Component 5 must account for this drift.

Key data facts driving the design: **Chicago replaced its violation scheme cleanly
on 2018-07-01** (June: 0 rows with Priority terminology, 415 with Critical/Serious;
July: 761 and 0); **`results` has seven values, not four**; **`Pass w/ Conditions`
carries priority violations 97.9% of the time** against 0.5% for `Pass`; the
violation number does *not* encode severity; **24.9% of `Out of Business` records
are followed by another inspection** at the same premises.

### Entity resolution (2026-08-16)

| Property | Value |
|---|---|
| Nodes | 51,099 |
| **Establishments** | **35,859** |
| Distinct usable licences (comparison) | 48,963 |
| Reduction ratio | 0.73 |
| Candidate pairs | 335,393 (29,280 strong, 2,915 probable, 747 ambiguous) |
| Runtime | 43 s |
| Single-inspection establishments | 6,084 — down 51% from 12,356 under naive licence grouping |
| Rows with unusable licence | 850 (0.27%) |
| Oversized blocks / blacklisted coordinates | 0 / 0 |

Key data facts driving the design: **18.47% of (name, address) pairs hold more
than one licence** (max 47); the `'0'` licence sentinel covers 323 distinct
names; case and whitespace alone collapse 33,261 address strings to 20,313;
coordinate spread within an address is exactly 0 m; 75.5% of same-place licence
pairs overlap in time rather than succeeding one another.

---

## Known Issues

### Component 7

1. **The improvement over Component 6 is small and not clearly real.** NDE 0.2326 → 0.2376
   is +2.1% relative, and the logistic model's observed value sits *inside* XGBoost's
   seasonality redraw interval [0.2224, 0.2444]. Component 6's gain over the heuristics
   survives the same test; this one does not clearly. Reported as a small improvement, not
   a decisive one.
2. **The logistic baseline wins 7 of 17 quarterly folds** (xgboost 5, lightgbm 5). The tree
   models win by more when they win (+0.047 at 2024Q2) than they lose by (−0.017 at 2025Q4),
   which is what produces the positive mean. A mean improvement is not a per-quarter one.
3. **The two fold sets use different hyperparameters**, by design (ADR 0017), so a
   quarterly-versus-shift comparison confounds the regime with the parameter set. They are
   reported separately and never averaged.
4. **`covid_shift` was tuned on two inner folds** — its eight-dimensional parameter set is
   less well determined than the quarterly one, and its results are a robustness observation
   rather than a measurement.
5. **Importances are a diagnostic, not an attribution.** Condition number 71.8 and one
   feature pair at 0.9888 correlation mean a tree splits credit according to which feature
   it reached first. Component 11 owns attribution; SHAP is deliberately not implemented.
6. **Every fit is single-threaded**, which is what makes them bit-reproducible. That will
   not scale to a snapshot an order of magnitude larger, and relaxing it means giving up
   bit-identity — a decision that needs its own ADR rather than a quiet change.
7. **Determinism holds within a fixed library set only.** A version bump may move every
   number in the findings document; the manifest records the versions so that is detectable.
8. **Inspector-effect modelling is blocked** — the dataset has no inspector field (ADR 0019).
   This is the project's most consequential blocked experiment: the gap between "a violation
   was cited" and "the establishment was unsafe" cannot be characterised anywhere in Sentinel,
   and Component 12's fairness audit inherits the same limitation.

### Component 6

1. **Coefficients are not feature importances.** Condition number 71.8, and
   `prior_canvass_count` / `prior_canvass_inspected_count` are correlated at 0.9888
   while carrying mean coefficients of +1.99 and -1.47 — one effect split across two
   terms. Seven of thirty terms change sign across folds, all with mean magnitude below
   0.118. Component 11 owns attribution.
2. **43.24% of violations are still discovered later** than business-as-usual under the
   best model, marginally worse than the best heuristic's 42.88%. Re-ordering under fixed
   capacity is zero-sum; any days-earlier headline must carry this number.
3. **PR-AUC 0.5321 against a 0.4307 floor is a modest gain.** ROC-AUC 0.6163 is a weak
   classifier by any general standard. Both are reported because Component 5 reports
   them, not because they are the operative quantity.
4. **Probabilities are uncalibrated** (ECE 0.0635, MCE 0.1664). Component 9 owns this.
5. **Bit-reproducibility is conditional** on library versions and BLAS thread count, not
   just on input. Recorded per run in the manifest. Verified on scikit-learn 1.9.0 /
   numpy 2.5.2.
6. **The missing-indicator encoding captures a level shift, not a differing slope.** An
   interaction in the missing group cannot be represented by this model class.
7. **`priority_at_last_canvass` sits 0.0056 from the median-fill boundary** (0.5056 in
   the last fold, drifting down from 0.6310). The constant-0 rule for nullable booleans
   exists because of this; if that rule is ever revisited, read findings §3.3 first.
8. **Model selection on the quarterly folds picks the wrong model for `covid_shift`.**
   Measured, not hypothetical. This is a property of the problem, not a defect.

### Component 5

1. **17 folds is not many.** Fold-to-fold SD is reported for every metric, but a
   genuinely unusual quarter moves the mean visibly. NDE for the best baseline
   spans 0.119 to 0.262 across folds.
2. **Folds are not independent samples.** The same premises appears in many test
   windows over eight years, so the reported SD is a fold-to-fold spread, not a
   confidence interval.
3. **The `covid_shift` fold is a single fold** and carries no variance estimate.
   Read it as an illustration.
4. **Calibration windows are unused** until Component 9 exists. Built early on
   purpose so calibration cannot end up on test.
5. **`days_since_last_canvass` is not the spec's "days overdue"** — the CDPH risk
   category is in raw but not in the feature table, and Component 5 may not add
   features. It measures elapsed time, not deficit against a statutory deadline.
6. **Temperature attribution is BLOCKED.** The 11.77 pp seasonal effect confounds
   temperature with daylight, holidays and staffing.
7. **The full run takes 164 s**, dominated by 1,000 sensitivity replications
   across 18 folds and 3 baselines in pure Python. `--sensitivity-replications`
   lowers it; `--folds-only` runs in under a second.
8. **Intra-day order is unrecoverable**, so business-as-usual has no within-day
   resolution and the tie-break within a date is arbitrary-but-deterministic.
   NOT VERIFIED that this is immaterial at finer capacity granularity than a day.

### Component 4

1. **The boundary discards some genuinely prior information** — up to 1,075
   same-day licence inspections that may have preceded their canvass. Deliberate.
2. **Priority features are NULL for 24.5% of rows** (14,162 with no prior
   code-era canvass). Correct, but a quarter of the table carries no priority
   history.
3. **A 365-day window is empty for 62% of rows**, a consequence of the 358-day
   median cycle.
4. **`days_since_any_inspection` partly encodes scheduling policy**, not risk.
   Labelled and separable, not removed.
5. **History can span a tenant change** for 15.9% of rows. Exposed via two
   features rather than resolved; "a rename means a new business" is **NOT
   VERIFIED** in this data.
6. **No text, spatial, weather or licence features.** Their datasets are not
   ingested, so adding them is a Component 1 extension.
7. **Feature usefulness is unmeasured.** No model exists, and selecting features
   by downstream accuracy would itself be leakage. **NOT VERIFIED** that any
   feature predicts the target.

### Component 3

1. **52% of the dataset cannot be labelled.** 172,879 rows predate the
   2018-07-01 code change, where Priority violations are undefined. They remain
   usable as *features* — only the *label* is impossible.
2. **The base rate drifts from 87.6% to 39.1%.** Flagged via `code_era_phase`,
   not corrected. Component 5 must handle it.
3. **The narrative-exclusion list is judgement.** Four patterns affecting 74
   entries and 10 inspection labels, enumerated in the data contract.
4. **8 `Pass w/ Conditions` rows are labelled negative** where the result implies
   otherwise, because the parser stays independent of `results`.
5. **Inspector write-up variation is unmeasurable.** A priority violation found
   but not labelled is a false negative and the open data has no ground truth to
   check against. **NOT VERIFIED.**
6. **Severity within positive is not represented.** One priority violation and
   twelve produce the same label.

### Component 2

7. **Same-name outlets at a dense address can still merge.** Two outlets of one
   chain at one address with no store number and no distinguishing `aka_name`
   are indistinguishable from the data. `MCDONALD'S` at O'Hare (22 nodes, 20
   licences, 5 names) is the known example. Bounded to mega-addresses and
   surfaced by the cluster-size and address-density checks.
2. **747 ambiguous pairs have never been manually adjudicated.** They are the
   intended review queue, recorded in the edges table.
3. **`establishment_id` is not stable across snapshots.** It is a deterministic
   function of one input file. A later snapshot can merge or split clusters,
   retiring ids. A crosswalk is future work (ADR 0007).
4. **Stadiums and arenas resolve to one establishment.** The United Center is
   one establishment holding 16 licences. Whether an arena is one premises or
   many is definitional, not a defect.
5. **No `--as-of` resolution mode.** Identity is reconstructed from the whole
   snapshot. This is argued to be legitimate rather than leakage, but a strict
   as-of mode is named as future work.
6. **All pages are held in memory before writing.** `ingest_food_inspections`
   accumulates every page's records and writes one Parquet at the end. Measured
   at ~966 MB peak for the full pull, so it was left alone; it would need
   revisiting if the dataset grew several-fold.
7. **A dev extract is not a random sample.** Ordering by `inspection_id`
   correlates with time, so `--limit N` returns the oldest N inspections. Do
   not estimate distributions or train on a dev extract.
8. **No incremental ingestion.** Every run is a full pull of whatever scope is
   requested. `X-SODA2-Truth-Last-Modified` and `ETag` are documented as leads
   but unused.
9. **No cross-run deduplication.** Two runs produce two independent files with
   overlapping content, by design (ADR 0005).
10. **Schema divergence warns, it does not fail.** An added or removed upstream
   column is logged as a warning and the data is kept. Whether that should be
   fatal is an open question for a later component.
11. **A same-second re-run can collide.** Filenames have one-second resolution.
   Two ingestions starting within the same second would target the same path
   and the second would overwrite the first. Not observed; would need
   sub-second or a counter to eliminate.
12. **CI has never run.** The workflow is committed but no push has yet
   triggered GitHub Actions. **NOT VERIFIED.**

---
## Next Component

**Component 10 - Inspector-effect modelling. BLOCKED.**

ADR 0019 stands: the dataset has 22 columns and no inspector field. Nothing in Component 9
changes that. The next *implementable* component is 11 (SHAP) or 12 (fairness), and Component 12
has an input waiting for it - Component 8's community-area ablation, built for exactly that
purpose under ADR 0023.

### What the next component consumes

* `data/processed/predictions/calibrated_predictions_<stamp>.parquet` - **start here, not from
  Components 6-8's raw scores.** 207,680 rows, five calibrated models x 18 folds, readable by
  `evaluation.contract.read_predictions` with no translation.
* `data/processed/calibration/calibrator_parameters_` and `calibrator_isotonic_breakpoints_` -
  enough to re-apply any calibrator without re-running Component 9.
* Everything Components 2, 3, 4 and 5 already own, unchanged.

### What it must not redo

* **Do not recalibrate.** The calibrators are frozen per (model, fold). Refitting one on a test
  quarter - including "just to check" - reintroduces the leak ADR 0012 built the calibration
  window to prevent.
* **Do not re-fit a base model.** Component 9 re-executed them to recover a missing recording,
  under a bit-identity gate. That is not licence to revisit them.
* **Do not read `trained_through` as `train_end`.** On the calibrated artifact it is
  `calibration_end`; the estimator's horizon is in `base_model_trained_through`.
* **Do not average `covid_shift` into a quarterly mean.** Its probabilities are the least
  trustworthy in the project (slope 0.75-0.90, ECE roughly double quarterly).
* **Do not promote `xgboost_chain_embeddings`** on the strength of its calibrated Brier. It is an
  experimental Component 8 derivative (ADR 0022) that lost on NDE.

### Still open

* **Which model should Sentinel carry forward?** MEMORY open question 13. Component 9 sharpened
  it without settling it: `neural_numeric_only` still wins NDE (0.2482) but now has the
  *second-worst* calibrated ECE, while `xgboost` has the best (0.0474). The four families remain
  within 0.0156 NDE, and the neural advantage remains smaller than its own seed noise.
* **Seed averaging**, deferred by decision - it would create a base model Component 8 never
  evaluated and break the bit-identity gate - rather than by oversight.
* **The retraining trigger** proposed in `calibration_findings.md` 12.8 is a design proposal and
  has not been validated against any outcome.

### What must not be re-derived

Identity is Component 2's, labels are Component 3's, features are Component 4's, the evaluation
contract is Component 5's, the fitting/prediction contract is Component 6's, the tuning protocol
is Component 7's, the experimental categorical layer is Component 8's, and the calibrators are
Component 9's. Join on `target_inspection_id`; do not recompute any of them.

**Nothing in `data/processed/evaluation/`, `data/processed/predictions/`,
`data/processed/tuning/`, `data/processed/neural/` or `data/processed/calibration/` may be
joined onto a training table.** ADR 0013, ADR 0014, ADR 0018, ADR 0022, ADR 0024.
