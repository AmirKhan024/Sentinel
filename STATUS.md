# Current Status

**Last updated:** 2026-09-05
**Current component:** 21 of 21 (of the operational-planning extension; see "Not Started" below
for the roadmap correction) — Supervisor plan review, adjustment, and approval ✅, followed by a
final product-coherence completion pass, fixes for two real Side A/Side B conflation bugs
("Today = April 1, 2026" and broken establishment navigation), and a final acceptance audit that
found and fixed two more real gaps (see below) — Sentinel is now considered **COMPLETE** for its
current scope.
**Full backend regression suite (2026-09-05): 3,384 tests pass, 3 deselected** — re-run twice, the
second time covering the audit pass's `geographic_organization` fixes; identical count both times.
`ruff check`/`ruff format --check`/`mypy` clean on `src/sentinel/` except the pre-existing,
documented Component 9 calibration-file formatting exception and the one unrelated
`entity_service.py` E501. Frontend: **120 tests pass** (`tsc --noEmit` and `oxlint` both clean).
See "The Sentinel Frontend" → "The 'Today = April 1, 2026' bug", "Fixed broken establishment
navigation", and "Final acceptance audit" below
for what changed most
recently.
**State (as of 2026-08-28, superseded by Components 17-21 below):** Components 1–9 and 11–14 and
16 complete and verified against the full 57,727-row feature table; Component 10 blocked (ADR
0019), Component 15 blocked (ADR 0019, ADR 0043). An
end-to-end integration verification (2026-08-27) exercised the real CLI, API and frontend against
committed artifacts and found and fixed two real integration bugs invisible to unit tests alone
— see `docs/analysis/integration_verification_20260827.md`. A follow-up frontend product-clarity
pass (2026-08-28) rebuilt the primary UI in plain language for a non-technical inspection
supervisor, found and fixed one more real bug in the process, and changed no backend contract —
see `docs/analysis/frontend_product_clarity_20260828.md`.
**3,190 tests pass**, 3 deselected; ruff and mypy (strict, `src/sentinel`) are clean. `ruff format
--check` fails on the same **10 Component 9 files** it has since Component 9 closed — 5 in
`src/sentinel/calibration/`, 4 calibration test modules, and `scripts/profile_calibration.py`.
Pre-existing, recorded in HANDOFF §16d, verified untouched by this work, and deliberately not
fixed here because Component 9 is closed. Every
Component 12, 13, 14 and 16 file is formatted.
`uv run mypy src/sentinel` (the CI gate) is clean; `uv run mypy` over `scripts/` as well
reports 14 pre-existing `attr-defined` errors in three older profiling scripts —
`profile_boosting.py`, `profile_calibration.py`, `profile_explanations.py` — none of them
Component 13's, and `scripts/profile_policy.py` is clean.

**Component 14's measured answer:** the coverage reserve Component 13 went to the most trouble
to make explicit is **substantially notional once a real calendar is applied**. Component 13
fills the risk block at ranks 1..n_risk and places the reserve after it, so the reserve is
*always* the tail of the rank order — checked, with no exception in any of the 273 reserve-bearing
cells. A strict-priority schedule fills the horizon from the top, so when the horizon falls short
the rows that fall off the end are the reserve rows, every time. **1,012 of 3,459 coverage-reserve
slots — 29.3% — are lost to the horizon; 136 of 273 cells lose some and 91 lose it entirely.**
Neither layer is wrong on its own terms: ADR 0037 priced the reserve in forgone citations and
granted it a slot count, and nothing in that decision said the slots had to sit at the *end* of
the queue. Component 14 **reports this and does not correct it** — promoting reserve rows would
be re-ranking, which Component 13 owns.

The second measurement is the one that decided the component's shape. Component 13's cutoffs
descend from a quarter-wide median, so a schedule built on that same median is feasible before
anything is measured. Built on the days Chicago actually worked it is not: **44 of 90 (fold,
capacity) cells cannot fit their approved queue into their own horizon — 784 inspections — while
under the flat median the backlog is zero in every cell.** Both modes ship, and the scenario is
labelled a scenario everywhere it appears.

**Component 13's measured answer:** the intervention everyone's intuition demands was aimed at
a problem that is not there. Establishments with no code-era inspection history are 10.4% of the
candidates and already take **40–58% of the top of the queue** under pure risk ranking — a
selection ratio of **3.96 to 5.57** across all four candidate models — because their citation
rate is genuinely higher (0.4883 against 0.4283). A coverage floor is therefore inert in **338
of 340** quarterly cells at the population share, and a forced reserve at twice that share buys
343 additional low-information inspections for **34 forgone Priority citations a week**. Seven
policies were compared and **no winner was declared**: the trade-off is published and the choice
is left as a governance decision. `xgboost_platt` is selected as the production model on
calibration, after Component 5's own sensitivity bands showed all four candidates
indistinguishable on discovery efficiency.

**The measured answer:** Sentinel's ranking works, and it works measurably less well in some
Chicago neighbourhoods than others. Within-group ROC-AUC spans **0.509 to 0.710** across the 51
supported community areas — a spread more than twenty times the difference between the project's
best and worst *model*. Component 9's global calibration improvement **did not reach every
group**: for `xgboost` and `lightgbm` 25 of 33 supported community areas improved, and for
`neural_numeric_only` only **17 of 33** did. And the 405 rows with no recoverable geography —
which are the rows with no inspection history — are ranked at chance (0.509), selected at
**0.20×** the city rate, and have **0.6%** of their violations captured by the top 5% against a
city-wide 7.0%.

None of that establishes discrimination, causality or a protected-class finding, and ADR 0035
says so in those words. **A green run means the audit is sound, not that Sentinel is fair.**

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

## Component 11 — Explainability and Feature Attribution ✅

Answers "why did Sentinel score this establishment this way?" for the models Components 6–8
fitted and Component 9 calibrated. Reads existing models and existing predictions; changes
neither. **Fits nothing, selects nothing, alters no prediction.**

### What it implemented

* `src/sentinel/explain/` — 11 modules: `definitions.py` (the support matrix and every frozen
  constant, import-time guard), `models.py`, `refit.py`, `background.py`, `sample.py`,
  `attribute.py`, `aggregate.py`, `validate.py`, `writer.py`, `figures.py`, `build.py`.
* One command: `sentinel explain`.
* A **seventh** processed layer, `data/processed/explanations/`, holding seven tables.
* 173 new tests (1,911 → 2,084), 19 error-severity checks + 1 advisory, 29 figures.
* ADR 0028 (the layer), 0029 (re-execution under ADR 0026's gate), 0030 (the attribution
  protocol), 0031 (the unsupported model).
* `scripts/profile_explanations.py` — 9 read-only profiles, run **before** the implementation,
  which is what fixed every frozen constant.

### The support matrix

| model | method | exact |
|---|---|---|
| `logistic_regression` | closed form, `coef_j * (z_j − E[z_j])` | yes |
| `xgboost` | native TreeSHAP (`pred_contribs`) | yes |
| `lightgbm` | native TreeSHAP (`pred_contrib`) | yes |
| `neural_numeric_only` | antithetic permutation, 8 rounds | **no** |
| `xgboost_chain_embeddings` | — | **unsupported** (ADR 0031) |

`xgboost_chain_embeddings`'s fitted booster is reachable only through
`neural.embed._scorer_for`, a private process-local stash. Component 8 is closed, so it is
reported unsupported with the measurement as its reason and the four-line public accessor
proposed rather than taken. Its rows carry **nulls, never zeros** — zero is a legitimate
attribution and a table of them would read as "this model used no features".

### Results, measured (2026-08-25)

72 re-executed fits in 438.6 s, attribution 647.3 s, ~19 minutes wall clock. **166,144 /
166,144 test scores bit-identical to the committed artifacts, zero mismatches** (ADR 0029).
21,600 explained predictions, 648,000 attribution values. All 20 checks pass.

**The headline is a disagreement.** Cross-model rank correlation of the quarterly importance
rankings:

| pair | rank ρ | top-10 Jaccard |
|---|---:|---:|
| `lightgbm` vs `xgboost` | **0.9871** | 0.818 |
| `logistic_regression` vs `neural_numeric_only` | 0.8033 | 0.667 |
| `neural_numeric_only` vs `xgboost` | 0.6822 | 0.667 |
| `lightgbm` vs `logistic_regression` | **0.4351** | 0.333 |

Component 8 measured these four models landing within **0.0156 NDE** of one another.
Component 11 shows those near-identical scores come from **materially different reasoning**.
That strengthens Component 7's "the ceiling is the representation, not the estimator", and it
removes any argument from explanation to model choice — there is no "the models agree" story.

**What they leaned on.** Prior inspection history dominates every model:
`prior_canvass_count_code_era` is rank 1 for three of four, `prior_canvass_count` for the
fourth. Notably, the **missingness indicator** `missing_no_code_era_canvass` ranks 3rd for the
logistic model and 2nd for the network — the *absence* of a record is among the most
informative things these models have.

**Stability.** Consecutive-fold rank ρ 0.9606–0.9753 for the tree and linear models, **0.8914**
for the network. First fold to last (four years apart) falls to 0.7495–0.9197 with top-10
overlap 0.538–0.818. Quarter to quarter the reasoning holds; over four years it measurably
drifts, and between a third and a half of each model's top ten changed.

**COVID, and it confirms Component 6.** Under the regime shift three of four models leaned
2–3× harder on `days_since_any_inspection` (xgboost 0.1499 → **0.3728**, rank 3 → **1**;
network 0.1232 → **0.3546**, rank 8 → **1**). Component 6 found the model *ordering inverting*
on that fold and inferred from an ablation that the feature encodes scheduling policy.
Component 11 shows the mechanism directly.

### Artifacts

```text
data/processed/explanations/
  explanation_values_<UTC>.parquet               648,000 x 20   the long grain
  explanation_cases_<UTC>.parquet                 21,600 x 36   additivity + provenance
  explanation_importance_<UTC>.parquet             2,400 x 16   per fold and per fold set
  explanation_stability_<UTC>.parquet                 68 x  9   rank agreement
  explanation_drift_<UTC>.parquet                    120 x 15   per-feature rank travel
  explanation_representative_cases_<UTC>.parquet      12 x 14   high / medium / low
  explanation_support_<UTC>.parquet                    5 x 15   incl. the unsupported model
  manifest_explanation_values_<UTC>.json
```

### Verification

```bash
uv run python scripts/profile_explanations.py   # read-only, fixes the constants
uv run sentinel explain --dry-run --report      # writes nothing
uv run sentinel explain --report                # ~19 min; NO OMP_NUM_THREADS override
```

Tests with teeth: a background row dated after `train_end` turns the temporal check red; a
later fold's background handed to an earlier fold is rejected; one recorded score perturbed by
a **single ULP** turns the identity check red; `feature_127` is rejected; a fabricated
attribution for the unsupported model is rejected; and `permutation_shap` is cross-checked
against **brute-force Shapley enumerated over all 2^M subsets** — the definition itself, not
another library. `tree_shap` and `linear_shap` match `shap` (dev-only) to **0.0**.

---

## Component 12 — Fairness and Geographic Equity Audit ✅

**Question:** Sentinel decides who gets inspected first. Does it behave the same way
everywhere in the city?

**Answer:** No, and in four separate ways that are measured independently rather than
collapsed into a score.

### What it implemented

* `src/sentinel/fairness/` — 14 modules: `definitions.py` (the group registry, including the
  **refused** definitions, and every frozen constant behind an import-time guard), `models.py`,
  `groups.py` (the group frame and its temporal proof), `support.py`, `metrics.py`,
  `priority.py`, `missingness.py`, `attribution.py`, `disparity.py`, `drift.py`, `validate.py`,
  `writer.py`, `figures.py`, `build.py`.
* One command: `sentinel audit-fairness`.
* An **eighth** processed layer, `data/processed/fairness/`, holding ten tables.
* 247 new tests (2,084 → **2,331**), 13 error-severity checks + 3 advisories, 72 figures.
* ADR 0032 (the layer), 0033 (the group frame and the refused geographies), 0034 (the support
  policy and the advisory boundary), 0035 (what this component does not claim).
* `scripts/profile_fairness.py` — 10 read-only profiles, run **before** the implementation,
  which is what fixed every frozen constant.

### The first component that re-executes nothing

Component 9 had to regenerate scores that were never recorded; Component 11 had to regenerate
the models themselves. Both did it behind ADR 0026's bit-identity gate. Every input Component
12 needs already exists on disk, so its integrity claim is the opposite one — **nothing
moved** — checked by re-reading every input's sha256 after the last table is written. No
refit, no gate, no thread sensitivity.

### What the data allowed

| geography | status | why |
|---|---|---|
| `community_area` | **audited** | boundaries fixed since the 1920s; ADR 0023 handed it over explicitly |
| `zip` | **audited** | better supported — 56 of 69 clear the floor against 51 of 78 |
| ward | **refused** | the two published ward layers disagree on **98.3%** of rows: a ward id is a property of a boundary *version*, not of a place |
| census tract | **refused** | 797 groups over 32,696 rows |
| point geography, city/state, facility type | **refused** | ADR 0033 records each |

**The as-of geography and the row's own recorded geography disagree on 0 of 57,041 community
area rows and 0 of 57,326 ZIP rows**, so the temporally safe choice cost nothing. The refusals
are rows in `fairness_group_definitions`, not sentences in a document.

### Support came before metrics, and it is why the component is shaped this way

The median (fold, community area) cell holds **16 rows**; 4 of 1,288 clear the 200-row floor.
So the reporting grain is the pooled fold set — still strictly held out, and labelled on every
row as *the system as operated over 2022Q2–2026Q2* rather than one estimator.

Floors frozen from the profiler before any result: 200 rows / 20 positives / 20 negatives for
ranking, **300** for calibration (15 equal-mass bins × 20 rows). **The bin count was not
reduced to let more groups through**, because a group ECE at a different bin count would be
incomparable with Component 9's global one — the exact comparison the component exists to make.

Pooled quarterly: `community_area` 51 of 78 supported (33 for calibration); `zip` 56 of 69 (41).
**The 27 excluded community areas are rows with real counts and a stated reason, never absent
rows**, and a check compares the support table against the values observed in the data.

### Results, measured (2026-08-25)

**5 models × 2 geographies × 18 folds, 207,680 audited rows, 145 s.** All 13 error checks pass;
13 advisory findings; inputs byte-identical before and after.

**Before any model is involved**, the outcome rate spans **0.2200 → 0.5658** across supported
community areas against a city-wide 0.4283. A working risk model is therefore *expected* to
select at different rates; parity would mean ignoring a measured difference.

**Ranking (calibrated, pooled quarterly, 51 community areas):**

| model | min | max | spread |
|---|---|---|---:|
| `xgboost_chain_embeddings_platt` ⚠ | 0.5112 (`__UNKNOWN__`) | 0.7094 (ca 53) | **0.1982** |
| `xgboost_platt` | 0.5092 (`__UNKNOWN__`) | 0.6971 | 0.1879 |
| `neural_numeric_only_platt` | 0.5322 (`__UNKNOWN__`) | 0.7095 | 0.1773 |
| `lightgbm_platt` | 0.5178 (`__UNKNOWN__`) | 0.6911 | 0.1733 |
| `logistic_regression_platt` | 0.5241 (ca 45) | 0.6880 | 0.1640 |

The within-city spread is **larger than the entire difference between the best and worst model
in this project**, and community area 53 is the best-ranked group for all five.

**Calibration — the finding section 18 of the brief exists to surface.** Component 9 cut global
quarterly ECE by 20–25%. Per group it did not reach everyone:

| model | community areas improved | mean ECE base → calibrated |
|---|---:|---|
| `lightgbm_platt` | 25 / 33 | 0.0934 → 0.0828 |
| `xgboost_platt` | 25 / 33 | 0.0948 → 0.0844 |
| `logistic_regression_platt` | 23 / 33 | 0.0966 → 0.0854 |
| `neural_numeric_only_platt` | **17 / 33** | 0.0884 → 0.0862 |

Coherent with Component 9 rather than against it: the network had the best *uncalibrated* ECE
and least to gain. **Nothing was fixed** — a per-group calibrator would change Component 9 and
is a fairness decision disguised as a repair (ADR 0034).

**The sharpest finding is a group with no geography at all.** 405 quarterly rows carry
`community_area = __UNKNOWN__`, and the chain closes with every link measured:

```text
59.5% no prior inspection of any kind (0.74% overall)  -- 80x
61.7% no code-era canvass history     (10.4% overall)  --  6x
      -> ROC-AUC 0.509 (chance) -> selected at 0.20x -> 0.6% capture vs 7.0% city-wide
```

Of its 166 real violations the top 5% found **one**. This is the missingness indicator
Component 11 ranked 2nd/3rd in importance, resolved by neighbourhood. It is a measurement: "we
have never inspected this place" is a true and relevant fact, and no causal direction is claimed.

**Capture ranges 0.006 → 0.151 across supported community areas** against an overall 0.070, and
selection rate and capture are reported separately throughout — a group can be prioritised often
and still have its violations missed.

**Temporal drift could not be answered.** Exactly **one** quarterly fold per (model, geography)
has enough support to compute a disparity at all, so every series is `insufficient_folds` rather
than a fitted line. `DRIFT_MIN_FOLDS = 3` was frozen before any series existed.

**`covid_shift`, separately.** 8,840 rows — more than any single quarter — supporting **11 of 78**
community areas and 5 for calibration. Reported as a stress-test observation; no trend claimed.

### The advisory boundary, and why it is the most deliberate decision here

```text
the audit is WRONG   -> the build fails      (13 checks)
the world is UNEVEN  -> recorded, exit 0     (3 advisories)
```

**No measured disparity can fail the build, and there is no flag to make one.** A red build is
a demand for action, and the actions available are to change the model, the metric or the
threshold — two of which are worse than the disparity. ADR 0034.

`test_an_enormous_disparity_is_advisory_and_never_an_error` asserts a 0.95 ECE spread leaves
every error check green and the exit code zero. That test is what keeps the component honest.

### Artifacts

```text
data/processed/fairness/
  fairness_group_metrics_<UTC>.parquet          122,850 rows   the long grain
  fairness_priority_audit_<UTC>.parquet         136,850        selection AND capture
  fairness_group_support_<UTC>.parquet                         every observed group
  fairness_group_calibration_<UTC>.parquet                     base -> calibrated, per group
  fairness_disparity_<UTC>.parquet                             four measures, never one score
  fairness_drift_<UTC>.parquet                                 mostly insufficient_folds
  fairness_group_missingness_<UTC>.parquet                     the Component 11 link
  fairness_attribution_profiles_<UTC>.parquet                  grouped, never regenerated
  fairness_bootstrap_<UTC>.parquet                             both resampling schemes
  fairness_group_definitions_<UTC>.parquet                     incl. the REFUSED geographies
  manifest_fairness_group_metrics_<UTC>.json
```

### What it must not be read as

`does_not_establish` travels in every manifest and is printed on every run: **not causality,
not discrimination, not the absence of bias, not legal compliance, not ethical acceptability,
not equal treatment, not an optimal fairness policy.** And ADR 0019's gap is inherited rather
than discovered — the target is that a violation was *cited*, and geography is close to the
strongest available proxy for who inspected, so **this component cannot separate establishment
risk from differential inspection practice.**

### Verification

```bash
uv run python scripts/profile_fairness.py            # read-only; fixes the constants
uv run sentinel audit-fairness --dry-run --report    # writes nothing
uv run sentinel audit-fairness --report              # ~145 s; no thread override needed
```

Tests with teeth: a group mapping dated after its row turns the temporal check red; a same-day
mapping does too; swapping base and calibrated is detected against the committed artifact with
`==`; removing a small group from the support table is detected; a pooled `fold_set` is
rejected; a changed input checksum is rejected; and shuffling the prediction rows produces a
byte-identical artifact — which it did **not** until a canonical sort was added, because `ece`
uses equal-mass bins and rows tied at a bin boundary were assigned by arrival order.

---

## Component 13 — Decision Policy and Deployment Governance ✅

**Question:** Components 1–12 produce a calibrated probability. A probability is not an action.
Given limited capacity, model uncertainty, coverage gaps and Component 12's audit, what exactly
should Sentinel recommend doing?

**Answer:** a deterministic policy layer that converts calibrated risk into a
capacity-constrained queue, records for every establishment whether the model or the policy put
it there, and prices every alternative in Priority citations. **Its most valuable result is a
negative one:** the coverage intervention Component 12's finding appears to demand is aimed at a
population the risk queue already over-serves fourfold.

### What it implemented

* `src/sentinel/policy/` — 12 modules: `definitions.py` (the frozen policy grid, the
  model-selection rule, the vocabularies and the boundary, all behind an import-time guard),
  `models.py`, `inputs.py`, `select.py`, `eligibility.py`, `allocation.py`, `evaluate.py`,
  `governance.py`, `validate.py`, `writer.py`, `figures.py`, `build.py`.
* One command: `sentinel decide`.
* A **ninth** processed layer, `data/processed/policy/`, holding eleven tables.
* 228 new tests (2,331 → **2,559**), 18 error-severity checks + 4 advisories, 4 figures.
* `scripts/profile_policy.py` — eight read-only profiles, run **before** any policy constant
  was frozen, and what caught the central surprise.

### The production model, settled at last

MEMORY open question 13 — which model Sentinel should carry — is **closed as a policy decision,
not as a scientific one**. A lexicographic rule frozen in advance: discovery efficiency, then
calibration, then precision at one day of capacity, then the model name.

| model | NDE | sensitivity band | vs leader | calibrated ECE |
|---|---:|---|---|---:|
| `neural_numeric_only_platt` | 0.2482 | [0.2311, 0.2527] | tied | 0.0524 |
| **`xgboost_platt`** | 0.2376 | [0.2224, 0.2444] | tied | **0.0474** |
| `lightgbm_platt` | 0.2355 | [0.2201, 0.2419] | tied | 0.0490 |
| `logistic_regression_platt` | 0.2326 | [0.2160, 0.2374] | tied | 0.0518 |

**Axis 1 separates nothing.** Under Component 5's 1,000-replication label-flip study every
candidate's NDE interval contains every other candidate's point estimate — corroborating
Component 8's own conclusion that the network's advantage is the size of its seed noise. The
rule fell to calibration and selected `xgboost_platt`.

⚠ **The tie rule decides the deployment and was fixed after its inputs were first read.** The
plan carried a placeholder borrowed from a *different metric* (a ROC-AUC spread of 0.0058), and
under it `neural_numeric_only_platt` would have been selected instead. Both outcomes are emitted
on every run and ADR 0039 records the sequence. `xgboost_chain_embeddings_platt` is excluded
before any number is read (ADR 0022, ADR 0031).

### The seven policies, and what they cost

Pooled over 17 quarterly folds, `xgboost_platt`, at one week of real capacity (2,780 slots):

| policy | reserve slots | citations | Δ | eligible served | Δ |
|---|---:|---:|---:|---:|---:|
| `pure_risk` | 0 | 1,657 | — | 1,170 | — |
| `coverage_floor_population_share` | 0 | 1,657 | 0 | 1,170 | 0 |
| `coverage_floor_double_share` | 2 | 1,657 | 0 | 1,172 | +2 |
| `coverage_forced_half_share` | 133 | 1,649 | **−8** | 1,246 | +76 |
| `coverage_forced_population_share` | 274 | 1,642 | **−15** | 1,325 | +155 |
| `coverage_forced_double_share` | 556 | 1,623 | **−34** | 1,513 | +343 |

The floor is nearly inert because the risk queue already clears it. The forced reserve buys
coverage at a price reported in citations, and the price grows with capacity. **The one-day
deltas (±1 to ±3 out of 348) are inside the noise and the findings document says so**, including
where the noise flatters the component.

### What it deliberately did not do

* **Nothing changed for the `__UNKNOWN__` group.** At one day of capacity it gets 2 of 556 slots
  and 1 citation found — **identical under all seven policies**. Eligibility is keyed to missing
  *history*, and only 3.2% of that population sits in the no-geography group. A reserve that
  reached it would be an allocation keyed to a failed geocode (ADR 0038).
* No score adjusted by geography, no group-specific threshold or calibrator, no quota, no
  probability threshold anywhere.
* **No policy winner declared.** Two policies survive at the primary operating point and neither
  dominates; the run prints *the data does not determine the correct policy* and
  `a_winner_was_determined` fires as an advisory.

### Reproducibility

11 of 11 tables **byte-identical** across two independent production runs. Shuffling the
prediction rows and the feature rows leaves the queue identical, asserted end to end and over a
window of pure ties. Input checksums compared before and after every run. Refits,
re-executions, bit-identity gates: **none** — this component reads artifacts and does
arithmetic. The determinism claim is scoped in the manifest to *identical inputs including the
override file*; human overrides are external decisions and are pinned by checksum rather than
claimed reproducible.

---

## In Progress

Nothing. Components 1 through 9 and 11 through 13 are closed.

---

## Not Started

**Correction (2026-09-02): the table below was stale.** Components 14 and 16 are implemented
(see "Component 14" and "Component 16" sections further down this file) — the blanket "no code
exists" claim above predates them and was never updated. Component 15 (OR-Tools routing) remains
genuinely blocked for the reason stated. The original roadmap's 17–21 (LangGraph orchestration,
LLM-generated briefings, deterministic briefing verification, audit trail, frontend demo) were
never built; the project's actual direction after Component 16 diverged from that plan and
instead extended the live operational pipeline — Components 17–20 (candidates, operational
scoring, operational selection, geographic organization) and Component 21 (supervisor plan
review) as described below. Only Component 15 keeps its original meaning and status.

| # | Component | State |
|---|---|---|
| 9 | Probability calibration | **Implemented** |
| 10 | Inspector-effect modelling | **Blocked** — no inspector field exists (ADR 0019) |
| 11 | SHAP explainability | **Implemented** — 4 of 5 candidates supported (ADR 0028–0031) |
| 12 | Fairness and geographic equity audit | **Implemented** — 2 geographies audited, 5 refused (ADR 0032–0035) |
| 13 | Decision policy and deployment governance | **Implemented** — 7 policies compared, production model selected, no policy winner declared (ADR 0036–0040) |
| 14 | Constrained scheduling | **Implemented** — see "Component 14" below |
| 15 | OR-Tools routing | **Blocked** — same missing inspector/travel-time data as Component 10 (ADR 0019) |
| 16 | Deferral / human-review gate | **Implemented** — see "Component 16" below |
| 17 | Operational candidates | **Implemented** — `src/sentinel/candidates/`, live planning-date-scoped feature table. No `docs/data_contracts/candidates.md` yet (known gap, deliberately deferred — see the 2026-09-05 final completion pass below) |
| 18 | Operational scoring | **Implemented** — `src/sentinel/operational_scoring/`, scores Component 17's candidates with the frozen production model. No `docs/data_contracts/operational_scoring.md` yet (same deferred gap) |
| 19 | Operational selection | **Implemented** — `src/sentinel/operational_selection/`, capacity-constrained selection via Component 13's own allocation engine. No `docs/data_contracts/operational_selection.md` yet (same deferred gap); its manifest's coverage counts (`ranked_candidate_count`, `selectable_candidate_count`, `selected_count`) are exposed read-only via `GET /v1/manifests/operational_selection` as of the 2026-09-05 pass |
| 20 | Geographic work organization | **Implemented** — `src/sentinel/geographic_organization/` (v2: work blocks, suggested order, organization-mode tradeoff); see `docs/data_contracts/geographic_organization.md` |
| 21 | Supervisor plan review | **Implemented** — `src/sentinel/plan_review/`; see `docs/data_contracts/plan_review.md` |

---

## Current Architecture

```text
src/sentinel/
  __init__.py            __version__, stamped into every manifest
  config.py              Pydantic Settings (env prefix SENTINEL_)
  logging_setup.py       configure_logging()
  cli.py                 argparse: ingest, query, resolve, build-target,
                         build-features, train-baselines, tune-boosting,
                         train-boosting, build-neural-categoricals, tune-neural,
                         train-neural, calibrate, explain, audit-fairness,
                         decide, evaluate
  manifest.py            generic sha256 / read / write helpers
  policy/                Component 13
    definitions.py       POLICY_GRID, the selection rule, the vocabularies,
                         DOES_NOT_ESTABLISH -- all behind an import-time guard
    models.py            PolicyWindow, Allocation, Override, the manifest
    inputs.py            the only module reading Parquet on the way in
    select.py            the lexicographic production-model rule
    eligibility.py       one column, one predicate, and what it refuses
    allocation.py        risk block, coverage reserve, decide(), ranks
    evaluate.py          cell metrics, opportunity cost, the frontier
    governance.py        warnings_for, parse_overrides, apply_overrides
    validate.py          18 error checks + 4 advisories; the ADR 0034 split
    writer.py            eleven schemas, eleven total sort keys
    figures.py           four figures; none marks a point optimal
    build.py             run_policy (the only module touching the clock)
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
  profile_calibration.py  read-only profiles over the calibration surface
  profile_explanations.py read-only profiles over the attribution surface
  profile_fairness.py    10 read-only profiles over the group-audit surface
  profile_policy.py       8 read-only profiles over the decision surface;
                          run BEFORE any policy constant was frozen
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

**Command:** `uv run pytest` · **Result: 2,559 passed, 3 deselected** (2026-08-26)

Component 2 added 265 tests, Component 3 added 201, Component 4 added 202,
Component 5 added 278, Component 6 added 231, Component 7 added 235, Component 8 added
287, Component 9 added 135, Component 11 added 173, Component 12 added 247, Component 13
added **228**. Quality gate, all passing:

```bash
uv run pytest                  # 2,559 passed, 3 deselected, 662 s
uv run ruff check .            # All checks passed
uv run ruff format --check .   # fails on 10 pre-existing Component 9 files; see Known Issues
uv run mypy src/sentinel       # no issues in 133 source files
```

| Component 13 area | Coverage |
|---|---|
| `test_policy_definitions.py` (24) | the frozen grid and the import-time guard shown raising on eleven distinct defects — a baseline that reserves capacity, a mechanism no policy exercises, an orphan reason code, a selection rule that cannot terminate, an empty boundary list |
| `test_policy_allocation.py` (31) | floor versus forced semantics on hand-built windows; the reserve solved rather than assumed; disjointness and `n_risk + n_reserve == k` over a grid; ties broken on the id and never on row order |
| `test_policy_evaluate.py` (18) | **precision, capture and lift matched exactly against Component 5's own top-k helpers on `pure_risk`** — the check that licenses computing over the queue instead of calling them — plus a case proving a coverage queue's number differs |
| `test_policy_governance.py` (24) | the override contract: a blank actor, reason or timestamp refuses the whole file; displacement named; no backfill; id-order application; an inclusion with nothing left to displace raises rather than raising capacity |
| `test_policy_leakage.py` (44) | the safety wall — a queue longer than capacity, an ineligible row in the reserve, a duplicated rank, a lost prediction row, a label column in a decision artifact, a warning input that moves the queue, an unattributed override, a changed input checksum. **And five tests asserting the opposite**: a reserve that gave up 34 citations, an inert reserve, a moved group share and the absence of a winner must all stay advisory |
| `test_policy_determinism.py` (13) | two full runs byte-identical; shuffled predictions and shuffled features leave the queue identical; a window of pure ties, which is where Component 12's real ordering bug lived; the memoised order proven keyed by content rather than identity |
| `test_policy_build.py` (22) | end to end over a synthetic snapshot; the manifest, the boundary, the checksum gate; **the whole component run twice, with and without the group artifacts, and the ranks compared** |
| `test_cli_policy.py` (16) | registration, every flag, the refused model rejected before any artifact is read, and an advisory finding proven not to change the exit code |

| Component 11 area | Coverage |
|---|---|
| `test_explain_definitions.py` (30) | the support matrix; the import-time guard shown raising on nine distinct defects, including a permutation method labelled exact and an unsupported model that still advertises one; every Component 4 feature proven to map to itself, so no undeclared aggregation is possible |
| `test_explain_attribution.py` (18) | tree and linear SHAP matched to `shap` at **0.0**; permutation SHAP matched to **brute-force Shapley over all 2^M subsets** on a deliberately non-additive model; and `test_additivity_holds_at_one_round_and_is_therefore_not_evidence_of_accuracy`, which asserts both that additivity holds at one round *and* that one round is visibly wrong |
| `test_explain_leakage.py` (19) | the safety wall — appended future rows leave a background bit-identical; a poisoned reference row turns the check red; a later fold's background is rejected; the sampler proven to ignore the label by corrupting every column it may not read |
| `test_explain_contract.py` (34) | end to end against a real artifact; a one-ULP score perturbation detected; `feature_127` rejected; a fabricated attribution for the unsupported model rejected; the manifest proven to record only the artifacts the run actually read |
| `test_explain_stability.py` (25) | rank/Spearman/Jaccard arithmetic against hand-computed cases; an extreme `covid_shift` fold proven not to move the quarterly aggregate by a single float |
| `test_explain_writer.py` (27) | seven schemas, full sort keys, missing and unknown columns refused, a null `feature_value` surviving, byte-identical re-writes |
| `test_explain_determinism.py` (7) | two full runs byte-identical across all seven tables, plus a control asserting the run is not trivially empty |
| `test_cli_explain.py` (20) | the flag surface, the exit-code matrix, warnings not failing a run, and an absent Component 9 artifact proven to be a supported state rather than an error |

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
   it reached first. Attribution is now Component 11's, in
   `data/processed/explanations/` — use that, not `boosted_importances_*.parquet`. The
   correlation caveat survives the upgrade: SHAP splits credit between correlated features
   too, and discloses nothing about how.
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
   0.118. Attribution is now Component 11's. Note that its measured answer keeps those two
   columns at ranks 1 and 2 for this model, which is the coefficient magnitude reappearing
   rather than being corrected: a SHAP value on a linear model is
   `coef_j * (z_j - E[z_j])`, so it inherits the collinearity exactly.
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

### Component 14 — Operational scheduling and execution planning

Turns Component 13's approved queue into a plan against the calendar Chicago actually worked.
Deterministic, fits nothing, scores nothing, re-ranks nothing, and adds **no dependency**.

* `scripts/profile_scheduling.py` — 8 read-only profiles run before any implementation code.
  `docs/analysis/scheduling_findings.md` holds the output, and every frozen constant in
  `definitions.py` traces to a number in it.
* `src/sentinel/scheduling/` — 15 modules: the frozen contracts, the typed structures, the input
  contract, the horizon, the allocator, the backlog, the two external human contracts, the
  re-planner, the measurements, the validator, the writer, the figures and the orchestrator.
* `sentinel schedule` — CLI, with `--capacity-mode`, `--adjustments`, `--execution`, `--dry-run`
  and `--report`. Deliberately **no** `--capacity`, `--slots-per-day`, `--horizon-days`,
  `--extend-horizon` or `--threshold`; the suite asserts each absence.
* Output under `data/processed/scheduling/` — a tenth processed layer, 13 tables plus a
  manifest. Contract in `docs/data_contracts/inspection_schedule.md`.
* ADRs 0041 (the tenth layer), 0042 (the five layers and the four boundaries), 0043 (temporal
  scheduling without a route and without a solver), 0044 (the horizon is the capacity rule read
  backwards), 0045 (strict priority, and its price falls on the reserve), 0046 (backlog is a
  first-class outcome), 0047 (three human layers; execution is external).
* **450 new tests** (2,559 → 3,009), including all 17 specified failure injections.

**The horizon rule introduces no new constant.** `ceil(k / test_median_daily_capacity)` is
Component 5's own capacity rule read backwards, and it reproduces both day-denominated cutoff
names exactly — `k_1_day` spans one day, `k_1_week` spans five. Verified total across all 90
(fold, capacity) cells: **0** demand more operating days than their fold contains.

**The calendar is read, never generated.** An operating day is a date the recommendation
universe carries. Three inspections in the snapshot fall on a weekend, so a synthesised
Monday-to-Friday calendar would be wrong at the edges — and the holiday list it would need is
something this project has no way to verify.

**Verified on the full artifact:** 1,260 cells (2 capacity modes × 7 policies × 18 folds),
141,582 queue rows, **136,094 scheduled**, **5,488 backlogged** across 308 cells, **11,067 idle
slots** across 679 cells, and **0 inversions**. 28 error checks pass and 7 advisories fire.
Inputs sha256-identical before and after. **13 of 13 tables byte-identical across two
independent production runs**, and identical again under shuffled recommendation, adjustment and
execution rows. Runs in ~12 s.

**What it does not do, and cannot.** No route optimisation, no inspector assignment, no
travel-time estimate: the raw table has 22 columns and none of them is an inspector, which is
the same absence ADR 0019 recorded when it blocked Component 10. Routing is Component 15's, and
Component 15 is blocked on that same missing data. The allocation is described as what it is —
deterministic greedy allocation down an approved rank order — and no objective function is
defined or solved anywhere in the component.

**The execution layer is a contract with an engine, not a record.** The external adjustment and
execution contracts are fully implemented, validated and tested, and both tables are **typed
empty** on every run nobody supplies a file for. **No row in this repository describes anything
that happened in Chicago.**

---
---
## Next Component

**Component 15 - OR-Tools routing.** Blocked, and blocked for the same reason Component 10 is.

Component 14 is closed, so 15 is the next component in the roadmap — and it cannot be built on
this data. Routing needs an inspector, a base location, a duration and a travel time, and the
snapshot has none of the four. ADR 0043 records the check rather than the assumption:
`scripts/profile_scheduling.py` profile 7 is the inventory, twelve operational fields a real
inspection department schedules against, all absent. Adding OR-Tools and feeding it invented
parameters would produce routes that look authoritative and describe nothing.

**Component 16 — deferral / human-review gate — is now implemented.** See "Component 16" below.
It used the two pieces Component 14 shipped for it: an external contract shaped like Component
13's override, and a re-planner that appends rather than mutates.

### What Component 14 leaves open

1. **Is the 29.3% reserve loss stable, or an artifact of `ceil(k / median)`?** A horizon one day
   longer would recover much of it, and nothing measures which length is right.
2. **Should Component 13 place the reserve at the tail at all?** That placement is what makes
   the reserve the first casualty of a short horizon. Changing it is a Component 13 policy
   change, not a Component 14 scheduling change, and this component was not entitled to make it.
3. **Is the observed per-day calendar usable as *planning* capacity, or only as *evaluation*
   capacity?** It is measured from the window it schedules, so it says what capacity existed,
   not what a planner could have known on day one. A live deployment would need a forecast this
   project has not built.
4. **Are the adjustment and execution contracts usable by a real supervisor or a real field
   inspector?** Tested against synthetic files only — the same open question Component 13
   recorded for its override contract.

### What Component 15 must not redo

* **Do not re-rank.** Component 13 owns the queue and Component 14 preserves it exactly. A
  router that reordered by risk would be a third layer with an opinion about priority.
* **Do not raise capacity.** Every horizon descends from the window's own measured median daily
  rate. There is no flag that raises a slot count and the suite asserts the absence.
* **Do not treat the coverage reserve as slack.** Component 14 measured what the calendar
  already costs it; a router that spent it to shorten a drive would be making a policy change
  invisibly.
* **Do not let an execution outcome edit a plan.** `inspection_schedule` has no
  `execution_status` column, deliberately, and that is what makes the guarantee structural.
* **Do not join anything in `data/processed/scheduling/` onto a feature table.** It holds the
  system's own past scheduling decisions, one layer further out than Component 13 already
  refused to close.
* **Do not fabricate an inspector.** Two components are now blocked on the absence. A third that
  invented one would make all three unfalsifiable.

### What Component 16 must not redo (now enforced, not merely planned)

* **Do not introduce a probability, score or confidence threshold.** ADR 0040 forbids it; there
  is no `--threshold` flag and the suite asserts the absence.
* **Do not create an override or an adjustment directly.** A `refer_to_override` /
  `refer_to_adjustment` resolution records a pointer only.
* **Do not reuse "defer"/"deferred" as a status or verb.** Component 14's
  `ScheduleStatus.DEFERRED` means something structurally different, and the import-time guard
  checks the literal substring, not just enum-value overlap.

---

## Component 16 — deferral / human-review gate

Two deterministic triggers, both boolean facts an upstream component already wrote, no numeric
threshold: `policy_warning_present` (a selected Component 13 recommendation carries a policy
warning) and `no_execution_record_on_scheduled_row` (an occupying Component 14 schedule row has
no matching row in the accumulated execution log). `queue_is_deterministically_rebuildable` proves
neither trigger reads `score`, `base_score` or `final_policy_rank`.

Measured against real data (2026-08-27): of 1,453,760 recommendation rows, 70,791 flagged —
39,652 by the warning trigger, 70,791 by the execution-gap trigger. The execution-gap count is not
an operational finding: the production `execution_log` is empty, so every occupying schedule row
is, by construction, "missing" a report nobody has filed. See ADR 0051.

A fourth human layer, disjoint from override/adjustment/execution by construction and by
import-time guard, including an explicit check that no Component 16 vocabulary value contains the
literal substring `"defer"`. The queue (`human_review_queue`) is rebuilt fresh each run; the
resolution log (`review_resolution_log`) is append-only and permanent. CLI: `sentinel review`.
API: `/v1/review/queue`, `/v1/review/resolutions` (stage-only, ADR 0049's pattern). See
`docs/data_contracts/human_review.md` and `docs/interview/component_16.md`.

---

## The Sentinel API

Built after Component 14, without claiming either name above. It is **not** "Component 15" and
**not** "Component 16" — both keep their meanings unchanged (routing, blocked; the
deferral/human-review gate, now built and described above). It is cross-cutting infrastructure
(`src/sentinel/api/`, `sentinel serve`): a validated read/write HTTP boundary over Components
1-16's existing artifacts, computing nothing.

**Reads** compose existing Parquet artifacts into product-shaped JSON, behind a mandatory
decision scope — a request that leaves out `policy_id`/`fold_id`/`k_name`/etc. gets a `422`
naming what's missing, never a silently guessed "latest" row (ADR 0050).

**Writes** (an override, a scheduling adjustment, an execution event) validate against the exact
contracts Components 13 and 14 already define, then append to a staging file this package owns.
Nothing is applied: turning a staged request into a new artifact is still a manual
`sentinel decide` / `sentinel schedule` run by an operator (ADR 0049). There is no `POST` that
recomputes a queue or a schedule, and no `PATCH`/`PUT` route exists on any resource — immutability
is structural.

**What it does not do:** no routing endpoint (same blocker as Component 15 — no inspector, no
travel time), no authentication (documented gap, not a silent omission), no new database or
message queue (a flat append-only file was sufficient; nothing measured yet justifies more).

See ADR 0048, ADR 0049, ADR 0050, `docs/data_contracts/sentinel_api.md` and
`docs/interview/api_layer.md`.

---

## The Sentinel Frontend

Built after the Sentinel API, initially for product testing only, as a separate, minimal,
read-only frontend. **Correction (2026-09-04): the sentence that used to stand here** ("not
Component 21 -- that roadmap row stays 'Not implemented'") **is stale.** Component 21 was built
after this section was originally written (see "Not Started" above) and the frontend below now
includes its supervisor-facing plan-review and approval pages; the roadmap table is the current
source of truth, not this paragraph's original wording. `frontend/`: React + TypeScript + Vite, no
fabricated data, no client-side sort-column picker.

**The one change to `src/sentinel/`:** CORS middleware added to `src/sentinel/api/app.py`
(allowed origins from the new `Settings.api_cors_origins`, default `localhost:5173`), plus one new
test file `tests/api/test_cors.py`. Nothing else under `src/sentinel/` was touched. Full Python
suite re-verified after the change: **3,065 tests pass, 3 deselected** (up from the
Component-14-era 3,009 by tests accumulated since, plus the 2 new CORS tests here); ruff and mypy
`--strict` remain clean on `src/sentinel`.

Frontend test suite (Vitest + React Testing Library + msw): **47 tests pass** (up from 33) across
API client error classification, the scope selector's no-request-until-complete gating, and all
six pages (Overview, Inspection Priorities, Establishment detail, Inspection Schedule, Waiting for
Capacity, Human Review) — including a regression test that `recommendation.decision_reason` and
`schedule.schedule_reason` render as two distinct fields, never merged (ADR 0042). `npm run
typecheck` (`tsc -b --noEmit`) and `npm run lint` (`oxlint`) are both clean.

### Product clarity pass (2026-08-28)

The frontend above was technically correct but assumed the visitor already knew Sentinel's
internal vocabulary (fold, policy id, raw reason codes). A second pass rebuilt the primary
experience in plain language — Overview leads with the product story and real operational
counts; every page's table now shows a plain-language status/reason beside the raw code, moved
into a collapsed "Technical details" section rather than deleted; a new Human Review page exposes
Component 16's existing, previously frontend-less API endpoints; the Establishment page now shows
a five-step journey. `useDefaultScope` fills in a real, verified scope automatically from the
live manifests so a first-time visit shows real data immediately. A genuine bug was found and
fixed during this pass: calling react-router's `setSearchParams` once per scope field inside one
effect silently dropped all but the last field, caught by the frontend's own tests (not manual
inspection) timing out on "does the page ever show real data" rather than only "does a loading
state appear." Fixed with a bulk `setScopeFields` setter. No backend file, ML model, or API
contract changed. Full detail: `docs/analysis/frontend_product_clarity_20260828.md`.

### Actionability & operational workflow pass (2026-08-28)

A product reality check (self-critical, not a bug report) found the previous frontend told a
supervisor what to investigate ("resolve this," "confirm what happened") with no mechanism to do
either — every page was read-only — and treated `is_selected` as if it were a risk verdict, when
it is a function of the current plan's capacity cutoff and can flip with nothing about an
establishment changing. This pass closed both gaps without inventing any new backend behavior.

**Backend (read-path only — no write, no policy, no scheduling logic touched):** four filters
added, mirroring the existing `establishment_id`/`schedule_status` filter precedent —
`target_inspection_id` on `GET /v1/policy/overrides`, `GET /v1/schedule/adjustments`,
`GET /v1/execution/events`, `GET /v1/review/resolutions`; `trigger` (a literal substring match
against the existing pipe-joined `trigger_reasons` column) on `GET /v1/review/queue`, so a caller
can ask for `policy_warning_present` or `no_execution_record_on_scheduled_row` cases separately
without any new classification existing server-side. 6 new backend tests (`tests/api`: 82 → 88,
all passing); `ruff`/`mypy --strict` clean on `src/sentinel`. A pre-existing documentation error was found and corrected while
reading these services: `OverrideLogRowOut.status`, `ResolutionLogRowOut.status` and
`ReviewCaseOut.status` all claimed a still-staged row would show as pending/`pending_review` —
none of the four log-read services ever read the staging store; `status` is unconditionally
`"committed"`. Docstrings and `docs/data_contracts/sentinel_api.md` now say so; the frontend's
"Decision history" panel works around it by querying `GET /v1/staged-requests` separately and
merging client-side, which is what actually works today.

**Frontend — four real write forms**, each submitting exactly its backend contract's fields,
validated the same way the batch CLI validates them, staged and never applied (ADR 0049, stated
explicitly on every confirmation): `OverrideForm` (force_include/force_exclude), `AdjustmentForm`
(defer/advance/cancel a planned inspection), `ExecutionOutcomeForm` (execution status options read
live from `GET /v1/execution/contract`, never hardcoded), `ResolutionForm` (acknowledge/refer to
override/refer to adjustment/escalate). A new "Decision history" section per establishment merges
committed and still-pending entries across all four contracts. A real, product-relevant bug was
found and fixed in the process: a background refetch (e.g. `useDefaultScope` filling in
`schedule_config_id` slightly after the establishment record itself had already loaded) cycled the
page's main query back through `'loading'`, which unmounted the entire journey — including
whatever action form a person had open — discarding their in-progress input. Fixed with a
stale-while-revalidate pattern (the journey renders from the last successful payload, updated
during render rather than in an effect, never from the query's live status) rather than a
component-boundary workaround, since the same class of bug existed identically for the review-case
lookup.

**Terminology correction.** "Recommended for inspection" is now "Selected for this plan," with an
explicit hint that a different capacity or plan could place the same establishment on either side
of the cutoff with nothing about it different. The raw calibrated score is no longer the primary
number shown on list pages — it reads as a probability or a fixed threshold, which the policy
contract makes no claim to be — replaced by `relativePriorityLabel` (rank + percentile from
`model_rank`/`n_universe`, both already computed pre-cutoff by Component 13, so the position is
stable regardless of today's capacity). A single, once-per-page "How to use this priority" note
states the project's own measured ROC-AUC (~0.61-0.62) rather than hiding or overselling it. Human
Review is now two sections — Decision Review and Missing Outcomes — using the new `trigger`
filter; Overview's single "Needs human review" count is now two counts, so a 100%-execution-gap
result (measured: 28 of 28 in the current production scope) can never again read as "28 suspicious
decisions." A capacity-honesty note (`capacityHonestyNote`, sourced from the scheduling manifest's
real `capacity_mode`) states plainly that the default `observed_calendar` mode's daily capacity is
a historical inspection count, not a live staffing feed.

Frontend test suite: 47 → **64 passing**, including regression tests for the remount bug, the
trigger split, and each write form's staged-not-applied confirmation. `npm run typecheck` and
`npm run lint` clean. Manually verified against the real production API and artifacts (not just
mocks): staged a real override, confirmed it appeared via `/v1/staged-requests`, then removed it
from `data/staging/` so no test data was left in the repository; confirmed the real trigger split
(2 decision concerns vs 28 missing outcomes, for the default scope) and the real `n_universe`
(1,638) behind the percentile display.

No route optimization, inspector assignment, authentication, or live staffing integration was
added — none of those exist on the backend to build against, and none is implied by anything new
here.

See `frontend/README.md` for run instructions and the full list of what was deliberately left out.

### Component 21 completion pass — operational priority and plan approval (2026-09-04)

Closed out Component 21's two remaining requirements from the original spec: a supervisor's
ability to adjust field-work order without touching Sentinel's own risk rank, and an explicit
plan-approval act that produces an immutable, authoritative artifact for Component 22.

**Backend.** New `PlanDecisionAction.ADJUST_OPERATIONAL_PRIORITY` (4th decision verb, requiring
`revised_operational_priority`); a new `operational_priority` column on `supervisor_plan_review`,
computed as `coalesce(supervisor_revised_operational_priority, policy_rank)` — verified end to end
against real data that `rank`/`policy_rank` stay byte-identical while `operational_priority`
reflects the override. New `PlanApprovalStatus.APPROVED`, reachable from any decision-coverage
state, derived from whether a committed `approved_operational_plan` artifact exists for the
`planning_date` (not from decision coverage — a plan need not be fully decided to be approved).
New `src/sentinel/plan_review/approval.py` (5-point readiness checklist: no duplicate
establishments, every row carries the machine recommendation, geographic provenance present,
every recorded decision has a reason, undecided rows default to the machine recommendation —
advisory only) and `build_approved_plan()` in `build.py`, writing an immutable, timestamped
`approved_operational_plan_<planning_date>_<stamp>.parquet` + manifest. New CLI command
`sentinel approve-plan`. New API: `GET /v1/plan-review/approval`, `POST /v1/plan-review/approve`
(staged-only, ADR 0049, committed only by `sentinel approve-plan`).

Two real bugs were found and fixed while wiring the API path against real data (not caught by
unit tests written in isolation, only by an end-to-end smoke test): (1)
`PLAN_APPROVAL_REQUIRED_FIELDS` and its `_guard_registry()` check both referenced
`source_plan_review_sha256`, a field that exists on `ApprovedPlanManifest` (computed
independently from the real file at commit time) but never on `PlanApprovalRequest` (the
supervisor's own minimal input) — every approval request was refused with a "blank field" error
regardless of payload. (2) `StagingService._KIND_CONFIG` had no `"plan_approval"` entry, so a
valid request that passed schema and governance validation still crashed with `KeyError` inside
the staging append itself. Both are covered by new regression tests
(`tests/api/test_plan_review_approval_api.py`) so this class of "the contract compiles but the
whole path 500s" bug cannot silently reappear.

**Frontend.** `PlanDecisionForm` gained the `adjust_operational_priority` action and its required
revised-order input. New `PlanApprovalPanel` component: shows a readiness-aware "Approve plan"
action when unapproved, or the committed approval's identity and final counts (active/deferred/
not-proceeding/undecided) once approved — reading `GET /v1/plan-review/approval`, never
recomputing readiness client-side. `SupervisorPlanReviewPage` now shows `operational_priority`
beside `policy_rank` whenever they differ, with an explicit note that the machine rank is
unchanged. `PlanRowOut`/`PlanDecisionIn` types, `planReview.ts` API client, and `copy.ts` extended
to match. New `SupervisorPlanReviewPage.test.tsx` (4 tests) plus new plan-review mock
fixtures/handlers (none existed before this pass). Frontend suite: 63 → **67 passing**; `tsc
--noEmit` and `oxlint` both clean.

Verified against real committed data end to end: built a fresh 30-establishment plan review,
applied a 3-action demo decisions file (keep, defer with reason, adjust-priority with reason),
approved it (30 total, 29 active, 1 deferred, 27 undecided, all readiness checks READY), confirmed
the blocked path (mismatched `planning_date` refused with a clear error) and determinism
(repeated approvals of identical input produce byte-identical content outside the intentionally
non-deterministic approval identity fields). Full backend regression suite re-run after all
changes; see the top of this file for the current pass count.

### Final completion pass — product coherence, navigation, and honesty audit (2026-09-05)

A dedicated end-to-end audit and fix pass, distinct from any single component: the goal was
making the already-correct C17→C18→C19→C20→C21 pipeline feel like one coherent product for a real
food-inspection user rather than a set of separately-built pages, without touching any backend
business logic. Two research passes (frontend page/copy survey, backend/docs API survey) found the
pipeline itself sound — no contract-level defects — but found the two newest, most operationally
real pages (`GeographicPlanPage`, `SupervisorPlanReviewPage`) structurally orphaned: unreachable
from `OverviewPage`/`TodayPage`/`EstablishmentDetailPage`, with no "which plan am I looking at"
context, and several raw technical identifiers leaking into primary (non-collapsed) UI.

**Navigation.** `NavBar` reordered into three visually-grouped sections (landing → live
operational plan → historical analysis) rather than one flat list. `OverviewPage` gained an
unconditional "Today's field plan" section linking to Field Plan and Plan Review — previously
every link on that page pointed only at the backtest-era pages. `WorkflowDiagram` (shown on
Overview) extended from 5 steps to 8, adding the operational-candidate, scoring, capacity-
selection, geographic-organization, and supervisor-review/approval steps it previously omitted
entirely. `GeographicPlanPage`/`SupervisorPlanReviewPage` both gained a plain "Field plan for
{date}"/"Plan for {date}" header, using `planning_date` their own API responses already return —
no new backend call. `EstablishmentDetailPage` gained a 7th journey step ("Current field plan")
linking generically into both live-plan pages, since no existing API resolves establishment_id →
target_inspection_id for a true deep link (a genuine, explicitly out-of-scope gap for a future
pass, not silently worked around).

**Raw-ID leaks fixed (7 locations).** `EstablishmentDetailPage`'s subtitle no longer shows the raw
`establishmentId` unconditionally (moved into its existing Technical Details, address stays
primary). New `lib/copy.ts::apiErrorCodeLabel()` translates every `ApiError` subclass's code
(`artifact_not_found`, `row_not_found`, `validation_refused`, `duplicate_key`,
`unknown_component`) into plain language; `ErrorState` now shows that as primary text with the raw
code de-emphasized alongside it, not dropped. `ResolutionForm`'s always-visible hint no longer
names a raw `review_id`. New `lib/copy.ts::workBlockDisplayLabel()` replaces a raw `work_block_id`
fallback heading with "Work block N" wherever Component 20 has no label.
`GeographicPlanPage`'s empty state no longer names a raw CLI command (`sentinel
organize-geography`); `PlanApprovalPanel`'s hint no longer names `sentinel approve-plan` directly
in primary text (moved into a nested Technical Details).

**Backend: one small, additive, read-only change.** `operational_selection`'s own manifest already
computes `ranked_candidate_count`/`selectable_candidate_count`/`selected_count` (and three more
counts) — it was simply never reachable through any API route (`meta_service._COMPONENTS`
whitelisted only `policy`/`scheduling`/`explanations`/`review`). Added one dict entry:
`"operational_selection": ("operational_selection_processed_dir", "operational_selection")` —
confirmed against real committed data: `GET /v1/manifests/operational_selection` returns
`ranked_candidate_count=35859, selectable_candidate_count=35859, selected_count=30` for the
2026-08-28 planning date, matching the manifest on disk exactly. No new computation anywhere;
purely a whitelist addition, identical mechanism to the four pre-existing entries. New
`lib/copy.ts::operationalCoverageNote()` turns those three counts into a plain sentence on
`SupervisorPlanReviewPage`, honestly worded ("The rest are not flagged as lower risk — they simply
did not fit in this plan"), never implying anyone not selected is safe.

**Documentation.** `frontend/README.md` was stale — it still called Component 21 an "unbuilt...
Frontend demo roadmap row," contradicting the root `README.md`/`HANDOFF.md`, and never mentioned
`SupervisorPlanReviewPage`/`GeographicPlanPage` or their two write actions at all. Rewritten to
match reality, including the two new staged-write contracts (`plan_decision`, `plan_approval`) in
the same inventory as the four backtest-side forms. `docs/data_contracts/candidates.md` /
`operational_scoring.md` / `operational_selection.md` remain genuinely absent (every other
implemented component has one) — deliberately deferred rather than rushed, noted honestly above in
the roadmap table rather than left silently unstated.

**Testing.** Two pages had literally zero test coverage before this pass and their MSW mock
handlers didn't exist yet either (`/v1/schedule/dates`, `/v1/plan-review/work-blocks`) — both
added along with new fixtures. New: `TodayPage.test.tsx` (3), `GeographicPlanPage.test.tsx` (5),
`ErrorState.test.tsx` (3), `NavBar.test.tsx` (2), `lib/copy.test.ts` (5). Extended:
`SupervisorPlanReviewPage.test.tsx` (4→7), `OverviewPage.test.tsx` (+2). Frontend suite: **67 →
89 passing** (16 test files, up from 11); `tsc --noEmit` and `oxlint` both clean throughout.
Backend: 2 new tests in `tests/api/test_manifests_runs.py`; full suite **3,384 passed, 3
deselected** (up from 3,382), `ruff check`/`mypy` clean on every touched file.

Verified against real committed data end to end, via the actual running API
(`create_app()` + `dependency_overrides[get_settings]`, not mocks): fetched Component 20's 22 real
work blocks, Component 21's real plan summary (30 selected, already approved from earlier
verification), the new operational-selection coverage manifest (35,859 ranked → 30 selected,
exactly matching what now renders in the UI), and staged one fresh plan decision successfully.
All staged-write test artifacts created during this pass (and two leftover from earlier sessions'
manual testing) were removed from `data/staging/plan_review/` afterward — nothing was ever
applied/committed (ADR 0049), so removing them changes no real artifact.

No auth, live staffing/traffic, route optimization, or Component 22 (execution/outcome recording
for the operational side) was added or implied — all explicitly out of scope for this pass, per
the same boundaries every prior component has held.

### The "Today = April 1, 2026" bug — root cause, fix, and real data generation (2026-09-05)

A real product-correctness bug, distinct from the completion pass above: the landing page
(`TodayPage.tsx`, route `/`) was, despite its name, entirely wired to **Side A** — the historical
backtest/evaluation pipeline (Components 1-14, 16), scoped by `(policy_id, fold_set, fold_id,
k_name, schedule_config_id)` — never to **Side B**, the live `planning_date`-scoped operational
pipeline (Components 17-21). Root cause, traced exactly: `useDefaultScope` auto-selects the *last*
fold in the hardcoded `FOLD_TABLE` (`frontend/src/api/folds.ts`) — `quarterly-2026Q2`, a simulated
Apr-Jun 2026 test window — and `TodayPage` then called Component 14's fold-scoped
`listScheduleDates` and took the *first* date in that fold's simulated calendar: April 1, 2026.
"Today" had never once been connected to a real current date anywhere in the frontend or backend
— no such abstraction existed at all.

**Fix, two parts, both required.** (1) `TodayPage.tsx` was rebuilt from scratch to read Side B
instead: it now calls the same no-scope `getPlanSummary`/`listPlanRows` (`frontend/src/api/planReview.ts`)
`GeographicPlanPage`/`SupervisorPlanReviewPage` already use, and a new `frontend/src/lib/today.ts`
(`currentOperationalDate()`, computed live from `new Date()`, never a hardcoded literal) lets it
honestly compare the plan's own `planning_date` against the real current date — `lib/copy.ts::planLabelForToday`
says plainly "Today's inspection plan" when they match, or "Plan for {date} — not today's plan
yet" when they don't, never silently relabeling a stale plan. (2) The genuinely honest way to make
"today" show real, current data — rather than a UI trick over stale data — was to actually run the
real CLI pipeline once for `planning_date=2026-09-04` (the real current date in this environment):
`plan-candidates` → `score-candidates` → `select-inspections --capacity 30 --policy pure_risk` →
`organize-geography` → `review-plan` → `approve-plan`, producing new, real, immutable artifacts
(35,859 real candidates, 30 selected, 21 real geographic work areas, approved) using the same real
Chicago data and the same frozen production model as every other run — no fabrication anywhere.
The prior `2026-08-28` artifacts are untouched on disk (12 files, immutable) and remain reachable
by anyone querying that specific date directly; they are simply no longer what "latest" resolves
to.

The old fold/day-selector view (real, valid Side-A analysis) was not deleted — it moved verbatim to
a new route `/schedule/day` (`frontend/src/pages/ScheduleDayPage.tsx`), explicitly labeled
"Historical Day View," reachable via a new link on `SchedulePage`. `NavBar`'s `/plan` label
changed from the ambiguous "Plan Summary" to "Backtest Summary," since it's `OverviewPage`'s
fold-scoped historical summary, not Side B's operational one. Three smaller, explicitly-requested
UX fixes rode along: `Area N` → `Work Area N` display labels (new `copy.ts::workAreaLabel`,
client-side only — Component 20's own raw label is unchanged); `SupervisorPlanReviewPage`'s
undecided-row text is now context-aware ("No decision recorded yet" pre-approval vs. "No decision
was recorded before this plan was approved" post-approval); and the ROC-AUC disclosure
(`HOW_TO_USE_PRIORITY`) on `RecommendationsPage`/`EstablishmentDetailPage`, previously always-visible,
is now collapsed behind "How Sentinel prioritizes locations," matching the rest of the app's
progressive-disclosure convention.

**Explicitly not built, on purpose:** no "generate today's plan" button or endpoint (builds stay a
CLI/operator action, ADR 0049 — the frontend only ever reads and honestly labels the latest built
artifact); no operational-`planning_date` picker or prev/next-day control for Side B (it never had
one; adding it is a materially larger change not required to fix this bug — "today always means
the latest Side-B artifact, labeled honestly" is sufficient and matches Side B's existing no-scope
convention).

New tests: `frontend/src/lib/today.test.ts` (6), `TodayPage.test.tsx` fully rewritten (6),
`ScheduleDayPage.test.tsx` (new, 4, the old `TodayPage.test.tsx` content moved verbatim),
`copy.test.ts` (+9: `planLabelForToday`, `planStalenessNote`, `workAreaLabel`), plus updates to
`NavBar.test.tsx`, `SchedulePage.test.tsx`, `GeographicPlanPage.test.tsx`,
`SupervisorPlanReviewPage.test.tsx`, `RecommendationsPage.test.tsx`, `EstablishmentDetailPage.test.tsx`
for the label/wording changes. Frontend suite: 89 → **110 passing** (17 → 18 files). `tsc --noEmit`
and `oxlint` clean. No backend code changed — verified via the real running API that
`GET /v1/plan-review/summary`, `/work-blocks`, and `/rows` all now agree on `planning_date=2026-09-04`
(30 selected, 22 work blocks, approved), exactly what Today/Field Plan/Plan Review each read.

### Fixed broken establishment navigation from the live plan (2026-09-05)

A second real bug, found immediately after the date fix above and caused by the exact same class
of Side A/Side B conflation: clicking an establishment from `TodayPage`/`GeographicPlanPage`
(both Side B) led to "We couldn't find that record." Root cause, confirmed by reading
`src/sentinel/api/services/establishment_service.py::get_establishment_history` directly: both
pages linked into `EstablishmentDetailPage.tsx` (route `/establishments/:establishmentId`),
whose backend call, `GET /v1/establishments/{id}`, filters **Component 13's fold-scoped**
`inspection_recommendations` table — a historical population entirely different from "every real
establishment Sentinel knows about." Whether a given establishment appeared there depended on
whether it happened to be scored for whatever historical fold `useDefaultScope` auto-selected, a
condition with no relationship to the live operational plan at all.

Fixed with a new, genuinely Side-B-scoped page, **`frontend/src/pages/EstablishmentPlanDetailPage.tsx`**
at route `/plan/establishments/:targetInspectionId`, built entirely from an already-existing,
already-tested endpoint that the frontend had never called: `GET /v1/plan-review/rows/{target_inspection_id}`
(Component 21's own single-row lookup). No new backend code. Shows Sentinel priority (worded
honestly as `#{rank} in today's plan` — real data confirmed `PlanRowOut.rank`/`.policy_rank` are
identical, 1..30, i.e. rank *within the plan*, not the 35,859-candidate universe; reusing Side-A's
`relativePriorityLabel` here would have silently misrepresented the number), why it's prioritized,
current-plan/work-area status, and the existing `PlanDecisionForm` reused unmodified — machine
recommendation and supervisor decision rendered as visibly separate, never merged. States plainly,
in a collapsed "More inspection history" section, that deeper historical feature detail and SHAP
explanations aren't available for live operational establishments today (Component 18 doesn't run
explainability) — an honest limitation, not a silently missing field. `GeographicPlanPage.tsx`,
`TodayPage.tsx` (the two broken call sites, grep-confirmed to be the complete list), and
`SupervisorPlanReviewPage.tsx` (added a link from the existing inline-expand row for consistency)
now all route through it.

Verified against real data, across multiple establishments, not just one: fetched real rows from
the current plan and confirmed `GET /v1/plan-review/rows/{id}` returns `200` for the first-ranked,
last-ranked, middle-ranked, a grouped, a singleton-group, and a missing-location establishment —
six distinct real records, all resolving correctly (no establishment in the current committed plan
yet carries a supervisor decision to include as a seventh real category; that path is covered by a
mocked-fixture test instead, reported honestly rather than claimed as real-data-verified).

New tests: `EstablishmentPlanDetailPage.test.tsx` (5, new), plus navigation-destination assertions
added to `TodayPage.test.tsx`, `GeographicPlanPage.test.tsx`, and `SupervisorPlanReviewPage.test.tsx`.
Frontend suite: 110 → **118 passing** (19 files). `tsc --noEmit`/`oxlint` clean. No backend files
touched; full backend regression suite re-run to confirm.

### Final acceptance audit (2026-09-05)

A strict, no-new-features audit against the full C17-21 pipeline and its frontend, using real
data end to end. Found and fixed two more genuine gaps beyond what prior passes covered, both
root-caused rather than patched:

**"Needs Attention" was a single merged list, not two separate sections.** `HumanReviewPage.tsx`
computed both trigger reasons (`policy_warning_present`, `no_execution_record_on_scheduled_row`)
into one de-duplicated list with an inline per-row tag distinguishing them — real separation, but
weaker than the "visually and semantically separate" bar the product requires, since a case with
both reasons showed them concatenated in one line. Restructured into two headed `<section>`s
("Decision review (N)" / "Missing outcomes (N)"), each with its own count, its own empty state,
and its own honesty statement ("this is not evidence Sentinel decided wrong" /
"a record-keeping gap, not evidence anything went wrong") — a case needing both now genuinely
appears in both sections rather than being merged into one ambiguous line.

**A decision recorded after approval gave no warning that it wouldn't retroactively change the
approved plan.** `PlanDecisionForm` (shared by `SupervisorPlanReviewPage` and the new
`EstablishmentPlanDetailPage`) gained an optional `planAlreadyApproved` prop; when true, both the
closed and open states of the form now state plainly that a new decision "will not change the
approved plan -- it will be included the next time this plan is reviewed and re-approved." Backend
semantics were already correct and already documented (approval immutability was never meant to
block further decisions, only to protect the already-written artifact) — this closes the gap
between that documented intent and what the UI actually communicated. `EstablishmentPlanDetailPage`
also gained its own `getPlanSummary` read so it states the same plan-level approval status
Today/Plan Review show, closing a real cross-page consistency gap (the establishment-level page
previously said nothing about plan-level approval at all).

**A genuine, previously-uncaught pre-existing code-quality gap, found only because this audit ran
`ruff format`/`ruff check`/`mypy` over the whole `src/sentinel/` tree rather than only
touched files**: `src/sentinel/geographic_organization/organization.py` had a real (if ultimately
behavior-preserving, confirmed by re-running its 20-test suite) closure-over-loop-variable pattern
flagged by `ruff`'s B023 rule, plus two files with missing generic type arguments and one
`object`-typed index access mypy had never been run against directly. Fixed with the standard
default-argument closure-binding idiom (no behavior change — the closure was already used
synchronously within the same loop iteration it was defined in, so this was a static-analysis
false positive, not a live bug, verified by tracing the control flow) and precise `dict[str, Any]`
/`cast` annotations. 20 of the 25 unformatted files (across Components 17-21 plus two API files)
were re-formatted to close a formatting-drift gap that had grown silently across this session's many
edits (the 5 Component-9 calibration files remain deliberately unformatted, per the pre-existing,
documented decision in this file). `ruff check`/`ruff format --check`/`mypy` on `src/sentinel/` are
now clean except that one pre-existing exception and the unrelated `entity_service.py` E501.

**Verified end to end with a real decision + re-approval cycle**, not just reads: staged and
committed a real `adjust_operational_priority` decision via `sentinel review-plan --decisions`
against the live 2026-09-04 plan, confirmed `rank`/`policy_rank` stayed byte-identical (6/6) while
`operational_priority` changed to 1 and a full audit record appeared in `plan_decision_log`
(actor, reason, timestamp, outcome=applied), then re-ran `sentinel approve-plan` and confirmed the
new state (`decisions_recorded: 1`, `approval_status: approved`, `final_undecided_count: 29`)
is what every Side-B API endpoint — and therefore every Side-B page — now agrees on. Re-verified
establishment navigation across 7 categories (first/middle/last-ranked, grouped, singleton-group,
missing-location, and now a real decision-bearing establishment) — all `200`.

Frontend: 118 → **120 passing**. `tsc --noEmit`/`oxlint` clean. Backend: full suite unaffected in
count (**3,384 passed, 3 deselected**, re-run twice to confirm, once before and once after the
`geographic_organization` fixes), `ruff check`/`ruff format --check`/`mypy` now clean on every file
this session touched. No staged test artifacts left behind (this pass used the real CLI directly,
never the staging store, so there was nothing to clean up).
