# Sentinel

Risk-prioritized food inspection scheduling using Chicago open data.

The goal of Sentinel is to help a public health department decide **which food
establishments to inspect next**, by combining a calibrated risk model with a
deterministic statutory policy engine and constrained scheduling. Most of that
system does not exist yet.

This repository is being built **one component at a time**. See
[STATUS.md](STATUS.md) for the authoritative project state.

---

## Current status

**Components 1-9 complete: ingestion, entity resolution, target construction,
as-of feature engineering, temporal evaluation, baseline risk models,
gradient-boosted risk models, a neural network with entity embeddings, and
probability calibration.**

What exists today:

* A reproducible Python 3.12 project managed with `uv`.
* An explicit, paginating, retrying client for the Chicago Data Portal's
  Socrata API.
* Raw ingestion of the Food Inspections dataset (`4ijn-s7e5`) into
  timestamped Parquet files. Verified end-to-end on the full 314,245 rows.
* A JSON manifest recording provenance for every ingestion and resolution run.
* A DuckDB query layer for inspecting the raw Parquet.
* **Entity resolution**: a deterministic, explainable mapping from each
  inspection to a stable `establishment_id` representing a physical premises,
  plus an audit table recording why every merge — and every declined merge —
  was decided the way it was.
* **Target construction**: a precise, leakage-safe prediction target -- for each
  establishment-date on which a routine canvass occurred, did that canvass find
  at least one Priority or Priority Foundation violation?
* **As-of feature engineering**: 26 historical features per prediction
  opportunity, each computed strictly from inspections dated *before* that
  inspection, with the boundary enforced in one place and re-checked on every row.
* **Temporal evaluation**: a rolling-origin backtest with 17 quarterly folds, a
  model-agnostic prediction contract, and a five-schedule re-ordering simulation
  that measures whether a risk ranking would have surfaced violations earlier
  than the city's actual inspection order -- with the calibration window placed
  strictly between training and test, before any model exists to need it.
* **Baseline risk models**: three L2 logistic regressions, refitted once per
  fold (54 fits), with all preprocessing statistics taken from the training
  window only. The best reaches NDE 0.2326 and +5.70 mean days-earlier against
  the strongest heuristic's 0.1845 and +4.47, winning on all 17 quarterly folds.
* **Gradient-boosted risk models**: XGBoost and LightGBM on the same 26 features,
  each tuned with 100 Optuna trials per fold set under a protocol where every
  hyperparameter is selected from data strictly earlier than any test window. The
  measured answer is a **small** improvement — NDE 0.2326 → 0.2376, about +2%
  relative — that the logistic baseline beats on 7 of 17 individual folds, and that
  sits inside the seasonality sensitivity band. Two very different nonlinear
  learners landing within 0.005 of a penalised GLM is evidence the ceiling is the
  feature representation, not the estimator.
* 1,911 unit and integration tests with mocked HTTP, plus an opt-in live smoke
  test.
* **Neural network with entity embeddings**: a PyTorch MLP (embeddings for chain,
  facility type, community area and ZIP, concatenated with the same 30 standardised
  columns Components 6 and 7 use, through 256 and 128 hidden units to a single logit),
  trained under the same rolling-origin folds with early stopping carved from the *end
  of each training window* so no calibration or test row can influence when a fit stops.
  Two results point opposite ways. The network **on the same 26 features and nothing
  else** posts the best NDE in the project — **0.2482** against XGBoost's 0.2376 — and
  the best calibration (Brier 0.2355, ECE 0.0563), winning **12 of 17 folds**. But
  **adding the entity embeddings made it markedly worse** (0.2215), every ablation
  improves on the full model, and the learned chain vectors are statistically
  indistinguishable from a random table. The neural advantage is also the same size as
  its own five-seed spread (0.0053 against 0.0058), so it is **suggestive, not
  decisive**, and is not a deployment recommendation.

* **Probability calibration**: the models ranked well but their probabilities were wrong
  in a specific direction — **every one of the five candidates was underconfident**, with a
  calibration slope of 0.61–0.79 where 1.0 is perfect. Platt scaling, fitted per fold on the
  calibration window that Components 5–8 deliberately left untouched, pulls the slope to
  **1.00–1.03** and cuts ECE by **20–25%** (e.g. XGBoost 0.0621 → 0.0474) and MCE by 17–34%.
  The Brier decomposition shows why that is the right shape of answer: **reliability falls
  16–46% while resolution is unchanged to five decimal places** — a monotone map cannot
  create the ability to separate risk, and the measurement says so exactly. **The ranking is
  bit-for-bit untouched**: PR-AUC, ROC-AUC, NDE and precision@k all move by exactly
  0.00e+00, verified by re-running the Component 5 evaluator on the calibrated artifact.
  Two things had to be got right rather than assumed. The scores to calibrate **did not
  exist** — no component had ever scored a calibration window and no fitted model was
  persisted anywhere — so Component 9 re-executes Components 6–8's unchanged fits behind a
  **bit-identity gate** (207,680 rows compared with `==`, zero mismatches). And the choice
  between Platt and isotonic is made on an *expanding prefix* of folds rather than a pool,
  because fold N's calibration window **is** fold N−1's test window. Platt won all 90
  (model, fold) cells under a rule frozen in an ADR before any test window was opened.

Nothing else. No scheduling, no policy engine, no agent. The
evaluation harness was built *before* any model existed, so the first model was
measured by a yardstick it did not get to shape — and the models report no metrics
of their own; they emit scores and hand them to the evaluator.

**Read the caveats with the headline.** PR-AUC 0.5321 sits against a no-skill floor
of 0.4307, so the gain is +0.10. 43.24% of violations are still discovered *later*
than the city's actual order, because re-ordering under a fixed capacity is
zero-sum. And on the distribution-shift fold the model ordering **inverts** —
selecting on the rolling folds would have picked the wrong model for 2020. Calibration
**helped** on that fold (ECE −10 to −23%) but reached a slope of only 0.75–0.90, because the
base rate moves 17 points between its calibration and test windows and **prior shift is not
something a monotone recalibration can fix**. And calibration made ECE *worse* on 16 of the
85 quarterly (model, fold) cells; that is reported by a check deliberately incapable of failing, because
a check that went red on a worse number would create pressure to tune until it went green.

---

## Architecture

Five components: one path from the API to a model-ready training table, and one
honest way to measure a ranking built on top of it.

```text
Chicago Data Portal (Socrata SODA 2.1)
  https://data.cityofchicago.org/resource/4ijn-s7e5.json
                │
                │  GET ?$limit=1        field discovery (see note below)
                │
                │  GET  $limit / $offset / $order=inspection_id / $select
                │  bounded retry on 429, 5xx, timeouts
                ▼
        SocrataClient.iter_pages()          src/sentinel/ingest/socrata.py
                │  generator, one page resident at a time
                ▼
        response validation                 JSON array of objects, or raise
                │  X-SODA2-Fields / X-SODA2-Types captured
                ▼
        records -> Polars DataFrame         all columns Utf8, no coercion
                │
                ▼
        data/raw/food_inspections/
          food_inspections_<UTC>.parquet    zstd, timestamped, never overwritten
          manifest_food_inspections_<UTC>.json
                │
                ▼
        DuckDB  read_parquet(...)           src/sentinel/query/duckdb_queries.py
                │
                ▼
              SQL result


        ---- Component 2: entity resolution ----------------------------

        data/raw/food_inspections/*.parquet
                │  identity columns only; no results/violations/risk/date
                ▼
        normalize names, addresses, coordinates   entity/normalize.py
                │
                ▼
        build nodes                               entity/nodes.py
                │  314,245 rows -> 51,099 distinct identity signatures
                ▼
        block on (zip, house), coordinate, licence entity/blocking.py
                │  every non-licence block requires location agreement
                ▼
        evaluate pairs against named rules         entity/evidence.py
                │  S1-S3 strong · P1-P2 probable · A1-A2 ambiguous · V1-V4 veto
                ▼
        union-find + cluster invariants            entity/cluster.py
                │  deterministic split ladder if an invariant fails
                ▼
        data/interim/entity_resolution/
          establishment_assignments_<UTC>.parquet   inspection_id -> establishment_id
          establishments_<UTC>.parquet              one row per establishment
          entity_resolution_edges_<UTC>.parquet     the audit trail
          manifest_establishment_assignments_<UTC>.json


        ---- Component 3: target construction --------------------------

        raw Parquet  +  establishment_assignments
                │  identity columns are Component 2's contract, never re-derived
                ▼
        eligibility gates                         target/construct.py
                │  era >= 2018-07-01 · type == CANVASS
                │  results in {Pass, Pass w/ Conditions, Fail}
                ▼
        parse and classify violations             target/violations.py
                │  PRIORITY / PRIORITY FOUNDATION markers,
                │  narrative text excluded, violation number NOT used
                ▼
        collapse same establishment-date          target/construct.py
                │  one scheduling decision = one row, target = OR
                ▼
        data/interim/target/
          inspection_targets_<UTC>.parquet         313,624 rows
          manifest_inspection_targets_<UTC>.json


        ---- Component 4: as-of feature engineering --------------------

        raw Parquet + assignments + targets
                │
                ▼
        one range join, one temporal condition    features/historical.py
                │  h.inspection_date < t.inspection_date
                │  strictly before, never same-day
                ▼
        26 features in six families               features/definitions.py
                │  canvass history · priority history · windows
                │  context · tenant change · observation
                ▼
        validate: boundary re-derived per row     features/validate.py
                │
                ▼
        data/processed/features/
          as_of_features_<UTC>.parquet             57,727 rows x 33 columns
          manifest_as_of_features_<UTC>.json


        ---- Component 5: temporal evaluation --------------------------

        as-of feature table
                │
                ▼
        rolling-origin folds                      evaluation/folds.py
                │  TRAIN (expanding) -> CAL -> TEST, quarterly
                │  17 folds, 2022Q2 .. 2026Q2, partial windows excluded
                ▼
        prediction contract                       evaluation/contract.py
                │  exact coverage · no imputation · declared horizon
                ▼
        five reference schedules                  evaluation/simulate.py
                │  optimal · model · business-as-usual · random · worst
                │  capacity held constant by construction
                ▼
        metrics + simulation + sensitivity        evaluation/metrics.py
                │  ROC-AUC · PR-AUC · precision@k · discovery curves
                │  days-earlier distribution · seasonal re-draw
                ▼
        data/processed/evaluation/
          evaluation_folds_<UTC>.parquet           18 folds
          evaluation_metrics_<UTC>.parquet         2,808 rows, tidy long
          discovery_curves_<UTC>.parquet           373,986 points
          simulation_summary_<UTC>.parquet         504 rows
          seasonality_ / sensitivity_<UTC>.parquet
          manifest_evaluation_folds_<UTC>.json


        ---- Component 6: baseline risk models -------------------------

        as-of feature table + the same folds
                │
                ▼
        one model per fold, refitted                modeling/train.py
                │  train = assign_split(...) == "train", sorted canonically
                │  impute + scale fitted on TRAINING ROWS ONLY
                │  3 models x 18 folds = 54 fits
                ▼
        scores                                      modeling/predict.py
                │  P(target = 1); higher = higher risk
                │  trained_through = fold.train_end, never calibration_end
                ▼
        data/processed/predictions/                 <- a third artifact kind
          baseline_predictions_<UTC>.parquet         124,608 rows
          baseline_coefficients_<UTC>.parquet        1,530 rows
          baseline_training_log_<UTC>.parquet        54 rows
          manifest_baseline_predictions_<UTC>.json
                │
                │  sentinel evaluate --predictions <path>
                ▼
        back through Component 5's contract and metrics — unchanged,
        model-agnostic, and the sole source of every reported number

        ---- Component 7: gradient-boosted risk models -----------------

        data/processed/features/as_of_features_<UTC>.parquet
                │
                │  sentinel tune-boosting --trials 100    boosting/tuning.py
                │  region per fold set = first fold's train_start..calibration_end
                │     quarterly    2018-07-01..2022-03-31  <  first test 2022-04-01
                │     covid_shift  2018-07-01..2020-05-31  <  first test 2020-06-01
                │  6 / 2 inner rolling-origin folds, each with the outer gap
                │  early stopping happens HERE and only here
                ▼
        data/processed/tuning/                      <- a fourth artifact kind
          tuning_trials_<UTC>.parquet                400 rows, 0 failed
          manifest_tuning_trials_<UTC>.json
                │
                │  the winner is FROZEN BY HAND into definitions.TUNED_PARAMS,
                │  as a literal, so the choice appears in a diff
                ▼
        one model per fold, refitted                boosting/train.py
                │  matrix = 26 features + 4 indicators
                │  NULL -> NaN, routed natively: NOT imputed, NOT scaled
                │  frozen n_estimators rounds, NO eval_set, NO early stopping
                │  3 models x 18 folds = 54 fits
                ▼
        scores                                      boosting/predict.py
                │  RAW predict_proba[:, 1]; higher = higher risk
                │  trained_through = fold.train_end — literally, not nearly
                ▼
        data/processed/predictions/                 <- its own slug
          boosted_predictions_<UTC>.parquet          124,608 rows
          boosted_importances_<UTC>.parquet          1,620 rows (diagnostic only)
          boosted_training_log_<UTC>.parquet         54 rows
          manifest_boosted_predictions_<UTC>.json
                │
                │  sentinel evaluate --predictions <path>
                ▼
        the same Component 5 contract, unchanged. Component 6's artifact is
        untouched — verified byte-identical — so "did C7 beat C6?" stays answerable
```

**Component 4 asks "can my features see the future?". Component 5 asks "can my
evaluation see the future?". Component 6 asks "can my model see the future?"
Component 7 asks "can *I* see the future?"**
They are three different questions. A feature table with a perfect temporal
boundary still produces a dishonest result if it is split randomly, or if
calibration touches the test period. A random split over this table would train on
2024 and score on 2022; at a real decision point in 2022, 2024 has not happened.

And the third has a failure mode the first two cannot catch: an imputation median or
a scaler mean computed over the whole table *before* splitting. The fold boundary is
respected by the fit while the transform already knows the future, and nothing about
the split looks wrong. Component 6 therefore re-derives every preprocessing
statistic from the training rows and compares it to what was fitted.

The fourth question is different in kind, because its failure mode is a **person**.
Fit a booster, read a test metric, change `max_depth`, refit, keep the better one —
and the resulting artifact passes every check above. The predictions cover the test
window exactly, the declared horizon is honest, no training row is misdated. The model
is simply better than it should be, by an amount nobody can measure. Component 7 makes
that structurally impossible rather than merely discouraged: each fold set's
hyperparameters come from a search confined to a region that ends before that fold set's
first test window, and a check re-derives both dates from the fold definitions on every
run. See ADR 0017.

The estimand is stated before any result: Component 5 measures the **re-ordering**
of inspections that actually occurred. There are no labels for establishments
nobody visited, so coverage cannot be evaluated and no causal claim is made.
Component 6 does not change this — a trained model re-orders the same inspections
under the same fixed capacity, which is why moving one establishment earlier moves
another later, and why 43.24% of violations are still found later than the city's
actual order even under the best model.

Reasoning in
[`docs/analysis/temporal_evaluation_findings.md`](docs/analysis/temporal_evaluation_findings.md),
[ADR 0012](docs/decisions/0012-rolling-origin-temporal-evaluation.md) and
[ADR 0013](docs/decisions/0013-evaluation-results-are-artifacts-not-inputs.md).

The **as-of rule** is the whole of Component 4: a feature for the row at
`inspection_date = d` may use only records dated **strictly before** `d`. The
boundary is exclusive because `inspection_date` carries no time component in this
dataset, so same-day records cannot be ordered — and 43 same-day canvass
re-inspections at reference dates provably happened *after* the canvass they
follow. Observable consequence: `days_since_last_canvass` has a minimum of 1 and
contains no zeros.

Reasoning in
[`docs/analysis/as_of_feature_engineering_findings.md`](docs/analysis/as_of_feature_engineering_findings.md)
and [ADR 0010](docs/decisions/0010-as-of-feature-construction.md).

The **target** is: for each establishment-date on which a routine canvass
occurred, did that canvass find at least one **Priority or Priority Foundation**
violation? Not `results == 'Fail'` -- among canvasses, priority violations appear
in 97.9% of `Pass w/ Conditions` inspections, so a result-based label would
mislabel 16,261 of them. The reasoning is in
[`docs/analysis/target_construction_findings.md`](docs/analysis/target_construction_findings.md).

`inspection_date` is the **as-of boundary**: Component 4 may use only information
dated strictly before it.

An **establishment** is a *physical food-service premises*, not a licence and not
a business name. Successive tenants at one address are the same establishment
with a changing name; a mobile-food commissary holding 47 cart permits is one
establishment. The reasoning is in
[`docs/analysis/entity_resolution_findings.md`](docs/analysis/entity_resolution_findings.md).

### Raw stays raw

The Socrata API returns **every value as a JSON string**, including columns it
declares as `number` and `calendar_date`:

```json
{"inspection_id": "2641210", "inspection_date": "2026-08-14T00:00:00.000"}
```

The raw Parquet preserves that exactly: every column is written as `Utf8`. No
casting, no parsing, no cleaning. Two reasons:

1. **Fidelity.** A cast turns any unexpected value into a silent null. The raw
   layer must match the source so that a data quality problem is discoverable
   rather than already destroyed.
2. **Separation of concerns.** Typing and semantic interpretation are modelling
   decisions. They belong in a later component where they can be tested and
   documented, not smuggled into the download step.

The nested `location` object is serialized to its JSON string rather than
flattened or dropped, so nothing is lost.

### Why there is a field-discovery request

Stable pagination requires `$order`. This endpoint, however, **drops its five
`:@computed_region_*` columns** (Socrata-generated ward, community area, census
tract and zip-code spatial joins) from both the response and the schema header
whenever `$order` is present — unless they are named explicitly in `$select`.

Hardcoding a 22-column `$select` would fix that but create a worse problem: a
new upstream column would then be silently excluded. So ingestion instead issues
one unordered `?$limit=1` request to discover the current field list, and
selects exactly those fields. One extra request buys a complete raw layer that
still adapts to upstream schema changes.

Controlled by `SENTINEL_INCLUDE_COMPUTED_REGIONS` (default `true`). Full detail
in [docs/api/socrata_findings.md](docs/api/socrata_findings.md) §6.

---

## Data source

**Chicago Data Portal**, powered by Socrata (SODA 2.1).

| | |
|---|---|
| Dataset | Food Inspections |
| Dataset ID | `4ijn-s7e5` |
| Endpoint | `https://data.cityofchicago.org/resource/4ijn-s7e5.json` |
| Authentication | None required |
| Pagination | `$limit` + `$offset`, ordered by `$order=inspection_id` |
| Portal page | <https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5> |

The API is used directly. No HTML scraping, no manually downloaded CSV, no
third-party SDK wrapping the pagination.

An application token is **not** required. Socrata app tokens only relieve
anonymous rate-limit throttling; they grant no additional data access. The
optional `SENTINEL_SOCRATA_APP_TOKEN` setting exists for that purpose alone.

Detailed API findings, verified against the live service, are in
[docs/api/socrata_findings.md](docs/api/socrata_findings.md). The raw data
contract is in
[docs/data_contracts/food_inspections_raw.md](docs/data_contracts/food_inspections_raw.md).

---

## Running locally

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
# install dependencies into .venv
uv sync

# optional: start from the documented defaults
cp .env.example .env
```

### Development ingestion

Small, fast pull. Use this while working.

```bash
uv run sentinel ingest --dev              # uses SENTINEL_DEV_ROW_LIMIT (default 5000)
uv run sentinel ingest --limit 1000       # explicit row cap
```

### Full ingestion

Pulls the entire dataset (~314k rows at time of writing).

```bash
uv run sentinel ingest --full
```

### Other options

```bash
uv run sentinel ingest --limit 5000 --page-size 1000 --log-level DEBUG
uv run sentinel ingest --limit 5000 --output-dir /tmp/sentinel-raw
```

One of `--dev`, `--limit`, or `--full` is required. There is no default scope,
so a bare `sentinel ingest` cannot accidentally trigger a full download.

### Querying the raw data

```bash
uv run sentinel query --list                       # show available queries
uv run sentinel query --name row_count             # uses the most recent raw file
uv run sentinel query --name inspection_types
uv run sentinel query --name results_breakdown
uv run sentinel query --name inspection_date_range
uv run sentinel query --name schema
uv run sentinel query --name row_count --parquet path/to/file.parquet
```

These queries only describe the raw data. They contain no Sentinel filtering or
business logic.

> Because every raw column is `VARCHAR`, aggregating over dates or numbers
> requires an explicit cast in SQL. That is intentional — see
> [ADR 0002](docs/decisions/0002-parquet-raw-storage.md).

> **A development pull is not a random sample.** Pages are ordered by
> `inspection_id`, which correlates with time, so `--limit 5000` returns the
> *oldest* 5,000 inspections (observed: 2010-01-04 to 2011-05-09). Never
> estimate distributions from a dev extract.

---

### Resolving establishment identities

```bash
# resolve the most recent raw snapshot, writing three tables + a manifest
uv run sentinel resolve

# compute and validate without writing anything, printing the full report
uv run sentinel resolve --dry-run --report

# resolve a specific file into a specific directory
uv run sentinel resolve --parquet path/to/raw.parquet --output-dir out/
```

The command exits non-zero if a structural validation check fails, so broken
identities stop the pipeline rather than being handed quietly to the next
component. On the full snapshot it takes about 45 seconds.

To ask why two inspections were or were not treated as the same establishment,
filter the edges table on their `node_id`:

```python
import duckdb

duckdb.sql("""
    SELECT rule_id, tier, same_license, same_addr_key, name_exact, left_name, right_name
    FROM read_parquet('data/interim/entity_resolution/entity_resolution_edges_*.parquet')
    WHERE left_node_id = 'N-...' OR right_node_id = 'N-...'
""").show()
```

---

### Building the prediction target

```bash
# build from the latest raw snapshot and the latest Component 2 assignments
uv run sentinel build-target

# construct and validate without writing anything
uv run sentinel build-target --dry-run --report
```

Exits non-zero if a structural validation check fails. Takes about 25 seconds on
the full snapshot.

To see why a row is labelled the way it is, read its evidence span:

```python
import duckdb

duckdb.sql("""
    SELECT establishment_id, inspection_date, target, results, evidence
    FROM read_parquet('data/interim/target/inspection_targets_*.parquet')
    WHERE target = 1
    LIMIT 5
""").show()
```

---

### Building the as-of feature table

```bash
# build from the latest raw snapshot, assignments and targets
uv run sentinel build-features

# construct and validate without writing anything
uv run sentinel build-features --dry-run --report
```

Exits non-zero if a validation check fails — including the temporal invariant,
which is re-derived independently and checked on every row. Takes about 15
seconds on the full snapshot.

```python
import duckdb

duckdb.sql("""
    SELECT establishment_id, inspection_date,
           prior_canvass_count, days_since_last_canvass,
           prior_canvass_priority_rate, target
    FROM read_parquet('data/processed/features/as_of_features_*.parquet')
    LIMIT 5
""").show()
```

---

### Training the baseline models

```bash
# three models x 18 folds = 54 fits, ~30 seconds
uv run sentinel train-baselines

# one model only; the flag is repeatable
uv run sentinel train-baselines --models logistic_regression

# train and validate without writing anything
uv run sentinel train-baselines --dry-run --report
```

Exits non-zero if any of the fifteen error-severity checks fails. The one worth
knowing about re-derives every imputation median from the fold's training rows and
compares it to the fitted imputer — because a preprocessing statistic computed over
the whole table before splitting is leakage that no fold boundary catches: the
boundary is respected by the *fit* while the *transform* already knows the future.

This command reports **no metrics**. It writes a prediction artifact and points you
at the evaluator.

---

### Tuning and training the boosted models

Tuning and training are separate commands, and separate from evaluation.

```bash
# 100 Optuna trials per model per fold set: 4 studies, 400 trials, ~9.5 minutes
uv run sentinel tune-boosting --trials 100 --report

# one model, one fold set; both flags are repeatable
uv run sentinel tune-boosting --models xgboost --fold-set quarterly --trials 20

# search without writing anything
uv run sentinel tune-boosting --dry-run --report
```

`tune-boosting` prints each study's search region beside its fold set's first test
start, so the safety property is legible rather than buried in a manifest:

```text
tuning regions (each must end before its fold set's first test start):
  xgboost-quarterly:   2018-07-01..2022-03-31  <  first test 2022-04-01
  xgboost-covid_shift: 2018-07-01..2020-05-31  <  first test 2020-06-01
```

It then prints the parameter block to paste into `boosting/definitions.py`. **It edits
no source file.** The manual freeze is deliberate: a parameter set loaded from disk at
training time could change without a diff, and freezing is only meaningful if it cannot.

```bash
# three models x 18 folds = 54 fits, ~21 seconds; uses the frozen parameters
uv run sentinel train-boosting

# one model only; repeatable
uv run sentinel train-boosting --models lightgbm

# train and validate without writing anything
uv run sentinel train-boosting --dry-run --report
```

Exits non-zero if any of the fifteen error-severity checks fails. Two are specific to
this component: one rebuilds each fold's matrix and asserts its NaN pattern equals the
source frame's NULL pattern cell-for-cell — catching an accidentally reintroduced
imputer — and one asserts no final fit carries an early-stopping parameter, because that
would mean reading a window later than the declared horizon.

Like `train-baselines`, this reports **no metrics**, and it writes to its own slug so
Component 6's artifact stays byte-identical.

### Training the neural models

```bash
# Component 8's experimental categorical layer. Carries chain, facility type, community
# area and ZIP forward from the establishment's most recent EARLIER inspection. Not a
# Component 4 feature table -- see ADR 0022. ~0.5 s
uv run sentinel build-neural-categoricals --report

# The learning-rate sweep: 5 rates x 8 inner folds over 2 studies, ~9 minutes. Prints a
# block to paste into neural/definitions.py; it edits no source file.
uv run sentinel tune-neural --report

# 9 models x 18 folds, plus 4 extra seeds x 18 folds for the reproducibility experiment.
# 234 fits, ~33 minutes single-threaded on CPU. Writes the figures too.
uv run sentinel train-neural --report

# Component 5 scores it, exactly as it scores Components 6 and 7.
uv run sentinel evaluate --predictions data/processed/predictions/neural_predictions_<stamp>.parquet --report

# Component 9 -- fit and freeze the probability calibrators (~25 min).
# Run WITHOUT an OMP_NUM_THREADS override: the bit-identity gate is thread-sensitive.
uv run sentinel calibrate --report
uv run sentinel evaluate \n  --predictions data/processed/predictions/calibrated_predictions_<stamp>.parquet --report
```

Every fit is single-threaded on the CPU with `torch.use_deterministic_algorithms(True)`,
so re-runs are bit-identical. A CUDA device on the build machine is deliberately unused:
GPU reductions are not bit-reproducible, and that is the standard every leakage test in
this repository is written against. See [ADR 0020](docs/decisions/0020-pytorch-and-matplotlib-as-runtime-dependencies.md).

### Running the temporal evaluation

```bash
# heuristics only: 17 quarterly folds + the distribution-shift fold, ~165 seconds
uv run sentinel evaluate

# heuristics plus the fitted models, ~238 seconds
uv run sentinel evaluate \
  --predictions data/processed/predictions/baseline_predictions_<stamp>.parquet

# just the fold table -- fast, and enough to audit the split
uv run sentinel evaluate --folds-only --report

# evaluate and validate without writing anything
uv run sentinel evaluate --dry-run --report
```

Exits non-zero if any of the fourteen error-severity checks fails — including
the seven leakage checks, the capacity-conservation check, and the
business-as-usual identity. A failure means the evaluation itself could see the
future, which would make every number it reports confidently wrong.

`--predictions` is where a model enters. The evaluator does not know, and must not
know, what produced a score: it validates the claim (exact coverage, no nulls, a
declared training horizon inside the fold) and then measures it exactly as it
measures the six built-in heuristics.

```python
import duckdb

# how did each schedule do, per fold?
duckdb.sql("""
    SELECT fold_id, schedule_name, model_name,
           normalized_discovery_efficiency, mean_days_earlier,
           std_days_earlier, fraction_worse
    FROM read_parquet('data/processed/evaluation/simulation_summary_*.parquet')
    WHERE fold_set = 'quarterly'
    ORDER BY fold_id, normalized_discovery_efficiency DESC
    LIMIT 10
""").show()
```

**Read the estimand before the numbers.** Component 5 measures the re-ordering of
inspections that actually occurred; it cannot evaluate coverage, and it makes no
causal claim. See
[`docs/data_contracts/temporal_evaluation.md`](docs/data_contracts/temporal_evaluation.md) §1.

---

## Output

Every ingestion run writes two files into `data/raw/food_inspections/`:

```text
food_inspections_20260815T143000Z.parquet          the data, zstd-compressed
manifest_food_inspections_20260815T143000Z.json    the provenance record
```

Filenames embed the UTC retrieval timestamp, so **a re-run never overwrites
previously downloaded raw data**. The raw layer is append-only by construction.

The manifest records source, dataset ID, retrieval timestamp, code version,
mode, row limit, page size, pages fetched, the exact request parameters for
every page, row count, column names, the Socrata-declared field types, the
resulting Parquet schema, output path, file size, and a SHA-256 checksum of the
Parquet file.

Parquet files are gitignored; manifests are committed, so the repository keeps a
history of what was ingested without carrying the bulk data.

Entity resolution writes three tables plus its own manifest under
`data/interim/entity_resolution/`, target construction writes one table plus a
manifest under `data/interim/target/`, and feature engineering writes the
model-ready table under `data/processed/features/` — all following the same
rules. [ADR 0011](docs/decisions/0011-processed-layer-for-model-ready-tables.md)
records what makes a table model-ready and therefore eligible for the processed
layer. The full schemas,
identifier semantics and stability guarantees are in
[`docs/data_contracts/establishment_assignments.md`](docs/data_contracts/establishment_assignments.md).
The short version: `establishment_assignments` maps every `inspection_id` to an
`establishment_id` and deliberately carries no dates, counts or outcomes, so a
downstream join cannot pull whole-history information into a training row.

The processed layer now holds **five** kinds of thing, each with its own directory and
its own prohibition:

```text
data/processed/features/      model-ready tables. Trainable.               ADR 0011
data/processed/predictions/   model outputs. Never trainable.              ADR 0014
data/processed/evaluation/    measurements about models. Never trainable.  ADR 0013
data/processed/tuning/        hyperparameter search trials.                ADR 0018
data/processed/neural/        Component 8's experimental categoricals.     ADR 0022
data/processed/calibration/   fitted calibrators and their diagnostics.    ADR 0024
```

The fifth is the newest and the most easily misread. Component 4's feature table has no
categorical column at all, so Component 8 carries chain, facility type, community area
and ZIP forward as-of from the raw snapshot into a layer of its own. It is **not a
feature table**, `feature_definition_version` is unchanged at `v1`, and no other
component may join it onto anything.

**Nothing in the last three may ever be joined onto a feature table**, and no number in
the tuning layer is a result — every one is measured on a validation window that is
*training* data for the folds the chosen parameters are then used on. A search's best
PR-AUC looks exactly like a result, which is why it is filed where it cannot be mistaken
for one.

---

## Testing

```bash
uv run pytest                 # unit tests, fully offline
uv run pytest -v
uv run pytest -m live         # opt-in: makes one real call to the Chicago API
```

1,776 tests pass and 3 live tests are deselected. Unit tests mock HTTP at the
transport layer with `respx`, so real request
construction, status handling, pagination and retry logic are all exercised
without touching the network. Live tests are marked and deselected by default,
so neither CI nor the normal test run depends on an external service.

Linting and type checking:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/sentinel scripts
```

---

## Repository layout

```text
src/sentinel/
  config.py                    configuration (env-driven, no hardcoded values)
  logging_setup.py             logging configuration
  cli.py                       argparse CLI: ingest, query, resolve,
                               build-target, build-features,
                               train-baselines, tune-boosting,
                               train-boosting, evaluate
  manifest.py                  generic manifest helpers (hash, read, write)
  ingest/
    socrata.py                 paginating, retrying Socrata client
    food_inspections.py        orchestration: pages -> Parquet + manifest
    manifest.py                ingestion provenance model
  query/
    duckdb_queries.py          DuckDB over the raw Parquet
  boosting/                    Component 7: gradient-boosted risk models
    definitions.py             BOOSTING_REGISTRY, the declared SEARCH_SPACE,
                               the frozen TUNED_PARAMS, import-time guard
    preprocess.py              the tree matrix: no imputation, no scaling
    tuning.py                  the search that cannot reach a test window
    train.py                   one fit per fold; no early stopping
    predict.py                 predict_proba -> raw score, direction fixed
    models.py                  FittedBooster: a typed facade over two libraries
    validate.py                twenty checks, each re-derived from the data
    writer.py                  four output schemas
    build.py                   tune_boosting / train_boosting (the only I/O)
  modeling/                    Component 6: baseline risk models
    definitions.py             MODEL_REGISTRY; partitions derived from
                               Component 4's FEATURE_SPECS; import-time guard
    preprocess.py              matrix construction + the train-only pipeline
    train.py                   one fit per fold; the canonical sort
    predict.py                 predict_proba -> score, direction fixed
    models.py                  FittedModel: a typed facade over sklearn
    validate.py                fifteen checks, each re-derived from the data
    writer.py                  three output schemas
    build.py                   orchestration (the only module doing I/O)
  evaluation/                  Component 5: temporal evaluation
    models.py                  FoldSpec (refuses a leaky fold), PredictionSet
    folds.py                   rolling-origin fold construction
    contract.py                the model-agnostic prediction contract
    metrics.py                 hand-rolled metrics, sklearn-verified in tests
    rankers.py                 deterministic baselines; nothing is fitted
    simulate.py                the slot model and the five schedules
    sensitivity.py             de-trended seasonality + label re-draw
    validate.py                the seven leakage checks, re-derived
    writer.py                  six output schemas
    build.py                   orchestration (the only module doing I/O)
  features/                    Component 4: as-of feature engineering
    definitions.py             FeatureSpec list; the single source of truth
    historical.py              the range join and the temporal boundary
    validate.py                checks, incl. the full-table temporal invariant
    writer.py                  schema derived from the specs
    build.py                   orchestration (the only module doing I/O)
  target/                      Component 3: target construction
    models.py                  frozen structures + the definition constants
    violations.py              deterministic violation parsing/classification
    construct.py               eligibility gates, labelling, same-day collapse
    validate.py                post-construction checks
    writer.py                  the output table schema
    build.py                   orchestration (the only module doing I/O)
  entity/                      Component 2: entity resolution
    models.py                  frozen data structures + DEFAULT_THRESHOLDS
    normalize.py               name / address / geo normalization
    nodes.py                   rows -> distinct identity signatures
    blocking.py                candidate pair generation
    evidence.py                signals, vetoes, named match rules
    unionfind.py               disjoint-set union (no networkx)
    cluster.py                 components, invariants, split ladder
    validate.py                post-resolution checks
    writer.py                  the three output tables
    resolve.py                 orchestration (the only module doing I/O)
scripts/profile_entities.py    read-only entity profiling (analysis, not library)
scripts/profile_target.py      read-only target profiling
scripts/profile_features.py    read-only history-availability profiling
scripts/profile_evaluation.py  read-only evaluation-surface profiling
scripts/profile_baselines.py   read-only model profiling; train windows ONLY
tests/                         unit + integration tests; tests/fixtures/ holds
                               real regression cases as literal Python
data/raw|interim|processed/    data layers; contents gitignored, manifests kept
docs/analysis/                 empirical findings
docs/api/                      verified API behaviour
docs/data_contracts/           one contract per component output
docs/decisions/                architecture decision records
```

---

## Project roadmap

Components 1-7 are implemented. Everything below them is **planned, not
implemented** — no code for any of it exists in this repository.

| # | Component | Status |
|---|---|---|
| 1 | Project foundation + Chicago data ingestion | **Implemented** |
| 2 | Entity resolution | **Implemented** |
| 3 | Target construction | **Implemented** |
| 4 | As-of feature engineering | **Implemented** |
| 5 | Temporal evaluation framework | **Implemented** |
| 6 | Baseline risk models | **Implemented** |
| 7 | XGBoost / LightGBM | **Implemented** |
| 8 | Neural baseline | **Implemented** |
| 9 | Probability calibration | **Implemented** |
| 10 | Inspector-effect modelling | **Blocked** — the dataset has no inspector field (ADR 0019) |
| 11 | SHAP explainability | Not implemented — C7 emits split-gain importances as a diagnostic only |
| 12 | Fairness auditing | Not implemented |
| 13 | Deterministic statutory policy engine | Not implemented |
| 14 | Constrained scheduling | Not implemented |
| 15 | OR-Tools routing | Not implemented |
| 16 | Deferral / human-review gate | Not implemented |
| 17 | LangGraph orchestration | Not implemented |
| 18 | LLM-generated inspector briefings | Not implemented |
| 19 | Deterministic briefing verification | Not implemented |
| 20 | Audit trail | Not implemented |
| 21 | Frontend demo | Not implemented |

Technologies for later components (PyTorch, OR-Tools, LangGraph, a frontend) are
deliberately absent from `pyproject.toml`. Each is introduced only when the component
that needs it is built — scikit-learn and numpy arrived with Component 6 (ADR 0015),
and xgboost, lightgbm and optuna with Component 7 (ADR 0016).

---

## License

MIT
