# Sentinel

Risk-prioritized food inspection scheduling using Chicago open data.

The goal of Sentinel is to help a public health department decide **which food
establishments to inspect next**, by combining a calibrated risk model with a
deterministic policy engine and constrained scheduling. The risk model and the
policy engine now exist; the scheduling and routing layers do not.

This repository is being built **one component at a time**. See
[STATUS.md](STATUS.md) for the authoritative project state.

---

## Current status

**Components 1-9 and 11-14 complete: ingestion, entity resolution, target construction,
as-of feature engineering, temporal evaluation, baseline risk models,
gradient-boosted risk models, a neural network with entity embeddings,
probability calibration, feature attribution, a geographic equity audit, a
deterministic decision policy with deployment governance, and an operational
schedule laid against the calendar Chicago actually worked.**
Component 10 is **blocked** — the dataset has no inspector field (ADR 0019).

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
* 3,009 unit and integration tests with mocked HTTP, plus an opt-in live smoke
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

* **Explainability and feature attribution**: for every supported model and every temporal
  fold, the evidence structure behind a score — which features contributed, in which
  direction, against a reference point that provably could not see the future. Exact
  TreeSHAP for the boosters, the exact closed form for the logistic model, and a seeded
  antithetic permutation game for the network, all in one output space (log-odds) so a
  cross-model comparison is a comparison of models rather than of units. **The headline
  finding is a disagreement**: the four models score within 0.0156 NDE of one another and
  their feature-importance rankings correlate at only **ρ = 0.44** between the logistic
  model and LightGBM, sharing 3 of their top 10 features. Near-identical accuracy,
  substantially different reasoning — which strengthens Component 7's "the ceiling is the
  representation, not the estimator" and removes any argument from explanation to model
  choice. Quarter to quarter the reasoning is stable (consecutive-fold rank ρ 0.89–0.98);
  over four years it drifts (first-to-last ρ 0.75–0.92). **Under the COVID regime shift
  three of four models leaned 2–3× harder on `days_since_any_inspection`**, which is the
  direct mechanism behind the model-ordering inversion Component 6 could only infer from an
  ablation. One of the five candidates, `xgboost_chain_embeddings`, is reported
  **unsupported** with a stated reason rather than explained through a private interface.
* **A geographic equity audit**: every ranking, calibration and top-k question asked again
  once per Chicago neighbourhood, over the two geographies this data can actually define —
  community area and ZIP — with five others refused for measured reasons (the two published
  ward layers disagree on **98.3%** of rows). 286 groups observed, **132 supported** at a
  200-row floor and 154 recorded as `insufficient_support` with their counts rather than
  dropped. **The within-city spread of ranking quality, 0.177 ROC-AUC, is larger than the
  entire difference between the best and worst model in this project.** The sharpest finding
  is about a group with no geography at all: `__UNKNOWN__` establishments are ranked at
  chance, and of their 166 citations the top 5% found one. Nothing was fixed — a measured
  disparity is advisory, never a build failure (ADR 0034).
* **A deterministic decision policy**: the layer that turns a calibrated probability into a
  capacity-constrained inspection queue, and prices every alternative. It settles which model
  Sentinel carries (`xgboost_platt`) from a rule fixed in advance — **and records that all
  four candidates were statistically indistinguishable on the headline metric**, so the tie
  fell to calibration. Seven policies compared: pure risk prioritisation against coverage
  floors and forced coverage reserves at half, exactly, and twice the measured no-history
  population share. **The result is a negative one and it is the component's most valuable
  output**: the risk queue already over-serves establishments with no code-era history by
  four to five times their population share, so a coverage floor is inert in 338 of 340
  quarterly cells, and a forced reserve at twice the share gives up **34 Priority citations a
  week** to serve 343 more of them. Every recommendation records whether the model or the
  policy put it there. No score is adjusted by geography, no quota exists, and where the
  evidence does not pick a policy the run prints *the data does not determine the correct
  policy* rather than picking one.

* **An operational schedule**: the layer that turns the approved queue into a plan, because a
  rank position is not something a person can execute. The horizon length is Component 5's own
  capacity rule **read backwards** — `ceil(k / median_daily_rate)`, which reproduces `k_1_day`
  as one day and `k_1_week` as five — and the operating days and their volumes are **read out of
  the data**, not assumed: the dates Chicago actually inspected on and the real number of
  inspections performed on each. So the component introduces no constant of its own. Two
  measurements come out of it. First, Component 13's capacity assumption is optimistic: in **44
  of 90** (fold, capacity) cells the real calendar cannot fit the queue that assumption
  approved — **784 inspections** — while under the flat median the backlog is zero in every
  cell, by construction. Second, and this is the component's most valuable output: Component 13
  places its **coverage reserve at the tail of the rank order**, so a short horizon takes it
  first — **1,012 of 3,459 reserve slots (29.3%) never get scheduled, and 91 of 273
  reserve-bearing cells lose it entirely**. Component 14 reports that and deliberately does not
  correct it, because correcting it would be re-ranking, which Component 13 owns. Priority is
  preserved exactly (**0 inversions** across 1,260 cells), every human decision and every
  real-world outcome is recorded in its own table, and **no route is optimised**: the dataset
  has 22 columns and none of them is an inspector.

Nothing else. No routing, no inspector assignment, no agent. The
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
And Component 13's headline cost figures are quarter-scale: the one-day policy deltas are
±1 to ±3 citations out of 348 across seventeen folds, which is inside the noise, and the
findings document says so rather than quoting the flattering one.

---

## Architecture

One path from the API to a model-ready training table, one honest way to measure a
ranking built on top of it, and one deterministic layer that turns the ranking into a
capacity-constrained recommendation.

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


        ---- Components 8-13: the same seam, five more times ------------

        Every component after 7 attaches at the same two points and is not
        redrawn above. Each one either produces scores that Component 5's
        contract accepts, or consumes an artifact and writes to a layer of
        its own:

          8  neural network      -> predictions/ + an experimental categorical layer
          9  calibration         -> predictions/ + calibration/   (probabilities)
         11  explainability      -> explanations/                 (why a score)
         12  fairness audit      -> fairness/                     (who it behaved how for)
         13  decision policy     -> policy/                       (who to inspect, and why)
         14  operational schedule -> scheduling/                  (when, and what it cost)

        Component 13 is the one that changes the shape of the output rather
        than adding to it. Everything above produces a DESCRIPTION; it
        produces an INSTRUCTION:

        data/processed/predictions/calibrated_predictions_<UTC>.parquet
                │  + the as-of history column that defines coverage eligibility
                │  + each window's measured median daily inspection rate
                ▼
        apply the frozen model-selection rule       policy/select.py
                │  NDE (tied) -> calibrated ECE -> xgboost_platt
                ▼
        for each policy x fold x capacity           policy/allocation.py
                │  risk block = top (k - reserve) by calibrated risk
                │  reserve    = eligible rows the risk block did not take
                │  NO SCORE IS WRITTEN, EVER
                ▼
        price every policy against pure_risk        policy/evaluate.py
                │  the delta is reported in Priority citations
                ▼
        data/processed/policy/
          inspection_recommendations_<UTC>.parquet  1,453,760 rows
          policy_comparison_<UTC>.parquet           what each policy cost
          policy_override_log_<UTC>.parquet         what a human decided instead
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

        ---- Component 14: operational scheduling ----------------------

        Component 13 says WHO, at a stated capacity, and why. It stops
        there, because its own capacity is a rank position:

            "capacity is a rank position derived from the window's
             measured median daily inspection rate"

        A rank position is not a plan. Nobody executes "you are the 137th
        most important inspection this quarter"; they execute "you are on
        Thursday". Component 14 crosses that gap, and refuses to cross any
        other.

        data/processed/policy/inspection_recommendations_<UTC>.parquet
                │  the approved queue and its ranks, carried verbatim
                │
                │  + inspection_date, grouped -> the fold's OBSERVED
                │    operating days and the real volume worked on each
                │  + test_median_daily_capacity (Component 5)
                ▼
        horizon = ceil(k / median_daily) observed operating days
                │  not a new constant: Component 5's capacity rule read
                │  backwards. k_1_day spans one day, k_1_week spans five.
                ▼
        deterministic greedy slot allocation, in final_policy_rank order
                │  reads no score, no mechanism, no geography
                ▼
        data/processed/scheduling/                    thirteen tables
          inspection_schedule_<UTC>.parquet           when, and in which slot
          schedule_backlog_<UTC>.parquet              approved, and not reached
          priority_preservation_<UTC>.parquet         what the calendar cost
          ...                                         + a manifest

        Two capacity modes run by default, and only one is a measurement.
        `observed_calendar` uses the volumes Chicago actually worked;
        `flat_median` uses the median every day, which is what Component
        13's cutoffs already assume -- and is therefore SATURATED BY
        CONSTRUCTION at k_1_day and k_1_week. The gap between them is the
        component's central measurement:

            44 of 90 (fold, capacity) cells cannot fit their approved
            queue into their own horizon -- 784 inspections in total.
            Under the flat median the backlog is zero in every cell.

        And the finding that surprised us, measured from inside Component
        14 about Component 13:

            Component 13 places the coverage reserve at the TAIL of the
            rank order. A strict-priority schedule fills from the top, so
            a short horizon takes the reserve first -- every time.
            1,012 of 3,459 reserve slots (29.3%) are lost to the horizon;
            91 of 273 reserve-bearing cells lose it entirely.

        Component 14 reports that and does not correct it. Promoting
        reserve rows would be re-ranking, which Component 13 owns.

        What it does NOT do: no route optimisation, no inspector
        assignment, no travel-time estimate. The dataset has 22 columns
        and none of them is an inspector (ADR 0019); routing is Component
        15's, and Component 15 is blocked on the same missing data.


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

# Component 11 -- attribute the predictions to features (~19 min).
# Same thread-sensitivity: the bit-identity gate re-proves the explained models are the
# committed ones (166,144 scores compared with `==`, zero mismatches).
uv run sentinel explain --report

# Component 12 -- audit group behaviour across geography (~145 s).
# Unlike calibrate and explain this has NO bit-identity gate and NO thread sensitivity:
# it fits nothing and re-executes nothing. Its integrity claim is the opposite one --
# every input's sha256 is compared before and after to prove nothing moved.
uv run sentinel audit-fairness --report

# audit and validate, writing nothing
uv run sentinel audit-fairness --dry-run --report

# one model, one geography
uv run sentinel audit-fairness --models xgboost_platt --group-definitions community_area

# Component 13 -- turn calibrated predictions into a recommended queue (~39 s).
# Same integrity model as Component 12: fits nothing, re-executes nothing, and proves it
# by comparing every input's sha256 before and after. Selects the production model from a
# rule frozen in advance, compares seven policies, and prices each one in citations.
uv run sentinel decide --report

# decide and validate, writing nothing
uv run sentinel decide --dry-run --report

# compare one policy against the baseline, on a nominated model
uv run sentinel decide --policies coverage_forced_double_share --model lightgbm_platt

# apply a reviewer's override file; the deterministic queue is written unchanged
# and every decision is logged beside it with its actor, reason and timestamp
uv run sentinel decide --overrides overrides.json --report
```

Every fit is single-threaded on the CPU with `torch.use_deterministic_algorithms(True)`,
so re-runs are bit-identical. A CUDA device on the build machine is deliberately unused:
GPU reductions are not bit-reproducible, and that is the standard every leakage test in
this repository is written against. See [ADR 0020](docs/decisions/0020-pytorch-and-matplotlib-as-runtime-dependencies.md).

### Turning the queue into a plan

```bash
# Component 14 -- lay the approved queue against the calendar Chicago actually worked (~28 s).
# Same integrity model as Components 12 and 13: fits nothing, scores nothing, re-ranks nothing,
# and proves it by comparing every input's sha256 before and after. Both capacity modes run by
# default so the scenario's divergence from the observed calendar is always visible.
uv run sentinel schedule --report

# plan and validate, writing nothing
uv run sentinel schedule --dry-run --report

# the measured calendar only, one policy, one capacity level
uv run sentinel schedule --capacity-mode observed_calendar --policies pure_risk --k-names k_1_week

# apply a supervisor's scheduling adjustments; the deterministic plan is written unchanged
# beside the log, and every offered change is recorded whether or not it changed anything
uv run sentinel schedule --adjustments adjustments.json --report

# record what the field says happened, and roll the plan forward from there
uv run sentinel schedule --execution execution.json --report
```

There is deliberately no `--capacity`, `--slots-per-day`, `--horizon-days`, `--extend-horizon`
or `--threshold` flag. Each would be a way to make a scheduling number better without scheduling
anything better, and the test suite asserts that each stays absent.

### Flagging cases for human review

```bash
# Component 16 -- flag deterministic review cases from the current queue and schedule.
# No probability threshold anywhere: both triggers are boolean facts Components 13 and 14
# already computed.
uv run sentinel review --report

# flag and validate, writing nothing
uv run sentinel review --dry-run --report

# apply a reviewer's resolutions; the queue is written unchanged beside the resolution log
uv run sentinel review --resolutions resolutions.json --report
```

There is deliberately no `--threshold`, `--probability-threshold` or `--confidence-threshold`
flag, and the test suite asserts each stays absent. See ADR 0051 and
[`docs/data_contracts/human_review.md`](docs/data_contracts/human_review.md).

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

### Running the Sentinel API

```bash
# a validated HTTP boundary over the artifacts above -- computes nothing, retrains nothing
uv run sentinel serve

# explicit bind, and autoreload for local development
uv run sentinel serve --host 0.0.0.0 --port 8080 --reload
```

Interactive OpenAPI docs are served at `/docs` once running. See
[`docs/data_contracts/sentinel_api.md`](docs/data_contracts/sentinel_api.md) for the endpoint
list, the decision-scope contract, and why writes are staged rather than applied (ADR 0048,
ADR 0049, ADR 0050). A minimal read-only frontend for testing this API end to end lives under
[`frontend/`](frontend/README.md) — see "The Sentinel Frontend" below.

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

The processed layer now holds **ten** kinds of thing, each with its own directory and
its own prohibition:

```text
data/processed/features/      model-ready tables. Trainable.               ADR 0011
data/processed/predictions/   model outputs. Never trainable.              ADR 0014
data/processed/evaluation/    measurements about models. Never trainable.  ADR 0013
data/processed/tuning/        hyperparameter search trials.                ADR 0018
data/processed/neural/        Component 8's experimental categoricals.     ADR 0022
data/processed/calibration/   fitted calibrators and their diagnostics.    ADR 0024
data/processed/explanations/  feature attributions and their analysis.     ADR 0028
data/processed/fairness/      group-conditional behaviour. Never a verdict. ADR 0032
data/processed/policy/        decisions: who to inspect, and why. ADR 0036
data/processed/scheduling/    plans: when, and what the calendar cost. ADR 0041
```

The ninth and tenth are the only ones that are not descriptions of the world. Every other
layer says what *is*; `policy/` says who to **inspect**, and `scheduling/` says **when**.
Descriptions and instructions change for different reasons, are wrong in different ways, and
are read by different people, so ADR 0036 and ADR 0041 keep them apart. Both carry the same
prohibition as the eighth and for a sharper reason: a recommendation is downstream of every
model here, and joined back onto training rows it would make the system's own past decisions
an input to its future ones — the exact feedback loop Component 12 measured and Component 13
was built to keep visible rather than to close.

The tenth is separated from the ninth on the same argument, one step further out. A policy
decision and a scheduled slot are different grains — the second is the first *plus a date plus
a slot plus a planning run* — and they change for different reasons: a queue changes when a
department changes its mind about coverage, a schedule changes when a Tuesday turns out to hold
sixteen inspections instead of twenty-eight. Filed together there would be no convention saying
which was which.

The eighth had to answer a warning ADR 0028 left behind — that the taxonomy is getting long
enough to be a burden. It earns its place the same way the sixth and seventh did: Component 5
emits a `roc_auc` per (model, fold), and Component 12 emits one per (model, fold, geography,
group, grain, stage). Filed in one directory there would be two authoritative answers for the
same cell with no convention saying which is which. Its tables are keyed by **group** rather
than by row, which is what stops them becoming per-establishment features — and a number
meaning *"the model was well calibrated in this neighbourhood last quarter"*, broadcast back
onto training rows, would be the most self-fulfilling input this project could construct.

The last is the easiest to misfile, and ADR 0028 argues the near-miss explicitly. An
attribution is not a *prediction* — it scores nothing and `evaluate --predictions` rejects
it. It is also not an *evaluation result*, which is the more tempting mistake: a large
`mean_abs_shap` says a model **relied** on a feature, not that it was right to, and filed
beside `roc_auc` that distinction would not survive contact with a reader.

The fifth is the newest and the most easily misread. Component 4's feature table has no
categorical column at all, so Component 8 carries chain, facility type, community area
and ZIP forward as-of from the raw snapshot into a layer of its own. It is **not a
feature table**, `feature_definition_version` is unchanged at `v1`, and no other
component may join it onto anything.

**Nothing in the last five may ever be joined onto a feature table**, and no number in
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

3,190 tests pass and 3 live tests are deselected. Unit tests mock HTTP at the
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
                               train-boosting, build-neural-categoricals,
                               tune-neural, train-neural, calibrate,
                               explain, audit-fairness, decide, schedule,
                               review, evaluate, serve
  manifest.py                  generic manifest helpers (hash, read, write)
  ingest/
    socrata.py                 paginating, retrying Socrata client
    food_inspections.py        orchestration: pages -> Parquet + manifest
    manifest.py                ingestion provenance model
  query/
    duckdb_queries.py          DuckDB over the raw Parquet
  api/                         the Sentinel API: a validated HTTP boundary over
                               Components 1-16's artifacts. Not a numbered
                               component -- ADR 0048, ADR 0049, ADR 0050, ADR 0051
    app.py                     FastAPI app factory; no state of its own
    deps.py                    Settings/pagination/scope dependency wiring
    errors.py                  exception -> HTTP status mapping, in one place
    schemas/                   request/response pydantic models per layer
    services/                  artifact lookup, scope checks, staged writes
    routers/                   HTTP only -- no business logic here
  explain/                     Component 11: explainability and attribution
    definitions.py             EXPLAIN_REGISTRY (the support matrix), the frozen
                               budget and tolerances, import-time guard
    refit.py                   re-executes the frozen fits; ADR 0026's gate, reused
    background.py              temporally safe reference rows (training window only)
    sample.py                  the label-blind, seeded explanation sample
    attribute.py               tree / linear / permutation SHAP; no shap import
    aggregate.py               importance, rank stability, drift, local cases
    validate.py                twenty checks, each re-derived from the data
    writer.py                  seven output schemas
    figures.py                 six figure kinds, drawn from the tables only
    build.py                   orchestration (the only module doing I/O)
  policy/                      Component 13: decision policy and governance
    definitions.py             the frozen grid, the selection rule, the boundary
    inputs.py                  loading nine closed components' artifacts
    select.py                  applying the production-model rule
    eligibility.py             the coverage contract: one column, one predicate
    allocation.py              risk block, coverage reserve, and the ranks
    evaluate.py                comparison, opportunity cost, dominated policies
    governance.py              warnings, and the human override layer
    validate.py                18 errors (the policy) vs 4 advisories (its price)
    writer.py                  eleven output schemas
    figures.py / build.py      orchestration (the only module doing I/O)
  scheduling/                  Component 14: operational scheduling
    definitions.py             the horizon rule, both capacity modes, the boundary
    models.py                  Horizon, QueueRow, Placement, SchedulePlan
    inputs.py                  the authoritative scheduling input contract
    horizon.py                 the observed calendar and its per-day capacity
    allocation.py              deterministic greedy placement in policy-rank order
    backlog.py                 approved and not reached, as a population
    adjustments.py             a human changing WHEN an approved row is worked
    execution.py               what the field reports happened; external, not computed
    replan.py                  appends a planning run, never mutates one
    evaluate.py                utilisation, inversions, wait, reserve survival
    validate.py                28 errors (the plan) vs 7 advisories (its price)
    writer.py                  thirteen output schemas
    figures.py / build.py      orchestration (the only module doing I/O)
  review/                      Component 16: deferral / human-review gate
    definitions.py             two triggers, no threshold, the fourth human layer
    models.py                  ReviewCase, ReviewResolution, ReviewManifest
    trigger.py                 the two deterministic, threshold-free triggers
    resolution.py              parsing and applying human review resolutions
    inputs.py                  loading Component 13/14's artifacts, read-only
    validate.py                8 errors (the queue) vs 2 advisories (its findings)
    writer.py                  three output schemas
    figures.py / build.py      orchestration (the only module doing I/O)
  fairness/                    Component 12: the group-behaviour audit
    definitions.py             the group registry incl. the REFUSED geographies,
                               the frozen support floors, import-time guard
    groups.py                  the group frame; the temporal leakage surface
    support.py                 decided BEFORE any metric, and it shapes everything
    metrics.py                 group-conditional; imports the canonical implementations
    priority.py                selection rate AND capture, deliberately never combined
    missingness.py             data availability by group; the Component 11 link
    attribution.py             groups C11's artifact, never regenerates it
    disparity.py / drift.py    four measures, never one score; trends only when earned
    validate.py                13 errors (the audit) vs 3 advisories (the world)
    writer.py                  ten output schemas
    figures.py / build.py      orchestration (the only module doing I/O)
  calibration/                 Component 9: probability calibration
    definitions.py             CANDIDATE_REGISTRY and the pre-registered protocol
    basescores.py              the regeneration seam and the bit-identity gate
    train.py / predict.py      Platt and isotonic; fit, select, apply
    metrics.py                 Brier decomposition, slope, bootstrap, ranking
    validate.py / writer.py / figures.py / build.py
  neural/                      Component 8: MLP with entity embeddings
    definitions.py             NEURAL_REGISTRY, architecture constants, two guards
    categoricals.py / encode.py   the experimental as-of categorical layer
    net.py / train.py          the network, and one fit per fold with early stopping
    embed.py                   the embeddings-into-XGBoost experiment
    preprocess.py / predict.py / tuning.py / models.py
    validate.py / writer.py / figures.py / build.py
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
scripts/profile_boosting.py    read-only tuning-surface profiling
scripts/profile_neural.py      read-only categorical-coverage profiling
scripts/profile_calibration.py read-only calibration-window profiling
scripts/profile_explanations.py read-only attribution-surface profiling; fixes
                               Component 11's frozen constants before it is built
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

Components 1-9, 11-14, and 16-21 are implemented; 10 and 15 are blocked, both on the same missing
inspector/travel-time data (ADR 0019, ADR 0043).

**Correction: 17-21 no longer refer to the original plan.** The original roadmap for 17-21
(LangGraph orchestration, LLM-generated inspector briefings, deterministic briefing verification,
an audit trail, and a frontend demo) was never built. After Component 16, the project's actual
direction diverged from that plan and instead extended the live operational pipeline forward —
Components 17-20 (operational candidate generation, scoring, capacity-constrained selection, and
geographic organization) and Component 21 (supervisor plan review, adjustment, and approval), all
described below. Only Component 15 keeps its original meaning.

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
| 11 | SHAP explainability | **Implemented** — 4 of 5 candidates supported; `xgboost_chain_embeddings` reported unsupported (ADR 0031) |
| 12 | Fairness and geographic equity audit | **Implemented** — 2 geographies audited, 5 refused with measurements (ADR 0032-0035) |
| 13 | Decision policy and deployment governance | **Implemented** — 7 policies compared, production model selected, no policy winner declared (ADR 0036-0040) |
| 14 | Constrained scheduling | **Implemented** — 13 tables, 2 capacity modes, 29.3% of the coverage reserve measured lost to the calendar (ADR 0041-0047) |
| 15 | OR-Tools routing | Not implemented — blocked on missing inspector/travel-time data (ADR 0019, ADR 0043) |
| 16 | Deferral / human-review gate | **Implemented** — two deterministic triggers, no probability threshold, a fourth human layer disjoint from override/adjustment/execution (ADR 0051) |
| 17 | Operational candidate generation | **Implemented** — `src/sentinel/candidates/`; a live `planning_date` fed through Component 4's own feature SQL, unmodified |
| 18 | Operational scoring | **Implemented** — `src/sentinel/operational_scoring/`; scores Component 17's candidates with the frozen production model, no retraining |
| 19 | Operational (capacity-constrained) selection | **Implemented** — `src/sentinel/operational_selection/`; reuses Component 13's own `allocate()`/`decide()` engine unmodified |
| 20 | Geographic organization | **Implemented** — `src/sentinel/geographic_organization/`; groups only the already-selected establishments, membership-preserving by construction |
| 21 | Supervisor plan review, adjustment, and approval | **Implemented** — `src/sentinel/plan_review/`; four decision verbs plus a 5-point approval readiness checklist producing an immutable `approved_operational_plan` |

Technologies for later components (PyTorch, OR-Tools, LangGraph, a frontend) are
deliberately absent from `pyproject.toml`. Each is introduced only when the component
that needs it is built — scikit-learn and numpy arrived with Component 6 (ADR 0015),
xgboost, lightgbm and optuna with Component 7 (ADR 0016), and torch and matplotlib with
Component 8 (ADR 0020). Component 11 added **no** runtime dependency: `xgboost` and
`lightgbm` already ship exact TreeSHAP, linear SHAP is a closed form, and `shap` itself is
dev-only, used as a test oracle exactly as scikit-learn was through Component 5 (ADR 0030).
Components 12, 13, 14 and 16 added none either: a group-conditional metric, a deterministic
allocation, a deterministic placement and a deterministic review-flagging pass are all arithmetic
over artifacts that already exist. Component 14 is where the solver was promised, and ADR 0043
records why it did not arrive: strict priority preservation has a closed form over a unique,
contiguous rank, and every constraint an optimiser would trade off — inspector, duration, travel
time, road network — is absent from the dataset. OR-Tools belongs to Component 15's routing,
which is blocked on that same missing data.

### The Sentinel API

A **Sentinel API** (`src/sentinel/api/`, run with `sentinel serve`) sits alongside these
components as cross-cutting infrastructure — a validated read/write HTTP boundary over the
artifacts Components 1-16 already produce deterministically. It is **not itself a numbered
component**: it introduces no model, no policy, no schedule, and no route, and it does not
occupy or redefine Component 15's place in the table above. Writes it accepts (overrides,
scheduling adjustments, execution events, review resolutions) are staged for a human operator to
apply through the existing `sentinel decide` / `sentinel schedule` / `sentinel review` commands,
never applied by the API itself. See ADR 0048, ADR 0049, ADR 0050, ADR 0051, and
`docs/data_contracts/sentinel_api.md` for the full contract.

### The Sentinel Frontend

A React + TypeScript + Vite frontend (`frontend/`) sits on top of the Sentinel API. It began as
product-testing infrastructure for the read-only backtest/evaluation pages (Overview,
Recommendations, Schedule, Backlog, Human Review, Establishment detail) and is, for those pages,
**not a numbered component**. **Correction: it is no longer purely read-only.** Component 21
(supervisor plan review, adjustment, and approval — see the roadmap table above) has its own
frontend pages here (`SupervisorPlanReviewPage`, `GeographicPlanPage`) with real write actions —
recording a decision, adjusting field-work order, approving a plan — each staged, never applied
directly (ADR 0049), exactly like the backtest-side write forms below. It computes nothing and
duplicates no model, policy or scheduling logic — every value rendered is read verbatim from an
API response, and every write is validated against the same contract the batch CLI enforces.

Built first for **product testing** (can a human browse every API resource and confirm it behaves
as documented?), then rebuilt for **product clarity** (can a non-technical inspection supervisor
open it and understand, in plain language, what Sentinel recommends and why, without first
learning what a "fold" or a "policy id" is?). Every technical field, code and identifier from the
first pass still exists -- it now lives one click away under each page's "Technical details"
rather than being the first thing a visitor has to parse. `useDefaultScope` fills in a real,
verified working scope automatically from the live manifests, so a first-time visit shows real
data immediately instead of an empty scope form. See
[`docs/analysis/frontend_product_clarity_20260828.md`](docs/analysis/frontend_product_clarity_20260828.md)
for the full before/after.

The backtest/evaluation pages remain **read-only**: no override, adjustment, execution-event or
review-resolution forms, only `GET` requests, on that side of the app. Its decision-scope UX
mirrors the API's own guarantee -- no request is fired
while a required scope field (`policy_id`, `fold_set`, `fold_id`, `k_name`, and
`schedule_config_id` for schedule/backlog views) is unset, and a `422 ambiguous_scope` response is
rendered from its actual body rather than hidden. The one CORS middleware addition in
`src/sentinel/api/app.py` (configurable via `Settings.api_cors_origins`) exists solely so a
browser can call the API from `http://localhost:5173` in local development; it changes no other
behavior. See [`frontend/README.md`](frontend/README.md) for how to run both processes together
and what was deliberately left out.

A final completion pass (2026-09-05) audited the whole system end to end for product coherence:
`GeographicPlanPage`/`SupervisorPlanReviewPage` were reachable only via the top nav bar, not
linked from `OverviewPage`/`TodayPage`/`EstablishmentDetailPage`, and the plain-language
`WorkflowDiagram` on Overview described only the old 5-step backtest flow. Fixed with navigation
links, a plain "for {date}" header on both pages (from data their own API responses already
return), and an 8-step diagram. Seven raw technical-ID leaks (a bare establishment id, raw API
error codes, a raw review id, a raw work-block-id fallback, two CLI command names in primary
copy) were found and fixed, each moved into a "Technical details" section or replaced with plain
language, never deleted. The one backend change: `operational_selection`'s manifest already
computed candidate/eligible/selected counts that no API route exposed — one whitelist entry now
makes them reachable, powering an honest "how many establishments were considered vs. selected"
note on the Plan Review page. See `STATUS.md`'s "Final completion pass" section for the full
detail, exact file list, and test counts.

---

## License

MIT
