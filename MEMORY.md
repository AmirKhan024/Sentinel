# MEMORY

Compact context for future Claude Code sessions. Not a project description —
see README.md for that and STATUS.md for current state.

**Read this first, then STATUS.md, then HANDOFF.md.**

---

## Working agreement

* **One component at a time.** 21 components are planned; Components 1-7
  exist. Never implement ahead. If something belongs to a later component,
  write it down as a TODO or an architectural note instead of building it.
* **No fake completion.** Never claim tests pass, ingestion works, or a schema
  is what it is, without having run the command. Anything unverified must be
  labelled `NOT VERIFIED` with the command that would verify it.
* **Introduce a technology only when the component needing it is being built.**
  This is why there is no pandas, no PostgreSQL, no OR-Tools, no LangGraph.
* **Explain before abstraction.** Simple explicit code beats clever layering.
* Update STATUS.md at every milestone. It must reflect reality.

---

## Hard constraints — do not change without a very good reason

1. **`$order=inspection_id` on every paged request.** Socrata does not
   guarantee row order; without a total order, `$offset` paging can duplicate or
   skip rows. `build_params()` raises if the order column is empty. This is the
   correctness backbone of ingestion.
2. **Raw Parquet is all `Utf8`.** No casting at ingestion, ever. The API returns
   every value as a string; the raw layer preserves that exactly. Typing belongs
   to a later component. (ADR 0002)
3. **`data/raw/` is append-only.** Timestamped filenames, never overwritten. No
   `latest.parquet` pointer — resolving "latest" is a read-time concern handled
   by `latest_parquet()`. (ADR 0005)
4. **Non-retryable 4xx must raise immediately.** Only 429, 5xx, timeouts and
   transport errors are retried. A 400 is our bug; retrying hides it.
5. **No `print()` in `src/sentinel`.** Logging only, except for the CLI's final
   result lines, which are deliberate stdout output.
6. **Live tests stay deselected by default** (`addopts = -m 'not live'`). CI
   must never depend on the Chicago API being reachable.
7. **Never commit raw data.** `.gitignore` excludes Parquet but whitelists
   `manifest_*.json`, so provenance is versioned and bulk data is not.

### Component 2 invariants - violating any of these silently corrupts history

8. **An establishment is a physical premises**, not a licence and not a business
   name. Successive tenants at one address are the *same* establishment with a
   changing name; a commissary holding 47 cart permits is *one* establishment.
   Component 3 must therefore not assume behavioural continuity across a tenant
   change - `n_names` and `n_licenses` are exposed so it can detect one.
9. **`license_` is not the entity key, and licence *inequality* is never
   evidence against a match.** 18.47% of establishments hold more than one
   licence (max 47). A rule keyed on licence agreement fractures them.
10. **Every non-licence merge requires address equivalence.** This is what makes
    247 Subways safe without a chain-name list, and what bounds transitive
    chaining. Do not add a name-only block or a name-only merge rule.
11. **Only identity columns reach the matcher.** `IDENTITY_COLUMNS` in
    `entity/nodes.py` is the leakage boundary; `results`, `violations`, `risk`
    and `inspection_date` are excluded, and a test asserts it.
12. **The assignments table carries no dates, counts or outcomes.** That absence
    is the anti-leakage guarantee. Do not "helpfully" add them.
13. **Resolution must stay deterministic.** Content-hashed node ids, canonically
    ordered pairs, components re-labelled by minimum member, ids from the
    earliest inspection. Verified against a seeded shuffle of all 314,245 rows.
14. **`establishment_id` is snapshot-scoped**, not a durable primary key. A
    later snapshot can merge or split clusters and retire ids (ADR 0007).

### Component 3 invariants — the target is a regulatory statement, not a convenience

15. **The target is: at an eligible routine canvass, was at least one Priority or
    Priority Foundation violation found?** Not `results == 'Fail'` — among
    canvasses, priority violations appear in 97.9% of `Pass w/ Conditions` rows,
    so a result-based label would mislabel 16,261 inspections.
16. **The label is read from the violation text, never from `results`.** Using
    the result would make the target partly circular. `results` is consulted only
    to decide whether a row is *labellable* (a `Fail` with no text is unknown).
17. **Eligibility starts 2018-07-01.** Chicago replaced Critical/Serious with
    Priority/Priority Foundation/Core on that date, cleanly. Before it the target
    is *undefined*, not sparse.
18. **`Out of Business`, `No Entry`, `Not Ready` and `Business Not Located` are
    ineligible, not negative.** No inspection occurred. Labelling them negative
    would teach that a closed establishment is a clean one.
19. **`inspection_date` is the as-of boundary.** Component 4 may use only
    information strictly before it. `target`, `results`, `evidence` and the
    `n_*_entries` columns describe the outcome and are forbidden as features —
    the set is `sentinel.target.writer.TARGET_EVENT_COLUMNS`.
20. **Pre-2018 inspections are usable as features even though they cannot be
    labelled.** The era boundary constrains labelling, not knowledge.
21. **One row per (establishment, date), target = OR over that day's canvasses.**
    "Inspect E on date D" is one scheduling decision.
22. **Never re-derive identity or labels.** Component 2 owns `establishment_id`;
    Component 3 owns `target`.

### Component 4 invariants - the as-of rule is the whole component

23. **A feature for the row at `inspection_date = d` may use only records dated
    STRICTLY BEFORE d.** Not on or before. An inspection dated on the reference
    date is never history, including the target's own.
24. **The boundary is `<` because dates carry no time component.** All 314,245
    rows have `T00:00:00.000`, so same-day records cannot be ordered; 43 same-day
    canvass re-inspections at reference dates provably follow their canvass.
25. **One range join carries the condition, in one place** (`historical.py`), and
    `validate.py` re-derives it independently on every row. Never
    `groupby(establishment_id)` then merge - that is the all-history bug.
26. **Four missing-value rules**: counts never NULL (0 is a true observation);
    recency NULL when the event never happened (0 would mean "today"); rates NULL
    when the denominator is 0; at-last flags NULL when there is no prior event.
    Every event count is emitted beside its inspection count so a 0 is legible.
27. **Priority features use code-era canvasses only** and are NULL for the 24.5%
    of rows with no prior code-era canvass. Absence of evidence, not evidence of
    absence.
28. **Priority is classified by Component 3's parser**, never a SQL substring
    match, so it means the same thing in the label and the feature.
29. **`FEATURE_COLUMNS` is the complete set of model inputs.** `target`,
    `target_status`, `inspection_date` and `code_era_phase` are not features.
30. **Never select features by downstream accuracy.** That is leakage by another
    route. Justify by domain reasoning and availability only.

### Component 5 invariants — the evaluation must not see the future either

31. **The primary split is chronological, never random.** `train_test_split`,
    `KFold` and `StratifiedKFold` are forbidden. `FoldSpec` refuses to construct
    a fold whose windows are not strictly ordered, so a leaky split is
    unrepresentable rather than merely discouraged.
32. **`TRAIN → CAL → TEST`, calibration strictly between.** Never
    `TRAIN + TEST → calibration`, never `TRAIN → TEST → calibration`. The window
    exists now, unused, so Component 9 has nowhere else to put a calibrator.
33. **Training is an expanding window anchored at 2018-07-01.** The anchor never
    moves; each fold's training end advances one quarter.
34. **Partial windows are excluded, never fabricated.** A metric over a
    two-thirds window is not comparable to one over a full window. The exclusion
    is named in the manifest.
35. **Capacity is held constant by construction.** Slots are the observed
    multiset of inspection dates; every schedule is a permutation over the same
    slots. The system changes the *order*, never the *number*.
36. **`bau_simulated_date == actual inspection_date`, always.** So "days earlier"
    means "earlier than what really happened". Re-derived on every fold of every
    run, not argued.
37. **Labels never move.** Reordering simulation, not causal simulator.
38. **Higher score = inspected sooner**, ties broken on `target_inspection_id`
    ascending. Probed on every run.
39. **Ranking metrics and probability metrics are disjoint.** A producer that
    declares `is_probability=False` is never handed a Brier score to make it fit
    an API — the rows simply do not exist.
40. **A missing score is a rejection, never an imputation**, and a prediction set
    must cover its fold's test window *exactly*.
41. **A declared `trained_through` later than the fold's calibration end is
    rejected at the door.**
42. **Nothing in `data/processed/evaluation/` may be joined onto a training
    table.** These are measurements about models; a feature derived from a test
    score is the worst leakage in the project. ADR 0013.
43. **Never report a days-earlier mean without its distribution.** SD, and the
    fractions improved / unchanged / worse. On this data the SD is 7.3× the mean
    and 42.9% of positives are found *later*.
44. **Read every base-rate-dependent metric beside its fold's prevalence.**
    PR-AUC, precision@k and first-half discovery all move with it; ROC-AUC and
    NDE do not.

### Component 6 invariants — the model must not see the future either

45. **The fit AND every preprocessing statistic come only from the fold's
    training window.** This is the half most easily lost: an imputation median or
    a scaler mean computed over the whole table before splitting is leakage no
    fold boundary catches, because the boundary is respected by the *fit* while
    the *transform* already knows the future.
    `validate._preprocessing_comes_from_train` re-derives every median.
46. **One model per fold, refitted from scratch.** Fold N's model is not fold
    N-1's with more data bolted on. 3 models x 18 folds = 54 fits.
47. **`trained_through = fold.train_end`, never `calibration_end`.** The contract
    permits the later date and the six heuristics declare it, but Component 6
    fits no calibrator and never reads that window. ADR 0014.
48. **Training rows are sorted by `(inspection_date, target_inspection_id)`
    before fitting.** Load-bearing, not decorative: without it the same 23,346
    rows in a different order move coefficients by up to **7.049e-09**.
    `random_state` has NO effect on `lbfgs` — the sort is what makes a re-run
    reproducible.
49. **Four family indicators, never ten per-column ones.** The null masks within
    a Component 4 null-rule family are byte-identical, so
    `SimpleImputer(add_indicator=True)` emits collinear duplicates and picks its
    indicator set by observation rather than declaration.
50. **Nullable booleans fill with constant 0.0, never a median.**
    `priority_at_last_canvass` drifts 0.6310 -> 0.5056 across the training
    windows and sits 0.0056 from flipping the median fill.
51. **`FEATURE_COLUMNS` per model is explicit and closed.** Never "all columns
    except the target" — a future Component 4 metadata column would silently
    become a model input.
52. **No class weighting, no SMOTE, no resampling.** Prevalence is 52.52%; there
    is nothing to correct, and resampling would corrupt the probability scale
    Component 9 depends on.
53. **Component 6 computes no metrics.** Component 5 evaluates, via
    `sentinel evaluate --predictions`. Two evaluators would mean two answers and
    no way to tell which is authoritative.
54. **Nothing in `data/processed/predictions/` may be joined onto a feature
    table.** Model outputs, never trainable. ADR 0014.
55. **Never describe Component 6's probabilities as calibrated.** ECE 0.0635,
    MCE 0.1664, measured. Component 9 owns calibration.
56. **`cdph_2015_approximation` is never presented as the CDPH 2015 model.** Only
    3 of its 10 input families are reachable. The `approximation_note` travels in
    the manifest.
57. **Never quote days-earlier without the fraction found later.** 43.24% under
    the best model — marginally worse than the best heuristic. Re-ordering under
    fixed capacity is zero-sum.
58. **Coefficients are not importances here.** Condition number 71.8; a
    0.9888-correlated pair splits into +1.99 / -1.47. Component 11 owns
    attribution.

---

### Component 7 invariants — the *human* must not see the future either

59. **Hyperparameters may only be selected from data strictly earlier than the
    fold set's first test window.** This is a leak of a different kind from every
    other one in this project: fit, read a test metric, adjust, refit, keep the
    better one — and the artifact passes every mechanical check. The predictions
    cover the test window, the horizon is honest, no row is misdated. The model
    is just better than it should be. Component 5 protects evaluation time; it
    cannot protect against a person. ADR 0017.
60. **Two studies, one per fold set, never one shared.** The `covid_shift` test
    window (2020-06-01..2021-12-31) sits *inside* the quarterly tuning region
    (2018-07-01..2022-03-31). A shared study would have picked parameters using
    the shift fold's own test labels — biasing the single number most likely to
    change a release decision. `tuned_params` raises rather than borrowing across
    fold sets.
61. **The tuning region is derived from the fold definitions, never a date
    literal.** `tuning.tuning_region` reads the first fold's
    `train_start..calibration_end`. A hardcoded region would stop being correct
    the moment the anchor or cadence moved, invisibly.
62. **Early stopping happens only inside the tuning objective.** The winning
    trial's mean `best_iteration` is frozen; `train.fit_fold` runs exactly that
    many rounds with **no `eval_set`**. That is what makes
    `trained_through = fold.train_end` literally true rather than nearly true —
    early stopping at fold-fit time would need a window later than the training
    data, and the only ones available are the fold's own calibration and test spans.
63. **No imputation and no scaling is fitted.** NULLs reach the estimator as NaN
    and are routed by a learned default direction. A NULL here is a *fact* —
    "there was no prior canvass" — and Component 6 had to replace it with the
    median of a population the establishment is not in. 3,404,772 NaN cells
    reached the estimators across 54 fits.
64. **The four family indicators are kept anyway**, even though a NaN-native
    learner does not need them. Dropping them would mean the boosted and baseline
    matrices differ, making every C6/C7 comparison ambiguous between "the
    estimator is better" and "the matrix is different".
65. **Row order is far more load-bearing than in Component 6.** Shuffling the same
    53,844 training rows moves a *prediction* by **1.12e-01** (xgboost) and
    **1.23e-01** (lightgbm), against 7.049e-09 for C6's coefficients — a booster
    draws row and column subsamples in row order. Re-sorting restores the fit
    exactly. `fit_fold` re-sorts unconditionally.
66. **Every fit is single-threaded** (`n_jobs=1`, plus LightGBM's `deterministic`
    and `force_row_wise`). A multi-threaded histogram reduction is only
    approximately reproducible and this project's standard for "unchanged" is
    bit-identical.
67. **The frozen parameters are source literals, not a file read at training
    time.** A parameter set loaded from disk could change without a diff, and
    freezing is only meaningful if it cannot. `tune-boosting` prints the block; a
    human pastes and commits.
68. **Component 6's artifact is never overwritten.** Separate registry, separate
    slug (`boosted_predictions`), separate manifest. "Did C7 beat C6?" is only
    answerable if C6's answer survives — verified: re-running `train-baselines`
    reproduces sha256 `a2bb9411…00ff5b44` byte for byte.
69. **Class weighting is an ablation, never a default.** `xgboost_class_weighted`
    posts the *best* quarterly NDE (0.2390) and is still not adopted: it costs ECE
    0.0621 → 0.0836. Prevalence is 52.52%, so it distorts a balanced problem to buy
    a margin smaller than the seasonality band. Adopting it would be tuning until
    something wins.
70. **Importances are a diagnostic, never an attribution.** Native split gain, with
    condition number 71.8 and a 0.9888-correlated pair. Component 11 owns
    attribution; SHAP is deliberately not implemented here.
71. **Never describe an UNcalibrated probability as calibrated.** Components 6-8's
    artifacts remain raw: quarterly ECE 0.0621/0.0644, covid_shift 0.1253/0.1518.
    Component 9's `calibrated_predictions_*.parquet` is the calibrated one (quarterly ECE
    0.0474-0.0524, slope 1.00-1.03). The two must never be conflated, which is why a
    calibrated row's `model_name` is `"<base>_<method>"` and never a bare base name.
72. **Nothing in `data/processed/tuning/` may be joined onto a feature table, and
    no number in it is a result.** Every one is measured on a validation window that
    is training data for the folds the parameters are used on. ADR 0018.
73. **Inspector-effect modelling is BLOCKED, not skipped.** The dataset publishes 22
    columns and none identifies an inspector. A random intercept over an unobserved
    grouping has no likelihood; a marginalisation over an unestimated effect is
    arithmetic on a made-up number. Proxies (verbosity, ward, day-of-week) are
    refused. ADR 0019.

---

### Component 8 invariants — a *representation* must not see the future either

70. **The four embedded categoricals are NOT Component 4 features.** Chain, facility type,
    community area and ZIP do not exist in the feature table. Component 8 carries them
    forward as-of into `data/processed/neural/`, an explicitly experimental fifth processed
    layer, and `feature_definition_version` stays `v1`. Nothing else may join that table.
    See ADR 0022.
71. **A categorical is never read off the row being predicted.** The as-of join is
    `join_asof(strategy="backward", allow_exact_matches=False)` — the flag is the whole
    temporal argument. `source_inspection_date` is emitted per row and validated strictly
    earlier. Measured minimum lag: **1 day**. A zero would mean a row supplied its own
    attributes.
72. **A vocabulary is a fitted statistic and leaks like one.** Refitted per fold on training
    rows only. Index 0 is `__UNKNOWN__` and its vector is **learned**, not masked — 401 rows
    genuinely have no prior inspection. Order is **sorted**, never insertion-ordered, because
    insertion order is row order.
73. **Chain membership is derived inside the fold.** A name is a chain only if two distinct
    establishments carry it *among that fold's training rows*. Computing it globally would
    let a location opened in 2025 make a 2022 row part of a chain. An unshared name is
    `__INDEPENDENT__`, a real category — not a null, and not `__UNKNOWN__`.
74. **`establishment_id` is REFUSED, not merely absent.** A closed `EntityFamily` allowlist
    plus `FORBIDDEN_COLUMNS`, enforced at import and restated at runtime. A per-establishment
    parameter is the largest leakage surface in the project and Component 4 excludes identity
    by design. `chain` is the substitute. See ADR 0021.
75. **Early stopping validates on the last ~15% of the TRAINING window**, cut on a whole day
    so no date straddles the split. Never the calibration window, never the test window. That
    is what keeps `trained_through = fold.train_end` literally true for the first component
    that early-stops. `inner_validation_start` is in the training log so it is checkable.
    Cost: a final fit uses ~85% of its fold's training rows.
76. **Every fitted statistic comes from the inner training rows only** — medians, scaler
    mean/scale, vocabularies, chain membership. Stricter than the fold requires, because the
    early-stopping signal is only honest if the validation rows influenced none of them.
77. **The network imputes and scales; Component 7 did neither.** There is no NaN-native path
    for a dense layer. The four null-rule family indicators are how missingness survives.
    Measured justification for scaling: the widest feature SD is **18,409x** the narrowest.
78. **CPU, one thread, `use_deterministic_algorithms(True)`, seeded generator.** A CUDA
    device is present and deliberately unused — GPU reductions are not bit-reproducible and
    this project's standard is bit-identity. See ADR 0020.
79. **Across seeds nothing is claimed; it is measured.** Five seeds x 18 folds, written to
    `neural_seed_variation`. The spread (0.0058 ROC-AUC) is the same size as the neural
    model's entire advantage over XGBoost (0.0053). **This governs every reading of the
    Component 8 result.**
80. **The embedding-fed booster must consume vectors from its OWN fold.** `embed.fit_fold`
    refuses a donor fitted elsewhere. It borrows Component 7's frozen parameters unchanged —
    re-tuning would confound "the embeddings helped" with "a second search helped".
81. **Every Component 8 model scores an identical id set** (41,536 rows), enforced by
    `every_model_scored_the_same_rows`. No comparison here is across populations.

## Key neural-model facts (measured 2026-08-18 on the full feature table)

* **234 fits** over 18 folds in 1,998.7 s, 4,306 epochs, CPU / one thread. Every fit stopped
  on patience; none hit the 200-epoch cap.
* Quarterly means (17 folds): `neural_numeric_only` NDE **0.2482**, ROC-AUC 0.6241, PR-AUC
  0.5343, Brier **0.2355**, ECE **0.0563**, precision@k_1_day 0.6273, +6.10 days earlier.
* **It beats XGBoost (0.2376) and wins 12 of 17 folds** — the first result here where a mean
  improvement and a per-fold improvement agree.
* **But the margin equals the noise**: five-seed ROC-AUC spread 0.0058 against a 0.0053 win.
  Its 0.2482 clears XGBoost's seasonality p95 (0.2444), but XGBoost's 0.2376 sits inside the
  neural model's [0.2311, 0.2527]. **Suggestive, not decisive.**
* **The embeddings LOST.** `neural_embeddings` NDE 0.2215, i.e. 0.0267 *below* the
  no-categoricals control. Every ablation beats the full model. The one-hot control is within
  0.0009 of it, so representation is not the problem — capacity is.
* Mean best epoch orders by parameter count: `neural_numeric_only` 10.4 (41,729 params),
  `neural_embeddings` 4.0 (67,985), `neural_onehot` 2.3 (337,665). More capacity, faster
  overfit.
* **The embeddings HELPED XGBoost**: `xgboost_chain_embeddings` NDE 0.2444, PR-AUC **0.5357**
  (best of any model).
* **The chain embedding is indistinguishable from random**: pairwise cosine mean 0.0018 /
  SD 0.2508 against a random Gaussian table's 0.0000 / 0.2504. t-SNE is a featureless blob.
  SUBWAY's nearest neighbours are SWEET MANDY BS, KFC, KANELA BREAKFAST CLUB.
* **`neural_no_community_area` (0.2258) beats the full model (0.2215)** — community area
  bought nothing. ADR 0023's non-retention rule therefore cost nothing here and still stands.
* covid_shift (1 fold) inverts again: `neural_onehot`, the *worst* quarterly neural model, is
  the best of any model (ROC 0.6456, PR 0.6528). Four components, four inversions.
* Metric ordering disagrees: `neural_numeric_only` wins NDE but loses precision@k_1_day to
  `lightgbm` (0.6598) and the GLM (0.6576).
* Sweep: 40 trials / 550.9 s. lr 3e-3 (quarterly), 1e-2 (covid_shift). The spec's 1e-3
  baseline is 0.0020 behind on quarterly — the result is only mildly rate-sensitive.
* Experimental categoricals: 57,727 rows, coverage 0.9881-0.9931, 401 rows with no prior
  inspection (exactly Component 4's null `days_since_any_inspection` count), OOV rates on test
  windows 0.00-2.04%, 950 chains covering 22.70% of rows.

## Key boosted-model facts (measured 2026-08-17 on the full feature table)

* **The result is a small improvement that does not clearly survive scrutiny.**
  Quarterly mean over 17 folds: `xgboost` NDE **0.2376** vs `logistic_regression`
  **0.2326** — +2.1% relative. ROC-AUC 0.6188 vs 0.6163. PR-AUC 0.5343 vs 0.5321.
  Days earlier +5.83 vs +5.70.
* **Per fold, logistic wins 7 of 17**; xgboost 5, lightgbm 5. The tree models win by
  more when they win (+0.047 at 2024Q2) than they lose by (−0.017 at 2025Q4) — which
  is exactly what produces the positive mean.
* **The gap is inside the seasonality band.** Redraw intervals: xgboost
  [0.2224, 0.2444], logistic [0.2160, 0.2374]. Each observed value lies inside the
  other's range. C6's gain over the heuristics did *not* have this problem — 0.1845
  sits well below 0.2160.
* **The two libraries agree to 0.0021 NDE** after 100 tuned trials each. Two very
  different nonlinear learners landing that close to each other, and ~0.005 above a
  penalised GLM, is the component's real finding: **the ceiling is the 26-feature
  representation, not the estimator.**
* **`covid_shift` has no single winner.** lightgbm takes NDE (0.2585 vs 0.2512) and
  ROC-AUC (0.6292); `logistic_regression` takes PR-AUC (0.6328, highest of any model)
  and precision@k_1day (0.9545, by a wide margin). One fold, k=22 slots,
  days-earlier SD 208.
* **42.89% of violations are still found later**, against C6's 43.24%. Effectively
  unchanged. Re-ordering under fixed capacity is zero-sum.
* **Boosted probabilities are NOT worse calibrated on the quarterly folds** — ECE
  0.0621 (xgboost) vs 0.0635 (logistic) — contradicting the expectation carried in
  HANDOFF.md, and matching a probe pre-registered on the calibration window *before*
  training. Under shift the expectation holds: 0.1518 vs 0.1124.
* **Every study chose shallow trees**: `max_depth` 3–4 in three of four studies, from
  a searchable range of 3–10. The data does not support deep interactions.
* **The capacity cap binds.** LightGBM's quarterly winner is `max_depth=4,
  num_leaves=16` — exactly 2^depth, the ceiling imposed to keep the two searches
  comparable.
* **Search cost:** 400 trials, 4 studies, 0 failed, 563.8 s. Training 54 fits, 21.4 s.
  Evaluation 68.4 s.
* **NaN density:** 10 of 30 matrix columns carry NaN; 25.74% of rows have no code-era
  canvass history; 4.82% of all matrix cells are NaN.


---

### Component 9 invariants — a CALIBRATOR is a fitted model too

82. **A calibrator is a fitted statistic, so it has a horizon.** The calibrated artifact
    declares `trained_through = fold.calibration_end`, which is LATER than the base
    estimator's `train_end`. Saying it was trained only through `train_end` would be false:
    the calibrator really did read the calibration window. Three columns carry the
    distinction - `base_model_trained_through`, `calibrator_fitted_through`,
    `calibrated_prediction_available_from`. ADR 0024.
83. **Fold N's calibration window IS fold N-1's test window.** Any protocol that pools folds
    to pick one calibration method reads test data through a window that is nominally
    calibration. Selection is an EXPANDING PREFIX over folds 1..k of one fold set, never a
    pool. ADR 0025, and a test asserts the rejected design is detectably leaky.
84. **The selection metric is log-loss, not ECE.** ECE at 15 equal-mass bins over a ~500-row
    inner-select window is 27-50 rows/bin, it is not a proper scoring rule, and its bin count
    is a tunable free parameter - a rule that can be tuned is not a rule. ADR 0025.
85. **`TIE_THRESHOLD = 0.005` nats, prefer Platt, frozen with a git date before the first
    production run.** One median paired-bootstrap-gap SD. The plan's 0.002 sat below the
    smallest observed SD and was corrected by measurement. ADR 0025.
86. **No fitted model was ever persisted, and no component had scored a calibration window.**
    Component 9 re-executes Components 6-8 UNCHANGED behind a bit-identity gate: 207,680
    regenerated test rows compared with `==`, zero mismatches, build raises on failure. A
    tolerance is not the remedy. ADR 0026.
87. **The gate is BLAS-thread-sensitive, and that is correct.** `OMP_NUM_THREADS=1` moves
    `logistic_regression` by up to 5e-10 and fails the gate. Run `sentinel calibrate` with no
    thread override, matching the committed manifests' "unset (library default)".
88. **The calibrator is fed `logit(p)` recovered from the committed probability**, never the
    native margin - so the calibrated artifact is a pure function of an artifact that already
    exists on disk. `xgboost` and the network compute in FLOAT32, so the recovery differs from
    their margin by up to 2.6e-5; that is their precision, not a defect. ADR 0027.
89. **Platt preserves the ranking exactly; isotonic ties.** Measured: Platt 0 inversions, rho
    exactly 1.0, PR-AUC/ROC-AUC/NDE/precision@k delta exactly 0.00e+00. Isotonic also has 0
    inversions but tied ~40,000 pairs, moving top-k membership and precision@k by up to 0.21.
    **A tie is not an inversion** and must not be reported as one.
90. **Calibration cannot create resolution.** Measured unchanged to five decimal places while
    reliability fell 16-46%. If resolution moves under a monotone calibrator, something is
    wrong. Uncertainty is a property of the labels and is identical everywhere (0.24362).
91. **Prior shift is not miscalibration.** On `covid_shift` the base rate moves 17 points
    between the calibration and test windows, so Platt reaches slope 0.75-0.90 rather than
    1.00 and ECE stays roughly double quarterly. No monotone map fitted on the earlier window
    can correct that.
92. **Never seed from `hash()` of a string.** Python salts `str` hashing per process, so a
    seed derived from a model name gives different resamples every run. C9's bootstrap was not
    byte-reproducible until the key became the candidate's registry position. Found by
    comparing two runs' sha256 per table, which is now the standard determinism check.
93. **The retraining trigger (ECE > 0.075 or slope outside [0.80, 1.25] for two consecutive
    quarters) is a DESIGN PROPOSAL**, written after seeing the drift series and validated
    against nothing. Unlike the tie rule, it selects nothing, which is why it was allowed to
    be written afterwards.
---

## Key baseline-model facts (measured 2026-08-17 on the full feature table)

Detail in `docs/analysis/baseline_models_findings.md`.

* **3 models x 18 folds = 54 fits**, 124,608 prediction rows, training **29.7 s**
  (22.3 s fitting), evaluation **237.8 s**. 15/15 Component 6 checks and 14/14
  Component 5 checks pass.
* **Exactly 10 of the 26 features are nullable, in 4 null-rule families.** The
  masks *within* a family are byte-identical, and each is exactly the zero-set of
  a paired never-null count — so an indicator adds no new information, only
  *linearly available* information.
* **Best model** `logistic_regression`: NDE **0.2326**, mean days-earlier
  **+5.70**, precision@k_1_day **0.6576**, lift 1.53, ROC-AUC 0.6163, PR-AUC
  0.5321. Beats the best heuristic on **17 of 17** quarterly folds.
* **The bands do not overlap**: 0.2326 [0.2160, 0.2374] vs the heuristic's 0.1845
  [0.1720, 0.1922] over 1,000 label re-draws. The gap survives the seasonal drift.
* **But it is a modest model.** PR-AUC floor = prevalence = **0.4307**, so +0.10.
  ROC-AUC 0.6163 is weak by any general standard. **43.24% of violations are
  still found later** — worse than the heuristic's 42.88%.
* **The ablation inverts under distribution shift**: on `covid_shift`
  `logistic_regression_no_scheduling` wins (ROC-AUC 0.6286 / NDE 0.2571) while
  losing on the quarterly folds. **Model selection on the rolling folds would
  have picked the wrong model.** Confirmed with fitted models what Component 5
  predicted with heuristics.
* **`days_since_any_inspection` is NOT worthless** — stable +0.24 coefficient, no
  sign flip, and keeping it helps on the quarterly folds (+0.0088 NDE). The
  design-time speculation was wrong; the data settled it.
* **Uncalibrated**: Brier 0.2382, log-loss 0.6723, ECE 0.0635, MCE 0.1664.
* **All 54 fits converged** in 43-83 of 1000 iterations. Scores span
  2.44e-06 … 0.999603, zero saturated.
* **Coefficients are stable in sign except for 7 of 30 terms, all with mean
  magnitude below 0.118** — the nested window features and the 401-row indicator.
  Largest: `prior_canvass_count` +1.99 against `prior_canvass_inspected_count`
  -1.47, a 0.9888-correlated pair estimating one effect.

---

## Key temporal-evaluation facts (measured 2026-08-16 on the full snapshot)

Detail in `docs/analysis/temporal_evaluation_findings.md`.

* **17 quarterly folds + 1 `covid_shift` fold**, test range 2022Q2 … 2026Q2.
  Count derived from the data; 2026Q3 excluded as partial.
* **Capacity: median 29 inspections/day** overall (p25 21, p75 37, max 68),
  22–45 across test windows. `k` is derived from this, never chosen.
* **Base rate drifts** 0.667 → 0.535 across training windows, 0.513 → 0.379
  across test windows.
* **Best untrained baseline** (`prior_canvass_priority_rate`): NDE **0.1845 ±
  0.0404**, mean days-earlier **+4.47**, SD **32.60**, **42.9% found later**,
  first-half 0.576, ROC-AUC 0.5915.
* **Business-as-usual: NDE 0.0066 ± 0.0422, ROC-AUC 0.5040** — statistically
  indistinguishable from random *within a quarter*. The bar is low; beating it is
  not evidence of a good model.
* **Analytic bounds land exactly**: optimal +1.0000, worst −1.0000, both with
  zero variance; 340 random fold-seeds average −0.0016.
* **`constant` scores ROC-AUC exactly 0.5000** and NDE 0.0065 ≈ business-as-usual
  — the tie-break on `target_inspection_id` reconstructs approximate date order.
* **Time invariance does NOT hold**: de-trended seasonal swing **11.77 pp**,
  peak August +6.36, trough December −5.41. Consistent with the temperature
  hypothesis but **not attributable** to it — no weather data is ingested.
* **The sensitivity band survives**: 1,000 label re-draws leave the headline NDE
  at [0.172, 0.192] against an observed 0.1845; 1.7–2.1% of labels flip.
* **Distribution shift inverts the baseline ordering**: on `covid_shift`,
  `days_since_last_canvass` is strongest (0.170) and on the quarterly folds it is
  weakest (0.077).
* **Runtime 164.2 s**, 14/14 error checks pass, 962 KB of artifacts.

---

## Key as-of feature facts (measured 2026-08-16 on the full snapshot)

Detail in `docs/analysis/as_of_feature_engineering_findings.md`.

* **57,727 eligible target rows -> 57,727 feature rows, 0 unmatched, 26
  features, 33 columns** in 15.6 s. Output in `data/processed/features/`.
* **`inspection_date` has exactly ONE distinct time component** across all
  314,245 rows (`T00:00:00.000`). This single measurement settled the boundary.
* On reference dates there are also 1,075 `License`, **43 `Canvass
  Re-Inspection`** and 42 `Complaint` records. 2,103 target rows have at least
  one same-day companion.
* History is abundant: only **401 rows (0.69%)** are cold-start, but **5,615
  (9.7%)** have no prior canvass and **14,162 (24.5%)** none in the code era.
  **80%** have pre-2018 history, which is usable for counts but not for priority.
* **Canvass cycle: 358-day median** (p25 251, p75 482). So a 365-day window is
  **empty for 62% of rows**; 730d for 22%, 1095d for 14.3%.
* **Any-type interval p25 is 9 DAYS** - the re-inspection pattern. This is why
  `days_since_any_inspection` is labelled policy-encoding context rather than the
  primary recency.
* Prior canvasses include **16,517 `Out of Business`** and **13,077 `No Entry`**;
  they are excluded from outcome denominators. `prior_canvass_fail_rate` has 346
  more nulls than `days_since_last_canvass` for exactly this reason.
* **`days_since_last_canvass` min = 1, zero zeros.** A zero-day recency is
  unconstructable; cheapest proof the boundary works.
* **15.9%** of target rows sit in a premises that changed name; **1,962** follow
  a change immediately.
* Data quality: zero null dates, zero unparseable dates, zero duplicate
  `inspection_id`.
* Performance: 2m14s -> 15.6s by materializing the aggregation as a TABLE (not a
  view) and parsing only code-era violation text. Range join is 793,200 pairs,
  ~0.1 s; the cost is the Python parser, not the temporal logic.

---

## Key target facts (measured 2026-08-16 on the full snapshot)

Detail in `docs/analysis/target_construction_findings.md`.

* **314,245 inspections -> 313,624 target rows -> 57,727 eligible, 30,316
  positive (52.52%)** in 25 s, across 15,144 establishments.
* **`results` has SEVEN values, not the four that were documented**: Pass
  162,607 · Fail 60,513 · Pass w/ Conditions 46,661 · Out of Business 25,767 ·
  No Entry 14,045 · Not Ready 4,557 · Business Not Located 95. No nulls, blanks
  or case variants.
* **The 2018-07-01 cutover is clean**: June 2018 has 0 rows using Priority
  terminology and 415 using Critical/Serious; July has 761 and 0.
* **Priority presence by result, among canvasses**: Fail 99.4%, Pass w/
  Conditions 97.9%, Pass 0.45%.
* **The violation number does NOT encode severity.** Item 10 is 42.2% Priority
  Foundation / 11.3% Priority / 46.5% unlabelled; the same item covers a hand
  sink with no hot water and a missing hand-washing sign.
* **Requiring a `7-38-xxx` citation code would create ~21,281 false negatives** —
  "PRIORITY FOUNDATION VIOLATION. NO CITATION ISSUED." is genuine.
* **72% of violation entries carry no severity label**, so unlabelled means
  UNCLASSIFIED, never "Core".
* **Narrative exclusions** (grace period, will-be-issued, no-priority) change 74
  entries and **10 inspection labels** out of 137,598 and 58,427.
* **24.9% of `Out of Business` records are followed by another inspection** at
  the same premises, median 273 days later.
* **Base rate drift**: 87.6% (2018 H2) -> 77.4 -> 59.4 -> 50.3 -> 46.5 -> 46.1 ->
  42.6 -> 39.2 -> 39.1% (2026). `code_era_phase` flags the adoption period.
* Exclusions: `ineligible_era` 172,879 · `ineligible_type` 70,848 ·
  `ineligible_result` 12,091 · `unknown_violations` 79.
* 111 distinct `inspection_type` values; only `Canvass` (70,518 in the code era)
  is eligible. `Canvass Re-Inspection` (16,998) is excluded because it exists
  only because something failed.

---

## Key entity-resolution facts (measured 2026-08-16 on the full snapshot)

Snapshot `7d3c4069...ad38`, 314,245 rows. Detail in
`docs/analysis/entity_resolution_findings.md`.

* **314,245 rows -> 51,099 nodes -> 35,859 establishments** in 43 s.
* **18.47%** of (name, address) pairs hold more than one licence, up to 47 (a
  mobile-food commissary with one permit per cart). A licence is often *finer*
  grained than an establishment.
* The **`'0'` licence sentinel** covers 323 distinct names across 364 addresses.
  850 rows (0.27%) have no usable licence at all.
* **Address normalization is where the leverage is**: case and whitespace alone
  collapse 33,261 address strings to 20,313 (-39%). Name normalization resolves
  only 0.21% of licences, which is why there is no fuzzy name matching.
* **There are no long-form street suffixes** and one long directional in 20,312
  addresses. The real defect is *missing and contradictory* suffixes
  (`1901 W MADISON` / `AVE` / `ST` are all the United Center), so the suffix is
  excluded from `addr_key` rather than canonicalized.
* **Coordinate spread within an address is exactly 0 m.** The city geocodes
  before the string variation appears, so a shared coordinate bridges variants
  no string rule can, including the 2021 Lake Shore Drive rename. 95.4% of
  coordinates map to one address key; the worst covers 4.
* **75.5% of same-place licence pairs overlap in time** rather than succeeding
  one another, which is why there is no temporal logic in matching.
* Chains are pervasive: 247 Subways, 184 Dunkin Donuts; one O'Hare address
  (`11601 W TOUHY AVE`) carries 219 distinct business names and 417 licences.
* **Single-inspection establishments fell 51%** versus naive licence grouping
  (12,356 -> 6,084), the clearest evidence real history was recovered.

---

## Key API facts (verified 2026-08-15, live)

* Endpoint `https://data.cityofchicago.org/resource/4ijn-s7e5.json`, no auth.
* **314,245 rows** total at that date.
* **Every value is a JSON string**, including `number` and `calendar_date`
  columns. `location` is the one nested object.
* `$limit=60000` works — there is no 50k cap on this endpoint.
* Pagination ends on a short page or an empty page. There is no cursor and no
  "has more" flag.
* Errors are JSON with an `errorCode` (e.g. `query.soql.no-such-column`) + 4xx.
* **`$order` suppresses the 5 `:@computed_region_*` columns** unless they are
  explicitly named in `$select`. This is why ingestion makes an extra unordered
  `?$limit=1` request to discover the field list, then selects it. Do not
  replace that with a hardcoded column list — a new upstream column would then
  be silently dropped. See `docs/api/socrata_findings.md` §6.
* Response headers `X-SODA2-Fields` / `X-SODA2-Types` carry the declared schema
  on every request. Positionally aligned arrays.
* `X-SODA2-Truth-Last-Modified` and `ETag` exist and are unused — leads for a
  future incremental-ingestion component.

---

## Technology decisions and why

* **torch (CPU) and matplotlib arrive with Component 8.** ADR 0016 required the dependency
  argument to be restated rather than assumed; ADR 0020 restates it. Reverse-mode autodiff
  over embedding lookups is a solver, not a formula, and a wrong gradient does not raise — it
  trains to a slightly worse optimum. The CPU wheel is deliberate. matplotlib is runtime
  rather than dev because the figures are deliverables of `train-neural`. t-SNE comes from
  scikit-learn; umap-learn was refused.

| Choice | Why | ADR |
|---|---|---|
| Python 3.12 + uv | downstream ML ecosystem is Python; uv is fast and gives a committed lockfile | 0001 |
| httpx (not requests/sodapy) | modern client, respx mocks it at transport level; SDKs hide the pagination we most need to see | 0004 |
| Hand-written pagination | highest-risk logic in the component; must be visible and unit-testable | 0004 |
| Parquet, all `Utf8` | columnar + self-describing + compresses well; strings keep the raw layer faithful | 0002 |
| Polars (not pandas) | fast, explicit schema control, `pl.Utf8` enforcement is trivial | 0002 |
| DuckDB in-memory | reads Parquet in place, no load step, no server, real analytical SQL | 0003 |
| argparse (not Typer) | ~4 flags across 2 subcommands; Typer would add 3 deps for help-text polish | — |
| respx | mocks httpx at the transport layer, so real request/status/retry code runs | — |
| JSON manifest sidecar | human-readable, diffable, greppable, zero infrastructure | — |
| scikit-learn as a *runtime* dep (Component 6) | a metric is arithmetic over two arrays and checkable against a reference; an L2 logistic regression is an iterative optimisation whose subtle defects look like a slightly worse model, not a wrong number. C7 needs the same API anyway | 0015 |
| numpy declared explicitly | imported directly by `modeling/`; an undeclared direct import breaks on a minimal install | 0015 |
| `sklearn.*` mypy override + a typed facade | scikit-learn ships no `py.typed`, so the estimator is treated as opaque behind `modeling.models.FittedModel` rather than letting `Any` spread | 0015 |
| `data/processed/predictions/` as a third kind | predictions fail ADR 0011's model-ready test (no features) *and* ADR 0013's evaluation test (produced before scoring, and an evaluator input) | 0014 |
| xgboost **and** lightgbm, both (Component 7) | the comparison between them *is* the component's question, not an implementation detail: depth-wise vs leaf-wise growth means a single result would be a fact about one library's bias. Same side of ADR 0015's line as sklearn, further along it | 0016 |
| optuna for the search | not because TPE finds better parameters than random at 100 trials, but because a seeded sampler owns reproducibility instead of every call site threading a seed correctly | 0016 |
| `n_jobs=1` everywhere, plus LightGBM `deterministic`/`force_row_wise` | a multi-threaded histogram reduction is only approximately reproducible; this project's standard for "unchanged" is bit-identical. Costs ~9.5 min for 400 trials, which is affordable at 57,727 rows and would not be at 10x | 0016 |
| CatBoost rejected | Component 4's contract has no categorical features — all 26 are counts, day-deltas, rates, booleans — so its distinguishing capability is inert | 0016 |
| MLflow rejected | every property tracking was wanted for (model, seed, params, feature version, fold, objective, trial, score) is already in the trials table and its manifest. A tracking server to re-express recorded information is expansion for its own sake | 0018 |
| `data/processed/tuning/` as a fourth kind | a trials table is not model-ready, not a model output, and not a measurement *about a model on a test window* — its numbers come from validation windows that are training data downstream | 0018 |
| `data/processed/calibration/` as a sixth kind | a C9 drift table carries an `ece` per (model, fold) on the test window, and so does `evaluation_metrics_*`. Filed together there would be two authoritative ECEs for the same cell. C5 stays the only producer of headline metrics | 0024 |
| calibrated predictions in `predictions/`, not the new layer | ADR 0014 said so in advance, and it is what lets `evaluate --predictions` read them with no change to Component 5 | 0024 |
| re-executing C6-C8 rather than retro-fitting model persistence | a missing recording of a deterministic computation is not the same obstacle as missing data (contrast ADR 0019). A bit-identical re-derivation is stronger evidence than a pickle: it proves the pipeline reproduces, not merely that something was saved | 0026 |
| the recovered logit over the native margin | the calibrated artifact is then a pure function of an already-committed column plus two floats, auditable years later with no live model | 0027 |
| temperature scaling deferred, not rejected | it is Platt with the intercept fixed at 0, so the fitted Platt intercept already answers what a temperature would have been; the parameters are persisted if a later component wants it | 0027 |

---

## Important paths

```text
src/sentinel/boosting/                   Component 7: the tuned tree models
  definitions.py                         TUNED_PARAMS lives here, as literals
  tuning.py                              the protocol that cannot reach a test window
src/sentinel/modeling/                   Component 6: the first fitted models
data/processed/predictions/              model outputs; NEVER joined onto features
  baseline_predictions_<UTC>.parquet     the artifact Component 5 consumes
  baseline_coefficients_<UTC>.parquet    standardised, + scaler_mean / scaler_scale
  baseline_training_log_<UTC>.parquet    one row per (model, fold)
  boosted_predictions_<UTC>.parquet      C7's artifact; C6's is NEVER overwritten
  boosted_importances_<UTC>.parquet      diagnostic only, NOT an attribution
  boosted_training_log_<UTC>.parquet     one row per (model, fold)
data/processed/tuning/                   search trials; NO number here is a result
data/processed/neural/                   C8 experimental categoricals; NOT features
data/processed/calibration/              C9 calibrators + diagnostics; NOT joinable
  tuning_trials_<UTC>.parquet            400 rows; validation windows, not test
docs/analysis/baseline_models_findings.md
docs/analysis/boosting_models_findings.md
docs/data_contracts/baseline_predictions.md
docs/data_contracts/boosted_predictions.md
docs/interview/component_7.md
scripts/profile_baselines.py             read-only, train windows only
scripts/profile_neural.py      read-only, train windows only
scripts/profile_boosting.py              read-only, train + calibration only
src/sentinel/ingest/socrata.py            the client. Most important file.
src/sentinel/ingest/food_inspections.py   orchestration
src/sentinel/ingest/manifest.py           provenance record
src/sentinel/query/duckdb_queries.py      NAMED_QUERIES live here
src/sentinel/config.py                    every tunable setting
data/raw/food_inspections/                output: parquet + manifest_*.json
docs/api/socrata_findings.md              verified API behaviour — read before
                                          touching the client
docs/data_contracts/food_inspections_raw.md   what a raw file guarantees
docs/decisions/                           19 ADRs

src/sentinel/entity/evidence.py           the match rules. Read the findings
                                          doc before changing any of them.
src/sentinel/entity/models.py             DEFAULT_THRESHOLDS - every tunable
                                          number, each citing its measurement
src/sentinel/entity/normalize.py          normalization rules
scripts/profile_entities.py               36 read-only data profiles
data/interim/entity_resolution/           output: 3 parquet + manifest_*.json
docs/analysis/entity_resolution_findings.md   why Component 2 works the way it
                                          does. Read before touching entity/.
docs/data_contracts/establishment_assignments.md   Component 2 output contract

src/sentinel/target/violations.py         the violation parser + the narrative
                                          exclusion list
src/sentinel/target/construct.py          eligibility gates and labelling
src/sentinel/target/models.py             CODE_ERA_START, INSPECTED_RESULTS,
                                          TARGET_DEFINITION_VERSION

src/sentinel/evaluation/simulate.py       the slot model and the five schedules.
                                          The BAU identity lives here.
src/sentinel/evaluation/folds.py          rolling-origin fold construction
src/sentinel/evaluation/contract.py       the seam Component 6+ hands scores through
src/sentinel/evaluation/metrics.py        hand-rolled; sklearn-verified in tests
src/sentinel/evaluation/validate.py       the seven leakage checks, re-derived
scripts/profile_evaluation.py             17 read-only evaluation profiles
data/processed/evaluation/                output: 6 parquet + manifest_*.json
docs/analysis/temporal_evaluation_findings.md   the measured results
docs/data_contracts/temporal_evaluation.md      the evaluation contract. §1 is
                                          the estimand — read it before quoting
                                          any number from this component.
scripts/profile_target.py                 31 read-only target profiles
data/interim/target/                      output: parquet + manifest_*.json
docs/analysis/target_construction_findings.md   why the target is defined this
                                          way. Read before touching target/.
docs/data_contracts/inspection_targets.md  Component 3 output contract

src/sentinel/features/definitions.py      FEATURE_SPECS: the single source of
                                          truth for every feature
src/sentinel/features/historical.py       the range join and THE boundary
src/sentinel/features/validate.py         temporal_boundary_holds lives here
scripts/profile_features.py               21 read-only history profiles
data/processed/features/                  output: parquet + manifest_*.json
docs/analysis/as_of_feature_engineering_findings.md   why the boundary is `<`
docs/data_contracts/as_of_features.md     the output contract and the leakage
                                          rules for Component 5
```

---

## Commands

```bash
uv sync                                       # setup
uv run sentinel ingest --dev                  # small pull (SENTINEL_DEV_ROW_LIMIT)
uv run sentinel ingest --limit 5000           # explicit cap
uv run sentinel ingest --full                 # entire dataset (~70 min, verified)
uv run sentinel query --list
uv run sentinel query --name row_count
uv run sentinel resolve                       # entity resolution (~45 s)
uv run sentinel resolve --dry-run --report    # validate, write nothing
uv run sentinel build-target                  # target construction (~25 s)
uv run sentinel build-target --dry-run --report
uv run sentinel build-features                # as-of features (~16 s)
uv run sentinel build-features --dry-run --report
uv run python scripts/profile_entities.py     # 36 read-only entity profiles
uv run python scripts/profile_target.py       # 31 read-only target profiles
uv run sentinel train-baselines               # 3 baseline models, 54 fits (~30 s)
uv run sentinel train-baselines --dry-run --report
uv run sentinel train-baselines --models logistic_regression   # repeatable flag
uv run sentinel tune-boosting --trials 100     # 4 studies, 400 trials (~564 s)
uv run sentinel tune-boosting --models xgboost --fold-set quarterly --trials 20
uv run sentinel tune-boosting --dry-run --report
uv run sentinel build-neural-categoricals --report   # 0.5 s, C8 experimental layer
uv run sentinel tune-neural --report                 # 550.9 s, then freeze by hand
uv run sentinel train-neural --report                # 1998.7 s, 234 fits, writes figures
uv run sentinel train-boosting                # 3 boosted models, 54 fits (~21 s)
uv run sentinel train-boosting --models lightgbm
uv run sentinel train-boosting --dry-run --report
uv run sentinel evaluate                      # heuristics only (~165 s)
uv run sentinel evaluate --predictions data/processed/predictions/baseline_predictions_<stamp>.parquet
                                              # + the C6 models (~238 s)
uv run sentinel evaluate --predictions data/processed/predictions/boosted_predictions_<stamp>.parquet
                                              # + the C7 models (~68 s)
uv run sentinel evaluate --folds-only --report   # the split only, < 1 s
uv run sentinel evaluate --dry-run --report
uv run python scripts/profile_features.py     # 21 read-only history profiles
uv run python scripts/profile_evaluation.py   # 17 read-only evaluation profiles
uv run python scripts/profile_baselines.py    # 10 profiles, TRAIN WINDOWS ONLY
uv run python scripts/profile_boosting.py     # 7 profiles, train + calibration only
uv run pytest                                 # 1,776 tests, offline
uv run pytest -m live                         # 3 live tests, hits the real API
uv run ruff check . && uv run ruff format --check .
uv run mypy src/sentinel scripts
```

One of `--dev`, `--limit`, `--full` is required — there is no default scope, so
a bare `sentinel ingest` cannot accidentally pull 314k rows.

---

## Naming conventions

* Raw files: `food_inspections_<YYYYMMDD>T<HHMMSS>Z.parquet` (UTC, start time).
* Manifests: `manifest_<parquet stem>.json`, same directory. The `manifest_`
  prefix is what `.gitignore` whitelists — keep it.
* Env vars: `SENTINEL_` prefix, matching the `Settings` field name.
* Private helpers are `_prefixed`; tests import public names only.

---

## Assumptions being made

* `inspection_id` is unique and totally orderable. Verified consistent with
  observed pagination; not proven across the whole dataset.
* The Chicago dataset stays publicly available without authentication.
* A single machine can hold a full pull in memory. **Unverified at 314k rows.**
* `:@computed_region_*` values are Socrata-derived, not authoritative source
  data, and are re-derivable from latitude/longitude.

---

## Open design questions

1. **Should schema divergence be fatal?** Currently a warning. If Chicago drops
   a column Sentinel depends on, ingestion succeeds with a null column. A
   later component may want a declared required-column set that fails loudly.
2. **Incremental ingestion.** Full pulls will get wasteful. `ETag` /
   `X-SODA2-Truth-Last-Modified` / a `$where` on `inspection_date` are the
   options. Deferred until a component needs freshness.
3. **Memory at full scale.** All pages accumulate before the single Parquet
   write. If `--full` proves heavy, write per-page row groups instead.
4. **Same-second filename collision.** One-second filename resolution means two
   runs starting in the same second collide. Not observed; needs sub-second
   precision or a counter if it ever matters.
5. ~~**Is `license_` ever a usable key?**~~ **Answered.** No - it fails in both
   directions and is now supporting evidence only. See the facts section above.
6. **Cross-snapshot identity.** `establishment_id` is stable for one snapshot
   only. A crosswalk mapping retired ids to successors is unbuilt (ADR 0007).
7. **Same-name outlets at a dense address.** Two outlets of one chain at one
   address with no store number and no distinguishing `aka_name` still merge.
   This is the residual false-merge risk; bounded to mega-addresses.
8. **Should there be an `--as-of DATE` resolution mode?** Identity is currently
   reconstructed from the whole snapshot, which is argued to be legitimate
   rather than leakage. A strict mode would cost one run per evaluation fold.
9. **Can the pre-2018 era support its own target?** 172,879 rows use the old
   Critical/Serious scheme. A separate target could be defined for them, with its
   own definition version. Not attempted.
10. **Should the target become a count or a severity grade?** `v1` is binary
    presence; a count reflects inspector verbosity as much as risk. Revisitable
    as `v2`.
11. **Should history reset at a tenant change?** Component 4 exposes the
    transition as features instead of resetting, diverging from spec §3.3. A
    reset variant is an ablation, not a redesign.
12. ~~**Does `days_since_any_inspection` help or just encode scheduling
    policy?**~~ **Answered — and the answer is "both".** Component 6 built the
    ablation as two fitted models. Keeping the feature helps on the quarterly
    folds (NDE 0.2326 vs 0.2238) and its coefficient is stable at +0.24 with no
    sign flip, so it is not worthless. But on `covid_shift` the ablation **wins**
    (NDE 0.2571 vs 0.2512): it does partly encode scheduling policy, and when the
    policy itself breaks a model leaning on it is the more fragile one.
13. **Should model selection prefer the shift-robust model?** STILL OPEN, and
    Component 7 made it harder rather than easier. C6 measured a clean inversion on
    `covid_shift`; C7 measured a *metric-dependent* ordering — lightgbm takes NDE and
    ROC-AUC, `logistic_regression` takes PR-AUC and precision@k. So there is now no
    single "shift-robust model" to prefer, and the project still has no rule for which
    fold set governs a release decision. Component 9 or a policy component will have to
    settle it.
14. **Does the missing-indicator encoding need an interaction term?** ANSWERED, and the
    answer is "not much". A tree gets the interaction for free, and both boosters
    finished within 0.005 NDE of the logistic model. If a differing slope in the missing
    group mattered materially, the tree models should have found it.
15. **Is the C6→C7 improvement real?** UNRESOLVED. NDE +0.0050 on the quarterly mean,
    but logistic wins 7 of 17 folds and the gap sits inside the seasonality redraw band.
    The project has the redraw and the per-fold table but no paired significance test
    across folds; a bootstrap over establishments would answer it more directly.
16. **What is the actual ceiling of this feature representation?** Two very different
    nonlinear learners and a penalised GLM land within 0.005 NDE. That is the strongest
    evidence the project has that the limit is the 26 features rather than the estimator
    — which makes the next high-value move a Component 4 change (311 complaints, weather,
    facility type, CDPH risk category, statutory days-overdue), not a fourth model.

---

## Lessons learned (Component 8)

* **When a leakage test fails, suspect the test first — again.** Seven of Component 8's
  leakage tests failed against correct code because the *fixture* assigned chains by row
  position, so appending or shuffling a row changed which establishment was in which chain.
  The advice Component 7 wrote down paid for itself on the first run.
* **And one of those failures was Component 7's own bug, reproduced verbatim.** The
  corruption test mutated rows after `train_end` — which includes the fold's own *test* rows,
  whose scores are *supposed* to change. HANDOFF.md records C7 shipping exactly that defect.
  Reading the warning was not enough to avoid writing it.
* **`id()` is only unique among live objects, and that was a real bug.** The live torch
  module for each `FittedNetwork` was kept in a dict keyed by `id()`. The multi-seed fits go
  out of scope, CPython reuses their addresses, and a later fold's record could be handed
  **another fold's network** — scoring a window with the wrong model and raising nothing.
  Caught by a test that builds a copy of a live record and expects a refusal. The training
  run in flight was killed and restarted.
* **Two identical background runs will silently halve your throughput.** A detached `nohup`
  attempt survived a shell exit and competed with the tracked run for the same single CPU
  thread, at the same fold, writing to two different artifact stamps. Killed both, ran one.
* **A capacity result can masquerade as a representation result.** The embeddings "failing"
  is really the network overfitting: best epoch orders almost perfectly by parameter count.
  The one-hot control is what made that diagnosable — without it the finding would have been
  "learned vectors don't work here", which is a different and wrong claim.
* **Measure the embedding geometry, do not just look at it.** A t-SNE blob is suggestive; a
  pairwise-cosine distribution identical to a random Gaussian table's is evidence. The
  measurement took ten lines and turned an impression into a finding.
* **Declare the tie-break before you need it.** On a coarse grid over a near-flat objective,
  two learning rates finishing within 1e-6 is likely, and without a declared rule the winner
  is decided by float summation order.

## Lessons learned (Component 7)

* **The most dangerous leak in a tuned component is the human, and it is the only one
  no artifact records.** Every other leak here is mechanically detectable after the
  fact. Fit-read-adjust-refit is not: the artifact passes every check and the model is
  simply better than it should be. That asymmetry is why the protocol has to make it
  structurally impossible rather than discouraged.
* **When a leakage test fails, suspect the test first.** Both of the two failures in
  this component were test bugs that looked exactly like leaks. One mutated rows after
  `train_end` — which includes the fold's own test rows, whose scores are *supposed* to
  change. The other built future rows from override dicts and concatenated them
  positionally, so every value landed in the wrong column. `model_feature_scenario`
  exists to prevent the second; I wasn't using it.
* **A guard that cannot fail is worse than no guard**, and Component 5 already shipped
  one. The tuning-region overlap check is unreachable through the public path, because
  the region and the horizon derive from the same fold. I kept it — the realistic way to
  break it is widening the region — and drove it via monkeypatch with a docstring saying
  so, rather than deleting it or pretending it was live.
* **Pre-register the expectation before opening the test window.** HANDOFF said boosted
  probabilities would be worse calibrated; a calibration-window probe run *before*
  training said otherwise, and the evaluation agreed with the probe. Because the probe
  came first, "boosters calibrated slightly better here" is a finding rather than a
  post-hoc rationalisation.
* **Row-order sensitivity scales with the estimator, not with the pipeline.** C6 measured
  7.049e-09 on coefficients; C7 measured **0.11 on a prediction**, because a booster
  subsamples in row order. The same canonical sort protects both, but only one of them
  would have been catastrophic without it.
* **Two libraries agreeing is a stronger result than either winning.** XGBoost and
  LightGBM finished 0.0021 apart. That agreement is what licenses the claim that the
  ceiling is the representation — a single booster's number could not support it.
* **A mean is not a result until you have looked at the folds.** +2.1% NDE looked like a
  clean win until the per-fold table showed logistic taking 7 of 17. Both numbers are
  true; only one of them is the headline.
* **Report the ablation that wins and still reject it.** `xgboost_class_weighted` has the
  best NDE in the whole project and is not adopted, because it degrades ECE by 35% to buy
  a margin smaller than the seasonality band. Writing that down is the difference between
  a protocol and a preference.
* **"Blocked" is a real deliverable when it is evidenced.** Inspector modelling could not
  be done; the honest output is a column inventory, a rejected-proxy list, an ADR and a
  regression test that fires if the field ever appears — not a proxy with the right
  vocabulary and the wrong meaning.

---

## Lessons learned (Component 6)

* **Profile before implementing, even when the design looks obvious.** Three
  decisions were reversed by measurement, and none would have been visible from
  reading code: `SimpleImputer(add_indicator=True)` emits collinear duplicates
  here because the within-family null masks are identical; a median fill for
  nullable booleans is 0.0056 from flipping; and row order moves coefficients by
  7e-09. All three would have shipped silently.
* **A "hypothetical" hazard is worth fixing when you can measure how close it
  is.** The boolean median fill has not flipped — but `priority_at_last_canvass`
  is at 0.5056 and falling monotonically. "One year of data from being wrong" is a
  much stronger argument than "could theoretically be wrong".
* **`assert` is stripped under `python -O`.** An import-time registry guard must
  `raise ValueError`. A guard that vanishes under an optimisation flag is not a
  guard.
* **Check whether an enum member matches more than you meant.**
  `columns_in_family(NullRule.NEVER)` returns the 16 never-null features, so
  `indicator_source_column` silently returned a never-null column as the source of
  a missingness indicator until a test caught it. Now it refuses `NEVER`.
* **A test that loops over a computed list can pass vacuously.** Three fold-isolation
  tests were green because the fixture spanned 1,400 days and `quarterly_folds`
  returned `[]`. The fix is a helper that asserts the list is non-empty; the lesson
  is to check that a loop body actually ran.
* **Prove the detector works, not just that the pipeline is clean.** A leakage
  suite that only asserts "nothing moved" is reassuring for the wrong reason if the
  model is too weak to exploit a leak. `test_the_leakage_detector_itself_works`
  overwrites a feature with the label and requires perfect separation.
* **A dead check is worse than no check**, because it reads as coverage.
  `scores_respect_the_decision_point` could not fail for the whole of Component 5's
  life; the horizon rejection it depended on was never recorded. Every new error
  check in Component 6 has a test that shows it failing.
* **Align persisted data by key, never by position.** A prediction artifact carries
  its producer's row order, not the evaluator's. Zipping them positionally would
  attach every score to the wrong establishment and still produce a plausible
  number — a defect with no symptom.
* **`ColumnTransformer` reorders its output to match the transformer list.** The
  coefficient at position *i* belongs to the *branch* order, not the input order.
  Getting it wrong mislabels every coefficient while predicting identically, so both
  orders are derived from one declared constant.
* **A standardised coefficient without its scaler statistics is unreadable.**
  Emitting `scaler_mean` and `scaler_scale` beside each term is what makes the
  interpretability artifact actually interpretable.
* **Wall clock in a Parquet file breaks byte-reproducibility.** `fit_seconds`
  belongs in the manifest beside `built_at`, where non-determinism already lives.
* **Narrow a reproducibility claim rather than overstating it.** Bit-identity holds
  for a fixed library set and BLAS thread count, not across them — so all three go
  in the manifest.
* **`random_state` is not always the thing that makes a fit deterministic.**
  `lbfgs` has no stochastic component; the canonical sort does the work. Recording
  the seed anyway documents intent, but the docstring has to say which one matters.

---

## Lessons learned (Component 5)

* **A helper named `test_*` is collected by pytest.** `folds.test_frame` had to
  become `window_frame` — any importable callable starting with `test_` is
  mistaken for a test case the moment a test module imports it.
* **Give polars an explicit schema when building a frame from records.** Columns
  that are null for some rows and populated for others (`seed` is null for
  deterministic schedules, an int for random ones) get typed from whichever kind
  appears first otherwise.
* **Polars column aggregates are typed as a wide union.** Narrow once —
  `folds.min_date` / `max_date` — rather than scattering `# type: ignore` over
  every date comparison.
* **The analytic denominator is worth deriving.** `A_random = 0.5` exactly
  because a random permutation's expected curve is the diagonal, and
  `A_optimal = 1 − P/(2N)`. Using the derivation rather than a sampled seed makes
  optimal land on exactly 1 and worst on exactly −1, which is the cleanest
  possible evidence the area code is right.
* **Prove a check can fail.** A check that always passes and a check that works
  are indistinguishable until one is shown failing, so the leakage suite includes
  deliberately broken inputs.
* **Choose a missing-value rule on a semantic argument, then measure what it
  cost.** Both null rules here are stated with their measured consequence — one
  helps, one hurts. Choosing the tier by its outcome would be fitting to the label.
* **A negative result can be the most interesting one.** Business-as-usual
  scoring at random within a quarter was not expected, and it reframes what
  Component 6 has to prove.
* **Implementing a metric and verifying it against sklearn beats importing it.**
  Zero runtime dependency, and "how do you know your PR-AUC is right?" has a real
  answer.

---

## Lessons learned (Component 4)

* **One measurement can settle a design argument.** Whether the boundary is `<`
  or `<=` looked like a judgement call until a single query showed
  `inspection_date` has one distinct time component. After that it was not a
  choice: same-day records are unorderable, so they cannot be history.
* **Put the temporal condition in exactly one place.** 26 features, one range
  join. If the predicate were repeated per feature there would be 26 chances to
  get it wrong, and a reviewer would have to check all of them.
* **Re-derive the invariant independently rather than trusting the code that
  produced it.** `validate.py` recomputes the latest contributing date from a
  separate query. A check that reuses the aggregation only proves the aggregation
  agrees with itself.
* **A NULL rendered as 0 is the quiet catastrophe.** 14,162 rows have no
  code-era history; writing 0 would tell a model a quarter of establishments have
  a clean priority record. Pair every event count with its inspection count so
  the difference is visible.
* **Materialize before validating.** Running dozens of small checks against a
  VIEW re-executed the whole join each time: 2m14s. As a TABLE: 15.6s, identical
  output.
* **Do not parse what cannot exist.** Priority did not exist before 2018-07-01,
  so pre-code rows need no classification at all - half the parser's work removed
  by a definition rather than an optimisation.
* **Resist feature-count inflation.** 26 features, each with a written reason.
  The measurement that a 365-day window is empty for 62% of rows is a reason to
  document it, not a reason to add four more windows.

---

## Lessons learned (Component 3)

* **Profile the value set before trusting a contract.** The raw data contract
  documented four `results` values; there are seven, and the three missing ones
  are exactly the "no inspection happened" cases that must not become negatives.
* **Look for regime changes before defining anything longitudinal.** The whole
  target hinged on noticing that "Priority" does not exist before 2018-07-01.
  A single `GROUP BY year` on terminology presence found it in seconds; without
  it, half the labels would have been silently wrong.
* **A plausible structural signal can be a trap.** The violation number looks
  like a severity code and is not one. Checking the association empirically
  (item 10: 42/11/47) took one query and prevented a badly wrong parser.
* **Prefer excluding narrow narrative spans to requiring strong evidence.**
  Requiring a citation code would have looked rigorous and produced ~21,281 false
  negatives. Excluding four boilerplate phrases changed 10 labels.
* **Absence of a label is not evidence of the opposite.** 72% of entries carry no
  severity marker, so the parser emits UNCLASSIFIED rather than "Core".
* **Distinguish "not measured" from "measured as zero".** `Pass` with no
  violation text is a true zero; `Fail` with no violation text is unknown. The
  same null means different things depending on the row.

---

## Lessons learned (Component 2)

* **Investigate before designing, and be willing to throw the design away.** Six
  planned decisions were reversed by measurement: licence turned out to be too
  *fine* grained rather than too coarse; the street-suffix canonicalization
  table would have been dead code; geographic *distance* thresholds were
  meaningless because within-address spread is exactly zero; the temporal
  succession rule addressed a quarter of the cases it assumed; and the fuzzy
  name tier with its 100-pair calibration was retired before it was written.
* **Aggregate metrics hide over-merges.** The first full run passed every
  structural check while fusing about twenty O'Hare restaurants into one
  establishment. Only reading the largest clusters by hand found it. Inspect the
  biggest outputs, never just the summary.
* **`dba_name` is not always the business.** At O'Hare it is the concessionaire
  (`HOST INTERNATIONAL INC`); elsewhere it is the holding company
  (`1918 WINTER STREET ILLINOIS LLC` for a Mariano's). `aka_name` often carries
  the real identity.
* **A veto written too broadly is as damaging as no veto.** The first version of
  the `aka_conflict` veto fired on any two differently-named neighbours and
  would have blocked legitimate merges. The unit tests caught it before it
  reached the data, which is the argument for tests that assert what must *not*
  happen.
* **Classify ordinary difference as a decision, not a doubt.** Marking
  same-address different-name pairs as ambiguous produced 108,597 of them and
  buried the real 747-pair review queue.

---

## Lessons learned (Component 1)

* **Probe the API before writing the client.** Two behaviours would have been
  wrong if assumed from documentation: the absent 50k page cap, and the
  string-encoding of every value.
* **`$order` changing the returned column set was not documented anywhere.** It
  was found only by comparing the field count between an exploratory request
  and the real paginated one. Compare what you *get* against what you *expected*
  at every step.
* **An empty result still carries schema headers.** The first implementation
  lost the schema on a zero-row pull because `iter_pages` returns before
  yielding an empty page. Fixed by retaining the last-seen schema on the client.
* **DuckDB's `DESCRIBE` columns are `column_name` / `column_type`**, not
  `name` / `type`.
