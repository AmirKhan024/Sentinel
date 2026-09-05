# MEMORY

Compact context for future Claude Code sessions. Not a project description —
see README.md for that and STATUS.md for current state.

**Read this first, then STATUS.md, then HANDOFF.md.**

---

## Working agreement

* **One component at a time.** 21 components are planned; Components 1-9 and 11-13
  exist (10 is blocked by ADR 0019). Never implement ahead. If something belongs to a later component,
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

### Component 11 invariants — an EXPLANATION has a horizon too

94. **The BACKGROUND is part of the explanation, and is the leakage surface.** A SHAP value
    says how far a feature moved the output *relative to a reference set*. Drawn from the
    test window it would encode the period the model is being judged on - and every value
    would stay finite, additive and plausible. Nothing raises. Reference rows come from
    `modeling.train.training_frame` ONLY, and TWO checks re-derive it: one on dates, one on
    split membership, because a date comparison is weaker than the split.
95. **Additivity is NOT accuracy for the permutation method.** Its path telescopes to
    `f(row) - f(background)`, so `base + sum(phi)` reconstructs the output exactly at ONE
    round. Measured residual **0.0 at every round count from 1 to 64**. A green additivity
    check proves the arithmetic, never the credit split. Say this out loud whenever anyone
    reads a passing check as evidence the values are right.
96. **What IS approximate is measured, and the global statistic converges much faster than
    any single value.** At 8 rounds vs a 64-round reference: median per-value error 1.00%,
    global importance rank rho **0.9964**. So the network's global ranking is quotable and
    its per-row values are not, and every neural row carries `is_exact = false`.
97. **Components 6 and 7 order the same 30 columns differently at 19 OF 30 POSITIONS.**
    `ordered_matrix_columns` (C6, ColumnTransformer branch order) vs `matrix_columns` (C7,
    natural order). Picking wrong produces a table whose every value is arithmetically
    correct and attached to the WRONG FEATURE - no exception, no failed additivity check,
    because a sum is invariant to a permutation of its terms. The choice is per-model in
    `EXPLAIN_REGISTRY.name_source` and re-derived independently by a check.
98. **SHAP explains the BASE score, never the calibrated probability.** Platt is a separate
    two-parameter monotone map. `explanation_cases` carries `base_score` and
    `calibrated_probability` side by side so a user sees both and neither is mistaken for
    the other. ADR 0030.
99. **One output space: log-odds.** Contributions add up in the margin and NOT in
    probability, because sigmoid is not linear. `OutputSpace` declares one member;
    probability space is refused in prose rather than declared-and-unreachable.
100. **`shap` is DEV-ONLY, and no runtime dependency was added.** xgboost and lightgbm ship
    exact TreeSHAP (`pred_contribs` / `pred_contrib`); linear SHAP is
    `coef_j * (z_j - E[z_j])`. Only the network needed an approximation. Measured agreement
    with the oracle: **0.0**. Same pattern as C5's metrics vs sklearn. `shap` needs
    `numba>=0.67` and `llvmlite>=0.49` pinned or the resolver breaks numpy 2.5.2 - which is
    what ADR 0026's gate is baselined on.
101. **Component 11 re-executes the frozen fits, under ADR 0026's gate, calling C9's
    comparison rather than reimplementing it.** 166,144 test rows, `==`, zero mismatches,
    build RAISES before computing a single attribution if it fails. One definition of "the
    same model" in the project, not two. ADR 0029.
102. **`xgboost_chain_embeddings` is UNSUPPORTED, and its rows are NULL not 0.0.** Its
    booster is reachable only via `neural.embed._scorer_for`, private; `neural.train.
    scorer_for` is public, which is the only reason the network could be explained. Zero is a
    legitimate attribution meaning "did not move the score", so placeholder zeros would read
    as a model that used no features. ADR 0031 proposes `embed.booster_for` and does not add
    it.
103. **Component 11 MUST NOT select a model**, and is recorded as blocked from doing so in
    every manifest it emits. Attribution describes reasoning, not correctness - a model can
    lean hard on a feature that is misleading it, which C6 measured happening under shift.
104. **An attribution is keyed by `target_inspection_id`, so joining it back onto a feature
    table is ONE LINE AWAY** and would be a model's own reasoning about a row becoming a
    feature of that row. The sharpest never-join rule in the project. ADR 0028.

---

### Component 12 invariants — a GROUP LABEL must not see the future either

105. **The attribute a row is audited under comes from an inspection STRICTLY EARLIER than
    that row.** This is the quietest leak in the project. A leaked feature makes one column
    wrong; a leaked group label leaves every number finite, additive, plausible and in range
    and changes only *which neighbourhood the number is about*. Nothing raises.
    `groups.check_temporal_validity` re-derives the inequality per row rather than reading
    Component 8's manifest, and `CANONICAL_SORT` is applied before any metric touches the
    frame. Measured minimum lag: **1 day**.
106. **The as-of geography and the row's own recorded geography NEVER disagree** — 0 of 57,041
    community-area rows, 0 of 57,326 ZIP rows. A restaurant does not move, so the temporally
    safe choice cost nothing. That is why it was taken, rather than on principle. ADR 0033.
107. **WARD IS REFUSED, and the dataset is the evidence.** The snapshot publishes two ward
    layers and they assign different region ids to **56,451 of 57,403** rows (98.3%). A ward
    id is a property of a boundary VERSION, not of a place. Census tract (797 groups over
    32,696 rows), point geography and city/state (312,957 of 314,245 say CHICAGO) are refused
    too. **The refusals are ROWS in `fairness_group_definitions`**, so they keep travelling
    when someone opens the Parquet instead of the ADR.
108. **`community_area` is a Socrata computed-region id, NOT the official community-area
    number.** No boundary file is ingested, so **no neighbourhood is named anywhere** in the
    artifact. Guessing the mapping would attribute a measured disparity to the wrong
    neighbourhood in the one document whose purpose is to be trusted about which neighbourhood.
109. **Support is decided BEFORE any metric, and the per-fold grain does not survive it.**
    Median (fold, community area) cell: **16 rows**; 4 of 1,288 clear the 200-row floor. The
    reporting grain is the pooled fold set — legitimate (every row still held out) but
    labelled on every row as *the system as operated over 2022Q2-2026Q2*, because the 17
    windows were scored by 17 differently-fitted models.
110. **Floors frozen from the profiler before any result: 200 rows / 20 positives / 20
    negatives for ranking, 300 for calibration.** 300 is arithmetic, not taste: 15 equal-mass
    bins x 20 rows. **The bin count was NOT reduced** to let 18 more community areas through,
    because a group ECE at a different bin count is incomparable with Component 9's global one
    — which is the exact comparison the component exists to make. ADR 0034.
111. **An unsupported group is a ROW with real counts, a null value and a stated reason. Never
    an absent row.** `validate.no_group_disappeared` compares the support table against the
    values observed in the data. "Equal performance across groups" may never rest on the
    groups that were dropped, and 27 of 78 community areas were below the floor.
112. **`__UNKNOWN__` IS A GROUP, not a null.** It is a superset of the rows with no prior
    inspection of any type — the same rows the null-rule indicators fire on — so dropping it
    would delete the most interesting row set from a missingness audit. It turned out to be
    the component's sharpest finding.
113. **A MEASURED DISPARITY IS ADVISORY AND CAN NEVER FAIL THE BUILD, and there is no flag to
    change that.** 13 error checks are about the audit's integrity; 3 advisories are about the
    world. A red build is a demand for action, and the actions available are to change the
    model, the metric or the threshold — two of which are worse than the disparity.
    `test_an_enormous_disparity_is_advisory_and_never_an_error` asserts a 0.95 ECE spread
    leaves every error check green and exit code 0. ADR 0034.
114. **There is deliberately NO single fairness score.** Calibration parity and selection-rate
    parity cannot both hold when base rates differ, and they differ by 34 points here. Four
    disparity measures are reported side by side; every ratio is measured against the **pooled
    population**, never a nominated group.
115. **Selection rate and capture rate are NEVER combined.** Representation ("was this group
    prioritised?") and effectiveness ("did that find its violations?") are different questions,
    and a group can be prioritised often and still have its violations missed. Capture is also
    NOT `recall_at_k`: the cutoff is city-wide and competitive, so capture is what a
    competition against every other group left a group.
116. **Component 12 re-executes NOTHING — the first component of which that is true.** Its
    integrity claim is the opposite of ADR 0026's: **nothing moved**. Every input's sha256 is
    read before and again after the last write, and a difference is an error. No gate, no
    refit, no BLAS thread sensitivity — run it with any thread count.
117. **Polars aggregates in parallel and float summation order is not stable.** Two runs
    produced `mean_abs_shap` differing at **4.4e-16** (every rank, correlation and count
    identical). Fixed by sorting before aggregating. **A table that is *nearly* reproducible is
    a table whose two-run checksum comparison has stopped being a detector.**
118. **`ece` uses equal-mass bins, so rows tied at a bin boundary are assigned by ARRIVAL
    ORDER.** Shuffling the prediction rows changed the pooled reference value and therefore
    every disparity, until `groups.CANONICAL_SORT` was added. Found by a test, not by reading.
119. **Component 9 and Component 11 name the same model differently** — `xgboost_platt` vs
    `xgboost`. Looking a profile up under the calibrated name found nothing and drew no
    figure: **a missing figure looks exactly like a figure the data could not support.**
    `figures.base_model_name` is the one place that translation lives.
120. **Component 12 MUST NOT select a model**, and is recorded as blocked from doing so in
    every manifest. It made open question 13 harder, not easier: the best-NDE model
    (`neural_numeric_only`) is the one whose calibration reached the fewest groups.
121. **Nothing in `data/processed/fairness/` may be joined onto a feature table.** These tables
    are keyed by GROUP rather than by row, which is what stops them becoming features — and a
    number meaning "the model was well calibrated in this neighbourhood last quarter",
    broadcast back onto training rows, would be the most self-fulfilling input this project
    could construct. ADR 0032.
122. **A green run means the audit is SOUND, not that Sentinel is FAIR.** `does_not_establish`
    travels in every manifest and prints on every run: not causality, not discrimination, not
    the absence of bias, not legal compliance, not ethical acceptability, not equal treatment,
    not an optimal fairness policy. ADR 0035.
123. **ADR 0019's gap is INHERITED, not discovered.** The target is that a violation was
    *cited*, and Chicago assigns inspectors by district — so geography is close to the
    strongest available proxy for who inspected, and **nothing in this project separates
    establishment risk from differential inspection practice.**

### Component 13 invariants — a POLICY must not see the future, or the model's own past

124. **Eligibility reads ONE as-of column and no outcome column.** `coverage_eligible <=>
    prior_canvass_count_code_era == 0`, a Component 4 feature built under ADR 0010.
    `eligibility.refuse_forbidden` raises before the predicate is built if `target` or
    `target_status` is offered, and no decision table carries a label column at all — so a
    future edit that wanted one would have to change the contract to get it.
125. **A NULL history count is NEVER eligible.** The column carries `NullRule.NEVER`, so a zero
    is a real observation of no history and a null means the count itself is missing. `fill_null(-1)`,
    never `fill_null(0)`: mapping a null to eligible would reserve capacity for rows a join
    quietly failed to match.
126. **NO Component 12 number reaches a rank.** The group label and support status are read
    *onto* recommendation rows and never back into a decision. Those numbers are computed from
    held-out outcomes, so ranking on one would be ranking on the future — and the artifact would
    look completely normal. `_queue_signature` rebuilds the whole queue with both withheld and
    `validate.warnings_do_not_change_the_queue` compares ranks exactly. The end-to-end test runs
    the entire component twice, with and without the group artifacts.
127. **NO SCORE IS WRITTEN ANYWHERE IN COMPONENT 13.** `allocation.py` reads scores to order
    rows and writes none. `base_score` and `score` are copied verbatim from Component 9. ADR 0037:
    once a score is adjusted, nobody can say whether an establishment is in the queue because the
    model thinks it is risky or because a policy promoted it.
128. **`model_rank` sits beside `final_policy_rank` on every row.** Where they agree the model
    decided; where they differ the policy did, and `decision_mechanism` names which mechanism
    moved it. That pair is the component's whole design.
129. **Capacity is a RANK POSITION, never a probability.** Every cutoff descends from
    `simulate.capacity_k_values` and the window's own measured median daily rate. No probability
    threshold exists and there is no flag to add one — refused by Component 12 in prose and by
    Component 13 in `CAPACITY_SEMANTICS`.
130. **`reserve_target = floor(share * k)`, truncated.** A reserve may never spend more than the
    share it declared. The consequence is that a small share at a small cutoff floors to zero
    slots, reported as an advisory rather than rounded away — a policy that quietly overspends
    its own budget is worse than one that is visibly inert.
131. **The two mechanisms are disjoint BY CONSTRUCTION**: the reserve is filled from rows the
    risk block did not take. Checked anyway, because "by construction" is a claim about code that
    was correct when it was written.
132. **Advisories NEVER fail a build, and there is no flag to change that.** ADR 0034's line,
    inherited and sharper here: the cheapest way to make a red "this reserve gave up 34
    citations" build green is to delete the reserve, and that is a decision about how a city
    allocates enforcement rather than a defect a CI runner may fix. Five tests assert the
    opposite of every red test for exactly this reason.
133. **The determinism claim is SCOPED to the inputs including the override file.** Human
    overrides are external decisions; the manifest pins the file by checksum rather than claiming
    a person's typing is reproducible.

### Component 14 invariants — a SCHEDULE must not re-rank, and must not raise capacity

130. **`final_policy_rank` is the ONLY ordering key.** `scheduling/allocation.py` reads no
    score, no probability, no mechanism, no eligibility flag and no geography. The type system
    helps — `Placement` carries no score to read — and `c13_provenance_is_preserved` re-reads
    Component 13's artifact after the run and compares eight columns row by row at error
    severity. A scheduler that reordered by risk would be a second policy layer with no ADR.
131. **Capacity is inherited and never created.** Every horizon is
    `ceil(k / test_median_daily_capacity)` over the fold's own observed operating days. There is
    no function parameter, no CLI flag and no branch that raises a slot count or extends a
    horizon, and `tests/test_cli_scheduling.py` asserts that `--capacity`, `--slots-per-day`,
    `--horizon-days`, `--extend-horizon` and `--threshold` all stay absent.
132. **The calendar is READ, never generated.** An operating day is a date Component 13's
    universe carries. Three inspections in the snapshot fall on a weekend, so a synthesised
    Monday-to-Friday calendar would be wrong at the edges — and the holiday list it would need
    is unverifiable here.
133. **`flat_median` is a SCENARIO and is tautological at two of five cutoffs.** The horizon is
    `k / median` days of `median` slots, so it holds exactly `k`: backlog zero, utilisation
    exactly 1.000, before anything is measured. Every scenario row carries `is_scenario`, an
    advisory fires whenever any is written, and **no summary or manifest field ever pools it
    with the observed calendar** — pooling would divide the headline by however many modes ran.
134. **`inspection_schedule` has NO `execution_status` column, deliberately.** An execution
    outcome must never retroactively change a plan, and the strongest form of that guarantee is
    not giving it a column to write into. A consumer who wants both facts joins on the key.
135. **A deferred row still occupies a slot.** `OCCUPYING_STATUSES = {scheduled, deferred}`. A
    deferral moves an inspection rather than removing it, so it still consumes capacity, and the
    accounting identity is `n_scheduled + n_backlog + n_cancelled == n_recommended` with
    `n_deferred` a *breakdown* of the scheduled block rather than a fourth term.
136. **"Not scheduled" is NEVER redefined as "not recommended".** A backlogged row keeps its
    rank, mechanism, reason code and eligibility. This is the easiest defect in the component to
    introduce and the hardest to see, because the resulting artifact is internally consistent.
137. **Three human layers, never merged.** Recommendation override (*who*, Component 13),
    scheduling adjustment (*when*, Component 14), execution deviation (*what happened*). Three
    id namespaces, three disjoint verb vocabularies, enforced at **import time** by
    `definitions._guard_registry` and re-checked from the artifacts.
138. **A re-plan appends a planning run; it never mutates one.** `SchedulePlan` is frozen and
    `replan()` returns a new one. Completed rows keep their slot forever; rows on a day before
    the re-plan point are frozen; `original_*` is copied forward untouched. The one exemption is
    narrow: a `not_performed` row moves even from a past day, because freezing it would strand
    the inspection the report exists to rescue.
139. **A re-plan backfills where an override does not, and the distinction is written down.** An
    excluded row is a human decision that the slot should not be used; a day that did not happen
    is capacity that still exists. A cancellation — by adjustment or in the field — is a removal
    and is never backfilled.
140. **A scheduling adjustment NEVER displaces a coverage-reserve row.** It always takes the
    lowest-ranked `risk_priority` row on the target day, and refuses the run if none exists.
    Taking the reserve's slot would convert a scheduling change into a coverage cut silently.
141. **No solver, and the reason is recorded rather than assumed.** ADR 0043: strict priority
    preservation has a closed form over a unique contiguous rank, so there is nothing to search
    over; every constraint an optimiser would trade off is absent; and a solver returns a
    search-order-dependent answer among equal objectives, which breaks byte-identity.

## Key operational-scheduling facts (measured 2026-08-26 on the full policy artifact)

142. **The observed calendar exists and is an observation.** `inspection_date` on the
    recommendation universe gives each fold's real operating days and the real volume worked on
    each. 2026Q2: 63 days, min 1 / median 28 / max 55. The median agrees with Component 5's
    `test_median_daily_capacity` by construction, and the profiler cross-checks it.
143. **The horizon rule is total.** `ceil(k / median)` demands more operating days than the fold
    contains in **0 of 90** (fold, capacity) cells. Widest: covid_shift `k_pct_10`, 41 of 390.
144. **44 of 90 cells cannot fit their approved queue** into their own horizon — 48.9% — for
    **784** inspections. Under `flat_median` the backlog is **zero in every cell**, by
    construction. That contrast fixed `DEFAULT_CAPACITY_MODE`.
145. **THE HEADLINE: 1,012 of 3,459 coverage-reserve slots (29.3%) are lost to the horizon.**
    136 of 273 reserve-bearing cells lose some; **91 lose it entirely**. Cause: Component 13
    fills the risk block at ranks `1..n_risk` and puts the reserve after it, so the reserve is
    *always* the rank tail — 0 exceptions across all 273 cells — and a strict-priority schedule
    takes the tail first. **Reported at advisory severity and deliberately not corrected**:
    promoting reserve rows is re-ranking. Per policy the share lost runs 25.2% to 51.7%.
146. **Establishments legitimately recur inside one fold.** 1,573 establishment-fold pairs hold
    more than one scored canvass; 1 of 2,890 rows in the `k_1_week` queue is a repeat. So
    uniqueness is an **error** check on `target_inspection_id` and only an **advisory** on
    `establishment_id` — the stronger invariant would have gone red on correct data.
147. **Quarter-opening days are systematically thin.** 20 of 180 horizons open on a day holding
    less than half the window's median rate; 2024Q1 and 2026Q1 both open on a single inspection
    against medians of 34 and 35. Every `k_1_day` number in those cells is dominated by one
    unrepresentative day, and an advisory names them rather than the horizon starting somewhere
    flattering.
148. **Production run:** 1,260 cells (2 modes × 7 policies × 18 folds), 141,582 queue rows,
    136,094 scheduled, 5,488 backlogged in 308 cells, 11,067 idle slots in 679 cells, **0
    inversions**, 27 error checks green, 7 advisories, ~12 s. **13 of 13 tables byte-identical
    across two independent runs**, and identical again under shuffled recommendation, adjustment
    and execution rows.
149. **No execution event in this repository describes anything that happened in Chicago.** The
    contracts and engines are fully implemented and tested against synthetic fixtures; the
    production run's execution and adjustment logs are **typed empty**, and an advisory says so
    so nobody mistakes a typed zero for a measurement.

## Key decision-policy facts (measured 2026-08-26 on the full calibrated artifact)

Detail in `docs/analysis/policy_findings.md`.

* **7 policies x 4 models x 18 folds x 5 capacities, 1,453,760 recommendation rows, 38.9 s.**
  22 checks: 0 errors, 4 advisories. **11 of 11 tables byte-identical across two independent
  production runs.** Refits, re-executions, bit-identity gates: none.
* **THE HEADLINE IS A REFUTATION.** Establishments with no code-era canvass history are
  **10.4%** of the quarterly candidates and take **40-58% of the top of the queue** under pure
  risk — a selection ratio of **3.96 to 5.57** across all four models. Their citation rate is
  genuinely higher: **0.4883 vs 0.4283**, and they hold 11.9% of the positives. **The risk queue
  over-serves this population fourfold; it does not neglect it.**
* **Component 12's finding is about a DIFFERENT population.** Only **456 of 14,162** eligible
  rows (3.2%) sit in `__UNKNOWN__`. "No history" and "no geography" overlap and are not the same
  thing.
* **A coverage FLOOR is inert in 338 of 340 quarterly cells** at the population share (2 slots
  granted); 0 at half the share; 101 at twice it. It binds only where risk does not clear it,
  which on this data is almost nowhere.
* **A FORCED reserve costs citations, and the price grows with capacity.** At one week
  (2,780 slots): half share −8 for +76 eligible; population share −15 for +155; **double share
  −34 for +343**. The cost holds for every model except one cell of one model.
* **THE ONE-DAY DELTAS ARE INSIDE THE NOISE**: ±1 to ±3 citations out of 348 across 17 folds.
  `coverage_forced_population_share` posting +2 on `xgboost_platt` is the outlier, not the
  pattern — `logistic_regression_platt` posts −9 in the same cell.
* **NOTHING CHANGED FOR `__UNKNOWN__`.** 405 rows, 166 positives, **2 selected and 1 citation
  found — identical under all seven policies** at one day of capacity.
* **NO POLICY WINNER.** Two policies survive at `k_1_day` and neither dominates. The run prints
  *the data does not determine the correct policy*, and `a_winner_was_determined` fires as an
  advisory.
* **Production model: `xgboost_platt`, decided on CALIBRATION, not on discovery.** All four
  candidates' NDE sensitivity bands overlap under Component 5's 1,000-replication study — the
  headline metric of this project cannot tell them apart. ⚠ The discarded tie band (a ROC-AUC
  spread of 0.0058, a unit error) would have selected `neural_numeric_only_platt`; both outcomes
  are emitted on every run.
* **The eligible share of the one-day queue swings 0.828 (2022Q2) to 0.036 (2026Q2)**
  non-monotonically (2025Q4 = 0.667). Volatility, not a trend with a slope — but the two most
  recent quarters are among the three lowest.
* **44.8% of the recommended queue carries a `limited_history` warning.** A warning is never an
  abstention: Sentinel has no predictive interval, so there is nothing to abstain on.
* **On `covid_shift` all 22 one-day slots go to eligible establishments** under pure risk —
  100% of the queue against 15.5% of the population. Never pooled with a quarterly number.

## Key fairness-audit facts (measured 2026-08-25 on the full calibrated artifact)

Detail in `docs/analysis/fairness_findings.md`.

* **5 models x 2 geographies x 18 folds, 207,680 audited rows, 145.1 s.** 13/13 error checks
  pass, 13 advisories, inputs byte-identical before and after.
* **The disparity exists in the data before any model does**: outcome rate spans **0.2200 to
  0.5658** across supported community areas (0.2350-0.6099 across ZIPs) against a city-wide
  0.4283. So unequal selection by a working risk model is EXPECTED, not a finding.
* **Ranking varies more between neighbourhoods than between models.** Within-group ROC-AUC
  spans **0.509 to 0.710** over 51 supported community areas — spread 0.164-0.198 by model,
  against ~0.008 between the project's best and worst model. Community area 53 is the
  best-ranked group for all five models.
* **Component 9's global calibration improvement did NOT reach every group**: 25/33 community
  areas improved for `xgboost` and `lightgbm`, 23/33 for the GLM, and only **17/33** for
  `neural_numeric_only`. Coherent with C9 rather than against it — the network had the best
  UNcalibrated ECE and least to gain. Group ECE levels are roughly **double** the global
  figure (0.082-0.094 vs 0.0474), because a pooled ECE averages over groups that partly cancel.
* **The sharpest finding is the group with no geography**, 405 quarterly rows: 59.5% have no
  prior inspection of any kind (0.74% overall, **80x**), 61.7% no code-era canvass history
  (10.4% overall, **6x**), ROC-AUC **0.509** (chance), selected at **0.20x** the city rate,
  and **0.6%** of its violations captured by the top 5% against a city-wide 7.0%. Of its 166
  real violations the top 5% found **one**.
* The group with no recoverable geography IS the group with no history — C8's as-of join can
  only carry a location forward from an earlier inspection. **No causal direction is claimed.**
* **Capture ranges 0.006 to 0.151** across supported community areas against an overall 0.070.
* **Support**: `community_area` 51 of 78 supported (33 for calibration); `zip` 56 of 69 (41).
  286 groups observed, 132 supported, **154 insufficient and all recorded**.
* **Drift could not be answered**: exactly ONE quarterly fold per (model, geography) has
  enough support to compute a disparity at all, so every series is `insufficient_folds`.
  `DRIFT_MIN_FOLDS = 3` was frozen before any series existed.
* **`covid_shift`**: 8,840 rows — more than any single quarter — supporting **11 of 78**
  community areas and 5 for calibration. Six components, six divergences.
* Determinism: two full production runs are **byte-identical across all ten tables**.
* 247 new tests (2,084 -> **2,331**), 72 figures.

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
src/sentinel/scheduling/                 Component 14: the operational schedule
  definitions.py                         horizon rule, both capacity modes, the boundary
  horizon.py                             the observed calendar; the ONLY module setting capacity
  allocation.py                          greedy placement in policy-rank order; reads no score
  adjustments.py / execution.py          the two external human contracts
  replan.py                              appends a planning run, never mutates one
data/processed/scheduling/               tenth layer; NEVER joined onto features
  inspection_schedule_<UTC>.parquet      when, and in which slot
  schedule_backlog_<UTC>.parquet         approved and not reached; still recommended
  priority_preservation_<UTC>.parquet    what the calendar cost the reserve
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
src/sentinel/explain/                    Component 11: feature attribution
  definitions.py                         EXPLAIN_REGISTRY = the support matrix
  refit.py                               re-execution + ADR 0026's gate, REUSED
  background.py                          the leakage surface; training window ONLY
  attribute.py                           tree / linear / permutation; no shap import
src/sentinel/policy/                     Component 13: the decision policy layer
  definitions.py                         POLICY_GRID, the selection rule, the boundary
  eligibility.py                         one column, one predicate, and what it refuses
  allocation.py                          risk block, coverage reserve; NO score written
  select.py                              the lexicographic production-model rule
  governance.py                          warnings, and the human override layer
  validate.py                            18 errors (the policy) vs 4 advisories (its price)
data/processed/policy/                   C13 decisions; keyed by ROW, never joinable
  inspection_recommendations_<UTC>.parquet  the queue + the whole universe, 1,453,760 rows
  policy_selection_allocation_<UTC>.parquet offered / satisfied / granted, three numbers
  policy_comparison_<UTC>.parquet        what each policy costs, for EVERY model
  policy_model_selection_<UTC>.parquet   both tie-rule outcomes, side by side
  policy_override_log_<UTC>.parquet      the human layer; empty on most runs
docs/analysis/policy_findings.md
docs/data_contracts/policy_decisions.md
docs/interview/component_13.md
scripts/profile_policy.py                read-only; CAUGHT C13's central refutation
src/sentinel/fairness/                   Component 12: the group-behaviour audit
  definitions.py                         the group registry, incl. the REFUSED geographies
  groups.py                              the group frame; the temporal leakage surface
  support.py                             decided BEFORE any metric, and it shapes everything
  priority.py                            selection rate AND capture, never combined
  validate.py                            13 errors (the audit) vs 3 advisories (the world)
data/processed/fairness/                 C12 group metrics; keyed by GROUP, never joinable
  fairness_group_support_<UTC>.parquet   every observed group, supported or not
  fairness_priority_audit_<UTC>.parquet  who reaches the top k, and what it captured
  fairness_group_definitions_<UTC>.parquet  incl. why ward and census tract are refused
docs/analysis/fairness_findings.md
docs/data_contracts/fairness_audit.md
docs/interview/component_12.md
scripts/profile_fairness.py              read-only; FIXES C12's frozen constants
data/processed/explanations/             C11 attributions; the SHARPEST never-join
  explanation_values_<UTC>.parquet       648,000 rows; one per (model,fold,row,feature)
  explanation_cases_<UTC>.parquet        additivity + provenance + calibration link
  explanation_support_<UTC>.parquet      incl. the model that could NOT be explained
docs/analysis/baseline_models_findings.md
docs/analysis/boosting_models_findings.md
docs/analysis/explainability_findings.md
docs/data_contracts/baseline_predictions.md
docs/data_contracts/boosted_predictions.md
docs/data_contracts/explanations.md
docs/interview/component_7.md
docs/interview/component_11.md
scripts/profile_baselines.py             read-only, train windows only
scripts/profile_neural.py                read-only, train windows only
scripts/profile_boosting.py              read-only, train + calibration only
scripts/profile_explanations.py          read-only; FIXES C11's frozen constants
src/sentinel/ingest/socrata.py            the client. Most important file.
src/sentinel/ingest/food_inspections.py   orchestration
src/sentinel/ingest/manifest.py           provenance record
src/sentinel/query/duckdb_queries.py      NAMED_QUERIES live here
src/sentinel/config.py                    every tunable setting
data/raw/food_inspections/                output: parquet + manifest_*.json
docs/api/socrata_findings.md              verified API behaviour — read before
                                          touching the client
docs/data_contracts/food_inspections_raw.md   what a raw file guarantees
docs/decisions/                           35 ADRs

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
uv run sentinel schedule                      # Component 14 plan (~12 s, both modes)
uv run sentinel schedule --dry-run --report   # plan and validate, writing nothing
uv run sentinel schedule --capacity-mode observed_calendar   # the measured mode only
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
uv run python scripts/profile_explanations.py  # 9 read-only attribution profiles
uv run sentinel explain --report               # C11, ~19 min; NO OMP_NUM_THREADS override
uv run sentinel explain --models xgboost --sample-size 40 --dry-run --report
uv run python scripts/profile_fairness.py      # 10 read-only group-audit profiles
uv run sentinel audit-fairness --report        # C12, ~145 s; NO thread override needed
uv run sentinel audit-fairness --dry-run --report
uv run sentinel audit-fairness --models xgboost_platt --group-definitions community_area

uv run sentinel decide --report                # C13, ~39 s; NO thread override needed
uv run sentinel decide --dry-run --report      # decide and validate, writing nothing
uv run sentinel decide --policies coverage_forced_double_share --model lightgbm_platt
uv run sentinel decide --overrides overrides.json --report   # the human layer, audited
uv run python scripts/profile_policy.py        # the 8 pre-implementation profiles
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
uv run pytest                                 # 3,009 tests, offline
uv run pytest -m live                         # 3 live tests, hits the real API
uv run ruff check . && uv run ruff format --check .
uv run mypy src/sentinel scripts
```

One of `--dev`, `--limit`, `--full` is required — there is no default scope, so
a bare `sentinel ingest` cannot accidentally pull 314k rows.

---

## Naming conventions

* `scheduling/` artifacts are `<table>_<UTC>.parquet` with the manifest keyed to
  `inspection_schedule`. `schedule_config_id` is `<strategy>__<capacity_mode>`, and
  `planning_run_id` is `PR-<12 hex>` — a **content hash of the cell and configuration**, never a
  clock and never random, because a timestamped id would make two runs over identical inputs
  differ in a column.


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
    **Component 11 supplied the mechanism C6 could only infer.** Under the shift,
    three of four models leaned **2-3x harder** on that feature: xgboost's mean
    |SHAP| went 0.1499 -> 0.3728 and its rank 3 -> **1**; the network's 0.1232 ->
    0.3546, rank 8 -> **1**. The models did not merely *have* the feature during
    COVID, they reallocated their reasoning onto it. The logistic model is the
    instructive exception - a linear model cannot reallocate, so its top two
    features simply grew instead.
13. **Should model selection prefer the shift-robust model?** **CLOSED by Component 13 — as
    a POLICY decision, not a scientific one.** `xgboost_platt` is the production model, selected
    by a lexicographic rule frozen in advance. The measurement that made the closure possible is
    also the uncomfortable one: **axis 1 separated nothing.** Under Component 5's own
    1,000-replication label-flip study all four candidates' NDE intervals overlap, so the
    headline metric of this entire project cannot tell them apart, and the rule fell to
    calibration. ⚠ **The tie band decides the answer**: the discarded band (Component 8's
    five-seed ROC-AUC spread of 0.0058, applied to an NDE difference — a unit error) selects
    `neural_numeric_only_platt` instead. Both outcomes are emitted on every run and ADR 0039
    records the sequence, because the rule was fixed *after* its inputs were first read.
    **The scientific question — which model is actually best — remains unanswered, and Component
    13's measurement is that it may not be answerable on this data.** The history that made it
    hard, kept for context: C6 measured a clean inversion
    on `covid_shift`; C7 measured a *metric-dependent* ordering — lightgbm takes NDE and
    ROC-AUC, `logistic_regression` takes PR-AUC and precision@k. C9 added that the best
    uncalibrated model (`neural_numeric_only`, ECE 0.0563) has the *second-worst*
    calibrated ECE. **C11 removes the last easy tiebreak**: the four models score within
    0.0156 NDE and reason *differently* — importance rank rho is only **0.4351** between
    `logistic_regression` and `lightgbm`, sharing 3 of their top 10 features — so there is
    no "the models agree that X matters" consensus to appeal to, and preferring the model
    with the tidiest attribution table would be selecting on legibility. C11 is explicitly
    forbidden from settling this. A policy component must.
14. **Does the missing-indicator encoding need an interaction term?** ANSWERED, and the
    answer is "not much". A tree gets the interaction for free, and both boosters
    finished within 0.005 NDE of the logistic model. If a differing slope in the missing
    group mattered materially, the tree models should have found it.
15. **Is the C6→C7 improvement real?** UNRESOLVED. NDE +0.0050 on the quarterly mean,
    but logistic wins 7 of 17 folds and the gap sits inside the seasonality redraw band.
    The project has the redraw and the per-fold table but no paired significance test
    across folds; a bootstrap over establishments would answer it more directly.
17. **Is a policy's opportunity cost distinguishable from zero?** OPEN. Component 5's
    sensitivity machinery perturbs labels for NDE, but nothing equivalent was run over the
    policy comparison — so "is −34 citations a real difference" has no interval attached to it.
    A bootstrap over establishments within each fold would answer it.
18. **Is the coverage decline real, or is it volatility?** OPEN. The eligible share of the
    one-day queue runs 0.828 → 0.036 across 17 quarters but non-monotonically. Seventeen points
    cannot separate a trend from noise; a wider capacity or a pooled multi-quarter window would.
19. **Can anything reach the `__UNKNOWN__` group without a geographic rule?** OPEN, and probably
    not within this project. Component 13 measured that no policy in its grid changes that
    group's treatment at one day of capacity, and ADR 0038 refuses an allocation keyed to a
    failed geocode. The honest fix is upstream: better geocoding in Component 1 or 2.
20. **Is the override contract usable by a real reviewer?** UNKNOWN. It is tested against
    synthetic files only, and no operations room has ever seen it.
16. **What is the actual ceiling of this feature representation?** Two very different
    nonlinear learners and a penalised GLM land within 0.005 NDE. That is the strongest
    evidence the project has that the limit is the 26 features rather than the estimator
    — which makes the next high-value move a Component 4 change (311 complaints, weather,
    facility type, CDPH risk category, statutory days-overdue), not a fourth model.

---

## Lessons learned (the Sentinel API)

* **This is not a numbered component, and saying so up front prevented a naming collision.**
  The roadmap already names "Component 15" (routing, blocked) and "Component 16" (the
  deferral/human-review gate, since built — see below). The user explicitly decided this work
  should claim neither slot — it is cross-cutting infrastructure beside the numbered pipeline. Ask
  before assuming a new deliverable gets the "next" number in a strict roadmap.
* **The hardest design question was resolved by reading the actual call sites, not by guessing.**
  Whether a write endpoint could call `apply_adjustments`/`record_execution`/`replan` directly was
  answered by reading `scheduling/build.py::run_schedule` and finding those functions are only
  ever correctly invoked inside its per-cell batch loop, which recomputes several other tables and
  checksums inputs before/after. That's what justified "stage, never apply" (ADR 0049) — a
  conclusion reached from evidence, not from a general REST-API instinct.
* **Reuse `writer.SCHEMAS`/`finalize` for test fixtures, not hand-rolled dicts.** Building minimal
  Parquet fixtures directly through each component's own writer (as the project's other tests do)
  caught missing-column mistakes immediately and loudly, rather than producing a fixture that
  silently drifted from the real contract.
* **`**dict[str, object]` unpacking into a pydantic model fails mypy strict; `.model_validate(row)`
  does not.** `polars.DataFrame.row(0, named=True)` returns an `Any`-typed dict that mypy doesn't
  check, but an explicitly `dict[str, object]`-typed list (from a function with that return
  annotation) does trigger per-field argument-type errors on `**` unpacking. Prefer
  `Model.model_validate(row)` uniformly when constructing a pydantic model from a dict of unknown
  provenance.
* **FastAPI's `Depends(...)`-as-default is exactly the pattern ruff's B008 exists to flag.**
  Needed `[tool.ruff.lint.flake8-bugbear] extend-immutable-calls = ["fastapi.Depends", ...]`
  rather than disabling B008 project-wide.

---

## Lessons learned (Component 16)

* **A join key that is "unique enough" for the writer is not unique enough for a cross-table
  validator.** `target_inspection_id` is unique within one (policy, fold, k_name) cell of
  `inspection_recommendations`, but repeats across different policies' cells for the same
  establishment. The first version of `warning_trigger_rows_are_selected_and_warned` joined on
  `target_inspection_id` alone and produced 583,854 false-positive failures at full scale — every
  one of them a row matched against a *different policy's* `is_selected`/`warnings` value for the
  same establishment id. The fix was joining on the full cell key. The lesson: a spot-check
  against a handful of synthetic rows in a unit test will not surface a many-to-many join defect
  that only shows up once the same id legitimately recurs across a real-scale artifact — running
  the real CLI against real data caught it, the unit tests did not.
* **The same defect nearly repeated one function over, and the execution contract's own key
  saved it.** The execution-gap anti-join was tempted to key on `target_inspection_id` alone too.
  Checking Component 14's actual `EXECUTION_REQUIRED_FIELDS` first — `schedule_config_id,
  policy_id, fold_id, k_name, target_inspection_id`, deliberately *not* including `model_name` or
  `fold_set` — gave the correct five-column join key before the bug could be written at all.
  Read the upstream contract's own required-fields tuple before inventing a join key.
* **Guard against a reserved word with a literal substring check, not just enum-value
  disjointness.** Component 14's `ScheduleStatus.DEFERRED` and the natural English word for "send
  to a human" are one letter apart from colliding. Checking only that `ReviewResolutionAction`'s
  *values* don't overlap `AdjustmentAction`'s would not have caught a hypothetical
  `DEFERRED_FOR_REVIEW` status — it collides with nothing in the other enums and would still be
  the exact confusion the layer separation exists to prevent. The guard checks the substring
  `"defer"` directly, on every vocabulary value, as its own explicit assertion.
* **When three docstrings elsewhere gesture at "a threshold" for your component, resolve the
  tension in the ADR rather than picking a number or ignoring it.** `evaluation/metrics.py`,
  `calibration/definitions.py` and `evaluation/build.py` all said "a threshold is genuinely needed
  only by the Component 16 deferral gate." Read literally that could mean a real probability
  cutoff — which ADR 0040 forbids outright, since this project has never built a predictive
  interval. Flagging this as a genuinely open interpretive question to the user before
  implementing (rather than silently choosing) was the right call; the user's answer (no numeric
  threshold) is now on the record in ADR 0051 rather than assumed.

---

## Lessons learned (Component 14)

* **When a validator fails on your own new code, read the validator before the code.** Five
  error checks went red the first time the adjustment and execution paths ran end to end. Four
  were real defects — a sort key that stopped being a total order once a re-plan appended a
  second plan, a displaced row landing back in its own slot, a check comparing the whole backlog
  table against only planning run 0, and an identity that double-counted deferred rows. The
  fifth was the *check* being wrong: the temporal boundary as first written forbade moving a
  `not_performed` row whose day had passed, which is exactly the operation re-planning exists to
  perform. Component 7 learned to suspect the test first; this is that lesson one layer out.
* **A profiler can refute an invariant you were about to assert.** Profile 5 was run to
  establish "no establishment occupies two slots" and found it false — 1,573 establishment-fold
  pairs hold more than one scored canvass. Asserting it would have produced a red build on
  correct data, which is the failure that makes a suite stop being believed.
* **Ship the tautology, labelled, rather than hiding it.** `flat_median` is saturated by
  construction at two of five cutoffs. Deleting it would have been tidier; keeping it is what
  makes the observed calendar's shortfall legible as a *finding* rather than a bare number.
* **A promise in a comment is a debt.** `pyproject.toml` had said since Component 13 that a
  solver would arrive with Component 14. It did not, and the promise was discharged explicitly
  in ADR 0043 and in the comment itself rather than left quietly unredeemed.
* **Measure the previous component's premise, again.** Component 13 refuted the finding it was
  scoped around. Component 14 found that Component 13's central mechanism is substantially
  notional once a calendar is applied. Twice now, the profiler has been the most valuable hour.
* **Do not pool a scenario into a headline.** The reserve-loss figure was briefly 14.6% because
  the manifest averaged the observed calendar with a mode that loses nothing by construction.
  The real number is 29.3%. A scenario that cannot exhibit the effect must never be in the
  denominator.

## Lessons learned (Component 13)

* **Profile the premise, not only the parameters.** The profiler existed to fix constants — which
  column defines eligibility, how big the reserve should be. What it actually caught was that the
  intervention the entire component had been scoped around was aimed at a population the risk
  queue already over-serves fourfold. A component that had gone straight to implementation would
  have shipped a working, well-tested, thoroughly documented coverage reserve and never learned
  that the problem it solved does not exist here. **This is the single highest-value thing the
  investigate-first rule has produced in the project.**
* **Two words that sound identical are two different policies.** "Reserve 10% of capacity" means
  either *guarantee at least 10% goes to this population* or *spend 10% on rows the ranking
  passed over*. On this data the first is inert in 338 of 340 cells and the second gives up 34
  citations a week. Implementing only one would have hidden the whole result — and which one you
  would have picked depends entirely on which sentence you had in your head.
* **The tie rule is the decision.** A lexicographic selection rule looks like it is decided by
  axis 1. It is actually decided by *when you declare axis 1 a tie*, and two defensible bands
  picked two different models. The band was also initially borrowed from the wrong metric, which
  is the kind of unit error that survives review precisely because both numbers are "about 0.005".
* **Report the number that flatters you with the same caveat as the one that does not.**
  `coverage_forced_population_share` posts +2 citations *and* +31 eligible selections on the
  production model — a result that would headline nicely. It is inside the noise, three other
  models post negatives in the same cell, and the findings document says so.
* **A negative result needs more documentation than a positive one, not less.** "We built the
  mechanism and it does almost nothing" is only useful if the reader can see the measurement that
  makes the inertness meaningful. Hence the floor/forced split, the per-model robustness table,
  and the frontier that publishes the trade-off rather than resolving it.
* **`floor()` on a small k is where every early test failure lived.** Three red tests in a row
  were fixtures that had not thought hard enough about `int(0.20 * 4) == 0`. Component 7's rule
  held again: when a leakage test fails, suspect the test first.
* **Build the light comparison artifact, not the heavy one.** The first cut of the
  warnings-do-not-change-the-queue check materialised a second full 1.45M-row universe to compare
  two columns. `_queue_signature` keeps eight columns and proves the same thing.

## Lessons learned (Component 11)

* **When a leakage test fails, suspect the test first — third time it has paid.** A first
  draft shifted `frame.head(500)` by 900 days to make "future" rows; they landed back inside
  fold 0's own *training* window, so the test failed while nothing was wrong. Date the
  synthetic future from the end of the whole table, not by an offset from each row.
* **A tolerance measured on one fold is a tolerance measured from a sample of one.** The
  probe fold gave a tree additivity residual of 8.92e-07; the production run reached
  1.66e-06. The frozen 1e-5 (three orders above the probe) absorbed it. Had it been set at
  1e-6 the run would have failed on arithmetic that was entirely correct.
* **Ask what a passing check actually proves.** The permutation method's additivity residual
  is 0.0 at *every* round count, because the path telescopes. It was tempting to report that
  as evidence the values were accurate; it is evidence only that the sums are sound. The
  measurement that mattered was a convergence sweep against an independent-seed reference,
  and it showed the global ranking converging (rho 0.9964) long before any single value did
  (median error 1.00%).
* **The most dangerous defect in an attribution component leaves no trace.** Components 6 and
  7 order the same 30 columns differently at 19 of 30 positions. The wrong list produces
  arithmetically perfect values attached to the wrong features: no exception, no failed
  additivity check, every figure wrong. Measure the size of a trap before building next to
  it — the profiling script did that first, which is why the registry names the function
  per model instead of inferring it.
* **A "small" change to a closed component is still a change to a closed component.**
  `embed.booster_for` would have been four lines and would have made the fifth model
  explainable. It was written into an ADR and not added. The cost of not taking it is one
  function call, recoverable whenever Component 8 is next opened for a reason of its own.
* **Zero and null are different answers.** Zero is a legitimate attribution meaning "this
  feature did not move the score". A placeholder table of zeros for an unexplainable model
  would read as a model that used no features — worse than an empty table, because a reader
  cannot tell.
* **Check whether the library already does it before adding the library.** `shap` was going
  to be a runtime dependency until the profiling script established that xgboost and lightgbm
  each ship exact TreeSHAP and that linear SHAP has a closed form. It ended up dev-only, as a
  test oracle, agreeing to **0.0** — the same shape C5 used for its metrics.
* **A dev dependency can still break the runtime.** `shap` pulls numba, and without
  `numba>=0.67` / `llvmlite>=0.49` floors the resolver backtracks numpy off 2.5.2 — the
  version ADR 0026's bit-identity gate is baselined on. Downgrading the whole project to
  satisfy a test oracle would have been the tail wagging the dog.

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
