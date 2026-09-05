# Component 7 — interview defence

Every answer below is grounded in something this repository actually measured. Where a number
appears it comes from `docs/analysis/boosting_models_findings.md`, which was produced by the
commands in its header. Where a claim is uncertain the answer says so.

---

## 60-second answer

Component 6 gave us a regularised logistic baseline that beat the best heuristic on all 17
quarterly folds. Component 7 asks whether a nonlinear model does better on the same 26 features and
the same evaluation. I tuned XGBoost and LightGBM with Optuna — 100 trials per model per fold set,
400 trials total — under a protocol where every hyperparameter is selected from data strictly
earlier than any test window. Component 5 remained the only evaluator; Component 7 just writes a
prediction artifact.

The answer is: **yes, but barely, and the margin does not survive scrutiny.** XGBoost improves
normalised discovery efficiency from 0.2326 to 0.2376 on the quarterly mean — about 2% relative.
But the logistic model wins the plurality of individual folds, 7 of 17. And the whole gap sits
inside the interval Component 5's seasonality sensitivity produces by re-drawing labels. So the
honest conclusion is that the simpler model remains defensible, and the ceiling here is the feature
representation rather than the estimator.

---

## 2-minute answer

The starting point was three open items from Component 6: thin margins on several folds, an
ordering that reversed under the COVID distribution shift, and 43.24% of violations still surfaced
later than under business-as-usual.

Boosting was the right next estimator for reasons specific to this data. The features are
collinear — condition number 71.8, one pair correlated at 0.9888 — which a linear model handles by
splitting weight between terms, whereas a tree conditions on one and splits on the other. And ten
of 26 features are nullable, where the NULL is a *fact*: `prior_canvass_priority_rate` is null
because there was no prior canvass. Component 6 had to fill that with a training-window median,
replacing a fact with the average of a population the establishment isn't in. A booster routes NaN
natively, so the fact survives. Nearly 3.4 million NaN cells reached the estimators.

I used both libraries because the comparison between them is the question, not an implementation
detail — XGBoost grows depth-wise, LightGBM leaf-wise, so a single result would be a fact about one
library's bias.

The hard part was tuning. Component 6 had no hyperparameters and so needed no protocol. A booster
has eight, and choosing them opens a leak nothing downstream can detect: fit, read a test metric,
adjust, refit. Every artifact still passes every check; the model is just better than it should be.
So the region a study may read is derived from the fold definitions and ends before that fold set's
first test window, and there are **two** studies — because the `covid_shift` test window sits
*inside* the quarterly tuning region, and one shared study would have contaminated the one number
this project most needs kept clean.

Results: XGBoost 0.2376 NDE, LightGBM 0.2355, logistic 0.2326. Per fold, logistic wins 7, XGBoost
5, LightGBM 5. The seasonality redraw puts XGBoost's interval at [0.2224, 0.2444] — the logistic
model's observed value sits comfortably inside it. So I report a small improvement that I would not
defend as decisive, and I say so.

I also had to block a specified feature: inspector-effect modelling. The dataset publishes 22
columns and none identifies an inspector, so a random intercept over inspectors and any
marginalisation over them are undefined. I documented that with evidence and a regression test
rather than substituting a proxy.

---

## Deep technical answer

### The estimand has not changed

Component 5 owns it and Component 7 inherits it verbatim: *under a fixed, observed daily inspection
capacity, how much earlier would violations have surfaced if the same inspections had been
performed in a different order?* Labels are held fixed. The simulation is retrodictive. Only
establishments actually inspected are evaluated. The target is "a Priority or Priority Foundation
violation was cited", not "the establishment was unsafe".

### Architecture

`src/sentinel/boosting/` is a sibling of `modeling/`, importing from it rather than forking it.
`training_frame` comes from Component 6, which delegates to `evaluation.folds.assign_split` — so
there is exactly one definition of "the training window" in the repository, the one Component 5's
`future_rows_never_enter_training` check independently re-derives. The matrix builder wraps
Component 6's `to_matrix`, which guarantees the boosted and baseline matrices have identical
columns in identical order. The tuning objective calls `evaluation.metrics.pr_auc` — Component 5's
implementation, not a second one, because the metric used to *select* a model disagreeing with the
one used to *report* it is the subtlest way to make a result meaningless.

A separate `BOOSTING_REGISTRY` and a separate `boosted_predictions` slug keep Component 6's
artifact byte-identical. I verified that: re-running `train-baselines` under the current library
set reproduces sha256 `a2bb9411…00ff5b44`, matching the committed manifest exactly.

### Preprocessing is deliberately empty

No imputation, no scaling, no fitted statistic of any kind. That removes Component 6's
`preprocessing_comes_from_train` check, so I replaced it with a stronger one: recompute the NULL
mask from the source frame and assert it equals the NaN mask of the matrix the estimator received,
cell for cell. Plus a companion check asserting the NaN count is non-zero, so the mask comparison
cannot pass vacuously on an all-present frame.

I kept the four null-rule family indicators even though a NaN-native learner doesn't need them.
Dropping them would mean the two components' matrices differ, and every comparison would be
ambiguous between "the estimator is better" and "the matrix is different".

### The tuning protocol, precisely

- Region per fold set = the **first** fold's `train_start..calibration_end`. Because quarterly folds
  expand from a fixed anchor, that span is a subset of every later fold's own train-plus-calibration.
- `quarterly`: 2018-07-01..2022-03-31 against a first test start of 2022-04-01.
  `covid_shift`: 2018-07-01..2020-05-31 against 2020-06-01.
- Inner folds are real `FoldSpec` objects — train, an **unused** calibration quarter, then a
  validation quarter — so `assign_split` and `window_frame` work on them unchanged. The gap is
  there because every outer fold has one; tuning without it would select parameters for a zero-gap
  regime and apply them to a one-gap regime. Six inner folds for quarterly, two for covid_shift.
- Objective: mean PR-AUC across inner validation windows. `TPESampler(seed=20260817)`.
- **Early stopping exists only inside the objective.** The winning trial's mean `best_iteration` is
  frozen, and the final fit runs exactly that many rounds with no eval set. That is what lets
  `trained_through = fold.train_end` be literally true rather than nearly true.
- Selected parameters are frozen as **source literals** in `definitions.py`, not loaded from disk.
  A file-loaded parameter set could change without a diff.

### Determinism

The profiler measured something I did not expect: fitting the same 53,844 rows in a shuffled order
moves a prediction by **0.11**. Component 6's coefficients moved by 7.049e-09 — seven orders of
magnitude less — because that was float-summation order, whereas a booster draws row and column
subsamples in row order. Re-sorting a shuffled frame restores the fit **exactly**, and `fit_fold`
re-sorts unconditionally rather than trusting the caller. Both estimators are pinned to one thread,
because a multi-threaded histogram reduction is only approximately reproducible and this project's
standard for "unchanged" is bit-identical.

### The result, stated carefully

Quarterly mean over 17 folds: XGBoost ROC-AUC 0.6188 / PR-AUC 0.5343 / NDE 0.2376 / +5.83 days;
logistic 0.6163 / 0.5321 / 0.2326 / +5.70. The `xgboost_class_weighted` ablation actually posts the
best NDE (0.2390) but degrades ECE by 35% (0.0836 vs 0.0621), so it is not adopted.

Per fold, logistic wins 7 of 17. The tree models win by more when they win (+0.047 at 2024Q2) than
they lose when they lose (−0.017 at 2025Q4), which is what produces the positive mean. And the
seasonality redraw gives XGBoost [0.2224, 0.2444], which contains the logistic model's 0.2326.

Under `covid_shift` the ordering is **metric-dependent**: LightGBM takes NDE and ROC-AUC, the
logistic model takes PR-AUC (0.6328, highest of any model) and precision@k (0.9545, by a wide
margin). That is Component 6's inversion finding in a different shape.

---

## Top 15 likely questions

**1. Why move from logistic regression to gradient boosting at all?**
Two data-specific reasons, not fashion. The features are collinear (condition number 71.8, one pair
at 0.9888), which a GLM handles by splitting weight while a tree conditions and splits. And ten of
26 features are nullable where the NULL carries information — "no prior canvass" is a fact, and
Component 6 had to overwrite it with a median. A booster branches on it instead. MEMORY.md's open
question 14 asked whether the missing-indicator encoding needs interaction terms; a tree answers
that for free.

**2. Why test both XGBoost and LightGBM rather than picking one?**
Because the comparison is the question. They grow differently — depth-wise versus leaf-wise — so a
single result would be a fact about one library's inductive bias. They finished 0.0021 apart in NDE
after 100 tuned trials each, and *that agreement is the finding*: two very different nonlinear
learners landing in the same place, 0.005 above a penalised GLM, is evidence that the ceiling is
the 26-feature representation rather than the estimator.

**3. This is a classifier. Why do you keep calling it a ranking problem?**
Because the operational decision is an ordering under fixed capacity, not a yes/no. Chicago
performs a measured median of 22–45 inspections a day whatever the model says; the model chooses
*which* ones go first. So the primary metric is normalised discovery efficiency — where the
discovery curve sits between a random schedule and a perfect one — not accuracy at a threshold. We
never pick a threshold; Component 9 owns that.

**4. Why can't you tune on the final test fold?**
Because that loop leaves no trace. Fit, read a test metric, change `max_depth`, refit, keep the
better one — the resulting artifact passes every check the repository has. Predictions cover the
test window exactly, the declared horizon is honest, no training row is misdated. The model is just
better than it should be, by an amount nobody can measure, because the *human* saw the test data.
Every other leak in this project is mechanically detectable; this one is only preventable.

**5. How did you make Optuna temporally valid?**
The region a study may read is derived from the fold definitions — the first fold's
`train_start..calibration_end` — and a check re-derives both that date and the fold set's first
test start on every run and asserts one precedes the other. Inner folds are real rolling-origin
folds carved inside the region, keeping the outer structure's unused calibration gap. And a test
collects every row id the objective can touch and intersects it with every test-window id, asserting
the intersection is empty — dates could be right while a join was wrong.

**6. Why two studies instead of one?**
The `covid_shift` fold tests on 2020-06-01 to 2021-12-31, which sits *inside* the quarterly tuning
region. One shared study would have picked hyperparameters using that fold's own test labels. Both
Component 5 and Component 6 measured the model ordering reversing on that fold, so it is the number
most likely to change a release decision — biasing it optimistically to save one study would be the
worst available trade. The two studies also landed in genuinely different places: shift XGBoost
chose depth 3 and learning rate 0.056, quarterly chose depth 4 and 0.193.

**7. How did you do early stopping without leaking?**
Early stopping only runs inside the tuning objective, against inner validation quarters that are
training data for every outer fold. The winning trial's mean `best_iteration` becomes a frozen
`n_estimators`, and the final per-fold fit runs exactly that many rounds with no eval set. A
validation check asserts no fit carries `early_stopping_rounds`; the training log records
`early_stopped = false` on every row; and a leakage test deletes the whole calibration window and
asserts the predictions are bit-identical.

**8. Why didn't you use SMOTE?**
Measured prevalence is 52.52%. There is no imbalance to correct. Resampling would distort the
probability scale Component 9 has to calibrate, and synthetic minority rows would break the temporal
structure the entire project rests on. I did run class weighting as an explicit named ablation, and
it makes the point: it buys NDE +0.0014 and costs ECE 0.0621 → 0.0836. Adopted, it would be tuning
until something wins.

**9. Why no feature standardisation?**
A tree splits on thresholds, so it's invariant to any monotone transform. Standardising would change
nothing about the fitted model and would add a fitted statistic that has to be carried, validated
and explained. More importantly it would add a place where a statistic could be computed from the
wrong window.

**10. What exactly is the inspector-effect problem, and what did you do about it?**
The observed label is `f(establishment risk, inspector strictness, time effects)`. An establishment
doesn't choose its inspector, so if inspector identity becomes an establishment feature, a
restaurant that drew a strict inspector carries a permanently higher score for something it can't
control. The specification asks for inspector as a random intercept, controlled for during fitting
and marginalised away at inference. **I could not implement it: the dataset publishes 22 columns and
none identifies an inspector.** A random intercept over an unobserved grouping has no likelihood to
maximise; a marginalisation over an effect nobody estimated is arithmetic on a made-up number. I
wrote ADR 0019 with the column inventory, and a regression test that re-derives the absence from the
raw contract and fails if such a column ever appears.

**11. Why not just use a proxy — ward, day of week, violation-text verbosity?**
Each is genuinely correlated with something. None identifies a person, which is what a per-inspector
effect requires. Ward is a route proxy fully confounded with establishment composition —
neighbourhoods differ in cuisine, chain penetration, building age — so marginalising it away would
remove real establishment risk along with the supposed nuisance. Verbosity is unattributable: it
tells you write-up style varies, never whose. Day-of-week is already confounded with weather,
holidays and staffing. A model labelled "inspector-adjusted" that adjusted for ward is the most
misleading artifact this project could ship, because it survives being quoted.

**12. How do you know XGBoost is actually better than logistic regression?**
I'd say I *don't* know that, and the document says so. It's better on the quarterly mean by 0.0050
NDE. It loses 7 of 17 individual folds. And the logistic model's observed value sits inside
XGBoost's seasonality redraw interval [0.2224, 0.2444]. Compare that to Component 6's own claim: the
heuristic's 0.1845 sits well *below* the logistic model's interval, so that improvement survives the
same test. Component 7's does not clearly.

**13. What if it wins on ROC-AUC but loses on NDE?**
Then it loses, because NDE is the operational quantity. ROC-AUC asks whether a random positive
outranks a random negative anywhere in the ordering; NDE asks how much earlier violations surface
under a fixed daily capacity. A model can improve the tail of the ranking — irrelevant, because
those establishments get inspected late either way — while doing nothing at the top where the
capacity actually binds. On `covid_shift` this exact divergence happened: LightGBM took ROC-AUC and
NDE while the logistic model took PR-AUC and precision@k.

**14. What happens under the COVID shift?**
The ordering becomes metric-dependent rather than simply reversing. LightGBM takes NDE (0.2585 vs
0.2512) and ROC-AUC; the logistic model takes PR-AUC (0.6328, the highest of any model) and
precision@k_1day (0.9545, by a wide margin). The caveats matter: one fold, `k_1_day` is 22 slots so
precision@k is extremely noisy, and days-earlier has SD 208 against a mean of 26. It answers "does
the ranking survive a regime change", not "which model is best" — and it is never averaged into the
headline.

**15. Can you claim causal impact?**
No. The simulation is retrodictive: it re-orders inspections that actually happened, holding the
observed label fixed. It does not show a violation would have been found had the inspector arrived
earlier — the establishment's state on the earlier date is unobserved. The correct phrasing is
"would have surfaced earlier under the retrospective ranking simulation", never "prevented
violations". And 42.89% of violations still surface *later* under the XGBoost ranking, which is the
number that has to travel with any days-earlier claim.

---

## "Why did you choose this?"

**Why NDE as the primary metric?** Because capacity is the binding constraint and it's measured, not
chosen — median 22–45 inspections a day. NDE normalises the discovery curve between a random
schedule and a perfect one, so it's comparable across folds with different prevalence, which
raw precision@k is not.

**Why a separate registry and artifact slug rather than adding to `MODEL_REGISTRY`?**
HANDOFF.md suggested appending. I didn't, because that would change the default output of
`train-baselines` and mix C6 and C7 rows in one file. The brief's requirement that Component 6's
benchmark stay visible is only satisfiable if its artifact is untouched — and I verified it
reproduces byte-identically.

**Why keep the four missing-indicators when the model doesn't need them?** So the two components'
matrices are identical. Otherwise any measured difference is ambiguous between the estimator and the
inputs, and the whole comparison is worthless.

**Why freeze parameters as source literals rather than reading the trials file?** Because a
file-loaded parameter set can change without a diff, and freezing is only meaningful if it can't.
The manual paste step is the design.

**Why does Component 5 remain the evaluator?** One definition of every metric, one definition of the
folds, and — decisively — the test window stays out of reach of the component that is allowed to fit
things. Two sets of numbers would mean two answers to every question and no way to say which is
authoritative.

**Why is calibration deferred to Component 9?** Because a calibrator is a fitted model and fitting
it requires the calibration window, which would push `trained_through` from `train_end` to
`calibration_end`. Keeping them separate is what lets Component 7 declare the earlier horizon
truthfully. I report ECE and MCE and correct neither.

---

## "Why didn't you choose X instead?"

**CatBoost?** Component 4's contract has no categorical features — all 26 are counts, day-deltas,
rates and booleans — so its distinguishing capability is inert. A third dependency for nothing
measurable.

**sklearn's `HistGradientBoostingClassifier`?** Already a dependency, NaN-native, zero new packages.
It would have answered "does *a* boosted model beat the GLM" while forfeiting the cross-library
agreement that makes the answer credible. It's the honest fallback if the dependency budget shrinks.

**Random or grid search?** A grid over eight dimensions is hopeless at useful resolution. Random
search is defensible and free; I preferred a seeded sampler because reproducibility then belongs to
the sampler rather than to every call site threading a seed correctly, and because Optuna's
structure made the auditable trials table natural.

**MLflow?** The specification mentions it. The repo already has a working provenance mechanism — a
manifest per run, pinned by sha256, with library versions and input checksum — and every property
tracking was wanted for is in the trials table. A tracking server plus backend store to re-express
recorded information is architectural expansion for its own sake. Worth revisiting if searches ever
span machines or people.

**Ensembling XGBoost with LightGBM?** They agree to 0.0021 NDE. Blending two models that agree that
closely buys nothing, and blending before each member is understood hides which one carries the
result. Ensembling comes after the individual model stages.

**Tuning per outer fold?** It would use more data at each decision point and is arguably what a
health department would do. But 17 different parameter sets means the per-fold models aren't *the
same model*, so a comparison against Component 6's single specification would confound estimator
with tuning. Rejected on comparability, not cost.

**Multi-threaded fits?** Would cut the search from 9.4 minutes to about two. Rejected because
Component 6 set bit-identity as the standard for "unchanged" and every leakage test asserts it.
Downgrading to `allclose` would mean a real leak producing a 1e-9 shift becomes indistinguishable
from thread scheduling.

---

## "What went wrong?"

**My first leakage test was wrong, and the failure was instructive.** I mutated every feature dated
after `train_end` and asserted the predictions didn't move. They moved. For about a minute that
looked like a leak — it was the test: rows between `train_end` and `test_end` *include the fold's
own test rows*, and changing a test row's features is supposed to change its score. The fix was to
mutate strictly after `test_end`. Worth mentioning because a test that fails for the wrong reason is
as dangerous as one that passes for the wrong reason.

**A second test bug was more embarrassing and more dangerous.** I built "future rows" from override
dicts and concatenated them positionally, but the dicts are ordered differently from Component 4's
schema — so every value landed in the wrong column, and the resulting fit moved. That looks exactly
like a leak. The fixture `model_feature_scenario` exists precisely to order rows against the schema;
I wasn't using it. Both bugs argue for the same discipline: when a leakage test fails, suspect the
test first.

**One guard I wrote could not fail.** The check that a tuning region doesn't overlap a test window
was unreachable through the public path, because `tuning_region` and `first_test_start` derive from
the same fold and a fold's calibration end always precedes its own test start. The repo has a
precedent for exactly this defect — Component 5 shipped `scores_respect_the_decision_point`
declared and unreachable, fixed in ADR 0014. I kept the guard, because the realistic way to break it
is widening the region ("use all the data before the last fold"), and wrote the test to drive it
that way via monkeypatch — with a docstring explaining why.

**A pre-registered expectation failed.** HANDOFF.md warned boosted probabilities are usually worse
calibrated. I ran a calibration-window probe before training to pre-register that, and on the
quarterly folds it was *false* — XGBoost's ECE came out marginally better than the logistic model's.
Reported rather than dropped. It held under shift.

**The result is weaker than the effort.** Four hundred tuning trials and two libraries bought about
2% relative NDE, inside the seasonality band. That's a real outcome, not a failure, but it's not the
outcome the work looks like it should produce.

---

## "What would you improve?"

**The comparison needs a proper significance treatment.** I have the seasonality redraw and the
per-fold table, and together they say the improvement is not clearly real — but a paired test across
the 17 folds, or a bootstrap over establishments, would say it more precisely. The redraw varies
labels under a seasonal model; it doesn't directly answer "is this difference distinguishable from
sampling noise".

**`covid_shift`'s two inner folds are too thin** for an eight-dimensional search. Either accept
weaker parameter determination and say so more loudly, or design a shift-specific protocol that
borrows structure without borrowing test data.

**Single-threading won't scale.** It's the right trade at 57,727 rows. At ten times that, someone has
to choose between bit-identity and tractable runtime, and this ADR should be revisited rather than
quietly relaxed.

**The class-weighting ablation deserves a fuller sweep.** I tested one weight — the training
window's own `(1−p)/p`. A small grid would say whether the NDE/ECE trade is monotone or whether
there's a weight that buys ranking without much calibration cost. That's Component 9's territory,
arguably.

**The strongest next move is not a better estimator.** Two very different nonlinear learners and a
penalised GLM land within 0.005 NDE of each other. That points at the representation. The features
that would plausibly matter — nearby 311 complaints, weather, facility type, the CDPH risk category,
statutory days-overdue — are either not ingested or sit in the raw snapshot but not in Component 4's
table. Every one of those is a Component 4 change behind a bumped `feature_definition_version`, and
I'd expect more from any of them than from a fourth estimator.

---

## Five-line memory cheat sheet

1. **The question:** can a nonlinear model beat the C6 logistic baseline on the same 26 features?
   Answer: XGBoost NDE 0.2376 vs 0.2326, **+2.1% relative** — real but small.
2. **The catch:** logistic wins **7 of 17 folds**, and its 0.2326 sits *inside* XGBoost's seasonality
   redraw interval [0.2224, 0.2444]. C6's gain over heuristics survives that test; C7's over C6 does not.
3. **The protocol:** 400 Optuna trials, **two studies** (one per fold set) because covid_shift's test
   window sits inside the quarterly tuning region; early stopping only inside the objective, so
   `trained_through = train_end` stays literally true.
4. **The blocked thing:** inspector-effect modelling — the dataset has **22 columns and no inspector**.
   Documented with evidence and a regression test; no proxy substituted (ADR 0019).
5. **The carry-forward:** 42.89% of violations still found later; prevalence 52.52%; PR-AUC floor
   0.4307 not 0.5; probabilities raw; simulation retrodictive. **Two boosters and a GLM within 0.005
   NDE means the ceiling is the representation, not the estimator.**
