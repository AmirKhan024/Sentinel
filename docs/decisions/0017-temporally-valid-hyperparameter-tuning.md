# ADR 0017 — Hyperparameter tuning happens strictly before the first test window

**Status:** Accepted · **Date:** 2026-08-17

## Context

Component 6 shipped with no hyperparameters, and `modeling/build.py` records that as a deliberate
omission rather than a blocked experiment: *"a baseline exists to be trustworthy rather than
optimal, and a tuning protocol belongs with the model that benefits from it."* STATUS.md names
Component 7 as the component that has to design that protocol.

A booster has roughly eight hyperparameters, and choosing them opens a leak with a property none
of the project's other leaks have: **it leaves no trace in any artifact.**

Every mechanical leak this repository defends against is detectable after the fact. A feature that
sees the future fails Component 4's checks. A fold whose windows overlap cannot be constructed —
`FoldSpec.__post_init__` refuses it. A model that trains past its horizon fails Component 5's
`future_rows_never_enter_training`. A prediction that claims a horizon it did not use is rejected
by `validate_predictions`.

But this loop —

```
fit → read a test metric → change max_depth → refit → read again → keep the better one
```

— produces an artifact that passes every one of those checks. The predictions cover the test
window exactly, the declared horizon is honest, no training row is misdated. The model is simply
better than it should be, by an amount nobody can measure, because the *human* saw the test data
and the model inherited what they learned. Component 5 protects evaluation time; it cannot protect
against a person.

`scripts/profile_baselines.py` already states the discipline for design choices generally:

> the only out-of-sample surface this script is allowed to touch is the **calibration** window —
> which exists precisely so design choices can be frozen before the test period is opened.

Hyperparameters are design choices. This ADR makes that discipline structural rather than
aspirational.

## Decision

### The search region is derived from the folds, and ends before the first test window

For a fold set, the tuning region is the **first** fold's `train_start .. calibration_end`.

For `quarterly` that is 2018-07-01 .. 2022-03-31; the first quarterly test window opens
2022-04-01. Because quarterly folds expand from a fixed anchor (ADR 0012), that span is a subset
of *every* later fold's own train-plus-calibration. So no fold is ever scored by parameters chosen
on its own test data, or on any other quarterly fold's.

The region is computed by `tuning.tuning_region` from the fold list, never written as a date
literal. A hardcoded region would silently stop being correct the moment the anchor, the cadence
or `MIN_TRAIN_QUARTERS` moved, and the failure would be invisible.

### Two studies, one per fold set

The `covid_shift` fold tests on 2020-06-01 .. 2021-12-31, which sits **inside** the quarterly
region. A single shared study would therefore have selected hyperparameters using that fold's own
test labels.

That would be the worst available trade. Both Component 5 and Component 6 measured the model
ordering *reversing* under the shift fold, and MEMORY.md's open question 13 records that the
project has no principled rule yet for which fold set governs a release decision. Biasing the one
number most likely to change a release decision, in the optimistic direction, to save one study,
is not a trade worth making.

So `covid_shift` gets its own study over its own region, 2018-07-01 .. 2020-05-31. Each fold set's
parameters are frozen separately and `tuned_params` **refuses** to borrow across fold sets — a
missing entry raises rather than falling back to a default, because quietly fitting library
defaults would produce a number indistinguishable from a tuned one.

The two studies did land in materially different places, which is itself informative: XGBoost's
shift parameters are shallower (depth 3 versus 4), use a quarter of the learning rate (0.056
versus 0.193) and nearly twice the rounds (192 versus 103).

### Inner folds are real folds, gap included

Each inner fold is a genuine `FoldSpec`: train, then an **unused calibration quarter**, then a
validation quarter. Reusing the type means `assign_split` and `window_frame` work on them
unchanged rather than needing a parallel implementation that could disagree.

The gap is not decoration. Every outer fold has one, so tuning without it would select parameters
for a zero-gap regime and apply them to a one-gap regime.

Measured: 6 inner folds for `quarterly` (validation quarters 2020Q4 … 2022Q1, from 8 training
quarters) and 2 for `covid_shift` (2019Q4 and 2020Q1, from 4). Two is thin and is reported as
thin. A study yielding fewer than two is **refused**, not run — one inner fold would make the
objective a single number from a single quarter, which is a preference rather than a measurement.

### The objective is Component 5's PR-AUC

`evaluation.metrics.pr_auc`, averaged across the inner validation windows. Component 5 owns that
definition and Component 7 does not get a second one. Two implementations of average precision
would eventually disagree, and the one used to *select* a model disagreeing with the one used to
*report* it is the subtlest way to make a tuning result meaningless.

### Early stopping exists only inside the objective

The number of boosting rounds is chosen by early stopping against an inner validation quarter —
which is training data for every outer fold. The winning trial's mean `best_iteration` is then
**frozen**, and `train.fit_fold` runs exactly that many rounds with no `eval_set` at all.

This is what lets a final fit declare `trained_through = fold.train_end` truthfully. Early
stopping at fold-fit time would need a window later than the training data, and the only such
windows are the fold's own calibration and test spans; reading either would make the declared
horizon false. `validate.final_fits_did_no_early_stopping` re-checks that no fit carries an
`early_stopping_rounds` or `eval_set` parameter, and `test_boosting_leakage.py` deletes the entire
calibration window and asserts the predictions do not move.

### The selected parameters are frozen as source literals

`tune-boosting` writes a trials table and **prints** the block to paste into
`boosting.definitions.TUNED_PARAMS`. It edits no source file.

The manual step is the design. A parameter set loaded from disk at training time could change
without a diff, and the entire value of freezing is that it cannot. `TUNED_PARAMS_PROVENANCE`
records the study artifact's sha256 so the frozen values can be traced back to the search that
produced them.

## Alternatives rejected

**Tune per outer fold, on each fold's own train-plus-calibration.** Attractive because it uses the
most data available at each decision point and is arguably what a health department would do. That
would mean 17 different parameter sets per model, 17 × 100 × 6 fits, and — decisively — the
resulting per-fold models would not be *the same model*, so a per-fold comparison against Component
6's single logistic specification would confound the estimator with the tuning. Rejected on
comparability, not cost.

**One shared study for both fold sets.** Rejected above: it would contaminate the shift result,
which is the number this project most needs to keep clean.

**Tune on the calibration windows of all 17 folds.** Attractive because it is a lot of data and
calibration windows are the designated design surface. Rejected because fold *N*'s calibration
quarter is fold *N−1*'s **test** quarter — the same rows. Selecting hyperparameters there and then
scoring on 2022Q1 would mean the parameters had seen those labels.

**Restrict the quarterly region to pre-2020-06 so one study serves both.** Clean for both fold
sets, and it was seriously considered. Rejected because it leaves only eight quarters, all
pre-COVID, at a base rate near 0.82 against test-window rates of 0.38–0.51. Parameters selected on
a regime that unlike every quarterly test window is a worse defect than running a second study.

**Random or grid search instead of a sampler.** See ADR 0016.

**Fewer than 100 trials.** The specification calls for 100+. The measured cost is 563.8s for all
four studies, so there was no computational reason to reduce it and none was taken. Unit tests use
a 3-trial configuration and say so explicitly; the distinction between a development search and the
production search is stated in `tuning`'s docstring, in `cli.DEFAULT_TRIALS`, and in the findings
document.

**Selecting the model on the final test score.** The thing this entire ADR exists to prevent.
Measured consequence: on the quarterly mean XGBoost wins, but per fold the logistic model wins 7 of
17. Had selection been done on test performance the reported margin would be larger and it would
mean nothing.

## Consequences

- Two parameter sets per model, four studies, 400 trials, 563.8s. All four regions end strictly
  before their fold set's first test start, re-derived from the fold definitions by
  `validate.tuning_never_reached_a_test_window` on every run.
- **A new fold set requires a new study and a new `MIN_INNER_TRAIN_QUARTERS` entry.** There is no
  default; `build_inner_folds` refuses a fold set it has no declared inner training length for,
  because a default would be an undocumented design choice.
- `covid_shift`'s two inner folds are a thin basis for an eight-dimensional search. Its parameters
  should be read as less well determined than the quarterly ones, and its results are a
  robustness observation rather than a selection criterion.
- The frozen parameters are literals under version control. Re-running `tune-boosting` does not
  change them; a human must paste and commit, which is what makes the change appear in a diff.
- Component 9's calibrator will fit on the calibration window, which this component leaves
  untouched precisely so it can.
