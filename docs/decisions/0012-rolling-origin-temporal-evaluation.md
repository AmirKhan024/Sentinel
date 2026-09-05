# ADR 0012 — Rolling-origin temporal evaluation, never random cross-validation

**Status:** Accepted · **Date:** 2026-08-16

## Context

ADR 0010 gave Component 4 a temporal boundary: a feature for the row at
`inspection_date = d` may use only records dated strictly before `d`. ADR 0011
then put the resulting table in `data/processed/` and closed with a warning aimed
directly at this component:

> Component 5 must not confuse "in processed" with "safe to split randomly". The
> table's temporal guarantee is about *feature construction*; honest evaluation
> additionally requires chronological splitting.

The two failures are genuinely different. A leaked feature makes one column
wrong, and usually shows up as an implausible value. A leaked *evaluation* leaves
every value plausible and every conclusion wrong.

The concrete failure is a random split. The feature table spans 2018-07-03 to
2026-08-14. `train_test_split` over those rows will cheerfully train on 2019,
2021, 2023 and 2025 and score on 2020 and 2022 — so the model learns how
establishments behaved *after* the period it is judged on. Sentinel can never
operate that way: at a real decision point in 2020, 2021 has not happened.

Two measured properties of this dataset make the stakes higher than usual.

**The base rate drifts hard.** The positive rate falls from 0.876 in 2018 H2 to
0.391 in 2026. A model that had seen the later, lower-prevalence years would
carry that information into a 2020 test window, and any pooled metric would
largely measure the drift rather than the model.

**Establishments recur.** The median canvass cycle is 358 days, and an
establishment appears once per quarter but many times across the eight years. A
random split therefore also puts the *same premises* on both sides of the line at
different dates, which leaks establishment-level information even where it does
not leak calendar-level information.

## Decision

**The primary evaluation is a rolling-origin backtest with an expanding training
window, quarterly cadence, and a calibration period sitting strictly between
training and test.**

```text
fold 1   train 2018-07-01..2021-12-31 | cal 2022Q1 | test 2022Q2
fold 2   train 2018-07-01..2022-03-31 | cal 2022Q2 | test 2022Q3
...
fold 17  train 2018-07-01..2026-03-31 | cal 2026Q1 | test 2026Q2
```

Four things are fixed by this decision.

**1. Chronological, never random.** `train_test_split`, `KFold` and
`StratifiedKFold` are forbidden for the primary evaluation. `FoldSpec` refuses to
construct a fold whose windows are not strictly ordered, so a leaky split is
unrepresentable rather than merely discouraged.

**2. Expanding, not sliding.** Training always starts at the code-era anchor
`2018-07-01` and grows forward. A regulator retrains on everything it has;
discarding 2019 to keep a fixed-width window would model a system nobody would
build. The cost is that early folds train on a period with a very different base
rate, which is reported per fold rather than smoothed away.

**3. Calibration is a separate, interposed window.** `TRAIN → CAL → TEST`, never
`TRAIN + TEST → calibration` and never `TRAIN → TEST → calibration`. Component 9
will fit Platt or isotonic scaling on the calibration window; fitting it on the
test period would make the reported probabilities self-fulfilling, and fitting it
on the training period would inherit the model's own overfitting. The window
exists now, empty of models, so that Component 9 has nowhere else to put it.

**4. Partial windows are excluded, never fabricated.** The snapshot ends
2026-08-14, so 2026Q3 exists as 328 rows over 32 days. Treating that as a quarter
would compare a two-thirds window against full ones. It is dropped, and the
exclusion is named in the manifest so a silent truncation cannot be mistaken for
full coverage.

The fold count is **derived from the data**, not hardcoded: 17 quarterly folds on
the current snapshot, and re-ingesting a later one adds folds without a code
change.

### Cadence

Quarterly, matching the project specification's stated fold structure. The first
calibration quarter is the 15th quarter after the anchor, which reproduces the
specification's Fold 1 exactly. Fourteen quarters of training before the first
fold is a judgement: enough history for a model to have something to learn, while
still leaving 17 test windows.

Quarterly windows contain roughly 1,600–2,500 inspections each, and an
establishment appears in a quarter essentially once (57,321 of 57,727
establishment-quarter groups hold a single row). So a test window is close to a
set of distinct premises, which is the right unit for a ranking.

### A second, non-rolling fold set

`covid_shift` is one fold — train 2018-07-01…2020-02-29, calibrate on 2020 Q1's
tail, test 2020-06-01…2021-12-31. It answers a different question from the
rolling set: not *is the ranking good on average* but *does it survive an
operational regime change*. It is kept in a separate `fold_set` so it can never
be averaged into the headline by accident.

## Alternatives rejected

**Random or stratified k-fold.** The standard choice, and wrong here for the
reasons above. Recorded explicitly because it is what a reader will assume was
used unless told otherwise.

**A single train/test split.** Simple, and invites the unanswerable question "how
do you know that isn't luck?". Seventeen folds give a mean and a standard
deviation, so fold-to-fold variance is visible rather than hidden.

**Sliding (fixed-width) training window.** Would hold the training base rate more
nearly constant, which is attractive given the drift. Rejected because it models
an operator who deliberately forgets, and because the drift is itself a finding
worth surfacing rather than engineering away.

**Semi-annual or annual windows.** Fewer, more stable folds, each covering more
of a full canvass cycle. Rejected as the primary because it would give 8 folds
rather than 17 and lose resolution on exactly the drift that matters here. The
cadence is a parameter, so this can be run as a robustness check later without a
contract change.

**Grouping by establishment instead of by time.** `GroupKFold` on
`establishment_id` would stop the same premises appearing on both sides. It does
not stop the *future* appearing on the training side, which is the failure that
matters, and it models a deployment that never happens — Sentinel scores
establishments it has already seen, repeatedly, forever.

**Fabricating a final partial fold to reach 18.** Rejected on principle. A
metric computed over a two-thirds window is not comparable to one computed over a
full window, and reporting them side by side would be worse than reporting one
fewer fold.

## Consequences

- Every metric is reported per fold and as mean ± SD, never as a single pooled
  number. The fold table is an auditable artifact, so a reader can see exactly
  which rows produced which result without reading code.
- **Base-rate-dependent metrics must be read beside their fold's prevalence.**
  PR-AUC, precision@k and first-half discovery all move with the base rate, which
  ranges 0.379 to 0.513 across test windows. ROC-AUC and normalized discovery
  efficiency are rank-based and therefore drift-robust, which is why they are the
  headline numbers for cross-fold aggregation.
- The calibration windows are currently unused — no model exists yet. That is
  deliberate: the structure is built before the thing that needs it, so Component
  9 cannot quietly calibrate on test.
- Component 6 onwards must accept the fold definitions as given. Changing the
  cadence, the anchor or the calibration placement is a contract change and needs
  its own ADR, because results computed under different fold sets are not
  comparable.
- **A model may only be scored on a fold if it declares a training horizon within
  that fold.** The prediction contract enforces it, so a Component 6 model
  trained through 2025 and pointed at the 2023 window is rejected at the door
  rather than quietly producing an excellent, meaningless number.
