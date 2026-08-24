# ADR 0025 — The calibrator selection protocol, pre-registered

**Status:** Accepted · **Date:** 2026-08-24

## Context

Component 9 fits two calibrators — Platt scaling and isotonic regression — and must choose between
them. Both are implemented and both are reported, because they carry different assumptions; but one
of them has to be applied to the test window, and something has to decide which.

The obvious way to decide is the forbidden one. Fitting both, scoring both on the test window and
keeping the lower ECE is test-set selection. It would produce the best-looking number in the
project and it would be meaningless, because the reported test metric would then be the maximum of
two draws rather than an estimate of either.

This ADR is written **before the first production run**, and its purpose is to make that claim
checkable rather than asserted. The constants it freezes come from
`docs/analysis/calibration_findings.md` §3, §6 and §9 — a profiling pass that touched only
calibration windows.

Two structural facts constrain any answer.

**Calibration windows are small.** 1,357–2,459 rows, against training windows of 23,346–53,844.
Whatever holds out data for a selection has very little to hold out.

**Fold *N*'s calibration window is fold *N−1*'s test window.** `quarterly-2022Q3` calibrates on
2022-04-01…06-30, which is exactly `quarterly-2022Q2`'s test period. ADR 0017 already used this to
reject tuning across all 17 folds. It is the trap this ADR exists to avoid.

## Decision

### 1. The inner split: chronological, whole-day, 70/30

Each fold's calibration window is cut into an **inner-fit** portion and an **inner-select** portion
by date. Calibrators are fitted on the inner-fit portion and compared on the inner-select portion.

```
INNER_SELECT_FRACTION = 0.30
MIN_INNER_FIT_ROWS    = 400
MIN_INNER_SELECT_ROWS = 250
```

The cut reuses `neural.train.inner_split_date`, which walks distinct dates backwards accumulating
rows rather than taking a row quantile. Component 8's reason applies unchanged: two inspections of
the same establishment days apart share almost all of their as-of history, so a mid-day cut splits
rows that are not independent.

0.30 rather than Component 8's 0.15 because a calibration window is an order of magnitude smaller
than a training window; at 0.15 the smallest inner-select portion would be about 204 rows. Measured
on this snapshot, 0.30 gives 409–756 select rows and 948–1,703 fit rows, and every fold clears both
minimums.

A fold that fails a minimum is **refused, not calibrated on a window too small to mean anything** —
the posture `neural.train.split_training_window` already takes. None do today; the guard exists so
that a future snapshot surfaces as a failure rather than a silent method switch.

### 2. The selection metric is mean inner-select log-loss

Not ECE, although ECE is the metric this component exists to improve. Three reasons, all fixed in
advance:

1. **Resolution.** 15 equal-mass bins over a 409–756-row inner-select window is 27–50 rows per bin,
   of which 11–21 are positives. The sampling noise in a bin's observed rate is comparable to the
   effect being measured.
2. **ECE is not a proper scoring rule.** A calibrator that reshuffles probabilities within a bin
   lowers ECE without being better, and a degenerate calibrator predicting the base rate for every
   row scores near-zero ECE while carrying no information.
3. **Log-loss has no free parameter.** A selection made on ECE could be changed by changing the bin
   count, and a rule that can be tuned is not a rule.

Log-loss also punishes exactly the failure mode isotonic exhibits here — a confident wrong plateau
— which is the failure a probability consumed by a cost threshold must not have.

ECE, MCE and Brier are recorded on the inner-select window in `calibrator_selection_*.parquet` as
diagnostics. **They decide nothing.**

### 3. Selection granularity: expanding prefix, per (model, fold_set)

For fold *k* in a fold set, ordered by `calibration_end`, the method is the one with the lower
**mean inner-select log-loss over folds 1…k**. Fold 1 falls back to its own inner-select result,
which is the honest degenerate case.

Every input to fold *k*'s decision has `rd ≤ fold_k.calibration_end`, so it is horizon-legal under
the rule `evaluation/contract.py` already enforces — `_training_horizon` returns
`fold.calibration_end`, and the contract rejects only a declared horizon *past* it. It also mirrors
the expanding training window the whole project uses.

`covid_shift` is its own fold set and never mixes with `quarterly`, the same separation ADR 0017
forces on tuning.

**The per-fold winner is logged beside the prefix winner** in `calibrator_selection_*.parquet`, so
method instability is a measured result and the alternative design is auditable without a re-run.

### 4. The tie rule, frozen with a date

```
TIE_THRESHOLD  = 0.005 nats
TIE_PREFERENCE = platt

choose isotonic  iff  mean_ll(isotonic) < mean_ll(platt) − TIE_THRESHOLD
otherwise        platt
```

Note the asymmetry: Platt wins ties, and also wins when it is merely not-worse-by-enough.

**Where 0.005 comes from.** `scripts/profile_calibration.py` resamples each inner-select window
1,000 times, scoring both calibrators on the *same* resample so their shared variation cancels. The
SD of that paired gap is the resolution of the comparison. Measured over 72 (model, fold) cells:
min 0.0022, **median 0.0054**, max 0.1595. The threshold is one median paired-gap SD.

The implementation plan proposed 0.002 before this was measured. That value is **below the smallest
observed paired-gap SD**, so it would have declared winners on differences finer than the noise of
the comparison. The correction is recorded in `calibration_findings.md` §6.

The threshold is deliberately conservative for later folds: the prefix mean over *k* folds has an
SD shrinking roughly as 1/√k, so a fixed 0.005 becomes a stricter bar as *k* grows. Being
conservative in favour of the simpler method is the declared preference.

**Why Platt is the preference**, independent of any result:

1. Two parameters against a step function with up to ~2,300 breakpoints, fitted on ~1,200 rows.
2. **Platt is strictly monotone; isotonic is only weakly monotone.** Isotonic's plateaus create
   ties, and `evaluation.metrics.top_k_indices` breaks ties by `target_inspection_id` ascending, so
   a plateau can move top-k membership without the calibrator being non-monotone. "Do not re-rank"
   is satisfied exactly by Platt and only approximately by isotonic.
3. Isotonic requires `out_of_bounds="clip"` and so has a hard floor and ceiling at the calibration
   window's observed extremes. Platt extrapolates smoothly.

### 5. Refit on the full window, then freeze

The method chosen on the inner-select portion is refitted on the **entire** calibration window —
the selection consumed the split, the production calibrator should not. That calibrator is frozen
and applied to the test window. Nothing after this point reads a test label.

The ordering, in full:

```
base model fitted on train
    → calibrator candidates fitted on inner-fit
    → compared on inner-select
    → method chosen on the expanding prefix
    → chosen method refitted on the full calibration window
    → frozen
    → applied to test
    → test evaluated, once
```

### 6. What the code must not allow

- `SELECTION_METRIC`, `TIE_THRESHOLD` and `TIE_PREFERENCE` are literals in
  `calibration/definitions.py`. An error-severity check asserts the manifest's copies match those
  literals, so a run cannot report a rule it did not use.
- `--method platt|isotonic` exists on the CLI as a **diagnostic override** and is recorded in the
  manifest when used. The production run does not pass it.
- Both calibrators are fitted and both are written for every fold, including where isotonic lost,
  so the counterfactual is answerable from the artifact rather than by re-running with a flag —
  which is how a selection quietly becomes a test-set selection.

## Alternatives rejected

**Choose on the test window's ECE.** The thing this ADR exists to forbid, named so it cannot be
adopted quietly. It would make the reported metric the maximum of two draws.

**Pool all 17 quarterly folds and pick one method per model.** Attractive: 17 windows of evidence
instead of one, and far more stable than a per-fold choice. **Rejected because it is leakage, and
subtle leakage.** `quarterly-2022Q3`'s calibration window *is* `quarterly-2022Q2`'s test window, so
pooling to choose fold 1's method would choose it using fold 1's test period. A dedicated test,
`test_a_pooled_global_selection_would_read_an_earlier_folds_test_window`, asserts that this
rejected design is detectably leaky — the rejection is executable, not just written down.

**Choose strictly per fold, on that fold's inner-select portion alone.** The most literal reading of
"select using only the calibration window", and genuinely leakage-free. Rejected because at 409–756
select rows the winner flips fold to fold on noise: the measured paired-gap SD (median 0.0054) is
the same size as many of the observed gaps. That produces a calibrated time series whose method
changes underneath it, and the ECE-drift analysis in `calibration_findings.md` could then not
distinguish drift from a method switch. The per-fold winner is still logged.

**Use cross-validation inside the calibration window.** Attractive because it uses every row for
both fitting and selecting, which matters when rows are scarce. Rejected because random k-fold
across a window destroys the temporal ordering this project is built on, and because establishments
recur within a window — a random fold would put two rows of one establishment on both sides.

**Select on inner-select ECE, since ECE is what the component optimises.** Rejected for the three
reasons in §2. Notably the third: with a tunable bin count, "isotonic wins" and "Platt wins" are
both reachable from the same data.

**Skip the selection: always use isotonic, because it is more flexible.** The common default, and
rejected by the measurement. On these windows isotonic's inner-select log-loss reaches 1.1028
against Platt's 0.6850 on the same cell — pool-adjacent-violators on ~1,200 rows produces plateaus
at exactly 0 and 1, and a wrong plateau costs `−log(ε)`.

**Skip the selection: always use Platt, since it wins most cells here.** Tempting after seeing §6,
and rejected because it would forgo the comparison the brief asks for, and because "Platt wins on
this snapshot's calibration windows" is not the same claim as "Platt should always be used". The
protocol is what makes the answer defensible; the answer is not the protocol.

## Consequences

- The selection is reproducible from `calibrator_selection_*.parquet` alone: every candidate's
  inner-select log-loss, the prefix mean, the gap, the threshold in force, and the reason.
- A method switch mid-series is visible rather than silent, and is marked on the ECE-drift figure.
- The reported test-window comparison of Platt against isotonic is an **observation**, not a
  selection. Whichever wins on test, the frozen method was chosen before the test window opened.
- If isotonic wins on test but Platt was selected, that is reported as-is. It is the price of a
  protocol that cannot be tuned, and reporting it is the evidence that it was not.
- Component 10 inherits the frozen method per (model, fold) and must not re-select. Refitting a
  calibrator on a test quarter — including "just to check" — reintroduces exactly the leak this ADR
  removes.
