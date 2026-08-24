# ADR 0024 — Component 9 writes to three layers, and its diagnostics get a sixth

**Status:** Accepted · **Date:** 2026-08-24

## Context

The processed layer holds five kinds of thing, each with its own directory and its own membership
test:

```
data/processed/features/      model-ready tables. Trainable.               ADR 0011
data/processed/predictions/   model outputs. Never trainable.              ADR 0014
data/processed/evaluation/    measurements about models. Never trainable.  ADR 0013
data/processed/tuning/        attempts at a design choice. Not results.    ADR 0018
data/processed/neural/        an experimental model input.                 ADR 0022
```

Component 9 produces three grains at once, and they do not share a home.

1. **Calibrated predictions** — one calibrated probability per (model, fold, scored test row).
2. **The calibrator selection log** — one row per (model, fold, candidate method), recording what
   each method scored on an inner-select window and which was frozen.
3. **Diagnostics** — the re-derived calibration-window base scores, the fitted calibrator
   parameters, the isotonic breakpoints, the per-quarter drift table, the ranking-preservation
   table, the Brier decomposition and the bootstrap intervals.

Two of the three were already assigned a home by earlier ADRs, in as many words. ADR 0014: *"Component
9 will add calibrated predictions. It should write its own slug under the same layer and reuse this
contract rather than mutating Component 6's artifact."* ADR 0018: *"Component 9 will tune a
calibrator and should write here too, under its own slug, rather than mutating this one."*

The third has no home. It is not model-ready — no features, no label. It is not a model output — its
grain is a fitted correction and a set of measurements, not a scored decision. It is not a search
attempt. And although it is closest to ADR 0013's evaluation layer, that is precisely where it must
not go.

## Decision

### Calibrated predictions go to `data/processed/predictions/`, under their own slug

`calibrated_predictions_<stamp>.parquet`, with the run's anchor manifest beside it.

This is what ADR 0014 instructed, and it earns its place on that ADR's own membership test: the
grain is one scored decision per model per fold; it is produced after fitting and before scoring;
and it is self-describing.

The practical consequence is the one that matters: the artifact is shaped so
`evaluation.contract.read_predictions` reads it **with no change to Component 5**. It carries the
contract's two columns plus the metadata every prediction file carries, and the extra Component 9
columns are invisible to the validator, which selects only `PREDICTION_COLUMNS` into a
`PredictionSet`. `sentinel evaluate --predictions calibrated_predictions_<stamp>.parquet` therefore
just works, and the headline PR-AUC, ROC-AUC, NDE and precision@k for a calibrated model come from
the same evaluator that produced every earlier number.

`model_name` is `"<base>_<method>"` — `neural_numeric_only_platt`, never a bare
`neural_numeric_only`. A calibrated row and its uncalibrated ancestor can then be held in one table
without either overwriting or being mistaken for the other.

Components 6, 7 and 8's artifacts are not touched.

### The selection log goes to `data/processed/tuning/`, under its own slug

`calibrator_selection_<stamp>.parquet`, with its own manifest.

ADR 0018's membership test passes exactly. The grain is one attempt at a design choice — should
this fold use Platt or isotonic? It is produced *before* the artifact that gets scored. And every
number in it was measured on an inner-select window carved out of the calibration period, which is
not a test window and whose log-loss is not a result.

It also inherits ADR 0018's narrow exception for a `seconds` column, for the same reason: how long
the selection took is the fact that justifies its shape.

### The diagnostics get a new directory, `data/processed/calibration/`

A sibling of the other five, not a child of any. `Settings.calibration_processed_dir` carries the
prohibition in its docstring, matching the pattern `predictions_processed_dir` established:

> **Nothing here may be joined onto a feature table, and the calibration-window scores in
> particular must never reach a fit.** These rows sit after `train_end`; a base model that saw them
> would have been fitted past its own declared horizon.

The membership test for this layer: the grain is a *fitted correction* or a measurement of one; it
is produced after the base model and before the evaluation; and it exists so that the calibrator can
be reproduced from artifacts alone rather than by re-running the component.

Seven tables: `calibration_base_scores`, `calibrator_parameters`,
`calibrator_isotonic_breakpoints`, `calibration_drift`, `calibration_ranking_preservation`,
`calibration_brier_decomposition`, `calibration_bootstrap`.

### The diagnostics may not go in `data/processed/evaluation/`, and this is the load-bearing part

Component 9's drift table carries an `ece` column, per model, per fold, measured on that fold's test
window. `evaluation_metrics_*.parquet` carries an `ece` column, per model, per fold, measured on
that fold's test window. Filed in one directory there would be **two authoritative ECEs for the same
(model, fold)** with no convention saying which is the project's answer, and a reader joining the
two would silently get a cross product.

Component 5 remains the only producer of the headline metrics. Component 9's tables are the
diagnostics of a correction, and their separate location is what keeps that boundary visible.

The general rule this component must not erode: **Component 9 computes only the metrics Component 5
does not have** — Brier decomposition, calibration slope and intercept, bootstrap intervals and
ranking-preservation deltas. For ECE, MCE, Brier and log-loss it *imports*
`evaluation.metrics` rather than reimplementing it, so a C9 number and a C5 number can never
disagree over binning. `evaluation/metrics.py` is not modified.

### The calibrator itself is persisted, not only its output

`calibrator_parameters` holds Platt's coefficient and intercept in the long form Component 6 used
for coefficients. `calibrator_isotonic_breakpoints` holds `X_thresholds_` / `y_thresholds_` plus the
clip bounds, which with `np.interp` reproduce the map exactly.

Isotonic breakpoints are written for **every fold where isotonic was fitted, not only where it
won**, and the drift table carries all four stages (`uncalibrated`, `platt`, `isotonic`,
`selected`). The counterfactual — "would isotonic have been better on this quarter?" — is then
answerable from the artifact instead of by re-running with a different flag, which is how a
selection quietly becomes a test-set selection.

## Alternatives rejected

**Put everything under `data/processed/calibration/`.** Simpler, one place to look, and rejected
because ADR 0014 and ADR 0018 each named a home for their grain in advance. Honouring them is also
what keeps `sentinel evaluate --predictions` working with no Component 5 change; moving the
calibrated predictions would either break that seam or require editing a closed component.

**Put the diagnostics in `data/processed/evaluation/`.** Attractive because they are metrics, and
ADR 0013's directory is where metrics live. Rejected for the reason above: two authoritative ECEs
for the same (model, fold). This is the same argument ADR 0018 used to keep validation PR-AUC out of
that directory, applied to a metric whose name collides exactly rather than merely reading like a
result.

**Put the calibration-window base scores in `data/processed/predictions/` beside the test scores.**
Attractive — same grain, same columns, one row per scored row. Rejected because they are scores over
a window that is *not* the test window, and the prediction contract's fourth rejection rule exists
precisely to guarantee that a file in that directory covers a fold's test window exactly. A file
there that did not would make the contract's strongest guarantee conditional on which slug you read.

**Extend `evaluation/metrics.py` with the decomposition and the bootstrap.** The obvious home for a
metric, and rejected because Component 5 is closed and its `RANKING_METRICS` /
`PROBABILITY_METRICS` tuples are part of its contract — appending to them would change what every
earlier component's artifacts are read as. The new functions live in `calibration/metrics.py` and
import the old ones.

**Persist only the calibrated probabilities and drop the calibrator.** Attractive because the
probabilities are what downstream components consume. Rejected because a calibrator that cannot be
re-applied is a black box: a later component could not calibrate a new prediction without re-running
Component 9, and nobody could check that the mapping was monotone.

## Consequences

- Six processed layers. The tree in README.md and STATUS.md must show all six, and each must state
  what may not be done with it.
- `.gitignore` already excludes the Parquet files and whitelists `manifest_*.json`, so the new
  directory's manifests are committed and its data is not — matching every other layer.
- One component writes to three directories, which is more surface than any earlier component. That
  is the cost of honouring two prior ADRs' instructions rather than consolidating; the membership
  tests genuinely differ, and each table is where its grain says it belongs.
- **No number in `data/processed/calibration/` is the project's headline.** The headline PR-AUC,
  ROC-AUC, NDE, precision@k and days-earlier for a calibrated model come from
  `sentinel evaluate --predictions`, exactly as they do for an uncalibrated one.
- Component 10 consumes `calibrated_predictions_*.parquet` and may read the calibrator parameters to
  understand the mapping. It must not join `calibration_base_scores_*.parquet` onto anything it
  fits.
