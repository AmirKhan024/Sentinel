# ADR 0032 — Group-audit artifacts are an eighth processed layer

**Status:** Accepted · **Date:** 2026-08-25

## Context

Component 12 produces a new kind of row: one metric, for one model, conditioned on one group,
at one grain — plus the support counts that say whether the number means anything, and the
disparity summaries built on top.

Seven processed layers already exist, each created by an ADR that had to argue why the new
thing was not one of the existing things:

```text
features/       ADR 0011   model-ready tables
evaluation/     ADR 0013   measurements of a model's performance
predictions/    ADR 0014   model outputs
tuning/         ADR 0018   what a search tried
neural/         ADR 0022   an experimental model input
calibration/    ADR 0024   a fitted correction and its diagnostics
explanations/   ADR 0028   feature attributions
```

ADR 0028 closed with a warning this ADR has to answer directly:

> The layer count is becoming a thing a reader has to hold in their head; a future component
> that adds an eighth should consider whether the taxonomy still earns its keep.

It does, and the reason is the same one that created the sixth and the seventh.

## Decision

**`data/processed/fairness/` is an eighth layer**, holding ten tables:
`fairness_group_definitions`, `fairness_group_support`, `fairness_group_metrics`,
`fairness_group_calibration`, `fairness_priority_audit`, `fairness_group_missingness`,
`fairness_attribution_profiles`, `fairness_disparity`, `fairness_drift` and
`fairness_bootstrap`, with one manifest keyed to `fairness_group_metrics`.

### Why not `evaluation/`, which is the tempting mistake again

Component 5 owns the question "is this model any good", and it answers it with a
`roc_auc` per `(model, fold)` in `evaluation_metrics_*.parquet`.

Component 12 emits a `roc_auc` per `(model, fold, group_definition, group_value, grain,
stage)`. Filed in the same directory, `evaluation_metrics_20260824T160045Z.parquet` and
`fairness_group_metrics_<stamp>.parquet` would both contain a column called `roc_auc`
describing the same model on the same fold, differing only in a group filter and in whether
the score is the base or the calibrated one. A reader who grouped one and joined the other
would get two authoritative answers with no convention saying which is which.

This is precisely the argument ADR 0024 made for calibration diagnostics —

> a C9 drift table carries an `ece` per (model, fold) on the test window, and so does
> `evaluation_metrics_*`. Filed together there would be two authoritative ECEs for the same
> cell.

— and ADR 0028 made for attributions. The taxonomy earns its keep by being the thing that
stops that collision, and it earns it a third time here.

**Component 5 remains the only producer of the headline metrics.** Component 12 computes no
un-conditioned number: every row it writes carries a `group_definition` and a `group_value`.
The pooled reference value each disparity is measured against is written into the disparity
table as `reference_value` rather than re-emitted as a metric row, so there is never a
Component 12 row that could be mistaken for a Component 5 headline.

### Why not `predictions/`

Nothing here scores anything. `evaluation.contract.PREDICTION_COLUMNS` is
`(target_inspection_id, score)` and no table below has either as a meaningful pair, so
`sentinel evaluate --predictions` would reject every one of them. And a per-group number
produced *from* a model does not belong in the directory holding numbers produced *by* it,
which is the confusion ADR 0014 created its layer to avoid.

### Why not inside `explanations/`

Only one of the ten tables reads Component 11's artifact, and it reads it the way any
consumer would — by grouping it. Filing the other nine beside it would file a support count
and a selection-rate ratio under a heading that means "feature attribution".

### What the layer's docstring says

`Settings.fairness_processed_dir` carries the rule every other layer carries, plus one this
component needs specifically:

> **Nothing here may be joined onto a feature table, and no number here is a verdict.** A
> group metric describes how a model behaved on a subset of held-out rows. It does not
> establish discrimination, causality, legal compliance or the absence of bias, and a
> per-group number joined back onto training rows would make a model's measured behaviour on
> a neighbourhood an input to how it treats that neighbourhood next time.

The second clause is the one worth having. `fairness_group_support` is keyed by group value
rather than by `target_inspection_id`, so it is one join *and* one broadcast away from
becoming a feature — and a feature meaning "the model was well calibrated here last quarter"
would be the most self-fulfilling input this project could construct.

### The grain, and why two of them are persisted

Every metric table carries `grain ∈ {fold, fold_set}`. The per-fold rows are the honest
record that the data is thin: `scripts/profile_fairness.py` measured a median of **16 rows**
per (fold, community area) cell, and only 4 of 1,288 cells reaching 200. Almost every
per-fold row will therefore carry `group_status = insufficient_support` and a null value.

Writing them anyway is the decision. A table containing only the cells that qualified would
report the same conclusions while making the shortage invisible, and "we measured 51 groups"
reads very differently from "we measured 51 of 78, and per quarter almost none".

## Alternatives rejected

**Add group columns to `evaluation_metrics_*` and let the group be null for the overall
row.** Compact, and it would put every metric in one place. Rejected: it changes Component 5's
schema, which is closed, and it makes the overall/conditional distinction a null check on a
column rather than a directory boundary. A `WHERE group_value IS NULL` that someone forgets is
an aggregate silently computed over a subset.

**One wide table, one row per (model, fold, group), one column per metric.** Rejected for the
reason ADR 0028 rejected the same shape: the column names would encode the metric set into the
schema, so adding a metric would be a schema change — and the tidy-long form is what
`evaluation_metrics_*` already uses, so a reader who can read one can read the other.

**Emit only a findings document and figures.** Explicitly refused by the brief, and rightly. A
figure cannot be re-aggregated, checksummed or audited, and Component 13 will need the numbers
rather than a picture of them.

**Fold the support table into the metrics table.** Rejected because support is
model-independent: rows, positives and base rate are properties of the fold and the group, not
of the estimator. Repeating them on every (model, stage, metric) row would multiply one
measured fact across roughly two hundred rows and invite the two copies to disagree.

## Consequences

- An eighth `Settings` property and an eighth entry in every directory listing in the README.
  The taxonomy is now large enough that a ninth should have to argue harder than this one did.
- `data/processed/fairness/` joins every other data directory in `.gitignore`, with its
  `manifest_*.json` whitelisted, matching the convention Component 2 established.
- Component 12 reads five artifacts and writes to exactly one layer — unlike Component 9,
  which had to write to three. That is because a group metric is not a prediction and not a
  trial, so there is nowhere else it could legitimately go.
- A downstream policy or UI component can read `fairness_group_support` and
  `fairness_priority_audit` and render an equity view without importing `sentinel.fairness`,
  which is why every provenance column such a renderer needs is on the table rather than in
  the manifest.
