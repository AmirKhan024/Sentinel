# ADR 0028 — Feature attributions are a seventh processed layer, not predictions and not results

**Status:** Accepted · **Date:** 2026-08-25

## Context

Component 11 produces a new kind of row: one model's contribution of one feature to one
prediction, plus the summaries built from those contributions — global importance, rank
stability, explanation drift and a set of representative local cases.

Six processed layers already exist, each created by an ADR that had to argue why the new
thing was not one of the existing things:

```text
features/       ADR 0011   model-ready tables
evaluation/     ADR 0013   measurements of a model's performance
predictions/    ADR 0014   model outputs
tuning/         ADR 0018   what a search tried
neural/         ADR 0022   an experimental model input
calibration/    ADR 0024   a fitted correction and its diagnostics
```

An attribution is none of them, and the two near-misses are worth stating because both would
have been defensible and both would have caused a specific harm later.

## Decision

**`data/processed/explanations/` is a seventh layer**, holding seven tables:
`explanation_values`, `explanation_cases`, `explanation_importance`,
`explanation_stability`, `explanation_drift`, `explanation_representative_cases` and
`explanation_support`, with one manifest keyed to `explanation_values`.

### Why not `predictions/`

An attribution is not a model output. It does not score anything; no scheduler could rank on
it; `sentinel evaluate --predictions` would reject it, because
`evaluation.contract.PREDICTION_COLUMNS` is `(target_inspection_id, score)` and an
attribution row has neither meaning. It *explains* an output that already exists in
`predictions/` and is joined to it by `(model_name, fold_id, target_inspection_id)`.

Filing it there would also put a per-establishment number produced *from* a model into the
same directory as the numbers produced *by* it, which is precisely the shape of confusion
ADR 0014 created its layer to avoid.

### Why not `evaluation/`, which is the more tempting mistake

Component 5 owns the question "is this model any good". An attribution answers a different
question — "what did this model lean on" — and it carries **no notion of correct**. A large
`mean_abs_shap` says the model relied on a feature. It does not say the model was right to,
and a model can lean hard on a feature that is misleading it: Component 6 measured
`days_since_any_inspection` helping on the quarterly folds and *hurting* under distribution
shift, which is exactly a case where high reliance and low value coincide.

Filed beside `evaluation_metrics_*.parquet`, that distinction would not survive contact with
a reader. A column called `mean_abs_shap` sitting next to `roc_auc` will eventually be read
as a quality measure, and then someone will pick a model because its attributions looked
tidier — which ADR 0013's separation and section 21 of Component 11's brief both forbid.

The same argument ADR 0024 made for calibration applies here in a sharper form: two
authoritative-looking numbers about the same model, in one directory, with no convention
saying which answers which question.

### What the layer's docstring says

`Settings.explanations_processed_dir` carries the rule every other layer carries:

> **Nothing here may be joined onto a feature table, and no number here is a result.** A SHAP
> value describes how a model used a feature; it does not measure the feature's effect on
> food safety, and a per-establishment attribution joined back onto training rows would be a
> model's own output re-entering it as an input.

That last clause is not hypothetical. `explanation_values` is keyed by
`target_inspection_id`, which is Component 4's key, so the join is one line away and would
be the most damaging leak available in the project — a model's own reasoning about a row
becoming a feature of that row.

### The grain, and why it is denormalised on purpose

`explanation_values` repeats `base_value`, `prediction_value` and `trained_through` on every
one of its thirty rows per prediction. That is deliberate. The artifact must be readable
**without loading a model or importing this package**, and a consumer asking "does this row's
decomposition add up, and what horizon produced it?" can then answer from one table rather
than from a join it might get wrong. zstd stores thirty identical floats for almost nothing;
a reader who joins on the wrong key and publishes a wrong reconstruction costs considerably
more.

## Alternatives rejected

**One wide table, one row per prediction, thirty attribution columns.** Compact and
convenient for a dashboard. Rejected because the column names would then encode Component
4's feature set into the *schema*, so adding a feature would be a schema change in Component
11 — and because a long table is the shape every aggregate in `aggregate.py` wants.

**Put the attributions in `predictions/` beside the scores they explain.** Rejected above.

**Put the summaries in `evaluation/` and only the raw values in a new layer.** The worst of
both: it splits one component's output across two layers on a boundary no reader could infer,
and it files the importance table — the one most likely to be misread as a quality measure —
in the one directory where that misreading is easiest.

**Emit only figures.** Explicitly refused by the brief, and rightly: a PNG cannot be joined,
checksummed, re-aggregated or audited, and a later scheduling or UI component needs the
numbers rather than a picture of them.

## Consequences

- A seventh `Settings` property, and a seventh entry in every directory listing in the
  README. The layer count is becoming a thing a reader has to hold in their head; a future
  component that adds an eighth should consider whether the taxonomy still earns its keep.
- The artifact is the largest the project has produced by row count — 30 attribution rows per
  explained prediction — which is why the explanation sample is bounded (ADR 0030) rather
  than covering all 41,536 test rows per model.
- `data/processed/explanations/` is added to `.gitignore` alongside every other data
  directory, with its `manifest_*.json` whitelisted, matching the convention Component 2
  established.
- A downstream scheduling or UI component can read `explanation_cases` and
  `explanation_values` and render a per-establishment explanation without importing
  `sentinel.explain` at all. That was a design goal, and it is why every provenance column
  the renderer needs is on the table rather than in the manifest.
