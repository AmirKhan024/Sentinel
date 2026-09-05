# ADR 0018 — Tuning trials are a fourth processed layer, and none of their numbers is a result

**Status:** Accepted · **Date:** 2026-08-17

## Context

The processed layer already holds three kinds of thing, each with its own directory and its own
membership test:

```
data/processed/features/      model-ready tables. Trainable.               ADR 0011
data/processed/predictions/   model outputs. Never trainable.              ADR 0014
data/processed/evaluation/    measurements about models. Never trainable.  ADR 0013
```

Component 7's hyperparameter search produces a fourth: one row per trial, carrying the parameters
drawn, the per-inner-fold PR-AUC, the mean, the frozen round count and the search region.

It fits none of the three. It is not model-ready — it has no features and no label. It is not a
model output — its grain is a *search attempt*, not a scored decision. And it is not a measurement
about a model in ADR 0013's sense, because ADR 0013's tables measure a model's behaviour on a
**test** window, whereas every number here comes from an inner validation window that is *training
data* for every fold the winning parameters are then used on.

That last distinction is the one that matters, and it is easy to lose. `mean_pr_auc = 0.6054` looks
exactly like a result. It has the right name, the right range, and it is the number the search
maximised. Filed beside `evaluation_metrics_*.parquet` it would eventually be read as one, and
reporting it as Component 7's PR-AUC would be reporting an in-sample number as an out-of-sample one.

## Decision

### A new directory, `data/processed/tuning/`

A sibling of the other three, not a child of any of them. `Settings.tuning_processed_dir` carries
the prohibition in its docstring, matching the pattern `predictions_processed_dir` established:

> **Nothing here may be joined onto a feature table, and no number in it is a result.** A
> validation PR-AUC read as a headline metric would be an in-sample number reported as an
> out-of-sample one.

The membership test for this layer: the grain is one attempt at a design choice; it is produced
*before* any model is fitted for scoring; and every number in it was measured on a window that is
training data downstream.

### One table, one manifest

`tuning_trials_<stamp>.parquet` with a sidecar `manifest_tuning_trials_<stamp>.json`, following the
convention every other component uses — one manifest per run, keyed to the anchor artifact.

The manifest records `tuning_regions` and `first_test_start` **per fold set**, side by side. That
pairing is the whole point of the record: a reader who wants to check that the search could not
have seen a test window compares two dates that are both written down, rather than trusting a
sentence. `sentinel tune-boosting` prints the same pairing, formatted as
`region  <  first test`.

### The trials table keeps failed trials

A trial whose parameters cannot be fitted is pruned, not crashed — one unfittable corner of the
space should not end a 100-trial search. But it is written to the table with `failed = true` and
the exception text, rather than dropped. A silently shorter table would read as "100 trials, all
successful", which would be a claim nobody checked.

Measured on the production run: 400 trials, 0 failed.

### This table carries a duration, and the others do not

`seconds` per trial, which breaks the rule `modeling/writer.py` states — no timestamp or duration
in a Parquet file, because two runs over identical inputs would then produce different bytes.

The exception is deliberate and narrow. How long a search took is the fact that justifies its trial
count, and it is the number that would have to be reported if a computational constraint ever
forced fewer than 100 trials. The trials table makes no determinism claim about its bytes; the
prediction and evaluation tables do, and they keep the rule.

### The selected parameters do not live here

They are frozen into `boosting.definitions.TUNED_PARAMS` as source literals. See ADR 0017 — a
parameter set loaded from this directory at training time could change without a diff, and the
value of freezing is that it cannot. This directory holds the *evidence* for the choice; the source
holds the choice.

## Alternatives rejected

**Write the trials into `data/processed/evaluation/`.** Attractive because they are metrics, and
ADR 0013's directory is where metrics live. Rejected because it is precisely the co-location that
invites the misreading: a `pr_auc` column in the evaluation directory is a result by every
convention this project has established, and this one is not.

**Write them into `data/processed/predictions/` beside the model outputs.** Rejected on the same
grounds ADR 0014 used to keep predictions out of `features/` — co-location is the invitation. It
would also mean two different grains under one slug.

**Keep them in `interim/`.** Attractive because a search is mid-pipeline. Rejected because ADR
0005 reserves interim for tables a *later component consumes*, and nothing consumes this one. It is
read by people, which is the processed layer's test.

**Do not persist trials at all; print the winner and move on.** Attractive, and it is what most
tuning workflows do. Rejected because the reproducibility claim in ADR 0017 rests on being able to
re-open the search: `TUNED_PARAMS_PROVENANCE` names this artifact's sha256, and without the artifact
the provenance points at nothing. It would also make "100 trials, 0 failed" unverifiable.

**Use Optuna's own storage (SQLite via SQLAlchemy).** Attractive because it comes free with the
dependency, supports resuming, and has a study browser. Rejected because it would introduce a second
persistence format into a project where every artifact is zstd Parquet plus a JSON manifest, and
because resumability is not wanted here — a study that can be resumed can be resumed *after*
someone has looked at a test number.

**Introduce MLflow for experiment tracking.** The specification mentions it and it is the obvious
tool. Rejected because the repository already has a working provenance mechanism — a manifest per
run, pinned by sha256, with the library versions and the input checksum — and MLflow would be a
second, parallel one. Every property the specification wanted from tracking (model, seed,
parameters, feature version, fold definition, objective, trial number, score) is in the trials table
and its manifest. Adding a tracking server, a backend store and a UI to re-express information
already recorded would be architectural expansion for its own sake. This is worth revisiting if the
project ever runs searches across machines or people, where MLflow's centralisation earns its cost.

## Consequences

- Four processed layers. The tree in README.md and STATUS.md must show all four, and each must
  state what may not be done with it.
- `.gitignore` already excludes `*.parquet` and whitelists `manifest_*.json`, so the trials
  manifest is committed and the trials themselves are not — matching every other layer.
- **No number in `tuning_trials_*.parquet` may appear in a results table.** The four best mean
  validation PR-AUCs (0.6054, 0.6033, 0.7786, 0.7777) are recorded in
  `boosting/definitions.py`'s provenance comment specifically so that they are documented
  *and* labelled as not-results in the same place.
- The `covid_shift` validation figures are the sharpest illustration: 0.7786 is far above any test
  number this project reports, because that region sits in the pre-2020 era when the base rate was
  around 0.82. Quoted without its region it would look like a breakthrough.
- Component 9 will tune a calibrator and should write here too, under its own slug, rather than
  mutating this one.
