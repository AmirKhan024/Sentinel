# ADR 0026 — Component 9 re-executes Components 6–8's fits, and proves bit-identity before calibrating

**Status:** Accepted · **Date:** 2026-08-24

## Context

A calibrator is fitted on a base model's scores over the fold's calibration window. Component 9
therefore needs, for every candidate model and every fold, the score that model would have assigned
to each calibration-window row.

**Those scores do not exist, and neither do the models that would produce them.**

Every prediction artifact on disk covers exactly the test window and nothing else — 41,536 rows per
model across the 18 folds, which is `sum(test_rows)` to the row. `sum(calibration_rows)` is 34,261
and appears nowhere. This is not an oversight: `modeling/predict.py` says so in its docstring
("Nothing here touches the calibration window"), and Components 6, 7 and 8 each write a column
literally named `calibration_end_unused` into their training logs, with a comment reading *"Recorded
to make visible that Component N did not use it. Component 9 will."* ADR 0012 built the window
empty on purpose so that this component would have nowhere else to put a calibrator.

The second half of the problem is the one with no precedent. **No fitted model object is persisted
anywhere in this repository.** `data/` holds no pickle, no booster dump and no `.pt` state dict.
What is persisted is the *readable* residue of a fit: `baseline_coefficients_*.parquet` (with the
scaler statistics and imputation fills), `boosted_importances_*.parquet` (split counts, a
diagnostic), and `neural_embeddings_*.parquet` (the learned embedding tables).
`modeling.models.FittedModel.pipeline` lives in memory for the duration of one process;
`neural.train._MODELS` and `embed._ESTIMATORS` are process-local dicts.

So the calibration-window scores cannot be produced by loading a model and scoring a different
window. The fits themselves have to run again.

That collides with the instruction Component 9 was given — *do not re-train the base models* — and
with the rule that has governed every component since the first: **Components 1–8 are closed; if you
find a defect upstream, document it and stop before modifying it.**

## Decision

### The missing artifact was reported before anything was built

The blocker was surfaced and the resolution approved before a line of Component 9 was written. This
ADR is the record of that, and it is the reason the work is not a silent rebuild.

### Component 9 re-executes the unchanged fit functions; it does not re-train

The distinction is not rhetorical, and it is the whole basis for proceeding.

`sentinel calibrate` imports `modeling.train.fit_fold`, `boosting.train.fit_fold`,
`neural.train.fit_fold` and `neural.embed.fit_fold` and calls them with the same registry spec, the
same seed (42), the same hyperparameters, the same canonical row order and the same training frame
— `modeling.train.training_frame`, which is the repository's one definition of "train". Nothing is
tuned, no hyperparameter is touched, no feature is added, no target is changed. **Not one line of
`modeling/`, `boosting/` or `neural/` is modified, and not one byte of their committed artifacts is
rewritten.** Component 9 is a consumer of those packages in exactly the way Component 5 is a
consumer of their output.

Every determinism guarantee those components claim is a guarantee about this re-execution:
`fit_fold` sorts training rows canonically because the sort — not `random_state`, which lbfgs
ignores — is what makes a re-run reproducible; `net.seed_everything` sets
`torch.use_deterministic_algorithms(True)` and a single thread.

### The bit-identity gate: the claim is proved, not asserted

Re-executing a fit and getting a *different* model would invalidate the entire component — a
calibrator fitted on scores that no committed artifact contains is a correction to nothing. So the
claim is not left as reasoning.

For every candidate and every fold, the re-executed fit scores **both** windows. The calibration
window is what Component 9 wanted. The test window is the control: it is compared against the
committed Component 6/7/8 artifact, joined on `(model_name, fold_id, target_inspection_id)`, with
`==` — bit-identity, not `math.isclose`.

- It runs as an **error-severity** validation check, `base_scores_reproduce_the_committed_artifact`.
- `build.py` **raises before fitting any calibrator** if it fails. A validation report alone would
  let the artifact be written and merely complain about it.
- Two pytest cases guard the gate itself: one asserts the reproduction holds, and
  `test_the_bit_identity_detector_itself_works` perturbs a single regenerated score by one ULP and
  asserts the check goes red — proving the comparison really is exact.
- The manifest records the outcome per model and pins each committed artifact by sha256.

This gate is strictly stronger evidence than persisting the models would have been. A pickle proves
a model was saved; a bit-identical re-derivation proves the *pipeline* is reproducible end to end.

### `xgboost_chain_embeddings` re-runs its donor rather than reading the persisted table

The chain embedding block *can* be reconstructed exactly from `neural_embeddings_*.parquet`:
`embed.embedding_rows` writes `float(value)` of a float32 weight, which round-trips through Parquet
Float64 without loss, and the donor's chain vocabulary is recoverable as
`set(categories) − {__UNKNOWN__, __INDEPENDENT__}` by construction.

It is nevertheless re-fitted, for three reasons. The XGBoost half is not persisted either, so the
reconstruction saves only the 18 donor MLP fits — on the order of 150 seconds. Using it would
require synthesising a `FittedNetwork`, a 30-field frozen dataclass, most of whose values would be
fabricated — inside an object whose purpose is to be readable by a leakage test. And re-fitting the
donor yields a second free bit-identity proof.

The persisted table is used instead as an **error-severity check**,
`chain_embeddings_reproduce_the_committed_table`, which asserts the re-fitted donor's chain vectors
equal the committed table bit for bit and that the vocabulary identity above holds.

### The environment is pinned, and a version bump is a re-baseline

Bit-identity is scoped to this feature table, this row order, this library set, one thread and CPU
— the same narrow claim `manifest_neural_predictions_*.json` already makes. The manifest records
scikit-learn 1.9.0, numpy 2.5.2, xgboost 3.4.1, lightgbm 4.7.0 and torch 2.13.0+cpu.

**On a different library build the gate will fail, and that is the correct behaviour.** It is an
explicit, documented re-baseline, never a reason to loosen the comparison to a tolerance.

## Alternatives rejected

**Retroactively add model persistence to Components 6, 7 and 8 and re-run them.** The cleanest
artifact contract, and rejected because it modifies three closed components and rewrites three
committed artifacts to serve a fourth. It would also make Component 9's correctness depend on a
change to the code it is meant to be calibrating, and leave the repository unable to say whether a
difference came from the calibrator or from the rewrite. The right time to persist a model is when
the component that needs it is being built; the right way to record that is this ADR, and a note in
the Component 10 handoff.

**Fit the calibrator on the training window instead.** No re-execution needed, since training-window
scores could be regenerated just as easily — but the model is fitted on those rows, so its scores
there are optimistic in exactly the way a calibrator would then bake in. ADR 0012 rejected this in
advance: *"fitting it on the training period would inherit the model's own overfitting."*

**Fit the calibrator on the test window.** The leak this project exists to prevent. ADR 0012: *"a
calibrator fitted on test makes the reported probabilities self-fulfilling."*

**Refit each base model on train + calibration, then calibrate.** Attractive because it uses more
data. Rejected twice over: it is a *different model* from the one Components 6–8 evaluated, so no
Component 9 number would be comparable with any earlier one; and it destroys the held-out property
of the very window the calibrator needs.

**Declare Component 9 blocked, as Component 10 is (ADR 0019).** The honest option if no faithful
route existed. Rejected because one does: ADR 0019's blocker is *absent data* — the dataset has 22
columns and no inspector, and nothing can conjure one. This blocker is an absent *artifact* whose
producer is deterministic and still present in the repository. A missing recording of a reproducible
computation is not the same kind of obstacle as a missing measurement, and treating it as one would
be a category error.

**Loosen the gate to `math.isclose` if it fails.** Rejected pre-emptively, and named here so it
cannot be adopted quietly later. A tolerance would convert the one check that makes this ADR safe
into a check that passes when the models differ.

## Consequences

- Component 9 cannot be reproduced without re-running the fits, which takes minutes rather than
  seconds. The data contract states this so a future reader does not go looking for a saved model.
- The calibration windows are read for the first time in the project's history. After this
  component, `calibration_end_unused` is no longer true of every artifact — only of Components 6, 7
  and 8's, where it remains accurate.
- **Components 6, 7 and 8's prediction artifacts remain byte-identical**, and their sha256s are
  recorded in `docs/data_contracts/calibrated_predictions.md` so the claim is checkable.
- The re-derived calibration-window scores are persisted in
  `data/processed/calibration/calibration_base_scores_*.parquet` (ADR 0024), so a later component
  never has to repeat this re-execution to audit the calibrator.
- The bit-identity gate is a standing regression test on Components 6–8's determinism. If a future
  dependency bump silently changes a model's arithmetic, this is the check that catches it — a
  benefit the project did not previously have.
- **Component 10 must not treat the re-execution as licence to re-fit anything else.** The base
  models are frozen; this component re-ran them to recover a missing recording, not to revisit them.
