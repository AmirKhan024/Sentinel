# ADR 0029 — Component 11 re-executes the frozen fits to attribute them, under ADR 0026's gate

**Status:** Accepted · **Date:** 2026-08-25

## Context

A Shapley value is a property of a **model**, not of a score. TreeSHAP walks the trees and
reads the cover recorded at each node. Linear SHAP reads the coefficients and the reference
mean. The permutation game calls the network, thousands of times, on synthetic rows that do
not appear in any table. None of that can be done from a Parquet file of probabilities.

And **no fitted model object is persisted anywhere in this repository.** ADR 0026 established
that at length for Component 9: `data/` holds no pickle, no booster dump and no `.pt` state
dict. What is persisted is the readable residue of a fit — coefficients, split-gain
importances, embedding tables. `modeling.models.FittedModel.pipeline` lives in memory for the
duration of one process.

So Component 11 faces ADR 0026's problem again, for a different reason. Component 9 needed
scores over a window nobody had scored. Component 11 needs the model itself.

This collides with ADR 0026's own closing consequence, which is quoted here rather than
paraphrased because it is the objection this ADR has to answer:

> **Component 10 must not treat the re-execution as licence to re-fit anything else.** The
> base models are frozen; this component re-ran them to recover a missing recording, not to
> revisit them.

## Decision

### Component 11 re-executes, under the same gate, and the gate is not reimplemented

`sentinel explain` imports `modeling.train.fit_fold`, `boosting.train.fit_fold` and
`neural.train.fit_fold` and calls them with the same registry spec, the same seed (42), the
same hyperparameters, the same canonical row order and the same training frame —
`modeling.train.training_frame`, the repository's one definition of "train". Nothing is
tuned, no hyperparameter is touched, no feature is added, no target is changed. **Not one
line of `modeling/`, `boosting/` or `neural/` is modified, and not one byte of their
committed artifacts is rewritten.**

The proof is ADR 0026's, called rather than copied. `explain.refit.check_reproduction`
assembles the re-executed test scores into a `calibration.models.BaseScores` and hands it to
`calibration.basescores.committed_test_scores` and
`calibration.basescores.reproduction_mismatches` — the same two public functions Component 9
uses, comparing with `==` rather than `math.isclose`. There is therefore **one definition of
"the same model" in this project**, not two that could drift apart.

- It runs as an error-severity check, `regenerated_scores_reproduce_the_committed_artifact`.
- `build.py` **raises before computing a single attribution** if it fails. A validation
  report alone would let the artifact be written and merely complain about it.
- `test_the_prediction_mismatch_detector_itself_works` perturbs one recorded score by a
  single ULP and asserts the check goes red, so the exactness of the comparison is
  executable rather than declared.

### Why this is not the thing ADR 0026 forbade

ADR 0026 forbade treating re-execution as *licence to revisit* the base models — to re-tune
them, refit them on more data, or change what they are. Component 11 does none of that, and
structurally cannot: it fits nothing of its own, selects nothing, and its output is refused
by `evaluate --predictions` because an attribution has no `score` column.

The distinction that matters is between **re-running a deterministic computation to read
something that was never recorded**, and **changing a model**. Component 9 re-ran the fits to
read scores over a window; Component 11 re-runs them to read the model's internal structure.
Both are reads. Neither produces a model any component would deploy, and both are gated on
producing bit-identical output to the committed artifact — which is the operational
definition of "did not change anything".

Put the other way: if the re-execution *had* changed a model, the gate would fail and the run
would abort. The forbidden outcome is not merely discouraged here, it is unreachable.

### Measured

On the production run: **41,536 test rows per model across 4 models and 18 folds — 166,144
comparisons, zero mismatches.** Every supported candidate reproduces bit for bit, including
`neural_numeric_only`, whose fit is a full gradient-descent training run with early stopping.

### What is *not* re-executed

`xgboost_chain_embeddings` has no path in `explain/refit.py` at all. Its fitted booster is
reachable only through a private helper, and Component 8 is closed. See ADR 0031.

## Alternatives rejected

**Add model persistence to Components 6, 7 and 8 now.** ADR 0026 rejected this for Component
9 and the argument has not weakened — it modifies three closed components and rewrites three
committed artifacts to serve a fourth. It has, however, now been the right answer twice in a
row, and a note is added to the handoff: the next component that needs a model object should
propose persistence as its own change rather than re-deriving it a third time.

**Reconstruct the models from their persisted residue.** Possible for the linear model —
`baseline_coefficients_*.parquet` carries the coefficients, the scaler statistics and the
imputation fills, which is everything linear SHAP needs. Rejected because it works for
exactly one of the four supported models and would leave the component with two code paths
whose agreement nobody checks; and because a synthesised `FittedModel` would be a 20-field
dataclass most of whose values were fabricated, inside an object a leakage test is supposed
to be able to read.

**Attribute without a model, using a model-agnostic explainer over the committed scores.**
Attractive-sounding and wrong: a model-agnostic explainer still has to *call the model* on
perturbed inputs, and the committed scores exist only for the rows that were actually scored.
There is no set of stored predictions from which a Shapley value can be computed.

**Loosen the gate to a tolerance if it fails.** Named here so it cannot be adopted quietly
later, exactly as ADR 0026 named it. A tolerance would convert the one check that makes this
ADR safe into a check that passes when the models differ.

**Declare Component 11 blocked, as Component 10 is.** ADR 0019's blocker is *absent data* —
the dataset has 22 columns and no inspector field, and nothing can conjure one. This is an
absent *artifact* whose producer is deterministic and still present. Treating the two as the
same kind of obstacle would be a category error, and ADR 0026 already made that argument.

## Consequences

- `sentinel explain` takes about **19 minutes** on the production snapshot: 72 re-executed
  fits in 438.6 s (the 18 network fits are most of it) and 647.3 s of attribution (almost
  entirely the permutation game; TreeSHAP and the closed form are milliseconds). Stated in the
  README so nobody expects a report command to return in seconds.
- **The run must not set `OMP_NUM_THREADS`.** ADR 0026 records the gate correctly failing on
  32,696 of 41,536 `logistic_regression` rows under a thread-count override, by 1e-13 to
  5e-10, because a different BLAS thread count is a different float summation order. The
  value in force is written into every Component 11 manifest so the two runs can be compared
  rather than assumed.
- The bit-identity gate is now a standing regression test on Components 6–8's determinism in
  a *second* place. A dependency bump that silently changed a model's arithmetic would be
  caught by both `sentinel calibrate` and `sentinel explain`.
- Components 6, 7, 8 and 9's artifacts remain byte-identical, and Component 11's manifest
  records each one's sha256 **before and after** its own run, so "Component 11 changed no
  prediction" is checkable rather than asserted.
