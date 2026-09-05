# ADR 0031 — `xgboost_chain_embeddings` is reported unsupported, and the fix is proposed rather than taken

**Status:** Accepted · **Date:** 2026-08-25

## Context

`xgboost_chain_embeddings` is, structurally, the easiest model in the project to explain. It
is a plain `XGBClassifier` over a 46-column matrix — Component 7's 30 columns plus the 16
learned chain-embedding dimensions — and exact TreeSHAP would run on it in under a second per
fold. Its column names are already public: `embed.augmented_columns` returns them, and
`embed.embedding_columns` names the block `chain_emb_00 … chain_emb_15`.

**The fitted booster cannot be reached.** Measured by
`scripts/profile_explanations.py`'s `embedding_booster_boundary` profile:

```text
FittedEmbeddingBooster has 19 fields and none of them is the estimator:
  spec, fold_set, fold_id, matrix_columns, params, n_estimators, trees_built,
  importances, embedding_columns, donor_model, donor_fold_id, train_rows,
  train_positive_rate, train_nan_cells, train_start, train_end, trained_through,
  calibration_end_unused, seed

neural.embed public functions: augmented_columns, augmented_matrix, embedding_columns,
  embedding_rows, fit_fold, lookup_vectors, score_window
  -- every one takes a FittedEmbeddingBooster or returns names/vectors; none returns
     the estimator.

The only route to the live booster is neural.embed._scorer_for, which is private and
reads a process-local dict keyed by id().
```

The asymmetry with the network is what makes this a boundary question rather than a capability
question. `neural.train.scorer_for` **is** public — it is in `neural.train.__all__` — which is
exactly how `neural_numeric_only` is reached and explained. `neural.embed._scorer_for` is
private. That difference is not a considered decision in Component 8; it is an accident of
which helper a public `predict` happened to need to name.

Section 28 of Component 11's brief, and section 0 of `HANDOFF.md`, both say what to do when a
required interface is missing: stop, document the boundary problem, propose the smallest
possible public extension, and **do not silently reach into private helpers.**

## Decision

### The model is reported `unsupported`, with the measurement as its reason

`explanation_support` carries one row for it:

```text
model_name           xgboost_chain_embeddings
explanation_status   unsupported
explanation_method   NULL
output_space         NULL
explained_rows       0
attribution_values   0
unsupported_reason   the fitted booster is reachable only through
                     neural.embed._scorer_for, a private process-local stash;
                     FittedEmbeddingBooster has 19 fields and none of them is the
                     estimator, and no public neural.embed function returns it.
                     Component 8 is closed, so Component 11 does not reach into a
                     private helper to explain this model. The minimal public
                     extension that would lift the restriction --
                     embed.booster_for(fitted) -- is proposed in ADR 0031 and
                     deliberately not taken here.
```

It appears **nowhere else**. No values, no cases, no importance rows, no figure. That is
enforced by `validate.unsupported_models_carry_no_attributions`, and driven by
`test_a_fabricated_attribution_for_an_unsupported_model_is_rejected`.

**Nulls, never zeros.** Zero is a legitimate attribution meaning "this feature did not move
the score", so a placeholder row of zeros would read as a model that used no features rather
than as a model nobody could explain. That is the specific dishonesty this ADR exists to
prevent, and it is the one a reader would be least likely to catch.

`explain.refit.regenerate_fold` refuses the model outright rather than skipping it quietly, so
`sentinel explain --models xgboost_chain_embeddings` fails with the reason above and exit code
1 rather than producing an empty artifact.

### The minimal public extension, proposed and not taken

For whoever reopens Component 8:

```python
def booster_for(fitted: FittedEmbeddingBooster) -> Any:
    """The live estimator for a fit, for a caller that needs the model rather than a score."""
    return _scorer_for(fitted)
```

Four lines, in `neural/embed.py`, mirroring `neural.train.scorer_for` exactly. No behavioural
change, no artifact change, no change to any number Component 8 published, and it would make
this model TreeSHAP-able immediately.

It is not taken here because Component 8 is closed, and "the change is small" is the argument
every unauthorised change to a closed component has ever been made with. The right time to add
it is when Component 8 is next opened for a reason of its own.

### What explaining it would have required beyond the accessor

Worth recording, because the accessor is not the whole cost and a future implementer should
know:

- **16 of the 46 columns are not interpretable features.** `chain_emb_07 = +0.03` is exactly
  the `feature_127 = +0.04` output section 7 of the brief forbids, and the origin map in
  `explain.definitions` would reject the names today.
- **Aggregating them is defensible but needs stating.** SHAP is additive, so summing the 16
  dimensions into one `chain_identity` contribution is mathematically valid. It is still an
  aggregation rule that would have to be documented, tested, and kept traceable back to the
  per-dimension values — the brief requires all three.
- **The donor network would have to be re-fitted too**, since `embed.fit_fold` takes a
  `FittedNetwork` donor, adding 18 MLP fits to a run that already takes forty minutes.

## Alternatives rejected

**Reach into `neural.embed._scorer_for`.** It would work today, it would take one line, and it
is the move `HANDOFF.md` section 0 exists to prevent. It also creates a dependency from an
open component onto a closed one's private surface, which is the thing that makes "closed"
stop meaning anything.

**Add `booster_for` to Component 8 as part of this component.** Genuinely tempting: a pure
addition, no behaviour change, and the model is Component 8's most interesting derivative.
Rejected because the brief's section 22 and 28 both say the boundary is the point, and because
a component that edits a closed one to improve its own coverage has established that coverage
outranks the boundary. The extension is written down here so the cost of *not* taking it is
one function call, recoverable at any time.

**Reconstruct the booster from `neural_embeddings_*.parquet`.** ADR 0026 already considered
and rejected the equivalent move for Component 9: the embedding table round-trips exactly, but
the XGBoost half is not persisted either, so the reconstruction saves nothing and would require
synthesising a 30-field `FittedNetwork` most of whose values would be fabricated.

**Explain it with a model-agnostic explainer instead, avoiding the estimator.** A permutation
game needs to *call* the model, and `embed.score_window` requires the same private stash. Same
boundary, more compute.

**Drop it from the registry entirely.** Then the artifact would not say why one of Component
9's five candidates is missing, and a reader would have to infer it from an absence.
Section 13 of the brief is explicit that honest unsupported behaviour is better than a fake
explanation — and an unexplained absence is a third thing, worse than either.

## Consequences

- Component 11 covers **4 of Component 9's 5 candidates**. The one it does not cover is the
  experimental one, which ADR 0022 already labels as not for adoption and which HANDOFF is
  explicit is not to be promoted — so nothing a policy component needs is missing.
- `xgboost_chain_embeddings` had the best PR-AUC of any model. Its reasoning is therefore the
  one piece of evidence Component 11 cannot supply, and that limitation is stated in the
  findings document rather than left to the support table.
- The support matrix is machine-readable, so a future UI can render "no explanation available"
  with the reason attached instead of rendering an empty chart.
- If Component 8 is reopened for any reason, adding `booster_for` and re-running
  `sentinel explain` is a five-minute change with no migration.
