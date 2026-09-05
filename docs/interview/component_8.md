# Component 8 — interview defence

Companion to `component_7.md`. Same structure: short answers first, then the technical detail,
then the questions that actually get asked.

The results in §"The result, stated carefully" and in the tables are filled from the production
run recorded in `docs/analysis/neural_models_findings.md`. Nothing here is estimated.

---

## 60-second answer

Components 6 and 7 established that a penalised GLM, XGBoost and LightGBM all land within 0.005
NDE of each other on the same 26 features. That is three very different learners agreeing, which
points at the **feature representation** as the ceiling rather than the estimator.

Component 8 is the fourth learner, and it is the only one that can do something genuinely new:
learn a dense representation of a *categorical entity*. So it does two things at once. It fits the
specified PyTorch network — embeddings for chain, facility type, community area and ZIP,
concatenated with the 30 standardised numeric columns, through 256 and 128 hidden units to a
single logit — and it runs the controls that make the answer interpretable: the same network with
no categoricals at all, the same network with one-hot columns instead of embeddings, four
single-family ablations, and the learned chain vectors handed to Component 7's XGBoost.

Everything runs under Component 5's rolling-origin folds, writes a `PredictionSet`-compatible
artifact, and is scored by `sentinel evaluate`. Component 8 reports **no metric of its own**.

---

## 2-minute answer

The three things worth knowing about how it was built.

**The features the specification asked for did not exist.** Chain, facility type, community area
and ZIP are not in Component 4's table — it is 26 numeric temporal-history columns and nothing
categorical. Facility type and ZIP are in the raw Socrata snapshot, community area only as a
Socrata computed region, and chain nowhere at all. The project's own rule says a missing feature
belongs in Component 4 behind a bumped `feature_definition_version`, and Component 4 was not to be
touched. That conflict was surfaced before any code was written and resolved by building a
**separate, explicitly experimental categorical layer** under `data/processed/neural/`, with
`feature_definition_version` unchanged. The model that carries every C6/C7/C8 comparison —
`neural_numeric_only` — is fitted without any of it.

**Early stopping is the one thing Component 8 does that 6 and 7 did not, and it is where the
temporal argument had to be rebuilt.** A network needs a validation signal to stop on. The two
obvious windows — the fold's calibration span and its test span — are both *later* than
`train_end`, and reading either would break the horizon the artifact declares. So the split is
carved from the **end of the training window**: the last ~15% of training rows, cut on a whole day
so no single date straddles the boundary. `trained_through = fold.train_end` stays literally true,
and `inner_validation_start` is written into the training log so a reader can check rather than
trust.

**`establishment_id` is refused, not merely absent.** It is the obvious thing for an
entity-embedding component to learn and it is the largest leakage surface in the project: a
per-establishment parameter absorbs everything the network learned about that establishment, and
an embedding table cached across folds would carry the future backwards. Component 4 excludes
identity by design and `FORBIDDEN_COLUMNS` enforces it. `chain` — a property of a *group* of
establishments, with membership recomputed inside each fold from that fold's training rows — is
the deliberate substitute.

---

## Deep technical answer

### The estimand has not changed

Same target (a canvass inspection citing ≥1 Priority or Priority Foundation violation), same
eligibility, same 18 folds in two fold sets, same `target_inspection_id` join key, same evaluation
contract. Only the estimator changes — and, for the explicitly labelled experimental models, the
input representation.

`validate._every_model_scored_the_same_rows` enforces that all nine models score an identical id
set, so no comparison in this component is across different populations.

### Architecture

```
chain      (vocab ≤919) ─ Embedding(16) ─┐
facility   (vocab ≤167) ─ Embedding(8)  ─┤
community  (vocab   78) ─ Embedding(8)  ─┼─ concat ─► Linear(→256)
zip        (vocab  ≤71) ─ Embedding(8)  ─┤              BatchNorm1d
                                         │              ReLU
26 features + 4 family indicators ───────┘              Dropout(0.3)
   (imputed, standardised)                              Linear(→128)
                                                        BatchNorm1d
                                                        ReLU
                                                        Dropout(0.3)
                                                        Linear(→1)   ► logit
```

The output is a **logit**. Sigmoid is applied in exactly one place, `neural/predict.py`, where the
artifact needs a probability. `BCEWithLogitsLoss` fuses the sigmoid into the loss for numerical
stability, so applying it early would train badly without raising.

An ablated family **narrows the first layer** rather than zeroing a block. A zeroed block would
still cost parameters and still receive gradient, and "without the chain embedding" would not mean
what it says.

### Preprocessing is deliberately *not* empty — and that is a real difference from C7

| | C6 logistic | C7 boosters | C8 network |
| --- | --- | --- | --- |
| imputation | median / constant-0 | **none** | median / constant-0 |
| scaling | StandardScaler | **none** | StandardScaler |
| NULLs | imputed + family indicator | NaN, routed by a learned split direction | imputed + family indicator |

Component 7's best argument for a booster was that a NULL means "there was no prior canvass" and a
tree can branch on it. **A dense layer has no such option** — a NaN propagates through every
weight and destroys the fit in one backward pass. So the network imputes and the four null-rule
family indicators carry the fact of missingness, exactly as they do for the GLM.

Scaling returns for a measurable reason: on `quarterly-2026Q2`'s training window the widest feature
standard deviation is **18,409×** the narrowest. A tree is invariant to that; a first weighted sum
is not.

Every fitted statistic — medians, scaler mean/scale, vocabularies, chain membership — comes from
the **inner training rows only**, which is stricter than the fold requires. The early-stopping
signal is only honest if the validation rows influenced none of them.

### The tuning protocol, reused not reinvented

`boosting.tuning.tuning_region`, `first_test_start` and `build_inner_folds` are imported
unchanged. ADR 0017 is the contract, and Component 8 adds no second protocol.

The one hyperparameter searched is the learning rate, over a five-point grid spanning two decades
around the specified 1e-3. A grid rather than TPE because the specification asks whether the
result is *sensitive* to the rate — a question a grid answers legibly and a concentrating sampler
answers worse, by leaving the tails unmeasured. Selection is by mean inner-validation PR-AUC, ties
within 1e-6 broken toward the 1e-3 baseline; the tie-break is declared rather than emergent so
that float summation order cannot decide it.

Two fold sets, two studies — the quarterly region contains the `covid_shift` *test* window, so a
shared search would bias the one result the project most needs to keep honest.

### Determinism

Bit-identical for a fixed input, row order, library set, **one torch thread** and **CPU**. Four
mechanisms: the canonical training sort re-applied inside `fit_fold`; `set_num_threads(1)` plus
`use_deterministic_algorithms(True)`; batch order from an explicitly seeded `torch.Generator`
rather than global state; declared sort keys on every output table.

A CUDA device is present on the build machine and **deliberately unused** — GPU reductions are not
bit-reproducible and several backward kernels have no deterministic implementation at all.

Across *seeds* nothing is claimed. It is measured: the primary model is refit under five seeds on
every fold and the spread is written to `neural_seed_variation`.

### The result, stated carefully

Quarterly means, 17 folds, from `sentinel evaluate`:

| model | NDE | ROC-AUC | PR-AUC | Brier | ECE | P@k_1day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **neural_numeric_only** | **0.2482** | **0.6241** | 0.5343 | **0.2355** | **0.0563** | 0.6273 |
| xgboost_chain_embeddings | 0.2444 | 0.6222 | **0.5357** | 0.2374 | 0.0619 | 0.6480 |
| xgboost (C7) | 0.2376 | 0.6188 | 0.5343 | 0.2379 | 0.0621 | 0.6308 |
| lightgbm (C7) | 0.2355 | 0.6177 | 0.5342 | 0.2383 | 0.0644 | **0.6598** |
| logistic (C6) | 0.2326 | 0.6163 | 0.5321 | 0.2382 | 0.0635 | 0.6576 |
| neural_embeddings | 0.2215 | 0.6107 | 0.5233 | 0.2401 | 0.0679 | 0.6217 |

Say it in this order, because the order is what keeps it honest:

**"The network on the same features won, and it is the best-calibrated model in the project."**
NDE 0.2482 against XGBoost's 0.2376, and it wins **12 of 17 folds** — the first time in this
project that a mean improvement and a per-fold improvement agree. Brier 0.2355 and ECE 0.0563
beat the penalised GLM, which **disproves** the pre-registered expectation that a network would
be worse calibrated.

**"The embeddings — the thing the component was for — lost."** 0.2215 against the
no-categoricals control's 0.2482. Every ablation beats the full model. The one-hot control lands
within 0.0009, so the *representation* is not what failed: capacity is. Mean best epoch orders
almost perfectly by parameter count — 10.4 at 41,729 params, 4.0 at 67,985, 2.3 at 337,665.

**"And the win is the same size as its own noise."** The five-seed ROC-AUC spread is **0.0058**;
the advantage over XGBoost is **0.0053**. Its NDE clears XGBoost's seasonality p95 (0.2444), but
XGBoost sits comfortably inside the neural model's own [0.2311, 0.2527]. **Suggestive, not
decisive** — and not a deployment recommendation.

**"But the embeddings helped XGBoost."** `xgboost_chain_embeddings` posts NDE 0.2444 and the best
PR-AUC of any model (0.5357). The representation that hurt the network which learned it improved
the estimator it was handed to.

Three caveats that travel with the headline: `neural_numeric_only` **loses** `precision@k_1_day`
to LightGBM and the GLM; the ordering **inverts** on `covid_shift`, where the worst quarterly
neural model wins; and **43.03% of violations are still surfaced later**, effectively unchanged
since Component 6.

---

## "Why did you choose this?"

**Why a neural network at all, after C7?** Because the roadmap called for it and because entity
embeddings are the one genuinely new capability — not because a network was expected to win. The
prior, written down in STATUS.md before any code existed, was "no material difference".

**Why PyTorch rather than `sklearn.MLPClassifier`?** `MLPClassifier` is already a dependency and
would have cost nothing, but it cannot express an embedding layer. Without one this component
would be a third dense model on the same 26 features, answering a question Component 7 already
answered.

**Why embeddings rather than one-hot?** That is not an assumption here, it is experiment B.
Measured: the one-hot control needs **1,235 indicator columns** to carry the same four families
that 40 embedding dimensions carry. The textbook argument is that the embedding is cheaper and
shares statistical strength across categories; the component measures whether it actually helps.

**Why those embedding dimensions?** The specification names them. They are not arbitrary relative
to cardinality — chain has by far the largest vocabulary and gets 16; the other three are an order
of magnitude smaller and get 8. They were not tuned: searching four widths on a dataset where
three model classes already agree within 0.005 NDE would be tuning noise.

**Why batch size 512?** Named by the specification. It is also a reasonable fit for the data:
the largest training window is ~46k inner-training rows, so an epoch is ~90 steps — enough
gradient updates per epoch to make progress, few enough that 200 epochs is affordable
single-threaded.

**Why AdamW rather than Adam or SGD?** Named by the specification, and the right default here.
The embedding tables hold ~17k of the network's parameters and a rare category's row is touched by
a handful of rows per epoch; Adam's per-parameter step sizes handle that unevenness far better
than one global learning rate. AdamW decouples weight decay from the gradient step, so the
regularisation strength does not depend on the gradient magnitude.

**Why gradient clipping at 1.0?** Same reason, sharpened. An embedding row updated by one unlucky
batch can produce a very large gradient, and one such step can move a well-fitted network a long
way. Clipping the global norm bounds the damage without changing the direction.

**Why `pos_weight` is not used.** Measured prevalence is 52.52% overall and 0.379–0.492 per test
window. There is no imbalance to correct, and weighting a balanced problem shifts every predicted
probability away from the base rate for no ranking benefit — while Component 9 has to calibrate
whatever this component emits. It exists as `neural_pos_weighted`, a named ablation, because "we
tested weighting and it did not help" is only sayable if it was tested.

---

## "Why didn't you choose X instead?"

**An `establishment_id` embedding?** Refused. See ADR 0021 — three reasons, of which the decisive
one is that Component 4 excludes identity by design and a deployed system that ranked an
establishment on *which one it is* rather than *what it has done* would be indefensible.

**UMAP as well as t-SNE?** A new runtime dependency (`numba`, `llvmlite`) for a second view of a
question where the findings say plainly that neither projection is evidence of semantic structure.

**Re-tuning XGBoost for the widened embedding matrix?** That would confound "the embeddings
helped" with "a second search helped". The experiment uses Component 7's frozen parameters
unchanged.

**Early stopping against the fold's calibration window?** It is later than `train_end`, so
`trained_through` would have had to become `calibration_end` — and it would consume the window
Component 9 exists to use.

**Freezing an epoch count from an inner search, C7-style, then refitting on the whole training
window?** Genuinely defensible and it would recover the ~15% of rows the current design gives up.
Rejected because the specification asks for per-fold early stopping and per-fold learning curves,
and doubling an already long run to recover 15% of rows was the worse trade. **This is the
component's clearest self-inflicted limitation and it is recorded as one.**

**A GPU?** Present, unused. Bit-identity is the standard every leakage test in this repository is
written against.

---

## "What went wrong?"

**A fixture bug produced seven false leakage failures.** The first version of
`neural_categoricals_for` assigned chains by row *position*, so shuffling or appending a row
changed which establishment was in which chain. Seven leakage tests failed against correct code.
Component 7's handoff says "when a leakage test fails, suspect the test first"; that advice paid
for itself immediately. The fixture now derives every value from the row's own identity.

**And one of those failures was Component 7's exact bug, reproduced.** The corruption test mutated
rows after `train_end` — which includes the fold's own *test* rows, whose scores are supposed to
change. HANDOFF.md records Component 7 shipping precisely that defect. It now mutates strictly
after `test_end`, and a separate test asserts the broader property that nothing after `train_end`
moves any fitted artifact.

**A real bug, found by a test: `id()` reuse.** The live torch module for each `FittedNetwork` was
kept in a dict keyed by `id()`. CPython reuses an address as soon as the object at it is
collected, and the multi-seed fits go out of scope — so a later fold's record could land on a
collected record's slot and be handed **another fold's network**, scoring a window with the wrong
model and reporting nothing. Caught by a test that constructs a copy of a live record and expects
a refusal. Fixed by retaining the record in the dict value and identity-checking on read. The
training run in flight at the time was killed and restarted.

**The unseen-category test expected the wrong answer.** An unseen chain *name* is not `__UNKNOWN__`
— it is `__INDEPENDENT__`, "not a chain this fold knows about". `__UNKNOWN__` is reserved for a row
with no prior inspection to carry any value from. The code was right and the expectation was
wrong.

---

## "What would you improve?"

**The representation, not the estimator.** Four learners now agree closely. The features that would
plausibly move the number — nearby 311 complaints, weather, the CDPH risk category, statutory
days-overdue — are either not ingested or sit in the raw snapshot outside Component 4's table.
Every one is a Component 4 change behind a bumped `feature_definition_version`, and any of them has
a better prior than a fifth estimator.

**Refit on the full training window after freezing the epoch count**, recovering the 15% the
early-stopping split gives up.

**More seeds, or seed-averaged predictions.** The seed spread is of the same order as the
between-model differences, which is itself the most important operational finding here.

---

## Five-line memory cheat sheet

1. C8 = PyTorch MLP with entity embeddings (chain 16, facility 8, community 8, zip 8) + 30
   standardised numeric columns → 256/BN/ReLU/Drop → 128/BN/ReLU/Drop → 1 logit.
2. The four categoricals **are not in Component 4's table**; C8 built its own experimental as-of
   layer under `data/processed/neural/` and left `feature_definition_version = v1` (ADR 0022).
3. Early stopping validates on the **last 15% of the training window**, never on calibration or
   test, so `trained_through = train_end` stays literally true (ADR 0021).
4. `establishment_id` is **refused**, not absent; `chain` is the substitute, with membership
   recomputed inside every fold.
5. Nine models, 18 folds, one artifact, **no metric** — `sentinel evaluate` reports the numbers.
