# Data contract — neural predictions

**Produced by:** Component 8 (`sentinel train-neural`, `sentinel tune-neural`)
**Layer:** `data/processed/predictions/` (scores and diagnostics), `data/processed/tuning/`
(sweep trials), `data/processed/neural/` (the experimental categorical input, its own contract)
**Consumed by:** Component 5 (`sentinel evaluate --predictions`); Component 9 will consume the
scores as the thing it calibrates
**Design rationale:** ADR 0014 (predictions are outputs), ADR 0017 (the tuning protocol), ADR 0020
(PyTorch, matplotlib, CPU determinism), ADR 0021 (what may be embedded), ADR 0022 (the
experimental categorical layer), ADR 0023 (community area)


> **Component 9 status (2026-08-24):** this artifact was consumed by Component 9 and is
> **unchanged and byte-identical** — Component 9 re-executed the fits behind a bit-identity
> gate rather than rewriting anything here. The calibrated scores are a separate artifact,
> `calibrated_predictions_<stamp>.parquet`; see `calibrated_predictions.md`. The
> probabilities in *this* file remain uncalibrated.
---

## 1. Identity and file naming

```
data/processed/predictions/neural_predictions_<stamp>.parquet     # the artifact C5 reads
data/processed/predictions/neural_training_log_<stamp>.parquet    # one row per (model, fold)
data/processed/predictions/neural_epoch_log_<stamp>.parquet       # one row per epoch
data/processed/predictions/neural_embeddings_<stamp>.parquet      # the learned vectors
data/processed/predictions/neural_seed_variation_<stamp>.parquet  # the reproducibility experiment
data/processed/predictions/manifest_neural_predictions_<stamp>.json
data/processed/tuning/neural_sweep_trials_<stamp>.parquet
data/processed/tuning/manifest_neural_sweep_trials_<stamp>.json
```

**The slug differs from Components 6's and 7's on purpose.** `baseline_predictions`,
`boosted_predictions` and `neural_predictions` are three separate files under three separate
slugs, so each earlier benchmark stays visible and byte-identical. "Did Component 8 improve on
Component 7?" is only answerable if Component 7's answer was not overwritten.

The sweep trials live in a **different directory** for the reason ADR 0018 gives: their
`pr_auc` is a *validation* number measured on windows that are training data for every fold the
selected learning rate is then used on. Filed beside the predictions it would eventually be read
as a result.

## 2. The models

Nine, in one registry, each answering a declared experiment.

| model | learner | encoding | experiment |
| --- | --- | --- | --- |
| `neural_embeddings` | MLP | embedding | **A** — the specified network; the primary model |
| `neural_numeric_only` | MLP | none | **A** — the fair-comparison control against C6/C7 |
| `neural_onehot` | MLP | one-hot | **B** — do learned representations beat indicators? |
| `neural_no_chain` | MLP | embedding | **C** — ablation |
| `neural_no_facility_type` | MLP | embedding | **C** — ablation |
| `neural_no_community_area` | MLP | embedding | **C + D** — the fairness-relevant ablation |
| `neural_no_zip` | MLP | embedding | **C** — ablation |
| `neural_pos_weighted` | MLP | embedding | class-weighting ablation |
| `xgboost_chain_embeddings` | XGBoost | embedding | embeddings → XGBoost |

**`neural_numeric_only` is the model any C6/C7/C8 claim rests on.** It sees exactly the 30 matrix
columns Components 6 and 7 see — 26 features plus the four null-rule family indicators — and no
categoricals at all. Every categorical-bearing model is reported beside it, never in place of it.

**`xgboost_chain_embeddings` is not a neural model.** It is Component 7's XGBoost, with Component
7's frozen parameters, on Component 7's tree matrix, widened by the 16 chain-embedding dimensions
`neural_embeddings` learned **on the same fold's training rows**. Re-tuning it would confound "the
embeddings helped" with "a second search helped".

## 3. Score semantics

**`score` is `sigmoid(logit)` = P(target = 1). Higher means higher predicted risk** of a Priority
or Priority Foundation citation. Component 5 assumes that direction and probes it live; inverting
it would produce a plausible, confidently wrong result rather than an error.

The network emits a **logit**. The sigmoid is applied in exactly one place — `neural/predict.py`,
where a probability is required for the artifact. It is never applied before the loss, because
`BCEWithLogitsLoss` fuses it in for numerical stability and applying it twice trains badly without
raising.

**`is_probability = True` means "this is a `sigmoid` output", not "this is calibrated."** A
network trained under BCE is typically overconfident, and a sigmoid over an unbounded logit
saturates more readily than a penalised GLM's link. Component 8 measures ECE and MCE through
Component 5 and **corrects none of it**. Component 9 owns calibration.

## 4. `trained_through`

Always `fold.train_end`. This needs more explanation than it did in Components 6 and 7, because
Component 8 is the first component that **early-stops**.

A network needs a validation signal to stop on. The two obvious candidates — the fold's
calibration window and its test window — are both **later than `train_end`**, and reading either
would mean the fit had learned from a date later than the horizon it declares. In the calibration
window's case it would also consume the window Component 9 exists to use.

So the split is internal: the training window's days are sorted, the **last ~15% of rows** become
the early-stopping validation set, and the rest is fitted. The cut falls on a whole day, so no
single date straddles it — two inspections days apart share almost all of their as-of history, and
splitting a day would leak near-duplicate rows across the boundary in the ordinary
machine-learning sense.

`inner_validation_start` is emitted in the training log so the claim is checkable, and every
observed value is at or before its fold's `train_end`. Measured share across the 18 folds:
0.1501–0.1514.

**The cost is real and is not hidden:** the weights kept are the best validation epoch's, so the
final model is fitted on ~85% of its fold's training rows rather than all of them. Component 7
avoided that cost by freezing a round count and refitting on everything. Component 8 does not,
because the specification asks for per-fold early stopping and per-fold learning curves, and
doubling an already long run to recover 15% of rows was judged the worse trade. It is a genuine
difference from Component 7's protocol and it is stated rather than argued away.

## 5. Preprocessing: median/constant imputation, then standardisation

This is the one place Component 8 differs materially from Component 7, and the difference is
forced rather than chosen.

| | Component 6 | Component 7 | Component 8 |
| --- | --- | --- | --- |
| imputation | median / constant-0 | **none** | median / constant-0 |
| scaling | `StandardScaler` | **none** | `StandardScaler` |
| NULL handling | imputed + family indicator | NaN routed by a learned split direction | imputed + family indicator |
| matrix columns | 30 | 30 | 30 (+ embeddings or indicators) |

**Why scaling returns.** A tree splits on thresholds and is invariant to any monotone transform,
so Component 7 fitted no scaler at all. A dense layer's first weighted sum is not invariant.
Measured on `quarterly-2026Q2`'s training window, the widest feature standard deviation is
**18,409×** the narrowest (`days_since_first_inspection` at 1,545.03 against
`missing_no_prior_inspection` at 0.0839). Without standardisation the optimiser would spend its
early epochs undoing the units.

**Why imputation returns.** There is no NaN-native path for a dense layer: a NaN propagates
through every weight and destroys the fit in one backward pass. So the network imputes, and **the
four null-rule family indicators are how the fact of missingness survives** — exactly as they do
in Components 6 and 7. The rules are copied deliberately, not blindly: Component 6's justification
for them (a nullable boolean's median sits 0.0056 from flipping across folds) is a fact about the
data, not about logistic regression.

A network can additionally learn an *interaction* between an indicator and its imputed column,
which is the open question Component 6's findings recorded and could not answer itself.

**Every statistic is fitted on the inner training rows only** — stricter than the fold requires,
because the early-stopping signal is only honest if the validation rows influenced neither the
scaler nor the vocabulary. `validate._preprocessing_comes_from_inner_train` re-derives every
median from the source window and compares to 1e-9.

## 6. The categorical block

Four families, from Component 8's own experimental layer (its own contract; ADR 0022):

| family | embedding dim | vocabulary, largest fold |
| --- | ---: | ---: |
| `chain` | 16 | 919 |
| `facility_type` | 8 | 167 |
| `community_area` | 8 | 78 |
| `zip` | 8 | 71 |

**`establishment_id` is refused, not merely absent.** It is the obvious thing to embed and it is
the largest leakage surface in the project; `FORBIDDEN_COLUMNS` and a second closed allowlist both
reject it, and `validate._entity_columns_are_never_identity` restates the refusal at runtime. See
ADR 0021.

Vocabularies are refitted **per fold on training rows only**. Index 0 is `__UNKNOWN__` and its
vector is **learned**, not masked — 401 rows genuinely have no prior inspection, so index 0
receives gradient and learns the "never seen before" offset. Measured out-of-vocabulary rates on
the test windows: 0.0000–0.0204 per family per fold.

Vocabulary order is **sorted**, never insertion-ordered, because insertion order is row order.

## 7. Schemas

### `neural_predictions`

Column-for-column identical to `baseline_predictions` and `boosted_predictions` except the
trailing version column, which is what lets `evaluation.contract.read_predictions` consume all
three without translation.

| column | type | note |
| --- | --- | --- |
| `target_inspection_id` | Utf8 | |
| `score` | Float64 | `sigmoid(logit)`, in [0, 1] |
| `model_name` | Utf8 | one of the nine |
| `model_version` | Utf8 | |
| `fold_set` | Utf8 | `quarterly` \| `covid_shift` |
| `fold_id` | Utf8 | |
| `trained_through` | Date | never null; always `fold.train_end` |
| `is_probability` | Boolean | never null; True means `sigmoid`, not calibrated |
| `neural_definition_version` | Utf8 | |

### `neural_training_log`

One row per (model, fold). Carries the fold dates, the row counts on both sides of the inner
split, `inner_validation_start`, the architecture widths, `parameter_count`, `vocabulary_total`,
`train_nan_cells`, `seed`, `learning_rate`, `pos_weight`, `best_epoch`, `final_epoch`,
`learning_rate_changes`, `stop_reason` and `saturated_scores`.

Fields with no meaning for `xgboost_chain_embeddings` are **null rather than zero**: a zero
`best_epoch` would read as "peaked immediately" and a zero `parameter_count` as "no model".

### `neural_epoch_log`

One row per (model, fold, epoch): `train_loss`, `validation_loss`, `learning_rate`,
`is_best_epoch`. This is the learning curve, as data.

⚠ **`validation_loss` is an in-sample number.** It is measured on rows carved from inside the
training window. It is the early-stopping and `ReduceLROnPlateau` signal, and it is a result in
exactly the sense Component 7's inner-fold PR-AUC is — which is to say, not one.

### `neural_embeddings`

One row per (model, fold, family, category, dimension): `category_index`, `dimension`, `value`.
Written for the representative model only. This is a **representation, not an explanation** —
attribution is Component 11's.

### `neural_seed_variation`

One row per (model, fold, seed): `pr_auc`, `roc_auc`, `best_epoch`. The reproducibility
experiment. These are test-window metrics, and they are here rather than in Component 5's output
because they measure *the fitting procedure's* variance rather than a model's performance; the
headline metrics come from `sentinel evaluate` as they do for every other component.

### `neural_sweep_trials`

One row per (study, learning rate, inner fold), in `data/processed/tuning/`. Carries `pr_auc`,
`best_epoch`, `selected`, `region_start`, `region_end`. **No number in this table is a result.**

## 8. Determinism

The claim is the same narrow one Components 6 and 7 make, plus one measurement they did not need.

**Bit-identical output** for: this feature table, this categorical table, this row order, this
library set, **one torch thread**, **CPU**. Four mechanisms carry it:

1. Canonical training sort (`inspection_date`, `target_inspection_id`), re-applied inside
   `fit_fold` rather than trusted to the caller.
2. `torch.set_num_threads(1)` and `torch.use_deterministic_algorithms(True)` — the direct
   analogue of Component 7's `n_jobs=1`.
3. Batch order drawn from an explicitly seeded `torch.Generator`, not global state; `random`,
   `numpy` and `torch` seeded together.
4. Declared sort keys on every output table.

**A CUDA device is present on the build machine and deliberately unused.** GPU reductions are not
bit-reproducible and several backward kernels have no deterministic implementation, so
`use_deterministic_algorithms(True)` would raise on them. ADR 0020.

**Across seeds nothing is claimed — it is measured.** `neural_seed_variation` refits the primary
model under five seeds on every fold. A network has strictly more seed-sensitivity than a booster
(initialisation, batch composition, dropout masks on top of summation order), and reporting one
seed's number as *the* result would be the most misleading thing this component could do.

## 9. Guarantees a consumer may rely on

1. Exactly one score per (model, fold, test row); every test row of every fold scored once.
2. **Every model scores an identical `target_inspection_id` set** — enforced by
   `validate._every_model_scored_the_same_rows`, so no comparison here is across populations.
3. Every score is finite and within [0, 1].
4. `trained_through` is never null and never exceeds `fold.train_end`.
5. No fit read any row dated after its fold's `train_end` — not for weights, not for the scaler,
   not for a vocabulary, not for the stopping epoch, not for the learning rate.
6. The file is readable by `evaluation.contract.read_predictions` without translation.

## 10. Known limitations

1. **Probabilities are uncalibrated** and must not be read as risk levels until Component 9.
2. **The final fit uses ~85% of its fold's training rows** (§4). A genuine difference from
   Component 7's protocol.
3. **The categorical layer is experimental** and is not a Component 4 feature table (ADR 0022).
4. **Community area is a candidate demographic proxy**, included only with a matched ablation, and
   a better score with it is not grounds for retaining it (ADR 0023).
5. **`establishment_id` is not embedded**, so this component cannot claim to have learned
   per-establishment risk (ADR 0021).
6. **The chain family's ceiling is its coverage**: 22.70% of rows belong to a chain, so roughly
   three rows in four receive the shared `__INDEPENDENT__` vector.
7. **Determinism holds within a fixed library set only.** A torch version bump may move every
   number; the manifest records the version so it is detectable.
8. **Single-threaded CPU training will not scale** to a snapshot an order of magnitude larger.
9. **Importances are not emitted** for the embedding-fed booster's appended columns as an
   attribution — attribution is Component 11's, and a 16-dimensional learned vector is exactly the
   kind of input a split-gain number cannot explain.

## 11. Reproducing

```bash
uv run sentinel build-neural-categoricals --report      # the experimental input layer
uv run sentinel tune-neural --report                    # the learning-rate sweep; then freeze
uv run sentinel train-neural --report                   # every model, every fold
uv run sentinel evaluate --predictions data/processed/predictions/neural_predictions_<stamp>.parquet --report
```

`train-neural` does not require re-running the sweep: the learning rates are frozen literals in
`neural/definitions.py`, and `TUNED_HYPERPARAMS_PROVENANCE` names the study artifact that produced
them.
