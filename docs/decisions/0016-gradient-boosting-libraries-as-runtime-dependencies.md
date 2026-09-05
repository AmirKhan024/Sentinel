# ADR 0016 — XGBoost, LightGBM and Optuna as runtime dependencies

**Status:** Accepted · **Date:** 2026-08-17

## Context

ADR 0015 promoted scikit-learn to a runtime dependency and set the rule this project uses for
buying one. The rule is not "is it convenient" but *what kind of thing is it*:

> Component 5's metrics are *arithmetic over two arrays* … verifiable against a reference
> implementation to floating-point tolerance. An L2 logistic regression is an iterative
> constrained optimisation … Hand-rolling that is not implementing a formula, it is maintaining
> a solver, and **a subtle defect in it would not be visible as a wrong number — it would be
> visible as a slightly worse model, which is indistinguishable from an honest result.**

Component 7 needs three new things: a gradient-boosted tree implementation, a second one, and a
hyperparameter sampler. Component 6's dependency comment already anticipated the first two —
"Component 7 needs the same estimator API" — but the choice still has to be made explicitly,
because this is the largest dependency increase the project has taken and two of the three are
arguably interchangeable.

The project's standing constraint (README.md) is that technologies arrive only with the component
that needs them. Before this ADR the runtime set was eight packages. It is now eleven, and the
three added between them pull in `scipy`, `sqlalchemy`, `alembic`, `colorlog`, `tqdm` and
`greenlet`.

## Decision

### Take xgboost and lightgbm, both, as runtime dependencies

Both, not one. The comparison between them **is** Component 7's question, not an implementation
detail of it. Component 6 established that the 26-feature representation carries linear signal;
Component 7 asks whether a nonlinear learner does better, and a single boosted implementation
answers that only for one library's inductive bias. XGBoost grows depth-wise and LightGBM
leaf-wise, which is a genuine difference in what "a tree of depth 4" means, and the two landing in
the same place is evidence the result is about the data rather than about one library.

They fall on the same side of ADR 0015's line as scikit-learn, and further along it. A boosted
ensemble is a greedy structure search over split points with second-order gradient statistics,
histogram binning, sparsity-aware default directions for missing values, and regularised leaf
weights. A hand-rolled version would not produce a *wrong number*; it would produce a slightly
worse model, and this component's entire output is a comparison of slightly-different model
qualities. A defect there is invisible by construction.

### Take optuna for the search

Component 7 has to select roughly eight hyperparameters per model. The alternatives are a grid,
a random draw, or a sampler that conditions on what it has already seen. The reason to prefer
the last is not that it finds better parameters — with 100 trials on eight dimensions the
difference is modest — but that it is *seeded and reproducible*, which is what makes "these are
the parameters the search chose" a claim someone can re-run rather than an anecdote.

`TPESampler(seed=20260817)` explores the same candidate sequence on every run at a fixed trial
count, and `tests/test_boosting_tuning.py` asserts both halves: same seed reproduces, different
seed does not.

### Pin every estimator to one thread

`n_jobs=1` for both libraries, plus LightGBM's `num_threads=1`, `deterministic=True` and
`force_row_wise=True`. Histogram construction reduces over threads, and a float reduction's
result depends on which thread finishes first, so a multi-threaded fit is reproducible only
approximately. Component 6's standard for "did not move" is bit-identical, and this component
inherits it. `force_row_wise` additionally pins a strategy LightGBM otherwise chooses from a
runtime timing probe — that is, from the machine's load, which is not an input we control.

The measured cost, from `scripts/profile_boosting.py`: 3.57s for a 200-round XGBoost fit and
0.66s for LightGBM on the widest 53,844-row training window. The full 400-trial search took
563.8s and the 54-fit training run 21.4s. Single-threading is affordable here; it would not be
on a dataset an order of magnitude larger, and that is a limitation to carry forward rather than
a property to assume.

### Record every version in the manifest

`xgboost_version`, `lightgbm_version`, `numpy_version`, `sklearn_version` and `blas_threads`.
The determinism claim is the same narrow one ADR 0015 makes: identical predictions for a fixed
input, a fixed row order, a fixed library set and a single thread — **not** across library
versions. Nothing in the repository pins the versions, so the manifest records what was actually
in effect.

## Alternatives rejected

**Hand-roll a gradient-boosted tree.** Attractive because Component 5 hand-rolled its metrics and
the exercise is educational. Rejected for exactly ADR 0015's reason, amplified: a metric can be
cross-checked against a reference to 1e-12, whereas a boosting implementation can only be checked
against *how good the resulting model is*, which is the quantity under study. The check and the
claim would be the same measurement.

**Ship only one booster.** Attractive because it halves the dependency surface and the runtime.
Rejected because the two libraries' growth strategies differ enough that a single result would be
a fact about one library. The measured outcome vindicates this: on the quarterly folds XGBoost and
LightGBM finish 0.0021 apart in NDE, which is itself the finding — the two agree, so the ceiling
is the representation rather than the learner.

**Add CatBoost as a third.** Attractive because it handles categoricals natively and often wins
benchmarks. Rejected because Component 4's contract has no categorical features — all 26 are
counts, day-deltas, rates and booleans — so its distinguishing capability is inert here. It would
be a third dependency bought for nothing measurable.

**Use scikit-learn's `HistGradientBoostingClassifier` instead of either.** Attractive because it
is already a dependency, is NaN-native, and would have cost zero new packages. Rejected because it
is a single implementation with a narrower tuning surface, and choosing it would answer "does *a*
boosted model beat the GLM" while forfeiting the cross-library agreement that makes the answer
credible. It is the honest fallback if the dependency budget ever has to shrink, and worth naming
as such.

**Grid search or random search instead of Optuna.** A grid over eight dimensions is combinatorially
hopeless at any useful resolution; random search is defensible and would have cost no dependency.
Rejected because random search's reproducibility rests on the caller threading a seed through
correctly at every call site, whereas a seeded sampler owns that, and because the trials table
Optuna's structure makes natural is what turns the search into an auditable artifact.

**Hyperopt or scikit-optimize.** Both would work. Optuna was chosen because the project
specification names it, it is the most actively maintained of the three, and its ask-and-tell
structure made the `TrialResult` record straightforward to build. This is a weak preference and
the decision would not change materially under either substitute.

**Multi-threaded fits, accepting approximate reproducibility.** Attractive because it would cut
the search from 9.4 minutes to perhaps two. Rejected because Component 6 established
bit-identity as this project's standard for "unchanged", and `tests/test_boosting_leakage.py`
asserts it on every leakage test. Downgrading to `np.allclose` would mean a genuine leak producing
a 1e-9 shift could no longer be distinguished from thread scheduling.

## Consequences

- The runtime dependency set is eleven packages; the transitive set grew by twelve.
- **Every fit in this project is single-threaded.** That is a deliberate correctness-over-speed
  trade and it will not scale to a much larger snapshot without revisiting this ADR.
- Determinism is claimed only within a fixed library set. A version bump may change every number
  in `docs/analysis/boosting_models_findings.md`, and the manifest is what makes that detectable.
- `mypy` treats all three as untyped (`ignore_missing_imports`), so fitted estimators are wrapped
  in the `boosting.models.FittedBooster` facade and every value crossing that boundary is
  converted explicitly — the same pattern ADR 0015 established for scikit-learn.
- Component 8's neural baseline will need PyTorch, which is a larger dependency than all three of
  these together. This ADR's reasoning applies there and should be re-stated rather than assumed.
