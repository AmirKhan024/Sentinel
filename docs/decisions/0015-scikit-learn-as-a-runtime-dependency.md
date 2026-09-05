# ADR 0015 — scikit-learn as a runtime dependency

**Status:** Accepted · **Date:** 2026-08-17

## Context

The project rule, stated in `pyproject.toml` and followed for five components, is that
a technology is introduced only when the component that needs it is actually being
built. Runtime dependencies stood at six: httpx, pydantic, pydantic-settings, polars,
pyarrow, duckdb.

scikit-learn has been present since Component 5, but as a **dev-only test oracle**,
with an explicit comment: "Component 5 implements its own metrics so that no runtime
dependency is added for them; the suite cross-checks every one against scikit-learn.
Nothing in src/sentinel imports it."

Component 6 fits models. That comment stops being true, and the question is whether to
make it true again by hand-rolling the estimator, or to promote the dependency.

There is a real precedent pulling toward hand-rolling: Component 5 *did* hand-roll
ROC-AUC, PR-AUC, Brier, log-loss, ECE, MCE and precision/recall@k rather than import
them, and cross-checks all of them against sklearn in the test suite. That was the
right call and it is worth asking whether the same reasoning extends here.

## Decision

**scikit-learn moves to `[project.dependencies]`, and numpy is declared explicitly
alongside it.**

The reasoning is a difference in kind, not in degree. Component 5's metrics are
*arithmetic over two arrays* — a few dozen lines each, fully specified by a formula,
and verifiable against a reference implementation to floating-point tolerance. An L2
logistic regression is an iterative constrained optimisation: it needs a solver, a
convergence criterion, numerically stable log-sum-exp, and correct handling of a
rank-deficient design matrix (this one has condition number 71.8 and a 0.9888-correlated
pair). Hand-rolling that is not implementing a formula, it is maintaining a solver, and
a subtle defect in it would not be visible as a wrong number — it would be visible as
a slightly worse model, which is indistinguishable from an honest result.

Three further reasons:

- **Component 7 needs the same estimator API surface.** XGBoost and LightGBM both ship
  sklearn-compatible wrappers, and `Pipeline` / `ColumnTransformer` is how the
  train-only preprocessing guarantee is expressed mechanically rather than by
  convention. Promoting now avoids promoting one component later anyway.
- **The dependency is already installed and already trusted.** It has been the
  correctness oracle for every Component 5 metric since that component shipped.
  Promoting it adds no new supply-chain surface; it changes which section of
  `pyproject.toml` names it.
- **numpy is a direct import, so it is declared.** It was previously an undeclared
  transitive dependency of scikit-learn and pyarrow. Component 6 imports it by name,
  and an undeclared direct import is exactly the kind of thing that breaks on a
  minimal install.

The dev group keeps using scikit-learn as a test oracle. The comment there is updated
rather than deleted, because the cross-check it describes is still running and still
valuable.

### mypy

scikit-learn ships no `py.typed` marker, so under `strict = true` every symbol it
exports is `Any`. An override is required:

```toml
[[tool.mypy.overrides]]
module = ["sklearn.*"]
ignore_missing_imports = true
```

The consequence is that type safety stops at the sklearn boundary, so the boundary is
made explicit rather than pretended away. The fitted estimator is treated as **opaque**
and wrapped in a typed facade, `modeling.models.FittedModel`, which carries the
coefficients, the scaler statistics, the iteration count and the convergence flag as
ordinary typed Python. Values crossing out of sklearn are converted explicitly —
`[float(v) for v in proba[:, 1]]`, not a bare return — because `strict` enables
`warn_return_any` and a bare return would silently type as `Any`.

### The reproducibility caveat, recorded here because the dependency causes it

Fitting the same fold twice on the same rows in a different order produces coefficients
differing by up to **7.049e-09** (measured, fold `quarterly-2022Q2`, 23,346 rows).
`StandardScaler` accumulates variance incrementally and the lbfgs gradient is a BLAS
reduction; both depend on float summation order. Sorting the training rows canonically
makes the fits bit-identical.

So the project's determinism claim is narrowed, precisely and in writing: **predictions
and coefficients are bit-reproducible for a fixed input, a fixed row order, a fixed
library set and a fixed BLAS thread count** — not across library versions or thread
counts. `sklearn_version`, `numpy_version` and `blas_threads` are therefore recorded in
`BaselineModelManifest`, because nothing else in the repository pins them and the
coefficients depend on them.

Related, and stated so no reader is misled: `random_state=42` has **no effect** on
`solver="lbfgs"`, which has no stochastic component. It is recorded because it
documents intent and because a future switch to `saga` would make it load-bearing. The
thing that makes Component 6 deterministic is the sort, not the seed.

## Alternatives rejected

**Hand-roll the logistic regression, as Component 5 hand-rolled its metrics.** Zero new
runtime dependencies, and consistent with the strictest reading of the project rule.
Rejected because it maintains a solver rather than implementing a formula (above), and
because the failure mode is silent: a subtly wrong optimiser produces a slightly worse
model, which cannot be distinguished from an honest baseline. The whole value of a
baseline is that later components can be measured against it.

**Keep scikit-learn dev-only and import it lazily inside function bodies**, so the
wheel does not declare it. Rejected as dishonest packaging: the code would not run
without it, and a dependency the code requires belongs in the dependency list. It would
also defer the failure from install time to first use.

**Use statsmodels instead**, which gives inference — standard errors and p-values on
the coefficients. Genuinely attractive for an interpretable baseline. Rejected because
§4 of the findings shows why: with condition number 71.8 and a 0.9888-correlated pair
splitting into +1.99 / −1.47, per-coefficient standard errors would invite exactly the
misreading the findings document warns against, and Component 11 owns attribution
properly. It would also be a second new dependency and would not serve Component 7.

**Add scipy explicitly too.** It is a scikit-learn dependency and is not imported
directly by any Sentinel module. Not declared, on the same rule that made numpy's
declaration necessary: declare direct imports, not transitive ones.

## Consequences

- Runtime dependencies go from six to eight. The `pyproject.toml` comment explaining
  the minimalism is updated to name Component 6 and cite this ADR, so the next reader
  sees why the rule was applied rather than abandoned.
- Type safety stops at the sklearn boundary, and the boundary is a named, tested facade
  (`FittedModel`) rather than an implicit `Any` spreading through the package.
- The project's determinism claim is now conditional and the conditions are written
  down and recorded per-run in the manifest. This is a narrowing of an earlier, broader
  claim, and it is the accurate one.
- Component 7 inherits the dependency and the estimator API, and needs no further
  packaging decision.
- The test suite continues to cross-check Component 5's hand-rolled metrics against
  scikit-learn. That cross-check is now checking a runtime dependency against itself in
  one direction only — sklearn is the oracle, the hand-rolled implementations are the
  subject — which is still the useful direction.
