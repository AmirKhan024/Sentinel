# ADR 0020 — PyTorch and matplotlib as runtime dependencies, and CPU-only determinism

**Status:** Accepted · **Date:** 2026-08-18

## Context

ADR 0016's final consequence names this decision and refuses to make it in advance:

> Component 8's neural baseline will need PyTorch, which is a larger dependency than all three of
> these together. This ADR's reasoning applies there and should be re-stated rather than assumed.

So it is re-stated here. ADR 0015 set the rule, and the rule is not "is it convenient" but *what
kind of thing is it*: a formula can be hand-rolled and verified against a reference to
floating-point tolerance; a solver cannot, because a subtle defect in one is invisible as a wrong
number and visible only as a slightly worse model — which is indistinguishable from an honest
result.

Component 8 needs two new things: a framework that can define an embedding layer, backpropagate
through it and run an optimiser, and something that can draw two figures the project
specification requires. Before this ADR the runtime set was eleven packages.

There is a third decision entangled with the first, and taking it implicitly would be the failure
STATUS.md warned about — "GPU or multi-threaded training will not reproduce bit-for-bit, and that
is a decision to make explicitly in an ADR rather than to discover." The build machine has an
NVIDIA RTX 4050 with 6 GB of VRAM. It is deliberately unused.

## Decision

### Take PyTorch as a runtime dependency

It falls on the same side of ADR 0015's line as scikit-learn and the boosters, and further along
it than either. What Component 8 needs is reverse-mode automatic differentiation over a graph
containing sparse embedding lookups, a fused numerically stable sigmoid-plus-binary-cross-entropy,
batch normalisation with separate training and evaluation behaviour, AdamW's decoupled weight
decay, gradient-norm clipping across a heterogeneous parameter set, and a plateau-triggered
learning-rate schedule.

Hand-rolling that is not implementing a formula. It is maintaining an autodiff engine, and the
failure mode is exactly the one ADR 0015 identified: a wrong gradient does not raise, it trains to
a slightly worse optimum. This component's entire purpose is to compare a neural model against
XGBoost, and a comparison whose neural side might be silently mis-differentiated answers nothing.

The CPU wheel is taken (`torch 2.13.0+cpu`), which is what `uv` resolved on this platform and
which is also what the determinism decision below requires.

### Take matplotlib, and take it as a runtime dependency rather than a dev one

The project specification requires a training/validation loss curve with the early-stopping point
marked, and a 2-D projection of the learned chain embeddings. Those are deliverables of the
component, not developer conveniences, and they are produced by `sentinel train-neural` — a
runtime path. Filing the library under `dev` would mean the shipped command could not run.

It is a smaller and better-behaved dependency than PyTorch: pure Python plus a C extension, no
solver, no numerics that could be silently wrong. The reason to take it rather than hand-roll SVG
is different from the reason to take PyTorch, and worth stating separately — a hand-rolled plot
would not be *wrong*, it would just be worse, and the effort is better spent on the leakage suite.

`Agg` is selected before `pyplot` is imported, so a figure renders identically on a headless
runner and on this machine.

### t-SNE comes from scikit-learn; UMAP is refused

The specification asks for "t-SNE or UMAP". scikit-learn is already a dependency and ships
`sklearn.manifold.TSNE`. `umap-learn` would be a new runtime dependency pulling `numba` and
`llvmlite` to produce a second view of the same question, and the question is not one where two
views would change any conclusion — the findings document says plainly that neither projection is
evidence of semantic structure.

### Pin the CPU, one thread, and deterministic algorithms

Every fit runs on the CPU with `torch.set_num_threads(1)` and
`torch.use_deterministic_algorithms(True)`. Batch order is drawn from an explicitly seeded
`torch.Generator` rather than from global state, and `random`, `numpy` and `torch` are all seeded
together in one function.

This is the direct analogue of Component 7's `n_jobs=1`, taken for the same reason: a float
reduction over threads depends on the order the threads finish in, so a multi-threaded fit is
reproducible only approximately — and this project's standard for "did not move" is bit-identity.
A leakage test that asserts a fold's predictions are *unchanged* when the future is mutated is
worthless if the predictions move by 1e-7 on every re-run for unrelated reasons.

**The GPU is refused, not unavailable.** CUDA reductions are not bit-reproducible, several
backward kernels have no deterministic implementation at all, and
`use_deterministic_algorithms(True)` would raise on them. The trade is stated plainly: this
component's full run takes tens of minutes on one CPU thread and would take a small fraction of
that on the RTX 4050, and the project takes the slower, reproducible option here exactly as it did
in Component 7.

### Measure the residual rather than asserting it away

Bit-identity is claimed only for a fixed input, a fixed row order, a fixed library set, one thread
and CPU. It is **not** claimed across seeds, and a network has strictly more seed-sensitivity than
a booster: weight initialisation, batch composition and dropout masks on top of the summation
order a booster already has.

So the final configuration is refit under five seeds on every fold and the spread is written to
`neural_seed_variation_<stamp>.parquet`. Reporting one seed's number as *the* result, when the
seed-to-seed spread is of the same order as the difference between models, would be the single
most misleading thing this component could do.

## Alternatives rejected

**Hand-roll the network in numpy.** A two-layer MLP's forward pass is a formula; its backward pass
through an embedding table, batch normalisation and a fused BCE-with-logits is not. Rejected
because a defect would present as a worse model, not an error — ADR 0015's rule, applied.

**Use scikit-learn's `MLPClassifier` instead.** It is already a dependency and would have cost
nothing. Rejected because it cannot express an embedding layer, and an embedding layer is the one
genuinely new capability Component 8 has. Without it this component would be a third dense model
on the same 26 features and would answer a question Component 7 already answered.

**Take PyTorch as a dev dependency and import it lazily.** Rejected for the reason ADR 0015
rejected the same shape for scikit-learn: `sentinel train-neural` is a shipped command, and a
shipped command that fails on a clean install is broken.

**Use the GPU and claim "approximate reproducibility".** Rejected. The project's leakage tests are
written against bit-identity, and weakening that standard to accommodate a faster fit would
weaken every safety claim Components 6 and 7 make, not just Component 8's.

**Use the GPU only for the multi-seed experiment**, where run-to-run variation is the subject
anyway. Rejected as a false economy: two execution paths would mean the seed spread was measured
under a different numerical regime than the headline fits, and the comparison would be confounded.

**Add `umap-learn` alongside t-SNE.** Rejected; see above.

**Add `tensorboard` or any experiment tracker.** Rejected. The epoch history is written to a
Parquet table with a declared schema, which is the project's existing convention for every other
artifact and is queryable without a server.

## Consequences

- The runtime dependency set is thirteen packages. PyTorch's CPU wheel is by far the largest
  single artifact the project installs.
- **Every fit in this project remains single-threaded**, now including a neural network. Component
  8's full nine-model run over eighteen folds is the longest-running command in the repository.
- `torch` ships `py.typed`, so unlike scikit-learn, xgboost, lightgbm and optuna it needs no
  `ignore_missing_imports` override. Two narrow narrowings were still required under
  `--strict`: `nn.Sequential.__call__` returns `Any`, and `nn.ModuleList.__getitem__` returns
  `Module` rather than `Embedding`. Both are handled in `neural/net.py` rather than by widening
  the mypy configuration.
- `matplotlib` has no `py.typed` marker; the `_matplotlib()` helper returns `Any` and the figure
  code is confined to `neural/figures.py` so that `Any` cannot spread.
- Determinism is claimed only within a fixed library set. A torch version bump may move every
  number in `docs/analysis/neural_models_findings.md`, and the manifest records the version so
  that it is detectable.
- The GPU on the build machine is unused and the manifest says so explicitly, so a future reader
  does not assume a CUDA run produced these numbers.
- Component 9 (calibration) inherits a set of raw, uncalibrated sigmoid outputs that saturate more
  readily than a penalised GLM's — a network's overconfidence is the thing that component will
  have to correct.
