# ADR 0027 — The calibrator's input is the logit recovered from the committed probability

**Status:** Accepted · **Date:** 2026-08-24

## Context

Platt scaling fits a one-variable logistic regression, and it needs a real-valued input on a scale
where a logistic link is the right shape. Conventionally that input is the base model's **raw
decision score** — the linear predictor of a GLM, a booster's margin, a network's pre-sigmoid logit.

No component in this repository persists one. Every prediction artifact stores the probability and
only the probability: Component 6 writes `predict_proba(...)[:, 1]`, Component 7 the booster's
probability, Component 8 `torch.sigmoid(logits)` — whose docstring says "The sigmoid is applied here
and nowhere else." There is no margin, logit or raw-score column in any writer schema.

Component 9 re-executes those fits to obtain calibration-window scores (ADR 0026), so it has the
fitted objects in memory and could take either route:

- **(i) the native margin** — reachable through public attributes with no change to any closed
  component: `pipeline.decision_function`, XGBoost's `output_margin=True`, LightGBM's
  `raw_score=True`, and `neural.train.scorer_for` before the sigmoid.
- **(ii) the recovered logit** — `logit(p)` computed from the committed probability.

The two are the same quantity in exact arithmetic. They are not the same artifact.

## Decision

### The calibrator input is the recovered logit

```python
def logit(p: float) -> float:
    return math.log(p) - math.log1p(-p)
```

Written that way rather than as `math.log(p / (1 - p))`: for p near 1 the subtraction `1 - p`
cancels catastrophically, while `log1p` is accurate there by construction.

- **Platt** receives `logit(p)` as an `(n, 1)` array.
- **Isotonic** receives `p` itself. Isotonic is invariant to any strictly monotone
  reparametrisation, so the choice is free, and `p` keeps the persisted breakpoints readable.

### Why (ii) and not (i): what the calibrated artifact is a function of

This is the decisive argument, and it is about provenance rather than numerics.

With the recovered logit, applying the frozen calibrator to the test window is a **pure function of
the already-committed `score` column plus two fitted floats**. The calibrated artifact can be
re-derived, audited or re-applied years later from artifacts alone — no live model, no re-execution,
no dependency on ADR 0026's regeneration having been correct.

With the native margin, the calibrated artifact would be a function of a value that exists nowhere
on disk. Persisting the margin would fix that, but only by making Component 9's output depend on the
regeneration being faithful — which is the thing the bit-identity gate is trying to *prove*, not
something to assume. Recovering the logit from the committed probability keeps the two independent.

### The measurement, and the surprise in it

The choice is not left as reasoning. `scripts/profile_calibration.py` captures the native margin
alongside the recovered logit for all 34,261 calibration-window rows per model and compares them
(`calibration_findings.md` §5):

| model | max abs difference | mean | rows over 1e-9 |
|---|---|---|---|
| `lightgbm` | 1.776e-15 | 1.367e-16 | 0 |
| `logistic_regression` | 2.602e-13 | 1.985e-16 | 0 |
| `xgboost` | 1.393e-06 | 7.122e-08 | 33,922 |
| `neural_numeric_only` | 2.615e-05 | 8.148e-08 | 33,898 |

The two float64 models round-trip to the last bits. The other two do not, and **the cause is not
the arithmetic in `logit()` — it is that those models compute in float32.** XGBoost's margin is
float32; the network's forward pass, logit and sigmoid are float32 throughout. Float32 carries about
1.2e-7 of relative precision. A logit is passed through a sigmoid, rounded to float32, widened to
float64 for storage, then inverted; the round trip cannot recover more than float32 carried, and the
residual is amplified in the tails where the sigmoid is flattest.

**The persisted probability is the float64 image of a float32 sigmoid, so `logit(score)` is the best
recovery obtainable from what exists on disk.** The discrepancy measures the base model's precision,
not the recovery's fidelity — which strengthens the decision rather than weakening it, because the
native margin has no more information than the probability does at float32.

The practical magnitude is negligible for the purpose: 2.6e-5 in logit space moves a calibrated
probability by roughly `2.6e-5 × slope × p(1−p)`, i.e. under 1e-5 — four orders of magnitude below
the ECE differences this component measures.

### No probability saturates, so nothing is clamped

Not one score of the 34,261 per model sits at exactly 0.0 or 1.0, so `logit(p)` is finite
everywhere. `LOGIT_EPSILON = 1e-15` (the value and rationale of
`evaluation.metrics.LOG_LOSS_EPSILON`) exists as a guard, and `logit_clamped_rows` is recorded in
the manifest. On this snapshot it is 0; a non-zero value on a later snapshot is a signal worth
seeing rather than swallowing.

### The margin is still captured, as a check

`calibration/basescores.py` captures the native margin anyway and writes it to
`calibration_base_scores_*.parquet`. A warn-severity check,
`recovered_logit_matches_the_native_margin`, asserts the maximum absolute difference stays below
**1e-4** and reports the observed maximum.

The plan pre-declared 1e-9 for this check. **That threshold was wrong** — set from a float64
expectation, it would have fired on 33,898 neural rows and 33,922 xgboost rows, every one of them
correct behaviour. It is set from the measurement instead: 1e-4 sits comfortably above the observed
2.6e-5 and remains tight enough to catch what the check exists for. A double sigmoid, a sign flip or
a mis-join produces an O(1) discrepancy, not an O(1e-5) one.

### The chain, stated once so no component has to infer it

```
base model  →  probability p          (committed by Components 6/7/8; float64 of a float32 or
                                       float64 computation, uncalibrated)
p           →  logit(p)               (Component 9, for Platt only; isotonic takes p)
logit(p)    →  calibrated probability (the frozen calibrator)
```

**The sigmoid is never applied twice, and an already-calibrated probability is never calibrated
again.** The calibrated artifact's `model_name` is `"<base>_<method>"`, so a second pass would have
to name a model that is not in the candidate registry, and the registry guard rejects it.

## Alternatives rejected

**Calibrate the native margin and persist it.** The textbook choice, and rejected on provenance:
the calibrated artifact would depend on a column no earlier component wrote, making Component 9's
output unreproducible from the committed artifacts. It would also require adding a margin column to
Components 6–8's schemas — a change to three closed components.

**Calibrate the probability directly with Platt (logistic on `p` rather than on `logit(p)`).**
Attractive for its simplicity, and rejected because it is the wrong link. A logistic regression on
`p ∈ (0, 1)` fits a sigmoid of a probability, which is not the family that contains the identity
map — so a perfectly calibrated model could not be left alone by its own calibrator. On `logit(p)`,
slope 1 and intercept 0 recover the input exactly, which is what makes the fitted slope
interpretable as a measure of over- or under-confidence.

**Use `sklearn.calibration.CalibratedClassifierCV`.** The library's own answer, and it implements
both methods. Rejected because it wants an *unfitted* estimator and performs its own
cross-validation to generate the calibration set — which is precisely the design ADR 0025 replaces
with a temporal split. Handing it a pre-fitted estimator with `cv="prefit"` would work, but it would
also hide the calibrator's input transform inside a wrapper at exactly the point this ADR is trying
to make explicit.

**Add temperature scaling as a third method.** Named alongside Platt and isotonic in Component 8's
`BLOCKED_EXPERIMENTS` list. Rejected for now because temperature scaling is Platt with the intercept
fixed at 0 and the slope constrained — a strict special case of what is already fitted, so the Platt
fit's coefficient already answers what a temperature would have been. If the fitted intercept turns
out to be negligible across folds, temperature is the simpler model and a later component may adopt
it; the parameters needed to make that call are persisted in `calibrator_parameters_*.parquet`.

**Clamp probabilities into `[ε, 1−ε]` before taking the logit, unconditionally.** Rejected as a
default because it would silently alter scores on a snapshot where nothing saturates, and because
the clamp count is more useful as a reported signal than as an invisible correction. The guard
exists; it is recorded when it fires.

## Consequences

- `calibrator_parameters_*.parquet` carries an `input_transform` column (`logit` for Platt,
  `identity` for isotonic), so the mapping is reproducible from the artifact without reading this
  ADR.
- A calibration slope of 1.0 and intercept of 0.0 means "already calibrated", and the fitted Platt
  coefficients are directly readable as over/under-confidence. Slope < 1 is overconfidence.
- The Platt calibrator applied to its own output must return slope 1.0 by construction; this is
  asserted as a warn-severity self-check.
- **Any future component that adds a base model must emit a probability, not a margin**, or state
  its own input transform. The recovery assumes the committed score is `sigmoid(margin)`.
- The float32 finding is worth carrying forward: `xgboost` and the network do not have float64
  precision in their probabilities, so no downstream component should treat a difference below about
  1e-6 in those models' scores as meaningful.
