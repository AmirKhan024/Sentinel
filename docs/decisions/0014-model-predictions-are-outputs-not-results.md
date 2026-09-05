# ADR 0014 — Model predictions are outputs, not results

**Status:** Accepted · **Date:** 2026-08-17

## Context

ADR 0011 defined what belongs in `data/processed/`: a table is model-ready when it
has one row per unit of decision, features and labels together, a settled temporal
contract, and explicit column roles. ADR 0013 then split the processed layer in two —
`features/` for trainable tables, `evaluation/` for measurements about models — and
gave `evaluation/` a three-part membership test.

Component 6 produces a third kind of thing, and it fits neither box.

A prediction table has one row per `(model, fold, target_inspection_id)` carrying a
score. It is not model-ready: it has no features, and joining it onto a training table
is precisely the leakage the project is built to prevent. But it also fails ADR 0013's
test for `evaluation/`, on two of three criteria. Criterion 2 says an evaluation
artifact "is produced **after** a model has been scored, so it cannot be an input to
that model without circularity" — predictions are produced *before* scoring and *are*
the evaluator's input. Criterion 3 says it "is read by people and reports, not by a
fitting routine" — predictions are read by a program, not a person.

ADR 0013 already anticipated this and said Component 6 "writes its predictions as a
*separate* artifact under its own slug", but it did not say where, or state the rule.

There is a second question this ADR has to settle, because it is the same question in
a different guise. Component 5 shipped `contract.read_predictions` with the docstring
"Load prediction artifacts written by a later component", and `PREDICTION_METADATA_COLUMNS`
exists solely to make an on-disk prediction file self-describing. Neither is called
from anywhere in the repository. The seam was designed and left unwired, and wiring it
exposes three latent defects that are latent only because no fitted model existed yet:

1. `_append_metrics` hardcodes `model_version=rankers.RANKER_VERSION`. Correct while
   the only producers are six rankers sharing a version; wrong the moment a second
   producer exists.
2. `Observations.horizon_rejections` is declared, documented and never populated, so
   the check `scores_respect_the_decision_point` **cannot fail**. A `trained_through`
   violation currently surfaces as a coverage failure instead — the right outcome for
   the wrong reason, and an unhelpful message.
3. `metrics.PROBABILITY_METRICS` is fully implemented, tested against scikit-learn,
   and never emitted, because it is only reachable when a producer declares
   `is_probability=True` and none did.

## Decision

**Component 6 writes to `data/processed/predictions/`, and this ADR records that the
processed layer holds three kinds of thing, not two.**

```text
data/processed/features/      model-ready tables. Trainable. ADR 0011's four tests.
data/processed/predictions/   model outputs. Never trainable, never joined onto features.
data/processed/evaluation/    measurements about models. Never trainable.
```

A table belongs in `processed/predictions/` when:

1. Its grain is a *model output* — one row per scored decision per model per fold.
2. It is produced **after** fitting and **before** scoring, so it is the evaluator's
   input and never the fitter's.
3. It carries enough provenance to be scored without its producer being present:
   which model, which version, which fold, and what the model was allowed to know.

The primary artifact is `baseline_predictions_<stamp>.parquet`, with one manifest
keyed to it, matching the convention Components 2 and 5 use for multi-table output.
`baseline_coefficients` and `baseline_training_log` share the stamp.

### `trained_through` is the training end, not the calibration end

`validate_predictions` requires `trained_through <= fold.calibration_end`, and the six
built-in rankers declare `calibration_end`. Component 6 declares **`fold.train_end`**.

The contract's ceiling is the calibration end because fitting a calibrator on the
calibration window is what that window is for. Component 6 fits no calibrator and
must not touch that window at all — it exists so Component 9 has somewhere to put one.
Declaring `calibration_end` would claim a horizon the model did not use, and would
leave Component 6's own validator unable to detect a future calibration-window leak.
So `trained_through == fold.train_end` is asserted as an error-severity check, and the
unused calibration end is recorded separately in `baseline_training_log` as
`calibration_end_unused`.

This diverges from the example snippet in `HANDOFF.md` §13, which shows
`calibration_end`. That snippet anticipates the post-Component-9 calibrated case; for
an uncalibrated model it overstates the horizon.

### The three latent defects are corrected as the smallest compatible change

- `contract.py` gains `PredictionHorizonError(PredictionContractError)`, raised from
  the `trained_through` branch. A **subclass**, so every existing
  `except PredictionContractError` keeps working unchanged, and the discriminated case
  becomes available to a caller that wants it.
- `_append_metrics` takes `model_version` and `is_probability` as parameters.
- `run_evaluation` gains `predictions_path: Path | None = None`, defaulting to exactly
  today's behaviour, with a regression test asserting the no-flag path produces an
  identical metrics table.

Probability metrics are emitted as `brier`, `log_loss`, `ece`, `mce` when a producer
declares `is_probability=True`. **`precision`, `recall` and `f1` are deliberately not
emitted**: they require a threshold, `METRICS_SCHEMA` has no threshold column, and the
only place to record 0.5 would be `k_name` — which would be a lie about what `k`
means. `metrics.precision_recall_f1` says itself that 0.5 "has no operational meaning"
here and that a threshold is genuinely needed only by the Component 16 deferral gate.

Nothing else in Component 5 changes: not fold cadence, the 2018-07-01 anchor,
calibration placement, the nine contract rules, simulation semantics, score direction,
metric definitions, or `EVALUATION_DEFINITION_VERSION`.

## Alternatives rejected

**Write predictions into `data/processed/evaluation/`.** One fewer directory, and they
are evaluation-adjacent. Rejected because ADR 0013's criteria exclude them on two of
three counts, and because `evaluation/` carries the rule "nothing in here may ever be
joined onto a training table" — a rule that reads as being about *measurements*.
Diluting that directory with a table of a different kind weakens the one sentence that
protects it.

**Write predictions into `data/processed/features/`.** Rejected for exactly the reason
ADR 0013 gave: co-location is an invitation to join, and a score joined onto a
training table is the most damaging leakage available.

**A second evaluator inside Component 6** (`modeling/evaluate.py` importing Component
5's pure metrics and looping folds itself). Zero lines of Component 5 changed, which
is genuinely attractive. Rejected because it leaves `read_predictions` and
`PREDICTION_METADATA_COLUMNS` as permanent dead code, puts two evaluation loops in one
repository, and duplicates the k-derivation, the row builders, the `SENSITIVITY_EXCLUDED`
policy and the metrics schema — four drift surfaces, each of which would eventually
disagree with Component 5 about a number. `evaluation/writer.py` says in its own
docstring that the metrics table is long "so Component 6 can add a model … without
reshaping the table". Adding a model to that table is the documented purpose.

**Component 6 calls `run_evaluation` internally.** Rejected outright: it inverts the
dependency, puts a modelling component in charge of invoking the evaluator, and makes
fit → evaluate → refit possible in a single process. That loop is what ADR 0013
exists to prevent.

**Fix the three latent defects by matching on the exception message** rather than
adding a subclass. Rejected as brittle in the ordinary way, and because a discriminated
failure mode deserves a type.

**Leave `trained_through = calibration_end` to match the rankers and the HANDOFF
snippet.** Consistent, and contract-legal. Rejected because it is a false declaration:
it claims the model saw a quarter it never read, and it disables the one check that
would catch a future calibration leak from inside Component 6.

## Consequences

- The processed layer now holds three kinds of thing with a stated test for each,
  rather than a judgement call when the next component arrives.
- **Nothing in `predictions/` may be joined onto a feature table**, stated here so a
  future component argues against an ADR rather than against a convention.
- A prediction artifact is self-describing: `read_predictions` can reconstruct a
  `PredictionSet` with its declared horizon from the file alone, with no access to the
  code that produced it.
- `EvaluationManifest` gains `predictions_path` and `predictions_sha256`, so an
  evaluation run is pinned to the exact prediction bytes it scored. Without this the
  manifest's promise — "exactly what was the model allowed to know?" — stops being
  answerable as soon as a fitted model is involved.
- The check `scores_respect_the_decision_point` can now actually fail, and a test
  proves it does. A check that cannot fail is worse than no check, because it reads as
  coverage.
- `.gitignore` needs no change: `!data/processed/**/` and
  `!data/processed/**/manifest_*.json` already cover a new subdirectory, so the Parquet
  is ignored and the provenance record is committed.
- Component 9 will add calibrated predictions. It should write its own slug under the
  same layer and reuse this contract rather than mutating Component 6's artifact.
