# ADR 0013 — Evaluation results are artifacts, not model inputs

**Status:** Accepted · **Date:** 2026-08-16

## Context

ADR 0011 defined what belongs in `data/processed/`: a table is model-ready when
it has one row per unit of decision, features and labels together, a settled
temporal contract, and explicit column roles. Component 4's feature table meets
all four and lives at `data/processed/features/`.

Component 5 produces six tables — fold definitions, metrics, discovery curves,
simulation summaries, seasonal effects and sensitivity bands. None of them meets
that test, and none should. They are *results*: measurements about a model, not
inputs to one. A discovery curve has one row per inspection slot per schedule;
a metrics table has one row per `(fold, model, metric)`. Neither is trainable and
neither should ever be trained on.

They also cannot go in `interim/`. ADR 0005 defines interim as intermediate
results of a multi-step transformation — something a later step consumes on its
way to producing the real output. These are the real output. Nothing downstream
transforms them further; they are read by humans, by a future demo frontend, and
by whatever writes the final report.

So there are two ways to get this wrong. Putting results in `features/` would
invite a later component to join them onto the training table, which is how a
test score becomes a feature. Putting them in `interim/` would say they are
half-finished when they are the deliverable.

## Decision

**Component 5 writes to `data/processed/evaluation/`, a sibling of
`processed/features/`, and this ADR records the distinction between the two.**

```text
data/processed/features/     model-ready tables. Trainable. ADR 0011's four tests.
data/processed/evaluation/   measurements about models. Never trainable.
```

A table belongs in `processed/evaluation/` when:

1. Its grain is a *measurement*, not a decision — a fold, a metric, a schedule, a
   curve point.
2. It is produced **after** a model has been scored, so it cannot be an input to
   that model without circularity.
3. It is read by people and reports, not by a fitting routine.

The six tables are `evaluation_folds`, `evaluation_metrics`, `discovery_curves`,
`simulation_summary`, `seasonality` and `sensitivity`. They share one stamp per
run and **one manifest**, keyed to `evaluation_folds` as the primary artifact —
the same convention Component 2 uses for its three tables.

`evaluation_folds` is the primary artifact because it is the one a reviewer
needs first: it answers "exactly what data was the model allowed to know when
this score was produced?" without opening any other file.

`.gitignore` already whitelists `data/processed/**/manifest_*.json`, so the new
directory needs no rule change: the Parquet is ignored, the provenance record is
committed.

## Alternatives rejected

**Write to `data/processed/features/` alongside the feature table.** One fewer
directory. Rejected because co-location is exactly the invitation to join. A
component that finds `evaluation_metrics` next to `as_of_features` and joins them
has produced a feature derived from a test score, which is the most damaging form
of leakage in the project and the hardest to spot afterwards.

**Write to `data/interim/evaluation/`.** Would match Components 2 and 3.
Rejected for the reason ADR 0011 gave in the opposite direction: those are
interim because their outputs genuinely are intermediate, not because interim is
the default. These are terminal.

**A new top-level layer, `data/results/`.** Honest about the distinction, and it
would put a fourth meaning into a three-layer scheme that ADR 0005 deliberately
kept small. The sibling directory carries the same information at less cost.

**One combined wide table instead of six.** Simpler to find. Rejected because
the six have genuinely different grains — a fold, a metric, a curve point, a
schedule, a month, a band — and forcing them together would mean nulls in most
columns of most rows, which is how a schema stops meaning anything.

**One manifest per table.** More granular provenance. Rejected for consistency
with Component 2, and because the six are produced by a single run from a single
input: six manifests would repeat the same checksum six times and invite them to
drift apart.

## Consequences

- The processed layer now has two kinds of thing in it, with a stated test for
  which is which, rather than a judgement call each time.
- **Nothing in `evaluation/` may ever be joined onto a training table.** Stated
  here so a future component has to argue against an ADR rather than merely
  against a convention.
- Component 6 writes its predictions as a *separate* artifact under its own slug
  and hands them to Component 5's contract; it does not write into
  `evaluation/`, and it does not read from it while fitting.
- The demo frontend (Component 21) reads `discovery_curves` and
  `simulation_summary` directly. Storing curves at full resolution rather than
  summarizing them is what makes that possible — 373,986 rows compress to 868 KB,
  so the cost of keeping the whole curve is negligible against the cost of a
  reader having to take the headline number on trust.
- A future re-run overwrites nothing: each run gets its own timestamp, and
  `latest_parquet(dir, prefix=...)` resolves the newest, exactly as for every
  other component.
