# ADR 0011 — The processed layer holds model-ready tables

**Status:** Accepted · **Date:** 2026-08-16

## Context

ADR 0005 created three data layers with enforced meanings:

```text
data/raw/         exactly what the source returned. Append-only.
data/interim/     intermediate results of multi-step transformations.
data/processed/   analysis- and model-ready outputs.
```

Three components later, `data/processed/` is still empty. Component 1 writes raw;
Components 2 and 3 both write interim, each with a short justification for why
their output is *not* model-ready:

- Component 2 emits an establishment crosswalk — a key mapping that Component 3
  consumes to build something else.
- Component 3 emits labels — genuinely useful, but not trainable on their own,
  because nothing can be learned from a target column with no features beside it.

Component 4 is the first output that does not have that excuse. It is one row per
prediction opportunity, carrying 26 features and the label, with a documented
temporal guarantee. A modelling component can read it and fit.

Leaving it in `interim/` would mean the processed layer never fills up, and its
definition would quietly become decorative — the exact erosion ADR 0005 exists to
prevent, arrived at from the opposite direction.

## Decision

**The as-of feature table is written to `data/processed/features/`, and this ADR
records the criterion for what belongs there.**

A table belongs in `processed/` when all four hold:

1. **One row per unit of decision.** The grain matches the question being asked,
   not an intermediate structure.
2. **Features and labels are both present**, on the same row, so the table is
   directly trainable without a further join to something else.
3. **The temporal contract is settled and enforced.** For Sentinel this means the
   as-of boundary is applied and validated, not left to the consumer.
4. **Column roles are explicit.** A consumer can tell, without reading code,
   which columns are inputs and which are outcomes.

Component 4's output satisfies all four:
`(establishment_id, inspection_date)` grain; `FEATURE_COLUMNS` and
`LABEL_COLUMNS` on the same row; a strictly-before boundary checked on every row
(ADR 0010); and the two sets enumerated in code and asserted disjoint.

The `.gitignore` whitelist is extended so `processed/` manifests are committed on
the same terms as `raw/` and `interim/` ones — the Parquet is ignored, the
provenance record is kept.

## Alternatives rejected

**Write to `data/interim/features/`.** Consistent with Components 2 and 3, and it
would have avoided this ADR entirely. Rejected because it is consistent for the
wrong reason: those two are interim because their outputs genuinely are not
model-ready, not because interim is the default. Applying the layer definitions
by habit rather than by test is how a layer scheme decays.

**Wait for Component 5 or 6 to be the first processed writer.** Component 5 is
temporal evaluation and Component 6 is modelling; neither produces a new table of
this kind — they consume this one. Deferring would mean the first processed table
arrives with an unstated rationale, or never.

**Redefine `processed/` to mean "the final artifact".** Would leave everything
before the last component in interim, which makes the distinction useless.

**Split: features to processed, labels left behind in interim.** Purer
separation, but it forces every consumer to redo a join that Component 4 has
already validated, and each of them could get the join wrong. The label columns
are carried and clearly marked instead.

## Consequences

- `data/processed/` is now in use, with a written test for future components
  rather than a judgement call each time.
- The layer boundary stays meaningful: Components 2 and 3 remain interim, and the
  reason is now a stated criterion rather than an accident.
- **A model-ready table is a sharper object than a clean one.** Cleanliness was
  never the criterion — Component 2's output is clean and belongs in interim.
- Future components producing a *different* model-ready table (for example a
  differently-grained one for the scheduling optimiser) should write to
  `processed/` alongside this one, under their own slug.
- Component 5 must not confuse "in processed" with "safe to split randomly". The
  table's temporal guarantee is about *feature construction*; honest evaluation
  additionally requires chronological splitting, which is Component 5's job.
