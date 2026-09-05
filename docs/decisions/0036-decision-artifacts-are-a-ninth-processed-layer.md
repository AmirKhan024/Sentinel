# ADR 0036 — Decision artifacts are a ninth processed layer

**Status:** Accepted · **Date:** 2026-08-26

## Context

Component 13 emits a table whose rows say *this establishment is the fourth inspection on
Tuesday*. Nothing in the eight existing processed layers has that grain, and three of them are
close enough to be dangerous.

`predictions/` holds a calibrated probability per (model, fold, scored row). A recommendation is
keyed by the same establishment and the same fold, carries a score column, and would sit in that
directory without anything looking wrong. `evaluation/` holds a `precision_at_k` per (model,
fold); Component 13 emits one per (policy, model, fold, capacity). `fairness/` holds
group-conditional measurements addressed, in Component 12's own words, to this component.

Every layer since ADR 0024 has had to avoid the same collision, and the argument has been
identical each time: two authoritative answers to the same cell, filed in one directory with no
convention saying which is which, is how a project starts quoting the flattering one.

There is a second reason here that the earlier layers did not have. A recommendation is the only
artifact in this project that is an **instruction to a person**. Everything upstream is a
description of the world; this says what to do about it. Descriptions and instructions change
for different reasons, are wrong in different ways, and are read by different people.

## Decision

**Component 13 writes to `data/processed/policy/`, a ninth processed layer whose grain is a
decision: one establishment, one operating period, one capacity assumption, one policy, together
with the mechanism that put it in the queue or kept it out.**

### A recommendation is not a prediction, and the directory says so

A prediction is a belief: *this establishment has a 0.62 chance of being cited*. A
recommendation is an instruction: *send an inspector here on Tuesday*. The first changes when the
model changes; the second changes when capacity changes, when a policy changes, or when a
supervisor overrides it — none of which touch the model at all.

Component 5's `evaluate --predictions` would refuse every table in this layer, which is the
mechanical form of the same statement: nothing here scores anything.

### Eleven tables, not one, and three of the splits are load-bearing

`inspection_recommendations` carries the whole prediction universe rather than only the queue.
It would be smaller to write the selected rows, and it would make the most important question
unanswerable — *why was this establishment not inspected?* A queue-only artifact can say who was
chosen; only a universe-grained one can say who was considered.

`policy_selection_allocation` is separate because how many slots the reserve was offered, how
many the risk block had already filled, and how many were finally granted are three different
numbers. Only together do they distinguish "the floor was satisfied" from "the floor was
ignored" from "there were not enough eligible establishments to satisfy it".

`policy_comparison` carries the opportunity cost as a column on the same row as the coverage
number, rather than in a table a reader has to join. The single most misreadable thing this
component could produce is a coverage figure without its price beside it.

### Nothing here may be joined onto a feature table

The prohibition every processed layer since ADR 0011 carries, and it is sharpest here. A
recommendation is downstream of every model in this project. Joined back onto training rows it
would make the system's own past decisions an input to its future ones — which is precisely the
feedback loop Component 12 measured in the `__UNKNOWN__` group, and which this component exists
to keep visible rather than to close.

### The manifest is keyed to the recommendation table

Following the convention every component has used: the manifest sidecar describes the table that
is the component's answer. For Component 12 that was the group metrics; here it is the queue.

## Alternatives rejected

**Write recommendations into `predictions/`.** The cheapest option and the most dangerous. The
two artifacts share a key, a fold and a score column, so a consumer reading the directory by
prefix would find both and have no convention for telling a belief from an instruction. ADR 0024
and ADR 0028 each rejected the same shortcut for the same reason.

**Write into `evaluation/`, since the comparison is metrics.** Half the tables are metrics and
half are decisions, so this would split the component across two layers by accident of column
type rather than by grain. It would also give `precision_at_k` two producers, which Component 5
has been the sole owner of since ADR 0013.

**One wide table instead of eleven.** A single table at the recommendation grain cannot hold a
per-policy pooled frontier row or a per-group audit row without nulling most of its columns, and
a reader could not tell a genuinely absent value from a value that does not apply at that grain.

**Keep advisories only in the manifest, as Component 12 did.** Rejected on volume. Component 12
emitted 13 advisory strings; a policy run produces one per (policy, capacity) cell that was inert
or costly, and "which cells were inert, and which gave up citations" is a question with a shape
that a list of formatted strings answers badly. `policy_advisories` is a table *and* the strings
still travel in the manifest.

## Consequences

- `Settings.policy_processed_dir` resolves to `data/processed/policy/`, with the layer's grain,
  its two near-collisions and the join prohibition stated in the property's docstring.
- Eleven tables, each with a declared schema, a declared total sort key and an entry in
  `docs/data_contracts/policy_decisions.md`. Column order is the contract.
- The manifest sidecar sits beside `inspection_recommendations_<stamp>.parquet` and carries the
  frozen policy grid, the selection rule, the selected model, the winner or its absence, and the
  `does_not_establish` boundary.
- `tests/test_policy_writer.py` asserts that no table in the layer carries `target` or
  `target_status`, so a future edit that wanted the label would have to change the contract to
  get it.
- A tenth layer will be needed if Component 14 emits a routed schedule; that is a different
  grain again — a decision plus a time and a vehicle — and this ADR does not license filing it
  here.
