# ADR 0007 — Establishment identifier scheme and its stability limits

**Status:** Accepted · **Date:** 2026-08-16

## Context

Component 2 emits `establishment_id`, the key every later component joins on.
The identifier has to satisfy three things that pull against each other:

1. **Deterministic.** The same input and code version must always produce the
   same ids, or every downstream artifact silently becomes unreproducible.
2. **Traceable.** When a merge looks wrong, a human needs to get from the id to
   the underlying records without running the pipeline.
3. **As stable as possible under data growth.** The dataset is live and will be
   re-ingested. Ids that churn on every refresh would force a full rebuild of
   everything downstream.

The third is the hard one, because cluster membership is a function of the whole
snapshot: new data can merge two clusters or split one.

## Decision

```text
establishment_id = "EST-" + zero-pad(min(inspection_id over the cluster), 11)
```

for example `EST-00000067435`. Inspection ids are compared **numerically**; a
non-numeric id raises rather than silently sorting as a string. Zero-padding to
11 digits sits comfortably above the 7-digit ids in the data.

Separately, each establishment carries **`cluster_content_sha256`** — the hash of
its sorted member node ids. This is explicitly *not* the identifier. It is a
change detector.

## Alternatives considered

| scheme | deterministic for one snapshot? | stable under new data? | traceable? |
|---|---|---|---|
| Sequential (`1, 2, 3…`) | only with a canonical sort | **no** — one new establishment shifts everything after it | no |
| `sha256` of member keys | yes | **no** — one added inspection changes the cluster's hash | no |
| **`min(inspection_id)`** | **yes** | **mostly** | **yes** |

The anchor scheme wins on the two properties the others fail. Inspection ids are
assigned monotonically over time, so appending a later snapshot introduces only
*larger* ids and cannot change which row is a cluster's earliest. And the id
points at exactly one raw row, so `EST-00000067435` can be looked up to see the
original name and address that anchored it — a hash cannot.

## The stability limit, stated plainly

Content-derived ids are stable only for a fixed input snapshot. Two things break
stability across snapshots:

1. **Merges.** If a later snapshot supplies evidence joining `EST-A` and
   `EST-B`, the merged cluster takes the smaller anchor. The other id is
   **retired**.
2. **Splits.** New data can trip a cluster invariant; the non-anchor half gets a
   new id.

A third breaks it without any data change at all:

3. **Rule changes.** Editing a normalization or matching rule changes cluster
   membership and therefore ids. `normalization_version` is recorded in the
   manifest so a diff between two outputs can be attributed to code rather than
   to data.

> `establishment_id` is a deterministic function of one raw snapshot, identified
> by its sha256. It is **not a durable primary key across snapshots** and must
> not be used as one without a crosswalk.

`cluster_content_sha256` makes this tractable rather than merely disclaimed.
Diffing two runs on `(establishment_id, cluster_content_sha256)` gives:

- same id, same hash → cluster unchanged
- same id, different hash → membership moved; downstream features need recomputing
- id absent from the later run → retired by a merge

## Consequences

- Determinism is verified directly rather than assumed: resolving a seeded random
  permutation of all 314,245 input rows produces a byte-identical
  `inspection_id → establishment_id` mapping. This rests on four properties —
  content-hashed node ids, canonically ordered candidate pairs, connected
  components re-labelled by their minimum member, and ids derived from the
  earliest inspection.
- A validation check asserts every id matches `^EST-\d{11}$`, is used by exactly
  one cluster, and anchors on its own earliest member.
- Ids are readable in logs and reports, and sort chronologically by first
  inspection, which is a small but real convenience.
- **Future work, not built:** a cross-snapshot crosswalk table mapping retired
  ids to their successors. Until it exists, re-ingesting and re-resolving means
  downstream artifacts keyed on `establishment_id` must be rebuilt, not
  incrementally updated.
- A cluster whose earliest inspection is later removed upstream would change id.
  The Chicago dataset is append-mostly and this has not been observed, but it is
  not prevented.
