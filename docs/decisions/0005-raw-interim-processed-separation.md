# ADR 0005 — Separate raw / interim / processed layers, with raw append-only

**Status:** Accepted · **Date:** 2026-08-15

## Context

Data pipelines that transform files in place become impossible to debug: when a
downstream number looks wrong, you cannot tell whether the source was wrong or
the transformation was.

## Decision

Three directories with distinct, enforced meanings:

```text
data/raw/         exactly what the source returned. Append-only. Never edited.
data/interim/     intermediate results of multi-step transformations.
data/processed/   analysis- and model-ready outputs.
```

Component 1 writes only to `data/raw/`. The other two are created empty, with a
`.gitkeep`, and stay empty until a component needs them.

Raw files are **timestamped and never overwritten**:

```text
food_inspections_20260815T145703Z.parquet
manifest_food_inspections_20260815T145703Z.json
```

Parquet is gitignored; manifests are committed.

## Rationale

* **Debuggability.** With raw preserved, any downstream discrepancy can be
  traced by re-running the transformation against the original bytes.
* **The source is live.** The Chicago dataset changes daily. Overwriting raw
  would destroy the only record of what the data looked like at training time,
  which makes a past model's behaviour unreproducible. That matters for a
  project whose eventual output influences real inspection scheduling.
* **Timestamped filenames are cheap versioning.** No content-addressed store, no
  DVC, no object storage. The format sorts lexicographically into chronological
  order, so "latest" is `sorted(...)[-1]`.
* **Committing manifests but not data** keeps a full, diffable history of what
  was ingested and when, without putting hundreds of megabytes into Git.

Resolving "which file is latest" is a *read-time* concern, handled by
`latest_parquet()` in the query layer. Ingestion deliberately does not maintain
a `latest.parquet` pointer, because that would be mutable state inside an
append-only layer.

## Alternatives rejected

* **One file, overwritten each run** — destroys history and makes past results
  unreproducible.
* **DVC or an object store** — real versioning infrastructure, unjustified for a
  single dataset on one machine. Revisit if the data outgrows local disk.
* **Committing raw Parquet** — bloats the repository with data that is
  reproducible from the API and already described by a committed manifest.

## Consequences

* Disk usage grows with each run. Old raw files can be deleted manually; the
  committed manifests preserve the record that they existed.
* Downstream code must resolve which raw file it is reading, and should record
  that path. `latest_parquet()` is the convenience path, not the only one.
* No component may write to `data/raw/` except an ingestion component.
