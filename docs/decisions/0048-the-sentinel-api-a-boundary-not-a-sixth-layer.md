# ADR 0048 — The Sentinel API: a boundary, not a sixth layer

**Status:** Accepted · **Date:** 2026-08-26

## Context

Components 1 through 14 produce a deterministic pipeline's worth of artifacts: timestamped
Parquet tables under `data/processed/`, each with a JSON manifest sidecar, each documented in
`docs/data_contracts/`. Nothing in the repository exposes them to a product consumer. A frontend,
or any client that is not a Python process importing `sentinel.*` directly, has no way to ask
"what does Sentinel recommend for this establishment" without reading internal Parquet files and
internal Python modules by hand — resolving the latest run itself, knowing which columns compose
into which product concept, and re-deriving the scope rules that keep one fold's decision from
being read as another's.

That gap is not a missing feature of any single component. It is a missing *boundary*: a place
where "the artifacts Components 1-14 wrote" turns into "a validated, paginated, documented HTTP
contract a frontend developer can build against without reading this repository's internals."

The obvious risk in closing that gap is closing it the wrong way — by teaching the boundary how
to compute something. A boundary that re-derives a recommendation, re-applies an override, or
picks a "current" fold when a caller under-specifies one has stopped being a boundary and started
being a sixth pipeline layer with none of the first five's discipline: no ADR, no validator, no
manifest, no determinism proof.

## Decision

**The Sentinel API is a read/write HTTP interface over Components 1-14's existing artifacts. It
computes nothing, and it is not a numbered component.**

### It is an interface, not a layer, in ADR 0042's sense

ADR 0042 names five layers — model, policy, recommendation, schedule, execution — and four
boundaries between them that must never collapse into each other. The Sentinel API adds no sixth
entry to that list. Every field a response carries was already written by Component 11, 13 or 14;
the API's own code contributes no score, no rank, no mechanism, no reason code and no schedule
date. Where a response composes several artifacts into one JSON object (see the establishment
detail endpoint), the fields keep the name and the source of the layer that produced them —
`recommendation.decision_reason` is Component 13's, `schedule.schedule_reason` is Component 14's,
and the two are never merged into one ambiguous `reason` field, for the same reason ADR 0047
refuses one generic override table.

### It is cross-cutting infrastructure, not a slot in the roadmap

This repository's roadmap already names a "Component 15" (OR-Tools routing, blocked on missing
inspector/travel-time data — ADR 0019, ADR 0043) and a "Component 16" (a deferral/human-review
gate, the next component the roadmap describes as implementable). The Sentinel API is neither. It
does not attempt routing, and it does not gate anything. It is built and documented as
infrastructure that sits *beside* the numbered pipeline — under `src/sentinel/api/`, described in
its own ADRs and its own data contract, referenced from `README.md` after the roadmap table
rather than as a row inside it. The roadmap's existing numbering is left untouched.

### The service layer is genuinely new code, and it is the only new code

No general-purpose "load the latest run and query it" reader existed anywhere in the repository
before this ADR — every existing loader (`policy/inputs.py`, `scheduling/inputs.py`) takes an
explicit path for one pipeline run's internal use and returns a raw Polars frame. The API's
`services/artifacts.py` module is what resolves "latest run for this table", reads it, and checks
a caller's scope; every other service function is composition on top of it. Column contracts are
never re-declared: every response schema mirrors the corresponding `writer.SCHEMAS` entry field
for field, and `sentinel.query.duckdb_queries.latest_parquet` is reused rather than reimplemented
for "find the newest timestamped file."

### Routing stays unfabricated, by omission

The API has no routing, inspector-assignment or travel-time endpoint. This is not a gap the API
apologizes for closing later — ADR 0019 and ADR 0043 already establish that the dataset carries
no inspector, no duration, no travel time and no road network, and nothing about adding an HTTP
layer changes that. An endpoint that promised a route would be promising an answer built from
fabricated inputs.

## Alternatives rejected

**Let a frontend read Parquet files directly, with a shared library of query helpers.** Pushes
scope validation, pagination, and artifact-resolution logic into every consumer, with no single
place to enforce "an ambiguous scope is a 422." Also exposes internal column names and file
layout as a public contract by accident.

**Give the API write access to trigger `sentinel decide`/`sentinel schedule` itself.** Considered
and rejected; see ADR 0049. In brief: those commands build the whole cell in one checksummed
batch, and calling them from a single HTTP request would either run a slow synchronous rebuild on
every write or introduce an async job system this project does not need yet.

**Number it "Component 15" and let the API absorb the old routing slot.** Rejected by the
project's own numbering discipline: "Component 15" already has a documented meaning (routing,
blocked) in `README.md`, `HANDOFF.md`, `STATUS.md` and the interview docs. Reassigning the name
would either require rewriting that history or produce two different things both called
"Component 15" depending on which document a reader opens.

## Consequences

* A frontend can build every page listed in the data contract (`docs/data_contracts/sentinel_api.md`)
  without ever opening a Parquet file or importing `sentinel.*` directly.
* The API has exactly one way to fail that is *not* a bug: an upstream artifact does not exist
  yet. That failure is a 404 with a clear message, never a 500.
* Every non-trivial number the API returns can be traced to a specific artifact file and manifest,
  because the API never computes one of its own.

## Limitations

* The API is read-mostly. Its write surface is narrow and never applies anything immediately —
  ADR 0049.
* No authentication or authorization exists. See ADR 0049's consequences and
  `docs/interview/api_layer.md` for what a real deployment would need to add.
* The API assumes a single-writer, single-reader local filesystem. It was not built to serve
  concurrent writers racing on the same staging file at meaningful scale.

## What this decision does NOT claim

* **Not that Components 1-14 needed an API to be correct.** They were complete and closed before
  this ADR; the API changes nothing about how they compute anything.
* **Not that this is a general-purpose API framework choice for the whole project.** It is scoped
  to exposing the existing artifacts; see ADR 0049 for why it does not become a general-purpose
  application server.
* **Not a claim that "Component 15" or "Component 16" have been renumbered, redefined or
  completed.** Both names keep their existing, unrelated meaning in the roadmap.
