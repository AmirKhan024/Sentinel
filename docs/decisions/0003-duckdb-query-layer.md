# ADR 0003 — DuckDB as the query layer over raw Parquet

**Status:** Accepted · **Date:** 2026-08-15

## Context

The raw layer needs to be inspectable. "How many rows? What date range? Which
inspection types?" must be answerable without writing a script each time.

## Decision

**DuckDB**, in-memory, reading Parquet directly via `read_parquet()`. No import
step, no persistent database file, no schema DDL.

A small set of named descriptive queries is exposed through
`sentinel query --name <name>`. They describe the raw data only, and contain no
Sentinel business logic.

## Rationale

* DuckDB reads Parquet in place. There is no load step to keep in sync, and no
  second copy of the data that could drift from the files on disk.
* It is an in-process library: no server, no port, no daemon.
* Full analytical SQL — window functions, CTEs, aggregates — which is what the
  later feature-engineering components will want.
* Zero cost when unused.

In-memory rather than a persistent `.duckdb` file, because the Parquet files
**are** the durable artifact. A persistent database would be a second source of
truth that could disagree with them.

## Alternatives rejected

* **PostgreSQL / PostGIS** — a server to run, a schema to migrate and a load
  step, all to answer `SELECT count(*)`. PostGIS may genuinely be warranted when
  routing arrives (Component 15). It is not warranted now.
* **SQLite** — requires importing the data, and its analytical SQL is much
  weaker.
* **Polars only** — Polars is already used to *write* the raw layer. Adding SQL
  gives an ad-hoc query surface that does not require writing Python, which is
  the point.

## Consequences

* Because the raw layer is all strings (ADR 0002), aggregating over dates or
  numbers requires an explicit cast in SQL. Expected.
* Paths are passed as bound query parameters, never interpolated into SQL.
* If a persistent analytical store is ever needed, DuckDB can write one without
  changing the raw layer.
