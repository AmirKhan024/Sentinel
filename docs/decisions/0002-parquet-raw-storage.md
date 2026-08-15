# ADR 0002 — Raw data stored as Parquet, all columns as strings

**Status:** Accepted · **Date:** 2026-08-15

## Context

The Socrata API returns JSON. We need a durable on-disk raw layer. Two
questions: what format, and what types.

A live finding forced the second question. The API returns **every value as a
JSON string**, even for columns it declares as `number` or `calendar_date`:

```json
{"inspection_id": "2641210", "latitude": "41.86568627741837"}
```

## Decision

**Format: Apache Parquet**, zstd-compressed, one timestamped file per run.

**Types: every column written as `Utf8`.** No casting at ingestion. The nested
`location` object is re-serialized to its compact JSON string. Missing keys
become null.

## Rationale

### Why Parquet

* Columnar, so a later component reading three of twenty-two columns pays for
  three.
* Self-describing: the schema travels with the file.
* Compresses far better than JSON on this data, which is full of repetitive
  categorical text. The 5,000-row development extract is 827,350 bytes.
* Read natively by DuckDB, Polars, pandas, PyArrow and Spark, so the raw layer
  never becomes a lock-in point.

### Why all strings

* **Casting is lossy in a way that hides itself.** A malformed date silently
  becomes null and is then indistinguishable from genuinely missing data. Once
  the raw file is written, that evidence is gone.
* **The raw layer must match the source.** If the bytes on disk differ in
  meaning from what the API returned, "raw" is a false label, and every
  downstream debugging session starts from a lie.
* **Typing is a modelling decision.** Whether `license_ = "0"` means "no
  licence", how to parse a floating timestamp, whether `""` is null — each
  deserves to be named, tested and documented in a component of its own.

## Alternatives rejected

* **Raw JSON / JSONL** — maximally faithful, but every downstream read pays
  full parsing cost and there is no schema. Parquet with all-string columns
  keeps the fidelity while adding a schema and compression.
* **Cast using `X-SODA2-Types`** — convenient downstream, but introduces silent
  nulls and breaks the raw guarantee. Rejected explicitly.
* **CSV** — no schema, ambiguous quoting, and the `violations` field contains
  newlines and delimiter-like characters.
* **A database table** — infrastructure ahead of a requirement, and it would
  make the raw layer mutable.

## Consequences

* Downstream components **must cast explicitly** and handle cast failures. This
  is a real, deliberate cost, paid to keep the raw layer honest.
* A DuckDB `DESCRIBE` over a raw file reports `VARCHAR` for all 22 columns.
  That is correct, not a bug.
* File size is larger than a typed encoding would give. Acceptable.
