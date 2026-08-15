# ADR 0004 — Use the Socrata API directly, with pagination written by hand

**Status:** Accepted · **Date:** 2026-08-15

## Context

The Chicago Food Inspections data can be obtained by scraping the portal,
downloading a CSV export by hand, using the `sodapy` SDK, or calling the SODA
API directly.

## Decision

Call the **SODA 2.1 API directly with httpx**, and implement pagination, retries
and error handling **in this repository** rather than delegating them.

`$order=inspection_id` is mandatory on every paged request, and is enforced in
code.

## Rationale

### Why the API rather than scraping or a manual CSV

* It is the officially supported, stable interface. Scraping HTML breaks on any
  portal redesign.
* A manually downloaded CSV is not reproducible: nobody can tell later what was
  downloaded, when, or with what filters. The API plus a manifest gives exact
  provenance.
* The API supports filtered and incremental retrieval, which later components
  may need.

### Why hand-written pagination rather than an SDK

The paging contract is the highest-risk thing in this component. If pages
silently overlap or skip rows, every downstream model trains on corrupted data,
and nothing in the pipeline would notice.

Writing the loop explicitly means the termination conditions, the `$order`
requirement and the retry policy are all visible, reviewable and directly
unit-testable. `sodapy` would hide exactly the behaviour that most needs to be
legible, and it is a thin wrapper over `requests` regardless.

### Retry policy

Verified that the API returns HTTP 400 with a JSON `errorCode` for a bad query.
So: retry 429, 5xx, timeouts and transport errors with bounded exponential
backoff; **raise immediately on any other 4xx**. A 400 is a defect in our own
request, and retrying it three times only delays and obscures the bug.

## Alternatives rejected

* **`sodapy`** — hides the critical logic, and is an extra dependency for a
  wrapper we would have to read anyway in order to trust it.
* **Scraping** — fragile, and unnecessary when an API exists.
* **Manual CSV download** — not reproducible; explicitly ruled out.
* **Unbounded retries** — turns an outage into a hang.

## Consequences

* We own the pagination code and must test it. Done: pagination, mid-page
  truncation, short-page and empty-page termination, and all four retry paths
  are covered by unit tests.
* An API change is our problem to detect. Mitigated by an opt-in live smoke test
  asserting that the endpoint, field list and string encoding still hold.
* One extra request per run, because the field list is discovered at runtime.
  See docs/api/socrata_findings.md §6 for why.
