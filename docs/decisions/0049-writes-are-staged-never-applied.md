# ADR 0049 — Writes are staged, never applied

**Status:** Accepted · **Date:** 2026-08-26

## Context

Components 13 and 14 already define three human-input contracts — a recommendation override, a
scheduling adjustment, and an execution event — each with a pydantic model, a strict all-or-
nothing parser, and an append-only log table (ADR 0047). All three are currently *batch* inputs:
a person edits a JSON file, and an operator passes it to `sentinel decide --overrides PATH` or
`sentinel schedule --adjustments PATH --execution PATH`, which reads the whole file, applies it,
and writes one new timestamped, checksummed artifact set for the whole cell.

The Sentinel API needs to let a caller submit one override, one adjustment, or one execution
event over HTTP. The tempting design is to have the write endpoint call straight through to the
same functions the CLI calls — `sentinel.policy.select`, `sentinel.scheduling.build.run_schedule`,
`replan`, `apply_adjustments`, `record_execution` — so that a `POST` produces an immediately
updated recommendation queue or schedule.

That design does not survive contact with how those functions actually work, which this ADR
checked directly rather than assumed:

* `apply_adjustments`, `record_execution` and `replan` are pure functions on a `SchedulePlan`, but
  every real call site is inside `run_schedule`'s per-cell batch loop. That loop needs the *full*
  approved queue for the cell, recomputes `schedule_slots`, `schedule_summary` and
  `capacity_utilization` for every cell in the run, checksums every upstream input before and
  after, and writes one manifest for the whole batch. Calling `apply_adjustments` or
  `record_execution` for a single cell outside that loop would either skip all of that
  bookkeeping or require reimplementing it inside the API — a second, thinner copy of
  `run_schedule` with none of its checks.
* The override, adjustment and execution parsers are deliberately **all-or-nothing**: one bad row
  in a file refuses the whole file (`adjustments.py`, `execution.py`, `policy/governance.py`).
  That rule exists because a partially applied file produces a schedule "nobody authorised." An
  API that applied one validated request at a time, outside that file-level contract, would be
  changing what "the whole file" means without changing the rule that assumes one.
* The checksum-before/after and single-manifest-per-run pattern *is* this project's
  reproducibility argument (`inputs_unchanged`, `input_sha256_after` on every manifest). A write
  path that bypassed it to answer one HTTP request quickly would be the first artifact-producing
  code path in the repository with no manifest at all.

## Decision

**A write endpoint validates one request against the existing contract for its layer, then
appends it to an append-only staging file the API owns. It never calls
`select`/`run_schedule`/`replan`/`apply_adjustments`/`record_execution`, and it never produces a
new committed artifact.**

### What "validate" means, concretely

Each of the three write endpoints — `POST /v1/policy/overrides`, `POST /v1/schedule/adjustments`,
`POST /v1/execution/events` — accepts exactly the fields of the corresponding pydantic model
(`Override`, `Adjustment`, `ExecutionEvent`; `extra="forbid"`), then runs the payload through the
*real* parser (`parse_overrides`, `parse_adjustments`, `parse_execution_events`) as a
one-element list, for validation only. If the batch CLI would refuse this row — an unknown
action, a cancel carrying a target date, the derived `no_execution_record` status offered as if a
person could supply it — the API refuses it with the parser's own message. No second copy of the
validation rules exists.

### What "staged" means, concretely

A validated request is appended as one JSON line to a file under `Settings.staging_dir`
(`data/staging/policy/overrides_pending.jsonl`,
`data/staging/scheduling/adjustments_pending.jsonl`,
`data/staging/scheduling/execution_events_pending.jsonl`), in exactly the JSON-list-of-objects
shape the batch CLI's `--overrides`/`--adjustments`/`--execution` flags already read — so an
operator can render the pending file to a list and hand it to the CLI with no format
translation. The line is appended, never rewritten: the staging store is append-only for the same
reason every committed log in this project is.

The endpoint returns `{request_id, kind, natural_id, status: "pending", staged_at}` — never a
recomputed schedule or queue, because none exists yet.

### There is no `POST /v1/replan`

Replanning is not exposed as a distinct write action. It happens later, when an operator re-runs
`sentinel schedule` against the accumulated staged execution events, and that re-run is what
produces the next `planning_run_id`/`replan_index`. The API can only ever get a caller as far as
"an execution event has been staged that would, once applied, trigger a re-plan" —
`GET /v1/staged-requests` and `GET /v1/execution/events` (with `status=pending`) make that
visible without the API doing the re-planning itself.

### Idempotency and duplicate detection

Staging is deliberately safe to retry and hostile to silent overwrite:

* The **same** natural id (`override_id`/`adjustment_id`/`execution_id`) with a byte-identical
  payload, re-posted, returns the original receipt rather than erroring — a caller's retry after
  a dropped response never produces a second, conflicting pending record.
* The same id with a **different** payload is refused (`409 duplicate_key`) — silently
  overwriting a pending human decision is exactly the failure ADR 0047's audit trail exists to
  prevent.
* An id that already exists in the **committed** log for that layer is refused the same way,
  whether or not it is also pending — staging it again would be indistinguishable from
  re-deciding something an operator already decided.

### Reconciliation is a read, not a side effect

`GET /v1/staged-requests` cross-references every pending id against the latest committed log for
its layer and reports `pending` or `applied` accordingly. This is pure comparison against
artifacts the API already reads elsewhere — it never writes to the staging store, and it never
triggers a batch run to "check."

## Alternatives rejected

**Call `run_schedule`/`select` synchronously inside the request.** Rejected above: it either
skips the batch's own checksum and manifest guarantees, or reimplements them inside the API,
producing two code paths that write the same kind of artifact with two different amounts of
rigor.

**Introduce a job queue (Celery, a background worker) so a write endpoint can trigger an
asynchronous batch run.** Rejected as infrastructure with no measured need yet — this project's
own convention (ADR 0001, restated in `pyproject.toml`'s dependency comments) is to add a
technology only when the component that needs it is being built, and nothing here needs a
scheduler beyond the one an operator already runs by hand.

**Mutate the pending file in place on a duplicate id (last write wins).** Would let a second API
caller silently overwrite a first caller's staged decision with no record that the first one ever
existed.

## Consequences

* A `POST` to any of the three write endpoints never changes what a concurrent `GET` to
  `/v1/recommendations`, `/v1/schedule` or any committed-log endpoint returns. Reads and writes
  are on entirely separate files.
* Turning a batch of staged requests into a new schedule or queue remains a manual, human-
  triggered operational step. This is a real limitation for a "real-time" product experience, not
  a simplification — see Limitations.
* The staging store itself needs an operational process (a cron job, a runbook step) to be
  drained regularly in any real deployment. None is built here.

## Limitations

* There is no automatic drain. A staged request can sit pending indefinitely if nobody runs the
  batch CLI against it. The API surfaces this honestly (`status: "pending"` forever) rather than
  hiding it.
* Concurrent writers appending to the same staging file are serialized only by the filesystem's
  own append semantics; this was not built or tested for high write concurrency.
* `GET /v1/staged-requests` reconciliation is a full-log scan per request. Fine at this project's
  scale (a handful of overrides/adjustments/events per run); would need indexing at a much larger
  one.

## What this decision does NOT claim

* **Not that this is a real-time product feature.** A caller who submits an override does not see
  it reflected in the recommendation queue until an operator runs `sentinel decide` again. The
  staged receipt says so explicitly (`status: "pending"`).
* **Not that staging is a database.** It is a flat, append-only JSON-lines file per contract, with
  no query engine, no indexing and no transactions beyond what a single `open(..., "a")` call
  gives for free.
* **Not that this design is what a production deployment should ship as its final form.** It is
  the smallest correct thing that preserves every guarantee Components 13 and 14 already made;
  `docs/interview/api_layer.md` names what a real deployment would need to add.
