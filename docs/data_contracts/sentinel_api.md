# Data contract — the Sentinel API

**Producer:** `sentinel serve` (`src/sentinel/api/`) · **Layer:** none — a read/write HTTP
boundary over `data/processed/policy/`, `data/processed/scheduling/` and
`data/processed/explanations/`, plus its own append-only store at `data/staging/`. See
ADR 0048, ADR 0049, ADR 0050.

This document describes the API's *own* request/response contracts. It does not restate the
underlying artifact schemas — those are `docs/data_contracts/policy_decisions.md`,
`docs/data_contracts/inspection_schedule.md` and `docs/data_contracts/explanations.md`. Every
field this API returns is named identically to the artifact column it came from.

---

## What this API is not

**It is not a sixth pipeline layer.** It computes nothing. Every non-trivial value in a response
was already written by a batch CLI command; see ADR 0048.

**It is not a way to apply an override, adjustment or execution event immediately.** A `POST`
stages a validated request; turning it into a new committed artifact remains a manual step
through `sentinel decide` / `sentinel schedule`. See ADR 0049.

**It does not guess a decision scope.** A request that does not fully specify which fold, policy,
capacity or planning run it means is refused with `422`, never silently resolved to "latest" or
"first." See ADR 0050.

**It is not a routing or inspector-assignment API.** No such endpoint exists, because the dataset
has no inspector, duration or travel time (ADR 0019, ADR 0043).

---

## Decision scope

Every endpoint below states its **required** scope fields. A request missing any of them returns:

```json
{
  "error": "ambiguous_scope",
  "detail": "Decision scope is missing required field(s): fold_id, k_name. ...",
  "missing_scope_fields": ["fold_id", "k_name"]
}
```

Available scope fields (all are query parameters, all optional on the model — required-ness is
per endpoint): `policy_id`, `model_name`, `fold_set`, `fold_id`, `k_name`, `schedule_config_id`,
`planning_run_id`, `replan_index`.

`planning_run_id`/`replan_index` are the one field pair with a stated default: omitted, they
resolve to the scoped cell's latest `replan_index`. Every row in the response still carries its
own `planning_run_id`/`replan_index`, so which one was picked is never ambiguous in the response
itself.

## Pagination envelope

Every list endpoint returns:

```json
{
  "data": [ ... ],
  "page": {"offset": 0, "limit": 50, "total": 137},
  "run": {"path": "...", "manifest_path": "...", "built_at": "2026-08-26T12:00:00+00:00"}
}
```

`limit` is capped at `Settings.api_max_page_size` (default 500) regardless of what a caller
requests — `inspection_recommendations` alone holds roughly 1.45M rows, and this endpoint never
performs an unbounded scan. Sorting is a per-endpoint whitelist of columns, never an arbitrary
caller-supplied field name.

## Endpoints

### Read

| Method & path | Required scope | Notes |
| --- | --- | --- |
| `GET /healthz` | — | Liveness only; touches no artifact. |
| `GET /v1/manifests/{component}` | — | `component` ∈ `policy`, `scheduling`, `explanations`, `review`. Returns the latest run's manifest JSON. |
| `GET /v1/runs` | — | `component` query param optional. Lists discoverable timestamped runs per component. |
| `GET /v1/recommendations` | `policy_id, fold_set, fold_id, k_name` | Filters: `establishment_id`, `is_selected`. Backed by `inspection_recommendations`. |
| `GET /v1/recommendations/{target_inspection_id}` | same | Single-row lookup. |
| `GET /v1/policy/selection-allocation` | same | `policy_selection_allocation`. |
| `GET /v1/policy/overrides` | same | Filter: `target_inspection_id`. `policy_override_log`, **committed rows only** — `status` is always `"committed"` here; see the correction below. |
| `GET /v1/schedule` | `schedule_config_id, policy_id, fold_set, fold_id, k_name` | Filters: `establishment_id`, `schedule_status`. Defaults to the latest `replan_index`. |
| `GET /v1/schedule/backlog` | same | `schedule_backlog`. |
| `GET /v1/schedule/summary` | same | `schedule_summary`, one row per cell, not paginated. |
| `GET /v1/schedule/capacity-utilization` | same | `capacity_utilization`. |
| `GET /v1/schedule/priority-preservation` | same | `priority_preservation`. |
| `GET /v1/schedule/replanning-runs` | same (no `planning_run_id`/`replan_index`) | `replanning_runs`, every replan for the cell, ordered. |
| `GET /v1/schedule/adjustments` | same | Filter: `target_inspection_id`. `schedule_adjustment_log`, **committed rows only**. |
| `GET /v1/execution/events` | same | Filter: `target_inspection_id`. `execution_log`, **committed rows only**. |
| `GET /v1/execution/summary` | same | `execution_summary`. |
| `GET /v1/execution/contract` | — | Read-only dump of `execution_contract` — the adjustment/execution file formats as data. The authoritative source for `execution_status`'s allowed values; a client should read this rather than hardcode the enum. |
| `GET /v1/review/queue` | `policy_id, fold_set, fold_id, k_name` | Filter: `trigger` — a literal substring match against the existing pipe-joined `trigger_reasons` column (e.g. `policy_warning_present`), not a new classification; a case carrying multiple triggers matches each of them. `human_review_queue`. Committed rows only; presentation `status` is always `"committed"` here. |
| `GET /v1/review/queue/{target_inspection_id}` | same | Single-row lookup. |
| `GET /v1/review/resolutions` | same | Filter: `target_inspection_id`. `review_resolution_log`, **committed rows only**. |
| `GET /v1/explanations/support` | — | `explanation_support` for every model. |
| `GET /v1/explanations/{target_inspection_id}` | `model_name, fold_set, fold_id` | 404 distinguishes "model not explainable" from "not in the sampled subset." |
| `GET /v1/establishments/{establishment_id}` | `policy_id, fold_set, fold_id, k_name` (+ `schedule_config_id` to include schedule) | Cross-artifact bundle; see below. |
| `GET /v1/staged-requests` | — | `kind`/`status` query params optional. Every pending or reconciled staged request. |

### Write (stage-only — see ADR 0049)

| Method & path | Body | Behavior |
| --- | --- | --- |
| `POST /v1/policy/overrides` | Exactly Component 13's `Override` fields | Validated via `parse_overrides`; staged; `201` with a `StagedRequestReceipt`. `409` on a colliding id (pending or committed). |
| `POST /v1/schedule/adjustments` | Exactly Component 14's `Adjustment` fields | Validated via `parse_adjustments`; staged the same way. |
| `POST /v1/execution/events` | Exactly Component 14's `ExecutionEvent` fields | Validated via `parse_execution_events`; staged the same way. Rejects `execution_status: no_execution_record` (a derived category, not a postable one). |
| `POST /v1/review/resolutions` | Exactly Component 16's `ReviewResolution` fields | Validated via `parse_resolutions`; staged the same way. Pointer-field mismatch for the given `resolution_action` is refused as `validation_refused`. |

No `PATCH`, `PUT` or `DELETE` route exists anywhere in the API. Immutability is structural: there
is no path capable of mutating a committed row, so those verbs return FastAPI's default `405` on
every resource rather than a hand-written rejection.

## Correction: the four log-read endpoints do not merge staged rows

Earlier text in this document (and in each schema's own docstring) described
`GET /v1/policy/overrides`, `GET /v1/schedule/adjustments`, `GET /v1/execution/events` and
`GET /v1/review/resolutions` as returning "committed rows plus pending staged ones, distinguished
by `status`." Reading `policy_service.get_override_log`, `scheduling_service.get_adjustment_log`,
`scheduling_service.get_execution_events` and `review_service.get_resolution_log` directly shows
this was never implemented: each unconditionally stamps `status = "committed"` on every row it
reads from the committed artifact, and none of the four reads the staging store at all. This was
found while building the frontend's write forms — a caller that wants "this establishment's
committed *and* pending decisions in one place" (the frontend's "Decision history" panel) must
call the relevant `GET` endpoint **and** `GET /v1/staged-requests?kind=<kind>` separately, then
merge client-side on `target_inspection_id`/`natural_id`. That is what
`frontend/src/components/actions/DecisionHistory.tsx` does. Fixing the four read services to merge
server-side, so the docstring's original claim becomes true, is future work — not done here, to
keep this pass to product surface and read filters rather than changing read-service behavior.

## The establishment-detail bundle

`GET /v1/establishments/{establishment_id}` composes, in one response:

```json
{
  "establishment_id": "...",
  "recommendation": { "...": "Component 13's row" },
  "schedule": { "...": "Component 14's row, or null" },
  "explanation": { "...": "Component 11's case + feature values, or null" },
  "explanation_unavailable_reason": "why explanation is null, or null",
  "history_factors": { "...": "a curated slice of Component 4's own as-of feature row, or null" },
  "history_factors_unavailable_reason": "why history_factors is null, or null"
}
```

* `schedule` is `null` whenever `schedule_config_id` was not supplied in the request scope, or
  when no schedule row matches (the establishment may still be in the recommendation queue but
  unscheduled/backlogged).
* `explanation` is `null` whenever the model has no attribution support at all, or the specific
  `target_inspection_id` was not in Component 11's sampled subset for that fold —
  `explanation_unavailable_reason` says which.
* `history_factors` is a fixed, curated subset of ten columns already present in Component 4's
  `as_of_features_*.parquet` (e.g. `prior_canvass_priority_rate`, `days_since_last_canvass`) —
  added so a product surface can state the concrete, non-technical reasons behind a
  recommendation (`RiskHistoryFactorsOut`, `src/sentinel/api/schemas/establishment.py`) without
  requiring a model explanation, which is only available for a sampled subset of rows. It is
  `null` only when no feature table has been built yet, or this `target_inspection_id` is outside
  it — never a silently empty object; `history_factors_unavailable_reason` says which. This field
  computes nothing: every value was already written by Component 4.
* If the establishment matches more than one recommendation row under the given scope (it was
  inspected more than once in the same fold), the endpoint returns `422 ambiguous_scope` with the
  candidate `target_inspection_id` values, rather than picking one.

`recommendation.decision_reason` and `schedule.schedule_reason` are never merged into a single
`reason` field — they are two different components' answers to two different questions (ADR 0042,
ADR 0050).

**A real, pre-existing naming detail this endpoint surfaces rather than hides:**
`recommendation.model_name` carries the *calibrated* name (e.g. `xgboost_platt`), while
Component 11's explanation artifacts are keyed by the *base* model name (e.g. `xgboost`) —
documented in `docs/data_contracts/explanations.md` §"the calibration boundary". Looking up an
explanation under the calibrated name legitimately finds nothing, and the bundle reports this as
`explanation_unavailable_reason` rather than silently trying both names. A caller that wants a
given establishment's explanation should pass the base model name to
`GET /v1/explanations/{id}` explicitly.

## Reproducibility

* Repeated, identical `GET` requests against unchanged artifacts return byte-identical JSON:
  deterministic sort order, no timestamp is injected into a response's `data` payload (only the
  `run.built_at` provenance field, sourced from the manifest, ever carries one).
* No `GET` request ever writes to `data/processed/` or `data/staging/`.
* No `POST` request ever writes to `data/processed/` — only to `data/staging/`.

## Errors

| HTTP | `error` | Meaning |
| --- | --- | --- |
| 404 | `artifact_not_found` | The upstream component has not produced this table yet. |
| 404 | `row_not_found` | The artifact exists; no row matches the request under the given scope. |
| 404 | `unknown_component` | `{component}` in a manifest/runs request is not one this API exposes. |
| 422 | `ambiguous_scope` | The scope does not pick out exactly one cell or row. Carries `missing_scope_fields` and/or `candidate_values`. |
| 422 | `validation_refused` | A write payload failed the real parser for its layer; carries that parser's own message. |
| 409 | `duplicate_key` | A write payload's natural id collides with a pending or committed record under a different payload. |
| 405 | — | A verb with no route on this path (structural immutability). |
| 500 | `internal_error` | An unexpected failure. No stack trace or internal path is ever included in the body. |
