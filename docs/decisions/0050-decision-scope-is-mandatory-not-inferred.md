# ADR 0050 — Decision scope is mandatory, not inferred

**Status:** Accepted · **Date:** 2026-08-26

## Context

The same establishment appears many times over in Components 13 and 14's artifacts: once per
policy (`pure_risk`, `coverage_floor_population_share`, ...), once per fold
(`quarterly-2026Q1`, `quarterly-2026Q2`, ...), once per capacity level (`k_1_day`, `k_1_week`),
and — once scheduling is added — once per `schedule_config_id` and once per planning run
(`planning_run_id`/`replan_index`) every time a re-plan appends a new one. A request for "the
recommendation for establishment E1" or "the schedule for E1" is not a well-formed question until
enough of those axes are pinned down that exactly one row could possibly answer it.

An API that tried to be helpful here has an obvious, wrong move available: default the missing
axis to "the latest one," or "the first one found," or "the production model's." Every one of
those defaults is a *policy decision made silently on the caller's behalf* — which fold counts as
current, which policy is "the" answer — and this project has already refused to make that kind of
decision silently once, in ADR 0042: the whole point of keeping five layers apart is that "who
should be inspected" and "who was inspected" and "who is scheduled" are different facts that
must never quietly collapse into one. An API endpoint that picked a fold for a caller who didn't
name one would be the same collapse one layer further out — a policy choice with no ADR behind
it, made inside a request handler instead of a governance document.

## Decision

**Every endpoint states exactly which scope fields disambiguate its answer. A request missing any
of them is refused with `422 ambiguous_scope` and the list of what's missing — never silently
resolved to "latest," "first," or "production."**

### The scope model

`DecisionScope` (`sentinel.api.schemas.common`) carries every field any endpoint might need:
`policy_id`, `model_name`, `fold_set`, `fold_id`, `k_name`, `schedule_config_id`,
`planning_run_id`, `replan_index`. All are optional on the model itself — which ones are
*required* is a per-endpoint decision made once, by `services.artifacts.require_scope`, and
never inferred by any individual handler. Recommendation and policy endpoints require
`policy_id, fold_set, fold_id, k_name`; schedule endpoints additionally require
`schedule_config_id`; explanation endpoints require `model_name, fold_set, fold_id`.

`planning_run_id`/`replan_index` are the one place a narrow, stated default exists rather than a
required field: when neither is given, the schedule and backlog endpoints resolve to the cell's
*latest* `replan_index` — and the response is unambiguous about which one that was, because every
row in it carries its own `planning_run_id` and `replan_index`. This is not an exception to the
rule; it is the rule applied to an axis where "the current plan for this cell" has one measurable,
stated meaning (the highest `replan_index` written), unlike "the current fold" or "the production
policy," which do not.

### Ambiguity is checked twice: before the read, and after it

`require_scope` catches the case where a caller didn't supply enough *fields*. A second check
catches the case where the supplied fields still weren't enough: an establishment can hold more
than one row even inside a fully-specified `(policy_id, fold_set, fold_id, k_name)` cell, because
it can have been inspected more than once in the same window. The establishment-detail and
recommendation-list-by-id paths check the *row count* after filtering and raise
`AmbiguousScope` — with the actual candidate values (`target_inspection_id`s, in that case) in
the error body — rather than returning the first match.

### The error is actionable, not just true

`AmbiguousScope`'s body always carries either `missing_scope_fields` (what to add) or
`candidate_values` (which of the values that already disambiguate a *different* row would resolve
this one) — never only "ambiguous." A caller that hits this error can retry correctly without
reading source code.

## Alternatives rejected

**Default to the most recently written run.** Silently answers a different question than the one
asked, and "most recent" is itself a choice about which axis (fold? policy? capacity?) to treat
as the mutable one and which to treat as fixed — a choice this project has consistently refused
to make outside a governance document (ADR 0034, ADR 0039, ADR 0042).

**Default to the production/selected model or policy.** Same objection, more consequential: it
would make "GET a recommendation" quietly depend on Component 13's selection rule, so a change to
that rule would silently change what an *unrelated* API request returns.

**Return all matching rows instead of erroring on ambiguity.** Considered for the
establishment-detail endpoint specifically. Rejected because "here are your two possible answers,
guess" is not meaningfully different from guessing on the caller's behalf, and it turns a
single-object response schema into a collection one depending on data the caller can't predict in
advance.

## Consequences

* Every list endpoint's response is reproducible: the same scope, against the same artifact
  files, returns the same rows in the same order, every time.
* A frontend must always know, and send, the scope it means. This is more upfront work per
  request than a "just give me the latest" API, and it is treated as the correct trade rather
  than friction to remove later.
* The 422 error body is part of the API's contract, not an afterthought — `missing_scope_fields`
  and `candidate_values` are documented fields on the error response, not free-text hints.

## Limitations

* The scope model is a flat set of string/int fields, not a typed union per endpoint. An endpoint
  that doesn't use `schedule_config_id` still exposes it as an acceptable (ignored) query
  parameter. Accepted for now as the simpler shape; a stricter per-endpoint scope type is a
  reasonable future tightening if it becomes a source of caller confusion.
* The default-to-latest-`replan_index` behavior is the one place this ADR's rule is relaxed, and
  a future reviewer should be able to point at this document to see that the relaxation was
  considered and scoped narrowly rather than crept in accidentally.

## What this decision does NOT claim

* **Not that every ambiguity in the underlying data is a defect.** An establishment legitimately
  can be inspected twice in one fold; the API's job is to say so clearly, not to prevent it from
  being true.
* **Not that scope fields map one-to-one onto a caller's mental model.** A frontend still needs to
  know what a `fold_id` or a `k_name` means to construct a sensible request; this ADR only
  commits the API to never guessing on the caller's behalf once those concepts exist.
