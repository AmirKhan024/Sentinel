# ADR 0051 — The human-review gate: a fourth human layer, and why it carries no threshold

**Status:** Accepted · **Date:** 2026-08-27

## Context

Component 13 computes `PolicyWarning`s per row — `limited_history`, `no_prior_inspection`,
`insufficient_group_audit_support`, `unknown_geography` — and ADR 0040 states plainly that a
warning is not an abstention: nothing today escalates one. Component 14 can produce a scheduled
inspection with no matching execution report, and nothing today surfaces that gap as a case
needing attention. Both components repeatedly gesture at a future layer: HANDOFF.md and STATUS.md
name Component 16 — the deferral / human-review gate — as "the next implementable component,"
and three separate docstrings (`sentinel.evaluation.metrics`, `sentinel.calibration.definitions`,
`sentinel.evaluation.build`) each say a sentence to the effect of *"a threshold is genuinely
needed only by the Component 16 deferral gate, which can record its own."*

That last sentence is a tension, not a plan. ADR 0040's `ABSTENTION_POLICY` is explicit and load
bearing: *"Sentinel never abstains... An abstention would require a per-row confidence estimate,
and this project has not built one — no predictive interval, no conformal set, no ensemble
spread. Emitting an abstention category anyway would be manufacturing the statistic that justifies
it."* Reading "a threshold" in those three docstrings as a probability or confidence cutoff would
mean building the very statistic ADR 0040 refuses. This ADR resolves the tension on the record,
rather than silently picking a number or silently ignoring the hint.

A second constraint comes from Component 14. `ScheduleStatus.DEFERRED` already exists and means
"a scheduled inspection was moved to a later operating day by an adjustment or a re-plan." A human
review gate is naturally described with the English word "defer," and using it as a controlled
vocabulary value here would make two structurally unrelated facts about the same establishment
look, in code and in data, like the same fact. ADR 0047 already established that the project's
three human layers (override, adjustment, execution) use disjoint verb vocabularies and separate
id namespaces, checked at import time. A fourth layer must extend that discipline, not relax it.

## Decision

**Two deterministic triggers, no numeric threshold, ever. A fourth human layer, with a verb
vocabulary and id namespace disjoint from the other three and from "defer"/"deferred."**

### The two triggers, and nothing else

1. `policy_warning_present` — a selected recommendation carries at least one Component 13
   `PolicyWarning`. This is the literal escalation ADR 0040 declined to build automatically:
   Component 13 records the fact, Component 16 is what routes it to a human.
2. `no_execution_record_on_scheduled_row` — an occupying schedule row has no matching row in the
   accumulated execution log, keyed on Component 14's own execution contract
   (`schedule_config_id, policy_id, fold_id, k_name, target_inspection_id`). This is the row-level
   version of the cell-grain count Component 14 already computes as `NO_EXECUTION_RECORD`.

Both are boolean facts an upstream component already wrote or a plain anti-join already computes.
Neither reads `score`, `base_score` or `final_policy_rank`. `queue_is_deterministically_rebuildable`
proves this by rebuilding the queue from the two triggers and comparing byte for byte against the
written artifact.

**Rejected as triggers, and why:** an explanation-unavailable trigger — the one unsupported model,
`xgboost_chain_embeddings_platt`, is already excluded from `CANDIDATE_MODELS`, so no row in
`inspection_recommendations` could ever be produced by it; wiring in a trigger for a condition no
row can satisfy is dead code. A "a validation check failed without stopping the run" trigger — no
such state exists anywhere in this pipeline; the `SEVERITY_ERROR`/advisory split is binary at the
*run* level, never at the row level. Any numeric score or probability threshold — forbidden by
ADR 0040 directly.

### Reading the "a threshold is Component 16's" docstrings

Those three sentences are read as naming *a decision boundary* in the general sense — the
flag/no-flag line the two triggers already draw — not a probability cutoff. `NO_THRESHOLD` states
this explicitly in `sentinel.review.definitions` and travels in every manifest. If a future
component genuinely needs a calibrated operating point (a real cost-sensitive decision with a
built predictive interval behind it), that is new work with its own ADR — it is not something
this component silently absorbs.

### The fourth human layer

| layer | owner | changes | verbs | id |
| --- | --- | --- | --- | --- |
| recommendation override | Component 13 | **who** is in the queue | `force_include`, `force_exclude` | `override_id` |
| scheduling adjustment | Component 14 | **when** an approved row is worked | `defer_to_date`, `advance_to_date`, `cancel` | `adjustment_id` |
| execution event | Component 14 | records **what happened** | `completed`, `not_performed`, `cancelled_in_field` | `execution_id` |
| review resolution | Component 16 | records what a human decided about a **flagged case** | `acknowledge`, `refer_to_override`, `refer_to_adjustment`, `escalate` | `review_id` |

`sentinel.review.definitions._guard_registry()` checks, at import time, that
`ReviewResolutionAction` shares no value with `OverrideAction` or `AdjustmentAction`, and —
stronger than mere set disjointness — that no `ReviewCaseStatus`, `ReviewTriggerReason` or
`ReviewResolutionAction` value contains the literal substring `"defer"`. The second check exists
because set disjointness alone would not have caught a hypothetical `DEFERRED_FOR_REVIEW` status
value: it collides with nothing in the other two enums, and would still be exactly the confusion
this ADR exists to prevent.

A review resolution never creates the override or adjustment it refers to. `refer_to_override`
and `refer_to_adjustment` record a pointer — the human's stated intent — and the override or
adjustment itself remains a separate submission through Component 13's or Component 14's own
contract, validated by that contract's own parser.

### Everything else inherited unchanged

The queue is rebuilt fresh each run (mirroring `inspection_recommendations`); the resolution log
is append-only (mirroring `policy_override_log`); resolutions parse all-or-nothing and apply in
`review_id` order, never file order; `actor`/`reason_code`/a human timestamp are mandatory; the
API stages, never applies (ADR 0049); `DecisionScope` needs no new field, since a review case is
scoped by the same `(policy_id, fold_set, fold_id, k_name)` recommendations already use.

## Alternatives rejected

**A numeric confidence/probability threshold.** Directly forbidden by ADR 0040; there is no
predictive interval anywhere in this project to threshold against.

**Reusing "defer"/"deferred" as the review-gate's verb.** Confusable with
`ScheduleStatus.DEFERRED` at both the code and data level — the exact failure mode ADR 0047's
disjoint-vocabulary discipline exists to prevent.

**Merging review resolution into the existing override or adjustment tables.** A review
resolution answers a different question ("should a human look at this?") than either existing
contract, and merging it would repeat the mistake ADR 0047 already rejected for adjustments and
execution events.

**An append-only, never-shrinking review queue.** Rejected in favor of a queue rebuilt fresh each
run, because the trigger conditions are themselves derived from mutable upstream state (an
execution gap can close). The resolution log, not the queue, is the permanent record.

## Consequences

* A fourth table family (`data/processed/review/`), a fourth API resource group
  (`/v1/review/...`), and a fourth CLI subcommand (`sentinel review`).
* On a run against real data (2026-08-27), 70,791 of 1,453,760 recommendation rows were flagged:
  39,652 by `policy_warning_present` and 70,791 by `no_execution_record_on_scheduled_row` (the
  production execution log is currently empty, so every occupying schedule row is a gap by
  construction — see Limitations).
* The chain original recommendation → approved recommendation → planned schedule → **flagged for
  review** → human resolution → override/adjustment/execution is reconstructible from artifacts
  alone.

## Limitations

* **No execution event in this repository describes anything that happened in Chicago**, exactly
  as ADR 0047 already recorded. Because of this, the measured `no_execution_record_on_scheduled_row`
  count above (70,791) is an artifact of an empty execution log, not a finding about real
  operational gaps — every occupying schedule row is, by construction, "missing" a report nobody
  was ever going to file. This is stated plainly rather than reported as a discovery.
* Whether the two triggers match what a real health department would actually want reviewed is
  untested — the same open question ADR 0047 recorded for its own two contracts.
* Whether `POINTER_MAY_FORWARD_REFERENCE` (a resolution may point at an override/adjustment id
  that has not yet been submitted) is the right operational ordering is a genuine open design
  choice, not a scientific finding; the alternative (refuse until the target exists) would force a
  strict submission order this repository does not currently enforce anywhere else.

## What this decision does NOT claim

* **Not that a flagged case is wrong, risky, or should be overridden.** A warning or an execution
  gap is a fact about what is known, not a verdict.
* **Not that an unflagged case needed no review.** The trigger set is two deterministic facts, not
  an exhaustive review checklist a real department would use.
* **Not that this component introduces any form of automated decision-making beyond routing.** It
  computes no score, fits no model, and makes no recommendation of its own — it only names which
  already-existing rows should stop in front of a human.
