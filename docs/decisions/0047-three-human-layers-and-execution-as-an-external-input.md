# ADR 0047 — Three human layers, never merged; execution is external and re-planning does not mutate history

**Status:** Accepted · **Date:** 2026-08-26

## Context

Component 13 introduced one human input: a recommendation override, which changes *who* is in
the approved queue. Component 14 needs two more — a way to change *when* an approved row is
worked, and a way to record *what actually happened*.

The tempting design is one generic "override" table with an action column wide enough for all
three. It is one contract to learn, one parser, one log. It also destroys the only thing that
makes the chain auditable. After the merge, "somebody changed something" is the most a reader
can reconstruct, and these three questions become indistinguishable:

* Did a supervisor decide this establishment should be in the queue at all?
* Did a supervisor decide it should be worked on Thursday rather than Tuesday?
* Did an inspector report that Tuesday did not happen?

The first is a policy decision, the second an operational one, and the third is not a decision
at all — it is an observation about the past, and it must never edit the plan it describes.

There is a second problem underneath. This project's reproducibility claim is byte-identity
across runs. Human and operational inputs are not reproducible computation, and a contract that
implied otherwise would be the easiest lie in the component.

## Decision

**Three contracts, three id namespaces, three disjoint verb vocabularies, three tables.
Execution events are external inputs, and a re-plan appends a planning run rather than mutating
one.**

### The three layers, kept apart mechanically

| layer | owner | changes | verbs | id |
| --- | --- | --- | --- | --- |
| recommendation override | Component 13 | **who** is in the queue | `force_include`, `force_exclude` | `override_id` |
| scheduling adjustment | Component 14 | **when** an approved row is worked | `defer_to_date`, `advance_to_date`, `cancel` | `adjustment_id` |
| execution deviation | Component 14 | records **what happened** | `completed`, `not_performed`, `cancelled_in_field` | `execution_id` |

The disjointness is enforced at **import time**: `definitions._guard_registry` fails if any
adjustment verb collides with an override verb, or if the two contracts share an id field name.
`adjustments_are_not_overrides` re-checks it from the written artifacts, including that the
adjustment log carries no queue column.

### Both new contracts inherit Component 13's override discipline verbatim

* A JSON list of objects. **Every field required**; a blank or missing one refuses the **whole
  file**, because a partially applied file produces a schedule nobody authorised.
* Applied in **id order, never file order**, so re-serialising the JSON cannot change the
  outcome. Tested both directions.
* `actor`, `reason_code` and a human timestamp are mandatory — an external change with no
  attribution is an anonymous change to who gets inspected when.
* The timestamp is the **person's**, never the run's clock.
* Duplicate ids refuse the file.
* Both files are pinned in the manifest by **sha256**, rather than by a claim that human typing
  is reproducible.
* Every offered change is logged **whether or not it changed anything**. "The supervisor asked
  and it made no difference" is an audit fact, and a log recording only effective changes would
  make a no-op indistinguishable from a request nobody made.

### An adjustment costs a displacement, and never spends the coverage reserve

Capacity is fixed, so moving a row onto a full day means another row gives up its slot. The
displaced row is always the **lowest-ranked `risk_priority` row** on that day, never a
`coverage_reserve` row — `policy/governance.py`'s rule for override inclusions, for the same
reason: taking the slot from the coverage allocation would quietly convert every scheduling
change into a coverage cut. If no displaceable risk row exists, the run **refuses** rather than
falling back to the reserve.

The displaced row takes the slot the mover vacated — a swap, not a search. Searching the horizon
could displace a third row and turn one supervisor's request into a cascade nobody asked for. If
it cannot land, it goes to backlog with `displaced_by_adjustment`. It never disappears.

`cancel` frees a slot and the slot is **not** backfilled, exactly as `force_exclude` is not: the
supervisor who struck a row did not ask for a replacement.

### The deterministic plan is written unchanged

An applied adjustment appends a **new planning run**; the plan at index 0 is written untouched
beside it. `the_deterministic_plan_is_intact` rebuilds the plan with both external files
withheld and compares byte for byte. Component 13 writes its queue unchanged beside its override
log, and the pattern is inherited deliberately.

### Execution records, and cannot edit

`inspection_schedule` has **no `execution_status` column**. There is nothing to write into, and
the check fails if such a column appears at all. Execution lives in `execution_log` and
`execution_summary`; a consumer who wants both joins on the decision key.

`scheduled_date` (what the field asserts) and `plan_scheduled_date` (what the plan held) are
both written and never merged. A field log that disagrees with the plan is a fact about
operations, and overwriting either value would destroy the only evidence of it.

A scheduled row that no event mentions is counted as `no_execution_record` — its own visible
category, never folded into "not completed". Silently treating an absent report as a failure
would manufacture a completion rate out of missing data.

### A re-plan appends; it never mutates

Frozen at every index: **completed** rows keep their slot forever; every row on an operating day
**before the re-plan point** is frozen whatever its status; `original_scheduled_date` and
`original_schedule_rank` are copied forward untouched, so *"where was this originally going to
be?"* survives any number of re-plans.

Ordering is `final_policy_rank` at every index. Nothing is re-ranked at any depth.

The temporal boundary carries exactly **one exemption**, and it is narrow: a row the field
reported as `not_performed` moves even though its day has passed. Freezing it would strand the
inspection the report was filed to rescue. Every other row before the boundary must be identical
between the two plans, and the check proves it by comparing them.

### Why a re-plan backfills where an override does not

The two look like the same operation and are not, so the distinction is written down rather than
left to be inferred. An **excluded** row is a human decision that the slot should not be used,
and re-filling it would overturn that decision. A **day that did not happen** is capacity that
still exists, and refusing to re-plan it would strand it for no reason anybody chose. A
cancellation — by adjustment or in the field — is a removal, and is never backfilled, on the
first rule.

### The reproducibility claim is scoped exactly

> identical recommendation artifact + identical scheduling configuration + identical external
> files → byte-identical tables.

Nothing broader. `DETERMINISM_SCOPE` says so in the manifest.

### The contract is emitted as data

`execution_contract` holds both file formats field by field, with allowed values and meanings,
so a reader who opens the Parquet layer instead of the markdown can still construct a valid
file. A contract that lives only in prose drifts from its parser.

## Alternatives rejected

**One generic override table with a wide action column.** The whole argument above.

**Let an execution event update the schedule row in place.** Easy to read, destroys the record.

**Infer an execution status for unreported rows.** Would manufacture the completion rate.

**Let a re-plan reorder by anything other than policy rank.** An unowned policy layer that
appears only on the second run — the worst possible place for one.

**Reconcile a disagreeing field date onto the plan.** Destroys the evidence that the two ever
differed.

## Consequences

* Three contracts to learn. The data contract documents all three and `execution_contract` ships
  them as data.
* On a run with no external files: two typed **empty** tables, one `replanning_runs` row per cell
  recording the original plan, and every execution number a typed zero. The advisory
  `an_execution_record_was_supplied` fires so nobody mistakes a zero for a measurement.
* The full chain is reconstructible from artifacts alone.

## Limitations

* **No execution event in this repository describes anything that happened in Chicago.** The
  engines are exercised by synthetic fixtures only, and the production run has an empty log.
* Whether the contracts are usable by a real supervisor or a real field inspector is untested —
  the same open question Component 13 recorded for its override contract.
* The adjustment engine handles one cell at a time; an adjustment referencing a cell that was
  not scheduled in this run is logged as `row_not_in_plan` rather than refused.
* Only one re-plan level is exercised in production. The chaining is tested to depth two.

## What this decision does NOT claim

* **Not that real-world execution is deterministic.** It is not, and the scoped claim says so.
* **Not that an execution report is accurate.** It is external, unverified and supplied by a
  person. Separation keeps it from contaminating the plan; it says nothing about its truth.
* **Not that the three-layer split is complete.** A real department would have more kinds of
  human decision than three. These are the three this pipeline can currently represent.
