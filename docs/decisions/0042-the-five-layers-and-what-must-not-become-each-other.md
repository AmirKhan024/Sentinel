# ADR 0042 — The five layers, and the four things that must not become each other

**Status:** Accepted · **Date:** 2026-08-26

## Context

By Component 14 the pipeline is five layers deep:

```text
model         a calibrated probability that this establishment will be cited
policy        who is in the approved queue at this capacity, and by which mechanism
recommendation the queue itself, ranked
schedule      which operating day and which slot each approved row occupies
execution     what a person reports actually happened
```

Each boundary is a place where a plausible shortcut collapses two different facts into one. All
four shortcuts produce artifacts that look right, and three of them are the kind of defect that
only becomes visible when somebody asks a question the artifact can no longer answer.

**A probability becoming a date.** The shortcut is to schedule directly off the score — sort by
probability, deal out days. It skips the policy layer entirely, and with it the coverage
allocation, the mechanism column and every reason code. The queue looks identical most of the
time, which is what makes it dangerous.

**A scheduling rule editing a probability or a rank.** The shortcut is to nudge a rank so a
schedule fits more tidily. After that, "why is this establishment fourth?" has one answer where
it should have two, and the question ADR 0037 exists to keep answerable — did the model decide
this, or did the policy? — acquires a third possible answer nobody can distinguish.

**An execution outcome changing a recommendation.** The shortcut is a single mutable row per
establishment that gets updated as things happen. It makes the current state easy to read and
destroys the historical record: after the update, nobody can say what was recommended before the
inspection was attempted, and the system's own past decisions become unreconstructible.

**An execution outcome changing a plan that was already written.** The same shortcut one layer
in. A plan that is edited in place cannot be compared with the plan that replaced it, so "what
did this Tuesday's cancellation actually cost?" has no answer.

## Decision

**The five layers are separate artifacts, and the four boundaries are enforced structurally
rather than by convention.**

### A probability never becomes a date without passing through a rank

`scheduling/allocation.py` consumes `final_policy_rank` and the horizon. It does not read
`score`, `base_score`, `decision_mechanism`, `coverage_eligible` or any geography, and the type
system helps: `Placement` carries no score for a placement function to reach. The scores travel
on the output row as provenance, copied verbatim, and are never read back.

### A scheduling rule never edits a probability, a rank or a mechanism

`c13_provenance_is_preserved` re-reads the Component 13 artifact after the run and compares
eight columns row by row at error severity. It is deliberate duplication of a promise the code
already keeps: the promise is about code that was correct when it was written.

### An execution outcome never changes a recommendation

`inspection_schedule` has **no `execution_status` column**. There is nothing for an execution
event to write into, and `execution_never_alters_a_recommendation` fails the run if such a
column appears at all — not merely if it holds a wrong value. A consumer who wants both facts
joins `execution_log` on the decision key, and the join is the moment where a reader can see
that they are two facts.

### An execution outcome never changes a plan that was already written

A `SchedulePlan` is frozen and `replan()` returns a new one. Both are written, both carry their
own `planning_run_id` and `replan_index`, and `no_execution_event_changes_an_earlier_schedule`
compares run *n* against run *n+1* row by row.

The comparison carries exactly one exemption, and it is narrow on purpose: a row the field
reported as `not_performed` may move even though its day has passed. Freezing it would strand
the inspection the report was filed to rescue. Every other row on a day before the boundary —
completed, cancelled, or simply unreported — must be identical between the two plans.

## Alternatives rejected

**One mutable "current state" table.** What most operational systems do. It answers "what is
happening now" beautifully and "what did we decide, and when, and what changed it" not at all.
This project's whole argument is that the second question is the one that matters.

**Enforce the boundaries in documentation and code review.** Tried implicitly in every project
that has ever lost an audit trail. The reason `inspection_schedule` has no `execution_status`
column is that a column which does not exist cannot be written to by a future edit that seemed
reasonable at the time.

**Let the scheduler break ties by score.** Tempting, and unnecessary: `final_policy_rank` is
unique and contiguous, so there are no ties. Reaching for a score to resolve a tie that cannot
occur would put a score-reading code path in a module whose whole claim is that it has none.

## Consequences

* Answering an operational question often takes a join. That is the cost, and it buys the
  ability to say which layer produced which number.
* The chain *original recommendation → approved recommendation after override → planned schedule
  → scheduling adjustment → execution outcome* is reconstructible end to end, from artifacts,
  without reading source.
* Three separate human-input contracts rather than one generic override (ADR 0047).

## Limitations

* Five layers and thirteen tables is a lot of surface for a reader to learn. The data contract
  is the entry point and it is long.
* The structural guarantees are strong at the boundaries and silent in the middle: nothing stops
  a future component from reading a score *out* of `inspection_schedule` and doing something
  unwise with it elsewhere.

## What this decision does NOT claim

* **Not that the layers are independently valid.** Every layer inherits every limitation of the
  ones above it, and Component 5's re-ordering caveat reaches all the way down.
* **Not that the separation makes any layer correct.** It makes them *distinguishable*. A wrong
  schedule built correctly from a wrong queue is still wrong, and the separation only means
  somebody can tell which of the two to fix.
* **Not that execution data is trustworthy because it is separate.** It is external, unverified
  and supplied by a person. Separation keeps it from contaminating the plan; it says nothing
  about whether the report is accurate.
