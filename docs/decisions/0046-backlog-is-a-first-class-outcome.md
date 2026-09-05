# ADR 0046 — Backlog is a first-class outcome, and insufficient capacity is not an error

**Status:** Accepted · **Date:** 2026-08-26

## Context

Component 13 approves `k` establishments. Component 14 lays them across a horizon whose slots
come from the days Chicago actually worked. In **44 of 90** (fold, capacity) cells the second
number is smaller than the first, and **784** approved inspections have nowhere to go.

There are three ways to handle that and two of them are wrong.

**Raise the horizon until everything fits.** Makes the problem disappear and makes every
capacity number meaningless. Forbidden by ADR 0044 and by `HANDOFF.md`'s "do not raise capacity".

**Fail the run.** Superficially rigorous. It would make the build red on 44 of 90 correct cells,
and the only way to make it green would be to lie about the calendar.

**Report it.** Which requires deciding what "it" is — a count, or a population.

There is a fourth failure mode that is subtler than all three: quietly treating an unscheduled
row as *unrecommended*. It is the easiest defect to introduce and the hardest to notice, because
the resulting artifact is internally consistent. "How many did we recommend?" and "how many did
we schedule?" would return the same number, and nobody would be able to see that a third of the
queue had been silently withdrawn.

## Decision

**"Not scheduled" is never redefined as "not recommended", and the backlog is a typed table
rather than a flag.**

### Insufficient capacity is an outcome, not an error

`allocation.place()` returns unreached rows with status `backlog` and reason
`capacity_exhausted_in_horizon`, and the run stays green. A component that raised here would be
refusing to report the thing it was built to find.

The advisory `every_recommendation_was_scheduled` fires and names the count. Its mirror image,
`capacity_is_fully_utilized`, fires on the 679 cells that left slots idle — the calendar being
*more* generous than the cutoff, which is a different problem with the same root and deserves to
be visible too.

### A backlogged row keeps everything Component 13 gave it

`final_policy_rank`, `decision_mechanism`, `decision_reason`, `coverage_eligible`, the scores.
All carried verbatim. The row is still selected; only the calendar ran out.

### The backlog is a population, because a count cannot be acted on

`schedule_backlog` carries, per row:

| column | question it answers |
| --- | --- |
| `backlog_position` | where in the queue's own order this row sits |
| `final_policy_rank` | what Component 13 ranked it |
| `decision_mechanism` | whether risk or coverage put it in the queue |
| `horizon_slots` / `slots_short` | how far short the horizon fell |
| `would_fit_on_day_index` | how much longer this would have taken |
| `first_available_date` | the fold's next operating day past the horizon |

"Ten rows did not fit" is much less useful than "they were ranks 131 to 140, the horizon was ten
slots short, and the next operating day was the 9th". A boolean on the schedule row could carry
none of it.

`would_fit_on_day_index` is **null** when the fold's remaining calendar cannot reach the row at
all. That is a different and worse answer than a large number, and it is the true one.

### The accounting identity is checked, and deferrals are not double-counted

`n_scheduled + n_backlog + n_cancelled == n_recommended`, asserted per cell by `counts_add_up`.

`n_deferred` is deliberately **not** a fourth term. A deferred row still holds a slot — a
deferral moves an inspection, it does not remove it — so it is already inside `n_scheduled`, and
adding it again would double-count exactly the rows a supervisor moved. It is reported as a
breakdown of the scheduled block, and the check asserts that relationship rather than assuming
it. `OCCUPYING_STATUSES` is the same rule expressed in code, so a capacity check cannot let a
day be overbooked by the rows somebody moved onto it.

### Nothing disappears, at any planning index

`backlog_is_exactly_the_unscheduled_remainder` compares the backlog table against the plan per
planning run. `every_selected_recommendation_is_accounted_for` proves every approved row appears
once per configuration. `replan()` raises if any row vanishes during a re-plan — "an
establishment nobody is accountable for" is the failure being prevented.

## Alternatives rejected

**A `scheduled` boolean on the recommendation table.** One column, no new table, and it cannot
carry a shortfall, a position or a next-available date. It would also put a Component 14 fact on
a Component 13 artifact, which ADR 0041 refuses.

**Emit only the count in the summary.** Cheap and useless to the person who has to do something
about it.

**Treat backlog as a failure and exit non-zero.** Red on 44 of 90 correct cells, with the only
remedy being dishonesty.

**Drop backlogged rows entirely.** The silent-withdrawal defect. Named here so it is on record
as considered and refused.

## Consequences

* `schedule_backlog` is non-empty in 308 of 1,260 cells on the observed calendar, and empty by
  construction in every `flat_median` cell — which is itself reported.
* Two advisories in opposite directions, so a reader can distinguish "the queue was too long"
  from "the queue was too short".
* Backlog is emitted **per planning run**, so comparing index 0 with index 1 shows what a
  not-performed day actually cost.

## Limitations

* The backlog is per cell and per horizon. It does not roll forward across folds; there is no
  notion here of a quarter's backlog becoming the next quarter's problem, because each fold is
  scheduled independently.
* `would_fit_on_day_index` extends the fold's own observed calendar at its own observed volumes,
  which is a projection, not a plan.
* A large backlog and a large idle count can co-occur across cells in the same run, and the
  summary does not net them — deliberately, because they are different cells.

## What this decision does NOT claim

* **Not that 784 establishments went uninspected.** Every one of them *was* inspected — this is
  a re-ordering study over inspections that already happened. The backlog is what a stated
  capacity rule would not have fitted.
* **Not that the backlog is anyone's fault.** It is the arithmetic of a quarter-wide median
  meeting particular days.
* **Not that a backlogged establishment is lower risk.** It is lower *ranked*, which is
  Component 13's claim, and it fell outside a horizon, which is this component's.
