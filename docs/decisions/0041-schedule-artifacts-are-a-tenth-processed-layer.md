# ADR 0041 — Schedule artifacts are a tenth processed layer

**Status:** Accepted · **Date:** 2026-08-26

## Context

Component 13 produces `inspection_recommendations`: for every scored inspection, under every
policy, at every capacity cutoff, whether it is in the queue and why. `HANDOFF.md` §16g calls it
"the first component whose output is an instruction rather than a description", and that is
right as far as it goes. But its own `CAPACITY_SEMANTICS` says what the instruction is missing:

```text
capacity is a rank position derived from the window's measured median daily inspection rate
```

A rank position is not a plan. Nobody can execute "you are the 137th most important inspection
this quarter". They can execute "you are on Thursday, and you are the fourth inspection that
day". Component 14 produces the second sentence, and the question is where it goes.

There were three candidate homes and all three were wrong.

**`policy/`** is the closest and the most dangerous. Its grain is a *decision*: one
establishment, one operating period, one capacity assumption, one policy. Component 14's grain
is a decision **plus a date plus a slot plus a planning run** — strictly finer, and finer in a
dimension `policy/` has no column for. Worse, the two change for different reasons: a queue
changes when a department changes its mind about coverage, a schedule changes when a Tuesday
turns out to hold sixteen inspections instead of twenty-eight. Filed together there would be no
convention saying which is which, and a reader who found two `n_selected`-shaped numbers in one
directory would have no way to tell which one answered their question.

**`evaluation/`** owns what a fold is, what a capacity cutoff is, and what a window's median
daily rate is. Component 14 consumes all three and redefines none. Writing a second set of
per-fold capacity numbers into that directory would put two authoritative answers to
"how much capacity did this window have" in one place, which is how a project starts quoting the
flattering one.

**`predictions/`** is refused for the reason every layer since ADR 0024 has been refused it. A
prediction is a belief about the world; a schedule is an instruction to a person. Components 6
to 9 stay the only producers of scores, and nothing written here is a score.

## Decision

**Component 14 writes to `data/processed/scheduling/`, a tenth processed layer.**

Thirteen tables, keyed to `inspection_schedule`, with the manifest as
`manifest_inspection_schedule_<stamp>.json`. `Settings.scheduling_processed_dir` states the
grain, the near-collisions above and the join prohibition below in its own docstring, so the
boundary is visible from the configuration rather than only from this file.

### The grain is a slot, and it is genuinely new

One row per (capacity mode, policy, model, fold, capacity level, planning run, scored
inspection). Two of those dimensions — planning run and capacity mode — do not exist anywhere
upstream, and the slot itself (`scheduled_date`, `day_index`, `slot_index`) is the thing the
component was built to produce.

### The layer holds the queue, not the universe

Component 13's recommendation table is universe-grained on purpose: only a universe-grained
artifact can answer *why was this establishment not inspected*. That question is already
answered one layer up, so repeating 1.4 million rows here would restate an answer rather than
add one. `inspection_schedule` therefore holds the approved queue — the rows Component 13
selected — and `schedule_backlog` holds the ones the calendar could not reach.

### Nothing in this layer may be joined onto a feature table

The same prohibition ADR 0036 placed on `policy/`, one layer further out and for a stronger
reason. A schedule is downstream of every model, every policy *and* every human decision in this
project. Joined back onto training rows it would make the system's own past scheduling decisions
an input to its future risk estimates — closing the feedback loop Component 12 measured and
Component 13 was built to keep visible.

## Alternatives rejected

**Add date and slot columns to `inspection_recommendations`.** Cheapest, and it destroys the
thing that makes Component 13 auditable. That table's contract says every cutoff is a rank
position; adding a date would make it a plan, and the fourteen upstream checks that police the
recommendation contract would silently be policing a different artifact. It would also make
Component 13's output non-reproducible without Component 14's inputs.

**One table with a `layer` discriminator column.** Sounds tidy and makes every schema the union
of two schemas, so every consumer filters before reading and a forgotten filter is a wrong
answer that looks right.

**Put the schedule in `interim/`.** ADR 0005 reserves `processed/` for analysis- and
model-ready tables, and a schedule is exactly that: it is the artifact an operations reader
opens. Nothing downstream consumes it as a mid-pipeline key mapping.

## Consequences

* Ten processed layers now. The count is a cost, and the alternative — layers whose grain is
  "roughly the previous one plus a bit" — is a worse one.
* A reader answering "when is this establishment being inspected, and why then?" joins
  `inspection_schedule` to `inspection_recommendations` on the decision key. The join is the
  point: it is the moment where the two facts are visibly two facts.
* Every table carries `schedule_definition_version`, so two artifacts can be told apart when the
  horizon rule or either external contract changes.

## Limitations

* The layer does not carry an execution status. That is ADR 0047's decision, not this one's, but
  it means a reader wanting "what actually happened" needs a second join.
* Ten layers is a lot to hold in one's head, and the README's repository-layout section is now
  the only place the whole set is listed together.

## What this decision does NOT claim

* **Not that the schedule is more authoritative than the recommendation.** It is strictly
  downstream. Where the two disagree about a rank, the recommendation is right and the schedule
  has a bug.
* **Not that a tenth layer will be the last.** Component 15's routing, if the data for it ever
  arrives, would be a different grain again — a slot plus a route position plus an inspector —
  and would need its own decision.
* **Not that the layer is safe to join anywhere.** It is safe to join to Component 13 on the
  decision key for reading. It is never safe to join onto a feature table, and that is a
  prohibition rather than a caution.
