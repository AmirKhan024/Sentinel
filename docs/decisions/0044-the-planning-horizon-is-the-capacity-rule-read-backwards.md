# ADR 0044 — The planning horizon is the capacity rule read backwards

**Status:** Accepted · **Date:** 2026-08-26

## Context

To lay a queue across days, a scheduler needs two things it does not obviously have: how many
days, and how many slots on each.

Both are places where an arbitrary constant would be invisible. "Plan a two-week horizon at
twenty inspections a day" reads like operations and is two numbers nobody measured, and every
utilisation, backlog and wait figure downstream would inherit them.

Component 5 already refused this once. `evaluation/simulate.py:capacity_k_values` says so:

> Nothing here is a round number chosen for convenience — inventing `k = 50` would make the
> headline operational metric an assumption.

So `k_1_day` is the window's measured median daily inspection rate and `k_1_week` is five times
it. The question for Component 14 is whether it can reach a horizon without adding a number of
its own.

`scripts/profile_scheduling.py` answers it, and finds something better than expected: the
recommendation universe carries `inspection_date`, the date each inspection actually happened.
So the operating days of a fold and the number of inspections performed on each are both
**observations**, not assumptions.

## Decision

**`horizon_days = ceil(k / test_median_daily_capacity)`, taken as a prefix of the fold's own
observed operating days.**

### The rule introduces no new constant

It is `capacity_k_values` read backwards. The two cutoffs that already carry a duration in their
names fall out of it exactly — `k_1_day` spans one day and `k_1_week` spans five, for every
median rate — which is why it is one rule rather than a table of cases. Profile 2 verifies it
across all 90 (fold, capacity) cells and finds **0** that demand more operating days than their
fold contains, so the rule is total and needs no fallback branch. A clamp exists anyway and
records a check, because "unreachable" is a claim about one snapshot.

### The calendar is read, never generated

An operating day is a date the universe carries for that fold. No working-week rule, no holiday
list, no synthesised date. Profile 4 is the argument against the alternative: three inspections
in the snapshot fall on a weekend, so a generated Monday-to-Friday calendar would be wrong at
the edges, and the holiday list it would need is something this project has no way to verify.
Reading the dates costs nothing and imports no assumption.

### Two capacity modes, and only one of them is a measurement

| mode | slots per day | status |
| --- | --- | --- |
| `observed_calendar` | the inspections Chicago actually performed on that date | **measured**, and the default |
| `flat_median` | `test_median_daily_capacity` on every day | **scenario**, labelled everywhere |

The horizon is **identical** in both modes; only the slot counts differ. If a mode changed the
calendar too, the two would not be comparable and the measurement below would be confounded.

### The measurement that makes the default obvious

Profile 3:

> In **44 of 90** (fold, capacity) cells the observed calendar supplies fewer slots than the
> approved queue needs — 48.9% — for a total of **784** recommended inspections that do not fit
> inside their own horizon. Under the flat median the backlog is **zero in every cell**.

The zero is not a better result. It is arithmetic: the horizon is defined as `k / median` days
of `median` slots, so it holds exactly `k`. At `k_1_day` and `k_1_week` the scenario is
**provably tautological** — backlog zero, utilisation exactly 1.000, before anything is
measured.

That is why `observed_calendar` is the default, and why `flat_median` is kept rather than
deleted: the contrast between them *is* the finding. A component that shipped only the scenario
would have reported perfect capacity utilisation on every cell and been describing its own
arithmetic. `CAPACITY_MODE_SCENARIO_CLAIM` states the tautology in the manifest, every scenario
row carries `is_scenario = true`, and an advisory fires whenever scenario rows are written.

### Capacity is never raised

There is no parameter on any function in `horizon.py` that could increase a slot count, no CLI
flag that reaches one, and no `--horizon-days`, `--capacity`, `--slots-per-day` or
`--extend-horizon`. `tests/test_cli_scheduling.py` asserts each absence, because the point of a
missing flag is that it stays missing.

### The horizon anchors on the window's first operating day

Not on the first day above a volume floor, and not on a representative day. Both alternatives
introduce an arbitrary constant and a selection effect at once. The cost is real and measured —
quarter-opening days are systematically thin, and 2024Q1 and 2026Q1 both open on a single
inspection against medians of 34 and 35 — so the advisory `the_horizon_opens_on_a_full_day`
fires and names the cells rather than the code quietly starting somewhere flattering.

## Alternatives rejected

**A configurable horizon length.** The obvious operational affordance, and a way to make any
utilisation number come out right. Rejected: the horizon would become the most consequential
untested parameter in the project.

**Generate the calendar from a working-week rule and a holiday list.** Rejected on profile 4.
Wrong at the edges and unverifiable in the middle.

**Ship only `observed_calendar`.** Loses the ability to say *what Component 13's own capacity
assumption hides*, which is the component's second-most useful output.

**Use the mean daily rate rather than the median.** Would silently change `k`, which is
Component 5's to define. Component 14 does not get to redefine a cutoff it inherited.

## Consequences

* Component 14 contains **no arbitrary constant**. Every number descends from a measurement.
* Backlog and idle capacity are both real, non-degenerate and frequently non-zero — 44 of 90
  cells each — which makes them worth reporting.
* Two configurations run by default, so the scenario's divergence is always visible rather than
  opt-in.
* `flat_median` rows are tautological at two of five cutoffs, and that is documented rather than
  suppressed.

## Limitations

* **The observed calendar is measured from the window it schedules.** A planner standing on day
  one of 2026Q2 would not know that 8 April would hold thirty inspections. It states what
  capacity *existed*, not what a planner could have known; a live deployment would need a
  forecast this project has not built. Component 13 already inherits a version of this — `k`
  descends from the same window — but per-*day* granularity is a step beyond per-*window*, and
  pretending otherwise would be the easiest lie in this component. It is recorded in
  `DOES_NOT_ESTABLISH` and travels in every manifest.
* The horizon is a prefix, so every cell is scheduled against the *start* of its window. A cell
  scheduled against the middle would get different numbers.
* The clamp branch is unexercised on this snapshot, so it is tested and unobserved.

## What this decision does NOT claim

* **Not that the observed calendar is the right planning capacity.** It is the honest
  retrospective one. Whether a department should plan against last quarter's realised volumes is
  an operations question this component does not answer.
* **Not that 784 inspections were missed.** Every number is a re-ordering of inspections that
  already happened. A backlog is what a stated capacity rule would not have fitted.
* **Not that the flat median is wrong.** It is exactly what Component 13's cutoffs assume. The
  finding is that the assumption is optimistic, not that it was unreasonable.
