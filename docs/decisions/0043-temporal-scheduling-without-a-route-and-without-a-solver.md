# ADR 0043 — Temporal scheduling without a route, and without a solver

**Status:** Accepted · **Date:** 2026-08-26

## Context

"Scheduling" invites two assumptions that this dataset does not support, and both would be
expensive to discover late.

The first is that a schedule assigns work to *people*. The raw Chicago food-inspection table has
22 columns and none of them is an inspector. There is no roster, no headcount, no shift, no
specialisation and no base location. That is the same absence ADR 0019 recorded when it blocked
Component 10, and nothing in Components 11, 12 or 13 changed it.

The second is that a schedule assigns work to *places in an order that minimises travel*. There
is no duration, no start or end time, no travel time, no road network and no distance matrix.
Latitude, longitude and community area do exist — and ADR 0033 and ADR 0038 already refuse
geography-keyed allocation on separate grounds, so even the geography that exists may not be
used this way.

`scripts/profile_scheduling.py` profile 7 is the inventory: twelve operational fields a real
inspection department schedules against, all absent.

There is also a promise outstanding. `pyproject.toml` says, in a comment written during
Component 13:

> A policy engine that needed a solver would be describing a different problem — **Component
> 14's** — and the dependency would arrive with it.

Component 14 has arrived. The promise has to be either kept or explicitly discharged.

## Decision

**Component 14 performs temporal and workload scheduling, not geographic route optimisation, and
it adds no solver and no dependency.**

### Routing is refused, and it is repository policy rather than improvisation

The README roadmap already assigns routing to **Component 15 ("OR-Tools routing")**. Declining
it here is therefore not a limitation this component invented to excuse itself; it is the
boundary the project drew before this component existed. Component 15 is itself blocked on the
same missing data.

`SCHEDULING_SEMANTICS` states it in the words the component must be described in, and travels in
every manifest:

> Component 14 performs temporal and workload scheduling, not geographic route optimisation. The
> dataset has no inspector, no shift, no duration, no travel time and no road network, so a
> route here is not underdetermined — it is unrepresented.

A day's slots are a **workload count**, never a route. Slot index 4 means "the fourth inspection
that day", not "the fourth stop".

### No solver, and the promise is kept by checking rather than by assuming

Four reasons, in decreasing order of how much they would have cost to discover late:

1. **Every constraint an optimiser would trade off is absent.** Travel time, duration, inspector
   count, skill matching, appointment windows. An OR-Tools model over Sentinel's columns would
   be a model over invented parameters, and its output would be an optimality *claim* backed by
   fabricated inputs — the exact thing the specification's "no fake optimization claims" rule
   forbids, and the same failure mode ADR 0034's advisory boundary exists to refuse.

2. **There is nothing to search over.** The only objective this component could state is
   "preserve Component 13's priority order", and strict priority preservation has a closed form:
   walk the days in date order, walk the queue in rank order, stop when one runs out. It is a
   prefix operation. `final_policy_rank` is unique and contiguous — checked by Component 13 at
   error severity — so the assignment is not merely optimal, it is *unique*.

3. **A solver would break determinism in the one way that matters.** CP-SAT with equal objective
   values returns a search-order-dependent solution. This project's byte-identity contract and
   its shuffled-input invariance both require the opposite.

4. **A dependency is a claim about the problem.** ADR 0015 and ADR 0016 both argue that a library
   arrives when the problem needs an *algorithm* rather than a formula. This problem needs a
   formula.

`pyproject.toml`'s comment block is amended to record that Component 14 arrived and the solver
did not, so the standing rule stays honest rather than quietly unredeemed.

### The allocation is described as what it is

`ALLOCATION_CLAIM`, in the manifest on every run:

> deterministic greedy slot allocation down an approved rank order. Not optimal, not optimised,
> and no objective function is defined or solved anywhere in this component.

### There is one scheduling strategy, and the count is a finding

The specification offers a "constraint-aware" strategy conditional on real or explicitly
configured operational constraints existing. Profile 7 shows none do, so a second strategy would
be a strategy over invented inputs. `NO_CONSTRAINT_AWARE_STRATEGY` records the omission and its
reason, and there is deliberately **no `constraint_adjusted` reason code** — a code no run can
emit is indistinguishable from one that is broken, which is the rule `policy/definitions.py`
already applies to mechanisms.

The supported way to move a row is the external adjustment contract, because that carries an
actor, a reason code and a timestamp: a constraint somebody is accountable for.

## Alternatives rejected

**Synthesise inspector assignments from a plausible headcount.** Divide the daily slots by an
assumed number of inspectors. Produces a schedule that looks operational, and every row of it is
a claim about staffing this project has no evidence for.

**Estimate travel time from straight-line distance and an assumed speed.** The most seductive
option, because lat/lon really are present. It would produce routes that look authoritative and
describe nothing — and it would additionally violate ADR 0033 and ADR 0038.

**Add OR-Tools now and use it trivially, so the dependency is in place for Component 15.**
Rejected on ADR 0015's rule: a dependency arrives with the component that needs it, and one
added early is one nobody has a reason to keep correct.

**Call the greedy allocation "optimised" because it is optimal for its objective.** True and
misleading. A reader who sees "optimised schedule" will assume travel, staffing and durations
were considered.

## Consequences

* Component 14 adds **zero** runtime dependencies, like Components 11, 12 and 13 before it.
* `scheduling/allocation.py` is small — the whole algorithm is one loop — and its smallness is
  the argument, not an embarrassment.
* Every artifact and the manifest state what is not done, so the boundary travels with the data
  rather than living only here.
* Component 15 remains blocked, and this ADR records why.

## Limitations

* A day's slot count says nothing about whether those inspections are geographically reachable
  in a day. If they are spread across the city, the plan may be infeasible in a way this
  component cannot see and does not claim to.
* `slot_index` orders inspections within a day, and that order is priority, not efficiency. A
  real department would reorder within the day for travel, and nothing here would notice.
* The refusal is contingent on the data. If a roster or a duration field is ever ingested, this
  decision should be revisited rather than treated as settled.

## What this decision does NOT claim

* **Not that routing does not matter.** It matters a great deal to a real inspection department.
  It is unrepresented here, which is a different statement.
* **Not that a solver would be wrong for this problem in general.** It would be wrong for *this
  problem on this data*, where there is nothing to search over and nothing to trade off.
* **Not that the schedule is optimal in any sense a reader would recognise.** It is the unique
  strict-priority assignment. That is a much narrower claim than "optimal".
* **Not that Component 15 is merely unbuilt.** It is unbuildable on this dataset, for the same
  reason Component 10 is.
