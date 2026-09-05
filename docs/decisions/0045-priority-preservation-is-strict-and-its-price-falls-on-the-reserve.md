# ADR 0045 — Priority preservation is strict, and its price falls on the coverage reserve

**Status:** Accepted · **Date:** 2026-08-26

## Context

`HANDOFF.md` §16g is explicit about what Component 14 may not do:

> **Re-rank.** Component 13 owns the queue. A scheduler that reordered by risk would be a second
> policy layer with no ADR behind it.

That settles the ordering. What it does not settle — because Component 13 had no calendar with
which to notice — is what happens to the *shape* of the queue when a horizon is too short to
hold it.

`scripts/profile_scheduling.py` profile 8 measured it, and the answer is the most consequential
thing this component found.

Component 13's allocator (`policy/allocation.py`) fills the risk block at ranks `1..n_risk` and
places the coverage reserve after it, at `n_risk+1..k`. **The reserve is therefore always the
tail of the rank order** — profile 8 checks this and finds no exception in any of the 273 cells
that allocate a reserve at all.

A strict-priority schedule fills the horizon from the top. So when the horizon falls short, the
rows that fall off the end are the reserve rows. Every time. Without the scheduler ever looking
at a mechanism.

## Decision

**The schedule preserves `final_policy_rank` exactly, and the measured cost of doing so is
reported at advisory severity and not corrected.**

### Strict priority, with nothing else consulted

`allocation.place()` walks the horizon in date order and the queue in `final_policy_rank` order.
It reads no score, no probability, no mechanism, no eligibility flag and no geography. Under
this strategy `schedule_rank == final_policy_rank` identically and **inversions are zero by
construction** — the production run over 1,260 cells reports 0.

`schedule_rank` is written **beside** `final_policy_rank` on every row rather than replacing it.
That is ADR 0037's pattern one layer further out: where the two agree the policy decided, where
they differ the scheduler did, and `inversion_reason` says which mechanism moved it. An
interviewer can point at any row and ask "did the policy or the calendar put this here?" and the
artifact answers without anyone reading source.

### The inversion machinery exists and measures zero

Spec §9 asks that no inversion be silent. The honest implementation of that requirement is
machinery that *runs* and reports zero, rather than a docstring asserting the property.
`count_inversions` is a Fenwick tree over the placed ranks, `no_inversion_without_a_reason_code`
fails at error severity on any out-of-order row carrying `InversionReason.NONE`, and inversions
arise only from an external adjustment or a re-plan — each of which stamps a reason code.

### The measured price

Across the 273 reserve-bearing cells, on the observed calendar:

> **1,012 of 3,459 coverage-reserve slots — 29.3% — are lost to the horizon. In 136 of 273 cells
> some of the reserve is lost, and in 91 it is lost entirely.**

Per policy, the share lost ranges from 25.2% (`coverage_forced_double_share`) to 51.7%
(`coverage_floor_double_share`).

The `flat_median` scenario loses **zero**, by construction, which is why the two modes are
reported separately and never pooled: averaging a tautology into the finding would halve it.

### It is an advisory, and it must stay one

`the_coverage_reserve_survived_scheduling` never fails a run. The reason is sharper than ADR
0034's general argument: **the cheapest way to turn a red build green here would be to make the
scheduler prefer reserve rows** — which is re-ranking, which `HANDOFF.md` forbids in those
words, and which would move a coverage decision into a layer that does not own policy.

The same logic covers the backlog advisory. 44 of 90 cells cannot fit their queue; a build that
went red on that would be a build that goes green only when the scheduler lies about the
calendar.

### Component 14 reports this and does not fix it

Neither layer is wrong on its own terms. ADR 0037 priced the reserve in forgone citations and
granted it a slot count; nothing in that decision said the slots had to sit at the *end* of the
queue, and Component 13 had no calendar with which to notice that it would matter. The cost
lands on the mechanism the policy layer went to the most trouble to make explicit.

Whether the reserve belongs at the head of the queue instead is a **Component 13 policy
question**, and changing it there is a policy change with its own trade-offs. Component 14 is
not entitled to make it, and making it here would put one coverage decision in two layers.

## Alternatives rejected

**Interleave the reserve through the queue so it survives a short horizon.** The fix everyone
reaches for, and it is re-ranking. It would also silently change what a coverage reserve *is*:
Component 13's reserve is an allocation of the *queue*, and interleaving would make it an
allocation of the *schedule*, which nobody authorised.

**Protect reserve rows from the horizon cutoff.** Same objection, differently dressed. It gives
reserve rows an effective priority they were never granted.

**Raise this to error severity because it is important.** It is important, which is exactly why
it must not be an error. An error creates pressure to make it go away, and the only cheap way to
make it go away is forbidden.

**Report it only in the findings document.** Then it would not travel with the artifact.
`reserve_slots_lost` is a column on `priority_preservation`, a count in the manifest, and a line
in the CLI summary.

## Consequences

* Every run prints the reserve loss. It is the component's headline and it leads `STATUS.md`.
* The `priority_preservation` table keeps mechanism preservation in its own column group,
  separate from priority, coverage and wait, so nothing sums them into one score.
* A green run means the plan was built correctly, and `GREEN_RUN_MEANS` says in the report that
  it does not mean the coverage reserve survived.
* Component 13's open questions gain one: whether the reserve belongs at the tail at all.

## Limitations

* The 29.3% figure depends on the horizon rule. A horizon one day longer would recover much of
  it, and nothing here measures which length is right.
* The measurement is retrospective, like everything else in this project. It says what a stated
  capacity rule would not have fitted, not that any establishment went uninspected.
* Inversions are zero here only because nothing external was supplied. A department using the
  adjustment contract in earnest would see a non-zero count, and its interpretation is unstudied.

## What this decision does NOT claim

* **Not that the reserve loss is Component 13's error.** It is what two individually correct
  layers do when composed.
* **Not that strict priority is the right scheduling policy.** It is the one this data supports
  and the one Component 13's ownership of the queue requires. Whether a department should ever
  depart from it is a question the adjustment contract exists to let them answer for themselves.
* **Not that zero inversions means the schedule is good.** It means the schedule is faithful to
  a queue whose own quality is Component 13's claim, not this one's.
