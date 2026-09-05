# Component 14 — Operational scheduling and execution planning

Plain language. No prior machine-learning knowledge assumed.

---

## 1. What problem does Component 14 solve?

After thirteen components, Sentinel can hand a department a ranked list: *these thirty
establishments, in this order, at this week's capacity, and here is why each one is on it*.

**A ranked list is not a plan.**

Component 13 says so itself, in the sentence that travels inside every one of its manifests:

> capacity is a rank position derived from the window's measured median daily inspection rate

A rank position is not something a person can execute. Nobody can act on *"you are the 137th most
important inspection this quarter"*. They can act on *"you are on Thursday, and you are the
fourth inspection that day"*.

Component 14 is the layer that turns the first sentence into the second — and, just as
importantly, refuses to do several things that would look like part of the same job.

```text
MODEL LAYER       data -> as-of features -> trained model -> calibrated probability
                  |  a probability is not an action
                  v
POLICY LAYER      capacity + eligibility + governance -> an approved, ranked queue
                  |  a rank is not a date
                  v
SCHEDULING LAYER  observed calendar + slot allocation -> a plan
                  |  a plan is not a record of what happened
                  v
EXECUTION LAYER   external reports -> what actually happened
```

---

## 2. The first question: what can we honestly schedule against?

Before writing any code, we asked what operational information the repository *actually* has.
This matters more than it sounds, because scheduling is a field full of plausible-looking
numbers.

**What exists.** The recommendation table carries `inspection_date` — the date each inspection
really happened. Group by it and you get, for free, the exact days Chicago inspected on and how
many inspections happened on each. For 2026 Q2: 63 operating days, Monday to Friday, with real
daily counts ranging from 1 to 55 around a median of 28.

That is an **observation**, not an assumption. It is the whole foundation of the component.

**What does not exist.** The raw table has 22 columns, and none of them is an inspector. There
is no roster, no headcount, no shift, no base location, no inspection duration, no travel time
and no road network. There is no closure calendar, no appointment window and no statutory
deadline.

So Component 14 can schedule **time and workload**. It cannot schedule **people or routes**, and
it says so in the words it must be described in:

> Component 14 performs temporal and workload scheduling, not geographic route optimisation. The
> dataset has no inspector, no shift, no duration, no travel time and no road network, so a
> route here is not underdetermined — it is unrepresented.

This is not a limitation the component invented to excuse itself. The project's own roadmap
already assigns routing to **Component 15**, and Component 15 is blocked on exactly the same
missing data that blocked Component 10.

---

## 3. How long is the plan, and how many inspections fit in a day?

Two numbers that would be very easy to invent. "Plan two weeks at twenty a day" reads like
operations and is two figures nobody measured — and every utilisation, backlog and delay number
downstream would silently inherit them.

Component 14 invents neither.

**How many days?** Component 5 built the capacity cutoffs from each window's measured median
daily rate: `k_1_day` *is* that rate, and `k_1_week` is five times it. Read that backwards and
you get the horizon:

```text
horizon_days = ceil(k / median_daily_rate)
```

It is not a new number — it is the existing rule inverted. And it reproduces the cutoff names
exactly: `k_1_day` spans one day, `k_1_week` spans five. Checked across all 90 (fold, capacity)
combinations: none demands more days than its fold actually has.

**How many slots per day?** Two answers, and only one of them is a measurement:

| mode | slots per day | what it is |
|---|---|---|
| `observed_calendar` | what Chicago really did that day | **a measurement** — the default |
| `flat_median` | the window's median, every day | **a scenario** — labelled everywhere |

**Why keep the scenario at all?** Because the gap between the two is the component's central
finding, and because the scenario is quietly circular. The horizon is `k / median` days of
`median` slots — so it holds exactly `k`, always. Backlog zero, utilisation exactly 1.000,
before anything is measured. Ship only that mode and the component would have reported perfect
capacity utilisation on every cell while describing nothing but its own arithmetic.

---

## 4. What the calendar actually does to the plan

> **In 44 of 90 (fold, capacity) cells the observed calendar supplies fewer slots than the
> approved queue needs — 48.9% of them — for 784 inspections that do not fit inside their own
> horizon. Under the flat median, the backlog is zero in every single cell.**

This is not a defect in Component 13 and not a defect in the scheduler. It is a fact about how
Chicago works: a median summarises a whole quarter, and the particular days at the *start* of a
quarter run below it. So a cutoff derived from the median promises capacity that the first week
does not have.

Worked example — 2026 Q2, `pure_risk`, one week of capacity:

```text
k                     140          (5 × the median of 28)
horizon               5 operating days
observed volumes      28, 18, 21, 33, 30   =  130 slots
scheduled             130
backlog                10          -- real, and measured

same cell, flat_median:  5 × 28 = 140 slots, backlog 0, utilisation exactly 1.000
```

---

## 5. The finding that surprised us

This is the one worth remembering, and it is a criticism of Component 13 discovered from inside
Component 14.

Component 13 builds its queue in two blocks: the risk block first, at ranks 1 to *n*, and then
the **coverage reserve** — capacity deliberately set aside for establishments the model knows
little about. That reserve was the centrepiece of Component 13. It was argued for, priced in
forgone citations, and given its own ADR.

But it sits at the **tail** of the rank order. Always — checked, with no exception in any of the
273 cells that allocate one.

And a strict-priority schedule fills the horizon from the top.

> **1,012 of 3,459 coverage-reserve slots — 29.3% — are lost to the horizon. 136 of 273 cells
> lose some of it. 91 lose it entirely.**

The scheduler never looks at a mechanism. It does not know which rows are reserve rows. They are
simply last, so they are simply first to fall off the end.

**Neither layer is wrong on its own terms.** ADR 0037 granted the reserve a slot count; nothing
in that decision said the slots had to be at the *end* of the queue, and Component 13 had no
calendar with which to notice that it would matter.

**And Component 14 does not fix it.** Promoting reserve rows in the schedule would be
re-ranking, which Component 13 owns and which the handoff forbids in those words. It would also
put one coverage decision in two different layers, which is how a system stops being auditable.
So the finding is measured, reported at advisory severity, and handed back as an open question.

---

## 6. Can a lower-priority establishment be scheduled first?

Under the reference strategy: **no, never**, and the run proves it rather than asserting it.
Across 1,260 cells the measured inversion count is **0**.

The plan carries `schedule_rank` **beside** `final_policy_rank` rather than replacing it. Where
the two agree, the policy decided. Where they differ, the scheduler did — and an
`inversion_reason` column names the mechanism that moved it. Point at any row and ask *"did the
policy or the calendar put this here?"* and the artifact answers without anyone reading code.

An inversion can only arise from a human adjustment or a re-plan, and every one carries a reason
code. A row that sits out of order carrying the token `none` fails the build.

---

## 7. What happens when things go wrong in the real world?

Three different things can happen, and Component 14 keeps them in three separate places on
purpose.

| what happened | whose decision | what it changes |
|---|---|---|
| a supervisor removes an establishment from the queue | Component 13 override | **who** |
| a supervisor moves an inspection to Thursday | Component 14 adjustment | **when** |
| an inspector reports the place was shut | execution event | **what happened** |

The tempting design is one "override" table for all three. It would destroy the audit trail: a
reader would only be able to reconstruct *"somebody changed something"*.

So there are three contracts with three id namespaces and three non-overlapping vocabularies —
and the code refuses to start if the vocabularies ever collide.

**Insufficient capacity is not an error.** An establishment that does not fit goes to the
**backlog**, keeping its rank, its mechanism and its reason code. "Not scheduled" is never
quietly redefined as "not recommended" — that is the easiest defect in this whole component to
introduce and the hardest to notice, because the resulting artifact looks perfectly consistent.

**A re-plan adds a plan; it never edits one.** Completed inspections keep their slot forever.
Anything on a day before the re-planning point is frozen. Both plans are written, so you can
diff them and see exactly what a cancelled Tuesday cost.

There is one deliberate exception, and it is narrow: an inspection the field reported as *not
performed* moves even though its day has passed. Freezing it would strand the very inspection
the report was filed to rescue.

---

## 8. Is the schedule optimal?

**No, and it is described as what it is:** deterministic greedy allocation down an approved rank
order. No objective function is defined anywhere in the component and none is solved.

We were explicitly expected to consider a solver here — `pyproject.toml` had promised since
Component 13 that "a policy engine that needed a solver would be describing a different problem
— Component 14's — and the dependency would arrive with it".

It did not arrive, and the promise was kept by checking rather than assuming:

1. **There is nothing to search over.** The only objective this component could state is
   "preserve Component 13's order", and that has a closed form: walk the days in date order,
   walk the queue in rank order. The rank is unique and gapless, so the answer is not just
   optimal — it is *unique*.
2. **Every constraint an optimiser would trade off is missing.** Travel time, duration,
   inspector count, skill matching. A solver over these columns would be optimising invented
   parameters and reporting an optimality claim backed by fabricated inputs.
3. **A solver would break reproducibility.** Given two equally good answers it returns whichever
   its search reached first. This project requires byte-identical output.

---

## 9. How do we know the numbers are real?

* **13 of 13 tables byte-identical** across two independent production runs.
* Identical again under **shuffled** recommendation rows, adjustment rows and execution rows —
  the order rows happen to sit in a file can never change the plan.
* Every input's checksum compared **before and after** the run: Component 14 modifies nothing
  upstream.
* **28 error checks** and **7 advisories**, with the split deliberate: a defect in the
  *computation* fails the build; a finding about what the schedule *cost* never does.
* **450 tests**, including all seventeen deliberately injected failures — a day one slot over
  capacity, an establishment booked twice, a row that silently vanishes, an execution outcome
  that edits a recommendation.

The advisory split matters more than it looks. If "the coverage reserve lost 1,012 slots" failed
the build, the cheapest way to make the build green would be to make the scheduler prefer
reserve rows — which is re-ranking, which is forbidden. A rule that makes the forbidden thing
the easy thing is a bad rule.

Every validation report opens with:

> **A green run means the plan was built correctly. It does not mean the city has enough
> capacity, it does not mean the schedule is the right one, and it does not mean the coverage
> reserve survived.**

---

## 10. What Component 14 explicitly does not claim

* **Not that it is the right schedule.** It is the schedule one stated strategy produces from
  one approved queue under one measured calendar.
* **Not that 784 inspections were missed.** Every one of those establishments *was* inspected —
  this is a re-ordering of history, not a forecast. The backlog is what a stated capacity rule
  would not have fitted.
* **Not that the calendar is knowable in advance.** It is measured from the very window it
  schedules. A planner standing on 1 April would not know that 8 April would hold thirty
  inspections. A live deployment would need a forecast this project has not built — and saying
  otherwise would be the easiest lie in the component.
* **Not that the reserve loss is Component 13's mistake.** It is what two individually correct
  layers do when composed.
* **Nothing at all about inspectors.** The dataset names none, and nothing here infers one.
* **Not that a day's inspections are geographically reachable.** They are a workload count, not
  a route, and the component cannot see the difference.

---

## 11. The 60-second version

> Sentinel could already tell you *who* to inspect and why. It could not tell you *when*, because
> its capacity was a rank position — and nobody can act on "you are 137th in the queue".
>
> Component 14 closes that gap using only real information. The operating calendar is read out
> of the data rather than assumed: the days Chicago actually inspected on, and the real number
> of inspections performed on each. The horizon length is the existing capacity rule read
> backwards, so the component introduces no new constant at all.
>
> Doing that surfaced two things. First, Component 13's capacity assumption is optimistic: in
> half of all cells the real calendar cannot fit the queue that assumption approved. Second — and
> this is the one I'd lead with — Component 13's coverage reserve sits at the *tail* of the rank
> order, so a short horizon eats it first. Nearly 30% of it never gets scheduled, and in a third
> of cells it vanishes completely.
>
> The component reports that and deliberately does not fix it, because fixing it would mean
> re-ranking, and Component 13 owns the ranking. It also refuses to do route optimisation, since
> the dataset has 22 columns and not one of them is an inspector.
>
> What it delivers is an auditable path from risk prediction to policy decision to recommended
> queue to a dated, slotted plan — with every human decision and every real-world outcome
> recorded separately, so you can always tell which layer produced which number.
