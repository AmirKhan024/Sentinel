# Component 14 — operational scheduling and execution planning: findings

**Snapshot:** `inspection_recommendations_20260826T075812Z.parquet` (1,453,760 rows) ·
`evaluation_folds_20260824T160045Z.parquet` (18 folds) ·
`food_inspections_20260816T070911Z.parquet` (22 columns)

**Produced by:** `scripts/profile_scheduling.py`, read-only, before any scheduling code was
written. Nothing here fits, scores, ranks or recalibrates anything: every model is frozen and
Component 13's queue is read, never recomputed.

---

## 0. The question, and the answer the calendar gave

Component 13 produces an ordered queue and stops. Its own `CAPACITY_SEMANTICS` says why:

```text
capacity is a rank position derived from the window's measured median daily inspection rate
```

A rank position is not a plan. Nobody can execute "you are the 137th most important inspection
this quarter" -- they can execute "you are on Thursday". The gap between those two sentences is
this component, and the first question is whether the repository holds enough real information
to cross it honestly.

**It does, for time. It does not, for people or routes.**

The operating calendar is an observation: `inspection_date` on the recommendation universe
gives the exact dates Chicago inspected on and how many inspections happened on each. The
inspector, the duration, the travel time and the route are not in the dataset at all -- the
same absence ADR 0019 recorded when it blocked Component 10.

So Component 14 schedules **time and workload**, and refuses **routing**. Profile 7 is the
inventory that decision rests on.

---

## 1. The measurement that decides the component

Component 13's cutoffs descend from a **quarter-wide median**. Build a schedule on that same
median and it is feasible before anything is measured: the horizon is `k / median` days of
`median` slots, so it holds exactly `k` and the backlog is zero by construction.

Build it on the days Chicago actually worked and it is not.

> **In 44 of 90 (fold, capacity) cells -- 48.9% -- the observed calendar supplies fewer slots
> than the approved queue needs, for a total of 784 recommended inspections that do not fit
> inside their own horizon.**

That is not a defect in Component 13 and not a defect in the scheduler. It is a property of
Chicago's operating pattern: a median summarises a quarter, and the particular days at the
start of a quarter run below it. A cutoff derived from the median therefore promises capacity
that the first week does not have.

This single number fixes three things:

| decision | fixed to | because |
|---|---|---|
| `DEFAULT_CAPACITY_MODE` | `observed_calendar` | it describes days that happened |
| `flat_median` | retained, labelled **scenario** | it reproduces C13's stated semantics, and the contrast *is* the finding |
| `horizon_capacity_meets_the_queue` | **advisory**, not error | a short week is a fact about the city, not a bug in the run |

Had only `flat_median` been implemented, the component would have reported 100% capacity
utilisation and zero backlog on every cell, and would have been describing its own arithmetic.

---

## 2. What the profiles fixed

| profile | fixes |
|---|---|
| 1 `operating_calendar` | that an observed-calendar mode is possible at all |
| 2 `horizon_rule` | `ceil(k / median_daily)`, and that it is **total** -- 0 of 90 cells overrun |
| 3 `observed_versus_flat` | `DEFAULT_CAPACITY_MODE`, and the advisory threshold |
| 4 `weekday_shape` | that the calendar is **read**, never synthesised from a working-week rule |
| 5 `queue_recurrence` | that establishment uniqueness is an **advisory**, inspection uniqueness an **error** |
| 6 `mechanism_mix` | that C13 provenance is carried forward, never recomputed |
| 7 `absent_operational_data` | what the component refuses to build |
| 8 `reserve_survives_scheduling` | **the headline** -- and the `the_coverage_reserve_survived_scheduling` advisory |

Two of these are refusals, and they matter as much as the rules.

**Profile 4** was run to reject a shortcut. A scheduler could synthesise Monday-to-Friday plus a
holiday list -- and it would be wrong at the edges (3 inspections fall on a weekend) and would
need a holiday calendar this project has no way to verify. Reading the dates out of the artifact
imports no assumption, so that is what the component does.

**Profile 5** was run because "no establishment occupies two slots" is the invariant a scheduler
*wants* to assert, and on this data it is false. 1,573 establishment-fold pairs hold more than
one scored canvass; 1 of 2,890 rows in the `k_1_week` queue is a repeat. Component 13's grain is
the scored inspection **event**, and two canvasses of one premises in one quarter are two real
opportunities. Asserting the stronger invariant would have produced a red build on correct data,
which is the failure mode that makes a suite stop being believed.

---

## 2a. The headline: what the calendar does to the coverage reserve

Component 13 allocates a coverage reserve as a slot count, fills the risk block at ranks
`1..n_risk`, and puts the reserve at ranks `n_risk+1..k`. **The reserve is therefore always the
tail of the rank order** -- checked here, with no exception in any of the 273 cells that
allocate one.

A strict-priority schedule fills the horizon from the top of that order. So when the horizon
falls short, the rows that fall off the end are the reserve rows, every time:

> **Strict-priority scheduling loses 1,012 of 3,459 coverage-reserve slots to the horizon --
> 29.3%. In 136 of 273 cells some of the reserve is lost, and in 91 it is lost entirely.**

Neither layer is wrong on its own terms. ADR 0037 priced the reserve in forgone citations and
granted it a slot count; nothing in that decision said the slots had to sit at the *end* of the
queue, and Component 13 had no calendar with which to notice that it mattered. The cost lands
on the mechanism the policy layer went to the most trouble to make explicit.

**Component 14 reports this and does not correct it.** Promoting reserve rows in the schedule
would be re-ranking -- which `HANDOFF.md` forbids in those words -- and would put one coverage
decision in two layers at once. The advisory fires; the build stays green. The cheapest way to
turn a red build green here would be to make the scheduler prefer reserve rows, which is
precisely the change nobody is entitled to make in this component.

This is the second time Sentinel has measured a prior component's premise instead of inheriting
it. Component 13 refuted the finding it was scoped around; Component 14 finds that Component
13's central mechanism is substantially notional once a real calendar is applied.

---

## 3. What these findings do not establish

* **Not that the schedule is optimal.** It is deterministic greedy allocation down an approved
  rank order. No objective function is defined and none is solved.
* **Not that 784 inspections were actually missed.** Every number here is a re-ordering of
  inspections that already happened -- Component 5's limitation, inherited whole. The backlog is
  what a stated capacity rule would not have fitted, not a count of neglected establishments.
* **Not that the observed calendar is the future calendar.** It is the calendar of a quarter
  that has already closed.
* **Not anything about inspectors.** There are none in this dataset, and nothing here infers
  one.
* **Not that the reserve loss is Component 13's error.** It is what two correct layers do
  when composed. Whether the reserve belongs at the head of the queue instead is a policy
  question this component is not entitled to answer.
* **Not that the observed calendar is knowable in advance.** It is measured from the window
  it schedules. It states what capacity existed, not what a planner could have known on day
  one; a live deployment would need a forecast this project has not built.

---

## 4. The profiles


### operating_calendar

**An operating calendar exists, and it is an observation rather than an assumption.**
Every fold's test window resolves to a set of distinct dates on which inspections were
actually performed, with a real count on each. The last column is Component 5's own
`test_median_daily_capacity`, recomputed here from the recommendation artifact as a
cross-check: it agrees with the median of the per-day counts, which is what it is
defined to be.

The spread is the part that matters. A fold whose median day holds 28 inspections has
days holding 1 and days holding 55, so a schedule built on the median alone is not
describing any particular day.

| fold | operating days | min/day | median/day | mean/day | max/day | C5 median |
|---|---|---|---|---|---|---|
| covid_shift-2020H2-2021 | 390 | 1 | 22.0 | 22.7 | 52 | 22 |
| quarterly-2022Q2 | 63 | 2 | 29.0 | 28.0 | 45 | 29 |
| quarterly-2022Q3 | 64 | 2 | 28.0 | 27.1 | 44 | 28 |
| quarterly-2022Q4 | 61 | 1 | 30.0 | 27.9 | 50 | 30 |
| quarterly-2023Q1 | 59 | 1 | 31.0 | 30.5 | 49 | 31 |
| quarterly-2023Q2 | 63 | 1 | 29.0 | 28.4 | 51 | 29 |
| quarterly-2023Q3 | 63 | 1 | 26.0 | 26.2 | 49 | 26 |
| quarterly-2023Q4 | 60 | 7 | 33.5 | 32.6 | 53 | 33 |
| quarterly-2024Q1 | 59 | 1 | 34.0 | 32.4 | 56 | 34 |
| quarterly-2024Q2 | 59 | 16 | 38.0 | 37.2 | 53 | 38 |
| quarterly-2024Q3 | 64 | 4 | 33.5 | 33.2 | 59 | 33 |
| quarterly-2024Q4 | 62 | 1 | 40.0 | 36.3 | 56 | 40 |
| quarterly-2025Q1 | 58 | 7 | 45.0 | 42.4 | 68 | 45 |
| quarterly-2025Q2 | 62 | 1 | 39.0 | 36.9 | 62 | 39 |
| quarterly-2025Q3 | 64 | 3 | 28.0 | 27.4 | 44 | 28 |
| quarterly-2025Q4 | 61 | 3 | 30.0 | 29.0 | 54 | 30 |
| quarterly-2026Q1 | 57 | 1 | 35.0 | 33.6 | 61 | 35 |
| quarterly-2026Q2 | 63 | 1 | 28.0 | 26.0 | 55 | 28 |

### horizon_rule

**The horizon rule is `ceil(k / test_median_daily_capacity)`, and it is total.**
Across every (fold, capacity) cell, **0** demand more operating days than
their fold contains. The rule therefore needs no fallback branch, and Component 14 can
refuse to invent one -- a fallback nothing exercises is a fallback nobody has tested.

The widest cell is `covid_shift-2020H2-2021 / k_pct_10`, which spans 41 of 390 available
days.

| fold | operating days | k_pct_01 | k_pct_05 | k_pct_10 | k_1_day | k_1_week |
|---|---|---|---|---|---|---|
| covid_shift-2020H2-2021 | 390 | 4 | 21 | 41 | 1 | 5 |
| quarterly-2022Q2 | 63 | 1 | 4 | 7 | 1 | 5 |
| quarterly-2022Q3 | 64 | 1 | 4 | 7 | 1 | 5 |
| quarterly-2022Q4 | 61 | 1 | 3 | 6 | 1 | 5 |
| quarterly-2023Q1 | 59 | 1 | 3 | 6 | 1 | 5 |
| quarterly-2023Q2 | 63 | 1 | 4 | 7 | 1 | 5 |
| quarterly-2023Q3 | 63 | 1 | 4 | 7 | 1 | 5 |
| quarterly-2023Q4 | 60 | 1 | 3 | 6 | 1 | 5 |
| quarterly-2024Q1 | 59 | 1 | 3 | 6 | 1 | 5 |
| quarterly-2024Q2 | 59 | 1 | 3 | 6 | 1 | 5 |
| quarterly-2024Q3 | 64 | 1 | 4 | 7 | 1 | 5 |
| quarterly-2024Q4 | 62 | 1 | 3 | 6 | 1 | 5 |
| quarterly-2025Q1 | 58 | 1 | 3 | 6 | 1 | 5 |
| quarterly-2025Q2 | 62 | 1 | 3 | 6 | 1 | 5 |
| quarterly-2025Q3 | 64 | 1 | 4 | 7 | 1 | 5 |
| quarterly-2025Q4 | 61 | 1 | 3 | 6 | 1 | 5 |
| quarterly-2026Q1 | 57 | 1 | 3 | 6 | 1 | 5 |
| quarterly-2026Q2 | 63 | 1 | 3 | 6 | 1 | 5 |

### observed_versus_flat

**In 44 of 90 (fold, capacity) cells the observed calendar
supplies fewer slots than the queue needs** -- 0.4889 of them -- for a total
of **784** recommended inspections that do not fit inside their own
horizon. Under the flat median the backlog is zero in every cell, by construction.

That contrast is the finding, and it is a fact about Chicago's operating pattern rather
than about the scheduler: early-quarter days run below the quarter's median, so a
cutoff derived from the median promises capacity the first week does not have.

It also fixes two decisions. `observed_calendar` is the **default** capacity mode,
because it describes days that happened; `flat_median` is retained as an explicitly
labelled **scenario**, because it reproduces Component 13's own stated capacity
semantics and the comparison between them is the measurement above. And the
`horizon_capacity_meets_the_queue` advisory fires on exactly the cells counted here.

Row-by-row for `quarterly-2026Q2`:

| capacity | k | horizon days | flat slots | observed slots | backlog flat | backlog observed |
|---|---|---|---|---|---|---|
| k_pct_01 | 16 | 1 | 28 | 28 | 0 | 0 |
| k_pct_05 | 82 | 3 | 84 | 67 | 0 | **15** |
| k_pct_10 | 164 | 6 | 168 | 160 | 0 | **4** |
| k_1_day | 28 | 1 | 28 | 28 | 0 | 0 |
| k_1_week | 140 | 5 | 140 | 130 | 0 | **10** |

### weekday_shape

**The calendar is read, never generated.** Inspections concentrate on weekdays, but
**3** fall on a Saturday or Sunday. A synthesised Monday-to-Friday calendar
would silently discard those days, and would also need a holiday list this project does
not have and has no way to verify.

Reading the operating dates out of the artifact costs nothing and imports no
assumption, so that is what Component 14 does.

| weekday | inspections | share |
|---|---|---|
| Mon | 8035 | 0.1934 |
| Tue | 8862 | 0.2134 |
| Wed | 9061 | 0.2181 |
| Thu | 8066 | 0.1942 |
| Fri | 7509 | 0.1808 |
| Sat | 2 | 0.0000 |
| Sun | 1 | 0.0000 |

### queue_recurrence

**An establishment can legitimately recur inside one fold, and rarely does inside one
queue.** 1573 establishment-fold pairs hold more than one scored canvass
across the universe, and the table below counts how many survive into each queue.

This fixes a validation decision. Uniqueness is an **error** check on
`target_inspection_id` -- booking one inspection into two slots is always a defect --
and only an **advisory** on `establishment_id`, because two canvasses of one premises
in one quarter is something Chicago did, not something the scheduler got wrong.

Asserting the stronger invariant would have produced a red build on correct data,
which is the failure mode that makes a suite stop being believed.

| capacity | selected rows | unique inspections | unique establishments | recurring |
|---|---|---|---|---|
| k_pct_01 | 415 | 415 | 394 | 0 |
| k_pct_05 | 2076 | 2076 | 1801 | 1 |
| k_pct_10 | 4154 | 4154 | 3418 | 3 |
| k_1_day | 578 | 578 | 527 | 0 |
| k_1_week | 2890 | 2890 | 2376 | 1 |

### mechanism_mix

**The mechanism travels on the row, so the schedule carries it forward unchanged.**
A scheduled inspection can therefore answer two separate questions -- *why was this
recommended* and *why is it on this day* -- without either answer being reconstructed.

Component 14 reports this mix and never alters it. Promoting a coverage-reserve row to
an earlier day, or demoting one to make a schedule tidier, would be a second policy
decision taken with no ADR behind it.

| policy | mechanism | selected rows | share of queue |
|---|---|---|---|
| coverage_floor_double_share | coverage_reserve | 29 | 0.0029 |
| coverage_floor_double_share | risk_priority | 10084 | 0.9971 |
| coverage_floor_half_share | risk_priority | 10113 | 1.0000 |
| coverage_floor_population_share | coverage_reserve | 1 | 0.0001 |
| coverage_floor_population_share | risk_priority | 10112 | 0.9999 |
| coverage_forced_double_share | coverage_reserve | 1994 | 0.1972 |
| coverage_forced_double_share | risk_priority | 8119 | 0.8028 |
| coverage_forced_half_share | coverage_reserve | 463 | 0.0458 |
| coverage_forced_half_share | risk_priority | 9650 | 0.9542 |
| coverage_forced_population_share | coverage_reserve | 972 | 0.0961 |
| coverage_forced_population_share | risk_priority | 9141 | 0.9039 |
| pure_risk | risk_priority | 10113 | 1.0000 |

### absent_operational_data

**The raw snapshot has 22 columns and none of them is an inspector.**
That is the same absence ADR 0019 recorded when it blocked Component 10, and it decides
the shape of this component: Sentinel can schedule *time* and *workload*, and it cannot
schedule *people* or *routes*.

Component 14 therefore performs temporal and workload scheduling, **not** geographic
route optimisation. The README roadmap already assigns routing to Component 15, so
declining it here is repository policy rather than an improvisation -- and Component 15
is itself blocked on the same missing data.

| operational field | in snapshot | note |
|---|---|---|
| inspector identity | absent | no inspector column anywhere in the snapshot (ADR 0019) |
| inspector roster / headcount | absent | not published with the inspection data |
| inspector working hours or shifts | absent | not published |
| inspector base location | absent | not published |
| inspection duration / time on site | absent | no start or end time, only a date |
| travel time between establishments | absent | no route, no distance, no timestamp |
| road network / drive-time matrix | absent | outside the scope of the Socrata dataset |
| service territory assignment | absent | not published; district is inferable only by geography |
| appointment windows | absent | inspections are unannounced; no such field exists |
| establishment closure / unavailability | absent | no availability calendar is published |
| statutory deadline per establishment | absent | not in the dataset |
| execution status of a planned inspection | absent | the dataset records completed inspections only |

What *is* present, and is what the component is built from:

| present and used | source |
|---|---|
| operating dates | `inspection_date` on the recommendation universe |
| inspections per operating day | the same column, grouped |
| median daily capacity | Component 5 `test_median_daily_capacity` |
| capacity cutoff k | Component 13 `k` |
| priority order | Component 13 `final_policy_rank` |
| selection provenance | Component 13 `decision_mechanism` / `decision_reason` |

### reserve_survives_scheduling

**The coverage reserve is always the tail of the rank order.** In 0 of
273 reserve-bearing cells does a reserve row outrank a risk row -- the
allocator places the risk block first and the reserve after it, by construction.

> **Strict-priority scheduling loses 1012 of 3459 coverage-reserve
> slots to the horizon -- 0.293. In
> 136 of 273 cells some of the reserve is lost, and in
> 91 it is lost entirely.**

This is a fact about Component 13's allocation meeting Component 14's calendar,
and neither layer is wrong on its own terms. ADR 0037 priced the reserve in
forgone citations and granted it a slot count; nothing in that decision said the
slots had to be at the *end* of the queue, and nothing in Component 13 had a
calendar to notice that it mattered.

**Component 14 reports this and does not correct it.** Promoting reserve rows in
the schedule would be re-ranking -- forbidden in those words -- and would put one
coverage decision in two layers. The advisory
`the_coverage_reserve_survived_scheduling` fires on exactly the cells counted
here, at advisory severity, because the cheapest way to turn such a build green
would be to make the scheduler prefer reserve rows.

| policy | cells | reserve recommended | reserve scheduled | lost | share lost |
|---|---|---|---|---|---|
| coverage_floor_double_share | 13 | 29 | 14 | 15 | 0.5172 |
| coverage_floor_population_share | 1 | 1 | 1 | 0 | 0.0000 |
| coverage_forced_double_share | 90 | 1994 | 1492 | 502 | 0.2518 |
| coverage_forced_half_share | 79 | 463 | 285 | 178 | 0.3844 |
| coverage_forced_population_share | 90 | 972 | 655 | 317 | 0.3261 |

