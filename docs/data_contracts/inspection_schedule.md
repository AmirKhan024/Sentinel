# Data contract — Component 14 operational scheduling

**Producer:** `sentinel schedule` · **Layer:** `data/processed/scheduling/` (tenth processed
layer, ADR 0041) · **Manifest:** `manifest_inspection_schedule_<stamp>.json`

Thirteen tables. Column order is part of the contract; changing it is a contract change.

---

## ⚠ What this artifact is not

**It is not a routing plan.** The dataset has no inspector, no shift, no duration, no travel
time and no road network, so a route here is not underdetermined — it is unrepresented. A day's
slots are a **workload count**; `slot_index` means "the fourth inspection that day", never "the
fourth stop". Routing is Component 15's, and Component 15 is blocked on the same missing data
(ADR 0043).

**It is not an optimisation.** It is deterministic greedy slot allocation down an approved rank
order. No objective function is defined anywhere in this component and none is solved. Nothing
here may be described as optimal, optimised or efficient.

**It is not a re-ranking.** `final_policy_rank` is the only ordering key. Nothing in this layer
reads a score, a probability, a mechanism or a geography to decide who goes first, and every
Component 13 provenance column is copied verbatim and checked against the source after the run.

**It is not a forecast.** Every operating day in it is a day that has already happened. The
observed calendar is measured from the same window it schedules, so it states what capacity
*existed*, not what a planner could have known on day one.

**It is not a claim that any establishment went uninspected.** Component 5's limitation is
inherited whole: this is a re-ordering study over inspections that already happened. A backlog
is what a stated capacity rule would not have fitted.

**It is not an execution record.** `inspection_schedule` has no `execution_status` column, by
design. What actually happened lives in `execution_log`, and no row in this repository describes
a real Chicago execution event.

**Nothing here may be joined onto a feature table.** A schedule is downstream of every model,
every policy and every human decision in this project; joined back onto training rows it would
close the feedback loop one layer further out than Component 13 already refused to close it.

**Capacity is never raised.** Every horizon descends from Component 5's measured median daily
rate by way of Component 13's `k`. There is no `--capacity`, `--slots-per-day`,
`--horizon-days`, `--extend-horizon` or `--threshold` flag, and the test suite asserts each
absence.

---

## Grain, keys and sorting

| table | grain | sort key (total order) |
| --- | --- | --- |
| `inspection_schedule` | (config, policy, model, fold, capacity, planning run, scored inspection) | `schedule_config_id, policy_id, model_name, fold_set, fold_id, k_name, replan_index, target_inspection_id` |
| `schedule_backlog` | same | same |
| `schedule_slots` | (config, fold, capacity, operating day) — **policy-independent** | `schedule_config_id, fold_set, fold_id, k_name, slot_date` |
| `schedule_summary` | (config, policy, model, fold, capacity) | the six cell keys |
| `capacity_utilization` | (cell, operating day) | the six cell keys, then `slot_date` |
| `priority_preservation` | (config, policy, model, fold, capacity) | the six cell keys |
| `schedule_configurations` | one row per configuration | `schedule_config_id` |
| `schedule_adjustment_log` | one row per adjustment offered | `adjustment_id` |
| `execution_contract` | one row per (contract, field) | `contract_name, field_name` |
| `execution_log` | one row per execution event offered | `execution_id` |
| `execution_summary` | (config, policy, model, fold, capacity) | the six cell keys |
| `replanning_runs` | (config, policy, fold, capacity, planning run) | those keys, then `replan_index` |
| `schedule_advisories` | one row per advisory finding | `code, scope` |

Every sort key is a **total order**: two runs over identical inputs produce byte-identical
files. `replan_index` is part of the schedule's grain because a re-plan **appends** a plan
beside the old one rather than editing it, so one scored inspection legitimately holds one row
per planning run.

---

## The horizon and the two capacity modes

```text
horizon_days = ceil(k / test_median_daily_capacity)
```

Not a new constant: the inverse of the rule `evaluation.simulate.capacity_k_values` used to
produce `k`, read backwards. It reproduces both day-denominated names exactly — `k_1_day` spans
one day, `k_1_week` spans five. Verified total across all 90 (fold, capacity) cells.

The horizon is a **prefix of the fold's own observed operating days**. An operating day is a
date Component 13's universe carries; no working-week rule, no holiday list, no generated date.

| mode | slots per day | status |
| --- | --- | --- |
| `observed_calendar` | inspections Chicago actually performed that date | **measured** · the default |
| `flat_median` | `test_median_daily_capacity` every day | **scenario** · labelled everywhere |

The horizon is identical in both modes; only slot counts differ, so the two stay comparable.

> ⚠ **`flat_median` is tautological at `k_1_day` and `k_1_week`.** The horizon is `k / median`
> days of `median` slots, so it holds exactly `k`: backlog zero and utilisation exactly 1.000,
> by construction. Every scenario row carries `is_scenario = true`, an advisory fires whenever
> any is written, and no summary pools it with the observed calendar.

---

## 1. `inspection_schedule` — the component's answer

The approved queue, laid against the calendar. Two provenance blocks that never mix.

**Component 13, verbatim:** `recommendation_date` (the as-of scoring date — **not** a schedule
date), `base_score`, `score`, `model_rank`, `final_policy_rank`, `decision_mechanism`,
`decision_reason`, `coverage_eligible`, `warnings`, `recommendation_override_id`,
`policy_definition_version`.

**Component 14:** `planning_run_id`, `replan_index`, `schedule_status`, `schedule_reason`,
`inversion_reason`, `scheduled_date`, `day_index`, `slot_index`, `schedule_rank`,
`wait_operating_days`, `original_scheduled_date`, `original_schedule_rank`, `adjustment_id`,
`is_scenario`.

`schedule_rank` sits **beside** `final_policy_rank` rather than replacing it. That pair is the
whole design: where they agree the policy decided, where they differ the scheduler did, and
`inversion_reason` names the mechanism. ADR 0037's pattern, one layer out.

`original_scheduled_date` and `original_schedule_rank` are **write-once**, set at first
placement and copied forward untouched at every later index.

---

## 2. Controlled vocabularies

**`schedule_status`** — what the *plan* says.

| value | meaning |
| --- | --- |
| `scheduled` | holds a slot on a horizon operating day |
| `backlog` | approved, but the horizon held no slot. Still recommended |
| `deferred` | moved to a later day by an adjustment or re-plan. **Still holds a slot** |
| `cancelled` | struck from the plan. The slot is not backfilled |

`recommended` is not here — it is Component 13's `is_selected`. `completed` is not here — it is
an execution fact, and a plan column execution can write into is the retroactive edit the
temporal boundary exists to prevent.

**`schedule_reason`** — `placed_in_priority_order` · `capacity_exhausted_in_horizon` ·
`deferred_by_adjustment` · `advanced_by_adjustment` · `displaced_by_adjustment` ·
`rescheduled_by_replan` · `cancelled_by_adjustment` · `cancelled_in_field`. Which reasons each
status may carry is frozen in `STATUS_REASONS` and checked at error severity.

There is deliberately **no `constraint_adjusted`**: no operational constraint exists in this
data, so no run could emit it, and a code no path reaches is indistinguishable from a broken one.

**`inversion_reason`** — `none` · `deferred_by_adjustment` · `advanced_by_adjustment` ·
`displaced_by_adjustment` · `rescheduled_by_replan`. `none` is a token, never a null: an empty
cell is ambiguous between "no inversion" and "inversions were not computed".

**`adjustment_action`** — `defer_to_date` · `advance_to_date` · `cancel`. Disjoint from
Component 13's `OverrideAction` by import-time guard.

**`execution_status`** — `completed` · `not_performed` · `cancelled_in_field`, plus the derived
`no_execution_record` that appears only in summary counts and can never be supplied.

---

## 3. Accounting

```text
n_scheduled + n_backlog + n_cancelled == n_recommended
```

`n_deferred` is **not** a fourth term. A deferred row still holds a slot, so it is already
inside `n_scheduled`; adding it again would double-count exactly the rows somebody moved. It is
a breakdown of the scheduled block, and `counts_add_up` asserts that relationship.

---

## 4. The two external contracts

Both mirror Component 13's override contract (`policy_decisions.md` §8) exactly: a JSON list,
**every field required**, a blank or missing field refuses the **whole file**, applied in **id
order** rather than file order, pinned in the manifest by sha256 rather than by a claim that
human typing is reproducible. Every offered change is logged whether or not it changed anything.

```json
// scheduling adjustment
[{"adjustment_id": "SA-2026-0031", "schedule_config_id": "strict_priority__observed_calendar",
  "policy_id": "pure_risk", "fold_id": "quarterly-2026Q2", "k_name": "k_1_week",
  "target_inspection_id": "2467913", "action": "defer_to_date", "target_date": "2026-04-08",
  "reason_code": "establishment_closed", "actor": "district.supervisor.4",
  "decided_at": "2026-08-26T09:00:00Z"}]

// execution event
[{"execution_id": "EX-2026-0104", "schedule_config_id": "strict_priority__observed_calendar",
  "policy_id": "pure_risk", "fold_id": "quarterly-2026Q2", "k_name": "k_1_week",
  "target_inspection_id": "2467913", "scheduled_date": "2026-04-02",
  "execution_status": "not_performed", "reason_code": "closed_on_arrival",
  "actor": "field.inspector.log", "observed_at": "2026-04-02T16:40:00Z"}]
```

`target_date` must be an operating day **inside the cell's horizon** — a date outside it would
extend the horizon, which is a capacity increase by another name. A move onto a full day
displaces the lowest-ranked **risk** row on that day, **never a coverage-reserve row**; if none
is displaceable the run refuses. The displaced row takes the slot the mover vacated, or goes to
backlog. `cancel` frees a slot and it is **not** backfilled.

The deterministic plan at `replan_index = 0` is written **unchanged**, and
`the_deterministic_plan_is_intact` rebuilds it with both files withheld to prove it.

Both contracts are also emitted as data in `execution_contract`, so a reader who never opens
this document can still construct a valid file.

---

## 5. Re-planning

A re-plan **appends** a planning run; it never mutates one. Preserved at every index: completed
rows keep their slot; every row on a day before the re-plan point is frozen whatever its status;
`original_*` is copied forward untouched. Placement is ordered by `final_policy_rank` at every
depth.

One narrow exemption: a row the field reported as `not_performed` moves even though its day has
passed. Freezing it would strand the inspection the report was filed to rescue.

A re-plan **backfills** freed capacity from the backlog, where Component 13's `force_exclude`
deliberately does not. The difference is what the freed capacity means: an excluded row is a
decision that the slot should not be used; a day that did not happen is capacity that still
exists. A cancellation is a removal and is never backfilled.

---

## 6. Determinism

**Byte-identical across runs given identical inputs**, verified on two independent production
runs, and asserted under shuffled recommendation rows, shuffled adjustment rows and shuffled
execution rows.

Row order is an explicit input contract, never an assumption: every ordering descends from
`final_policy_rank` with a secondary key on `target_inspection_id`, and never from Parquet row
order. `planning_run_id` is a **content hash** of the cell and configuration — never a clock,
never random — because a timestamped id would make two runs over identical inputs differ.

**The claim is scoped:** identical recommendation artifact + identical scheduling configuration
+ identical external files → byte-identical tables. Adjustments and execution events are
external human and operational inputs; the manifest pins them by checksum rather than claiming
that what inspectors did last Tuesday is reproducible computation.

---

## 7. Values: descriptive, normative, operational or evidence

| table | kind |
| --- | --- |
| `inspection_schedule`, `schedule_backlog`, `schedule_adjustment_log`, `replanning_runs` | **operational** — what the system plans, and what a human changed |
| `execution_log` | **execution evidence** — what a person reports happened; external, not reproducible |
| `execution_contract` | **external input contract** — the file format, as data |
| `schedule_configurations` | **normative** — choices this component made, with their rules attached |
| `schedule_slots`, `schedule_summary`, `capacity_utilization`, `priority_preservation`, `execution_summary`, `schedule_advisories` | **descriptive** — measurements of what the schedule did |

---

## Validation

**35 checks — 28 errors and 7 advisories. Error severity fails the run; advisory severity never does, and there is no flag to
change that** (ADR 0034, inherited through ADR 0041).

Errors cover: sort order and duplicate keys · every scheduled row originating in the approved
queue · every approved row accounted for · no double-booked slot · no inspection in two slots ·
no day over capacity · slot counts matching their declared mode · horizon ordered, contiguous
and real · unique contiguous schedule ranks · slot order following policy rank · no inversion
without a reason code · valid status/reason pairing · backlog equal to the unscheduled remainder
· counts adding up · never scheduling more than `k` · Component 13 provenance preserved · no
outcome column anywhere · completed rows never rescheduled · execution never altering a
recommendation · nothing before a re-plan point moving · unique chained planning runs ·
adjustments distinguishable from overrides · original assignments preserved · the reserve never
displaced · external changes fully attributed · the deterministic plan intact · the frozen grid
· inputs unmodified.

Advisories cover: idle capacity · backlog · **coverage-reserve slots lost to the horizon** ·
scenario rows written · whether an execution record was supplied · thin opening days ·
establishments recurring within a horizon.

The report prints this line first, on every run:

> **A green run means the plan was built correctly. It does not mean the city has enough
> capacity, it does not mean the schedule is the right one, and it does not mean the coverage
> reserve survived.**
