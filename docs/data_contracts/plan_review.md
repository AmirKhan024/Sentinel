# Data contract — Component 21 supervisor plan review

**Producer:** `sentinel review-plan` (review + decisions) and `sentinel approve-plan` (approval)
**Layer:** `data/processed/plan_review/`
**Manifest:** `manifest_supervisor_plan_review_<planning_date>_<stamp>.json`,
`manifest_approved_operational_plan_<planning_date>_<stamp>.json`
**Definition version:** `v1`

Three tables: `supervisor_plan_review` (the joined, supervisor-facing view), `plan_decision_log`
(the audit trail — written only when at least one decision applied), and
`approved_operational_plan` (written only by `sentinel approve-plan`, once — see "Approval" below).

---

## ⚠ What this artifact is not

**It is not Component 16.** Component 16's human-review gate reads Component 13 (historical
policy recommendations) and Component 14 (scheduling) — a different, backtest-scoped pipeline
keyed on `policy_id/fold_id/k_name`. Component 21 reads only Component 20's live,
`planning_date`-scoped plan. The two are deliberately separate components with disjoint
vocabularies (checked at import time), so a backtest cell and a live plan can never be confused.

**It is not a trigger-based exception queue.** Every establishment in Component 20's plan is
automatically in scope for supervisor review — a supervisor reviews the whole proposed workload,
not a second flagged subset. There is no `trigger.py`.

**It never edits a risk, policy, or geographic field.** `calibrated_score`, `base_score`, `rank`,
`policy_rank`, `selection_reason`, `selection_mechanism`, `geographic_group_id`, and
`work_block_id` are copied verbatim from Component 20 and checked byte-identical after every run.

**A supervisor decision never overwrites Sentinel's own recommendation.** Both are visible on the
same row, side by side, always. `check_original_recommendation_never_overwritten` proves it.

**It never creates a Component 13 override or a Component 14 adjustment.** A
`move_to_later_workday` or `do_not_proceed_as_planned` decision records the supervisor's stated
intent as an additional, audited fact. Turning that intent into an actual capacity or schedule
change remains a separate submission through Component 13's or Component 14's own contract.

**`draft` / `under_supervisor_review` / `adjusted` are derived, never stored** — computed at read
time from how many establishments in the plan have a recorded decision. `approved` is the one
exception: it is not a derived function of decision coverage (a fully-decided plan is `adjusted`,
not `approved`) — it reflects whether a committed `approved_operational_plan` artifact exists for
this `planning_date`, checked independently each time (see "Approval" below). There is still no
`execution`/`outcome_recorded` state, because nothing downstream of this component currently
consumes one.

**`adjust_operational_priority` never changes a risk or policy field.** It sets
`operational_priority`, a display-only field-work ordering column, computed as
`coalesce(supervisor_revised_operational_priority, policy_rank)`. `rank` and `policy_rank` are
unaffected and remain byte-identical to Component 20's output, checked by the same
`IMMUTABLE_FIELDS` invariant as every other field above. "risk_rank = 7,
supervisor_operational_priority = 2" is a valid, expected state; there is no way to make Sentinel
itself report `policy_rank = 2` through this action.

**An approval does not run the plan a second time.** `build_approved_plan` never calls
Component 17–20 again, never rescoring or reselecting anything — it validates the existing
`supervisor_plan_review` frame against a 5-point readiness checklist, then writes it, unedited
except for the addition of the approval's own identity fields (`approval_id`, `approved_by`,
`approved_at`), as `approved_operational_plan`. If the checklist fails, nothing is written — an
approval is all-or-nothing, exactly like a decisions-file submission.

---

## Grain, keys and sorting

`supervisor_plan_review`: one row per establishment in Component 20's plan (same grain,
additive columns only). Sorted the same way Component 20 sorts.

`plan_decision_log`: one row per submitted decision (applied or rejected), sorted by
`decision_id`.

`supervisor_plan_review` is **rebuilt fresh each run** from Component 20's current plan and
whatever decisions file was given — it is not itself an accumulating log. `plan_decision_log` for
one run reflects exactly the decisions file given to that run; an operator maintains their own
accumulated decisions file across runs, the same convention as Component 16's `--resolutions`.

---

## The decision vocabulary

| action | meaning | required extra field |
| --- | --- | --- |
| `keep_selected` | proceed with Sentinel's recommendation and Component 20's placement exactly as proposed | — |
| `move_to_later_workday` | not proceeding on the proposed date/block; states an intended later date | `revised_planned_date` |
| `adjust_operational_priority` | changes the display-only field-work order for this establishment; never Sentinel's own `rank`/`policy_rank` | `revised_operational_priority` |
| `do_not_proceed_as_planned` | not proceeding with this establishment as planned at all | — |

Disjoint by construction and import-time guard from `OverrideAction` (Component 13),
`AdjustmentAction` (Component 14), and `ReviewResolutionAction` (Component 16). No value contains
the substring `"defer"`, reserved by Component 14's `ScheduleStatus.DEFERRED`.

A `do_not_proceed_as_planned` decision does **not** remove the establishment from Component 19's
selected set or Component 20's plan — both remain visible and unedited; only the decision log
records the supervisor's intent.

---

## `supervisor_plan_review` columns (additive over Component 20's plan)

| column | meaning |
| --- | --- |
| every Component 20 column | copied verbatim, immutable |
| `plan_review_definition_version` | contract version |
| `supervisor_decision_id` | the decision that decided this row, if any; null otherwise |
| `supervisor_decision_action`, `_reason_code`, `_actor`, `_decided_at` | the decision's own fields, joined in |
| `supervisor_revised_planned_date`, `supervisor_revised_work_block_id`, `supervisor_revised_operational_priority` | populated only when the action supplies them |
| `operational_priority` | `coalesce(supervisor_revised_operational_priority, policy_rank)` — display-only field-work order, always present, never a substitute for `policy_rank` |

## `plan_decision_log` columns

| column | meaning |
| --- | --- |
| `decision_id` | human-supplied natural id, unique |
| `planning_date`, `target_inspection_id` | the establishment being decided about |
| `decision_action`, `reason_code`, `actor`, `decided_at` | mandatory attribution; `decided_at` is the supervisor's own timestamp |
| `revised_planned_date`, `revised_work_block_id`, `revised_operational_priority` | present only for actions that use them |
| `outcome` | `applied`, `no_op_already_decided`, or `establishment_not_in_plan` |
| `plan_review_definition_version` | contract version |

## `approved_operational_plan` columns

Every `supervisor_plan_review` column, unedited, plus three identity fields added at approval
time: `approval_id`, `approved_by`, `approved_at`. Sorted by `operational_priority` (nulls last),
then `target_inspection_id` — the order a field team would actually work the plan in.

---

## The human decision contract

A JSON list of objects, mirroring Component 16's resolution contract:

* Every required field (`decision_id, planning_date, target_inspection_id, decision_action,
  reason_code, actor, decided_at`) must be present and non-blank; a blank or missing one refuses
  the **whole file**.
* `move_to_later_workday` additionally requires a non-blank `revised_planned_date`.
* `adjust_operational_priority` additionally requires `revised_operational_priority`.
* Applied in **`decision_id` order, never file order** — re-serialising the file cannot change
  which decision "wins" an establishment two decisions both address.
* Duplicate `decision_id`s refuse the file.
* A decision addressed to an establishment not in the plan is not an error — it is logged with
  outcome `establishment_not_in_plan`, and the run still succeeds (surfaced as a manifest
  warning).

---

## Determinism

> identical Component 20 plan + identical decisions file → byte-identical `supervisor_plan_review`
> and `plan_decision_log`.

Nothing broader. Decisions are external human input; the manifest pins the decisions file by
checksum rather than claiming a human decision is reproducible computation.

---

## API surface

`GET /v1/plan-review/summary`, `GET /v1/plan-review/rows`,
`GET /v1/plan-review/rows/{target_inspection_id}`, `GET /v1/plan-review/work-blocks`,
`GET /v1/plan-review/decisions`, `GET /v1/plan-review/approval` (all accept an optional
`planning_date` query parameter; without it, the most recent plan review is used; `/approval`
404s until a plan has actually been approved). `POST /v1/plan-review/decisions` and
`POST /v1/plan-review/approve` are both stage-only (ADR 0049): validated through the same
`parse_decisions`/`parse_approval` parsers the batch CLI uses, then appended to
`data/staging/plan_review/decisions_pending.jsonl` / `approvals_pending.jsonl`; neither is ever
applied by the API itself — an operator commits staged approvals through `sentinel approve-plan`,
which re-runs the full readiness checklist before writing anything.

---

## Validation summary

Error-severity: the establishment set is unchanged from Component 20; `IMMUTABLE_FIELDS` (risk,
policy, and geographic columns) byte-identical; every decided row still carries Sentinel's own
`policy_rank`/`selection_reason`; no establishment decided twice.

Advisory (never fails a build): a decision addressed to an establishment outside this plan.

---

## Approval

`sentinel approve-plan` (or a committed `POST /v1/plan-review/approve` staged request) runs a
5-point readiness checklist against the `supervisor_plan_review` frame before writing anything,
implemented independently of the build-time checks above rather than trusting them:

1. `no_duplicate_establishments`
2. `every_row_carries_the_machine_recommendation` — `policy_rank`/`selection_reason` present on
   every row
3. `geographic_provenance_present` — `work_block_id`/`location_status` present on every row
4. `every_recorded_decision_has_a_reason`
5. `undecided_rows_default_to_the_machine_recommendation` — advisory only, never blocks

Any error-severity failure refuses the **entire** approval; nothing is written. A plan does not
need every row decided to be approved — an undecided row proceeds exactly as Sentinel proposed,
and the readiness check states that explicitly rather than silently.

An approval requires only `approval_id`, `planning_date`, `approved_by`, `approved_at` (optional
`note`) from the supervisor — it is not a second place to restate individual reasons, which
already live in the decision log. `source_plan_review_sha256` (which plan-review file was
approved) is computed independently from the real file at commit time, never supplied by the
request.

The written `approved_operational_plan` artifact is never rewritten in place — true here the same
way it is true for every other Sentinel artifact, by never opening one to edit it. A supervisor
who amends the plan after approval produces a new `supervisor_plan_review` snapshot and, if they
choose, a new approval event; the original approved artifact is untouched and remains a permanent
record of exactly what was handed to Component 22 at that moment, named by its own source
checksum.

---

## Lifecycle

```
draft  →  under_supervisor_review  →  adjusted
                                          |
                                          v
                                      approved
```

`draft`/`under_supervisor_review`/`adjusted` are derived from decision coverage, not stored.
`approved` is derived from whether a committed `approved_operational_plan` exists for this
`planning_date` — reached from any of the other three states (a plan need not be `adjusted`,
i.e. fully decided, to be approved). There is still no `execution`/`outcome_recorded` state in
this component — that would require a downstream consumer this project does not yet have; adding
one prematurely would be exactly the fabricated capability this project avoids elsewhere.
