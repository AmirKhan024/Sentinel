# Data contract — Component 16 deferral / human-review gate

**Producer:** `sentinel review` · **Layer:** `data/processed/review/` (eleventh processed layer)
**Manifest:** `manifest_human_review_queue_<stamp>.json`

Three tables. Column order is part of the contract; changing it is a contract change.

---

## ⚠ What this artifact is not

**It is not a confidence score, and it carries no threshold.** Sentinel has never built a
predictive interval, a conformal set or an ensemble spread (ADR 0040), and this component does
not fabricate one. Both triggers below are boolean facts an upstream component already wrote.
There is no `--threshold`, `--probability-threshold` or `--confidence-threshold` flag, and the
test suite asserts each absence.

**It is not an abstention.** Every flagged row already carries a rank and a recommendation from
Component 13. Flagging a case for review annotates that recommendation; it never withholds it,
blanks it, or replaces it with a "no decision" placeholder while review is pending.

**It is not Component 14's `ScheduleStatus.DEFERRED`.** That status means a scheduled inspection
was moved to a later operating day. A Component 16 review case means a human should look at a row
before it continues to be treated as automatically sufficient. The two ideas are unrelated, and no
Component 16 vocabulary value contains the word "defer" — checked at import time.

**It does not create an override or an adjustment.** A `refer_to_override` or
`refer_to_adjustment` resolution records a pointer to a human's stated intent. The override or
adjustment itself is a separate submission through Component 13's or Component 14's own contract.

**It does not re-rank, re-date or re-score anything.** Nothing here reads `score`, `base_score`
or `final_policy_rank` to decide queue membership, and `queue_is_deterministically_rebuildable`
proves it by rebuilding the queue from the two triggers and comparing byte for byte.

**It is not a verdict.** A flagged case is not necessarily wrong, and an unflagged case is not
necessarily fine. The trigger set is two deterministic facts, not a review checklist.

---

## Grain, keys and sorting

| table | grain | sort key (total order) |
| --- | --- | --- |
| `human_review_queue` | (policy, model, fold, capacity, flagged establishment) | `policy_id, model_name, fold_set, fold_id, k_name, target_inspection_id` |
| `review_resolution_log` | one row per resolution offered | `review_id` |
| `review_advisories` | one row per advisory finding | `code, scope` |

`human_review_queue` is **rebuilt fresh each run** from current Component 13/14 state, not
accumulated — a case whose trigger condition later resolves itself (an execution record arrives)
drops off the next run's queue. `review_resolution_log` is **append-only**: the permanent record
of what a human decided, whether or not the case that prompted it is still on the live queue.

---

## The two deterministic triggers

| trigger | condition | source |
| --- | --- | --- |
| `policy_warning_present` | `is_selected = true` and `warnings != "none"` | Component 13's `inspection_recommendations.warnings` |
| `no_execution_record_on_scheduled_row` | occupying schedule row (`scheduled` or `deferred`, at the cell's latest `replan_index`) with no matching row in `execution_log` | anti-join on `(schedule_config_id, policy_id, fold_id, k_name, target_inspection_id)` — Component 14's own execution contract key |

A row satisfying both triggers carries both reasons, sorted and pipe-joined
(`no_execution_record_on_scheduled_row|policy_warning_present`) — the same convention Component
13 uses for `warnings`. Neither trigger reads a numeric score.

`--schedule` and `--execution` are optional CLI inputs. Without a schedule, only the
`policy_warning_present` trigger runs. Without an execution log, every occupying schedule row is
treated as a gap (there is nothing to anti-join against).

---

## `human_review_queue` columns

| column | meaning |
| --- | --- |
| `policy_id`, `model_name`, `fold_set`, `fold_id`, `k_name`, `target_inspection_id` | Component 13's cell and row keys, verbatim |
| `establishment_id`, `final_policy_rank`, `decision_mechanism`, `decision_reason`, `warnings` | Component 13's own columns, carried unedited |
| `trigger_reasons` | sorted, pipe-joined `ReviewTriggerReason` set; never blank on a real row |
| `schedule_config_id`, `planning_run_id`, `replan_index`, `scheduled_date` | populated only for a case reached through the execution-gap trigger; blank otherwise |
| `review_status` | `flagged` or `resolved`, joined from `review_resolution_log` |
| `review_id`, `resolution_action` | the resolution that resolved this case, if any; blank otherwise |
| `review_definition_version` | contract version, currently `v1` |

## `review_resolution_log` columns

| column | meaning |
| --- | --- |
| `review_id` | human-supplied natural id, unique |
| `policy_id`, `fold_id`, `k_name`, `target_inspection_id` | the case being resolved |
| `resolution_action` | `acknowledge`, `refer_to_override`, `refer_to_adjustment`, `escalate` |
| `reason_code`, `actor`, `decided_at` | mandatory attribution; `decided_at` is the reviewer's own timestamp, never the run's clock |
| `referenced_override_id`, `referenced_adjustment_id` | exactly one populated when the action requires it, both blank otherwise |
| `escalation_note` | free text, only meaningful for `escalate` |
| `original_status`, `final_status`, `outcome` | `applied`, `no_op_already_resolved`, or `case_not_in_queue` |
| `review_definition_version` | contract version |

---

## The human resolution contract

A JSON list of objects, mirroring Component 13's override contract and Component 14's adjustment
contract exactly:

* **Every field required**; a blank or missing one refuses the **whole file**.
* Applied in **`review_id` order, never file order**, so re-serialising the JSON cannot change
  which resolution "wins" a case two resolutions both address.
* `actor`, `reason_code` and a human timestamp are mandatory.
* Duplicate `review_id`s refuse the file.
* The pointer field required by `resolution_action` (`referenced_override_id` for
  `refer_to_override`, `referenced_adjustment_id` for `refer_to_adjustment`) must be present and
  non-blank; the other pointer field must be blank.
* Pointer validity — whether the referenced override or adjustment has actually been submitted —
  is checked only as an **advisory**. A human may record intent before submitting the override or
  adjustment itself through its own contract.

---

## Determinism

> identical Component 13 recommendations + identical Component 14 schedule/execution log +
> identical resolutions file → byte-identical tables.

Nothing broader. Resolutions are external human input; the manifest pins the resolutions file by
checksum rather than claiming a human decision is reproducible computation.

---

## API surface

`GET /v1/review/queue`, `GET /v1/review/queue/{target_inspection_id}`,
`GET /v1/review/resolutions` (all require `policy_id`, `fold_set`, `fold_id`, `k_name` — ADR
0050), and `POST /v1/review/resolutions` (stage-only; validated through the same
`parse_resolutions` parser the batch CLI uses, then appended to
`data/staging/review/resolutions_pending.jsonl`; never applied directly — ADR 0049). See
`docs/data_contracts/sentinel_api.md`.

---

## Validation summary

Error-severity: every case carries a trigger; warning-triggered rows were selected and warned
(re-derived from source); the queue rebuilds byte-identical from the two triggers; no duplicate
`review_id`; pointer fields are mutually exclusive per action; `review_status` reflects exactly
one applied resolution; resolution verbs never collide with `OverrideAction`/`AdjustmentAction`;
every input file is byte-identical after the run.

Advisory (never fails a build): a pointer that does not yet resolve to a committed override or
adjustment; the count of cases flagged this run, by trigger.

See `docs/decisions/0051-the-human-review-gate-and-why-it-carries-no-threshold.md`.
