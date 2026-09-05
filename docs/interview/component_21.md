# Component 21 — Supervisor plan review, adjustment, and approval

Plain language. No prior machine-learning knowledge assumed.

---

## 1. What problem does Component 21 solve?

Component 20 produces a geographically organized plan: which establishments Sentinel selected,
grouped into work blocks a field team could visit together. Nobody has looked at it yet. Component
21 is the screen a supervisor actually uses: see the proposed workload, understand why Sentinel
organized it this way, keep or change individual establishments with a reason, adjust the order
field staff should visit them in without touching Sentinel's own risk ranking, and finally approve
the whole plan so it becomes the one authoritative record a downstream execution step (not yet
built) could consume.

The one rule everything else follows from: a supervisor's decision is recorded *beside* Sentinel's
recommendation, never in place of it. Both are always visible. Nothing here retrains a model,
recomputes a score, or reselects who is in the plan.

---

## 2. Where does it sit in the architecture?

```text
candidates (17) -> scoring (18) -> selection (19) -> geography (20) -> PLAN REVIEW (21)
                                                                             |
                                                                             v
                                                keep / defer / adjust priority / remove
                                                                             |
                                                                             v
                                                                         approve
                                                                             |
                                                                             v
                                                          approved_operational_plan
                                                          (the Component 22 contract)
```

It reads only Component 20's output — never Components 18 or 19 directly, so a plan decision can
never bypass geographic organization or capacity/policy selection. It writes its own artifacts:
`supervisor_plan_review` (rebuilt fresh each run), `plan_decision_log` (the audit trail), and,
only when a supervisor approves, `approved_operational_plan`.

---

## 3. Why is this a fifth human-decision layer, not a fourth?

Sentinel already had three: Component 13's `OverrideAction` (force-include/force-exclude a row
in the *historical, backtest* selection), Component 14's `AdjustmentAction`/`ExecutionStatus`
(move or cancel a *scheduled* inspection, report what happened), and Component 16's
`ReviewResolutionAction` (resolve a *flagged historical case*). All three are backtest-scoped —
they operate on a fold, not a live date.

Component 21's `PlanDecisionAction` is the first decision vocabulary that operates on a live
`planning_date` at all. It is checked at import time against all three other vocabularies for verb
collisions, and against the literal substring `"defer"`, reserved by Component 14's
`ScheduleStatus.DEFERRED` — so a plan decision and a schedule status can never be typo'd into
meaning the same thing in code.

---

## 4. What can a supervisor actually do?

Four decision verbs, each requiring `reason_code`, `actor`, `decided_at`:

* **`keep_selected`** — proceed with Sentinel's recommendation and Component 20's placement
  exactly as proposed. No extra field.
* **`move_to_later_workday`** — not proceeding on the proposed date/block; states an intended
  later date (`revised_planned_date`, required). Does not remove the establishment from the plan
  or create a Component 14 schedule adjustment — it records intent only.
* **`adjust_operational_priority`** — changes the display order field staff would work the plan
  in (`revised_operational_priority`, required). This is the one action people most expect to
  quietly become a rank edit, and it deliberately does not: it sets a new column,
  `operational_priority = coalesce(supervisor_revised_operational_priority, policy_rank)`.
  Sentinel's own `rank` and `policy_rank` are unedited and checked byte-identical to Component
  20's output by the same invariant that protects every other risk/policy/geographic field.
  "risk_rank = 7, supervisor_operational_priority = 2" is a valid, expected state; there is no
  code path that can make Sentinel itself report `policy_rank = 2`.
* **`do_not_proceed_as_planned`** — not proceeding with this establishment at all, for this plan.
  Still does not remove it from the underlying selected set; only the decision log records it.

Then, separately: **approve the plan.**

---

## 5. Why is approval a separate act instead of "every row decided = approved"?

Because that would make approval an accidental consequence of data entry rather than a deliberate
supervisor act. A plan where every establishment has a recorded decision reaches
`PlanApprovalStatus.ADJUSTED` — not `APPROVED`. Approval requires its own minimal, explicit
request (`approval_id`, `planning_date`, `approved_by`, `approved_at`, optional `note`), and it
can be submitted from *any* decision-coverage state, including a plan with zero decisions — an
undecided row simply proceeds exactly as Sentinel proposed, and the approval readiness check says
so explicitly (as an advisory note, not a blocker) rather than silently.

---

## 6. What does approval actually check, and what happens if it fails?

A 5-point readiness checklist, run independently of the build-time invariants that already
protect `supervisor_plan_review` — not trusted from an upstream flag, the same posture every
other validator in this project takes:

1. `no_duplicate_establishments`
2. `every_row_carries_the_machine_recommendation` — `policy_rank`/`selection_reason` present
3. `geographic_provenance_present` — `work_block_id`/`location_status` present
4. `every_recorded_decision_has_a_reason`
5. `undecided_rows_default_to_the_machine_recommendation` — advisory only, never blocks

Any error-severity failure refuses the **entire** approval — nothing is written, not a partial
artifact. This was tested directly: submitting an approval for a `planning_date` that doesn't
match the plan review's own `planning_date` is refused with a specific error before the checklist
even runs; corrupting a decision to have no `reason_code` fails check 4 and blocks approval end
to end.

---

## 7. What does "the approved artifact is immutable" actually mean, mechanically?

It means the same thing immutability means everywhere else in this project: the file that gets
written is never opened and edited. `sentinel approve-plan` writes
`approved_operational_plan_<planning_date>_<stamp>.parquet` once, named by a fresh timestamp, and
nothing in the codebase ever rewrites an existing one. If a supervisor changes their mind after
approving — decides to add a decision, or re-approve with a correction — that produces a *new*
`supervisor_plan_review` snapshot and, if they choose, a *new* approval event with a new
`approval_id`. The original approved file is untouched, still on disk, still checksummed against
the exact `supervisor_plan_review` it approved (`source_plan_review_sha256`, computed
independently from the real file at commit time, never supplied by the request). Nothing needed a
database lock or a versioning scheme invented for this — the project's existing
"artifacts are never edited in place" convention already provides it.

---

## 8. What two real bugs did wiring this into the API actually surface?

Both were found only by literally sending an HTTP request to the running app — `create_app()`
plus `app.dependency_overrides[get_settings]`, following the exact pattern the test suite's own
`conftest.py` fixture uses — not by running the unit test suite, which passed the whole time.

1. **A required-field list named a field the request model doesn't have.**
   `PLAN_APPROVAL_REQUIRED_FIELDS` (and the import-time guard enforcing it) required
   `source_plan_review_sha256` — but that field lives only on `ApprovedPlanManifest`, computed
   from the real plan-review file at commit time. `PlanApprovalRequest`, the supervisor's own
   minimal input, never carries it. Every single approval request was refused with "field is
   blank," regardless of payload, until the required-fields tuple was corrected to name only the
   fields the request actually has.
2. **The staging layer had no entry for the new kind at all.** `StagingService._KIND_CONFIG` is a
   small dict mapping each staged-write "kind" to its natural-id field and file path — a fourth
   kind, `plan_decision`, was already registered from earlier work, but the fifth, `plan_approval`,
   was never added. A request that passed schema validation and the (now-fixed) governance check
   still crashed with a bare `KeyError` inside `staging.append()` itself.

Neither bug was a logic error inside any one function — every function involved was individually
correct in isolation. The seam between them was wrong, and only an end-to-end request exercises
the seam. Both are now covered by regression tests
(`tests/api/test_plan_review_approval_api.py`) that POST through the actual router, not just call
the service function directly.

---

## 9. What did a real run against production data show?

A fresh plan review for `planning_date=2026-08-28` (30 establishments), a 3-action demo decisions
file (keep one, defer one with a reason and revised date, adjust one's operational priority to 1
with a reason), then approval: 30 total, 29 active, 1 deferred, 0 not proceeding, 27 undecided,
all 5 readiness checks READY. The adjusted establishment's `rank`/`policy_rank` stayed exactly 3
throughout — unchanged by the priority edit — while its `operational_priority` correctly showed 1.
Two approvals of byte-identical input produced byte-identical output outside the approval identity
fields themselves (`approval_id`, timestamps), which are new by construction on every submission.

---

## 10. Difficult interviewer questions

**"Why not just let `adjust_operational_priority` write directly to `policy_rank`? Wouldn't that
be simpler?"**
Because it would destroy the one fact this whole component exists to preserve: what Sentinel
actually recommended. If a supervisor's reordering silently became the model's own rank, nobody
could ever again answer "what did the model say, before a human touched it?" — which is exactly
the audit question a food-safety oversight body would ask first. A second column costs one line of
`coalesce()`; conflating the two costs the entire audit trail's credibility.

**"What stops a supervisor from approving a plan nobody actually reviewed?"**
Nothing forces a human to look at every row before approving — the readiness checklist explicitly
allows approving with rows undecided, treating "no decision" as "proceed as Sentinel proposed,"
which is itself a legitimate, honestly-labeled outcome (see check 5). What the checklist *does*
guarantee is that whatever is approved is structurally complete: no duplicate establishments,
every row still carries Sentinel's own recommendation, geography is present, and any decision that
was recorded carries a reason. Whether a supervisor actually read the plan is a human process
question this component cannot and does not claim to answer.

**"Couldn't the two integration bugs have been caught by better unit tests?"**
Not by unit tests of the same kind that were already passing — both functions involved
(`parse_approval`, `staging.append`) were individually correct against their own existing tests.
The bug was in configuration data shared across modules (`PLAN_APPROVAL_REQUIRED_FIELDS`, the
`_KIND_CONFIG` dict) that only manifests when the full request path — schema, governance check,
staging append — runs together. This is the specific argument for keeping an end-to-end smoke
test in the workflow even when the unit suite is green, not a criticism of the unit tests that did
exist.

**"Why does approval read `check_readiness` independently instead of trusting the checks that
already ran when the plan review was built?"**
Because a plan review can be rebuilt, amended, or hand-edited between when it was written and when
someone tries to approve it, and because this project's own established posture (every validator
here re-derives its answer rather than trusting an upstream flag) says a stored "this was already
checked" boolean is exactly the kind of state that silently goes stale. Re-running the checklist
at approval time costs nothing and closes that gap.

---

## 11. 60–90 second answer

"Component 21 is the screen a supervisor actually uses on top of Sentinel's machine-generated
geographic plan. They can keep, defer, adjust the field-work order, or decline to proceed with any
establishment — each with a required reason — and none of those actions can ever overwrite
Sentinel's own recommendation; both are always visible on the same row. Adjusting field-work order
is the one action people expect to secretly become a rank edit, and it doesn't — it writes a
separate display-only column, and Sentinel's own risk rank is checked byte-identical after every
run. Approval is a deliberate, separate act — not an automatic consequence of deciding every row —
gated by a five-point readiness check that refuses the whole approval outright if anything's
wrong, and the approved artifact, once written, is never edited in place; an amendment produces a
new snapshot, leaving the original as a permanent, checksummed record of exactly what was handed
downstream. Wiring the API path surfaced two real bugs — a required-field mismatch and a missing
staging-service entry — that no unit test caught because each one was individually correct; only
an actual end-to-end request through the running app exposed the seam between them, which is why
that kind of smoke test stayed part of finishing this component rather than an optional extra."
