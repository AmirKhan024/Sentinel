# Data contract — Component 13 decision policy

**Producer:** `sentinel decide` · **Layer:** `data/processed/policy/` (ninth processed layer,
ADR 0036) · **Manifest:** `manifest_inspection_recommendations_<stamp>.json`

Eleven tables. Column order is part of the contract; changing it is a contract change.

---

## ⚠ What this artifact is not

**It is not a prediction.** Nothing in this layer scores anything. `score` and `base_score` are
copied verbatim from Component 9 and are never recomputed, adjusted or reweighted. Component 5's
`evaluate --predictions` would refuse every table here.

**It is not a statement that the recommended queue is the correct queue.** It is the queue one
stated policy produces from one selected model under one capacity assumption.

**It is not a claim that the selected model is the best model.** Four candidates were
statistically indistinguishable on the headline metric and the rule broke the tie on a secondary
axis. A different, equally defensible tie rule selects a different model, and both outcomes are
recorded in `policy_model_selection` (ADR 0039).

**It is not a fairness intervention and establishes no fairness claim.** No score is adjusted by
geography, no group-specific threshold or calibrator exists, and no quota is applied. ADR 0035's
boundary is inherited unchanged: no protected characteristic is observed anywhere in this
project.

**It is not an autonomous enforcement decision.** Every row is a recommendation to a person, and
the override contract exists because operations will legitimately depart from it.

**Nothing here may be joined onto a feature table.** A recommendation is downstream of every
model in this project; joined back onto training rows it would make the system's own past
decisions an input to its future ones.

**Every capacity cutoff is a rank position**, derived by `evaluation.simulate.capacity_k_values`
from each window's own measured median daily inspection rate. There is no probability threshold
anywhere in this component and no flag to add one.

---

## Downstream consumer

**Component 14 reads this artifact read-only** and never modifies it — `inputs_were_not_modified`
compares every input's sha256 before and after its run. It consumes `is_selected`,
`final_policy_rank`, `inspection_date` (from which it derives each fold's observed operating
calendar) and the full provenance block, which it carries onto every scheduled row verbatim.
`policy_override_log` is read as **provenance evidence only**: an applied override stamps an
`recommendation_override_id` onto the schedule row and never changes a rank. Nothing in this
contract changes. See `docs/data_contracts/inspection_schedule.md` and ADR 0041.

---

## Grain, keys and sorting

| table | grain | sort key (total order) |
| --- | --- | --- |
| `inspection_recommendations` | (policy, model, fold, capacity, scored establishment) | `policy_id, model_name, fold_set, fold_id, k_name, target_inspection_id` |
| `policy_selection_allocation` | (policy, model, fold, capacity) | `policy_id, model_name, fold_set, fold_id, k_name` |
| `policy_comparison` | (policy, model, fold, capacity) | `policy_id, model_name, fold_set, fold_id, k_name` |
| `policy_frontier` | (policy, model, fold set, capacity) pooled | `model_name, fold_set, k_name, policy_id` |
| `policy_group_audit` | (policy, model, geography, group, fold, capacity) | `policy_id, model_name, group_definition, fold_set, fold_id, k_name, group_value` |
| `policy_decision_reasons` | (policy, model, fold, capacity, mechanism, reason, warning set) | as listed, in that order |
| `policy_coverage_eligibility` | (grain, fold) — model-independent | `grain, fold_set, fold_id` |
| `policy_configurations` | one row per candidate policy | `policy_id` |
| `policy_model_selection` | one row per registered model | `model_name` |
| `policy_advisories` | one row per advisory finding | `code, scope` |
| `policy_override_log` | one row per override offered | `override_id` |

Every sort key is a **total order**: two runs over identical inputs produce byte-identical
files. A partial key would leave ties resolved by append order, which is not a contract —
Component 12 shipped exactly that defect once.

---

## 1. `inspection_recommendations` — the component's answer

One row per (policy, capacity, fold, scored establishment). **The whole prediction universe, not
only the queue.** A queue-only artifact could say who was chosen; only a universe-grained one can
answer *why was this establishment not inspected?*

| column | type | meaning |
| --- | --- | --- |
| `policy_id` | Utf8 | which of the seven candidate policies produced this row |
| `model_name` | Utf8 | the selected production model; one model per artifact |
| `fold_set`, `fold_id` | Utf8 | Component 5's operating period |
| `k_name`, `k` | Utf8, Int64 | the capacity level and its realised slot count |
| `target_inspection_id` | Utf8 | Component 3's key; also the canonical tie-break column |
| `establishment_id` | Utf8 | Component 2's stable identifier |
| `inspection_date` | Date | the as-of scoring date |
| `base_score` | Float64 | Component 9's uncalibrated score, verbatim |
| `score` | Float64 | Component 9's calibrated probability, verbatim — **the ranking key** |
| `model_rank` | Int64 | position in the pure model ordering, 1-based |
| `final_policy_rank` | Int64 | position in this policy's queue, 1..k; **null when not selected** |
| `is_selected` | Boolean | in the queue at this capacity |
| `decision_mechanism` | Utf8 | `risk_priority` \| `coverage_reserve` \| `not_selected` |
| `decision_reason` | Utf8 | see the vocabulary below |
| `coverage_eligible` | Boolean | `prior_canvass_count_code_era == 0` |
| `secondary_no_history` | Boolean | `prior_inspection_count_any_type == 0`; reporting only |
| `warnings` | Utf8 | sorted, `\|`-joined set, or the literal `none` |
| `group_value`, `group_status` | Utf8 | Component 12's as-of community area and support status; **advisory** |
| `policy_definition_version` | Utf8 | bumped when the contract changes |

**`model_rank` beside `final_policy_rank` is the component's central pair.** Where they agree the
model decided; where they differ the policy did, and `decision_mechanism` names which mechanism
moved it.

**`group_value` and `group_status` are read onto the row and never read back.** The queue is
rebuilt on every run with both withheld and the ranks compared exactly
(`validate.warnings_do_not_change_the_queue`, error severity).

Row count on the production run: **1,453,760** = 7 policies × 5 capacities × 41,536 scored rows.

---

## 2. Controlled vocabularies

### `decision_mechanism`

| value | meaning |
| --- | --- |
| `risk_priority` | in the top `k - n_reserve` by calibrated risk |
| `coverage_reserve` | coverage-eligible, outside the risk cutoff, inside the reserve allocation |
| `not_selected` | not in the queue at this capacity |

Every selected row carries exactly one of the first two. The two are disjoint by construction —
the reserve is filled from rows the risk block did not take — and the disjointness is checked
rather than trusted.

### `decision_reason`

| value | meaning |
| --- | --- |
| `selected_by_risk_rank` | the model's ranking put it here |
| `selected_by_coverage_reserve` | the policy's allocation put it here |
| `not_selected_capacity_exhausted` | outranked, and not coverage-eligible |
| `not_selected_reserve_exhausted` | coverage-eligible, outranked, and the reserve did not reach it |

The last two are separated because "you were outranked" and "the reserve ran out" are different
facts, and only the second is a statement about this component's policy. It is what makes *was
the reserve too small?* answerable from the artifact.

**No reason code is generated text.** No language model writes any value in this vocabulary.

### `warnings`

| code | meaning |
| --- | --- |
| `limited_history` | coverage-eligible: no canvass since 2018-07-01 |
| `no_prior_inspection` | no inspection of any type on record |
| `unknown_geography` | Component 12's `__UNKNOWN__` token: no community area recoverable |
| `insufficient_group_audit_support` | the audit could not measure this neighbourhood |
| `none` | nothing to flag |

A sorted set joined with `|`, not a precedence — choosing one to display would choose which fact
a reviewer is allowed to see. **A warning is not an abstention:** every row still receives a
recommendation and a rank. There is no abstention category, because it would require a per-row
confidence estimate this project has not built (ADR 0040).

### `reserve_mechanism`

| value | meaning |
| --- | --- |
| `none` | no reserve; the risk ranking is the queue |
| `floor` | guarantee at least `share × k` eligible selections; inert when risk already delivers |
| `forced` | spend `share × k` slots on eligible rows the risk ranking passed over |

---

## 3. `policy_selection_allocation` — the policy's account of itself

One row per (policy, model, fold, capacity). Separate from the comparison because three of its
columns cannot be reconstructed from anywhere else without reimplementing the allocator.

Key columns: `reserve_target` (offered), `n_eligible_in_risk_top_k` (already satisfied),
`n_reserve` (granted), `n_risk`, `n_selected`, `n_eligible_available`, `reserve_inert`.

**Those three numbers together distinguish** "the floor was satisfied" from "the floor was
ignored" from "there were not enough eligible establishments to satisfy it". A table recording
only the granted count could not tell them apart.

Invariants, all checked at error severity:
`n_risk + n_reserve == n_selected == k`, `n_reserve <= reserve_target`,
`n_reserve <= n_eligible_available`.

`reserve_target` is `floor(share × k)` — **truncated**, so a reserve can never spend more than
the share it declared. The consequence is that a small share at a small cutoff floors to zero
slots; that is reported as an advisory rather than rounded away.

---

## 4. `policy_comparison` — effectiveness and its price, on one row

One row per (policy, model, fold, capacity), for **every admissible model**, not only the
selected one — so a reader can check that the conclusion about what coverage costs survives the
choice of model.

Effectiveness: `positives_selected`, `precision_at_k`, `capture_rate`, `lift_at_k`, `nde`.
Coverage: `eligible_selected`, `eligible_selected_share`, `eligible_positives_selected`,
`eligible_capture_rate`.
Price: `delta_positives`, `delta_precision`, `delta_capture`, `delta_nde`,
`delta_eligible_selected`, `delta_eligible_capture`.

**Every `delta_` column is measured against `pure_risk` at the identical model, fold and
capacity** — cell by cell, never as a difference of pooled means. `delta_positives` is the
headline: the number of Priority citations the policy gave up, in the unit that means something
to a department. **A reserve is described as free only where this is exactly zero.**

`nde` is computed over a schedule that is the policy's queue followed by the model's ranking for
the tail. Only the first `k` entries are a policy statement; the tail convention is a measurement
choice and is stated rather than buried, because a different one would give the same policy a
different NDE.

Precision, capture and lift are computed **over the queue that was built**, not by calling
Component 5's top-k helpers — those re-derive the top `k` from the scores, which is the
definition of `pure_risk`, so handing a coverage policy's queue to them would silently measure
the baseline. `tests/test_policy_evaluate.py` runs both paths on `pure_risk` and requires exact
agreement.

---

## 5. `policy_frontier` — dominated policies, and nothing more

One row per (policy, model, fold set, capacity), pooled. Marks `is_dominated` and names
`dominated_by`.

Domination is over two axes — `positives_selected` and `eligible_selected` — and they are
**deliberately not summed**. A single "policy score" would embed an exchange rate between a
missed Priority citation and an uninspected establishment with no history, and nothing in this
project measures that rate.

A winner is named only when one policy is the unique non-dominated policy at the primary
operating point. On the production run it is not, and the manifest records
`policy_winner: null` with `no_winner_statement: "the data does not determine the correct
policy"`. **That is a published result, not a failure to conclude.**

---

## 6. `policy_group_audit` — descriptive, and Component 12's status carried through

One row per (policy, model, geography, group, fold, capacity). Selection share, selection rate,
capture and their support counts.

**`group_status` is Component 12's, carried unchanged.** Nothing in this component filters on it,
and `validate.unsupported_groups_are_preserved` fails the run if a group Component 12 called
unsupported is relabelled here — because the easiest way to produce a flattering group table
would be to promote the groups nobody could measure.

Nothing here is optimised against. A coverage policy changes who is inspected by construction, so
it changes group shares by construction; the numbers exist so the change is visible, not so it is
minimised.

---

## 7. `policy_model_selection` — the deployment decision, re-derivable

One row per registered calibrated model, **admissible and refused**. The refusal is data so a
reader who opens the Parquet instead of ADR 0039 still finds out why there are four candidates
and not five.

Carries every axis (`nde`, `nde_p05`, `nde_p95`, `ece`, `precision_at_k_1_day`), `tied_on_nde`,
`decided_on_axis`, `is_selected` and **`selected_under_discarded_band`**.

The last column exists because the tie rule decides which model is deployed and the rule was
fixed after its inputs were first read. Recording only the outcome would hide that another
defensible rule gives a different answer.

---

## 8. `policy_override_log` — the human layer

One row per override offered, applied or not. Carries the **original** recommendation
(`original_is_selected`, `original_mechanism`, `original_reason`, `original_policy_rank`) beside
the **final** decision (`final_is_selected`, `displaced_target_inspection_id`) and the
attribution (`actor`, `reason_code`, `decided_at`, `override_id`).

`outcome` ∈ `applied` | `no_op_already_selected` | `no_op_not_selected` | `row_not_in_window`.

Empty on every run nobody supplied overrides for — a typed empty table, not a missing file.

### The override input contract

A JSON list of objects. Every field is required; a blank or missing field refuses the **whole
file**, because a partially applied override file produces a queue nobody authorised.

```json
[
  {
    "override_id": "OV-2026-0417",
    "policy_id": "pure_risk",
    "fold_id": "quarterly-2026Q2",
    "k_name": "k_1_day",
    "target_inspection_id": "2467913",
    "action": "force_include",
    "reason_code": "outbreak_investigation",
    "actor": "district.supervisor.4",
    "decided_at": "2026-08-26T09:00:00Z"
  }
]
```

`force_include` displaces the lowest-ranked **risk** selection still standing — capacity is
fixed, so an inclusion costs an exclusion, and the displaced establishment is named. It never
raids the coverage reserve, which would quietly convert every override into a coverage cut.

`force_exclude` frees a slot and the slot is **not** backfilled. Backfilling would be the policy
making a second decision on the back of a human one.

Overrides are applied in `override_id` order, not file order, so re-serialising cannot change the
queue.

**The deterministic queue artifact is written unchanged.** The override log sits beside it, and
`validate.overrides_left_the_deterministic_queue_intact` checks that at error severity.

---

## 9. Determinism

**Byte-identical across runs given identical inputs.** Verified: 11 of 11 tables matched across
two independent production runs, and `tests/test_policy_determinism.py` asserts it end to end
plus under shuffled prediction rows and shuffled feature rows.

**Row order is an explicit input contract, not an assumption.** Every ordering descends from
`evaluation.metrics.top_k_indices` — descending score, ties ascending `target_inspection_id` —
and never from Parquet row order.

**The claim is scoped in the manifest:** identical *given identical inputs including the override
file*. Human overrides are external decisions; the manifest pins the file by checksum rather than
claiming a person's typing is reproducible.

---

## 10. Values: descriptive, normative or operational

| table | kind |
| --- | --- |
| `policy_coverage_eligibility`, `policy_group_audit`, `policy_comparison`, `policy_frontier` | **descriptive** — measurements of what each policy does |
| `policy_configurations`, `policy_model_selection` | **normative** — choices this component made, with their rules attached |
| `inspection_recommendations`, `policy_selection_allocation`, `policy_override_log` | **operational** — what the system recommends, and what a human decided |
| `policy_advisories` | **descriptive** — findings that never fail a run |

This distinction matters. A number in the first group describes the world. A number in the second
is a decision somebody can disagree with. A row in the third is an instruction to a person.

---

## Validation

22 checks. **Error severity fails the run; advisory severity never does, and there is no flag to
change that** (ADR 0034, inherited).

Errors cover: sort order and duplicate keys · exact coverage of the prediction universe ·
selected count equals capacity · allocation arithmetic and the reserve ceiling · no
double-selection · valid mechanism and reason pairing · reserve eligibility · the risk-prefix
contract · unique contiguous ranks · eligibility re-derived from its column · no outcome column
in any decision artifact · the queue unchanged with the warning inputs withheld · the grid
matching the frozen definitions · comparison covering every allocated cell · unsupported groups
preserved · overrides attributed · overrides not rewriting the queue · inputs unmodified.

Advisories cover: inert reserve cells · citations given up · group representation movement · the
absence of a policy winner.

> **A green run means the policy was applied correctly. It does not mean the policy is the right
> one.** That sentence is printed by the validation report on every run.
