# ADR 0038 — Coverage eligibility is missing history, not geography

**Status:** Accepted · **Date:** 2026-08-26

## Context

Component 12 handed this component a finding about a *place*: the `__UNKNOWN__` community area,
where the model ranks at chance and the top 5% finds one of 166 citations. The obvious response
is to reserve capacity for that group.

It would also be the single worst decision available here.

`__UNKNOWN__` is not a place. Component 12 says so in those words: it is the token Component 8
writes when no geography could be carried forward, so it is a data-quality artifact — a
population defined by a join that failed. Reserving municipal inspection capacity for
establishments whose address could not be geocoded is a rule that cannot be defended to an
inspector, an alderman or a court, and it would be an allocation keyed to a geographic label,
which ADR 0034 and ADR 0035 both foreclose.

The finding is nonetheless real. The question is what the *right* variable is.

## Decision

**Coverage eligibility is a per-row fact from Component 4's as-of feature table:
`prior_canvass_count_code_era == 0`. It is never a geographic group, never Component 12's
`__UNKNOWN__` token, and never a group-conditional number joined back onto a row.**

### The column was chosen because it is the cause, not a correlate

`scripts/profile_policy.py` swept four candidate missing-history rules. This one is the exact
condition under which `prior_canvass_priority_count`, `prior_canvass_priority_foundation_count`,
`prior_canvass_priority_rate` and `priority_at_last_canvass` are NULL — the four features that
encode the outcome the model is predicting. When it is zero the model is not making a weak
judgement about the establishment; it is making no judgement, because the evidence it would use
does not exist.

Component 11 measured that the corresponding missingness indicator ranks second or third in
importance for two of four models, which is the same fact from the other direction.

The other three candidates were rejected on measurement. `no_prior_inspection` is the strictest
and most intuitive rule and covers 401 of 57,727 rows (0.69%) — at one day of real capacity that
is a reserve of zero or one slot, which is a mechanism nobody can measure and therefore nobody
can defend. It is carried on every row as `secondary_no_history` for reporting, because it names
a genuinely distinct population.

### The two populations overlap and are not the same thing

Profile 8 measured it: 66.5% of `__UNKNOWN__` rows are coverage-eligible against 24.0% of rows
with a named community area, and of the 14,162 eligible rows only 456 — 3.2% — sit in
`__UNKNOWN__`. The overlap is what makes the distinction necessary to state rather than what
makes it unnecessary.

A reserve keyed to missing inspection history is a rule about **what the model can and cannot
know**. A reserve keyed to `__UNKNOWN__` is a rule about **whose address failed to geocode**.
The first survives being read aloud in a room where somebody disagrees with it.

### A null is never eligible

The column carries `NullRule.NEVER` in Component 4, so a zero is a positive observation that no
code-era canvass exists. A null would mean the count itself is missing — a join that failed, a
column that moved. `_eligible_expr` maps a null to a value the predicate cannot match rather
than to zero, because reserving capacity for rows about which nothing at all is known is the
defect this component exists to make impossible.

### The audit still reaches the artifact — as a label, never as a score

Component 12's group value and support status are read onto every recommendation row and
surfaced as `warnings`. They tell a reviewer that this establishment sits in a neighbourhood the
audit could not measure, or in none at all. They never enter a rank.

That is checked rather than claimed. `_queue_signature` rebuilds the entire queue with the group
label and support status absent, and `validate.warnings_do_not_change_the_queue` compares the
ranks exactly. `tests/test_policy_build.py` runs the whole component twice — once with the group
artifacts and once without — and asserts the recommendation ranks are identical.

### ADR 0035 delegated a choice here, and this is the answer to it

ADR 0035 closed with a sentence that names this component:

> the standard criteria are mutually incompatible when base rates differ — and they differ
> here, from 0.220 to 0.566 across supported community areas — so "optimal" is undefined until
> someone chooses which criterion to prefer. **That choice is Component 13's.**

**Component 13 declines to choose one, and the refusal is the answer rather than an evasion.**
Three measured reasons, in order of weight.

**No criterion has an objective this component is authorised to optimise.** Choosing between
demographic parity, equalised odds and calibration-within-groups is choosing what a city owes
which neighbourhoods. It is a political judgement with a technical surface, and nothing in this
project measures the quantity that would decide it.

**The criteria are defined over protected characteristics, and none is observed.** ADR 0035's
own boundary: community area and ZIP correlate with race and income by construction, but a
correlate is not the attribute. A criterion optimised over a proxy is not that criterion; it is
a different one nobody named.

**And the finding that motivated the delegation does not survive contact with the data.** ADR
0035 wrote that sentence because Component 12 had measured the `__UNKNOWN__` group ranked at
chance and barely selected. Component 13's profiler measured the population that finding
appears to be about — establishments the model has no history for — and found the risk queue
**over-serves** it by four to five times its population share. There is no under-service to
correct at the level a fairness criterion would operate on, and only 3.2% of that population is
in `__UNKNOWN__` at all.

What Component 13 does instead is publish the trade-off and refuse to collapse it:
`policy_frontier` marks dominated policies on two axes and stops, because ranking the survivors
needs an exchange rate between a missed Priority citation and an uninspected establishment with
no history. When no policy is uniquely non-dominated the run prints *the data does not determine
the correct policy*. **That is the honest discharge of ADR 0035's delegation: the choice was
handed here, this component measured what each option costs, and it hands the choice back with
the price list attached.**

### Temporal safety is inherited, not re-derived

`prior_canvass_count_code_era` is an ADR 0010 as-of feature: it counts canvasses strictly before
the row's own reference date, from records that existed by then. This component reads that
column and adds nothing to it, which is the only way to be sure eligibility cannot see further
than the model did. `validate.eligibility_matches_the_declared_rule` re-derives the flag from the
column on every run and compares.

## Alternatives rejected

**Reserve capacity for the `__UNKNOWN__` community area.** The direct response to Component 12's
finding, and the one this ADR exists to refuse. It is an allocation keyed to a geographic label,
applied to a group defined by a failed join, and it is exactly the "score-adjustment table"
reading of a fairness audit that ADR 0035 warned the numbers would attract.

**Define eligibility on Component 12's group-conditional numbers** — say, groups where measured
ROC-AUC is near chance. Rejected on two grounds. HANDOFF forbids joining a `fairness/` table onto
a per-row artifact, and the reason is that a number meaning "the model was well calibrated in
this neighbourhood last quarter", broadcast back onto rows, is the most self-fulfilling input
this project could construct. It is also a leak: those numbers are computed from held-out
outcomes.

**Use `no_prior_inspection`, the strictest and most intuitive rule.** Rejected on size, not on
principle: 401 rows across eight years cannot support a measurable allocation. Retained as a
reported flag.

**Use the union of all four missing-history families.** Rejected because three of the four are
nested inside the fourth in practice and the union adds population without adding a distinct
claim — and because a rule that names four columns is four times as easy to get wrong in an edit.

**Define eligibility as "the model ranks this population poorly".** Unfalsifiable, and circular:
the population would be defined by the behaviour the policy is meant to respond to, so the
policy could never be wrong.

## Consequences

- `ELIGIBILITY_COLUMN`, `ELIGIBILITY_RULE` and `ELIGIBILITY_IS_NOT_GEOGRAPHY` are frozen in
  `policy/definitions.py` and written into every manifest.
- `eligibility.refuse_forbidden` raises before the predicate is built if a label column is
  offered, and `tests/test_policy_eligibility.py` drives that red for `target` and
  `target_status`.
- `validate.warnings_do_not_change_the_queue` is an error-severity check, and
  `test_a_warning_input_that_changes_the_queue_turns_the_check_red` proves it can fire.
- `validate.unsupported_groups_are_preserved` fails the run if a group Component 12 called
  unsupported is relabelled here, so the easiest way to produce a flattering group table is
  closed.
- Component 12's finding about `__UNKNOWN__` is **not** acted on as a geographic rule and **is**
  reported: the group carries the `unknown_geography` warning on every row, and
  `policy_group_audit` reports its selection share and capture under each policy.
- A future component that wants a genuinely geographic intervention must ingest a protected
  characteristic and argue for it separately. ADR 0035 took that position for inspector identity
  and this ADR takes it here, for the same reason: a correlate is not the attribute.
