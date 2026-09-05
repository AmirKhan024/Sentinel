# ADR 0037 — Coverage is an explicit allocation, never a score adjustment

**Status:** Accepted · **Date:** 2026-08-26

## Context

Component 12 measured a closed loop. Establishments in the `__UNKNOWN__` community area have no
prior inspection of any type in 59.5% of cases against 0.74% overall; 61.7% are missing code-era
canvass history against 10.4%; the models rank them at ROC-AUC 0.509–0.532, which is chance; and
at the top 5% they are selected at 0.20× the overall rate and their citations are found at 0.006
against 0.070. Of 166 positives, the top 5% found one.

Something in the system has to respond to that, and there are two families of response.

The first adjusts the **estimate**: add a bonus to the score of an establishment with little
history, fit a group-specific calibrator, apply a lower probability threshold to the group. The
second adjusts the **allocation**: leave every estimate exactly as it is and change how capacity
is divided.

The first family is easier to implement, produces a single ranked list with no extra machinery,
and is what most systems do. It is also unauditable. Once a score has been adjusted, nobody
looking at the queue can say whether an establishment is there because the model thinks it is
risky or because a policy decided to promote it, and the two facts have become one number.

## Decision

**A coverage policy changes which establishments are inspected. It never changes how risky any
establishment is held to be.**

### No score is written anywhere in Component 13

`allocation.py` reads scores to order rows and writes none. The recommendation artifact carries
`base_score` and `score` copied verbatim from Component 9, and carries `model_rank` beside
`final_policy_rank` so the two orderings sit side by side on every row. Where they agree the
model decided; where they differ the policy did, and the `decision_mechanism` column says which
mechanism moved it.

That pair of columns is the whole design. An interviewer can point at any row and ask "did the
model make this decision or did the policy?" and the artifact answers without anyone reading
source code.

### The reserve is a slot count, not a bonus

A coverage policy declares a share of capacity. At capacity *k* the reserve target is
`floor(share * k)` — truncated, so a reserve can never spend more than the share it declared.
The risk block takes the top `k - granted` by calibrated risk; the reserve takes eligible rows
the risk block did not take. The two mechanisms are disjoint by construction, every selected row
carries exactly one of them, and `policy_selection_allocation` records offered, already-satisfied
and granted as three separate numbers.

### Ordering inside the reserve is calibrated risk, and the cost of that is stated

The reserve spends its slots on the eligible establishments the model likes most. Component 12
measured the model's ranking *inside* this population at roughly chance, so "the ones the model
likes most" is doing modest work — but the alternative, a canonical date-and-id order, is a
lottery, and a lottery is defensible only if one believes the ranking carries no information at
all, which is a stronger claim than the measurement supports.

Using risk to order the reserve is **not** using risk to size it. The reserve changes allocation;
the estimate is untouched.

### Two mechanisms, because they are different policies

A **floor** guarantees an outcome: at least `share * k` of the queue is coverage-eligible, and
nothing happens when the risk ranking already delivers that. A **forced** reserve guarantees a
spend: `share * k` slots go to eligible rows the risk ranking passed over, whether or not it
already selected others.

Most people mean the second when they say "reserve some capacity", and the first is what a
coverage guarantee actually is. On this data they diverge almost completely — at the population
share the floor is inert in 84 of 85 quarterly cells and grants 2 slots across all four models,
while the forced reserve moves 274 slots at a week of capacity for a measured cost of 15
Priority citations. Implementing only one would have hidden the entire result: only the floor,
and the component reports "the mechanism does nothing" without being able to say what a
mechanism that *did* something would cost; only the forced reserve, and it never discovers that
the guarantee everyone actually means is already satisfied.

### Every policy is priced against doing nothing

`pure_risk` is in the grid as a candidate rather than as a control. Every coverage policy's cell
is differenced against it at the identical model, fold and capacity, and the difference is
reported in **citations** first: `delta_positives`, alongside `delta_precision`,
`delta_capture`, `delta_nde` and the coverage gained. A reserve is described as free only where
the measured delta is zero.

## Alternatives rejected

**Add a score bonus for establishments with little history.** The obvious implementation, and
rejected because it destroys the audit. After a bonus, "why is this establishment in the queue?"
has one answer where it should have two, and no downstream consumer — a supervisor, an
inspector, a court — can separate the model's judgement from the department's policy.

**Fit a group-specific calibrator.** ADR 0034 already refused this for Component 12 and gave the
reason: it would change Component 9. It is also a score adjustment wearing statistical clothes.

**Apply a lower probability threshold to the low-history population.** Refused twice over. It is
a score adjustment, and Component 12's `THRESHOLD_POLICY` records that this project has never
derived a probability cutoff from anything. Every cutoff in Component 13 is a rank position
derived from the window's own measured median daily inspection rate.

**Implement the floor only, since it is the defensible mechanism.** The most tempting
alternative. Rejected because the floor turned out to be inert almost everywhere, so a component
that implemented only the floor would have reported "the reserve does nothing" without being
able to say what a reserve that *did* something would cost. The forced mechanism is the
counterfactual that makes the finding legible.

**Optimise the reserve share against a coverage objective.** Rejected as out of scope and out of
authority. Optimising requires an exchange rate between a missed Priority citation and an
uninspected establishment with no history; nothing in this project measures one, and inventing
one would be this component setting a city's enforcement priorities by choosing a constant.

## Consequences

- Every recommendation row carries `model_rank`, `final_policy_rank`, `decision_mechanism` and
  `decision_reason`, so the model/policy boundary is answerable per row from the artifact alone.
- `tests/test_policy_allocation.py` asserts the two mechanisms are disjoint, that every reserve
  selection is coverage-eligible, and that `n_risk + n_reserve == k` across a grid of policies
  and capacities.
- `validate.reserve_rows_are_eligible` and `validate.risk_rows_satisfy_the_risk_contract` fail
  the run at error severity, and `tests/test_policy_leakage.py` drives both red on purpose.
- `validate.coverage_is_not_free` fires as an **advisory** whenever a policy gave up citations,
  and `test_a_reserve_that_gave_up_citations_is_advisory_and_never_an_error` asserts it never
  fails a build. The cheapest way to make such a build green is to delete the reserve.
- The manifest carries `capacity_semantics`, stating that every cutoff is a rank position and
  that no probability threshold exists — with no flag to add one.
- A future component that wants a scored intervention must revisit this ADR explicitly. Nothing
  here licenses one, and the columns to hold it do not exist.
