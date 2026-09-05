# ADR 0040 — Human overrides, and why a warning is not an abstention

**Status:** Accepted · **Date:** 2026-08-26

## Context

Component 13 produces an ordered inspection queue. Two things follow that the previous twelve
components never had to face.

First, real operations will not follow it exactly. An outbreak is reported, a complaint arrives,
a court orders an inspection, a facility opens outside anything the model has seen. A system that
cannot absorb those without being switched off is a system that gets switched off.

Second, the queue's rows are not equally well evidenced. Some establishments have eight years of
canvass history; some have none. A reviewer working the queue should be told which is which.

Both invite the same mistake — treating "we know less about this row" as a quantity — and the
project has no such quantity. Sentinel has never built a predictive interval, a conformal set or
an ensemble spread. There is nothing to abstain *on*.

## Decision

**Sentinel never abstains. Every row receives a recommendation and a rank; a warning annotates
that recommendation rather than withholding it. Human overrides are external inputs, recorded
beside the deterministic queue and never inside it.**

### A warning is a fact about evidence, not an estimate of uncertainty

Four codes, each a deterministic fact somebody already measured:

```text
limited_history                    no canvass since 2018-07-01; the priority features are NULL
no_prior_inspection                no inspection of any type on record
unknown_geography                  no community area could be recovered (Component 12's token)
insufficient_group_audit_support   the audit could not measure this neighbourhood
```

None is a probability. None enters a rank. The third and fourth say what Component 12 could not
establish about a row's surroundings — which is a different claim from "the model is wrong here",
and the one a reviewer actually needs.

The column is a sorted, pipe-joined set rather than a single highest-precedence code. Choosing
one to display would mean choosing which fact about an establishment a reviewer is allowed to
see, and "we have never inspected this place" and "we cannot measure how the model behaves in
this neighbourhood" are different problems for different readers. A row with nothing to flag
carries the literal token `none`, because an empty cell is ambiguous between "no warning" and
"warnings were not computed".

### There is no abstention category, and the reason is stated rather than assumed

Emitting `insufficient_confidence` would require a per-row confidence estimate. Manufacturing
one — from score distance to 0.5, from an ensemble the project does not have, from the number of
non-null features — would be inventing the statistic that justifies the abstention. The refusal
is recorded in `ABSTENTION_POLICY` and travels in every manifest, following ADR 0030's treatment
of probability-space attribution: refused in prose rather than declared and left unreachable.

### An override changes the decision. It never changes the evidence

Two verbs, both auditable:

- `force_include` puts an establishment in the queue. Capacity is fixed, so it **displaces** the
  lowest-ranked risk selection still standing, and the displaced establishment is named in the
  log rather than absorbed.
- `force_exclude` takes one out. The freed slot is **not** backfilled.

The no-backfill rule is deliberate and is the one most likely to be questioned. Backfilling would
be the policy making a second decision on the back of a human one, and the reviewer who struck a
row did not ask for a replacement. A supervisor who wants the slot re-used can add a
`force_include` and the log will show both decisions.

`force_include` raids the risk block rather than the coverage reserve. Taking the slot from the
reserve would quietly convert every override into a coverage cut — a policy change nobody made.

### The deterministic artifact is written unchanged

`inspection_recommendations` is produced from the policy computation and never edited.
`policy_override_log` sits beside it with the original recommendation, its mechanism, its rank,
the final decision, the displaced row, the actor, the reason code and the timestamp. An audit
never asks only what happened — it asks what would have happened, what happened instead, and who
decided — and both halves have to be readable years later.

`validate.overrides_left_the_deterministic_queue_intact` checks it at error severity.

### Every field is required, and one bad row refuses the file

An override with no actor is an anonymous change to who gets inspected; one with no reason code
is a change nobody can review. Neither is something to fill in helpfully, so both are validation
errors.

A malformed row refuses the **whole file** rather than being skipped, because a partially applied
override file produces a queue nobody authorised: the reviewer believes they made five changes
and four happened.

Overrides are applied in `override_id` order, not file order, so re-serialising the file cannot
change the queue.

### The determinism claim is scoped precisely

Written into every manifest as `determinism_scope`:

> the policy computation is deterministic: identical inputs produce byte-identical tables, and
> shuffling the input rows changes nothing. Overrides are external human decisions, so a run is
> byte-identical only given the identical override file, and the manifest pins that file by
> checksum rather than claiming it is reproducible.

Claiming byte-identity across runs for something a person typed would be the easiest lie
available in this component.

### The layer boundary this component ends at

```text
MODEL LAYER      historical data -> as-of features -> trained model -> calibrated probability
                 |
POLICY LAYER     capacity + eligibility + governance + deterministic allocation -> queue
                 |
HUMAN LAYER      review + external constraint + documented override + audit log
```

Component 13 owns the middle band and hands the third an artifact plus a contract. It does not
own the third, and it does not pretend the third does not exist.

## Alternatives rejected

**Add an abstention category so the system can decline low-evidence rows.** Superficially the
responsible choice, and rejected because it needs a confidence estimate the project has not
built. It would also be operationally strange: every establishment in the universe is inspected
eventually, so there is nothing to decline — the question is only *when*.

**Derive confidence from the calibrated probability's distance from 0.5.** Rejected. That is a
measure of predicted risk, not of evidential support, and an establishment with no history can
score 0.9 with nothing behind it.

**Backfill the slot a `force_exclude` frees.** The most-requested behaviour and rejected on
authority. The system would be choosing an additional inspection nobody asked for, and the
displaced-and-replaced chain would make the log's causal story unreadable.

**Let `force_include` raid the coverage reserve when the risk block is small.** Rejected: it
converts an operational override into a silent policy change.

**Skip malformed override rows and process the rest.** Rejected. Partial application produces a
queue that neither the policy nor the reviewer authorised, and the reviewer is not told.

**Implement the contract but not the application.** Considered, and rejected because a governance
mechanism that has never been executed is a document, not a mechanism. The failure paths — an
inclusion with nothing left to displace, an override for a row outside the window — are only
observable if the code runs.

**Build a review UI.** Out of scope for this component and for this project. A structured JSON
contract plus an audited log is the interface; what renders it is Component 21's problem.

## Consequences

- `--overrides PATH` on `sentinel decide`; `OVERRIDE_REQUIRED_FIELDS`, `OverrideAction`,
  `OVERRIDE_CANNOT` and `ABSTENTION_POLICY` are frozen in `definitions.py`.
- The override file is pinned by checksum in the manifest alongside `overrides_applied`, and is
  re-checksummed after the run like every other input.
- `policy_override_log` is written on every run, empty when no overrides were supplied — a typed
  empty table rather than a missing file, so a reader meets the schema and concludes "none".
- `validate.overrides_are_fully_attributed` and
  `validate.overrides_left_the_deterministic_queue_intact` fail the run at error severity;
  `tests/test_policy_leakage.py` drives both red, including one parametrised case per required
  field.
- `tests/test_policy_governance.py` asserts the displacement, the absence of backfill, both no-op
  outcomes, id-order application, and that an inclusion with nothing left to displace raises
  rather than raising capacity.
- **A green run means the policy was applied correctly. It does not mean the policy is the right
  one, and it says nothing about whether the overrides were justified.** That sentence is printed
  by the validation report on every run.
