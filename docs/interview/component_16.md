# Component 16 — Deferral / human-review gate

Plain language. No prior machine-learning knowledge assumed.

---

## 1. What problem does Component 16 solve?

Sentinel can now predict risk, rank a queue, schedule it against a real calendar, and log human
overrides, adjustments and execution reports. What it has never done is stop and say: *"a human
should look at this case before the automated pipeline is treated as sufficient."*

Two concrete gaps motivated this:

* Component 13 already computes a `warnings` column — a row might say `limited_history`, meaning
  the model is ranking an establishment it has almost no evidence about. Today, a warning is
  annotation only. Nothing escalates it.
* Component 14 can schedule an inspection and never learn whether it happened. A row can sit in
  `inspection_schedule` forever with no matching row in `execution_log`, and nothing surfaces that
  as an operational gap somebody should chase down.

Component 16 is the mechanism that turns both of those into a visible, named case a human can act
on — without touching the score, the rank, the schedule, or any upstream artifact.

---

## 2. Where does it sit in the architecture?

```text
model -> policy (Component 13) -> schedule (Component 14) -> HUMAN REVIEW (Component 16)
                                                                    |
                                                                    v
                                            acknowledge / refer to override / refer to
                                            adjustment / escalate
```

It reads Component 13's `inspection_recommendations` and Component 14's `inspection_schedule` /
`execution_log`. It writes nothing back to either. It produces its own new artifact,
`human_review_queue`, plus a permanent log of what humans decided about flagged cases,
`review_resolution_log`.

---

## 3. What actually triggers a flag?

Exactly two conditions, both already-computed facts:

1. **A selected recommendation carries a policy warning.** `is_selected = true` and
   `warnings != "none"`.
2. **A scheduled inspection has no execution report.** An occupying schedule row (`scheduled` or
   `deferred`) with no matching row in the accumulated execution log.

Nothing here is a probability threshold. There is no `--threshold` flag, and there never will be
one added quietly — Sentinel has never built a predictive interval, and ADR 0040 states plainly
that manufacturing one to gate on would be fabricating a statistic.

---

## 4. Why not just add a confidence score and threshold it?

Because there is no confidence score to threshold. This project's models produce a calibrated
*probability*, and calibration (Component 9) is not the same thing as a *predictive interval* —
knowing that a probability is well-calibrated on average across a quarter says nothing about how
confident the model is about any one establishment. Building a per-row uncertainty estimate would
be new statistical work, with its own validation and its own ADR. Component 16 does not
manufacture that work to look complete; it uses the two facts that are actually available.

---

## 5. Why is this not the same thing as Component 14's "deferred" status?

Component 14 already has a schedule status called `deferred` — it means a scheduled inspection
was moved to a later operating day by a human adjustment or a re-plan. That is a scheduling-timing
fact.

A Component 16 review case means something completely different: a human should look at this row
before it keeps being treated as automatically sufficient. The two ideas are unrelated, and mixing
them up in code or in data would be a real defect — so the code enforces the difference
mechanically. At import time, Component 16's vocabulary is checked against Component 13's and
Component 14's own verb sets, and separately checked so that no Component 16 status or action ever
contains the literal word "defer".

---

## 6. What does a human actually do with a flagged case?

Four possible resolutions, each an explicit, attributed decision:

* **`acknowledge`** — looked at it, no further action through this component.
* **`refer_to_override`** — record that the decision will be (or has been) handled as a Component
  13 override. Carries the `override_id` as a pointer only; it does not create the override.
* **`refer_to_adjustment`** — same idea, for a Component 14 scheduling adjustment.
* **`escalate`** — beyond what this component's contracts represent.

A resolution is submitted the same way an override or an adjustment is: a JSON file with every
field required, applied in `review_id` order (never file order, so re-serializing the file cannot
change who "wins" a case two resolutions both address), with a mandatory actor, reason code, and
the reviewer's own timestamp.

---

## 7. How does a case stop being flagged?

Two different ways, and they mean different things:

* **A human resolves it.** The case moves from `flagged` to `resolved` in the queue, and the
  resolution stays in `review_resolution_log` forever.
* **The underlying condition disappears.** If an execution report shows up later, the
  `no_execution_record_on_scheduled_row` trigger no longer fires for that row, and it silently
  drops off the next run's queue — because the queue is rebuilt fresh from current state every
  time, the same way `inspection_recommendations` is. Nothing is lost: if a human ever resolved
  that case while it was flagged, that resolution is still in the permanent log.

---

## 8. What did a real run against production data find?

39,652 of 1,453,760 recommendation rows carried a policy warning on a selected row. 70,791 rows
were flagged in total (a union of both triggers). But the execution-gap count is not a genuine
operational finding yet — the production execution log is currently empty, so *every* occupying
schedule row is counted as "missing" a report, because nobody has filed one for any row. That is
stated honestly in the manifest rather than reported as if it were a discovery about real
inspector performance.

---

## 9. Difficult interviewer questions

**"Why didn't you just use a probability cutoff — it's the obvious design?"**
Because the project has no per-row confidence estimate to cut against. A cutoff on a calibrated
probability would be treating "average calibration across a quarter" as if it meant "this specific
row's uncertainty is known," which it does not. ADR 0040 already drew that line for the whole
project; this component holds it.

**"How do you know these are the right two triggers?"**
We don't claim they're complete — the interview doc and the ADR both say so. They're the two
conditions that are (a) already computed by an upstream component and (b) row-level and
queryable. A real health department almost certainly has more review criteria than this; adding
one requires it to be equally deterministic and equally traceable to a real fact, not a vibe.

**"Doesn't rebuilding the queue fresh each run risk losing track of a case?"**
No — the resolution log is append-only and permanent regardless of what happens to the live
queue. What disappears from the queue is the *flag*, not the record of what a human decided.

**"What stops 'refer to override' from silently becoming the override itself?"**
Nothing in this component creates an override. `refer_to_override` only records a pointer id; the
Component 13 override contract is a separate submission, validated by Component 13's own parser.
An invariant test checks this directly.

**"Isn't 70,791 execution gaps evidence of a real operational problem?"**
No — the production execution log has zero rows, so this count is fully explained by "nobody has
submitted any execution report yet," not by any actual failure to inspect. This is called out
explicitly rather than presented as a finding.

---

## 10. 60–90 second answer

"Component 16 is a deterministic gate that decides when an already-computed recommendation or
schedule should stop and go to a human instead of continuing on autopilot. It doesn't score
anything or predict anything — it flags a case only when an upstream component already recorded a
fact worth a second look: a policy warning on a row that was actually selected, or a scheduled
inspection with no matching execution report. There's no confidence threshold anywhere, because
this project has never built a real uncertainty estimate, and inventing one just to gate on would
be fabricating the exact statistic an earlier design decision explicitly refused to fabricate.
Humans resolve a flagged case with one of four attributed actions — acknowledge, refer to an
override, refer to a scheduling adjustment, or escalate — and those referrals are pointers, never
duplicate implementations of Component 13's or Component 14's own contracts. The queue itself
recomputes fresh every run; the record of what a human actually decided is permanent."
