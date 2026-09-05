# Component 13 — Decision policy and deployment governance

Plain language. No prior machine-learning knowledge assumed.

---

## 1. What problem does Component 13 solve?

After twelve components, Sentinel can look at a Chicago food establishment and say *there is a
62% chance this one gets cited for a Priority violation if we inspect it now*. That number is
honest — Component 9 checked that a predicted 62% happens about 62% of the time.

**A number is not an action.** Somebody still has to decide who gets inspected on Tuesday.

The obvious move is to sort by that number and take the top thirty. That looks like no decision
at all — it looks like just using the model. It is actually a decision, and it silently assumes
all of the following:

- higher estimated risk should always beat every other consideration
- an establishment the model knows nothing about should accept whatever rank it gets
- nobody needs to be told which recommendations rest on thin evidence
- a supervisor who wants to override the list has no supported way to do it
- and if the queue systematically skips a whole category of establishment, that is fine

Component 13 stops making those assumptions silently and makes them arguable instead.

```text
MODEL LAYER      data -> as-of features -> trained model -> calibrated probability
                 |
                 |  a probability is not yet an action
                 v
POLICY LAYER     capacity + eligibility rules + governance + allocation -> a queue
                 |
                 v
HUMAN LAYER      review + external constraints + documented overrides + audit log
```

The component owns the middle band. The test of whether it succeeded is simple: point at any
recommendation and ask **"did the model decide this, or did the policy?"** The artifact has to
answer.

---

## 2. What is this component *not*?

Deliberately near the top, because a table of who to inspect invites more misreading than
anything else this project produces.

**It is not a new model.** Nothing is trained, refitted, recalibrated or rescored. The component
reads nine closed components' files, does arithmetic, and writes decisions. It checksums every
input before it starts and again after it finishes, and fails if a single byte moved.

**It does not say the recommended queue is the correct queue.** It is the queue one stated
policy produces from one selected model under one capacity assumption.

**It does not say the model it selected is the best model.** More on that in §4 — the honest
answer is that four models were indistinguishable and a secondary criterion broke the tie.

**It is not a fairness fix.** No score is adjusted by neighbourhood, no group gets its own
threshold, no quota exists. Component 12 measured differences across Chicago; this component
does not act on geography at all, and §7 explains why that is the right call rather than an
evasion.

**It does not decide anything by itself.** Every row is a recommendation to a person, and the
override contract exists because operations will legitimately depart from it.

---

## 3. The surprise: the thing we built this for turned out not to be true

Component 12 handed over an uncomfortable finding. In the group of establishments whose
neighbourhood could not be identified, the model ranks at roughly **chance**, and of 166
establishments that were cited, the top 5% of the queue found **one**.

The obvious reading is: *the model neglects establishments it knows nothing about, so let's
reserve some inspection capacity for them.*

Before writing any policy code, we measured whether that reading was true. **It is not.**

Establishments with no inspection history since Chicago's 2018 food code took effect are **10.4%
of the candidates** and take **40% to 58% of the top of the queue** — between four and five and a
half times their share of the population, and it holds for all four candidate models.

The models are not neglecting them. They are prioritising them heavily, and they are right to:
that population's citation rate is **48.8% against the city-wide 42.8%**.

So why does Component 12's finding exist? Because it is about a **different group**. "No
neighbourhood on record" and "no inspection history on record" overlap but are not the same
thing: only **3.2%** of the no-history population sits in the no-geography group.

This is the component's most valuable result, and it is a negative one. The intervention
everybody's intuition demands would have been aimed at a problem that is not there.

---

## 4. Which model does the queue use?

Nine components produced five candidate models and never picked one. Components 11 and 12 were
explicitly forbidden from picking, and it landed here because a department cannot work four
queues.

The rule was written down before it was run:

```text
1. discovery efficiency  (does the ranking find violations early?)
2. calibration quality   (do the probabilities mean what they say?)
3. precision at one day of real capacity
4. the model's name      (so the rule always terminates)
```

Try each in order; move to the next only if the previous cannot separate.

**Step 1 could not separate anything.** Component 5 already publishes an uncertainty band for
discovery efficiency, built by scrambling 1,000 sets of labels per quarter and seeing how much
the number moves. Every candidate's band overlaps every other candidate's. The headline metric
of this entire project cannot tell these four models apart.

So the rule fell to calibration and selected **`xgboost_platt`**.

### The uncomfortable part, which is written down rather than smoothed over

Deciding *when two models are tied* is what decides which model gets deployed.

The plan originally carried a different tie band, borrowed from a different metric — and using a
ROC-AUC spread to judge an NDE difference is a units error, noticed only after the numbers had
already been looked at. It was replaced with the right quantity.

**The two rules pick different models.** Under the discarded one, the neural network wins.

That could have been quietly tidied away. Instead, every run emits both answers side by side,
and ADR 0039 records the sequence. A rule chosen after you have seen what it decides is
defensible only if the choosing is visible.

---

## 5. The two kinds of "reserve some capacity", and why both were built

Everyone means slightly different things by *reserve capacity for establishments we know nothing
about*, and the difference turns out to be the whole result.

**A floor** is a guarantee: *at least 10% of each day's inspections go to no-history
establishments.* If the ranking already does that, the floor does nothing.

**A forced reserve** is a spend: *set aside 10% of each day's slots for no-history
establishments the ranking passed over*, whether or not it already picked others.

Seven policies were compared: doing nothing, plus each mechanism at half, exactly, and twice the
measured 10.4% population share. Those three sizes are the only reason a number near 10% appears
anywhere in the component — it is a measurement, not a round number somebody liked.

---

## 6. What it cost

Pooled over seventeen quarters, at one week of real inspection capacity (2,780 inspections):

| policy | reserve slots used | citations found | change | no-history establishments served | change |
| --- | ---: | ---: | ---: | ---: | ---: |
| do nothing | 0 | 1,657 | — | 1,170 | — |
| floor at 10% | 0 | 1,657 | 0 | 1,170 | 0 |
| floor at 20% | 2 | 1,657 | 0 | 1,172 | +2 |
| forced 5% | 133 | 1,649 | **−8** | 1,246 | +76 |
| forced 10% | 274 | 1,642 | **−15** | 1,325 | +155 |
| forced 20% | 556 | 1,623 | **−34** | 1,513 | +343 |

Two things to read here.

**The floor does almost nothing** — across every model and every quarter it granted 2 slots out
of 340 opportunities at the 10% setting. That follows directly from §3: the ranking already
clears the bar four times over.

**The forced reserve buys coverage at a real price.** Reserving twice the population share
serves 343 more low-information establishments and gives up **34 Priority violations that would
otherwise have been found**. Not "a small efficiency loss" — 34 inspections that would have
caught something and did not.

Nothing here is described as free. The number is reported whatever it says, and it is reported
in citations because that is the unit a health department can argue about.

---

## 7. What happens to the group Component 12 was worried about

Nothing. And that is stated rather than glossed.

At one day of capacity, the no-geography group gets **2 of 556 slots and 1 citation found — under
every single one of the seven policies**, including the most aggressive reserve.

The reason is structural. The reserve is keyed to missing *history*, and only 3.2% of the
no-history population is in the no-geography group. A reserve that actually reached that group
would have to be keyed to the geography — which means reserving public inspection capacity for
establishments **whose address failed to geocode**. That is not a rule anyone can defend to an
inspector, an alderman or a court. It is a data-quality artifact wearing a policy's clothes.

So the component does the two things it can honestly do: it flags every row in that group in the
operational artifact, and it reports that the problem is unsolved. It does not invent a fix that
would look like progress.

---

## 8. So which policy should Chicago use?

**The component declines to say, and that is the answer it publishes.**

Two policies survive the comparison at one day of capacity, and neither beats the other on both
things that matter — one finds slightly more violations, the other serves considerably more
low-information establishments.

Picking between them requires an exchange rate: *how many missed Priority violations is one
inspection of an establishment we know nothing about worth?* Nothing in this project measures
that, and inventing a number would be a piece of software setting a city's enforcement
priorities by choosing a constant.

So the run prints, in these words:

> **POLICY WINNER: the data does not determine the correct policy.**

The full trade-off is published; the choice is left where it belongs.

There is a specific obligation being discharged here. Component 12's closing decision record
ended by saying that choosing between the standard fairness criteria was *this* component's job.
Component 13 declines, and says why: no criterion has an objective this component is authorised
to optimise, all of them are defined over protected characteristics that this project never
observes, and — most concretely — the under-service that made the question seem urgent turns out
not to exist at the level a criterion would operate on. The choice was handed here; it is handed
back with a price list attached.

---

## 9. What a person working the queue actually sees

Each recommendation carries its rank, the mechanism that put it there, a reason code, and any
warnings. At one day of capacity, nearly half the queue carries a warning:

| warning | rows out of 556 |
| --- | ---: |
| none | 286 |
| limited history | 220 |
| the audit could not measure this neighbourhood | 21 |
| both of the above | 27 |
| limited history + no neighbourhood on record | 2 |

**A warning is not a refusal.** Every row still gets a recommendation and a rank. Sentinel never
says "I don't know" — and the reason is not confidence, it is honesty: doing so properly needs a
per-row uncertainty estimate, and this project has never built one. Making one up to justify an
abstention would be inventing the statistic that licenses it.

---

## 10. When a human disagrees

A supervisor can force an establishment into the queue or out of it. Two rules make it
auditable.

**Adding costs something.** Capacity is fixed, so an inclusion pushes out the lowest-ranked
risk-based selection — and the log names which one. The system never quietly finds an extra
inspector.

**Removing does not backfill.** Strike a row and the slot stays empty. Promoting the next
establishment would be the software making a second decision on the back of a human one, and the
supervisor who struck that row did not ask for a replacement.

Every override needs an id, an actor, a reason code and a timestamp. Miss any one and the whole
file is rejected — a half-applied override file produces a queue nobody authorised.

The machine's recommendation is written **unchanged**, and the human decision sits in a separate
log beside it with both versions. An audit never asks only what happened; it asks what would
have happened, what happened instead, and who decided.

And the reproducibility claim is scoped honestly: the policy computation is byte-identical across
runs, *given the same override file*. A person's typing is not reproducible and the component
does not pretend it is.

---

## 11. How this component avoided fooling itself

**Everything was measured before anything was decided.** A read-only profiling script ran first
and produced the findings document; the eligibility column, the reserve sizes and the selection
axes were all set from those numbers. That is how §3's surprise was caught before it could
become a feature nobody needed.

**The advisory geography is provably advisory.** The whole queue is rebuilt on every run with the
neighbourhood label and the audit status removed, and the ranks compared exactly. If a Component
12 number ever leaked into a ranking decision, the run goes red — the artifact would look
completely normal otherwise.

**Forty-four tests break things on purpose.** A queue longer than the day's capacity; an
establishment with full history smuggled into the low-information reserve; the same
establishment selected twice; a rank appearing twice; a lost row; an override with no actor. Each
would produce an entirely plausible-looking file, and each is proven to turn the build red.

**And five tests assert the opposite.** A reserve that gave up 34 citations must **not** fail the
build. Neither must an inert reserve, a group whose share of the queue moved, or the absence of a
winner. The reasoning is short: the cheapest way to make a red "this reserve cost 34 citations"
build green is to delete the reserve — and that is a decision about how a city allocates
enforcement, not one a CI runner is entitled to take.

**Two full production runs produced byte-identical files** across all eleven tables, and shuffling
the input rows changes nothing — including on a window where every score is tied, which is where
Component 12 once found a real ordering bug.

---

## 12. What was found and deliberately not fixed

- The no-geography group is served no better under any policy. Reported, not worked around.
- The one-day cost numbers are inside the noise (±1 to ±3 citations out of 348). Said plainly,
  including where the noise happens to flatter the component.
- No confidence interval is placed on any policy cost, so "is −34 citations distinguishable from
  zero?" is not answered.
- The model choice is not robust: two defensible tie rules pick two different models.
- The distribution-shift window orders the models differently again, and is excluded from the
  rule and named as a limitation.

---

## The sentence to leave with

> The model estimates risk; it does not decide who gets inspected. A deterministic policy layer
> turns those estimates into a capacity-constrained queue, records for every establishment
> whether the model or the policy put it there, and prices every alternative in the only unit
> that matters — Priority violations found or missed. We tested the intervention everyone's
> intuition demands and found the queue already over-serves the population it was meant to help
> by four to five times, so the guarantee is nearly inert and the aggressive version costs 34
> citations a week. No score is adjusted by neighbourhood, no quota exists, no fairness claim is
> made, and where the evidence does not pick a policy the system says so instead of picking one.
