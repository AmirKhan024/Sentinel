# Component 12 — Fairness and geographic equity audit

Plain language. No prior machine-learning knowledge assumed.

---

## 1. What problem does Component 12 solve?

Sentinel decides who gets inspected first. Chicago's health department can work about thirty
food inspections a day, and there are 35,859 establishments, so the order is the whole
decision — every establishment moved up the queue moves another one down.

Components 5 through 11 established that the ordering works, that the probabilities are
honest, and that the models can be explained. **None of that answers whether the system
behaves the same way everywhere in the city.**

That is the gap. A model can be right on average and consistently poor for one neighbourhood,
and every number the earlier components report would look exactly the same either way,
because they are all averages over the whole city.

So this component takes each earlier question and asks it again, once per neighbourhood:

```text
Component 5 asked   "does the ranking work?"     -> does it work HERE?
Component 9 asked   "are the numbers honest?"    -> are they honest HERE?
Component 11 asked  "what did the model use?"    -> does it use different things HERE?
                    "who gets inspected first?"  -> and how much of THIS neighbourhood's
                                                    risk did that actually find?
```

---

## 2. What is this component *not*?

This is the most important section, and it is deliberately near the top.

The word *fairness* means considerably more in public than what this code measures, and a
table called `fairness_group_metrics` showing per-neighbourhood numbers for a system that
allocates municipal inspections will be read by somebody as a finding about discrimination.
It is not one.

**It does not establish discrimination.** None of the models has a geographic input at all —
Sentinel's feature table is 26 numbers about a place's inspection history and contains no
address, no ZIP, no neighbourhood. Any difference the audit finds arises through features that
happen to be distributed unevenly, not through the model reading a group label.

**But the reverse is not true either**, and this is the trap the component exists to avoid.
"The model does not use community area, therefore it is fair" is a claim this audit was built
to be *able to contradict*. Fairness through unawareness is not fairness. That is why the
audit measures behaviour rather than inspecting a feature list.

**It does not establish causality.** Everything is observational. Nothing is randomised.

**It is not a protected-class audit.** This project ingests no demographic data of any kind —
no race, no income, no census variable. Chicago's community areas correlate strongly with race
and income (that is what the city publishes statistics against them for), but a correlate is
not the attribute. A legal disparate-impact finding requires the protected characteristic, and
this audit does not have it.

**It does not prove the absence of bias.** 27 of the 78 community areas were too small to
measure and were excluded from every comparison. A system can look even across the groups you
could measure and fail badly for one you could not.

**A green run means the audit is sound. It does not mean Sentinel is fair.** That sentence is
printed by the command on every run, not just written here.

---

## 3. What group data actually existed?

The honest answer is: less than the brief assumed, and it had to be checked rather than
guessed.

Chicago's inspection dataset publishes ten geographic columns. **Not one of them survives into
the model.** They exist only in the raw snapshot, and in one experimental table Component 8
built.

Two were usable:

| geography | why it was usable |
| --- | --- |
| **community area** | Chicago's 77 community areas have had fixed boundaries since the 1920s. That stability is what makes it safe to attach one to a 2019 inspection |
| **ZIP code** | recorded directly on the inspection record, and better supported than community area |

Five were refused, each with a measurement rather than an opinion:

**Ward** was the interesting one. The dataset publishes *two* ward columns — the current
boundaries and the 2003–2015 boundaries — and they disagree about which ward a place is in on
**98.3% of rows**. A ward is a property of a *boundary version*, not of a place. Labelling a
2019 inspection with today's ward assigns it to a district that did not exist when it
happened. That the publisher ships two vintages is itself the evidence.

**Census tract** would give 797 groups over 32,696 evaluated rows — about 41 rows each. Nothing
would be measurable, and the resulting table of blanks would look thorough while saying
nothing.

**City and state** are degenerate: 312,957 of 314,245 rows say `CHICAGO`, and the 95 distinct
values include `Chicago`, `chicago` and `CCHICAGO`. A group definition whose variety comes
from typing errors is not a group definition.

**The refusals are rows in the artifact, not sentences in a document.** Somebody who opens the
data file instead of reading this page still finds out why there is no ward breakdown, and
finds the number that decided it.

---

## 4. One subtlety worth understanding: where the neighbourhood comes from

An inspection record carries the establishment's location. So why not just read it?

Because this project has a rule, applied everywhere since Component 4: a row may only be
described using information that existed *before* it. The location is written onto the
inspection record at inspection time, so reading it off the row being audited would be the
one place in the repository that breaks the rule.

So the audit uses the neighbourhood recorded at that establishment's **previous** inspection.

Then it checked whether the safer choice cost anything:

| geography | rows where both values exist | times they disagreed |
| --- | ---: | ---: |
| community area | 57,041 | **0** |
| ZIP | 57,326 | **0** |

They never disagree — a restaurant does not move. **The safe option was free**, which is a much
better reason to take it than a principle would have been.

---

## 5. Before comparing anything: is there enough data to compare?

This is the step that shaped the entire component, and skipping it is how a fairness audit
ends up publishing a dramatic-looking ratio computed from twelve rows.

Chicago has 77 community areas. A quarterly test window holds about 1,800 inspections. Divide
one by the other:

> the **median** (quarter, neighbourhood) cell holds **16 rows**.

Sixteen rows cannot support an accuracy measurement. Only 4 of 1,288 such cells reach 200 rows.

So the audit reports at a different level: it pools the 17 quarters. That is legitimate —
every row is still one the model never saw during training — but it costs something, and the
cost is written onto every pooled row: those 17 quarters were scored by 17 separately-trained
models, so a pooled number describes *the system as it operated from 2022 to 2026*, not one
model.

Pooled, the thresholds were fixed **before** any result was looked at:

```text
at least 200 rows, and at least 20 of each outcome  -> can measure ranking accuracy
at least 300 rows                                   -> can measure probability honesty
```

Which leaves:

| geography | groups | can measure ranking | can measure calibration |
| --- | ---: | ---: | ---: |
| community area | 78 | **51** | 33 |
| ZIP | 69 | **56** | 41 |

**The 27 community areas that did not qualify are still in the artifact**, each with its real
counts and a sentence saying which threshold it missed. That is deliberate. A table containing
only the groups that qualified would report identical conclusions while making the shortage
invisible — and "we measured 51 of 78" reads very differently from "we measured 51".

---

## 6. What the data looked like before the model was even involved

Among the neighbourhoods with enough data, the rate at which inspections find a serious
violation ranges from **22.0% to 56.6%**, against a city-wide 42.8%.

That is a thirty-four point difference, and it exists in the outcomes themselves. Two things
follow.

**First, unequal treatment by the model is the expected behaviour of a working one.** If
violations really are three times more common in one neighbourhood than another, a risk model
that inspected both at the same rate would be ignoring a measured difference. So "the model
selects some neighbourhoods more often" is not, on its own, a finding.

**Second — and this is the limitation that bounds everything else — what is being measured is
that a violation was *written down*, not that a restaurant was unsafe.** The dataset does not
say who inspected. Chicago assigns inspectors by district, so geography is close to the
strongest available stand-in for *which inspector*. A neighbourhood with a higher citation rate
might have riskier restaurants, or stricter inspectors, and **nothing in this project can tell
those apart.**

That was known in advance — Component 7 recorded it in ADR 0019 and predicted this component
would inherit it — so it is stated here rather than discovered here.

---

## 7. What the audit found

### The ranking works much better in some places than others

Measuring how well each model ranks *within* each neighbourhood, the spread across
neighbourhoods is about **0.17 to 0.20** on a 0.5-to-1.0 scale.

For scale: the difference between Sentinel's best and worst *model* is about 0.008. **The
difference between neighbourhoods is more than twenty times the difference between models.**

### Calibration improved globally and got worse in a third of neighbourhoods

Component 9's headline was that calibration cut the error by about a quarter city-wide. Per
neighbourhood:

| model | neighbourhoods that improved |
| --- | ---: |
| lightgbm | 25 of 33 |
| xgboost | 25 of 33 |
| logistic regression | 23 of 33 |
| **neural network** | **17 of 33** |

For the best models, **8 of 33 neighbourhoods got worse**. For the neural network — the model
Component 8 called the best in the project — barely half improved.

This is not a contradiction of Component 9. It is the same fact seen at a finer resolution.
Calibration applies one correction to everybody; a correction that is right on average is
wrong for the places furthest from average, and the neural network was already closest to
correct so it had the least to gain and the most to lose.

**Nothing was done about it**, and that was a decision rather than an oversight. Fitting a
separate correction per neighbourhood would make the reported probability depend on where a
restaurant is — which is the thing the project already declined to let the *model* do.

### The sharpest finding is about a group that has no geography at all

405 of the evaluated inspections could not be assigned a neighbourhood, because the
establishment had no earlier inspection to carry one forward from.

It would have been easy to drop them as missing data. They were kept as a group, and they turn
out to be the most interesting rows in the audit:

```text
59.5% have no prior inspection of any kind    (city-wide: 0.74%)   -- 80x
61.7% have no recent violation history        (city-wide: 10.4%)   --  6x
   |
   v
ranking quality 0.51            -- indistinguishable from random
selected at 0.20x the city rate -- one fifth as often as average
6 in 1,000 of their violations found by the top 5%  (city-wide: 70 in 1,000)
```

Of the 166 real violations in that group, prioritising the top 5% of the city found **one**.

The chain is complete and every link is measured: *no record → no feature → nothing to rank on
→ deprioritised → still not inspected → still no record.*

**And it is a measurement, not an accusation.** "We have never inspected this place" is a true
and useful fact, and deleting that feature would not undo the inequality in inspection history
behind it — it would only stop the model seeing it. Which way the causation runs is not
answerable here.

### Two different questions about the top of the queue

The audit reports these separately and never combines them:

```text
selection rate  -- was this neighbourhood put near the front?
capture rate    -- and did that actually find its violations?
```

A neighbourhood can be prioritised often and still have its violations missed. Collapsing both
into one "fairness at top-5%" number would average two different things into something with no
meaning. Across neighbourhoods, capture ranges from **0.6% to 15.1%** against a city-wide 7.0%.

---

## 8. Why there is no single "fairness score"

Because the standard fairness criteria are **mathematically incompatible** when the underlying
rates differ — and here they differ by thirty-four points.

You can have a system whose probabilities are equally honest everywhere, or one that selects
every neighbourhood at the same rate, but when the real violation rates differ you cannot have
both. Any single score would be quietly choosing which one matters, on behalf of a reader who
cannot see the choice being made.

So the artifact reports **four** disparity measures side by side, and a policy component
decides what to do about them.

---

## 9. How this component avoided fooling itself

**Every disparity is advisory. None of them can fail the build.**

That sounds backwards for a fairness audit, and it is the most deliberate decision in the
component. If a large disparity turned the build red, then every future change to this
repository would be made under pressure to make that number smaller — and the three ways to
make it smaller are to change the model, change the metric, or move the threshold. Two of
those are worse than the disparity.

So the line is drawn somewhere else:

```text
the audit is WRONG      -> the build fails
the world is UNEVEN     -> recorded, printed, reported, and the build passes
```

Thirteen checks fail the build, and they are all about the audit's own integrity: a row
measured under the wrong quarter, a neighbourhood label taken from the future, the calibrated
and uncalibrated numbers swapped, a group silently dropped, an input file that changed while
the audit was reading it.

**Every one of those was deliberately broken in a test to prove it fires.** A check whose
failure has never been observed is indistinguishable from one that cannot fire — this project
shipped exactly that defect once, and has answered it in every component since.

And there is a test asserting the opposite of all the others: **a catastrophic 0.95 disparity
must leave every error check green and the exit code zero.** That test is what keeps the
component honest.

---

## 10. What was found and deliberately not fixed

* Calibration made a third of neighbourhoods worse. **Measured, reported, left alone** — the
  fix would be a substantive fairness decision disguised as a repair.
* The no-history group is ranked at random and almost never inspected. **Reported.** Fixing it
  is a policy question about how a city allocates attention to places it knows nothing about.
* Whether disparities are growing over time **could not be answered.** Only one quarter in
  seventeen had enough data per neighbourhood to compute the comparison at all. A line through
  one point is not a trend, so the artifact says `insufficient_folds` rather than fitting one.

---

## The sentence to leave with

> Sentinel behaves measurably differently across Chicago's neighbourhoods — the ranking varies
> more between neighbourhoods than between models, calibration improved on average while
> getting worse in a third of them, and the establishments with no inspection history are
> ranked at chance and almost never reach the top of the queue. This audit shows where that is
> true. It does not show why, it does not show that anyone was discriminated against, and it
> does not say what should be done about it.
