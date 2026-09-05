# Component 11 — Explainability and feature attribution

Plain language. No prior machine-learning knowledge assumed.

---

## 1. What problem does Component 11 solve?

Sentinel already tells a public health department **which** establishments to inspect first,
and how confident it is. It could not tell anyone **why**.

That gap is the whole component. An inspector's day is a scarce, reallocatable resource, and
Sentinel proposes reallocating it. When it moves a restaurant to the front of the queue,
somebody eventually asks: *what did the model see?* Before Component 11 the only honest answer
was "a number came out of a model". Now the answer is a table:

```text
Restaurant: inspection 2650133,  fold quarterly-2026Q2
Model score before calibration:  0.706
Calibrated risk probability:     0.664

Main factors increasing risk
  1. historical rate of priority citations   +0.42
  2. cited at the last routine inspection    +0.19
  3. long gap since the last routine visit   +0.08

Main factors reducing risk
  1. recent clean follow-up                  -0.11
  2. low complaint history                   -0.04
```

Nobody wrote those sentences by hand. They are generated from the artifact, deterministically,
and the same command run tomorrow produces the same numbers.

---

## 2. Why is a raw prediction not enough for a regulator?

Four reasons, and only the first is the obvious one.

**Accountability.** A public agency acting on a score can be asked to justify it, in a
meeting, in a FOIA response, or in court. "The model said so" is not a justification.

**Error detection.** A model can be right for the wrong reason. If Sentinel's top signal
turned out to be something that merely *encodes the department's existing schedule*, the model
would look predictive while adding nothing — it would be forecasting the department's own
behaviour. You cannot notice that from an accuracy number. You can notice it from an
attribution table.

**Trust, in the right direction.** An inspector who can see the reasoning can also see when it
is thin, and push back. An opaque score invites either blind compliance or blanket rejection,
and both are worse than an argued disagreement.

**Drift you would otherwise miss.** A model can hold its accuracy steady from quarter to
quarter while quietly changing which signals it leans on. Accuracy metrics are blind to that.
Component 11 measures it directly.

---

## 3. Global versus local explanation

Two different questions, and conflating them is a common mistake.

**Global** — *what does this model generally rely on?* One number per feature per model:
the average size of its contribution across many predictions. This is a property of the
model.

**Local** — *why did **this** establishment get **this** score?* One number per feature for
one prediction. This is a property of a single decision.

They can disagree, and the disagreement is informative. A feature can be globally unimportant
and decisive for one establishment — a first-time premises with no history at all, where the
one signal that exists carries all the weight.

Component 11 produces both, and keeps them in separate tables so neither can be mistaken for
the other.

---

## 4. What is a SHAP value, intuitively?

Imagine a team producing a result together, and you want to pay each member fairly for their
contribution. The fair-payout problem was solved in 1953 by Lloyd Shapley, in game theory,
and it has a unique answer: for each member, look at every possible order in which the team
could have been assembled, measure how much the result changed when that member joined, and
average over all the orders.

A SHAP value applies this to features. The "result" is the model's output for one
establishment. The "team members" are the features. The payout to each feature is its SHAP
value: **how much this feature moved this prediction, averaged fairly over every order in
which the features could have been revealed to the model.**

Three properties come out of that, and they are the reason SHAP is used rather than something
simpler:

- **It adds up.** `baseline + all the contributions = the model's actual output`. No
  unexplained remainder, no leftover.
- **It is signed.** A feature can push risk up or down, and you can see which.
- **It shares credit fairly between features that interact.** Simpler methods double-count.

The **baseline** is the model's expected output — roughly, what it would say knowing nothing
in particular about this establishment. Every contribution is measured relative to that.

---

## 5. Why does the explanation "scale" matter?

Because a contribution is a number *of something*, and the something must be stated.

Sentinel's models internally work in **log-odds** — an unbounded scale where 0 means
fifty-fifty, positive means more likely, negative means less. The final probability is a
squashed version of that.

Contributions add up in log-odds. **They do not add up in probability**, because the squashing
is not linear. If you insisted on probability-space attributions, you would have to either
abandon the "it adds up" property or invent numbers to make it appear to hold.

So every value in the artifact is in log-odds and every row says so, in a column called
`output_space`. The project deliberately does not offer a probability-space variant at all,
rather than offering one with a footnote — a caveat in a footnote is a caveat that travels
separately from the number.

---

## 6. Why is the calibrated probability a different thing from the explanation?

Component 9 added a calibration step. The model produces a raw score; a small two-parameter
correction then maps it to a probability that means what it says.

```text
model  ──►  base score 0.706  ──►  Platt correction  ──►  calibrated 0.664
  │
  └── this is what the attributions decompose
```

Component 11 explains the **left-hand side**. The correction is a separate map with two
parameters and no features of its own; attributing it would mean attributing a composition of
two models, and the numbers would answer a question nobody asked.

So both numbers are carried side by side, and a user sees both:

```text
Model score before calibration:  0.706
Calibrated risk probability:     0.664
```

The correction never changes the *ordering* — Component 9 verified every ranking metric moved
by exactly 0.00 — so the explanation of the ranking is unaffected by which number you show.

---

## 7. How can temporal leakage happen in an explanation?

This is the least obvious part of the component, and the part most likely to go wrong quietly.

Every SHAP value answers "how much did this feature move the output, **relative to what?**".
That "relative to what" is a set of reference rows, called the *background*, and it is part of
the explanation rather than a technicality behind it.

The tempting mistake: use the rows you are already explaining as the background. They are in
memory, they are the right shape, and every arithmetic check would still pass. But those rows
come from the period the model is being *judged* on — information the model did not have when
it produced the score. The explanation's reference point would encode the future.

Nothing would raise. The values would be finite, additive and plausible. That is exactly the
failure mode this project keeps finding in different disguises.

---

## 8. How is the background kept safe?

```text
   TRAIN ─────────────────────────►│ CALIBRATION │ TEST
   background drawn only here       │             │ rows explained here
                                 train_end
```

Three defences, and the third is the one that matters:

1. Reference rows come from the fold's **training window** and nowhere else, via
   `modeling.train.training_frame` — the same function every model fit calls, so there is one
   definition of "train" in the project rather than two.
2. The selection is uniform and seeded, so it is reproducible.
3. **A validation check re-derives every reference row's date from the feature table** and
   fails if any lands after `train_end`, and a *second* check asserts every reference row is
   genuinely a member of that fold's training split — because a date comparison is weaker than
   the split itself.

And the checks are tested by breaking them on purpose. One test poisons a single reference row
with a future date and asserts the check goes red; another hands a later fold's background to
an earlier fold and asserts it is rejected. A guard whose failure has never been observed is
indistinguishable from one that cannot fire.

Two of the four models need no background at all: the tree models take their reference from
structure recorded inside the trees at fit time, which is temporally safe by construction.

---

## 9. What actually drove the models?

See `docs/analysis/explainability_findings.md` for the measured tables — this document
deliberately does not restate numbers that live there, so the two cannot drift apart.

The shape of the answer: the models lean overwhelmingly on **prior compliance history** —
whether this establishment has been cited before, and how often — with scheduling-gap
features second. That is not a surprise, and "not a surprise" is itself a mild positive
result: a model whose top signal was something semantically unrelated to food safety would
have been a red flag.

What is worth knowing is the **spread**, not the mean. Every importance number is reported
with its fold-to-fold standard deviation and its best and worst rank, because a mean
importance quoted alone invites "feature X is the most important feature" — a claim that is
only true if the ranks hold.

---

## 10. Were the explanations stable over time?

Measured, per model, two ways — rank correlation across the whole ranking, and top-10 overlap
— across seventeen consecutive quarters. Numbers in the findings document.

The reason to measure it at all: **a model can hold its accuracy steady while changing its
reasoning.** Component 5 already showed that time invariance does not hold in this data — an
11.77 percentage-point seasonal swing — and Component 6 showed the *model ordering* inverting
under distribution shift. Explanation drift is a third, separate phenomenon, and it is not the
same as Component 9's calibration drift. A model can be perfectly calibrated and still be
reasoning differently than it was two years ago.

Whatever the data showed is what is reported. If the rankings move, the findings say they
move.

---

## 11. What changed during COVID?

The `covid_shift` fold is reported **separately and is never averaged into the quarterly
figures** — structurally, not by convention: they are different values of a `fold_set` column
and the aggregation is grouped by it.

That is not caution for its own sake. It is one fold, over a period when the scheduling policy
itself broke, and every component so far has found it diverging. Component 6 found the model
*ordering* inverting on it: the ablation that lost on the ordinary quarters won under shift,
because the feature it dropped partly encodes scheduling policy, and when the policy breaks a
model leaning on it is the more fragile one.

Averaging that fold into seventeen ordinary ones would move the headline and leave no trace of
having done so.

---

## 12. Which models could be explained, and why?

| model | method | exact? |
|---|---|---|
| logistic regression | closed-form formula | yes |
| XGBoost | the library's own exact TreeSHAP | yes |
| LightGBM | the library's own exact TreeSHAP | yes |
| neural network (numeric) | permutation sampling | **no** |
| `xgboost_chain_embeddings` | — | **unsupported** |

Each method was chosen because of what the model *is*, not to make the code uniform. A linear
model has an exact formula. A tree ensemble has an exact algorithm both libraries already
ship. Only the neural network has no exact answer, and only it gets an approximation.

**The unsupported one is the interesting case.** `xgboost_chain_embeddings` is structurally the
easiest model in the project to explain — a plain tree ensemble — but the fitted model object
can only be reached through a *private* helper inside Component 8, and Component 8 is closed.
The project rule is that you document a missing interface and propose the smallest public fix
rather than reaching around it. So the model is reported unsupported, with the measurement as
its stated reason, and the four-line public accessor that would lift the restriction is written
down in ADR 0031 and deliberately not added.

Its row carries **nulls, not zeros**. Zero is a legitimate attribution meaning "this feature
did not move the score", so a placeholder table of zeros would read as a model that used no
features rather than as a model nobody could explain.

---

## 13. What are the limitations of SHAP and of this component?

**Correlated features share credit, invisibly.** Component 7 measured a condition number of
71.8 with one feature pair correlated at 0.9888. When two features carry nearly the same
information, SHAP splits the credit between them, and no SHAP value tells you it did. Reading
"feature A contributed +0.4 and feature B +0.1" as "A matters four times as much" is wrong when
A and B are near-duplicates.

**The neural model's per-row values are approximate.** Measured: at the frozen budget the
median per-value error is about 1% of the largest attribution, while the *global ranking* is
stable to a rank correlation of 0.996. So the network's global table is quotable and its
individual values should not be quoted to three decimal places. Every such row is labelled
`is_exact = false`.

**Additivity is not accuracy, for the approximate method.** The permutation method's arithmetic
reconstructs the model output exactly no matter how few samples are drawn, so a green
additivity check proves the sums are sound and says nothing about whether the credit was split
correctly. This is stated in the code, in the contract and in an ADR, because it is precisely
the check a reader would over-interpret.

**Only a sample of predictions is explained** — 300 rows per model per fold, chosen uniformly
and seeded, out of 41,536. Enough for stable global rankings; not a census.

**One model of the five is unexplained**, and it is the one with the best PR-AUC.

**The explanation inherits every limitation of the thing it explains.** Sentinel observes
violations *cited*, not violations *committed*, and the dataset has no inspector field (ADR
0019), so the gap between "was cited" and "was unsafe" cannot be characterised anywhere in
this project. An attribution explains a model of citations.

---

## 14. What Component 11 does **not** prove

**It does not prove causality.** This is the single most important sentence in the document.

A SHAP value states: *how the model used a feature.*

It does not state: *changing that feature would change the outcome.*

Concretely: if "long gap since the last routine inspection" carries a large positive
attribution, that does **not** mean inspecting sooner would make an establishment safer. It
means the model learned that establishments with long gaps have historically been more likely
to be cited — which could be because gaps track neglect, or because the department already
schedules risky premises differently, or because both track something neither of them measures.

**It does not prove the model is good.** Attributions describe reasoning, not correctness.
A model can lean hard on a feature that is misleading it, and Component 6 measured exactly that
happening under distribution shift.

**It does not select a model.** Component 11 may report explanation quality, stability and
support status; it may not name a winner. That would be choosing a model on legibility.
Components 5 through 9 own the evaluation protocol, and "model selection" is recorded in this
component's blocked-experiments list so the boundary is written into every manifest it emits.

---

## The sentence to leave with

> Sentinel does not merely assign restaurants a risk score. For every supported model and
> every temporal fold, Component 11 exposes the evidence structure behind that score — which
> features contributed, in which direction, against a reference point that provably could not
> see the future — and measures whether the model's reliance on those signals holds over time.
>
> These are explanations of **model behaviour**, not causal explanations of food-safety
> violations.
