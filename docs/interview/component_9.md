# Component 9 — interview defence

Companion to `component_7.md` and `component_8.md`. Same structure: short answers first, then
the technical detail, then the questions that actually get asked. Every number here is measured;
where something is uncertain it says so.

## 60-second answer

Components 6–8 built a **ranking** — which restaurants to inspect first. Component 9 asks a
different question: **when Sentinel says 0.30, does it happen 30% of the time?**

It did not. Every one of the five candidate models was **underconfident**: the calibration slope
sat at 0.61–0.79 where 1.0 is perfect, meaning they all hedged toward the base rate. Platt
scaling, fitted on a window that no earlier component had ever touched, pulls the slope to
**1.00–1.03** and cuts ECE by 20–25%.

The important part is what did **not** change. PR-AUC, ROC-AUC, NDE and precision@k are
**identical to the last bit** — delta exactly 0.00e+00, verified by re-running Component 5's
evaluator on the calibrated artifact. That is the success case, not a null result: a monotone
map cannot reorder anything, and the Brier decomposition confirms it — reliability fell 16–46%
while **resolution was unchanged to five decimal places**.

## 2-minute answer

**The blocker first, because it shaped the component.** A calibrator is fitted on a base model's
scores over the calibration window. Those scores did not exist — every prediction artifact
covered exactly the test window — and no fitted model was persisted anywhere. So Component 9
re-executes Components 6–8's *unchanged* fit functions, and proves they are the same models by
re-deriving the test window and comparing it to the committed artifact **bit for bit**: 207,680
rows, zero mismatches. The build refuses to fit any calibrator if that gate fails.

**The protocol.** Both Platt and isotonic are fitted for every fold. The choice between them is
made on an inner split of the calibration window, using mean log-loss over an *expanding prefix*
of folds — never a pool over all folds, because **fold N's calibration window is fold N−1's test
window**, so pooling would choose fold 1's method using fold 1's test period. The tie rule
(0.005 nats, prefer Platt) was frozen in ADR 0025 with a git date, from a bootstrap noise
measurement, before any test window was opened.

**The result.** Platt won all 90 (model, fold) cells. Isotonic was not close: its log-loss was
*worse than the uncalibrated model's* on four of five candidates, because pool-adjacent-violators
on a ~1,200-row window produces plateaus at exactly 0 and 1, and a wrong plateau costs −log(ε).
Per fold in isolation isotonic would have won 16 of 90 — which is exactly what the expanding
prefix exists to smooth.

**The honest caveats.** `covid_shift` calibrates to a slope of only 0.75–0.90 and roughly double
the ECE, because its base rate moved 17 points between the calibration and test windows and no
monotone recalibration can fix prior shift. Calibration made ECE *worse* on 16 of 90 cells. And
the model ordering by ECE inverted: `neural_numeric_only` had the best uncalibrated ECE and now
has the second worst, simply because it started closest to calibrated.

## Deep technical answer

### The estimand has not changed

Still: re-ordering inspections that actually occurred, under the real daily capacity. Component 9
changes the *number attached to* each position, never the position.

### What was actually fitted

Per (model, fold): a two-parameter logistic regression of the label on `logit(p)`, unpenalised
(`C=1e10`), fitted on the fold's full calibration window — 1,357 to 2,459 rows. 90 frozen
calibrators, 180 fitted in total because the losing method is persisted too.

### The calibrator's input

No component persists a logit. Component 9 calibrates the logit **recovered** from the committed
probability, `log(p) − log1p(−p)`, not the model's native decision margin — even though the
margin is reachable through public attributes. The reason is provenance: the calibrated artifact
is then a pure function of the already-committed `score` column plus two floats, reproducible
years later from artifacts alone.

The cross-check produced the run's most interesting surprise. For the two float64 models the
recovery is exact to 1e-13. For `xgboost` and the network it is not — up to **2.6e-5** — and the
cause is not the arithmetic: **those models compute in float32.** A float32 sigmoid cannot be
inverted to more than float32 precision. The persisted probability is the float64 image of a
float32 computation, so `logit(score)` is the best recovery that exists.

That finding also invalidated a threshold I had pre-declared at 1e-9, which would have fired on
33,898 correct rows. It is now 1e-4, set from the measurement.

### The temporal protocol, precisely

```
train ──────────────────────────►│ calibration │ test
      base model fitted here      │             │
                                  │ inner-fit → inner-select (method chosen here)
                                  │ full window → calibrator refitted, frozen
                                                │ evaluated once, never fitted on
```

`trained_through` on the calibrated artifact is `calibration_end`, **later** than the base
model's `train_end`. That is honest rather than convenient: the calibrator really did read the
calibration window. It sits exactly at the contract's ceiling —
`evaluation.contract._training_horizon` returns `calibration_end` — so Component 5 accepts it
with no change. Three separate provenance columns record the estimator's horizon, the
calibrator's horizon and the first operationally available date.

### Determinism

Isotonic is exactly order-invariant. Platt is not: shuffling the calibration rows moves an applied
probability by **one ULP** (2.2e-16), the same lbfgs summation-order sensitivity Component 6
documented for its own coefficients. Moot in production because the window is canonically sorted.

The bit-identity gate is sensitive to the same thing, and it caught it: a first run under
`OMP_NUM_THREADS=1` failed on 32,696 of 41,536 `logistic_regression` rows, by 1e-13 to 5e-10.
Nothing was wrong with the model — the committed run used the library default thread count, and a
different count is a different summation order. That failure is the best evidence the gate works.

### The result, stated carefully

Quarterly mean over 17 folds:

| model | ECE before → after | MCE before → after | Brier before → after | slope before → after |
|---|---|---|---|---|
| `xgboost` | 0.0621 → **0.0474** | 0.1741 → 0.1150 | 0.2379 → 0.2350 | 0.640 → 1.005 |
| `lightgbm` | 0.0644 → **0.0490** | 0.1755 → 0.1260 | 0.2383 → 0.2351 | 0.618 → 1.015 |
| `logistic_regression` | 0.0635 → **0.0518** | 0.1664 → 0.1297 | 0.2382 → 0.2358 | 0.611 → 1.015 |
| `neural_numeric_only` | 0.0563 → **0.0524** | 0.1444 → 0.1201 | 0.2355 → 0.2347 | 0.791 → 1.003 |
| `xgboost_chain_embeddings` ⚠ | 0.0619 → **0.0481** | 0.1767 → 0.1236 | 0.2374 → 0.2346 | 0.651 → 1.029 |

Ranking, all five models, verified through Component 5: **every delta exactly 0.00e+00.**

---

## "Why did you choose this?"

**Why Platt and isotonic, and not temperature scaling?** Temperature scaling is Platt with the
intercept fixed at zero — a strict special case. Fitting Platt answers what a temperature would
have been for free, and the fitted intercepts (−0.000 to +0.011 after calibration) say it would
have been nearly as good. It is recorded as available-if-wanted rather than run.

**Why log-loss to select, when ECE is the thing you are improving?** Three reasons, all fixed in
advance. Fifteen equal-mass bins over a ~500-row inner-select window is 27–50 rows per bin, of
which 11–21 are positives — the noise is the size of the effect. ECE is not a proper scoring
rule: a calibrator that reshuffles within a bin lowers it without being better, and a degenerate
calibrator predicting the base rate everywhere scores near zero. And ECE's bin count is a free
parameter, so a rule based on it can be tuned until it gives the answer you want. Log-loss has
none of those properties, and it punishes exactly the failure isotonic exhibits here.

**Why an expanding prefix rather than per-fold selection?** Per-fold, isotonic would have won 16
of 90 cells and the method would have flipped 16 times mid-series — making the drift plot
unreadable, because an ECE step could be drift or could be the switch. The prefix uses folds
1…k, every one of which ends at or before fold k's calibration end, so it is horizon-legal by the
contract the project already enforces.

**Why 0.005 for the tie threshold?** It is one median paired-gap SD, measured by resampling each
inner-select window 1,000 times and scoring both calibrators on the *same* resample so their
shared variation cancels. Min 0.0022, median 0.0054, max 0.1595 over 72 cells. My implementation
plan had proposed 0.002 — which is below the smallest observed SD, so it would have declared
winners on differences finer than the noise of the comparison. The profiling caught it.

## "Why didn't you choose X instead?"

**Why not just use isotonic? It is more flexible and usually wins.** Because it lost, badly, and
for a legible reason. Its inner-select log-loss reached **1.1028** against Platt's 0.6850 on the
same cell; its test log-loss is worse than the *uncalibrated* model's on four of five candidates;
and its post-calibration slope collapses to 0.42–0.58, meaning it introduces underconfidence
rather than removing it. Pool-adjacent-violators on ~1,200 rows produces plateaus at exactly 0
and 1, and one wrong plateau costs −log(ε).

**Why not pick whichever method has the lower test ECE?** That is test-set selection. The
reported metric would then be the maximum of two draws rather than an estimate of either. The
evidence that I did not do it is in the results: on `covid_shift`, isotonic beat Platt on ECE for
`logistic_regression` (0.0861 against 0.0973), and Platt was still frozen — because the rule was
decided on calibration-window evidence before that number existed.

**Why not fit the calibrator on the training window?** The model is fitted on those rows, so its
scores there are optimistic in exactly the way a calibrator would bake in. ADR 0012 rejected this
in advance.

**Why not refit the base models on train + calibration to use more data?** That is a *different
model* from the one Components 6–8 evaluated, so no Component 9 number would be comparable with
any earlier one — and it destroys the held-out property of the window the calibrator needs.

**Why not persist the models retroactively so you did not have to re-fit?** It would modify three
closed components and rewrite three committed artifacts to serve a fourth. It would also make
Component 9's correctness depend on a change to the code it is calibrating. The bit-identity gate
is stronger evidence anyway: a pickle proves a model was saved; a bit-identical re-derivation
proves the whole pipeline is reproducible.

## "What went wrong?"

**Three thresholds I declared before measuring were wrong, and the measurements caught all
three.** The tie threshold (0.002 → 0.005), the logit-recovery tolerance (1e-9 → 1e-4, because
two base models compute in float32), and the Platt self-check tolerance (1e-6 → 1e-3, because
`C=1e10` is large but not infinite). Each is recorded with the number that forced it. The general
lesson: a threshold set from expectation rather than measurement is a guess wearing a decimal
point.

**The bit-identity gate failed on the first run**, on 32,696 rows, because I had set
`OMP_NUM_THREADS=1` and the committed run had not. Component 7's lesson — *when a leakage test
fails, suspect the test first* — generalised: when a determinism gate fails, suspect the
environment first.

**The bootstrap was 100× too slow as first written.** It refits sklearn 1,000 times per cell for
the calibration slope, which is ~10⁶ fits across the run. Fixed by dropping the slope from the
bootstrap (its point estimate is still reported everywhere) and vectorising the other three
metrics — a numpy twin verified against Component 5's implementations, agreeing to ~1 ULP.

**ADR 0020's prediction was wrong.** It expected the neural network to be overconfident. Every
model was *under*confident, and the network was the least miscalibrated of the five. The data
wins and the divergence is recorded.

## "What would you improve?"

1. **Persist the fitted estimators.** The whole re-execution exists because nobody did. It is a
   25-minute cost on every reproduction.
2. **Average the neural model over seeds before calibrating.** Its advantage (0.0053 ROC-AUC) is
   smaller than its own seed spread (0.0058), so the headline is partly seed luck. Deliberately
   deferred here because it would create a base model Component 8 never evaluated and break the
   bit-identity gate.
3. **Handle prior shift explicitly.** On `covid_shift` most of the residual error is a 17-point
   base-rate move that a monotone recalibration structurally cannot fix. A prior-correction term
   would.
4. **Validate the retraining trigger.** The thresholds proposed are a design proposal, chosen to
   sit outside the observed operating range. Nothing downstream consumes them yet.

---

## The questions that actually get asked

**What is calibration?** Making the number mean what it says. A model that outputs 0.30 for a
thousand restaurants is calibrated if about 300 of them turn out to have a Priority violation.

**Why isn't accuracy enough?** Accuracy needs a threshold, and Sentinel has none — every
establishment is inspected eventually, so there is nothing to accept or reject. It also says
nothing about the number: a model can be 70% accurate while every probability it emits is wrong.

**Why isn't ROC-AUC enough?** ROC-AUC only sees the *order*. Multiply every probability by 0.5
and ROC-AUC is unchanged while every probability is now half what it should be. That is exactly
the failure this component found: ROC-AUC 0.62 with a calibration slope of 0.64.

**What does a predicted 0.70 mean?** After Component 9: of the establishments Sentinel scores
0.70, about 70% are cited. Before it, that number was closer to 0.60 — the models hedged.

**Why does Sentinel need calibrated probabilities?** Because the components after it consume the
number, not the order. A cost threshold ("inspect if risk × cost of a missed violation exceeds
the cost of an inspection") is arithmetic on a probability. A deferral gate ("send to human
review if the model is uncertain") is a statement about a probability. Both are meaningless if
0.30 does not mean 30%.

**How is that different from ranking?** Ranking answers *who first*; calibration answers *how
likely*. They are independent: a perfectly ordered model can be badly calibrated, and a perfectly
calibrated model can rank at chance. Sentinel needed both, and had only the first.

**What is Platt scaling?** Simply: fit a small correction curve that maps the model's claimed
probabilities to what actually happened, using held-out data, and it has only two knobs — how
steep and how shifted. Technically: a one-variable logistic regression of the outcome on the
model's logit, `P(y=1) = sigmoid(a·logit(p) + b)`, fitted by maximum likelihood.

**What is isotonic regression?** Simply: sort the predictions, then draw the best staircase you
can that never goes down. Technically: the pool-adjacent-violators algorithm finds the
least-squares fit subject only to being non-decreasing — no functional form at all.

**Why did you test both?** They carry different assumptions and fail differently. Platt assumes
the miscalibration is a smooth sigmoidal distortion; isotonic assumes nothing but monotonicity.
Implementing only the one that won would have made "Platt is better here" an assertion rather
than a measurement.

**Which is more flexible?** Isotonic, by a lot — up to ~2,300 breakpoints against Platt's two
parameters.

**Which is more likely to overfit?** Isotonic, and it did: fitted on ~1,200 rows it produced
plateaus at exactly 0 and 1, and its log-loss on held-out calibration data reached 1.10 against
Platt's 0.69.

**How did you choose between them without test leakage?** An inner chronological split of the
calibration window: fit both on the earlier 70%, compare on the later 30%, choose on the mean
over an expanding prefix of folds, refit the winner on the full window, freeze, then apply to
test. The rule and its threshold were committed to git before the first production run.

**What exactly is the calibration window?** One quarter sitting between training and test in
every fold. It was built in Component 5, before any model existed, precisely so Component 9 would
have nowhere else to fit a calibrator — Components 6, 7 and 8 each write a column literally named
`calibration_end_unused` to prove they did not touch it.

**Why can't you calibrate on the test set?** The reported probabilities would be self-fulfilling.
You would be measuring how well a correction fits the data it was fitted on, and reporting it as
how well it will work on data it has not seen.

**How did you prevent temporal leakage?** Structurally, not by discipline. `FoldSpec` cannot be
constructed with overlapping windows. The calibration frame is a filter over the same
`assign_split` every other component uses, so there is one definition of every window. The
selection uses only folds ending at or before the current one. And 21 runtime checks re-derive
these properties from the data rather than trusting what the orchestrator reported.

**What is Brier score?** The mean squared error of a probability: average of (predicted −
outcome)². 0 is perfect, 0.25 is a coin flip. It rewards being both well-ordered and
well-calibrated, which is why it is reported beside ECE rather than instead of it.

**What is ECE?** Expected calibration error: sort the predictions, cut them into 15 equal-sized
bins, and in each bin compare what the model claimed on average to what actually happened. ECE is
the mass-weighted mean of those gaps. An ECE of 0.05 means the model's claim is off by about 5
percentage points on average. Example: a bin whose mean prediction is 0.40 but where 33% were
cited contributes a gap of 0.07.

**Why equal-mass bins?** Because the score distribution is concentrated — `lightgbm` never says
less than 0.045 or more than 0.951 — so most equal-*width* bins would be empty, and an empty bin
contributes nothing while looking perfectly calibrated. Equal-mass cuts by position, so every bin
carries real weight. The spec asked for 15 equal-mass bins and Component 5 implements exactly
that; Component 9 imports it rather than writing a second version.

**What is MCE, and how does it differ from ECE?** MCE is the *worst* bin, unweighted. ECE is the
average; MCE is the tail. A model can have a good ECE and a bad MCE if it is well behaved
everywhere except one region — which matters operationally, because that region might be exactly
the high-risk end. Here both improved: ECE by 20–25%, MCE by 17–34%.

**What does a reliability diagram show?** Predicted probability on the x-axis, observed frequency
on the y-axis, with the 45° line as perfect. A curve *flatter* than the diagonal means
underconfidence (the model hedges); *steeper* means overconfidence. Our before-panels are visibly
flatter; the after-panels sit on the line.

**What is Brier decomposition?** Brier = reliability − resolution + uncertainty.

- **Reliability** — are the claims true? How far each bin's prediction sits from its outcome
  rate. Lower is better, and this is what calibration fixes: it fell 16–46%.
- **Resolution** — does the model separate anything? How far the bins' outcome rates sit from the
  overall base rate. Higher is better, and a monotone calibrator cannot create it — measured
  **unchanged to five decimal places**.
- **Uncertainty** — how hard is the problem? The Brier score of always predicting the base rate.
  A property of the data; identical (0.24362) for every model and stage.

We also report the within-bin residual rather than hiding it: the three-term identity is exact
only for a forecast constant within each bin, and ours are continuous.

**Can calibration improve ROC-AUC or PR-AUC?** Not with a strictly monotone map — measured delta
exactly 0.00e+00 for both. Isotonic can move ROC-AUC slightly (up to 0.0127 here) because its
plateaus create ties, and ties count as half a concordant pair.

**Can calibration change ranking?** Platt cannot: 0 inversions, Spearman ρ exactly 1.0. Isotonic
did not *invert* anything either, but it tied ~40,000 pairs, and because top-k ties are broken by
`target_inspection_id`, that moved top-k membership 226–265 times and precision@k by up to 0.21.
A tie is not an inversion, and reporting it as one would misdescribe a correct calibrator.

**What happened to NDE after calibration?** Nothing, exactly. 0.248212 → 0.248212 for the
network. That is the expected and correct result.

**What happened to COVID calibration?** It helped — ECE fell 10–23% on every model — but the
level stayed roughly double the quarterly level and the slope only reached 0.75–0.90 instead of
1.00. The residual is prior shift: the base rate fell from 0.683 in its calibration window to
0.513 in test, and no monotone recalibration fitted on the earlier window can correct a 17-point
move in the prior.

**What is calibration drift?** A calibrator learns the score-to-probability relationship on one
quarter and is applied to the next. When the environment moves — inspection priorities,
establishment mix, base rate — that relationship moves too. Measured here: the post-calibration
slope wanders between 0.77 and 1.64 across 17 quarters.

**When would you retrain?** Proposed: when quarterly ECE exceeds 0.075, or the slope leaves
[0.80, 1.25], in two consecutive quarters. Two quarters because a single quarter's ECE moves
±0.01 on sampling noise alone. **This is a design proposal, not a validated threshold** — it was
written after seeing the drift series, and I say so rather than dressing it up.

**Why not simply recalibrate every quarter?** Operationally you might. But you cannot *evaluate*
that way: refitting on a test quarter to measure that quarter is the self-fulfilling loop the
whole temporal structure exists to prevent. In production the honest version is to refit on the
most recent *closed* window and accept that it is always one step behind.

**Which model should Sentinel use after C9?** On the evidence, **`neural_numeric_only_platt`** —
but the honest answer is that the evidence is weak, and I would say so in the room. See "Final
decision" below.

**Does calibration make the model safer?** It makes the model's *claims* honest, which is a
precondition for safe use, not safety itself. A well-calibrated model can still be
systematically wrong about a subgroup while being right on average.

**Does calibration solve fairness?** No. A single global calibrator can be perfect overall and
badly miscalibrated within a neighbourhood or a facility type. Sentinel has not measured that —
it is Component 12's, and Component 8 built the community-area ablation for exactly that purpose.

**Does calibration solve inspector bias?** No, and it cannot. The dataset has 22 columns and no
inspector field at all (ADR 0019). Whatever inspector-to-inspector variation exists is inside the
labels, and a calibrator fitted on those labels inherits it.

**What would you do if calibration got worse over time?** First check whether it is drift or
prior shift — they look alike and have different fixes. Prior shift needs a base-rate correction;
genuine drift needs a refit on a more recent window. Then check whether the *base model* has
drifted too, because a calibrator cannot rescue a model whose ordering has decayed.

**Why not just use isotonic because it gets the lowest ECE?** On the quarterly folds it does not
— Platt is lower on three of five, and isotonic's log-loss is worse than the uncalibrated
model's on four of five. But the deeper answer is that choosing on a test metric is the thing
that makes the test metric meaningless.

**How would you explain calibration to a city official?** "Before, when our system said a
restaurant had a 70% chance of a serious violation, only about 60% of them did — the system was
being cautious in a way that made the numbers hard to plan with. Now 70% means about 70%. We
haven't changed which restaurants we inspect first, or in what order. We've changed what the
number on the screen means, so you can use it in a budget."

---

## "Explain Component 9 in 30 seconds"

Our models ranked restaurants well but their probabilities were wrong in a specific way — all
five were underconfident, saying 0.60 when they meant 0.70. We fitted a two-parameter correction
on a quarter of data that sat between training and testing and had never been touched by any
earlier component. Calibration error dropped 20–25%, and the ranking is bit-for-bit identical,
which is exactly what should happen.

## "Explain Component 9 to a non-ML interviewer"

Imagine a weather forecaster who is very good at telling you which days are rainier than which,
but whose numbers are systematically timid: on the days they say "60% chance", it actually rains
70% of the time. The *order* of the days is perfect; the *numbers* are not.

That was our system. It could tell you restaurant A was riskier than restaurant B, but the
percentages were pulled toward the middle.

Component 9 fixes the numbers without touching the order. We took a slice of history the models
had never seen — deliberately set aside two components ago — compared what they claimed to what
actually happened, and learned a small correction curve. Apply that curve, and the percentages
line up with reality.

We checked carefully that the order really is unchanged, because that is the property the
inspection schedule depends on. It is identical, to the last decimal place. And we checked that
the correction was not learned from the data we test on, because that would be marking your own
homework.

## "Explain Component 9 to a senior ML engineer"

Per-fold Platt and isotonic on a held-out calibration quarter, `TRAIN → CAL → TEST`, expanding
origin, 17 quarterly folds plus a COVID distribution-shift fold.

Three things worth your attention.

**The blocker.** No component persisted a fitted estimator and none had ever scored a calibration
window, so the calibrator inputs did not exist. Rather than retro-fitting model persistence into
three closed components, C9 re-executes their unchanged fit functions and gates on bit-identity
of the re-derived test window against the committed artifact — 207,680 rows, `==` not `isclose`,
build raises before fitting anything if it fails. That gate caught a BLAS thread-count difference
at 1e-13 magnitude, which is a decent proxy for its sensitivity.

**The selection.** Not per-fold, and not pooled. Pooled is leakage that looks clean: fold N's
calibration window *is* fold N−1's test window, so a global choice reads test data through a
window that is nominally calibration. Expanding prefix over folds 1…k keeps everything at or
before `fold_k.calibration_end`, which is the horizon `evaluation.contract` already enforces.
Selection metric is log-loss, not ECE — ECE at 15 equal-mass bins over ~500 inner-select rows is
27–50 rows/bin, it is improper, and its bin count is a tunable free parameter. Tie threshold
0.005 nats, set at one median paired-bootstrap-gap SD, frozen in an ADR with a git date.

**The results are boring in the right way.** Slopes 0.61–0.79 → 1.00–1.03. Reliability −16 to
−46%, resolution unchanged to 5dp, uncertainty invariant — so the entire Brier gain is the term
theory says it should be. Ranking metrics delta exactly 0.0 through C5's own evaluator. Isotonic
loses everywhere and legibly: PAVA on ~1,200 rows saturates at 0/1, log-loss 1.10 vs 0.69 in the
worst cell, post-hoc slope 0.42–0.58.

The parts I would push on if I were you: `covid_shift` reaches slope 0.87 and 2× ECE, which is
prior shift a monotone map structurally cannot correct; the improvement is a 17-fold-mean effect,
not visible in a single quarter's bootstrap CI; and the neural model was calibrated at seed 42
only, while its advantage over XGBoost is smaller than its own seed spread.

---

## Five-line memory cheat sheet

1. **The question:** can the probability be trusted, not can the ranking be improved. Answer:
   slope **0.61–0.79 → 1.00–1.03**, ECE **−20 to −25%**, MCE **−17 to −34%**.
2. **The proof it changed nothing else:** PR-AUC, ROC-AUC, NDE, precision@k all delta **exactly
   0.00e+00** through Component 5's own evaluator; Brier decomposition shows **resolution
   unchanged to 5dp** while reliability fell 16–46%.
3. **The blocker:** no calibration-window scores and no persisted models existed. C9 re-executes
   C6–C8 unchanged behind a **bit-identity gate — 207,680 rows, zero mismatches** (ADR 0026).
4. **The protocol:** Platt vs isotonic on an inner split, **expanding prefix** (pooling folds
   leaks — fold N's cal window is fold N−1's test window), tie rule **0.005 nats, prefer Platt**,
   frozen in ADR 0025 before any test window opened. **Platt won all 90 cells; isotonic would
   have won 16 per-fold.**
5. **The honest bits:** `covid_shift` only reaches slope **0.87 at 2× ECE** (prior shift, not
   fixable by a monotone map); calibration made ECE **worse on 16 of 90 cells**; the model
   ordering by ECE **inverted** (the neural net went from best to second-worst); and three
   thresholds I pre-declared were wrong and the measurements corrected them.
