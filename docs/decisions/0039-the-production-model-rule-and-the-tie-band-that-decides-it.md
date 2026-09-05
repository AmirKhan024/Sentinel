# ADR 0039 — The production-model rule, and the tie band that decides it

**Status:** Accepted · **Date:** 2026-08-26

## Context

Nine components produced five calibrated models and no decision about which one Chicago should
carry. MEMORY open question 13 stayed open through Component 11 and Component 12, both of which
are recorded as blocked from model selection in every manifest they emit. HANDOFF §16f states
that a policy component will have to settle it, and forbids using any Component 12 number to do
so.

It lands here because a department cannot work four queues.

The question is genuinely hard, and Component 12 made it harder rather than easier. The model
with the best NDE, `neural_numeric_only`, has the second-worst calibrated ECE and reached the
fewest neighbourhoods with its calibration (17 of 33). `xgboost` has the best calibrated ECE and
the second-best NDE. `lightgbm` wins precision at one day of capacity. `logistic_regression` has
the narrowest across-neighbourhood ROC-AUC spread. Four axes, four winners.

## Decision

**A lexicographic rule over Components 5 and 9's own published artifacts, frozen in
`definitions.py`, applied by `select.py`, and emitted in full to `policy_model_selection` so
anyone who distrusts the answer can re-derive it.**

```text
candidates   calibrated models that Component 11 could explain and that are not experimental
             -> lightgbm_platt, logistic_regression_platt, neural_numeric_only_platt, xgboost_platt
axis 1       quarterly-mean NDE; two models are TIED when their Component 5 sensitivity
             intervals overlap
axis 2       quarterly-mean calibrated ECE, lower wins
axis 3       quarterly-mean precision@k_1_day, higher wins
axis 4       model name ascending -- so the rule always terminates
```

### The exclusion happens before any number is read

`xgboost_chain_embeddings_platt` has the best PR-AUC of any model in the project. It is refused
as a deployment candidate because ADR 0022 makes it experimental and ADR 0031 records that
Component 11 could not explain it. A model whose recommendations cannot be explained to the
inspector acting on them is not a deployment candidate, whatever it scores. The refusal is a row
in `policy_model_selection` with its reason, not a filter in a comprehension.

### Axis 1 separates nothing, and that is the finding

Component 5's `sensitivity` artifact perturbs the labels 1,000 times per fold and publishes each
model's NDE p05–p95 interval. Measured on the production run:

| model | NDE | sensitivity band | vs leader |
| --- | ---: | --- | --- |
| `neural_numeric_only_platt` | 0.2482 | [0.2311, 0.2527] | leader |
| `xgboost_platt` | 0.2376 | [0.2224, 0.2444] | overlaps |
| `lightgbm_platt` | 0.2355 | [0.2201, 0.2419] | overlaps |
| `logistic_regression_platt` | 0.2326 | [0.2160, 0.2374] | overlaps |

**All four bands overlap the leader's.** The headline operational metric of this entire project
cannot tell these four models apart, which corroborates Component 8's own conclusion that the
network's advantage is the size of its seed noise. The rule therefore falls to axis 2 and selects
**`xgboost_platt`** on a calibrated ECE of 0.0474. Axis 3 and the name terminator are never
reached.

### Calibration is axis 2 because a policy layer publishes probabilities to people

Discovery efficiency first, because ranking under capacity is the operational problem.
Calibration second, because this component hands a probability to a human reviewer beside a
recommendation, and a probability that does not mean what it says is worse than no probability.
Precision at one day of capacity third, because it is the most concrete operational number. The
name last, so two identical models still produce one answer.

### The tie rule was fixed after its inputs were first read, and that is recorded

This is the uncomfortable part and it is stated rather than smoothed over.

The implementation plan carried a placeholder band: Component 8's five-seed **ROC-AUC** spread of
0.0058. Thresholding an **NDE** difference with a ROC-AUC spread is a unit error, and it was
noticed after the NDE column had already been read. It was replaced with Component 5's own NDE
sensitivity interval — the right quantity, on the right metric, from an artifact that already
existed.

**The two rules select different models.** Under the discarded 0.0058 band,
`neural_numeric_only_platt` separates on axis 1 and is selected. Under band overlap the four tie
and `xgboost_platt` is selected on calibration.

Because the tie rule decides the deployment, both outcomes are emitted on every run:
`policy_model_selection` carries `selected_under_discarded_band` alongside `is_selected`, and the
manifest carries `discarded_tie_band` and `selected_model_under_discarded_band`. A rule chosen
after seeing what it decides is defensible only when the choosing is visible.

Band overlap is preferred for two reasons that do not depend on which model it picks. It compares
NDE against an NDE-derived interval rather than against a spread measured on a different metric.
And interval overlap is the method `baseline_models_findings.md` already used to decide whether
two NDE numbers differ — so the rule is this repository's existing precedent rather than a new
threshold invented for this component.

### The result is an operating choice, not a finding

Written into every manifest as `production_model_claim`, in these words: *an operating choice of
the policy layer, applied from a rule fixed before it was run and recorded with its inputs.
Revisable. NOT a finding that this model is the best one.*

### The shift fold is reported beside the rule and never averaged into it

Component 7 measured that the `covid_shift` fold orders these models differently — `lightgbm`
takes NDE there, and `logistic_regression` posts the highest PR-AUC of any model in the project.
Pooling one 18-month episode with seventeen quarters would let a single unusual period outvote
four years of ordinary ones. It is carried as a named limitation on the choice, and it is the
main reason the choice is described as revisable.

## Alternatives rejected

**Decline to select, and emit four queues.** The most honest-looking option and the one that
fails the component's purpose: Sentinel would still be unable to answer "which K establishments
do you recommend?" with a single answer, and open question 13 would move to Component 14
unchanged. The comparison *is* emitted for all four models, so the evidence survives; only the
queue is singular.

**Select on the shift fold, since selection on rolling folds picks the wrong model under
shift.** Component 7's finding is real and this is the strongest alternative. Rejected because
one held-out shift episode is one observation. A selection rule carried by a single unusual
period is not more robust than one carried by seventeen ordinary ones; it is differently fragile.

**Use Component 12's group-calibration coverage as an axis.** `lightgbm` and `xgboost` reach 25
of 33 community areas and `neural_numeric_only` reaches 17, which is a real difference and would
be a defensible tiebreaker. Forbidden by HANDOFF §16f, and rightly: Component 12 is recorded as
blocked from model selection in every manifest it emits, and reaching for its numbers here would
retroactively make it a selection component.

**Keep the 0.0058 band and not mention the change.** Rejected. The band decides which model a
city deploys, and a rule whose provenance is hidden is a rule nobody can audit.

**Combine the axes into a weighted score.** Rejected for the reason the frontier is not collapsed
either: a weighting is an exchange rate between discovery and calibration, and nothing in this
project measures one.

## Consequences

- `SELECTION_AXES`, `SELECTION_TIE_RULE`, `SELECTION_FOLD_SET` and `DISCARDED_TIE_BAND` are
  frozen in `definitions.py`; a registry guard refuses a rule whose last axis is not the model
  name, so it cannot fail to terminate.
- `policy_model_selection` holds one row per registered model — admissible and refused — with
  every axis value, the sensitivity band, the tie flag, both outcomes, and which axis decided.
- The manifest records `selected_model`, `selection_decided_on_axis`,
  `selected_model_under_discarded_band` and `production_model_claim`.
- `--model NAME` overrides the rule as a diagnostic, and refuses an inadmissible model by name.
  Following ADR 0025's `--method` precedent: the override exists, and it is recorded.
- `tests/test_policy_select.py` asserts the rule terminates, falls through each axis in order,
  refuses a candidate with a missing measurement, and never pools the shift fold.
- **MEMORY open question 13 is closed as a policy decision, not as a scientific one.** The
  measurement it asked for — which model is best — remains unanswered, and this ADR records that
  the four candidates are statistically indistinguishable on the metric the question was about.
