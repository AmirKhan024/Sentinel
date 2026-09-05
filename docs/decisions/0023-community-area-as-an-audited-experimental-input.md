# ADR 0023 — Community area is an audited experimental input, and a better score does not retain it

**Status:** Accepted · **Date:** 2026-08-18

## Context

The project specification permits community area in the neural-network embedding experiment, and
attaches a condition to the permission: demographic variables remain excluded from the model, and
community area must be audited as a possible demographic proxy.

Chicago's 77 community areas are the city's standard unit for publishing demographic statistics.
They are strongly associated with race and income by construction — that is what they are used
for. A model conditioning on community area is therefore conditioning on something a
demographic variable would also explain, whether or not any demographic variable is present.

Sentinel's position on this is already recorded in two places and neither is neutral.

ADR 0019, refusing community area as an inspector proxy:

> ward and community area are route proxies fully confounded with establishment composition

and its consequence, which Component 12 inherits:

> disparate citation rates across geography and time … cannot be separated from disparate
> treatment. That is a material limitation on what a fairness audit can conclude here.

Component 4's tests assert that no model-derived or demographic feature exists in the feature
table. Component 8 does not change that: no race, income, or ACS variable is used anywhere in this
component. The question is narrower — whether a *geographic* identifier, which is a legitimate
operational fact about an establishment, may be embedded when it is known to correlate with
protected characteristics.

## Decision

**Community area is included as an explicitly labelled experimental embedding, with a matched
ablation, and a better score with it is not grounds for retaining it.**

### It is experimental, not production

It enters through Component 8's own categorical layer (ADR 0022), not through Component 4's
feature table. `feature_definition_version` stays `v1`, and Components 6 and 7 remain unaware of
it. Nothing about this decision promotes community area into the production feature set, and doing
so later requires a Component 4 release and a Component 12 finding, in that order.

### The ablation is a registered model, not a post-hoc adjustment

`neural_no_community_area` is a first-class entry in `NEURAL_REGISTRY`, identical to
`neural_embeddings` in every respect except that it drops the community-area family. Both are
fitted on every fold, both are scored on the same rows, and both appear in the results table.

It is a separate *fit* rather than a column zeroed at scoring time for the reason Component 6
recorded when it built `logistic_regression_no_scheduling` the same way: refitting is the only way
to see what a model does when it cannot lean on an input, whereas dropping a column at scoring
time measures a mis-specified model.

### The decision rule is declared in advance

**If the community-area embedding improves the score, that is not a reason to keep it.** It is a
finding to hand to Component 12, which owns the fairness audit and the question of whether the
improvement is (a) genuine geographic risk signal, (b) a proxy for the demographics of a
neighbourhood, or (c) a proxy for *inspection practice* in a neighbourhood — the third being
indistinguishable from the first two in this data, per ADR 0019.

Declaring this before the numbers were measured is the point. A rule adopted after seeing that the
input helped would be a rationalisation.

### What this component may and may not conclude

**May:** how much predictive signal the community-area embedding carries, measured as the gap
between two otherwise-identical models under the same temporal evaluation.

**May not:** whether including it is fair, whether the signal is causal, or whether it should
ship. Component 8 has no demographic data to correlate against, no per-area outcome decomposition,
and no disparate-impact metric. Producing any of those here would be Component 12's work done
badly and early.

### No demographic variable is introduced

No race, income, ACS, census or deprivation variable is used anywhere in Component 8. The
`BLOCKED_EXPERIMENTS` list in `neural/build.py` records this in every manifest, so the claim
travels with the artifact rather than living only in this document.

## Alternatives rejected

**Exclude community area entirely.** The safest option, and rejected because it would leave the
question unanswered. A future component asking "would geography have helped?" would have to build
this experiment anyway, and it is cheaper and more honest to measure it now under a declared
non-retention rule than to leave it as an open assumption.

**Include it without an ablation.** Rejected outright: the specification requires the comparison,
and without it the component could report that its best model uses community area while being
unable to say what that cost or bought.

**Substitute ZIP as a "less demographic" geography.** Rejected as false comfort. Chicago ZIPs are
also strongly associated with race and income; ZIP is included as its own family with its own
ablation, and neither is treated as a neutral geography.

**Substitute latitude/longitude, or a distance-to-city-centre feature.** Rejected: a continuous
geography is not less of a proxy, only less legible — and a less legible proxy is worse for an
audit, not better.

**Adopt a fairness metric now** (demographic parity across community areas, say). Rejected. That
is Component 12, it needs data this project has not ingested, and ADR 0019 already records that
the inspector-effect gap limits what any such metric could conclude here.

**Decide retention on the measured result.** Rejected — this is the alternative the ADR exists to
foreclose.

## Consequences

- Component 8 reports two headline embedding models: with community area and without. Neither is
  described as "the" model.
- Whatever the measured gap turns out to be, the recommendation carried forward is unchanged:
  **community area is not promoted to the production feature set by this component.**
- Component 12 inherits a quantified, reproducible starting point — a fitted pair of models
  differing in exactly one input, over 18 temporal folds — instead of an assumption.
- The `zip` family carries the same caveat and the same ablation, and should be read the same way.
- If Component 12 clears community area, promoting it is a Component 4 change behind a bumped
  `feature_definition_version`, followed by re-running Components 6, 7 and 8. That sequence is the
  cost of having kept it experimental, and it is the right cost.
