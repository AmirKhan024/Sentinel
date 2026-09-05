# ADR 0035 — What Component 12 does not claim

**Status:** Accepted · **Date:** 2026-08-25

## Context

Every other ADR in this repository records a choice between implementable options. This one
records a boundary, because Component 12 is the first component whose output carries a word —
*fairness* — that means considerably more in public than what the code measures.

The risk is specific and it is not hypothetical. A table titled `fairness_group_metrics`, in a
directory called `fairness/`, showing per-neighbourhood ROC-AUC and calibration for a system
that allocates municipal inspections, will be read by someone as a finding about
discrimination. It is not one, and the distance between what it measures and what it will be
read as is larger than for any artifact this project has produced.

Writing the limits into an ADR, the data contract, the manifest and the CLI summary is the
only way the boundary travels with the numbers.

## Decision

**Component 12 measures observable differences in model and ranking behaviour across the
geographic groups this data can define. It establishes none of the following, and the
documentation says so in these words.**

### It does not establish causality

Every number is observational. Nothing is randomised, no counterfactual is constructed, and no
intervention is simulated. A group with worse calibration is a group where the measured
probabilities matched the measured outcomes less well; the audit cannot say why, and cannot
say what would happen if anything changed.

Component 5's own limitation is inherited whole: this is a re-ordering study over inspections
that actually happened. No establishment nobody inspected has a label, so nothing here speaks
to coverage.

### It does not establish discrimination

Different calibration across groups is not the model discriminating against a group. The four
models carry no geographic input at all — Component 4's table is 26 numeric history features —
so any difference arises through correlated features rather than through a group attribute.

The converse is equally important and is why the audit exists: **the absence of a group
feature does not prove the absence of a disparity.** Fairness through unawareness is not
fairness, and "the model does not use community area, therefore it is fair" is a claim this
component was built to be able to contradict rather than to support.

### It does not establish the absence of bias

Equal measured performance across supported groups would not be evidence of fairness. It would
be evidence that one set of metrics, on one set of groups, at one grain, over one snapshot,
did not separate. Groups below the support floor are excluded from every comparison — 27 of 78
community areas at the 200-row floor — and a system can be globally even while failing badly
for a small group nobody could measure.

### It does not establish legal or regulatory compliance

No protected class is observed anywhere in this project. Community area and ZIP correlate
strongly with race and income by construction — that is what the city publishes statistics
against — but a correlate is not the attribute. A disparate-impact finding requires the
protected characteristic, and this audit does not have it, so no output here is evidence for
or against a compliance position.

### It does not establish ethical acceptability, or equal treatment

Whether a measured difference is acceptable is a policy judgement about how a city allocates
enforcement, and it depends on facts this project does not hold. The audit's job is to make
the trade-off visible. Deciding it is not delegated here, and Component 12 is recorded as
blocked from model selection in every manifest it emits.

### It does not establish an optimal fairness policy

No fairness intervention is implemented, tested or recommended. No reweighting, no threshold
adjustment, no per-group calibrator, no constraint. The formal reason is that the standard
criteria are mutually incompatible when base rates differ — and they differ here, from 0.220
to 0.566 across supported community areas — so "optimal" is undefined until someone chooses
which criterion to prefer. That choice is Component 13's.

### The limitation it inherits, and cannot work around

ADR 0019 recorded that this dataset publishes 22 columns and none identifies an inspector, and
named the consequence for this component in advance:

> Component 12's fairness audit inherits the same gap: disparate citation rates across
> establishment groups cannot be decomposed into establishment risk versus differential
> inspector treatment.

The target is that a Priority violation was **cited**, not that an establishment was unsafe.
Chicago assigns inspectors by district, so geography is close to the strongest available proxy
for who inspected — which means a measured geographic difference in outcome rate is confounded
with inspection practice by construction, and this project cannot separate the two.

ADR 0023 said the same thing about the community-area signal specifically, listing three
candidate explanations for it — genuine geographic risk, a demographic proxy, or a proxy for
inspection practice — and recording that the third is indistinguishable from the first two in
this data. **Component 12 does not distinguish them.** It was handed that question and it is
reporting that the question is not answerable here, which is the honest outcome and not a
failure to try.

### Nothing was quietly fixed

This is an audit. It retrains nothing, recalibrates nothing, reweights nothing, and modifies
no prediction. The input artifacts' checksums are compared before and after every run and a
change is an error-severity failure.

Where an uncomfortable result was found it was reported. Where a group could not be measured,
that is recorded as `insufficient_support` with its counts rather than dropped. Where a
statistic would have been prettier under a different bin count or a different threshold, the
canonical one was kept.

## Alternatives rejected

**Say nothing and let the numbers speak.** They do not speak; they get quoted. An artifact
called `fairness_group_metrics` with no attached boundary is one screenshot away from being
presented as a bias audit.

**Soften the language so the component sounds more useful.** Rejected. "We audited fairness
across Chicago neighbourhoods" is a sentence this code cannot support, and a component whose
value proposition needs it should not exist.

**Put the limitations only in the findings document.** Rejected for the reason ADR 0019 gave
for putting its block into every manifest: a document can be lost, superseded or not read,
and the caveat has to travel with the artifact. The `BLOCKED_EXPERIMENTS` list and the
manifest's `does_not_establish` field carry it into every run.

**Refuse to build the component at all, since it cannot certify fairness.** The most defensible
alternative, and rejected. Measuring where a deployed ranking behaves differently is useful
even when the causes are unidentifiable, and it is strictly better than the status quo of
having never looked. What is not acceptable is measuring it and then describing the result as
something it is not.

## Consequences

- Every Component 12 manifest carries a `does_not_establish` list naming causality,
  discrimination, absence of bias, legal compliance, ethical acceptability, equal treatment
  and optimal policy.
- The CLI summary prints the boundary on every run, next to the counts, so it is not possible
  to read a run's output without it.
- The data contract opens with a `⚠ What this artifact is not` section, and the findings
  document ends with limitations rather than a conclusion.
- **A green run means the audit is sound, not that Sentinel is fair.** That sentence appears
  in the summary, the contract and the findings document, in those words.
- A future component that wants a protected-class finding must ingest the protected
  characteristic. That is a Component 1 extension with its own provenance and vintage
  questions, and this ADR does not license a proxy in the meantime — the same position
  ADR 0019 took on inspector identity, for the same reason.
