# ADR 0019 — Inspector-effect modelling is blocked: the dataset has no inspector

**Status:** Accepted · **Date:** 2026-08-17

## Context

The published audit of Chicago's 2015 food-inspection model identified inspector identity as a
serious problem, and this project has carried that finding since Component 5. The mechanism is
straightforward. An observed violation is not a measurement of establishment risk alone:

```
P(cited)  =  f(establishment risk, inspector strictness, time and day effects)
```

An establishment does not choose its inspector. So if inspector identity becomes an establishment
feature, a restaurant that happened to draw a strict inspector acquires a permanently higher risk
score for a reason it cannot control and cannot correct — the model would encode an assignment as
a property. That is the failure mode the audit named, and it is why the project specification asks
for inspector strictness to be treated as a **nuisance effect**: estimated separately (a
mixed-effects logistic regression, establishment features as fixed effects, inspector as a random
intercept), controlled for during fitting, and **marginalised away at inference**, so the final
ranking answers *"what is this establishment's risk under a typical inspector?"* rather than
*"what is its risk because inspector X visited last?"*.

Component 7 is where the specification places that work, and it is a good design. It is also not
implementable on this data.

## Decision

**Component 7 does not model inspector effects, does not estimate inspector strictness, and does
not marginalise over an inspector effect. The field does not exist.**

### The evidence

The Chicago Food Inspections dataset (Socrata `4ijn-s7e5`) publishes **22 columns**, verified from
the ingested snapshot's own manifest, where `column_names` equals the endpoint's declared
`X-SODA2-Fields`:

```
inspection_id, dba_name, aka_name, license_, facility_type, risk, address, city, state, zip,
inspection_date, inspection_type, results, violations, latitude, longitude, location,
:@computed_region_awaf_s7ux, :@computed_region_6mkv_f3dw, :@computed_region_vrxf_vc4k,
:@computed_region_bdys_3d7i, :@computed_region_43wa_7qmu
```

None identifies a person. Not a name, not a badge number, not a pseudonymous id, not a team code.
Component 1's ingestion takes the full declared projection and drops nothing
(`ingest/food_inspections.py::_resolve_columns`), so this is an absence in the source rather than a
narrowing we applied.

Every downstream schema inherits the absence: Component 2's assignments, Component 3's targets,
Component 4's 33 columns, Component 5's folds and metrics, Component 6's predictions. The repository
already says so in prose — `docs/data_contracts/temporal_evaluation.md` records that inspector data
"are not ingested at all", and `modeling/definitions.py` lists `"inspector identity (deliberately
excluded -- audit Finding 1)"` among the 2015 model's unreachable inputs.

One wording correction this ADR makes: **"deliberately excluded" understates it.** It reads as a
modelling choice that could be reversed. The truth is stronger — there is nothing to exclude.

### Why an absent grouping cannot be modelled

A random intercept is an estimate of between-group variance over an *observed* grouping. With no
grouping variable there is no likelihood to maximise, no variance component to estimate and no
posterior mean to marginalise. Writing code that produced a number here would not be a model with
wide error bars; it would be arithmetic on a quantity nobody measured.

Marginalisation is worse, because it *looks* like it would work. Marginalising a tree model over an
inspector effect means averaging predictions across the inspector distribution — which requires
knowing the distribution, which requires the field. Substituting any other variable produces a
well-formed number that answers a different question, and nothing downstream would reveal the
substitution.

### What is recorded instead

1. A `BLOCKED_EXPERIMENTS` entry in every Component 7 manifest, naming the absence, the specific
   methods it blocks, the proxies refused and this ADR. Every run carries it, so the record cannot
   be lost when a document is.
2. `tests/test_boosting_inspector_blocked.py`, which re-derives the absence from the raw data
   contract, the ingested manifest, Component 4's feature list and every registered model's feature
   columns — rather than asserting a sentence. It fails if an inspector-like column ever appears,
   so the block cannot quietly outlive its reason.
3. This ADR.

## Alternatives rejected

**Use a proxy for inspector strictness.** Several are available and each is superficially
reasonable:

- **Violation-text verbosity** (`n_violation_entries`, per-entry length). Genuinely carries
  inspector-behaviour signal — ADR 0008 rejected the violation *count* as a target for exactly
  this reason, recording that "the count depends on inspector write-up verbosity more than on
  establishment state". But verbosity is unattributable: it tells you *that* write-up style varies,
  never *whose*. Grouping by it groups inspections, not inspectors.
- **Ward or community area** (`:@computed_region_43wa_7qmu`, `:@computed_region_vrxf_vc4k`).
  Chicago assigns inspectors by district, so geography is the closest thing to a "who inspected"
  grouping. It is also fully confounded with establishment composition — neighbourhoods differ in
  cuisine, chain penetration, building age and median income — so a "strictness" estimate would be
  substantially a neighbourhood-risk estimate. Marginalising it away would remove real
  establishment risk along with the supposed nuisance.
- **Day of week and month.** Already used by Component 5's time-invariance sensitivity, which
  explicitly warns the seasonal term "confound[s] temperature with daylight, holidays and staffing".
  Staffing is the inspector-workforce channel, so this is a proxy for a mixture that *includes* the
  target quantity and several others.

All three are rejected together, and the reason is one sentence: **none identifies a person, so
none supports a per-inspector random effect, and a marginalisation over any of them answers a
question nobody asked while carrying the vocabulary of the question they did.** A model labelled
"inspector-adjusted" that adjusted for ward would be the most misleading artifact this project
could ship.

**Ship the three-condition comparison (with inspector / marginalised / without) using a proxy in
place of inspector.** The specification asks for this comparison and it is the most tempting
substitute, because the table would look right. Rejected: the "with inspector" and "marginalised"
columns would both be fabrications, and the "score inflation attributable to inspector identity"
metric would be measuring inflation attributable to geography. A well-formed table of wrong numbers
is worse than an empty one, because it survives being quoted.

**Estimate a latent strictness effect over ward × quarter cells without calling it inspector.**
This is defensible *as its own thing* — unobserved heterogeneity in citation propensity across
geography and time is a real and interesting quantity, and Component 12's fairness audit may want
it. Rejected here because it is not what Component 7 was asked for, because naming it anything
near "inspector" would invite the misreading this ADR exists to prevent, and because Component 7's
boundary is gradient boosting.

**Ingest an external source carrying inspector identity.** The correct fix, and the one that would
unblock everything above. Rejected for Component 7 because no such source has been identified: the
open dataset does not carry it, and obtaining it would mean a FOIA request or a data-sharing
agreement with CDPH. That is a Component 1 extension and a procurement question, not a modelling
decision.

**Say nothing and move on.** Rejected because the audit finding is one of the two reasons this
project exists, and a reader who knows the audit will ask what happened to it. An evidenced block
is a better answer than silence, and a much better answer than a proxy.

## Consequences

- **Sentinel cannot separate establishment risk from inspector strictness on this data, and every
  model in the project inherits that.** The measured target is "a Priority or Priority Foundation
  violation was *cited*", not "the establishment was unsafe", and the gap between those two
  contains inspector variation that nothing here can quantify. This limitation belongs beside every
  result the project reports, not only Component 7's.
- The models are, in a narrow sense, safe from the audit's specific failure by construction: no
  registered model can carry an inspector feature because none exists, and the import-time guard
  restricts every model to Component 4's 26 declared columns. That is safety by absence, not by
  design, and it should be described that way.
- **This ADR does not license a proxy later.** A future component wanting inspector adjustment must
  either obtain the field or re-open this decision explicitly.
- To unblock: ingest a source with a per-inspection inspector identifier, add it behind a bumped
  `feature_definition_version` in Component 4, then implement the mixed-effects estimate and the
  marginalisation as originally specified. The regression test in
  `tests/test_boosting_inspector_blocked.py` will fail on the day such a column appears, which is
  the intended trigger for that conversation.
- Component 12's fairness audit inherits the same gap: disparate citation rates across
  establishment groups cannot be decomposed into establishment risk versus differential inspector
  treatment. That is a material limitation on what a fairness audit can conclude here, and it
  should be stated in that component rather than discovered during it.
