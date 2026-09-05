# ADR 0021 — Entity embeddings: what may be embedded, and why not `establishment_id`

**Status:** Accepted · **Date:** 2026-08-18

## Context

STATUS.md and HANDOFF.md both name entity embeddings as Component 8's one genuinely new
capability and, in the same breath, as the largest leakage risk in the project:

> An embedding of `establishment_id` is exactly what Components 6 and 7 cannot learn. It is also a
> per-establishment parameter fitted across rows, so an embedding fitted without respecting the
> fold boundary carries future information about that establishment backwards. Component 4
> deliberately excludes establishment identity as a feature; re-introducing it through an
> embedding needs its own ADR and its own leakage suite, not a paragraph.

This is that ADR and that suite.

There is a hard constraint in the code already. `modeling.definitions.FORBIDDEN_COLUMNS` contains
the three key columns, the two label columns and the two provenance columns, and both
`modeling.definitions._guard_registry` and `boosting.definitions._guard_registry` raise at import
time if any spec's `feature_columns` intersects it. `establishment_id` is in that set. A naive
`NeuralSpec(feature_columns=(..., "establishment_id"))` would not import — which is the correct
behaviour and is not, on its own, enough: an embedding is not a feature column, and a guard that
only inspects `feature_columns` would not see one.

## Decision

### Categorical inputs live in a second field, guarded separately

`NeuralSpec` carries two column lists:

- `feature_columns` — the 26 Component 4 features, guarded against `FORBIDDEN_COLUMNS` exactly as
  Components 6 and 7 guard theirs, by importing the same constant.
- `entity_columns` — the categorical families, guarded against its own closed allowlist.

They are separate fields rather than one list because the two have **different safety arguments**,
and collapsing them would let one borrow the other's justification. A categorical also becomes
unreachable by any code that iterates `feature_columns`, which is most of the preprocessing.

### The allowlist is a closed enum, and identity is not on it

`EntityFamily` has exactly four members: `chain`, `facility_type`, `community_area`, `zip`.
`_guard_registry` refuses any `entity_columns` entry that is not one of them, with an error
message that names `establishment_id` specifically. Adding a fifth family means editing the enum,
the embedding-dimension table and this ADR — not editing a spec.

### `establishment_id` is refused

It is the obvious thing for an entity-embedding component to learn, and it is the thing this
component must not do. Three reasons, in order of severity:

**It is a per-row-group parameter fitted across the whole training window.** An establishment's
vector absorbs whatever the network learned from every row that establishment appears in. Within a
fold that is legitimate — it is target encoding, confined to training data, no worse than a
training-window median. Across folds it is not, and the temptation to cache an embedding table
between folds (for speed, or to "warm start") would be a leak that no metric would reveal.

**Component 4 excludes identity by design.** `FORBIDDEN_COLUMNS` is not an implementation detail;
it encodes the decision that Sentinel ranks establishments by *what they have done*, not by *which
one they are*. An establishment embedding would reintroduce identity through a side door, and a
deployed system that scored an establishment high because of its identifier rather than its
history would be indefensible to the department that had to act on it.

**It would not even be learnable for most rows.** 15,144 distinct establishments across 57,727
rows is a mean of 3.8 rows each; a 16-dimensional vector per establishment is more parameters than
data for most of them.

### `chain` is the deliberate substitute

A chain is a *group* of establishments sharing Component 2's normalised name (`name_key`). It is
the entity concept that is genuinely new relative to Components 6 and 7, has enough rows per
category to learn from, and carries a plausible mechanism — corporate food-safety practice is
shared across a brand's locations in a way it is not across unrelated businesses.

Measured on the current snapshot: **950 names are carried by more than one establishment**,
covering 13,103 of 57,727 rows (22.70%). The largest are `SUBWAY` (159 establishments),
`DUNKIN DONUTS` (136) and `MCDONALDS` (51).

**Membership is derived per fold, from training rows only.** Whether a name counts as a chain
depends on which *other* establishments exist, so a global membership set would let a location
opened in 2025 make a 2022 row part of a chain. An establishment whose name is unshared within the
fold's training window is `__INDEPENDENT__` — a real category with a learned vector, because "not
part of a chain" is a fact worth conditioning on rather than a null.

### Index 0 is `__UNKNOWN__`, and its vector is learned

Any category absent from a fold's training rows maps to index 0. That row is **not** masked or
frozen at zero: genuine unknowns exist in training — 401 rows have no prior inspection of any type
to carry a categorical forward from — so index 0 receives gradient and learns the "never seen this
establishment before" offset. A category appearing only later maps onto that same row, which is
the honest default: the model has no basis to say anything else about it.

Measured out-of-vocabulary rates on the test windows are low (0.00%–2.04% per family per fold), so
this fallback is not doing most of the work.

### Every claim above has a test that drives it into failure

`tests/test_neural_leakage.py` covers the ten properties the specification names. The ones
specific to this ADR:

- a category planted only after `train_end` acquires no index, and rows carrying it fall back;
- a name that becomes shared only after `train_end` is not treated as a chain;
- every vocabulary entry re-derives from the window it was fitted on;
- vocabulary order is sorted, not insertion-ordered, because insertion order is row order;
- the embeddings-into-XGBoost experiment refuses a donor network fitted on a different fold;
- and a planted-label test proves the fixture can transmit a signal at all, so the tests above are
  measuring a protected pipeline rather than a weak model.

`validate._entity_columns_are_never_identity` restates the import-time guard at runtime. One guard
for this is not enough.

## Alternatives rejected

**Embed `establishment_id` with a minimum-row threshold** (e.g. only establishments with ≥20
rows). Rejected: it reduces the parameter-count objection without touching the other two, and the
threshold itself would be a fitted quantity needing its own temporal treatment.

**Embed `license_` instead.** Rejected: it is a near-synonym for establishment identity with the
same objections, and Component 2 already treats licence as an identity signal.

**Use a hashing trick for chains** to avoid a per-fold vocabulary. Rejected: collisions would be
untraceable, and the vocabulary being explicit is what lets a leakage test enumerate it.

**Compute chain membership once, globally, and document the caveat.** Rejected. This is the exact
shape of leak that leaves no trace in any artifact, and "documented" is not a control.

**Freeze the UNKNOWN row at zero.** Rejected: there are real unknowns in training, so the row is
learnable, and a frozen zero vector would assert that "no history" means "average", which is the
opposite of what Component 4's null-rule indicators say.

## Consequences

- Component 8 embeds four families and not establishment identity. The component's headline
  therefore cannot be "we learned per-establishment risk", and should not be read as such.
- The chain family's usefulness is bounded by its coverage: 22.70% of rows belong to a chain, so
  roughly three rows in four receive the `__INDEPENDENT__` vector. That is a ceiling on what this
  family can contribute, and it is a measurement rather than a guess.
- A future component wanting per-establishment structure must revisit this ADR rather than
  extending a spec.
- `neural_numeric_only` exists so that the C6/C7/C8 comparison is unaffected by any of this: it
  sees the same 30 matrix columns Components 6 and 7 see and no categoricals at all.
- Component 12's fairness audit inherits an explicit, ablatable community-area input rather than a
  hidden one. See ADR 0023.
