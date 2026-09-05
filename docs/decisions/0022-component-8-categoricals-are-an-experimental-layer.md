# ADR 0022 — Component 8's categoricals are an experimental layer, not a Component 4 change

**Status:** Accepted · **Date:** 2026-08-18

## Context

The project specification for Component 8 names four categorical families to embed: chain,
facility type, community area and ZIP. **None of them exists in Component 4's feature table.**

Verified against `as_of_features_20260816T150313Z.parquet` (57,727 rows, 33 columns): all 26
features are numeric temporal-history counts, recencies and rates. There is no categorical column
of any kind. Where the four actually live:

| family | where it is | distinct, over the 57,727 feature rows |
| --- | --- | ---: |
| `facility_type` | raw Socrata snapshot; dropped at `target/build.py:48` | 169 |
| `zip` | raw Socrata snapshot | 72 |
| `community_area` | raw only, as Socrata computed region `:@computed_region_vrxf_vc4k` | 78 |
| `chain` | **nowhere**; derived from Component 2's `name_key` | 950 chains, 22.70% of rows |

This collides head-on with a rule the project has stated twice, in STATUS.md and HANDOFF.md:

> **Do not add a feature.** If one is missing it belongs in Component 4 behind a bumped
> `feature_definition_version`.

and with the instruction governing this component, that Components 1–7 must not be modified. Taken
together those two rules make the specified experiment impossible as literally written: the
features are not there, and the place they belong is a component that may not be touched.

The conflict was surfaced before any code was written rather than resolved silently.

## Decision

**Component 8 builds its own categorical layer, explicitly labelled experimental, and Component 4
is not modified.**

Concretely:

- A new module `neural/categoricals.py` and a new command
  `sentinel build-neural-categoricals` produce `neural_categoricals_<stamp>.parquet`.
- It lands in a **new processed layer**, `data/processed/neural/`, reached by
  `Settings.neural_processed_dir` — a sibling of `features/`, `evaluation/`, `predictions/` and
  `tuning/`, not a child of any of them.
- `feature_definition_version` stays `v1`. Nothing in the new table is a Component 4 feature and
  nothing may treat it as one.
- Component 4's table, Component 6's artifacts and Component 7's artifacts are untouched.

### Why a separate layer rather than a Component 4 release

A `feature_definition_version = v2` would be the *correct* home for these columns if they were
production features. They are not, yet, and promoting them would have three costs the component's
question does not justify:

1. **It would invalidate the comparison.** Components 6 and 7 are measured on v1. Re-running them
   on v2 to keep the comparison honest is a much larger change than Component 8, and not
   re-running them would mean comparing models across different feature tables — the exact
   confound the fair-comparison rule exists to prevent.
2. **It would promote community area into the production feature set** before Component 12 has
   audited it as a demographic proxy. See ADR 0023.
3. **The experiment might fail.** The whole point of Component 8 is to find out whether learned
   representations of these families help. Shipping them as features first and measuring second
   inverts that.

### Why the comparison is still fair

`neural_numeric_only` is registered precisely for this. It sees the same 30 matrix columns
Components 6 and 7 see — 26 features plus the four null-rule family indicators — and no
categoricals at all. It is the model that carries any claim of the form "the neural estimator
achieved X while XGBoost achieved Y", and every categorical-bearing model is reported beside it
rather than in place of it.

`validate._every_model_scored_the_same_rows` enforces that every Component 8 model scores an
identical `target_inspection_id` set, so no comparison in this component is over different
populations.

### The as-of rule applies unchanged

Each categorical is the value recorded at the establishment's most recent inspection of **any
type, strictly before** the row's own `inspection_date`. The target row never supplies its own
attributes.

This is stricter than it strictly needs to be. Facility type and address are genuinely known
before an inspection happens, so reading them off the target row would be arguably legitimate. But
they are *recorded on the inspection record*, which is written at inspection time, and this
project does not build features from the row being predicted. Carrying the last observed value
forward needs no exception to ADR 0010 and is directly testable.

The table emits `source_inspection_id`, `source_inspection_date` and `days_since_source` beside
every value, and `validate._categoricals_are_strictly_as_of` re-derives the strict inequality on
every row. Measured: minimum lag **1 day**, median 357, maximum 5,416. A zero would mean a row had
supplied its own attributes.

The cost is stated rather than hidden: **401 rows have no prior inspection of any type** and get
`__UNKNOWN__` for all four families. That is exactly the number of rows Component 4 marks with a
null `days_since_any_inspection` — the two components independently agree on which establishments
have no history, which is the consistency check worth having.

## Alternatives rejected

**Bump `feature_definition_version` to v2 and add the four columns.** Rejected for the three
reasons above. This remains the right move *if* Component 8 finds the families useful and
Component 12 clears community area — and that ordering is the point.

**Read the categoricals straight off the target inspection row.** Simpler, and defensible on the
grounds that facility type is known in advance. Rejected because it would make Component 8 the
only component in the repository that reads a field from the row it is predicting, and the
precedent is worth more than the 401 rows it would save.

**Join the categoricals inside `train_neural` without an artifact.** Rejected: the join is the one
part of this component that reaches outside Component 4's contract, and it should be a step a
human runs and can inspect, not a silent step inside training.

**Put the table in `data/processed/features/`.** Rejected. Co-location with the feature table is
exactly the invitation to join, and ADR 0014 already records that co-location is how the most
damaging leakage happens.

**Derive chain from `dba_name` directly rather than Component 2's `name_key`.** Rejected: it would
put a second name-normalisation in the repository, and "the same name" must mean the same thing in
Component 8 as it does in entity resolution.

**Merge facility-type synonyms** (`GROCERY STORE` / `GROCERY`). Rejected: deciding two free-text
values mean the same business is a judgement this module has no basis for, and it would be baked
invisibly into every result. Normalisation is limited to case, trimming and whitespace collapse.

## Consequences

- The processed layer has five kinds of artifact. `data/processed/neural/` holds exactly one
  table and is not a general-purpose home for anything else.
- **Nothing in `data/processed/neural/` may be joined onto a feature table by any other
  component**, and the `Settings` property's docstring says so.
- Component 8's reproduction sequence gains a step: `build-neural-categoricals` runs before
  `tune-neural` and `train-neural`, and both read the artifact by path.
- The categorical table is pinned by checksum in the neural manifest alongside the feature table,
  so a run is traceable to both inputs.
- If Component 4 later adopts these families behind `v2`, this layer should be deleted rather than
  left to drift beside the real one.
- A reader who finds `facility_type` in this project will find it in exactly two places — the raw
  snapshot and this experimental table — and in neither is it a feature.
