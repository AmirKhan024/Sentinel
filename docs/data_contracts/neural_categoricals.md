# Data contract — Component 8 experimental categoricals

**Produced by:** Component 8 (`sentinel build-neural-categoricals`)
**Layer:** `data/processed/neural/` — a fifth processed layer, deliberately separate
**Consumed by:** Component 8 (`sentinel tune-neural`, `sentinel train-neural`) and
Component 12 (`sentinel audit-fairness`), for reporting only -- see §0a
**Design rationale:** ADR 0022 (why this is not a Component 4 change), ADR 0021 (what may be
embedded), ADR 0010 (the as-of rule this inherits)

---

## 0. ⚠ This is not a feature table

Component 4's contract has 26 features and **not one of them is categorical**. The four families
Component 8 embeds do not exist there, and this table is how they are brought in. It is
quarantined on purpose:

- `feature_definition_version` **stays `v1`**. Component 4's table was not modified to produce
  this and no column here is a Component 4 feature.
- It lives in `data/processed/neural/`, a sibling of `features/`, not inside it. Co-location with
  the feature table is exactly the invitation to join, and ADR 0014 records that co-location is
  how the most damaging leakage happens.
- **Nothing here may be joined onto a feature table by any component.** Component 12 joins it
  onto *predictions*, for reporting, and never onto anything trainable -- see §0a.
- Component 8's own fair-comparison model, `neural_numeric_only`, is fitted without any of it.

If Component 4 later adopts these families behind a bumped `feature_definition_version`, this
layer should be **deleted** rather than left to drift beside the real one.

## 0a. Component 12 reads this table, and what that does and does not permit

ADR 0033 records the decision. Component 12's audit needs a *group* for each evaluated row,
and `community_area` and `zip` exist nowhere else in this project -- Component 4's table is 26
numeric history features and carries no location at all.

**What that permits:** joining these two columns onto Component 9's calibrated predictions, to
partition held-out rows for reporting. Nothing is fitted on them, no model sees them, and they
appear in no trainable table.

**What it does not change:** §0's rule stands in full. This is still not a feature table,
`feature_definition_version` is still `v1`, and the prohibition on joining it onto one is
unchanged rather than relaxed.

**Why the as-of value rather than the row's own.** Component 12 measured both before choosing:
the two disagree on **0 of 57,041** community-area rows and **0 of 57,326** ZIP rows. A
restaurant does not move, so carrying the last observed value forward reproduces the
contemporaneous one exactly -- and the safe option therefore needed no exception to ADR 0010.

**If this layer is ever deleted** -- as §0 says it should be when Component 4 adopts these
families behind a bumped `feature_definition_version` -- Component 12's group frame must be
re-pointed at the Component 4 columns **in the same change**.

## 1. Identity and file naming

```
data/processed/neural/neural_categoricals_<YYYYMMDDTHHMMSSZ>.parquet
data/processed/neural/manifest_neural_categoricals_<YYYYMMDDTHHMMSSZ>.json
```

zstd-compressed Parquet, sorted by `target_inspection_id`. One row per Component 4 feature row,
one-to-one, enforced by `validate._categoricals_cover_every_row`.

Primary key: `target_inspection_id`. It is Component 3's identifier and Component 4's primary key,
and it is also the raw Socrata `inspection_id` — which is what makes the join to the raw snapshot
possible without inventing a key.

## 2. The as-of rule

**Every value is the one recorded at the establishment's most recent inspection of any type,
strictly before the row's own `inspection_date`. The target row never supplies its own
attributes.**

This is stricter than the data strictly requires. Facility type and address are genuinely known
before an inspection happens, so reading them off the target row would be arguably legitimate —
but they are *recorded on the inspection record*, which is written at inspection time, and this
project does not build features from the row being predicted. Carrying the last observed value
forward needs no exception to ADR 0010.

The join is `polars.join_asof(strategy="backward", allow_exact_matches=False)`, partitioned by
`establishment_id`. **`allow_exact_matches=False` is the whole temporal argument.** With exact
matches allowed, the target inspection would supply its own categoricals on every row that has
one, every leakage test in the component would still pass, and the model would be reading the
present.

Three columns are emitted so the claim is checkable *per row* rather than asserted once:
`source_inspection_id`, `source_inspection_date`, `days_since_source`.

Measured on `neural_categoricals_20260818T125631Z.parquet` (57,727 rows):

| rows with a source | mean lag | median | p25 | p75 | **min** | max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 57,326 | 392.6 | 357 | 264 | 448 | **1** | 5,416 |

**The minimum of 1 day is the observable.** A zero would mean a row had supplied its own
attributes.

## 3. Missingness is a token, never a null

An establishment with no prior inspection of any type has nothing to carry forward. Those rows get
`__UNKNOWN__` — a **real category with a learned embedding row**, not a null and not an
imputation. "We have never seen this place before" is a fact about the establishment, and it is
the same fact Component 4 encodes with a null-rule family indicator.

A null would instead reach the encoder and be coerced somewhere, and where it landed would depend
on which code path saw it first. `validate._categoricals_are_never_null` refuses one.

Measured coverage:

| family | distinct | with a value | coverage | `__UNKNOWN__` |
| --- | ---: | ---: | ---: | ---: |
| `chain_key` | 13,303 | 57,326 | 0.9931 | 401 |
| `facility_type` | 169 | 57,274 | 0.9922 | 453 |
| `community_area` | 78 | 57,041 | 0.9881 | 686 |
| `zip` | 72 | 57,326 | 0.9931 | 401 |

**401 rows have no prior inspection at all**, which is exactly the number of rows Component 4
marks with a null `days_since_any_inspection`. The two components independently agree about which
establishments have no history; that agreement is the consistency check worth having.

`facility_type` and `community_area` lose a few more rows because a prior inspection can exist
while its own facility-type text is blank, or its coordinates missing.

## 4. Sources and normalisation

| emitted column | source | normalisation |
| --- | --- | --- |
| `chain_key` | Component 2 `establishment_assignments.name_key` | none — reused as-is, so "the same name" means the same thing here as in entity resolution |
| `facility_type` | raw `facility_type` | trim, upper-case, collapse internal whitespace; blank → `__UNKNOWN__` |
| `community_area` | raw `:@computed_region_vrxf_vc4k` | leading integer, or `__UNKNOWN__` |
| `zip` | raw `zip` | leading five digits, or `__UNKNOWN__` |

**Synonyms are deliberately not merged.** Deciding that `GROCERY STORE` and `GROCERY` are the same
business is a judgement this module has no basis for, and it would be baked invisibly into every
downstream result.

`community_area` is a **Socrata computed region**, not city-supplied source data: Socrata joins
the row's coordinates against a boundary layer, and the column is absent whenever the coordinates
are. That is documented behaviour, recorded rather than patched over, and it is why community area
has the lowest coverage of the four.

## 5. ⚠ `chain` is not in this table

This table emits `chain_key` — the as-of normalised name. The **chain category** is derived later,
inside each fold, by `neural.encode.chain_membership`:

- `__UNKNOWN__` when nothing could be carried forward;
- the name itself when that name is carried by **more than one establishment among the fold's
  training rows**;
- `__INDEPENDENT__` otherwise — a real category, because "not part of a chain" is a fact worth
  conditioning on.

Membership is a property of a *set* of establishments, so computing it once over the whole
snapshot would let a second location opened in 2025 make a 2022 row part of a chain. It is
therefore not precomputed here, and cannot be.

For description only (the model never sees this): **950 names span more than one establishment**,
covering 13,103 of 57,727 rows (22.70%). Largest: `SUBWAY` (159 establishments), `DUNKIN DONUTS`
(136), `MCDONALDS` (51), `7 ELEVEN` (47), `JIMMY JOHNS` (43).

## 6. Schema

### `neural_categoricals`

| column | type | null | meaning |
| --- | --- | --- | --- |
| `target_inspection_id` | Utf8 | never | Primary key. Component 3's id; also the raw `inspection_id`. |
| `establishment_id` | Utf8 | never | Component 2's identity. Carried for the as-of partition and for audit — **not an input to any model**. |
| `inspection_date` | Utf8 | never | The decision point, `YYYY-MM-DD`. Matches Component 4. |
| `chain_key` | Utf8 | never | As-of normalised name, or `__UNKNOWN__`. Not the chain category — see §5. |
| `facility_type` | Utf8 | never | As-of, normalised, or `__UNKNOWN__`. |
| `community_area` | Utf8 | never | As-of Socrata computed region, or `__UNKNOWN__`. |
| `zip` | Utf8 | never | As-of five-digit ZIP, or `__UNKNOWN__`. |
| `source_inspection_id` | Utf8 | **yes** | Which earlier inspection supplied the values. Null exactly when there was none. |
| `source_inspection_date` | Date | **yes** | Its date. **Always strictly earlier** than `inspection_date`. |
| `days_since_source` | Int32 | **yes** | The lag. Minimum observed 1. |

Sort key: `target_inspection_id`.

## 7. Validation

Four error-severity checks, every one re-derived from the data rather than read from a manifest:

| check | what it re-derives |
| --- | --- |
| `categoricals_cover_every_row` | one row per feature row, no extras |
| `categoricals_are_strictly_as_of` | `source_inspection_date < inspection_date`, **per row** |
| `categoricals_are_never_null` | absence is a token, never a null |
| `categoricals_carry_no_label` | no label or provenance column smuggled alongside |

All four passed on the shipped artifact.

## 8. Guarantees a consumer may rely on

1. One row per `target_inspection_id`, covering Component 4's table exactly.
2. Every categorical is a non-null string; absence is `__UNKNOWN__`.
3. Every non-null `source_inspection_date` is strictly earlier than the row's `inspection_date`.
4. No outcome, label or provenance column is present. Nothing here was derived from a target.
5. Re-running against the same three inputs produces a byte-identical file.

## 9. Known limitations

1. **This is not a feature table and its contents are not validated as features.** No null-rule
   family, no dtype contract, no Component 4 test covers them.
2. **Community area is a candidate demographic proxy** and is included only as an audited
   experimental input with a matched ablation. See ADR 0023.
3. **Facility type is free text**, normalised only for case and whitespace. Synonyms remain
   distinct categories.
4. **`community_area` is a derived spatial join**, not authoritative city data, and is absent
   wherever coordinates were.
5. **The carried value can be very stale** — up to 5,416 days. Facility type and address are
   stable attributes, but a stale value is still stale.
6. **`chain_key` cardinality (13,303) is not chain cardinality (950).** Most normalised names
   belong to exactly one establishment and become `__INDEPENDENT__` inside a fold.
7. **The 401 no-history rows are the same rows** Component 4 cannot compute a recency for. They
   are the hardest rows in the dataset for every model, not just this one.

## 10. Reproducing

```bash
uv run sentinel build-neural-categoricals --report
```

Resolves the most recent feature table, raw snapshot and Component 2 assignment table unless
`--features`, `--raw` and `--assignments` are given. Runtime ~0.5 s.
