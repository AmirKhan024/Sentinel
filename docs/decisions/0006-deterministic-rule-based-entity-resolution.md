# ADR 0006 — Deterministic rule-based entity resolution

**Status:** Accepted · **Date:** 2026-08-16

## Context

Every component after this one — targets, historical features, temporal
evaluation, modelling, calibration, policy, scheduling — depends on knowing which
inspection records describe the same physical establishment. Nothing in the raw
data provides that key.

An empirical investigation of all 314,245 rows
(`docs/analysis/entity_resolution_findings.md`) established the shape of the
problem, and several findings contradicted the intuitions the design started
from:

- `license_` fails as an entity key in **both** directions. One licence can
  cover several successive businesses (a school and a bar share licence 1932),
  and — the larger problem — **18.47% of apparent establishments hold more than
  one licence**, up to 47 for a mobile-food commissary with one permit per cart.
  A licence is frequently *finer*-grained than an establishment.
- The `'0'` licence sentinel is attached to 323 distinct business names across
  364 addresses.
- Name normalization has low yield: full normalization resolves 0.21% of
  licences. Address normalization has high yield: case and whitespace alone
  collapse 33,261 distinct address strings to 20,313.
- Chains are pervasive: 247 Subways, 184 Dunkin' Donuts, and one O'Hare address
  carrying 219 distinct business names.
- The false-merge and false-split costs are asymmetric. A wrong merge pools
  years of inspection history onto one identity and corrupts every downstream
  feature for every establishment involved. A wrong split only costs
  statistical power.

## Decision

Entity resolution is a **deterministic, rule-based, auditable** pipeline:

```text
normalize → build nodes → block → evaluate pairs against named rules
         → union-find over accepted edges → check cluster invariants
         → split deterministically if an invariant fails
```

Five properties are load-bearing.

**1. Matching operates on nodes, not inspections.** A node is one distinct
identity signature (licence, names, address, unit, zip, coordinate). 314,245 rows
collapse to 51,099 nodes, and an audit row becomes a statement about two ways of
recording a place rather than about two individual visits.

**2. Address equivalence is required for every non-licence merge.** Two nodes at
different places are never compared on name at all. This is what makes 247
Subways harmless *without* a chain-name list, and it is what bounds transitive
chaining: a chain can only spread through the address graph via licence hops,
and those are capped by the cluster invariants.

Address equivalence has two routes: an identical normalized address key, or an
identical geocoded coordinate within one zip. The second exists because
coordinate spread within an address is exactly 0 m and address variants share a
single coordinate — the city's geocoder is a better address normalizer than any
string rule, and it bridges cases string rules cannot, including Chicago's 2021
rename of Lake Shore Drive.

**3. Licence equality is supporting evidence, and licence inequality is never
evidence against.** Treating licence disagreement as a signal would fracture 18%
of establishments.

**4. Name matching is exact after normalization**, over both `dba_name` and
`aka_name`. Token-containment is admitted as a weaker tier, at an identical
address only.

**5. Vetoes outrank agreement.** Conflicting directionals, suites, store numbers,
or trade names defeat any merge rule, because each of those is a positive
statement that two records describe different places.

Rules are named (`S1`–`S3`, `P1`–`P2`, `A1`–`A2`, `V1`–`V4`, `N0`–`N2`), the
deciding rule is recorded on every edge, and edges are written to an audit table.

Ambiguity is a first-class outcome. A pair with real but insufficient evidence is
recorded and **not merged**.

## Alternatives rejected

**Grouping by `license_`.** Measured to fail, in both directions (see Context).
Not a judgement call — the numbers are in findings §3.

**Grouping by exact `(name, address)` strings.** Measured to fail: 33,261 raw
address strings describe 20,313 places, so this over-splits massively before any
of the harder cases.

**Fuzzy name matching with a calibrated similarity threshold.** This was the
original plan, including a 100-pair manual calibration exercise. The measurements
retired it: full name normalization resolves 0.21% of licences, so the achievable
upside is small, while the downside — 247 Subways and 219 business names at one
address — is large. The plan named `T1 = 1.0` (exact match only) as its
documented fallback if calibration did not justify a fuzzy tier; the measurements
reached that conclusion before calibration rather than after. **This is a
measured outcome, not an omission.**

**Probabilistic record linkage (Fellegi–Sunter, `splink`).** Needs labelled
match/non-match pairs, which do not exist for this dataset, and produces a
probability that cannot be explained to a public-health inspector asking why two
restaurants were treated as one. The asymmetric cost structure here wants a
conservative rule that can be read, not a calibrated score.

**Embedding-based or density clustering (DBSCAN, agglomerative on vectors).**
Opaque, sensitive to initialisation and library version, and unable to answer
"why were these two merged?" with anything better than a distance. Determinism
across runs would depend on a numerical library's version.

**An LLM adjudicator.** Non-deterministic, unauditable, unreproducible, and
impossible to regression-test. Explicitly out of scope.

**`rapidfuzz` or `jellyfish` for string similarity.** No new dependency was
added. With address-based blocking the pair count is 335,393, and the primary
comparison is token-set equality over pre-computed frozensets — exact, stdlib,
and trivially explainable. Total resolution runtime is 43 s. Buying a dependency
for a workload that already runs in well under a minute would contradict the
project rule of introducing a technology only when a component requires it.

**`networkx` for connected components.** Union-find is forty lines and is the
only graph operation needed. Writing it out means no dependency whose version
could change a grouping between runs.

**Temporal succession logic** (merging on temporally disjoint licence spans at
one address). Planned, then dropped: 75.5% of same-place licence pairs *overlap*
in time rather than succeeding one another, so disjointness is a weak
discriminator. Dropping it also removed the last use of `inspection_date` from
the matcher, which is a leakage benefit.

## Consequences

- Every merge and every declined merge is explainable from the output alone, by
  filtering the edges table on a node id.
- Resolution is reproducible: the same snapshot and code version always produce
  the same identities, verified on the full 314,245 rows against a seeded random
  permutation of the input.
- The system under-merges rather than over-merges. 747 ambiguous pairs are
  recorded for review rather than forced into a decision.
- Rules are visible and cheap to change, but changing one changes
  `establishment_id` values (see ADR 0007), so `normalization_version` is
  recorded in the manifest.
- Exact-only name matching means genuine typos in a business name will split an
  establishment unless the licence, address key or coordinate carries it. This
  is the accepted cost of not adding a fuzzy tier, and it is revisitable: the
  measurement that would overturn it is a materially large class of
  character-level typos among same-address pairs.
- The residual false-merge risk is concentrated at dense addresses, where two
  same-name outlets with no store number and no distinguishing trade name are
  indistinguishable from the data. This is documented rather than hidden, and
  surfaced by the address-density and cluster-size checks.
