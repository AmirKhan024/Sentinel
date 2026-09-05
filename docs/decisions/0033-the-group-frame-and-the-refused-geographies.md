# ADR 0033 — The group frame is as-of geography, and ward is refused

**Status:** Accepted · **Date:** 2026-08-25

## Context

A fairness audit needs a group. This project has no demographic variable of any kind — no
race, income, ACS, census or deprivation field is ingested anywhere, and ADR 0023 records the
refusal to add one. What it does have is geography, in the raw snapshot only.

`scripts/profile_fairness.py` inventoried every geographic column before anything was chosen,
measured on the 314,245-row snapshot:

| raw column | concept | distinct | nulls |
|---|---|---:|---:|
| `zip` | postal code as typed | 140 | 42 |
| `:@computed_region_vrxf_vc4k` | community area | 78 | 1,365 |
| `:@computed_region_6mkv_f3dw` | ZIP-code region | 66 | 1,042 |
| `:@computed_region_43wa_7qmu` | ward, current boundaries | 51 | 1,365 |
| `:@computed_region_awaf_s7ux` | ward, 2003–2015 boundaries | 51 | 1,365 |
| `:@computed_region_bdys_3d7i` | census tract | 797 | 1,089 |
| `latitude` / `longitude` | point geography | 18,931 | 1,042 |
| `address` | street address | 33,261 | 0 |
| `city` | municipality as typed | 95 | 181 |
| `state` | state as typed | 8 | 69 |

**None of them is a model feature.** Component 4's table is 33 columns of which 26 are
numeric inspection history, and it carries no location at all.

Two questions therefore had to be settled before any metric was computed: which column
supplies the group, and where the value is read from.

## Decision

**The audited group definitions are `community_area` and `zip`, both read from Component 8's
as-of categorical layer. Ward, census tract, point geography and city/state are refused, and
the refusals are written into the artifact rather than only into this document.**

### The value comes from the as-of layer, and it costs nothing

`data/processed/neural/neural_categoricals_<stamp>.parquet` carries, for every one of
Component 4's 57,727 rows, the community area and ZIP recorded at that establishment's most
recent inspection of any type **strictly before** the row's own `inspection_date`
(`join_asof(strategy="backward", allow_exact_matches=False)`, ADR 0022). Component 8 validates
the strict inequality per row; the measured minimum lag is 1 day and a zero would mean a row
had supplied its own attributes.

The alternative was to read the value off the row being audited. That is contemporaneous
rather than future information, it is used only to partition held-out rows for reporting, and
it would have been defensible. It was still not taken, and the reason is a measurement rather
than a principle:

| group definition | rows where both values exist | disagreements |
|---|---:|---:|
| `community_area` | 57,041 | **0** |
| `zip` | 57,326 | **0** |

**The two never disagree.** A community area is an attribute of a fixed premises, so carrying
the last observed value forward reproduces the contemporaneous one exactly. The temporally
safe option is therefore free, and taking it means this component needs no exception to
ADR 0010, introduces no new join against raw, and inherits a frame that is already validated
strictly as-of. The cost is 362 rows of coverage — `community_area` is `__UNKNOWN__` on 686
rows against 324 nulls on the row's own value — and that cost is recorded rather than hidden.

### `__UNKNOWN__` is a group, not a null

401 rows have no prior inspection of any type and a further ~285 had one whose coordinates
were missing. Those rows carry `__UNKNOWN__`, and it is audited as a first-class group value
with its own support counts and its own metrics.

Dropping it would have been the single most misleading choice available here, because it is
not a random 1.2% of the data. It is a superset of exactly the rows Component 4 cannot compute
a recency for — the same rows the null-rule family indicators fire on, and the same indicators
Component 11 measured ranking second and third in importance for two of four models. The
group with no geography is the group with no history, and the audit's own missingness section
depends on it being present.

### Ward is refused, and the dataset proves why

The snapshot publishes **two** ward layers, and they assign different region ids to
**56,451 of 57,403** rows — 98.3%.

That the publisher ships a 2003–2015 vintage alongside a current one is the point. A ward
identifier is a property of a boundary version, not of a place, so attaching the current ward
to a 2019 row assigns that row to a district that did not exist when it was inspected. That is
exactly the "present-day attribute attached to a historical row" hazard the brief's data-
provenance section names, and it is a leak of group identity rather than of outcome — subtler,
because every number downstream would still look finite and plausible.

Chicago's 77 community areas have been fixed since the 1920s, which is why the city publishes
statistics against them, and that stability is the whole reason one geography is admissible
and the other is not.

ADR 0019 separately refused ward and community area as *inspector* proxies. This ADR does not
overturn that: neither is used to estimate an inspector effect here, and nothing in Component
12 is labelled inspector-adjusted.

### The other refusals

- **Census tract** — 797 groups over 32,696 quarterly test rows is ~41 rows each, before any
  fold split. Nothing would clear a support floor and reporting it would be a table of nulls.
- **Latitude/longitude** — ADR 0023 already rejected continuous geography, and the reason
  transfers exactly: it is not less of a proxy, only less legible, and a less legible proxy is
  worse for an audit rather than better.
- **`city` / `state`** — degenerate. 312,957 of 314,245 rows say `CHICAGO`, and the 95
  distinct values include `Chicago`, `chicago` and `CCHICAGO`. A group definition whose
  cardinality comes from typing errors is not a group definition.
- **`:@computed_region_6mkv_f3dw`** — a second, coarser ZIP geography. Auditing both it and
  `zip` would report one thing twice and invite the two to be read as corroborating each other.
- **`facility_type`** — available, and not geography. It is recorded in the registry as
  refused-for-scope rather than refused-for-quality, so a later component wanting an
  operational rather than geographic dimension has the entry to turn on.

### The refusals are data

`fairness_group_definitions` carries one row per candidate — audited **and** refused — with
its source column, cardinality, missingness, the temporal-stability evidence, whether it is a
model feature, and the reason. A refusal that lives only in prose is a refusal that stops
travelling the moment someone reads the Parquet instead of the ADR, and the registry guard
raises at import if a refused definition is requested.

### Community area is an opaque region id, and is not given neighbourhood names

`:@computed_region_vrxf_vc4k` is a Socrata **computed region**: the platform joins the row's
coordinates against a boundary layer and returns that layer's row index. It is stable, and it
is not necessarily the city's official community-area number.

This project has ingested no boundary file, so the mapping from region index to neighbourhood
name is not established here. The audit therefore reports region ids and does not print
"Austin" or "Lincoln Park" anywhere. Naming them would be the most useful thing this component
could do for a reader and the easiest place for it to be quietly wrong — an off-by-one in a
boundary index would attribute a measured disparity to the wrong neighbourhood, in a document
whose entire purpose is to be trusted about which neighbourhood.

Unblocking it is one ingestion away and is recorded as such.

## Alternatives rejected

**Read the group off the row being audited.** Rejected above, on the measurement rather than
on principle. It would also have made Component 12 the second component to read a field from
the row it is processing, and Component 8 declined to be the first (ADR 0022).

**Derive the group frame with a fresh as-of join against raw rather than reusing Component
8's.** Would gain ward and census tract, both of which are refused anyway, and would put a
second as-of geography join in the repository — two implementations of "the same place", which
is precisely why ADR 0022 refused to derive `chain` from `dba_name` a second time.

**Promote community area into Component 4 so the audit reads a feature table.** Rejected, and
foreclosed in advance: ADR 0023 records that promoting it requires a Component 4 release *and*
a Component 12 finding, in that order. This component is the finding, not the release.

**Audit only community area.** Tighter, and it is exactly the question ADR 0023 handed over.
Rejected because ZIP is measurably better supported — 56 of 69 ZIPs clear the 200-row floor
against 51 of 78 community areas — and because a disparity that appears under one geography
and not the other is information about how robust the finding is. ADR 0023 already requires
ZIP to be read with the same caveat, so auditing it costs no additional claim.

**Construct a demographic group by joining an external ACS table.** The thing that would make
this a protected-class audit rather than a geographic one. Rejected here: it is a Component 1
ingestion with its own provenance, vintage and boundary-alignment questions, and ADR 0023
already recorded that adopting a demographic fairness metric "needs data this project has not
ingested". Doing it badly inside Component 12 would be worse than not doing it, because the
resulting table would carry the vocabulary of a protected-class finding.

## Consequences

- **This is a geographic group audit, not a protected-class fairness certification**, and
  every document says so in those words. Community areas correlate strongly with race and
  income by construction — that is what the city uses them for — but a correlate is not the
  attribute, and no result here supports a statement about a protected class.
- The `neural/` layer now has a second reader. ADR 0022's rule that nothing there may be
  joined onto a **feature table** is untouched: Component 12 joins it onto *predictions*, for
  reporting, and never onto anything trainable. The contract is amended to say so rather than
  left ambiguous.
- If Component 8's categorical artifact is ever deleted, as ADR 0022 says it should be when
  Component 4 adopts the families, Component 12's group frame must be re-pointed at the
  Component 4 columns in the same change.
- Ward, census tract and point geography stay refused until someone re-opens this ADR. The
  ward measurement is re-derived by a test, so the refusal cannot outlive its reason silently.
