# Entity resolution: empirical findings

Component 2 investigation. Every number below was measured against one specific
snapshot with `scripts/profile_entities.py`. Nothing here is assumed, inferred
from a sample, or carried over from documentation.

**Read this before changing `src/sentinel/entity/`.** The rules in that package
exist because of specific measurements recorded here, and several of them are
deliberately the *opposite* of the textbook default.

---

## 0. The snapshot these measurements describe

| Property | Value |
|---|---|
| Command | `uv run sentinel ingest --full --log-level INFO` |
| File | `data/raw/food_inspections/food_inspections_20260816T070911Z.parquet` |
| sha256 | `7d3c4069340a68d197204c6cca9fca6399c6565bc3668760f145f43cd377ad38` |
| Rows | 314,245 |
| Columns | 22, all `Utf8` (ADR 0002) |
| Size | 48,801,672 bytes (zstd) |
| Pages | 7 (6 × 50,000 + a 14,245-row short page that terminated pagination) |
| Date range | 2010-01-04 → 2026-08-14 |
| Wall time | 69 m 24 s |
| Retries | 1 (`ReadTimeout` at `$offset=100000`, recovered on attempt 1 of 4) |
| Peak RSS | ~966 MB (sampled during page 6) |

**Every measurement in this document is a measurement of that sha256 and no
other file.** The dataset is live; a later pull will differ.

Two operational notes from the run:

- The 69-minute wall time is almost entirely server-side. Page 3 alone took 62
  minutes including one read timeout. `Settings.request_timeout` is a per-socket-
  operation timeout, not a deadline for the whole request, so a slowly trickling
  response can hold a connection far longer than 60 s without tripping it. Worth
  knowing before anyone treats `--full` as a quick command.
- Peak memory came in well under the level that would have justified refactoring
  Component 1's accumulate-then-concatenate ingestion. It was left alone.

The previously committed 5,000-row development extract was **not** used for any
measurement here. It is the oldest 5,000 rows by `inspection_id` and its
distributions are not representative.

---

## 1. Data quality overview

| Field | Null | Blank | Notes |
|---|---|---|---|
| `inspection_id` | 0 | 0 | 314,245 rows, 314,245 distinct, all numeric |
| `dba_name` | 0 | 0 | 35,062 distinct |
| `aka_name` | 2,419 | 2,419 | 0.77% missing |
| `license_` | 19 | 0 | plus 831 rows of the `'0'` sentinel |
| `address` | 0 | 3 | 33,261 distinct raw strings |
| `zip` | — | — | see §9 |
| `latitude`/`longitude` | 1,042 | — | 0.33%; **0 unparseable, 0 outside the Chicago bounding box** |
| `facility_type` | — | 5,345 | blank is a real value here |

`inspection_id` is unique across all 314,245 rows and is always numeric. The
assignment table can therefore use it as a primary key without qualification.

---

## 2. Licence completeness (§5.1)

```
total_rows            314,245
null license               19   (0.006%)
empty string                0
whitespace-only             0
license_ = '0'            831   (0.264%)
non-numeric                 0
leading-zero forms          0
distinct license_      48,964
```

**The `'0'` sentinel is garbage, and provably so.** Those 831 rows carry:

| license value | rows | distinct names | distinct addresses | distinct zips |
|---|---|---|---|---|
| `0` | 831 | 323 | 364 | 56 |
| `<NULL>` | 19 | 7 | 9 | 7 |

323 unrelated businesses across 364 addresses and 56 zip codes share the literal
value `'0'`. Any rule that treated licence equality as identity evidence would
merge all of them into a single establishment.

> **Decision.** `license_is_usable` ⇔ non-null, non-blank after trim, and not
> `'0'`. 850 rows (0.27%) fail this and carry no licence evidence at all. They
> can still resolve on name and address.

Licence length distribution is bimodal — 255,962 rows at 7 digits, 48,516 at 5 —
reflecting two eras of licence numbering. This is cosmetic and needs no handling.

---

## 3. Can `license_` be the entity key? (§5.2)

No. Two independent measurements rule it out, in **opposite directions**.

### 3.1 One licence can cover several successive businesses

| attribute | licences | % single-valued | p90 | p99 | p99.5 | max |
|---|---|---|---|---|---|---|
| `dba_name` | 48,963 | 97.38 | 1 | 2 | 2 | 7 |
| `address` | 48,963 | 64.91 | 2 | 2 | 2 | 5 |
| `zip` | 48,963 | 99.75 | 1 | 1 | 1 | 2 |
| `facility_type` | 48,963 | 90.12 | 1 | 1 | 1 | 4 |

97.4% of licences carry one name, so licence is *mostly* stable. But the 2.6%
that are not include genuine business turnover, not formatting noise:

| licence | address | names over time |
|---|---|---|
| 1514802 | 1208 N WELLS ST | OLD TOWN BURGER SALOON → RABBIT HOLE → Saluki Bar → TAVERN ON WELLS |
| 2088536 | 4810 N BROADWAY | CAIRO → CARAVAN → LE NOCTURNE |
| 2220904 | 355 E OHIO ST | FLOUR & STONE → L AVENTINO → STREETERVILLE PIZZERIA & TAP → URBAN CROSTA |
| 1932 | (2 addresses) | ST ELIZABETH ELEMENTARY SCHOOL → STETSON'S |

Licence 1932 is the clearest failure: a school and a bar share one licence
number across two addresses. Grouping by licence would fuse them.

The `address` column looks alarming at 64.91% single-valued, but §7 shows most
of that is trailing-whitespace duplication, not real movement.

### 3.2 One establishment routinely holds *many* licences — the bigger problem

This is the finding that most changed the design.

```
(name, address) pairs         40,161
  with 1 licence              32,744
  with 2 licences              5,688
  with 3-5 licences            1,616
  with more than 5               113
  percent with >1            18.47%
  max licences                    47
```

**18.47% of apparent establishments hold more than one licence**, and the tail
is extreme:

| name | address | licences | rows |
|---|---|---|---|
| TRIPLE A SERVICES, INC. | 2637 S THROOP ST | 47 | 64 |
| BAMBI | 2051 W 47TH ST | 23 | 33 |
| THUNDERBIRD CATERING | 1204 W 36TH PL | 22 | 36 |
| UNITED CLUB | 11601 W TOUHY AVE | 21 | 108 |
| PALETERIA AZTECA #2 | 3119 W CERMAK RD | 19 | 36 |
| CHICAGO HILTON & TOWERS | 720 S MICHIGAN AVE | 16 | 87 |
| MCDONALD'S | 11601 W TOUHY AVE | 12 | 67 |

A licence in this dataset is frequently **finer-grained than an establishment**.
Mobile-food operators hold one licence per cart or truck, all registered to a
single commissary address; hotels and airport terminals hold one licence per
food outlet. `TRIPLE A SERVICES` is not 47 businesses, it is one commissary with
47 permits.

> **Decision.** `license_` is neither necessary nor sufficient for identity.
> Licence equality is *supporting* evidence, never the key. Crucially, licence
> *inequality* is **not** evidence against a match — that would fracture 18% of
> establishments.

### 3.3 Does one licence wander geographically?

| coordinate span within a licence | licences | % |
|---|---|---|
| exactly 0 m | 48,502 | 99.54 |
| 0–25 m | 65 | 0.13 |
| 25–50 m | 29 | 0.06 |
| 50–250 m | 27 | 0.06 |
| 250 m – 1 km | 18 | 0.04 |
| 1–10 km | 44 | 0.09 |
| over 10 km | 42 | 0.09 |

99.5% of licences never move. The 86 licences spanning more than 1 km are real
relocations or reassignments, and are exactly the cases where merging on licence
alone would join two physically distant places.

---

## 4. Name variation (§5.4)

```
distinct dba_name                        35,062
after upper() + trim()                   34,672   (-1.1%)
after stripping all non-alphanumerics    33,749   (-3.7%)
```

Name normalization has **low yield**. Contrast this with addresses (§7), where
the same operations collapse 39%.

Measuring it where it matters — how many licences carry exactly one name:

| stage | licences with a single name |
|---|---|
| raw | 47,682 |
| after `upper()`/`trim()` | 47,714 |
| after alphanumeric-only | 47,770 |
| after stripping LLC/INC/CORP/LTD | 47,785 |

Full name normalization resolves **103 licences out of 48,963 (0.21%)**. The
remaining multi-name licences are genuine business turnover, not formatting.

> **Conclusion.** Name normalization is worth doing — it is cheap and it does
> fix real cases like licence 14616 (`ILLINOIS SPORTSERVICE INC` / `ILLINOIS
> SPORTSERVICE, INC` / `ILLINOIS SPORT SERVICE INC.` → 7 strings collapse to 2)
> — but it is not where the leverage is. **Aggressive fuzzy name matching would
> buy almost nothing and risk a great deal.** See §12.

### 4.1 Corporate suffixes, validated against the data

Trailing tokens across distinct names, from `name_trailing_tokens`:

| token | distinct names | corporate? |
|---|---|---|
| INC | 4,454 | yes |
| RESTAURANT | 1,213 | **no — descriptive** |
| LLC | 1,175 | yes |
| CAFE | 766 | **no** |
| MART | 762 | **no** |
| CORP | 213 | yes |
| CO | 168 | yes |
| COMPANY | 152 | yes |
| CORPORATION | 114 | yes |
| LTD | 106 | yes |

The plan's proposed suffix list is confirmed by frequency: `INC`, `LLC`, `CORP`,
`CO`, `COMPANY`, `CORPORATION`, `LTD` all clear 100 distinct names. Descriptive
trailing words like `RESTAURANT`, `CAFE`, `MART` and `GRILL` are *more* common
than most corporate suffixes and must **not** be stripped — they carry real
distinguishing meaning (`JOE'S GRILL` ≠ `JOE'S MART`).

`S` appears as a trailing token in 595 distinct names and as a token in 4,846.
That is possessive splitting (`JOE'S` → `JOE S`), which is why possessives must
be folded *before* punctuation is replaced by spaces.

### 4.2 Chain names are pervasive

| name | distinct addresses | licences | rows |
|---|---|---|---|
| SUBWAY | 247 | 369 | 4,272 |
| DUNKIN DONUTS | 184 | 226 | 2,203 |
| 7-ELEVEN | 78 | 101 | 655 |
| MCDONALD'S | 66 | 101 | 1,019 |
| MCDONALDS | 40 | 49 | 541 |
| CITGO | 58 | 66 | 276 |

`MCDONALD'S` and `MCDONALDS` are separate raw values that normalization
correctly unifies — and that unification is exactly what makes a name-only match
rule catastrophic. There are 247 Subways.

> **Decision.** No name-only blocking and no name-only merge rule, ever. Every
> non-licence merge requires address equivalence. Chains are then harmless
> *across* addresses; the residual risk is two same-name outlets at one address,
> handled in §12.

`TBD` occurs as a `dba_name` at 19 distinct addresses — a placeholder, not a
business. Generic-name spread is measured rather than hand-listed for this
reason.

---

## 5. `aka_name` (§5.5)

```
total rows                                 314,245
null / blank                                 2,419   (0.77%)
identical to dba_name                      228,480   (72.7%)
differs from dba_name                       83,346   (26.5%)
  of which identical after normalization      4,893
```

The surprise is *how* they differ. `aka_name` is frequently the **trade name**
while `dba_name` is the **legal entity**:

| `dba_name` | `aka_name` |
|---|---|
| 1918 WINTER STREET ILLINOIS LLC | MARIANO'S |
| 1053 W 103RD INC | CITGO |
| 1250 SOUTH LLC | CRAVE CAFE & LOUNGE |
| 14 W. HUBBARD, LLC | LAO EIGHTEEN |
| 1910 N MILWAUKEE OPERATIONS, LLC | REMEDY |
| 103 CHICKEN & MORE, INC. | KRISPY KRUNCHY CHICKEN |
| #1 CHOP SUEY RESTAURANT, INC | #1 CHOP SUEY |

This matters because the legal entity changes on ownership transfer while the
physical restaurant does not. A resolver that only reads `dba_name` will split an
establishment every time the holding company is renamed.

But `aka_name` is also *more* generic than `dba_name` in places — `CITGO`,
`FOODA`, `SUBWAY` — so matching on it across addresses would be worse than
matching on `dba_name`.

> **Decision.** Treat `aka_name` as a **second name candidate** for the same
> record. Two records match on name if any of their normalized name keys agree
> (dba↔dba, aka↔aka, or dba↔aka). Because every non-licence rule already
> requires address equivalence, the generic-alias risk is confined to a single
> address and is acceptable. `aka_name` is never used for blocking.

---

## 6. Address completeness

```
total rows                          314,245
null address                              0
blank address                             3
rows with no leading house number        11
distinct raw address strings         33,261
distinct after upper() + trim()      20,313
```

Only 14 rows in 314,245 lack a usable address. Address is the most complete
identity field in the dataset.

---

## 7. Address variation (§5.6) — where the real leverage is

**`upper()` + whitespace collapse alone reduces 33,261 distinct address strings
to 20,313 — a 39% collapse.** This single measurement explains why the licence
→ address fan-out in §3.1 looked so bad: most "multiple addresses per licence"
cases are the same string with a trailing space.

| licence | recorded addresses |
|---|---|
| 1354323 | `1410 S MUSEUM CAMPUS `, `1410 S MUSEUM CAMPUS DR`, `1410 S MUSEUM CAMPUS DR `, `1410 S MUSEUM CAMPUS DR. `, `425 E MC FETRIDGE BLDG ` |
| 2488196 | `4701 N KEDZIE AVE BLDG`, `4701 N KEDZIE AVE BLDG `, `4707-4713 N KEDZIE AVE`, `4707-4713 N KEDZIE AVE ` |

Pattern census over the 20,312 distinct upper/trimmed addresses:

| pattern | count | % |
|---|---|---|
| short directional (`N`/`S`/`E`/`W`) | 17,818 | 87.7 |
| **long directional (`NORTH`/`SOUTH`/…)** | **1** | **0.005** |
| unit word (`STE`/`SUITE`/`APT`/`FL`/`BLDG`/…) | 399 | 1.96 |
| `#` marker | 1 | 0.005 |
| ranged house number (`4707-4713`) | 2,231 | 10.98 |
| fractional (`1234 1/2`) | 63 | 0.31 |
| ampersand | 13 | 0.06 |
| period | 47 | 0.23 |
| internal double space | 87 | 0.43 |
| non-ASCII | 0 | 0 |

Final-token census (street suffixes):

| token | distinct addresses |
|---|---|
| AVE | 9,397 |
| ST | 6,960 |
| RD | 1,285 |
| BROADWAY | 362 |
| BLVD | 325 |
| DR | 265 |
| BLDG | 189 |
| PL | 160 |
| PKWY | 103 |

**There are no long-form street suffixes.** No `STREET`, no `AVENUE`, no
`BOULEVARD` anywhere in the top 40. The city's data entry is already
USPS-abbreviated.

> **Decision, contradicting the plan.** The large street-suffix and directional
> canonicalization tables the plan called for would be **dead code**. One address
> in 20,312 has a long directional and zero have long suffixes. A minimal mapping
> is retained for safety, but it is explicitly documented as near-zero-yield
> rather than presented as load-bearing.

### 7.1 The suffix problem is *absence and disagreement*, not long forms

The real address defect is that the suffix is sometimes missing or simply wrong:

| normalized without suffix | recorded variants |
|---|---|
| 1901 W MADISON | `1901 W MADISON`, `1901 W MADISON AVE`, `1901 W MADISON ST` |
| 2300 S THROOP | `2300 S THROOP`, `2300 S THROOP AVE`, `2300 S THROOP ST` |
| 135 N KEDZIE | `135 N KEDZIE`, `135 N KEDZIE AVE`, `135 N KEDZIE ST` |
| 324 N LEAVITT | `324 N LEAVITT`, `324 N LEAVITT AVE`, `324 N LEAVITT ST` |
| 600 E GRAND | `600 E GRAND AVE`, `600 E GRAND ST` |

`1901 W MADISON` is the United Center. `2300 S THROOP` and `324 N LEAVITT` are
two of the largest shared-kitchen addresses in the city (§10). These are not
edge cases; they are among the busiest addresses in the dataset.

Dropping the final suffix token from the address key collapses 20,317 keys to
20,184 — **133 addresses merged**. Inspecting every group with 3+ variants
showed only genuine same-place variants and no false merges.

> **Decision.** The street suffix is **excluded from `addr_key`** and retained
> as a separate attribute. The risk that this fuses two distinct streets is
> negligible in Chicago's grid, where directional + street name is effectively
> unique, and `addr_key` additionally pins the house number and zip.

Other real patterns requiring handling: ranged house numbers in three separate
styles (`4749-4753`, `4749-51`, `3000 -3002`, `436 - 440`); ordinals split by a
space (`2100 W 22 ND PL` vs `2100 W 22ND PL`); trailing floor/building junk
(`2333 N MILWAUKEE AVE BLDG 1STFL.& BSMT.`); and one genuine street rename
(`5700 S LAKE SHORE DR` → `5700 S JEAN BAPTISTE POINTE DUSABLE LAKE SHORE DR`,
Chicago, 2021), which string normalization cannot fix but §8 can.

---

## 8. Geographic evidence (§5.7) — better than expected, and used differently

```
rows                                    314,245
null latitude                             1,042   (0.33%)
unparseable                                   0
outside the Chicago bounding box              0
distinct coordinate pairs                18,930
```

Latitude precision is 13–15 decimal places for 98.7% of rows. The coordinates
are not rounded, not truncated, and not centroid-snapped at any material rate:
only 10 coordinate pairs are shared by more than 3 distinct addresses, the worst
being 7.

The decisive measurement:

```
geo_spread_within_address:  20,117 addresses,  max span = 0.00 m
```

**Every distinct address string maps to exactly one coordinate pair, with zero
variance.** The coordinate is a deterministic function of the address string as
recorded — it carries no independent surveying information.

That sounds like it makes geo useless. It does the opposite. Because the city
geocodes *before* the string variation is introduced, address variants collapse
to a single coordinate:

| address key (suffix dropped) | zip | distinct strings | distinct coordinates |
|---|---|---|---|
| 1901 W MADISON | 60612 | 3 | **1** |
| 135 N KEDZIE | 60612 | 3 | **1** |
| 2300 S THROOP | 60608 | 3 | **1** |
| 324 N LEAVITT | 60612 | 3 | **1** |
| 2502 1 2 W DEVON | 60659 | 2 | **1** |

**The geocoder is a better address normalizer than any string transformation I
could write**, because it resolves cases string rules cannot — including the
Lake Shore Drive rename.

False-merge risk, measured. Grouping the 18,930 coordinates by how many distinct
normalized address keys each covers:

| coordinate covers | count | % |
|---|---|---|
| 1 address key | 18,061 | 95.41 |
| 2 address keys | 816 | 4.31 |
| more than 2 | 46 | 0.24 |
| worst case | 4 | |

4.6% of coordinates span more than one address key, so **coordinate equality
alone is not sufficient for a merge**.

> **Decision.** Exact coordinate equality (string equality on the raw
> `latitude`/`longitude`, plus zip agreement) is accepted as an *alternative
> route to address equivalence*, never as a merge reason on its own. It only
> ever fires inside a rule that independently requires name or licence evidence.
> Coordinates shared by more than 4 distinct address keys are blacklisted as
> geocoder artefacts.
>
> Note this also means **geographic distance thresholds are pointless here**.
> The plan called for deriving `D_veto`/`D_warn` from a distance distribution;
> that distribution is degenerate (all zeros within an address). Distance-based
> vetoes are replaced by exact-equality evidence plus a cluster-level bounding
> box check that exists only to catch bugs.

---

## 9. Zip and other supporting fields

`zip` is single-valued for 99.75% of licences (max 2). It is a reliable
partitioning key and is included in `addr_key`.

`facility_type` is single-valued for **99.54%** of licences (44,127 of 44,331
with a non-blank value; max 4). Value casing is inconsistent (`Restaurant` vs
`TAVERN` vs `GAS STATION`) and 5,345 rows are blank.

> **Decision.** `facility_type` is a good corroborator — stable enough to
> demand agreement in the weaker merge rules — but never a primary key, and a
> blank value must count as "unknown", never as a mismatch.

`city`/`state` are near-constant and contribute nothing to identity.

---

## 10. Address density — the mega-address problem

| address | zip | licences | names | rows |
|---|---|---|---|---|
| 11601 W TOUHY AVE | 60666 | 417 | 219 | 3,910 |
| 324 N LEAVITT ST | 60612 | 194 | 169 | 469 |
| 2300 S THROOP ST | 60608 | 192 | 188 | 809 |
| 5700 S CICERO AVE | 60638 | 132 | 84 | 904 |
| 131 N CLINTON ST | 60661 | 103 | 96 | 586 |
| 600 E GRAND AVE | 60611 | 99 | 66 | 417 |

`11601 W TOUHY AVE` is O'Hare International Airport: 219 distinct business names
at one address, including 12 licences for `MCDONALD'S` and 21 for `UNITED CLUB`.
`324 N LEAVITT` and `2300 S THROOP` are shared commercial kitchens. `600 E GRAND`
is Navy Pier.

These addresses are legitimate, not errors, and they are where a same-address
merge rule is most dangerous: two `MCDONALD'S` in different O'Hare terminals
share a name, an address, and a zip.

> **Consequence.** Cluster-size and address-density checks must *report* these
> rather than fail on them, and the pair-evaluation cost must be bounded (a
> 417-member block yields ~87,000 pairs — cheap, but the bound must exist so a
> pathological future block cannot blow up).

---

## 11. Temporal behaviour (§5.8)

Licence pairs attached to the same (name, address):

```
pairs                     17,847
temporally disjoint        4,365   (24.5%)
overlapping               13,482   (75.5%)
```

**Three quarters of multi-licence establishments hold their licences
concurrently, not successively.** This directly refutes the "licence
renumbering / succession" model the plan assumed. Concurrency is the norm
because, per §3.2, the extra licences are per-cart and per-outlet permits.

Business succession at one licence is also real but rarer (§3.1), and when it
happens it looks like this at a fixed address:

| address | licence | names in time order |
|---|---|---|
| 1208 N WELLS ST | 1514802 | OLD TOWN BURGER SALOON → RABBIT HOLE → SALUKI BAR → TAVERN ON WELLS |
| 4810 N BROADWAY | 2088536 | CAIRO → CARAVAN → LE NOCTURNE |
| 15 W DIVISION ST | 1196 | FINN MC COOL'S / ALUMNI CLUB → HOPSMITH TAVERN → SNUGGERY / APARTMENT |
| 7600 S EXCHANGE AVE | 1494617 | CITGO → GOLO FUEL MINI MART → WMI SYRIA INC |

> **Decision, contradicting the plan.** The planned successor rule (`same
> address ∧ temporally disjoint licence spans ∧ similar name`) is **dropped**.
> Disjointness describes only a quarter of the multi-licence population, so it
> is a weak discriminator; and the cases it would catch are exactly the
> successive-different-business cases above, which we do **not** want to merge
> on name similarity. Temporal spans are recorded in the output for downstream
> use but do not drive any merge decision.
>
> This also removes the last use of `inspection_date` from the matching logic,
> which is a leakage benefit (§14).

### 11.1 What "the same establishment" is taken to mean

The turnover examples force the question explicitly. When `1208 N WELLS ST`
houses four bars in fifteen years, is that one establishment or four?

Sentinel exists to decide **where to send an inspector**. The unit of
inspection is a physical food-handling premises. A new tenant does not make the
kitchen, the grease trap or the walk-in cooler new.

> **Definition adopted.** An establishment is a **physical food-service
> premises**, identified by location. Successive businesses at one premises are
> the *same* establishment with a changing name. Concurrent permits at one
> premises are the *same* establishment with multiple licences.
>
> This is a deliberate choice with a real cost: it means an establishment's
> history can span a change of owner, cuisine and name. Component 3 must not
> assume behavioural continuity across such a transition, and the output
> therefore exposes name-change and licence-change counts so downstream code can
> detect it. The alternative — treating each business as a separate entity —
> would discard the physical-premises risk signal that motivates the project.

---

## 12. False-merge and false-split risk

### False merges (the dangerous direction)

| risk | evidence | mitigation |
|---|---|---|
| Two chain outlets at one mega-address (`MCDONALD'S` ×12 at O'Hare) | §10 | store-number digits are retained in the name key and a digit conflict vetoes; identical-name-identical-address outlets at a *dense* address are the residual risk, accepted and reported |
| Generic names (`TBD` at 19 addresses, `RESTAURANT`) | §4.2 | descriptive tokens are never stripped; no name-only rule exists |
| The `'0'` licence sentinel (323 names) | §2 | hard-excluded from licence evidence |
| A coordinate covering 2–4 address keys | §8 | coordinate equality never merges alone; blacklist above 4 |
| Different suites in one building | §7 | unit is extracted and retained; unit conflict vetoes |

### False splits (the costly direction)

| risk | evidence | mitigation |
|---|---|---|
| Trailing whitespace on the address | §7 (39% collapse) | whitespace normalization |
| Missing or wrong street suffix | §7.1 (133 addresses) | suffix excluded from `addr_key` |
| Ranged house-number styles | §7 (2,231 addresses, 3 styles) | ranges normalized to the low endpoint |
| Legal entity renamed, premises unchanged | §5 (MARIANO'S / 1918 WINTER STREET ILLINOIS LLC) | `aka_name` used as a second name key |
| Licence renumbered or split into per-cart permits | §3.2 (18.47%) | licence inequality is never evidence against a match |
| Corporate suffix drift (`X` vs `X LLC`) | §4.1 | trailing corporate suffixes stripped |
| Street renamed (Lake Shore Drive) | §7.1 | coordinate equality bridges it |

---

## 13. Recommended resolution strategy

Follows directly from the measurements above.

1. **Unit of matching is a node**, not an inspection: the distinct tuple
   (licence key, name keys, address key, unit, zip, coordinate). 314,245 rows
   collapse to far fewer nodes, and the audit table stays readable.
2. **Address equivalence is required for every non-licence merge.** Two nodes at
   different places never merge on name. This is what makes 247 Subways safe.
   Address equivalence means equal `addr_key`, *or* equal coordinate with equal
   zip (§8).
3. **Licence equality is supporting evidence, not the key** (§3). Licence
   inequality is never evidence against.
4. **Name matching is exact-after-normalization**, over both `dba_name` and
   `aka_name` (§4, §5). Fuzzy similarity is deliberately *not* used: it would
   resolve 0.21% of licences (§4) while putting the chain and mega-address cases
   (§4.2, §10) at risk. Token-containment is admitted only as a weaker tier at
   an identical address.
5. **Vetoes beat everything**: directional conflict, unit conflict, store-number
   digit conflict.
6. **Clustering is union-find over accepted edges**, with cluster invariants and
   a deterministic degradation ladder if an invariant fails.
7. **No temporal logic in matching** (§11), which keeps identity reconstruction
   cleanly separated from as-of feature availability.

### Thresholds

The plan called for a 100-pair manual calibration to set a fuzzy name-similarity
threshold `T1`. §4 makes that exercise unnecessary and the design retires it:
with fuzzy name matching removed, there is no similarity threshold to calibrate.
The plan explicitly named `T1 = 1.0` (exact match only) as the documented
fallback if calibration did not justify a fuzzy tier; the measurements retired
the tier before calibration rather than after. This is recorded in ADR 0006 as a
measured outcome, not an omission.

The parameters that remain are structural, and each is set from a measurement:

| parameter | value | derived from |
|---|---|---|
| licence sentinel set | `{'0'}`, null, blank | §2 (323 names under `'0'`) |
| max address keys per cluster | 4 | §3.1 (max 5 raw, mostly whitespace duplicates) |
| max distinct zips per cluster | 1 | §9 (99.75% single) |
| coordinate blacklist threshold | > 4 address keys | §8 (max observed 4) |
| max block size | 5,000 | §10 (largest real block 417) |
| bounding-box sanity radius | 2,000 m | §8 (within-address span is 0 m; this only catches bugs) |

---

## 14. Temporal leakage

Three distinctions, stated because Components 3–5 depend on them.

1. **Retrospective identity reconstruction is legitimate.** Resolving that a
   2012 inspection happened at the premises now called *Tavern on Wells* uses
   the whole snapshot, including later rows. That is not leakage: identity is a
   property of the place, not a prediction about it. The building at 1208 N
   Wells was the same building in 2012 whether or not we learned it in 2025.

2. **As-of feature availability is a different thing entirely.** Any *quantity*
   per establishment — inspection counts, prior failure rates, days since last
   inspection — is only meaningful relative to a reference date. Component 2
   computes none of these, and the assignments table deliberately carries no
   dates, counts or outcomes, so a downstream join cannot accidentally pull a
   full-history aggregate into a training row.

3. **The residual channel, and its bound.** Cluster membership does encode
   information distributed across the whole history. Two bounds are enforced:
   - Matching reads **identity fields only**: `license_`, `dba_name`,
     `aka_name`, `address`, `zip`, `latitude`, `longitude`, `facility_type`.
     **`results`, `violations` and `risk` are never inputs to matching.**
     Following §11, `inspection_date` is not an input either.
   - Aggregate columns on the establishments table are marked audit-only in the
     data contract and are not model features.

   A strict `--as-of DATE` mode (resolve using only rows on or before a date) is
   named as future work; it would cost one resolution run per evaluation fold.

---

## 15. What changed relative to the pre-investigation plan

Recorded because the plan is in the repository's history and the divergences are
the point of doing the investigation.

| Plan assumed | Data showed | Change |
|---|---|---|
| Licence may be too coarse (several businesses per licence) | It is mostly too **fine** — 18.47% of establishments hold several licences, up to 47 | Licence demoted to corroborator; licence inequality never counts against a match |
| Large street-suffix / directional canonicalization tables needed | 1 long directional and 0 long suffixes in 20,312 addresses | Tables reduced to a documented near-no-op |
| Suffixes should be canonicalized long → short | The defect is *missing and conflicting* suffixes | Suffix excluded from the address key entirely |
| Geo needs a distance threshold, may be centroid-pinned | Within-address span is exactly 0 m; coordinates are a clean function of the address | Distance thresholds dropped; exact coordinate equality adopted as an address-equivalence route |
| A successor rule keyed on disjoint licence spans | 75.5% of same-place licence pairs *overlap* | Rule dropped; no temporal logic in matching |
| Fuzzy name similarity with a calibrated threshold `T1` | Full name normalization resolves 0.21% of licences | Fuzzy tier dropped; exact-after-normalization only |
| 100-pair manual calibration | Nothing left to calibrate | Retired, with the reasoning recorded here and in ADR 0006 |

---

<!-- §16 is appended after the resolver has been run against this snapshot. -->

## 16. Post-resolution results

Written after running the finished resolver against the same snapshot
(`sentinel resolve --report`, 43 s wall time). This closes the loop between the
investigation above and what the implementation actually produced.

### 16.1 Headline

| Quantity | Value |
|---|---|
| Input rows | 314,245 |
| Distinct nodes (identity signatures) | 51,099 |
| **Establishments** | **35,859** |
| Distinct usable licences (for comparison) | 48,963 |
| Reduction ratio (establishments ÷ licences) | 0.73 |
| Candidate pairs evaluated | 335,393 |
| Oversized blocks skipped | 0 |
| Blacklisted coordinates | 0 |
| Rows with no usable licence | 850 |
| Distinct address keys | 19,287 |

The reduction ratio is the key sanity number. At ≈1.0 the matcher would be doing
nothing; far below, it would be over-merging. 0.73 says roughly a quarter of
licences were folded into an establishment that already existed — consistent
with §3.2's measurement that 18.47% of places hold more than one licence.

### 16.2 Edge outcomes

| rule | tier | pairs | meaning |
|---|---|---|---|
| V3 | no_match | 54,462 | conflicting store numbers |
| S2 | strong | 21,381 | same place, same name |
| S1 | strong | 7,869 | same place, same licence |
| V4 | no_match | 2,997 | shared operator name, disagreeing trade names |
| P2 | probable | 2,908 | name containment at one place |
| A2 | ambiguous | 375 | containment with conflicting facility types |
| A1 | ambiguous | 372 | one licence at two different places |
| V1 | no_match | 239 | conflicting directionals |
| S3 | strong | 30 | licence + name, house number off by ≤2 |
| P1 | probable | 7 | licence, house number off by ≤2 |
| V2 | no_match | 3 | conflicting suites |

**S2 does most of the work** (21,381 of 29,280 strong edges). That is the rule
that repairs the multi-licence establishments, and it is also the rule carrying
the most false-merge risk — which is why V3 and V4 exist and why V3 fires more
often than any merge rule.

**Only 747 pairs are ambiguous**, a genuinely reviewable queue. An earlier
iteration classified "same address, same facility type, unrelated names" as
ambiguous and produced 108,597 — inspecting those showed ordinary strip-mall
neighbours and successive tenants, so they were reclassified as rule N2, a
decision rather than a doubt.

`V2` firing only 3 times confirms §7's prediction: units appear on 1.96% of
addresses, so the unit veto is nearly inert. It was retained because when it
does fire it is right, and it costs nothing.

### 16.3 Confidence and tiers

| establishment confidence | count |
|---|---|
| high (strong edges only) | 34,401 |
| medium (at least one probable edge) | 1,438 |
| reduced (survived a cluster split) | 20 |

| assignment row tier | rows |
|---|---|
| singleton | 176,076 |
| high | 110,686 |
| medium | 27,436 |
| reduced | 47 |

Only 20 clusters out of 35,859 tripped an invariant and had to be rebuilt — all
of them on `conflicting_units`, all resolved by the degradation ladder.

### 16.4 Inspection history per establishment

| inspections | establishments |
|---|---|
| 1 | 6,084 (17.0%) |
| 2–5 | 12,448 |
| 6–20 | 13,018 |
| more than 20 | 4,309 |
| maximum | 286 |
| mean | 8.76 |

**The single-inspection rate fell from 12,356 (licence-only grouping) to 6,084 —
a 51% reduction.** That is the clearest evidence the resolver is recovering real
history: half the establishments that looked like one-off inspections under a
naive licence grouping are actually places with a longer record under another
licence or spelling. Component 3 has materially more history to work with as a
direct result.

8,931 establishments (24.9%) hold more than one licence, the largest holding 62.
3,071 (8.6%) have carried more than one name, the largest 13.

### 16.5 Largest establishments, checked by hand

| establishment | nodes | inspections | licences | names | place |
|---|---|---|---|---|---|
| EST-00000068356 | 86 | 185 | 1 | 3 | Illinois Sportservice, Guaranteed Rate Field |
| EST-00000058536 | 68 | 105 | 16 | 3 | The United Center |
| EST-00000068276 | 64 | 84 | 62 | 6 | Triple A Services commissary |
| EST-00000112420 | 53 | 191 | 1 | 3 | Sportservice, Soldier Field |
| EST-00000068349 | 34 | 54 | 32 | 1 | Thunderbird Catering |
| EST-00001152088 | 23 | 33 | 23 | 1 | Bambi (pushcart operator) |

Every one is a real single premises: two stadiums, an arena, and three
mobile-food commissaries whose many licences are per-cart permits. None is a
chained over-merge.

### 16.6 The over-merge this run caught, and the fix

The **first** full run produced a 47-node cluster at O'Hare containing 23
distinct business names — Starbucks in Terminal 3, Johnny Rockets in Terminal 2,
Chili's Too, Goose Island, La Tapenade, Brioche Dorée and more. All carried
`dba_name = HOST INTERNATIONAL INC`, the concessionaire, with the actual
identity in `aka_name`. Rule S2 matched them on the operator name and transitive
merging chained the rest.

This is precisely the false merge §12 warned about: it would have pooled about
twenty restaurants' inspection histories onto one identity and corrupted every
downstream feature for all of them.

Veto **V4** was added in response: when the name evidence driving a merge is a
shared name but the trade names actively disagree, the pair does not merge. It
is waived when the licence agrees, because one licence at one address is a
single premises even across a rename. After the fix those 291 rows resolve to 14
establishments, one per outlet, each retaining its multi-year history.

A first version of V4 was too broad — it fired on any two differently-named
neighbours and would have blocked legitimate containment merges such as
`HOT WOK CHINESE KITCHEN` / `NEW HOT WOK CHINESE KITCHEN`. The unit test suite
caught that before it reached the data. The shipped version requires that name
evidence is actually driving the merge, and lets trade names corroborate by
containment as well as equality.

**Both the over-merge and the over-correction are now regression tests**
(`tests/fixtures/real_cases.py`).

### 16.7 Output artifacts

| file | rows | bytes |
|---|---|---|
| `establishment_assignments_20260816T085729Z.parquet` | 314,245 | 5,933,587 |
| `establishments_20260816T085729Z.parquet` | 35,859 | 2,068,779 |
| `entity_resolution_edges_20260816T085729Z.parquet` | 90,643 | 1,271,247 |
| `manifest_establishment_assignments_20260816T085729Z.json` | — | 6,062 |

All nine error-severity validation checks pass. Determinism was verified
directly on the full snapshot: resolving a seeded random permutation of all
314,245 input rows produced a byte-identical `inspection_id → establishment_id`
mapping.

### 16.8 What remains unresolved

- **Chains at mega-addresses.** Two same-name outlets at one address with no
  store number and no distinguishing trade name will still merge. `MCDONALD'S`
  at O'Hare is 22 nodes across 20 licences and 5 names; some of that may be
  more than one physical counter. This is the residual false-merge risk and it
  is bounded to dense addresses.
- **Stadiums and arenas.** The United Center resolves to one establishment
  holding 16 licences. Whether an arena is one premises or many is a genuine
  definitional question, not a bug; the current answer follows the
  physical-premises definition in §11.1.
- **A2 and A1 pairs** (747 total) have never been manually adjudicated. They are
  the intended review queue and are recorded in the edges table for exactly
  that purpose.
