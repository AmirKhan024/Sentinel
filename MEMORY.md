# MEMORY

Compact context for future Claude Code sessions. Not a project description —
see README.md for that and STATUS.md for current state.

**Read this first, then STATUS.md, then HANDOFF.md.**

---

## Working agreement

* **One component at a time.** 21 components are planned; Components 1-4
  exist. Never implement ahead. If something belongs to a later component,
  write it down as a TODO or an architectural note instead of building it.
* **No fake completion.** Never claim tests pass, ingestion works, or a schema
  is what it is, without having run the command. Anything unverified must be
  labelled `NOT VERIFIED` with the command that would verify it.
* **Introduce a technology only when the component needing it is being built.**
  This is why there is no pandas, no PostgreSQL, no OR-Tools, no LangGraph.
* **Explain before abstraction.** Simple explicit code beats clever layering.
* Update STATUS.md at every milestone. It must reflect reality.

---

## Hard constraints — do not change without a very good reason

1. **`$order=inspection_id` on every paged request.** Socrata does not
   guarantee row order; without a total order, `$offset` paging can duplicate or
   skip rows. `build_params()` raises if the order column is empty. This is the
   correctness backbone of ingestion.
2. **Raw Parquet is all `Utf8`.** No casting at ingestion, ever. The API returns
   every value as a string; the raw layer preserves that exactly. Typing belongs
   to a later component. (ADR 0002)
3. **`data/raw/` is append-only.** Timestamped filenames, never overwritten. No
   `latest.parquet` pointer — resolving "latest" is a read-time concern handled
   by `latest_parquet()`. (ADR 0005)
4. **Non-retryable 4xx must raise immediately.** Only 429, 5xx, timeouts and
   transport errors are retried. A 400 is our bug; retrying hides it.
5. **No `print()` in `src/sentinel`.** Logging only, except for the CLI's final
   result lines, which are deliberate stdout output.
6. **Live tests stay deselected by default** (`addopts = -m 'not live'`). CI
   must never depend on the Chicago API being reachable.
7. **Never commit raw data.** `.gitignore` excludes Parquet but whitelists
   `manifest_*.json`, so provenance is versioned and bulk data is not.

### Component 2 invariants - violating any of these silently corrupts history

8. **An establishment is a physical premises**, not a licence and not a business
   name. Successive tenants at one address are the *same* establishment with a
   changing name; a commissary holding 47 cart permits is *one* establishment.
   Component 3 must therefore not assume behavioural continuity across a tenant
   change - `n_names` and `n_licenses` are exposed so it can detect one.
9. **`license_` is not the entity key, and licence *inequality* is never
   evidence against a match.** 18.47% of establishments hold more than one
   licence (max 47). A rule keyed on licence agreement fractures them.
10. **Every non-licence merge requires address equivalence.** This is what makes
    247 Subways safe without a chain-name list, and what bounds transitive
    chaining. Do not add a name-only block or a name-only merge rule.
11. **Only identity columns reach the matcher.** `IDENTITY_COLUMNS` in
    `entity/nodes.py` is the leakage boundary; `results`, `violations`, `risk`
    and `inspection_date` are excluded, and a test asserts it.
12. **The assignments table carries no dates, counts or outcomes.** That absence
    is the anti-leakage guarantee. Do not "helpfully" add them.
13. **Resolution must stay deterministic.** Content-hashed node ids, canonically
    ordered pairs, components re-labelled by minimum member, ids from the
    earliest inspection. Verified against a seeded shuffle of all 314,245 rows.
14. **`establishment_id` is snapshot-scoped**, not a durable primary key. A
    later snapshot can merge or split clusters and retire ids (ADR 0007).

### Component 3 invariants — the target is a regulatory statement, not a convenience

15. **The target is: at an eligible routine canvass, was at least one Priority or
    Priority Foundation violation found?** Not `results == 'Fail'` — among
    canvasses, priority violations appear in 97.9% of `Pass w/ Conditions` rows,
    so a result-based label would mislabel 16,261 inspections.
16. **The label is read from the violation text, never from `results`.** Using
    the result would make the target partly circular. `results` is consulted only
    to decide whether a row is *labellable* (a `Fail` with no text is unknown).
17. **Eligibility starts 2018-07-01.** Chicago replaced Critical/Serious with
    Priority/Priority Foundation/Core on that date, cleanly. Before it the target
    is *undefined*, not sparse.
18. **`Out of Business`, `No Entry`, `Not Ready` and `Business Not Located` are
    ineligible, not negative.** No inspection occurred. Labelling them negative
    would teach that a closed establishment is a clean one.
19. **`inspection_date` is the as-of boundary.** Component 4 may use only
    information strictly before it. `target`, `results`, `evidence` and the
    `n_*_entries` columns describe the outcome and are forbidden as features —
    the set is `sentinel.target.writer.TARGET_EVENT_COLUMNS`.
20. **Pre-2018 inspections are usable as features even though they cannot be
    labelled.** The era boundary constrains labelling, not knowledge.
21. **One row per (establishment, date), target = OR over that day's canvasses.**
    "Inspect E on date D" is one scheduling decision.
22. **Never re-derive identity or labels.** Component 2 owns `establishment_id`;
    Component 3 owns `target`.

### Component 4 invariants - the as-of rule is the whole component

23. **A feature for the row at `inspection_date = d` may use only records dated
    STRICTLY BEFORE d.** Not on or before. An inspection dated on the reference
    date is never history, including the target's own.
24. **The boundary is `<` because dates carry no time component.** All 314,245
    rows have `T00:00:00.000`, so same-day records cannot be ordered; 43 same-day
    canvass re-inspections at reference dates provably follow their canvass.
25. **One range join carries the condition, in one place** (`historical.py`), and
    `validate.py` re-derives it independently on every row. Never
    `groupby(establishment_id)` then merge - that is the all-history bug.
26. **Four missing-value rules**: counts never NULL (0 is a true observation);
    recency NULL when the event never happened (0 would mean "today"); rates NULL
    when the denominator is 0; at-last flags NULL when there is no prior event.
    Every event count is emitted beside its inspection count so a 0 is legible.
27. **Priority features use code-era canvasses only** and are NULL for the 24.5%
    of rows with no prior code-era canvass. Absence of evidence, not evidence of
    absence.
28. **Priority is classified by Component 3's parser**, never a SQL substring
    match, so it means the same thing in the label and the feature.
29. **`FEATURE_COLUMNS` is the complete set of model inputs.** `target`,
    `target_status`, `inspection_date` and `code_era_phase` are not features.
30. **Never select features by downstream accuracy.** That is leakage by another
    route. Justify by domain reasoning and availability only.

---

## Key as-of feature facts (measured 2026-08-16 on the full snapshot)

Detail in `docs/analysis/as_of_feature_engineering_findings.md`.

* **57,727 eligible target rows -> 57,727 feature rows, 0 unmatched, 26
  features, 33 columns** in 15.6 s. Output in `data/processed/features/`.
* **`inspection_date` has exactly ONE distinct time component** across all
  314,245 rows (`T00:00:00.000`). This single measurement settled the boundary.
* On reference dates there are also 1,075 `License`, **43 `Canvass
  Re-Inspection`** and 42 `Complaint` records. 2,103 target rows have at least
  one same-day companion.
* History is abundant: only **401 rows (0.69%)** are cold-start, but **5,615
  (9.7%)** have no prior canvass and **14,162 (24.5%)** none in the code era.
  **80%** have pre-2018 history, which is usable for counts but not for priority.
* **Canvass cycle: 358-day median** (p25 251, p75 482). So a 365-day window is
  **empty for 62% of rows**; 730d for 22%, 1095d for 14.3%.
* **Any-type interval p25 is 9 DAYS** - the re-inspection pattern. This is why
  `days_since_any_inspection` is labelled policy-encoding context rather than the
  primary recency.
* Prior canvasses include **16,517 `Out of Business`** and **13,077 `No Entry`**;
  they are excluded from outcome denominators. `prior_canvass_fail_rate` has 346
  more nulls than `days_since_last_canvass` for exactly this reason.
* **`days_since_last_canvass` min = 1, zero zeros.** A zero-day recency is
  unconstructable; cheapest proof the boundary works.
* **15.9%** of target rows sit in a premises that changed name; **1,962** follow
  a change immediately.
* Data quality: zero null dates, zero unparseable dates, zero duplicate
  `inspection_id`.
* Performance: 2m14s -> 15.6s by materializing the aggregation as a TABLE (not a
  view) and parsing only code-era violation text. Range join is 793,200 pairs,
  ~0.1 s; the cost is the Python parser, not the temporal logic.

---

## Key target facts (measured 2026-08-16 on the full snapshot)

Detail in `docs/analysis/target_construction_findings.md`.

* **314,245 inspections -> 313,624 target rows -> 57,727 eligible, 30,316
  positive (52.52%)** in 25 s, across 15,144 establishments.
* **`results` has SEVEN values, not the four that were documented**: Pass
  162,607 · Fail 60,513 · Pass w/ Conditions 46,661 · Out of Business 25,767 ·
  No Entry 14,045 · Not Ready 4,557 · Business Not Located 95. No nulls, blanks
  or case variants.
* **The 2018-07-01 cutover is clean**: June 2018 has 0 rows using Priority
  terminology and 415 using Critical/Serious; July has 761 and 0.
* **Priority presence by result, among canvasses**: Fail 99.4%, Pass w/
  Conditions 97.9%, Pass 0.45%.
* **The violation number does NOT encode severity.** Item 10 is 42.2% Priority
  Foundation / 11.3% Priority / 46.5% unlabelled; the same item covers a hand
  sink with no hot water and a missing hand-washing sign.
* **Requiring a `7-38-xxx` citation code would create ~21,281 false negatives** —
  "PRIORITY FOUNDATION VIOLATION. NO CITATION ISSUED." is genuine.
* **72% of violation entries carry no severity label**, so unlabelled means
  UNCLASSIFIED, never "Core".
* **Narrative exclusions** (grace period, will-be-issued, no-priority) change 74
  entries and **10 inspection labels** out of 137,598 and 58,427.
* **24.9% of `Out of Business` records are followed by another inspection** at
  the same premises, median 273 days later.
* **Base rate drift**: 87.6% (2018 H2) -> 77.4 -> 59.4 -> 50.3 -> 46.5 -> 46.1 ->
  42.6 -> 39.2 -> 39.1% (2026). `code_era_phase` flags the adoption period.
* Exclusions: `ineligible_era` 172,879 · `ineligible_type` 70,848 ·
  `ineligible_result` 12,091 · `unknown_violations` 79.
* 111 distinct `inspection_type` values; only `Canvass` (70,518 in the code era)
  is eligible. `Canvass Re-Inspection` (16,998) is excluded because it exists
  only because something failed.

---

## Key entity-resolution facts (measured 2026-08-16 on the full snapshot)

Snapshot `7d3c4069...ad38`, 314,245 rows. Detail in
`docs/analysis/entity_resolution_findings.md`.

* **314,245 rows -> 51,099 nodes -> 35,859 establishments** in 43 s.
* **18.47%** of (name, address) pairs hold more than one licence, up to 47 (a
  mobile-food commissary with one permit per cart). A licence is often *finer*
  grained than an establishment.
* The **`'0'` licence sentinel** covers 323 distinct names across 364 addresses.
  850 rows (0.27%) have no usable licence at all.
* **Address normalization is where the leverage is**: case and whitespace alone
  collapse 33,261 address strings to 20,313 (-39%). Name normalization resolves
  only 0.21% of licences, which is why there is no fuzzy name matching.
* **There are no long-form street suffixes** and one long directional in 20,312
  addresses. The real defect is *missing and contradictory* suffixes
  (`1901 W MADISON` / `AVE` / `ST` are all the United Center), so the suffix is
  excluded from `addr_key` rather than canonicalized.
* **Coordinate spread within an address is exactly 0 m.** The city geocodes
  before the string variation appears, so a shared coordinate bridges variants
  no string rule can, including the 2021 Lake Shore Drive rename. 95.4% of
  coordinates map to one address key; the worst covers 4.
* **75.5% of same-place licence pairs overlap in time** rather than succeeding
  one another, which is why there is no temporal logic in matching.
* Chains are pervasive: 247 Subways, 184 Dunkin Donuts; one O'Hare address
  (`11601 W TOUHY AVE`) carries 219 distinct business names and 417 licences.
* **Single-inspection establishments fell 51%** versus naive licence grouping
  (12,356 -> 6,084), the clearest evidence real history was recovered.

---

## Key API facts (verified 2026-08-15, live)

* Endpoint `https://data.cityofchicago.org/resource/4ijn-s7e5.json`, no auth.
* **314,245 rows** total at that date.
* **Every value is a JSON string**, including `number` and `calendar_date`
  columns. `location` is the one nested object.
* `$limit=60000` works — there is no 50k cap on this endpoint.
* Pagination ends on a short page or an empty page. There is no cursor and no
  "has more" flag.
* Errors are JSON with an `errorCode` (e.g. `query.soql.no-such-column`) + 4xx.
* **`$order` suppresses the 5 `:@computed_region_*` columns** unless they are
  explicitly named in `$select`. This is why ingestion makes an extra unordered
  `?$limit=1` request to discover the field list, then selects it. Do not
  replace that with a hardcoded column list — a new upstream column would then
  be silently dropped. See `docs/api/socrata_findings.md` §6.
* Response headers `X-SODA2-Fields` / `X-SODA2-Types` carry the declared schema
  on every request. Positionally aligned arrays.
* `X-SODA2-Truth-Last-Modified` and `ETag` exist and are unused — leads for a
  future incremental-ingestion component.

---

## Technology decisions and why

| Choice | Why | ADR |
|---|---|---|
| Python 3.12 + uv | downstream ML ecosystem is Python; uv is fast and gives a committed lockfile | 0001 |
| httpx (not requests/sodapy) | modern client, respx mocks it at transport level; SDKs hide the pagination we most need to see | 0004 |
| Hand-written pagination | highest-risk logic in the component; must be visible and unit-testable | 0004 |
| Parquet, all `Utf8` | columnar + self-describing + compresses well; strings keep the raw layer faithful | 0002 |
| Polars (not pandas) | fast, explicit schema control, `pl.Utf8` enforcement is trivial | 0002 |
| DuckDB in-memory | reads Parquet in place, no load step, no server, real analytical SQL | 0003 |
| argparse (not Typer) | ~4 flags across 2 subcommands; Typer would add 3 deps for help-text polish | — |
| respx | mocks httpx at the transport layer, so real request/status/retry code runs | — |
| JSON manifest sidecar | human-readable, diffable, greppable, zero infrastructure | — |

---

## Important paths

```text
src/sentinel/ingest/socrata.py            the client. Most important file.
src/sentinel/ingest/food_inspections.py   orchestration
src/sentinel/ingest/manifest.py           provenance record
src/sentinel/query/duckdb_queries.py      NAMED_QUERIES live here
src/sentinel/config.py                    every tunable setting
data/raw/food_inspections/                output: parquet + manifest_*.json
docs/api/socrata_findings.md              verified API behaviour — read before
                                          touching the client
docs/data_contracts/food_inspections_raw.md   what a raw file guarantees
docs/decisions/                           11 ADRs

src/sentinel/entity/evidence.py           the match rules. Read the findings
                                          doc before changing any of them.
src/sentinel/entity/models.py             DEFAULT_THRESHOLDS - every tunable
                                          number, each citing its measurement
src/sentinel/entity/normalize.py          normalization rules
scripts/profile_entities.py               36 read-only data profiles
data/interim/entity_resolution/           output: 3 parquet + manifest_*.json
docs/analysis/entity_resolution_findings.md   why Component 2 works the way it
                                          does. Read before touching entity/.
docs/data_contracts/establishment_assignments.md   Component 2 output contract

src/sentinel/target/violations.py         the violation parser + the narrative
                                          exclusion list
src/sentinel/target/construct.py          eligibility gates and labelling
src/sentinel/target/models.py             CODE_ERA_START, INSPECTED_RESULTS,
                                          TARGET_DEFINITION_VERSION
scripts/profile_target.py                 31 read-only target profiles
data/interim/target/                      output: parquet + manifest_*.json
docs/analysis/target_construction_findings.md   why the target is defined this
                                          way. Read before touching target/.
docs/data_contracts/inspection_targets.md  Component 3 output contract

src/sentinel/features/definitions.py      FEATURE_SPECS: the single source of
                                          truth for every feature
src/sentinel/features/historical.py       the range join and THE boundary
src/sentinel/features/validate.py         temporal_boundary_holds lives here
scripts/profile_features.py               21 read-only history profiles
data/processed/features/                  output: parquet + manifest_*.json
docs/analysis/as_of_feature_engineering_findings.md   why the boundary is `<`
docs/data_contracts/as_of_features.md     the output contract and the leakage
                                          rules for Component 5
```

---

## Commands

```bash
uv sync                                       # setup
uv run sentinel ingest --dev                  # small pull (SENTINEL_DEV_ROW_LIMIT)
uv run sentinel ingest --limit 5000           # explicit cap
uv run sentinel ingest --full                 # entire dataset (~70 min, verified)
uv run sentinel query --list
uv run sentinel query --name row_count
uv run sentinel resolve                       # entity resolution (~45 s)
uv run sentinel resolve --dry-run --report    # validate, write nothing
uv run sentinel build-target                  # target construction (~25 s)
uv run sentinel build-target --dry-run --report
uv run sentinel build-features                # as-of features (~16 s)
uv run sentinel build-features --dry-run --report
uv run python scripts/profile_entities.py     # 36 read-only entity profiles
uv run python scripts/profile_target.py       # 31 read-only target profiles
uv run python scripts/profile_features.py     # 21 read-only history profiles
uv run pytest                                 # 745 tests, offline
uv run pytest -m live                         # 3 live tests, hits the real API
uv run ruff check . && uv run ruff format --check .
uv run mypy src/sentinel scripts
```

One of `--dev`, `--limit`, `--full` is required — there is no default scope, so
a bare `sentinel ingest` cannot accidentally pull 314k rows.

---

## Naming conventions

* Raw files: `food_inspections_<YYYYMMDD>T<HHMMSS>Z.parquet` (UTC, start time).
* Manifests: `manifest_<parquet stem>.json`, same directory. The `manifest_`
  prefix is what `.gitignore` whitelists — keep it.
* Env vars: `SENTINEL_` prefix, matching the `Settings` field name.
* Private helpers are `_prefixed`; tests import public names only.

---

## Assumptions being made

* `inspection_id` is unique and totally orderable. Verified consistent with
  observed pagination; not proven across the whole dataset.
* The Chicago dataset stays publicly available without authentication.
* A single machine can hold a full pull in memory. **Unverified at 314k rows.**
* `:@computed_region_*` values are Socrata-derived, not authoritative source
  data, and are re-derivable from latitude/longitude.

---

## Open design questions

1. **Should schema divergence be fatal?** Currently a warning. If Chicago drops
   a column Sentinel depends on, ingestion succeeds with a null column. A
   later component may want a declared required-column set that fails loudly.
2. **Incremental ingestion.** Full pulls will get wasteful. `ETag` /
   `X-SODA2-Truth-Last-Modified` / a `$where` on `inspection_date` are the
   options. Deferred until a component needs freshness.
3. **Memory at full scale.** All pages accumulate before the single Parquet
   write. If `--full` proves heavy, write per-page row groups instead.
4. **Same-second filename collision.** One-second filename resolution means two
   runs starting in the same second collide. Not observed; needs sub-second
   precision or a counter if it ever matters.
5. ~~**Is `license_` ever a usable key?**~~ **Answered.** No - it fails in both
   directions and is now supporting evidence only. See the facts section above.
6. **Cross-snapshot identity.** `establishment_id` is stable for one snapshot
   only. A crosswalk mapping retired ids to successors is unbuilt (ADR 0007).
7. **Same-name outlets at a dense address.** Two outlets of one chain at one
   address with no store number and no distinguishing `aka_name` still merge.
   This is the residual false-merge risk; bounded to mega-addresses.
8. **Should there be an `--as-of DATE` resolution mode?** Identity is currently
   reconstructed from the whole snapshot, which is argued to be legitimate
   rather than leakage. A strict mode would cost one run per evaluation fold.
9. **Can the pre-2018 era support its own target?** 172,879 rows use the old
   Critical/Serious scheme. A separate target could be defined for them, with its
   own definition version. Not attempted.
10. **Should the target become a count or a severity grade?** `v1` is binary
    presence; a count reflects inspector verbosity as much as risk. Revisitable
    as `v2`.
11. **Should history reset at a tenant change?** Component 4 exposes the
    transition as features instead of resetting, diverging from spec §3.3. A
    reset variant is an ablation, not a redesign.
12. **Does `days_since_any_inspection` help or just encode scheduling policy?**
    Emitted so Component 5 can ablate it; unanswerable without a model.

---

## Lessons learned (Component 4)

* **One measurement can settle a design argument.** Whether the boundary is `<`
  or `<=` looked like a judgement call until a single query showed
  `inspection_date` has one distinct time component. After that it was not a
  choice: same-day records are unorderable, so they cannot be history.
* **Put the temporal condition in exactly one place.** 26 features, one range
  join. If the predicate were repeated per feature there would be 26 chances to
  get it wrong, and a reviewer would have to check all of them.
* **Re-derive the invariant independently rather than trusting the code that
  produced it.** `validate.py` recomputes the latest contributing date from a
  separate query. A check that reuses the aggregation only proves the aggregation
  agrees with itself.
* **A NULL rendered as 0 is the quiet catastrophe.** 14,162 rows have no
  code-era history; writing 0 would tell a model a quarter of establishments have
  a clean priority record. Pair every event count with its inspection count so
  the difference is visible.
* **Materialize before validating.** Running dozens of small checks against a
  VIEW re-executed the whole join each time: 2m14s. As a TABLE: 15.6s, identical
  output.
* **Do not parse what cannot exist.** Priority did not exist before 2018-07-01,
  so pre-code rows need no classification at all - half the parser's work removed
  by a definition rather than an optimisation.
* **Resist feature-count inflation.** 26 features, each with a written reason.
  The measurement that a 365-day window is empty for 62% of rows is a reason to
  document it, not a reason to add four more windows.

---

## Lessons learned (Component 3)

* **Profile the value set before trusting a contract.** The raw data contract
  documented four `results` values; there are seven, and the three missing ones
  are exactly the "no inspection happened" cases that must not become negatives.
* **Look for regime changes before defining anything longitudinal.** The whole
  target hinged on noticing that "Priority" does not exist before 2018-07-01.
  A single `GROUP BY year` on terminology presence found it in seconds; without
  it, half the labels would have been silently wrong.
* **A plausible structural signal can be a trap.** The violation number looks
  like a severity code and is not one. Checking the association empirically
  (item 10: 42/11/47) took one query and prevented a badly wrong parser.
* **Prefer excluding narrow narrative spans to requiring strong evidence.**
  Requiring a citation code would have looked rigorous and produced ~21,281 false
  negatives. Excluding four boilerplate phrases changed 10 labels.
* **Absence of a label is not evidence of the opposite.** 72% of entries carry no
  severity marker, so the parser emits UNCLASSIFIED rather than "Core".
* **Distinguish "not measured" from "measured as zero".** `Pass` with no
  violation text is a true zero; `Fail` with no violation text is unknown. The
  same null means different things depending on the row.

---

## Lessons learned (Component 2)

* **Investigate before designing, and be willing to throw the design away.** Six
  planned decisions were reversed by measurement: licence turned out to be too
  *fine* grained rather than too coarse; the street-suffix canonicalization
  table would have been dead code; geographic *distance* thresholds were
  meaningless because within-address spread is exactly zero; the temporal
  succession rule addressed a quarter of the cases it assumed; and the fuzzy
  name tier with its 100-pair calibration was retired before it was written.
* **Aggregate metrics hide over-merges.** The first full run passed every
  structural check while fusing about twenty O'Hare restaurants into one
  establishment. Only reading the largest clusters by hand found it. Inspect the
  biggest outputs, never just the summary.
* **`dba_name` is not always the business.** At O'Hare it is the concessionaire
  (`HOST INTERNATIONAL INC`); elsewhere it is the holding company
  (`1918 WINTER STREET ILLINOIS LLC` for a Mariano's). `aka_name` often carries
  the real identity.
* **A veto written too broadly is as damaging as no veto.** The first version of
  the `aka_conflict` veto fired on any two differently-named neighbours and
  would have blocked legitimate merges. The unit tests caught it before it
  reached the data, which is the argument for tests that assert what must *not*
  happen.
* **Classify ordinary difference as a decision, not a doubt.** Marking
  same-address different-name pairs as ambiguous produced 108,597 of them and
  buried the real 747-pair review queue.

---

## Lessons learned (Component 1)

* **Probe the API before writing the client.** Two behaviours would have been
  wrong if assumed from documentation: the absent 50k page cap, and the
  string-encoding of every value.
* **`$order` changing the returned column set was not documented anywhere.** It
  was found only by comparing the field count between an exploratory request
  and the real paginated one. Compare what you *get* against what you *expected*
  at every step.
* **An empty result still carries schema headers.** The first implementation
  lost the schema on a zero-row pull because `iter_pages` returns before
  yielding an empty page. Fixed by retaining the last-seen schema on the client.
* **DuckDB's `DESCRIBE` columns are `column_name` / `column_type`**, not
  `name` / `type`.
