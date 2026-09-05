# Data contract — Component 20 geographic work organization

**Producer:** `sentinel organize-geography` · **Layer:** `data/processed/geographic_organization/`
**Manifest:** `manifest_geographic_inspection_plan_<planning_date>_<stamp>.json`
**Definition version:** `v2`

---

## ⚠ What this artifact is not

**It is not a driving route, and it never claims to be.** Every distance is Haversine
(great-circle) between raw coordinates. There is no road-network, traffic, or travel-time source
anywhere in this project, and the code, the CLI help text, and this document say so explicitly.
Nothing here is called "optimal," "fastest," or "shortest."

**A geographic work block is not a workday.** Whether the establishments in one block fit into
one inspector's day is a capacity/staffing question this component does not model — see
`NON_GOALS`. A work block is a geographically coherent grouping a supervisor can *consider*
together; treating it as a confirmed day's assignment is a supervisor decision, not a Sentinel
claim.

**It never re-ranks a risk decision.** `calibrated_score`, `base_score`, `rank`, `policy_rank`,
`selection_reason`, and `selection_mechanism` are copied verbatim from Component 19 and checked
byte-identical after every run (`validate.check_risk_and_policy_fields_unchanged`). Geography
groups and labels; it never decides who is more urgent.

**It never fabricates a location.** An establishment with a null latitude or longitude is placed
in the `unmapped` pseudo-block, preserved, and never given an invented coordinate
(`validate.check_no_fabricated_coordinates`).

**It does not claim confirmed capacity.** The "selected inspection workload" language is
deliberate — Component 19's capacity is derived from historical inspection activity, not
confirmed future staffing, and Component 20 never upgrades that into "N confirmed inspections."

---

## Grain, keys and sorting

One row per selected (`is_selected == True`) establishment from Component 19. Sorted by
`work_block_id`, then `suggested_order_in_block` (nulls — the `unmapped` block — last).

The grain is the **selected set only**, not the full ranked queue: unlike Components 18/19,
which preserve their whole input for audit purposes, Component 20's job is to organize a bounded
plan a supervisor is about to review. A consumer needing the wider context joins back to
Component 19's artifact by `target_inspection_id`.

---

## The algorithm

**Grouping** (unchanged since v1): distance-threshold connected components via Union-Find.
Two establishments are in the same group if their Haversine distance is at most `threshold_km`.
This produces *connected-component* groups, not radius-bounded clusters — chained establishments
can end up in one group even if the two ends are farther apart than the threshold. The
`max_within_block_distance_km` metric quantifies the actual spread.

**Threshold**: `--threshold-km <KM>` or `--threshold-preset {tight,balanced,broad}` (1.0 / 1.5 /
5.0 km respectively; `balanced` equals the historical default). The two flags are mutually
exclusive. The threshold is an operational heuristic, not a validated travel distance — it is
meant to be tuned by supervisors against real field experience.

**Suggested work order within a block** (new in v2): controlled by `--organization-mode`.

| mode | rule | can reorder relative to `policy_rank`? |
| --- | --- | --- |
| `risk_first` (default) | exactly `policy_rank` ascending | never |
| `geography_assisted` | greedy nearest-neighbour, seeded at the block's highest-priority member | yes, within the block only |

`risk_first` is mechanically proven never to reorder
(`validate.check_risk_first_never_reorders`) — the conservative default the product requires.
`geography_assisted` is a deterministic heuristic over straight-line distance; it is never called
a route.

**Honesty about singleton blocks:** when most resulting blocks contain exactly one
establishment, the manifest's `notes` say so explicitly, name the current threshold, and point at
`--threshold-preset broad` as the tradeoff — this is never hidden.

---

## Column reference (additive over v1)

| column | meaning |
| --- | --- |
| `geographic_group_id`, `geographic_group_label` | v1 columns, unchanged: `area_N` / `unmapped`, ordered northernmost first |
| `location_status` | `location_available` / `location_unavailable` |
| `work_block_id`, `work_block_label` | **new** — alias `geographic_group_id`/`geographic_group_label` under the operational vocabulary; same value |
| `suggested_order_in_block` | **new** — 1-indexed position within the block; null for the `unmapped` block (no geography to order by) |
| `organization_mode` | **new** — `risk_first` or `geography_assisted`, the mode used for this run |
| `highest_sentinel_rank_in_block` | **new** — the block's best (lowest) `policy_rank`, read-only |
| every Component 18/19 column (`calibrated_score`, `base_score`, `rank`, `policy_rank`, `selection_reason`, `selection_mechanism`, coordinates, identity fields, …) | copied verbatim, immutable |

---

## Invariants (`validate.run_all_checks`, error-severity unless noted)

1. `selected_ids_unchanged_by_geography` — the most important check. Geography never adds,
   drops, or substitutes an establishment.
2. `risk_and_policy_fields_unchanged` — `IMMUTABLE_FIELDS` byte-identical to Component 19.
3. `coordinates_are_never_fabricated_or_altered`.
4. `no_duplicate_group_membership`.
5. `location_coverage_counts_match_the_plan` — independently re-derived, not trusted.
6. `unmapped_establishments_preserved_not_dropped`.
7. `group_ids_ordered_by_centroid_latitude` (warn if no mapped groups).
8. `suggested_order_is_permutation_of_block` — **new**: within every mapped block, the suggested
   order covers each member exactly once.
9. `risk_first_never_reorders` — **new**: mechanically proves the conservative default.

---

## Limitations

Inherited from Component 19: capacity reflects historical activity, not confirmed staffing.
Own limitations: straight-line distance only (no road network, traffic, or travel time); no
inspector start locations, working hours, or inspection-duration modeling; a work block is not a
workday. See `NON_GOALS` in `geographic_organization/definitions.py` for the full, checked list.

---

## CLI

`sentinel organize-geography [--selection PATH] [--threshold-km KM | --threshold-preset NAME]
[--organization-mode {risk_first,geography_assisted}] [--output-dir DIR] [--dry-run] [--report]`
