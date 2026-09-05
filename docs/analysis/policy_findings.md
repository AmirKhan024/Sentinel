# Component 13 — decision policy and deployment governance: findings

**Snapshot:** `as_of_features_20260816T150313Z.parquet` (57,727 rows) ·
`calibrated_predictions_20260824T163608Z.parquet` (207,680 rows) ·
`simulation_summary_/evaluation_metrics_/sensitivity_20260824T160045Z.parquet` ·
`neural_categoricals_20260818T125631Z.parquet` ·
`fairness_group_support_20260825T173633Z.parquet`

**Production run:** 2026-08-26, 38.9 s, 11 tables, 1,453,760 recommendation rows,
22 checks — **0 errors, 4 advisory findings**. Inputs sha256-identical before and after.
**11 of 11 tables byte-identical across two independent production runs.**
Refits, re-executions, bit-identity gates: **none**.

---

## 0. The question, and the answer that surprised us

Component 12 measured a closed loop and handed it here:

```text
group              __UNKNOWN__ community area, 405 quarterly test rows
data availability  59.5% have no prior inspection of any type, against 0.74% overall
feature missingness 61.7% missing code-era canvass history, against 10.4% overall
model reliance     Component 11: that indicator ranks 2nd/3rd in importance for two models
group behaviour    ROC-AUC 0.509-0.532 (random), selection ratio 0.20, capture 0.006 vs 0.070
```

The natural reading is *the model neglects establishments it knows nothing about, so reserve
some capacity for them*. **On this data that reading is wrong, and the profiler said so before a
line of policy code was written.**

The coverage-eligible population — establishments with no canvass since the 2018 code era —
is 10.4% of the quarterly test rows and takes **40% to 58%** of the top-k under pure risk
ranking, a selection ratio of **3.96 to 5.57** across all four candidate models. The models
lean hard on the missingness indicator, and they are right to: the eligible population's
outcome rate is **0.4883 against the window's 0.4283**, and it holds **11.9% of the positives
while being 10.4% of the rows**.

So the risk queue does not neglect this population. It over-serves it by four to five times.

What Component 12 found is real, and it is about a **different** population: `__UNKNOWN__` is a
geography that failed to resolve, and only **456 of 14,162** coverage-eligible rows (3.2%) sit
in it. The two populations overlap and are not the same thing — which is exactly why the policy
is defined on the history column and never on the geography (ADR 0038).

---

## 1. The eligibility contract

Four candidate rules were swept before one was frozen (profile 1):

| rule | column | eligible rows | share | base rate | nulls |
| --- | --- | ---: | ---: | ---: | ---: |
| **`no_code_era_canvass`** | **`prior_canvass_count_code_era`** | **14,162** | **0.2453** | **0.7284** | **0** |
| `no_prior_canvass` | `prior_canvass_count` | 5,615 | 0.0973 | 0.5907 | 0 |
| `no_inspected_canvass` | `prior_canvass_inspected_count` | 5,961 | 0.1033 | 0.5932 | 0 |
| `no_prior_inspection` | `prior_inspection_count_any_type` | 401 | 0.0069 | 0.5461 | 0 |

City-wide outcome rate over all 57,727 rows: **0.5252**.

`no_code_era_canvass` is frozen because it is the *cause* of the ranking difficulty rather than
a correlate: it is the exact condition under which the four priority features are NULL.
`no_prior_inspection` is rejected as a gate on size — 401 rows is zero or one slot at a day of
capacity — and is carried on every row as a reporting flag instead.

The `nulls` column is the assertion that matters. Every candidate carries `NullRule.NEVER` in
Component 4, so a zero is a real observation of no history. `_eligible_expr` maps a null to a
value the predicate cannot match, and `validate.eligibility_matches_the_declared_rule`
re-derives the flag on every run.

### Eligibility in the evaluated windows

| fold set | rows | eligible | share | base rate | eligible base rate | share of positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `quarterly` | 32,696 | 3,410 | **0.1043** | 0.4283 | **0.4883** | 0.1189 |
| `covid_shift` | 8,840 | 1,373 | 0.1553 | 0.5127 | **0.6635** | 0.2010 |

**The 0.1043 is the anchor for the whole policy grid** and the only reason a value near 0.10
appears anywhere in this component. The grid is half that, that, and twice that.

---

## 2. The production model

MEMORY open question 13 lands here because a department cannot work four queues. The rule is
lexicographic, frozen before it ran, and applied to Components 5 and 9's own artifacts:

| model | NDE (quarterly mean) | NDE sensitivity band | vs leader | calibrated ECE | precision@k_1_day |
| --- | ---: | --- | --- | ---: | ---: |
| `neural_numeric_only_platt` | 0.2482 | [0.2311, 0.2527] | tied | 0.0524 | 0.6273 |
| `xgboost_chain_embeddings_platt` ⚠ | 0.2444 | [0.2285, 0.2502] | excluded | 0.0481 | 0.6480 |
| **`xgboost_platt`** | 0.2376 | [0.2224, 0.2444] | tied | **0.0474** | 0.6308 |
| `lightgbm_platt` | 0.2355 | [0.2201, 0.2419] | tied | 0.0490 | 0.6598 |
| `logistic_regression_platt` | 0.2326 | [0.2160, 0.2374] | tied | 0.0518 | 0.6576 |

**Axis 1 separates nothing: all four candidate bands overlap the leader's.** Under Component 5's
1,000-replication label-flip study, every candidate's NDE interval contains every other
candidate's point estimate. The headline operational metric of this entire project cannot tell
these four models apart — which corroborates Component 8's own conclusion that the network's
advantage is the size of its seed noise.

The rule therefore falls to axis 2 and selects **`xgboost_platt`** on a calibrated ECE of
0.0474. Axis 3 and the name terminator are never reached.

⚠ **The tie rule decides the outcome, and it was fixed after its inputs were first read.** The
plan carried a placeholder — Component 8's five-seed *ROC-AUC* spread of 0.0058 — and
thresholding an *NDE* difference with a ROC-AUC spread is a unit error. **Under that discarded
band, `neural_numeric_only_platt` separates on axis 1 and would have been selected instead.**
Both outcomes are emitted on every run (`selected_under_discarded_band`), and ADR 0039 records
the sequence. A rule chosen after seeing what it decides is defensible only when the choosing is
visible.

`xgboost_chain_embeddings_platt` has the best PR-AUC in the project and is excluded before any
number is read: ADR 0022 makes it experimental and ADR 0031 records that Component 11 could not
explain it.

**The `covid_shift` fold orders these models differently** — `lightgbm` 0.2585, `xgboost` 0.2572,
`neural_numeric_only` 0.2569, `logistic_regression` 0.2512. One 18-month episode cannot carry a
selection rule, so it is a named limitation on the choice rather than an input to it.

---

## 3. The policy grid

Seven configurations: the null policy, and two mechanisms at three shares anchored on 0.1043.

| `policy_id` | mechanism | share | what it does |
| --- | --- | ---: | --- |
| `pure_risk` | none | 0.00 | the implicit policy, made explicit |
| `coverage_floor_half_share` | floor | 0.05 | guarantee half the population share |
| `coverage_floor_population_share` | floor | 0.10 | guarantee proportional access |
| `coverage_floor_double_share` | floor | 0.20 | guarantee twice proportional |
| `coverage_forced_half_share` | forced | 0.05 | *spend* half the population share |
| `coverage_forced_population_share` | forced | 0.10 | *spend* the population share |
| `coverage_forced_double_share` | forced | 0.20 | *spend* twice it |

A **floor** guarantees an outcome and does nothing when risk already delivers it. A **forced**
reserve guarantees a spend. On this data they diverge almost completely, and implementing only
one would have hidden the entire result.

---

## 4. What each policy costs — the central table

`xgboost_platt`, pooled over the 17 quarterly test windows. `Δ` is against `pure_risk` at the
identical model, fold and capacity.

### One day of real capacity (`k_1_day`, 556 slots)

| policy | reserve slots | citations | Δ citations | eligible selected | Δ eligible |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pure_risk` | 0 | 348 | — | 249 | — |
| `coverage_floor_half_share` | 0 | 348 | 0 | 249 | 0 |
| `coverage_floor_population_share` | 1 | 349 | **+1** | 250 | +1 |
| `coverage_floor_double_share` | 8 | 349 | **+1** | 257 | +8 |
| `coverage_forced_half_share` | 19 | 345 | **−3** | 261 | +12 |
| `coverage_forced_population_share` | 47 | 350 | **+2** | 280 | +31 |
| `coverage_forced_double_share` | 104 | 347 | **−1** | 313 | +64 |

### One week of real capacity (`k_1_week`, 2,780 slots)

| policy | reserve slots | citations | Δ citations | eligible selected | Δ eligible |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pure_risk` | 0 | 1,657 | — | 1,170 | — |
| `coverage_floor_half_share` | 0 | 1,657 | 0 | 1,170 | 0 |
| `coverage_floor_population_share` | 0 | 1,657 | 0 | 1,170 | 0 |
| `coverage_floor_double_share` | 2 | 1,657 | 0 | 1,172 | +2 |
| `coverage_forced_half_share` | 133 | 1,649 | **−8** | 1,246 | +76 |
| `coverage_forced_population_share` | 274 | 1,642 | **−15** | 1,325 | +155 |
| `coverage_forced_double_share` | 556 | 1,623 | **−34** | 1,513 | +343 |

### Reading these two tables

**The floor is nearly inert.** Across all four models and all quarterly cells: at the population
share it grants **2 slots in 340 cells** (inert in 338); at half the share it grants **zero**; at
twice the share it grants 101. That is the direct consequence of the risk queue already
over-serving this population fourfold.

**The forced reserve buys coverage at a real price, and the price grows with capacity.** At a
week of capacity, twice the population share buys 343 additional eligible selections and gives
up **34 Priority citations**. That number is the honest unit: 34 inspections that would have
found a violation and did not.

**At one day of capacity the deltas are inside the noise, and should not be read as a result.**
±1 to ±3 citations out of 348, over 17 folds, is not a measurable difference. In particular
`coverage_forced_population_share` posting **+2** — buying 31 extra eligible selections *and*
two more citations — is a pleasing number that this study cannot support. The `k_1_week` and
`k_pct_05` columns, where the counts are five times larger, are the ones to read.

### Does the conclusion survive the model choice?

`k_1_day`, Δ citations, all four candidates:

| policy | lightgbm | logistic | neural | xgboost |
| --- | ---: | ---: | ---: | ---: |
| `coverage_floor_population_share` | 0 | 0 | 0 | +1 |
| `coverage_forced_half_share` | −3 | −4 | 0 | −3 |
| `coverage_forced_population_share` | −5 | −9 | 0 | **+2** |
| `coverage_forced_double_share` | −1 | −19 | −2 | −1 |

**The floor is inert for every model. The forced reserve costs citations for every model except
one cell of one model.** `xgboost_platt`'s +2 is the outlier, not the pattern — which is the
clearest possible demonstration of why the comparison is run for all four candidates rather than
only for the one the selection rule picked.

---

## 5. The frontier, and the absence of a winner

Domination on two axes — citations discovered and coverage-eligible establishments served —
with no exchange rate assumed between them.

At `k_1_day`, two policies survive: `coverage_forced_population_share` (350, 280) and
`coverage_forced_double_share` (347, 313). **`pure_risk` is dominated** by the first. At
`k_pct_05` all seven survive. At `k_1_week` four survive.

The rule declares a winner only when one policy is the unique non-dominated policy at the primary
operating point. It is not, so:

> **POLICY WINNER: the data does not determine the correct policy.**

That is published as the result, in the summary and the manifest, and
`validate.a_winner_was_determined` fires as an **advisory** rather than an error. Forcing a
winner would require an exchange rate between a missed Priority citation and an uninspected
establishment with no history, and nothing in this project measures one.

---

## 6. The trend the pooled numbers hide

Eligible share of the queue, per fold, at one day of capacity, under `pure_risk`:

| fold | k | eligible selected | share | fold | k | eligible selected | share |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 2022Q2 | 29 | 24 | 0.828 | 2024Q3 | 33 | 23 | 0.697 |
| 2022Q3 | 28 | 20 | 0.714 | 2024Q4 | 40 | 29 | 0.725 |
| 2022Q4 | 30 | 23 | 0.767 | 2025Q1 | 45 | 8 | 0.178 |
| 2023Q1 | 31 | 13 | 0.419 | 2025Q2 | 39 | 9 | 0.231 |
| 2023Q2 | 29 | 14 | 0.483 | 2025Q3 | 28 | 6 | 0.214 |
| 2023Q3 | 26 | 11 | 0.423 | 2025Q4 | 30 | 20 | 0.667 |
| 2023Q4 | 33 | 24 | 0.727 | 2026Q1 | 35 | 6 | 0.171 |
| 2024Q1 | 34 | 4 | 0.118 | **2026Q2** | **28** | **1** | **0.036** |

**In the most recent quarter the pure-risk queue put one no-history establishment into a day's
28 slots** — 3.6%, against a pooled mean of 0.489 over the earlier folds and 0.299 over the last
four. The series is volatile rather than monotone (2025Q4 is 0.667), so this is a wide swing
rather than a trend with a slope, but the two most recent quarters are among the three lowest in
the series.

**This is why the floor is worth implementing even though it is inert on average.** A mechanism
that does nothing for thirteen quarters and then binds in the fourteenth is not a no-op; it is a
guarantee. Reporting only the pooled ratio would have retired the mechanism on the strength of a
number that describes 2022.

What this does **not** say: that the decline is a problem, or that the model is wrong to have
made it. The eligible population's base rate falls over the same period, so a risk ranking that
selects it less often may be tracking a real change.

---

## 7. The `__UNKNOWN__` group: what Component 13 changed, and did not

At one day of capacity, pooled quarterly, `xgboost_platt`:

| policy | rows | positives | selected | citations found | status |
| --- | ---: | ---: | ---: | ---: | --- |
| **every one of the seven** | 405 | 166 | **2** | **1** | supported |

**Component 13 changes nothing for this group at one day of capacity, and the number is
identical under all seven policies.** Component 12's most uncomfortable finding — of 166
positives, the top 5% found one — is not fixed here, and the component says so rather than
implying otherwise.

The reason is structural and is the whole of ADR 0038. The reserve is keyed to missing
*history*, and only 3.2% of the eligible population sits in `__UNKNOWN__`. A reserve that
reached this group would have to be keyed to the geography — an allocation rule applied to a
population defined by a failed geocode, which is not defensible to an inspector, an alderman or
a court.

What Component 13 does instead is make the group **visible in the operational artifact**: every
row in it carries the `unknown_geography` warning, `policy_group_audit` reports its selection
share and capture under each policy, and `validate.unsupported_groups_are_preserved` fails the
run if a group Component 12 could not measure is quietly relabelled here.

---

## 8. What a reviewer sees

Warnings on the 556 selected rows at one day of capacity under `pure_risk`:

| warnings | rows | share |
| --- | ---: | ---: |
| `none` | 286 | 0.514 |
| `limited_history` | 220 | 0.396 |
| `insufficient_group_audit_support` `|` `limited_history` | 27 | 0.049 |
| `insufficient_group_audit_support` | 21 | 0.038 |
| `limited_history` `|` `unknown_geography` | 2 | 0.004 |

**Nearly half of the recommended queue carries a warning**, and 44.8% of it is flagged
`limited_history` — the direct consequence of the risk ranking over-serving that population.
A warning is not an abstention: every one of these rows still receives a recommendation and a
rank. Sentinel has never built a predictive interval, so there is nothing to abstain on, and
manufacturing one would be inventing the statistic that justifies it (ADR 0040).

### Where every slot came from

`k_1_day`, quarterly, 32,696 candidate rows per policy:

| policy | risk_priority | coverage_reserve | not_selected (capacity) | not_selected (reserve) |
| --- | ---: | ---: | ---: | ---: |
| `pure_risk` | 556 | 0 | 28,979 | 3,161 |
| `coverage_forced_double_share` | 452 | 104 | 29,043 | 3,097 |

Every selected row names the mechanism that put it there, and every unselected eligible row is
distinguished from an unselected ineligible one — so "was the reserve too small?" is answerable
from the artifact rather than from the code.

---

## 9. The `covid_shift` fold, reported separately and never pooled

At one day of capacity the shift window's 22 slots go to **22 coverage-eligible establishments
under pure risk** — 100% of the queue, against 15.5% of the population — and every policy
produces an identical result. The eligible population's outcome rate there is 0.6635 against the
window's 0.5127.

The shift episode is the extreme case of the quarterly finding, not a counterexample to it: when
the world changes, recent history is what the model has least of, and it leans entirely on the
establishments that have none.

Never averaged into a quarterly number, per HANDOFF's standing instruction.

---

## 10. Advisories from the production run

Four fired, all advisory, all exit-code zero:

| advisory | detail |
| --- | --- |
| `reserve_is_not_inert` | 1,000 of 2,072 reserving cells granted no slots |
| `coverage_is_not_free` | cells across the grid gave up citations; the measured price of coverage |
| `group_representation_is_stable` | supported groups whose share of the queue moved past 0.05 |
| `a_winner_was_determined` | the data does not determine the correct policy |

**None of these fails a build, and there is deliberately no flag to make one.** The cheapest way
to turn a red "this reserve gave up 34 citations" build green is to delete the reserve — a policy
decision about how a city allocates enforcement, and not one a CI runner is entitled to take.
`tests/test_policy_leakage.py` asserts the opposite of every red test for exactly this reason.

---

## 11. Reproducibility

- **11 of 11 tables byte-identical** across two independent production runs.
- Shuffling the prediction rows, and separately the feature rows, leaves the queue identical —
  asserted end to end in `tests/test_policy_determinism.py`, and over a window of pure ties,
  which is where Component 12's equal-mass binning defect lived.
- Input checksums taken before the first read and again after the last write;
  `inputs_were_not_modified` is an error-severity check.
- No refits, no re-executions, no bit-identity gates — this component reads artifacts and does
  arithmetic.
- The determinism claim is scoped in the manifest: byte-identical *given identical inputs
  including the override file*. Human overrides are external inputs and are pinned by checksum
  rather than claimed to be reproducible.

---

## 12. Figures

| figure | question |
| --- | --- |
| `policy_frontier_xgboost_platt_quarterly_k_1_day.png` | citations against coverage; no point marked optimal |
| `policy_opportunity_cost_xgboost_platt_quarterly.png` | citations given up, per policy, per capacity |
| `policy_coverage_trend_xgboost_platt_k_1_day.png` | who the queue serves, fold by fold, against the 0.1043 line |
| `policy_mechanism_composition_xgboost_platt_quarterly_k_1_day.png` | risk slots against reserve slots |

No figure marks a point optimal, and the cost figure is drawn in citations rather than in a
normalised index: "−34" is a number a department can argue about.

---

## Limitations

1. **This is a re-ordering study.** Component 5's limitation inherited whole: no establishment
   nobody inspected has a label, so nothing here speaks to coverage of the uninspected. Labels
   never move when a policy reorders the queue.
2. **The one-day deltas are inside the noise.** ±1 to ±3 citations out of 348 over 17 folds is
   not a measurable difference, and `coverage_forced_population_share`'s +2 should not be quoted
   as a benefit. The week-scale numbers are the reliable ones.
3. **No confidence interval is placed on any policy delta.** Component 5's sensitivity machinery
   perturbs labels for NDE; nothing equivalent was run over the policy comparison, so "is −34
   citations distinguishable from zero?" is not answered here.
4. **The tie rule decides the production model.** Two defensible rules select two different
   models. The choice is recorded and both outcomes are emitted, but the selection is not robust.
5. **`covid_shift` orders the models differently from the quarterly folds** and is excluded from
   the rule. Selecting on the rolling folds may pick the wrong model under shift; Component 7
   measured that and this component does not resolve it.
6. **The coverage trend is volatile, not monotone.** 2025Q4 sits at 0.667 between two quarters
   near 0.18. Calling it a decline overstates what 17 points support.
7. **Nothing here is fixed for the `__UNKNOWN__` group.** Every policy leaves its treatment
   identical at one day of capacity. The component reports that rather than working around it.
8. **The eligible population is defined by one column.** A different missing-history rule would
   produce a different population and a different price; profile 1 sweeps four but only one is
   implemented.
9. **No fairness criterion is optimised and none is claimed — and ADR 0035 explicitly
   delegated that choice here.** Component 13 declines it, for three measured reasons set out
   in ADR 0038: no criterion has an objective this component is authorised to optimise; the
   criteria are defined over protected characteristics and none is observed anywhere in this
   project; and the under-service that motivated the delegation is not present at the level a
   criterion would operate on — the risk queue over-serves the no-history population fourfold.
   The trade-off is published with its price list instead, and the choice is handed back.
10. **The override mechanism has never been exercised by a real reviewer.** It is tested against
    synthetic override files; whether the contract is usable in an operations room is unknown.
11. **A green run means the policy was applied correctly. It does not mean the policy is the
    right one.** That sentence is printed by the validation report on every run.

---

## Reproducing

```bash
uv run python scripts/profile_policy.py > /tmp/profile.md   # the pre-implementation profiles
uv run sentinel decide --report                              # the production run
uv run sentinel decide --output-dir /tmp/run2 --no-figures   # for byte-comparison
uv run pytest tests/test_policy_*.py tests/test_cli_policy.py -q
```
