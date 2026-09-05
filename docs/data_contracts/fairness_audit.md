# Data contract — group-behaviour audit (Component 12)

**Producer:** `sentinel audit-fairness` · **Layer:** `data/processed/fairness/` (ADR 0032)
**Definition version:** `fairness_definition_version = "v1"`

Ten Parquet tables and one manifest, all timestamped `<slug>_<UTC>.parquet` with the manifest
keyed to `fairness_group_metrics`. Column order is part of this contract; changing it is a
contract change.

---

## 0. ⚠ What this artifact is, and what it is not

It **is** a measurement of how Sentinel's held-out predictions, calibrated probabilities,
ranking behaviour and top-k prioritisation differ across the geographic groups this data can
define.

It is **not** any of the following, and ADR 0035 records why for each:

* **not evidence of discrimination** — no model in this project has a geographic input at
  all, so a measured difference arises through correlated features rather than through a
  group attribute. The converse matters equally: the *absence* of a group feature does not
  prove the absence of a disparity, which is why this component measures behaviour rather
  than inspecting a feature list.
* **not causal** — every number is observational. Nothing is randomised, no counterfactual is
  constructed, and no intervention is simulated.
* **not a protected-class finding** — no race, income, ACS, census or deprivation variable is
  ingested anywhere in this project. Community areas correlate strongly with race and income
  by construction, and a correlate is not the attribute.
* **not evidence of the absence of bias** — 27 of 78 community areas fall below the support
  floor and are excluded from every comparison. A system can be even across the groups it
  could measure and fail badly for one it could not.
* **not a legal compliance position, an ethical judgement, or a fairness policy.**

And one limitation inherited rather than discovered. ADR 0019 records that this dataset
publishes 22 columns and none identifies an inspector. The target is that a Priority violation
was **cited**, not that an establishment was unsafe, and Chicago assigns inspectors by
district — so a geographic difference in citation rate is confounded with inspection practice
by construction. **Nothing in this artifact separates the two.**

**A green validation run means the audit is internally sound. It does not mean Sentinel is
fair.**

---

## 1. Identity and file naming

```
data/processed/fairness/fairness_group_definitions_<YYYYMMDDTHHMMSSZ>.parquet
data/processed/fairness/fairness_group_support_<stamp>.parquet
data/processed/fairness/fairness_group_metrics_<stamp>.parquet
data/processed/fairness/fairness_group_calibration_<stamp>.parquet
data/processed/fairness/fairness_priority_audit_<stamp>.parquet
data/processed/fairness/fairness_group_missingness_<stamp>.parquet
data/processed/fairness/fairness_attribution_profiles_<stamp>.parquet
data/processed/fairness/fairness_disparity_<stamp>.parquet
data/processed/fairness/fairness_drift_<stamp>.parquet
data/processed/fairness/fairness_bootstrap_<stamp>.parquet
data/processed/fairness/manifest_fairness_group_metrics_<stamp>.json
```

zstd-compressed Parquet. Every table is written in a **total** sort order, so two runs over
identical inputs produce byte-identical files.

## 2. The inputs, and the fact that none of them moved

| input | why it is read |
| --- | --- |
| `calibrated_predictions_<stamp>.parquet` | the audited rows. Carries `score` **and** `base_score` on one row, which is what lets the two stages be compared with no join |
| `as_of_features_<stamp>.parquet` | the outcome label, the reference date, and the null-rule families the missingness audit reads |
| `neural_categoricals_<stamp>.parquet` | where `community_area` and `zip` live (ADR 0022) |
| `explanation_values_<stamp>.parquet` | Component 11's attributions, grouped rather than regenerated. **Optional** |

**This component re-executes nothing.** Component 9 had to regenerate scores that were never
recorded and Component 11 had to regenerate the models themselves, both behind ADR 0026's
bit-identity gate. Every input here already exists on disk, so the integrity claim is the
opposite one — *nothing moved* — and it is checked by re-reading every input's sha256 after
the last table is written. `inputs_unchanged` on the manifest is that comparison, and a
mismatch is an error-severity failure.

## 3. Group definitions, and why two of them and not seven

Audited: **`community_area`** and **`zip`**, both read from Component 8's as-of layer, where
each value is the one recorded at the establishment's most recent inspection of any type
**strictly before** the row's own date.

The alternative was the value recorded on the row itself. Measured, before choosing:

| group definition | rows where both values exist | disagreements |
| --- | ---: | ---: |
| `community_area` | 57,041 | **0** |
| `zip` | 57,326 | **0** |

The two never disagree, so the temporally safe option is free. See ADR 0033.

**Refused, and the refusals are rows in `fairness_group_definitions` rather than prose:**
ward (the two published ward layers assign different region ids to 98.3% of rows — a ward
identifier is a property of a boundary version, not of a place), census tract (797 groups over
32,696 rows), point geography, city/state (degenerate), facility type (out of scope: not
geography).

**`community_area` is a Socrata computed region id, not a neighbourhood name.** No boundary
file is ingested by this project, so no neighbourhood name is printed anywhere in this
artifact. Guessing the mapping would attribute a measured disparity to the wrong neighbourhood
in the one document whose purpose is to be trusted about which neighbourhood.

**`__UNKNOWN__` is a group value, not a null.** It is a superset of the rows with no prior
inspection of any type — exactly the rows the null-rule family indicators fire on — so
dropping it would remove the most interesting row set from a missingness audit.

## 4. Support: the gate every number passes through

```text
SUPPORT_MIN_ROWS       = 200    ranking and threshold metrics
SUPPORT_MIN_POSITIVE   =  20
SUPPORT_MIN_NEGATIVE   =  20
CALIBRATION_MIN_ROWS   = 300    ECE, MCE, slope, intercept, Brier, log loss
```

Frozen from `scripts/profile_fairness.py` before any disparity was computed (ADR 0034). The
calibration floor is arithmetic rather than taste: `evaluation.metrics.ece` uses 15 equal-mass
bins, and 20 rows per bin needs 300 rows. **The bin count is not reduced to let more groups
through**, because a group ECE at a different bin count would be incomparable with Component
9's global figure — which is the comparison this component exists to make.

**A group below a floor is a row with real counts, a null value, a status and a stated
reason. It is never an absent row.** `validate.no_group_disappeared` compares the support
table against the group values observed in the data.

## 5. Two grains, and what pooling costs

| `grain` | what it covers |
| --- | --- |
| `fold` | one fold's test window |
| `fold_set` | every test window in one fold set, pooled. **The reporting grain** |

Pooling is measurement, not the leak ADR 0025 forbids: every pooled row is strictly held out
for its own fold. What it costs is stated on the rows rather than in a footnote — the 17
quarterly windows were scored by 17 differently-fitted models, so **a pooled number describes
the system as operated over 2022Q2–2026Q2, not one estimator.**

`quarterly` and `covid_shift` are never pooled together, and `validate.covid_was_not_pooled`
enforces it.

The per-fold grain is thin by measurement: the median (fold, community area) cell holds **16
rows**, and 4 of 1,288 reach the 200-row floor. Per-fold metric rows are therefore emitted
only for groups that cleared their floor; the *support* table still carries every observed
group at every grain, which is what keeps the shortage visible.

---

## 6. `fairness_group_definitions` — provenance, including the refusals

| column | type | meaning |
| --- | --- | --- |
| `group_definition` | Utf8 | the candidate's name |
| `status` | Utf8 | `audited` or `refused` |
| `source_column` | Utf8 | where the value is (or would be) read from |
| `provenance` | Utf8 | how the value is derived, and its caveats |
| `rationale` | Utf8 | why it was a candidate at all |
| `is_model_feature` | Boolean | **false on every row.** Emitted anyway: it is the fact that stops "the model does not use community area, therefore it is fair" being read into the artifact |
| `refusal_reason` | Utf8 | the measurement that refused it; empty for an audited definition |
| `distinct_values` | Int64 | observed in the audited rows; 0 for a refused definition |
| `unknown_rows` | Int64 | rows carrying `__UNKNOWN__` |
| `audited_rows` | Int64 | rows this run measured |
| `fairness_definition_version` | Utf8 | `v1` |

Sort key: `group_definition`.

## 7. `fairness_group_support` — counts, base rates and the support decision

| column | type | meaning |
| --- | --- | --- |
| `group_definition`, `group_value` | Utf8 | the group |
| `grain`, `fold_set`, `fold_id` | Utf8 | `fold_id` is `""` at `fold_set` grain, never null |
| `n_rows`, `n_positive`, `n_negative` | Int64 | always real, whatever the status |
| `base_rate` | Float64 | null on a zero-row group; 0.0 is a legitimate rate and means something else |
| `representation_share` | Float64 | this group's share of the rows evaluated at this grain |
| `ranking_status` | Utf8 | `supported` / `insufficient_support`, against the 200-row floor |
| `calibration_status` | Utf8 | the same, against the 300-row floor |
| `insufficient_reason` | Utf8 | every floor that was missed, not only the first |
| `fairness_definition_version` | Utf8 | `v1` |

Sort key: `group_definition, grain, fold_set, fold_id, group_value`.

**Model-independent by construction.** Rows and positives are properties of the fold and the
group, and every calibrated model scores an identical id set — which this component checks
rather than assumes.

## 8. `fairness_group_metrics` — the long grain

| column | type | meaning |
| --- | --- | --- |
| `model_name` | Utf8 | the calibrated name, e.g. `xgboost_platt` |
| `stage` | Utf8 | `base` (the uncalibrated `base_score`) or `calibrated` (the `score`) |
| `group_definition`, `group_value`, `grain`, `fold_set`, `fold_id` | Utf8 | the cell |
| `metric` | Utf8 | see below |
| `metric_kind` | Utf8 | `ranking` / `probability` / `threshold_audit` |
| `k_name`, `k` | Utf8, Int64 | `""` and 0 for a metric that takes no capacity |
| `value` | Float64 | **null** when the group missed its floor or the metric is undefined |
| `n_rows`, `n_positive`, `n_negative` | Int64 | the support behind the value |
| `group_status`, `insufficient_reason` | Utf8 | why a null is null |
| `fairness_definition_version` | Utf8 | `v1` |

Sort key: `model_name, stage, group_definition, grain, fold_set, fold_id, metric, k_name,
group_value`.

Metrics, all computed by the implementation that **owns** them rather than reimplemented:

* ranking — `roc_auc`, `pr_auc` (`evaluation.metrics`), `nde` (`evaluation.simulate`)
* probability — `brier`, `log_loss`, `ece`, `mce` (`evaluation.metrics`),
  `calibration_slope`, `calibration_intercept` (`calibration.metrics`)
* threshold audit — `precision_at_k`, `recall_at_k`, `lift_at_k` (`evaluation.metrics`), plus
  `true_positive_rate`, `false_positive_rate`, `false_discovery_rate`

**`nde` here is a *within-group* efficiency**: if only this group's inspections were
reordered, how efficiently would its positives surface. That is deliberately not the same
question as the group's share of a city-wide top-k, which is §10's.

## 9. `fairness_group_calibration` — did Platt reach this group?

| column | type | meaning |
| --- | --- | --- |
| `model_name`, `group_definition`, `group_value`, `grain`, `fold_set`, `fold_id` | Utf8 | the cell |
| `metric` | Utf8 | one of the probability metrics |
| `base_value`, `calibrated_value`, `delta` | Float64 | before, after, and the difference |
| `improved` | Boolean | **null when either side is null, never false.** "We could not tell" and "it got worse" are different answers. For `calibration_slope` it means *closer to 1.0*, because 0.6 and 1.4 are both miscalibrated |
| `n_rows`, `n_positive` | Int64 | support |
| `group_status` | Utf8 | the calibration status |
| `fairness_definition_version` | Utf8 | `v1` |

Sort key: `model_name, group_definition, grain, fold_set, fold_id, metric, group_value`.

Its own table rather than a pivot over §8, because "did the global improvement reach this
group?" should be one column rather than a join a reader has to construct correctly.

## 10. `fairness_priority_audit` — who is prioritised, and what it captured

Two quantities, deliberately not combined:

```text
selection_rate   n_selected / n_rows            representation: was this group prioritised?
capture_rate     positives_selected / n_pos     effectiveness:  was that useful to them?
```

| column | type | meaning |
| --- | --- | --- |
| `model_name`, `stage`, `group_definition`, `group_value`, `grain`, `fold_set`, `fold_id` | Utf8 | the cell |
| `k_name`, `k` | Utf8, Int64 | the cutoff, from `simulate.capacity_k_values` |
| `n_rows`, `n_positive`, `population_share` | Int64/Float64 | the group in the population |
| `n_selected`, `selected_share`, `selection_rate` | Int64/Float64 | the group in the top k |
| `selection_rate_ratio` | Float64 | `selection_rate / (k/N)`. 1.0 = proportionate. **Null rather than infinite** on a zero denominator |
| `positives_selected`, `precision_in_selected` | Int64/Float64 | null when nothing was selected |
| `capture_rate` | Float64 | **null when the group has no positives** — never 0.0, which would read as total failure rather than nothing to capture |
| `overall_capture_rate` | Float64 | the same quantity over every row at this grain |
| `group_status`, `insufficient_reason` | Utf8 | support |
| `fairness_definition_version` | Utf8 | `v1` |

Sort key: `model_name, stage, group_definition, grain, fold_set, fold_id, k_name, group_value`.

**The cutoff is city-wide and competitive**: taken over every audited row, then groups counted
inside it. That is what makes `capture_rate` different from `recall_at_k`, which selects its
top k *within* the rows it is handed — a group's capture rate is what a competition against
every other group left it.

**Neither number is a target.** Outcome rates differ from 0.220 to 0.566 across supported
community areas, so a working risk model is *expected* to select at different rates; parity
would require ignoring a measured difference in outcomes.

**⚠ Every threshold figure here is a descriptive threshold audit, not a deployment policy.**
The cutoffs are rank positions derived from real inspection capacity. No probability threshold
is offered, and there is no flag to add one — Component 13 owns decision policy.

## 11. `fairness_group_missingness` — data availability as a fairness surface

| column | type | meaning |
| --- | --- | --- |
| `group_definition`, `group_value`, `grain`, `fold_set`, `fold_id` | Utf8 | the cell |
| `indicator`, `source_column` | Utf8 | the null-rule family, and the column whose mask defines it |
| `n_rows`, `n_missing`, `missing_rate` | Int64/Float64 | the distribution |
| `overall_missing_rate`, `deviation` | Float64 | against the pooled rate; signed |
| `missing_rate_in_top_k`, `k_name` | Float64, Utf8 | the same rate among the group's prioritised rows; null when it placed nobody |
| `group_status` | Utf8 | support |
| `fairness_definition_version` | Utf8 | `v1` |

Sort key: `group_definition, grain, fold_set, fold_id, indicator, group_value`.

Component 11 measured `missing_no_code_era_canvass` ranking **third** for the logistic model
and **second** for the network — the *absence* of a record is among the most informative
signals either has. This table measures how that absence is distributed and **stops there**.
Missingness is not unfair by definition: "we have never inspected this place" is a true and
relevant fact, and removing the feature would not undo the inequality in inspection history
behind it.

## 12. `fairness_attribution_profiles` — does the model reason differently by group?

| column | type | meaning |
| --- | --- | --- |
| `model_name` | Utf8 | the **base** name (`xgboost`), matching Component 11's artifact |
| `group_definition`, `group_value`, `fold_set` | Utf8 | the cell. Pooled; no per-fold grain |
| `feature_name`, `mean_abs_shap`, `mean_shap` | Utf8/Float64 | the profile |
| `rank`, `overall_rank`, `rank_delta` | Int64 | within the group, overall, and the travel |
| `n_rows` | Int64 | explained rows behind the profile |
| `profile_spearman` | Float64 | the group's whole ranking against the model's overall one |
| `is_exact` | Boolean | **false for the network.** Its per-row values are permutation estimates |
| `group_status` | Utf8 | always `supported`; the floor is `ATTRIBUTION_MIN_ROWS = 100` |
| `fairness_definition_version` | Utf8 | `v1` |

Sort key: `model_name, group_definition, fold_set, group_value, rank, feature_name`.

**Component 11's artifact is grouped, never regenerated.** Re-running `sentinel explain` at a
different `--sample-size` would change the rows every published Component 11 number rests on.
Measured: the median (model, community area) cell holds 40 explained rows and 56 of 312 clear
100 — enough for a *profile* comparison, whose global statistic Component 11 measured
converging far faster than any individual value, and enough for nothing more.

**⚠ An attribution is not a quality measure** (ADR 0030). A model can lean hard on a feature
that is misleading it, which Component 6 measured happening under distribution shift. A
difference between two groups' profiles is a difference in model *reliance*, not evidence of
discrimination and not causal.

## 13. `fairness_disparity` — four measures, never one score

| column | type | meaning |
| --- | --- | --- |
| `model_name`, `stage`, `group_definition`, `grain`, `fold_set`, `fold_id`, `metric`, `k_name` | Utf8 | the comparable cell |
| `measure` | Utf8 | `spread`, `ratio`, `max_deviation`, `weighted_sd` |
| `value` | Float64 | null when undefined, with a reason |
| `reference_value` | Float64 | the **pooled population value** over the same rows, never a nominated group |
| `max_value`, `max_group`, `max_group_rows` | Float64/Utf8/Int64 | the highest group, with its support |
| `min_value`, `min_group`, `min_group_rows` | Float64/Utf8/Int64 | the lowest, likewise |
| `n_groups_supported` | Int64 | how many groups the measure was computed over |
| `n_groups_unsupported` | Int64 | **how many were excluded.** A spread over 51 of 78 is a different claim from one over all 78 |
| `undefined_reason` | Utf8 | too few groups, a vanished denominator, or no reference |
| `fairness_definition_version` | Utf8 | `v1` |

Sort key: `model_name, stage, group_definition, grain, fold_set, fold_id, metric, k_name,
measure`.

`capture_rate` is folded in from §10 because it is a group metric in every respect except
which table it lives in.

**There is deliberately no single fairness score.** Calibration parity and selection-rate
parity cannot both hold when base rates differ, and they differ here from 0.220 to 0.566 — so
a scalar would be a hidden weighting of incompatible criteria, chosen by whoever wrote it and
invisible to whoever read it.

## 14. `fairness_drift` — is the gap itself moving?

| column | type | meaning |
| --- | --- | --- |
| `model_name`, `stage`, `group_definition`, `fold_set`, `metric`, `k_name`, `measure` | Utf8 | the series |
| `folds_measured` / `folds_total` | Int64 | **how many folds the disparity was computable in**, against how many existed |
| `mean_spread`, `min_spread`, `max_spread` | Float64 | over the measured folds |
| `sd_spread` | Float64 | **a fold-to-fold spread, NOT a confidence interval.** The folds overlap and share establishments on a 358-day median canvass cycle |
| `first_fold_id`, `first_spread`, `last_fold_id`, `last_spread` | Utf8/Float64 | the endpoints |
| `relative_change` | Float64 | null when the series started at exactly zero |
| `trend` | Utf8 | `stable` / `widening` / `narrowing` / **`insufficient_folds`** |
| `fairness_definition_version` | Utf8 | `v1` |

Sort key: `model_name, stage, group_definition, fold_set, metric, k_name, measure`.

`insufficient_folds` is the common case and it names itself rather than hiding inside
`stable` — a series that could not be measured is not a series that did not move.

## 15. `fairness_bootstrap` — deterministic intervals, for two metrics only

| column | type | meaning |
| --- | --- | --- |
| `model_name`, `stage`, `group_definition`, `group_value`, `grain`, `fold_set`, `metric`, `k_name` | Utf8 | the cell |
| `point_estimate`, `lower`, `upper` | Float64 | percentile interval |
| `replications`, `level` | Int64/Float64 | 1,000 and 0.95 |
| `seed` | Int64 | derived from the model's position plus a frozen base — **never `hash()` of a name**, because Python salts `str` hashing per process (MEMORY invariant 92) |
| `n_rows` | Int64 | support |
| `scheme` | Utf8 | `row` or `establishment_block` |
| `fairness_definition_version` | Utf8 | `v1` |

Sort key: `model_name, stage, group_definition, grain, fold_set, metric, k_name, group_value,
scheme`.

**Both schemes are run for every interval.** Establishments recur inside a neighbourhood and
their rows share an as-of history, so an i.i.d. row bootstrap understates the standard error;
running both settles that with a measurement rather than a caveat, exactly as Component 9 did.

Only `ece` and `capture_rate` get intervals. Bootstrapping every cell would triple the runtime
to decorate numbers nobody would read differently, and a small group stays flagged by its
support regardless of any interval.

---

## 16. ⚠ What a consumer may not use

1. **Nothing here may be joined onto a feature table.** These tables are keyed by *group*
   rather than by row, which is what stops them becoming per-establishment features — and a
   number meaning "the model was well calibrated in this neighbourhood last quarter", joined
   back onto training rows, would be the most self-fulfilling input this project could build.
2. **No number here is a verdict.** See §0.
3. **Do not read `stage = base` as a calibrated probability, or the reverse.** MEMORY
   invariant 71, and `validate.stages_are_not_confused` re-derives the distinction against
   the committed artifact with `==`.
4. **Do not read a `fold_set`-grain number as a statement about one model.** It pools 17
   differently-fitted models.
5. **Do not average `covid_shift` into a quarterly mean**, and do not claim a trend from it.
6. **Do not quote a disparity without `n_groups_supported` and `n_groups_unsupported`.**
7. **Do not quote an extreme group without its row count.** Every extreme in §13 carries one.
8. **Do not treat a null as a zero.** Nulls here are load-bearing: an unsupported group, a
   ratio with a vanished denominator, a group with no positives to capture.
9. **Do not select a model on these numbers.** Component 12 produces evidence; MEMORY open
   question 13 is a policy component's to settle, and this component is recorded as blocked
   from it in every manifest it emits.
10. **Do not read `community_area` as an official neighbourhood number.** See §3.

## Provenance and integrity

The manifest pins every input by sha256 **before** the run and records every one again
**after** the last table is written, so "Component 12 changed nothing" is a checkable claim.
It also restates, so a consumer never has to import this package: the audited and refused
group definitions with their reasons, the support policy and its floors, the models and
stages, the k levels, the threshold policy, the disparity reference, the bootstrap
configuration, `does_not_establish`, `blocked`, and `inherited_limitations`.

### Guarantees a consumer may rely on

1. Every audited row corresponds to exactly one row of the committed prediction artifact, and
   the two id sets are **equal** rather than one containing the other.
2. Every group value appears in the declared source column; none was invented or remapped.
3. Every group value was recorded at an inspection **strictly before** the row it labels.
4. Every group observed in the data has a support row at every grain.
5. No metric row carries a value without its support counts, and no unsupported row carries a
   value.
6. Every support status is re-derivable from the counts beside it and the frozen floors.
7. No outcome, score or probability column appears in any table.
8. Every table is in a total sort order with no duplicate key.
9. The input artifacts are byte-identical before and after the run.
10. Re-running against the same inputs produces byte-identical tables.

### Reproducing

```bash
uv run python scripts/profile_fairness.py            # read-only; fixes the frozen constants
uv run sentinel audit-fairness --dry-run --report    # audit and validate, write nothing
uv run sentinel audit-fairness --report              # ~162 s
```

Resolves the most recent feature table, calibrated predictions, categoricals and explanations
unless `--features`, `--calibrated-predictions`, `--categoricals` and `--explanations` are
given. Unlike `calibrate` and `explain`, this command has **no bit-identity gate and no
thread sensitivity**: it fits nothing and re-executes nothing.
