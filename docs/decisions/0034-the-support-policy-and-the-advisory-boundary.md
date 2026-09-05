# ADR 0034 — The support policy, and why a measured disparity is advisory rather than an error

**Status:** Accepted · **Date:** 2026-08-25

## Context

Two questions decide whether a fairness audit is honest, and neither is about which metric to
compute.

The first is **how little data is too little**. Chicago has 77 community areas and Sentinel's
quarterly test windows hold 1,638 to 2,459 rows each. Dividing one by the other gives cells
that cannot support an ROC-AUC, let alone a calibration curve. An audit that computed a number
anyway would publish 78 four-decimal figures of which most are noise, and the loudest
disparity in the table would be the smallest group's.

The second is **what happens when the audit finds something**. If a large group disparity
fails the build, then every future change to this repository is made under pressure to move
that number, and the cheapest way to move it is to change the measurement.

Both were settled from `scripts/profile_fairness.py` before any metric was implemented, in the
order the working agreement requires.

## Decision

### The support thresholds, frozen from measurement

```text
SUPPORT_MIN_ROWS       = 200    ranking and error metrics
SUPPORT_MIN_POSITIVE   =  20
SUPPORT_MIN_NEGATIVE   =  20
CALIBRATION_MIN_ROWS   = 300    ECE, MCE, slope, intercept, Brier, log loss
```

Measured, pooled over the 17 quarterly test windows:

| floor | community areas clearing it | ZIPs clearing it |
|---:|---:|---:|
| 100 rows | 69 / 78 | 58 / 69 |
| **200 rows** | **51 / 78** | **56 / 69** |
| 300 rows | 33 / 78 | 41 / 69 |
| 500 rows | 20 / 78 | 30 / 69 |
| 20 positives **and** 20 negatives | 74 / 78 | 59 / 69 |

**200 rows is where the row floor stops being the binding constraint for ZIP and still
excludes the tail for community area.** The class floors are set below the row floor
deliberately: a group of 250 rows with 4 positives supports no ranking statement, and ROC-AUC
is undefined outright on a single-class group. Twenty of each gives 400 discordant pairs,
which is the smallest number at which the metric is doing arithmetic rather than reporting an
accident.

**The calibration floor is arithmetic rather than taste.** `evaluation.metrics.ece` uses 15
equal-mass bins. Component 9 recorded 27–50 rows per bin as already thin for a selection rule.
Twenty rows per bin needs 300 rows, and that is the floor.

**The bin count is not reduced to make more groups qualify.** Ten bins would let 41 more
community areas through, and every one of their ECEs would be incomparable with Component 9's
global figure — which is the exact comparison section 18 of the brief asks for. A threshold
loosened until the answer arrives is not a threshold.

### The per-fold grain almost never qualifies, and is written anyway

Measured: the median (fold, community area) cell holds **16 rows**. Of 1,288 such cells, 327
reach 30 rows, 57 reach 100, and **4 reach 200**.

So essentially no per-fold group metric will be supported, and the pooled fold-set grain is
the reporting grain. The per-fold rows are still computed and persisted, carrying
`group_status = insufficient_support` and a null value, because a table containing only the
cells that qualified would report identical conclusions while making the shortage invisible.

### Pooling across folds is measurement, not the ADR 0025 leak

Every pooled row is strictly held out for its own fold: it was scored by a model that never
saw it. ADR 0025's prohibition is on *selecting* something using a pooled window — fold N's
calibration window is fold N−1's test window, so a pooled selection reads an earlier fold's
test period. Component 12 selects nothing. It fits nothing. It has no free parameter that
could absorb information from the rows it reads.

What pooling does cost is stated on every pooled row rather than in a footnote: the 17 windows
were scored by 17 differently-fitted models, so a pooled number describes **the system as
operated over 2022Q2–2026Q2**, not one estimator.

`quarterly` and `covid_shift` are never pooled together. Only 11 of 78 community areas clear
200 rows inside the COVID window, so it is reported as a separate stress-test observation.

### Unsupported is a recorded state, never an absence

A group below threshold produces a row with a null `value`, its true `n_rows`,
`n_positive` and `n_negative`, `group_status = insufficient_support`, and an
`insufficient_reason` naming the floor it failed. `validate` has an error-severity check that
the set of group values in the metrics table equals the set observed in the data.

**"Equal performance across groups" may not be claimed on the strength of groups that were
excluded**, and the artifact is built so that a reader can see how many were.

### Every ratio is measured against the pooled population, never against a chosen group

`max/min` ratios report both ends with their group ids and their supports. Deviation is
measured against the pooled reference, which is written into the disparity row as
`reference_value`. Selection-rate ratio is a group's selection rate divided by the overall
selection rate `k/N`.

No group is nominated as the reference. A reference group chosen after seeing the results
would be a conclusion wearing a criterion's clothes, and choosing one before would still be
choosing which neighbourhood counts as normal.

### Zero denominators are handled explicitly, and null is not zero

A ratio with a zero denominator is `null` with a stated reason, never `inf`, never `0.0`, and
never silently dropped. A group with zero positives has no capture rate and no recall — those
are null, and `n_positive = 0` is on the row so the null is legible. Component 11 established
the principle when it wrote nulls rather than zeros for its unsupported model: zero is a
legitimate value here too, so a placeholder zero would read as a real measurement.

### Disparity is advisory. Data integrity is an error.

```text
ERROR      the audit is wrong          -> fails the run, exit 1
ADVISORY   the world is uneven         -> recorded, exit 0
```

**ERROR:** a scored row missing from the group frame; a group value not present in the
declared source; a group mapping dated on or after its row; a fold id that does not match
`assign_split` re-derived from the data; base and calibrated stages confused; a metric row
without support counts; a group observed in the data but absent from the support table; an
input artifact whose sha256 changed during the run; a duplicate primary key; a label or
feature column smuggled into the output.

**ADVISORY:** a wide group ECE spread; a large selection-rate ratio; a wide capture spread; a
group's representation drifting; a group losing support between folds.

**No threshold on a measured disparity fails this build**, and there is deliberately no flag
to make one. The reason is not that disparities do not matter — this component exists because
they do. It is that a red build is a demand for action, and the only actions available to
someone facing a red fairness check are to change the model, change the metric, or change the
threshold. Two of those three are worse than the disparity.

An advisory is recorded in the artifact, printed in the report, and reported in the findings
document. Component 13 owns what to do about it.

### No group-specific calibrator is fitted, and no prediction is modified

Fitting a calibrator per community area would change Component 9, which is closed. It would
also be a substantive fairness decision disguised as a fix: per-group calibration trades
overall calibration for group calibration, needs a per-group calibration window most groups do
not have the rows for, and makes the probability a function of the neighbourhood — which is
the thing ADR 0023 declined to let the *model* do.

If Platt improved the global ECE while worsening it for some groups, this component reports
that and stops.

## Alternatives rejected

**A single fairness score.** One number, easy to track, and the reason the brief forbids it.
Equal calibration and equal selection rates cannot both hold when base rates differ — measured
here at 0.220 to 0.566 across supported community areas — so any scalar is a hidden weighting
of incompatible criteria, chosen by whoever wrote it and invisible to whoever reads it.

**Drop unsupported groups.** Cleaner tables and a better-looking audit. Rejected: it is the
specific failure the brief names, and the resulting document would say "no disparity found" in
a font indistinguishable from "no disparity looked for".

**Bootstrap everything to rescue small groups.** A confidence interval on 12 rows is honest
about its own width and still invites the point estimate to be quoted. Bootstrapping is
applied only to group ECE and top-k capture — the two metrics where sampling variability
materially changes the reading — with a deterministic seed derived from the registry position,
never from `hash()` of a string (MEMORY invariant 92). Small groups are flagged by support
regardless of any interval.

**Set the thresholds after seeing the disparity results.** The alternative this ADR exists to
foreclose, and the one Component 9 had to correct itself on three times in the other
direction. The numbers above come from a profiling script that reads no metric.

**Make an extreme disparity an error "just for the worst case".** Rejected. There is no
principled place to put that line, and wherever it went it would become the number the project
optimises against.

## Consequences

- A green Component 12 run means the audit is internally sound. It does not mean Sentinel is
  fair, and the CLI summary says so on every run rather than only in the documentation.
- Roughly two thirds of the per-fold metric rows will be null. That is the measurement, and
  the artifact is larger for it.
- The drift analysis is thinner than the brief's section 12 imagines, because per-fold group
  metrics mostly do not exist. It is computed over the cells that qualify and reports how many
  did, rather than over an imputed series.
- The thresholds are frozen source literals in `fairness/definitions.py` with an import-time
  guard, following ADR 0025's precedent. Changing one is a source diff.
