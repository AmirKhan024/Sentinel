"""Typed structures for Component 12. No behaviour, no I/O, no clock.

Every measurement structure carries its own support counts. That is deliberate and it is the
component's central discipline: a metric value without the row count behind it is a number
that cannot be read, and the single easiest way for a fairness audit to mislead is to report
a dramatic ratio from a group of twelve rows without saying so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from sentinel.fairness.definitions import (
    GroupDefinitionSpec,
    GroupStatus,
    MetricKind,
    Stage,
)

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"

#: Offenders listed on a failing check before the list is truncated. Matching Components 9
#: and 11: a report that prints ten thousand ids is a report nobody reads.
MAX_OFFENDERS = 20


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One assertion about the audit or the artifact holding it.

    ``severity`` decides whether a failure stops the run. ADR 0034 draws the line: a defect
    in the audit is an error, and a disparity the audit measured is an advisory. A measured
    inequality must never fail a build, because the cheapest way to turn such a build green
    is to change the measurement.
    """

    name: str
    passed: bool
    severity: str
    detail: str
    offenders: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GroupSupport:
    """How much data one group has at one grain, and whether that is enough.

    Model-independent by construction: rows, positives and base rate are properties of the
    fold and the group, not of the estimator. Every calibrated model scores an identical id
    set, which Component 12 checks rather than assumes, so one support record serves every
    model and the two copies cannot drift apart.
    """

    group_definition: str
    group_value: str
    grain: str
    fold_set: str
    #: Empty string at ``fold_set`` grain. Empty rather than null so the column's meaning
    #: never depends on the grain, matching how Component 5 handles ``k_name``.
    fold_id: str

    n_rows: int
    n_positive: int
    n_negative: int
    #: ``None`` when the group has no rows at all -- never 0.0, which is a legitimate rate.
    base_rate: float | None
    #: This group's share of the rows evaluated at this grain.
    representation_share: float

    ranking_status: GroupStatus
    calibration_status: GroupStatus
    #: Why a floor was missed, naming the floor. Empty when both statuses are supported.
    insufficient_reason: str

    @property
    def is_single_class(self) -> bool:
        """True when every row shares one label, so ROC-AUC and NDE are undefined."""
        return self.n_positive == 0 or self.n_negative == 0


@dataclass(frozen=True, slots=True)
class GroupMetric:
    """One metric, for one model, at one stage, restricted to one group.

    ``value`` is ``None`` whenever the group missed its floor or the metric is mathematically
    undefined on the group's rows, and the support counts are populated regardless. An
    unsupported group is a row with a null and a reason, never an absent row -- so a reader
    can always count how many groups were excluded from a comparison, and "equal performance
    across groups" can never rest silently on the ones that were dropped.
    """

    model_name: str
    stage: Stage
    group_definition: str
    group_value: str
    grain: str
    fold_set: str
    fold_id: str

    metric: str
    metric_kind: MetricKind
    #: Empty for a metric that takes no capacity, matching ``evaluation_metrics``.
    k_name: str
    #: 0 when ``k_name`` is empty.
    k: int
    value: float | None

    n_rows: int
    n_positive: int
    n_negative: int
    group_status: GroupStatus
    insufficient_reason: str


@dataclass(frozen=True, slots=True)
class CalibrationComparison:
    """One group's calibration before and after Component 9's Platt map.

    Its own structure rather than two ``GroupMetric`` rows, because the question section 18
    of the brief asks -- *did the global improvement reach this group?* -- should be one
    column rather than a pivot a reader has to get right. ``improved`` is null when either
    side is null, never false: "we could not tell" and "it got worse" are different answers.
    """

    model_name: str
    group_definition: str
    group_value: str
    grain: str
    fold_set: str
    fold_id: str

    metric: str
    base_value: float | None
    calibrated_value: float | None
    delta: float | None
    #: True when calibration moved the metric in the direction that counts as better for it.
    improved: bool | None

    n_rows: int
    n_positive: int
    group_status: GroupStatus


@dataclass(frozen=True, slots=True)
class PriorityRow:
    """Who appears in the top ``k``, and how much of their risk it captured.

    Two different questions live here on purpose, and conflating them is the mistake this
    structure exists to prevent.

    ``selection_rate`` is *representation*: what fraction of this group's rows were
    prioritised. ``capture_rate`` is *effectiveness*: what fraction of this group's actual
    positive outcomes the prioritised set contained. A group can be over-represented in the
    top k while the ranking captures less of its risk than average, and only reporting both
    makes that visible.

    Neither is a target. Base rates differ from 0.220 to 0.566 across supported community
    areas, so a working risk model is *expected* to select at different rates -- equal
    selection would require ignoring a measured difference in outcomes.
    """

    model_name: str
    stage: Stage
    group_definition: str
    group_value: str
    grain: str
    fold_set: str
    fold_id: str

    k_name: str
    k: int

    n_rows: int
    n_positive: int
    #: This group's share of the rows the top-k was drawn from.
    population_share: float
    n_selected: int
    #: This group's share of the selected set.
    selected_share: float
    #: ``n_selected / n_rows``.
    selection_rate: float | None
    #: ``selection_rate / (k / N)``. 1.0 means selected exactly in proportion to presence.
    #: ``None`` rather than infinite when the overall rate is zero.
    selection_rate_ratio: float | None
    #: Positives among this group's selected rows.
    positives_selected: int
    #: ``positives_selected / n_selected`` -- precision inside this group's selected rows.
    precision_in_selected: float | None
    #: ``positives_selected / n_positive`` -- the capture metric. ``None`` when the group has
    #: no positives, never 0.0.
    capture_rate: float | None
    #: The same quantity over every row at this grain, so a group reads against a reference.
    overall_capture_rate: float | None

    group_status: GroupStatus
    insufficient_reason: str


@dataclass(frozen=True, slots=True)
class DisparityRow:
    """One disparity summary over the supported groups of one metric.

    ``n_groups_unsupported`` is not decoration. A spread computed over 51 of 78 community
    areas is a different claim from one computed over all of them, and the two are
    indistinguishable without this column.
    """

    model_name: str
    stage: Stage
    group_definition: str
    grain: str
    fold_set: str
    fold_id: str

    metric: str
    k_name: str
    measure: str
    value: float | None

    #: The pooled population value the deviation measures are taken against. Never a
    #: nominated group -- see ``DISPARITY_REFERENCE``.
    reference_value: float | None
    max_value: float | None
    max_group: str
    max_group_rows: int
    min_value: float | None
    min_group: str
    min_group_rows: int

    n_groups_supported: int
    n_groups_unsupported: int
    #: Why a measure is null when it is: a zero denominator, or too few supported groups.
    undefined_reason: str


@dataclass(frozen=True, slots=True)
class DriftRow:
    """How one metric's group disparity moved across a fold set's folds.

    Only the folds in which the disparity was computable contribute, and ``folds_measured``
    says how many that was. The support policy means it is usually few, so ``trend`` reports
    ``insufficient_folds`` rather than fitting a line through two points.
    """

    model_name: str
    stage: Stage
    group_definition: str
    fold_set: str
    metric: str
    k_name: str
    measure: str

    folds_measured: int
    folds_total: int
    mean_spread: float | None
    sd_spread: float | None
    min_spread: float | None
    max_spread: float | None
    first_fold_id: str
    first_spread: float | None
    last_fold_id: str
    last_spread: float | None
    relative_change: float | None
    #: ``stable`` | ``widening`` | ``narrowing`` | ``insufficient_folds``.
    trend: str


@dataclass(frozen=True, slots=True)
class MissingnessRow:
    """How often one null-rule family is missing inside one group.

    The link Component 11 made available: it measured a missingness indicator ranking second
    and third in importance for two of four models, so an unevenly distributed absence is an
    unevenly distributed *input*. This records the distribution and stops there -- a
    missingness feature is not unfair by definition, and "we have never inspected this place"
    is a true and relevant fact whose removal would not undo the inequality behind it.
    """

    group_definition: str
    group_value: str
    grain: str
    fold_set: str
    fold_id: str

    indicator: str
    source_column: str
    n_rows: int
    n_missing: int
    missing_rate: float
    overall_missing_rate: float
    #: ``missing_rate - overall_missing_rate``. Signed, because which way it goes matters.
    deviation: float
    #: The same rate among this group's rows selected into the top 5%, or ``None`` when the
    #: group had no selected rows.
    missing_rate_in_top_k: float | None
    k_name: str

    group_status: GroupStatus


@dataclass(frozen=True, slots=True)
class AttributionProfileRow:
    """One feature's mean absolute attribution inside one group, for one model.

    Descriptive, pooled across a fold set, and support-gated. A difference between two
    groups' profiles says a model's *reliance* on features differs across those populations.
    Per ADR 0030 an attribution is not a quality measure in the first place -- a model can
    lean hard on a feature that is misleading it, which Component 6 measured happening under
    distribution shift -- so this establishes neither discrimination nor causality.
    """

    model_name: str
    group_definition: str
    group_value: str
    fold_set: str
    feature_name: str
    mean_abs_shap: float
    mean_shap: float
    rank: int
    #: This feature's rank in the same model's profile over every audited row.
    overall_rank: int
    rank_delta: int
    n_rows: int
    #: Spearman rank correlation of this group's whole profile against the overall profile.
    #: Repeated on each of the group's feature rows so the table is readable without a join.
    profile_spearman: float | None
    is_exact: bool
    group_status: GroupStatus


@dataclass(frozen=True, slots=True)
class BootstrapRow:
    """A deterministic interval for one group metric.

    Applied only to the two metrics where sampling variability materially changes the
    reading. Bootstrapping everything because uncertainty sounds sophisticated would triple
    the runtime to decorate numbers nobody would read differently, and a small group stays
    flagged by its support regardless of any interval.
    """

    model_name: str
    stage: Stage
    group_definition: str
    group_value: str
    grain: str
    fold_set: str
    metric: str
    k_name: str

    point_estimate: float | None
    lower: float | None
    upper: float | None
    replications: int
    level: float
    seed: int
    n_rows: int
    #: What is resampled: rows within the group, with replacement.
    scheme: str


@dataclass(slots=True)
class FairnessStats:
    """Counters describing one run. Mutable by design; ``build`` fills it in as it goes."""

    models: int = 0
    group_definitions: int = 0
    folds: int = 0
    audited_rows: int = 0

    groups_observed: int = 0
    groups_supported: int = 0
    groups_insufficient: int = 0

    metric_rows: int = 0
    metric_rows_null: int = 0
    priority_rows: int = 0
    calibration_rows: int = 0
    missingness_rows: int = 0
    attribution_rows: int = 0
    disparity_rows: int = 0
    drift_rows: int = 0
    bootstrap_rows: int = 0

    advisories: int = 0
    seconds: float = 0.0
    #: Set false if any input artifact's checksum changed during the run. An error.
    inputs_unchanged: bool = True


class ArtifactRecord(BaseModel):
    """Provenance for one written file."""

    path: str
    bytes: int
    sha256: str
    row_count: int
    schema_: dict[str, str] = Field(alias="schema")

    model_config = {"populate_by_name": True}


class FairnessManifest(BaseModel):
    """Self-contained provenance and QA record for one group-audit run.

    Pins every input by checksum and records each one again *after* the run, so "Component 12
    changed nothing" is a checkable claim rather than an assurance. That matters more here
    than anywhere else in the project: this component's entire value rests on it being an
    observer, and an observer that quietly rewrote what it observed would invalidate every
    number in the artifact and every earlier component's too.

    ``does_not_establish`` travels in every manifest by decision, not convention. ADR 0035
    records why: a table called ``fairness_group_metrics`` will be read as a finding about
    discrimination unless the boundary is attached to it, and a document can be lost while a
    manifest sits beside the data.
    """

    component: str = "fairness"
    code_version: str
    fairness_definition_version: str
    built_at: str

    features_path: str
    features_sha256: str
    feature_definition_version: str
    calibrated_predictions_path: str
    calibrated_predictions_sha256: str
    categoricals_path: str
    categoricals_sha256: str
    explanations_path: str | None
    explanations_sha256: str | None

    #: Every input checksum re-read after the last table was written. Equal to the values
    #: above on any correct run.
    inputs_unchanged: bool
    input_sha256_after: dict[str, str]

    # The group frame, restated so a consumer never has to import this package.
    audited_group_definitions: list[str]
    refused_group_definitions: dict[str, str]
    group_provenance: dict[str, str]
    group_source_is_a_model_feature: dict[str, bool]

    # The support policy, likewise.
    support_min_rows: int
    support_min_positive: int
    support_min_negative: int
    calibration_min_rows: int
    calibration_bins: int
    groups_observed: int
    groups_supported: int
    groups_insufficient: int

    # The audited surface.
    models: list[str]
    experimental_models: list[str]
    stages: list[str]
    fold_sets: list[str]
    k_levels: list[str]
    threshold_policy: str
    disparity_reference: str

    # Uncertainty.
    bootstrap_replications: int
    bootstrap_seed: int
    bootstrap_metrics: list[str]
    ci_level: float

    # The boundary.
    does_not_establish: list[str]
    blocked: list[str]
    inherited_limitations: list[str]

    checks: list[dict[str, object]]
    advisories: list[str]
    artifacts: list[ArtifactRecord]

    row_counts: dict[str, int]
    seconds: float


@dataclass(slots=True)
class GroupFrame:
    """The audited rows and the provenance of the group values attached to them.

    Carried together rather than as a bare frame because the provenance is what makes the
    frame admissible: which column each value came from, that every one was recorded strictly
    before the row it labels, and that no key mapped to two values.
    """

    #: One row per (model, fold, audited prediction), with a column per group definition.
    frame: object
    definitions: tuple[GroupDefinitionSpec, ...]
    #: Rows in the group source whose value was carried from a strictly earlier inspection.
    as_of_rows: int
    #: The smallest observed lag in days. A zero would mean a row supplied its own attributes.
    min_source_lag_days: int | None
    #: Group values observed per definition, including ``__UNKNOWN__``.
    observed_values: dict[str, tuple[str, ...]] = field(default_factory=dict)


__all__ = [
    "MAX_OFFENDERS",
    "SEVERITY_ERROR",
    "SEVERITY_WARN",
    "ArtifactRecord",
    "AttributionProfileRow",
    "BootstrapRow",
    "CalibrationComparison",
    "DisparityRow",
    "DriftRow",
    "FairnessManifest",
    "FairnessStats",
    "GroupFrame",
    "GroupMetric",
    "GroupSupport",
    "MissingnessRow",
    "PriorityRow",
    "ValidationCheck",
]
