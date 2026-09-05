"""Frozen specifications and every pre-declared constant for Component 12.

Nothing else in this package may hold a magic number. Components 9 and 11 established the
rule and it binds hardest here, because a support threshold chosen after seeing which
neighbourhoods came out badly is not a threshold -- it is a result being selected. Every
constant below is justified by a measurement in ``docs/analysis/fairness_findings.md``,
produced by ``scripts/profile_fairness.py`` before this module existed, and frozen in
ADR 0034.

**The group registry is the component's most important declaration**, and it holds the
*refused* definitions as well as the audited ones. Ward, census tract, point geography and
city/state are entries here with their reasons attached, not omissions -- so a reader asking
"why is there no ward breakdown?" finds the measurement rather than silence, and
``fairness_group_definitions`` can carry the refusals into the artifact where a lost document
cannot take them with it. See ADR 0033.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sentinel.calibration.definitions import CANDIDATE_REGISTRY
from sentinel.evaluation.metrics import DEFAULT_CALIBRATION_BINS

#: Bumped when anything in this module changes the meaning of an emitted column.
FAIRNESS_DEFINITION_VERSION = "v1"

#: The absence token Component 8 writes when nothing could be carried forward. It is a real
#: group value here, never a null to be dropped -- see :data:`UNKNOWN_IS_A_GROUP`.
UNKNOWN_GROUP = "__UNKNOWN__"

#: Why the unknown-geography group is audited rather than filtered out. Measured: 686 of
#: 57,727 rows carry it, and the set is a superset of the 401 rows with no prior inspection
#: of any type -- exactly the rows Component 4 cannot compute a recency for, and exactly the
#: rows the null-rule family indicators fire on. Component 11 measured two of four models
#: ranking one of those indicators second or third in importance. The group with no geography
#: is the group with no history, so dropping it would remove the single most interesting row
#: set from a missingness audit.
UNKNOWN_IS_A_GROUP = (
    "__UNKNOWN__ is a first-class group value, not a null. It is a superset of the rows with "
    "no prior inspection of any type, which is the same population the null-rule family "
    "indicators encode and Component 11 measured the models leaning on."
)


class GroupStatus(StrEnum):
    """Whether a group had enough data for the metric on the row to mean anything."""

    #: Cleared every applicable floor. The value is populated.
    SUPPORTED = "supported"
    #: Below a floor. The value is null, the counts are real, and the reason is stated.
    INSUFFICIENT_SUPPORT = "insufficient_support"


class GroupDefinitionStatus(StrEnum):
    """Whether a candidate group definition is audited or refused."""

    AUDITED = "audited"
    REFUSED = "refused"


class Grain(StrEnum):
    """The temporal aggregation a measurement was taken at.

    Both are persisted. ``FOLD`` rows are almost all ``insufficient_support`` -- the median
    (fold, community area) cell holds 16 rows -- and they are written anyway, because a table
    containing only the cells that qualified would report identical conclusions while making
    the shortage invisible.
    """

    #: One fold's test window.
    FOLD = "fold"
    #: Every test window in one fold set, pooled. The reporting grain.
    #:
    #: Legitimate, and not the leak ADR 0025 forbids: every pooled row is strictly held out
    #: for its own fold. What it costs is that the windows were scored by differently-fitted
    #: models, so a pooled number describes the system as operated rather than one estimator.
    FOLD_SET = "fold_set"


class Stage(StrEnum):
    """Which of the two probabilities on a calibrated prediction row was measured.

    Both live on the same row of Component 9's artifact -- ``base_score`` and ``score`` -- so
    the audit needs no join to compare them, and confusing them is the failure mode
    ``validate.stages_are_not_confused`` exists to catch. MEMORY invariant 71: an
    uncalibrated probability must never be described as calibrated.
    """

    #: ``base_score``: the uncalibrated output of the Component 6/7/8 estimator.
    BASE = "base"
    #: ``score``: the Platt-calibrated probability Component 9 froze.
    CALIBRATED = "calibrated"


class MetricKind(StrEnum):
    """Which family a metric belongs to, and therefore which support floor gates it.

    Component 5's ``RANKING_METRICS`` / ``PROBABILITY_METRICS`` separation is load-bearing
    there and is inherited here, with a third member for the top-k error behaviour that is
    reported only at capacity-derived cutoffs.
    """

    RANKING = "ranking"
    PROBABILITY = "probability"
    #: Confusion-matrix behaviour at a descriptive top-k cutoff. Never a deployment policy.
    THRESHOLD_AUDIT = "threshold_audit"


@dataclass(frozen=True, slots=True)
class GroupDefinitionSpec:
    """One candidate way of grouping the audited rows, audited or refused.

    ``source_column`` names where the value is read from. For an audited definition that is a
    column of Component 8's as-of categorical table; for a refused one it is the raw column
    that would have supplied it, so the refusal points at something a reader can go and look
    at rather than at nothing.

    ``is_model_feature`` is false for every entry and is emitted anyway. It is the fact that
    stops "the model does not use community area, therefore it is fair" from being read into
    the artifact: a model with no geographic input can still behave differently across
    geography, which is the entire reason this component measures behaviour.
    """

    name: str
    status: GroupDefinitionStatus
    source_column: str
    provenance: str
    rationale: str
    is_model_feature: bool = False
    refusal_reason: str = ""
    version: str = "v1"


#: Every geographic grouping this dataset could support, audited and refused alike.
#:
#: The two audited definitions are read from ``neural_categoricals_<stamp>.parquet``, where
#: each value is the one recorded at the establishment's most recent inspection of any type
#: **strictly before** the row's own date. Measured: that value and the value recorded on the
#: row itself disagree on **0 of 57,041** community-area rows and **0 of 57,326** ZIP rows, so
#: the temporally safe choice costs nothing. See ADR 0033.
GROUP_DEFINITION_REGISTRY: tuple[GroupDefinitionSpec, ...] = (
    GroupDefinitionSpec(
        name="community_area",
        status=GroupDefinitionStatus.AUDITED,
        source_column="community_area",
        provenance=(
            "Component 8 as-of layer, carried from raw ':@computed_region_vrxf_vc4k'. A "
            "Socrata computed region -- the platform joins the row's coordinates against a "
            "boundary layer and returns that layer's row index -- so it is a stable "
            "identifier but NOT necessarily the city's official community-area number. No "
            "boundary file is ingested by this project, so no neighbourhood name is printed "
            "anywhere: an off-by-one in a boundary index would attribute a measured "
            "disparity to the wrong neighbourhood in the one document whose purpose is to "
            "be trusted about which neighbourhood."
        ),
        rationale=(
            "Chicago's 77 community areas have been fixed since the 1920s, which is why the "
            "city publishes statistics against them, and that stability is what makes "
            "attaching one to a 2019 row temporally safe. ADR 0023 handed this definition to "
            "Component 12 explicitly."
        ),
    ),
    GroupDefinitionSpec(
        name="zip",
        status=GroupDefinitionStatus.AUDITED,
        source_column="zip",
        provenance=(
            "Component 8 as-of layer, carried from the raw 'zip' column's leading five "
            "digits. Recorded on the inspection record rather than spatially derived, which "
            "is why its coverage is better than community area's."
        ),
        rationale=(
            "Measurably better supported than community area -- 56 of 69 ZIPs clear the "
            "200-row floor against 51 of 78 community areas -- so a disparity appearing "
            "under one geography and not the other is information about how robust the "
            "finding is. ADR 0023 requires it to be read with the same demographic-proxy "
            "caveat, so auditing it adds no new claim."
        ),
    ),
    GroupDefinitionSpec(
        name="ward",
        status=GroupDefinitionStatus.REFUSED,
        source_column=":@computed_region_43wa_7qmu",
        provenance="raw only; a second, 2003-2015 vintage exists at ':@computed_region_awaf_s7ux'",
        rationale=(
            "Chicago assigns inspectors by district, so ward is the closest available "
            "thing to a route grouping."
        ),
        refusal_reason=(
            "Not temporally stable. The two published ward layers assign different region "
            "ids to 56,451 of 57,403 rows (98.3%). A ward identifier is a property of a "
            "boundary version, not of a place, so attaching the current ward to a 2019 row "
            "assigns it to a district that did not exist when it was inspected. That the "
            "publisher ships two vintages at all is the evidence. ADR 0019 separately "
            "refused ward as an inspector proxy."
        ),
    ),
    GroupDefinitionSpec(
        name="census_tract",
        status=GroupDefinitionStatus.REFUSED,
        source_column=":@computed_region_bdys_3d7i",
        provenance="raw only",
        rationale="The finest published geography, and the one demographic data is keyed to.",
        refusal_reason=(
            "797 groups over 32,696 quarterly test rows is roughly 41 rows each before any "
            "fold split. No group would clear the 200-row floor, so the table would be a "
            "table of nulls -- which reports nothing while looking thorough."
        ),
    ),
    GroupDefinitionSpec(
        name="point_geography",
        status=GroupDefinitionStatus.REFUSED,
        source_column="latitude",
        provenance="raw only; 18,931 distinct coordinate pairs",
        rationale="A continuous geography needs no boundary file and never goes stale.",
        refusal_reason=(
            "ADR 0023 already rejected continuous geography for Component 8 and the reason "
            "transfers exactly: it is not less of a proxy for a protected characteristic, "
            "only less legible -- and a less legible proxy is worse for an audit, not better."
        ),
    ),
    GroupDefinitionSpec(
        name="municipality",
        status=GroupDefinitionStatus.REFUSED,
        source_column="city",
        provenance="raw only",
        rationale="The obvious top of any geographic hierarchy.",
        refusal_reason=(
            "Degenerate. 312,957 of 314,245 rows say CHICAGO, and the 95 distinct values "
            "include 'Chicago', 'chicago' and 'CCHICAGO'. A group definition whose "
            "cardinality comes from typing errors is not a group definition."
        ),
    ),
    GroupDefinitionSpec(
        name="facility_type",
        status=GroupDefinitionStatus.REFUSED,
        source_column="facility_type",
        provenance=(
            "Component 8 as-of layer; 169 distinct values after case and whitespace normalisation"
        ),
        rationale="A meaningful operational dimension with good support.",
        refusal_reason=(
            "Out of scope rather than unusable: this component audits geographic equity, and "
            "facility type is not geography. It is also free text whose synonyms are "
            "deliberately not merged (ADR 0022), so 'GROCERY' and 'GROCERY STORE' are "
            "distinct groups. A later component wanting an operational dimension has this "
            "entry to turn on."
        ),
    ),
)

GROUP_DEFINITIONS_BY_NAME: dict[str, GroupDefinitionSpec] = {
    spec.name: spec for spec in GROUP_DEFINITION_REGISTRY
}

#: The definitions actually audited, in declared order.
AUDITED_GROUP_DEFINITIONS: tuple[str, ...] = tuple(
    spec.name for spec in GROUP_DEFINITION_REGISTRY if spec.status is GroupDefinitionStatus.AUDITED
)


# --- the support policy, frozen from measurement (ADR 0034) ------------------

#: Minimum rows for a ranking or threshold metric. Measured: 51 of 78 community areas and 56
#: of 69 ZIPs clear this pooled over the 17 quarterly folds. 200 is where the row floor stops
#: being the binding constraint for ZIP while still excluding community area's tail.
SUPPORT_MIN_ROWS = 200

#: Minimum positives. ROC-AUC is undefined on a single-class group, and a group of 250 rows
#: with four positives supports no ranking statement. Twenty of each gives 400 discordant
#: pairs -- the smallest count at which the metric is arithmetic rather than an accident.
SUPPORT_MIN_POSITIVE = 20

#: Minimum negatives, for the same reason and by the same argument.
SUPPORT_MIN_NEGATIVE = 20

#: Minimum rows for a calibration metric. Arithmetic, not taste: ``evaluation.metrics.ece``
#: uses 15 equal-mass bins and Component 9 recorded 27-50 rows per bin as already thin for a
#: selection rule. Twenty rows per bin needs 300 rows.
#:
#: **The bin count is deliberately not reduced.** Ten bins would let 41 more community areas
#: through and every one of their ECEs would be incomparable with Component 9's global
#: figure -- which is the exact comparison this component exists to make.
CALIBRATION_MIN_ROWS = DEFAULT_CALIBRATION_BINS * 20

#: Minimum explained rows for a per-group attribution profile. Measured: 56 of 312
#: (model, community area) cells and 84 of 252 (model, ZIP) cells clear it, pooled over the
#: quarterly folds. Component 11's sample is 300 rows per (model, fold) and re-running it
#: larger is forbidden -- it would change the rows every published number rests on -- so this
#: floor is set against the sample that exists.
ATTRIBUTION_MIN_ROWS = 100

#: The bins a group reliability curve is computed at. The canonical value, imported rather
#: than restated, so a group ECE is the same arithmetic as Component 9's global one.
GROUP_CALIBRATION_BINS = DEFAULT_CALIBRATION_BINS


# --- the priority audit ------------------------------------------------------

#: The top-k cutoffs the priority and capture audit reports at.
#:
#: Both families are kept. The percentage cutoffs make groups comparable across folds whose
#: sizes range 1,638 to 8,840 rows; the capacity cutoffs are what the city could actually
#: work in a day and in a week, which is the operational question. Neither is invented --
#: ``evaluation.simulate.capacity_k_values`` derives them from each window's own measured
#: median daily rate, and this component calls it rather than reimplementing it.
K_LEVELS: tuple[str, ...] = ("k_pct_01", "k_pct_05", "k_pct_10", "k_1_day", "k_1_week")

#: Why no probability threshold is offered, and why there is no flag to add one.
#:
#: A cutoff at p = 0.5 is a number this project has never derived from anything, and
#: per-group error rates reported at one would read as a deployment policy. Component 13 owns
#: decision policy. Following ADR 0030's treatment of probability-space attribution, the
#: option is refused in prose rather than declared and left unreachable -- a rejection that
#: has never been observed is indistinguishable from one that cannot happen.
THRESHOLD_POLICY = (
    "descriptive threshold audit: error rates are reported only at capacity-derived and "
    "percentage top-k cutoffs, never at a probability threshold. Not a deployment policy."
)


# --- disparity measures ------------------------------------------------------


class DisparityMeasure(StrEnum):
    """The interpretable disparity summaries. Deliberately several, never one score.

    A single fairness number would be a hidden weighting of mutually incompatible criteria.
    Equal calibration and equal selection rates cannot both hold when base rates differ, and
    they differ here from 0.220 to 0.566 across supported community areas -- so any scalar
    would encode a choice made by whoever wrote it and invisible to whoever read it.
    """

    #: ``max - min`` over supported groups. Absolute, in the metric's own units.
    SPREAD = "spread"
    #: ``max / min``. Null rather than infinite when the minimum is zero.
    RATIO = "ratio"
    #: ``max |group - reference|``, the largest deviation from the pooled population value.
    MAX_DEVIATION = "max_deviation"
    #: Rows-weighted standard deviation across supported groups.
    WEIGHTED_SD = "weighted_sd"


#: What every ratio and deviation is measured against.
#:
#: The pooled population value, never a nominated group. A reference group chosen after
#: seeing the results would be a conclusion wearing a criterion's clothes; choosing one
#: beforehand would still be choosing which neighbourhood counts as normal.
DISPARITY_REFERENCE = "pooled population value over the same rows, never a nominated group"


# --- drift -------------------------------------------------------------------

#: Folds a metric must be supported in before its disparity series is called stable or
#: drifting. Below this the series is reported with ``trend = "insufficient_folds"``.
#:
#: Three, matching Component 11's ``MIN_STABILITY_FOLDS``. Two points are a line through any
#: two numbers, and the brief is explicit that a trend may not be claimed from one or two
#: folds.
DRIFT_MIN_FOLDS = 3

#: Relative change in a disparity spread, first supported fold to last, above which the
#: series is called ``widening`` or ``narrowing`` rather than ``stable``.
#:
#: A quarter, chosen before any disparity series existed. A threshold set afterwards would
#: be a conclusion with a criterion bolted on -- the mistake ADR 0030 named when it froze
#: ``RANK_DRIFT_THRESHOLD`` in advance for the same reason.
DRIFT_MATERIAL_CHANGE = 0.25


# --- uncertainty -------------------------------------------------------------

#: Replications per bootstrap interval, matching Component 9's ``BOOTSTRAP_REPLICATIONS``.
BOOTSTRAP_REPLICATIONS = 1000

#: Seed base. An integer literal, never ``hash()`` of a string: Python salts ``str`` hashing
#: per process, which is what made Component 9's bootstrap non-reproducible until the key
#: became a registry position. MEMORY invariant 92.
BOOTSTRAP_SEED = 20260826

#: Two-sided interval level.
CI_LEVEL = 0.95

#: Both resampling schemes, run for every interval rather than one chosen in advance.
#:
#: Component 9's reasoning transfers directly and matters more here: establishments recur
#: inside a neighbourhood on a 358-day median canvass cycle and their rows share an as-of
#: history, so an i.i.d. row bootstrap understates the standard error of a group metric.
#: Running both settles the objection with a measurement rather than a caveat, and a reader
#: who distrusts one can read the other.
BOOTSTRAP_SCHEMES: tuple[str, ...] = ("row", "establishment_block")

#: The metrics that get an interval, and only these.
#:
#: Bootstrapping everything because uncertainty sounds sophisticated would triple the runtime
#: to decorate numbers whose reading does not change. These two are where sampling
#: variability materially affects interpretation: a group ECE over 300 rows and a capture
#: rate over a handful of captured positives. Small groups stay flagged by support regardless
#: of any interval -- an interval on twelve rows is honest about its own width and still
#: invites the point estimate to be quoted.
BOOTSTRAP_METRICS: tuple[str, ...] = ("ece", "capture_rate")


# --- figures -----------------------------------------------------------------

#: Supported groups drawn per figure. Seventy-eight tiny labels is an unreadable chart, and
#: an unreadable chart is a decorative one. The full table is the source of truth and the
#: contract records this display policy so a reader knows a figure is a view rather than the
#: data.
DISPLAY_TOP_N = 20

#: Groups given their own reliability diagram: the best-supported, per model.
RELIABILITY_GROUPS = 4


# --- advisory thresholds (ADR 0034) ------------------------------------------

#: Group ECE spread above which an advisory is recorded. Advisory only: it never fails a run.
ADVISORY_ECE_SPREAD = 0.05

#: Selection-rate ratio outside ``[1/x, x]`` at which an advisory is recorded.
ADVISORY_SELECTION_RATIO = 2.0

#: Top-k capture-rate spread above which an advisory is recorded.
ADVISORY_CAPTURE_SPREAD = 0.20

#: Representation-share travel across folds above which an advisory is recorded.
ADVISORY_REPRESENTATION_TRAVEL = 0.05


# --- the boundary ------------------------------------------------------------

#: What a Component 12 result does not establish. Written into every manifest and printed by
#: the CLI on every run, so the boundary travels with the numbers rather than living only in
#: a document that can be lost or not read. See ADR 0035.
DOES_NOT_ESTABLISH: tuple[str, ...] = (
    "causality -- every number is observational; nothing is randomised and no counterfactual "
    "is constructed",
    "discrimination -- no model here has a geographic input at all, and a measured difference "
    "arises through correlated features rather than through a group attribute",
    "absence of bias -- equal measured performance on supported groups is not evidence of "
    "fairness, and 27 of 78 community areas fall below the support floor",
    "legal or regulatory compliance -- no protected characteristic is observed anywhere in "
    "this project, and a correlate is not the attribute",
    "ethical acceptability -- whether a measured difference is acceptable is a policy "
    "judgement this component is not delegated",
    "equal treatment -- the target is that a violation was CITED, not that an establishment "
    "was unsafe, and ADR 0019 records that the two cannot be separated on this data",
    "an optimal fairness policy -- no intervention is implemented, tested or recommended, and "
    "the standard criteria are mutually incompatible when base rates differ",
)

#: Experiments deliberately not run, with the reason. Carried in every manifest, matching the
#: convention Components 7, 8 and 11 established.
BLOCKED_EXPERIMENTS: tuple[str, ...] = (
    "protected-class fairness metrics -- no race, income, ACS, census or deprivation variable "
    "is ingested anywhere in this project (ADR 0023). Adding one is a Component 1 extension "
    "with its own provenance and boundary-vintage questions, and this component does not "
    "license a proxy in the meantime",
    "inspector-effect decomposition -- the dataset publishes 22 columns and none identifies a "
    "person (ADR 0019), so a group difference in citation rate cannot be split into "
    "establishment risk versus differential inspection practice",
    "ward and census-tract group definitions -- refused with measurements, see the registry "
    "above and ADR 0033",
    "neighbourhood names for community-area region ids -- no boundary file is ingested, and "
    "guessing the mapping would attribute a measured disparity to the wrong neighbourhood",
    "group-specific recalibration -- would change Component 9, which is closed, and is a "
    "substantive fairness decision disguised as a fix (ADR 0034)",
    "any debiasing, reweighting, threshold adjustment or prediction modification -- this "
    "component is an audit and modifies nothing",
    "model selection on fairness grounds -- Component 12 produces evidence; a policy "
    "component settles MEMORY open question 13",
)


class FairnessDefinitionError(ValueError):
    """A registry or constant that could mislead, leak, or claim more than it delivers."""


def group_definition_for(name: str) -> GroupDefinitionSpec:
    """The spec for one group definition, audited or refused.

    Raises on a refused definition rather than returning it, so a caller cannot audit
    ``ward`` by asking politely. The refusal reason is in the message, because a caller who
    asked deserves to be told why rather than that they may not.
    """
    spec = GROUP_DEFINITIONS_BY_NAME.get(name)
    if spec is None:
        known = ", ".join(sorted(GROUP_DEFINITIONS_BY_NAME))
        raise FairnessDefinitionError(f"unknown group definition {name!r}; known: {known}")
    if spec.status is GroupDefinitionStatus.REFUSED:
        raise FairnessDefinitionError(
            f"group definition {name!r} is refused and may not be audited: {spec.refusal_reason}"
        )
    return spec


def support_floor_for(kind: MetricKind) -> int:
    """The row floor gating one metric family.

    Probability metrics need more rows than ranking metrics because a binned calibration
    statistic spends its rows across bins, and this returns the difference rather than
    letting each call site remember it.
    """
    if kind is MetricKind.PROBABILITY:
        return CALIBRATION_MIN_ROWS
    return SUPPORT_MIN_ROWS


def _guard_registry() -> None:
    """Reject a registry or constant set that could mislead, leak or over-claim.

    Raises rather than asserts, at import time, so a bad spec cannot be constructed and then
    used -- ``python -O`` strips asserts and would strip this. Mirrors the guards Components
    6 through 11 each run, and adds the checks only a fairness component needs.
    """
    seen: set[str] = set()
    for spec in GROUP_DEFINITION_REGISTRY:
        if spec.name in seen:
            raise FairnessDefinitionError(f"duplicate group definition: {spec.name}")
        seen.add(spec.name)

        if spec.status is GroupDefinitionStatus.REFUSED and not spec.refusal_reason:
            raise FairnessDefinitionError(
                f"{spec.name}: refused without a reason. A refusal that states no measurement "
                "is indistinguishable from an omission, and the whole point of holding "
                "refused definitions in the registry is that the reason travels."
            )
        if spec.status is GroupDefinitionStatus.AUDITED and spec.refusal_reason:
            raise FairnessDefinitionError(f"{spec.name}: audited but carries a refusal reason")
        if spec.is_model_feature:
            raise FairnessDefinitionError(
                f"{spec.name}: declared as a model feature. No geographic column reaches "
                "Component 4's table, and a registry claiming otherwise would license the "
                "'the model does not use it, therefore it is fair' reading this component "
                "exists to refute."
            )

    if not AUDITED_GROUP_DEFINITIONS:
        raise FairnessDefinitionError("no group definition is audited")

    if SUPPORT_MIN_POSITIVE < 1 or SUPPORT_MIN_NEGATIVE < 1:
        raise FairnessDefinitionError(
            "a class floor below 1 would admit a single-class group, on which ROC-AUC is "
            "undefined and PR-AUC is degenerate"
        )
    if CALIBRATION_MIN_ROWS < SUPPORT_MIN_ROWS:
        raise FairnessDefinitionError(
            "the calibration floor must be at least the ranking floor: a binned statistic "
            "spends its rows across bins and cannot need fewer of them"
        )
    if CALIBRATION_MIN_ROWS < GROUP_CALIBRATION_BINS:
        raise FairnessDefinitionError(
            "fewer rows than bins would leave empty bins, and an ECE over empty bins is a "
            "mean over nothing"
        )
    if GROUP_CALIBRATION_BINS != DEFAULT_CALIBRATION_BINS:
        raise FairnessDefinitionError(
            "the group bin count must equal Component 5's, or a group ECE is not comparable "
            "with Component 9's global ECE -- which is the comparison this component exists "
            "to make"
        )
    if not 0.0 < CI_LEVEL < 1.0:
        raise FairnessDefinitionError(f"CI_LEVEL must be in (0, 1), got {CI_LEVEL}")
    if DRIFT_MIN_FOLDS < 3:
        raise FairnessDefinitionError(
            "a trend may not be claimed from one or two folds; two points are a line through "
            "any two numbers"
        )

    known_metrics = {spec.name for spec in CANDIDATE_REGISTRY}
    if not known_metrics:
        raise FairnessDefinitionError("Component 9's candidate registry is empty")

    if not DOES_NOT_ESTABLISH:
        raise FairnessDefinitionError(
            "the boundary list is empty. It is written into every manifest and printed on "
            "every run precisely so it cannot be dropped."
        )


_guard_registry()


__all__ = [
    "ADVISORY_CAPTURE_SPREAD",
    "ADVISORY_ECE_SPREAD",
    "ADVISORY_REPRESENTATION_TRAVEL",
    "ADVISORY_SELECTION_RATIO",
    "ATTRIBUTION_MIN_ROWS",
    "AUDITED_GROUP_DEFINITIONS",
    "BLOCKED_EXPERIMENTS",
    "BOOTSTRAP_METRICS",
    "BOOTSTRAP_REPLICATIONS",
    "BOOTSTRAP_SCHEMES",
    "BOOTSTRAP_SEED",
    "CALIBRATION_MIN_ROWS",
    "CI_LEVEL",
    "DISPARITY_REFERENCE",
    "DISPLAY_TOP_N",
    "DOES_NOT_ESTABLISH",
    "DRIFT_MATERIAL_CHANGE",
    "DRIFT_MIN_FOLDS",
    "FAIRNESS_DEFINITION_VERSION",
    "GROUP_CALIBRATION_BINS",
    "GROUP_DEFINITIONS_BY_NAME",
    "GROUP_DEFINITION_REGISTRY",
    "K_LEVELS",
    "RELIABILITY_GROUPS",
    "SUPPORT_MIN_NEGATIVE",
    "SUPPORT_MIN_POSITIVE",
    "SUPPORT_MIN_ROWS",
    "THRESHOLD_POLICY",
    "UNKNOWN_GROUP",
    "UNKNOWN_IS_A_GROUP",
    "DisparityMeasure",
    "FairnessDefinitionError",
    "Grain",
    "GroupDefinitionSpec",
    "GroupDefinitionStatus",
    "GroupStatus",
    "MetricKind",
    "Stage",
    "group_definition_for",
    "support_floor_for",
]
