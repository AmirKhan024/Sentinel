"""Feature specifications: the single source of truth for the feature table.

Every feature is a declared :class:`FeatureSpec` rather than an ad-hoc column, so
the SQL, the output schema, the data contract and the tests all read from one
list. Adding a column without adding a spec is not possible: the writer builds
its schema from ``FEATURE_SPECS``.

Each spec carries its own missing-value rule, because the rule is the part most
easily got wrong. Findings §6.1: 14,162 target rows have no prior code-era
canvass while 6,442 have code-era history and no priority violation. Those two
groups mean completely different things, and a table that renders both as ``0``
teaches a model that a quarter of establishments have a clean record when nothing
is known about them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# Bumped whenever a feature's formula, the temporal boundary, or a missing-value
# rule changes. Two runs that disagree on this value did not build the same
# features, and the manifest records it.
FEATURE_DEFINITION_VERSION = "v1"

# Trailing windows, in days. Findings §5: the canvass cycle has a 358-day median,
# so a 365-day window is empty for 62% of rows and a 1095-day window for 14.3%.
# The three match the project spec's 1y/2y/3y. More would add collinearity
# without information -- a 365-day count is a subset of the 730-day count.
WINDOW_DAYS: tuple[int, ...] = (365, 730, 1095)


class NullRule(StrEnum):
    """When a feature is NULL. Four rules, applied uniformly (findings §7)."""

    #: Counts are never NULL. 0 means "none observed", which is a true
    #: observation. Every subtype count is emitted beside its total so a reader
    #: can distinguish "none of these" from "none at all".
    NEVER = "never"
    #: Recency is NULL when the event never happened. 0 would mean "today",
    #: which is the opposite of "never".
    NO_PRIOR_CANVASS = "no prior canvass"
    NO_PRIOR_INSPECTION = "no prior inspection of any type"
    #: Rates are NULL when the denominator is zero. 0/0 is not 0.
    NO_INSPECTED_CANVASS = "no prior canvass that was actually inspected"
    #: Priority history is undefined before the 2018 food code (ADR 0009).
    NO_CODE_ERA_CANVASS = "no prior canvass on or after 2018-07-01"


class Family(StrEnum):
    """Grouping used for reporting and for the data contract."""

    CANVASS_HISTORY = "canvass_history"
    PRIORITY_HISTORY = "priority_history"
    WINDOW = "window"
    CONTEXT = "context"
    TENANT = "tenant"
    OBSERVATION = "observation"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One feature, fully specified.

    ``description`` states what the value measures *and* why it could plausibly
    predict the target, because a feature that cannot answer the second question
    should not exist.
    """

    name: str
    dtype: str
    family: Family
    null_rule: NullRule
    description: str
    sources: tuple[str, ...]


_INSPECTED = "(results in Pass / Pass w-Conditions / Fail)"

FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # -- canvass history: the routine series, comparable to the target --------
    FeatureSpec(
        name="prior_canvass_count",
        dtype="int32",
        family=Family.CANVASS_HISTORY,
        null_rule=NullRule.NEVER,
        description=(
            "Routine canvasses recorded for this establishment strictly before the "
            "reference date. Establishes how much is known about the premises at all; "
            "0 is a true observation meaning no prior canvass exists."
        ),
        sources=("inspection_type", "inspection_date"),
    ),
    FeatureSpec(
        name="prior_canvass_count_code_era",
        dtype="int32",
        family=Family.CANVASS_HISTORY,
        null_rule=NullRule.NEVER,
        description=(
            "Prior canvasses on or after 2018-07-01. Exists so a consumer can tell "
            "'no priority violations found' from 'no priority evidence available' -- "
            "the priority features are NULL exactly when this is 0."
        ),
        sources=("inspection_type", "inspection_date"),
    ),
    FeatureSpec(
        name="prior_canvass_inspected_count",
        dtype="int32",
        family=Family.CANVASS_HISTORY,
        null_rule=NullRule.NEVER,
        description=(
            f"Prior canvasses where an inspection actually happened {_INSPECTED}. "
            "Findings §6: 16,517 prior canvasses are Out of Business and 13,077 are "
            "No Entry; folding those into an outcome denominator would count a locked "
            "door as a clean result."
        ),
        sources=("inspection_type", "inspection_date", "results"),
    ),
    FeatureSpec(
        name="days_since_last_canvass",
        dtype="int32",
        family=Family.CANVASS_HISTORY,
        null_rule=NullRule.NO_PRIOR_CANVASS,
        description=(
            "Days from the most recent prior canvass to the reference date. The "
            "cleanest recency signal, because the canvass series is the routine cycle "
            "rather than the event-driven one. NULL when no prior canvass exists -- 0 "
            "would falsely mean 'inspected today'."
        ),
        sources=("inspection_type", "inspection_date"),
    ),
    FeatureSpec(
        name="prior_canvass_fail_count",
        dtype="int32",
        family=Family.CANVASS_HISTORY,
        null_rule=NullRule.NEVER,
        description=(
            "Prior canvasses with result Fail. A history of outright failure is the "
            "most direct evidence of recurring problems, and it is available for the "
            "whole record including the pre-2018 era."
        ),
        sources=("inspection_type", "inspection_date", "results"),
    ),
    FeatureSpec(
        name="prior_canvass_pass_w_conditions_count",
        dtype="int32",
        family=Family.CANVASS_HISTORY,
        null_rule=NullRule.NEVER,
        description=(
            "Prior canvasses with result Pass w/ Conditions. Kept separate from Fail "
            "because Component 3 measured that it carries priority violations 97.9% of "
            "the time -- it behaves like a failure, and pooling it into 'pass' would "
            "hide that."
        ),
        sources=("inspection_type", "inspection_date", "results"),
    ),
    FeatureSpec(
        name="prior_canvass_fail_rate",
        dtype="float64",
        family=Family.CANVASS_HISTORY,
        null_rule=NullRule.NO_INSPECTED_CANVASS,
        description=(
            "prior_canvass_fail_count divided by prior_canvass_inspected_count. "
            "Normalises the failure history by exposure, so a long-observed "
            "establishment is not penalised for having simply been around longer. "
            "NULL when the denominator is 0."
        ),
        sources=("inspection_type", "inspection_date", "results"),
    ),
    FeatureSpec(
        name="fail_at_last_canvass",
        dtype="bool",
        family=Family.CANVASS_HISTORY,
        null_rule=NullRule.NO_PRIOR_CANVASS,
        description=(
            "Whether the most recent prior canvass resulted in Fail. The single most "
            "recent outcome, which the 2015 Chicago model also used and which is the "
            "obvious heuristic baseline. NULL when there is no prior canvass."
        ),
        sources=("inspection_type", "inspection_date", "results"),
    ),
    # -- priority history: code-era only --------------------------------------
    FeatureSpec(
        name="prior_canvass_priority_count",
        dtype="int32",
        family=Family.PRIORITY_HISTORY,
        null_rule=NullRule.NO_CODE_ERA_CANVASS,
        description=(
            "Prior code-era canvasses that found at least one Priority or Priority "
            "Foundation violation. The direct historical analogue of the target. NULL "
            "when no code-era canvass exists, because Priority did not exist before "
            "2018-07-01 (ADR 0009) -- absence of evidence, not evidence of absence."
        ),
        sources=("inspection_type", "inspection_date", "violations"),
    ),
    FeatureSpec(
        name="prior_canvass_priority_foundation_count",
        dtype="int32",
        family=Family.PRIORITY_HISTORY,
        null_rule=NullRule.NO_CODE_ERA_CANVASS,
        description=(
            "Prior code-era canvasses that found at least one Priority Foundation "
            "violation specifically. Separated from the combined count because "
            "Priority Foundation is the more procedural category and may carry "
            "different information."
        ),
        sources=("inspection_type", "inspection_date", "violations"),
    ),
    FeatureSpec(
        name="prior_canvass_priority_rate",
        dtype="float64",
        family=Family.PRIORITY_HISTORY,
        null_rule=NullRule.NO_CODE_ERA_CANVASS,
        description=(
            "prior_canvass_priority_count divided by prior_canvass_count_code_era. The "
            "establishment's historical base rate for the exact event being predicted, "
            "normalised by exposure."
        ),
        sources=("inspection_type", "inspection_date", "violations"),
    ),
    FeatureSpec(
        name="priority_at_last_canvass",
        dtype="bool",
        family=Family.PRIORITY_HISTORY,
        null_rule=NullRule.NO_CODE_ERA_CANVASS,
        description=(
            "Whether the most recent prior code-era canvass found a Priority or "
            "Priority Foundation violation. The most recent evidence of the target "
            "event itself."
        ),
        sources=("inspection_type", "inspection_date", "violations"),
    ),
    # -- context: a different series, labelled as such ------------------------
    FeatureSpec(
        name="prior_inspection_count_any_type",
        dtype="int32",
        family=Family.CONTEXT,
        null_rule=NullRule.NEVER,
        description=(
            "Every prior inspection of any type. Broader than the canvass count and "
            "far denser -- only 401 target rows have none, against 5,615 with no prior "
            "canvass -- so it carries signal for establishments the canvass series "
            "cannot describe."
        ),
        sources=("inspection_date",),
    ),
    FeatureSpec(
        name="days_since_any_inspection",
        dtype="int32",
        family=Family.CONTEXT,
        null_rule=NullRule.NO_PRIOR_INSPECTION,
        description=(
            "Days since the most recent prior inspection of any type. Retained as "
            "labelled context rather than as the primary recency: findings §5 measured "
            "a p25 of 9 days, which is the re-inspection pattern, so this partly "
            "encodes Chicago's scheduling policy rather than the establishment's risk. "
            "Emitted so a later component can ablate it deliberately."
        ),
        sources=("inspection_date",),
    ),
    FeatureSpec(
        name="prior_complaint_count",
        dtype="int32",
        family=Family.CONTEXT,
        null_rule=NullRule.NEVER,
        description=(
            "Prior complaint-driven inspections of any variant. Complaints are "
            "excluded from the target because they are a biased data-generating "
            "process, but a history of them is genuine prior information about the "
            "premises."
        ),
        sources=("inspection_type", "inspection_date"),
    ),
    FeatureSpec(
        name="prior_reinspection_count",
        dtype="int32",
        family=Family.CONTEXT,
        null_rule=NullRule.NEVER,
        description=(
            "Prior re-inspections of any variant. A re-inspection only happens after a "
            "failure, so the count is a proxy for how often this establishment has "
            "needed follow-up."
        ),
        sources=("inspection_type", "inspection_date"),
    ),
    FeatureSpec(
        name="prior_license_inspection_count",
        dtype="int32",
        family=Family.CONTEXT,
        null_rule=NullRule.NEVER,
        description=(
            "Prior licence-related inspections. These cluster around openings and "
            "ownership transfers, so the count is an indirect signal of churn at the "
            "premises."
        ),
        sources=("inspection_type", "inspection_date"),
    ),
    # -- tenant change: the spec §3.3 signal, exposed not acted on ------------
    FeatureSpec(
        name="name_changed_since_last_canvass",
        dtype="bool",
        family=Family.TENANT,
        null_rule=NullRule.NO_PRIOR_CANVASS,
        description=(
            "Whether the business name on the reference inspection differs from the "
            "name on the most recent prior canvass. Component 2 tracks physical "
            "premises, so history can span a change of tenant; this exposes the "
            "transition instead of silently resetting history (findings §8). Measured "
            "at 4.96% of consecutive canvass pairs."
        ),
        sources=("dba_name", "inspection_type", "inspection_date"),
    ),
    FeatureSpec(
        name="prior_canvass_count_current_name",
        dtype="int32",
        family=Family.TENANT,
        null_rule=NullRule.NEVER,
        description=(
            "Prior canvasses carrying the same business name as the reference "
            "inspection. Read beside prior_canvass_count, it says how much of the "
            "premises' history belongs to the current tenant."
        ),
        sources=("dba_name", "inspection_type", "inspection_date"),
    ),
    # -- observation window ---------------------------------------------------
    FeatureSpec(
        name="days_since_first_inspection",
        dtype="int32",
        family=Family.OBSERVATION,
        null_rule=NullRule.NO_PRIOR_INSPECTION,
        description=(
            "Days from the earliest recorded inspection at this establishment to the "
            "reference date. Named for what it is: time since the first *record*, not "
            "the age of the business, which this dataset cannot observe. Provides the "
            "exposure denominator for reading the raw counts."
        ),
        sources=("inspection_date",),
    ),
)


def _window_specs() -> tuple[FeatureSpec, ...]:
    """Build the windowed features, paired so an empty window stays legible.

    Each priority-event count is emitted beside its canvass count for the same
    window. Findings §5.1: a 365-day window is empty for 62% of rows, so a bare
    ``0`` events would be indistinguishable from "clean". The pair resolves it.
    """
    specs: list[FeatureSpec] = []
    for days in WINDOW_DAYS:
        specs.append(
            FeatureSpec(
                name=f"canvasses_last_{days}d",
                dtype="int32",
                family=Family.WINDOW,
                null_rule=NullRule.NEVER,
                description=(
                    f"Canvasses in the half-open window [reference date - {days} days, "
                    f"reference date). Read beside the priority-event count for the "
                    f"same window, it distinguishes 'not inspected' from 'inspected "
                    f"and clean'."
                ),
                sources=("inspection_type", "inspection_date"),
            )
        )
        specs.append(
            FeatureSpec(
                name=f"canvass_priority_events_last_{days}d",
                dtype="int32",
                family=Family.WINDOW,
                null_rule=NullRule.NEVER,
                description=(
                    f"Code-era canvasses in the last {days} days that found a Priority "
                    f"or Priority Foundation violation. Recent problems are more "
                    f"informative than old ones; 0 means none in the window, which the "
                    f"paired canvass count makes interpretable."
                ),
                sources=("inspection_type", "inspection_date", "violations"),
            )
        )
    return tuple(specs)


FEATURE_SPECS = FEATURE_SPECS + _window_specs()

#: The only columns a model may consume.
FEATURE_COLUMNS: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)

#: Identity and timing. ``inspection_date`` is the as-of boundary, not a feature.
KEY_COLUMNS: tuple[str, ...] = (
    "establishment_id",
    "inspection_date",
    "target_inspection_id",
)

#: Carried so the table is directly trainable. Never inputs.
LABEL_COLUMNS: tuple[str, ...] = ("target", "target_status")

#: Provenance. ``code_era_phase`` is a stratification variable for Component 5,
#: not a predictor -- it describes the regulatory regime, not the establishment.
PROVENANCE_COLUMNS: tuple[str, ...] = ("code_era_phase", "feature_definition_version")


def spec_by_name(name: str) -> FeatureSpec:
    """Look up one feature specification."""
    for spec in FEATURE_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown feature: {name}")


def specs_in_family(family: Family) -> tuple[FeatureSpec, ...]:
    """Every specification in one family."""
    return tuple(spec for spec in FEATURE_SPECS if spec.family is family)
