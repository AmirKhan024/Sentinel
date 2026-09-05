"""Model specifications: the single source of truth for what Component 6 fits.

Every model is a declared :class:`ModelSpec` rather than an estimator constructed at a
call site, so the registry, the manifest, the data contract and the tests all read from
one list. There is no dynamic import and no way to fit something the registry does not
name.

The feature partitions here are **derived** from ``FEATURE_SPECS``, never hand-listed.
Component 4 owns which features exist and which can be NULL; if it adds a feature or
changes a null rule, that change arrives here automatically and the tests that pin the
partition fail loudly rather than the preprocessing silently doing the wrong thing to a
column nobody declared.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from sentinel.features.definitions import (
    FEATURE_COLUMNS,
    FEATURE_SPECS,
    KEY_COLUMNS,
    LABEL_COLUMNS,
    PROVENANCE_COLUMNS,
    Family,
    NullRule,
)

#: Bumped whenever a model's feature set, hyperparameters or preprocessing changes in a
#: way that makes two runs incomparable. Recorded in the manifest and in every
#: coefficient row.
MODEL_DEFINITION_VERSION = "v1"

#: Prefix for the missingness indicators built in ``preprocess``. One per null-rule
#: family, never one per column -- see the package docstring.
INDICATOR_PREFIX = "missing_"

#: Fixed for every Component 6 model. Recorded rather than tuned: a baseline exists to
#: be trustworthy, not optimal, and tuning needs a train/calibration protocol that
#: belongs with the model that benefits from it.
#:
#: ``class_weight=None`` is deliberate. Measured prevalence is 52.52% overall and
#: 0.379-0.492 per test window, so there is no imbalance to correct, and resampling
#: would corrupt the probability scale Component 9 depends on.
#:
#: ``random_state`` has **no effect** on ``solver="lbfgs"``, which has no stochastic
#: component. It is recorded because it documents intent and because a future switch to
#: ``saga`` would make it load-bearing. What makes this component reproducible is the
#: canonical sort in ``train.fit_fold``, not this value.
LOGISTIC_PARAMS: Mapping[str, object] = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
}

DEFAULT_SEED = 42

#: Columns that are never model inputs, whatever a spec claims. Identity, the label and
#: provenance. ``inspection_date`` is the as-of boundary and a legitimate *sort* key;
#: ``code_era_phase`` is a legitimate *stratification* variable. Neither is a predictor.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset((*KEY_COLUMNS, *LABEL_COLUMNS, *PROVENANCE_COLUMNS))

#: The 2015 CDPH model's inputs, and whether the current data contract can reach them.
#: Reproduced from ``docs/analysis/temporal_evaluation_findings.md`` §10 and
#: ``evaluation/rankers.py`` so the table travels attached to the model that needs it.
#: Three of ten families are reachable, so a faithful replication is not possible and
#: what Component 6 builds is labelled an approximation.
CDPH_2015_UNREACHABLE_INPUTS: tuple[str, ...] = (
    "inspector identity (deliberately excluded -- audit Finding 1)",
    "nearby 311 complaints (not ingested)",
    "burglary intensity (not ingested)",
    "alcohol / tobacco licence (not ingested)",
    "weather / temperature (not ingested; needs NOAA GHCN USW00094846)",
    "facility type (in the raw snapshot, not in Component 4's feature table)",
    "CDPH risk category (in the raw snapshot, not in Component 4's feature table)",
)

CDPH_APPROXIMATION_NOTE = (
    "APPROXIMATION, not a replication of Chicago's deployed 2015 model. Only 3 of that "
    "model's 10 input families are reachable from the current data contract (prior "
    "violation history, time since last inspection, business tenure). Unreachable: "
    + "; ".join(CDPH_2015_UNREACHABLE_INPUTS)
    + ". The 2015 published result also used a different food code, a different target "
    "definition and a 14.1% base rate against this data's 52.52%, so its numbers are "
    "not a benchmark this project can be measured against. No coefficient, input or "
    "historical behaviour is fabricated."
)

#: The feature families the 2015 input list can actually reach. Canvass history covers
#: prior violation history and time-since-last-inspection; priority history and the
#: trailing windows cover violation history over time; observation is business tenure.
#: Excluded: CONTEXT (all-type inspection counts -- complaint/reinspection/licence
#: inspections are not the 2015 model's 311 complaints or licence records) and TENANT
#: (name change, which has no 2015 counterpart).
CDPH_FAMILIES: tuple[Family, ...] = (
    Family.CANVASS_HISTORY,
    Family.PRIORITY_HISTORY,
    Family.WINDOW,
    Family.OBSERVATION,
)

#: The one feature the Component 5 handoff singled out for ablation. It partly encodes
#: scheduling policy rather than establishment risk (p25 of 9 days is the re-inspection
#: pattern), so the ablation is built as a second fitted model and the harness decides.
ABLATED_FEATURE = "days_since_any_inspection"


class MissingStrategy(StrEnum):
    """How one column's NULLs are filled. Declared per column, never inferred."""

    #: Never NULL by Component 4's contract. 0 is a true observation; pass it through.
    PASSTHROUGH = "passthrough"
    #: Train-window median. Partition-based, so exactly order-invariant.
    MEDIAN = "median"
    #: Constant 0.0. Used for nullable booleans, where a median would be filled with
    #: whichever class dominates that fold's training window. Measured:
    #: ``priority_at_last_canvass`` drifts 0.6310 -> 0.5056 across the 17 quarterly
    #: folds, so the median fill sits 0.0056 from flipping and would reverse the
    #: encoding of "unknown" mid-sequence for no substantive reason.
    CONSTANT_FALSE = "constant_false"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One baseline model, fully specified.

    ``feature_columns`` is explicit and closed. It is never "every column except the
    target": a future Component 4 release adding a metadata column would silently
    become a model input under that rule.

    ``approximation_note`` is non-None only for a model that cannot be what its name
    suggests, and it is written into the manifest and the data contract so the caveat
    travels with the artifact rather than living in a document.
    """

    name: str
    version: str
    description: str
    feature_columns: tuple[str, ...]
    is_probability: bool
    seed: int
    params: Mapping[str, object]
    approximation_note: str | None = None


def _cdph_feature_columns() -> tuple[str, ...]:
    """Features corresponding to the 2015 input families this data can reach."""
    return tuple(spec.name for spec in FEATURE_SPECS if spec.family in CDPH_FAMILIES)


def _without_ablated_feature() -> tuple[str, ...]:
    return tuple(name for name in FEATURE_COLUMNS if name != ABLATED_FEATURE)


MODEL_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="logistic_regression",
        version="v1",
        description=(
            "L2-penalised logistic regression on all 26 as-of features plus four "
            "null-rule family indicators. The primary Component 6 baseline, and the "
            "model the project specification names."
        ),
        feature_columns=FEATURE_COLUMNS,
        is_probability=True,
        seed=DEFAULT_SEED,
        params=LOGISTIC_PARAMS,
    ),
    ModelSpec(
        name="logistic_regression_no_scheduling",
        version="v1",
        description=(
            f"Identical to logistic_regression but without {ABLATED_FEATURE}. Built as "
            "a separate fitted model rather than a post-hoc adjustment: refitting is "
            "the only way to see what the model does when it cannot lean on the "
            "feature, whereas dropping a column at scoring time measures a "
            "mis-specified model instead."
        ),
        feature_columns=_without_ablated_feature(),
        is_probability=True,
        seed=DEFAULT_SEED,
        params=LOGISTIC_PARAMS,
    ),
    ModelSpec(
        name="cdph_2015_approximation",
        version="v1",
        description=(
            "L2-penalised logistic regression restricted to the feature families the "
            "2015 CDPH model's input list can actually reach. An approximation, "
            "explicitly labelled; see approximation_note."
        ),
        feature_columns=_cdph_feature_columns(),
        is_probability=True,
        seed=DEFAULT_SEED,
        params=LOGISTIC_PARAMS,
        approximation_note=CDPH_APPROXIMATION_NOTE,
    ),
)

MODELS_BY_NAME: dict[str, ModelSpec] = {spec.name: spec for spec in MODEL_REGISTRY}


# --- derived feature partitions ---------------------------------------------


def nullable_columns() -> tuple[str, ...]:
    """Features that can be NULL, in ``FEATURE_SPECS`` order. Measured: 10 of 26."""
    return tuple(spec.name for spec in FEATURE_SPECS if spec.null_rule is not NullRule.NEVER)


def never_null_columns() -> tuple[str, ...]:
    """Features that are never NULL. For these, 0 is a real observation."""
    return tuple(spec.name for spec in FEATURE_SPECS if spec.null_rule is NullRule.NEVER)


def null_families() -> tuple[NullRule, ...]:
    """The distinct non-NEVER null rules, in a stable order. Measured: 4.

    Sorted by value so the indicator column order is a declared property of the rule
    set rather than an accident of how ``FEATURE_SPECS`` happens to be written.
    """
    present = {spec.null_rule for spec in FEATURE_SPECS if spec.null_rule is not NullRule.NEVER}
    return tuple(sorted(present, key=lambda rule: rule.value))


def columns_in_family(rule: NullRule) -> tuple[str, ...]:
    """Every feature sharing one null rule.

    Passing ``NullRule.NEVER`` returns the 16 never-null features, which is a coherent
    answer to the question asked -- but ``NEVER`` is not an indicator family, so
    :func:`indicator_source_column` refuses it rather than returning one of those.
    """
    return tuple(spec.name for spec in FEATURE_SPECS if spec.null_rule is rule)


def family_indicator_name(rule: NullRule) -> str:
    """The indicator column for one null-rule family.

    Named from the enum member, not its value, because the value is prose with spaces
    and a date in it.
    """
    return f"{INDICATOR_PREFIX}{rule.name.lower()}"


def indicator_columns() -> tuple[str, ...]:
    """The four family indicators, in declared order."""
    return tuple(family_indicator_name(rule) for rule in null_families())


def indicator_source_column(rule: NullRule) -> str:
    """The column whose null mask defines a family's indicator.

    Any member would do: the masks within a family are byte-identical on the full
    snapshot, and ``validate`` re-asserts that per fold rather than trusting it.

    ``NullRule.NEVER`` is refused. It has 16 member features, so returning ``members[0]``
    would hand back a never-null column as the source of a missingness indicator -- a
    silently wrong answer rather than an error.
    """
    if rule is NullRule.NEVER:
        raise KeyError(
            "NullRule.NEVER is not an indicator family: a never-null column has no "
            "missingness to indicate"
        )
    members = columns_in_family(rule)
    if not members:
        raise KeyError(f"null rule {rule.value!r} has no member features")
    return members[0]


def missing_strategy_for(column: str) -> MissingStrategy:
    """How one feature's NULLs are filled.

    Booleans get a constant rather than a median for the drift reason recorded on
    :class:`MissingStrategy`.
    """
    spec = next((s for s in FEATURE_SPECS if s.name == column), None)
    if spec is None:
        raise KeyError(f"Unknown feature: {column}")
    if spec.null_rule is NullRule.NEVER:
        return MissingStrategy.PASSTHROUGH
    if spec.dtype == "bool":
        return MissingStrategy.CONSTANT_FALSE
    return MissingStrategy.MEDIAN


def boolean_columns() -> tuple[str, ...]:
    """Features stored as booleans. Cast to float before any estimator sees them."""
    return tuple(spec.name for spec in FEATURE_SPECS if spec.dtype == "bool")


def max_iter_of(spec: ModelSpec) -> int:
    """The solver's iteration cap.

    ``params`` is a ``Mapping[str, object]`` because it is passed straight to an untyped
    estimator, so the one value the convergence check needs is narrowed here rather than
    at each call site.
    """
    value = spec.params.get("max_iter", 0)
    return value if isinstance(value, int) else 0


def spec_for(name: str) -> ModelSpec:
    """Look up one model specification."""
    if name not in MODELS_BY_NAME:
        raise KeyError(f"Unknown model: {name}. Registered: {', '.join(sorted(MODELS_BY_NAME))}")
    return MODELS_BY_NAME[name]


def _guard_registry() -> None:
    """Reject a registry that could leak or that names a column Component 4 lacks.

    Raises rather than asserts: ``assert`` is stripped under ``python -O``, and a guard
    that disappears under an optimisation flag is not a guard. Runs at import time, so
    a bad spec cannot be constructed and then used.
    """
    known = set(FEATURE_COLUMNS)
    seen: set[str] = set()
    for spec in MODEL_REGISTRY:
        if spec.name in seen:
            raise ValueError(f"duplicate model name in registry: {spec.name}")
        seen.add(spec.name)
        if not spec.feature_columns:
            raise ValueError(f"{spec.name}: feature_columns is empty")
        leaked = sorted(set(spec.feature_columns) & FORBIDDEN_COLUMNS)
        if leaked:
            raise ValueError(
                f"{spec.name}: feature_columns contains non-feature column(s) "
                f"{', '.join(leaked)}. Identifiers, labels and provenance are never "
                "model inputs."
            )
        unknown = sorted(set(spec.feature_columns) - known)
        if unknown:
            raise ValueError(
                f"{spec.name}: feature_columns names column(s) absent from Component "
                f"4's contract: {', '.join(unknown)}"
            )
        duplicated = len(spec.feature_columns) != len(set(spec.feature_columns))
        if duplicated:
            raise ValueError(f"{spec.name}: feature_columns contains a duplicate")

    if len(indicator_columns()) != len(null_families()):
        raise ValueError("indicator names collided; one family would lose its indicator")
    overlap = set(indicator_columns()) & set(FEATURE_COLUMNS)
    if overlap:
        raise ValueError(
            f"indicator name(s) collide with a Component 4 feature: {', '.join(sorted(overlap))}"
        )


_guard_registry()


__all__ = [
    "ABLATED_FEATURE",
    "CDPH_2015_UNREACHABLE_INPUTS",
    "CDPH_APPROXIMATION_NOTE",
    "FORBIDDEN_COLUMNS",
    "INDICATOR_PREFIX",
    "LOGISTIC_PARAMS",
    "MODELS_BY_NAME",
    "MODEL_DEFINITION_VERSION",
    "MODEL_REGISTRY",
    "MissingStrategy",
    "ModelSpec",
    "boolean_columns",
    "columns_in_family",
    "family_indicator_name",
    "indicator_columns",
    "indicator_source_column",
    "max_iter_of",
    "missing_strategy_for",
    "never_null_columns",
    "null_families",
    "nullable_columns",
    "spec_for",
]
