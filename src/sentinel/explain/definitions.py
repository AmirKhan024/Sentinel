"""Frozen specifications and every pre-declared constant for Component 11.

Nothing else in this package may hold a magic number. Component 9 established the rule and
the reason applies here with more force, not less: an attribution budget that was chosen
after seeing which features came out on top is not a budget, it is a result being tuned.
Every constant below is justified by a measurement in
``docs/analysis/explainability_findings.md``, produced by ``scripts/profile_explanations.py``
before this module existed, and frozen in ADR 0030.

**The support matrix is the component's most important declaration.** Four models can be
explained faithfully and one cannot, and which is which follows from what each estimator is
and what Component 8 chose to make public -- not from what would make the results look
complete. :data:`EXPLAIN_REGISTRY` states the method, the output space, the exactness and,
for the unsupported model, the reason, so a reader never has to infer any of it.

**On the name-recovery functions.** Components 6 and 7 order the same thirty columns
differently -- the profiling script measures the two orders disagreeing at 19 of 30
positions -- and picking the wrong one produces a table whose every value is arithmetically
correct and attached to the wrong feature. Nothing raises, and no additivity check can
catch it, because a sum is invariant to a permutation of its terms. So the choice is
recorded per model here rather than inferred at the call site.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from sentinel.calibration.definitions import CANDIDATE_REGISTRY, Family
from sentinel.features.definitions import FEATURE_COLUMNS
from sentinel.modeling.definitions import (
    FORBIDDEN_COLUMNS,
    columns_in_family,
    family_indicator_name,
    indicator_columns,
    null_families,
)

#: Bumped when anything in this module changes the meaning of an emitted column.
EXPLAIN_DEFINITION_VERSION = "v1"


class OutputSpace(StrEnum):
    """The space an attribution is expressed in.

    One member, deliberately. All four supported models expose a natural log-odds output --
    ``decision_function`` for the logistic model, ``output_margin``/``raw_score`` for the
    boosters, the pre-sigmoid logit for the network -- so every value in the artifact is
    comparable with every other, and a cross-model importance table is a comparison of
    models rather than of units.

    **Probability space was considered and rejected**, and is not declared here. A Shapley
    decomposition of ``sigmoid(margin)`` is not additive in the margin's own contributions,
    because ``sigmoid`` is not linear. A probability-space table would therefore have to
    either abandon additivity or fabricate it, and declaring a member nothing can reach
    would repeat the defect ADR 0014 records: a rejection that has never been observed is
    indistinguishable from one that cannot happen.
    """

    LOG_ODDS = "log_odds"


class ExplanationMethod(StrEnum):
    """How a model's attributions are computed. Dispatched on in ``attribute.py``."""

    #: Exact TreeSHAP, from the booster's own implementation.
    TREE_SHAP = "tree_shap"
    #: The closed form for a linear model under an interventional reference.
    LINEAR_SHAP = "linear_shap"
    #: Antithetic permutation sampling. Additive by construction, approximate in how the
    #: credit is split among columns.
    PERMUTATION_SHAP = "permutation_shap"


class ExplanationStatus(StrEnum):
    """Whether a model could be explained at all."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class FeatureKind(StrEnum):
    """What a matrix column is.

    The distinction matters for tracing a value back to Component 4: a ``FEATURE`` column
    *is* a Component 4 feature, while a ``FAMILY_INDICATOR`` summarises the null mask that
    several of them share.
    """

    FEATURE = "feature"
    FAMILY_INDICATOR = "family_indicator"


@dataclass(frozen=True, slots=True)
class ExplanationSpec:
    """One model Component 11 attempts to explain, and how.

    ``name_source`` is prose naming the function that recovers this model's column names.
    It is not a callable: the callables live in three different closed packages with three
    different spec types, so the dispatch is in ``attribute.py`` where the spec objects are
    already in hand, and this field is what makes the choice auditable from the registry
    alone.
    """

    name: str
    family: Family
    component: int
    source_slug: str
    status: ExplanationStatus
    method: ExplanationMethod | None
    output_space: OutputSpace | None
    is_exact: bool
    name_source: str
    rationale: str
    is_experimental: bool = False
    unsupported_reason: str = ""
    version: str = "v1"


#: Why ``xgboost_chain_embeddings`` carries no attributions. Measured by
#: ``scripts/profile_explanations.py``'s ``embedding_booster_boundary`` profile and frozen
#: in ADR 0031. Stated once, here, and copied into the artifact and the manifest so a
#: consumer never has to read this file to learn why a row is missing.
EMBEDDING_BOOSTER_UNSUPPORTED_REASON = (
    "the fitted booster is reachable only through neural.embed._scorer_for, a private "
    "process-local stash; FittedEmbeddingBooster has 19 fields and none of them is the "
    "estimator, and no public neural.embed function returns it. Component 8 is closed, so "
    "Component 11 does not reach into a private helper to explain this model. The minimal "
    "public extension that would lift the restriction -- embed.booster_for(fitted) -- is "
    "proposed in ADR 0031 and deliberately not taken here."
)


#: The support matrix. One entry per Component 9 candidate, in ``CANDIDATE_REGISTRY``
#: order, so the two components explain and calibrate the same five models and a reader can
#: line the tables up without a join.
#:
#: Reusing Component 9's candidate set rather than inventing a sixth model list is itself a
#: decision: twelve models have prediction artifacts on disk, and the five here are the ones
#: that disagree about which is best. Explaining an ablation nobody would deploy would be an
#: artifact dump rather than an answer.
EXPLAIN_REGISTRY: tuple[ExplanationSpec, ...] = (
    ExplanationSpec(
        name="logistic_regression",
        family=Family.LOGISTIC,
        component=6,
        source_slug="baseline_predictions",
        status=ExplanationStatus.SUPPORTED,
        method=ExplanationMethod.LINEAR_SHAP,
        output_space=OutputSpace.LOG_ODDS,
        is_exact=True,
        name_source="modeling.preprocess.ordered_matrix_columns",
        rationale=(
            "a linear model's Shapley values under an interventional reference have a "
            "closed form, coef_j * (z_j - E[z_j]), so this is the one model whose "
            "attribution is arithmetic rather than an algorithm"
        ),
    ),
    ExplanationSpec(
        name="xgboost",
        family=Family.BOOSTED,
        component=7,
        source_slug="boosted_predictions",
        status=ExplanationStatus.SUPPORTED,
        method=ExplanationMethod.TREE_SHAP,
        output_space=OutputSpace.LOG_ODDS,
        is_exact=True,
        name_source="boosting.preprocess.matrix_columns",
        rationale=(
            "exact TreeSHAP from the booster's own pred_contribs, which needs no "
            "background dataset: the conditional expectation is taken over the tree's "
            "recorded cover"
        ),
    ),
    ExplanationSpec(
        name="lightgbm",
        family=Family.BOOSTED,
        component=7,
        source_slug="boosted_predictions",
        status=ExplanationStatus.SUPPORTED,
        method=ExplanationMethod.TREE_SHAP,
        output_space=OutputSpace.LOG_ODDS,
        is_exact=True,
        name_source="boosting.preprocess.matrix_columns",
        rationale="exact TreeSHAP from the booster's own pred_contrib, computed in float64",
    ),
    ExplanationSpec(
        name="neural_numeric_only",
        family=Family.NEURAL_MLP,
        component=8,
        source_slug="neural_predictions",
        status=ExplanationStatus.SUPPORTED,
        method=ExplanationMethod.PERMUTATION_SHAP,
        output_space=OutputSpace.LOG_ODDS,
        is_exact=False,
        name_source="neural.preprocess.transformed_columns",
        rationale=(
            "no exact attribution exists for a multi-layer network, so an antithetic "
            "permutation game is played over the same thirty columns. The live module is "
            "reached through neural.train.scorer_for, which is public"
        ),
    ),
    ExplanationSpec(
        name="xgboost_chain_embeddings",
        family=Family.NEURAL_EMBEDDING_BOOSTER,
        component=8,
        source_slug="neural_predictions",
        status=ExplanationStatus.UNSUPPORTED,
        method=None,
        output_space=None,
        is_exact=False,
        name_source="neural.embed.augmented_columns",
        rationale=(
            "a plain XGBoost model over 30 base columns plus 16 chain embedding "
            "dimensions, and therefore TreeSHAP-able in principle -- but not reachable"
        ),
        is_experimental=True,
        unsupported_reason=EMBEDDING_BOOSTER_UNSUPPORTED_REASON,
    ),
)

SPECS_BY_NAME: dict[str, ExplanationSpec] = {s.name: s for s in EXPLAIN_REGISTRY}

#: The models an ordinary run explains.
SUPPORTED_MODELS: tuple[str, ...] = tuple(
    s.name for s in EXPLAIN_REGISTRY if s.status is ExplanationStatus.SUPPORTED
)


# --- the sampling budget (ADR 0030) ------------------------------------------

#: Rows explained per (model, fold). Frozen from the ``permutation_cost`` profile: the
#: population is 41,536 rows over 18 folds, the tree and linear methods could explain all of
#: them for nothing, and the network could not.
#:
#: The same sampled ids are used for **every** model. Explaining all rows for the cheap
#: models and a sample for the network would compare importance across two different
#: populations, and the difference between the two tables would then be partly a sampling
#: artifact and partly a real difference, with no way to tell which.
SAMPLE_SIZE = 300

#: The seed the explanation sample is drawn under. A frozen constant rather than a CLI flag,
#: for the reason ``TUNING_SEED`` is: a seed a caller can change is a seed that cannot be
#: cited. Not derived from ``hash()`` of anything -- Python salts ``str`` hashing per
#: process, and Component 9 lost a day of reproducibility to exactly that.
SAMPLING_SEED = 20260825

#: Prose recorded in every row and in the manifest, so the artifact states its own sampling
#: rule without a reader having to find this module.
SAMPLE_STRATEGY = (
    "uniform without replacement from the fold's test window, sorted canonically by "
    "(rd, target_inspection_id) before drawing, under a frozen seed; identical ids for "
    "every model; no label, outcome or score participates in the selection"
)

#: The population a sample is drawn from. Named in the artifact so "300 rows" is never read
#: as "300 rows of something unspecified".
SAMPLING_POPULATION = "fold test window"

#: Reference rows for the permutation explainer, drawn from the fold's **training** window.
#: Not the test window: a background is part of the explanation, and a background containing
#: rows the model was never allowed to see at fit time would answer a counterfactual the
#: model was never asked.
BACKGROUND_SIZE = 64
BACKGROUND_SEED = 20260826
BACKGROUND_STRATEGY = (
    "uniform without replacement from the fold's training window as defined by "
    "modeling.train.training_frame, under a frozen seed; every row is dated on or before "
    "fold.train_end by construction, and validate re-derives that from the frame rather "
    "than trusting it"
)

#: Antithetic permutation pairs per explained row. Each round walks one permutation
#: forwards and backwards, so a round costs ``2 * (M + 1) * BACKGROUND_SIZE`` forward rows.
#:
#: Frozen at 8 from the ``permutation_convergence`` sweep, which measured, against a
#: 64-round reference drawn at an independent seed:
#:
#: ===== ================= ================== =================
#: round median local err  max global err     global rank rho
#: ===== ================= ================== =================
#: 1     2.76%             16.61%             0.9706
#: 2     1.90%              8.14%             0.9902
#: 4     1.29%              5.63%             0.9942
#: 8     1.00%              2.85%             0.9964
#: 16    0.71%              4.41%             0.9973
#: 32    0.53%              1.67%             0.9991
#: ===== ================= ================== =================
#:
#: 8 is where the global importance *ranking* -- the statistic the findings document
#: actually reports -- has stabilised, at a cost of roughly fourteen minutes over the
#: 18-fold population. Going to 32 buys 0.0027 of rank correlation for four times the
#: compute, and buys the per-row values a halving of an error that is still too large to
#: quote to three decimal places either way.
#:
#: **The per-row values remain approximate at any round count in this range**, and that is
#: recorded as ``is_exact = false`` on every neural row rather than softened.
PERMUTATION_ROUNDS = 8


# --- additivity tolerances (ADR 0030) ----------------------------------------

#: ``|base + sum(phi) - model output|`` must not exceed these. Per method, because the three
#: differ by orders of magnitude and a single tolerance would be either vacuous for two of
#: them or unmeetable for the third.
#:
#: Every one of these is an **arithmetic** tolerance, not a modelling one. TreeSHAP and
#: linear SHAP are exact and the residual is float summation order; the permutation game's
#: path telescopes, so its additivity is exact at one round too. What is approximate about
#: the permutation method is how credit is divided among columns, which additivity cannot
#: see and which is recorded instead as ``is_exact = false``.
#:
#: XGBoost computes in float32 and LightGBM in float64, so the tree tolerance is set from
#: the worse of the two rather than from an average.
#:
#: Each is the measured maximum on the probe fold, rounded up by three to five orders of
#: magnitude to absorb fold-to-fold variation in tree count and window size -- and no
#: further, because a tolerance generous enough never to trip is a check that has stopped
#: checking:
#:
#: ==================  ==================  =========
#: method              measured max        frozen at
#: ==================  ==================  =========
#: tree_shap           8.920e-07 (xgb)     1e-5
#: linear_shap         1.332e-15           1e-10
#: permutation_shap    6.112e-10           1e-6
#: ==================  ==================  =========
ADDITIVITY_TOLERANCE: Mapping[ExplanationMethod, float] = {
    ExplanationMethod.TREE_SHAP: 1e-5,
    ExplanationMethod.LINEAR_SHAP: 1e-10,
    ExplanationMethod.PERMUTATION_SHAP: 1e-6,
}


# --- analysis parameters (ADR 0030) ------------------------------------------

#: How many features a local explanation shows on each side, and the width of the top-k
#: overlap statistic. Ten of thirty, so the statistic is neither the whole ranking (where
#: overlap is trivially 1.0) nor its noisy tail.
TOP_K = 10

#: Rank positions a feature's importance must move across the quarterly folds before the
#: findings document is allowed to call the change material. Declared before the ranks were
#: computed. With 30 features, a threshold of 5 is a sixth of the ranking.
RANK_DRIFT_THRESHOLD = 5

#: Predicted-score quantiles that define the representative local cases. **Quantiles of the
#: prediction, never of the outcome.** Choosing a case because the model was right about it
#: would be outcome-driven storytelling with a deterministic-looking rule on top.
REPRESENTATIVE_QUANTILES: Mapping[str, float] = {"high": 0.90, "medium": 0.50, "low": 0.10}

#: The fold representative figures are drawn on. The project's standing choice since
#: Component 8: the last quarterly fold, plus ``covid_shift`` reported separately and never
#: averaged in.
REPRESENTATIVE_FOLD_SET = "quarterly"

#: Minimum Spearman correlation between the sampled and full-population importance rankings
#: before the advisory check complains. Advisory, not error: a low value means the sample is
#: too small to speak for the population, which is a fact about the budget rather than a
#: defect in the attribution.
SAMPLE_FIDELITY_MIN_RHO = 0.95


# --- experiments this component does not run ---------------------------------

#: Recorded in every manifest, so the scope reads as a decision rather than as whatever
#: happened to fit in the time available.
BLOCKED_EXPERIMENTS: Mapping[str, str] = {
    "model selection": (
        "Component 11 may not name a winner. Explainability is diagnostic evidence about "
        "how a model reasons; it does not measure whether the reasoning is right, and "
        "Components 5 to 9 own that question. A model chosen because its attributions "
        "looked tidier would have been chosen on legibility."
    ),
    "recalibration": (
        "the calibrators are frozen per (model, fold). This component reads base_score and "
        "the calibrated probability side by side and fits nothing."
    ),
    "causal interpretation": (
        "a SHAP value states how the model used a feature. It does not state that changing "
        "the feature would change the outcome, and no artifact, figure or sentence this "
        "component emits may be read that way."
    ),
    "interaction values": (
        "TreeSHAP can emit an [n, M, M] interaction tensor and both boosters support it. "
        "Not computed: it is 30x the storage for a question nobody has asked yet, and the "
        "two models that could produce one are not the two the project is likeliest to "
        "deploy. A later component can add it without changing anything here."
    ),
    "attribution over the learned embedding": (
        "Component 8's own BLOCKED list says the embedding tables are a representation "
        "rather than an explanation. Nothing here contradicts that, and the one model that "
        "consumes them is unsupported for an unrelated reason."
    ),
    "explaining the calibrated probability directly": (
        "Platt is a monotone two-parameter map applied after the fact. Attributions are "
        "computed on the base model's log-odds and the calibrated probability is carried "
        "alongside; pretending SHAP decomposed the calibrated number would misdescribe "
        "which model was explained. ADR 0030."
    ),
}


# --- derived feature partitions ----------------------------------------------


def _feature_family_map() -> dict[str, tuple[str, str]]:
    """Every matrix column mapped to ``(original_feature_name, derived_from)``.

    Derived from Component 4's null rules rather than restated, so a change there surfaces
    here as a guard failure instead of as a quietly wrong label.
    """
    mapping = {name: (name, name) for name in FEATURE_COLUMNS}
    for rule in null_families():
        indicator = family_indicator_name(rule)
        members = columns_in_family(rule)
        mapping[indicator] = (rule.name.lower(), ",".join(sorted(members)))
    return mapping


#: ``matrix column -> (original_feature_name, derived_from)``.
#:
#: A plain feature maps to itself: the transformation applied to it -- median or constant
#: imputation, then standardisation -- is monotone and one-to-one, so the attribution
#: belongs to the same feature it started as.
#:
#: A family indicator does not map to a single column, and pretending otherwise would be
#: the false aggregation the brief warns about. It maps to its null-rule family, and
#: ``derived_from`` lists every Component 4 column that family covers, so a value can always
#: be traced back to the columns it summarises.
FEATURE_ORIGIN: Mapping[str, tuple[str, str]] = _feature_family_map()

#: Every column name Component 11 may ever emit. A name absent from this set is rejected by
#: ``validate``, which is what makes ``feature_127``-style output unrepresentable rather
#: than merely discouraged.
KNOWN_FEATURE_NAMES: frozenset[str] = frozenset(FEATURE_ORIGIN)


def origin_of(column: str) -> tuple[str, str]:
    """``(original_feature_name, derived_from)`` for one matrix column."""
    try:
        return FEATURE_ORIGIN[column]
    except KeyError:
        raise KeyError(
            f"{column!r} is not a known feature representation. Component 11 refuses to "
            "emit an attribution it cannot name; an anonymous column means the matrix and "
            "its name list have drifted apart."
        ) from None


def kind_of(column: str) -> FeatureKind:
    """Whether a matrix column is a Component 4 feature or a null-rule family indicator."""
    if column in FEATURE_COLUMNS:
        return FeatureKind.FEATURE
    if column in indicator_columns():
        return FeatureKind.FAMILY_INDICATOR
    raise KeyError(f"{column!r} is neither a Component 4 feature nor a family indicator")


def spec_for(name: str) -> ExplanationSpec:
    """One model's explanation spec, by name."""
    try:
        return SPECS_BY_NAME[name]
    except KeyError:
        known = ", ".join(sorted(SPECS_BY_NAME))
        raise KeyError(f"Unknown model for Component 11: {name!r}. Known: {known}") from None


def tolerance_for(method: ExplanationMethod) -> float:
    """The additivity tolerance frozen for one method."""
    return ADDITIVITY_TOLERANCE[method]


def _guard_registry() -> None:
    """Reject a registry that could mislabel, leak or claim more than it delivers.

    Raises rather than asserts, at import time, so a bad spec cannot be constructed and then
    used -- ``python -O`` strips asserts and would strip this. Mirrors the three guards
    Components 6, 7 and 8 each run, and adds the checks only an explanation component needs.
    """
    seen: set[str] = set()
    calibration_names = {c.name for c in CANDIDATE_REGISTRY}

    for spec in EXPLAIN_REGISTRY:
        if spec.name in seen:
            raise ValueError(f"duplicate model name in explain registry: {spec.name}")
        seen.add(spec.name)

        supported = spec.status is ExplanationStatus.SUPPORTED
        if supported and spec.method is None:
            raise ValueError(f"{spec.name}: supported but declares no explanation method")
        if supported and spec.output_space is None:
            raise ValueError(
                f"{spec.name}: supported but declares no output space. An attribution "
                "whose space is unstated is a number without units."
            )
        if not supported:
            if spec.method is not None or spec.output_space is not None:
                raise ValueError(
                    f"{spec.name}: unsupported but declares a method or an output space. "
                    "An unsupported model must carry nothing that looks like an "
                    "explanation, including the promise of one."
                )
            if not spec.unsupported_reason:
                raise ValueError(
                    f"{spec.name}: unsupported with no stated reason. Honest unsupported "
                    "behaviour is only honest if it says why."
                )
        if supported and spec.unsupported_reason:
            raise ValueError(f"{spec.name}: supported but carries an unsupported_reason")
        if spec.is_exact and spec.method is ExplanationMethod.PERMUTATION_SHAP:
            raise ValueError(
                f"{spec.name}: permutation sampling is never exact. Labelling an "
                "approximation exact is the one claim this component must not make."
            )

    unknown = sorted(seen - calibration_names)
    if unknown:
        raise ValueError(
            f"explain registry names model(s) absent from Component 9's candidate set: "
            f"{', '.join(unknown)}. The two components explain and calibrate the same "
            "models so their tables line up without a join."
        )
    missing = sorted(calibration_names - seen)
    if missing:
        raise ValueError(
            f"Component 9 candidate(s) {', '.join(missing)} have no explain spec. Every "
            "candidate must be accounted for, as supported or as unsupported with a reason."
        )

    if not SUPPORTED_MODELS:
        raise ValueError("no model is supported; Component 11 would have nothing to compute")

    leaked = sorted(KNOWN_FEATURE_NAMES & FORBIDDEN_COLUMNS)
    if leaked:
        raise ValueError(
            f"feature origin map names non-feature column(s) {', '.join(leaked)}. "
            "Identifiers, labels and provenance are never attributed."
        )

    for column, (original, derived) in FEATURE_ORIGIN.items():
        if not original or not derived:
            raise ValueError(f"{column}: incomplete origin mapping")
        if column in FEATURE_COLUMNS and original != column:
            raise ValueError(
                f"{column}: a Component 4 feature must map to itself. Mapping it elsewhere "
                "would be an undeclared aggregation."
            )

    for method in ExplanationMethod:
        if method not in ADDITIVITY_TOLERANCE:
            raise ValueError(f"no additivity tolerance declared for {method.value}")
        if ADDITIVITY_TOLERANCE[method] <= 0:
            raise ValueError(f"{method.value}: additivity tolerance must be positive")

    for tier, quantile in REPRESENTATIVE_QUANTILES.items():
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"representative quantile {tier} is out of range: {quantile}")

    for name, value in (
        ("SAMPLE_SIZE", SAMPLE_SIZE),
        ("BACKGROUND_SIZE", BACKGROUND_SIZE),
        ("PERMUTATION_ROUNDS", PERMUTATION_ROUNDS),
        ("TOP_K", TOP_K),
        ("RANK_DRIFT_THRESHOLD", RANK_DRIFT_THRESHOLD),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if len(KNOWN_FEATURE_NAMES) < TOP_K:
        raise ValueError("TOP_K exceeds the number of features, so overlap would be trivially 1")


_guard_registry()


__all__ = [
    "ADDITIVITY_TOLERANCE",
    "BACKGROUND_SEED",
    "BACKGROUND_SIZE",
    "BACKGROUND_STRATEGY",
    "BLOCKED_EXPERIMENTS",
    "EMBEDDING_BOOSTER_UNSUPPORTED_REASON",
    "EXPLAIN_DEFINITION_VERSION",
    "EXPLAIN_REGISTRY",
    "FEATURE_ORIGIN",
    "KNOWN_FEATURE_NAMES",
    "PERMUTATION_ROUNDS",
    "RANK_DRIFT_THRESHOLD",
    "REPRESENTATIVE_FOLD_SET",
    "REPRESENTATIVE_QUANTILES",
    "SAMPLE_FIDELITY_MIN_RHO",
    "SAMPLE_STRATEGY",
    "SAMPLE_SIZE",
    "SAMPLING_POPULATION",
    "SAMPLING_SEED",
    "SPECS_BY_NAME",
    "SUPPORTED_MODELS",
    "TOP_K",
    "ExplanationMethod",
    "ExplanationSpec",
    "ExplanationStatus",
    "FeatureKind",
    "OutputSpace",
    "kind_of",
    "origin_of",
    "spec_for",
    "tolerance_for",
]
