"""Neural model specifications, the architecture, and the frozen training constants.

Mirrors ``modeling.definitions`` and ``boosting.definitions`` deliberately: a declared
registry, feature partitions inherited from Component 4 rather than hand-listed, and an
import-time guard that raises (never asserts, because ``assert`` is stripped under
``python -O``) on any spec that could leak or that names a column Component 4 lacks.

Three things here have no Component 6 or 7 counterpart.

**A second, separately guarded column list.** ``NeuralSpec`` carries ``feature_columns``
-- the same 26 numeric features every other model sees, guarded against the same
``FORBIDDEN_COLUMNS`` -- and ``entity_columns``, which name categorical families that do
**not** live in Component 4's table. Keeping them in a different field is the point: a
categorical can never be reached by code that iterates ``feature_columns``, and the guard
for each is written separately because the two have different safety arguments. See ADR
0021.

**``establishment_id`` is refused.** It is the obvious thing to embed and it is exactly
what this component must not do. It is a forbidden column, Component 4 excludes identity
by design, and a per-establishment parameter fitted across rows is the largest leakage
surface in the project: an establishment's embedding carries whatever the network learned
about it from every row, including rows a later fold has not reached yet. ``chain`` is the
deliberate lower-cardinality substitute -- it is a property of a *group* of
establishments, derived per fold from training rows only. See ADR 0021.

**The architecture is a constant, not a search space.** The project specification names
the layer widths, the dropout rate and the embedding dimensions, so they are recorded here
as literals rather than tuned. The one thing Component 8 searches is the learning rate,
because the specification asks whether the result is sensitive to it -- and that search
runs under Component 7's protocol, not a second one. See ADR 0017.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from sentinel.features.definitions import FEATURE_COLUMNS
from sentinel.modeling.definitions import FORBIDDEN_COLUMNS

#: Bumped whenever a network's feature set, architecture, preprocessing or training
#: protocol changes in a way that makes two runs incomparable.
NEURAL_DEFINITION_VERSION = "v1"

#: One seed for weight initialisation and batch order, one for the learning-rate search.
#: Both are load-bearing, unlike Component 6 where ``random_state`` was inert: a network
#: initialises randomly and consumes its training rows in a shuffled order.
DEFAULT_SEED = 42
TUNING_SEED = 20260818

#: Seeds for the reproducibility experiment. The final configuration is refit under each
#: and the spread is reported, because a single seed's number would be a claim this
#: project cannot support -- Component 7 already established that reproducibility is a
#: real issue here, and a network has strictly more stochasticity than a booster.
SEED_SWEEP: tuple[int, ...] = (42, 43, 44, 45, 46)


class Learner(StrEnum):
    """What actually fits. Closed set."""

    #: The specified embedding network.
    MLP = "mlp"
    #: XGBoost on the tree matrix augmented with the network's learned chain vectors.
    #: Not a neural model; it is the specification's "feed the embeddings into XGBoost"
    #: experiment, and it lives in this registry because the embeddings are Component 8's.
    XGBOOST_EMBEDDING = "xgboost_embedding"


class CategoricalEncoding(StrEnum):
    """How a spec represents its categorical families."""

    #: Learned dense vectors, the component's subject.
    EMBEDDING = "embedding"
    #: Indicator columns, one per (family, category) seen in training. The control.
    ONE_HOT = "one_hot"
    #: No categoricals at all -- the 26 numeric features and nothing else.
    NONE = "none"


class EntityFamily(StrEnum):
    """The categorical families Component 8 may embed.

    A closed set, and the closure is the safety property: ``_guard_registry`` refuses any
    ``entity_columns`` entry that is not a member, so a categorical cannot be introduced
    by editing a spec alone. Adding one means editing this enum, the embedding-dimension
    table and the ADR that justifies it.
    """

    CHAIN = "chain"
    FACILITY_TYPE = "facility_type"
    COMMUNITY_AREA = "community_area"
    ZIP = "zip"


#: Embedding width per family, exactly as the project specification names them. Not
#: tuned: the specification fixes them, and searching four widths on a dataset where
#: three model classes already agree within 0.005 NDE would be tuning noise.
#:
#: The widths are not arbitrary relative to cardinality. Measured on the 57,727 feature
#: rows: chain has the largest vocabulary and gets 16; facility type (182), community
#: area (78) and zip (97) are an order of magnitude smaller and get 8.
EMBEDDING_DIMS: Mapping[EntityFamily, int] = {
    EntityFamily.CHAIN: 16,
    EntityFamily.FACILITY_TYPE: 8,
    EntityFamily.COMMUNITY_AREA: 8,
    EntityFamily.ZIP: 8,
}

#: Every entity family, in declared order. The order is the concatenation order of the
#: embedding block and therefore part of the matrix contract; it is derived from the enum
#: rather than written twice.
ENTITY_COLUMNS: tuple[str, ...] = tuple(family.value for family in EntityFamily)

#: Reserved vocabulary index 0. Any category not seen in a fold's *training* rows maps
#: here at validation and test time.
#:
#: The row is **learned, not frozen at zero**. Genuine unknowns exist in training --
#: ``categoricals`` emits UNKNOWN for a row with no prior inspection to carry a value
#: forward from -- so index 0 receives gradient from rows that really are unknown, and
#: the vector it learns is the "no history" group's offset rather than an arbitrary
#: constant. See ADR 0021 and the ``encode`` docstring.
UNKNOWN_CATEGORY = "__UNKNOWN__"

#: The chain value for an establishment whose normalised name is not shared with any
#: other establishment *in the fold's training window*. A real category, not a null: "is
#: not part of a chain" is a fact about an establishment and the model should be able to
#: condition on it.
INDEPENDENT_CHAIN = "__INDEPENDENT__"


# --- architecture ------------------------------------------------------------

#: Hidden widths, in order. Named by the project specification.
HIDDEN_SIZES: tuple[int, ...] = (256, 128)

#: Dropout probability after each hidden block. Named by the specification.
DROPOUT = 0.3

#: The output is a single logit. Sigmoid is applied only where a probability is required
#: for reporting or evaluation, never inside the loss -- ``BCEWithLogitsLoss`` fuses the
#: sigmoid into the loss for numerical stability, and applying it twice would be a silent
#: defect that merely trains badly rather than raising.
OUTPUT_SIZE = 1


# --- training constants ------------------------------------------------------

#: The specification's optimiser and its baseline learning rate. The rate is the one
#: thing Component 8 searches; the rest are fixed.
OPTIMIZER = "AdamW"
BASELINE_LEARNING_RATE = 1e-3

#: AdamW's decoupled weight decay. Left at the library default and recorded rather than
#: tuned, so the manifest states what was used instead of implying a search happened.
WEIGHT_DECAY = 0.01

BATCH_SIZE = 512
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 15
GRADIENT_CLIP_NORM = 1.0

#: ``ReduceLROnPlateau`` settings. Monitors the inner-validation loss, which is computed
#: on rows strictly inside the training window -- see ``INNER_VALIDATION_FRACTION``.
SCHEDULER = "ReduceLROnPlateau"
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-6

#: The loss. Takes logits, not probabilities.
LOSS = "BCEWithLogitsLoss"

#: ``pos_weight`` is **not** applied by the primary models.
#:
#: Measured prevalence is 52.52% overall and 0.379-0.492 per test window, so there is no
#: imbalance to correct. Weighting a balanced problem shifts every predicted probability
#: away from the base rate for no ranking benefit, and Component 9 has to calibrate what
#: this component emits. It exists as a separately named ablation -- exactly as Component
#: 7 carries ``xgboost_class_weighted`` -- because "we tested weighting and it did not
#: help" is only sayable if it was actually tested.
POS_WEIGHT_DEFAULT: float | None = None

#: The fraction of each fold's training window, taken from its **end** by date, held out
#: to early-stop on.
#:
#: This is the decision that keeps ``trained_through = fold.train_end`` literally true.
#: Early stopping needs a validation window later than the rows being fitted. The obvious
#: candidates -- the fold's calibration window and its test window -- are both later than
#: ``train_end``, and reading either would mean the fit had learned from a date later than
#: the horizon it declares. So the split is carved *inside* the training window: the last
#: 15% of training days validate, the first 85% fit. Nothing later than ``train_end`` is
#: read by any final fit, and the fold's calibration window is untouched, exactly as in
#: Components 6 and 7. See ADR 0017 and ADR 0021.
INNER_VALIDATION_FRACTION = 0.15

#: Below this many rows on either side of the inner split, a fold is refused rather than
#: fitted. An early-stopping signal from a handful of rows is noise, and silently
#: training without one would make two folds incomparable.
MIN_INNER_SPLIT_ROWS = 200

#: The learning rates the sweep evaluates. A decade either side of the specified
#: baseline, which is enough to show whether the result is sensitive without becoming an
#: open-ended search. The specification asks for a demonstration, not an optimisation.
LEARNING_RATE_GRID: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


#: Where ``TUNED_HYPERPARAMS`` came from. A reader can tell a frozen search result from a
#: placeholder without reading git history.
TUNED_HYPERPARAMS_PROVENANCE = (
    "sentinel tune-neural, 40 trials over 2 studies (5 rates x 6 quarterly inner folds, "
    "5 rates x 2 covid_shift inner folds), 550.9s. Study artifact "
    "neural_sweep_trials_20260818T125643Z.parquet "
    "sha256 ebd51e08f9401661d1c35aaf83bd2280e19aa9dc37eb938bd9291891511e01ce, "
    "from as_of_features_20260816T150313Z.parquet (57,727 rows) and "
    "neural_categoricals_20260818T125631Z.parquet. Seed 20260818; torch 2.13.0+cpu, CPU, "
    "one thread."
)

#: Best learning rate per fold set, selected on inner validation windows strictly earlier
#: than that fold set's first test window. Two fold sets means two searches, for the same
#: reason Component 7 runs two: the quarterly tuning region contains the covid_shift
#: *test* window, so one shared search would make the shift result optimistically biased.
#: See ADR 0017.
TUNED_HYPERPARAMS: Mapping[str, Mapping[str, float]] = {
    # Mean inner-validation PR-AUC 0.607026 over 6 inner folds ending 2022-03-31, the day
    # before the first quarterly test window opens. The grid scored
    # 1e-4:0.5918, 3e-4:0.5943, 1e-3:0.6050, 3e-3:0.6070, 1e-2:0.5964 -- a range of
    # 0.0152, so the result is mildly sensitive to the rate and the specification's own
    # 1e-3 baseline sits 0.0020 below the winner. That is a small enough margin that the
    # honest reading is "anywhere in 1e-3..3e-3 is equivalent", and it is recorded rather
    # than presented as a tuning gain.
    "quarterly": {"learning_rate": 0.003},
    # Mean inner-validation PR-AUC 0.792758 over 2 inner folds ending 2020-05-31. A
    # separate study, not a copy of the quarterly one: the quarterly region contains this
    # fold's entire test window, so sharing a rate would make the shift result
    # optimistically biased. Note the two regimes disagree about the rate, which is itself
    # a reason not to share. Two inner folds is the declared minimum -- these numbers are
    # a weaker measurement than the quarterly ones and are treated as such.
    "covid_shift": {"learning_rate": 0.01},
}


@dataclass(frozen=True, slots=True)
class NeuralSpec:
    """One neural experiment, fully specified.

    ``feature_columns`` is the numeric half and is explicit and closed, for the same
    reason Components 6 and 7 keep theirs so: "every column except the target" would
    silently absorb a future Component 4 metadata column into the model.

    ``entity_columns`` is the categorical half. It is a separate field rather than more
    entries in ``feature_columns`` because the two are guarded differently and because
    the categoricals are **not Component 4 features** -- they come from Component 8's own
    experimental join and must stay visibly separable from the production feature set.

    ``encoding`` decides what happens to ``entity_columns``: learned vectors, indicator
    columns, or nothing. A spec with ``CategoricalEncoding.NONE`` must have empty
    ``entity_columns``, and the guard enforces it -- the alternative is a spec that
    declares categoricals and silently ignores them.
    """

    name: str
    version: str
    description: str
    learner: Learner
    encoding: CategoricalEncoding
    feature_columns: tuple[str, ...]
    entity_columns: tuple[str, ...]
    is_probability: bool
    seed: int
    pos_weighted: bool = False
    experiment: str = ""


def _entities(*families: EntityFamily) -> tuple[str, ...]:
    """Entity column names in declared order, whatever order the caller names them.

    The concatenation order of the embedding block is a property of
    :class:`EntityFamily`, not of how a spec happens to be written, so an ablation that
    drops one family cannot also permute the rest.
    """
    chosen = {family.value for family in families}
    return tuple(name for name in ENTITY_COLUMNS if name in chosen)


ALL_ENTITIES: tuple[str, ...] = ENTITY_COLUMNS


NEURAL_REGISTRY: tuple[NeuralSpec, ...] = (
    NeuralSpec(
        name="neural_embeddings",
        version="v1",
        description=(
            "The specified network: learned embeddings for chain (16), facility type "
            "(8), community area (8) and zip (8), concatenated with the 26 standardised "
            "as-of features and the four null-rule family indicators, then "
            "256-BatchNorm-ReLU-Dropout(0.3), 128-BatchNorm-ReLU-Dropout(0.3), and a "
            "single logit. The primary Component 8 model."
        ),
        learner=Learner.MLP,
        encoding=CategoricalEncoding.EMBEDDING,
        feature_columns=FEATURE_COLUMNS,
        entity_columns=ALL_ENTITIES,
        is_probability=True,
        seed=DEFAULT_SEED,
        experiment="A -- neural baseline",
    ),
    NeuralSpec(
        name="neural_numeric_only",
        version="v1",
        description=(
            "The same network on exactly the matrix Components 6 and 7 see: 26 features "
            "plus four indicators, no categoricals at all. This is the model that makes "
            "the C6/C7/C8 comparison unambiguous -- it changes the estimator and "
            "nothing else, so a difference against XGBoost cannot be attributed to the "
            "extra columns the embedding models get."
        ),
        learner=Learner.MLP,
        encoding=CategoricalEncoding.NONE,
        feature_columns=FEATURE_COLUMNS,
        entity_columns=(),
        is_probability=True,
        seed=DEFAULT_SEED,
        experiment="A -- fair-comparison control",
    ),
    NeuralSpec(
        name="neural_onehot",
        version="v1",
        description=(
            "The control for the embedding question: identical architecture, identical "
            "features, identical categorical families -- represented as indicator "
            "columns fitted on training rows instead of learned vectors. The only thing "
            "that differs from neural_embeddings is how a category becomes a number, so "
            "the difference between the two is what a learned representation bought."
        ),
        learner=Learner.MLP,
        encoding=CategoricalEncoding.ONE_HOT,
        feature_columns=FEATURE_COLUMNS,
        entity_columns=ALL_ENTITIES,
        is_probability=True,
        seed=DEFAULT_SEED,
        experiment="B -- without entity embeddings",
    ),
    NeuralSpec(
        name="neural_no_chain",
        version="v1",
        description="neural_embeddings without the chain embedding. Ablation.",
        learner=Learner.MLP,
        encoding=CategoricalEncoding.EMBEDDING,
        feature_columns=FEATURE_COLUMNS,
        entity_columns=_entities(
            EntityFamily.FACILITY_TYPE, EntityFamily.COMMUNITY_AREA, EntityFamily.ZIP
        ),
        is_probability=True,
        seed=DEFAULT_SEED,
        experiment="C -- embedding ablation",
    ),
    NeuralSpec(
        name="neural_no_facility_type",
        version="v1",
        description="neural_embeddings without the facility-type embedding. Ablation.",
        learner=Learner.MLP,
        encoding=CategoricalEncoding.EMBEDDING,
        feature_columns=FEATURE_COLUMNS,
        entity_columns=_entities(EntityFamily.CHAIN, EntityFamily.COMMUNITY_AREA, EntityFamily.ZIP),
        is_probability=True,
        seed=DEFAULT_SEED,
        experiment="C -- embedding ablation",
    ),
    NeuralSpec(
        name="neural_no_community_area",
        version="v1",
        description=(
            "neural_embeddings without the community-area embedding. Both an embedding "
            "ablation and the fairness-relevant comparison: community area is a "
            "candidate demographic proxy, so what the model loses without it is the "
            "quantity Component 12 needs. A better score WITH it is not grounds for "
            "retaining it. See ADR 0023."
        ),
        learner=Learner.MLP,
        encoding=CategoricalEncoding.EMBEDDING,
        feature_columns=FEATURE_COLUMNS,
        entity_columns=_entities(EntityFamily.CHAIN, EntityFamily.FACILITY_TYPE, EntityFamily.ZIP),
        is_probability=True,
        seed=DEFAULT_SEED,
        experiment="C + D -- community-area ablation",
    ),
    NeuralSpec(
        name="neural_no_zip",
        version="v1",
        description="neural_embeddings without the zip embedding. Ablation.",
        learner=Learner.MLP,
        encoding=CategoricalEncoding.EMBEDDING,
        feature_columns=FEATURE_COLUMNS,
        entity_columns=_entities(
            EntityFamily.CHAIN, EntityFamily.FACILITY_TYPE, EntityFamily.COMMUNITY_AREA
        ),
        is_probability=True,
        seed=DEFAULT_SEED,
        experiment="C -- embedding ablation",
    ),
    NeuralSpec(
        name="neural_pos_weighted",
        version="v1",
        description=(
            "The class-weighting ablation: neural_embeddings with BCEWithLogitsLoss "
            "pos_weight set from the training window's own prevalence, and otherwise "
            "identical. Not a default -- prevalence is 52.52%, so there is no imbalance "
            "to correct. It exists so the claim that weighting does not help is a "
            "measurement rather than an assumption."
        ),
        learner=Learner.MLP,
        encoding=CategoricalEncoding.EMBEDDING,
        feature_columns=FEATURE_COLUMNS,
        entity_columns=ALL_ENTITIES,
        is_probability=True,
        seed=DEFAULT_SEED,
        pos_weighted=True,
        experiment="pos_weight ablation",
    ),
    NeuralSpec(
        name="xgboost_chain_embeddings",
        version="v1",
        description=(
            "The specification's embeddings-into-XGBoost experiment. Component 7's "
            "frozen XGBoost parameters on Component 7's tree matrix, widened by the 16 "
            "chain-embedding dimensions this component learned on the SAME fold's "
            "training rows. Not a neural model; it measures whether the learned "
            "representation is useful to the estimator that currently wins."
        ),
        learner=Learner.XGBOOST_EMBEDDING,
        encoding=CategoricalEncoding.EMBEDDING,
        feature_columns=FEATURE_COLUMNS,
        entity_columns=_entities(EntityFamily.CHAIN),
        is_probability=True,
        seed=DEFAULT_SEED,
        experiment="embeddings -> XGBoost",
    ),
)

SPECS_BY_NAME: dict[str, NeuralSpec] = {spec.name: spec for spec in NEURAL_REGISTRY}

#: Which fitted network supplies the embedding table an ``XGBOOST_EMBEDDING`` spec feeds
#: on. Declared here rather than discovered in ``build`` so the dependency is a property
#: of the registry, and so the guard can prove the donor exists.
#:
#: The donor is the primary network fitted on the **same fold**, so the vectors XGBoost
#: receives were learned from that fold's training rows and nothing else. Reusing the
#: donor rather than fitting a second network is not only cheaper -- it is what makes the
#: experiment answer the question asked, which is whether *this component's* embeddings
#: help the estimator that currently wins.
EMBEDDING_DONOR: Mapping[str, str] = {"xgboost_chain_embeddings": "neural_embeddings"}

#: Models that are networks, in registry order. The ones ``train.fit_fold`` can fit.
MLP_MODELS: tuple[str, ...] = tuple(
    spec.name for spec in NEURAL_REGISTRY if spec.learner is Learner.MLP
)

#: The model whose learning curves, embeddings and seed spread are reported. Named rather
#: than inferred, so the figures cannot silently start describing a different model.
REPRESENTATIVE_MODEL = "neural_embeddings"


def spec_for(name: str) -> NeuralSpec:
    """Look up one neural specification."""
    if name not in SPECS_BY_NAME:
        raise KeyError(
            f"Unknown neural model: {name}. Registered: {', '.join(sorted(SPECS_BY_NAME))}"
        )
    return SPECS_BY_NAME[name]


def embedding_dim(column: str) -> int:
    """The embedding width for one entity column."""
    return EMBEDDING_DIMS[EntityFamily(column)]


def embedding_width(spec: NeuralSpec) -> int:
    """Total width the concatenated embedding block contributes for one spec."""
    if spec.encoding is not CategoricalEncoding.EMBEDDING:
        return 0
    return sum(embedding_dim(name) for name in spec.entity_columns)


def learning_rate_for(fold_set: str) -> float:
    """The frozen learning rate for one fold set.

    Falls back to the specified baseline for an unknown fold set rather than raising:
    a fold set with no frozen entry has not been searched, and the baseline is the
    specification's own value. The manifest records which was used.
    """
    entry = TUNED_HYPERPARAMS.get(fold_set)
    if entry is None:
        return BASELINE_LEARNING_RATE
    return float(entry.get("learning_rate", BASELINE_LEARNING_RATE))


def _guard_registry() -> None:
    """Reject a registry that could leak, or that names a column Component 4 lacks.

    Raises rather than asserts, and runs at import time, so a bad spec cannot be
    constructed and then used. Three separate guards, because ``feature_columns`` and
    ``entity_columns`` have different safety arguments and collapsing them would let one
    borrow the other's justification.
    """
    known = set(FEATURE_COLUMNS)
    allowed_entities = set(ENTITY_COLUMNS)
    seen: set[str] = set()

    for spec in NEURAL_REGISTRY:
        if spec.name in seen:
            raise ValueError(f"duplicate model name in registry: {spec.name}")
        seen.add(spec.name)

        # --- the numeric half: identical to Components 6 and 7 ---
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
        if len(spec.feature_columns) != len(set(spec.feature_columns)):
            raise ValueError(f"{spec.name}: feature_columns contains a duplicate")

        # --- the categorical half: a closed allowlist, and identity is not on it ---
        stray = sorted(set(spec.entity_columns) - allowed_entities)
        if stray:
            raise ValueError(
                f"{spec.name}: entity_columns names {', '.join(stray)}, which is not a "
                f"declared EntityFamily. Permitted: {', '.join(sorted(allowed_entities))}. "
                "establishment_id in particular is refused: a per-establishment "
                "parameter carries that establishment's future backwards. See ADR 0021."
            )
        forbidden = sorted(set(spec.entity_columns) & FORBIDDEN_COLUMNS)
        if forbidden:
            raise ValueError(
                f"{spec.name}: entity_columns contains identity, label or provenance "
                f"column(s) {', '.join(forbidden)}"
            )
        overlap = sorted(set(spec.entity_columns) & set(spec.feature_columns))
        if overlap:
            raise ValueError(
                f"{spec.name}: {', '.join(overlap)} appears as both a numeric feature "
                "and an entity column, so it would enter the network twice"
            )
        if len(spec.entity_columns) != len(set(spec.entity_columns)):
            raise ValueError(f"{spec.name}: entity_columns contains a duplicate")

        # --- the encoding and the columns must agree ---
        if spec.encoding is CategoricalEncoding.NONE and spec.entity_columns:
            raise ValueError(
                f"{spec.name}: encoding is NONE but entity_columns names "
                f"{', '.join(spec.entity_columns)}. Those columns would be declared and "
                "then silently ignored."
            )
        if spec.encoding is not CategoricalEncoding.NONE and not spec.entity_columns:
            raise ValueError(
                f"{spec.name}: encoding is {spec.encoding.value} but entity_columns is "
                "empty. Use CategoricalEncoding.NONE to mean 'no categoricals'."
            )
        if spec.learner is Learner.XGBOOST_EMBEDDING and spec.encoding is not (
            CategoricalEncoding.EMBEDDING
        ):
            raise ValueError(
                f"{spec.name}: the embeddings-into-XGBoost experiment requires learned "
                "embeddings to feed it"
            )

    for dependent, donor in EMBEDDING_DONOR.items():
        if dependent not in SPECS_BY_NAME:
            raise ValueError(f"EMBEDDING_DONOR names unknown model {dependent!r}")
        if donor not in SPECS_BY_NAME:
            raise ValueError(
                f"{dependent}: donor {donor!r} is not registered, so the embeddings it "
                "is supposed to consume would never be fitted"
            )
        donor_spec = SPECS_BY_NAME[donor]
        if donor_spec.learner is not Learner.MLP:
            raise ValueError(f"{dependent}: donor {donor!r} is not a network")
        borrowed = set(SPECS_BY_NAME[dependent].entity_columns) - set(donor_spec.entity_columns)
        if borrowed:
            raise ValueError(
                f"{dependent}: needs embedding(s) for {', '.join(sorted(borrowed))} that "
                f"its donor {donor!r} does not learn"
            )
    for spec in NEURAL_REGISTRY:
        if spec.learner is Learner.XGBOOST_EMBEDDING and spec.name not in EMBEDDING_DONOR:
            raise ValueError(
                f"{spec.name}: an embedding-fed booster with no declared donor could "
                "only get its vectors by fitting something undeclared"
            )

    if REPRESENTATIVE_MODEL not in SPECS_BY_NAME:
        raise ValueError(
            f"REPRESENTATIVE_MODEL {REPRESENTATIVE_MODEL!r} is not in the registry, so "
            "the figures would describe nothing"
        )
    missing_dims = sorted(set(ENTITY_COLUMNS) - {f.value for f in EMBEDDING_DIMS})
    if missing_dims:
        raise ValueError(f"no embedding dimension declared for: {', '.join(missing_dims)}")


_guard_registry()


__all__ = [
    "ALL_ENTITIES",
    "BASELINE_LEARNING_RATE",
    "BATCH_SIZE",
    "DEFAULT_SEED",
    "DROPOUT",
    "EARLY_STOPPING_PATIENCE",
    "EMBEDDING_DIMS",
    "EMBEDDING_DONOR",
    "ENTITY_COLUMNS",
    "GRADIENT_CLIP_NORM",
    "HIDDEN_SIZES",
    "INDEPENDENT_CHAIN",
    "INNER_VALIDATION_FRACTION",
    "LEARNING_RATE_GRID",
    "LOSS",
    "MAX_EPOCHS",
    "MLP_MODELS",
    "MIN_INNER_SPLIT_ROWS",
    "NEURAL_DEFINITION_VERSION",
    "NEURAL_REGISTRY",
    "OPTIMIZER",
    "OUTPUT_SIZE",
    "POS_WEIGHT_DEFAULT",
    "REPRESENTATIVE_MODEL",
    "SCHEDULER",
    "SCHEDULER_FACTOR",
    "SCHEDULER_MIN_LR",
    "SCHEDULER_PATIENCE",
    "SEED_SWEEP",
    "SPECS_BY_NAME",
    "TUNED_HYPERPARAMS",
    "TUNED_HYPERPARAMS_PROVENANCE",
    "TUNING_SEED",
    "UNKNOWN_CATEGORY",
    "WEIGHT_DECAY",
    "CategoricalEncoding",
    "EntityFamily",
    "Learner",
    "NeuralSpec",
    "embedding_dim",
    "embedding_width",
    "learning_rate_for",
    "spec_for",
]
