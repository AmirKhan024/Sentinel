"""Boosted model specifications, the search space, and the frozen tuned parameters.

Mirrors ``modeling.definitions`` deliberately: a declared registry, feature partitions
derived from Component 4 rather than hand-listed, and an import-time guard that raises
(never asserts, because ``assert`` is stripped under ``python -O``) on any spec that
could leak or that names a column Component 4 does not have.

Two things here have no Component 6 counterpart.

**A search space.** Component 6's ``LOGISTIC_PARAMS`` is a constant because a baseline
exists to be trustworthy rather than optimal. A booster has no defensible default, so the
space is declared here as data -- one row per tuned dimension -- and ``tuning`` drives
Optuna from it. Declaring rather than hard-coding the suggests means the space is a thing
tests and documentation can read, and the XGBoost/LightGBM mapping is checkable rather
than asserted in prose.

**Frozen tuned parameters.** ``TUNED_PARAMS`` holds the outcome of the search, keyed by
model and fold set, so ``train-boosting`` reproduces without re-running it -- exactly as
``LOGISTIC_PARAMS`` works. The values are literals under version control with a
provenance string naming the study that produced them. That is the point: once frozen,
no test-window number can change them without the change appearing in a diff.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from sentinel.features.definitions import FEATURE_COLUMNS
from sentinel.modeling.definitions import FORBIDDEN_COLUMNS

#: Bumped whenever a booster's feature set, search space, fixed parameters or
#: preprocessing changes in a way that makes two runs incomparable.
BOOSTING_DEFINITION_VERSION = "v1"

#: One seed for the estimators, one for the sampler. Both are load-bearing here, unlike
#: Component 6 where ``random_state`` was inert under lbfgs: a booster subsamples rows
#: and columns, and TPE draws candidates from a distribution.
DEFAULT_SEED = 42
TUNING_SEED = 20260817


class Estimator(StrEnum):
    """Which library fits the model. Closed set -- there is no third option in C7."""

    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


# --- fixed parameters --------------------------------------------------------

#: Never tuned, never varied. Split into a separate mapping from the search space so the
#: reason each one is fixed can be stated once, and so a test can assert that tuning
#: cannot reach them.
#:
#: The single-thread settings are the determinism guarantee. Histogram construction in
#: both libraries reduces over threads, and a float reduction's result depends on the
#: order the threads finish in -- so a multi-threaded fit is reproducible only
#: approximately, and Component 6's standard for "did not move" is bit-identical. One
#: thread is slower and exactly reproducible; that is the trade this project takes
#: everywhere. See ADR 0016.
FIXED_PARAMS: Mapping[Estimator, Mapping[str, object]] = {
    Estimator.XGBOOST: {
        "objective": "binary:logistic",
        # The label is 0/1 and the score must be P(target=1). An eval metric of ``aucpr``
        # matches the tuning objective, so early stopping and the study agree about what
        # "better" means.
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "n_jobs": 1,
        # NaN is routed to a learned default direction at each split rather than
        # imputed. Component 4's NULLs mean "no prior canvass", which is information;
        # ``missing=nan`` is what lets the model use it as such.
        "missing": float("nan"),
    },
    Estimator.LIGHTGBM: {
        "objective": "binary",
        "metric": "average_precision",
        "n_jobs": 1,
        "num_threads": 1,
        # LightGBM's own reproducibility switches. ``force_row_wise`` pins the histogram
        # construction strategy, which is otherwise chosen at runtime from a timing
        # probe -- i.e. from the machine's load, which is not an input we control.
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    },
}

#: Upper bound on boosting rounds during the search. Early stopping inside the tuning
#: objective decides the actual count; this only stops a pathological trial from running
#: forever at a very low learning rate.
MAX_BOOSTING_ROUNDS = 2000

#: Rounds without validation improvement before the search stops a fit. Used *only*
#: inside the tuning objective, against an inner validation window that is training data
#: for every outer fold. Final fits use a fixed ``n_estimators`` and stop nothing.
EARLY_STOPPING_ROUNDS = 50


@dataclass(frozen=True, slots=True)
class SearchDimension:
    """One tuned hyperparameter.

    ``concept`` is the model-independent thing being tuned; ``name`` is the library's
    own parameter name. Keeping both is what makes the two searches comparable: the
    claim "XGBoost and LightGBM explored the same space" is checkable by comparing
    concepts, not by hoping two parameter lists mean the same thing.
    """

    concept: str
    name: str
    kind: str
    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"int", "float"}:
            raise ValueError(f"{self.name}: kind must be 'int' or 'float', got {self.kind!r}")
        if self.low > self.high:
            raise ValueError(f"{self.name}: low {self.low} exceeds high {self.high}")


#: The tuned space, per estimator. Ranges follow the project specification's intended
#: space; the parameter *names* are each library's own, because copying XGBoost's names
#: into LightGBM would silently tune nothing.
SEARCH_SPACE: Mapping[Estimator, tuple[SearchDimension, ...]] = {
    Estimator.XGBOOST: (
        SearchDimension("tree_depth", "max_depth", "int", 3, 10),
        SearchDimension("learning_rate", "learning_rate", "float", 0.01, 0.3, log=True),
        SearchDimension("leaf_size_regularizer", "min_child_weight", "float", 1.0, 20.0, log=True),
        SearchDimension("row_subsample", "subsample", "float", 0.6, 1.0),
        SearchDimension("column_subsample", "colsample_bytree", "float", 0.6, 1.0),
        SearchDimension("l1_penalty", "reg_alpha", "float", 1e-8, 10.0, log=True),
        SearchDimension("l2_penalty", "reg_lambda", "float", 1e-8, 10.0, log=True),
    ),
    Estimator.LIGHTGBM: (
        SearchDimension("tree_depth", "max_depth", "int", 3, 10),
        SearchDimension("leaf_count", "num_leaves", "int", 8, 256, log=True),
        SearchDimension("learning_rate", "learning_rate", "float", 0.01, 0.3, log=True),
        SearchDimension("leaf_size_regularizer", "min_child_samples", "int", 5, 100, log=True),
        SearchDimension("row_subsample", "bagging_fraction", "float", 0.6, 1.0),
        SearchDimension("column_subsample", "feature_fraction", "float", 0.6, 1.0),
        SearchDimension("l1_penalty", "lambda_l1", "float", 1e-8, 10.0, log=True),
        SearchDimension("l2_penalty", "lambda_l2", "float", 1e-8, 10.0, log=True),
    ),
}

#: The concept-to-parameter mapping, written out because it is the part of Component 7
#: most easily got wrong and least visible when it is. Reproduced in the data contract.
#:
#: The asymmetry is real, not an oversight. XGBoost grows depth-wise, so ``max_depth``
#: alone bounds a tree at 2^depth leaves. LightGBM grows leaf-wise, so ``max_depth``
#: bounds nothing on its own and ``num_leaves`` is the actual capacity knob -- which is
#: why LightGBM has one extra dimension and why ``tuning`` caps ``num_leaves`` at
#: 2^``max_depth``. Without that cap the two searches would explore different capacity
#: ranges and the comparison between them would be measuring the space, not the model.
PARAM_MAPPING: tuple[tuple[str, str | None, str | None, str], ...] = (
    ("tree_depth", "max_depth", "max_depth", "same name, different force: see leaf_count"),
    (
        "leaf_count",
        None,
        "num_leaves",
        "LightGBM only. XGBoost's depth-wise growth implies at most 2^max_depth leaves, "
        "so capping num_leaves at 2^max_depth keeps the two capacity ranges equal.",
    ),
    ("learning_rate", "learning_rate", "learning_rate", "identical semantics"),
    (
        "leaf_size_regularizer",
        "min_child_weight",
        "min_child_samples",
        "not interchangeable: XGBoost sums the Hessian in a child, LightGBM counts rows. "
        "Both restrict how small a leaf may get, on different scales, so each is tuned "
        "over its own natural range rather than a shared one.",
    ),
    ("row_subsample", "subsample", "bagging_fraction", "LightGBM also needs bagging_freq=1"),
    ("column_subsample", "colsample_bytree", "feature_fraction", "per-tree in both"),
    ("l1_penalty", "reg_alpha", "lambda_l1", "identical semantics"),
    ("l2_penalty", "reg_lambda", "lambda_l2", "identical semantics"),
    (
        "boosting_rounds",
        "n_estimators",
        "n_estimators",
        "not searched. Set from early stopping inside the tuning objective and then "
        "frozen, so no final fit reads a validation window.",
    ),
)

#: Where ``TUNED_PARAMS`` came from. A reader can tell a frozen search result from a
#: placeholder without reading git history, and can re-open the study that produced it.
#:
#: The four best mean validation PR-AUCs were xgboost-quarterly 0.6054 (trial 36),
#: lightgbm-quarterly 0.6033 (trial 28), xgboost-covid_shift 0.7786 (trial 63) and
#: lightgbm-covid_shift 0.7777 (trial 74). **None of those is a result.** They are
#: measured on inner validation quarters that are training data for every outer fold
#: these parameters are then used on, and the covid_shift figures are higher only
#: because its region sits in the pre-2020 era when the base rate was around 0.82.
#: Quoting either as Component 7's performance would be reporting an in-sample number
#: as an out-of-sample one.
TUNED_PARAMS_PROVENANCE = (
    "sentinel tune-boosting --trials 100, 400 trials over 4 studies, 0 failed, 563.8s. "
    "Study artifact tuning_trials_20260817T155315Z.parquet "
    "sha256 a77687b7e56e34954b3463c64a071b3708ba2220b613efe041cd5235adec8b14, "
    "from as_of_features_20260816T150313Z.parquet (57,727 rows). "
    "TPESampler seed 20260817; xgboost 3.4.1, lightgbm 4.7.0, optuna 4.9.0."
)

#: Best parameters per model per fold set. Two fold sets means two studies, because the
#: quarterly tuning region contains the covid_shift *test* window: one shared study would
#: make the shift result optimistically biased and therefore not comparable to Component
#: 6's clean one, and the shift result is the single number this project most needs to
#: stay honest. See ADR 0017.
TUNED_PARAMS: Mapping[str, Mapping[str, Mapping[str, object]]] = {
    Estimator.XGBOOST.value: {
        # Trial 36 of 100. Mean inner-validation PR-AUC 0.6054 over 6 quarterly inner
        # folds ending 2022-03-31, the day before the first quarterly test window opens.
        "quarterly": {
            "max_depth": 4,
            "learning_rate": 0.1926324953901243,
            "min_child_weight": 12.607179102564166,
            "subsample": 0.7877818882255597,
            "colsample_bytree": 0.668974715332733,
            "reg_alpha": 1.0910568566546592e-08,
            "reg_lambda": 0.005889806219464979,
            "n_estimators": 103,
        },
        # Trial 63 of 100. Mean inner-validation PR-AUC 0.7786 over 2 inner folds ending
        # 2020-05-31. A separate study, not a copy of the quarterly one: the quarterly
        # region contains this fold's entire test window, so sharing parameters would
        # make the shift result optimistically biased. Note the search landed somewhere
        # genuinely different -- shallower trees, a quarter of the learning rate and
        # nearly twice the rounds -- which is the pre-COVID regime's own answer.
        "covid_shift": {
            "max_depth": 3,
            "learning_rate": 0.05598704173922894,
            "min_child_weight": 2.649550642462423,
            "subsample": 0.762140912352976,
            "colsample_bytree": 0.6173166616787783,
            "reg_alpha": 1.994200257477126e-07,
            "reg_lambda": 0.02981913787127491,
            "n_estimators": 192,
        },
    },
    Estimator.LIGHTGBM.value: {
        # Trial 28 of 100. Mean inner-validation PR-AUC 0.6033.
        # ``num_leaves`` 16 is exactly 2^``max_depth``, i.e. the capacity cap binding:
        # the search wanted a wider tree than a depth-4 XGBoost tree could build, and
        # was held to the same ceiling so the comparison stays about the algorithm.
        "quarterly": {
            "max_depth": 4,
            "num_leaves": 16,
            "learning_rate": 0.2992200240188712,
            "min_child_samples": 54,
            "bagging_fraction": 0.7781136640014453,
            "bagging_freq": 1,
            "feature_fraction": 0.6046947274523935,
            "lambda_l1": 0.00021894473605833144,
            "lambda_l2": 7.095252963920867e-07,
            "n_estimators": 63,
        },
        # Trial 74 of 100. Mean inner-validation PR-AUC 0.7777.
        "covid_shift": {
            "max_depth": 8,
            "num_leaves": 10,
            "learning_rate": 0.057811212728145885,
            "min_child_samples": 44,
            "bagging_fraction": 0.6320623097532436,
            "bagging_freq": 1,
            "feature_fraction": 0.8934942988225112,
            "lambda_l1": 1.4203794299154232e-05,
            "lambda_l2": 0.2919393588867272,
            "n_estimators": 54,
        },
    },
}


@dataclass(frozen=True, slots=True)
class BoostingSpec:
    """One boosted model, fully specified.

    ``feature_columns`` is explicit and closed for the same reason Component 6's is:
    "every column except the target" would silently absorb a future Component 4 metadata
    column into the model.

    ``class_weighted`` is the one ablation this component carries. It is False for both
    primary models -- measured prevalence is 52.52%, so there is no imbalance to correct,
    and weighting would distort the probability scale Component 9 has to calibrate. It
    exists as an explicit, separately-named variant rather than a default because "we
    tested weighting and it did not help" is only sayable if it was actually tested.
    """

    name: str
    version: str
    description: str
    estimator: Estimator
    feature_columns: tuple[str, ...]
    is_probability: bool
    seed: int
    class_weighted: bool = False


BOOSTING_REGISTRY: tuple[BoostingSpec, ...] = (
    BoostingSpec(
        name="xgboost",
        version="v1",
        description=(
            "Gradient-boosted trees (XGBoost, hist) on all 26 as-of features plus the "
            "four null-rule family indicators. NULLs reach the model as NaN and are "
            "routed by a learned default direction rather than imputed."
        ),
        estimator=Estimator.XGBOOST,
        feature_columns=FEATURE_COLUMNS,
        is_probability=True,
        seed=DEFAULT_SEED,
    ),
    BoostingSpec(
        name="lightgbm",
        version="v1",
        description=(
            "Gradient-boosted trees (LightGBM, leaf-wise) on the same 26 features and "
            "the same four indicators as xgboost, tuned over a concept-matched search "
            "space so the comparison measures the algorithm rather than the space."
        ),
        estimator=Estimator.LIGHTGBM,
        feature_columns=FEATURE_COLUMNS,
        is_probability=True,
        seed=DEFAULT_SEED,
    ),
    BoostingSpec(
        name="xgboost_class_weighted",
        version="v1",
        description=(
            "The class-weighting ablation: xgboost with scale_pos_weight set from the "
            "training window's own prevalence, and otherwise identical parameters. Not "
            "a default and not separately tuned -- it reuses xgboost's frozen "
            "parameters so the only difference between the two is the weighting."
        ),
        estimator=Estimator.XGBOOST,
        feature_columns=FEATURE_COLUMNS,
        is_probability=True,
        seed=DEFAULT_SEED,
        class_weighted=True,
    ),
)

BOOSTING_MODELS_BY_NAME: dict[str, BoostingSpec] = {s.name: s for s in BOOSTING_REGISTRY}

#: Models that are tuned in their own right. An ablation is excluded: it borrows its
#: donor's parameters, so searching for it separately would vary two things at once.
TUNABLE_MODELS: tuple[str, ...] = tuple(s.name for s in BOOSTING_REGISTRY if not s.class_weighted)

#: The model whose frozen parameters an ablation borrows. Keyed by ablation name so the
#: borrowing is declared rather than inferred from a name prefix.
PARAMETER_DONOR: Mapping[str, str] = {"xgboost_class_weighted": "xgboost"}


def spec_for(name: str) -> BoostingSpec:
    """Look up one boosted model specification."""
    if name not in BOOSTING_MODELS_BY_NAME:
        raise KeyError(
            f"Unknown boosted model: {name}. "
            f"Registered: {', '.join(sorted(BOOSTING_MODELS_BY_NAME))}"
        )
    return BOOSTING_MODELS_BY_NAME[name]


def search_space(estimator: Estimator) -> tuple[SearchDimension, ...]:
    """The tuned dimensions for one library."""
    return SEARCH_SPACE[estimator]


def tuned_params(spec: BoostingSpec, fold_set: str) -> dict[str, object]:
    """The frozen search result for one model on one fold set.

    An ablation borrows its donor's parameters, so the ablation measures the one thing
    it varies. Raises rather than falling back to a default: a missing entry means the
    study for that fold set was never run, and quietly fitting library defaults would
    produce a number indistinguishable from a tuned one.
    """
    source = PARAMETER_DONOR.get(spec.name, spec.name)
    by_fold_set = TUNED_PARAMS.get(source)
    if by_fold_set is None:
        raise KeyError(f"no frozen parameters for model {source!r}")
    params = by_fold_set.get(fold_set)
    if params is None:
        raise KeyError(
            f"no frozen parameters for model {source!r} on fold set {fold_set!r}. "
            f"Known fold sets: {', '.join(sorted(by_fold_set))}. Parameters are never "
            "borrowed across fold sets -- each study is confined to a region earlier "
            "than its own fold set's first test window."
        )
    return dict(params)


def estimator_params(spec: BoostingSpec, fold_set: str) -> dict[str, object]:
    """Everything the estimator constructor receives: fixed, then tuned, then the seeds.

    Merge order is deliberate and guarded at import: the search space and the fixed
    parameters are disjoint, so a tuned value can never silently overwrite a determinism
    or objective setting.
    """
    params: dict[str, object] = dict(FIXED_PARAMS[spec.estimator])
    params.update(tuned_params(spec, fold_set))
    params["random_state"] = spec.seed
    if spec.estimator is Estimator.LIGHTGBM:
        # LightGBM reads three further seeds independently of ``random_state``; leaving
        # them unset makes bagging and feature sampling irreproducible.
        params["bagging_seed"] = spec.seed
        params["feature_fraction_seed"] = spec.seed
        params["data_random_seed"] = spec.seed
    return params


def n_estimators_of(spec: BoostingSpec, fold_set: str) -> int:
    """The frozen boosting-round count, narrowed out of an untyped mapping."""
    value = tuned_params(spec, fold_set).get("n_estimators", 0)
    return value if isinstance(value, int) else 0


def _guard_registry() -> None:
    """Reject a registry or a search space that could leak, mislabel or collide.

    Raises rather than asserts, at import time, so a bad spec cannot be constructed and
    then used. Mirrors ``modeling.definitions._guard_registry`` and adds the three checks
    that only a tuned component needs.
    """
    known = set(FEATURE_COLUMNS)
    registered = {s.name for s in BOOSTING_REGISTRY}
    seen: set[str] = set()
    for spec in BOOSTING_REGISTRY:
        if spec.name in seen:
            raise ValueError(f"duplicate model name in boosting registry: {spec.name}")
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
        if len(spec.feature_columns) != len(set(spec.feature_columns)):
            raise ValueError(f"{spec.name}: feature_columns contains a duplicate")

    for ablation, donor in PARAMETER_DONOR.items():
        if ablation not in registered:
            raise ValueError(f"parameter donor declared for unregistered model {ablation!r}")
        if donor not in registered:
            raise ValueError(f"{ablation}: parameter donor {donor!r} is not a registered model")

    for estimator, dimensions in SEARCH_SPACE.items():
        names = [d.name for d in dimensions]
        if len(names) != len(set(names)):
            raise ValueError(f"{estimator.value}: search space names a parameter twice")
        collision = sorted(set(names) & set(FIXED_PARAMS[estimator]))
        if collision:
            raise ValueError(
                f"{estimator.value}: search space would overwrite fixed parameter(s) "
                f"{', '.join(collision)}. A tuned value must never reach a determinism "
                "or objective setting."
            )
        if "n_estimators" in names:
            raise ValueError(
                f"{estimator.value}: n_estimators must not be a searched dimension. It "
                "is set from early stopping inside the objective and then frozen."
            )

    mapped = {row[0] for row in PARAM_MAPPING}
    for estimator, dimensions in SEARCH_SPACE.items():
        undocumented = sorted({d.concept for d in dimensions} - mapped)
        if undocumented:
            raise ValueError(
                f"{estimator.value}: search dimension(s) {', '.join(undocumented)} are "
                "absent from PARAM_MAPPING, so the two searches could not be shown "
                "comparable."
            )


_guard_registry()


__all__ = [
    "BOOSTING_DEFINITION_VERSION",
    "BOOSTING_MODELS_BY_NAME",
    "BOOSTING_REGISTRY",
    "DEFAULT_SEED",
    "EARLY_STOPPING_ROUNDS",
    "FIXED_PARAMS",
    "MAX_BOOSTING_ROUNDS",
    "PARAMETER_DONOR",
    "PARAM_MAPPING",
    "SEARCH_SPACE",
    "TUNABLE_MODELS",
    "TUNED_PARAMS",
    "TUNED_PARAMS_PROVENANCE",
    "TUNING_SEED",
    "BoostingSpec",
    "Estimator",
    "SearchDimension",
    "estimator_params",
    "n_estimators_of",
    "search_space",
    "spec_for",
    "tuned_params",
]
