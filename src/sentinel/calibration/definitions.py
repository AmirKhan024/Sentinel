"""Frozen specifications and every pre-declared constant for Component 9.

Nothing else in this package may hold a magic number. That is not a style preference: the
whole defensibility of a calibrator selection rests on the rule having been fixed *before*
the test window was opened, and a threshold written inline at its point of use is a
threshold nobody can date. Every constant here is justified by a measurement in
``docs/analysis/calibration_findings.md`` and frozen in ADR 0025 or ADR 0027, and a
validation check asserts the manifest's copies match these literals -- so a run cannot
report a rule it did not use.

The candidate set is deliberately small. Twelve models have prediction artifacts on disk;
calibrating all of them would be an artifact dump rather than a principled comparison. The
five here are the ones that disagree about which is best, each for a stated reason.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

#: Bumped when anything in this module changes the meaning of an emitted column.
CALIBRATION_DEFINITION_VERSION = "v1"


class Method(StrEnum):
    """The two calibration methods. Both are always fitted; one is frozen per fold."""

    PLATT = "platt"
    ISOTONIC = "isotonic"


class Family(StrEnum):
    """Which component's fit function produces a candidate's base scores.

    Not cosmetic -- ``basescores.py`` dispatches on it, and each family reaches its raw
    decision margin differently (ADR 0027).
    """

    LOGISTIC = "logistic"
    BOOSTED = "boosted"
    NEURAL_MLP = "neural_mlp"
    NEURAL_EMBEDDING_BOOSTER = "neural_embedding_booster"


#: Stages a metric can be measured at. ``SELECTED`` is whichever of the two the protocol
#: froze for that fold, duplicated so a consumer never has to join the selection log to
#: read the production number.
STAGE_UNCALIBRATED = "uncalibrated"
STAGE_SELECTED = "selected"
STAGES: tuple[str, ...] = (STAGE_UNCALIBRATED, Method.PLATT, Method.ISOTONIC, STAGE_SELECTED)


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One base model Component 9 calibrates, and why it is in the set."""

    name: str
    family: Family
    component: int
    source_slug: str
    rationale: str
    is_experimental: bool = False
    version: str = "v1"

    def calibrated_name(self, method: Method) -> str:
        """The ``model_name`` a calibrated row carries.

        Never the bare base name: a calibrated row and its uncalibrated ancestor must be
        able to sit in one results table without either being mistaken for the other.
        """
        return f"{self.name}_{method.value}"


#: The candidate set, with the reason each is included recorded beside it.
#:
#: The three metrics disagree about the winner, and that disagreement is the reason to
#: calibrate more than one: `neural_numeric_only` has the best NDE, Brier and ECE,
#: `lightgbm` the best precision@k_1_day, `xgboost_chain_embeddings` the best PR-AUC.
#: Choosing before calibrating would foreclose the comparison.
CANDIDATE_REGISTRY: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        name="logistic_regression",
        family=Family.LOGISTIC,
        component=6,
        source_slug="baseline_predictions",
        rationale=(
            "the reference simple model, and the only candidate whose scores are float64 "
            "end to end. Best MCE in the project (0.1664) before any calibration."
        ),
    ),
    CandidateSpec(
        name="xgboost",
        family=Family.BOOSTED,
        component=7,
        source_slug="boosted_predictions",
        rationale="the stronger of the two tuned boosters on NDE (0.2376) and ROC-AUC.",
    ),
    CandidateSpec(
        name="lightgbm",
        family=Family.BOOSTED,
        component=7,
        source_slug="boosted_predictions",
        rationale=(
            "best precision@k_1_day, which is the metric closest to a day's real capacity, "
            "and the worst uncalibrated ECE of the four non-experimental candidates (0.0644)."
        ),
    ),
    CandidateSpec(
        name="neural_numeric_only",
        family=Family.NEURAL_MLP,
        component=8,
        source_slug="neural_predictions",
        rationale=(
            "best uncalibrated Brier (0.2355), ECE (0.0563) and NDE (0.2482) in the project. "
            "The model most likely to be carried forward, and therefore the one whose "
            "calibration matters most."
        ),
    ),
    CandidateSpec(
        name="xgboost_chain_embeddings",
        family=Family.NEURAL_EMBEDDING_BOOSTER,
        component=8,
        source_slug="neural_predictions",
        rationale=(
            "best PR-AUC, the specification's primary classification metric. Carried as an "
            "explicitly experimental Component 8 derivative under ADR 0022's labelling "
            "regime: it lost on NDE, and calibrating it well must not make it the headline."
        ),
        is_experimental=True,
    ),
)

CANDIDATES_BY_NAME: dict[str, CandidateSpec] = {c.name: c for c in CANDIDATE_REGISTRY}

#: Component 8's donor network for the embedding-fed booster. Restated rather than imported
#: so this module lists every model Component 9 will fit, including the one it fits only as
#: an input to another.
EMBEDDING_DONOR = "neural_embeddings"

#: Models deliberately *not* calibrated, and why. Recorded so the candidate set reads as a
#: decision rather than as whatever happened to be convenient.
EXCLUDED_MODELS: Mapping[str, str] = {
    "logistic_regression_no_scheduling": "an ablation of a candidate, not a competing model",
    "cdph_2015_approximation": (
        "reaches only 3 of the deployed 2015 model's 10 input families; calibrating an "
        "acknowledged approximation would lend it an authority it does not have"
    ),
    "xgboost_class_weighted": (
        "a class-weighting ablation; its uncalibrated ECE is among the worst measured"
    ),
    "neural_embeddings": (
        "fitted as the donor for xgboost_chain_embeddings, but not calibrated in its own "
        "right: its embeddings lost, and HANDOFF is explicit that they are not to be adopted"
    ),
    "neural_onehot": "an encoding ablation",
    "neural_no_chain": "a Component 8 ablation, retained there for the embedding question",
    "neural_no_facility_type": "a Component 8 ablation",
    "neural_no_community_area": "a Component 12 fairness input, not a model choice (ADR 0023)",
    "neural_no_zip": "a Component 8 ablation",
    "neural_pos_weighted": "measures what class weighting costs; ECE 0.1002, the worst measured",
}


# --- the selection protocol (ADR 0025) --------------------------------------
#
# Every constant below was fixed by a profiling pass over calibration windows only, and
# committed before the first production run. See calibration_findings.md sections 3 and 6.

#: Share of each calibration window held back to choose between the two methods. 0.30
#: rather than Component 8's 0.15 because a calibration window is an order of magnitude
#: smaller than a training window; at 0.15 the smallest inner-select portion would be ~204
#: rows. Measured: 409-756 select rows across the 18 folds.
INNER_SELECT_FRACTION = 0.30

#: Below these a fold is refused rather than calibrated on a window too small to mean
#: anything, matching ``neural.train.split_training_window``. Smallest measured on this
#: snapshot: 948 inner-fit, 409 inner-select -- so the guards are real but not binding.
MIN_INNER_FIT_ROWS = 400
MIN_INNER_SELECT_ROWS = 250

#: Mean inner-select log-loss decides the method. Not ECE: 15 equal-mass bins over a
#: ~500-row inner-select window is 27-50 rows per bin, ECE is not a proper scoring rule,
#: and its bin count is a free parameter -- a selection rule that can be tuned is not a
#: rule. ECE, MCE and Brier are recorded on the same window as diagnostics that decide
#: nothing.
SELECTION_METRIC = "inner_select_log_loss"

#: Selection granularity. Fold k's method is the winner on the mean over folds 1..k of the
#: same fold set. Every input then has ``rd <= fold_k.calibration_end``, which is exactly
#: the horizon ``evaluation.contract`` already enforces. Pooling *all* folds instead would
#: be leakage: fold N's calibration window is fold N-1's TEST window.
SELECTION_GRANULARITY = "expanding prefix over folds 1..k, per (model, fold_set)"

#: Absolute mean-log-loss gap below which the two methods are declared tied, in nats.
#:
#: Measured, not chosen: the paired bootstrap gap SD over 72 (model, fold) cells is
#: min 0.0022, median 0.0054, max 0.1595. This is one median paired-gap SD. The
#: implementation plan proposed 0.002, which sits *below* the smallest observed SD and
#: would have declared winners on differences finer than the noise of the comparison.
TIE_THRESHOLD = 0.005

#: Platt wins ties, and also wins when it is merely not-worse-by-enough. Three reasons,
#: none of them a result: two parameters against up to ~2,300 breakpoints on a ~1,200-row
#: fit; Platt is *strictly* monotone where isotonic is only weakly monotone, so isotonic's
#: plateaus can move top-k membership through the tie-break; and isotonic needs
#: ``out_of_bounds="clip"`` and so has a hard floor and ceiling at the calibration window's
#: extremes.
TIE_PREFERENCE = Method.PLATT

#: The rule, in one line, so the manifest can state it without prose drift.
SELECTION_RULE = (
    f"isotonic iff mean_{SELECTION_METRIC}(isotonic) < "
    f"mean_{SELECTION_METRIC}(platt) - {TIE_THRESHOLD}; otherwise {TIE_PREFERENCE.value}"
)


# --- the calibrators themselves (ADR 0027) -----------------------------------

#: Platt is a two-parameter maximum-likelihood fit, so it is deliberately unpenalised.
#: scikit-learn's default ``C=1.0`` would shrink the slope toward zero and *cause* the
#: under-confidence the calibrator exists to remove.
PLATT_PARAMS: Mapping[str, object] = {
    "C": 1e10,
    "solver": "lbfgs",
    "max_iter": 1000,
    "fit_intercept": True,
}

#: ``out_of_bounds="clip"`` is mandatory, not a default worth reconsidering: a test-window
#: score outside the calibration window's observed range would otherwise map to NaN, which
#: the prediction contract rejects as a null score.
ISOTONIC_PARAMS: Mapping[str, object] = {
    "y_min": 0.0,
    "y_max": 1.0,
    "increasing": True,
    "out_of_bounds": "clip",
}

#: How far the recalibration slope of a Platt-calibrated window may sit from 1.0.
#:
#: Refitting the Cox regression on Platt's own output must return slope 1.0 by
#: construction -- it is the same estimator on the same rows -- so this is a free self-check
#: that the calibrator was applied the way it was fitted. It is not *exactly* 1.0 because
#: ``C = 1e10`` is a large but finite penalty: the measured residual is 4.5e-5. The
#: tolerance is set above that and far below the departure a misapplication would cause,
#: which is O(0.1) or worse.
PLATT_SELF_CHECK_TOLERANCE = 1e-3

#: Platt receives the recovered logit; isotonic receives the probability. Isotonic is
#: invariant to a monotone reparametrisation, so the choice is free and ``p`` keeps the
#: persisted breakpoints readable.
INPUT_TRANSFORM: Mapping[Method, str] = {
    Method.PLATT: "logit",
    Method.ISOTONIC: "identity",
}

#: Guard for ``logit(p)`` at the extremes. The value and rationale of
#: ``evaluation.metrics.LOG_LOSS_EPSILON``. On this snapshot nothing saturates and the
#: clamp never fires; ``logit_clamped_rows`` is reported so a later snapshot that does is
#: visible rather than silently corrected.
LOGIT_EPSILON = 1e-15

#: Tolerance for the warn-severity check that ``logit(p)`` agrees with the base model's own
#: decision margin.
#:
#: Set from measurement, not from expectation. xgboost and the network compute in float32,
#: so the sigmoid round trip loses float32 precision: the observed maximum discrepancy is
#: 2.6e-5 (neural) and 1.4e-6 (xgboost), against 1e-13 for the two float64 models. The plan
#: proposed 1e-9, which would have fired on 33,898 correct rows. 1e-4 sits above the
#: observed maximum and still catches what the check exists for -- a double sigmoid, a sign
#: flip or a mis-join is an O(1) discrepancy, not an O(1e-5) one.
MARGIN_TOLERANCE = 1e-4


# --- uncertainty -------------------------------------------------------------

BOOTSTRAP_REPLICATIONS = 1000
#: Date-style seed, matching ``TUNING_SEED`` and ``TSNE_SEED``.
BOOTSTRAP_SEED = 20260824
CI_LEVEL = 0.95

#: Both schemes are run for every reported interval. Rows are not independent within a
#: window -- an establishment can appear more than once and its rows share an as-of history
#: -- so an i.i.d. row bootstrap understates the standard error. Running both settles the
#: objection with a measurement instead of a caveat.
BOOTSTRAP_SCHEME_ROW = "row"
BOOTSTRAP_SCHEME_BLOCK = "establishment_block"
BOOTSTRAP_SCHEMES: tuple[str, ...] = (BOOTSTRAP_SCHEME_ROW, BOOTSTRAP_SCHEME_BLOCK)

#: Above this share of single-class resamples the interval is written as null rather than
#: computed from whatever survived.
MAX_DEGENERATE_SHARE = 0.05

BOOTSTRAP_CAVEAT = (
    "The within-fold percentile interval is the confidence interval. The across-fold SD is "
    "a DISPERSION, not a confidence interval: the folds share an expanding training window, "
    "so fold 17's training rows are a superset of fold 1's, the per-fold estimates are "
    "strongly positively dependent, and a t-interval built from this SD would be "
    "anticonservative. covid_shift is never pooled with the quarterly folds."
)


# --- provenance semantics ----------------------------------------------------
#
# Carried as prose into the manifest so a consumer never has to infer them.

TRAINED_THROUGH_SEMANTICS = (
    "fold.calibration_end -- and that is LATER than the base estimator's horizon. The "
    "estimator's weights were fitted through fold.train_end and are bit-identical to the "
    "committed Component 6/7/8 artifact; the calibrator was then fitted on the fold's "
    "calibration window, which is what that window exists for (ADR 0012, ADR 0014). This "
    "artifact must NOT be described as trained only through train_end. The contract's "
    "ceiling is exactly calibration_end, so it sits at the ceiling rather than past it."
)

AVAILABLE_FROM_SEMANTICS = (
    "fold.test_start -- the first date on which this score could have been produced in "
    "operation, being the day after the last row the calibrator was allowed to learn from."
)

PROBABILITY_SEMANTICS = (
    "A CALIBRATED probability of a Priority or Priority Foundation citation, produced by "
    "applying a frozen per-fold calibrator to the base model's committed uncalibrated "
    "score. The base score is carried alongside in `base_score` so the correction is "
    "always visible. The calibrator is never applied twice: model_name is "
    "'<base>_<method>', which is not a name any base model carries."
)

SCORE_DIRECTION = "descending: higher score = higher predicted violation risk"

#: Experiments Component 9 does not run, reported rather than faked -- the convention every
#: component since Component 7 follows.
BLOCKED_EXPERIMENTS: tuple[str, ...] = (
    "cost-sensitive thresholding, the deferral gate and any operating threshold: Component "
    "9 makes the probability trustworthy and stops there. A threshold is Component 16's.",
    "seed averaging of the neural candidate: deferred by decision, not oversight. Neural "
    "seed noise (0.0058 ROC-AUC) exceeds that family's entire advantage over XGBoost, so a "
    "seed-averaged model would be a better model -- but it would be a NEW base model that "
    "Component 8 never evaluated, and it would break the bit-identity gate ADR 0026 rests "
    "on. Carried to the Component 10 handoff.",
    "temperature scaling: a strict special case of Platt with the intercept fixed at zero, "
    "so the fitted Platt intercept already answers what a temperature would have been. The "
    "parameters needed to make that call are persisted (ADR 0027).",
    "recalibrating per test quarter: the test windows are evaluation only. A calibrator "
    "refitted on a test quarter would make the reported probabilities self-fulfilling.",
    "ensembling the calibrated candidates: Component 9 calibrates models, it does not "
    "combine them.",
)


class CalibrationDefinitionError(ValueError):
    """Raised when the frozen specifications are internally inconsistent."""


def candidate_index(name: str) -> int:
    """This candidate's position in the frozen registry.

    Used as a bootstrap seed component. **Never ``hash(name)``**: Python salts string hashing
    per process (PYTHONHASHSEED), so a seed derived from it gives different resamples on every
    run -- which a byte-for-byte determinism check caught it doing. The registry position is
    stable across processes, and stable across runs that subset the candidates with
    ``--models``, because it indexes the registry rather than the selection.
    """
    return list(CANDIDATES_BY_NAME).index(spec_for(name).name)


def spec_for(name: str) -> CandidateSpec:
    """Look up a candidate by name, failing loudly on a typo."""
    try:
        return CANDIDATES_BY_NAME[name]
    except KeyError:
        known = ", ".join(sorted(CANDIDATES_BY_NAME))
        raise CalibrationDefinitionError(
            f"Unknown calibration candidate {name!r}; known: {known}"
        ) from None


def _guard_registry() -> None:
    """Assert the frozen specifications agree with themselves, at import time.

    Mirrors ``boosting.definitions._guard_registry``. These are the invariants a typo in a
    literal would break silently, and every one of them is load-bearing somewhere else in
    the package.
    """
    names = [c.name for c in CANDIDATE_REGISTRY]
    if len(set(names)) != len(names):
        raise CalibrationDefinitionError("duplicate candidate name in CANDIDATE_REGISTRY")

    overlap = set(names) & set(EXCLUDED_MODELS)
    if overlap:
        raise CalibrationDefinitionError(
            f"model(s) both calibrated and excluded: {', '.join(sorted(overlap))}"
        )
    if EMBEDDING_DONOR in names:
        raise CalibrationDefinitionError(
            f"{EMBEDDING_DONOR} is fitted as a donor only and must not be a candidate"
        )

    for candidate in CANDIDATE_REGISTRY:
        if not candidate.rationale:
            raise CalibrationDefinitionError(
                f"{candidate.name}: every candidate must record why it is in the set"
            )
        for method in Method:
            if candidate.calibrated_name(method) in CANDIDATES_BY_NAME:
                raise CalibrationDefinitionError(
                    f"{candidate.name}: calibrated name collides with a base model name, so "
                    "a calibrated row could be applied to a calibrator a second time"
                )

    # The tie rule must be able to express a preference. A zero threshold would make the
    # preference unreachable, and a negative one would invert the rule.
    if TIE_THRESHOLD <= 0.0:
        raise CalibrationDefinitionError("TIE_THRESHOLD must be positive; see ADR 0025")
    if not 0.0 < INNER_SELECT_FRACTION < 1.0:
        raise CalibrationDefinitionError("INNER_SELECT_FRACTION must lie strictly in (0, 1)")
    if MIN_INNER_FIT_ROWS < 1 or MIN_INNER_SELECT_ROWS < 1:
        raise CalibrationDefinitionError("inner-split minimums must be positive")
    if set(INPUT_TRANSFORM) != set(Method):
        raise CalibrationDefinitionError("every method must declare its input transform")
    if STAGE_UNCALIBRATED in {m.value for m in Method}:
        raise CalibrationDefinitionError("the uncalibrated stage must not name a method")


_guard_registry()


__all__ = [
    "AVAILABLE_FROM_SEMANTICS",
    "BLOCKED_EXPERIMENTS",
    "BOOTSTRAP_CAVEAT",
    "BOOTSTRAP_REPLICATIONS",
    "BOOTSTRAP_SCHEMES",
    "BOOTSTRAP_SCHEME_BLOCK",
    "BOOTSTRAP_SCHEME_ROW",
    "BOOTSTRAP_SEED",
    "CALIBRATION_DEFINITION_VERSION",
    "CANDIDATES_BY_NAME",
    "CANDIDATE_REGISTRY",
    "CI_LEVEL",
    "EMBEDDING_DONOR",
    "EXCLUDED_MODELS",
    "INNER_SELECT_FRACTION",
    "INPUT_TRANSFORM",
    "ISOTONIC_PARAMS",
    "LOGIT_EPSILON",
    "MARGIN_TOLERANCE",
    "MAX_DEGENERATE_SHARE",
    "MIN_INNER_FIT_ROWS",
    "MIN_INNER_SELECT_ROWS",
    "PLATT_PARAMS",
    "PLATT_SELF_CHECK_TOLERANCE",
    "PROBABILITY_SEMANTICS",
    "SCORE_DIRECTION",
    "SELECTION_GRANULARITY",
    "SELECTION_METRIC",
    "SELECTION_RULE",
    "STAGES",
    "STAGE_SELECTED",
    "STAGE_UNCALIBRATED",
    "TIE_PREFERENCE",
    "TIE_THRESHOLD",
    "TRAINED_THROUGH_SEMANTICS",
    "CalibrationDefinitionError",
    "CandidateSpec",
    "candidate_index",
    "Family",
    "Method",
    "spec_for",
]
