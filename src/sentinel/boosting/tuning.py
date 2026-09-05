"""Temporally valid hyperparameter search. The part of Component 7 that can go wrong
silently, so it is the part with the most structure.

Component 6 had no hyperparameters and therefore needed no protocol. A booster has a
dozen, and selecting them opens a leak that no artifact records and no check downstream
can detect: fit a model, read a test metric, change a parameter, refit. Each individual
step is legitimate; the loop is leakage, and it leaves the model looking better than it
is with nothing in the repository to show why. Component 5 protects evaluation time, but
it cannot protect against a human.

The protocol makes the leak structurally impossible rather than merely discouraged.

**One region per fold set, strictly earlier than that fold set's first test window.**
The region is the *first* fold's ``train_start .. calibration_end``. Because quarterly
folds expand from a fixed anchor, that span is a subset of every later fold's own
train-plus-calibration, so no fold is ever scored by parameters chosen on its own test
data -- or on any other fold's.

**Two regions, not one.** ``covid_shift`` tests on 2020-06-01..2021-12-31, which sits
*inside* the quarterly region. A single shared study would therefore have selected
parameters using the shift fold's own test labels, making Component 7's shift number
optimistically biased and not comparable with Component 6's clean one. Since the
reversal of model ordering under shift is the most consequential finding this project
has, biasing it would be the worst available trade. So each fold set gets its own study.
See ADR 0017.

**Inner folds mirror the outer structure, gap included.** Each inner fold is a real
``FoldSpec``: train, then an unused calibration quarter, then a validation quarter. The
gap is not decoration -- every outer fold has one, so tuning without it would select
parameters for a zero-gap regime and apply them to a one-gap regime.

**Early stopping lives here and only here.** The number of boosting rounds is chosen by
early stopping against an inner *validation* quarter, which is training data for every
outer fold. The winning trial's round count is then frozen, and ``train.fit_fold`` runs
exactly that many rounds with no eval set at all. That is what lets a final fit declare
``trained_through = fold.train_end`` truthfully.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from sentinel.boosting import preprocess
from sentinel.boosting.definitions import (
    EARLY_STOPPING_ROUNDS,
    FIXED_PARAMS,
    MAX_BOOSTING_ROUNDS,
    TUNING_SEED,
    BoostingSpec,
    Estimator,
    search_space,
)
from sentinel.boosting.models import InnerFoldScore, StudyResult, TrialResult
from sentinel.boosting.train import BoostingTrainError, build_estimator, fold_labels
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation import metrics
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling.train import training_frame

logger = logging.getLogger(__name__)

#: Training quarters before the first inner validation window, per fold set. Chosen so
#: each region yields at least two inner folds while leaving a training window large
#: enough to fit a tree ensemble on. ``covid_shift``'s region is only eight quarters
#: long, so its value is necessarily smaller and its fold count necessarily thinner --
#: reported, never padded.
MIN_INNER_TRAIN_QUARTERS: dict[str, int] = {
    folds_module.QUARTERLY: 8,
    folds_module.COVID_SHIFT: 4,
}

#: Below this, a study is refused rather than run. A single inner fold would make the
#: objective one number from one quarter, which is a preference rather than a measurement.
MIN_INNER_FOLDS = 2

OBJECTIVE = (
    "mean PR-AUC across temporally ordered inner validation windows, each strictly "
    "later than its own inner training window and all strictly earlier than the fold "
    "set's first test window"
)

SAMPLER = "optuna.samplers.TPESampler"


class TuningError(RuntimeError):
    """Raised when a search cannot be run, or cannot be run honestly."""


# --- 1. the region and the inner folds ---------------------------------------


def tuning_region(fold_set: str, outer_folds: Sequence[FoldSpec]) -> tuple[date, date]:
    """The span a study for ``fold_set`` may read, derived from the folds themselves.

    Deliberately computed from the fold set rather than written down as a date literal.
    A hardcoded region would silently stop being correct the moment the anchor, the
    cadence or the minimum training length changed, and the failure would be invisible.
    """
    members = [f for f in outer_folds if f.fold_set == fold_set]
    if not members:
        raise TuningError(f"no folds in fold set {fold_set!r}; a region cannot be derived")
    first = min(members, key=lambda f: (f.test_start, f.fold_id))
    return first.train_start, first.calibration_end


def first_test_start(fold_set: str, outer_folds: Sequence[FoldSpec]) -> date:
    """The earliest date any model in ``fold_set`` will be scored on."""
    members = [f for f in outer_folds if f.fold_set == fold_set]
    if not members:
        raise TuningError(f"no folds in fold set {fold_set!r}")
    return min(f.test_start for f in members)


def inner_folds(
    *,
    fold_set: str,
    region_start: date,
    region_end: date,
    min_train_quarters: int,
) -> list[FoldSpec]:
    """Rolling-origin folds carved inside one tuning region.

    Built with ``evaluation.folds``'s own quarter arithmetic, so "a quarter" means the
    same thing here as everywhere else in the project. The validation window is the
    ``FoldSpec``'s *test* window -- these are genuine folds, and reusing the type means
    ``assign_split`` and ``window_frame`` work on them unchanged rather than needing a
    parallel implementation that could disagree.
    """
    if region_start > region_end:
        raise TuningError(f"tuning region {region_start}..{region_end} is empty")

    anchor = folds_module.quarter_start(region_start)
    calibration = folds_module.add_quarters(anchor, min_train_quarters)
    built: list[FoldSpec] = []
    while True:
        validation = folds_module.add_quarters(calibration, 1)
        validation_end = folds_module.quarter_end(validation)
        if validation_end > region_end:
            break
        built.append(
            FoldSpec(
                fold_set=f"tuning-{fold_set}",
                fold_id=f"tuning-{fold_set}-{folds_module.quarter_key(validation)}",
                train_start=region_start,
                train_end=calibration - timedelta(days=1),
                calibration_start=calibration,
                calibration_end=folds_module.quarter_end(calibration),
                test_start=validation,
                test_end=validation_end,
            )
        )
        calibration = folds_module.add_quarters(calibration, 1)
    return built


def build_inner_folds(fold_set: str, outer_folds: Sequence[FoldSpec]) -> list[FoldSpec]:
    """The inner folds for one fold set, with the safety property re-derived, not assumed.

    The last line is the whole protocol in one assertion: every inner window ends before
    the fold set's first test window begins. It is checked here, against dates computed
    from the data, rather than recorded in a manifest and trusted.
    """
    region_start, region_end = tuning_region(fold_set, outer_folds)
    horizon = first_test_start(fold_set, outer_folds)
    if region_end >= horizon:
        raise TuningError(
            f"{fold_set}: tuning region ends {region_end} but the first test window "
            f"starts {horizon}. A study over this region would select hyperparameters "
            "using test labels."
        )
    minimum = MIN_INNER_TRAIN_QUARTERS.get(fold_set)
    if minimum is None:
        raise TuningError(
            f"{fold_set}: no inner training length declared. A default would be an "
            f"undocumented design choice; add an entry to MIN_INNER_TRAIN_QUARTERS."
        )
    built = inner_folds(
        fold_set=fold_set,
        region_start=region_start,
        region_end=region_end,
        min_train_quarters=minimum,
    )
    if len(built) < MIN_INNER_FOLDS:
        raise TuningError(
            f"{fold_set}: region {region_start}..{region_end} yields {len(built)} inner "
            f"fold(s), fewer than the {MIN_INNER_FOLDS} required. Folds are never "
            "fabricated and the region is never widened to reach a count."
        )
    latest = max(f.test_end for f in built)
    if latest >= horizon:
        raise TuningError(
            f"{fold_set}: inner validation reaches {latest}, on or after the first test "
            f"start {horizon}"
        )
    logger.info(
        "%s: %d inner fold(s) over %s..%s, all ending before %s",
        fold_set,
        len(built),
        region_start,
        region_end,
        horizon,
    )
    return built


# --- 2. one trial ------------------------------------------------------------


def suggest_params(spec: BoostingSpec, trial: Any) -> dict[str, object]:
    """Draw one point from the declared search space.

    Driven from ``SEARCH_SPACE`` rather than a hand-written block of ``suggest_*`` calls,
    so the space a study explored is the space the documentation and the tests read.

    The one conditional is LightGBM's ``num_leaves``, capped at ``2 ** max_depth``.
    LightGBM grows leaf-wise, so ``max_depth`` alone bounds nothing; without the cap the
    two libraries would explore different capacity ranges and any difference between
    them would be a fact about the search space rather than about the algorithm.
    """
    params: dict[str, object] = {}
    for dimension in search_space(spec.estimator):
        if dimension.kind == "int":
            params[dimension.name] = trial.suggest_int(
                dimension.name, int(dimension.low), int(dimension.high), log=dimension.log
            )
        else:
            params[dimension.name] = trial.suggest_float(
                dimension.name, dimension.low, dimension.high, log=dimension.log
            )

    if spec.estimator is Estimator.LIGHTGBM:
        depth = params.get("max_depth")
        leaves = params.get("num_leaves")
        if isinstance(depth, int) and isinstance(leaves, int):
            params["num_leaves"] = min(leaves, 2**depth)
        params["bagging_freq"] = 1
    return params


def trial_params(spec: BoostingSpec, drawn: dict[str, object]) -> dict[str, object]:
    """Fixed parameters, then the drawn point, then the seeds and the round cap."""
    params: dict[str, object] = dict(FIXED_PARAMS[spec.estimator])
    params.update(drawn)
    params["n_estimators"] = MAX_BOOSTING_ROUNDS
    params["random_state"] = spec.seed
    if spec.estimator is Estimator.LIGHTGBM:
        params["bagging_seed"] = spec.seed
        params["feature_fraction_seed"] = spec.seed
        params["data_random_seed"] = spec.seed
    return params


def _fit_early_stopped(
    spec: BoostingSpec,
    params: dict[str, object],
    train_matrix: NDArray[np.float64],
    train_labels: NDArray[np.int64],
    validation_matrix: NDArray[np.float64],
    validation_labels: NDArray[np.int64],
) -> tuple[Any, int]:
    """Fit with early stopping against an inner validation window. Returns the round count.

    This is the only place in Component 7 where an ``eval_set`` exists. The window it
    points at is inside the tuning region, so it is training data for every outer fold
    the resulting parameters will be used on.
    """
    if spec.estimator is Estimator.XGBOOST:
        estimator = build_estimator(
            spec, {**params, "early_stopping_rounds": EARLY_STOPPING_ROUNDS}
        )
        estimator.fit(
            train_matrix,
            train_labels,
            eval_set=[(validation_matrix, validation_labels)],
            verbose=False,
        )
        best = getattr(estimator, "best_iteration", None)
        rounds = int(best) + 1 if best is not None else MAX_BOOSTING_ROUNDS
        return estimator, rounds

    import lightgbm as lgb

    estimator = build_estimator(spec, params)
    estimator.fit(
        train_matrix,
        train_labels,
        # ``eval_X``/``eval_y`` rather than the older ``eval_set``, which LightGBM 4.7
        # deprecates. Named separately here because the two libraries disagree about the
        # shape of this argument and hiding that behind a shared helper would make the
        # next version bump silently pass the wrong thing.
        eval_X=validation_matrix,
        eval_y=validation_labels,
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    best = getattr(estimator, "best_iteration_", None)
    rounds = int(best) if best else MAX_BOOSTING_ROUNDS
    return estimator, rounds


def score_inner_fold(
    spec: BoostingSpec,
    frame: pl.DataFrame,
    fold: FoldSpec,
    params: dict[str, object],
) -> InnerFoldScore:
    """Fit on one inner training window and score its validation window.

    PR-AUC comes from ``evaluation.metrics.pr_auc``. Component 5 owns that definition
    and Component 7 does not get a second one: two implementations of average precision
    would eventually disagree, and the one used to *select* a model disagreeing with the
    one used to *report* it is the subtlest way to make a tuning result meaningless.
    """
    train = training_frame(frame, fold)
    validation = folds_module.window_frame(frame, fold)
    if train.height == 0 or validation.height == 0:
        raise TuningError(
            f"{spec.name}: inner fold {fold.fold_id} has {train.height} training and "
            f"{validation.height} validation rows; a fold with an empty window is a "
            "defect rather than something to score around."
        )

    train_labels = fold_labels(train, spec.name, fold.fold_id)
    validation_labels = fold_labels(validation, spec.name, fold.fold_id)
    if len(set(train_labels.tolist())) < 2:
        raise TuningError(f"{spec.name}: inner fold {fold.fold_id} training window is one class")

    estimator, rounds = _fit_early_stopped(
        spec,
        params,
        preprocess.tree_matrix(train, spec),
        train_labels,
        preprocess.tree_matrix(validation, spec),
        validation_labels,
    )
    proba = estimator.predict_proba(preprocess.tree_matrix(validation, spec))
    scores = [float(v) for v in proba[:, 1]]
    value = metrics.pr_auc(validation_labels.tolist(), scores)
    if value is None:
        raise TuningError(
            f"{spec.name}: inner fold {fold.fold_id} validation window is one class, so "
            "PR-AUC is undefined"
        )
    return InnerFoldScore(
        fold_id=fold.fold_id,
        train_rows=train.height,
        validation_rows=validation.height,
        pr_auc=value,
        best_iteration=rounds,
    )


# --- 3. the study ------------------------------------------------------------


def run_study(
    spec: BoostingSpec,
    frame: pl.DataFrame,
    outer_folds: Sequence[FoldSpec],
    *,
    fold_set: str,
    trials: int,
    seed: int = TUNING_SEED,
) -> StudyResult:
    """Search for ``spec``'s hyperparameters on one fold set. Never touches a test window.

    The sampler is seeded, so a re-run with the same trial count explores the same
    sequence of candidates. That is what makes "these are the best parameters" a
    reproducible claim rather than an anecdote about one afternoon.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if trials < 1:
        raise TuningError(f"{spec.name}: a study needs at least one trial, got {trials}")

    started = datetime.now(UTC)
    region_start, region_end = tuning_region(fold_set, outer_folds)
    inner = build_inner_folds(fold_set, outer_folds)
    study_name = f"{spec.name}-{fold_set}"
    collected: list[TrialResult] = []

    def objective(trial: Any) -> float:
        trial_started = datetime.now(UTC)
        drawn = suggest_params(spec, trial)
        params = trial_params(spec, drawn)
        try:
            scores = tuple(score_inner_fold(spec, frame, fold, params) for fold in inner)
        except (TuningError, BoostingTrainError) as exc:
            elapsed = (datetime.now(UTC) - trial_started).total_seconds()
            collected.append(
                TrialResult(
                    study=study_name,
                    model_name=spec.name,
                    fold_set=fold_set,
                    number=trial.number,
                    params=dict(drawn),
                    inner_scores=(),
                    mean_pr_auc=float("nan"),
                    n_estimators=0,
                    seconds=round(elapsed, 3),
                    failed=True,
                    failure=str(exc),
                )
            )
            # Pruned, not crashed: one unfittable corner of the space should not end a
            # 100-trial search, but it must be visible in the trials table rather than
            # scored as if it had succeeded.
            raise optuna.TrialPruned() from exc

        mean = float(np.mean([s.pr_auc for s in scores]))
        rounds = max(1, int(round(float(np.mean([s.best_iteration for s in scores])))))
        elapsed = (datetime.now(UTC) - trial_started).total_seconds()
        collected.append(
            TrialResult(
                study=study_name,
                model_name=spec.name,
                fold_set=fold_set,
                number=trial.number,
                params=dict(drawn),
                inner_scores=scores,
                mean_pr_auc=mean,
                n_estimators=rounds,
                seconds=round(elapsed, 3),
            )
        )
        return mean

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=trials)

    succeeded = [t for t in collected if not t.failed]
    if not succeeded:
        raise TuningError(
            f"{study_name}: every one of {trials} trial(s) failed. No parameters are "
            "frozen from a study that never produced a score."
        )
    best = max(succeeded, key=lambda t: (t.mean_pr_auc, -t.number))
    elapsed = (datetime.now(UTC) - started).total_seconds()
    logger.info(
        "%s: %d/%d trial(s) scored, best mean PR-AUC %.4f at trial %d (%d rounds) in %.1fs",
        study_name,
        len(succeeded),
        trials,
        best.mean_pr_auc,
        best.number,
        best.n_estimators,
        elapsed,
    )
    return StudyResult(
        study=study_name,
        model_name=spec.name,
        fold_set=fold_set,
        region_start=region_start,
        region_end=region_end,
        inner_folds=tuple(f.fold_id for f in inner),
        trials=tuple(sorted(collected, key=lambda t: t.number)),
        best=best,
        sampler_seed=seed,
        seconds=round(elapsed, 3),
    )


def frozen_params(result: StudyResult) -> dict[str, object]:
    """The winning trial rendered as a ``TUNED_PARAMS`` entry, round count included.

    Emitted so the tuning command can print exactly what should be pasted into
    ``definitions.TUNED_PARAMS``. Freezing by hand into a literal under version control
    is deliberate: a parameter set read from a file at training time could change
    without a diff, and the whole point of freezing is that it cannot.
    """
    params = dict(result.best.params)
    params["n_estimators"] = result.best.n_estimators
    return params


__all__ = [
    "MIN_INNER_FOLDS",
    "MIN_INNER_TRAIN_QUARTERS",
    "OBJECTIVE",
    "SAMPLER",
    "TuningError",
    "build_inner_folds",
    "first_test_start",
    "frozen_params",
    "inner_folds",
    "run_study",
    "score_inner_fold",
    "suggest_params",
    "trial_params",
    "tuning_region",
]
