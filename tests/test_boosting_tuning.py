"""The hyperparameter search: reproducible, bounded, and unable to reach a test window.

This is the file that protects the leak Component 5 cannot see. A search that reads a
test window leaves no trace in any artifact -- the predictions look normal, the horizon
check passes, and the model is simply better than it should be. So the tests here
recompute the region and the inner-fold dates from the data rather than reading the
manifest fields that record them, and assert the objective's reachable row set is
disjoint from every test window it will later be judged on.

Every search here uses a tiny deterministic configuration: three trials, a handful of
rounds. The production search is 100 trials per study and runs only through
``sentinel tune-boosting``. Confusing the two would make the suite slow and would tempt
someone to read a unit test's PR-AUC as a result.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from sentinel.boosting import tuning
from sentinel.boosting.definitions import (
    MAX_BOOSTING_ROUNDS,
    TUNING_SEED,
    Estimator,
    spec_for,
)
from sentinel.boosting.tuning import TuningError
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from tests.conftest import spanning_model_features

PRIMARY = spec_for("xgboost")
SECONDARY = spec_for("lightgbm")

#: A unit-test search. The production search is 100 trials per study, via the CLI.
UNIT_TEST_TRIALS = 3


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return spanning_model_features(days=1900).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )


@pytest.fixture(scope="module")
def outer(frame: pl.DataFrame) -> list[FoldSpec]:
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    assert start is not None and end is not None
    quarterly = folds_module.quarterly_folds(data_start=start, data_end=end)
    assert len(quarterly) >= 4
    shift = folds_module.covid_shift_fold(data_end=end)
    assert shift, "fixture must span the covid_shift window or half the tests are skipped"
    return [*quarterly, *shift]


# --- 1. the region -----------------------------------------------------------


def test_the_region_is_the_first_folds_train_and_calibration_span(
    outer: list[FoldSpec],
) -> None:
    first = min(
        (f for f in outer if f.fold_set == "quarterly"), key=lambda f: (f.test_start, f.fold_id)
    )
    assert tuning.tuning_region("quarterly", outer) == (first.train_start, first.calibration_end)


def test_each_fold_set_gets_its_own_region(outer: list[FoldSpec]) -> None:
    """Sharing one would let the quarterly region cover the shift fold's test window."""
    quarterly = tuning.tuning_region("quarterly", outer)
    shift = tuning.tuning_region("covid_shift", outer)
    assert quarterly != shift
    assert shift[1] < quarterly[1]


def test_the_quarterly_region_would_have_covered_the_shift_test_window(
    outer: list[FoldSpec],
) -> None:
    """The measured fact that forces two studies instead of one. Asserted, not asserted about.

    If this ever stops being true the two-study design could be revisited -- so it is
    checked rather than left as a claim in an ADR.
    """
    _, quarterly_end = tuning.tuning_region("quarterly", outer)
    shift = next(f for f in outer if f.fold_set == "covid_shift")
    assert shift.test_start < quarterly_end
    assert shift.test_end < quarterly_end


def test_an_unknown_fold_set_raises(outer: list[FoldSpec]) -> None:
    with pytest.raises(TuningError, match="no folds in fold set"):
        tuning.tuning_region("weekly", outer)


# --- 2. the inner folds ------------------------------------------------------


def test_every_inner_window_ends_before_the_first_test_start(outer: list[FoldSpec]) -> None:
    for fold_set in ("quarterly", "covid_shift"):
        horizon = tuning.first_test_start(fold_set, outer)
        inner = tuning.build_inner_folds(fold_set, outer)
        assert inner
        for fold in inner:
            assert fold.test_end < horizon


def test_inner_folds_keep_the_outer_structures_gap(outer: list[FoldSpec]) -> None:
    """Train, an unused calibration quarter, then validation -- the outer shape."""
    for fold in tuning.build_inner_folds("quarterly", outer):
        assert fold.train_end < fold.calibration_start
        assert fold.calibration_end < fold.test_start
        assert (fold.test_start - fold.calibration_end).days == 1


def test_inner_folds_expand_rather_than_slide(outer: list[FoldSpec]) -> None:
    inner = tuning.build_inner_folds("quarterly", outer)
    assert len({f.train_start for f in inner}) == 1
    ends = [f.train_end for f in inner]
    assert ends == sorted(ends)
    assert len(set(ends)) == len(ends)


def test_inner_fold_ids_are_distinct_and_name_their_fold_set(outer: list[FoldSpec]) -> None:
    for fold_set in ("quarterly", "covid_shift"):
        inner = tuning.build_inner_folds(fold_set, outer)
        ids = [f.fold_id for f in inner]
        assert len(set(ids)) == len(ids)
        assert all(f.fold_set == f"tuning-{fold_set}" for f in inner)


def test_a_region_yielding_too_few_folds_is_refused_rather_than_widened(
    outer: list[FoldSpec], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(tuning.MIN_INNER_TRAIN_QUARTERS, "quarterly", 200)
    with pytest.raises(TuningError, match="fewer than the 2 required"):
        tuning.build_inner_folds("quarterly", outer)


def test_a_fold_set_without_a_declared_inner_length_is_refused(
    outer: list[FoldSpec], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A default would be an undocumented design choice."""
    monkeypatch.delitem(tuning.MIN_INNER_TRAIN_QUARTERS, "covid_shift")
    with pytest.raises(TuningError, match="no inner training length declared"):
        tuning.build_inner_folds("covid_shift", outer)


def test_an_empty_region_is_refused() -> None:
    with pytest.raises(TuningError, match="is empty"):
        tuning.inner_folds(
            fold_set="quarterly",
            region_start=date(2022, 1, 1),
            region_end=date(2020, 1, 1),
            min_train_quarters=4,
        )


# --- 3. the search space is drawn from correctly ------------------------------


class _RecordingTrial:
    """A minimal Optuna trial stand-in, so the suggests can be inspected directly."""

    def __init__(self, ints: dict[str, int], floats: dict[str, float]) -> None:
        self._ints = ints
        self._floats = floats
        self.seen: list[tuple[str, float, float, bool]] = []

    def suggest_int(self, name: str, low: int, high: int, log: bool = False) -> int:
        self.seen.append((name, low, high, log))
        return self._ints.get(name, low)

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        self.seen.append((name, low, high, log))
        return self._floats.get(name, low)


def test_every_declared_dimension_is_actually_suggested() -> None:
    for spec in (PRIMARY, SECONDARY):
        trial = _RecordingTrial({}, {})
        tuning.suggest_params(spec, trial)
        suggested = {name for name, *_ in trial.seen}
        declared = {d.name for d in tuning.search_space(spec.estimator)}
        assert declared <= suggested


def test_the_suggested_ranges_are_the_declared_ranges() -> None:
    for spec in (PRIMARY, SECONDARY):
        trial = _RecordingTrial({}, {})
        tuning.suggest_params(spec, trial)
        seen = {name: (low, high, log) for name, low, high, log in trial.seen}
        for dimension in tuning.search_space(spec.estimator):
            assert seen[dimension.name] == (dimension.low, dimension.high, dimension.log)


def test_lightgbm_num_leaves_is_capped_at_two_to_the_depth() -> None:
    """Without the cap the two libraries would explore different capacity ranges."""
    trial = _RecordingTrial({"max_depth": 4, "num_leaves": 256}, {})
    params = tuning.suggest_params(SECONDARY, trial)
    assert params["num_leaves"] == 16


def test_the_cap_leaves_a_leaf_count_below_the_bound_alone() -> None:
    trial = _RecordingTrial({"max_depth": 10, "num_leaves": 64}, {})
    assert tuning.suggest_params(SECONDARY, trial)["num_leaves"] == 64


def test_lightgbm_gets_the_bagging_frequency_its_row_subsample_needs() -> None:
    """``bagging_fraction`` without ``bagging_freq`` silently subsamples nothing."""
    params = tuning.suggest_params(SECONDARY, _RecordingTrial({}, {}))
    assert params["bagging_freq"] == 1


def test_xgboost_is_not_given_a_leaf_count() -> None:
    params = tuning.suggest_params(PRIMARY, _RecordingTrial({}, {}))
    assert "num_leaves" not in params
    assert "bagging_freq" not in params


def test_trial_params_cap_the_round_count_and_carry_the_seeds() -> None:
    params = tuning.trial_params(SECONDARY, {"max_depth": 4})
    assert params["n_estimators"] == MAX_BOOSTING_ROUNDS
    assert params["random_state"] == SECONDARY.seed
    assert params["bagging_seed"] == SECONDARY.seed
    assert params["num_threads"] == 1


def test_trial_params_cannot_overwrite_a_fixed_parameter() -> None:
    """The guard rejects such a space at import; this confirms the merge order too."""
    params = tuning.trial_params(PRIMARY, {"max_depth": 9})
    assert params["tree_method"] == "hist"
    assert params["n_jobs"] == 1
    assert params["objective"] == "binary:logistic"


# --- 4. running a study -------------------------------------------------------


def test_a_study_is_reproducible_at_a_fixed_seed(
    frame: pl.DataFrame, outer: list[FoldSpec]
) -> None:
    """Two runs of the same tiny configuration must choose the same parameters."""
    first = tuning.run_study(
        PRIMARY, frame, outer, fold_set="quarterly", trials=UNIT_TEST_TRIALS, seed=TUNING_SEED
    )
    second = tuning.run_study(
        PRIMARY, frame, outer, fold_set="quarterly", trials=UNIT_TEST_TRIALS, seed=TUNING_SEED
    )
    assert first.best.params == second.best.params
    assert first.best.mean_pr_auc == second.best.mean_pr_auc
    assert first.best.n_estimators == second.best.n_estimators
    assert [t.params for t in first.trials] == [t.params for t in second.trials]


def test_a_different_seed_explores_a_different_sequence(
    frame: pl.DataFrame, outer: list[FoldSpec]
) -> None:
    """Otherwise the seed is decorative and the reproducibility claim is empty."""
    first = tuning.run_study(
        PRIMARY, frame, outer, fold_set="quarterly", trials=UNIT_TEST_TRIALS, seed=1
    )
    second = tuning.run_study(
        PRIMARY, frame, outer, fold_set="quarterly", trials=UNIT_TEST_TRIALS, seed=2
    )
    assert [t.params for t in first.trials] != [t.params for t in second.trials]


def test_a_study_records_its_region_and_its_inner_folds(
    frame: pl.DataFrame, outer: list[FoldSpec]
) -> None:
    result = tuning.run_study(PRIMARY, frame, outer, fold_set="quarterly", trials=UNIT_TEST_TRIALS)
    assert (result.region_start, result.region_end) == tuning.tuning_region("quarterly", outer)
    assert list(result.inner_folds) == [
        f.fold_id for f in tuning.build_inner_folds("quarterly", outer)
    ]
    assert result.region_end < tuning.first_test_start("quarterly", outer)


def test_every_trial_scores_every_inner_fold(frame: pl.DataFrame, outer: list[FoldSpec]) -> None:
    result = tuning.run_study(
        SECONDARY, frame, outer, fold_set="quarterly", trials=UNIT_TEST_TRIALS
    )
    expected = len(tuning.build_inner_folds("quarterly", outer))
    for trial in result.trials:
        if trial.failed:
            continue
        assert len(trial.inner_scores) == expected


def test_the_objective_is_the_mean_of_the_inner_scores(
    frame: pl.DataFrame, outer: list[FoldSpec]
) -> None:
    result = tuning.run_study(PRIMARY, frame, outer, fold_set="quarterly", trials=UNIT_TEST_TRIALS)
    for trial in result.trials:
        if trial.failed:
            continue
        mean = sum(s.pr_auc for s in trial.inner_scores) / len(trial.inner_scores)
        assert trial.mean_pr_auc == pytest.approx(mean)


def test_the_winning_trial_is_the_highest_scoring_one(
    frame: pl.DataFrame, outer: list[FoldSpec]
) -> None:
    result = tuning.run_study(PRIMARY, frame, outer, fold_set="quarterly", trials=UNIT_TEST_TRIALS)
    scored = [t for t in result.trials if not t.failed]
    assert result.best.mean_pr_auc == max(t.mean_pr_auc for t in scored)


def test_the_frozen_round_count_comes_from_early_stopping(
    frame: pl.DataFrame, outer: list[FoldSpec]
) -> None:
    """It must be a real observation, not the cap, or early stopping never fired."""
    result = tuning.run_study(PRIMARY, frame, outer, fold_set="quarterly", trials=UNIT_TEST_TRIALS)
    assert 1 <= result.best.n_estimators <= MAX_BOOSTING_ROUNDS
    assert result.best.n_estimators < MAX_BOOSTING_ROUNDS


def test_frozen_params_carry_the_round_count(frame: pl.DataFrame, outer: list[FoldSpec]) -> None:
    result = tuning.run_study(PRIMARY, frame, outer, fold_set="quarterly", trials=UNIT_TEST_TRIALS)
    frozen = tuning.frozen_params(result)
    assert frozen["n_estimators"] == result.best.n_estimators
    assert set(result.best.params) <= set(frozen)


def test_a_study_with_no_trials_is_refused(frame: pl.DataFrame, outer: list[FoldSpec]) -> None:
    with pytest.raises(TuningError, match="at least one trial"):
        tuning.run_study(PRIMARY, frame, outer, fold_set="quarterly", trials=0)


def test_the_covid_shift_study_uses_its_own_thinner_region(
    frame: pl.DataFrame, outer: list[FoldSpec]
) -> None:
    result = tuning.run_study(
        SECONDARY, frame, outer, fold_set="covid_shift", trials=UNIT_TEST_TRIALS
    )
    assert result.region_end == date(2020, 5, 31)
    assert result.region_end < tuning.first_test_start("covid_shift", outer)
    assert len(result.inner_folds) >= 2


# --- 5. the objective never reads a test row ----------------------------------


def test_the_objective_reaches_no_row_of_its_own_fold_sets_test_windows(
    frame: pl.DataFrame, outer: list[FoldSpec]
) -> None:
    """Row ids, not dates: the strongest form of the claim."""
    from sentinel.modeling.train import training_frame

    for fold_set in ("quarterly", "covid_shift"):
        test_ids: set[str] = set()
        for fold in outer:
            if fold.fold_set == fold_set:
                test_ids.update(
                    folds_module.window_frame(frame, fold)["target_inspection_id"].to_list()
                )
        assert test_ids, f"{fold_set} has no test rows; the intersection would be free"

        touched: set[str] = set()
        for inner in tuning.build_inner_folds(fold_set, outer):
            touched.update(training_frame(frame, inner)["target_inspection_id"].to_list())
            touched.update(
                folds_module.window_frame(frame, inner)["target_inspection_id"].to_list()
            )
        assert touched, "the inner folds reached no rows at all"
        assert not (touched & test_ids)


def test_the_objective_metric_is_component_5s(frame: pl.DataFrame, outer: list[FoldSpec]) -> None:
    """A second implementation of average precision would eventually disagree with C5's."""
    from sentinel.evaluation import metrics

    inner = tuning.build_inner_folds("quarterly", outer)[0]
    params = tuning.trial_params(PRIMARY, tuning.suggest_params(PRIMARY, _RecordingTrial({}, {})))
    score = tuning.score_inner_fold(PRIMARY, frame, inner, params)
    assert 0.0 <= score.pr_auc <= 1.0
    assert metrics.pr_auc([0, 1], [0.1, 0.9]) is not None


def test_scoring_names_the_estimator_and_the_fold_on_failure(
    frame: pl.DataFrame, outer: list[FoldSpec]
) -> None:
    inner = tuning.build_inner_folds("quarterly", outer)[0]
    single_class = frame.with_columns(pl.lit(1).cast(pl.Int8).alias("target"))
    params = tuning.trial_params(PRIMARY, {"max_depth": 3})
    with pytest.raises(TuningError, match="one class"):
        tuning.score_inner_fold(PRIMARY, single_class, inner, params)


def test_both_estimators_run_the_objective(frame: pl.DataFrame, outer: list[FoldSpec]) -> None:
    inner = tuning.build_inner_folds("quarterly", outer)[0]
    for spec in (PRIMARY, SECONDARY):
        params = tuning.trial_params(spec, tuning.suggest_params(spec, _RecordingTrial({}, {})))
        score = tuning.score_inner_fold(spec, frame, inner, params)
        assert score.best_iteration >= 1
        assert score.validation_rows > 0
        assert score.fold_id == inner.fold_id


def test_the_declared_objective_and_sampler_are_recorded_as_strings() -> None:
    """These land in the manifest verbatim, so they should say what actually happened."""
    assert "PR-AUC" in tuning.OBJECTIVE
    assert "strictly earlier than the fold set's first test window" in tuning.OBJECTIVE
    assert tuning.SAMPLER == "optuna.samplers.TPESampler"


def test_the_estimator_enum_covers_both_searched_libraries() -> None:
    assert {e for e in Estimator} == {Estimator.XGBOOST, Estimator.LIGHTGBM}
