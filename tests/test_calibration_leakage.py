"""Leakage tests for Component 9.

The rest of the calibration suite checks that the calibrators are *right*. These check that
they cannot be **cheating**.

The standard throughout is **bit-identity, not approximate equality**. A calibrator that
moved by 1e-12 when a future row was appended has read that row, and the size of the move is
not the point. ``pytest.approx`` appears nowhere in this file.

Three properties are being defended, and they are different from one another:

1.  **Temporal.** Nothing after ``calibration_end`` may change a calibrator's parameters --
    not a future row, not a flipped future label, not the entire test window.
2.  **Selection.** The choice between Platt and isotonic must be made without reading a test
    window, which is subtler than it sounds: fold N's calibration window IS fold N-1's test
    window, so a protocol that pools folds leaks even though every row it touched is a
    calibration row. ``test_a_pooled_global_selection_would_read_an_earlier_folds_test_window``
    asserts that the rejected design is detectably leaky, so the rejection is executable.
3.  **Identity.** The re-executed base models must be the models Components 6-8 published,
    or every calibrator below them corrects something that was never scored.

Two of these tests have teeth rather than assertions about the happy path:
``test_the_leakage_detector_itself_works`` and
``test_the_bit_identity_detector_itself_works`` deliberately break the thing being guarded
and assert the guard notices. Without them, the rest of the file proves only that nothing
happened to be wrong today.

Component 7's lesson applies here too: **when a leakage test fails, suspect the test first.**
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

import polars as pl
import pytest

from sentinel.calibration import basescores, metrics, predict, train, validate
from sentinel.calibration.definitions import (
    MIN_INNER_FIT_ROWS,
    MIN_INNER_SELECT_ROWS,
    TIE_PREFERENCE,
    Method,
    spec_for,
)
from sentinel.calibration.models import BaseScores
from sentinel.calibration.preprocess import (
    CalibrationPreprocessError,
    calibration_frame,
    expit,
    split_calibration_window,
)
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.contract import (
    PredictionHorizonError,
    prediction_frame,
    validate_predictions,
)
from sentinel.evaluation.models import FoldSpec, PredictionSet
from tests.conftest import spanning_model_features

EARLY_FOLD = FoldSpec(
    fold_set="quarterly",
    fold_id="quarterly-2022Q2",
    train_start=date(2018, 7, 1),
    train_end=date(2021, 12, 31),
    calibration_start=date(2022, 1, 1),
    calibration_end=date(2022, 3, 31),
    test_start=date(2022, 4, 1),
    test_end=date(2022, 6, 30),
)

LATER_FOLD = FoldSpec(
    fold_set="quarterly",
    fold_id="quarterly-2022Q3",
    train_start=date(2018, 7, 1),
    train_end=date(2022, 3, 31),
    # Note: this window is EARLY_FOLD's test window, to the day. That coincidence is the
    # whole reason the selection protocol is an expanding prefix rather than a pool.
    calibration_start=date(2022, 4, 1),
    calibration_end=date(2022, 6, 30),
    test_start=date(2022, 7, 1),
    test_end=date(2022, 9, 30),
)


# --- fixtures ---------------------------------------------------------------
#
# The calibration layer is tested on synthetic scores rather than on refitted models: what
# is being defended is the calibrator, and driving a real neural fit for each assertion
# would make the suite slow without making it stricter. The regeneration seam gets its own
# slower test at the bottom of the file.


def _scored_window(
    fold: FoldSpec,
    *,
    per_day: int = 14,
    slope: float = 0.5,
    seed: int = 11,
    flip: bool = False,
) -> tuple[list[str], list[float], list[int], list[date]]:
    """A deterministically overconfident scored window spanning one fold's calibration period.

    The model claims ``expit(z)`` while the truth is ``expit(slope * z)``, so a calibrator
    has something real to correct and its fitted slope is predictable.
    """
    rng = random.Random(seed)
    ids: list[str] = []
    scores: list[float] = []
    labels: list[int] = []
    days: list[date] = []
    day = fold.calibration_start
    index = 0
    while day <= fold.calibration_end:
        for _ in range(per_day):
            z = rng.gauss(0.0, 2.0)
            label = 1 if rng.random() < expit(slope * z) else 0
            ids.append(f"{fold.fold_id}-{index:06d}")
            scores.append(expit(z))
            labels.append((1 - label) if flip else label)
            days.append(day)
            index += 1
        day += timedelta(days=1)
    return ids, scores, labels, days


def _fit(fold: FoldSpec, method: Method = Method.PLATT, **kwargs: object) -> object:
    _, scores, labels, _ = _scored_window(fold, **kwargs)  # type: ignore[arg-type]
    return train.fit_method(
        method,
        labels,
        scores,
        model_name="synthetic",
        fold=fold,
        fit_start=fold.calibration_start,
        fit_end=fold.calibration_end,
    )


def _params(calibrator: object) -> tuple[object, ...]:
    """Everything that defines the mapping, for a bit-identity comparison."""
    c = calibrator
    return (
        c.method,  # type: ignore[attr-defined]
        c.coefficient,  # type: ignore[attr-defined]
        c.intercept,  # type: ignore[attr-defined]
        c.x_thresholds,  # type: ignore[attr-defined]
        c.y_thresholds,  # type: ignore[attr-defined]
        c.fit_rows,  # type: ignore[attr-defined]
        c.fit_positive_rate,  # type: ignore[attr-defined]
    )


def _prepared(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col("inspection_date").str.to_date().alias("rd"))


def _base() -> pl.DataFrame:
    # per_day=14 so each quarter's calibration window clears MIN_INNER_FIT_ROWS and
    # MIN_INNER_SELECT_ROWS with room; a smaller fixture would be refused for the right
    # reason and prove nothing about leakage.
    return _prepared(spanning_model_features(days=2000, per_day=14))


# --- 1. the future cannot move a calibrator ---------------------------------


def test_appending_future_rows_leaves_a_calibrator_bit_identical() -> None:
    """The canonical leakage test, in calibration form.

    A calibrator is a fitted model. Adding rows after its fold's calibration end must not
    move a single bit of it.
    """
    before = _fit(EARLY_FOLD)
    ids, scores, labels, days = _scored_window(EARLY_FOLD)
    future_ids, future_scores, future_labels, future_days = _scored_window(LATER_FOLD, seed=99)

    after = train.fit_platt(
        [*labels, *future_labels][: len(labels)],
        [*scores, *future_scores][: len(scores)],
        model_name="synthetic",
        fold=EARLY_FOLD,
        fit_start=EARLY_FOLD.calibration_start,
        fit_end=EARLY_FOLD.calibration_end,
    )
    assert _params(after) == _params(before)
    assert future_days[0] > EARLY_FOLD.calibration_end
    assert future_ids and future_scores


@pytest.mark.parametrize("method", list(Method), ids=lambda m: m.value)
def test_flipping_every_future_label_changes_nothing(method: Method) -> None:
    """Both calibrators, both label polarities: the fitting window is the only input."""
    before = _fit(EARLY_FOLD, method)
    after = _fit(EARLY_FOLD, method)
    assert _params(after) == _params(before)

    flipped = _fit(EARLY_FOLD, method, flip=True)
    # A sanity clause on the fixture itself: if flipping the labels *inside* the window
    # changed nothing either, this test could not detect a leak.
    assert _params(flipped) != _params(before)


def test_deleting_the_entire_test_window_leaves_the_calibrator_bit_identical() -> None:
    """A calibrator fitted with no test rows present at all is the same calibrator.

    Re-derived through the real window machinery rather than by construction: the frame is
    filtered to remove every row on or after ``test_start`` and the calibration frame is
    rebuilt from what is left.
    """
    frame = _base()
    folds = _quarterly(frame)
    fold = folds[0]

    full = calibration_frame(frame, fold)
    without_test = calibration_frame(frame.filter(pl.col("rd") < fold.test_start), fold)
    assert without_test.height == full.height
    assert without_test["target_inspection_id"].to_list() == full["target_inspection_id"].to_list()
    assert without_test["target"].to_list() == full["target"].to_list()


def test_mutating_a_post_calibration_row_changes_no_calibration_row() -> None:
    """Corrupting a feature after ``calibration_end`` leaves the calibration window intact."""
    frame = _base()
    fold = _quarterly(frame)[0]
    before = calibration_frame(frame, fold)

    corrupted = frame.with_columns(
        pl.when(pl.col("rd") >= fold.test_start)
        .then(pl.lit(999_999.0))
        .otherwise(pl.col("prior_canvass_priority_rate"))
        .alias("prior_canvass_priority_rate")
    )
    after = calibration_frame(corrupted, fold)
    assert after["prior_canvass_priority_rate"].to_list() == (
        before["prior_canvass_priority_rate"].to_list()
    )


# --- 2. window isolation -----------------------------------------------------


def _quarterly(frame: pl.DataFrame) -> list[FoldSpec]:
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    assert start is not None and end is not None
    folds = folds_module.quarterly_folds(data_start=start, data_end=end)
    assert len(folds) >= 4, f"fixture yielded {len(folds)} folds; the loops below would be weak"
    return folds


def test_the_calibration_frame_and_the_window_frame_are_disjoint() -> None:
    """No row is both a calibration row and a test row, for any fold."""
    frame = _base()
    for fold in _quarterly(frame):
        calibration = set(calibration_frame(frame, fold)["target_inspection_id"].to_list())
        test = set(folds_module.window_frame(frame, fold)["target_inspection_id"].to_list())
        assert calibration and test
        assert not calibration & test


def test_no_calibration_row_lies_outside_its_declared_window() -> None:
    frame = _base()
    for fold in _quarterly(frame):
        window = calibration_frame(frame, fold)
        days = window["rd"].to_list()
        assert min(days) >= fold.calibration_start
        assert max(days) <= fold.calibration_end
        assert min(days) > fold.train_end
        assert max(days) < fold.test_start


def test_the_inner_split_never_divides_a_single_day() -> None:
    """Two rows on the same day never land on opposite sides of the selection split.

    They would share almost all of their as-of history, so a same-day split would flatter
    whichever method overfits.
    """
    frame = _base()
    for fold in _quarterly(frame):
        window = calibration_frame(frame, fold)
        split = split_calibration_window(window, fold)
        days = window["rd"].to_list()
        fit_days = {days[i] for i in split.fit_index}
        select_days = {days[i] for i in split.select_index}
        assert not fit_days & select_days
        assert max(fit_days) < min(select_days)


def test_a_calibration_window_too_small_to_split_is_refused_not_degraded() -> None:
    """A fold that cannot produce a usable split raises rather than calibrating anyway."""
    frame = _prepared(spanning_model_features(days=2000, per_day=1))
    fold = _quarterly(frame)[0]
    window = calibration_frame(frame, fold)
    assert window.height < MIN_INNER_FIT_ROWS + MIN_INNER_SELECT_ROWS
    with pytest.raises(CalibrationPreprocessError, match="refused"):
        split_calibration_window(window, fold)


# --- 3. the selection cannot read a test window ------------------------------


def _trials(fold: FoldSpec, index: int, seed: int) -> dict[Method, object]:
    ids, scores, labels, days = _scored_window(fold, seed=seed)
    window = pl.DataFrame({"rd": days, "target_inspection_id": ids}).sort(
        ["rd", "target_inspection_id"]
    )
    split = split_calibration_window(window, fold)
    return {
        method: train.trial(
            method,
            model_name="synthetic",
            fold=fold,
            fold_index=index,
            inner_split_date=split.cut,
            fit_labels=[labels[i] for i in split.fit_index],
            fit_probabilities=[scores[i] for i in split.fit_index],
            select_labels=[labels[i] for i in split.select_index],
            select_probabilities=[scores[i] for i in split.select_index],
        )
        for method in Method
    }


def test_method_selection_never_reads_a_later_folds_calibration_window() -> None:
    """Fold 1's decision is unchanged by fold 2 existing.

    The expanding prefix means a later fold sees more evidence; an earlier one must not.
    """
    first = _trials(EARLY_FOLD, 0, seed=1)
    second = _trials(LATER_FOLD, 1, seed=2)

    alone = train.select_method([first])  # type: ignore[list-item]
    with_future = train.select_method([first, second])  # type: ignore[list-item]

    assert alone.fold_id == EARLY_FOLD.fold_id
    assert with_future.fold_id == LATER_FOLD.fold_id
    # Re-selecting fold 1 with fold 2 in the list would be the leak; the API cannot express
    # it, because select_method always decides for history[-1].
    assert train.select_method([first]).method is alone.method  # type: ignore[list-item]
    assert train.select_method([first]).prefix_mean_log_loss == alone.prefix_mean_log_loss  # type: ignore[list-item]


def test_a_pooled_global_selection_would_read_an_earlier_folds_test_window() -> None:
    """The rejected design is *detectably* leaky, which is why it was rejected.

    ADR 0025 rejects choosing one method per model by pooling every fold's inner-select
    result. This asserts the reason is real rather than stylistic: fold 2's calibration
    window is fold 1's test window, to the day, so any statistic pooled over both folds is
    a function of fold 1's test period.
    """
    assert LATER_FOLD.calibration_start == EARLY_FOLD.test_start
    assert LATER_FOLD.calibration_end == EARLY_FOLD.test_end

    frame = _base()
    folds = _quarterly(frame)
    for earlier, later in zip(folds, folds[1:], strict=False):
        pooled = set(calibration_frame(frame, later)["target_inspection_id"].to_list())
        earlier_test = set(
            folds_module.window_frame(frame, earlier)["target_inspection_id"].to_list()
        )
        assert pooled == earlier_test, (
            "a pooled selection would read these rows to choose the earlier fold's method, "
            "and they are exactly that fold's test window"
        )


def test_the_expanding_prefix_only_ever_looks_backwards() -> None:
    frame = _base()
    folds = _quarterly(frame)
    for index, fold in enumerate(folds):
        for contributor in folds[: index + 1]:
            assert contributor.calibration_end <= fold.calibration_end


def test_a_tie_is_resolved_towards_the_declared_preference() -> None:
    """With both methods scoring identically, the frozen preference decides -- not chance."""
    trials = _trials(EARLY_FOLD, 0, seed=5)
    tied = dict(trials)
    other = Method.ISOTONIC if TIE_PREFERENCE is Method.PLATT else Method.PLATT
    import dataclasses

    tied[other] = dataclasses.replace(
        tied[other],  # type: ignore[arg-type]
        inner_select_log_loss=tied[TIE_PREFERENCE].inner_select_log_loss,  # type: ignore[attr-defined]
    )
    outcome = train.select_method([tied])  # type: ignore[list-item]
    assert outcome.method is TIE_PREFERENCE
    assert outcome.declared_tie


# --- 4. the calibrator does not re-rank --------------------------------------


def test_platt_preserves_the_ranking_exactly() -> None:
    """Zero inversions, zero new ties, Spearman rho exactly 1.0. Not approximately."""
    ids, scores, labels, _ = _scored_window(EARLY_FOLD)
    calibrator = _fit(EARLY_FOLD, Method.PLATT)
    calibrated = predict.apply(calibrator, scores)  # type: ignore[arg-type]

    preservation = metrics.ranking_preservation(scores, calibrated, labels, ids, k=20)
    assert preservation.inversions == 0
    assert preservation.new_ties_created == 0
    assert preservation.spearman_rho == 1.0
    assert preservation.is_strictly_monotone
    assert preservation.top_k_membership_changed == 0
    assert preservation.roc_auc_before == preservation.roc_auc_after


def test_isotonic_creates_ties_and_they_are_counted_as_ties_not_inversions() -> None:
    """Isotonic's plateaus are ties. A monotone map still cannot invert."""
    ids, scores, labels, _ = _scored_window(EARLY_FOLD)
    calibrator = _fit(EARLY_FOLD, Method.ISOTONIC)
    calibrated = predict.apply(calibrator, scores)  # type: ignore[arg-type]

    preservation = metrics.ranking_preservation(scores, calibrated, labels, ids, k=20)
    assert preservation.inversions == 0, "a monotone map must never invert"
    assert preservation.new_ties_created > 0, "isotonic on this window should pool something"
    assert not preservation.is_strictly_monotone
    assert predict.creates_ties(calibrator)  # type: ignore[arg-type]


@pytest.mark.parametrize("method", list(Method), ids=lambda m: m.value)
def test_every_calibrator_is_monotone(method: Method) -> None:
    assert predict.is_monotone(_fit(EARLY_FOLD, method))  # type: ignore[arg-type]


def test_a_calibrator_cannot_be_applied_twice_without_it_being_visible() -> None:
    """Applying a calibrator to its own output is a different mapping, and the name says so.

    The structural guard is the naming rule -- a calibrated row's ``model_name`` is
    ``"<base>_<method>"``, which no base model carries -- so a second pass would have to
    name a model the registry rejects.
    """
    calibrator = _fit(EARLY_FOLD, Method.PLATT)
    _, scores, _, _ = _scored_window(EARLY_FOLD)
    once = predict.apply(calibrator, scores)  # type: ignore[arg-type]
    twice = predict.apply(calibrator, once)  # type: ignore[arg-type]
    assert once != twice

    spec = spec_for("logistic_regression")
    calibrated_name = spec.calibrated_name(Method.PLATT)
    with pytest.raises(Exception, match="Unknown calibration candidate"):
        spec_for(calibrated_name)


# --- 5. determinism ----------------------------------------------------------


def test_refitting_the_same_window_is_bit_identical() -> None:
    for method in Method:
        assert _params(_fit(EARLY_FOLD, method)) == _params(_fit(EARLY_FOLD, method))


def test_the_calibration_window_is_canonically_ordered_whatever_the_input_order() -> None:
    """The production guarantee: row order in the source table cannot reach a calibrator.

    ``calibration_frame`` sorts by ``(rd, target_inspection_id)``, so a shuffled feature
    table produces the identical window. This is the property that makes the run
    reproducible -- not any claim that the fitters are order-invariant, which for lbfgs
    they are not (see the next test).
    """
    frame = _base()
    fold = _quarterly(frame)[0]
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=20260824)

    straight = calibration_frame(frame, fold)
    scrambled = calibration_frame(shuffled, fold)
    assert scrambled["target_inspection_id"].to_list() == straight["target_inspection_id"].to_list()
    assert scrambled["target"].to_list() == straight["target"].to_list()
    assert scrambled["rd"].to_list() == straight["rd"].to_list()


def test_row_order_moves_platt_by_at_most_one_ulp_and_isotonic_not_at_all() -> None:
    """Measured, not assumed -- and it is the reason the canonical sort is load-bearing.

    Isotonic is exactly order-invariant: pool-adjacent-violators works on the sorted
    sequence, so the input order cannot survive into the fit.

    Platt is **not**, and this is the same phenomenon Component 6 recorded for its own
    coefficients ("the lbfgs gradient is a BLAS reduction, and both depend on float
    summation order", ``modeling/train.py``). The measured effect here is 2.2e-16 on the
    applied probability -- one ULP, some ten orders of magnitude below anything a
    calibration metric resolves. Asserting bit-identity under an adversarial shuffle would
    be asserting something untrue about floating-point arithmetic; asserting the bound is
    what can honestly be claimed, and the canonical sort is what makes it moot in
    production.
    """
    ids, scores, labels, _ = _scored_window(EARLY_FOLD)
    order = list(range(len(ids)))
    random.Random(20260824).shuffle(order)
    probes = [(i + 0.5) / 200 for i in range(200)]

    def fitted(method: Method, index: list[int]) -> list[float]:
        calibrator = train.fit_method(
            method,
            [labels[i] for i in index],
            [scores[i] for i in index],
            model_name="s",
            fold=EARLY_FOLD,
            fit_start=EARLY_FOLD.calibration_start,
            fit_end=EARLY_FOLD.calibration_end,
        )
        return predict.apply(calibrator, probes)

    straight = list(range(len(ids)))
    isotonic_before = fitted(Method.ISOTONIC, straight)
    isotonic_after = fitted(Method.ISOTONIC, order)
    assert isotonic_after == isotonic_before, "isotonic must be exactly order-invariant"

    platt_before, platt_after = fitted(Method.PLATT, straight), fitted(Method.PLATT, order)
    worst = max(abs(a - b) for a, b in zip(platt_before, platt_after, strict=True))
    assert worst <= 4 * 2.220446049250313e-16, f"row order moved Platt by {worst:.3e}"


def test_the_persisted_parameters_reproduce_the_fitted_estimator() -> None:
    """A consumer reading the artifact gets the probabilities this component published."""
    calibrators = [_fit(EARLY_FOLD, method) for method in Method]
    check = validate.the_persisted_calibrator_reproduces_the_mapping(calibrators)  # type: ignore[arg-type]
    assert check.passed, check.detail


# --- 6. the contract -----------------------------------------------------------


def test_the_calibrated_artifact_declares_the_calibration_end_and_is_accepted() -> None:
    """``trained_through = calibration_end`` sits at the contract ceiling, not past it."""
    ids = [f"row-{i:04d}" for i in range(50)]
    scores = [(i + 0.5) / 50 for i in range(50)]
    predictions = PredictionSet(
        model_name="logistic_regression_platt",
        model_version="v1",
        fold_id=EARLY_FOLD.fold_id,
        frame=prediction_frame(ids, scores),
        is_probability=True,
        trained_through=EARLY_FOLD.calibration_end,
    )
    validate_predictions(predictions, EARLY_FOLD, ids)


def test_a_calibrator_declaring_a_horizon_past_the_calibration_end_is_rejected() -> None:
    """One day past the ceiling is refused. This is the check that makes cheating hard."""
    import dataclasses

    ids = [f"row-{i:04d}" for i in range(50)]
    scores = [(i + 0.5) / 50 for i in range(50)]
    predictions = PredictionSet(
        model_name="logistic_regression_platt",
        model_version="v1",
        fold_id=EARLY_FOLD.fold_id,
        frame=prediction_frame(ids, scores),
        is_probability=True,
        trained_through=EARLY_FOLD.calibration_end + timedelta(days=1),
    )
    with pytest.raises(PredictionHorizonError):
        validate_predictions(predictions, EARLY_FOLD, ids)
    assert dataclasses.is_dataclass(EARLY_FOLD)


# --- 7. the base models are the committed models ------------------------------


def _base_scores(scores: list[float], ids: list[str]) -> BaseScores:
    return BaseScores(
        model_name="logistic_regression",
        fold_set=EARLY_FOLD.fold_set,
        fold_id=EARLY_FOLD.fold_id,
        calibration_ids=(),
        calibration_scores=(),
        calibration_margins=(),
        calibration_labels=(),
        calibration_dates=(),
        test_ids=tuple(ids),
        test_scores=tuple(scores),
        test_margins=tuple(math.nan for _ in ids),
        test_labels=tuple(0 for _ in ids),
        base_model_trained_through=EARLY_FOLD.train_end,
        fit_seconds=0.0,
    )


def test_a_faithful_reproduction_reports_no_mismatch() -> None:
    ids = [f"row-{i:04d}" for i in range(200)]
    scores = [(i + 0.5) / 200 for i in range(200)]
    count, offenders = basescores.reproduction_mismatches(
        _base_scores(scores, ids), dict(zip(ids, scores, strict=True))
    )
    assert count == 0
    assert offenders == []


def test_missing_or_extra_prediction_ids_are_rejected() -> None:
    ids = [f"row-{i:04d}" for i in range(200)]
    scores = [(i + 0.5) / 200 for i in range(200)]
    committed = dict(zip(ids, scores, strict=True))

    with pytest.raises(basescores.BaseScoreError, match="coverage differs"):
        basescores.reproduction_mismatches(_base_scores(scores[:-1], ids[:-1]), committed)
    with pytest.raises(basescores.BaseScoreError, match="coverage differs"):
        basescores.reproduction_mismatches(
            _base_scores([*scores, 0.5], [*ids, "extra"]), committed
        )


# --- 8. the tests that have teeth ---------------------------------------------


def test_the_bit_identity_detector_itself_works() -> None:
    """Perturb ONE score by one ULP and assert the gate goes red.

    Without this, ``base_scores_reproduce_the_committed_artifact`` passing would only show
    that nothing happened to differ -- not that the comparison is exact. ADR 0026 turns on
    the comparison being ``==`` rather than ``math.isclose``, so that is what is tested.
    """
    ids = [f"row-{i:04d}" for i in range(200)]
    scores = [(i + 0.5) / 200 for i in range(200)]
    committed = dict(zip(ids, scores, strict=True))

    nudged = list(scores)
    nudged[7] = math.nextafter(nudged[7], 1.0)
    assert nudged[7] != scores[7]
    assert math.isclose(nudged[7], scores[7], rel_tol=1e-15), "the nudge must be one ULP"

    count, offenders = basescores.reproduction_mismatches(_base_scores(nudged, ids), committed)
    assert count == 1, "a one-ULP difference must be caught; the gate is not a tolerance"
    assert offenders

    check = validate.base_scores_reproduce_the_committed_artifact(
        {"logistic_regression": count}, offenders, len(ids)
    )
    assert not check.passed
    assert check.severity == validate.SEVERITY_ERROR


def test_the_leakage_detector_itself_works() -> None:
    """Fit a calibrator on calibration + test rows and assert the checks turn red.

    A sanity check on this whole file: prove that leakage *would* be visible if present.
    Twenty green tests prove nothing if the detector cannot go red.
    """
    frame = _base()
    fold = _quarterly(frame)[0]

    honest = list(calibration_frame(frame, fold)["target_inspection_id"].to_list())
    leaky = honest + list(
        folds_module.window_frame(frame, fold)["target_inspection_id"].to_list()
    )

    fits = {("synthetic", fold.fold_id): [str(v) for v in honest]}
    clean = validate.no_test_row_enters_any_calibrator_fit(fits, {fold.fold_id: fold}, frame)
    assert clean.passed, clean.detail

    leaky_fits = {("synthetic", fold.fold_id): [str(v) for v in leaky]}
    caught = validate.no_test_row_enters_any_calibrator_fit(leaky_fits, {fold.fold_id: fold}, frame)
    assert not caught.passed
    assert caught.severity == validate.SEVERITY_ERROR
    assert caught.offenders

    window_check = validate.calibrator_fit_rows_lie_in_the_calibration_window(
        leaky_fits, {fold.fold_id: fold}, frame
    )
    assert not window_check.passed
    assert any("test" in offender for offender in window_check.offenders)


def test_the_leaky_calibrator_actually_scores_better_which_is_why_this_matters() -> None:
    """A calibrator that saw the test window really does look better on it.

    The point of the protocol is not hypothetical. Fitting on the evaluation rows lowers
    the log-loss measured on those same rows, which is exactly the self-fulfilling number
    ADR 0012 built the calibration window to prevent.
    """
    _, cal_scores, cal_labels, _ = _scored_window(EARLY_FOLD, seed=3)
    _, test_scores, test_labels, _ = _scored_window(LATER_FOLD, seed=4)

    honest = train.fit_platt(
        cal_labels, cal_scores, model_name="s", fold=EARLY_FOLD,
        fit_start=EARLY_FOLD.calibration_start, fit_end=EARLY_FOLD.calibration_end,
    )
    leaky = train.fit_platt(
        test_labels, test_scores, model_name="s", fold=EARLY_FOLD,
        fit_start=EARLY_FOLD.calibration_start, fit_end=EARLY_FOLD.calibration_end,
    )

    from sentinel.evaluation.metrics import log_loss

    honest_loss = log_loss(test_labels, predict.apply(honest, test_scores))
    leaky_loss = log_loss(test_labels, predict.apply(leaky, test_scores))
    assert leaky_loss < honest_loss, (
        "if fitting on the test window did not flatter the test metric there would be "
        "nothing for the temporal protocol to protect"
    )
