"""Fitting one network to one fold: the split, the loop, and what is recorded.

The temporal properties live in ``test_neural_leakage.py``. This file is about the
mechanics: does the early-stopping split land where it should, does patience actually
stop a run, are the restored weights the best epoch's rather than the last epoch's, and
is every number the manifest reports actually measured.

Every fit here uses a small epoch budget. The architecture is fixed by the specification
and is exercised in ``test_neural_net.py``; running 200 epochs to observe a training loop
would make the suite slow for no additional coverage.
"""

from __future__ import annotations

import polars as pl
import pytest

from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling.train import training_frame
from sentinel.neural import predict, train
from sentinel.neural.definitions import INNER_VALIDATION_FRACTION, spec_for
from tests.conftest import neural_categoricals_for, spanning_model_features

PRIMARY = spec_for("neural_embeddings")
NUMERIC_ONLY = spec_for("neural_numeric_only")
WEIGHTED = spec_for("neural_pos_weighted")


def _base() -> pl.DataFrame:
    return spanning_model_features(days=1600, per_day=2).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )


def _fold(frame: pl.DataFrame) -> FoldSpec:
    built = folds_module.quarterly_folds(
        data_start=folds_module.min_date(frame, "rd"),
        data_end=folds_module.max_date(frame, "rd"),
    )
    assert built
    return built[0]


# --- 1. the inner split ------------------------------------------------------


def test_the_split_holds_back_roughly_the_declared_fraction() -> None:
    frame = _base()
    fold = _fold(frame)
    window = training_frame(frame, fold)
    cut = train.inner_split_date(window)
    validation = window.filter(pl.col("rd") >= cut)
    share = validation.height / window.height
    assert INNER_VALIDATION_FRACTION * 0.6 <= share <= INNER_VALIDATION_FRACTION * 2.0, (
        f"held back {share:.3f} of training rows, far from the declared {INNER_VALIDATION_FRACTION}"
    )


def test_the_split_falls_on_a_whole_day() -> None:
    frame = _base()
    fold = _fold(frame)
    window = training_frame(frame, fold)
    cut = train.inner_split_date(window)
    left = window.filter(pl.col("rd") < cut)
    right = window.filter(pl.col("rd") >= cut)
    assert set(left["rd"].to_list()).isdisjoint(set(right["rd"].to_list()))


def test_the_split_is_temporally_ordered() -> None:
    frame = _base()
    fold = _fold(frame)
    window = training_frame(frame, fold)
    left, right, cut = train.split_training_window(window, PRIMARY, fold)
    assert max(left["rd"].to_list()) < cut <= min(right["rd"].to_list())
    assert cut <= fold.train_end


def test_a_window_too_small_to_split_is_refused_not_silently_fitted() -> None:
    """Two folds trained under different stopping rules are not comparable."""
    frame = _base()
    fold = _fold(frame)
    tiny = training_frame(frame, fold).head(50)
    with pytest.raises(train.NeuralTrainError, match="at least"):
        train.split_training_window(tiny, PRIMARY, fold)


def test_a_frame_without_a_reference_date_is_refused() -> None:
    with pytest.raises(train.NeuralTrainError, match="parsed reference date"):
        train.inner_split_date(pl.DataFrame({"x": [1, 2, 3]}))


# --- 2. the training loop ----------------------------------------------------


def _fit(frame: pl.DataFrame, fold: FoldSpec, spec=PRIMARY, **kwargs):  # type: ignore[no-untyped-def]
    return train.fit_fold(
        spec,
        training_frame(frame, fold),
        fold,
        categoricals=neural_categoricals_for(frame),
        **kwargs,
    )


def test_every_epoch_is_recorded_with_both_losses() -> None:
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold, max_epochs=4)
    assert len(fitted.epochs) == fitted.final_epoch
    for record in fitted.epochs:
        assert record.train_loss > 0.0
        assert record.validation_loss > 0.0
        assert record.learning_rate > 0.0
    assert [r.epoch for r in fitted.epochs] == list(range(1, fitted.final_epoch + 1))


def test_the_epoch_budget_is_respected() -> None:
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold, max_epochs=3)
    assert fitted.final_epoch <= 3
    assert fitted.stop_reason == train.STOP_BUDGET


def test_patience_stops_a_run_before_the_budget() -> None:
    """Patience must stop a run whose validation loss stops improving.

    Driven with a deliberately large learning rate rather than the frozen one. On this
    fixture the signal is clean enough that a well-chosen rate improves the validation
    loss monotonically for dozens of epochs -- which is correct behaviour, and would make
    a patience test written against it silently vacuous. A rate that overshoots makes the
    loss bounce, which is the condition patience exists to detect.
    """
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold, max_epochs=40, patience=1, learning_rate=0.5)
    assert fitted.stop_reason == train.STOP_EARLY, (
        "patience never fired even at a diverging learning rate"
    )
    assert fitted.final_epoch < 40
    assert fitted.best_epoch <= fitted.final_epoch


def test_a_run_that_keeps_improving_uses_its_whole_budget() -> None:
    """The other half of the same mechanism, and the reason the test above needs care.

    A monotonically improving fit must NOT stop early. Recording this explicitly means a
    future change that made patience fire spuriously would be caught.
    """
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold, max_epochs=6, patience=15)
    assert fitted.final_epoch == 6
    assert fitted.stop_reason == train.STOP_BUDGET


def test_the_restored_epoch_is_the_best_validation_epoch() -> None:
    """Without restoration, patience would detect overfitting and return the overfit model."""
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold, max_epochs=12, patience=3)
    losses = {r.epoch: r.validation_loss for r in fitted.epochs}
    assert fitted.best_epoch == min(losses, key=lambda e: losses[e])


def test_the_recorded_learning_rate_is_the_frozen_one_for_the_fold_set() -> None:
    from sentinel.neural.definitions import learning_rate_for

    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold, max_epochs=2)
    assert fitted.learning_rate == learning_rate_for(fold.fold_set)


def test_an_explicit_learning_rate_overrides_the_frozen_one() -> None:
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold, max_epochs=2, learning_rate=0.05)
    assert fitted.learning_rate == 0.05
    assert fitted.epochs[0].learning_rate == 0.05


def test_an_empty_training_window_is_refused() -> None:
    frame = _base()
    fold = _fold(frame)
    with pytest.raises(train.NeuralTrainError, match="no training rows"):
        train.fit_fold(
            PRIMARY,
            training_frame(frame, fold).head(0),
            fold,
            categoricals=neural_categoricals_for(frame),
        )


def test_a_single_class_training_window_is_refused() -> None:
    """A constant probability is a reference schedule, not a model."""
    frame = _base().with_columns(pl.lit(1, dtype=pl.Int8).alias("target"))
    fold = _fold(frame)
    with pytest.raises(train.NeuralTrainError, match="single class"):
        _fit(frame, fold, max_epochs=2)


def test_a_null_target_is_refused() -> None:
    frame = _base()
    fold = _fold(frame)
    broken = frame.with_columns(
        pl.when(pl.col("rd") == pl.col("rd").min())
        .then(None)
        .otherwise(pl.col("target"))
        .cast(pl.Int8)
        .alias("target")
    )
    with pytest.raises(train.NeuralTrainError, match="null target"):
        _fit(broken, fold, max_epochs=2)


# --- 3. what the fit records -------------------------------------------------


def test_the_fit_records_what_it_actually_saw() -> None:
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold, max_epochs=3)

    window = training_frame(frame, fold)
    assert fitted.train_rows == window.height
    assert fitted.inner_train_rows + fitted.inner_validation_rows == window.height
    assert fitted.train_start == fold.train_start
    assert fitted.train_end == fold.train_end
    assert fitted.trained_through == fold.train_end
    assert fitted.calibration_end_unused == fold.calibration_end
    assert 0.0 < (fitted.train_positive_rate or 0.0) < 1.0


def test_the_fit_records_the_architecture_it_built() -> None:
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold, max_epochs=2)
    assert fitted.dense_width == 30
    assert fitted.embedding_width == 16 + 8 + 8 + 8
    assert fitted.parameter_count > 0
    assert len(fitted.embeddings) == len(PRIMARY.entity_columns)


def test_an_embedding_table_has_one_vector_per_category() -> None:
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold, max_epochs=2)
    for table in fitted.embeddings:
        vocab = fitted.encoding.vocabulary_for(table.column)
        assert len(table.vectors) == vocab.size
        assert len(table.categories) == vocab.size
        assert table.categories == vocab.categories
        assert table.dim > 0


def test_a_numeric_only_fit_has_no_embeddings_at_all() -> None:
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold, NUMERIC_ONLY, max_epochs=2)
    assert fitted.embeddings == ()
    assert fitted.embedding_width == 0
    assert fitted.encoding.vocabularies == ()


def test_pos_weight_is_applied_only_by_the_declared_ablation() -> None:
    frame = _base()
    fold = _fold(frame)
    plain = _fit(frame, fold, PRIMARY, max_epochs=2)
    weighted = _fit(frame, fold, WEIGHTED, max_epochs=2)
    assert plain.pos_weight is None
    assert weighted.pos_weight is not None and weighted.pos_weight > 0.0


def test_pos_weight_comes_from_the_training_windows_own_prevalence() -> None:
    """Never a global constant; the ablation is only interpretable if it is fold-local."""
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold, WEIGHTED, max_epochs=2)
    window = training_frame(frame, fold).filter(pl.col("rd") < fitted.inner_validation_start)
    positives = int((window["target"] == 1).sum())
    negatives = int((window["target"] == 0).sum())
    assert fitted.pos_weight == pytest.approx(negatives / positives)


def test_a_missing_categorical_table_is_refused_for_an_embedding_spec() -> None:
    frame = _base()
    fold = _fold(frame)
    with pytest.raises(train.NeuralTrainError, match="requires categoricals"):
        train.fit_fold(PRIMARY, training_frame(frame, fold), fold, categoricals=None)


def test_a_numeric_only_spec_needs_no_categorical_table() -> None:
    frame = _base()
    fold = _fold(frame)
    fitted = train.fit_fold(
        NUMERIC_ONLY, training_frame(frame, fold), fold, categoricals=None, max_epochs=2
    )
    assert fitted.spec.name == "neural_numeric_only"


# --- 4. scoring --------------------------------------------------------------


def test_scores_are_probabilities_in_the_unit_interval() -> None:
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold, max_epochs=3)
    window = folds_module.window_frame(frame, fold)
    ids, scores = predict.score_window(fitted, window, categoricals=neural_categoricals_for(frame))
    assert len(ids) == len(scores) == window.height
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_scores_align_positionally_with_the_windows_own_order() -> None:
    """Component 5 depends on this; a mis-join would be silent."""
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold, max_epochs=2)
    window = folds_module.window_frame(frame, fold)
    ids, _ = predict.score_window(fitted, window, categoricals=neural_categoricals_for(frame))
    assert ids == [str(v) for v in window["target_inspection_id"].to_list()]


def test_scoring_an_empty_window_is_refused() -> None:
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold, max_epochs=2)
    with pytest.raises(predict.NeuralPredictError, match="empty"):
        predict.score_window(
            fitted,
            folds_module.window_frame(frame, fold).head(0),
            categoricals=neural_categoricals_for(frame),
        )


def test_scoring_needs_the_categorical_table_the_vocabularies_were_fitted_against() -> None:
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold, max_epochs=2)
    with pytest.raises(predict.NeuralPredictError, match="requires the categorical table"):
        predict.score_window(fitted, folds_module.window_frame(frame, fold), categoricals=None)


def test_saturated_scores_are_counted_not_rejected() -> None:
    assert predict.saturated_count([0.0, 0.5, 1.0, 0.99]) == 2
    assert predict.saturated_count([0.5]) == 0
