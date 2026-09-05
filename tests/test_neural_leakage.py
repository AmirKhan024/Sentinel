"""Component 8's safety wall. The ten properties the specification requires, driven.

Modelled on ``test_boosting_leakage.py`` and inheriting its standard: *a fold's fit must
be **bit-identical** when anything it was not allowed to see changes.* Not "close", not
"within tolerance" -- identical. A network has more moving parts than a booster, so the
standard matters more here, not less.

Component 8 adds three exposures Components 6 and 7 did not have, and each gets its own
section:

* **A vocabulary is a fitted statistic.** An embedding index that exists because its
  category appears in a test window is future information even though it is not a label.
* **Chain membership depends on which other establishments exist**, so it can only be
  computed inside a fold.
* **Early stopping needs a validation signal**, and the only windows later than the
  training data are the ones the fold forbids. The signal is carved from inside the
  training window instead, and that is checked rather than asserted.

The last test in this file plants the label in a feature and proves the fixture reveals
it. Without that, every test above it could be passing because the model is weak rather
than because the pipeline is protected. Component 7 learned this the hard way: both of
its leakage-test failures were bugs in the tests, not in the code.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import polars as pl
import pytest

from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling.train import training_frame
from sentinel.neural import embed, encode, predict, preprocess, train
from sentinel.neural.definitions import (
    INDEPENDENT_CHAIN,
    UNKNOWN_CATEGORY,
    CategoricalEncoding,
    NeuralSpec,
    spec_for,
)
from tests.conftest import neural_categoricals_for, spanning_model_features

#: Small enough that a fit takes a second, large enough that a fold is real.
EPOCHS = 3

PRIMARY = spec_for("neural_embeddings")
NUMERIC_ONLY = spec_for("neural_numeric_only")


def _base() -> pl.DataFrame:
    frame = spanning_model_features(days=1600, per_day=2)
    return frame.with_columns(pl.col("inspection_date").str.to_date().alias("rd"))


def _fold(frame: pl.DataFrame) -> FoldSpec:
    built = folds_module.quarterly_folds(
        data_start=folds_module.min_date(frame, "rd"),
        data_end=folds_module.max_date(frame, "rd"),
    )
    assert built, "fixture produced no quarterly folds; every test below would be vacuous"
    return built[0]


def _fit(
    frame: pl.DataFrame,
    fold: FoldSpec,
    spec: NeuralSpec = PRIMARY,
    categoricals: pl.DataFrame | None = None,
) -> Any:
    cats = neural_categoricals_for(frame) if categoricals is None else categoricals
    return train.fit_fold(
        spec,
        training_frame(frame, fold),
        fold,
        categoricals=cats,
        max_epochs=EPOCHS,
    )


def _scores(
    frame: pl.DataFrame,
    fold: FoldSpec,
    spec: NeuralSpec = PRIMARY,
    categoricals: pl.DataFrame | None = None,
) -> list[float]:
    cats = neural_categoricals_for(frame) if categoricals is None else categoricals
    fitted = _fit(frame, fold, spec, cats)
    _, scores = predict.score_window(
        fitted, folds_module.window_frame(frame, fold), categoricals=cats
    )
    return scores


def _future_frame(frame: pl.DataFrame, fold: FoldSpec, *, target: int) -> pl.DataFrame:
    """The same table with extra rows appended strictly after the fold's TEST window.

    After ``test_end``, not merely after ``train_end``. Rows after ``train_end`` include
    the fold's own calibration and test rows, and a test row's score is *supposed* to
    change when that row changes -- comparing predictions across such a mutation
    measures the evaluator, not the training boundary. Component 7 shipped exactly this
    bug once; its handoff records it, and reproducing it here was how this comment got
    written.
    """
    later = frame.filter(pl.col("rd") > fold.test_end)
    assert later.height > 0, "fixture has no rows after test_end; the test would be vacuous"
    appended = later.head(50).with_columns(
        pl.col("target_inspection_id") + "_X",
        pl.lit(target, dtype=pl.Int8).alias("target"),
    )
    return pl.concat([frame, appended]).sort(["rd", "target_inspection_id"])


# --- 1. test-period mutation cannot change a training artifact ----------------


@pytest.mark.parametrize("target", [0, 1])
def test_appending_future_rows_leaves_a_fold_bit_identical(target: int) -> None:
    """Rows after train_end must be invisible to the fit, of either class.

    Both classes are exercised because a leak that only moved the fit when future
    positives arrived would be invisible to a single-class version of this test.
    """
    frame = _base()
    fold = _fold(frame)
    before = _scores(frame, fold)
    after = _scores(_future_frame(frame, fold, target=target), fold)
    assert before == after, (
        "appending rows after train_end changed this fold's predictions, so something "
        "the fold was not allowed to see reached the fit"
    )


def test_flipping_every_future_label_leaves_a_fold_bit_identical() -> None:
    """The strongest form: invert every label after train_end and refit."""
    frame = _base()
    fold = _fold(frame)
    before = _scores(frame, fold)

    flipped = frame.with_columns(
        pl.when(pl.col("rd") > pl.lit(fold.train_end))
        .then(1 - pl.col("target"))
        .otherwise(pl.col("target"))
        .cast(pl.Int8)
        .alias("target")
    )
    changed = int(frame.filter(pl.col("rd") > fold.train_end).height)
    assert changed > 0, "no future rows to flip; the test would be vacuous"

    after = _scores(flipped, fold)
    assert before == after, (
        f"flipping {changed} future label(s) changed this fold's predictions, which means "
        "a label from after train_end reached the model"
    )


def test_corrupting_a_post_training_feature_leaves_a_fold_bit_identical() -> None:
    """A feature value after train_end must not reach the fit either."""
    frame = _base()
    fold = _fold(frame)
    before = _scores(frame, fold)

    corrupted = frame.with_columns(
        pl.when(pl.col("rd") > pl.lit(fold.test_end))
        .then(pl.lit(999999.0))
        .otherwise(pl.col("prior_canvass_fail_rate"))
        .alias("prior_canvass_fail_rate")
    )
    changed = frame.filter(pl.col("rd") > fold.test_end).height
    assert changed > 0, "no rows after test_end to corrupt; the test would be vacuous"
    after = _scores(corrupted, fold)
    assert before == after, "a feature value from beyond the test window reached the fit"


def test_every_fitted_artifact_is_unchanged_by_anything_after_train_end() -> None:
    """The broader property, stated over artifacts rather than predictions.

    A test row's *score* legitimately changes when that row's features change -- it is
    the row being predicted. What must never change is anything the fit *learned*: the
    scaler, the imputation values, the vocabularies, the chain set and the learned
    vectors. This corrupts every row after ``train_end`` (calibration and test included)
    and asserts all five are identical.
    """
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    before = _fit(frame, fold, PRIMARY, cats)

    corrupted = frame.with_columns(
        pl.when(pl.col("rd") > pl.lit(fold.train_end))
        .then(pl.lit(999999.0))
        .otherwise(pl.col("prior_canvass_fail_rate"))
        .alias("prior_canvass_fail_rate"),
        pl.when(pl.col("rd") > pl.lit(fold.train_end))
        .then(1 - pl.col("target"))
        .otherwise(pl.col("target"))
        .cast(pl.Int8)
        .alias("target"),
    )
    assert frame.filter(pl.col("rd") > fold.train_end).height > 0
    after = _fit(corrupted, fold, PRIMARY, cats)

    assert before.scaler_mean == after.scaler_mean
    assert before.scaler_scale == after.scaler_scale
    assert before.imputed_values == after.imputed_values
    assert before.encoding.chains == after.encoding.chains
    assert before.encoding.sizes == after.encoding.sizes
    for left, right in zip(before.embeddings, after.embeddings, strict=True):
        assert left.categories == right.categories
        assert left.vectors == right.vectors, (
            f"{left.column}: the learned vectors moved when data after train_end changed"
        )


# --- 2. the calibration window is genuinely unused ---------------------------


def test_deleting_the_calibration_window_changes_nothing() -> None:
    """Component 8 early-stops, and this proves it did not early-stop on calibration.

    The fold's calibration window exists for Component 9. If removing it moved a single
    prediction, the early-stopping signal -- or a scaler statistic, or a vocabulary --
    had reached into it.
    """
    frame = _base()
    fold = _fold(frame)
    before = _scores(frame, fold)

    without = frame.filter(
        (pl.col("rd") < fold.calibration_start) | (pl.col("rd") > fold.calibration_end)
    )
    removed = frame.height - without.height
    assert removed > 0, "fixture has no calibration rows; the test would be vacuous"

    after = _scores(without, fold)
    assert before == after, (
        f"deleting {removed} calibration row(s) changed this fold's predictions, so the "
        "fit read a window it declared unused"
    )


def test_the_early_stopping_window_lies_inside_the_training_window() -> None:
    """The property that makes ``trained_through = train_end`` true for C8."""
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold)

    assert fold.train_start < fitted.inner_validation_start <= fold.train_end, (
        f"early stopping validated from {fitted.inner_validation_start}, which is outside "
        f"the training window {fold.train_start}..{fold.train_end}"
    )
    assert fitted.inner_validation_start < fold.calibration_start
    assert fitted.trained_through == fold.train_end
    assert fitted.inner_train_rows > 0 and fitted.inner_validation_rows > 0


def test_the_inner_split_never_divides_a_single_day() -> None:
    """Two inspections on one day share almost all their as-of history.

    Splitting a day across the early-stopping boundary would leak near-duplicate rows in
    the ordinary machine-learning sense, which no temporal check would catch.
    """
    frame = _base()
    fold = _fold(frame)
    window = training_frame(frame, fold)
    cut = train.inner_split_date(window)

    left = {d for d in window.filter(pl.col("rd") < cut)["rd"].to_list()}
    right = {d for d in window.filter(pl.col("rd") >= cut)["rd"].to_list()}
    assert left and right, "one side of the split is empty"
    assert not (left & right), "a single date appears on both sides of the inner split"
    assert max(left) < min(right)


# --- 3. preprocessing statistics come from the permitted rows only -----------


def test_scaler_and_imputer_are_unchanged_by_future_rows() -> None:
    """Every fitted statistic must be identical when the future changes."""
    frame = _base()
    fold = _fold(frame)
    before = _fit(frame, fold)
    after = _fit(_future_frame(frame, fold, target=1), fold)

    assert before.scaler_mean == after.scaler_mean, "the scaler's means moved"
    assert before.scaler_scale == after.scaler_scale, "the scaler's scales moved"
    assert before.imputed_values == after.imputed_values, "an imputation value moved"


def test_imputation_values_re_derive_from_the_inner_training_window() -> None:
    """Recompute each median independently and compare, as Component 6 does."""
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold)

    source = frame.filter(
        (pl.col("rd") >= fold.train_start) & (pl.col("rd") < fitted.inner_validation_start)
    )
    assert source.height > 0
    checked = 0
    for column, fill in fitted.imputed_values.items():
        if column not in source.columns:
            continue
        strategy = preprocess.baseline_preprocess.strategy_for(column)
        if strategy.value == "constant_false":
            assert fill == 0.0, f"{column}: nullable boolean filled with {fill}, not 0.0"
            checked += 1
            continue
        median = source[column].cast(pl.Float64).median()
        if median is None:
            continue
        assert abs(fill - float(median)) < 1e-9, (
            f"{column}: fitted fill {fill} does not match the inner training window's "
            f"median {median}"
        )
        checked += 1
    assert checked >= 4, f"only {checked} statistic(s) checked; the test is too weak"


# --- 4. categorical mappings are temporally safe -----------------------------


def test_a_test_only_category_never_enters_the_vocabulary() -> None:
    """The leak specific to an embedding, driven directly.

    A category is planted that appears **only** after ``train_end``. It must not acquire
    an index, and rows carrying it must map to the learned UNKNOWN row.
    """
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame).with_columns(
        pl.when(pl.col("inspection_date").str.to_date() > pl.lit(fold.train_end))
        .then(pl.lit("FUTURE_ONLY_CHAIN"))
        .otherwise(pl.col("chain_key"))
        .alias("chain_key")
    )
    planted = cats.filter(pl.col("chain_key") == "FUTURE_ONLY_CHAIN").height
    assert planted > 0, "no future-only category was planted; the test would be vacuous"

    fitted = _fit(frame, fold, PRIMARY, cats)
    vocabulary = fitted.encoding.vocabulary_for("chain").categories
    assert "FUTURE_ONLY_CHAIN" not in vocabulary, (
        "a category that appears only after train_end acquired its own embedding row"
    )

    window = folds_module.window_frame(frame, fold)
    joined = window.join(
        cats.select("target_inspection_id", "chain_key", "facility_type", "community_area", "zip"),
        on="target_inspection_id",
        how="left",
    )
    codes = encode.index_matrix(joined, PRIMARY, fitted.encoding)
    position = PRIMARY.entity_columns.index("chain")
    vocabulary_list = list(fitted.encoding.vocabulary_for("chain").categories)
    independent = vocabulary_list.index(INDEPENDENT_CHAIN)

    # An unseen *name* is not "unknown" -- it is "not a chain this fold knows about",
    # which is a different and more informative statement. UNKNOWN is reserved for a row
    # with no prior inspection to carry any value forward from. What matters for leakage
    # is that the unseen name got no row of its own, which is asserted above.
    assert (codes[:, position] == independent).all(), (
        "test rows carrying a name unseen in training did not fall back to the shared "
        f"{INDEPENDENT_CHAIN} row"
    )
    assert independent != encode.UNKNOWN_INDEX


def test_chain_membership_is_derived_inside_the_fold() -> None:
    """A name that becomes shared only later must not be a chain earlier.

    This is the failure that would leave every metric looking normal: an establishment
    marked as chained on the strength of a sibling location that does not exist yet.
    """
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)

    # Make one name shared by two establishments, but only after train_end.
    late = cats.filter(pl.col("inspection_date").str.to_date() > pl.lit(fold.train_end))
    assert late.height > 2
    late_ids = set(late["target_inspection_id"].to_list()[:40])
    planted = cats.with_columns(
        pl.when(pl.col("target_inspection_id").is_in(list(late_ids)))
        .then(pl.lit("LATE_CHAIN"))
        .otherwise(pl.col("chain_key"))
        .alias("chain_key")
    )

    fitted = _fit(frame, fold, PRIMARY, planted)
    assert "LATE_CHAIN" not in fitted.encoding.chains, (
        "a name that only became shared after train_end was treated as a chain"
    )
    assert "LATE_CHAIN" not in fitted.encoding.vocabulary_for("chain").categories


def test_the_vocabulary_re_derives_from_the_inner_training_window() -> None:
    """Every embedding row must correspond to a category in the fitting window."""
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    fitted = _fit(frame, fold, PRIMARY, cats)

    source = frame.filter(
        (pl.col("rd") >= fold.train_start) & (pl.col("rd") < fitted.inner_validation_start)
    ).join(
        cats.select("target_inspection_id", "chain_key", "facility_type", "community_area", "zip"),
        on="target_inspection_id",
        how="left",
    )
    resolved = encode.resolve_categories(source, fitted.encoding.chains)

    for vocab in fitted.encoding.vocabularies:
        observed = {str(v) for v in resolved[vocab.column].drop_nulls().to_list()}
        observed.add(UNKNOWN_CATEGORY)
        unexplained = set(vocab.categories) - observed
        assert not unexplained, (
            f"{vocab.column}: {len(unexplained)} vocabulary entr(y/ies) are not present in "
            f"the window the vocabulary was fitted on, e.g. {sorted(unexplained)[:3]}"
        )
        assert vocab.categories[0] == UNKNOWN_CATEGORY


def test_vocabulary_order_is_sorted_not_insertion_ordered() -> None:
    """Insertion order is row order, and row order must never reach a fitted artifact."""
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold)
    for vocab in fitted.encoding.vocabularies:
        rest = list(vocab.categories[1:])
        assert rest == sorted(rest), f"{vocab.column}: vocabulary is not sorted"


# --- 5. the embeddings-into-XGBoost experiment -------------------------------


def test_the_booster_refuses_an_embedding_table_from_another_fold() -> None:
    """The experiment's entire temporal guarantee, driven rather than trusted."""
    frame = _base()
    built = folds_module.quarterly_folds(
        data_start=folds_module.min_date(frame, "rd"),
        data_end=folds_module.max_date(frame, "rd"),
    )
    assert len(built) >= 2, "need two folds to cross the donor over"
    first, second = built[0], built[1]
    cats = neural_categoricals_for(frame)

    donor = _fit(frame, first, PRIMARY, cats)
    spec = spec_for("xgboost_chain_embeddings")
    with pytest.raises(embed.EmbedError, match="was fitted on fold"):
        embed.fit_fold(
            spec,
            training_frame(frame, second),
            second,
            donor=donor,
            categoricals=cats,
        )


def test_embedding_fed_boosting_is_unchanged_by_future_rows() -> None:
    """The whole experiment, refitted with the future mutated."""
    frame = _base()
    fold = _fold(frame)
    spec = spec_for("xgboost_chain_embeddings")

    def run(table: pl.DataFrame) -> list[float]:
        cats = neural_categoricals_for(table)
        donor = _fit(table, fold, PRIMARY, cats)
        fitted = embed.fit_fold(
            spec, training_frame(table, fold), fold, donor=donor, categoricals=cats
        )
        _, scores = embed.score_window(
            fitted,
            folds_module.window_frame(table, fold),
            donor=donor,
            categoricals=cats,
        )
        return scores

    before = run(frame)
    after = run(_future_frame(frame, fold, target=1))
    assert before == after, (
        "the embedding-fed booster's predictions moved when rows after train_end changed"
    )


def test_the_augmented_matrix_is_the_tree_matrix_plus_the_embedding_block() -> None:
    """Column identity and order, so an importance can never be mislabelled."""
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    donor = _fit(frame, fold, PRIMARY, cats)
    spec = spec_for("xgboost_chain_embeddings")

    columns = embed.augmented_columns(spec, donor)
    base = preprocess.matrix_columns(spec)
    assert columns[: len(base)] == base, "the tree matrix's columns were reordered"
    extra = columns[len(base) :]
    assert len(extra) == donor.embedding_for("chain").dim == 16
    assert list(extra) == sorted(extra), "embedding column names are not sortable in order"


# --- 6. determinism and ordering ---------------------------------------------


def test_matrix_column_order_is_deterministic_across_fits() -> None:
    """Feature ordering must be a property of the spec, not of a run."""
    frame = _base()
    fold = _fold(frame)
    first = _fit(frame, fold)
    second = _fit(frame, fold)
    assert first.matrix_columns == second.matrix_columns
    assert first.encoding.columns == second.encoding.columns == PRIMARY.entity_columns


def test_two_fits_of_the_same_fold_are_bit_identical() -> None:
    """The determinism claim, driven. Seeded generator, one thread, CPU."""
    frame = _base()
    fold = _fold(frame)
    assert _scores(frame, fold) == _scores(frame, fold)


def test_shuffling_the_input_row_order_does_not_change_the_fit() -> None:
    """``fit_fold`` re-sorts canonically, so a caller's order cannot reach the weights."""
    frame = _base()
    fold = _fold(frame)
    before = _scores(frame, fold)
    shuffled = frame.sort("target_inspection_id", descending=True)
    after = _scores(shuffled, fold)
    assert before == after, "the caller's row order reached the fitted network"


def test_every_model_scores_the_same_rows() -> None:
    """The alignment that makes the C6/C7/C8 comparison meaningful.

    Every Component 8 model must score exactly the test window's
    ``target_inspection_id`` set -- no more, no fewer, in the same order.
    """
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    window = folds_module.window_frame(frame, fold)
    expected = [str(v) for v in window["target_inspection_id"].to_list()]

    for name in ("neural_embeddings", "neural_numeric_only", "neural_onehot"):
        spec = spec_for(name)
        fitted = _fit(frame, fold, spec, cats)
        ids, scores = predict.score_window(fitted, window, categoricals=cats)
        assert ids == expected, f"{name} scored a different row set or a different order"
        assert len(scores) == len(expected)


def test_the_numeric_only_model_sees_exactly_the_component_7_matrix() -> None:
    """The fair-comparison control must not gain a column from the categorical layer."""
    frame = _base()
    fold = _fold(frame)
    fitted = _fit(frame, fold, NUMERIC_ONLY)

    assert NUMERIC_ONLY.encoding is CategoricalEncoding.NONE
    assert NUMERIC_ONLY.entity_columns == ()
    assert fitted.embedding_width == 0
    assert fitted.encoding.vocabularies == ()
    assert fitted.dense_width == len(preprocess.matrix_columns(NUMERIC_ONLY)) == 30


# --- 7. the detector itself works --------------------------------------------


def test_the_leakage_detector_itself_works() -> None:
    """Plant the label in a feature and confirm the fixture reveals it.

    Every test above asserts that something did *not* change. All of them would pass
    against a model that learns nothing at all. This one proves the fixture and the
    pipeline can transmit a signal, which is what makes the rest meaningful.
    """
    frame = _base().with_columns(
        pl.col("target").cast(pl.Float64).alias("prior_canvass_priority_rate")
    )
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    fitted = _fit(frame, fold, NUMERIC_ONLY, cats)
    window = folds_module.window_frame(frame, fold)
    _, scores = predict.score_window(fitted, window, categoricals=cats)

    labels = [int(v) for v in window["target"].to_list()]
    positives = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    negatives = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    assert positives and negatives, "the planted window has only one class"
    assert min(positives) > max(negatives), (
        "a planted label did not separate the classes, so the leakage tests above are "
        "measuring a weak model rather than a protected pipeline"
    )


def test_the_categorical_detector_itself_works() -> None:
    """The same proof for the embedding path: a planted category must be learnable.

    Without this, ``test_a_test_only_category_never_enters_the_vocabulary`` could pass
    because embeddings do nothing at all rather than because the vocabulary is protected.
    """
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame).join(
        frame.select("target_inspection_id", "target"), on="target_inspection_id", how="left"
    )
    # A category that IS the label, present in training. The vocabulary must contain both
    # values -- which is the mechanism the protective tests rely on working.
    planted = cats.with_columns(
        pl.when(pl.col("target") == 1)
        .then(pl.lit("LABEL_ONE"))
        .otherwise(pl.lit("LABEL_ZERO"))
        .alias("facility_type")
    ).drop("target")

    fitted = _fit(frame, fold, PRIMARY, planted)
    vocabulary = set(fitted.encoding.vocabulary_for("facility_type").categories)
    assert {"LABEL_ONE", "LABEL_ZERO"} <= vocabulary, (
        "a category present throughout the training window did not reach the vocabulary, "
        "so the vocabulary tests above are not testing what they claim"
    )


def test_a_non_strict_as_of_source_is_detectable() -> None:
    """The categorical validator must reject a value carried from the row's own date."""
    from sentinel.neural import validate as neural_validate

    frame = _base()
    cats = neural_categoricals_for(frame)
    broken = cats.with_columns(
        pl.col("inspection_date").str.to_date().alias("source_inspection_date")
    )
    check = next(
        c
        for c in neural_validate.validate_categoricals(frame, broken)
        if c.name == "categoricals_are_strictly_as_of"
    )
    assert not check.passed, (
        "a categorical carried from the row's own inspection date was accepted, so the "
        "as-of check cannot detect the failure it exists for"
    )
    good = next(
        c
        for c in neural_validate.validate_categoricals(frame, cats)
        if c.name == "categoricals_are_strictly_as_of"
    )
    assert good.passed


def test_a_future_date_in_the_categorical_layer_is_detectable() -> None:
    """And the same for a source date after the row's own date."""
    from sentinel.neural import validate as neural_validate

    frame = _base()
    cats = neural_categoricals_for(frame)
    broken = cats.with_columns(
        (pl.col("inspection_date").str.to_date() + pl.duration(days=1)).alias(
            "source_inspection_date"
        )
    )
    check = next(
        c
        for c in neural_validate.validate_categoricals(frame, broken)
        if c.name == "categoricals_are_strictly_as_of"
    )
    assert not check.passed


def test_the_fixture_actually_contains_chains() -> None:
    """A vacuity guard for every chain test above.

    If the fixture produced no shared names, ``chain_membership`` would return an empty
    set, every chain test would trivially pass, and the ablation would be measuring
    nothing.
    """
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    joined = training_frame(frame, fold).join(
        cats.select("target_inspection_id", "chain_key", "establishment_id"),
        on="target_inspection_id",
        how="left",
        suffix="_cat",
    )
    chains = encode.chain_membership(joined)
    assert len(chains) >= 4, (
        f"fixture yielded {len(chains)} chain(s); the chain tests would be weak or vacuous"
    )


def test_the_fixture_produces_enough_folds() -> None:
    """A vacuity guard for the fold-crossing test."""
    frame = _base()
    built = folds_module.quarterly_folds(
        data_start=folds_module.min_date(frame, "rd"),
        data_end=folds_module.max_date(frame, "rd"),
    )
    assert len(built) >= 2, f"fixture yielded {len(built)} fold(s); loops would be weak"
    for fold in built:
        assert fold.train_end < fold.calibration_start < fold.test_start


def test_a_date_gap_separates_training_from_test() -> None:
    """Restated here because every test in this file depends on it."""
    frame = _base()
    fold = _fold(frame)
    assert fold.train_end < fold.calibration_start
    assert fold.calibration_end < fold.test_start
    assert (fold.test_start - fold.train_end) > timedelta(days=1)
    assert isinstance(fold.train_end, date)
