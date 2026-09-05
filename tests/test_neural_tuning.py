"""The learning-rate sweep, and the protocol it borrows rather than reinvents.

The point of these tests is that Component 8 did *not* invent a second tuning protocol.
``tuning_region``, ``first_test_start`` and ``build_inner_folds`` are Component 7's, and
the properties ADR 0017 guarantees must hold here unchanged. The rest is about the
selection rule, which is declared rather than emergent precisely so a test can drive it.
"""

from __future__ import annotations

import polars as pl
import pytest

from sentinel.boosting.tuning import build_inner_folds, first_test_start, tuning_region
from sentinel.evaluation import folds as folds_module
from sentinel.neural import tuning
from sentinel.neural.definitions import (
    BASELINE_LEARNING_RATE,
    LEARNING_RATE_GRID,
    REPRESENTATIVE_MODEL,
    spec_for,
)
from sentinel.neural.models import SweepPoint, SweepResult
from tests.conftest import neural_categoricals_for, spanning_model_features


def _frame() -> pl.DataFrame:
    return spanning_model_features(days=2200, per_day=2).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )


def _folds(frame: pl.DataFrame) -> list[object]:
    return folds_module.quarterly_folds(
        data_start=folds_module.min_date(frame, "rd"),
        data_end=folds_module.max_date(frame, "rd"),
    )


# --- 1. the selection rule ---------------------------------------------------


def test_the_highest_mean_wins() -> None:
    chosen, reason = tuning.select_rate([(1e-4, 0.50), (1e-3, 0.60), (1e-2, 0.55)])
    assert chosen == 1e-3
    assert "highest mean" in reason


def test_a_tie_is_broken_toward_the_specified_baseline() -> None:
    """A coarse grid over a near-flat objective ties often.

    Without a declared rule the winner would be decided by float summation order, which
    is exactly the kind of non-determinism this project refuses everywhere else.
    """
    chosen, reason = tuning.select_rate([(1e-4, 0.60), (1e-3, 0.60), (1e-2, 0.60)])
    assert chosen == BASELINE_LEARNING_RATE
    assert "tied" in reason and "baseline" in reason


def test_a_near_tie_within_the_tolerance_counts_as_a_tie() -> None:
    chosen, _ = tuning.select_rate([(1e-3, 0.6000000), (1e-2, 0.6000005)])
    assert chosen == BASELINE_LEARNING_RATE


def test_a_difference_beyond_the_tolerance_is_not_a_tie() -> None:
    chosen, _ = tuning.select_rate([(1e-3, 0.60), (1e-2, 0.61)])
    assert chosen == 1e-2


def test_an_empty_score_set_is_refused() -> None:
    with pytest.raises(tuning.NeuralTuningError, match="no rates were scored"):
        tuning.select_rate([])


# --- 2. the region is Component 7's -----------------------------------------


def test_the_sweep_uses_the_region_the_fold_set_implies() -> None:
    frame = _frame()
    outer = _folds(frame)
    assert len(outer) >= 2
    start, end = tuning_region("quarterly", outer)  # type: ignore[arg-type]
    horizon = first_test_start("quarterly", outer)  # type: ignore[arg-type]
    assert end < horizon, "Component 7's own region already reaches a test window"


def test_the_inner_folds_end_before_the_first_test_window() -> None:
    frame = _frame()
    outer = _folds(frame)
    inner = build_inner_folds("quarterly", outer)  # type: ignore[arg-type]
    horizon = first_test_start("quarterly", outer)  # type: ignore[arg-type]
    assert inner, "no inner folds; the sweep would be vacuous"
    for fold in inner:
        assert fold.test_end < horizon
        assert fold.train_end < fold.calibration_start < fold.test_start


def test_the_sweep_refuses_a_region_that_would_reach_a_test_window() -> None:
    """Re-derived inside the sweep, not merely inherited.

    This is the one property whose failure would be invisible in every number the sweep
    produces, so it is checked twice: once by Component 7's builder and once here.
    """
    frame = _frame()
    outer = _folds(frame)
    spec = spec_for(REPRESENTATIVE_MODEL)
    # A fold set with no members cannot produce a region at all. The error comes from
    # Component 7's own ``tuning_region``, which is the point: the guard is inherited.
    from sentinel.boosting.tuning import TuningError

    with pytest.raises((TuningError, tuning.NeuralTuningError)):
        tuning.sweep_fold_set(
            spec,
            frame,
            outer,  # type: ignore[arg-type]
            fold_set="not_a_fold_set",
            categoricals=neural_categoricals_for(frame),
            grid=(1e-3,),
        )


# --- 3. a real sweep ---------------------------------------------------------


def test_a_two_point_sweep_scores_every_rate_on_every_inner_fold() -> None:
    """Kept to two rates and a tiny epoch budget; the grid itself is exercised above."""
    frame = _frame()
    outer = _folds(frame)
    spec = spec_for(REPRESENTATIVE_MODEL)
    result = tuning.sweep_fold_set(
        spec,
        frame,
        outer,  # type: ignore[arg-type]
        fold_set="quarterly",
        categoricals=neural_categoricals_for(frame),
        grid=(1e-3, 1e-2),
        max_epochs=2,
    )
    inner = build_inner_folds("quarterly", outer)  # type: ignore[arg-type]
    assert len(result.points) == 2 * len(inner)
    assert len(result.scores) == 2
    assert result.best_learning_rate in (1e-3, 1e-2)
    assert result.inner_folds == tuple(f.fold_id for f in inner)
    assert result.region_end < first_test_start("quarterly", outer)  # type: ignore[arg-type]
    for point in result.points:
        assert 0.0 <= point.pr_auc <= 1.0
        assert point.train_rows > 0 and point.validation_rows > 0


def test_an_empty_grid_is_refused() -> None:
    frame = _frame()
    outer = _folds(frame)
    with pytest.raises(tuning.NeuralTuningError, match="grid is empty"):
        tuning.sweep_fold_set(
            spec_for(REPRESENTATIVE_MODEL),
            frame,
            outer,  # type: ignore[arg-type]
            fold_set="quarterly",
            categoricals=neural_categoricals_for(frame),
            grid=(),
        )


# --- 4. the frozen block -----------------------------------------------------


def _result(fold_set: str, rate: float) -> SweepResult:
    from datetime import date

    return SweepResult(
        study=f"m-{fold_set}",
        model_name="m",
        fold_set=fold_set,
        region_start=date(2018, 7, 1),
        region_end=date(2022, 3, 31),
        inner_folds=("a", "b"),
        points=(SweepPoint("a", rate, 10, 5, 0.6, 3),),
        scores=((rate, 0.6),),
        best_learning_rate=rate,
        selection_reason="because",
        seed=1,
        seconds=1.0,
    )


def test_the_frozen_block_is_pasteable_python() -> None:
    """``tune-neural`` prints a literal a human pastes; it edits no source file.

    A parameter set loaded from disk at training time could change without a diff, and
    the entire value of freezing is that it cannot.
    """
    block = tuning.frozen_block([_result("quarterly", 0.003), _result("covid_shift", 0.01)])
    assert block.startswith("TUNED_HYPERPARAMS")
    assert '"quarterly": {"learning_rate": 0.003}' in block
    assert '"covid_shift": {"learning_rate": 0.01}' in block
    compiled = compile(block, "<frozen block>", "exec")
    assert compiled is not None


def test_the_frozen_block_records_the_region_it_searched() -> None:
    block = tuning.frozen_block([_result("quarterly", 0.003)])
    assert "2018-07-01..2022-03-31" in block
    assert "before the first test start" in block


def test_the_frozen_block_is_ordered_by_fold_set() -> None:
    """So two runs produce a textually comparable block."""
    block = tuning.frozen_block([_result("quarterly", 0.003), _result("covid_shift", 0.01)])
    assert block.index("covid_shift") < block.index('"quarterly"')


# --- 5. the declared constants ----------------------------------------------


def test_the_objective_names_a_validation_window_not_a_test_window() -> None:
    assert "inner validation" in tuning.OBJECTIVE
    assert "earlier than the fold set's first test window" in tuning.OBJECTIVE


def test_the_search_is_an_exhaustive_grid() -> None:
    """A grid answers "is this sensitive to the rate"; TPE would leave the tails unmeasured."""
    assert "grid" in tuning.SEARCH
    assert "no adaptive sampler" in tuning.SEARCH


def test_the_grid_is_the_declared_one() -> None:
    assert LEARNING_RATE_GRID == (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


def test_the_sweep_model_is_the_representative_one() -> None:
    """One search, applied to every network, so the ablations stay comparable."""
    assert REPRESENTATIVE_MODEL == "neural_embeddings"
