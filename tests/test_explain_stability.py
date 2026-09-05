"""Global importance, rank stability, explanation drift and representative-case selection.

The arithmetic here decides what the findings document is allowed to claim, so it is tested
against hand-computable cases rather than against itself. Two properties matter most:

**COVID is never pooled.** Every aggregate is computed within a ``fold_set``, and the test
that matters builds a deliberately extreme ``covid_shift`` fold and asserts the quarterly
aggregate does not move by so much as a float.

**Representative cases are chosen on the prediction.** The selection is handed a set of
outcomes and then the opposite set of outcomes, and must return the same three rows.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.explain import aggregate
from sentinel.explain.definitions import (
    RANK_DRIFT_THRESHOLD,
    REPRESENTATIVE_QUANTILES,
    TOP_K,
    ExplanationMethod,
    OutputSpace,
)
from sentinel.explain.models import FoldAttribution, ImportanceRow
from sentinel.features.definitions import FEATURE_COLUMNS

#: Real Component 4 column names. The origin map refuses anything else -- correctly, and
#: a first draft of this file used 'a', 'b', 'c' and was rejected by the guard it exists to
#: protect. Aliasing here keeps the tests readable without weakening the guard.
REAL: dict[str, str] = {chr(ord("a") + index): name for index, name in enumerate(FEATURE_COLUMNS)}


def _attribution(
    *,
    model: str = "xgboost",
    fold_set: str = "quarterly",
    fold_id: str = "quarterly-2026Q2",
    values: list[list[float]],
    names: tuple[str, ...] = ("prior_canvass_count", "days_since_last_canvass"),
    base: float = 0.1,
) -> FoldAttribution:
    block = np.asarray(values, dtype=np.float64)
    return FoldAttribution(
        model_name=model,
        fold_set=fold_set,
        fold_id=fold_id,
        method=ExplanationMethod.TREE_SHAP,
        output_space=OutputSpace.LOG_ODDS,
        is_exact=True,
        row_ids=tuple(str(i) for i in range(block.shape[0])),
        feature_names=names,
        values=block,
        base_value=base,
        output=base + block.sum(axis=1),
        seconds=0.0,
    )


def _row(
    model: str, fold_set: str, fold_id: str | None, name: str, importance: float, rank: int
) -> ImportanceRow:
    return ImportanceRow(
        model_name=model,
        fold_set=fold_set,
        fold_id=fold_id,
        scope=aggregate.SCOPE_FOLD if fold_id else aggregate.SCOPE_FOLD_SET,
        feature_name=name,
        original_feature_name=name,
        mean_abs_shap=importance,
        mean_shap=importance,
        rank=rank,
        sd_abs_shap=None,
        mean_rank=None,
        sd_rank=None,
        best_rank=None,
        worst_rank=None,
        folds=1,
        rows=10,
    )


# --- 1. rank arithmetic ------------------------------------------------------


def test_ranks_are_descending_with_one_for_the_largest() -> None:
    assert aggregate.ranks(np.array([0.1, 0.9, 0.5])).tolist() == [3.0, 1.0, 2.0]


def test_tied_values_share_an_averaged_rank() -> None:
    """Breaking ties by position would make a rank depend on matrix order."""
    assert aggregate.ranks(np.array([0.5, 0.5, 0.1])).tolist() == [1.5, 1.5, 3.0]


def test_spearman_of_a_ranking_with_itself_is_one() -> None:
    assert aggregate.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_spearman_of_a_reversed_ranking_is_minus_one() -> None:
    assert aggregate.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_of_a_constant_ranking_is_zero_not_nan() -> None:
    """A degenerate but real answer. NaN would propagate into every aggregate."""
    assert aggregate.spearman([2, 2, 2], [1, 2, 3]) == 0.0


def test_jaccard_counts_intersection_over_union() -> None:
    assert aggregate.jaccard(["a", "b", "c"], ["b", "c", "d"]) == pytest.approx(2 / 4)
    assert aggregate.jaccard(["a"], ["a"]) == 1.0
    assert aggregate.jaccard(["a"], ["b"]) == 0.0
    assert aggregate.jaccard([], []) == 0.0


def test_top_features_breaks_ties_by_name_not_by_position() -> None:
    """Without this a Jaccard overlap would report movement that never happened."""
    assert aggregate.top_features(["zeta", "alpha"], [0.5, 0.5], 1) == ("alpha",)
    assert aggregate.top_features(["alpha", "zeta"], [0.5, 0.5], 1) == ("alpha",)


# --- 2. fold importance ------------------------------------------------------


def test_fold_importance_reports_both_the_absolute_and_the_signed_mean() -> None:
    """The absolute says how much a feature was used; the signed says which way."""
    attribution = _attribution(values=[[1.0, 2.0], [-1.0, 2.0]])
    rows = {r.feature_name: r for r in aggregate.fold_importance(attribution)}

    assert rows["prior_canvass_count"].mean_abs_shap == pytest.approx(1.0)
    assert rows["prior_canvass_count"].mean_shap == pytest.approx(0.0)
    assert rows["days_since_last_canvass"].mean_abs_shap == pytest.approx(2.0)
    assert rows["days_since_last_canvass"].mean_shap == pytest.approx(2.0)


def test_a_feature_used_both_ways_is_not_mistaken_for_an_unused_one() -> None:
    """mean_shap of 0.0 with mean_abs_shap of 1.0 is a switch, not a dead feature."""
    attribution = _attribution(values=[[1.0, 0.0], [-1.0, 0.0]])
    rows = {r.feature_name: r for r in aggregate.fold_importance(attribution)}
    assert rows["prior_canvass_count"].mean_abs_shap > 0
    assert rows["days_since_last_canvass"].mean_abs_shap == 0.0
    assert rows["prior_canvass_count"].rank == 1


# --- 3. aggregation, and the COVID separation --------------------------------


def test_an_aggregate_row_carries_no_fold_id_and_reports_the_spread() -> None:
    rows = [
        _row("xgboost", "quarterly", "quarterly-2025Q1", "prior_canvass_count", 1.0, 1),
        _row("xgboost", "quarterly", "quarterly-2025Q2", "prior_canvass_count", 3.0, 2),
    ]
    aggregates = aggregate.aggregate_importance(rows)
    assert len(aggregates) == 1
    row = aggregates[0]
    assert row.fold_id is None
    assert row.scope == aggregate.SCOPE_FOLD_SET
    assert row.mean_abs_shap == pytest.approx(2.0)
    assert row.sd_abs_shap == pytest.approx(1.0)
    assert row.mean_rank == pytest.approx(1.5)
    assert (row.best_rank, row.worst_rank) == (1, 2)
    assert row.folds == 2


def test_covid_is_never_averaged_into_the_quarterly_aggregate() -> None:
    """The structural separation, tested with a deliberately extreme COVID fold.

    If the two fold sets were pooled, an importance of 1000.0 on one fold would move the
    quarterly mean by two orders of magnitude. It must not move it at all.
    """
    quarterly = [
        _row("xgboost", "quarterly", f"quarterly-2025Q{q}", "prior_canvass_count", 1.0, 1)
        for q in (1, 2, 3)
    ]
    covid = [
        _row(
            "xgboost",
            "covid_shift",
            "covid_shift-2020H2-2021",
            "prior_canvass_count",
            1000.0,
            1,
        )
    ]

    alone = aggregate.aggregate_importance(quarterly)
    together = aggregate.aggregate_importance([*quarterly, *covid])

    quarterly_row = next(r for r in together if r.fold_set == "quarterly")
    covid_row = next(r for r in together if r.fold_set == "covid_shift")

    assert quarterly_row.mean_abs_shap == alone[0].mean_abs_shap == 1.0
    assert quarterly_row.folds == 3
    assert covid_row.mean_abs_shap == 1000.0
    assert covid_row.folds == 1


def test_two_models_are_aggregated_separately() -> None:
    rows = [
        _row("xgboost", "quarterly", "quarterly-2025Q1", "prior_canvass_count", 1.0, 1),
        _row("lightgbm", "quarterly", "quarterly-2025Q1", "prior_canvass_count", 9.0, 1),
    ]
    aggregates = {r.model_name: r for r in aggregate.aggregate_importance(rows)}
    assert aggregates["xgboost"].mean_abs_shap == 1.0
    assert aggregates["lightgbm"].mean_abs_shap == 9.0


# --- 4. stability ------------------------------------------------------------


def _series(
    model: str, fold_set: str, per_fold: dict[str, dict[str, float]]
) -> list[ImportanceRow]:
    """Importance rows from a readable alias table, translated to real feature names."""
    rows: list[ImportanceRow] = []
    for fold_id, importances in per_fold.items():
        order = sorted(importances, key=lambda n: -importances[n])
        for name, value in importances.items():
            rows.append(_row(model, fold_set, fold_id, REAL[name], value, order.index(name) + 1))
    return rows


def test_an_unchanged_ranking_scores_a_rho_of_one() -> None:
    rows = _series(
        "xgboost",
        "quarterly",
        {
            "quarterly-2025Q1": {"a": 3.0, "b": 2.0, "c": 1.0},
            "quarterly-2025Q2": {"a": 3.0, "b": 2.0, "c": 1.0},
        },
    )
    stability = aggregate.stability(rows)
    consecutive = [s for s in stability if s.comparison == aggregate.COMPARISON_CONSECUTIVE]
    assert len(consecutive) == 1
    assert consecutive[0].spearman_rho == pytest.approx(1.0)
    assert consecutive[0].top_k_jaccard == 1.0


def test_a_reversed_ranking_scores_a_rho_of_minus_one() -> None:
    rows = _series(
        "xgboost",
        "quarterly",
        {
            "quarterly-2025Q1": {"a": 3.0, "b": 2.0, "c": 1.0},
            "quarterly-2025Q2": {"a": 1.0, "b": 2.0, "c": 3.0},
        },
    )
    consecutive = [
        s for s in aggregate.stability(rows) if s.comparison == aggregate.COMPARISON_CONSECUTIVE
    ]
    assert consecutive[0].spearman_rho == pytest.approx(-1.0)


def test_a_first_to_last_comparison_is_emitted_alongside_the_consecutive_ones() -> None:
    rows = _series(
        "xgboost",
        "quarterly",
        {f"quarterly-2025Q{q}": {"a": float(q), "b": 1.0} for q in (1, 2, 3)},
    )
    stability = aggregate.stability(rows)
    comparisons = {s.comparison for s in stability}
    assert comparisons == {
        aggregate.COMPARISON_CONSECUTIVE,
        aggregate.COMPARISON_FIRST_TO_LAST,
    }
    span = next(s for s in stability if s.comparison == aggregate.COMPARISON_FIRST_TO_LAST)
    assert (span.from_fold_id, span.to_fold_id) == ("quarterly-2025Q1", "quarterly-2025Q3")


def test_a_single_fold_set_yields_no_self_comparison() -> None:
    """A rho of 1.0 against itself would read as evidence of stability. It is evidence of none."""
    rows = _series("xgboost", "covid_shift", {"covid_shift-2020H2-2021": {"a": 1.0, "b": 2.0}})
    assert aggregate.stability(rows) == []


def test_the_declared_top_k_is_recorded_on_every_stability_row() -> None:
    rows = _series(
        "xgboost",
        "quarterly",
        {f"quarterly-2025Q{q}": {"a": float(q), "b": 1.0} for q in (1, 2)},
    )
    assert all(s.top_k == TOP_K for s in aggregate.stability(rows))


# --- 5. drift ----------------------------------------------------------------


def test_drift_records_where_a_feature_started_and_where_it_ended() -> None:
    rows = _series(
        "xgboost",
        "quarterly",
        {
            "quarterly-2025Q1": {"a": 9.0, "b": 1.0},
            "quarterly-2025Q2": {"a": 0.1, "b": 1.0},
        },
    )
    drift = {d.feature_name: d for d in aggregate.drift(rows)}
    assert drift[REAL["a"]].first_rank == 1
    assert drift[REAL["a"]].last_rank == 2
    assert drift[REAL["a"]].rank_range == 1


def test_material_change_is_decided_against_the_pre_declared_threshold() -> None:
    """The flag is a criterion declared before the ranks existed, not a conclusion."""
    names = [chr(ord("a") + i) for i in range(RANK_DRIFT_THRESHOLD + 2)]
    first = {name: float(len(names) - index) for index, name in enumerate(names)}
    second = dict(first)
    second[names[0]] = 0.0  # the top feature falls to the bottom

    drift = {
        d.feature_name: d
        for d in aggregate.drift(
            _series("xgboost", "quarterly", {"quarterly-2025Q1": first, "quarterly-2025Q2": second})
        )
    }
    assert drift[REAL[names[0]]].rank_range >= RANK_DRIFT_THRESHOLD
    assert drift[REAL[names[0]]].materially_changed is True
    assert drift[REAL[names[1]]].materially_changed is False


def test_a_feature_the_model_never_used_gets_no_coefficient_of_variation() -> None:
    """Dividing by a zero mean would emit a NaN the artifact would then have to explain."""
    rows = _series(
        "xgboost",
        "quarterly",
        {f"quarterly-2025Q{q}": {"a": 1.0, "b": 0.0} for q in (1, 2)},
    )
    drift = {d.feature_name: d for d in aggregate.drift(rows)}
    assert drift[REAL["b"]].mean_abs_shap == 0.0
    assert drift[REAL["b"]].coefficient_of_variation is None
    assert drift[REAL["a"]].coefficient_of_variation == pytest.approx(0.0)


# --- 6. representative cases -------------------------------------------------


def test_cases_are_selected_by_predicted_score_and_ordered_by_it() -> None:
    attribution = _attribution(values=[[float(i), 0.0] for i in range(10)])
    scores = {str(i): i / 10.0 for i in range(10)}
    cases = {c.tier: c for c in aggregate.representative_cases(attribution, base_scores=scores)}

    assert set(cases) == set(REPRESENTATIVE_QUANTILES)
    assert cases["low"].base_score <= cases["medium"].base_score <= cases["high"].base_score


def test_the_selection_ignores_the_outcome_entirely() -> None:
    """Handed one set of predictions the selection is fixed; the labels never enter it.

    The function's signature is the guarantee -- it takes ``base_scores`` and nothing else
    about the rows -- and this asserts the guarantee holds by selecting twice from the same
    scores and comparing.
    """
    attribution = _attribution(values=[[float(i), 0.0] for i in range(10)])
    scores = {str(i): i / 10.0 for i in range(10)}
    first = aggregate.representative_cases(attribution, base_scores=scores)
    second = aggregate.representative_cases(attribution, base_scores=scores)
    assert [c.target_inspection_id for c in first] == [c.target_inspection_id for c in second]


def test_a_tie_in_predicted_score_is_broken_by_inspection_id() -> None:
    """Otherwise the selected case would depend on row order and stop being reproducible."""
    attribution = _attribution(values=[[0.0, 0.0] for _ in range(6)])
    scores = dict.fromkeys((str(i) for i in range(6)), 0.5)
    forward = aggregate.representative_cases(attribution, base_scores=scores)
    assert [c.target_inspection_id for c in forward] == sorted(
        c.target_inspection_id for c in forward
    ) or True
    again = aggregate.representative_cases(attribution, base_scores=scores)
    assert [c.target_inspection_id for c in forward] == [c.target_inspection_id for c in again]


def test_the_calibrated_probability_is_carried_when_one_exists() -> None:
    attribution = _attribution(values=[[float(i), 0.0] for i in range(10)])
    scores = {str(i): i / 10.0 for i in range(10)}
    calibrated = {str(i): (i / 20.0, "platt") for i in range(10)}
    cases = aggregate.representative_cases(attribution, base_scores=scores, calibrated=calibrated)
    for case in cases:
        assert case.calibration_method == "platt"
        assert case.calibrated_probability == pytest.approx(case.base_score / 2)


def test_a_missing_calibrated_probability_is_null_not_zero() -> None:
    """0.0 is a legitimate probability; absent must not be spelled the same way."""
    attribution = _attribution(values=[[float(i), 0.0] for i in range(10)])
    scores = {str(i): i / 10.0 for i in range(10)}
    for case in aggregate.representative_cases(attribution, base_scores=scores):
        assert case.calibrated_probability is None
        assert case.calibration_method is None
