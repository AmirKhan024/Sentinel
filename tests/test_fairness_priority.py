"""The top-k audit: who gets prioritised, and what that prioritisation was worth to them.

Two quantities live in this module and the tests keep them apart on purpose:

```text
selection rate   representation -- was this group prioritised?
capture rate     effectiveness  -- was that prioritisation useful?
```

A group can be over-represented in the top k while the ranking finds a smaller share of its
violations than average. Every test that touches one checks the other has not moved with it.

The selection is **city-wide and competitive**: the cutoff is taken over every audited row and
groups are counted inside it. That is what makes capture different from ``recall_at_k``, and
the difference is the effect this audit exists to find.
"""

from __future__ import annotations

import polars as pl
import pytest

from sentinel.fairness import priority
from sentinel.fairness.definitions import Grain, GroupStatus, Stage, group_definition_for
from sentinel.fairness.models import GroupSupport

COMMUNITY_AREA = group_definition_for("community_area")


def _frame(rows: list[tuple[str, str, int, float]]) -> pl.DataFrame:
    """``(id, group, label, score)`` laid out explicitly, so every test is readable."""
    return pl.DataFrame(
        {
            "target_inspection_id": [r[0] for r in rows],
            "community_area": [r[1] for r in rows],
            "target": [r[2] for r in rows],
            "calibrated_probability": [r[3] for r in rows],
        }
    )


def _support(**counts: tuple[int, int]) -> dict[str, GroupSupport]:
    """Support records forced to SUPPORTED, so a test isolates the priority arithmetic."""
    return {
        value: GroupSupport(
            group_definition="community_area",
            group_value=value,
            grain=Grain.FOLD_SET.value,
            fold_set="quarterly",
            fold_id="",
            n_rows=n_rows,
            n_positive=n_positive,
            n_negative=n_rows - n_positive,
            base_rate=n_positive / n_rows if n_rows else None,
            representation_share=1.0,
            ranking_status=GroupStatus.SUPPORTED,
            calibration_status=GroupStatus.SUPPORTED,
            insufficient_reason="",
        )
        for value, (n_rows, n_positive) in counts.items()
    }


def _audit(frame: pl.DataFrame, support: dict[str, GroupSupport], k: int) -> dict[str, object]:
    rows = priority.audit(
        frame,
        COMMUNITY_AREA,
        support,
        model_name="m",
        stage=Stage.CALIBRATED,
        score_column="calibrated_probability",
        grain=Grain.FOLD_SET,
        fold_set="quarterly",
        fold_id="",
        k_name="k_pct_05",
        k=k,
    )
    return {row.group_value: row for row in rows}


# --- 1. top-k selection is deterministic and canonically tie-broken -------------


def test_selection_takes_the_highest_scores() -> None:
    frame = _frame(
        [("T003", "A", 1, 0.9), ("T002", "A", 0, 0.2), ("T001", "B", 1, 0.8), ("T000", "B", 0, 0.1)]
    )
    chosen = priority.select_top_k(frame, "calibrated_probability", 2)
    assert set(chosen["target_inspection_id"].to_list()) == {"T003", "T001"}


def test_ties_are_settled_on_the_canonical_tie_break_not_on_frame_order() -> None:
    """Two runs over the same rows shuffled must select the same establishments.

    Parquet row order is not a contract, and this reuses Component 5's `top_k_indices`
    rather than adding a second tie-breaking rule to the project.
    """
    rows = [("T003", "A", 1, 0.5), ("T002", "A", 0, 0.5), ("T001", "B", 1, 0.5)]
    ordered = priority.select_top_k(_frame(rows), "calibrated_probability", 2)
    shuffled = priority.select_top_k(_frame(list(reversed(rows))), "calibrated_probability", 2)
    assert sorted(ordered["target_inspection_id"].to_list()) == ["T001", "T002"]
    assert sorted(shuffled["target_inspection_id"].to_list()) == ["T001", "T002"]


def test_a_k_below_one_is_rejected() -> None:
    with pytest.raises(priority.PriorityError, match="k must be at least 1"):
        priority.select_top_k(_frame([("T000", "A", 1, 0.5)]), "calibrated_probability", 0)


# --- 2. representation --------------------------------------------------------------


def test_selection_rate_and_ratio_match_arithmetic_done_by_hand() -> None:
    """Ten rows: six in A, four in B. B's best outscores everything; then A's four best.

    A: 4 of 6 selected -> rate 4/6 = 0.667; overall rate 5/10 = 0.5; ratio 1.333
    B: 1 of 4 selected -> rate 0.25;                                  ratio 0.5
    """
    rows = [(f"T{i:03d}", "A", i % 2, 0.9 - i * 0.01) for i in range(6)]
    # B's top row clears every A row; its other three sit below all of them.
    rows += [("T006", "B", 0, 0.95)]
    rows += [(f"T{i:03d}", "B", i % 2, 0.5 - i * 0.01) for i in range(7, 10)]
    audited = _audit(_frame(rows), _support(A=(6, 3), B=(4, 2)), 5)

    assert audited["A"].n_selected == 4
    assert audited["A"].selection_rate == pytest.approx(4 / 6)
    assert audited["A"].selection_rate_ratio == pytest.approx((4 / 6) / 0.5)
    assert audited["B"].n_selected == 1
    assert audited["B"].selection_rate_ratio == pytest.approx(0.25 / 0.5)


def test_population_and_selected_shares_each_sum_to_one() -> None:
    rows = [(f"T{i:03d}", "A" if i < 6 else "B", i % 2, 0.9 - i * 0.01) for i in range(10)]
    audited = _audit(_frame(rows), _support(A=(6, 3), B=(4, 2)), 5)
    assert sum(r.population_share for r in audited.values()) == pytest.approx(1.0)  # type: ignore[attr-defined]
    assert sum(r.selected_share for r in audited.values()) == pytest.approx(1.0)  # type: ignore[attr-defined]


def test_a_group_that_places_nobody_in_the_top_k_still_gets_a_row() -> None:
    """The most interesting row in the table, and the one an inner join would delete."""
    rows = [(f"T{i:03d}", "A", 1, 0.9) for i in range(4)]
    rows += [(f"T{i:03d}", "B", 0, 0.1) for i in range(4, 8)]
    audited = _audit(_frame(rows), _support(A=(4, 4), B=(4, 0)), 2)

    assert "B" in audited
    assert audited["B"].n_selected == 0
    assert audited["B"].selection_rate == 0.0
    # None rather than 0.0: "none of the zero rows we picked were positive" is not a
    # precision of zero.
    assert audited["B"].precision_in_selected is None


# --- 3. capture ----------------------------------------------------------------------


def test_capture_is_measured_against_the_groups_own_positives() -> None:
    """A: 3 positives, 2 of them selected -> 2/3. B: 2 positives, none selected -> 0.0."""
    rows = [
        ("T009", "A", 1, 0.95),
        ("T008", "A", 1, 0.90),
        ("T007", "A", 1, 0.20),
        ("T006", "A", 0, 0.10),
        ("T005", "B", 1, 0.15),
        ("T004", "B", 1, 0.05),
    ]
    audited = _audit(_frame(rows), _support(A=(4, 3), B=(2, 2)), 2)
    assert audited["A"].capture_rate == pytest.approx(2 / 3)
    assert audited["B"].capture_rate == pytest.approx(0.0)


def test_a_group_with_no_positives_has_a_null_capture_rate() -> None:
    """None, never 0.0: there was nothing to capture, which is not a failure to capture."""
    rows = [
        ("T003", "A", 1, 0.9),
        ("T002", "A", 1, 0.8),
        ("T001", "B", 0, 0.7),
        ("T000", "B", 0, 0.6),
    ]
    audited = _audit(_frame(rows), _support(A=(2, 2), B=(2, 0)), 2)
    assert audited["B"].capture_rate is None


def test_the_overall_capture_rate_is_carried_on_every_row_as_a_reference() -> None:
    rows = [
        (f"T{i:03d}", "A" if i < 3 else "B", 1 if i < 4 else 0, 0.9 - i * 0.1) for i in range(6)
    ]
    audited = _audit(_frame(rows), _support(A=(3, 3), B=(3, 1)), 3)
    values = {r.overall_capture_rate for r in audited.values()}  # type: ignore[attr-defined]
    assert len(values) == 1


def test_over_representation_and_under_capture_can_happen_together() -> None:
    """The reason the two are never combined into one number.

    Six rows, three in each group. B takes two of the three top slots -- selection rate
    2/3 against an overall 1/2, so a ratio of 1.33 -- and the ranking still finds only half
    of B's violations while finding all of A's.
    """
    rows = [
        ("T005", "B", 0, 0.99),
        ("T004", "B", 1, 0.98),
        ("T003", "A", 1, 0.97),
        ("T002", "B", 1, 0.10),
        ("T001", "A", 0, 0.05),
        ("T000", "A", 0, 0.01),
    ]
    audited = _audit(_frame(rows), _support(A=(3, 1), B=(3, 2)), 3)

    assert audited["B"].selection_rate_ratio == pytest.approx((2 / 3) / 0.5)
    assert audited["B"].capture_rate == pytest.approx(0.5)
    assert audited["A"].capture_rate == pytest.approx(1.0), "and A does better on capture"


# --- 4. support is carried through, never applied as a filter --------------------------


def test_an_unsupported_group_gets_a_row_with_its_status_and_reason() -> None:
    rows = [(f"T{i:03d}", "A" if i < 8 else "B", i % 2, 0.9 - i * 0.05) for i in range(10)]
    support = _support(A=(8, 4))
    support["B"] = GroupSupport(
        group_definition="community_area",
        group_value="B",
        grain=Grain.FOLD_SET.value,
        fold_set="quarterly",
        fold_id="",
        n_rows=2,
        n_positive=1,
        n_negative=1,
        base_rate=0.5,
        representation_share=0.2,
        ranking_status=GroupStatus.INSUFFICIENT_SUPPORT,
        calibration_status=GroupStatus.INSUFFICIENT_SUPPORT,
        insufficient_reason="2 rows < 200",
    )
    audited = _audit(_frame(rows), support, 4)

    assert audited["B"].group_status is GroupStatus.INSUFFICIENT_SUPPORT
    assert audited["B"].insufficient_reason == "2 rows < 200"
    # Counts are still real: support gates the reading, not the arithmetic.
    assert audited["B"].n_rows == 2


def test_a_group_absent_from_the_support_table_is_marked_rather_than_dropped() -> None:
    rows = [(f"T{i:03d}", "A" if i < 3 else "C", i % 2, 0.9 - i * 0.1) for i in range(6)]
    audited = _audit(_frame(rows), _support(A=(3, 1)), 2)
    assert audited["C"].group_status is GroupStatus.INSUFFICIENT_SUPPORT
    assert "absent from the support table" in audited["C"].insufficient_reason


def test_supported_capture_selects_only_the_quotable_rows() -> None:
    rows = [
        ("T003", "A", 1, 0.9),
        ("T002", "A", 0, 0.8),
        ("T001", "B", 0, 0.2),
        ("T000", "B", 0, 0.1),
    ]
    audited = priority.audit(
        _frame(rows),
        COMMUNITY_AREA,
        _support(A=(2, 1), B=(2, 0)),
        model_name="m",
        stage=Stage.CALIBRATED,
        score_column="calibrated_probability",
        grain=Grain.FOLD_SET,
        fold_set="quarterly",
        fold_id="",
        k_name="k_pct_05",
        k=2,
    )
    quotable = priority.supported_capture(audited)
    assert [r.group_value for r in quotable] == ["A"], "B has no positives, so no capture rate"


def test_an_empty_frame_produces_no_rows_rather_than_raising() -> None:
    assert _audit(_frame([]), {}, 1) == {}
