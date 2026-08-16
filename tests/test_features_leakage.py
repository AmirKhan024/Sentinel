"""Temporal leakage: the safety wall.

This file exists so that reintroducing future information into a feature is
impossible without a test failing. It is separate from the other feature tests
deliberately — the rest check that features are *right*, these check that they
cannot be *cheating*.

Each test follows the same shape: build features, perturb the future or the
present, rebuild, and assert an earlier row did not move. If any of these ever
fails, stop and find the join condition that lost its date predicate.
"""

from __future__ import annotations

import random
from pathlib import Path

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.features.build import build_features
from sentinel.features.definitions import FEATURE_COLUMNS
from tests.conftest import assignment_frame, make_inspection_record, target_scenario

PRIORITY = "3. MANAGEMENT - Comments: NO POLICY. PRIORITY FOUNDATION 7-38-010."
CORE = "55. PHYSICAL FACILITIES - Comments: DIRTY FLOOR."

EST = "EST-A"

# One establishment, inspected in 2019 and again in 2020. The 2020 row is the
# one under test; everything after it is the "future" these tests perturb.
BASE_ROWS = [
    make_inspection_record(
        1,
        inspection_id="1001",
        inspection_date="2019-03-01T00:00:00.000",
        violations=CORE,
        results="Pass",
    ),
    make_inspection_record(
        2,
        inspection_id="1002",
        inspection_date="2020-06-15T00:00:00.000",
        violations=CORE,
        results="Pass",
    ),
]
BASE_PAIRS = [("1001", EST), ("1002", EST)]


def _targets_for(rows: list[dict[str, object]]) -> pl.DataFrame:
    """A Component 3-shaped target table for the given canvass rows."""
    return pl.DataFrame(
        {
            "establishment_id": [EST] * len(rows),
            "inspection_date": [str(r["inspection_date"]) for r in rows],
            "target_inspection_id": [str(r["inspection_id"]) for r in rows],
            "target": [1] * len(rows),
            "target_status": ["eligible"] * len(rows),
            "code_era_phase": ["stable"] * len(rows),
        },
        schema={
            "establishment_id": pl.Utf8,
            "inspection_date": pl.Utf8,
            "target_inspection_id": pl.Utf8,
            "target": pl.Int8,
            "target_status": pl.Utf8,
            "code_era_phase": pl.Utf8,
        },
    )


def _build(
    settings: Settings,
    tmp_path: Path,
    rows: list[dict[str, object]],
    pairs: list[tuple[str, str]],
    target_rows: list[dict[str, object]],
    tag: str,
) -> pl.DataFrame:
    raw = tmp_path / f"raw_{tag}.parquet"
    asg = tmp_path / f"asg_{tag}.parquet"
    tgt = tmp_path / f"tgt_{tag}.parquet"
    target_scenario(rows).write_parquet(raw)
    assignment_frame(pairs).write_parquet(asg)
    _targets_for(target_rows).write_parquet(tgt)
    return build_features(
        settings, parquet_path=raw, assignments_path=asg, targets_path=tgt, dry_run=True
    ).features


def _row_for(frame: pl.DataFrame, inspection_id: str) -> dict[str, object]:
    matched = frame.filter(pl.col("target_inspection_id") == inspection_id)
    assert matched.height == 1, f"expected exactly one row for {inspection_id}"
    return matched.row(0, named=True)


# --- 1. a future inspection cannot change a past row ----------------------


def test_appending_a_future_inspection_changes_nothing(settings: Settings, tmp_path: Path) -> None:
    """The canonical leakage test. Add 2025 data; the 2020 row must not move."""
    before = _build(settings, tmp_path, BASE_ROWS, BASE_PAIRS, [BASE_ROWS[1]], "a")

    future = [
        *BASE_ROWS,
        make_inspection_record(
            9,
            inspection_id="1099",
            inspection_date="2025-01-01T00:00:00.000",
            violations=PRIORITY,
            results="Fail",
        ),
    ]
    after = _build(settings, tmp_path, future, [*BASE_PAIRS, ("1099", EST)], [BASE_ROWS[1]], "b")

    assert _row_for(before, "1002") == _row_for(after, "1002")


def test_appending_many_future_inspections_changes_nothing(
    settings: Settings, tmp_path: Path
) -> None:
    future = list(BASE_ROWS)
    pairs = list(BASE_PAIRS)
    for i in range(5):
        ident = f"20{i:02d}"
        future.append(
            make_inspection_record(
                50 + i,
                inspection_id=ident,
                inspection_date=f"202{4 + (i % 2)}-0{i + 1}-01T00:00:00.000",
                violations=PRIORITY,
                results="Fail",
            )
        )
        pairs.append((ident, EST))

    before = _build(settings, tmp_path, BASE_ROWS, BASE_PAIRS, [BASE_ROWS[1]], "c")
    after = _build(settings, tmp_path, future, pairs, [BASE_ROWS[1]], "d")
    assert _row_for(before, "1002") == _row_for(after, "1002")


# --- 2. mutating a future outcome cannot change a past row ----------------


def test_changing_a_future_outcome_changes_nothing(settings: Settings, tmp_path: Path) -> None:
    """Flip a 2025 inspection from clean to a priority failure."""
    clean_future = [
        *BASE_ROWS,
        make_inspection_record(
            9,
            inspection_id="1099",
            inspection_date="2025-01-01T00:00:00.000",
            violations=CORE,
            results="Pass",
        ),
    ]
    dirty_future = [
        *BASE_ROWS,
        make_inspection_record(
            9,
            inspection_id="1099",
            inspection_date="2025-01-01T00:00:00.000",
            violations=PRIORITY,
            results="Fail",
        ),
    ]
    pairs = [*BASE_PAIRS, ("1099", EST)]

    clean = _build(settings, tmp_path, clean_future, pairs, [BASE_ROWS[1]], "e")
    dirty = _build(settings, tmp_path, dirty_future, pairs, [BASE_ROWS[1]], "f")
    assert _row_for(clean, "1002") == _row_for(dirty, "1002")


# --- 3. the target's own inspection is never its own history --------------


def test_target_inspection_never_enters_its_own_features(
    settings: Settings, tmp_path: Path
) -> None:
    """The most direct form of leakage: a row seeing its own outcome.

    The 2020 inspection is given a priority violation. Its own feature row must
    not count it, and its recency must not collapse to 0.
    """
    rows = [
        BASE_ROWS[0],
        make_inspection_record(
            2,
            inspection_id="1002",
            inspection_date="2020-06-15T00:00:00.000",
            violations=PRIORITY,
            results="Fail",
        ),
    ]
    frame = _build(settings, tmp_path, rows, BASE_PAIRS, [rows[1]], "g")
    row = _row_for(frame, "1002")

    assert row["prior_canvass_count"] == 1  # only the 2019 inspection
    assert row["prior_canvass_fail_count"] == 0  # its own Fail is not counted
    assert row["days_since_last_canvass"] != 0  # not "today"
    assert row["days_since_last_canvass"] == 472  # 2019-03-01 -> 2020-06-15
    # The 2019 canvass is pre-priority-history only in the sense that it is
    # code-era and clean, so the count is a genuine zero rather than the row's
    # own violation leaking in.
    assert row["prior_canvass_priority_count"] == 0


def test_a_positive_target_does_not_inflate_its_own_history(
    settings: Settings, tmp_path: Path
) -> None:
    """Same row, two different target outcomes, identical features."""
    clean = [
        BASE_ROWS[0],
        make_inspection_record(
            2,
            inspection_id="1002",
            inspection_date="2020-06-15T00:00:00.000",
            violations=CORE,
            results="Pass",
        ),
    ]
    dirty = [
        BASE_ROWS[0],
        make_inspection_record(
            2,
            inspection_id="1002",
            inspection_date="2020-06-15T00:00:00.000",
            violations=PRIORITY,
            results="Fail",
        ),
    ]

    a = _build(settings, tmp_path, clean, BASE_PAIRS, [clean[1]], "h")
    b = _build(settings, tmp_path, dirty, BASE_PAIRS, [dirty[1]], "i")

    feature_only = [c for c in FEATURE_COLUMNS]
    assert {k: _row_for(a, "1002")[k] for k in feature_only} == {
        k: _row_for(b, "1002")[k] for k in feature_only
    }


# --- 4. same-day records are never history --------------------------------


def test_same_day_records_are_excluded(settings: Settings, tmp_path: Path) -> None:
    """Findings §4: dates carry no time component, so same-day cannot be ordered.

    A same-day complaint and a same-day canvass re-inspection are added. Neither
    may enter any feature — the re-inspection in particular can only have
    happened *after* the canvass it follows.
    """
    rows = [
        *BASE_ROWS,
        make_inspection_record(
            3,
            inspection_id="1003",
            inspection_date="2020-06-15T00:00:00.000",
            inspection_type="Complaint",
            violations=PRIORITY,
            results="Fail",
        ),
        make_inspection_record(
            4,
            inspection_id="1004",
            inspection_date="2020-06-15T00:00:00.000",
            inspection_type="Canvass Re-Inspection",
            violations=PRIORITY,
            results="Fail",
        ),
    ]
    pairs = [*BASE_PAIRS, ("1003", EST), ("1004", EST)]

    baseline = _build(settings, tmp_path, BASE_ROWS, BASE_PAIRS, [BASE_ROWS[1]], "j")
    with_same_day = _build(settings, tmp_path, rows, pairs, [BASE_ROWS[1]], "k")

    assert _row_for(baseline, "1002") == _row_for(with_same_day, "1002")
    row = _row_for(with_same_day, "1002")
    assert row["prior_complaint_count"] == 0
    assert row["prior_reinspection_count"] == 0


def test_a_record_one_day_earlier_is_included(settings: Settings, tmp_path: Path) -> None:
    """The boundary is exclusive, not absent: the day before does count."""
    rows = [
        *BASE_ROWS,
        make_inspection_record(
            3,
            inspection_id="1003",
            inspection_date="2020-06-14T00:00:00.000",
            inspection_type="Complaint",
            violations=CORE,
            results="Pass",
        ),
    ]
    frame = _build(settings, tmp_path, rows, [*BASE_PAIRS, ("1003", EST)], [BASE_ROWS[1]], "l")
    row = _row_for(frame, "1002")
    assert row["prior_complaint_count"] == 1
    assert row["days_since_any_inspection"] == 1


# --- 5. determinism -------------------------------------------------------


def test_shuffling_the_input_changes_nothing(settings: Settings, tmp_path: Path) -> None:
    """Row order must not reach the features."""
    rows = [
        *BASE_ROWS,
        make_inspection_record(
            3,
            inspection_id="1003",
            inspection_date="2019-09-01T00:00:00.000",
            violations=PRIORITY,
            results="Fail",
        ),
        make_inspection_record(
            4,
            inspection_id="1004",
            inspection_date="2018-11-01T00:00:00.000",
            inspection_type="Complaint",
            violations=CORE,
        ),
    ]
    pairs = [*BASE_PAIRS, ("1003", EST), ("1004", EST)]

    ordered = _build(settings, tmp_path, rows, pairs, [BASE_ROWS[1]], "m")

    shuffled_rows = rows[:]
    random.Random(20260816).shuffle(shuffled_rows)
    shuffled = _build(settings, tmp_path, shuffled_rows, pairs, [BASE_ROWS[1]], "n")

    assert ordered.equals(shuffled)


def test_two_builds_are_identical(settings: Settings, tmp_path: Path) -> None:
    first = _build(settings, tmp_path, BASE_ROWS, BASE_PAIRS, [BASE_ROWS[1]], "o")
    second = _build(settings, tmp_path, BASE_ROWS, BASE_PAIRS, [BASE_ROWS[1]], "p")
    assert first.equals(second)


# --- 6. the invariant, asserted directly ----------------------------------


def test_every_feature_row_only_saw_earlier_inspections(settings: Settings, tmp_path: Path) -> None:
    """An end-to-end restatement of the rule the whole component exists for."""
    rows = [
        make_inspection_record(
            i,
            inspection_id=str(2000 + i),
            inspection_date=f"20{18 + i}-05-01T00:00:00.000",
            violations=PRIORITY if i % 2 else CORE,
            results="Fail" if i % 2 else "Pass",
        )
        for i in range(6)
    ]
    pairs = [(str(2000 + i), EST) for i in range(6)]
    frame = _build(settings, tmp_path, rows, pairs, rows[1:], "q")

    dates = sorted(str(r["inspection_date"])[:10] for r in rows)
    for feature_row in frame.iter_rows(named=True):
        reference = feature_row["inspection_date"]
        earlier = [d for d in dates if d < reference]
        assert feature_row["prior_inspection_count_any_type"] == len(earlier), (
            f"row at {reference} counted {feature_row['prior_inspection_count_any_type']} "
            f"prior inspections; only {len(earlier)} are strictly earlier"
        )


@pytest.mark.parametrize("column", ["target", "target_status"])
def test_label_columns_are_not_features(column: str) -> None:
    """A label must never be reachable as a model input."""
    assert column not in FEATURE_COLUMNS
