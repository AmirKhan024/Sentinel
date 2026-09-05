"""Temporal leakage and determinism for Component 17.

Mirrors ``tests/test_features_leakage.py``'s shape deliberately: build candidates,
perturb the future (or the present, or the planning date itself), rebuild, and
assert an earlier candidate's features did not move. The one addition specific to
this component is that there is no future *row* to omit in the first place --
these tests instead perturb what happens on or after the planning date, since
that is the only kind of "future" an operational candidate can have.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from sentinel.candidates.build import build_candidates
from sentinel.candidates.universe import CandidateGenerationError
from sentinel.config import Settings
from tests.conftest import assignment_frame, make_inspection_record, target_scenario

CORE = "55. PHYSICAL FACILITIES - Comments: DIRTY FLOOR."
PRIORITY = "3. MANAGEMENT - Comments: NO POLICY. PRIORITY FOUNDATION 7-38-010."

EST = "EST-A"
PLANNING_DATE = "2021-01-01"

# One establishment inspected twice before the planning date. Everything from
# the planning date onward is "the future" these tests perturb.
BASE_ROWS = [
    make_inspection_record(
        1, inspection_id="1001", inspection_date="2019-03-01T00:00:00.000",
        violations=CORE, results="Pass",
    ),
    make_inspection_record(
        2, inspection_id="1002", inspection_date="2020-06-15T00:00:00.000",
        violations=CORE, results="Pass",
    ),
]
BASE_PAIRS = [("1001", EST), ("1002", EST)]


def _establishments_for(establishment_ids: list[str]) -> pl.DataFrame:
    """A minimal Component 2-shaped establishments table."""
    return pl.DataFrame(
        {
            "establishment_id": establishment_ids,
            "canonical_name": [f"NAME-{e}" for e in establishment_ids],
            "canonical_address": [f"ADDR-{e}" for e in establishment_ids],
            "canonical_zip": ["60601"] * len(establishment_ids),
        },
        schema={
            "establishment_id": pl.Utf8,
            "canonical_name": pl.Utf8,
            "canonical_address": pl.Utf8,
            "canonical_zip": pl.Utf8,
        },
    )


def _build(
    settings: Settings,
    tmp_path: Path,
    rows: list[dict[str, object]],
    pairs: list[tuple[str, str]],
    tag: str,
    *,
    planning_date: str = PLANNING_DATE,
    establishment_ids: list[str] | None = None,
) -> pl.DataFrame:
    raw = tmp_path / f"raw_{tag}.parquet"
    asg = tmp_path / f"asg_{tag}.parquet"
    est = tmp_path / f"est_{tag}.parquet"
    target_scenario(rows).write_parquet(raw)
    assignment_frame(pairs).write_parquet(asg)
    ids = establishment_ids or sorted({e for _, e in pairs})
    _establishments_for(ids).write_parquet(est)
    return build_candidates(
        settings,
        planning_date=planning_date,
        parquet_path=raw,
        assignments_path=asg,
        establishments_path=est,
        dry_run=True,
    ).candidates


def _row_for(frame: pl.DataFrame, establishment_id: str) -> dict[str, object]:
    matched = frame.filter(pl.col("establishment_id") == establishment_id)
    assert matched.height == 1, f"expected exactly one candidate row for {establishment_id}"
    return matched.row(0, named=True)


FEATURE_ONLY_COLUMNS = [
    "prior_canvass_count",
    "prior_canvass_count_code_era",
    "prior_canvass_fail_count",
    "days_since_last_canvass",
    "prior_canvass_priority_count",
]


# --- 1. a record on or after the planning date cannot change a candidate ---


def test_appending_a_future_record_changes_nothing(settings: Settings, tmp_path: Path) -> None:
    """The canonical leakage test, adapted: add a 2025 record; the candidate must not move."""
    before = _build(settings, tmp_path, BASE_ROWS, BASE_PAIRS, "a")

    future = [
        *BASE_ROWS,
        make_inspection_record(
            9, inspection_id="1099", inspection_date="2025-01-01T00:00:00.000",
            violations=PRIORITY, results="Fail",
        ),
    ]
    after = _build(settings, tmp_path, future, [*BASE_PAIRS, ("1099", EST)], "b")

    before_row = {k: _row_for(before, EST)[k] for k in FEATURE_ONLY_COLUMNS}
    after_row = {k: _row_for(after, EST)[k] for k in FEATURE_ONLY_COLUMNS}
    assert before_row == after_row


def test_a_record_on_the_planning_date_itself_is_excluded(
    settings: Settings, tmp_path: Path
) -> None:
    """The boundary is strict: a record dated exactly on planning_date is not history.

    No real inspection could ever be dated on a future planning date in practice,
    but the join condition must refuse it structurally rather than by accident of
    the data never containing one.
    """
    baseline = _build(settings, tmp_path, BASE_ROWS, BASE_PAIRS, "c")

    same_day = [
        *BASE_ROWS,
        make_inspection_record(
            9, inspection_id="1099", inspection_date=f"{PLANNING_DATE}T00:00:00.000",
            violations=PRIORITY, results="Fail",
        ),
    ]
    with_same_day = _build(settings, tmp_path, same_day, [*BASE_PAIRS, ("1099", EST)], "d")

    assert _row_for(baseline, EST)["prior_canvass_count"] == _row_for(with_same_day, EST)[
        "prior_canvass_count"
    ]
    assert _row_for(with_same_day, EST)["prior_canvass_priority_count"] == 0


def test_a_record_one_day_before_the_planning_date_is_included(
    settings: Settings, tmp_path: Path
) -> None:
    """The boundary is exclusive, not absent: the day before does count."""
    rows = [
        *BASE_ROWS,
        make_inspection_record(
            9, inspection_id="1099", inspection_date="2020-12-31T00:00:00.000",
            violations=CORE, results="Pass",
        ),
    ]
    frame = _build(settings, tmp_path, rows, [*BASE_PAIRS, ("1099", EST)], "e")
    row = _row_for(frame, EST)
    assert row["prior_canvass_count"] == 3
    assert row["days_since_last_canvass"] == 1


# --- 2. moving the planning date earlier cannot see later data ------------


def test_an_earlier_planning_date_cannot_see_later_records(
    settings: Settings, tmp_path: Path
) -> None:
    """Move planning_date itself earlier; the later record must vanish from history."""
    later = _build(settings, tmp_path, BASE_ROWS, BASE_PAIRS, "f", planning_date="2021-01-01")
    earlier = _build(
        settings, tmp_path, BASE_ROWS, BASE_PAIRS, "g", planning_date="2019-06-01"
    )

    assert _row_for(later, EST)["prior_canvass_count"] == 2
    assert _row_for(earlier, EST)["prior_canvass_count"] == 1  # only the 2019-03-01 canvass


# --- 3. determinism ---------------------------------------------------------


def test_two_builds_are_identical(settings: Settings, tmp_path: Path) -> None:
    first = _build(settings, tmp_path, BASE_ROWS, BASE_PAIRS, "h")
    second = _build(settings, tmp_path, BASE_ROWS, BASE_PAIRS, "i")
    assert first.equals(second)


def test_candidate_ids_are_a_pure_function_of_date_and_establishment(
    settings: Settings, tmp_path: Path
) -> None:
    from sentinel.candidates.definitions import synthetic_candidate_id

    frame = _build(settings, tmp_path, BASE_ROWS, BASE_PAIRS, "j")
    expected = synthetic_candidate_id(planning_date=PLANNING_DATE, establishment_id=EST)
    assert _row_for(frame, EST)["target_inspection_id"] == expected


# --- 4. planning date coverage / validation --------------------------------


def test_planning_date_on_or_before_the_earliest_record_is_rejected(
    settings: Settings, tmp_path: Path
) -> None:
    with pytest.raises(CandidateGenerationError, match="not supportable"):
        _build(settings, tmp_path, BASE_ROWS, BASE_PAIRS, "k", planning_date="2019-03-01")


def test_malformed_planning_date_is_rejected(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(CandidateGenerationError, match="not a valid ISO date"):
        _build(settings, tmp_path, BASE_ROWS, BASE_PAIRS, "l", planning_date="not-a-date")


def test_planning_date_beyond_the_latest_record_warns_but_does_not_fail(
    settings: Settings, tmp_path: Path
) -> None:
    raw = tmp_path / "raw_m.parquet"
    asg = tmp_path / "asg_m.parquet"
    est = tmp_path / "est_m.parquet"
    target_scenario(BASE_ROWS).write_parquet(raw)
    assignment_frame(BASE_PAIRS).write_parquet(asg)
    _establishments_for([EST]).write_parquet(est)

    result = build_candidates(
        settings,
        planning_date="2030-01-01",
        parquet_path=raw,
        assignments_path=asg,
        establishments_path=est,
        dry_run=True,
    )
    assert any("planning_date_beyond_ingested_data" in w for w in result.manifest.warnings)
    assert result.candidates.height == 1


# --- 5. no establishment enters the pool without prior history -------------


def test_an_establishment_with_no_prior_record_is_not_a_candidate(
    settings: Settings, tmp_path: Path
) -> None:
    """An establishment whose only record is on/after the planning date is excluded."""
    other = make_inspection_record(
        3, inspection_id="2001", inspection_date="2022-01-01T00:00:00.000",
        violations=CORE, results="Pass",
    )
    rows = [*BASE_ROWS, other]
    pairs = [*BASE_PAIRS, ("2001", "EST-B")]
    frame = _build(
        settings, tmp_path, rows, pairs, "n", establishment_ids=[EST, "EST-B"]
    )
    assert set(frame["establishment_id"].to_list()) == {EST}


# --- 6. missing location is preserved honestly, never fabricated -----------


def test_missing_coordinates_are_null_not_fabricated(settings: Settings, tmp_path: Path) -> None:
    rows = [
        make_inspection_record(
            1, inspection_id="1001", inspection_date="2019-03-01T00:00:00.000",
            violations=CORE, results="Pass", latitude=None, longitude=None,
        ),
    ]
    frame = _build(settings, tmp_path, rows, [("1001", EST)], "o")
    row = _row_for(frame, EST)
    assert row["as_of_latitude"] is None
    assert row["as_of_longitude"] is None
    assert row["has_location"] is False
    # The candidate is still present: missing location never excludes a row.
    assert frame.height == 1


def test_present_coordinates_are_carried_through_unchanged(
    settings: Settings, tmp_path: Path
) -> None:
    frame = _build(settings, tmp_path, BASE_ROWS, BASE_PAIRS, "p")
    row = _row_for(frame, EST)
    assert row["as_of_latitude"] == pytest.approx(41.8781)
    assert row["as_of_longitude"] == pytest.approx(-87.6298)
    assert row["has_location"] is True


# --- 7. candidate metadata never touches the feature/label columns ---------


def test_candidate_metadata_columns_are_disjoint_from_feature_columns(
    settings: Settings, tmp_path: Path
) -> None:
    from sentinel.features.definitions import FEATURE_COLUMNS

    frame = _build(settings, tmp_path, BASE_ROWS, BASE_PAIRS, "q")
    metadata_columns = {
        "as_of_dba_name", "as_of_address", "as_of_zip", "as_of_latitude",
        "as_of_longitude", "has_location", "n_prior_records", "first_known_date",
        "last_known_date", "canonical_name", "canonical_address", "canonical_zip",
        "planning_date", "candidate_definition_version",
    }
    assert metadata_columns.isdisjoint(FEATURE_COLUMNS)
    assert metadata_columns.issubset(set(frame.columns))


def test_target_is_always_null_and_status_is_never_a_real_status(
    settings: Settings, tmp_path: Path
) -> None:
    """A candidate must never look like a real, labelled Component 3 row."""
    frame = _build(settings, tmp_path, BASE_ROWS, BASE_PAIRS, "r")
    row = _row_for(frame, EST)
    assert row["target"] is None
    assert row["target_status"] == "operational_candidate"
