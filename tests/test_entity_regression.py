"""Regression tests built from real cases found in the Chicago snapshot.

Unit tests prove the rules do what they say. These prove the rules do the right
thing to data that actually exists. Every case was copied verbatim out of
``food_inspections_20260816T070911Z.parquet`` while inspecting the first full
resolution run, and several of them are cases the first run got wrong.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.entity.resolve import resolve_establishments
from tests.conftest import entity_scenario, make_entity_record
from tests.fixtures.real_cases import REAL_CASES, RealCase


def _resolve_pair(settings: Settings, tmp_path: Path, case: RealCase) -> bool:
    """Resolve the two records of a case and report whether they merged."""
    rows = [
        make_entity_record(1, inspection_id="1000001", **case.left),
        make_entity_record(2, inspection_id="1000002", **case.right),
    ]
    path = tmp_path / f"{case.case_id}.parquet"
    entity_scenario(rows).write_parquet(path)

    result = resolve_establishments(settings, parquet_path=path, dry_run=True)
    mapping = dict(
        zip(
            result.assignments["inspection_id"],
            result.assignments["establishment_id"],
            strict=True,
        )
    )
    return mapping["1000001"] == mapping["1000002"]


@pytest.mark.parametrize("case", REAL_CASES, ids=lambda c: c.case_id)
def test_real_case_resolves_as_expected(settings: Settings, tmp_path: Path, case: RealCase) -> None:
    merged = _resolve_pair(settings, tmp_path, case)
    verb = "merge" if case.should_merge else "stay separate"
    assert merged is case.should_merge, (
        f"{case.case_id}: expected these records to {verb}.\n{case.why}"
    )


@pytest.mark.parametrize(
    "case", [c for c in REAL_CASES if not c.should_merge], ids=lambda c: c.case_id
)
def test_declined_merges_are_recorded_in_the_audit_table(
    settings: Settings, tmp_path: Path, case: RealCase
) -> None:
    """A false merge avoided is only useful if we can explain why."""
    rows = [
        make_entity_record(1, inspection_id="1000001", **case.left),
        make_entity_record(2, inspection_id="1000002", **case.right),
    ]
    path = tmp_path / f"{case.case_id}_audit.parquet"
    entity_scenario(rows).write_parquet(path)

    result = resolve_establishments(settings, parquet_path=path, dry_run=True)
    if result.edges.height == 0:
        # The two records shared no block at all, which is itself a complete
        # explanation: they were never plausible candidates.
        return
    assert set(result.edges["tier"]) <= {"no_match", "ambiguous"}


def test_every_case_carries_an_explanation() -> None:
    for case in REAL_CASES:
        assert len(case.why) > 40, f"{case.case_id} needs a real explanation"


def test_cases_cover_both_directions_of_error() -> None:
    """False merges and false splits are both represented."""
    assert any(c.should_merge for c in REAL_CASES)
    assert any(not c.should_merge for c in REAL_CASES)


def test_ohare_concessionaire_case_is_present() -> None:
    """The specific over-merge the first full run produced."""
    case = next(c for c in REAL_CASES if c.case_id == "ohare_host_international_concessionaire")
    assert not case.should_merge
    assert "aka" in case.tags


def test_real_cases_use_valid_raw_shapes(settings: Settings, tmp_path: Path) -> None:
    """Each case must be loadable as a raw-shaped, all-Utf8 frame."""
    for case in REAL_CASES:
        frame = entity_scenario(
            [
                make_entity_record(1, inspection_id="1", **case.left),
                make_entity_record(2, inspection_id="2", **case.right),
            ]
        )
        assert frame.height == 2
        assert all(dtype == pl.Utf8 for dtype in frame.dtypes)
