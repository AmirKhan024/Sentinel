"""Determinism, end to end and under shuffled inputs.

The claim being tested: identical Component 13/14 artifacts + identical resolutions file produce
byte-identical review tables. A resolution file is external human input, so the claim is scoped
to "given the same file" rather than to reproducibility of the human decision itself.
"""

from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.manifest import compute_sha256
from sentinel.review.build import run_review


def _recommendations(n: int = 6) -> pl.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "policy_id": "pure_risk",
                "model_name": "lightgbm_platt",
                "fold_set": "quarterly",
                "fold_id": "2026Q1",
                "k_name": "k_1_day",
                "k": n,
                "target_inspection_id": f"t{i:03d}",
                "establishment_id": f"e{i:03d}",
                "inspection_date": date(2026, 1, 5),
                "base_score": 0.5,
                "score": 0.6,
                "model_rank": i + 1,
                "final_policy_rank": i + 1,
                "is_selected": True,
                "decision_mechanism": "risk_priority",
                "decision_reason": "selected_by_risk_rank",
                "coverage_eligible": False,
                "secondary_no_history": False,
                "warnings": "limited_history" if i % 2 == 0 else "none",
                "group_value": "",
                "group_status": "",
                "policy_definition_version": "v1",
            }
        )
    return pl.DataFrame(rows)


@pytest.fixture
def recommendations_path(tmp_path: Path) -> Path:
    path = tmp_path / "inspection_recommendations_20260101T000000Z.parquet"
    _recommendations().write_parquet(path)
    return path


def test_two_runs_over_identical_inputs_produce_identical_tables(
    settings: Settings, recommendations_path: Path
) -> None:
    first = run_review(
        settings, recommendations_path=recommendations_path, write_figures=False, dry_run=True
    )
    second = run_review(
        settings, recommendations_path=recommendations_path, write_figures=False, dry_run=True
    )
    for table in ("human_review_queue", "review_resolution_log", "review_advisories"):
        assert first.tables[table].equals(second.tables[table]), table


@pytest.mark.parametrize("seed", range(3))
def test_shuffled_recommendation_rows_produce_an_identical_queue(
    settings: Settings, recommendations_path: Path, seed: int, tmp_path: Path
) -> None:
    first = run_review(
        settings, recommendations_path=recommendations_path, write_figures=False, dry_run=True
    ).tables["human_review_queue"]

    shuffled = pl.read_parquet(recommendations_path).sample(fraction=1.0, shuffle=True, seed=seed)
    shuffled_path = tmp_path / f"shuffled{seed}_20260101T000001Z.parquet"
    shuffled.write_parquet(shuffled_path)

    second = run_review(
        settings, recommendations_path=shuffled_path, write_figures=False, dry_run=True
    ).tables["human_review_queue"]
    assert first.equals(second)


def test_written_files_are_byte_identical_across_runs(
    settings: Settings, recommendations_path: Path, tmp_path: Path
) -> None:
    digests = []
    for run_index in range(2):
        destination = tmp_path / f"out{run_index}"
        result = run_review(
            settings,
            recommendations_path=recommendations_path,
            output_dir=destination,
            write_figures=False,
        )
        parquet = sorted(p for p in result.written if p.suffix == ".parquet")
        digests.append([compute_sha256(p) for p in parquet])
    assert digests[0] == digests[1]


def test_resolutions_apply_in_id_order_regardless_of_file_order(
    settings: Settings, recommendations_path: Path, tmp_path: Path
) -> None:
    resolutions = [
        {
            "review_id": "R1",
            "policy_id": "pure_risk",
            "fold_id": "2026Q1",
            "k_name": "k_1_day",
            "target_inspection_id": "t000",
            "resolution_action": "acknowledge",
            "reason_code": "reviewed",
            "actor": "alice",
            "decided_at": "2026-01-06T00:00:00Z",
        },
        {
            "review_id": "R2",
            "policy_id": "pure_risk",
            "fold_id": "2026Q1",
            "k_name": "k_1_day",
            "target_inspection_id": "t000",
            "resolution_action": "escalate",
            "reason_code": "second_opinion",
            "actor": "bob",
            "decided_at": "2026-01-06T01:00:00Z",
        },
    ]
    forward = tmp_path / "forward.json"
    backward = tmp_path / "backward.json"
    forward.write_text(json.dumps(resolutions), encoding="utf-8")
    backward.write_text(json.dumps(list(reversed(resolutions))), encoding="utf-8")

    first = run_review(
        settings,
        recommendations_path=recommendations_path,
        resolutions_path=forward,
        write_figures=False,
        dry_run=True,
    )
    second = run_review(
        settings,
        recommendations_path=recommendations_path,
        resolutions_path=backward,
        write_figures=False,
        dry_run=True,
    )
    assert first.tables["human_review_queue"].equals(second.tables["human_review_queue"])
    assert first.tables["review_resolution_log"].equals(second.tables["review_resolution_log"])


def test_random_orderings_all_agree(
    settings: Settings, recommendations_path: Path, tmp_path: Path
) -> None:
    frame = pl.read_parquet(recommendations_path)
    expected = run_review(
        settings, recommendations_path=recommendations_path, write_figures=False, dry_run=True
    ).tables["human_review_queue"]
    for seed in range(3):
        rows = frame.to_dicts()
        random.Random(seed).shuffle(rows)
        path = tmp_path / f"perm{seed}_20260101T00000{seed}Z.parquet"
        pl.DataFrame(rows, schema=frame.schema).write_parquet(path)
        got = run_review(
            settings, recommendations_path=path, write_figures=False, dry_run=True
        ).tables["human_review_queue"]
        assert got.equals(expected), seed
