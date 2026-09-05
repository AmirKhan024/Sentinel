"""Determinism, end to end and under shuffled inputs.

The claim being tested is the one the manifest scopes: *identical recommendation artifact +
identical scheduling configuration + identical external files produce byte-identical tables*.
Human and operational inputs are not reproducible computation, and the tests are careful not to
claim they are -- what is asserted is that given the same files, the output does not move.

Shuffle-invariance is the property no artifact can show on its own, so it is proved by doing it:
the same rows in a different order must produce the same plan, because the ordering comes from
``final_policy_rank`` and never from Parquet row order.
"""

from __future__ import annotations

import random
from pathlib import Path

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.scheduling.build import _planning_run_id, run_schedule
from sentinel.scheduling.definitions import CONFIG_GRID
from sentinel.scheduling.writer import LAYERS

from .conftest import make_adjustment, make_execution_event, scheduling_json_for


def _recommendations(n_days: int = 3, per_day: int = 4, k: int = 8) -> pl.DataFrame:
    """A small Component 13 artifact, shaped exactly like the real one."""
    from datetime import date

    rows = []
    index = 0
    for day in range(n_days):
        for _ in range(per_day):
            rows.append(
                {
                    "policy_id": "pure_risk",
                    "model_name": "xgboost_platt",
                    "fold_set": "quarterly",
                    "fold_id": "quarterly-2026Q2",
                    "k_name": "k_1_week",
                    "k": k,
                    "target_inspection_id": f"T{index:05d}",
                    "establishment_id": f"EST-{index:05d}",
                    "inspection_date": date(2026, 4, day + 1),
                    "model_rank": index + 1,
                    "final_policy_rank": index + 1 if index < k else None,
                    "is_selected": index < k,
                    "decision_mechanism": "risk_priority" if index < k else "not_selected",
                    "decision_reason": (
                        "selected_by_risk_rank" if index < k else "not_selected_capacity_exhausted"
                    ),
                    "coverage_eligible": False,
                    "score": 0.9 - index * 0.001,
                    "base_score": 0.8 - index * 0.001,
                    "warnings": "none",
                    "policy_definition_version": "v1",
                }
            )
            index += 1
    return pl.DataFrame(rows)


def _folds() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "fold_set": "quarterly",
                "fold_id": "quarterly-2026Q2",
                "test_median_daily_capacity": 4.0,
                "evaluation_definition_version": "v1",
            }
        ]
    )


@pytest.fixture
def artifacts(tmp_path: Path) -> tuple[Path, Path]:
    recommendations = tmp_path / "inspection_recommendations_20260826T000000Z.parquet"
    folds = tmp_path / "evaluation_folds_20260826T000000Z.parquet"
    _recommendations().write_parquet(recommendations)
    _folds().write_parquet(folds)
    return recommendations, folds


def _run(settings: Settings, artifacts: tuple[Path, Path], **kw: object):
    recommendations, folds = artifacts
    return run_schedule(
        settings,
        recommendations_path=recommendations,
        folds_path=folds,
        no_figures=True,
        dry_run=True,
        **kw,  # type: ignore[arg-type]
    )


class TestPlanningRunIds:
    def test_the_id_is_a_content_hash_and_not_a_clock(self) -> None:
        """A clock-derived id would make two runs over identical inputs differ in a column."""
        assert _planning_run_id("a", "b", "0") == _planning_run_id("a", "b", "0")

    def test_a_different_cell_gets_a_different_id(self) -> None:
        assert _planning_run_id("a", "b", "0") != _planning_run_id("a", "b", "1")


class TestShuffledInputs:
    @pytest.mark.parametrize("seed", range(5))
    def test_shuffled_recommendation_rows_produce_an_identical_plan(
        self, settings: Settings, artifacts: tuple[Path, Path], seed: int, tmp_path: Path
    ) -> None:
        first = _run(settings, artifacts).tables["inspection_schedule"]

        recommendations, folds = artifacts
        frame = pl.read_parquet(recommendations)
        shuffled = frame.sample(fraction=1.0, shuffle=True, seed=seed)
        shuffled_path = tmp_path / "shuffled_20260826T000001Z.parquet"
        shuffled.write_parquet(shuffled_path)

        second = run_schedule(
            settings,
            recommendations_path=shuffled_path,
            folds_path=folds,
            no_figures=True,
            dry_run=True,
        ).tables["inspection_schedule"]
        assert first.equals(second)

    def test_shuffled_adjustment_rows_produce_an_identical_plan(
        self, settings: Settings, artifacts: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Adjustments are applied in id order, so re-serialising cannot change the schedule."""
        # Both target day 2, which is the last day of this cell's two-day horizon. Day 2 is
        # already full, so each move costs a displacement -- which is what makes the ordering
        # observable: applied in a different order they would displace different rows.
        rows = [
            make_adjustment(1, target_inspection_id="T00000", target_date="2026-04-02"),
            make_adjustment(2, target_inspection_id="T00001", target_date="2026-04-02"),
        ]
        forward = tmp_path / "adj_forward.json"
        backward = tmp_path / "adj_backward.json"
        forward.write_text(scheduling_json_for(rows), encoding="utf-8")
        backward.write_text(scheduling_json_for(list(reversed(rows))), encoding="utf-8")

        first = _run(settings, artifacts, adjustments_path=forward).tables["inspection_schedule"]
        second = _run(settings, artifacts, adjustments_path=backward).tables["inspection_schedule"]
        assert first.equals(second)

    def test_shuffled_execution_rows_produce_an_identical_plan(
        self, settings: Settings, artifacts: tuple[Path, Path], tmp_path: Path
    ) -> None:
        rows = [
            make_execution_event(1, target_inspection_id="T00000"),
            make_execution_event(
                2, target_inspection_id="T00001", execution_status="not_performed"
            ),
        ]
        forward = tmp_path / "exec_forward.json"
        backward = tmp_path / "exec_backward.json"
        forward.write_text(scheduling_json_for(rows), encoding="utf-8")
        backward.write_text(scheduling_json_for(list(reversed(rows))), encoding="utf-8")

        first = _run(settings, artifacts, execution_path=forward).tables["inspection_schedule"]
        second = _run(settings, artifacts, execution_path=backward).tables["inspection_schedule"]
        assert first.equals(second)


class TestRepeatedRuns:
    def test_two_runs_over_identical_inputs_produce_identical_tables(
        self, settings: Settings, artifacts: tuple[Path, Path]
    ) -> None:
        first = _run(settings, artifacts)
        second = _run(settings, artifacts)
        for table in LAYERS:
            assert first.tables[table].equals(second.tables[table]), table

    def test_both_capacity_modes_are_deterministic(
        self, settings: Settings, artifacts: tuple[Path, Path]
    ) -> None:
        for spec in CONFIG_GRID:
            first = _run(settings, artifacts, configs=[spec])
            second = _run(settings, artifacts, configs=[spec])
            assert first.tables["inspection_schedule"].equals(second.tables["inspection_schedule"])

    def test_written_files_are_byte_identical_across_runs(
        self, settings: Settings, artifacts: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """The end-to-end contract, checked on disk rather than in memory."""
        from sentinel.manifest import compute_sha256

        recommendations, folds = artifacts
        digests = []
        for run_index in range(2):
            destination = tmp_path / f"out{run_index}"
            result = run_schedule(
                settings,
                recommendations_path=recommendations,
                folds_path=folds,
                output_dir=destination,
                no_figures=True,
            )
            parquet = sorted(p for p in result.written if p.suffix == ".parquet")
            digests.append([compute_sha256(p) for p in parquet])
        assert digests[0] == digests[1]


class TestTheDeterministicPlanSurvivesExternalFiles:
    def test_the_plan_at_index_zero_is_unchanged_by_an_adjustment(
        self, settings: Settings, artifacts: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Component 13 writes its queue unchanged beside its override log; so does this."""
        clean = _run(settings, artifacts).tables["inspection_schedule"]

        path = tmp_path / "adj.json"
        path.write_text(
            scheduling_json_for(
                [make_adjustment(1, target_inspection_id="T00000", target_date="2026-04-02")]
            ),
            encoding="utf-8",
        )
        adjusted = _run(settings, artifacts, adjustments_path=path).tables["inspection_schedule"]
        baseline = adjusted.filter(pl.col("replan_index") == 0)
        assert baseline.equals(clean)

    def test_an_adjustment_appends_a_planning_run_rather_than_editing_one(
        self, settings: Settings, artifacts: tuple[Path, Path], tmp_path: Path
    ) -> None:
        path = tmp_path / "adj.json"
        path.write_text(
            scheduling_json_for(
                [make_adjustment(1, target_inspection_id="T00000", target_date="2026-04-02")]
            ),
            encoding="utf-8",
        )
        result = _run(settings, artifacts, adjustments_path=path)
        indices = set(result.tables["inspection_schedule"]["replan_index"].to_list())
        assert indices == {0, 1}
        assert set(result.tables["replanning_runs"]["trigger"].to_list()) == {
            "original_plan",
            "scheduling_adjustment",
        }


class TestCanonicalOrdering:
    def test_the_plan_never_depends_on_parquet_row_order(
        self, settings: Settings, artifacts: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Sorted descending by id -- the least natural order -- and the plan must not move."""
        recommendations, folds = artifacts
        reversed_frame = pl.read_parquet(recommendations).sort(
            "target_inspection_id", descending=True
        )
        path = tmp_path / "reversed_20260826T000002Z.parquet"
        reversed_frame.write_parquet(path)

        first = _run(settings, artifacts).tables["inspection_schedule"]
        second = run_schedule(
            settings,
            recommendations_path=path,
            folds_path=folds,
            no_figures=True,
            dry_run=True,
        ).tables["inspection_schedule"]
        assert first.equals(second)

    def test_random_orderings_all_agree(
        self, settings: Settings, artifacts: tuple[Path, Path], tmp_path: Path
    ) -> None:
        recommendations, folds = artifacts
        frame = pl.read_parquet(recommendations)
        expected = _run(settings, artifacts).tables["inspection_schedule"]
        for seed in range(3):
            rows = frame.to_dicts()
            random.Random(seed).shuffle(rows)
            path = tmp_path / f"perm{seed}_20260826T00000{seed}Z.parquet"
            pl.DataFrame(rows, schema=frame.schema).write_parquet(path)
            got = run_schedule(
                settings,
                recommendations_path=path,
                folds_path=folds,
                no_figures=True,
                dry_run=True,
            ).tables["inspection_schedule"]
            assert got.equals(expected), seed
