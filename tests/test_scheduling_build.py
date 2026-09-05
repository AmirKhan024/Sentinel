"""End to end: read a Component 13 artifact, plan it, validate it, write the layer.

The input-contract section is the important half. A scheduling layer is exactly the place where
a quietly incomplete queue would go unnoticed -- a policy artifact that dropped rows would
produce a shorter queue, a fuller horizon and a better utilisation number, all for a reason that
has nothing to do with scheduling. So every one of those defects is a refusal on the way in.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.scheduling import validate as v
from sentinel.scheduling.build import run_schedule, summarize
from sentinel.scheduling.definitions import CONFIG_GRID
from sentinel.scheduling.inputs import (
    ScheduleInputError,
    observed_calendars,
    read_recommendations,
    validate_folds_against_recommendations,
    validate_recommendations,
)
from sentinel.scheduling.writer import LAYERS

from .conftest import make_adjustment, make_execution_event, scheduling_json_for


def _recommendation_rows(n_days: int = 3, per_day: int = 4, k: int = 8) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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
    return rows


@pytest.fixture
def artifacts(tmp_path: Path) -> tuple[Path, Path]:
    recommendations = tmp_path / "inspection_recommendations_20260826T000000Z.parquet"
    folds = tmp_path / "evaluation_folds_20260826T000000Z.parquet"
    pl.DataFrame(_recommendation_rows()).write_parquet(recommendations)
    pl.DataFrame(
        [
            {
                "fold_set": "quarterly",
                "fold_id": "quarterly-2026Q2",
                "test_median_daily_capacity": 4.0,
                "evaluation_definition_version": "v1",
            }
        ]
    ).write_parquet(folds)
    return recommendations, folds


class TestTheInputContract:
    def _frame(self, **changes: object) -> pl.DataFrame:
        rows = _recommendation_rows()
        for key, value in changes.items():
            rows[0][key] = value
        return pl.DataFrame(rows)

    def test_a_well_formed_artifact_passes(self) -> None:
        validate_recommendations(pl.DataFrame(_recommendation_rows()), source="test")

    def test_two_models_in_one_artifact_are_refused(self) -> None:
        """A schedule over two models would silently be two schedules."""
        rows = _recommendation_rows()
        rows[0]["model_name"] = "lightgbm_platt"
        with pytest.raises(ScheduleInputError, match="carries 2 models"):
            validate_recommendations(pl.DataFrame(rows), source="test")

    def test_an_unknown_policy_is_refused(self) -> None:
        rows = _recommendation_rows()
        for row in rows:
            row["policy_id"] = "coverage_vibes"
        with pytest.raises(ScheduleInputError, match="unknown policy_id"):
            validate_recommendations(pl.DataFrame(rows), source="test")

    def test_a_duplicate_decision_key_is_refused(self) -> None:
        rows = _recommendation_rows()
        rows.append(dict(rows[0]))
        with pytest.raises(ScheduleInputError, match="duplicate decision key"):
            validate_recommendations(pl.DataFrame(rows), source="test")

    def test_a_null_inspection_date_is_refused(self) -> None:
        """The calendar is derived from that column; a null drops or invents an operating day."""
        with pytest.raises(ScheduleInputError, match="no inspection_date"):
            validate_recommendations(self._frame(inspection_date=None), source="test")

    def test_a_selected_row_without_a_rank_is_refused(self) -> None:
        with pytest.raises(ScheduleInputError, match="no final_policy_rank"):
            validate_recommendations(self._frame(final_policy_rank=None), source="test")

    def test_an_unselected_row_carrying_a_rank_is_refused(self) -> None:
        """A queue position for an inspection nobody approved."""
        rows = _recommendation_rows()
        rows[-1]["final_policy_rank"] = 99
        with pytest.raises(ScheduleInputError, match="unselected row"):
            validate_recommendations(pl.DataFrame(rows), source="test")

    def test_a_selected_count_other_than_k_is_refused(self) -> None:
        rows = _recommendation_rows()
        rows[0]["is_selected"] = False
        rows[0]["final_policy_rank"] = None
        with pytest.raises(ScheduleInputError, match="other than k"):
            validate_recommendations(pl.DataFrame(rows), source="test")

    def test_a_gapped_rank_is_refused(self) -> None:
        rows = _recommendation_rows()
        rows[0]["final_policy_rank"] = 99
        with pytest.raises(ScheduleInputError, match="not unique and contiguous"):
            validate_recommendations(pl.DataFrame(rows), source="test")

    def test_an_unknown_mechanism_is_refused(self) -> None:
        with pytest.raises(ScheduleInputError, match="unknown decision_mechanism"):
            validate_recommendations(self._frame(decision_mechanism="vibes"), source="test")

    def test_a_mechanism_reason_pair_component_13_forbids_is_refused(self) -> None:
        with pytest.raises(ScheduleInputError, match="Component"):
            validate_recommendations(
                self._frame(decision_reason="selected_by_coverage_reserve"), source="test"
            )

    def test_an_outcome_column_is_refused_on_the_way_in(self, tmp_path: Path) -> None:
        """A scheduler that could read the label could order by it."""
        path = tmp_path / "contaminated.parquet"
        frame = pl.DataFrame(_recommendation_rows()).with_columns(pl.lit(1).alias("target"))
        frame.write_parquet(path)
        with pytest.raises(ScheduleInputError, match="outcome column"):
            read_recommendations(path)

    def test_a_missing_provenance_column_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "thin.parquet"
        pl.DataFrame(_recommendation_rows()).drop("decision_mechanism").write_parquet(path)
        with pytest.raises(ScheduleInputError, match="missing"):
            read_recommendations(path)


class TestTheFoldCrossCheck:
    def test_agreement_passes(self) -> None:
        frame = pl.DataFrame(_recommendation_rows())
        day = frame.with_columns(pl.lit("k_1_day").alias("k_name"), pl.lit(4).alias("k"))
        validate_folds_against_recommendations(day, {"quarterly-2026Q2": 4}, source="test")

    def test_a_disagreement_is_refused(self) -> None:
        """``k_1_day`` *is* the median. If they differ, the two artifacts are different snapshots.

        This is the check that stops a stale fold table producing a plausible schedule for the
        wrong calendar.
        """
        frame = pl.DataFrame(_recommendation_rows())
        day = frame.with_columns(pl.lit("k_1_day").alias("k_name"), pl.lit(4).alias("k"))
        with pytest.raises(ScheduleInputError, match="different snapshots"):
            validate_folds_against_recommendations(day, {"quarterly-2026Q2": 9}, source="test")

    def test_a_fold_absent_from_the_fold_table_is_refused(self) -> None:
        frame = pl.DataFrame(_recommendation_rows())
        day = frame.with_columns(pl.lit("k_1_day").alias("k_name"), pl.lit(4).alias("k"))
        with pytest.raises(ScheduleInputError, match="absent from the fold table"):
            validate_folds_against_recommendations(day, {}, source="test")


class TestTheObservedCalendar:
    def test_the_calendar_counts_the_universe_not_the_queue(self) -> None:
        """Every inspection on a date is a slot, whether or not the policy selected it."""
        calendars = observed_calendars(pl.DataFrame(_recommendation_rows()))
        assert calendars["quarterly-2026Q2"] == (
            (date(2026, 4, 1), 4),
            (date(2026, 4, 2), 4),
            (date(2026, 4, 3), 4),
        )


class TestEndToEnd:
    def test_a_clean_run_produces_every_table(
        self, settings: Settings, artifacts: tuple[Path, Path]
    ) -> None:
        recommendations, folds = artifacts
        result = run_schedule(
            settings,
            recommendations_path=recommendations,
            folds_path=folds,
            no_figures=True,
            dry_run=True,
        )
        assert set(result.tables) == set(LAYERS)
        assert not v.has_failures(result.checks)

    def test_the_queue_is_scheduled_against_the_observed_calendar(
        self, settings: Settings, artifacts: tuple[Path, Path]
    ) -> None:
        recommendations, folds = artifacts
        result = run_schedule(
            settings,
            recommendations_path=recommendations,
            folds_path=folds,
            configs=[c for c in CONFIG_GRID if not c.is_scenario],
            no_figures=True,
            dry_run=True,
        )
        summary = result.tables["schedule_summary"].row(0, named=True)
        assert summary["horizon_days"] == 2
        assert summary["horizon_slots"] == 8
        assert summary["n_scheduled"] == 8
        assert summary["n_backlog"] == 0

    def test_the_external_logs_are_typed_empty_when_no_file_is_supplied(
        self, settings: Settings, artifacts: tuple[Path, Path]
    ) -> None:
        """A typed empty table says "nobody adjusted anything"; a missing file says nothing."""
        recommendations, folds = artifacts
        result = run_schedule(
            settings,
            recommendations_path=recommendations,
            folds_path=folds,
            no_figures=True,
            dry_run=True,
        )
        for table in ("schedule_adjustment_log", "execution_log"):
            assert result.tables[table].height == 0
            assert result.tables[table].width > 0

    def test_the_original_plan_is_itself_a_planning_run(
        self, settings: Settings, artifacts: tuple[Path, Path]
    ) -> None:
        recommendations, folds = artifacts
        result = run_schedule(
            settings,
            recommendations_path=recommendations,
            folds_path=folds,
            no_figures=True,
            dry_run=True,
        )
        assert result.tables["replanning_runs"].height > 0
        assert set(result.tables["replanning_runs"]["trigger"].to_list()) == {"original_plan"}

    def test_the_contract_table_is_always_populated(
        self, settings: Settings, artifacts: tuple[Path, Path]
    ) -> None:
        recommendations, folds = artifacts
        result = run_schedule(
            settings,
            recommendations_path=recommendations,
            folds_path=folds,
            no_figures=True,
            dry_run=True,
        )
        assert result.tables["execution_contract"].height == 22

    def test_a_dry_run_writes_nothing(
        self, settings: Settings, artifacts: tuple[Path, Path], tmp_path: Path
    ) -> None:
        recommendations, folds = artifacts
        destination = tmp_path / "out"
        result = run_schedule(
            settings,
            recommendations_path=recommendations,
            folds_path=folds,
            output_dir=destination,
            no_figures=True,
            dry_run=True,
        )
        assert result.written == []
        assert not destination.exists()

    def test_a_real_run_writes_the_layer_and_a_manifest(
        self, settings: Settings, artifacts: tuple[Path, Path], tmp_path: Path
    ) -> None:
        recommendations, folds = artifacts
        destination = tmp_path / "out"
        result = run_schedule(
            settings,
            recommendations_path=recommendations,
            folds_path=folds,
            output_dir=destination,
            no_figures=True,
        )
        parquet = [p for p in result.written if p.suffix == ".parquet"]
        manifests = [p for p in result.written if p.suffix == ".json"]
        assert len(parquet) == len(LAYERS)
        assert len(manifests) == 1

    def test_the_manifest_carries_the_boundary_and_the_headline(
        self, settings: Settings, artifacts: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """The boundary must travel with the artifact, not only in documentation."""
        recommendations, folds = artifacts
        result = run_schedule(
            settings,
            recommendations_path=recommendations,
            folds_path=folds,
            output_dir=tmp_path / "out",
            no_figures=True,
        )
        manifest = next(p for p in result.written if p.suffix == ".json")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for key in (
            "blocked",
            "does_not_establish",
            "inherited_limitations",
            "scheduling_semantics",
            "no_solver",
            "horizon_rule",
            "capacity_mode_scenario_claim",
            "three_human_layers",
            "temporal_boundary",
            "determinism_scope",
            "green_run_means",
            "reserve_slots_lost",
        ):
            assert key in payload, key
        assert payload["inputs_unchanged"] is True

    def test_the_inputs_are_not_modified(
        self, settings: Settings, artifacts: tuple[Path, Path], tmp_path: Path
    ) -> None:
        from sentinel.manifest import compute_sha256

        recommendations, folds = artifacts
        before = (compute_sha256(recommendations), compute_sha256(folds))
        run_schedule(
            settings,
            recommendations_path=recommendations,
            folds_path=folds,
            output_dir=tmp_path / "out",
            no_figures=True,
        )
        assert (compute_sha256(recommendations), compute_sha256(folds)) == before

    def test_restricting_the_run_does_not_report_a_shortfall_the_user_asked_for(
        self, settings: Settings, artifacts: tuple[Path, Path]
    ) -> None:
        """Validated against the cells requested, not against every cell Component 13 wrote."""
        recommendations, folds = artifacts
        result = run_schedule(
            settings,
            recommendations_path=recommendations,
            folds_path=folds,
            policies=["pure_risk"],
            k_names=["k_1_week"],
            no_figures=True,
            dry_run=True,
        )
        assert not v.has_failures(result.checks)


class TestTheExternalPathsEndToEnd:
    def test_an_adjustment_and_an_execution_log_leave_the_run_green(
        self, settings: Settings, artifacts: tuple[Path, Path], tmp_path: Path
    ) -> None:
        recommendations, folds = artifacts
        adjustments = tmp_path / "adj.json"
        execution = tmp_path / "exec.json"
        adjustments.write_text(
            scheduling_json_for(
                [make_adjustment(1, target_inspection_id="T00000", target_date="2026-04-02")]
            ),
            encoding="utf-8",
        )
        execution.write_text(
            scheduling_json_for(
                [
                    make_execution_event(1, target_inspection_id="T00001"),
                    make_execution_event(
                        2, target_inspection_id="T00002", execution_status="not_performed"
                    ),
                ]
            ),
            encoding="utf-8",
        )
        result = run_schedule(
            settings,
            recommendations_path=recommendations,
            folds_path=folds,
            configs=[c for c in CONFIG_GRID if not c.is_scenario],
            adjustments_path=adjustments,
            execution_path=execution,
            no_figures=True,
            dry_run=True,
        )
        assert not v.has_failures(result.checks)
        assert result.tables["schedule_adjustment_log"].height == 1
        assert result.tables["execution_log"].height == 2

    def test_a_malformed_adjustment_file_refuses_the_run(
        self, settings: Settings, artifacts: tuple[Path, Path], tmp_path: Path
    ) -> None:
        from sentinel.scheduling.adjustments import AdjustmentError

        recommendations, folds = artifacts
        path = tmp_path / "adj.json"
        path.write_text(scheduling_json_for([make_adjustment(1, actor="")]), encoding="utf-8")
        with pytest.raises(AdjustmentError):
            run_schedule(
                settings,
                recommendations_path=recommendations,
                folds_path=folds,
                adjustments_path=path,
                no_figures=True,
                dry_run=True,
            )

    def test_a_non_list_payload_is_refused(
        self, settings: Settings, artifacts: tuple[Path, Path], tmp_path: Path
    ) -> None:
        recommendations, folds = artifacts
        path = tmp_path / "adj.json"
        path.write_text(json.dumps({"adjustment_id": "SA-1"}), encoding="utf-8")
        with pytest.raises(ScheduleInputError, match="JSON list"):
            run_schedule(
                settings,
                recommendations_path=recommendations,
                folds_path=folds,
                adjustments_path=path,
                no_figures=True,
                dry_run=True,
            )


class TestTheSummary:
    def test_the_summary_reports_the_reserve_on_the_observed_calendar_only(
        self, settings: Settings, artifacts: tuple[Path, Path]
    ) -> None:
        """Pooling the scenario would average a tautology into the headline and halve it."""
        recommendations, folds = artifacts
        result = run_schedule(
            settings,
            recommendations_path=recommendations,
            folds_path=folds,
            no_figures=True,
            dry_run=True,
        )
        text = summarize(result)
        assert "observed calendar only" in text
        assert "coverage reserve" in text
