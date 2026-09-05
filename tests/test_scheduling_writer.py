"""The Parquet contract: column order, sort keys, and the typed empty tables.

Column order is part of the data contract, so it is asserted rather than assumed. The typed
empty frames matter as much: the adjustment log and the execution log are empty on every run
nobody supplied a file for, which is most of them, and a missing file would be ambiguous
between "nobody adjusted anything" and "this build does not support adjustments".
"""

from __future__ import annotations

import polars as pl
import pytest

from sentinel.scheduling.writer import (
    DATASET_SLUG,
    LAYERS,
    SCHEMAS,
    SORT_KEYS,
    empty,
    finalize,
    schema_of,
    write_table,
)


class TestTheSchemaRegistry:
    def test_every_table_declares_a_sort_key(self) -> None:
        assert set(SCHEMAS) == set(SORT_KEYS)

    def test_every_table_is_listed_in_the_manifest_order(self) -> None:
        assert set(LAYERS) == set(SCHEMAS)

    def test_the_manifest_is_keyed_to_the_schedule(self) -> None:
        assert DATASET_SLUG == "inspection_schedule"
        assert LAYERS[0] == DATASET_SLUG

    def test_every_sort_key_names_real_columns(self) -> None:
        for table, keys in SORT_KEYS.items():
            for key in keys:
                assert key in SCHEMAS[table], f"{table}.{key}"

    def test_every_table_carries_its_definition_version(self) -> None:
        """So a reader can tell whether two artifacts are comparable."""
        for table, schema in SCHEMAS.items():
            assert "schedule_definition_version" in schema, table

    def test_no_table_carries_an_outcome_column(self) -> None:
        for table, schema in SCHEMAS.items():
            assert "target" not in schema, table
            assert "target_status" not in schema, table

    def test_the_schedule_has_no_execution_status_column(self) -> None:
        """The structural guarantee: execution has nowhere to write into a plan."""
        assert "execution_status" not in SCHEMAS["inspection_schedule"]

    def test_the_execution_log_does_have_one(self) -> None:
        assert "execution_status" in SCHEMAS["execution_log"]

    def test_the_schedule_sort_key_includes_the_planning_run(self) -> None:
        """A re-plan appends a plan, so one inspection legitimately holds one row per run.

        Without the index the sort is not a total order and two plans would be
        indistinguishable in the artifact.
        """
        assert "replan_index" in SORT_KEYS["inspection_schedule"]
        assert "replan_index" in SORT_KEYS["schedule_backlog"]

    def test_component_13_provenance_columns_are_present_on_the_schedule(self) -> None:
        schema = SCHEMAS["inspection_schedule"]
        for column in (
            "final_policy_rank",
            "model_rank",
            "decision_mechanism",
            "decision_reason",
            "score",
            "base_score",
            "coverage_eligible",
            "warnings",
            "policy_definition_version",
        ):
            assert column in schema, column

    def test_scheduling_provenance_is_separate_from_recommendation_provenance(self) -> None:
        """Both blocks present, neither overwriting the other."""
        schema = SCHEMAS["inspection_schedule"]
        assert "final_policy_rank" in schema
        assert "schedule_rank" in schema
        assert "recommendation_date" in schema
        assert "scheduled_date" in schema


class TestEmpty:
    @pytest.mark.parametrize("table", sorted(SCHEMAS))
    def test_every_table_has_a_typed_empty_form(self, table: str) -> None:
        frame = empty(table)
        assert frame.height == 0
        assert frame.columns == list(SCHEMAS[table])

    @pytest.mark.parametrize("table", sorted(SCHEMAS))
    def test_the_empty_form_keeps_its_dtypes(self, table: str) -> None:
        frame = empty(table)
        assert list(frame.schema.values()) == list(SCHEMAS[table].values())

    def test_an_unknown_table_is_refused(self) -> None:
        with pytest.raises(KeyError):
            empty("schedule_wishes")


class TestFinalize:
    def _row(self) -> dict[str, object]:
        return {
            "code": "a_finding",
            "severity": "warn",
            "scope": "run",
            "n_cells": 3,
            "detail": "something worth reporting",
            "schedule_definition_version": "v1",
        }

    def test_column_order_follows_the_contract(self) -> None:
        frame = finalize([self._row()], "schedule_advisories")
        assert frame.columns == list(SCHEMAS["schedule_advisories"])

    def test_rows_are_sorted_by_the_declared_key(self) -> None:
        rows = [
            {**self._row(), "code": "z_finding"},
            {**self._row(), "code": "a_finding"},
        ]
        frame = finalize(rows, "schedule_advisories")
        assert frame["code"].to_list() == ["a_finding", "z_finding"]

    def test_no_rows_produces_the_typed_empty_frame(self) -> None:
        assert finalize([], "schedule_advisories").columns == list(SCHEMAS["schedule_advisories"])

    def test_a_missing_column_is_refused(self) -> None:
        row = self._row()
        del row["detail"]
        with pytest.raises(ValueError, match="missing columns"):
            finalize([row], "schedule_advisories")

    def test_an_unknown_column_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown columns"):
            finalize([{**self._row(), "surprise": 1}], "schedule_advisories")

    def test_an_unknown_table_is_refused(self) -> None:
        with pytest.raises(KeyError):
            finalize([self._row()], "schedule_wishes")

    def test_a_column_that_is_null_throughout_keeps_its_declared_type(self) -> None:
        """Inference would type it from whichever value arrived first, and there is none.

        A date column that is null for the first rows would arrive as ``Null`` rather than
        ``Date``, and the file would no longer match its contract.
        """
        rows = [
            {
                "schedule_config_id": "c",
                "policy_id": "p",
                "model_name": "m",
                "fold_set": "quarterly",
                "fold_id": "f",
                "k_name": "k_1_week",
                "k": 1,
                "target_inspection_id": "T",
                "establishment_id": "E",
                "final_policy_rank": 1,
                "decision_mechanism": "risk_priority",
                "decision_reason": "selected_by_risk_rank",
                "coverage_eligible": False,
                "backlog_position": 1,
                "backlog_reason": "capacity_exhausted_in_horizon",
                "horizon_slots": 0,
                "slots_short": 1,
                "would_fit_on_day_index": None,
                "first_available_date": None,
                "planning_run_id": "PR",
                "replan_index": 0,
                "is_scenario": False,
                "schedule_definition_version": "v1",
            }
        ]
        frame = finalize(rows, "schedule_backlog")
        assert frame.schema["first_available_date"] == pl.Date
        assert frame.schema["would_fit_on_day_index"] == pl.Int64


class TestWriting:
    def test_a_table_round_trips_through_parquet(self, tmp_path: object) -> None:
        frame = finalize(
            [
                {
                    "code": "a_finding",
                    "severity": "warn",
                    "scope": "run",
                    "n_cells": 1,
                    "detail": "d",
                    "schedule_definition_version": "v1",
                }
            ],
            "schedule_advisories",
        )
        path = tmp_path / "advisories.parquet"  # type: ignore[operator]
        write_table(frame, path)
        assert pl.read_parquet(path).equals(frame)

    def test_schema_of_reports_the_dtypes_as_strings(self) -> None:
        frame = empty("schedule_advisories")
        assert schema_of(frame)["n_cells"] == "Int64"


class TestTheContractTableIsSelfDescribing:
    def test_the_execution_contract_names_both_external_files(self) -> None:
        """A contract that lives only in prose drifts from its parser."""
        from sentinel.scheduling.build import _contract_rows

        contracts = {row["contract_name"] for row in _contract_rows()}
        assert contracts == {"scheduling_adjustment", "execution_event"}

    def test_every_contract_field_is_required(self) -> None:
        from sentinel.scheduling.build import _contract_rows

        assert all(row["required"] for row in _contract_rows())

    def test_the_verb_vocabularies_are_published_in_the_table(self) -> None:
        from sentinel.scheduling.build import _contract_rows

        rows = {(r["contract_name"], r["field_name"]): r for r in _contract_rows()}
        assert "defer_to_date" in str(rows[("scheduling_adjustment", "action")]["allowed_values"])
        assert "completed" in str(rows[("execution_event", "execution_status")]["allowed_values"])

    def test_the_contract_table_finalizes(self) -> None:
        from sentinel.scheduling.build import _contract_rows

        frame = finalize(_contract_rows(), "execution_contract")
        assert frame.height == 22
        assert frame.columns == list(SCHEMAS["execution_contract"])
