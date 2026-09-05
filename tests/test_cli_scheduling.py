"""The ``sentinel schedule`` command: registration, flags, error paths and exit codes.

The exit-code section is the one that matters. An advisory finding -- a backlog, idle capacity,
a coverage reserve lost to a short horizon -- must leave the run green, because the cheapest way
to turn such a build green is to make the scheduler prefer reserve rows, and that is re-ranking.

The absent-flag section matters almost as much. There is deliberately no way from the command
line to raise capacity, extend a horizon or introduce a probability threshold, and a test that
asserts the absence is what stops one being added for convenience later.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from sentinel import cli
from sentinel.config import Settings
from sentinel.scheduling.definitions import CONFIG_GRID, K_LEVELS, CapacityMode
from sentinel.scheduling.models import SEVERITY_ERROR, SEVERITY_WARN, ValidationCheck


def _args(**overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = {
        "command": "schedule",
        "recommendations": None,
        "folds": None,
        "override_log": None,
        "adjustments": None,
        "execution": None,
        "capacity_mode": "both",
        "policies": None,
        "k_names": None,
        "output_dir": None,
        "no_figures": True,
        "figures_dir": None,
        "dry_run": True,
        "report": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestRegistration:
    def test_the_command_parses(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["schedule", "--dry-run"])
        assert args.command == "schedule"
        assert args.dry_run

    def test_the_capacity_mode_defaults_to_both(self) -> None:
        """So the scenario's divergence from the real calendar is always visible, not opt-in."""
        args = cli.build_parser().parse_args(["schedule"])
        assert args.capacity_mode == "both"

    @pytest.mark.parametrize("mode", [str(m) for m in CapacityMode] + ["both"])
    def test_every_capacity_mode_is_accepted(self, mode: str) -> None:
        args = cli.build_parser().parse_args(["schedule", "--capacity-mode", mode])
        assert args.capacity_mode == mode

    def test_an_unknown_capacity_mode_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["schedule", "--capacity-mode", "wishful"])

    def test_policies_and_k_names_are_repeatable(self) -> None:
        args = cli.build_parser().parse_args(
            ["schedule", "--policies", "pure_risk", "--k-names", "k_1_day", "--k-names", "k_1_week"]
        )
        assert args.policies == ["pure_risk"]
        assert args.k_names == ["k_1_day", "k_1_week"]


class TestTheAbsentFlags:
    """Each of these would be a way to make a scheduling number better without scheduling better."""

    @pytest.mark.parametrize(
        "flag",
        [
            "--capacity",
            "--slots-per-day",
            "--horizon-days",
            "--extend-horizon",
            "--threshold",
            "--probability-threshold",
            "--reserve-share",
        ],
    )
    def test_no_flag_can_raise_capacity_or_introduce_a_threshold(self, flag: str) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["schedule", flag, "1"])


class TestFlagValidation:
    def test_an_empty_policy_list_is_refused(self, settings: Settings) -> None:
        with pytest.raises(SystemExit, match="at least one policy"):
            cli._run_schedule(_args(policies=[]), settings)

    def test_an_empty_capacity_list_is_refused(self, settings: Settings) -> None:
        with pytest.raises(SystemExit, match="at least one capacity"):
            cli._run_schedule(_args(k_names=[]), settings)

    def test_an_unknown_capacity_level_is_refused_and_the_known_ones_named(
        self, settings: Settings
    ) -> None:
        with pytest.raises(SystemExit, match="known:"):
            cli._run_schedule(_args(k_names=["k_1_fortnight"]), settings)

    def test_a_known_capacity_level_passes_the_flag_check(self, settings: Settings) -> None:
        """Reaches the artifact lookup and fails there, which is the next step, not this one."""
        assert cli._run_schedule(_args(k_names=[K_LEVELS[0]]), settings) == 1

    def test_flags_are_checked_before_artifacts_are_resolved(self, settings: Settings) -> None:
        """A malformed flag is wrong whether or not the data happens to be on disk."""
        with pytest.raises(SystemExit):
            cli._run_schedule(_args(k_names=["nonsense"]), settings)


class TestArtifactResolution:
    def test_a_missing_recommendation_artifact_exits_one(self, settings: Settings) -> None:
        assert cli._run_schedule(_args(), settings) == 1

    def test_a_named_missing_file_exits_one(self, settings: Settings, tmp_path: Path) -> None:
        assert cli._run_schedule(_args(recommendations=tmp_path / "absent.parquet"), settings) == 1


class TestCapacityModeSelection:
    @pytest.mark.parametrize("mode", [str(m) for m in CapacityMode])
    def test_a_single_mode_selects_one_configuration(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run(_settings: Settings, **kwargs: Any) -> Any:
            captured.update(kwargs)
            raise FileNotFoundError("stop here")

        monkeypatch.setattr(cli, "run_schedule", fake_run)
        monkeypatch.setattr(cli, "_latest", lambda *a, **k: Path("x.parquet"))
        cli._run_schedule(_args(capacity_mode=mode), settings)
        assert [str(c.capacity_mode) for c in captured["configs"]] == [mode]

    def test_both_selects_the_whole_grid(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run(_settings: Settings, **kwargs: Any) -> Any:
            captured.update(kwargs)
            raise FileNotFoundError("stop here")

        monkeypatch.setattr(cli, "run_schedule", fake_run)
        monkeypatch.setattr(cli, "_latest", lambda *a, **k: Path("x.parquet"))
        cli._run_schedule(_args(capacity_mode="both"), settings)
        assert len(captured["configs"]) == len(CONFIG_GRID)


class TestExitCodes:
    def _result(self, checks: list[ValidationCheck]) -> Any:
        from sentinel.scheduling.build import ScheduleResult
        from sentinel.scheduling.models import ScheduleStats
        from sentinel.scheduling.writer import LAYERS, empty

        return ScheduleResult(
            tables={name: empty(name) for name in LAYERS},
            checks=checks,
            stats=ScheduleStats(),
        )

    def _run_with(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch, checks: list[ValidationCheck]
    ) -> int:
        result = self._result(checks)
        monkeypatch.setattr(cli, "run_schedule", lambda *a, **k: result)
        monkeypatch.setattr(cli, "_latest", lambda *a, **k: Path("x.parquet"))
        return cli._run_schedule(_args(), settings)

    def test_a_clean_run_exits_zero(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checks = [ValidationCheck("ok", True, SEVERITY_ERROR, "fine")]
        assert self._run_with(settings, monkeypatch, checks) == 0

    def test_an_advisory_finding_still_exits_zero(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The line ADR 0034 drew, inherited one layer further out.

        The cheapest way to turn a red "this schedule lost 1,012 reserve slots" build green is
        to make the scheduler prefer reserve rows -- which is re-ranking, and not a change a CI
        runner is entitled to make.
        """
        checks = [
            ValidationCheck(
                "the_coverage_reserve_survived_scheduling", False, SEVERITY_WARN, "1012 lost"
            ),
            ValidationCheck("every_recommendation_was_scheduled", False, SEVERITY_WARN, "5488"),
        ]
        assert self._run_with(settings, monkeypatch, checks) == 0

    def test_an_error_finding_exits_one(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checks = [ValidationCheck("no_day_exceeds_its_capacity", False, SEVERITY_ERROR, "bad")]
        assert self._run_with(settings, monkeypatch, checks) == 1

    def test_the_report_prints_on_failure_without_being_asked(
        self,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        checks = [ValidationCheck("no_day_exceeds_its_capacity", False, SEVERITY_ERROR, "bad")]
        self._run_with(settings, monkeypatch, checks)
        assert "no_day_exceeds_its_capacity" in capsys.readouterr().out

    def test_the_summary_always_prints(
        self,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run_with(settings, monkeypatch, [ValidationCheck("ok", True, SEVERITY_ERROR, "f")])
        assert "coverage reserve" in capsys.readouterr().out


class TestTheHelpText:
    def test_the_help_names_the_scenario_as_a_scenario(self) -> None:
        parser = cli.build_parser()
        for action in parser._subparsers._group_actions[0].choices["schedule"]._actions:  # type: ignore[union-attr,attr-defined]
            if action.dest == "capacity_mode":
                assert "scenario" in (action.help or "")
                return
        raise AssertionError("--capacity-mode is not registered")
