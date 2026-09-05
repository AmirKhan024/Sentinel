"""The `sentinel audit-fairness` command: flags, defaults, exit codes and error shape.

Two things this file cares about beyond wiring.

**An expected failure must not produce a traceback.** A missing artifact and a refused group
definition are ordinary outcomes of running the command wrong, and both should print a line
and exit 1.

**An advisory finding must not change the exit code.** That is ADR 0034 expressed at the one
place a build system reads: if a measured disparity turned the exit code red, every future
change would be made under pressure to move the number.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from sentinel import cli
from sentinel.config import Settings
from sentinel.fairness.models import SEVERITY_ERROR, SEVERITY_WARN, ValidationCheck


def _args(**overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = {
        "features": None,
        "calibrated_predictions": None,
        "categoricals": None,
        "explanations": None,
        "output_dir": None,
        "models": None,
        "group_definitions": None,
        "no_figures": True,
        "figures_dir": None,
        "dry_run": True,
        "report": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# --- 1. the parser ---------------------------------------------------------------


def test_the_command_is_registered() -> None:
    args = cli.build_parser().parse_args(["audit-fairness"])
    assert args.command == "audit-fairness"


def test_the_path_flags_default_to_none_so_the_runner_resolves_the_latest() -> None:
    args = cli.build_parser().parse_args(["audit-fairness"])
    assert args.features is None
    assert args.calibrated_predictions is None
    assert args.categoricals is None
    assert args.explanations is None


def test_the_repeatable_flags_accumulate() -> None:
    args = cli.build_parser().parse_args(
        [
            "audit-fairness",
            "--models",
            "xgboost_platt",
            "--models",
            "lightgbm_platt",
            "--group-definitions",
            "community_area",
        ]
    )
    assert args.models == ["xgboost_platt", "lightgbm_platt"]
    assert args.group_definitions == ["community_area"]


def test_the_store_true_flags_default_off() -> None:
    args = cli.build_parser().parse_args(["audit-fairness"])
    assert args.dry_run is False
    assert args.report is False
    assert args.no_figures is False


def test_paths_are_parsed_as_paths() -> None:
    args = cli.build_parser().parse_args(
        ["audit-fairness", "--features", "f.parquet", "--output-dir", "out"]
    )
    assert args.features == Path("f.parquet")
    assert args.output_dir == Path("out")


def test_the_help_names_the_refused_definitions() -> None:
    """A user who types `ward` should be able to find out why not from the help alone."""
    assert "ward" in cli.REFUSED_DEFINITIONS
    assert "census_tract" in cli.REFUSED_DEFINITIONS


def test_the_module_docstring_lists_the_command() -> None:
    assert "sentinel audit-fairness" in (cli.__doc__ or "")


# --- 2. missing inputs fail without a traceback ------------------------------------


def test_a_missing_artifact_logs_and_returns_one(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _absent(*args: Any, **kwargs: Any) -> Path:
        raise FileNotFoundError("no calibrated predictions found")

    monkeypatch.setattr(cli, "_latest", _absent)
    assert cli._run_audit_fairness(_args(), settings) == 1


def test_an_empty_models_list_is_a_flag_error_not_a_traceback(settings: Settings) -> None:
    with pytest.raises(SystemExit, match="--models requires at least one"):
        cli._run_audit_fairness(_args(models=[]), settings)


def test_an_empty_group_definitions_list_is_a_flag_error(settings: Settings) -> None:
    with pytest.raises(SystemExit, match="--group-definitions requires at least one"):
        cli._run_audit_fairness(_args(group_definitions=[]), settings)


def test_a_refused_group_definition_logs_and_returns_one(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The registry raises; the CLI turns that into a line and an exit code."""
    monkeypatch.setattr(cli, "_latest", lambda *a, **k: tmp_path / "x.parquet")
    assert cli._run_audit_fairness(_args(group_definitions=["ward"]), settings) == 1


# --- 3. exit codes ------------------------------------------------------------------


class _Result:
    """Just enough of a FairnessResult for the runner's tail."""

    def __init__(self, checks: list[ValidationCheck]) -> None:
        self.checks = checks
        self.advisories: list[str] = []
        self.written: list[Path] = []
        self.figure_paths: list[Path] = []
        self.manifest_path: Path | None = None
        self.dry_run = True
        self.definitions = ["community_area"]
        from sentinel.fairness.models import FairnessStats

        self.stats = FairnessStats()


def _run_with(
    checks: list[ValidationCheck],
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    report: bool = False,
) -> int:
    monkeypatch.setattr(cli, "_latest", lambda *a, **k: tmp_path / "x.parquet")
    monkeypatch.setattr(cli, "run_fairness_audit", lambda *a, **k: _Result(checks))
    return cli._run_audit_fairness(_args(report=report), settings)


def test_an_advisory_finding_does_not_change_the_exit_code(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ADR 0034, expressed at the one place a build system reads.

    If a measured disparity turned the exit code red, the cheapest ways to turn it green
    would be to change the metric or move the threshold -- both worse than the disparity.
    """
    advisory = ValidationCheck(
        name="group_calibration_spread_is_modest",
        passed=False,
        severity=SEVERITY_WARN,
        detail="every cell exceeds the spread",
    )
    assert _run_with([advisory], settings, monkeypatch, tmp_path) == 0


def test_an_error_severity_failure_returns_one(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failure = ValidationCheck(
        name="inputs_were_not_modified",
        passed=False,
        severity=SEVERITY_ERROR,
        detail="an input changed during the run",
    )
    assert _run_with([failure], settings, monkeypatch, tmp_path) == 1


def test_an_all_green_run_returns_zero(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    passing = ValidationCheck(
        name="no_group_disappeared", passed=True, severity=SEVERITY_ERROR, detail="all present"
    )
    assert _run_with([passing], settings, monkeypatch, tmp_path) == 0


def test_the_report_flag_prints_the_boundary(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    passing = ValidationCheck(
        name="no_group_disappeared", passed=True, severity=SEVERITY_ERROR, detail="all present"
    )
    _run_with([passing], settings, monkeypatch, tmp_path, report=True)
    printed = capsys.readouterr().out
    assert "does NOT mean Sentinel is" in printed
    assert "DOES NOT ESTABLISH" in printed


def test_a_failing_run_prints_the_report_even_without_the_flag(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    failure = ValidationCheck(
        name="stages_are_not_confused",
        passed=False,
        severity=SEVERITY_ERROR,
        detail="base and calibrated were swapped",
    )
    _run_with([failure], settings, monkeypatch, tmp_path)
    assert "stages_are_not_confused" in capsys.readouterr().out


# --- 4. dispatch ----------------------------------------------------------------------


def test_main_dispatches_to_the_fairness_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _record(args: argparse.Namespace, settings: Settings) -> int:
        seen["command"] = args.command
        return 0

    monkeypatch.setattr(cli, "_run_audit_fairness", _record)
    assert cli.main(["audit-fairness"]) == 0
    assert seen["command"] == "audit-fairness"
