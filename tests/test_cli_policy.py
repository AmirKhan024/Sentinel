"""The ``sentinel decide`` command: registration, flags, error paths and exit codes.

The exit-code section is the one that matters. An advisory finding -- a reserve that cost
citations, a frontier with no winner -- must leave the run green, because the cheapest way to
turn such a build green is to delete the reserve, and that is a policy decision.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from sentinel import cli
from sentinel.config import Settings
from sentinel.policy.definitions import CANDIDATE_MODELS, REFUSED_MODELS
from sentinel.policy.models import SEVERITY_ERROR, SEVERITY_WARN, ValidationCheck


def _args(**overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = {
        "command": "decide",
        "features": None,
        "calibrated_predictions": None,
        "folds": None,
        "simulation": None,
        "metrics": None,
        "sensitivity": None,
        "categoricals": None,
        "fairness_support": None,
        "overrides": None,
        "output_dir": None,
        "policies": None,
        "model": None,
        "no_figures": True,
        "figures_dir": None,
        "dry_run": True,
        "report": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# --- 1. the parser -----------------------------------------------------------------


def test_the_command_is_registered() -> None:
    args = cli.build_parser().parse_args(["decide"])
    assert args.command == "decide"


def test_the_module_docstring_lists_the_command() -> None:
    """The docstring is the first thing a reader meets; a command absent from it is hidden."""
    assert cli.__doc__ is not None
    assert "sentinel decide" in cli.__doc__


def test_every_artifact_flag_takes_a_path() -> None:
    args = cli.build_parser().parse_args(
        [
            "decide",
            "--features",
            "f.parquet",
            "--calibrated-predictions",
            "c.parquet",
            "--folds",
            "d.parquet",
            "--simulation",
            "s.parquet",
            "--metrics",
            "m.parquet",
            "--sensitivity",
            "n.parquet",
            "--overrides",
            "o.json",
        ]
    )
    for value in (
        args.features,
        args.calibrated_predictions,
        args.folds,
        args.simulation,
        args.metrics,
        args.sensitivity,
        args.overrides,
    ):
        assert isinstance(value, Path)


def test_every_artifact_flag_defaults_to_none_so_the_runner_resolves_the_latest() -> None:
    args = cli.build_parser().parse_args(["decide"])
    assert args.features is None
    assert args.calibrated_predictions is None
    assert args.overrides is None


def test_the_policy_flag_accumulates() -> None:
    args = cli.build_parser().parse_args(
        ["decide", "--policies", "pure_risk", "--policies", "coverage_floor_half_share"]
    )
    assert args.policies == ["pure_risk", "coverage_floor_half_share"]


def test_the_switches_default_off() -> None:
    args = cli.build_parser().parse_args(["decide"])
    assert args.no_figures is False
    assert args.dry_run is False
    assert args.report is False


def test_the_model_help_names_the_admissible_and_the_refused() -> None:
    """A user offered a flag should be told which values it accepts and which are refused."""
    parser = cli.build_parser()
    text = parser.format_help()
    assert "decide" in text
    decide_help = next(
        action
        for action in parser._subparsers._group_actions[0].choices["decide"]._actions  # type: ignore[union-attr]
        if action.dest == "model"
    ).help
    assert decide_help is not None
    assert CANDIDATE_MODELS[0] in decide_help
    assert REFUSED_MODELS[0] in decide_help


# --- 2. error paths ------------------------------------------------------------------


def test_a_missing_artifact_logs_and_returns_one(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing input is a message, not a traceback."""
    with caplog.at_level("ERROR"):
        assert cli._run_decide(_args(), settings) == 1


def test_an_empty_policy_flag_is_a_flag_error_not_a_traceback(settings: Settings) -> None:
    with pytest.raises(SystemExit, match="--policies"):
        cli._run_decide(_args(policies=[]), settings)


def test_a_refused_model_is_rejected_before_any_artifact_is_read(settings: Settings) -> None:
    """Checked before resolution: a bad flag is wrong whether or not the data is on disk."""
    with pytest.raises(SystemExit, match="refused"):
        cli._run_decide(_args(model=REFUSED_MODELS[0]), settings)


def test_an_unknown_model_is_rejected_with_the_admissible_list(settings: Settings) -> None:
    with pytest.raises(SystemExit, match=CANDIDATE_MODELS[0]):
        cli._run_decide(_args(model="not_a_model"), settings)


# --- 3. exit codes -------------------------------------------------------------------


class _Result:
    """The minimum surface ``_run_decide`` touches after a successful run."""

    def __init__(self, checks: list[ValidationCheck]) -> None:
        self.checks = checks
        self.stats = type(
            "S",
            (),
            {
                "selected_model": "xgboost_platt",
                "policies": 7,
                "folds": 18,
                "fold_sets": ["quarterly"],
                "universe_rows": 10,
                "eligible_rows": 2,
                "queue_rows": 3,
                "reserve_rows": 1,
                "inert_cells": 0,
                "overrides_applied": 0,
                "inputs_unchanged": True,
                "advisories": 0,
                "seconds": 1.0,
            },
        )()
        self.selection = type(
            "Sel", (), {"decided_on_axis": "ece", "n_tied_on_nde": 4, "under_discarded_band": "x"}
        )()
        self.winner = None
        self.tables = {"inspection_recommendations": type("T", (), {"height": 10})()}
        self.dry_run = True
        self.written: list[Path] = []
        self.manifest_path = None
        self.figure_paths: list[Path] = []


def _patched(monkeypatch: pytest.MonkeyPatch, checks: list[ValidationCheck]) -> None:
    """Stand in for the run itself, so these tests are about the exit code and nothing else.

    ``_latest`` is patched too: the runner resolves every input path before it calls
    ``run_policy``, so without it these tests would fail on a missing artifact and prove
    nothing about how a check's severity reaches the exit code.
    """
    monkeypatch.setattr(cli, "_latest", lambda *a, **k: Path("stand-in.parquet"))
    monkeypatch.setattr(cli, "run_policy", lambda *a, **k: _Result(checks))
    monkeypatch.setattr(cli, "summarize_policy", lambda result: "SUMMARY")


def test_an_advisory_finding_does_not_change_the_exit_code(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The line ADR 0034 drew, held one component later.

    A reserve that gave up citations is a finding about how a city allocates enforcement. A CI
    runner is not entitled to demand it be deleted.
    """
    advisory = ValidationCheck(
        name="coverage_is_not_free",
        passed=False,
        severity=SEVERITY_WARN,
        detail="gave up 34 citations",
    )
    _patched(monkeypatch, [advisory])
    assert cli._run_decide(_args(), settings) == 0


def test_an_error_severity_failure_returns_one_and_prints_the_report(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    failure = ValidationCheck(
        name="selected_counts_equal_capacity",
        passed=False,
        severity=SEVERITY_ERROR,
        detail="3 cells select more than k",
    )
    _patched(monkeypatch, [failure])
    assert cli._run_decide(_args(), settings) == 1
    assert "selected_counts_equal_capacity" in capsys.readouterr().out


def test_a_clean_run_returns_zero_and_prints_no_report(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    passing = ValidationCheck(
        name="selected_counts_equal_capacity", passed=True, severity=SEVERITY_ERROR, detail="ok"
    )
    _patched(monkeypatch, [passing])
    assert cli._run_decide(_args(), settings) == 0
    out = capsys.readouterr().out
    assert "SUMMARY" in out
    assert "decision policy validation" not in out


def test_the_report_flag_prints_the_boundary_on_a_green_run(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    passing = ValidationCheck(
        name="selected_counts_equal_capacity", passed=True, severity=SEVERITY_ERROR, detail="ok"
    )
    _patched(monkeypatch, [passing])
    assert cli._run_decide(_args(report=True), settings) == 0
    out = capsys.readouterr().out
    assert "It does not mean the policy is the right one." in out


# --- 4. dispatch -----------------------------------------------------------------------


def test_main_dispatches_decide_to_its_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {}

    def _fake(args: argparse.Namespace, settings: Settings) -> int:
        called["yes"] = True
        return 0

    monkeypatch.setattr(cli, "_run_decide", _fake)
    assert cli.main(["decide"]) == 0
    assert called == {"yes": True}
