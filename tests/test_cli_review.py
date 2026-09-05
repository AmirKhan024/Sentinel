"""The ``sentinel review`` command: registration, flags, error paths and exit codes.

The absent-flag section is the one that encodes the "no fabricated threshold" invariant at the
CLI surface. There is deliberately no way from the command line to introduce a probability or
confidence cutoff.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from sentinel import cli
from sentinel.config import Settings
from sentinel.review.models import SEVERITY_ERROR, SEVERITY_WARN, ValidationCheck


def _args(**overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = {
        "command": "review",
        "recommendations": None,
        "schedule": None,
        "execution": None,
        "resolutions": None,
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
        args = parser.parse_args(["review", "--dry-run"])
        assert args.command == "review"
        assert args.dry_run

    def test_policies_and_k_names_are_repeatable(self) -> None:
        args = cli.build_parser().parse_args(
            ["review", "--policies", "pure_risk", "--k-names", "k_1_day", "--k-names", "k_1_week"]
        )
        assert args.policies == ["pure_risk"]
        assert args.k_names == ["k_1_day", "k_1_week"]

    def test_optional_paths_default_to_none(self) -> None:
        args = cli.build_parser().parse_args(["review"])
        assert args.schedule is None
        assert args.execution is None
        assert args.resolutions is None


class TestTheAbsentFlags:
    """Each of these would be exactly the invented statistic ADR 0040 refuses."""

    @pytest.mark.parametrize(
        "flag",
        [
            "--threshold",
            "--probability-threshold",
            "--confidence-threshold",
            "--score-cutoff",
            "--min-confidence",
        ],
    )
    def test_no_flag_can_introduce_a_threshold(self, flag: str) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["review", flag, "0.5"])


class TestFlagValidation:
    def test_an_empty_policy_list_is_refused(self, settings: Settings) -> None:
        with pytest.raises(SystemExit, match="at least one policy"):
            cli._run_review(_args(policies=[]), settings)

    def test_an_empty_capacity_list_is_refused(self, settings: Settings) -> None:
        with pytest.raises(SystemExit, match="at least one capacity"):
            cli._run_review(_args(k_names=[]), settings)

    def test_flags_are_checked_before_artifacts_are_resolved(self, settings: Settings) -> None:
        """A malformed flag is wrong whether or not the data happens to be on disk."""
        with pytest.raises(SystemExit):
            cli._run_review(_args(policies=[]), settings)


class TestArtifactResolution:
    def test_a_missing_recommendation_artifact_exits_one(self, settings: Settings) -> None:
        assert cli._run_review(_args(), settings) == 1

    def test_a_named_missing_file_exits_one(self, settings: Settings, tmp_path: Path) -> None:
        assert cli._run_review(_args(recommendations=tmp_path / "absent.parquet"), settings) == 1


class TestExitCodes:
    def _result(self, checks: list[ValidationCheck]) -> Any:
        from sentinel.review.build import ReviewResult
        from sentinel.review.models import ReviewStats
        from sentinel.review.writer import SCHEMAS, empty

        return ReviewResult(
            tables={name: empty(name) for name in SCHEMAS},
            checks=checks,
            manifest=None,  # type: ignore[arg-type]
            stats=ReviewStats(),
            advisories=[],
        )

    def _run_with(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch, checks: list[ValidationCheck]
    ) -> int:
        result = self._result(checks)
        monkeypatch.setattr(cli, "run_review", lambda *a, **k: result)
        monkeypatch.setattr(cli, "_latest", lambda *a, **k: Path("x.parquet"))
        return cli._run_review(_args(), settings)

    def test_a_clean_run_exits_zero(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checks = [ValidationCheck("ok", True, SEVERITY_ERROR, "fine")]
        assert self._run_with(settings, monkeypatch, checks) == 0

    def test_an_advisory_finding_still_exits_zero(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checks = [ValidationCheck("cases_flagged_by_trigger", False, SEVERITY_WARN, "70791")]
        assert self._run_with(settings, monkeypatch, checks) == 0

    def test_an_error_finding_exits_one(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checks = [ValidationCheck("every_case_carries_a_trigger", False, SEVERITY_ERROR, "bad")]
        assert self._run_with(settings, monkeypatch, checks) == 1

    def test_the_report_prints_on_failure_without_being_asked(
        self,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        checks = [ValidationCheck("every_case_carries_a_trigger", False, SEVERITY_ERROR, "bad")]
        self._run_with(settings, monkeypatch, checks)
        assert "every_case_carries_a_trigger" in capsys.readouterr().out

    def test_the_summary_always_prints(
        self,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run_with(settings, monkeypatch, [ValidationCheck("ok", True, SEVERITY_ERROR, "f")])
        assert "deferral / human-review gate" in capsys.readouterr().out
