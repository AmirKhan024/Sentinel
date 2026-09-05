"""``sentinel explain``: the parser, and the handler's exit-code matrix.

Split the way every CLI test file in this project is. The parser tests assert the flag
surface a user meets; the handler tests stub the builder and assert the contract between an
outcome and an exit code -- 0 clean, 1 for an expected failure or a failed error-severity
check, 2 for argparse. A failed *warning* must not fail the command, and that is asserted
too: a warn-level check that took the build red would make every advisory a blocker.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from sentinel import cli
from sentinel.explain.build import ExplainBuildError
from sentinel.explain.definitions import EXPLAIN_REGISTRY, SAMPLE_SIZE, SUPPORTED_MODELS
from sentinel.explain.models import ExplainStats, ValidationCheck
from sentinel.explain.refit import RefitError
from sentinel.explain.validate import SEVERITY_ERROR, SEVERITY_WARN


def _parse(argv: list[str]) -> argparse.Namespace:
    return cli.build_parser().parse_args(argv)


# --- 1. the parser -----------------------------------------------------------


def test_explain_parses_with_no_flags_and_sensible_defaults() -> None:
    args = _parse(["explain"])
    assert args.command == "explain"
    assert args.features is None
    assert args.calibrated_predictions is None
    assert args.models is None
    assert args.sample_size == SAMPLE_SIZE
    assert args.no_figures is False
    assert args.dry_run is False
    assert args.report is False


def test_models_is_repeatable() -> None:
    args = _parse(["explain", "--models", "xgboost", "--models", "lightgbm"])
    assert args.models == ["xgboost", "lightgbm"]


def test_every_path_flag_is_a_path() -> None:
    args = _parse(
        [
            "explain",
            "--features",
            "f.parquet",
            "--baseline-predictions",
            "b.parquet",
            "--boosted-predictions",
            "g.parquet",
            "--neural-predictions",
            "n.parquet",
            "--calibrated-predictions",
            "c.parquet",
            "--output-dir",
            "out",
            "--figures-dir",
            "figs",
        ]
    )
    for value in (
        args.features,
        args.baseline_predictions,
        args.boosted_predictions,
        args.neural_predictions,
        args.calibrated_predictions,
        args.output_dir,
        args.figures_dir,
    ):
        assert isinstance(value, Path)


def test_sample_size_is_an_integer() -> None:
    assert _parse(["explain", "--sample-size", "50"]).sample_size == 50


def test_there_is_no_seed_flag() -> None:
    """The sampling seed is a frozen constant. A seed a caller can change cannot be cited."""
    with pytest.raises(SystemExit):
        _parse(["explain", "--seed", "7"])


def test_the_help_names_every_supported_and_unsupported_model() -> None:
    """A user must be able to learn the support matrix without reading the source."""
    parser = cli.build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    text = subparsers.choices["explain"].format_help()
    for name in SUPPORTED_MODELS:
        assert name in text
    for spec in EXPLAIN_REGISTRY:
        if spec.name not in SUPPORTED_MODELS:
            assert spec.name in text, "an unsupported model must still be discoverable"


def test_the_module_docstring_documents_the_command() -> None:
    """The docstring is the CLI reference, so it is test-enforced like every other one."""
    assert "sentinel explain" in (cli.__doc__ or "")


def test_the_log_level_flag_works_on_either_side_of_the_subcommand() -> None:
    assert _parse(["--log-level", "DEBUG", "explain"]).log_level == "DEBUG"
    assert _parse(["explain", "--log-level", "DEBUG"]).log_level == "DEBUG"


# --- 2. the handler ----------------------------------------------------------


class _Result:
    """The subset of ``ExplainResult`` the handler touches."""

    def __init__(self, checks: list[ValidationCheck]) -> None:
        self.checks = checks
        self.stats = ExplainStats(
            folds=18,
            fold_sets={"quarterly": 17, "covid_shift": 1},
            feature_rows=100,
            models_supported=4,
            models_unsupported=1,
            refits=72,
            explained_rows=5400,
            attribution_values=162000,
            reproduction_rows=41536,
            reproduction_mismatches=0,
            refit_seconds=1.0,
            attribute_seconds=1.0,
        )
        self.tables: dict[str, Any] = {}
        self.values_path: Path | None = Path("explanation_values.parquet")
        self.manifest_path: Path | None = None
        self.written: list[Path] = []
        self.figure_paths: list[Path] = []
        self.dry_run = False


def _check(passed: bool, severity: str = SEVERITY_ERROR) -> ValidationCheck:
    return ValidationCheck(name="a_check", passed=passed, severity=severity, detail="detail")


def _namespace(**overrides: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "features": Path("f.parquet"),
        "baseline_predictions": Path("b.parquet"),
        "boosted_predictions": Path("g.parquet"),
        "neural_predictions": Path("n.parquet"),
        "calibrated_predictions": Path("c.parquet"),
        "output_dir": None,
        "models": None,
        "sample_size": SAMPLE_SIZE,
        "no_figures": False,
        "figures_dir": None,
        "dry_run": False,
        "report": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _patch(monkeypatch: pytest.MonkeyPatch, result: object | Exception) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_run(_settings: Any, **kwargs: Any) -> object:
        seen.update(kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(cli, "run_explanations", fake_run)
    monkeypatch.setattr(cli, "summarize_explanations", lambda _r: "summary")
    return seen


def test_a_clean_run_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _Result([_check(True)]))
    assert cli._run_explain(_namespace(), cli.load_settings()) == 0


def test_a_failed_error_check_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An explanation that is confidently about the wrong thing is worse than none."""
    _patch(monkeypatch, _Result([_check(False)]))
    assert cli._run_explain(_namespace(), cli.load_settings()) == 1
    assert "a_check" in capsys.readouterr().out, "a failure prints the report unasked"


def test_a_failed_warning_still_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warnings never fail a run; otherwise every advisory becomes a blocker."""
    _patch(monkeypatch, _Result([_check(False, SEVERITY_WARN)]))
    assert cli._run_explain(_namespace(), cli.load_settings()) == 0


def test_report_prints_the_full_report_without_changing_the_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch(monkeypatch, _Result([_check(True)]))
    assert cli._run_explain(_namespace(report=True), cli.load_settings()) == 0
    assert "a_check" in capsys.readouterr().out


def test_an_empty_models_list_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _Result([_check(True)]))
    with pytest.raises(SystemExit, match="at least one model"):
        cli._run_explain(_namespace(models=[]), cli.load_settings())


def test_a_non_positive_sample_size_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _Result([_check(True)]))
    with pytest.raises(SystemExit, match="positive integer"):
        cli._run_explain(_namespace(sample_size=0), cli.load_settings())


def test_a_missing_artifact_is_reported_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, FileNotFoundError("no features"))
    assert cli._run_explain(_namespace(), cli.load_settings()) == 1


def test_a_failed_bit_identity_gate_is_reported_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The most important expected failure: the explained model is not the committed one."""
    _patch(monkeypatch, ExplainBuildError("re-executed test scores differ"))
    assert cli._run_explain(_namespace(), cli.load_settings()) == 1


def test_asking_to_explain_an_unsupported_model_is_reported_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, RefitError("is not supported by Component 11"))
    assert cli._run_explain(_namespace(), cli.load_settings()) == 1


def test_the_flags_reach_the_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch(monkeypatch, _Result([_check(True)]))
    cli._run_explain(
        _namespace(dry_run=True, no_figures=True, sample_size=25, models=["xgboost"]),
        cli.load_settings(),
    )
    assert seen["dry_run"] is True
    assert seen["write_figures"] is False
    assert seen["sample_size"] == 25
    assert seen["models"] == ["xgboost"]
    assert seen["calibrated_path"] == Path("c.parquet")


def test_an_absent_calibrated_artifact_is_a_supported_state_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Component 9's output is carried alongside an explanation, never explained by it."""
    seen = _patch(monkeypatch, _Result([_check(True)]))
    monkeypatch.setattr(
        cli,
        "_latest",
        lambda _s, _d, prefix, label: (
            (_ for _ in ()).throw(FileNotFoundError(label))
            if prefix == "calibrated_predictions_"
            else tmp_path / f"{prefix}x.parquet"
        ),
    )
    assert (
        cli._run_explain(
            _namespace(
                features=None,
                baseline_predictions=None,
                boosted_predictions=None,
                neural_predictions=None,
                calibrated_predictions=None,
            ),
            cli.load_settings(),
        )
        == 0
    )
    assert seen["calibrated_path"] is None


def test_explain_is_dispatched_from_main(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {}

    def fake(_args: argparse.Namespace, _settings: Any) -> int:
        called["yes"] = True
        return 0

    monkeypatch.setattr(cli, "_run_explain", fake)
    assert cli.main(["explain"]) == 0
    assert called == {"yes": True}
