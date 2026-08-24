"""The ``sentinel calibrate`` subcommand: argument handling, exit codes and defaults.

The exit code is the part that matters operationally. A failed error-severity check means
either the re-executed base model is not the one Components 6-8 published, or a calibrator
read a window it was not allowed to. Both make every probability in the artifact
untrustworthy, so the command must exit non-zero rather than writing a plausible file and
printing a warning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from sentinel import cli
from sentinel.calibration.definitions import BOOTSTRAP_REPLICATIONS, CANDIDATE_REGISTRY, Method
from sentinel.calibration.models import ValidationCheck


def _parse(argv: list[str]) -> argparse.Namespace:
    return cli.build_parser().parse_args(argv)


# --- the parser ---------------------------------------------------------------


def test_calibrate_is_registered_with_sensible_defaults() -> None:
    args = _parse(["calibrate"])
    assert args.command == "calibrate"
    assert args.features is None
    assert args.models is None
    assert args.method is None
    assert args.dry_run is False
    assert args.report is False
    assert args.no_figures is False
    assert args.bootstrap_replications == BOOTSTRAP_REPLICATIONS


def test_models_is_repeatable() -> None:
    args = _parse(["calibrate", "--models", "xgboost", "--models", "lightgbm"])
    assert args.models == ["xgboost", "lightgbm"]


def test_every_candidate_is_named_in_the_help_text() -> None:
    """A reader should not have to open the source to learn what will be calibrated."""
    help_text = cli.build_parser().format_help()
    assert "calibrate" in help_text
    for spec in CANDIDATE_REGISTRY:
        assert spec.name in cli.build_parser()._subparsers._group_actions[0].choices[  # type: ignore[union-attr]
            "calibrate"
        ].format_help()


def test_method_only_accepts_an_implemented_method() -> None:
    assert _parse(["calibrate", "--method", "platt"]).method == "platt"
    assert _parse(["calibrate", "--method", "isotonic"]).method == "isotonic"
    with pytest.raises(SystemExit):
        _parse(["calibrate", "--method", "temperature"])


def test_paths_are_parsed_as_paths() -> None:
    args = _parse([
        "calibrate",
        "--features", "a.parquet",
        "--categoricals", "b.parquet",
        "--baseline-predictions", "c.parquet",
        "--boosted-predictions", "d.parquet",
        "--neural-predictions", "e.parquet",
        "--output-dir", "out",
        "--figures-dir", "figs",
    ])
    for value in (
        args.features, args.categoricals, args.baseline_predictions,
        args.boosted_predictions, args.neural_predictions, args.output_dir, args.figures_dir,
    ):
        assert isinstance(value, Path)


def test_the_usage_block_documents_the_command() -> None:
    assert "sentinel calibrate" in (cli.__doc__ or "")


# --- the handler --------------------------------------------------------------


class _Result:
    """The minimum surface ``_run_calibrate`` touches."""

    def __init__(self, checks: list[ValidationCheck]) -> None:
        self.checks = checks
        self.predictions_path = None
        self.manifest_path = None
        self.figure_paths: list[Path] = []


def _namespace(**overrides: Any) -> argparse.Namespace:
    base = dict(
        features=Path("f.parquet"),
        categoricals=Path("c.parquet"),
        baseline_predictions=Path("b.parquet"),
        boosted_predictions=Path("g.parquet"),
        neural_predictions=Path("n.parquet"),
        output_dir=None,
        models=None,
        method=None,
        bootstrap_replications=10,
        no_figures=True,
        figures_dir=None,
        dry_run=True,
        report=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _patch(monkeypatch: pytest.MonkeyPatch, checks: list[ValidationCheck]) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_run(settings: Any, **kwargs: Any) -> _Result:
        seen.update(kwargs)
        return _Result(checks)

    monkeypatch.setattr(cli, "run_calibration", fake_run)
    monkeypatch.setattr(cli, "summarize_calibration", lambda result: "summary")
    return seen


def test_a_clean_run_exits_zero(monkeypatch: pytest.MonkeyPatch, settings: Any) -> None:
    _patch(monkeypatch, [ValidationCheck("ok", True, "error", "fine")])
    assert cli._run_calibrate(_namespace(), settings) == 0


def test_a_failed_error_check_exits_one(monkeypatch: pytest.MonkeyPatch, settings: Any) -> None:
    """The gate is operational, not decorative."""
    _patch(
        monkeypatch,
        [ValidationCheck("base_scores_reproduce_the_committed_artifact", False, "error", "no")],
    )
    assert cli._run_calibrate(_namespace(), settings) == 1


def test_a_failed_warn_check_still_exits_zero(
    monkeypatch: pytest.MonkeyPatch, settings: Any
) -> None:
    """Isotonic creating ties is a note, not a failure."""
    _patch(monkeypatch, [ValidationCheck("isotonic_ties", False, "warn", "note")])
    assert cli._run_calibrate(_namespace(), settings) == 0


def test_the_method_override_is_passed_through_as_an_enum(
    monkeypatch: pytest.MonkeyPatch, settings: Any
) -> None:
    seen = _patch(monkeypatch, [])
    cli._run_calibrate(_namespace(method="isotonic"), settings)
    assert seen["method_override"] is Method.ISOTONIC


def test_no_override_means_the_protocol_decides(
    monkeypatch: pytest.MonkeyPatch, settings: Any
) -> None:
    seen = _patch(monkeypatch, [])
    cli._run_calibrate(_namespace(), settings)
    assert seen["method_override"] is None


def test_an_empty_models_list_is_rejected(monkeypatch: pytest.MonkeyPatch, settings: Any) -> None:
    _patch(monkeypatch, [])
    with pytest.raises(SystemExit, match="at least one model"):
        cli._run_calibrate(_namespace(models=[]), settings)


def test_a_non_positive_replication_count_is_rejected(
    monkeypatch: pytest.MonkeyPatch, settings: Any
) -> None:
    _patch(monkeypatch, [])
    with pytest.raises(SystemExit, match="positive integer"):
        cli._run_calibrate(_namespace(bootstrap_replications=0), settings)


def test_a_missing_input_is_reported_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, settings: Any
) -> None:
    """Expected failures are logged and returned as exit 1, never raised at the user."""
    def fake_run(settings_: Any, **kwargs: Any) -> _Result:
        raise FileNotFoundError("features: nothing matching as_of_features_")

    monkeypatch.setattr(cli, "run_calibration", fake_run)
    monkeypatch.setattr(cli, "summarize_calibration", lambda result: "summary")
    assert cli._run_calibrate(_namespace(), settings) == 1


def test_a_build_error_is_reported_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, settings: Any
) -> None:
    def fake_run(settings_: Any, **kwargs: Any) -> _Result:
        raise cli.CalibrationBuildError("regenerated scores differ from the committed artifact")

    monkeypatch.setattr(cli, "run_calibration", fake_run)
    monkeypatch.setattr(cli, "summarize_calibration", lambda result: "summary")
    assert cli._run_calibrate(_namespace(), settings) == 1


def test_the_dry_run_flag_reaches_the_builder(
    monkeypatch: pytest.MonkeyPatch, settings: Any
) -> None:
    seen = _patch(monkeypatch, [])
    cli._run_calibrate(_namespace(dry_run=True), settings)
    assert seen["dry_run"] is True


def test_no_figures_reaches_the_builder_inverted(
    monkeypatch: pytest.MonkeyPatch, settings: Any
) -> None:
    seen = _patch(monkeypatch, [])
    cli._run_calibrate(_namespace(no_figures=True), settings)
    assert seen["write_figures"] is False
