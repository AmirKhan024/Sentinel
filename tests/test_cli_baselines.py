"""CLI coverage for `train-baselines` and `evaluate --predictions`.

Kept beside `test_cli.py` rather than inside it: that file is already 600 lines and
organised per subcommand, and the two Component 6 entry points are one story -- train,
then hand the artifact to the evaluator.

The parsing tests matter as much as the end-to-end ones. `--models` is `action="append"`,
which is easy to get wrong in a way that silently trains one model instead of three.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from sentinel.cli import build_parser, main
from sentinel.config import Settings
from sentinel.modeling.definitions import MODEL_REGISTRY


def parse(argv: list[str]):  # type: ignore[no-untyped-def]
    return build_parser().parse_args(argv)


def _features(tmp_path: Path, *, days: int = 1900) -> Path:
    from tests.conftest import spanning_model_features

    path = tmp_path / "as_of_features_20260817T000000Z.parquet"
    spanning_model_features(days=days).write_parquet(path)
    return path


# --- parsing ----------------------------------------------------------------


def test_train_baselines_defaults_to_every_model() -> None:
    args = parse(["train-baselines"])
    assert args.command == "train-baselines"
    assert args.models is None  # type: ignore[attr-defined]
    assert args.features is None  # type: ignore[attr-defined]
    assert args.dry_run is False  # type: ignore[attr-defined]
    assert args.report is False  # type: ignore[attr-defined]


def test_models_flag_is_repeatable() -> None:
    args = parse(
        [
            "train-baselines",
            "--models",
            "logistic_regression",
            "--models",
            "cdph_2015_approximation",
        ]
    )
    assert args.models == ["logistic_regression", "cdph_2015_approximation"]  # type: ignore[attr-defined]


def test_train_baselines_accepts_paths_and_flags(tmp_path: Path) -> None:
    args = parse(
        [
            "train-baselines",
            "--features",
            str(tmp_path / "f.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--report",
        ]
    )
    assert args.features == tmp_path / "f.parquet"  # type: ignore[attr-defined]
    assert args.output_dir == tmp_path / "out"  # type: ignore[attr-defined]
    assert args.dry_run is True  # type: ignore[attr-defined]
    assert args.report is True  # type: ignore[attr-defined]


def test_train_baselines_log_level_works_on_either_side() -> None:
    before = parse(["--log-level", "DEBUG", "train-baselines"])
    after = parse(["train-baselines", "--log-level", "DEBUG"])
    assert before.log_level == "DEBUG"  # type: ignore[attr-defined]
    assert after.log_level == "DEBUG"  # type: ignore[attr-defined]


def test_evaluate_predictions_defaults_to_none() -> None:
    assert parse(["evaluate"]).predictions is None  # type: ignore[attr-defined]


def test_evaluate_accepts_a_predictions_path(tmp_path: Path) -> None:
    args = parse(["evaluate", "--predictions", str(tmp_path / "p.parquet")])
    assert args.predictions == tmp_path / "p.parquet"  # type: ignore[attr-defined]


def test_the_command_catalogue_lists_both_invocations() -> None:
    """The module docstring is the canonical catalogue; a new command must appear in it."""
    import sentinel.cli as cli

    assert cli.__doc__ is not None
    assert "sentinel train-baselines" in cli.__doc__
    assert "sentinel evaluate --predictions" in cli.__doc__


# --- train-baselines end to end ---------------------------------------------


def test_train_baselines_writes_and_exits_zero(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = _features(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)

    out = tmp_path / "predictions"
    code = main(["train-baselines", "--features", str(features), "--output-dir", str(out)])
    assert code == 0
    assert list(out.glob("baseline_predictions_*.parquet"))
    assert list(out.glob("baseline_coefficients_*.parquet"))
    assert list(out.glob("baseline_training_log_*.parquet"))
    assert list(out.glob("manifest_baseline_predictions_*.json"))


def test_train_baselines_dry_run_writes_nothing(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = _features(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)

    out = tmp_path / "predictions"
    code = main(
        ["train-baselines", "--features", str(features), "--output-dir", str(out), "--dry-run"]
    )
    assert code == 0
    assert not out.exists()


def test_train_baselines_honours_a_single_model(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = _features(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)

    out = tmp_path / "predictions"
    code = main(
        [
            "train-baselines",
            "--features",
            str(features),
            "--output-dir",
            str(out),
            "--models",
            "logistic_regression",
        ]
    )
    assert code == 0
    written = next(iter(out.glob("baseline_predictions_*.parquet")))
    assert set(pl.read_parquet(written)["model_name"].unique()) == {"logistic_regression"}


def test_train_baselines_reports_an_unknown_model_without_a_traceback(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = _features(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)
    code = main(
        ["train-baselines", "--features", str(features), "--models", "xgboost", "--dry-run"]
    )
    assert code == 1


def test_train_baselines_reports_a_missing_input_without_a_traceback(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)
    code = main(["train-baselines", "--features", str(tmp_path / "absent.parquet"), "--dry-run"])
    assert code == 1


def test_train_baselines_reports_when_no_upstream_files_exist(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)
    assert main(["train-baselines", "--dry-run"]) == 1


def test_train_baselines_exits_one_when_a_check_fails(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed check means a model may have seen data it was not allowed to."""
    from sentinel.modeling.models import ValidationCheck

    features = _features(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)
    monkeypatch.setattr(
        "sentinel.modeling.build.validate.validate_baselines",
        lambda *a, **k: [
            ValidationCheck(name="forced", passed=False, severity="error", detail="forced")
        ],
    )
    code = main(["train-baselines", "--features", str(features), "--dry-run"])
    assert code == 1


def test_train_baselines_prints_no_metric(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Component 6 must not report a number Component 5 owns."""
    features = _features(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)
    main(["train-baselines", "--features", str(features), "--dry-run"])
    out = capsys.readouterr().out
    assert "evaluate --predictions" in out
    for metric in ("roc_auc", "ROC-AUC", "pr_auc", "NDE", "precision@k"):
        assert metric not in out


def test_report_flag_prints_the_validation_report(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    features = _features(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)
    main(["train-baselines", "--features", str(features), "--dry-run", "--report"])
    out = capsys.readouterr().out
    assert "Baseline model validation report" in out
    assert "trained_through_is_the_training_end" in out


# --- the two commands together ----------------------------------------------


def test_train_then_evaluate_the_artifact(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented Component 6 workflow, end to end through the CLI."""
    features = _features(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)

    predictions_dir = tmp_path / "predictions"
    assert (
        main(
            [
                "train-baselines",
                "--features",
                str(features),
                "--output-dir",
                str(predictions_dir),
            ]
        )
        == 0
    )
    artifact = next(iter(predictions_dir.glob("baseline_predictions_*.parquet")))

    evaluation_dir = tmp_path / "evaluation"
    code = main(
        [
            "evaluate",
            "--features",
            str(features),
            "--predictions",
            str(artifact),
            "--output-dir",
            str(evaluation_dir),
            "--seeds",
            "2",
            "--sensitivity-replications",
            "3",
        ]
    )
    assert code == 0

    metrics = next(iter(evaluation_dir.glob("evaluation_metrics_*.parquet")))
    scored = set(pl.read_parquet(metrics)["model_name"].unique())
    for spec in MODEL_REGISTRY:
        assert spec.name in scored


def test_evaluate_reports_a_missing_predictions_file_without_a_traceback(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = _features(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)
    code = main(
        [
            "evaluate",
            "--features",
            str(features),
            "--predictions",
            str(tmp_path / "absent.parquet"),
            "--dry-run",
            "--seeds",
            "2",
            "--sensitivity-replications",
            "2",
        ]
    )
    assert code == 1
