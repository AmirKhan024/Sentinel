"""The two Component 7 subcommands: parsing, exit codes, and what they must not print.

Kept beside ``test_cli_baselines.py`` rather than folded into ``test_cli.py``, which is
already long. The pattern is that file's: parse-only tests that never touch the
filesystem, end-to-end tests that do, and one test asserting the training command reports
no metric Component 5 owns -- because two answers to "how good is this model?" is one
answer too many.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from sentinel import cli
from sentinel.boosting.definitions import BOOSTING_REGISTRY, TUNABLE_MODELS, TUNING_SEED
from tests.conftest import spanning_model_features


@pytest.fixture(scope="module")
def features(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("cli_boosting_input")
    path = tmp / "as_of_features_20260101T000000Z.parquet"
    spanning_model_features(days=1900).write_parquet(path)
    return path


# --- 1. parsing --------------------------------------------------------------


def test_both_subcommands_are_registered() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["train-boosting"])
    assert args.command == "train-boosting"
    args = parser.parse_args(["tune-boosting"])
    assert args.command == "tune-boosting"


def test_the_docstring_catalogues_both_commands() -> None:
    """A command absent from the catalogue is undiscoverable, so the docstring is tested."""
    assert cli.__doc__ is not None
    assert "sentinel train-boosting" in cli.__doc__
    assert "sentinel tune-boosting" in cli.__doc__


def test_models_is_repeatable_on_both_commands() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["train-boosting", "--models", "xgboost", "--models", "lightgbm"])
    assert args.models == ["xgboost", "lightgbm"]
    args = parser.parse_args(["tune-boosting", "--models", "xgboost"])
    assert args.models == ["xgboost"]


def test_models_defaults_to_none_meaning_every_registered_model() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["train-boosting"]).models is None


def test_fold_set_is_repeatable() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        ["tune-boosting", "--fold-set", "quarterly", "--fold-set", "covid_shift"]
    )
    assert args.fold_sets == ["quarterly", "covid_shift"]


def test_the_trial_budget_defaults_to_the_documented_production_value() -> None:
    """Lowering it silently would turn a production search into a development one."""
    parser = cli.build_parser()
    assert parser.parse_args(["tune-boosting"]).trials == cli.DEFAULT_TRIALS
    assert cli.DEFAULT_TRIALS == 100


def test_the_sampler_seed_defaults_to_the_declared_one() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["tune-boosting"]).seed == TUNING_SEED


def test_the_help_text_names_the_registered_models() -> None:
    """So ``--help`` cannot drift from the registry as models are added or renamed."""
    import contextlib
    import io

    for command, expected in (
        ("train-boosting", [s.name for s in BOOSTING_REGISTRY]),
        ("tune-boosting", list(TUNABLE_MODELS)),
    ):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.suppress(SystemExit):
            cli.main([command, "--help"])
        text = buffer.getvalue()
        for name in expected:
            assert name in text, f"{command} --help does not mention {name}"


def test_the_tuning_help_does_not_offer_the_untunable_ablation() -> None:
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.suppress(SystemExit):
        cli.main(["tune-boosting", "--help"])
    # argparse reflows help text, so compare on normalised whitespace rather than on
    # the exact line breaks it happens to choose for this terminal width.
    flattened = " ".join(buffer.getvalue().split())
    assert "borrows its donor's parameters and is not tunable on its own" in flattened


def test_log_level_works_on_either_side_of_the_subcommand() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["--log-level", "DEBUG", "train-boosting"]).log_level == "DEBUG"
    assert parser.parse_args(["tune-boosting", "--log-level", "ERROR"]).log_level == "ERROR"


# --- 2. argument validation --------------------------------------------------


def test_an_empty_models_list_is_rejected_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = cli.build_parser().parse_args(["train-boosting"])
    args.models = []
    with pytest.raises(SystemExit, match="at least one model name"):
        cli._run_train_boosting(args, cli.load_settings())


def test_an_empty_fold_set_list_is_rejected() -> None:
    args = cli.build_parser().parse_args(["tune-boosting"])
    args.fold_sets = []
    with pytest.raises(SystemExit, match="at least one fold set name"):
        cli._run_tune_boosting(args, cli.load_settings())


def test_a_non_positive_trial_budget_is_rejected() -> None:
    args = cli.build_parser().parse_args(["tune-boosting", "--trials", "0"])
    with pytest.raises(SystemExit, match="must be a positive integer"):
        cli._run_tune_boosting(args, cli.load_settings())


# --- 3. end to end -----------------------------------------------------------


def test_train_boosting_writes_three_tables_and_a_manifest(
    tmp_path: Path, features: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["train-boosting", "--features", str(features), "--output-dir", str(tmp_path)])
    assert code == 0
    assert len(list(tmp_path.glob("*.parquet"))) == 3
    assert len(list(tmp_path.glob("manifest_*.json"))) == 1


def test_train_boosting_prints_no_metric_component_5_owns(
    tmp_path: Path, features: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Component 7 predicts; Component 5 evaluates."""
    cli.main(["train-boosting", "--features", str(features), "--output-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "evaluate --predictions" in out
    for metric in ("roc_auc", "ROC-AUC", "pr_auc", "PR-AUC", "NDE", "precision@k"):
        assert metric not in out


def test_train_boosting_honours_a_single_model(tmp_path: Path, features: Path) -> None:
    code = cli.main(
        [
            "train-boosting",
            "--features",
            str(features),
            "--output-dir",
            str(tmp_path),
            "--models",
            "lightgbm",
        ]
    )
    assert code == 0
    predictions = sorted(tmp_path.glob("boosted_predictions_*.parquet"))[-1]
    names = set(pl.read_parquet(predictions)["model_name"].unique().to_list())
    assert names == {"lightgbm"}


def test_train_boosting_dry_run_writes_nothing(tmp_path: Path, features: Path) -> None:
    code = cli.main(
        ["train-boosting", "--features", str(features), "--output-dir", str(tmp_path), "--dry-run"]
    )
    assert code == 0
    assert not list(tmp_path.glob("*.parquet"))


def test_tune_boosting_writes_a_trials_table(tmp_path: Path, features: Path) -> None:
    code = cli.main(
        [
            "tune-boosting",
            "--features",
            str(features),
            "--output-dir",
            str(tmp_path),
            "--trials",
            "2",
            "--models",
            "lightgbm",
            "--fold-set",
            "quarterly",
        ]
    )
    assert code == 0
    trials = sorted(tmp_path.glob("tuning_trials_*.parquet"))
    assert len(trials) == 1
    frame = pl.read_parquet(trials[-1])
    assert frame.height == 2
    assert set(frame["study"].unique().to_list()) == {"lightgbm-quarterly"}


def test_tune_boosting_prints_the_block_to_freeze(
    tmp_path: Path, features: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The manual freeze step is the design: a file-loaded parameter set has no diff."""
    cli.main(
        [
            "tune-boosting",
            "--features",
            str(features),
            "--output-dir",
            str(tmp_path),
            "--trials",
            "2",
            "--models",
            "xgboost",
            "--fold-set",
            "quarterly",
        ]
    )
    out = capsys.readouterr().out
    assert "freeze these into boosting.definitions.TUNED_PARAMS" in out
    assert "n_estimators=" in out


def test_tune_boosting_shows_each_region_against_its_test_horizon(
    tmp_path: Path, features: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The protocol should be legible in the output, not only in a manifest field."""
    cli.main(
        [
            "tune-boosting",
            "--features",
            str(features),
            "--output-dir",
            str(tmp_path),
            "--trials",
            "2",
            "--models",
            "xgboost",
        ]
    )
    out = capsys.readouterr().out
    assert "first test" in out
    assert "NOT a result" in out


def test_tune_boosting_dry_run_writes_nothing(tmp_path: Path, features: Path) -> None:
    code = cli.main(
        [
            "tune-boosting",
            "--features",
            str(features),
            "--output-dir",
            str(tmp_path),
            "--trials",
            "2",
            "--models",
            "xgboost",
            "--fold-set",
            "quarterly",
            "--dry-run",
        ]
    )
    assert code == 0
    assert not list(tmp_path.glob("*.parquet"))


# --- 4. failures exit 1 without a traceback -----------------------------------


def test_an_unknown_model_exits_one_without_a_traceback(
    tmp_path: Path, features: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        [
            "train-boosting",
            "--features",
            str(features),
            "--output-dir",
            str(tmp_path),
            "--models",
            "catboost",
        ]
    )
    assert code == 1
    assert "Traceback" not in capsys.readouterr().err


def test_tuning_an_untunable_ablation_exits_one(tmp_path: Path, features: Path) -> None:
    """It borrows its donor's parameters; searching for it would vary two things."""
    code = cli.main(
        [
            "tune-boosting",
            "--features",
            str(features),
            "--output-dir",
            str(tmp_path),
            "--trials",
            "2",
            "--models",
            "xgboost_class_weighted",
        ]
    )
    assert code == 1
    assert "xgboost_class_weighted" not in TUNABLE_MODELS


def test_a_missing_feature_table_exits_one(tmp_path: Path) -> None:
    code = cli.main(
        [
            "train-boosting",
            "--features",
            str(tmp_path / "absent.parquet"),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert code == 1


def test_an_unknown_fold_set_exits_one(tmp_path: Path, features: Path) -> None:
    code = cli.main(
        [
            "tune-boosting",
            "--features",
            str(features),
            "--output-dir",
            str(tmp_path),
            "--trials",
            "2",
            "--fold-set",
            "weekly",
        ]
    )
    assert code == 1


def test_a_failed_check_exits_one(
    tmp_path: Path, features: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forced, because no real input in the suite makes an error-severity check fail."""
    from sentinel.boosting.models import ValidationCheck

    def failing(*args: object, **kwargs: object) -> list[ValidationCheck]:
        return [
            ValidationCheck(
                name="forced", passed=False, severity="error", detail="deliberate failure"
            )
        ]

    monkeypatch.setattr("sentinel.boosting.build.validate.validate_boosting", failing)
    code = cli.main(["train-boosting", "--features", str(features), "--output-dir", str(tmp_path)])
    assert code == 1


# --- 5. the two components stay separate ---------------------------------------


def test_the_boosted_run_does_not_touch_component_6s_artifact(
    tmp_path: Path, features: Path
) -> None:
    """C6's benchmark must remain visible, so the two write different slugs."""
    cli.main(["train-boosting", "--features", str(features), "--output-dir", str(tmp_path)])
    assert not list(tmp_path.glob("baseline_predictions_*.parquet"))
    assert list(tmp_path.glob("boosted_predictions_*.parquet"))


def test_train_then_evaluate_the_boosted_artifact(tmp_path: Path, features: Path) -> None:
    """The whole seam through the CLI, which is how a person actually runs it."""
    assert (
        cli.main(["train-boosting", "--features", str(features), "--output-dir", str(tmp_path)])
        == 0
    )
    predictions = sorted(tmp_path.glob("boosted_predictions_*.parquet"))[-1]
    evaluation = tmp_path / "evaluation"
    code = cli.main(
        [
            "evaluate",
            "--features",
            str(features),
            "--output-dir",
            str(evaluation),
            "--predictions",
            str(predictions),
            "--seeds",
            "2",
            "--sensitivity-replications",
            "3",
        ]
    )
    assert code == 0
    metrics = pl.read_parquet(sorted(evaluation.glob("evaluation_metrics_*.parquet"))[-1])
    scored = set(metrics["model_name"].unique().to_list())
    for spec in BOOSTING_REGISTRY:
        assert spec.name in scored
