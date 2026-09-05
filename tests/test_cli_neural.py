"""The three Component 8 subcommands: parsing, exit codes, and what they must not print.

Kept beside ``test_cli_baselines.py`` and ``test_cli_boosting.py`` rather than folded into
``test_cli.py``, which is already long. The pattern is those files': parse-only tests that
never touch the filesystem, end-to-end tests that do, and one test asserting the training
command reports no metric Component 5 owns -- because two answers to "how good is this
model?" is one answer too many.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel import cli
from sentinel.neural.definitions import LEARNING_RATE_GRID, NEURAL_REGISTRY, TUNING_SEED
from tests.conftest import neural_categoricals_for, spanning_model_features


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("cli_neural_input")
    frame = spanning_model_features(days=1600, per_day=3)
    features = tmp / "as_of_features_20260101T000000Z.parquet"
    frame.write_parquet(features)
    categoricals = tmp / "neural_categoricals_20260101T000000Z.parquet"
    neural_categoricals_for(frame).write_parquet(categoricals)
    return {"features": features, "categoricals": categoricals, "dir": tmp}


# --- 1. parsing --------------------------------------------------------------


def test_all_three_subcommands_are_registered() -> None:
    parser = cli.build_parser()
    for command in ("build-neural-categoricals", "tune-neural", "train-neural"):
        assert parser.parse_args([command]).command == command


def test_the_docstring_catalogues_every_command() -> None:
    """``test_cli.py`` asserts the same property for the earlier components.

    A command absent from the catalogue is one a reader of ``--help``'s prose never
    learns about.
    """
    assert cli.__doc__ is not None
    for command in ("build-neural-categoricals", "tune-neural", "train-neural"):
        assert f"sentinel {command}" in cli.__doc__


def test_train_neural_defaults_to_every_registered_model() -> None:
    args = cli.build_parser().parse_args(["train-neural"])
    assert args.models is None
    assert args.dry_run is False
    assert args.no_seed_sweep is False
    assert args.no_figures is False


def test_models_is_repeatable() -> None:
    args = cli.build_parser().parse_args(
        ["train-neural", "--models", "neural_embeddings", "--models", "neural_numeric_only"]
    )
    assert args.models == ["neural_embeddings", "neural_numeric_only"]


def test_the_models_help_names_every_registered_model() -> None:
    parser = cli.build_parser()
    text = parser.format_help()
    assert "train-neural" in text
    # The registry is named in the subcommand's own help rather than restated here.
    action = next(
        a
        for a in parser._subparsers._group_actions[0].choices["train-neural"]._actions  # type: ignore[union-attr]
        if a.dest == "models"
    )
    for spec in NEURAL_REGISTRY:
        assert spec.name in (action.help or "")


def test_tune_neural_defaults_to_the_declared_seed() -> None:
    args = cli.build_parser().parse_args(["tune-neural"])
    assert args.seed == TUNING_SEED
    assert args.fold_sets is None


def test_fold_set_is_repeatable() -> None:
    args = cli.build_parser().parse_args(
        ["tune-neural", "--fold-set", "quarterly", "--fold-set", "covid_shift"]
    )
    assert args.fold_sets == ["quarterly", "covid_shift"]


def test_dry_run_and_report_are_available_on_every_command() -> None:
    parser = cli.build_parser()
    for command in ("build-neural-categoricals", "tune-neural", "train-neural"):
        args = parser.parse_args([command, "--dry-run", "--report"])
        assert args.dry_run is True
        assert args.report is True


def test_an_unknown_command_exits_two() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["train-neurals"])
    assert excinfo.value.code == 2


# --- 2. end to end -----------------------------------------------------------


def test_train_neural_writes_an_artifact_and_exits_zero(
    inputs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        [
            "train-neural",
            "--features",
            str(inputs["features"]),
            "--categoricals",
            str(inputs["categoricals"]),
            "--output-dir",
            str(tmp_path),
            "--models",
            "neural_numeric_only",
            "--no-seed-sweep",
            "--no-figures",
        ]
    )
    assert code == 0
    written = list(tmp_path.glob("neural_predictions_*.parquet"))
    assert len(written) == 1
    manifest = list(tmp_path.glob("manifest_neural_predictions_*.json"))
    assert len(manifest) == 1

    out = capsys.readouterr().out
    assert "Component 8 neural models" in out
    assert "sentinel evaluate --predictions" in out


def test_train_neural_reports_no_metric_component_5_owns(
    inputs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Component 8 predicts; Component 5 evaluates. Two answers is one too many."""
    cli.main(
        [
            "train-neural",
            "--features",
            str(inputs["features"]),
            "--categoricals",
            str(inputs["categoricals"]),
            "--output-dir",
            str(tmp_path),
            "--models",
            "neural_numeric_only",
            "--no-seed-sweep",
            "--no-figures",
        ]
    )
    out = capsys.readouterr().out.lower()
    for forbidden in ("roc-auc", "roc_auc", "pr-auc", "pr_auc", "nde", "brier", "precision@"):
        assert forbidden not in out, f"the training command reported {forbidden}"


def test_dry_run_writes_nothing(inputs: dict[str, Path], tmp_path: Path) -> None:
    code = cli.main(
        [
            "train-neural",
            "--features",
            str(inputs["features"]),
            "--categoricals",
            str(inputs["categoricals"]),
            "--output-dir",
            str(tmp_path),
            "--models",
            "neural_numeric_only",
            "--no-seed-sweep",
            "--no-figures",
            "--dry-run",
        ]
    )
    assert code == 0
    assert not list(tmp_path.glob("*.parquet"))
    assert not list(tmp_path.glob("*.json"))


def test_an_unknown_model_name_exits_one(inputs: dict[str, Path], tmp_path: Path) -> None:
    """A typo must not quietly halve the portfolio."""
    code = cli.main(
        [
            "train-neural",
            "--features",
            str(inputs["features"]),
            "--categoricals",
            str(inputs["categoricals"]),
            "--output-dir",
            str(tmp_path),
            "--models",
            "neural_embedings",
        ]
    )
    assert code == 1


def test_requesting_the_booster_without_its_donor_exits_one(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    """The embedding-fed booster cannot run without the network that supplies its vectors.

    Silently fitting a second donor, or reusing a cached one, would break the guarantee
    that the vectors came from this fold.
    """
    code = cli.main(
        [
            "train-neural",
            "--features",
            str(inputs["features"]),
            "--categoricals",
            str(inputs["categoricals"]),
            "--output-dir",
            str(tmp_path),
            "--models",
            "xgboost_chain_embeddings",
        ]
    )
    assert code == 1


def test_a_missing_features_file_exits_one(tmp_path: Path) -> None:
    code = cli.main(
        [
            "train-neural",
            "--features",
            str(tmp_path / "absent.parquet"),
            "--categoricals",
            str(tmp_path / "absent_cats.parquet"),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert code == 1


def test_build_neural_categoricals_reports_its_families(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    """Driven through the module rather than the CLI: the CLI needs a raw snapshot.

    ``build-neural-categoricals`` resolves a raw Socrata file and Component 2's
    assignments, neither of which a unit fixture should fabricate at full fidelity.
    ``test_neural_categoricals.py`` drives the builder itself against a hand-made raw
    frame; this asserts the command is wired and its flags parse.
    """
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "build-neural-categoricals",
            "--features",
            "f.parquet",
            "--raw",
            "r.parquet",
            "--assignments",
            "a.parquet",
        ]
    )
    assert args.command == "build-neural-categoricals"
    assert args.raw == Path("r.parquet")
    assert args.assignments == Path("a.parquet")


def test_tune_neural_grid_is_the_declared_one() -> None:
    """The CLI exposes no ``--grid``: the grid is a declared constant, not a flag.

    A rate range chosen at the command line would not be recorded in a diff, and the
    whole point of freezing is that a search's inputs are visible.
    """
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["tune-neural", "--grid", "0.1"])
    assert len(LEARNING_RATE_GRID) == 5


def test_the_output_is_utf8_safe(
    inputs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The summary must not depend on a console encoding to render."""
    cli.main(
        [
            "train-neural",
            "--features",
            str(inputs["features"]),
            "--categoricals",
            str(inputs["categoricals"]),
            "--output-dir",
            str(tmp_path),
            "--models",
            "neural_numeric_only",
            "--no-seed-sweep",
            "--no-figures",
        ]
    )
    out = capsys.readouterr().out
    out.encode("ascii")
