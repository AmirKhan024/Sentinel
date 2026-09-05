"""Tests for CLI argument parsing and its wiring into configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.cli import _resolve_row_limit, build_parser, main
from sentinel.config import Settings
from tests.conftest import entity_scenario, make_entity_record


def parse(argv: list[str]) -> object:
    return build_parser().parse_args(argv)


# --- scope flags -----------------------------------------------------------


def test_ingest_requires_a_scope_flag() -> None:
    """A bare `sentinel ingest` must not silently start a full download."""
    with pytest.raises(SystemExit):
        parse(["ingest"])


def test_dev_and_full_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse(["ingest", "--dev", "--full"])


def test_limit_and_full_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse(["ingest", "--limit", "10", "--full"])


def test_dev_uses_configured_dev_row_limit(settings: Settings) -> None:
    args = parse(["ingest", "--dev"])
    assert _resolve_row_limit(args, settings) == settings.dev_row_limit  # type: ignore[arg-type]


def test_explicit_limit_overrides_dev_default(settings: Settings) -> None:
    args = parse(["ingest", "--limit", "123"])
    assert _resolve_row_limit(args, settings) == 123  # type: ignore[arg-type]


def test_full_means_no_row_limit(settings: Settings) -> None:
    args = parse(["ingest", "--full"])
    assert _resolve_row_limit(args, settings) is None  # type: ignore[arg-type]


def test_non_positive_limit_is_rejected(settings: Settings) -> None:
    args = parse(["ingest", "--limit", "0"])
    with pytest.raises(SystemExit, match="--limit must be a positive integer"):
        _resolve_row_limit(args, settings)  # type: ignore[arg-type]


# --- other options ---------------------------------------------------------


def test_page_size_and_output_dir_are_parsed() -> None:
    args = parse(["ingest", "--limit", "5", "--page-size", "50", "--output-dir", "out"])
    assert args.page_size == 50  # type: ignore[attr-defined]
    assert str(args.output_dir) == "out"  # type: ignore[attr-defined]


def test_page_size_override_applies_to_settings(settings: Settings) -> None:
    """model_copy must produce new settings without mutating the original."""
    updated = settings.model_copy(update={"page_size": 999})
    assert updated.page_size == 999
    assert settings.page_size != 999


def test_log_level_is_parsed() -> None:
    args = parse(["--log-level", "DEBUG", "ingest", "--dev"])
    assert args.log_level == "DEBUG"  # type: ignore[attr-defined]


def test_invalid_log_level_rejected() -> None:
    with pytest.raises(SystemExit):
        parse(["--log-level", "LOUD", "ingest", "--dev"])


# --- query subcommand ------------------------------------------------------


def test_query_list_flag() -> None:
    args = parse(["query", "--list"])
    assert args.list_queries is True  # type: ignore[attr-defined]


def test_query_name_and_parquet() -> None:
    args = parse(["query", "--name", "row_count", "--parquet", "a.parquet"])
    assert args.name == "row_count"  # type: ignore[attr-defined]
    assert str(args.parquet) == "a.parquet"  # type: ignore[attr-defined]


def test_unknown_subcommand_rejected() -> None:
    with pytest.raises(SystemExit):
        parse(["nonsense"])


def test_log_level_accepted_after_subcommand() -> None:
    args = parse(["ingest", "--dev", "--log-level", "DEBUG"])
    assert args.log_level == "DEBUG"  # type: ignore[attr-defined]


def test_log_level_before_subcommand_survives_omission_after() -> None:
    """The subparser copy uses SUPPRESS so it cannot clobber the global value."""
    args = parse(["--log-level", "WARNING", "ingest", "--dev"])
    assert args.log_level == "WARNING"  # type: ignore[attr-defined]


# --- resolve subcommand ----------------------------------------------------


def test_resolve_parses_with_no_flags() -> None:
    args = parse(["resolve"])
    assert args.command == "resolve"  # type: ignore[attr-defined]
    assert args.parquet is None  # type: ignore[attr-defined]
    assert args.dry_run is False  # type: ignore[attr-defined]


def test_resolve_accepts_a_parquet_override() -> None:
    args = parse(["resolve", "--parquet", "some/file.parquet"])
    assert args.parquet == Path("some/file.parquet")  # type: ignore[attr-defined]


def test_resolve_accepts_an_output_dir() -> None:
    args = parse(["resolve", "--output-dir", "out"])
    assert args.output_dir == Path("out")  # type: ignore[attr-defined]


def test_resolve_accepts_dry_run_and_report() -> None:
    args = parse(["resolve", "--dry-run", "--report"])
    assert args.dry_run is True  # type: ignore[attr-defined]
    assert args.report is True  # type: ignore[attr-defined]


def test_resolve_accepts_log_level_on_either_side() -> None:
    before = parse(["--log-level", "DEBUG", "resolve"])
    after = parse(["resolve", "--log-level", "DEBUG"])
    assert before.log_level == after.log_level == "DEBUG"  # type: ignore[attr-defined]


def test_resolve_writes_tables_and_exits_zero(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "food_inspections_20260816T000000Z.parquet"
    entity_scenario(
        [
            make_entity_record(1, inspection_id="1", dba_name="ONE", aka_name="ONE"),
            make_entity_record(2, inspection_id="2", dba_name="TWO", aka_name="TWO"),
        ]
    ).write_parquet(raw)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)

    out = tmp_path / "resolved"
    code = main(["resolve", "--parquet", str(raw), "--output-dir", str(out)])
    assert code == 0
    assert list(out.glob("establishment_assignments_*.parquet"))
    assert list(out.glob("manifest_establishment_assignments_*.json"))


def test_resolve_dry_run_writes_nothing(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "food_inspections_20260816T000000Z.parquet"
    entity_scenario([make_entity_record(1, inspection_id="1")]).write_parquet(raw)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)

    out = tmp_path / "resolved"
    assert main(["resolve", "--parquet", str(raw), "--output-dir", str(out), "--dry-run"]) == 0
    assert not out.exists()


def test_resolve_reports_a_missing_parquet_without_a_traceback(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)
    assert main(["resolve", "--parquet", str(tmp_path / "absent.parquet")]) == 1


def test_resolve_reports_when_no_raw_file_exists(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)
    assert main(["resolve"]) == 1


# --- build-target subcommand -----------------------------------------------


def test_build_target_parses_with_no_flags() -> None:
    args = parse(["build-target"])
    assert args.command == "build-target"  # type: ignore[attr-defined]
    assert args.parquet is None  # type: ignore[attr-defined]
    assert args.assignments is None  # type: ignore[attr-defined]
    assert args.dry_run is False  # type: ignore[attr-defined]


def test_build_target_accepts_both_input_overrides() -> None:
    args = parse(["build-target", "--parquet", "raw.parquet", "--assignments", "asg.parquet"])
    assert args.parquet == Path("raw.parquet")  # type: ignore[attr-defined]
    assert args.assignments == Path("asg.parquet")  # type: ignore[attr-defined]


def test_build_target_accepts_dry_run_and_report() -> None:
    args = parse(["build-target", "--dry-run", "--report"])
    assert args.dry_run is True  # type: ignore[attr-defined]
    assert args.report is True  # type: ignore[attr-defined]


def test_build_target_accepts_log_level_on_either_side() -> None:
    before = parse(["--log-level", "DEBUG", "build-target"])
    after = parse(["build-target", "--log-level", "DEBUG"])
    assert before.log_level == after.log_level == "DEBUG"  # type: ignore[attr-defined]


def _tiny_target_inputs(tmp_path: Path) -> tuple[Path, Path]:
    from tests.conftest import assignment_frame, make_inspection_record, target_scenario

    raw = tmp_path / "food_inspections_20260816T000000Z.parquet"
    asg = tmp_path / "establishment_assignments_20260816T000000Z.parquet"
    target_scenario(
        [
            make_inspection_record(
                1,
                inspection_id="1",
                violations="3. A - Comments: PRIORITY FOUNDATION 7-38-010.",
                results="Fail",
            ),
            make_inspection_record(2, inspection_id="2"),
        ]
    ).write_parquet(raw)
    assignment_frame([("1", "EST-A"), ("2", "EST-B")]).write_parquet(asg)
    return raw, asg


def test_build_target_writes_and_exits_zero(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, asg = _tiny_target_inputs(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)

    out = tmp_path / "targets"
    code = main(
        ["build-target", "--parquet", str(raw), "--assignments", str(asg), "--output-dir", str(out)]
    )
    assert code == 0
    assert list(out.glob("inspection_targets_*.parquet"))
    assert list(out.glob("manifest_inspection_targets_*.json"))


def test_build_target_dry_run_writes_nothing(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, asg = _tiny_target_inputs(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)

    out = tmp_path / "targets"
    assert (
        main(
            [
                "build-target",
                "--parquet",
                str(raw),
                "--assignments",
                str(asg),
                "--output-dir",
                str(out),
                "--dry-run",
            ]
        )
        == 0
    )
    assert not out.exists()


def test_build_target_reports_a_missing_input_without_a_traceback(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, asg = _tiny_target_inputs(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)
    assert (
        main(
            [
                "build-target",
                "--parquet",
                str(tmp_path / "absent.parquet"),
                "--assignments",
                str(asg),
            ]
        )
        == 1
    )


def test_build_target_reports_when_no_raw_file_exists(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)
    assert main(["build-target"]) == 1


# --- build-features subcommand ---------------------------------------------


def test_build_features_parses_with_no_flags() -> None:
    args = parse(["build-features"])
    assert args.command == "build-features"  # type: ignore[attr-defined]
    assert args.parquet is None  # type: ignore[attr-defined]
    assert args.assignments is None  # type: ignore[attr-defined]
    assert args.targets is None  # type: ignore[attr-defined]


def test_build_features_accepts_all_three_input_overrides() -> None:
    args = parse(
        [
            "build-features",
            "--parquet",
            "raw.parquet",
            "--assignments",
            "asg.parquet",
            "--targets",
            "tgt.parquet",
        ]
    )
    assert args.parquet == Path("raw.parquet")  # type: ignore[attr-defined]
    assert args.assignments == Path("asg.parquet")  # type: ignore[attr-defined]
    assert args.targets == Path("tgt.parquet")  # type: ignore[attr-defined]


def test_build_features_accepts_dry_run_and_report() -> None:
    args = parse(["build-features", "--dry-run", "--report"])
    assert args.dry_run is True  # type: ignore[attr-defined]
    assert args.report is True  # type: ignore[attr-defined]


def _tiny_feature_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    import polars as pl

    from tests.conftest import assignment_frame, make_inspection_record, target_scenario

    raw = tmp_path / "food_inspections_20260816T000000Z.parquet"
    asg = tmp_path / "establishment_assignments_20260816T000000Z.parquet"
    tgt = tmp_path / "inspection_targets_20260816T000000Z.parquet"
    rows = [
        make_inspection_record(1, inspection_id="1", inspection_date="2020-01-01T00:00:00.000"),
        make_inspection_record(2, inspection_id="2", inspection_date="2022-01-01T00:00:00.000"),
    ]
    target_scenario(rows).write_parquet(raw)
    assignment_frame([("1", "EST-A"), ("2", "EST-A")]).write_parquet(asg)
    pl.DataFrame(
        {
            "establishment_id": ["EST-A"],
            "inspection_date": ["2022-01-01T00:00:00.000"],
            "target_inspection_id": ["2"],
            "target": [1],
            "target_status": ["eligible"],
            "code_era_phase": ["stable"],
        },
        schema={
            "establishment_id": pl.Utf8,
            "inspection_date": pl.Utf8,
            "target_inspection_id": pl.Utf8,
            "target": pl.Int8,
            "target_status": pl.Utf8,
            "code_era_phase": pl.Utf8,
        },
    ).write_parquet(tgt)
    return raw, asg, tgt


def test_build_features_writes_and_exits_zero(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, asg, tgt = _tiny_feature_inputs(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)

    out = tmp_path / "features"
    code = main(
        [
            "build-features",
            "--parquet",
            str(raw),
            "--assignments",
            str(asg),
            "--targets",
            str(tgt),
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    assert list(out.glob("as_of_features_*.parquet"))
    assert list(out.glob("manifest_as_of_features_*.json"))


def test_build_features_dry_run_writes_nothing(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, asg, tgt = _tiny_feature_inputs(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)

    out = tmp_path / "features"
    code = main(
        [
            "build-features",
            "--parquet",
            str(raw),
            "--assignments",
            str(asg),
            "--targets",
            str(tgt),
            "--output-dir",
            str(out),
            "--dry-run",
        ]
    )
    assert code == 0
    assert not out.exists()


def test_build_features_reports_a_missing_input_without_a_traceback(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, asg, tgt = _tiny_feature_inputs(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)
    assert (
        main(
            [
                "build-features",
                "--parquet",
                str(tmp_path / "absent.parquet"),
                "--assignments",
                str(asg),
                "--targets",
                str(tgt),
            ]
        )
        == 1
    )


def test_build_features_reports_when_no_upstream_files_exist(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)
    assert main(["build-features"]) == 1


# --- evaluate subcommand ----------------------------------------------------


def test_evaluate_parses_with_no_flags() -> None:
    args = parse(["evaluate"])
    assert args.command == "evaluate"  # type: ignore[attr-defined]
    assert args.features is None  # type: ignore[attr-defined]
    assert args.folds_only is False  # type: ignore[attr-defined]


def test_evaluate_accepts_its_input_and_output_overrides() -> None:
    args = parse(["evaluate", "--features", "f.parquet", "--output-dir", "out"])
    assert args.features == Path("f.parquet")  # type: ignore[attr-defined]
    assert args.output_dir == Path("out")  # type: ignore[attr-defined]


def test_evaluate_accepts_dry_run_report_and_folds_only() -> None:
    args = parse(["evaluate", "--dry-run", "--report", "--folds-only"])
    assert args.dry_run is True  # type: ignore[attr-defined]
    assert args.report is True  # type: ignore[attr-defined]
    assert args.folds_only is True  # type: ignore[attr-defined]


def test_evaluate_seed_and_replication_counts_default_to_the_declared_values() -> None:
    from sentinel.evaluation.sensitivity import DEFAULT_REPLICATIONS
    from sentinel.evaluation.simulate import DEFAULT_RANDOM_REPLICATIONS

    args = parse(["evaluate"])
    assert args.seeds == DEFAULT_RANDOM_REPLICATIONS  # type: ignore[attr-defined]
    assert args.sensitivity_replications == DEFAULT_REPLICATIONS  # type: ignore[attr-defined]


def test_evaluate_log_level_works_on_either_side_of_the_subcommand() -> None:
    before = parse(["--log-level", "DEBUG", "evaluate"])
    after = parse(["evaluate", "--log-level", "DEBUG"])
    assert before.log_level == "DEBUG"  # type: ignore[attr-defined]
    assert after.log_level == "DEBUG"  # type: ignore[attr-defined]


def _tiny_evaluation_input(tmp_path: Path) -> Path:
    from tests.conftest import spanning_features

    path = tmp_path / "as_of_features_20260816T150313Z.parquet"
    spanning_features(days=1800, per_day=2).write_parquet(path)
    return path


def test_evaluate_writes_and_exits_zero(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = _tiny_evaluation_input(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)

    out = tmp_path / "evaluation"
    code = main(
        [
            "evaluate",
            "--features",
            str(features),
            "--output-dir",
            str(out),
            "--seeds",
            "2",
            "--sensitivity-replications",
            "3",
        ]
    )
    assert code == 0
    assert list(out.glob("evaluation_folds_*.parquet"))
    assert list(out.glob("evaluation_metrics_*.parquet"))
    assert list(out.glob("discovery_curves_*.parquet"))
    assert list(out.glob("simulation_summary_*.parquet"))
    assert list(out.glob("manifest_evaluation_folds_*.json"))


def test_evaluate_dry_run_writes_nothing(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = _tiny_evaluation_input(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)

    out = tmp_path / "evaluation"
    code = main(
        [
            "evaluate",
            "--features",
            str(features),
            "--output-dir",
            str(out),
            "--dry-run",
            "--seeds",
            "2",
            "--sensitivity-replications",
            "3",
        ]
    )
    assert code == 0
    assert not out.exists()


def test_evaluate_folds_only_is_fast_and_still_writes_the_split(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = _tiny_evaluation_input(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)

    out = tmp_path / "evaluation"
    assert (
        main(["evaluate", "--features", str(features), "--output-dir", str(out), "--folds-only"])
        == 0
    )
    import polars as pl

    (folds_file,) = list(out.glob("evaluation_folds_*.parquet"))
    (metrics_file,) = list(out.glob("evaluation_metrics_*.parquet"))
    assert pl.read_parquet(folds_file).height > 0
    assert pl.read_parquet(metrics_file).height == 0


def test_evaluate_reports_a_missing_input_without_a_traceback(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)
    assert main(["evaluate", "--features", str(tmp_path / "absent.parquet")]) == 1


def test_evaluate_reports_when_no_feature_table_exists(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)
    assert main(["evaluate"]) == 1


def test_evaluate_refuses_a_non_positive_seed_count(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = _tiny_evaluation_input(tmp_path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)
    with pytest.raises(SystemExit):
        main(["evaluate", "--features", str(features), "--seeds", "0"])


def test_evaluate_refuses_a_snapshot_too_short_to_fold(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Folds are never fabricated to fill a report."""
    from tests.conftest import spanning_features

    path = tmp_path / "as_of_features_short.parquet"
    spanning_features(days=200).write_parquet(path)
    monkeypatch.setattr("sentinel.cli.load_settings", lambda: settings)
    assert main(["evaluate", "--features", str(path)]) == 1


# --- serve -------------------------------------------------------------------


def test_serve_parses_with_no_flags() -> None:
    args = parse(["serve"])
    assert args.command == "serve"  # type: ignore[attr-defined]
    assert args.host is None  # type: ignore[attr-defined]
    assert args.port is None  # type: ignore[attr-defined]
    assert args.reload is False  # type: ignore[attr-defined]


def test_serve_accepts_host_port_and_reload() -> None:
    args = parse(["serve", "--host", "0.0.0.0", "--port", "9000", "--reload"])
    assert args.host == "0.0.0.0"  # type: ignore[attr-defined]
    assert args.port == 9000  # type: ignore[attr-defined]
    assert args.reload is True  # type: ignore[attr-defined]


def test_run_serve_binds_using_settings_defaults_when_unset(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_run_serve` never starts a real server here -- `uvicorn.run` is replaced with a spy."""
    from sentinel.cli import _run_serve

    calls: list[dict[str, object]] = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.append({"app": app, **kwargs}))
    args = parse(["serve"])
    assert _run_serve(args, settings) == 0
    assert calls[0]["host"] == settings.api_host
    assert calls[0]["port"] == settings.api_port
    assert calls[0]["reload"] is False
