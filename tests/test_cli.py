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
