"""CLI argument-parsing tests for ``organize-geography``'s v2 flags and the new
``review-plan`` command. Neither Component 20 nor Component 21 previously had a dedicated
CLI test file; execution-level behavior is covered by the build-level test suites."""

from __future__ import annotations

from sentinel import cli


class TestOrganizeGeographyFlags:
    def test_organization_mode_defaults_to_risk_first(self) -> None:
        args = cli.build_parser().parse_args(["organize-geography"])
        assert args.organization_mode == "risk_first"

    def test_organization_mode_accepts_geography_assisted(self) -> None:
        args = cli.build_parser().parse_args(
            ["organize-geography", "--organization-mode", "geography_assisted"]
        )
        assert args.organization_mode == "geography_assisted"

    def test_threshold_km_and_threshold_preset_default_to_none(self) -> None:
        args = cli.build_parser().parse_args(["organize-geography"])
        assert args.threshold_km is None
        assert args.threshold_preset is None

    def test_threshold_preset_accepts_named_values(self) -> None:
        for name in ("tight", "balanced", "broad"):
            args = cli.build_parser().parse_args(
                ["organize-geography", "--threshold-preset", name]
            )
            assert args.threshold_preset == name

    def test_threshold_km_and_threshold_preset_together_is_a_parse_error(self) -> None:
        import pytest

        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(
                [
                    "organize-geography",
                    "--threshold-km",
                    "2.0",
                    "--threshold-preset",
                    "broad",
                ]
            )

    def test_unknown_threshold_preset_is_a_parse_error(self) -> None:
        import pytest

        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["organize-geography", "--threshold-preset", "extreme"])

    def test_unknown_organization_mode_is_a_parse_error(self) -> None:
        import pytest

        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(
                ["organize-geography", "--organization-mode", "fastest_route"]
            )


class TestReviewPlanRegistration:
    def test_the_command_parses(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["review-plan", "--dry-run"])
        assert args.command == "review-plan"
        assert args.dry_run

    def test_optional_paths_default_to_none(self) -> None:
        args = cli.build_parser().parse_args(["review-plan"])
        assert args.plan is None
        assert args.decisions is None
        assert args.output_dir is None

    def test_decisions_path_is_accepted(self) -> None:
        args = cli.build_parser().parse_args(["review-plan", "--decisions", "decisions.json"])
        assert str(args.decisions) == "decisions.json"

    def test_report_flag(self) -> None:
        args = cli.build_parser().parse_args(["review-plan", "--report"])
        assert args.report is True
