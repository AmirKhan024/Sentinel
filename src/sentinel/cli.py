"""Command line interface for Sentinel.

Built on ``argparse`` from the standard library. This component needs a handful
of flags across two subcommands; argparse covers that completely. Adding Typer
(and with it click and rich) would mean three dependencies bought for help-text
ergonomics we do not yet need, which contradicts the project rule of
introducing a technology only when a component requires it. If the CLI grows
several more subcommands, revisiting this is reasonable.

Commands
--------
    sentinel ingest --dev                      small development pull
    sentinel ingest --limit 5000               explicit row cap
    sentinel ingest --full                     entire dataset
    sentinel query --list                      show available named queries
    sentinel query --name row_count            query the latest raw Parquet
    sentinel resolve                           resolve establishment identities
    sentinel resolve --dry-run --report        resolve without writing anything
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sentinel import __version__
from sentinel.config import Settings, load_settings
from sentinel.entity import validate
from sentinel.entity.resolve import EntityResolutionError, resolve_establishments, summarize
from sentinel.ingest.food_inspections import ingest_food_inspections
from sentinel.ingest.socrata import SocrataError
from sentinel.logging_setup import configure_logging
from sentinel.query import duckdb_queries

logger = logging.getLogger("sentinel.cli")

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description=(
            "Sentinel - risk-prioritized food inspection scheduling "
            "(ingestion and entity resolution)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"sentinel {__version__}")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=LOG_LEVELS,
        help="Override the configured log level.",
    )

    # Shared options are also attached to each subcommand so `--log-level`
    # works on either side of the subcommand name. The subparser copy defaults
    # to SUPPRESS, so when it is omitted there it does not clobber a value
    # given before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--log-level",
        default=argparse.SUPPRESS,
        choices=LOG_LEVELS,
        help="Override the configured log level.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- ingest -----------------------------------------------------------
    ingest = subparsers.add_parser(
        "ingest",
        parents=[common],
        help="Download Chicago food inspections into the raw data layer.",
    )
    # Exactly one scope flag is required. Making the scope explicit prevents an
    # accidental full 300k-row pull from a bare `sentinel ingest`.
    scope = ingest.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--dev",
        action="store_true",
        help="Development pull using the configured dev row limit.",
    )
    scope.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Development pull capped at N rows.",
    )
    scope.add_argument(
        "--full",
        action="store_true",
        help="Full pull of the entire dataset (no row limit).",
    )
    ingest.add_argument(
        "--page-size",
        type=int,
        metavar="N",
        help="Rows requested per API page ($limit). Overrides configuration.",
    )
    ingest.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the Parquet file and manifest.",
    )

    # --- query ------------------------------------------------------------
    query = subparsers.add_parser(
        "query",
        parents=[common],
        help="Run a descriptive DuckDB query against a raw Parquet file.",
    )
    query.add_argument(
        "--name",
        help="Named query to run. Use --list to see the options.",
    )
    query.add_argument(
        "--list",
        action="store_true",
        dest="list_queries",
        help="List the available named queries and exit.",
    )
    query.add_argument(
        "--parquet",
        type=Path,
        metavar="PATH",
        help="Parquet file to query. Defaults to the most recent raw file.",
    )

    # --- resolve ----------------------------------------------------------
    resolve = subparsers.add_parser(
        "resolve",
        parents=[common],
        help="Resolve inspections into stable establishment identities.",
    )
    resolve.add_argument(
        "--parquet",
        type=Path,
        metavar="PATH",
        help="Raw Parquet to resolve. Defaults to the most recent raw file.",
    )
    resolve.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the resolved tables and manifest.",
    )
    resolve.add_argument(
        "--dry-run", action="store_true", help="Resolve and validate, but write nothing."
    )
    resolve.add_argument(
        "--report", action="store_true", help="Print the full validation report, not only failures."
    )

    return parser


def _resolve_row_limit(args: argparse.Namespace, settings: Settings) -> int | None:
    """Translate the mutually exclusive scope flags into a row limit."""
    if args.full:
        return None
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be a positive integer")
        return int(args.limit)
    return settings.dev_row_limit


def _run_ingest(args: argparse.Namespace, settings: Settings) -> int:
    if args.page_size is not None:
        if args.page_size <= 0:
            raise SystemExit("--page-size must be a positive integer")
        settings = settings.model_copy(update={"page_size": args.page_size})

    row_limit = _resolve_row_limit(args, settings)

    try:
        result = ingest_food_inspections(
            settings,
            row_limit=row_limit,
            output_dir=args.output_dir,
        )
    except SocrataError as exc:
        # Fail loudly, but without a traceback for an expected class of failure.
        logger.error("Ingestion failed: %s", exc)
        return 1

    print(f"rows:     {result.row_count}")
    print(f"parquet:  {result.parquet_path}")
    print(f"manifest: {result.manifest_path}")
    print(f"sha256:   {result.manifest.sha256}")
    return 0


def _render_table(columns: list[str], rows: list[tuple[object, ...]]) -> str:
    """Render query output as a plain fixed-width table."""
    header = [str(c) for c in columns]
    body = [["" if v is None else str(v) for v in row] for row in rows]
    widths = [
        max(len(header[i]), *(len(r[i]) for r in body)) if body else len(header[i])
        for i in range(len(header))
    ]
    lines = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(header)),
        "  ".join("-" * widths[i] for i in range(len(header))),
    ]
    lines.extend("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in body)
    return "\n".join(lines)


def _run_query(args: argparse.Namespace, settings: Settings) -> int:
    if args.list_queries:
        print("Available named queries:")
        for name in sorted(duckdb_queries.NAMED_QUERIES):
            print(f"  {name}")
        return 0

    if not args.name:
        raise SystemExit("query requires --name (or --list to see the options)")

    parquet_path = args.parquet
    if parquet_path is None:
        try:
            parquet_path = duckdb_queries.latest_parquet(settings.food_inspections_raw_dir)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 1
        logger.info("Using most recent raw file: %s", parquet_path)

    try:
        result = duckdb_queries.run_named_query(parquet_path, args.name)
    except (KeyError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1

    print(_render_table(result.columns, result.rows))
    return 0


def _run_resolve(args: argparse.Namespace, settings: Settings) -> int:
    parquet_path = args.parquet
    if parquet_path is None:
        try:
            parquet_path = duckdb_queries.latest_parquet(settings.food_inspections_raw_dir)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 1
        logger.info("Using most recent raw file: %s", parquet_path)

    try:
        result = resolve_establishments(
            settings,
            parquet_path=parquet_path,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, EntityResolutionError, ValueError) as exc:
        logger.error("Entity resolution failed: %s", exc)
        return 1

    print(summarize(result))

    failed = validate.has_failures(result.checks)
    if args.report or failed:
        print(validate.format_report(result.checks))
    # A failed structural check means the identities are wrong, so the command
    # fails loudly rather than leaving quietly broken output for Component 3.
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_logging(getattr(args, "log_level", None) or settings.log_level)

    if args.command == "ingest":
        return _run_ingest(args, settings)
    if args.command == "query":
        return _run_query(args, settings)
    if args.command == "resolve":
        return _run_resolve(args, settings)

    # argparse enforces `required=True` on the subparser, so this is defensive
    # only. parser.error() exits with status 2.
    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
