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
    sentinel build-target                      construct the prediction target
    sentinel build-target --dry-run --report   build without writing anything
    sentinel build-features                    construct the as-of feature table
    sentinel build-features --dry-run --report build without writing anything
    sentinel train-baselines                   fit the baseline risk models
    sentinel train-baselines --models NAME      fit only the named model(s)
    sentinel train-baselines --dry-run --report train without writing anything
    sentinel tune-boosting                     search boosted hyperparameters
    sentinel tune-boosting --trials 100        explicit trial budget per study
    sentinel tune-boosting --fold-set NAME     search one fold set only
    sentinel train-boosting                    fit the boosted risk models
    sentinel train-boosting --models NAME       fit only the named model(s)
    sentinel train-boosting --dry-run --report  train without writing anything
    sentinel build-neural-categoricals         build Component 8's as-of categoricals
    sentinel build-neural-categoricals --report  build and print the full check report
    sentinel tune-neural                       search the neural learning rate
    sentinel tune-neural --fold-set NAME       search one fold set only
    sentinel train-neural                      fit the neural models with embeddings
    sentinel train-neural --models NAME         fit only the named model(s)
    sentinel train-neural --no-seed-sweep       skip the multi-seed reproducibility run
    sentinel train-neural --dry-run --report    train without writing anything
    sentinel calibrate                         fit and freeze probability calibrators
    sentinel calibrate --models NAME            calibrate only the named base model(s)
    sentinel calibrate --method platt           force one method, skipping selection
    sentinel calibrate --no-figures             skip the reliability and drift figures
    sentinel calibrate --dry-run --report       calibrate without writing anything
    sentinel evaluate                          run the temporal evaluation harness
    sentinel evaluate --folds-only             emit the fold table and stop
    sentinel evaluate --predictions PATH       also score a prediction artifact
    sentinel evaluate --dry-run --report       evaluate without writing anything

Components 6 and 7 train and Component 5 evaluates, and they are separate commands on
purpose. ``train-baselines`` and ``train-boosting`` each write a prediction artifact and
report no metric; ``evaluate --predictions`` reads one and reports the numbers. Anyone
tempted to collapse them should read ADR 0013 first.

``tune-boosting`` is separate again, and separate from training. It searches, writes a
trials table and prints the parameter block to freeze into
``boosting.definitions.TUNED_PARAMS``; it edits no source file. A search that silently
rewrote the parameters it had just chosen would make "these are frozen" untrue. See
ADR 0017.

Component 8 follows the same shape, with one command more. ``build-neural-categoricals``
is separate from ``train-neural`` because it is the one step that reaches outside
Component 4's contract: it carries chain, facility type, community area and zip forward
as-of from the raw snapshot into an explicitly experimental artifact. Making it a step a
human runs and can inspect -- rather than a silent join inside training -- is the point.
Component 4's table and its ``feature_definition_version`` are untouched. See ADR 0022.
``tune-neural`` searches the learning rate under ADR 0017's protocol and prints a block
to freeze, exactly as ``tune-boosting`` does.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sentinel import __version__
from sentinel.boosting import validate as boosting_validate
from sentinel.boosting.build import (
    BoostingBuildError,
    summarize_tuning,
    train_boosting,
    tune_boosting,
)
from sentinel.boosting.build import summarize as summarize_boosting
from sentinel.boosting.definitions import BOOSTING_REGISTRY, TUNABLE_MODELS, TUNING_SEED
from sentinel.boosting.tuning import TuningError
from sentinel.calibration import validate as calibration_validate
from sentinel.calibration.basescores import BaseScoreError
from sentinel.calibration.build import CalibrationBuildError, run_calibration
from sentinel.calibration.build import summarize as summarize_calibration
from sentinel.calibration.definitions import (
    BOOTSTRAP_REPLICATIONS,
    CANDIDATE_REGISTRY,
    Method,
)
from sentinel.calibration.preprocess import CalibrationPreprocessError
from sentinel.calibration.train import CalibrationTrainError
from sentinel.config import Settings, load_settings
from sentinel.entity import validate
from sentinel.entity.resolve import EntityResolutionError, resolve_establishments, summarize
from sentinel.evaluation import validate as evaluation_validate
from sentinel.evaluation.build import EvaluationError, run_evaluation
from sentinel.evaluation.build import summarize as summarize_evaluation
from sentinel.evaluation.sensitivity import DEFAULT_REPLICATIONS
from sentinel.evaluation.simulate import DEFAULT_RANDOM_REPLICATIONS
from sentinel.features import validate as feature_validate
from sentinel.features.build import (
    FeatureConstructionError,
    build_features,
)
from sentinel.features.build import (
    summarize as summarize_features,
)
from sentinel.ingest.food_inspections import ingest_food_inspections
from sentinel.ingest.socrata import SocrataError
from sentinel.logging_setup import configure_logging
from sentinel.modeling import validate as modeling_validate
from sentinel.modeling.build import BaselineTrainingError, train_baselines
from sentinel.modeling.build import summarize as summarize_baselines
from sentinel.modeling.definitions import MODEL_REGISTRY
from sentinel.neural import validate as neural_validate
from sentinel.neural.build import (
    NeuralBuildError,
    build_neural_categoricals,
    train_neural,
    tune_neural,
)
from sentinel.neural.build import summarize as summarize_neural
from sentinel.neural.build import summarize_categoricals as summarize_neural_categoricals
from sentinel.neural.build import summarize_tuning as summarize_neural_tuning
from sentinel.neural.categoricals import CategoricalBuildError
from sentinel.neural.definitions import NEURAL_REGISTRY
from sentinel.neural.definitions import TUNING_SEED as NEURAL_TUNING_SEED
from sentinel.neural.train import NeuralTrainError
from sentinel.neural.tuning import NeuralTuningError
from sentinel.query import duckdb_queries
from sentinel.target import validate as target_validate
from sentinel.target.build import TargetConstructionError, build_targets
from sentinel.target.build import summarize as summarize_target

logger = logging.getLogger("sentinel.cli")

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

#: Optuna trials per study. The documented production search; `--trials` lowers it for
#: development, and the findings document distinguishes the two runs explicitly.
DEFAULT_TRIALS = 100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description=(
            "Sentinel - risk-prioritized food inspection scheduling "
            "(ingestion, entity resolution, target and feature construction, "
            "temporal evaluation)."
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

    # --- build-target -----------------------------------------------------
    build_target = subparsers.add_parser(
        "build-target",
        parents=[common],
        help="Construct the prediction target from resolved inspections.",
    )
    build_target.add_argument(
        "--parquet",
        type=Path,
        metavar="PATH",
        help="Raw Parquet. Defaults to the most recent raw file.",
    )
    build_target.add_argument(
        "--assignments",
        type=Path,
        metavar="PATH",
        help="Component 2 assignments. Defaults to the most recent.",
    )
    build_target.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the target table and manifest.",
    )
    build_target.add_argument(
        "--dry-run", action="store_true", help="Construct and validate, but write nothing."
    )
    build_target.add_argument(
        "--report", action="store_true", help="Print the full validation report, not only failures."
    )

    # --- build-features ---------------------------------------------------
    build_feat = subparsers.add_parser(
        "build-features",
        parents=[common],
        help="Construct as-of historical features for each prediction opportunity.",
    )
    build_feat.add_argument(
        "--parquet",
        type=Path,
        metavar="PATH",
        help="Raw Parquet. Defaults to the most recent raw file.",
    )
    build_feat.add_argument(
        "--assignments",
        type=Path,
        metavar="PATH",
        help="Component 2 assignments. Defaults to the most recent.",
    )
    build_feat.add_argument(
        "--targets",
        type=Path,
        metavar="PATH",
        help="Component 3 targets. Defaults to the most recent.",
    )
    build_feat.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the feature table and manifest.",
    )
    build_feat.add_argument(
        "--dry-run", action="store_true", help="Construct and validate, but write nothing."
    )
    build_feat.add_argument(
        "--report", action="store_true", help="Print the full validation report, not only failures."
    )

    # --- train-baselines --------------------------------------------------
    train = subparsers.add_parser(
        "train-baselines",
        parents=[common],
        help="Fit the baseline risk models and write fold-specific predictions.",
    )
    train.add_argument(
        "--features",
        type=Path,
        metavar="PATH",
        help="Component 4 feature table. Defaults to the most recent.",
    )
    train.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the prediction tables and manifest.",
    )
    train.add_argument(
        "--models",
        action="append",
        metavar="NAME",
        help=(
            "Model to fit; repeatable. Defaults to every registered model "
            f"({', '.join(spec.name for spec in MODEL_REGISTRY)})."
        ),
    )
    train.add_argument(
        "--dry-run", action="store_true", help="Train and validate, but write nothing."
    )
    train.add_argument(
        "--report", action="store_true", help="Print the full validation report, not only failures."
    )

    # --- tune-boosting ----------------------------------------------------
    tune = subparsers.add_parser(
        "tune-boosting",
        parents=[common],
        help="Search boosted hyperparameters on temporally valid inner folds.",
    )
    tune.add_argument(
        "--features",
        type=Path,
        metavar="PATH",
        help="Component 4 feature table. Defaults to the most recent.",
    )
    tune.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the trials table and manifest.",
    )
    tune.add_argument(
        "--models",
        action="append",
        metavar="NAME",
        help=(
            "Model to tune; repeatable. Defaults to every tunable model "
            f"({', '.join(TUNABLE_MODELS)}). An ablation borrows its donor's "
            "parameters and is not tunable on its own."
        ),
    )
    tune.add_argument(
        "--fold-set",
        action="append",
        metavar="NAME",
        dest="fold_sets",
        help=(
            "Fold set to tune for; repeatable. Defaults to every fold set present. "
            "Each gets its own study over a region ending before its own first test "
            "window, so the two are never mixed."
        ),
    )
    tune.add_argument(
        "--trials",
        type=int,
        metavar="N",
        default=DEFAULT_TRIALS,
        help=(
            f"Optuna trials per study (default {DEFAULT_TRIALS}). Lower values are for "
            "development; the documented production search uses the default."
        ),
    )
    tune.add_argument(
        "--seed",
        type=int,
        metavar="N",
        default=TUNING_SEED,
        help=f"Sampler seed (default {TUNING_SEED}). Fixed so a search is reproducible.",
    )
    tune.add_argument("--dry-run", action="store_true", help="Search, but write nothing.")
    tune.add_argument(
        "--report", action="store_true", help="Print the full validation report, not only failures."
    )

    # --- train-boosting ---------------------------------------------------
    train_boost = subparsers.add_parser(
        "train-boosting",
        parents=[common],
        help="Fit the boosted risk models and write fold-specific predictions.",
    )
    train_boost.add_argument(
        "--features",
        type=Path,
        metavar="PATH",
        help="Component 4 feature table. Defaults to the most recent.",
    )
    train_boost.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the prediction tables and manifest.",
    )
    train_boost.add_argument(
        "--models",
        action="append",
        metavar="NAME",
        help=(
            "Model to fit; repeatable. Defaults to every registered boosted model "
            f"({', '.join(spec.name for spec in BOOSTING_REGISTRY)})."
        ),
    )
    train_boost.add_argument(
        "--dry-run", action="store_true", help="Train and validate, but write nothing."
    )
    train_boost.add_argument(
        "--report", action="store_true", help="Print the full validation report, not only failures."
    )

    # --- build-neural-categoricals ----------------------------------------
    neural_cats = subparsers.add_parser(
        "build-neural-categoricals",
        parents=[common],
        help="Build Component 8's experimental as-of categorical table.",
    )
    neural_cats.add_argument(
        "--features",
        type=Path,
        metavar="PATH",
        help="Component 4 feature table. Defaults to the most recent.",
    )
    neural_cats.add_argument(
        "--raw",
        type=Path,
        metavar="PATH",
        help="Raw food-inspections snapshot. Defaults to the most recent.",
    )
    neural_cats.add_argument(
        "--assignments",
        type=Path,
        metavar="PATH",
        help="Component 2 establishment assignments. Defaults to the most recent.",
    )
    neural_cats.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the categorical table and manifest.",
    )
    neural_cats.add_argument(
        "--dry-run", action="store_true", help="Build and validate, but write nothing."
    )
    neural_cats.add_argument(
        "--report",
        action="store_true",
        help="Print the full validation report, not only failures.",
    )

    # --- tune-neural ------------------------------------------------------
    tune_net = subparsers.add_parser(
        "tune-neural",
        parents=[common],
        help="Search the neural learning rate on windows earlier than any test period.",
    )
    tune_net.add_argument(
        "--features",
        type=Path,
        metavar="PATH",
        help="Component 4 feature table. Defaults to the most recent.",
    )
    tune_net.add_argument(
        "--categoricals",
        type=Path,
        metavar="PATH",
        help="Component 8 categorical table. Defaults to the most recent.",
    )
    tune_net.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the sweep trials table and manifest.",
    )
    tune_net.add_argument(
        "--fold-set",
        action="append",
        dest="fold_sets",
        metavar="NAME",
        help="Fold set to search; repeatable. Defaults to every fold set present.",
    )
    tune_net.add_argument(
        "--seed",
        type=int,
        default=NEURAL_TUNING_SEED,
        metavar="N",
        help=f"Seed for every fit in the sweep (default: {NEURAL_TUNING_SEED}).",
    )
    tune_net.add_argument(
        "--dry-run", action="store_true", help="Search and validate, but write nothing."
    )
    tune_net.add_argument(
        "--report",
        action="store_true",
        help="Print the full validation report, not only failures.",
    )

    # --- train-neural -----------------------------------------------------
    train_net = subparsers.add_parser(
        "train-neural",
        parents=[common],
        help="Fit the neural models and write fold-specific predictions.",
    )
    train_net.add_argument(
        "--features",
        type=Path,
        metavar="PATH",
        help="Component 4 feature table. Defaults to the most recent.",
    )
    train_net.add_argument(
        "--categoricals",
        type=Path,
        metavar="PATH",
        help="Component 8 categorical table. Defaults to the most recent.",
    )
    train_net.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the prediction tables and manifest.",
    )
    train_net.add_argument(
        "--models",
        action="append",
        metavar="NAME",
        help=(
            "Model to fit; repeatable. Defaults to every registered neural model "
            f"({', '.join(spec.name for spec in NEURAL_REGISTRY)})."
        ),
    )
    train_net.add_argument(
        "--no-seed-sweep",
        action="store_true",
        help=(
            "Skip the multi-seed reproducibility run. Faster; leaves run-to-run "
            "variation unmeasured."
        ),
    )
    train_net.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip rendering the learning-curve and embedding figures.",
    )
    train_net.add_argument(
        "--figures-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the figures.",
    )
    train_net.add_argument(
        "--dry-run", action="store_true", help="Train and validate, but write nothing."
    )
    train_net.add_argument(
        "--report",
        action="store_true",
        help="Print the full validation report, not only failures.",
    )

    # --- calibrate --------------------------------------------------------
    calibrate = subparsers.add_parser(
        "calibrate",
        parents=[common],
        help="Fit probability calibrators on each fold's calibration window and freeze them.",
    )
    calibrate.add_argument(
        "--features", type=Path, metavar="PATH",
        help="Component 4 feature table. Defaults to the most recent.")
    calibrate.add_argument(
        "--categoricals", type=Path, metavar="PATH",
        help=("Component 8 categorical table, needed only by xgboost_chain_embeddings. "
              "Defaults to the most recent."))
    calibrate.add_argument(
        "--baseline-predictions", type=Path, metavar="PATH",
        help="Component 6 artifact, for the bit-identity gate. Defaults to the most recent.")
    calibrate.add_argument(
        "--boosted-predictions", type=Path, metavar="PATH",
        help="Component 7 artifact, for the bit-identity gate. Defaults to the most recent.")
    calibrate.add_argument(
        "--neural-predictions", type=Path, metavar="PATH",
        help="Component 8 artifact, for the bit-identity gate. Defaults to the most recent.")
    calibrate.add_argument(
        "--output-dir", type=Path, metavar="DIR",
        help="Destination for the calibration tables and manifest.")
    calibrate.add_argument(
        "--models", action="append", metavar="NAME",
        help=("Base model to calibrate; repeatable. Defaults to every candidate "
              f"({', '.join(spec.name for spec in CANDIDATE_REGISTRY)})."))
    calibrate.add_argument(
        "--method", choices=[m.value for m in Method], default=None,
        help=("Force one calibration method and skip the selection protocol. Diagnostic "
              "only -- the production run selects, and a forced run is recorded as such "
              "in the manifest."))
    calibrate.add_argument(
        "--bootstrap-replications", type=int, metavar="N", default=BOOTSTRAP_REPLICATIONS,
        help=f"Replications per bootstrap interval. Default {BOOTSTRAP_REPLICATIONS}.")
    calibrate.add_argument(
        "--no-figures", action="store_true",
        help="Skip the reliability diagrams and the drift figure.")
    calibrate.add_argument(
        "--figures-dir", type=Path, metavar="DIR",
        help="Destination for the figures. Defaults to docs/analysis/figures.")
    calibrate.add_argument(
        "--dry-run", action="store_true",
        help="Calibrate and validate, but write nothing.")
    calibrate.add_argument(
        "--report", action="store_true",
        help="Print the full validation report, not only failures.")

    # --- evaluate ---------------------------------------------------------
    evaluate = subparsers.add_parser(
        "evaluate",
        parents=[common],
        help="Run the rolling-origin backtest and the re-ordering simulation.",
    )
    evaluate.add_argument(
        "--features",
        type=Path,
        metavar="PATH",
        help="Component 4 feature table. Defaults to the most recent.",
    )
    evaluate.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the evaluation tables and manifest.",
    )
    evaluate.add_argument(
        "--folds-only",
        action="store_true",
        help="Emit the fold table and stop. Fast, and enough to audit the split.",
    )
    evaluate.add_argument(
        "--predictions",
        type=Path,
        metavar="PATH",
        help=(
            "Prediction artifact from a modelling component, scored alongside the "
            "built-in baselines. Omit to evaluate only the heuristics."
        ),
    )
    evaluate.add_argument(
        "--seeds",
        type=int,
        metavar="N",
        default=DEFAULT_RANDOM_REPLICATIONS,
        help=f"Random-schedule replications (default {DEFAULT_RANDOM_REPLICATIONS}).",
    )
    evaluate.add_argument(
        "--sensitivity-replications",
        type=int,
        metavar="N",
        default=DEFAULT_REPLICATIONS,
        help=f"Time-invariance label re-draws (default {DEFAULT_REPLICATIONS}).",
    )
    evaluate.add_argument(
        "--dry-run", action="store_true", help="Evaluate and validate, but write nothing."
    )
    evaluate.add_argument(
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


def _run_build_target(args: argparse.Namespace, settings: Settings) -> int:
    parquet_path = args.parquet
    assignments_path = args.assignments
    try:
        if parquet_path is None:
            parquet_path = duckdb_queries.latest_parquet(settings.food_inspections_raw_dir)
            logger.info("Using most recent raw file: %s", parquet_path)
        if assignments_path is None:
            assignments_path = duckdb_queries.latest_parquet(
                settings.entity_resolution_interim_dir,
                prefix="establishment_assignments_",
            )
            logger.info("Using most recent assignments: %s", assignments_path)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    try:
        result = build_targets(
            settings,
            parquet_path=parquet_path,
            assignments_path=assignments_path,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, TargetConstructionError, ValueError) as exc:
        logger.error("Target construction failed: %s", exc)
        return 1

    print(summarize_target(result))

    failed = target_validate.has_failures(result.checks)
    if args.report or failed:
        print(target_validate.format_report(result.checks))
    # A failed structural check means the labels are wrong, so the command fails
    # loudly rather than handing quietly broken targets to Component 4.
    return 1 if failed else 0


def _run_build_features(args: argparse.Namespace, settings: Settings) -> int:
    parquet_path = args.parquet
    assignments_path = args.assignments
    targets_path = args.targets
    try:
        if parquet_path is None:
            parquet_path = duckdb_queries.latest_parquet(settings.food_inspections_raw_dir)
            logger.info("Using most recent raw file: %s", parquet_path)
        if assignments_path is None:
            assignments_path = duckdb_queries.latest_parquet(
                settings.entity_resolution_interim_dir,
                prefix="establishment_assignments_",
            )
            logger.info("Using most recent assignments: %s", assignments_path)
        if targets_path is None:
            targets_path = duckdb_queries.latest_parquet(
                settings.target_interim_dir, prefix="inspection_targets_"
            )
            logger.info("Using most recent targets: %s", targets_path)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    try:
        result = build_features(
            settings,
            parquet_path=parquet_path,
            assignments_path=assignments_path,
            targets_path=targets_path,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, FeatureConstructionError, ValueError) as exc:
        logger.error("Feature construction failed: %s", exc)
        return 1

    print(summarize_features(result))

    failed = feature_validate.has_failures(result.checks)
    if args.report or failed:
        print(feature_validate.format_report(result.checks))
    # A failed check means a feature may contain future information, which is
    # the one defect that would silently invalidate every downstream result.
    return 1 if failed else 0


def _run_train_baselines(args: argparse.Namespace, settings: Settings) -> int:
    features_path = args.features
    try:
        if features_path is None:
            features_path = duckdb_queries.latest_parquet(
                settings.features_processed_dir, prefix="as_of_features_"
            )
            logger.info("Using most recent features: %s", features_path)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if args.models is not None and not args.models:
        raise SystemExit("--models requires at least one model name")

    try:
        result = train_baselines(
            settings,
            features_path=features_path,
            output_dir=args.output_dir,
            models=args.models,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, BaselineTrainingError, ValueError) as exc:
        logger.error("Baseline training failed: %s", exc)
        return 1

    print(summarize_baselines(result))

    failed = modeling_validate.has_failures(result.checks)
    if args.report or failed:
        print(modeling_validate.format_report(result.checks))
    # A failed check means a model may have been fitted on data it was not allowed to
    # see, or that its scores cannot be attributed to the right rows. Either makes
    # every number Component 5 would then report meaningless.
    return 1 if failed else 0


def _run_tune_boosting(args: argparse.Namespace, settings: Settings) -> int:
    features_path = args.features
    try:
        if features_path is None:
            features_path = duckdb_queries.latest_parquet(
                settings.features_processed_dir, prefix="as_of_features_"
            )
            logger.info("Using most recent features: %s", features_path)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if args.models is not None and not args.models:
        raise SystemExit("--models requires at least one model name")
    if args.fold_sets is not None and not args.fold_sets:
        raise SystemExit("--fold-set requires at least one fold set name")
    if args.trials < 1:
        raise SystemExit("--trials must be a positive integer")

    try:
        result = tune_boosting(
            settings,
            features_path=features_path,
            output_dir=args.output_dir,
            models=args.models,
            fold_sets=args.fold_sets,
            trials=args.trials,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, BoostingBuildError, TuningError, ValueError) as exc:
        logger.error("Hyperparameter search failed: %s", exc)
        return 1

    print(summarize_tuning(result))

    failed = boosting_validate.has_failures(result.checks)
    if args.report or failed:
        print(boosting_validate.format_report(result.checks))
    # A failed check means the search could have selected hyperparameters using a test
    # window, which is the one leak no downstream component can detect.
    return 1 if failed else 0


def _run_train_boosting(args: argparse.Namespace, settings: Settings) -> int:
    features_path = args.features
    try:
        if features_path is None:
            features_path = duckdb_queries.latest_parquet(
                settings.features_processed_dir, prefix="as_of_features_"
            )
            logger.info("Using most recent features: %s", features_path)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if args.models is not None and not args.models:
        raise SystemExit("--models requires at least one model name")

    try:
        result = train_boosting(
            settings,
            features_path=features_path,
            output_dir=args.output_dir,
            models=args.models,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, BoostingBuildError, ValueError) as exc:
        logger.error("Boosted training failed: %s", exc)
        return 1

    print(summarize_boosting(result))

    failed = boosting_validate.has_failures(result.checks)
    if args.report or failed:
        print(boosting_validate.format_report(result.checks))
    # A failed check means a model may have been fitted on data it was not allowed to
    # see, or that its scores cannot be attributed to the right rows. Either makes
    # every number Component 5 would then report meaningless.
    return 1 if failed else 0


def _latest(settings: Settings, directory: Path, prefix: str, label: str) -> Path:
    """Resolve the most recent artifact under one prefix, or explain what is missing."""
    try:
        path = duckdb_queries.latest_parquet(directory, prefix=prefix)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label}: {exc}") from exc
    logger.info("Using most recent %s: %s", label, path)
    return path


def _run_build_neural_categoricals(args: argparse.Namespace, settings: Settings) -> int:
    try:
        features_path = args.features or _latest(
            settings, settings.features_processed_dir, "as_of_features_", "features"
        )
        raw_path = args.raw or _latest(
            settings, settings.food_inspections_raw_dir, "food_inspections_", "raw snapshot"
        )
        assignments_path = args.assignments or _latest(
            settings,
            settings.entity_resolution_interim_dir,
            "establishment_assignments_",
            "establishment assignments",
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    try:
        result = build_neural_categoricals(
            settings,
            features_path=features_path,
            raw_path=raw_path,
            assignments_path=assignments_path,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, CategoricalBuildError, NeuralBuildError, ValueError) as exc:
        logger.error("Categorical build failed: %s", exc)
        return 1

    print(summarize_neural_categoricals(result))

    failed = neural_validate.has_failures(result.checks)
    if args.report or failed:
        print(neural_validate.format_report(result.checks))
    return 1 if failed else 0


def _run_tune_neural(args: argparse.Namespace, settings: Settings) -> int:
    try:
        features_path = args.features or _latest(
            settings, settings.features_processed_dir, "as_of_features_", "features"
        )
        categoricals_path = args.categoricals or _latest(
            settings, settings.neural_processed_dir, "neural_categoricals_", "categoricals"
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if args.fold_sets is not None and not args.fold_sets:
        raise SystemExit("--fold-set requires at least one fold set name")

    try:
        result = tune_neural(
            settings,
            features_path=features_path,
            categoricals_path=categoricals_path,
            output_dir=args.output_dir,
            fold_sets=args.fold_sets,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    except (
        FileNotFoundError,
        NeuralTuningError,
        NeuralBuildError,
        TuningError,
        ValueError,
    ) as exc:
        logger.error("Neural tuning failed: %s", exc)
        return 1

    print(summarize_neural_tuning(result))

    failed = neural_validate.has_failures(result.checks)
    if args.report or failed:
        print(neural_validate.format_report(result.checks))
    return 1 if failed else 0


def _run_train_neural(args: argparse.Namespace, settings: Settings) -> int:
    try:
        features_path = args.features or _latest(
            settings, settings.features_processed_dir, "as_of_features_", "features"
        )
        categoricals_path = args.categoricals or _latest(
            settings, settings.neural_processed_dir, "neural_categoricals_", "categoricals"
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if args.models is not None and not args.models:
        raise SystemExit("--models requires at least one model name")

    try:
        result = train_neural(
            settings,
            features_path=features_path,
            categoricals_path=categoricals_path,
            output_dir=args.output_dir,
            models=args.models,
            seed_sweep=not args.no_seed_sweep,
            render_figures=not args.no_figures,
            figures_dir=args.figures_dir,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, NeuralBuildError, NeuralTrainError, ValueError) as exc:
        logger.error("Neural training failed: %s", exc)
        return 1

    print(summarize_neural(result))

    failed = neural_validate.has_failures(result.checks)
    if args.report or failed:
        print(neural_validate.format_report(result.checks))
    return 1 if failed else 0


def _run_calibrate(args: argparse.Namespace, settings: Settings) -> int:
    try:
        features_path = args.features or _latest(
            settings, settings.features_processed_dir, "as_of_features_", "features"
        )
        categoricals_path = args.categoricals or _latest(
            settings, settings.neural_processed_dir, "neural_categoricals_", "categoricals"
        )
        prediction_paths = {
            "baseline_predictions": args.baseline_predictions
            or _latest(
                settings, settings.predictions_processed_dir,
                "baseline_predictions_", "Component 6 predictions",
            ),
            "boosted_predictions": args.boosted_predictions
            or _latest(
                settings, settings.predictions_processed_dir,
                "boosted_predictions_", "Component 7 predictions",
            ),
            "neural_predictions": args.neural_predictions
            or _latest(
                settings, settings.predictions_processed_dir,
                "neural_predictions_", "Component 8 predictions",
            ),
        }
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if args.models is not None and not args.models:
        raise SystemExit("--models requires at least one model name")
    if args.bootstrap_replications < 1:
        raise SystemExit("--bootstrap-replications must be a positive integer")

    try:
        result = run_calibration(
            settings,
            features_path=features_path,
            categoricals_path=categoricals_path,
            prediction_paths=prediction_paths,
            output_dir=args.output_dir,
            models=args.models,
            method_override=Method(args.method) if args.method else None,
            bootstrap_replications=args.bootstrap_replications,
            figures_dir=args.figures_dir,
            write_figures=not args.no_figures,
            dry_run=args.dry_run,
        )
    except (
        FileNotFoundError,
        CalibrationBuildError,
        CalibrationTrainError,
        CalibrationPreprocessError,
        BaseScoreError,
        ValueError,
    ) as exc:
        logger.error("Calibration failed: %s", exc)
        return 1

    print(summarize_calibration(result))

    failed = calibration_validate.has_failures(result.checks)
    if args.report or failed:
        print(calibration_validate.format_report(result.checks))
    # A failed check means either the re-executed base model is not the one Components 6-8
    # published -- so the calibrator corrects something that was never scored -- or a
    # calibrator read a window it was not allowed to. Both make every probability in the
    # artifact untrustworthy, which is worse than having no artifact.
    return 1 if failed else 0


def _run_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    features_path = args.features
    try:
        if features_path is None:
            features_path = duckdb_queries.latest_parquet(
                settings.features_processed_dir, prefix="as_of_features_"
            )
            logger.info("Using most recent features: %s", features_path)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if args.seeds < 1:
        raise SystemExit("--seeds must be a positive integer")
    if args.sensitivity_replications < 0:
        raise SystemExit("--sensitivity-replications must not be negative")

    try:
        result = run_evaluation(
            settings,
            features_path=features_path,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            folds_only=args.folds_only,
            random_seeds=args.seeds,
            sensitivity_replications=args.sensitivity_replications,
            predictions_path=args.predictions,
        )
    except (FileNotFoundError, EvaluationError, ValueError) as exc:
        logger.error("Evaluation failed: %s", exc)
        return 1

    print(summarize_evaluation(result))

    failed = evaluation_validate.has_failures(result.checks)
    if args.report or failed:
        print(evaluation_validate.format_report(result.checks))
    # A failed check means the evaluation itself could see the future, which
    # would make every number it reports confidently wrong.
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
    if args.command == "build-target":
        return _run_build_target(args, settings)
    if args.command == "build-features":
        return _run_build_features(args, settings)
    if args.command == "train-baselines":
        return _run_train_baselines(args, settings)
    if args.command == "tune-boosting":
        return _run_tune_boosting(args, settings)
    if args.command == "train-boosting":
        return _run_train_boosting(args, settings)
    if args.command == "build-neural-categoricals":
        return _run_build_neural_categoricals(args, settings)
    if args.command == "tune-neural":
        return _run_tune_neural(args, settings)
    if args.command == "train-neural":
        return _run_train_neural(args, settings)
    if args.command == "calibrate":
        return _run_calibrate(args, settings)
    if args.command == "evaluate":
        return _run_evaluate(args, settings)

    # argparse enforces `required=True` on the subparser, so this is defensive
    # only. parser.error() exits with status 2.
    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
