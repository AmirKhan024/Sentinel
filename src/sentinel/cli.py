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
    sentinel plan-candidates --planning-date D  build Component 17's operational candidates
    sentinel plan-candidates --planning-date D --dry-run --report  build without writing
    sentinel score-candidates                  Component 18: score candidates, rank priority
    sentinel score-candidates --dry-run --report  score and validate without writing
    sentinel select-inspections --capacity 30  Component 19: capacity-constrained selection
    sentinel select-inspections --capacity 30 --policy coverage_floor_population_share
    sentinel organize-geography                Component 20: geographic work blocks
    sentinel organize-geography --threshold-km 2.0
    sentinel organize-geography --threshold-preset broad
    sentinel organize-geography --organization-mode geography_assisted
    sentinel review-plan                       Component 21: supervisor plan review summary
    sentinel review-plan --decisions PATH      apply staged supervisor decisions
    sentinel approve-plan --approval-id ID --planning-date D --approved-by NAME
                                                Component 21: approve a plan for Component 22
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
    sentinel explain                           attribute predictions to features (SHAP)
    sentinel explain --models NAME              explain only the named model(s)
    sentinel explain --sample-size N            rows explained per model per fold
    sentinel explain --no-figures               skip the importance and local-case figures
    sentinel explain --dry-run --report         attribute and validate without writing
    sentinel audit-fairness                    audit group behaviour across geography
    sentinel audit-fairness --models NAME       audit only the named model(s)
    sentinel audit-fairness --group-definitions NAME   audit one group definition
    sentinel audit-fairness --no-figures        skip the equity figures
    sentinel audit-fairness --dry-run --report  audit and validate without writing
    sentinel decide                            build the recommended inspection queue
    sentinel decide --policies NAME             compare only the named policy (plus pure_risk)
    sentinel decide --model NAME                override the selected production model
    sentinel decide --overrides PATH            apply a human override file, audited
    sentinel decide --no-figures                skip the policy trade-off figures
    sentinel decide --dry-run --report          decide and validate without writing
    sentinel evaluate                          run the temporal evaluation harness
    sentinel evaluate --folds-only             emit the fold table and stop
    sentinel evaluate --predictions PATH       also score a prediction artifact
    sentinel evaluate --dry-run --report       evaluate without writing anything

``plan-candidates`` is Component 17. It shares Component 4's feature engine unmodified --
the same range join, the same missing-value rules, the same ``feature_definition_version``
-- run against a planning date instead of a real Component 3 target row. No future
information is ever read: only records with ``inspection_date < planning_date`` enter the
computation. A planning date on or before the raw snapshot's earliest record is refused
outright; a planning date after the most recent ingested record is accepted but flagged, in
the manifest, as reflecting the last ingest rather than a live feed. It writes to its own
artifact family and is never joined onto ``features/``.

``score-candidates`` is Component 18. It re-executes Components 6/7/8's own frozen fit
functions -- unmodified -- on a training window ending strictly before the planning date
(``fold_set="operational"``, never a real evaluation fold), scores Component 17's
candidates, and applies Component 9's *persisted* calibrator parameters (no refitting: a
calibrator is genuinely an artifact in this project, unlike a base model -- ADR 0026).
Which base model to use is decided by Component 13's own frozen selection rule
(``policy.select``), so operational mode and the historical recommendation queue can
never name two different "production models." No fitted model is ever pickled or
otherwise persisted; the same planning date against the same committed data reproduces
the same scores by construction, the same way Component 9's own bit-identity gate does.

``select-inspections`` is Component 19. It takes Component 18's full, unbounded priority
ranking and an explicit ``--capacity`` (inspection slots, never an inspector count -- nothing
in this project defines that conversion) and calls ``policy.allocation.allocate()`` /
``.decide()`` unmodified: the same risk-block-plus-coverage-reserve engine ``sentinel decide``
uses, fed a window built from Component 18's real scores instead of a historical fold. Every
Component 18 row survives in the output, selected or not, each carrying why (Component 13's
own ``DecisionMechanism``/``DecisionReason`` vocabulary). No establishment is ever fabricated
to fill a shortfall; a capacity that exceeds the selectable pool is reported as unfilled.

``organize-geography`` is Component 20. It reads Component 19's output only -- never
Component 18's -- so location can never bypass capacity/policy selection. It groups the
``is_selected == True`` rows into deterministic geographic work blocks (Haversine
distance-threshold connected components, reusing ``entity.evidence.haversine_m`` and
``entity.unionfind.UnionFind``), never changes which establishments were selected, and
never rewrites a Component 18/19 risk or policy field. An establishment with no usable
coordinates is preserved in an explicit "unmapped" group, never dropped and never given
a fabricated location. This is geographic distance, not driving distance or travel time;
no routing, travel-time, or inspector-assignment capability exists here or is implied. A
work block is not a workday: capacity/staffing remain a separate, unmodeled constraint.
Within each block, a suggested work order is produced by ``--organization-mode``:
``risk_first`` (default) is exactly Sentinel's own ``policy_rank`` order; geography never
reorders it. ``geography_assisted`` accepts a small amount of reordering, via a
deterministic nearest-neighbour heuristic, to reduce spatial back-and-forth -- it is a
heuristic ordering, never a route. ``--threshold-km``/``--threshold-preset`` (tight/
balanced/broad) are mutually exclusive ways to set the same grouping threshold; when most
resulting blocks are singletons, the manifest says so explicitly rather than hiding it.

``review-plan`` is Component 21. It reads Component 20's output only, and never edits a
geographic, risk, or policy field. It computes the supervisor-facing plan summary (workload,
mapped/unmapped, work-block count, decisions recorded) and, when given ``--decisions``, joins
in a batch of human plan decisions (``keep_selected``/``move_to_later_workday``/
``do_not_proceed_as_planned``/``adjust_operational_priority``), validated all-or-nothing the
same way Component 16 validates review resolutions. A plan decision is a distinct, additional,
audited fact -- it never overwrites Sentinel's own recommendation, and it never creates a
Component 13 override or Component 14 adjustment itself. ``adjust_operational_priority`` sets
a separate, display-only ``operational_priority`` field for field-work ordering; it never
touches ``rank`` or ``policy_rank``.

``approve-plan`` is also Component 21. It is a distinct, explicit act -- not a side effect of
every row having a decision -- that runs a readiness checklist (every row carries the machine
recommendation and geographic provenance, every recorded decision has a reason, no duplicate
establishment) and, only if every check passes, writes an ``approved_operational_plan``
artifact: a permanent, never-rewritten record of exactly what was handed to Component 22,
naming the exact ``supervisor_plan_review`` snapshot it approved by checksum. If the plan is
not ready, approval is refused outright with the specific failing check, never partially
applied.

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

``explain`` is Component 11 and is a *reader*. It re-executes each model's frozen fit to
obtain a model object -- no fitted model is persisted anywhere (ADR 0026) -- proves the
re-execution is the committed model by comparing its test scores bit for bit, and then
attributes those predictions to features. It fits nothing, changes no prediction, and
must never be read as selecting a model: attributions say how a model reasoned, not
whether it was right, and Components 5 to 9 own that question. See ADR 0029, ADR 0030.
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
from sentinel.candidates.build import build_candidates
from sentinel.candidates.build import summarize as summarize_candidates
from sentinel.candidates.universe import CandidateGenerationError
from sentinel.config import Settings, load_settings
from sentinel.entity import validate
from sentinel.entity.resolve import EntityResolutionError, resolve_establishments, summarize
from sentinel.evaluation import validate as evaluation_validate
from sentinel.evaluation.build import EvaluationError, run_evaluation
from sentinel.evaluation.build import summarize as summarize_evaluation
from sentinel.evaluation.sensitivity import DEFAULT_REPLICATIONS
from sentinel.evaluation.simulate import DEFAULT_RANDOM_REPLICATIONS
from sentinel.explain import validate as explain_validate
from sentinel.explain.attribute import AttributionError
from sentinel.explain.background import BackgroundError
from sentinel.explain.build import ExplainBuildError, run_explanations
from sentinel.explain.build import summarize as summarize_explanations
from sentinel.explain.definitions import EXPLAIN_REGISTRY, SAMPLE_SIZE, SUPPORTED_MODELS
from sentinel.explain.refit import RefitError
from sentinel.explain.sample import SampleError
from sentinel.fairness import validate as fairness_validate
from sentinel.fairness.attribution import AttributionError as FairnessAttributionError
from sentinel.fairness.build import FairnessBuildError, run_fairness_audit
from sentinel.fairness.build import summarize as summarize_fairness
from sentinel.fairness.definitions import (
    AUDITED_GROUP_DEFINITIONS,
    GROUP_DEFINITION_REGISTRY,
    FairnessDefinitionError,
    GroupDefinitionStatus,
)
from sentinel.fairness.disparity import DisparityError
from sentinel.fairness.drift import DriftError
from sentinel.fairness.groups import GroupFrameError
from sentinel.fairness.metrics import GroupMetricError
from sentinel.fairness.missingness import MissingnessError
from sentinel.fairness.priority import PriorityError
from sentinel.features import validate as feature_validate
from sentinel.features.build import (
    FeatureConstructionError,
    build_features,
)
from sentinel.features.build import (
    summarize as summarize_features,
)
from sentinel.geographic_organization import validate as geographic_organization_validate
from sentinel.geographic_organization.build import (
    GeographicOrganizationBuildError,
    build_geographic_plan,
)
from sentinel.geographic_organization.build import summarize as summarize_geographic_plan
from sentinel.geographic_organization.definitions import (
    DEFAULT_GEO_THRESHOLD_KM,
    GEO_THRESHOLD_PRESETS,
    OrganizationMode,
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
from sentinel.operational_scoring import validate as operational_scoring_validate
from sentinel.operational_scoring.build import (
    OperationalPriorityBuildError,
    build_operational_priorities,
)
from sentinel.operational_scoring.build import summarize as summarize_operational_priorities
from sentinel.operational_selection import validate as operational_selection_validate
from sentinel.operational_selection.build import (
    OperationalSelectionBuildError,
    build_operational_selection,
)
from sentinel.operational_selection.build import summarize as summarize_operational_selection
from sentinel.plan_review import validate as plan_review_validate
from sentinel.plan_review.approval import PlanApprovalGovernanceError, format_readiness_report
from sentinel.plan_review.build import (
    PlanApprovalBuildError,
    PlanReviewBuildError,
    build_approved_plan,
    build_plan_review,
    summarize_approval,
)
from sentinel.plan_review.build import summarize as summarize_plan_review
from sentinel.plan_review.models import PlanApprovalRequest
from sentinel.policy import validate as policy_validate
from sentinel.policy.allocation import AllocationError
from sentinel.policy.build import PolicyBuildError, run_policy
from sentinel.policy.build import summarize as summarize_policy
from sentinel.policy.definitions import (
    CANDIDATE_MODELS,
    POLICY_GRID,
    REFUSED_MODELS,
    PolicyDefinitionError,
)
from sentinel.policy.eligibility import EligibilityError
from sentinel.policy.evaluate import EvaluationError as PolicyEvaluationError
from sentinel.policy.governance import GovernanceError
from sentinel.policy.inputs import PolicyInputError
from sentinel.policy.select import SelectionError
from sentinel.query import duckdb_queries
from sentinel.review import validate as review_validate
from sentinel.review.build import ReviewBuildError, run_review
from sentinel.review.build import summarize as summarize_review
from sentinel.review.definitions import ReviewDefinitionError
from sentinel.review.inputs import ReviewInputError
from sentinel.review.resolution import ReviewGovernanceError
from sentinel.scheduling import validate as schedule_validate
from sentinel.scheduling.adjustments import AdjustmentError
from sentinel.scheduling.allocation import ScheduleAllocationError
from sentinel.scheduling.build import run_schedule
from sentinel.scheduling.build import summarize as summarize_schedule
from sentinel.scheduling.definitions import CONFIG_GRID as SCHEDULE_CONFIG_GRID
from sentinel.scheduling.definitions import K_LEVELS as SCHEDULE_K_LEVELS
from sentinel.scheduling.definitions import CapacityMode, SchedulingDefinitionError
from sentinel.scheduling.execution import ExecutionError
from sentinel.scheduling.horizon import HorizonError
from sentinel.scheduling.inputs import ScheduleInputError
from sentinel.scheduling.replan import ReplanError
from sentinel.target import validate as target_validate
from sentinel.target.build import TargetConstructionError, build_targets
from sentinel.target.build import summarize as summarize_target

logger = logging.getLogger("sentinel.cli")

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

#: Optuna trials per study. The documented production search; `--trials` lowers it for
#: development, and the findings document distinguishes the two runs explicitly.
DEFAULT_TRIALS = 100

#: Group definitions the registry refuses, named in the CLI help so a user who types one
#: is told why rather than that they may not. See ADR 0033.
REFUSED_DEFINITIONS = ", ".join(
    spec.name for spec in GROUP_DEFINITION_REGISTRY if spec.status is GroupDefinitionStatus.REFUSED
)


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

    # --- plan-candidates ----------------------------------------------------
    plan_candidates = subparsers.add_parser(
        "plan-candidates",
        parents=[common],
        help=(
            "Component 17: build the operational candidate/feature table for a "
            "planning date, using only records strictly before it."
        ),
    )
    plan_candidates.add_argument(
        "--planning-date",
        required=True,
        metavar="YYYY-MM-DD",
        help="The operational planning date. No future information is ever read.",
    )
    plan_candidates.add_argument(
        "--parquet",
        type=Path,
        metavar="PATH",
        help="Raw Parquet. Defaults to the most recent raw file.",
    )
    plan_candidates.add_argument(
        "--assignments",
        type=Path,
        metavar="PATH",
        help="Component 2 assignments. Defaults to the most recent.",
    )
    plan_candidates.add_argument(
        "--establishments",
        type=Path,
        metavar="PATH",
        help="Component 2 establishments. Defaults to the most recent.",
    )
    plan_candidates.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the candidate table and manifest.",
    )
    plan_candidates.add_argument(
        "--dry-run", action="store_true", help="Construct and validate, but write nothing."
    )
    plan_candidates.add_argument(
        "--report", action="store_true", help="Print the full validation report, not only failures."
    )

    # --- score-candidates -----------------------------------------------------
    score_candidates_parser = subparsers.add_parser(
        "score-candidates",
        parents=[common],
        help=(
            "Component 18: score a Component 17 candidate set with Sentinel's validated "
            "production model and produce a deterministic operational priority ranking."
        ),
    )
    score_candidates_parser.add_argument(
        "--candidates",
        type=Path,
        metavar="PATH",
        help="Component 17 operational candidate table. Defaults to the most recent.",
    )
    score_candidates_parser.add_argument(
        "--features",
        type=Path,
        metavar="PATH",
        help="Component 4 historical feature table. Defaults to the most recent.",
    )
    score_candidates_parser.add_argument(
        "--simulation",
        type=Path,
        metavar="PATH",
        help="Component 5 simulation summary. Defaults to the most recent.",
    )
    score_candidates_parser.add_argument(
        "--metrics",
        type=Path,
        metavar="PATH",
        help="Component 5 evaluation metrics. Defaults to the most recent.",
    )
    score_candidates_parser.add_argument(
        "--sensitivity",
        type=Path,
        metavar="PATH",
        help="Component 5 NDE sensitivity bands. Defaults to the most recent.",
    )
    score_candidates_parser.add_argument(
        "--calibrated-predictions",
        type=Path,
        metavar="PATH",
        help="Component 9 calibrated predictions. Defaults to the most recent.",
    )
    score_candidates_parser.add_argument(
        "--calibrator-parameters",
        type=Path,
        metavar="PATH",
        help="Component 9 calibrator parameters. Defaults to the most recent.",
    )
    score_candidates_parser.add_argument(
        "--calibrator-breakpoints",
        type=Path,
        metavar="PATH",
        help="Component 9 isotonic breakpoints. Defaults to the most recent.",
    )
    score_candidates_parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the priority table and manifest.",
    )
    score_candidates_parser.add_argument(
        "--dry-run", action="store_true", help="Score and validate, but write nothing."
    )
    score_candidates_parser.add_argument(
        "--report", action="store_true", help="Print the full validation report, not only failures."
    )

    # --- select-inspections -----------------------------------------------------
    select_inspections_parser = subparsers.add_parser(
        "select-inspections",
        parents=[common],
        help=(
            "Component 19: select a capacity-constrained inspection plan from a "
            "Component 18 priority set, using Component 13's own allocation engine."
        ),
    )
    select_inspections_parser.add_argument(
        "--capacity",
        type=int,
        required=True,
        metavar="N",
        help="Maximum inspections this planning run can perform. Not an inspector count.",
    )
    select_inspections_parser.add_argument(
        "--priority-set",
        type=Path,
        metavar="PATH",
        help="Component 18 operational priority set. Defaults to the most recent.",
    )
    select_inspections_parser.add_argument(
        "--planning-date",
        metavar="YYYY-MM-DD",
        help="Must match the priority set's own planning date if given.",
    )
    select_inspections_parser.add_argument(
        "--policy",
        default="",
        metavar="POLICY_ID",
        help="A sentinel.policy.definitions.POLICY_GRID id. Defaults to the grid's "
        "baseline (plain top-k, no coverage reserve).",
    )
    select_inspections_parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the selection table and manifest.",
    )
    select_inspections_parser.add_argument(
        "--dry-run", action="store_true", help="Select and validate, but write nothing."
    )
    select_inspections_parser.add_argument(
        "--report", action="store_true", help="Print the full validation report, not only failures."
    )

    # --- organize-geography -----------------------------------------------------
    organize_geography_parser = subparsers.add_parser(
        "organize-geography",
        parents=[common],
        help=(
            "Component 20: geographically organize a Component 19 selected set "
            "into deterministic proximity groups. Never changes who was selected."
        ),
    )
    organize_geography_parser.add_argument(
        "--selection",
        type=Path,
        metavar="PATH",
        help="Component 19 selection set. Defaults to the most recent.",
    )
    geo_threshold_group = organize_geography_parser.add_mutually_exclusive_group()
    geo_threshold_group.add_argument(
        "--threshold-km",
        type=float,
        default=None,
        metavar="KM",
        help=f"Geographic proximity threshold in kilometres. Default {DEFAULT_GEO_THRESHOLD_KM} "
        "-- a configurable operational heuristic, not a validated travel distance. "
        "Mutually exclusive with --threshold-preset.",
    )
    geo_threshold_group.add_argument(
        "--threshold-preset",
        choices=sorted(GEO_THRESHOLD_PRESETS),
        default=None,
        help="Named threshold label instead of an explicit --threshold-km: "
        + ", ".join(f"{name}={km} km" for name, km in sorted(GEO_THRESHOLD_PRESETS.items())),
    )
    organize_geography_parser.add_argument(
        "--organization-mode",
        choices=[m.value for m in OrganizationMode],
        default=OrganizationMode.RISK_FIRST.value,
        help="How the suggested order within a geographic work block is produced. "
        "risk_first (default) preserves Sentinel's priority ordering exactly; "
        "geography_assisted accepts a small amount of reordering for geographic coherence. "
        "Neither ever changes which establishments are grouped into a block.",
    )
    organize_geography_parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the geographic plan and manifest.",
    )
    organize_geography_parser.add_argument(
        "--dry-run", action="store_true", help="Organize and validate, but write nothing."
    )
    organize_geography_parser.add_argument(
        "--report", action="store_true", help="Print the full validation report, not only failures."
    )

    # --- review-plan -------------------------------------------------------------
    review_plan_parser = subparsers.add_parser(
        "review-plan",
        parents=[common],
        help=(
            "Component 21: supervisor plan review summary for a Component 20 geographic "
            "plan, optionally joined with a batch of human plan decisions."
        ),
    )
    review_plan_parser.add_argument(
        "--plan",
        type=Path,
        metavar="PATH",
        help="Component 20 geographic plan. Defaults to the most recent.",
    )
    review_plan_parser.add_argument(
        "--decisions",
        type=Path,
        metavar="PATH",
        help="JSON file of supervisor plan decisions (the operator's own accumulated "
        "file, matching --resolutions for `sentinel review`). Optional.",
    )
    review_plan_parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the plan review and decision log.",
    )
    review_plan_parser.add_argument(
        "--dry-run", action="store_true", help="Review and validate, but write nothing."
    )
    review_plan_parser.add_argument(
        "--report", action="store_true", help="Print the full validation report, not only failures."
    )

    # --- approve-plan --------------------------------------------------------------
    approve_plan_parser = subparsers.add_parser(
        "approve-plan",
        parents=[common],
        help=(
            "Component 21: approve a supervisor plan review, producing the immutable "
            "approved_operational_plan Component 22 consumes. Refuses outright, never "
            "partially, if the plan is not ready."
        ),
    )
    approve_plan_parser.add_argument(
        "--review",
        type=Path,
        metavar="PATH",
        help="Supervisor plan review (supervisor_plan_review_*.parquet). Defaults to the "
        "most recent.",
    )
    approve_plan_parser.add_argument(
        "--decision-log",
        type=Path,
        metavar="PATH",
        help="The plan_decision_log artifact this review was built with, for provenance. Optional.",
    )
    approve_plan_parser.add_argument("--approval-id", required=True, metavar="ID")
    approve_plan_parser.add_argument("--planning-date", required=True, metavar="YYYY-MM-DD")
    approve_plan_parser.add_argument(
        "--approved-by", required=True, metavar="NAME", help="The approving supervisor's name/id."
    )
    approve_plan_parser.add_argument(
        "--approved-at",
        metavar="ISO8601",
        help="Defaults to the current time if omitted.",
    )
    approve_plan_parser.add_argument("--note", metavar="TEXT", default=None)
    approve_plan_parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination directory for the approved plan and manifest.",
    )
    approve_plan_parser.add_argument(
        "--dry-run", action="store_true", help="Check readiness, but write nothing."
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
        "--features",
        type=Path,
        metavar="PATH",
        help="Component 4 feature table. Defaults to the most recent.",
    )
    calibrate.add_argument(
        "--categoricals",
        type=Path,
        metavar="PATH",
        help=(
            "Component 8 categorical table, needed only by xgboost_chain_embeddings. "
            "Defaults to the most recent."
        ),
    )
    calibrate.add_argument(
        "--baseline-predictions",
        type=Path,
        metavar="PATH",
        help="Component 6 artifact, for the bit-identity gate. Defaults to the most recent.",
    )
    calibrate.add_argument(
        "--boosted-predictions",
        type=Path,
        metavar="PATH",
        help="Component 7 artifact, for the bit-identity gate. Defaults to the most recent.",
    )
    calibrate.add_argument(
        "--neural-predictions",
        type=Path,
        metavar="PATH",
        help="Component 8 artifact, for the bit-identity gate. Defaults to the most recent.",
    )
    calibrate.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination for the calibration tables and manifest.",
    )
    calibrate.add_argument(
        "--models",
        action="append",
        metavar="NAME",
        help=(
            "Base model to calibrate; repeatable. Defaults to every candidate "
            f"({', '.join(spec.name for spec in CANDIDATE_REGISTRY)})."
        ),
    )
    calibrate.add_argument(
        "--method",
        choices=[m.value for m in Method],
        default=None,
        help=(
            "Force one calibration method and skip the selection protocol. Diagnostic "
            "only -- the production run selects, and a forced run is recorded as such "
            "in the manifest."
        ),
    )
    calibrate.add_argument(
        "--bootstrap-replications",
        type=int,
        metavar="N",
        default=BOOTSTRAP_REPLICATIONS,
        help=f"Replications per bootstrap interval. Default {BOOTSTRAP_REPLICATIONS}.",
    )
    calibrate.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip the reliability diagrams and the drift figure.",
    )
    calibrate.add_argument(
        "--figures-dir",
        type=Path,
        metavar="DIR",
        help="Destination for the figures. Defaults to docs/analysis/figures.",
    )
    calibrate.add_argument(
        "--dry-run", action="store_true", help="Calibrate and validate, but write nothing."
    )
    calibrate.add_argument(
        "--report", action="store_true", help="Print the full validation report, not only failures."
    )

    # --- explain ----------------------------------------------------------
    explain = subparsers.add_parser(
        "explain",
        parents=[common],
        help="Attribute each model's predictions to features, per fold, with SHAP.",
    )
    explain.add_argument(
        "--features",
        type=Path,
        metavar="PATH",
        help="Component 4 feature table. Defaults to the most recent.",
    )
    explain.add_argument(
        "--baseline-predictions",
        type=Path,
        metavar="PATH",
        help="Component 6 artifact, for the bit-identity gate. Defaults to the most recent.",
    )
    explain.add_argument(
        "--boosted-predictions",
        type=Path,
        metavar="PATH",
        help="Component 7 artifact, for the bit-identity gate. Defaults to the most recent.",
    )
    explain.add_argument(
        "--neural-predictions",
        type=Path,
        metavar="PATH",
        help="Component 8 artifact, for the bit-identity gate. Defaults to the most recent.",
    )
    explain.add_argument(
        "--calibrated-predictions",
        type=Path,
        metavar="PATH",
        help=(
            "Component 9 artifact, so each explanation carries its calibrated probability "
            "beside the base score. Optional; defaults to the most recent if one exists."
        ),
    )
    explain.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination for the explanation tables and manifest.",
    )
    explain.add_argument(
        "--models",
        action="append",
        metavar="NAME",
        help=(
            "Model to explain; repeatable. Defaults to every supported model "
            f"({', '.join(SUPPORTED_MODELS)}). Unsupported, with the reason recorded in "
            "the artifact: "
            f"{', '.join(x.name for x in EXPLAIN_REGISTRY if x.name not in SUPPORTED_MODELS)}."
        ),
    )
    explain.add_argument(
        "--sample-size",
        type=int,
        metavar="N",
        default=SAMPLE_SIZE,
        help=(
            f"Rows explained per model per fold. Default {SAMPLE_SIZE}. Changing it "
            "changes the artifact, and the value used is recorded in every row."
        ),
    )
    explain.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip the importance, stability, beeswarm and local-case figures.",
    )
    explain.add_argument(
        "--figures-dir",
        type=Path,
        metavar="DIR",
        help="Destination for the figures. Defaults to docs/analysis/figures.",
    )
    explain.add_argument(
        "--dry-run", action="store_true", help="Attribute and validate, but write nothing."
    )
    explain.add_argument(
        "--report", action="store_true", help="Print the full validation report, not only failures."
    )

    # --- audit-fairness ---------------------------------------------------
    fairness = subparsers.add_parser(
        "audit-fairness",
        parents=[common],
        help="Audit group behaviour across the geographies this data can define.",
    )
    fairness.add_argument(
        "--features",
        type=Path,
        metavar="PATH",
        help="Component 4 feature table, for the outcome label and the null-rule families.",
    )
    fairness.add_argument(
        "--calibrated-predictions",
        type=Path,
        metavar="PATH",
        help=(
            "Component 9 artifact. Carries the base score beside the calibrated one, which "
            "is what lets the audit compare the two stages without a join. Defaults to the "
            "most recent."
        ),
    )
    fairness.add_argument(
        "--categoricals",
        type=Path,
        metavar="PATH",
        help=(
            "Component 8 as-of categorical table, where community area and ZIP live "
            "(ADR 0022). Defaults to the most recent."
        ),
    )
    fairness.add_argument(
        "--explanations",
        type=Path,
        metavar="PATH",
        help=(
            "Component 11 attributions, for the per-group feature-reliance profiles. "
            "Optional; absent is a supported state and leaves that table empty."
        ),
    )
    fairness.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination for the fairness tables and manifest.",
    )
    fairness.add_argument(
        "--models",
        action="append",
        metavar="NAME",
        help="Calibrated model to audit; repeatable. Defaults to every model in the artifact.",
    )
    fairness.add_argument(
        "--group-definitions",
        action="append",
        metavar="NAME",
        help=(
            "Group definition to audit; repeatable. Defaults to "
            f"{', '.join(AUDITED_GROUP_DEFINITIONS)}. Refused with a measured reason: "
            f"{REFUSED_DEFINITIONS}."
        ),
    )
    fairness.add_argument("--no-figures", action="store_true", help="Skip the equity figures.")
    fairness.add_argument(
        "--figures-dir",
        type=Path,
        metavar="DIR",
        help="Destination for the figures. Defaults to docs/analysis/figures.",
    )
    fairness.add_argument(
        "--dry-run", action="store_true", help="Audit and validate, but write nothing."
    )
    fairness.add_argument(
        "--report",
        action="store_true",
        help="Print the full validation report, not only failures.",
    )

    # --- decide -----------------------------------------------------------
    decide = subparsers.add_parser(
        "decide",
        parents=[common],
        help="Turn calibrated predictions into a capacity-constrained inspection queue.",
    )
    decide.add_argument(
        "--features",
        type=Path,
        metavar="PATH",
        help=(
            "Component 4 feature table. Supplies the outcome label, the establishment "
            "identifier and the as-of history column coverage eligibility is defined on. "
            "Defaults to the most recent."
        ),
    )
    decide.add_argument(
        "--calibrated-predictions",
        type=Path,
        metavar="PATH",
        help="Component 9 artifact. The queue is ranked on its calibrated score.",
    )
    decide.add_argument(
        "--folds",
        type=Path,
        metavar="PATH",
        help=(
            "Component 5 fold table. Read to confirm the feature snapshot and the evaluation "
            "run describe the same periods, not to derive the folds."
        ),
    )
    decide.add_argument(
        "--simulation",
        type=Path,
        metavar="PATH",
        help="Component 5 simulation summary. Axis 1 of the model-selection rule.",
    )
    decide.add_argument(
        "--metrics",
        type=Path,
        metavar="PATH",
        help="Component 5 metric table. Axes 2 and 3 of the model-selection rule.",
    )
    decide.add_argument(
        "--sensitivity",
        type=Path,
        metavar="PATH",
        help=(
            "Component 5 NDE sensitivity bands. Decides which models are tied on axis 1, "
            "which is what stops noise choosing the production model."
        ),
    )
    decide.add_argument(
        "--categoricals",
        type=Path,
        metavar="PATH",
        help=(
            "Component 8 as-of categoricals, for the advisory geography label. Optional: "
            "without it the queue is identical and the group columns are blank."
        ),
    )
    decide.add_argument(
        "--fairness-support",
        type=Path,
        metavar="PATH",
        help=(
            "Component 12 support table, read as evidence for the advisory warnings. Never "
            "read into a score, a rank or an allocation."
        ),
    )
    decide.add_argument(
        "--overrides",
        type=Path,
        metavar="PATH",
        help=(
            "JSON override file. Applied beside the deterministic queue and logged in full; "
            "the recommendation artifact is written unchanged."
        ),
    )
    decide.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination for the policy tables and manifest.",
    )
    decide.add_argument(
        "--policies",
        action="append",
        metavar="NAME",
        help=(
            f"Policy to compare; repeatable. Defaults to all "
            f"{len(POLICY_GRID)}. The baseline pure_risk is always included, because every "
            "opportunity cost is measured against it."
        ),
    )
    decide.add_argument(
        "--model",
        metavar="NAME",
        help=(
            f"Override the model the selection rule chose. Admissible: "
            f"{', '.join(CANDIDATE_MODELS)}. Refused: {', '.join(REFUSED_MODELS)}."
        ),
    )
    decide.add_argument("--no-figures", action="store_true", help="Skip the policy figures.")
    decide.add_argument(
        "--figures-dir",
        type=Path,
        metavar="DIR",
        help="Destination for the figures. Defaults to docs/analysis/figures.",
    )
    decide.add_argument(
        "--dry-run", action="store_true", help="Decide and validate, but write nothing."
    )
    decide.add_argument(
        "--report", action="store_true", help="Print the full validation report, not only failures."
    )

    # --- schedule ---------------------------------------------------------
    schedule = subparsers.add_parser(
        "schedule",
        parents=[common],
        help="Turn the recommended queue into an operating plan over the observed calendar.",
    )
    schedule.add_argument(
        "--recommendations",
        type=Path,
        metavar="PATH",
        help=(
            "Component 13 recommendation table. The approved queue, its ranks and its "
            "provenance. Defaults to the most recent."
        ),
    )
    schedule.add_argument(
        "--folds",
        type=Path,
        metavar="PATH",
        help=(
            "Component 5 fold table. Supplies test_median_daily_capacity, from which every "
            "horizon length in this component descends."
        ),
    )
    schedule.add_argument(
        "--override-log",
        type=Path,
        metavar="PATH",
        help=(
            "Component 13 override log, read as provenance evidence only. It stamps an "
            "override id onto the schedule row and never changes a rank."
        ),
    )
    schedule.add_argument(
        "--adjustments",
        type=Path,
        metavar="PATH",
        help=(
            "JSON scheduling-adjustment file: a human changing when an approved row is "
            "worked. Audited in full; the deterministic plan is written unchanged beside it."
        ),
    )
    schedule.add_argument(
        "--execution",
        type=Path,
        metavar="PATH",
        help=(
            "JSON execution-event file: what the field reports happened. Drives re-planning "
            "of later days only, and never edits a recommendation or an earlier plan."
        ),
    )
    schedule.add_argument(
        "--capacity-mode",
        choices=[*[str(mode) for mode in CapacityMode], "both"],
        default="both",
        help=(
            "Which capacity mode to plan against. observed_calendar is measured and is the "
            "default; flat_median is an explicitly labelled scenario. Both are emitted by "
            "default so the scenario's divergence is always visible rather than opt-in."
        ),
    )
    schedule.add_argument(
        "--policies",
        action="append",
        metavar="NAME",
        help="Policy to schedule; repeatable. Defaults to every policy in the artifact.",
    )
    schedule.add_argument(
        "--k-names",
        action="append",
        metavar="NAME",
        help=(
            f"Capacity level to schedule; repeatable. Defaults to all "
            f"{len(SCHEDULE_K_LEVELS)}: {', '.join(SCHEDULE_K_LEVELS)}."
        ),
    )
    schedule.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination for the scheduling tables and manifest.",
    )
    schedule.add_argument("--no-figures", action="store_true", help="Skip the schedule figures.")
    schedule.add_argument(
        "--figures-dir",
        type=Path,
        metavar="DIR",
        help="Destination for the figures. Defaults to docs/analysis/figures.",
    )
    schedule.add_argument(
        "--dry-run", action="store_true", help="Plan and validate, but write nothing."
    )
    schedule.add_argument(
        "--report", action="store_true", help="Print the full validation report, not only failures."
    )

    # --- review -------------------------------------------------------------
    review = subparsers.add_parser(
        "review",
        parents=[common],
        help=(
            "Flag deterministic cases for human review from Component 13's queue and "
            "Component 14's schedule. Component 16 -- the deferral / human-review gate."
        ),
    )
    review.add_argument(
        "--recommendations",
        type=Path,
        metavar="PATH",
        help="Component 13 recommendation table. Required. Defaults to the most recent.",
    )
    review.add_argument(
        "--schedule",
        type=Path,
        metavar="PATH",
        help=(
            "Component 14 schedule table. Optional: without it the execution-gap trigger does "
            "not run and only the policy-warning trigger flags cases."
        ),
    )
    review.add_argument(
        "--execution",
        type=Path,
        metavar="PATH",
        help=(
            "Component 14 accumulated execution log. Optional: without it every occupying "
            "schedule row is treated as an execution gap."
        ),
    )
    review.add_argument(
        "--resolutions",
        type=Path,
        metavar="PATH",
        help=(
            "JSON review-resolution file: a human's decision about a flagged case. Audited in "
            "full beside the queue; the queue is written unchanged."
        ),
    )
    review.add_argument(
        "--policies",
        action="append",
        metavar="NAME",
        help="Policy to review; repeatable. Defaults to every policy in the artifact.",
    )
    review.add_argument(
        "--k-names",
        action="append",
        metavar="NAME",
        help="Capacity level to review; repeatable. Defaults to every level in the artifact.",
    )
    review.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Destination for the review tables and manifest.",
    )
    review.add_argument("--no-figures", action="store_true", help="Skip the review figures.")
    review.add_argument(
        "--figures-dir",
        type=Path,
        metavar="DIR",
        help="Destination for the figures. Defaults to docs/analysis/figures.",
    )
    review.add_argument(
        "--dry-run", action="store_true", help="Build the queue and validate, but write nothing."
    )
    review.add_argument(
        "--report", action="store_true", help="Print the full validation report, not only failures."
    )

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

    serve = subparsers.add_parser(
        "serve",
        parents=[common],
        help="Run the Sentinel API: a read/write HTTP boundary over Components 1-16's artifacts.",
    )
    serve.add_argument(
        "--host", type=str, default=None, metavar="HOST", help="Bind host (default: settings)."
    )
    serve.add_argument(
        "--port", type=int, default=None, metavar="PORT", help="Bind port (default: settings)."
    )
    serve.add_argument(
        "--reload", action="store_true", help="Autoreload on source change (development only)."
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


def _run_plan_candidates(args: argparse.Namespace, settings: Settings) -> int:
    parquet_path = args.parquet
    assignments_path = args.assignments
    establishments_path = args.establishments
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
        if establishments_path is None:
            establishments_path = duckdb_queries.latest_parquet(
                settings.entity_resolution_interim_dir,
                prefix="establishments_",
            )
            logger.info("Using most recent establishments: %s", establishments_path)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    try:
        result = build_candidates(
            settings,
            planning_date=args.planning_date,
            parquet_path=parquet_path,
            assignments_path=assignments_path,
            establishments_path=establishments_path,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, CandidateGenerationError, ValueError) as exc:
        logger.error("Candidate generation failed: %s", exc)
        return 1

    print(summarize_candidates(result))

    # result.checks merges Component 4's reused checks with this component's own two
    # (candidates_validate.py); both modules' has_failures() apply the identical rule
    # (any error-severity check failed), so either suffices over the combined list.
    failed = feature_validate.has_failures(result.checks)
    if args.report or failed:
        print(feature_validate.format_report(result.checks))
    return 1 if failed else 0


def _run_score_candidates(args: argparse.Namespace, settings: Settings) -> int:
    try:
        candidates_path = args.candidates or duckdb_queries.latest_parquet(
            settings.operational_candidates_processed_dir, prefix="operational_candidates_"
        )
        features_path = args.features or duckdb_queries.latest_parquet(
            settings.features_processed_dir, prefix="as_of_features_"
        )
        simulation_path = args.simulation or duckdb_queries.latest_parquet(
            settings.evaluation_processed_dir, prefix="simulation_summary_"
        )
        metrics_path = args.metrics or duckdb_queries.latest_parquet(
            settings.evaluation_processed_dir, prefix="evaluation_metrics_"
        )
        sensitivity_path = args.sensitivity or duckdb_queries.latest_parquet(
            settings.evaluation_processed_dir, prefix="sensitivity_"
        )
        calibrated_predictions_path = args.calibrated_predictions or duckdb_queries.latest_parquet(
            settings.predictions_processed_dir, prefix="calibrated_predictions_"
        )
        calibrator_parameters_path = args.calibrator_parameters or duckdb_queries.latest_parquet(
            settings.calibration_processed_dir, prefix="calibrator_parameters_"
        )
        calibrator_breakpoints_path = args.calibrator_breakpoints or duckdb_queries.latest_parquet(
            settings.calibration_processed_dir,
            prefix="calibrator_isotonic_breakpoints_",
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    try:
        result = build_operational_priorities(
            settings,
            candidates_path=candidates_path,
            historical_features_path=features_path,
            simulation_path=simulation_path,
            metrics_path=metrics_path,
            sensitivity_path=sensitivity_path,
            calibrated_predictions_path=calibrated_predictions_path,
            calibrator_parameters_path=calibrator_parameters_path,
            calibrator_isotonic_breakpoints_path=calibrator_breakpoints_path,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, OperationalPriorityBuildError, ValueError) as exc:
        logger.error("Operational scoring failed: %s", exc)
        return 1

    print(summarize_operational_priorities(result))

    failed = operational_scoring_validate.has_failures(result.checks)
    if args.report or failed:
        print(operational_scoring_validate.format_report(result.checks))
    return 1 if failed else 0


def _run_select_inspections(args: argparse.Namespace, settings: Settings) -> int:
    if args.capacity < 0:
        raise SystemExit(f"--capacity must be non-negative, got {args.capacity}")

    try:
        priority_path = args.priority_set or duckdb_queries.latest_parquet(
            settings.operational_scoring_processed_dir, prefix="operational_priority_"
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    try:
        result = build_operational_selection(
            settings,
            priority_path=priority_path,
            maximum_inspections=args.capacity,
            planning_date=args.planning_date,
            policy_id=args.policy,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, OperationalSelectionBuildError, ValueError) as exc:
        logger.error("Operational selection failed: %s", exc)
        return 1

    print(summarize_operational_selection(result))

    failed = operational_selection_validate.has_failures(result.checks)
    if args.report or failed:
        print(operational_selection_validate.format_report(result.checks))
    return 1 if failed else 0


def _run_organize_geography(args: argparse.Namespace, settings: Settings) -> int:
    try:
        selection_path = args.selection or duckdb_queries.latest_parquet(
            settings.operational_selection_processed_dir, prefix="operational_selection_"
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    try:
        result = build_geographic_plan(
            settings,
            selection_path=selection_path,
            threshold_km=args.threshold_km,
            threshold_preset=args.threshold_preset,
            organization_mode=OrganizationMode(args.organization_mode),
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, GeographicOrganizationBuildError, ValueError) as exc:
        logger.error("Geographic organization failed: %s", exc)
        return 1

    print(summarize_geographic_plan(result))

    failed = geographic_organization_validate.has_failures(result.checks)
    if args.report or failed:
        print(geographic_organization_validate.format_report(result.checks))
    return 1 if failed else 0


def _run_review_plan(args: argparse.Namespace, settings: Settings) -> int:
    try:
        plan_path = args.plan or duckdb_queries.latest_parquet(
            settings.geographic_organization_processed_dir, prefix="geographic_inspection_plan_"
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    try:
        result = build_plan_review(
            settings,
            plan_path=plan_path,
            decisions_path=args.decisions,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, PlanReviewBuildError, ValueError) as exc:
        logger.error("Plan review failed: %s", exc)
        return 1

    print(summarize_plan_review(result))

    failed = plan_review_validate.has_failures(result.checks)
    if args.report or failed:
        print(plan_review_validate.format_report(result.checks))
    return 1 if failed else 0


def _run_approve_plan(args: argparse.Namespace, settings: Settings) -> int:
    from datetime import UTC, datetime

    try:
        review_path = args.review or duckdb_queries.latest_parquet(
            settings.plan_review_processed_dir, prefix="supervisor_plan_review_"
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    approved_at = args.approved_at or datetime.now(UTC).isoformat()
    try:
        request = PlanApprovalRequest(
            approval_id=args.approval_id,
            planning_date=args.planning_date,
            approved_by=args.approved_by,
            approved_at=approved_at,
            note=args.note,
        )
    except ValueError as exc:
        logger.error("Approval request rejected: %s", exc)
        return 1

    try:
        result = build_approved_plan(
            settings,
            review_path=review_path,
            approval_request=request,
            decision_log_path=args.decision_log,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except (
        FileNotFoundError,
        PlanApprovalBuildError,
        PlanApprovalGovernanceError,
        ValueError,
    ) as exc:
        logger.error("Plan approval failed: %s", exc)
        return 1

    print(summarize_approval(result))
    print(format_readiness_report(result.checks))
    return 0


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
                settings,
                settings.predictions_processed_dir,
                "baseline_predictions_",
                "Component 6 predictions",
            ),
            "boosted_predictions": args.boosted_predictions
            or _latest(
                settings,
                settings.predictions_processed_dir,
                "boosted_predictions_",
                "Component 7 predictions",
            ),
            "neural_predictions": args.neural_predictions
            or _latest(
                settings,
                settings.predictions_processed_dir,
                "neural_predictions_",
                "Component 8 predictions",
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


def _run_explain(args: argparse.Namespace, settings: Settings) -> int:
    try:
        features_path = args.features or _latest(
            settings, settings.features_processed_dir, "as_of_features_", "features"
        )
        prediction_paths = {
            "baseline_predictions": args.baseline_predictions
            or _latest(
                settings,
                settings.predictions_processed_dir,
                "baseline_predictions_",
                "Component 6 predictions",
            ),
            "boosted_predictions": args.boosted_predictions
            or _latest(
                settings,
                settings.predictions_processed_dir,
                "boosted_predictions_",
                "Component 7 predictions",
            ),
            "neural_predictions": args.neural_predictions
            or _latest(
                settings,
                settings.predictions_processed_dir,
                "neural_predictions_",
                "Component 8 predictions",
            ),
        }
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    # Optional, and absent is a supported state rather than an error: a calibrated
    # probability is carried *alongside* an explanation and never explained by it, so a run
    # without Component 9's artifact is complete and simply leaves that column null.
    calibrated_path = args.calibrated_predictions
    if calibrated_path is None:
        try:
            calibrated_path = _latest(
                settings,
                settings.predictions_processed_dir,
                "calibrated_predictions_",
                "Component 9 predictions",
            )
        except FileNotFoundError:
            logger.info(
                "No calibrated prediction artifact found; explanations will carry a base "
                "score and a null calibrated probability."
            )

    if args.models is not None and not args.models:
        raise SystemExit("--models requires at least one model name")
    if args.sample_size < 1:
        raise SystemExit("--sample-size must be a positive integer")

    try:
        result = run_explanations(
            settings,
            features_path=features_path,
            prediction_paths=prediction_paths,
            calibrated_path=calibrated_path,
            output_dir=args.output_dir,
            models=args.models,
            sample_size=args.sample_size,
            figures_dir=args.figures_dir,
            write_figures=not args.no_figures,
            dry_run=args.dry_run,
        )
    except (
        FileNotFoundError,
        ExplainBuildError,
        RefitError,
        AttributionError,
        BackgroundError,
        SampleError,
        BaseScoreError,
        KeyError,
        ValueError,
    ) as exc:
        logger.error("Explanation failed: %s", exc)
        return 1

    print(summarize_explanations(result))

    failed = explain_validate.has_failures(result.checks)
    if args.report or failed:
        print(explain_validate.format_report(result.checks))
    # A failed check means either the explained model is not the one Components 6-8
    # published, or a value is attached to the wrong feature, the wrong establishment or the
    # wrong horizon. An explanation that is confidently about the wrong thing is worse than
    # no explanation, because a reader has no way to tell.
    return 1 if failed else 0


def _run_decide(args: argparse.Namespace, settings: Settings) -> int:
    """Build the recommended inspection queue, price every alternative policy, and report."""
    # Flag values are checked before any artifact is resolved. A malformed flag is wrong
    # whether or not the data happens to be on disk, and reporting a missing file first would
    # send a user looking for the wrong problem.
    if args.policies is not None and not args.policies:
        raise SystemExit("--policies requires at least one policy id")
    if args.model is not None and args.model not in CANDIDATE_MODELS:
        raise SystemExit(
            f"--model must be one of {', '.join(CANDIDATE_MODELS)}; "
            f"{', '.join(REFUSED_MODELS)} is refused"
        )

    try:
        features_path = args.features or _latest(
            settings, settings.features_processed_dir, "as_of_features_", "features"
        )
        calibrated_path = args.calibrated_predictions or _latest(
            settings,
            settings.predictions_processed_dir,
            "calibrated_predictions_",
            "Component 9 predictions",
        )
        folds_path = args.folds or _latest(
            settings, settings.evaluation_processed_dir, "evaluation_folds_", "evaluation folds"
        )
        simulation_path = args.simulation or _latest(
            settings,
            settings.evaluation_processed_dir,
            "simulation_summary_",
            "simulation summary",
        )
        metrics_path = args.metrics or _latest(
            settings, settings.evaluation_processed_dir, "evaluation_metrics_", "evaluation metrics"
        )
        sensitivity_path = args.sensitivity or _latest(
            settings, settings.evaluation_processed_dir, "sensitivity_", "NDE sensitivity bands"
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    # The two advisory inputs are optional by design: without them the queue is byte-identical
    # and only the warning columns go blank, which is the strongest possible statement that
    # Component 12's output never reaches an allocation decision.
    categoricals_path = args.categoricals
    if categoricals_path is None:
        try:
            categoricals_path = _latest(
                settings,
                settings.neural_processed_dir,
                "neural_categoricals_",
                "as-of categoricals",
            )
        except FileNotFoundError:
            logger.info("No as-of categoricals; the advisory geography columns will be blank")
    fairness_support_path = args.fairness_support
    if fairness_support_path is None:
        try:
            fairness_support_path = _latest(
                settings,
                settings.fairness_processed_dir,
                "fairness_group_support_",
                "Component 12 support",
            )
        except FileNotFoundError:
            logger.info("No Component 12 support table; group-support warnings are unavailable")

    try:
        result = run_policy(
            settings,
            features_path=features_path,
            calibrated_path=calibrated_path,
            folds_path=folds_path,
            simulation_path=simulation_path,
            metrics_path=metrics_path,
            sensitivity_path=sensitivity_path,
            categoricals_path=categoricals_path,
            fairness_support_path=fairness_support_path,
            overrides_path=args.overrides,
            output_dir=args.output_dir,
            policies=args.policies,
            model=args.model,
            figures_dir=args.figures_dir,
            write_figures=not args.no_figures,
            dry_run=args.dry_run,
        )
    except (
        FileNotFoundError,
        AllocationError,
        EligibilityError,
        GovernanceError,
        PolicyBuildError,
        PolicyDefinitionError,
        PolicyEvaluationError,
        PolicyInputError,
        SelectionError,
    ) as exc:
        logger.error("Decision policy failed: %s", exc)
        return 1

    print(summarize_policy(result))

    # An advisory finding never fails the run. The cheapest way to turn a red "this reserve
    # gave up 34 citations" build green is to delete the reserve, and that is a policy decision
    # about how a city allocates enforcement rather than a defect a CI runner may fix.
    failed = policy_validate.has_failures(result.checks)
    if args.report or failed:
        print(policy_validate.format_report(result.checks))
    return 1 if failed else 0


def _run_schedule(args: argparse.Namespace, settings: Settings) -> int:
    """Lay the approved queue against the observed calendar, measure it, and report.

    There is deliberately no flag here that raises capacity, extends a horizon, adds an
    operating day or introduces a probability threshold. Each would be a way to make a
    scheduling number better without scheduling anything better, and the absence is recorded in
    the manifest's blocked list so the boundary travels with the artifact.
    """
    if args.policies is not None and not args.policies:
        raise SystemExit("--policies requires at least one policy id")
    if args.k_names is not None and not args.k_names:
        raise SystemExit("--k-names requires at least one capacity level")
    unknown = [name for name in (args.k_names or []) if name not in SCHEDULE_K_LEVELS]
    if unknown:
        raise SystemExit(
            f"unknown capacity level(s) {', '.join(unknown)}; known: {', '.join(SCHEDULE_K_LEVELS)}"
        )

    if args.capacity_mode == "both":
        configs = list(SCHEDULE_CONFIG_GRID)
    else:
        configs = [
            spec for spec in SCHEDULE_CONFIG_GRID if str(spec.capacity_mode) == args.capacity_mode
        ]

    try:
        recommendations_path = args.recommendations or _latest(
            settings,
            settings.policy_processed_dir,
            "inspection_recommendations_",
            "Component 13 recommendations",
        )
        folds_path = args.folds or _latest(
            settings, settings.evaluation_processed_dir, "evaluation_folds_", "evaluation folds"
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    # Optional by design. Without the override log the plan is byte-identical and only the
    # provenance column goes blank, which is the strongest statement that a Component 13
    # override never reaches a scheduling decision.
    override_log_path = args.override_log
    if override_log_path is None:
        try:
            override_log_path = _latest(
                settings,
                settings.policy_processed_dir,
                "policy_override_log_",
                "Component 13 override log",
            )
        except FileNotFoundError:
            logger.info("No Component 13 override log; the override provenance column is blank")

    try:
        result = run_schedule(
            settings,
            recommendations_path=recommendations_path,
            folds_path=folds_path,
            override_log_path=override_log_path,
            adjustments_path=args.adjustments,
            execution_path=args.execution,
            configs=configs,
            policies=args.policies,
            k_names=args.k_names,
            output_dir=args.output_dir,
            figures_dir=args.figures_dir,
            no_figures=args.no_figures,
            dry_run=args.dry_run,
        )
    except (
        FileNotFoundError,
        AdjustmentError,
        ExecutionError,
        HorizonError,
        ReplanError,
        ScheduleAllocationError,
        ScheduleInputError,
        SchedulingDefinitionError,
    ) as exc:
        logger.error("Scheduling failed: %s", exc)
        return 1

    print(summarize_schedule(result))

    # An advisory finding never fails the run. The cheapest way to turn a red "this schedule
    # lost 1,012 coverage-reserve slots" build green is to make the scheduler prefer reserve
    # rows, which is re-ranking -- forbidden, and not a change a CI runner may make.
    failed = schedule_validate.has_failures(result.checks)
    if args.report or failed:
        print(schedule_validate.format_report(result.checks))
    return 1 if failed else 0


def _run_review(args: argparse.Namespace, settings: Settings) -> int:
    """Flag deterministic review cases from the current queue and schedule, and report.

    There is deliberately no flag here that introduces a probability or confidence threshold.
    Both triggers are boolean facts an upstream component already computed, and the absence of
    a threshold flag is recorded in the manifest's blocked list so the boundary travels with the
    artifact.
    """
    if args.policies is not None and not args.policies:
        raise SystemExit("--policies requires at least one policy id")
    if args.k_names is not None and not args.k_names:
        raise SystemExit("--k-names requires at least one capacity level")

    try:
        recommendations_path = args.recommendations or _latest(
            settings,
            settings.policy_processed_dir,
            "inspection_recommendations_",
            "Component 13 recommendations",
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    # Optional by design. Without the schedule the execution-gap trigger simply does not run,
    # which is the strongest possible statement that this component invents no state Component
    # 14 has not already produced.
    schedule_path = args.schedule
    if schedule_path is None:
        try:
            schedule_path = _latest(
                settings,
                settings.scheduling_processed_dir,
                "inspection_schedule_",
                "Component 14 schedule",
            )
        except FileNotFoundError:
            logger.info("No Component 14 schedule; the execution-gap trigger will not run")
    execution_path = args.execution
    if execution_path is None and schedule_path is not None:
        try:
            execution_path = _latest(
                settings, settings.scheduling_processed_dir, "execution_log_", "execution log"
            )
        except FileNotFoundError:
            logger.info(
                "No Component 14 execution log; every occupying schedule row is treated as a gap"
            )

    try:
        result = run_review(
            settings,
            recommendations_path=recommendations_path,
            schedule_path=schedule_path,
            execution_log_path=execution_path,
            resolutions_path=args.resolutions,
            output_dir=args.output_dir,
            policies=args.policies,
            k_names=args.k_names,
            figures_dir=args.figures_dir,
            write_figures=not args.no_figures,
            dry_run=args.dry_run,
        )
    except (
        FileNotFoundError,
        ReviewBuildError,
        ReviewDefinitionError,
        ReviewGovernanceError,
        ReviewInputError,
    ) as exc:
        logger.error("Component 16 failed: %s", exc)
        return 1

    print(summarize_review(result))

    failed = review_validate.has_failures(result.checks)
    if args.report or failed:
        print(review_validate.format_report(result.checks))
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


def _run_audit_fairness(args: argparse.Namespace, settings: Settings) -> int:
    # Flag values are checked before any artifact is resolved. A malformed flag is wrong
    # whether or not the data happens to be on disk, and reporting a missing file first
    # would send a user looking for the wrong problem.
    if args.models is not None and not args.models:
        raise SystemExit("--models requires at least one model name")
    if args.group_definitions is not None and not args.group_definitions:
        raise SystemExit("--group-definitions requires at least one definition name")

    try:
        features_path = args.features or _latest(
            settings, settings.features_processed_dir, "as_of_features_", "features"
        )
        calibrated_path = args.calibrated_predictions or _latest(
            settings,
            settings.predictions_processed_dir,
            "calibrated_predictions_",
            "Component 9 predictions",
        )
        categoricals_path = args.categoricals or _latest(
            settings,
            settings.neural_processed_dir,
            "neural_categoricals_",
            "Component 8 categoricals",
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    # Optional, and absent is a supported state rather than an error: the attribution profile
    # is one of ten tables, and a run without Component 11's artifact is a complete audit that
    # simply leaves that table empty.
    explanations_path = args.explanations
    if explanations_path is None:
        try:
            explanations_path = _latest(
                settings,
                settings.explanations_processed_dir,
                "explanation_values_",
                "Component 11 attributions",
            )
        except FileNotFoundError:
            logger.info(
                "No attribution artifact found; the per-group feature-reliance table will "
                "be empty and every other table is unaffected."
            )

    try:
        result = run_fairness_audit(
            settings,
            features_path=features_path,
            calibrated_path=calibrated_path,
            categoricals_path=categoricals_path,
            explanations_path=explanations_path,
            output_dir=args.output_dir,
            models=args.models,
            group_definitions=args.group_definitions,
            figures_dir=args.figures_dir,
            write_figures=not args.no_figures,
            dry_run=args.dry_run,
        )
    except (
        FileNotFoundError,
        FairnessBuildError,
        FairnessDefinitionError,
        GroupFrameError,
        GroupMetricError,
        DisparityError,
        DriftError,
        MissingnessError,
        PriorityError,
        FairnessAttributionError,
    ) as exc:
        logger.error("Fairness audit failed: %s", exc)
        return 1

    print(summarize_fairness(result))

    # An advisory finding never fails the run. ADR 0034: a red build is a demand for action,
    # and the actions available to whoever faces a red fairness check are to change the model,
    # the metric, or the threshold -- two of which are worse than the disparity.
    failed = fairness_validate.has_failures(result.checks)
    if args.report or failed:
        print(fairness_validate.format_report(result.checks))
    return 1 if failed else 0


def _run_serve(args: argparse.Namespace, settings: Settings) -> int:
    # Imported here, not at module level: every other command in this file runs with no web
    # framework installed, and importing FastAPI/uvicorn eagerly would make `sentinel decide`
    # fail in an environment that never asked to run the API. See pyproject.toml.
    import uvicorn

    from sentinel.api.app import create_app

    uvicorn.run(
        create_app(),
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        reload=args.reload,
    )
    return 0


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
    if args.command == "plan-candidates":
        return _run_plan_candidates(args, settings)
    if args.command == "score-candidates":
        return _run_score_candidates(args, settings)
    if args.command == "select-inspections":
        return _run_select_inspections(args, settings)
    if args.command == "organize-geography":
        return _run_organize_geography(args, settings)
    if args.command == "review-plan":
        return _run_review_plan(args, settings)
    if args.command == "approve-plan":
        return _run_approve_plan(args, settings)
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
    if args.command == "explain":
        return _run_explain(args, settings)
    if args.command == "audit-fairness":
        return _run_audit_fairness(args, settings)
    if args.command == "decide":
        return _run_decide(args, settings)
    if args.command == "schedule":
        return _run_schedule(args, settings)
    if args.command == "review":
        return _run_review(args, settings)
    if args.command == "evaluate":
        return _run_evaluate(args, settings)
    if args.command == "serve":
        return _run_serve(args, settings)

    # argparse enforces `required=True` on the subparser, so this is defensive
    # only. parser.error() exits with status 2.
    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
