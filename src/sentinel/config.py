"""Configuration for Sentinel.

Principle: nothing about the data source is hardcoded at a call site. Dataset
ID, endpoint, paths, page size, timeouts and retry budget all live here and are
overridable via environment variables (prefix ``SENTINEL_``) or a ``.env`` file.

Defaults match the live API behaviour verified in docs/api/socrata_findings.md.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root, derived from this file's location: src/sentinel/config.py -> ../../
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration, populated from environment / .env with defaults."""

    model_config = SettingsConfigDict(
        env_prefix="SENTINEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Socrata / Chicago Data Portal ---------------------------------
    socrata_domain: str = "data.cityofchicago.org"
    dataset_id: str = "4ijn-s7e5"
    dataset_name: str = "Food Inspections"

    # Optional. Only relieves anonymous throttling; grants no extra data.
    socrata_app_token: str | None = None

    # --- Paths ----------------------------------------------------------
    data_dir: Path = Field(default=REPO_ROOT / "data")

    # --- Ingestion behaviour --------------------------------------------
    page_size: int = Field(default=50_000, gt=0)
    dev_row_limit: int = Field(default=5_000, gt=0)
    request_timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_backoff: float = Field(default=1.0, ge=0)

    # Column used to impose a total order on the dataset before paginating.
    # Socrata offset pagination is only stable under an explicit $order; see
    # docs/api/socrata_findings.md.
    order_column: str = "inspection_id"

    # This endpoint drops its `:@computed_region_*` columns (Socrata-generated
    # ward / community area / census tract / zip spatial joins) whenever
    # $order is present, unless they are explicitly selected. When true, the
    # client discovers the full field list first and selects it, costing one
    # extra request and keeping the raw layer complete.
    include_computed_regions: bool = True

    # --- Logging ---------------------------------------------------------
    log_level: str = "INFO"

    # --- The Sentinel API -------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, gt=0)

    #: Hard ceiling on any list endpoint's page size. `inspection_recommendations` alone is
    #: ~1.45M rows; a caller-supplied `limit` above this is silently capped, never rejected,
    #: because the alternative -- an unbounded scan -- is the failure mode this exists to close.
    api_max_page_size: int = Field(default=500, gt=0)
    api_default_page_size: int = Field(default=50, gt=0)

    #: Browser origins allowed to call this API cross-origin (CORS). Exists for the local
    #: product-testing frontend under `frontend/`; the API itself has no notion of a frontend.
    api_cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    @property
    def resource_url(self) -> str:
        """Full SODA 2.1 resource URL for the configured dataset."""
        return f"https://{self.socrata_domain}/resource/{self.dataset_id}.json"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_processed_dir(self) -> Path:
        """Where Component 4 writes the as-of feature table.

        Processed rather than interim: ADR 0005 reserves the processed layer for
        analysis- and model-ready outputs, and this is the first table that
        qualifies -- features joined to labels, one row per prediction
        opportunity, directly trainable. See ADR 0011.
        """
        return self.processed_dir / "features"

    @property
    def evaluation_processed_dir(self) -> Path:
        """Where Component 5 writes fold definitions, metrics and curves.

        Processed rather than interim, but beside ``features/`` rather than
        inside it: these are evaluation *results*, not a model-ready table, and
        nothing downstream trains on them. See ADR 0013.
        """
        return self.processed_dir / "evaluation"

    @property
    def predictions_processed_dir(self) -> Path:
        """Where Component 6 writes model scores, coefficients and training logs.

        A third kind of processed artifact, and a sibling of the other two rather
        than a child of either. Predictions are model *outputs*: they fail ADR
        0011's model-ready test because they carry no features, and they fail ADR
        0013's evaluation test because they are produced *before* scoring and are
        the evaluator's input rather than its result.

        **Nothing here may be joined onto a feature table.** Co-location with
        ``features/`` is exactly the invitation to join, and a score joined onto a
        training table is the most damaging leakage available. See ADR 0014.
        """
        return self.processed_dir / "predictions"

    @property
    def tuning_processed_dir(self) -> Path:
        """Where Component 7 writes hyperparameter search trials.

        A fourth kind of processed artifact. A trials table is neither a
        model-ready table (ADR 0011), nor a model output (ADR 0014), nor a
        measurement of a model's performance (ADR 0013): it records what a
        search tried and what each attempt scored on a *validation* window that
        is training data for every fold the chosen parameters will be used on.

        **Nothing here may be joined onto a feature table, and no number in it
        is a result.** A validation PR-AUC read as a headline metric would be an
        in-sample number reported as an out-of-sample one. See ADR 0018.
        """
        return self.processed_dir / "tuning"

    @property
    def neural_processed_dir(self) -> Path:
        """Where Component 8 writes its experimental categorical table.

        A fifth kind of processed artifact, and the only one in the project that
        is a model *input* without being a Component 4 feature table.

        Component 4's contract has no categorical column. The four families
        Component 8 embeds -- chain, facility type, community area and zip --
        are carried as-of from the raw snapshot and from Component 2's entity
        resolution by Component 8 itself, and they are deliberately not promoted
        into ``features/``. Putting them there would make them features, and
        adding a feature belongs to Component 4 behind a bumped
        ``feature_definition_version``.

        **Nothing here is a feature and nothing here may be joined onto a
        feature table by any other component.** It exists so that one
        experiment's inputs are visible, auditable and separable from the
        production feature set. See ADR 0022.
        """
        return self.processed_dir / "neural"

    @property
    def calibration_processed_dir(self) -> Path:
        """Where Component 9 writes its calibrators and their diagnostics.

        A sixth kind of processed artifact. The grain is a *fitted correction*
        or a measurement of one: the calibration-window base scores, the fitted
        Platt parameters, the isotonic breakpoints, the per-quarter drift, the
        Brier decomposition, the ranking-preservation deltas and the bootstrap
        intervals.

        Not ``evaluation/``, and that separation is load-bearing. Component 9's
        drift table carries an ``ece`` per (model, fold) measured on the test
        window, and so does ``evaluation_metrics_*.parquet``. Filed together
        there would be two authoritative ECEs for the same cell, with no
        convention saying which is the project's answer. Component 5 remains the
        only producer of the headline metrics; these are the diagnostics of a
        correction.

        The calibrated predictions themselves are **not** here -- they are in
        ``predictions/`` under their own slug, where ADR 0014 said to put them,
        which is what lets ``sentinel evaluate --predictions`` read them with no
        change to Component 5. The selection log is in ``tuning/``, where ADR
        0018 said to put it.

        **Nothing here may be joined onto a feature table, and the
        calibration-window scores in particular must never reach a fit.** Those
        rows sit after ``train_end``; a base model that saw them would have been
        fitted past its own declared horizon. See ADR 0024.
        """
        return self.processed_dir / "calibration"

    @property
    def explanations_processed_dir(self) -> Path:
        """Where Component 11 writes feature attributions and their analysis.

        A seventh kind of processed artifact. The grain is an *attribution*: one model's
        contribution of one feature to one prediction, plus the summaries built from those
        contributions -- global importance, rank stability, explanation drift and the
        representative local cases.

        Not ``predictions/``, and the separation is the same one ADR 0024 drew for
        Component 9. An attribution is not a model output: it does not score anything, no
        scheduler could rank on it, and ``evaluate --predictions`` would refuse it. It
        explains an output that already exists elsewhere.

        Not ``evaluation/`` either, and that separation matters more. Component 5 owns the
        question "is this model any good", and an attribution answers a different question
        entirely -- "what did this model lean on" -- which carries no notion of correct.
        Filed beside the metrics, a large ``mean_abs_shap`` would eventually be read as
        evidence of quality. It is evidence of *reliance*, and a model can lean hard on a
        feature that is misleading it.

        **Nothing here may be joined onto a feature table, and no number here is a result.**
        A SHAP value describes how a model used a feature; it does not measure the
        feature's effect on food safety, and a per-establishment attribution joined back
        onto training rows would be a model's own output re-entering it as an input. See
        ADR 0028.
        """
        return self.processed_dir / "explanations"

    @property
    def fairness_processed_dir(self) -> Path:
        """Where Component 12 writes the group-behaviour audit.

        An eighth kind of processed artifact. The grain is a *group-conditional
        measurement*: one metric, for one model, at one prediction stage, restricted to one
        group of one group definition, at one grain -- together with the support counts that
        say whether the number means anything at all.

        Not ``evaluation/``, and this is the same collision ADR 0024 and ADR 0028 each had
        to avoid. Component 5 emits a ``roc_auc`` per (model, fold); this component emits a
        ``roc_auc`` per (model, fold, group definition, group, grain, stage). Filed in one
        directory there would be two authoritative answers for the same cell with no
        convention saying which is which. Component 5 stays the only producer of the
        headline metrics, and no row written here is un-conditioned.

        Not ``predictions/`` either: nothing here scores anything, and
        ``evaluate --predictions`` would refuse every table in it.

        **Nothing here may be joined onto a feature table, and no number here is a
        verdict.** A group metric describes how a model behaved on a subset of held-out
        rows. It does not establish discrimination, causality, legal compliance or the
        absence of bias. And a per-group number joined back onto training rows would make a
        model's measured behaviour on a neighbourhood an input to how it treats that
        neighbourhood next time -- a self-fulfilling feature, and the sharpest leak
        available here because these tables are keyed by group rather than by row. See
        ADR 0032 and ADR 0035.
        """
        return self.processed_dir / "fairness"

    @property
    def policy_processed_dir(self) -> Path:
        """Where Component 13 writes the decision policy's output.

        A ninth kind of processed artifact. The grain is a *decision*: one establishment, one
        operating period, one capacity assumption, one policy -- together with the mechanism
        that put it in the queue or kept it out, and the reason code for that mechanism.

        Not ``predictions/``, and this is the collision every layer since ADR 0024 has had to
        avoid. A prediction says an establishment has a 0.62 chance of being cited. A
        recommendation says it is the fourth inspection on Tuesday. The first is a belief about
        the world and the second is an instruction to a person, they change for entirely
        different reasons, and filed together there would be no convention saying which is
        which. Components 6 to 9 stay the only producers of scores, and nothing written here is
        a score.

        Not ``evaluation/`` either: Component 5 emits a ``precision_at_k`` per (model, fold),
        and this component emits one per (policy, model, fold, capacity). Two authoritative
        answers to the same cell, in one directory, is how a project starts quoting the
        flattering one.

        **Nothing here may be joined onto a feature table, and no row here is a prediction.** A
        recommendation is downstream of every model in this project; joined back onto training
        rows it would make the system's own past decisions an input to its future ones, which
        is the feedback loop Component 12 measured and this component exists to keep visible
        rather than to close. See ADR 0036.
        """
        return self.processed_dir / "policy"

    @property
    def scheduling_processed_dir(self) -> Path:
        """Where Component 14 writes the operational schedule.

        A tenth kind of processed artifact. The grain is a *slot*: one recommended inspection,
        one planning run, one operating period, one capacity mode -- together with the reason
        that put it in that period, or left it in the backlog.

        Not ``policy/``, and the distinction is the component's whole point. Component 13 says
        *who* should be inspected at a given capacity and why. Component 14 says *when* the
        approved queue is executed. Those two facts change for different reasons -- a policy
        changes when a department changes its mind about coverage, a schedule changes when a
        Tuesday turns out to hold sixteen inspections instead of twenty-eight -- and filed
        together there would be no convention saying which is which.

        Not ``evaluation/`` either: Component 5 owns what a fold and a capacity cutoff are, and
        this layer consumes both without redefining either.

        **Nothing here may be joined onto a feature table.** A schedule is downstream of every
        model, every policy and every human decision in this project; joined back onto training
        rows it would close the feedback loop Component 12 measured, one layer further out than
        Component 13 already refused to close it. See ADR 0041.
        """
        return self.processed_dir / "scheduling"

    @property
    def review_processed_dir(self) -> Path:
        """Where Component 16 writes the human-review queue.

        An eleventh kind of processed artifact. The grain is a *case*: one recommended or
        scheduled row, flagged for review by a named deterministic trigger, together with
        whatever resolution a human has since recorded for it.

        Not ``policy/`` and not ``scheduling/``: a review case names a reason a human should
        look at a row that those two components already decided about. It never re-ranks the
        row, never re-dates it, and never carries a probability threshold -- there is no flag
        to add one. See ADR 0051.
        """
        return self.processed_dir / "review"

    @property
    def operational_candidates_processed_dir(self) -> Path:
        """Where Component 17 writes the operational candidate/feature table.

        A twelfth kind of processed artifact, and the first whose grain is a *planning
        date* rather than a fold. The candidate table carries the identical feature
        contract Component 4 produces -- same ``FEATURE_SPECS``, same temporal boundary,
        same ``feature_definition_version`` -- so a model fitted on ``features/`` can
        score a row from here with no change. It differs only in what supplies the as-of
        reference date: Component 4 reads it from a real historical inspection that
        already happened; this component reads it from a planning date a supervisor
        chose, and there is no future inspection row behind it.

        Not ``features/`` itself. ``features/``'s grain is one row per real historical
        prediction opportunity, byte-for-byte reproducible from committed history. A row
        here is one row per real establishment that existed as of a chosen planning date,
        and a different planning date choice changes which rows exist -- a property
        ``features/`` deliberately does not have. Filed together there would be no way to
        tell a historical training row from a hypothetical planning-date row without
        reading every row's ``target_status``.

        **Nothing here may be joined onto ``features/``, and nothing here carries a real
        label.** ``target`` is always NULL and ``target_status`` always reads
        ``operational_candidate`` -- nothing has happened yet, by construction.
        """
        return self.processed_dir / "operational_candidates"

    @property
    def operational_scoring_processed_dir(self) -> Path:
        """Where Component 18 writes the scored, ranked `OperationalPrioritySet`.

        A thirteenth kind of processed artifact. The grain is a *scored candidate*: one
        Component 17 candidate, one planning date, the model that scored it, its
        calibrated score and its deterministic rank -- together with the reused
        Component 13 coverage-eligibility classification.

        Not `operational_candidates/`: that table carries no score, because Component 17
        deliberately stops before scoring. Not `predictions/`: Component 6/7/8's scores
        there are keyed by a real evaluation fold, and this table's rows are keyed by a
        planning date with no fold behind it at all -- filed together there would be no
        way to tell an evaluation score from an operational one without reading every
        row's `model_name`.

        **Nothing here may be joined onto `features/` or `predictions/`, and no fitted
        model is written here.** The re-executed base model is refit in memory for the
        run and never serialized; only its scores, and the frozen calibrator parameters
        it was combined with, are persisted, exactly as Component 9 already persists a
        calibrator.
        """
        return self.processed_dir / "operational_scoring"

    @property
    def operational_selection_processed_dir(self) -> Path:
        """Where Component 19 writes the capacity-constrained `OperationalSelectionSet`.

        A fourteenth kind of processed artifact. The grain is a *selection decision*: one
        Component 18 scored candidate, one planning run, one requested capacity, one
        policy -- together with the reused Component 13 mechanism/reason that put it in
        the plan or kept it out. Every Component 18 row is present, selected or not: this
        directory holds the bounded plan, not a replacement for the unbounded ranking in
        ``operational_scoring/``.

        Not `operational_scoring/`: a priority set has no notion of capacity, and a
        selection set with no capacity concept would not be one. Not `policy/`: that
        directory is Component 13's own historical-fold recommendation queue, and filing
        an operational selection there would let a reader mistake a live planning run for
        a backtest cell.

        **Nothing here may be joined onto `features/` or `predictions/`, and no
        establishment is ever added that Component 18 did not already rank.**
        """
        return self.processed_dir / "operational_selection"

    @property
    def geographic_organization_processed_dir(self) -> Path:
        """Where Component 20 writes the `GeographicInspectionPlan`.

        A fifteenth kind of processed artifact. The grain is a *selected*
        establishment (Component 19's `is_selected == True` rows only) annotated with
        a geographic proximity group -- never the full ranked/priority queue, which
        `operational_scoring/` and `operational_selection/` already preserve.

        Not `operational_selection/`: a selection set carries no geography beyond raw
        as-of coordinates it never reads; this directory's rows are the same
        establishments with a `geographic_group_id` decided by real coordinates. Not
        `predictions/`, `policy/`, or `scheduling/`: nothing here is a score, a
        capacity decision, or a calendar placement -- geography is strictly
        downstream of all three.

        **Nothing here may change a `calibrated_score`, a `rank`, a `policy_rank`, a
        `selection_reason`, or the selected-establishment-id set.** Component 20 adds
        columns; it never rewrites one it did not add, and the set of establishments
        organized here must equal the set Component 19 selected, exactly.
        """
        return self.processed_dir / "geographic_organization"

    @property
    def plan_review_processed_dir(self) -> Path:
        """Where Component 21 writes the supervisor plan review and its decision log.

        Reads only Component 20's `GeographicInspectionPlan` -- never Component 19's or
        Component 18's directly, so a supervisor decision can never bypass geographic
        organization or capacity/policy selection. A sixteenth kind of processed artifact.

        **Nothing here may change a `calibrated_score`, a `rank`, a `policy_rank`, a
        `selection_reason`, a `geographic_group_id`, or a `work_block_id`.** Component 21
        adds decision columns; it never rewrites one it did not add, and it never creates
        a Component 13 override or a Component 14 adjustment itself.
        """
        return self.processed_dir / "plan_review"

    @property
    def target_interim_dir(self) -> Path:
        """Where Component 3 writes the prediction target.

        Interim rather than processed: ADR 0005 reserves the processed layer for
        model-ready tables, and this is labels only -- Component 4 must still
        join as-of features onto it before anything can be trained.
        """
        return self.interim_dir / "target"

    @property
    def entity_resolution_interim_dir(self) -> Path:
        """Where Component 2 writes its outputs.

        Interim rather than processed: ADR 0005 reserves the processed layer for
        analysis- and model-ready tables, and an establishment crosswalk is a
        mid-pipeline key mapping that Component 3 consumes.
        """
        return self.interim_dir / "entity_resolution"

    @property
    def food_inspections_raw_dir(self) -> Path:
        """Where raw food-inspection Parquet files and their manifests land."""
        return self.raw_dir / "food_inspections"

    @property
    def staging_dir(self) -> Path:
        """Where the Sentinel API appends staged human-input requests.

        Not a processed layer and not interim: nothing here was produced by a batch component,
        and nothing here has been applied to any artifact. It is user-generated, append-only,
        and read by no pipeline component -- only by an operator who chooses to feed a staged
        file into `sentinel decide` / `sentinel schedule` by hand. See ADR 0049.
        """
        return self.data_dir / "staging"


def load_settings() -> Settings:
    """Load settings from environment / .env.

    Thin wrapper so callers depend on a function rather than on constructor
    behaviour, and so tests can inject a Settings instance directly.
    """
    return Settings()
