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


def load_settings() -> Settings:
    """Load settings from environment / .env.

    Thin wrapper so callers depend on a function rather than on constructor
    behaviour, and so tests can inject a Settings instance directly.
    """
    return Settings()
