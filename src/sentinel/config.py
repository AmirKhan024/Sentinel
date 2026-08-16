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
