"""Ingestion manifest: the provenance record for one raw file.

Why a JSON sidecar rather than a database table or a metadata service:

* It is human-readable and greppable. Six months from now, "what exactly is in
  this Parquet file?" is answered by opening one small file.
* It diffs cleanly in Git, so provenance is version-controlled alongside the
  code that produced it, while the bulk data is not.
* It needs zero infrastructure. Introducing a metadata store here would be
  infrastructure ahead of a requirement.

The manifest is written next to the Parquet file it describes and is named
after it, so the pair can never drift apart on disk.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from sentinel.manifest import compute_sha256, manifest_path_for, read_manifest_as, write_manifest

logger = logging.getLogger(__name__)

# The generic mechanics moved to sentinel.manifest when Component 2 began
# writing artifacts of its own. They are re-exported here so callers and tests
# that predate the split keep working unchanged.
__all__ = [
    "IngestionManifest",
    "compute_sha256",
    "manifest_path_for",
    "read_manifest",
    "write_manifest",
]


class IngestionManifest(BaseModel):
    """Everything needed to reproduce and verify one raw ingestion."""

    # --- provenance ------------------------------------------------------
    source: str
    source_url: str
    dataset_id: str
    dataset_name: str
    retrieved_at: datetime
    code_version: str

    # --- request ----------------------------------------------------------
    mode: str  # "dev" | "full"
    row_limit: int | None
    page_size: int
    pages_fetched: int
    order_column: str
    request_params: list[dict[str, str]] = Field(default_factory=list)

    # --- result -----------------------------------------------------------
    row_count: int
    column_names: list[str]
    socrata_field_names: list[str] = Field(default_factory=list)
    socrata_field_types: list[str] = Field(default_factory=list)
    parquet_schema: dict[str, str]
    output_path: str
    output_bytes: int
    sha256: str


def read_manifest(path: Path) -> IngestionManifest:
    """Load an ingestion manifest back, validating it against the model."""
    return read_manifest_as(IngestionManifest, path)
