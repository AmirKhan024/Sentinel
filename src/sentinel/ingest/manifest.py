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

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Read in chunks so hashing a multi-hundred-MB full pull does not load the
# whole file into memory.
_HASH_CHUNK_BYTES = 1024 * 1024


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


def compute_sha256(path: Path) -> str:
    """Checksum a file so "is this the file the manifest describes?" is answerable."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path_for(parquet_path: Path) -> Path:
    """Sidecar path for a given Parquet file.

    ``food_inspections_20260815T140000Z.parquet``
    -> ``manifest_food_inspections_20260815T140000Z.json``

    The ``manifest_`` prefix is what .gitignore whitelists, so manifests are
    committed while the Parquet files beside them are not.
    """
    return parquet_path.with_name(f"manifest_{parquet_path.stem}.json")


def write_manifest(manifest: IngestionManifest, path: Path) -> Path:
    """Write the manifest as indented JSON. Returns the path written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote manifest: %s", path)
    return path


def read_manifest(path: Path) -> IngestionManifest:
    """Load a manifest back, validating it against the model."""
    return IngestionManifest.model_validate_json(path.read_text(encoding="utf-8"))
