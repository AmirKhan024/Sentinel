"""Manifest helpers shared by every component that writes a data artifact.

Component 1 established the pattern: each Parquet file gets a JSON sidecar
recording where it came from, what is in it, and its checksum. Component 2
writes artifacts too, so the mechanics live here and the component-specific
*models* stay with their components — ``IngestionManifest`` in
``sentinel.ingest.manifest``, ``ResolutionManifest`` in
``sentinel.entity.models``.

The split is only about the mechanics. The reasoning for JSON sidecars is
unchanged and documented in ``sentinel.ingest.manifest``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Read in chunks so hashing a multi-hundred-MB file does not load it all into
# memory.
HASH_CHUNK_BYTES = 1024 * 1024


def compute_sha256(path: Path) -> str:
    """Checksum a file so "is this the file the manifest describes?" is answerable."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path_for(artifact_path: Path) -> Path:
    """Sidecar path for a given artifact.

    ``food_inspections_20260815T140000Z.parquet``
    -> ``manifest_food_inspections_20260815T140000Z.json``

    The ``manifest_`` prefix is what .gitignore whitelists, so manifests are
    committed while the Parquet files beside them are not.
    """
    return artifact_path.with_name(f"manifest_{artifact_path.stem}.json")


def write_manifest(manifest: BaseModel, path: Path) -> Path:
    """Write a manifest as indented, key-sorted JSON. Returns the path written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json", by_alias=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote manifest: %s", path)
    return path


def read_manifest_as[M: BaseModel](model_type: type[M], path: Path) -> M:
    """Load a manifest back, validating it against the given model."""
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))
