"""Tests for the ingestion manifest: completeness, checksum, round-trip."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import respx

from sentinel import __version__
from sentinel.config import Settings
from sentinel.ingest.food_inspections import ingest_food_inspections
from sentinel.ingest.manifest import (
    compute_sha256,
    manifest_path_for,
    read_manifest,
)
from sentinel.ingest.socrata import SocrataClient
from tests.conftest import (
    FIELD_NAMES,
    FIELD_TYPES,
    SODA_HEADERS,
    TEST_RESOURCE_URL,
    discovery_response,
    make_records,
)


def make_client() -> SocrataClient:
    return SocrataClient(
        resource_url=TEST_RESOURCE_URL,
        page_size=10,
        order_column="inspection_id",
        max_retries=0,
        sleep=lambda _s: None,
    )


def test_compute_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    payload = b"sentinel" * 1000
    path.write_bytes(payload)

    assert compute_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_manifest_path_is_sidecar_of_parquet() -> None:
    parquet = Path("/data/raw/food_inspections/food_inspections_20260815T120000Z.parquet")
    expected = "manifest_food_inspections_20260815T120000Z.json"

    result = manifest_path_for(parquet)

    assert result.name == expected
    assert result.parent == parquet.parent


@respx.mock
def test_manifest_records_full_provenance(settings: Settings) -> None:
    respx.get(TEST_RESOURCE_URL).mock(
        side_effect=[
            discovery_response(),
            httpx.Response(200, json=make_records(10, start=0), headers=SODA_HEADERS),
            httpx.Response(200, json=make_records(5, start=10), headers=SODA_HEADERS),
        ]
    )

    result = ingest_food_inspections(settings, row_limit=None, client=make_client())
    m = result.manifest

    assert m.source_url == settings.resource_url
    assert m.dataset_id == "4ijn-s7e5"
    assert m.dataset_name == "Food Inspections"
    assert m.code_version == __version__
    assert m.row_count == 15
    assert m.pages_fetched == 2
    assert m.page_size == settings.page_size
    assert m.order_column == "inspection_id"
    assert m.column_names == FIELD_NAMES
    assert m.socrata_field_names == FIELD_NAMES
    assert m.socrata_field_types == FIELD_TYPES
    assert m.output_path == str(result.parquet_path)
    assert m.output_bytes == result.parquet_path.stat().st_size
    assert m.retrieved_at.tzinfo is not None  # timezone-aware UTC


@respx.mock
def test_manifest_schema_reports_all_utf8(settings: Settings) -> None:
    respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=make_records(3), headers=SODA_HEADERS)
    )

    result = ingest_food_inspections(settings, row_limit=3, client=make_client())

    assert set(result.manifest.parquet_schema) == set(FIELD_NAMES)
    assert set(result.manifest.parquet_schema.values()) == {"String"}


@respx.mock
def test_manifest_checksum_matches_the_file_on_disk(settings: Settings) -> None:
    respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=make_records(4), headers=SODA_HEADERS)
    )

    result = ingest_food_inspections(settings, row_limit=4, client=make_client())

    assert result.manifest.sha256 == compute_sha256(result.parquet_path)


@respx.mock
def test_manifest_round_trips_through_json(settings: Settings) -> None:
    respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=make_records(2), headers=SODA_HEADERS)
    )

    result = ingest_food_inspections(settings, row_limit=2, client=make_client())
    reloaded = read_manifest(result.manifest_path)

    assert reloaded == result.manifest


@respx.mock
def test_manifest_file_is_readable_json(settings: Settings) -> None:
    respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=make_records(2), headers=SODA_HEADERS)
    )

    result = ingest_food_inspections(settings, row_limit=2, client=make_client())
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert payload["row_count"] == 2
    assert payload["dataset_id"] == "4ijn-s7e5"
    assert isinstance(payload["sha256"], str)
    assert len(payload["sha256"]) == 64


# --- generic helpers after the Component 2 refactor ------------------------
#
# compute_sha256/manifest_path_for/write_manifest moved to sentinel.manifest so
# entity resolution could write artifacts too. These assert the split did not
# change behaviour and that the typed reader works for a second model.


def test_generic_helpers_are_still_importable_from_ingest() -> None:
    """Existing callers and tests import these from the ingest module."""
    from sentinel.ingest import manifest as ingest_manifest

    assert ingest_manifest.compute_sha256 is not None
    assert ingest_manifest.manifest_path_for is not None
    assert ingest_manifest.write_manifest is not None


def test_generic_and_ingest_helpers_are_the_same_object() -> None:
    from sentinel import manifest as generic
    from sentinel.ingest import manifest as ingest_manifest

    assert ingest_manifest.compute_sha256 is generic.compute_sha256
    assert ingest_manifest.manifest_path_for is generic.manifest_path_for


def test_read_manifest_as_round_trips_a_resolution_manifest(tmp_path: Path) -> None:
    from sentinel.entity.models import ResolutionManifest
    from sentinel.manifest import read_manifest_as, write_manifest

    manifest = ResolutionManifest(
        code_version="0.1.0",
        normalization_version="1",
        resolved_at="2026-08-16T00:00:00+00:00",
        source_path="food_inspections_20260816T070911Z.parquet",
        source_sha256="7d3c4069",
        source_row_count=314245,
        thresholds={"max_zips_per_cluster": 1.0},
        node_count=51099,
        establishment_count=35859,
        singleton_establishment_count=26357,
        candidate_pair_count=335393,
        edges_by_tier={"strong": 29280},
        edges_by_rule={"S2": 21000},
        splits_by_reason={},
        oversized_block_count=0,
        blacklisted_coordinate_count=0,
        unusable_license_rows=850,
        unusable_address_rows=14,
        artifacts=[],
        checks=[],
    )
    path = tmp_path / "manifest_resolution.json"
    write_manifest(manifest, path)
    assert read_manifest_as(ResolutionManifest, path) == manifest
