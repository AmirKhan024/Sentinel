"""End-to-end resolution, the output contract, and determinism.

The determinism tests are the load-bearing ones. Component 3 onward will join on
``establishment_id``, so an id that moved between runs would silently invalidate
every downstream artifact.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.entity.models import ResolutionManifest
from sentinel.entity.resolve import resolve_establishments, summarize
from sentinel.entity.writer import ASSIGNMENTS_SCHEMA, EDGES_SCHEMA, ESTABLISHMENTS_SCHEMA
from sentinel.manifest import read_manifest_as
from tests.conftest import entity_scenario, make_entity_record

# A deliberate scenario with a known correct answer:
#   1-3  one premises, three licences, name and address variants   -> 1
#   4-5  a second premises at the same street, different number    -> 1
#   6-7  two different businesses sharing one address              -> 2
#   8-9  two numbered franchises at that address, distinct         -> 2
#   10   a row with the '0' licence sentinel                       -> 1
SCENARIO = [
    make_entity_record(
        1,
        inspection_id="1001",
        dba_name="ABC RESTAURANT",
        aka_name="ABC RESTAURANT",
        license_="10001",
        address="123 N MAIN ST",
        zip="60601",
    ),
    make_entity_record(
        2,
        inspection_id="1002",
        dba_name="ABC RESTAURANT LLC",
        aka_name="ABC RESTAURANT",
        license_="20002",
        address="123 N MAIN STREET",
        zip="60601",
    ),
    make_entity_record(
        3,
        inspection_id="1003",
        dba_name="Abc Restaurant",
        aka_name="ABC RESTAURANT",
        license_="30003",
        address="123 N MAIN ST  ",
        zip="60601",
    ),
    make_entity_record(
        4,
        inspection_id="1004",
        dba_name="DINER ONE",
        aka_name="DINER ONE",
        license_="40004",
        address="500 N MAIN ST",
        zip="60601",
        latitude="41.8800",
        longitude="-87.6300",
    ),
    make_entity_record(
        5,
        inspection_id="1005",
        dba_name="DINER ONE",
        aka_name="DINER ONE",
        license_="40004",
        address="500 N MAIN ST",
        zip="60601",
        latitude="41.8800",
        longitude="-87.6300",
    ),
    make_entity_record(
        6,
        inspection_id="1006",
        dba_name="KARMA MINI MART",
        aka_name="KARMA MINI MART",
        license_="60006",
        address="900 N MAIN ST",
        zip="60601",
        latitude="41.8900",
        longitude="-87.6400",
    ),
    make_entity_record(
        7,
        inspection_id="1007",
        dba_name="MB AND S MARKET",
        aka_name="MB AND S MARKET",
        license_="70007",
        address="900 N MAIN ST",
        zip="60601",
        latitude="41.8900",
        longitude="-87.6400",
    ),
    make_entity_record(
        8,
        inspection_id="1008",
        dba_name="SUBWAY 4321",
        aka_name="SUBWAY 4321",
        license_="80008",
        address="900 N MAIN ST",
        zip="60601",
        latitude="41.8900",
        longitude="-87.6400",
    ),
    make_entity_record(
        9,
        inspection_id="1009",
        dba_name="SUBWAY 9999",
        aka_name="SUBWAY 9999",
        license_="90009",
        address="900 N MAIN ST",
        zip="60601",
        latitude="41.8900",
        longitude="-87.6400",
    ),
    make_entity_record(
        10,
        inspection_id="1010",
        dba_name="ORPHAN CAFE",
        aka_name="ORPHAN CAFE",
        license_="0",
        address="1200 N MAIN ST",
        zip="60601",
        latitude="41.8950",
        longitude="-87.6450",
    ),
]

EXPECTED_ESTABLISHMENTS = 7


@pytest.fixture
def scenario_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "food_inspections_20260816T000000Z.parquet"
    entity_scenario(SCENARIO).write_parquet(path, compression="zstd")
    return path


def resolve(settings: Settings, path: Path, **kwargs: object):  # type: ignore[no-untyped-def]
    return resolve_establishments(settings, parquet_path=path, **kwargs)  # type: ignore[arg-type]


# --- the scenario resolves the way the design says it should -------------


def test_scenario_produces_the_expected_grouping(
    settings: Settings, scenario_parquet: Path
) -> None:
    result = resolve(settings, scenario_parquet, dry_run=True)
    mapping = dict(
        zip(
            result.assignments["inspection_id"],
            result.assignments["establishment_id"],
            strict=True,
        )
    )

    # Name and address variants across three licences are one establishment.
    assert mapping["1001"] == mapping["1002"] == mapping["1003"]
    # Repeat inspections of one place stay together.
    assert mapping["1004"] == mapping["1005"]
    # Unrelated neighbours at one address stay apart.
    assert mapping["1006"] != mapping["1007"]
    # A numbered franchise does not absorb its neighbours.
    assert mapping["1008"] not in {mapping["1006"], mapping["1007"]}
    # SUBWAY 4321 and SUBWAY 9999 are different franchise locations.
    assert mapping["1008"] != mapping["1009"]
    assert len(set(mapping.values())) == EXPECTED_ESTABLISHMENTS


def test_establishment_id_anchors_on_the_earliest_member(
    settings: Settings, scenario_parquet: Path
) -> None:
    result = resolve(settings, scenario_parquet, dry_run=True)
    mapping = dict(
        zip(
            result.assignments["inspection_id"],
            result.assignments["establishment_id"],
            strict=True,
        )
    )
    assert mapping["1001"] == "EST-00000001001"


def test_sentinel_licence_row_still_resolves(settings: Settings, scenario_parquet: Path) -> None:
    result = resolve(settings, scenario_parquet, dry_run=True)
    row = result.assignments.filter(pl.col("inspection_id") == "1010")
    assert row["establishment_id"][0] is not None
    assert row["license_key"][0] is None


def test_every_inspection_appears_exactly_once(settings: Settings, scenario_parquet: Path) -> None:
    result = resolve(settings, scenario_parquet, dry_run=True)
    assert result.assignments.height == len(SCENARIO)
    assert result.assignments["inspection_id"].n_unique() == len(SCENARIO)


def test_validation_passes_on_a_clean_scenario(settings: Settings, scenario_parquet: Path) -> None:
    result = resolve(settings, scenario_parquet, dry_run=True)
    failures = [c for c in result.checks if not c.passed and c.severity == "error"]
    assert failures == []


# --- determinism ---------------------------------------------------------


def test_two_runs_produce_identical_mappings(settings: Settings, scenario_parquet: Path) -> None:
    first = resolve(settings, scenario_parquet, dry_run=True).assignments
    second = resolve(settings, scenario_parquet, dry_run=True).assignments
    assert first.equals(second)


def test_shuffled_input_produces_an_identical_mapping(settings: Settings, tmp_path: Path) -> None:
    """The real determinism guarantee: row order must not reach the output."""
    ordered = tmp_path / "ordered.parquet"
    shuffled = tmp_path / "shuffled.parquet"
    entity_scenario(SCENARIO).write_parquet(ordered)

    rows = SCENARIO[:]
    random.Random(20260816).shuffle(rows)
    entity_scenario(rows).write_parquet(shuffled)

    a = resolve(settings, ordered, dry_run=True).assignments.sort("inspection_id")
    b = resolve(settings, shuffled, dry_run=True).assignments.sort("inspection_id")
    assert a.equals(b)


def test_node_ids_are_stable_across_runs(settings: Settings, scenario_parquet: Path) -> None:
    first = resolve(settings, scenario_parquet, dry_run=True).assignments
    second = resolve(settings, scenario_parquet, dry_run=True).assignments
    assert list(first["node_id"]) == list(second["node_id"])


def test_content_hashes_are_stable_across_runs(settings: Settings, scenario_parquet: Path) -> None:
    first = resolve(settings, scenario_parquet, dry_run=True).establishments
    second = resolve(settings, scenario_parquet, dry_run=True).establishments
    assert list(first["cluster_content_sha256"]) == list(second["cluster_content_sha256"])


# --- output contract -----------------------------------------------------


def test_assignments_schema_matches_the_contract(
    settings: Settings, scenario_parquet: Path
) -> None:
    frame = resolve(settings, scenario_parquet, dry_run=True).assignments
    assert list(frame.columns) == list(ASSIGNMENTS_SCHEMA)
    assert [str(d) for d in frame.dtypes] == [str(d) for d in ASSIGNMENTS_SCHEMA.values()]


def test_establishments_schema_matches_the_contract(
    settings: Settings, scenario_parquet: Path
) -> None:
    frame = resolve(settings, scenario_parquet, dry_run=True).establishments
    assert list(frame.columns) == list(ESTABLISHMENTS_SCHEMA)
    assert [str(d) for d in frame.dtypes] == [str(d) for d in ESTABLISHMENTS_SCHEMA.values()]


def test_edges_schema_matches_the_contract(settings: Settings, scenario_parquet: Path) -> None:
    frame = resolve(settings, scenario_parquet, dry_run=True).edges
    assert list(frame.columns) == list(EDGES_SCHEMA)
    assert [str(d) for d in frame.dtypes] == [str(d) for d in EDGES_SCHEMA.values()]


def test_assignments_carry_no_dates_counts_or_outcomes(
    settings: Settings, scenario_parquet: Path
) -> None:
    """Anti-leakage hygiene (findings §14), asserted on the actual output."""
    columns = set(resolve(settings, scenario_parquet, dry_run=True).assignments.columns)
    forbidden = {"inspection_date", "results", "violations", "risk", "n_inspections"}
    assert columns & forbidden == set()


def test_identifiers_stay_utf8(settings: Settings, scenario_parquet: Path) -> None:
    frame = resolve(settings, scenario_parquet, dry_run=True).assignments
    for column in ("inspection_id", "establishment_id", "license_key"):
        assert frame.schema[column] == pl.Utf8


# --- writing -------------------------------------------------------------


def test_writes_three_tables_and_a_manifest(
    settings: Settings, scenario_parquet: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    result = resolve(settings, scenario_parquet, output_dir=out)
    for path in (
        result.assignments_path,
        result.establishments_path,
        result.edges_path,
        result.manifest_path,
    ):
        assert path is not None
        assert path.exists()


def test_manifest_round_trips_and_pins_the_source(
    settings: Settings, scenario_parquet: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    result = resolve(settings, scenario_parquet, output_dir=out)
    assert result.manifest_path is not None
    loaded = read_manifest_as(ResolutionManifest, result.manifest_path)
    assert loaded.source_path == scenario_parquet.name
    assert loaded.source_row_count == len(SCENARIO)
    assert loaded.establishment_count == EXPECTED_ESTABLISHMENTS
    assert len(loaded.artifacts) == 3


def test_manifest_is_readable_json(
    settings: Settings, scenario_parquet: Path, tmp_path: Path
) -> None:
    result = resolve(settings, scenario_parquet, output_dir=tmp_path / "out")
    assert result.manifest_path is not None
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["component"] == "entity_resolution"
    assert payload["normalization_version"]


def test_written_tables_reload_identically(
    settings: Settings, scenario_parquet: Path, tmp_path: Path
) -> None:
    result = resolve(settings, scenario_parquet, output_dir=tmp_path / "out")
    assert result.assignments_path is not None
    assert pl.read_parquet(result.assignments_path).equals(result.assignments)


def test_dry_run_writes_nothing(settings: Settings, scenario_parquet: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = resolve(settings, scenario_parquet, output_dir=out, dry_run=True)
    assert result.assignments_path is None
    assert result.manifest_path is None
    assert not out.exists()


def test_missing_source_raises(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve(settings, tmp_path / "absent.parquet", dry_run=True)


def test_empty_input_produces_empty_outputs(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "empty.parquet"
    entity_scenario([]).write_parquet(path)
    result = resolve(settings, path, dry_run=True)
    assert result.assignments.height == 0
    assert result.establishments.height == 0
    assert list(result.assignments.columns) == list(ASSIGNMENTS_SCHEMA)


def test_all_licences_unusable_still_resolves(settings: Settings, tmp_path: Path) -> None:
    rows = [
        make_entity_record(
            1,
            inspection_id="1",
            license_="0",
            dba_name="ONLY PLACE",
            aka_name="ONLY PLACE",
            address="1 N MAIN ST",
        ),
        make_entity_record(
            2,
            inspection_id="2",
            license_="0",
            dba_name="ONLY PLACE",
            aka_name="ONLY PLACE",
            address="1 N MAIN ST",
        ),
    ]
    path = tmp_path / "nolicence.parquet"
    entity_scenario(rows).write_parquet(path)
    result = resolve(settings, path, dry_run=True)
    assert result.establishments.height == 1


def test_summarize_reports_the_headline_counts(settings: Settings, scenario_parquet: Path) -> None:
    text = summarize(resolve(settings, scenario_parquet, dry_run=True))
    assert f"establishments:   {EXPECTED_ESTABLISHMENTS}" in text
    assert "source rows:      10" in text


# --- audit trail ---------------------------------------------------------


def test_edges_explain_why_two_nodes_merged(settings: Settings, scenario_parquet: Path) -> None:
    """'Why was this inspection assigned to this establishment?' answered from
    the output alone."""
    result = resolve(settings, scenario_parquet, dry_run=True)
    merged = result.edges.filter(pl.col("tier") == "strong")
    assert merged.height >= 1
    assert set(merged["rule_id"]) <= {"S1", "S2", "S3"}


def test_edges_record_declined_merges(settings: Settings, scenario_parquet: Path) -> None:
    """'Why were these two NOT merged?' is one filter away."""
    result = resolve(settings, scenario_parquet, dry_run=True)
    vetoed = result.edges.filter(pl.col("rule_id").str.starts_with("V"))
    assert vetoed.height >= 1
