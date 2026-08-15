"""Shared fixtures.

The fake records mirror the real API's encoding exactly: every value is a JSON
string, and `location` is a nested object. Tests that assumed clean typed JSON
would pass while the real pipeline broke.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from sentinel.config import Settings

# The dataset's real field list, as reported by X-SODA2-Fields.
FIELD_NAMES = [
    "inspection_id",
    "dba_name",
    "aka_name",
    "license_",
    "facility_type",
    "risk",
    "address",
    "city",
    "state",
    "zip",
    "inspection_date",
    "inspection_type",
    "results",
    "violations",
    "latitude",
    "longitude",
    "location",
]

FIELD_TYPES = [
    "number",
    "text",
    "text",
    "number",
    "text",
    "text",
    "text",
    "text",
    "text",
    "number",
    "floating_timestamp",
    "text",
    "text",
    "text",
    "number",
    "number",
    "location",
]

TEST_RESOURCE_URL = "https://data.cityofchicago.org/resource/4ijn-s7e5.json"


def make_record(index: int) -> dict[str, Any]:
    """One synthetic record shaped exactly like a real Socrata row."""
    return {
        "inspection_id": str(100000 + index),
        "dba_name": f"TEST ESTABLISHMENT {index}",
        "aka_name": f"TEST ESTABLISHMENT {index}",
        "license_": str(2000000 + index),
        "facility_type": "Restaurant",
        "risk": "Risk 1 (High)",
        "address": f"{index} W TEST ST",
        "city": "CHICAGO",
        "state": "IL",
        "zip": "60601",
        "inspection_date": "2026-08-14T00:00:00.000",
        "inspection_type": "Canvass",
        "results": "Pass",
        "violations": "3. MANAGEMENT - Comments: none",
        "latitude": "41.8781",
        "longitude": "-87.6298",
        # Nested object, as the real API returns it.
        "location": {
            "latitude": "41.8781",
            "longitude": "-87.6298",
            "human_address": '{"address": "", "city": "", "state": "", "zip": ""}',
        },
    }


def make_records(count: int, *, start: int = 0) -> list[dict[str, Any]]:
    return [make_record(start + i) for i in range(count)]


SODA_HEADERS = {
    "X-SODA2-Fields": json.dumps(FIELD_NAMES),
    "X-SODA2-Types": json.dumps(FIELD_TYPES),
    "Content-Type": "application/json;charset=utf-8",
}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a temp data directory, with fast retries."""
    return Settings(
        data_dir=tmp_path / "data",
        page_size=10,
        dev_row_limit=25,
        max_retries=2,
        retry_backoff=0.0,
        request_timeout=5.0,
    )


@pytest.fixture
def no_sleep() -> object:
    """A sleep function that records delays instead of waiting."""

    class Recorder:
        def __init__(self) -> None:
            self.delays: list[float] = []

        def __call__(self, seconds: float) -> None:
            self.delays.append(seconds)

    return Recorder()


def discovery_response() -> httpx.Response:
    """The unordered single-row response used for field discovery.

    Ingestion issues this before paginating (see SocrataClient.discover_fields),
    so mocked side_effect sequences must account for it.
    """
    return httpx.Response(200, json=[make_record(0)], headers=SODA_HEADERS)
