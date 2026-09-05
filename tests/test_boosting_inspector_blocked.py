"""Inspector-effect modelling is BLOCKED because the field does not exist. Proven, not asserted.

The published audit of Chicago's 2015 model identified inspector identity as a serious
confound: an observed violation depends partly on establishment risk and partly on which
inspector attended, and an establishment does not choose its inspector. The project
specification therefore calls for inspector strictness to be modelled as a nuisance
effect -- a mixed-effects logistic regression with establishment features as fixed
effects and inspector as a random intercept -- and for tree-model predictions to be
evaluated at a marginalised inspector effect.

**None of that is implementable here, because the Chicago Food Inspections dataset
(`4ijn-s7e5`) publishes no inspector identifier.** Not a name, not a badge number, not a
pseudonymous id. The 22 raw columns are listed in
``docs/data_contracts/food_inspections_raw.md`` and none of them identifies a person.

A random intercept over an unobserved grouping is not a model; it is a fabrication. A
marginalisation over an effect that was never estimated is arithmetic on a number nobody
measured. So Component 7 ships the blocked record instead, and this file is the machine
-checkable half of it.

Two things are tested:

1. **The absence is real**, re-derived from the raw contract and every downstream schema
   rather than taken from prose. Existing documents already say inspector data are "not
   ingested"; this checks the data rather than the sentence.
2. **The absence would be noticed if it changed.** If Chicago ever publishes an inspector
   column, or a future component ingests one, these tests fail and someone has to decide
   what to do -- rather than the blocked note quietly outliving its reason.

See ADR 0019.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from sentinel.boosting.build import BLOCKED_EXPERIMENTS
from sentinel.boosting.definitions import BOOSTING_REGISTRY
from sentinel.features.definitions import FEATURE_COLUMNS
from sentinel.modeling.definitions import CDPH_2015_UNREACHABLE_INPUTS

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Anything that could name or pseudonymise the person who performed an inspection.
#: Deliberately broad: the point is to catch a column nobody thought to tell us about.
INSPECTOR_PATTERNS = (
    "inspector",
    "examiner",
    "sanitarian",
    "officer",
    "staff",
    "employee",
    "badge",
    "inspected_by",
)


def _looks_like_an_inspector(column: str) -> bool:
    lowered = column.lower()
    return any(pattern in lowered for pattern in INSPECTOR_PATTERNS)


def _raw_manifest_columns() -> list[str] | None:
    """The ingested raw column list, from Component 1's committed manifest.

    Returns ``None`` when no snapshot has been ingested, so the suite still runs on a
    clean checkout -- but the schema tests below cover the contract regardless.
    """
    directory = REPO_ROOT / "data" / "raw" / "food_inspections"
    manifests = sorted(directory.glob("manifest_*.json")) if directory.exists() else []
    if not manifests:
        return None
    payload = json.loads(manifests[-1].read_text(encoding="utf-8"))
    columns = payload.get("column_names") or payload.get("socrata_field_names")
    return list(columns) if columns else None


# --- 1. the absence is real -----------------------------------------------------


def test_the_raw_data_contract_lists_no_inspector_column() -> None:
    """Parsed from the contract document, so a future column addition fails here."""
    text = (REPO_ROOT / "docs" / "data_contracts" / "food_inspections_raw.md").read_text(
        encoding="utf-8"
    )
    backticked = set(re.findall(r"`([a-z_:@0-9]+)`", text))
    offenders = sorted(c for c in backticked if _looks_like_an_inspector(c))
    assert not offenders, (
        f"the raw data contract now names {offenders}. If Chicago has begun publishing "
        "inspector identity, ADR 0019's blocked record needs revisiting rather than "
        "leaving this test red."
    )


@pytest.mark.skipif(_raw_manifest_columns() is None, reason="no ingested snapshot on disk")
def test_the_ingested_snapshot_carries_no_inspector_column() -> None:
    """The data itself, not a document about the data."""
    columns = _raw_manifest_columns()
    assert columns is not None
    offenders = sorted(c for c in columns if _looks_like_an_inspector(c))
    assert not offenders, f"the ingested snapshot now carries {offenders}"


def test_component_4_declares_no_inspector_derived_feature() -> None:
    offenders = sorted(c for c in FEATURE_COLUMNS if _looks_like_an_inspector(c))
    assert not offenders


def test_no_boosted_model_names_an_inspector_feature() -> None:
    for spec in BOOSTING_REGISTRY:
        assert not [c for c in spec.feature_columns if _looks_like_an_inspector(c)]


def test_the_target_and_evaluation_contracts_carry_no_inspector_column() -> None:
    from sentinel.evaluation.contract import PREDICTION_COLUMNS, PREDICTION_METADATA_COLUMNS

    for column in (*PREDICTION_COLUMNS, *PREDICTION_METADATA_COLUMNS):
        assert not _looks_like_an_inspector(column)


# --- 2. the blocked record travels with the artifact -----------------------------


def test_the_manifest_records_the_block_and_says_why() -> None:
    """Every run carries the record, so it cannot be lost with a document."""
    entry = next((b for b in BLOCKED_EXPERIMENTS if "inspector" in b), None)
    assert entry is not None
    assert "none identifies an inspector" in entry
    assert "random intercept" in entry
    assert "ADR 0019" in entry


def test_the_block_refuses_a_proxy_rather_than_offering_one() -> None:
    """The failure mode this record exists to prevent is a plausible substitute.

    Ward, day-of-week and violation-text verbosity all correlate with *something*, and
    any of them could be dressed up as "inspector strictness". None identifies a person,
    all are confounded with establishment composition, and a marginalisation over them
    would answer a question nobody asked. The blocked note has to say so.
    """
    entry = next(b for b in BLOCKED_EXPERIMENTS if "inspector" in b)
    assert "not approximated by a proxy" in entry


def test_the_earlier_components_already_named_this_as_unreachable() -> None:
    """Component 6 recorded it in the 2015-model input list; Component 7 inherits it."""
    assert any("inspector identity" in item for item in CDPH_2015_UNREACHABLE_INPUTS)


def test_adr_0019_exists_and_documents_the_block() -> None:
    path = REPO_ROOT / "docs" / "decisions" / "0019-inspector-effect-modelling-is-blocked.md"
    assert path.exists(), "ADR 0019 is referenced by the manifest and must exist"
    text = path.read_text(encoding="utf-8")
    assert "## Context" in text
    assert "## Decision" in text
    assert "## Alternatives rejected" in text
    assert "## Consequences" in text
