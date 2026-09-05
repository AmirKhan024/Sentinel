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
import polars as pl
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


# --- Component 2 (entity resolution) fixtures ---------------------------
#
# `make_record` deliberately makes every field distinct, which is the right
# shape for ingestion tests and useless for entity resolution: the whole problem
# is near-duplicates. These helpers let a test state only the identity fields it
# cares about and inherit sane values for the rest.


def make_entity_record(index: int, **overrides: Any) -> dict[str, Any]:
    """A raw-shaped record with identity fields overridable.

    Pass ``dba_name=...``, ``address=...``, ``license_=...`` and so on to build
    a deliberate near-duplicate. ``None`` is a meaningful override (it sets the
    field to null); omitted keys keep `make_record`'s default.
    """
    record = make_record(index)
    record.update(overrides)
    return record


def entity_scenario(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Turn override dicts into a raw-shaped, all-Utf8 frame."""
    from sentinel.ingest.food_inspections import records_to_frame

    return records_to_frame(rows, columns=FIELD_NAMES)


# --- Component 3 (target construction) fixtures -------------------------


def make_inspection_record(index: int, **overrides: Any) -> dict[str, Any]:
    """A raw-shaped record with outcome fields overridable.

    Defaults to an eligible, negative canvass in the code era, so a test only
    has to state the one thing it is about. Pass ``results=``, ``violations=``,
    ``inspection_type=`` or ``inspection_date=`` to move it off that baseline.
    """
    record = make_record(index)
    record.update(
        {
            "inspection_date": "2022-03-14T00:00:00.000",
            "inspection_type": "Canvass",
            "results": "Pass",
            "violations": "55. PHYSICAL FACILITIES - Comments: DIRTY FLOOR.",
        }
    )
    record.update(overrides)
    return record


def target_scenario(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Turn override dicts into a raw-shaped, all-Utf8 frame."""
    from sentinel.ingest.food_inspections import records_to_frame

    return records_to_frame(rows, columns=FIELD_NAMES)


def assignment_frame(pairs: list[tuple[str, str]]) -> pl.DataFrame:
    """A minimal Component 2 assignments frame: (inspection_id, establishment_id)."""
    return pl.DataFrame(
        {
            "inspection_id": [i for i, _ in pairs],
            "establishment_id": [e for _, e in pairs],
        },
        schema={"inspection_id": pl.Utf8, "establishment_id": pl.Utf8},
    )


# --- Component 5 (temporal evaluation) fixtures -------------------------


def make_feature_row(index: int, **overrides: Any) -> dict[str, Any]:
    """One row of a Component 4-shaped feature table.

    Only the columns Component 5 actually reads are present: the keys, the
    label, provenance, and the three columns the deterministic baselines
    consume. The other 23 features are irrelevant to the evaluation harness and
    including them would suggest it looks at them.
    """
    row: dict[str, Any] = {
        "establishment_id": f"EST-{index:011d}",
        "inspection_date": "2022-05-19",
        "target_inspection_id": f"{2000000 + index}",
        "target": 0,
        "code_era_phase": "stable",
        "feature_definition_version": "v1",
        "days_since_last_canvass": 365,
        "priority_at_last_canvass": False,
        "prior_canvass_priority_rate": 0.0,
    }
    row.update(overrides)
    return row


def feature_scenario(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Turn override dicts into a Component 4-shaped feature frame."""
    return pl.DataFrame(
        rows,
        schema={
            "establishment_id": pl.Utf8,
            "inspection_date": pl.Utf8,
            "target_inspection_id": pl.Utf8,
            "target": pl.Int8,
            "code_era_phase": pl.Utf8,
            "feature_definition_version": pl.Utf8,
            "days_since_last_canvass": pl.Int32,
            "priority_at_last_canvass": pl.Boolean,
            "prior_canvass_priority_rate": pl.Float64,
        },
    )


def spanning_features(
    *,
    start: str = "2018-07-02",
    days: int = 3000,
    per_day: int = 3,
    positive_every: int = 2,
) -> pl.DataFrame:
    """A feature table long enough to build real quarterly folds from.

    ``days`` weekday-ish steps from ``start``, ``per_day`` inspections on each,
    every ``positive_every``-th one positive. Deterministic, so a test can assert
    exact fold counts and row counts rather than approximate ones.
    """
    from datetime import date, timedelta

    first = date.fromisoformat(start)
    rows: list[dict[str, Any]] = []
    index = 0
    for offset in range(days):
        day = first + timedelta(days=offset)
        for _ in range(per_day):
            rows.append(
                make_feature_row(
                    index,
                    inspection_date=day.isoformat(),
                    target=1 if index % positive_every == 0 else 0,
                )
            )
            index += 1
    return feature_scenario(rows)


# --- Component 6 (baseline models) fixtures -----------------------------
#
# Component 5's `make_feature_row` is a deliberate 9-column subset -- the harness
# reads only the keys, the label, provenance and the three columns its heuristics
# consume. A model consumes all 26 features, so these fixtures build the full
# Component 4 shape instead. They derive their schema from
# `features.writer.output_schema()` rather than restating it, so a Component 4
# column change surfaces here as a failure rather than as a silently narrower
# fixture.


def make_model_feature_row(index: int, **overrides: Any) -> dict[str, Any]:
    """One row of a full Component 4 feature table, all 33 columns.

    Defaults describe an establishment with ordinary history, so a test only has to
    state the one thing it is about. Pass ``history="none"`` /
    ``history="no_code_era"`` / ``history="no_inspected_canvass"`` to move the row
    onto one of Component 4's NULL patterns; those set every column in the relevant
    null-rule family together, which is the property the four-indicator design
    depends on.
    """
    from sentinel.features.definitions import FEATURE_SPECS, NullRule

    history = overrides.pop("history", "full")
    row: dict[str, Any] = {
        "establishment_id": f"EST-{index:011d}",
        "inspection_date": "2022-05-19",
        "target_inspection_id": f"{2000000 + index}",
        "target": 0,
        "target_status": "eligible",
        "code_era_phase": "stable",
        "feature_definition_version": "v1",
    }

    # Ordinary-history defaults, by dtype. Small non-zero counts so a zero in a test
    # is visibly deliberate.
    for spec in FEATURE_SPECS:
        if spec.dtype == "bool":
            row[spec.name] = False
        elif spec.dtype == "float64":
            row[spec.name] = 0.25
        else:
            row[spec.name] = 4 if spec.name.startswith("prior_") else 200

    nulled: set[NullRule] = set()
    if history == "none":
        nulled = {
            NullRule.NO_PRIOR_CANVASS,
            NullRule.NO_CODE_ERA_CANVASS,
            NullRule.NO_INSPECTED_CANVASS,
            NullRule.NO_PRIOR_INSPECTION,
        }
    elif history == "no_code_era":
        nulled = {NullRule.NO_CODE_ERA_CANVASS}
    elif history == "no_inspected_canvass":
        nulled = {NullRule.NO_INSPECTED_CANVASS}
    elif history != "full":
        raise ValueError(f"unknown history pattern: {history}")

    for spec in FEATURE_SPECS:
        if spec.null_rule in nulled:
            row[spec.name] = None
    # Component 4's paired-count invariant: a family's null mask is exactly the
    # zero-set of a never-null count. Keep the fixture consistent with it, so a test
    # that re-derives the mask from the count agrees with the one from the nulls.
    if NullRule.NO_PRIOR_CANVASS in nulled:
        row["prior_canvass_count"] = 0
    if NullRule.NO_CODE_ERA_CANVASS in nulled:
        row["prior_canvass_count_code_era"] = 0
    if NullRule.NO_INSPECTED_CANVASS in nulled:
        row["prior_canvass_inspected_count"] = 0
    if NullRule.NO_PRIOR_INSPECTION in nulled:
        row["prior_inspection_count_any_type"] = 0

    row.update(overrides)
    return row


def model_feature_scenario(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Turn override dicts into a full Component 4-shaped feature frame."""
    from sentinel.features.writer import output_schema

    schema = output_schema()
    ordered = [{name: row.get(name) for name in schema} for row in rows]
    return pl.DataFrame(ordered, schema=schema)


def spanning_model_features(
    *,
    start: str = "2018-07-02",
    days: int = 3000,
    per_day: int = 3,
    signal_strength: float = 0.6,
) -> pl.DataFrame:
    """A full-width feature table long enough to build real quarterly folds from.

    ``prior_canvass_priority_rate`` is correlated with the target at
    ``signal_strength`` so a fitted model has something to find and the score
    direction is testable; everything else is constant, so any measured effect is
    attributable. Deterministic -- no RNG -- so a test can assert exact values.
    """
    from datetime import date, timedelta

    first = date.fromisoformat(start)
    rows: list[dict[str, Any]] = []
    index = 0
    for offset in range(days):
        day = first + timedelta(days=offset)
        for slot in range(per_day):
            # A deterministic pseudo-pattern: two of every three rows positive on
            # even days, one of three on odd days, so the base rate moves over time
            # the way the real table's does.
            positive = (index + offset) % 3 != 0
            rate = signal_strength if positive else 1.0 - signal_strength
            history = "none" if index % 97 == 0 else ("no_code_era" if index % 31 == 0 else "full")
            row = make_model_feature_row(
                index,
                history=history,
                inspection_date=day.isoformat(),
                target=1 if positive else 0,
            )
            if row["prior_canvass_priority_rate"] is not None:
                row["prior_canvass_priority_rate"] = rate
            row["prior_canvass_count"] = row["prior_canvass_count"] or (2 + slot)
            rows.append(row)
            index += 1
    return model_feature_scenario(rows)


# --- Component 8 (neural models) fixtures -------------------------------
#
# Component 8 needs a second table beside the feature table: its experimental
# categorical layer. It is built here rather than by calling the real
# ``categoricals.build_categoricals``, because that function needs a raw Socrata
# snapshot and Component 2's assignments, and a fixture that required both would
# be testing the join rather than using it. ``test_neural_categoricals.py``
# exercises the real builder against a small hand-made raw frame.
#
# The chain structure is deliberate: names are assigned so that a handful are
# shared across establishments (real chains) and the rest are not, and the shared
# set GROWS over time, so a test can prove membership is derived per fold rather
# than globally.


def make_neural_categorical_row(index: int, **overrides: Any) -> dict[str, Any]:
    """One row of Component 8's experimental categorical table.

    Mirrors ``neural.writer.CATEGORICALS_SCHEMA``. ``source_inspection_date`` is
    always strictly earlier than ``inspection_date`` because that is the property
    the whole layer rests on; a test that wants to violate it overrides it
    explicitly and expects a check to fail.
    """
    from datetime import date, timedelta

    day = date.fromisoformat(overrides.get("inspection_date", "2018-07-02"))
    source = day - timedelta(days=30)
    row: dict[str, Any] = {
        "target_inspection_id": f"T{index:07d}",
        "establishment_id": f"EST-{index % 500:011d}",
        "inspection_date": day.isoformat(),
        "chain_key": f"NAME{index % 40:03d}",
        "facility_type": ["RESTAURANT", "GROCERY STORE", "SCHOOL", "BAKERY"][index % 4],
        "community_area": str(index % 17),
        "zip": f"606{index % 12:02d}",
        "source_inspection_id": f"S{index:07d}",
        "source_inspection_date": source,
        "days_since_source": 30,
    }
    row.update(overrides)
    return row


def neural_categoricals_for(features: pl.DataFrame) -> pl.DataFrame:
    """A categorical table covering exactly one feature table's rows.

    One-to-one on ``target_inspection_id``, which is what
    ``neural.build._load_categoricals`` requires and what
    ``validate._categoricals_cover_every_row`` re-derives.

    **Every value is derived from the row's own identity, never from its position.**
    That is not a stylistic choice: a position-derived fixture makes
    ``neural_categoricals_for(frame)`` and
    ``neural_categoricals_for(frame.sort(...))`` disagree about which row is in which
    chain, and every leakage test that appends, deletes or reorders a row would then
    fail against correct code. The first draft of this fixture did exactly that and
    produced seven false leakage failures.

    Chain membership is engineered rather than random: one row in five is assigned to
    one of eight shared names and the rest get a unique name, so
    ``encode.chain_membership`` finds a real chain set and an ablation that removes it
    is measuring something.
    """
    from datetime import date, timedelta

    def key_of(row_id: str) -> int:
        digits = "".join(ch for ch in row_id if ch.isdigit())
        return int(digits) if digits else abs(hash(row_id))

    rows: list[dict[str, Any]] = []
    for record in features.iter_rows(named=True):
        row_id = str(record["target_inspection_id"])
        key = key_of(row_id)
        day = date.fromisoformat(str(record["inspection_date"]))
        chain = f"NAME{key % 8:03d}" if key % 5 == 0 else f"SOLO{key:06d}"
        rows.append(
            {
                "target_inspection_id": row_id,
                "establishment_id": str(record["establishment_id"]),
                "inspection_date": day.isoformat(),
                "chain_key": chain,
                "facility_type": ["RESTAURANT", "GROCERY STORE", "SCHOOL", "BAKERY"][key % 4],
                "community_area": str(key % 17),
                "zip": f"606{key % 12:02d}",
                "source_inspection_id": f"S{key:07d}",
                "source_inspection_date": day - timedelta(days=30),
                "days_since_source": 30,
            }
        )
    from sentinel.neural.writer import finalize

    return finalize(rows, "neural_categoricals")


def make_raw_inspection_row(index: int, **overrides: Any) -> dict[str, Any]:
    """One row of the raw Socrata snapshot, in the shape Component 8 reads.

    Only the five columns ``categoricals.RAW_COLUMNS`` names are populated with
    meaningful values; the rest of the 22-column snapshot is irrelevant to this
    component and is deliberately absent so a test fails loudly if the module
    starts reading something it did not declare.
    """
    row: dict[str, Any] = {
        "inspection_id": f"S{index:07d}",
        "inspection_date": "2018-01-05T00:00:00.000",
        "facility_type": "Restaurant",
        "zip": "60601",
        ":@computed_region_vrxf_vc4k": "8",
    }
    row.update(overrides)
    return row


# --- Component 12 (fairness audit) fixtures -----------------------------
#
# Component 12 reads Component 9's calibrated artifact rather than any model, so
# its tests need a frame in that shape and never need a fitted estimator. The
# builder below produces one directly instead of running Components 6-9, for the
# same reason ``neural_categoricals_for`` builds its table rather than calling the
# real join: a fixture that ran the whole pipeline would be testing the pipeline.
#
# The score is a deterministic function of the row's own identity and label, so it
# is reproducible, correlated with the outcome (the ranking metrics have something
# to find), and NOT a function of row position -- a position-derived score makes
# ``calibrated_predictions_for(frame)`` and the same call on a shuffled frame
# disagree about which establishment scored what, which is the defect that produced
# seven false leakage failures in Component 8.


def make_calibrated_prediction_row(index: int, **overrides: Any) -> dict[str, Any]:
    """One row of Component 9's calibrated artifact.

    Mirrors ``calibration.writer``'s prediction schema. ``score`` and ``base_score``
    are deliberately different on every row: Component 9 measured Platt moving all
    207,680 of them, and a fixture where the two were equal would let a stage mix-up
    pass every check.
    """
    from datetime import date

    row: dict[str, Any] = {
        "target_inspection_id": f"T{index:07d}",
        "score": 0.5,
        "model_name": "logistic_regression_platt",
        "model_version": "v1",
        "fold_set": "quarterly",
        "fold_id": "quarterly-2022Q2",
        "trained_through": date(2022, 3, 31),
        "is_probability": True,
        "base_model_name": "logistic_regression",
        "base_model_version": "v1",
        "base_score": 0.4,
        "base_model_trained_through": date(2021, 12, 31),
        "calibrator_fitted_through": date(2022, 3, 31),
        "calibrated_prediction_available_from": date(2022, 4, 1),
        "method": "platt",
        "is_experimental": False,
        "calibration_definition_version": "v1",
    }
    row.update(overrides)
    return row


def calibrated_predictions_for(
    features: pl.DataFrame,
    *,
    models: tuple[str, ...] = ("logistic_regression_platt", "xgboost_platt"),
    signal: float = 0.35,
) -> pl.DataFrame:
    """A calibrated artifact covering every test-window row of a feature table's folds.

    Covers exactly the rows Components 6-9 would have scored: each fold's test window
    and nothing else. That matters, because ``every_audited_row_has_a_prediction``
    compares id sets rather than checking containment, so a fixture covering too many
    rows would make the check pass vacuously.

    The score carries real signal (``base + signal`` on a positive row) so that
    ROC-AUC is above 0.5 and a test asserting the score direction is asserting
    something. ``score`` is a monotone but non-identity transform of ``base_score``,
    which is what Platt is.
    """

    from sentinel.evaluation import folds as folds_module

    dated = features.with_columns(pl.col("inspection_date").str.to_date().alias("rd"))
    start = folds_module.min_date(dated, "rd")
    end = folds_module.max_date(dated, "rd")
    if start is None or end is None:
        raise ValueError("feature table has no usable reference dates")
    specs = [
        *folds_module.quarterly_folds(data_start=start, data_end=end),
        *folds_module.covid_shift_fold(data_end=end),
    ]

    def key_of(row_id: str) -> int:
        digits = "".join(ch for ch in row_id if ch.isdigit())
        return int(digits) if digits else abs(hash(row_id))

    rows: list[dict[str, Any]] = []
    for spec in specs:
        window = folds_module.window_frame(dated, spec)
        for record in window.iter_rows(named=True):
            row_id = str(record["target_inspection_id"])
            key = key_of(row_id)
            label = int(record["target"])
            for offset, model in enumerate(models):
                # Deterministic, identity-derived, and label-correlated. The 0.02 offset
                # per model keeps the models distinguishable without changing the ranking.
                base = 0.30 + (key % 40) / 100.0 + (signal if label else 0.0) + offset * 0.02
                base = min(max(base, 0.01), 0.98)
                rows.append(
                    make_calibrated_prediction_row(
                        key,
                        target_inspection_id=row_id,
                        model_name=model,
                        base_model_name=model.removesuffix("_platt"),
                        fold_set=spec.fold_set,
                        fold_id=spec.fold_id,
                        trained_through=spec.calibration_end,
                        base_model_trained_through=spec.train_end,
                        calibrator_fitted_through=spec.calibration_end,
                        calibrated_prediction_available_from=spec.test_start,
                        base_score=base,
                        # Platt: monotone, two parameters, never the identity.
                        score=min(max(0.5 + 0.85 * (base - 0.5), 0.01), 0.99),
                    )
                )
    if not rows:
        raise ValueError("no fold produced a test window")
    return pl.DataFrame(rows).sort(["model_name", "fold_id", "target_inspection_id"])


def explanation_values_for(
    predictions: pl.DataFrame,
    *,
    features_per_row: tuple[str, ...] = (
        "prior_canvass_count",
        "days_since_last_canvass",
        "missing_no_prior_canvass",
    ),
) -> pl.DataFrame:
    """A Component 11 attribution frame over a subset of a prediction frame's rows.

    Only the columns ``fairness.attribution.EXPLANATION_COLUMNS`` names, because that
    module reads the artifact by column name and does not import ``sentinel.explain``
    -- so a fixture carrying the full 20-column schema would hide a dependency on a
    column the module never declared.
    """

    def key_of(row_id: str) -> int:
        digits = "".join(ch for ch in row_id if ch.isdigit())
        return int(digits) if digits else abs(hash(row_id))

    rows: list[dict[str, Any]] = []
    for record in predictions.iter_rows(named=True):
        row_id = str(record["target_inspection_id"])
        key = key_of(row_id)
        base = str(record["base_model_name"])
        for position, feature in enumerate(features_per_row):
            rows.append(
                {
                    "model_name": base,
                    "fold_set": str(record["fold_set"]),
                    "fold_id": str(record["fold_id"]),
                    "target_inspection_id": row_id,
                    "feature_name": feature,
                    "shap_value": ((key % (7 + position)) - 3) / 10.0,
                    "is_exact": base != "neural_numeric_only",
                }
            )
    return pl.DataFrame(rows)


# --- Component 13 (decision policy) fixtures ----------------------------
#
# A policy runs over `PolicyWindow`s, not frames: an allocation is index arithmetic
# and a frame invites a reorder halfway through it. These builders make a window
# directly, so an allocation test can state the one property it is about -- which
# rows are eligible, where the scores tie -- without routing through a feature
# table and a fold construction that are not what the test is checking.


def make_policy_window(
    *,
    scores: list[float],
    eligible: list[bool] | None = None,
    labels: list[int] | None = None,
    ids: list[str] | None = None,
    secondary: list[bool] | None = None,
    fold_id: str = "quarterly-2022Q2",
    fold_set: str = "quarterly",
    median_daily: int = 5,
) -> Any:
    """A `PolicyWindow` from parallel lists, in the order given.

    The order given, deliberately *not* re-sorted into canonical order. A test that
    wants to prove the allocator is independent of input order needs to be able to
    hand it a window that is not in canonical order, and a builder that silently
    sorted would make that test assert nothing.
    """
    from datetime import date, timedelta

    from sentinel.policy.models import PolicyWindow

    n = len(scores)
    row_ids = ids if ids is not None else [f"T{i:05d}" for i in range(n)]
    return PolicyWindow(
        fold_set=fold_set,
        fold_id=fold_id,
        ids=tuple(row_ids),
        scores=tuple(float(s) for s in scores),
        base_scores=tuple(float(s) * 0.9 for s in scores),
        labels=tuple(labels if labels is not None else [0] * n),
        dates=tuple(date(2022, 4, 1) + timedelta(days=i % 30) for i in range(n)),
        eligible=tuple(eligible if eligible is not None else [False] * n),
        secondary_no_history=tuple(secondary if secondary is not None else [False] * n),
        median_daily_capacity=median_daily,
    )


def make_override(index: int, **overrides: Any) -> dict[str, Any]:
    """One row of the human override contract, as it arrives from JSON."""
    row: dict[str, Any] = {
        "override_id": f"OV-{index:04d}",
        "policy_id": "pure_risk",
        "fold_id": "quarterly-2022Q2",
        "k_name": "k_1_day",
        "target_inspection_id": "T00000",
        "action": "force_include",
        "reason_code": "outbreak_investigation",
        "actor": "inspector.smith",
        "decided_at": "2026-08-26T09:00:00Z",
    }
    row.update(overrides)
    return row


def policy_overrides_for(rows: list[dict[str, Any]]) -> str:
    """The override file's JSON text, for a test that writes one to disk."""
    import json

    return json.dumps(rows, indent=2)


# --- Component 14 (operational scheduling) fixtures ---------------------
#
# A schedule runs over a `Horizon` and a tuple of `QueueRow`s, not frames: placement is index
# arithmetic over an approved rank order, and a frame invites a reorder halfway through it --
# the same defect Component 13's fixtures were shaped to avoid. These builders make the two
# directly, so a placement test can state the one property it is about -- where the day
# boundaries fall, what happens when the horizon is short -- without routing through a policy
# artifact and a fold construction that are not what the test is checking.


def make_queue_row(index: int, **overrides: Any) -> Any:
    """One approved recommendation, as Component 13 hands it over.

    ``final_policy_rank`` defaults to ``index + 1`` so a list built by comprehension is already
    a well-formed queue, and a test that wants a broken one has to say so.
    """
    from datetime import date as _date

    from sentinel.scheduling.models import QueueRow

    fields: dict[str, Any] = {
        "target_inspection_id": f"T{index:05d}",
        "establishment_id": f"EST-{index:05d}",
        "final_policy_rank": index + 1,
        "model_rank": index + 1,
        "decision_mechanism": "risk_priority",
        "decision_reason": "selected_by_risk_rank",
        "score": 0.9 - index * 0.001,
        "base_score": 0.8 - index * 0.001,
        "coverage_eligible": False,
        "warnings": "none",
        "recommendation_date": _date(2026, 4, 1),
    }
    fields.update(overrides)
    return QueueRow(**fields)


def make_queue(n: int, *, reserve_tail: int = 0) -> tuple[Any, ...]:
    """An approved queue of ``n`` rows, optionally with a coverage reserve at the tail.

    The reserve goes at the *tail* because that is where Component 13 puts it -- the risk block
    fills ranks 1..n_risk and the reserve follows -- and that placement is the whole reason a
    short horizon takes the reserve first.
    """
    rows = []
    for index in range(n):
        is_reserve = index >= n - reserve_tail
        rows.append(
            make_queue_row(
                index,
                decision_mechanism="coverage_reserve" if is_reserve else "risk_priority",
                decision_reason=(
                    "selected_by_coverage_reserve" if is_reserve else "selected_by_risk_rank"
                ),
                coverage_eligible=is_reserve,
            )
        )
    return tuple(rows)


def make_calendar(counts: list[int], *, start_day: int = 1) -> tuple[tuple[Any, int], ...]:
    """An observed operating calendar from a list of per-day inspection counts.

    Consecutive April dates. The dates themselves carry no meaning in these tests -- what
    matters is that they are distinct and ascending, which is what the horizon requires.
    """
    from datetime import date as _date

    return tuple((_date(2026, 4, start_day + offset), count) for offset, count in enumerate(counts))


def make_horizon(counts: list[int], *, k: int, median: int = 5, mode: Any = None) -> Any:
    """A horizon over a hand-written calendar, in whichever capacity mode is asked for."""
    from sentinel.scheduling.definitions import CapacityMode
    from sentinel.scheduling.horizon import build_horizon

    return build_horizon(
        fold_set="quarterly",
        fold_id="quarterly-2026Q2",
        k_name="k_1_week",
        k=k,
        median_daily_capacity=median,
        calendar=make_calendar(counts),
        capacity_mode=mode or CapacityMode.OBSERVED_CALENDAR,
    )


def make_plan(counts: list[int], *, k: int, reserve_tail: int = 0, median: int = 5) -> Any:
    """A placed plan over a hand-written calendar. The unit most scheduling tests assert on."""
    from sentinel.scheduling.allocation import place
    from sentinel.scheduling.models import SchedulePlan

    queue = make_queue(k, reserve_tail=reserve_tail)
    horizon = make_horizon(counts, k=k, median=median)
    return SchedulePlan(
        schedule_config_id="strict_priority__observed_calendar",
        policy_id="pure_risk",
        model_name="xgboost_platt",
        fold_set="quarterly",
        fold_id="quarterly-2026Q2",
        k_name="k_1_week",
        k=k,
        horizon=horizon,
        placements=place(queue, horizon),
        planning_run_id="PR-test",
    )


def make_adjustment(index: int, **overrides: Any) -> dict[str, Any]:
    """One row of the scheduling-adjustment contract, as it arrives from JSON."""
    row: dict[str, Any] = {
        "adjustment_id": f"SA-{index:04d}",
        "schedule_config_id": "strict_priority__observed_calendar",
        "policy_id": "pure_risk",
        "fold_id": "quarterly-2026Q2",
        "k_name": "k_1_week",
        "target_inspection_id": "T00000",
        "action": "defer_to_date",
        "target_date": "2026-04-03",
        "reason_code": "establishment_closed",
        "actor": "district.supervisor.4",
        "decided_at": "2026-08-26T09:00:00Z",
    }
    row.update(overrides)
    return row


def make_execution_event(index: int, **overrides: Any) -> dict[str, Any]:
    """One row of the execution-event contract, as it arrives from JSON."""
    row: dict[str, Any] = {
        "execution_id": f"EX-{index:04d}",
        "schedule_config_id": "strict_priority__observed_calendar",
        "policy_id": "pure_risk",
        "fold_id": "quarterly-2026Q2",
        "k_name": "k_1_week",
        "target_inspection_id": "T00000",
        "scheduled_date": "2026-04-01",
        "execution_status": "completed",
        "reason_code": "routine",
        "actor": "field.inspector.log",
        "observed_at": "2026-04-01T12:00:00Z",
    }
    row.update(overrides)
    return row


def scheduling_json_for(rows: list[dict[str, Any]]) -> str:
    """The external file's JSON text, for a test that writes one to disk."""
    import json

    return json.dumps(rows, indent=2)
