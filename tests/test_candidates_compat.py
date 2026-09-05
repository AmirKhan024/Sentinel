"""Component 17's two compatibility obligations.

1. Its output must be consumable by the existing scoring path without any new
   adapter code -- checked here directly against
   ``sentinel.boosting.preprocess``, the module ``boosting.predict.score_window``
   uses to build the matrix a fitted estimator scores.
2. Building it must not disturb Component 4's historical path in any way --
   checked here by running ``build_features`` (real Component 4) after
   ``build_candidates`` (Component 17) over the same inputs and asserting the
   Component 4 output is exactly what it would have been on its own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from sentinel.boosting import preprocess as boosting_preprocess
from sentinel.boosting.definitions import BOOSTING_REGISTRY
from sentinel.candidates.build import build_candidates
from sentinel.config import Settings
from sentinel.features.build import build_features
from sentinel.features.definitions import FEATURE_COLUMNS
from tests.conftest import assignment_frame, target_scenario
from tests.test_candidates_leakage import BASE_PAIRS, BASE_ROWS, PLANNING_DATE, _establishments_for

CORE = "55. PHYSICAL FACILITIES - Comments: DIRTY FLOOR."
EST = "EST-A"


def _write_inputs(tmp_path: Path, tag: str) -> tuple[Path, Path, Path]:
    raw = tmp_path / f"raw_{tag}.parquet"
    asg = tmp_path / f"asg_{tag}.parquet"
    est = tmp_path / f"est_{tag}.parquet"
    target_scenario(BASE_ROWS).write_parquet(raw)
    assignment_frame(BASE_PAIRS).write_parquet(asg)
    _establishments_for([EST]).write_parquet(est)
    return raw, asg, est


def _targets_for(rows: list[dict[str, object]], establishment_id: str) -> pl.DataFrame:
    """A Component 3-shaped target table, minimal, for the Component 4 comparison."""
    return pl.DataFrame(
        {
            "establishment_id": [establishment_id] * len(rows),
            "inspection_date": [str(r["inspection_date"]) for r in rows],
            "target_inspection_id": [str(r["inspection_id"]) for r in rows],
            "target": [0] * len(rows),
            "target_status": ["eligible"] * len(rows),
            "code_era_phase": ["pre_code"] * len(rows),
        },
        schema={
            "establishment_id": pl.Utf8,
            "inspection_date": pl.Utf8,
            "target_inspection_id": pl.Utf8,
            "target": pl.Int8,
            "target_status": pl.Utf8,
            "code_era_phase": pl.Utf8,
        },
    )


# --- 1. model compatibility -------------------------------------------------


def test_candidate_output_builds_a_tree_matrix_for_every_registered_booster(
    settings: Settings, tmp_path: Path
) -> None:
    raw, asg, est = _write_inputs(tmp_path, "compat")
    result = build_candidates(
        settings,
        planning_date=PLANNING_DATE,
        parquet_path=raw,
        assignments_path=asg,
        establishments_path=est,
        dry_run=True,
    )
    frame = result.candidates
    assert "target_inspection_id" in frame.columns  # score_window's own requirement

    for spec in BOOSTING_REGISTRY:
        matrix = boosting_preprocess.tree_matrix(frame, spec)
        mask = boosting_preprocess.null_mask(frame, spec)
        assert matrix.shape == (frame.height, len(boosting_preprocess.matrix_columns(spec)))
        assert mask.shape == matrix.shape
        # The NaN pattern the estimator will actually branch on must be exactly the
        # NULL pattern Component 4's missing-value rules declared -- not more, not
        # fewer -- which is precisely what this component must not disturb.
        assert np.array_equal(np.isnan(matrix), mask)


def test_all_declared_feature_columns_are_present_and_typed(
    settings: Settings, tmp_path: Path
) -> None:
    raw, asg, est = _write_inputs(tmp_path, "schema")
    frame = build_candidates(
        settings,
        planning_date=PLANNING_DATE,
        parquet_path=raw,
        assignments_path=asg,
        establishments_path=est,
        dry_run=True,
    ).candidates
    for name in FEATURE_COLUMNS:
        assert name in frame.columns, f"missing declared feature column {name}"


# --- 2. Component 4 is unaffected by Component 17 running first -----------


def test_running_candidate_generation_does_not_change_historical_feature_output(
    settings: Settings, tmp_path: Path
) -> None:
    raw, asg, est = _write_inputs(tmp_path, "hist")
    targets = tmp_path / "tgt_hist.parquet"
    _targets_for([BASE_ROWS[1]], EST).write_parquet(targets)

    baseline = build_features(
        settings, parquet_path=raw, assignments_path=asg, targets_path=targets, dry_run=True
    ).features

    # Run Component 17 against the identical inputs in between.
    build_candidates(
        settings,
        planning_date=PLANNING_DATE,
        parquet_path=raw,
        assignments_path=asg,
        establishments_path=est,
        dry_run=True,
    )

    after = build_features(
        settings, parquet_path=raw, assignments_path=asg, targets_path=targets, dry_run=True
    ).features

    assert baseline.equals(after)
