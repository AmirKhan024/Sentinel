"""Determinism: shuffled inputs, repeated runs, and the ordering contract.

Component 12 found that shuffling the prediction rows changed every disparity it reported,
because equal-mass binning assigned boundary ties by arrival order. Row order is therefore
treated here as an explicit input contract rather than an assumption, and these tests are what
enforce it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.policy import writer
from sentinel.policy.allocation import allocate, decide, risk_order
from sentinel.policy.definitions import POLICY_GRID, PolicySpec, ReserveMechanism
from tests.conftest import make_policy_window

FORCED = PolicySpec("forced", ReserveMechanism.FORCED, 0.20, "spend a fifth")


# --- 1. the allocator does not read row order -------------------------------------


@pytest.mark.parametrize("spec", list(POLICY_GRID))
def test_shuffling_the_window_does_not_change_the_selected_establishments(
    spec: PolicySpec,
) -> None:
    """The property every policy in the grid must hold, checked for every policy in the grid."""
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    eligible = [False, True, False, True, False, True, False, True]
    ids = [f"T{i}" for i in range(len(scores))]
    forward = make_policy_window(scores=scores, eligible=eligible, ids=ids)
    order = [7, 2, 5, 0, 3, 6, 1, 4]
    shuffled = make_policy_window(
        scores=[scores[i] for i in order],
        eligible=[eligible[i] for i in order],
        ids=[ids[i] for i in order],
    )

    left = allocate(forward, spec, k_name="k_1_day", k=5)
    right = allocate(shuffled, spec, k_name="k_1_day", k=5)
    assert [forward.ids[i] for i in left.risk_indices] == [
        shuffled.ids[i] for i in right.risk_indices
    ]
    assert [forward.ids[i] for i in left.reserve_indices] == [
        shuffled.ids[i] for i in right.reserve_indices
    ]


def test_shuffling_a_window_of_pure_ties_does_not_change_the_selection() -> None:
    """The hardest case, and the one Component 12's defect lived in.

    With every score identical there is nothing but the tie-break to decide, so if row order
    leaked in anywhere this is where it would show.
    """
    ids = [f"T{i}" for i in range(6)]
    forward = make_policy_window(scores=[0.5] * 6, ids=ids, eligible=[False, True] * 3)
    reversed_window = make_policy_window(
        scores=[0.5] * 6, ids=list(reversed(ids)), eligible=list(reversed([False, True] * 3))
    )
    left = allocate(forward, FORCED, k_name="k_1_day", k=4)
    right = allocate(reversed_window, FORCED, k_name="k_1_day", k=4)
    assert sorted(forward.ids[i] for i in left.risk_indices) == sorted(
        reversed_window.ids[i] for i in right.risk_indices
    )


def test_shuffling_does_not_change_the_final_ranks() -> None:
    ids = [f"T{i}" for i in range(6)]
    scores = [0.9, 0.5, 0.5, 0.5, 0.2, 0.1]
    forward = make_policy_window(scores=scores, ids=ids)
    order = [4, 1, 5, 0, 3, 2]
    shuffled = make_policy_window(scores=[scores[i] for i in order], ids=[ids[i] for i in order])
    spec = POLICY_GRID[0]
    _m1, _r1, ranks_a = decide(forward, allocate(forward, spec, k_name="k_1_day", k=3))
    _m2, _r2, ranks_b = decide(shuffled, allocate(shuffled, spec, k_name="k_1_day", k=3))
    left = {forward.ids[i]: ranks_a[i] for i in range(forward.n)}
    right = {shuffled.ids[i]: ranks_b[i] for i in range(shuffled.n)}
    assert left == right


def test_the_cached_order_is_keyed_by_content_not_by_object_identity() -> None:
    """``risk_order`` is memoised. A cache keyed by identity would serve a stale answer."""
    first = make_policy_window(scores=[0.1, 0.9], ids=["A", "B"])
    second = make_policy_window(scores=[0.9, 0.1], ids=["A", "B"])
    assert risk_order(first) == (1, 0)
    assert risk_order(second) == (0, 1)


# --- 2. the whole component, run twice ----------------------------------------------


def test_two_runs_over_identical_inputs_produce_byte_identical_tables(
    settings: Settings, tmp_path: Path
) -> None:
    """The reproducibility claim, exercised end to end rather than asserted in a docstring.

    The real production runs are compared the same way and all eleven tables match; this is the
    version that runs in CI, on a synthetic snapshot, so a regression is caught before a
    release rather than after one.
    """
    from sentinel.policy.build import run_policy
    from sentinel.policy.definitions import CANDIDATE_MODELS
    from tests.conftest import (
        calibrated_predictions_for,
        neural_categoricals_for,
        spanning_model_features,
    )
    from tests.test_policy_build import _evaluation_artifacts, _support_table

    directory = tmp_path / "inputs"
    directory.mkdir()
    features = spanning_model_features(days=1500, per_day=3)
    features.write_parquet(directory / "features.parquet")
    calibrated_predictions_for(features, models=CANDIDATE_MODELS).write_parquet(
        directory / "calibrated.parquet"
    )
    categoricals = neural_categoricals_for(features)
    categoricals.write_parquet(directory / "categoricals.parquet")
    evaluation = _evaluation_artifacts(features, directory)
    support = _support_table(categoricals, directory)

    def _once(destination: Path) -> dict[str, bytes]:
        run_policy(
            settings,
            features_path=directory / "features.parquet",
            calibrated_path=directory / "calibrated.parquet",
            folds_path=evaluation["folds"],
            simulation_path=evaluation["simulation"],
            metrics_path=evaluation["metrics"],
            sensitivity_path=evaluation["sensitivity"],
            categoricals_path=directory / "categoricals.parquet",
            fairness_support_path=support,
            output_dir=destination,
            write_figures=False,
            dry_run=False,
        )
        return {
            path.name.rsplit("_", 1)[0]: path.read_bytes()
            for path in sorted(destination.glob("*.parquet"))
        }

    first = _once(tmp_path / "run1")
    second = _once(tmp_path / "run2")
    assert set(first) == set(writer.SCHEMAS)
    assert first == second


def test_shuffling_the_prediction_rows_does_not_change_the_queue(
    settings: Settings, tmp_path: Path
) -> None:
    """Parquet row order is not a contract, and the queue must not depend on it.

    Shuffled with a fixed permutation rather than a seeded RNG, so the test itself is
    deterministic and a failure is reproducible from the file alone.
    """
    from sentinel.policy.build import run_policy
    from sentinel.policy.definitions import CANDIDATE_MODELS
    from tests.conftest import calibrated_predictions_for, spanning_model_features
    from tests.test_policy_build import _evaluation_artifacts

    directory = tmp_path / "inputs"
    directory.mkdir()
    features = spanning_model_features(days=1500, per_day=3)
    features.write_parquet(directory / "features.parquet")
    predictions = calibrated_predictions_for(features, models=CANDIDATE_MODELS)
    predictions.write_parquet(directory / "ordered.parquet")
    predictions.reverse().write_parquet(directory / "shuffled.parquet")
    evaluation = _evaluation_artifacts(features, directory)

    def _queue(calibrated: str) -> pl.DataFrame:
        result = run_policy(
            settings,
            features_path=directory / "features.parquet",
            calibrated_path=directory / calibrated,
            folds_path=evaluation["folds"],
            simulation_path=evaluation["simulation"],
            metrics_path=evaluation["metrics"],
            sensitivity_path=evaluation["sensitivity"],
            write_figures=False,
            dry_run=True,
        )
        return result.tables["inspection_recommendations"].select(
            "policy_id", "fold_id", "k_name", "target_inspection_id", "final_policy_rank"
        )

    assert _queue("ordered.parquet").equals(_queue("shuffled.parquet"))


def test_shuffling_the_feature_rows_does_not_change_the_queue(
    settings: Settings, tmp_path: Path
) -> None:
    """The other half of the ordering contract: the label and history side."""
    from sentinel.policy.build import run_policy
    from sentinel.policy.definitions import CANDIDATE_MODELS
    from tests.conftest import calibrated_predictions_for, spanning_model_features
    from tests.test_policy_build import _evaluation_artifacts

    directory = tmp_path / "inputs"
    directory.mkdir()
    features = spanning_model_features(days=1500, per_day=3)
    features.write_parquet(directory / "ordered.parquet")
    features.reverse().write_parquet(directory / "shuffled.parquet")
    calibrated_predictions_for(features, models=CANDIDATE_MODELS).write_parquet(
        directory / "calibrated.parquet"
    )
    evaluation = _evaluation_artifacts(features, directory)

    def _queue(name: str) -> Any:
        result = run_policy(
            settings,
            features_path=directory / name,
            calibrated_path=directory / "calibrated.parquet",
            folds_path=evaluation["folds"],
            simulation_path=evaluation["simulation"],
            metrics_path=evaluation["metrics"],
            sensitivity_path=evaluation["sensitivity"],
            write_figures=False,
            dry_run=True,
        )
        return result.tables["inspection_recommendations"].select(
            "policy_id", "fold_id", "k_name", "target_inspection_id", "final_policy_rank"
        )

    assert _queue("ordered.parquet").equals(_queue("shuffled.parquet"))
