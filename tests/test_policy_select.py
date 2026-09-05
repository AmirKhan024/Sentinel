"""The model-selection rule: applied, terminating, and honest about the tie band.

MEMORY open question 13 -- which model Sentinel should carry -- stayed open through nine
components. It lands here because a queue needs exactly one model. These tests check that the
rule is a rule: it reads only the declared axes, it always terminates, and it records the
answer the discarded tie band would have given.
"""

from __future__ import annotations

import polars as pl
import pytest

from sentinel.policy.definitions import CANDIDATE_MODELS, DISCARDED_TIE_BAND
from sentinel.policy.select import SelectionError, axis_table, select

MODELS = ("model_a", "model_b", "model_c")


def _simulation(nde: dict[str, float], fold_set: str = "quarterly") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_name": list(nde),
            "schedule_name": ["model"] * len(nde),
            "fold_set": [fold_set] * len(nde),
            "normalized_discovery_efficiency": list(nde.values()),
        }
    )


def _sensitivity(bands: dict[str, tuple[float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_name": list(bands),
            "fold_set": ["quarterly"] * len(bands),
            "p05": [low for low, _ in bands.values()],
            "p95": [high for _, high in bands.values()],
        }
    )


def _metrics(ece: dict[str, float], precision: dict[str, float]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for model, value in ece.items():
        rows.append(
            {
                "model_name": model,
                "fold_set": "quarterly",
                "metric": "ece",
                "k_name": "",
                "value": value,
            }
        )
    for model, value in precision.items():
        rows.append(
            {
                "model_name": model,
                "fold_set": "quarterly",
                "metric": "precision_at_k",
                "k_name": "k_1_day",
                "value": value,
            }
        )
    return pl.DataFrame(rows)


def _select(
    nde: dict[str, float],
    bands: dict[str, tuple[float, float]],
    ece: dict[str, float],
    precision: dict[str, float],
) -> object:
    return select(
        simulation=_simulation(nde),
        metrics=_metrics(ece, precision),
        sensitivity=_sensitivity(bands),
        definition_version="v1",
        models=tuple(nde),
    )


# --- 1. axis 1: separation, and the absence of it ---------------------------------


def test_a_model_whose_band_does_not_overlap_the_leader_is_separated() -> None:
    """When NDE genuinely separates, the rule stops on axis 1 and calibration never runs."""
    selection = _select(
        nde={"model_a": 0.40, "model_b": 0.20},
        bands={"model_a": (0.38, 0.42), "model_b": (0.18, 0.22)},
        ece={"model_a": 0.09, "model_b": 0.01},
        precision={"model_a": 0.5, "model_b": 0.9},
    )
    assert selection.model_name == "model_a"
    assert selection.decided_on_axis == "nde"
    assert selection.n_tied_on_nde == 1


def test_overlapping_bands_are_tied_and_the_rule_falls_to_calibration() -> None:
    """The real case. Component 5's 1,000-replication study cannot separate the candidates.

    That the leader on NDE is *not* the model selected is the point: an NDE difference smaller
    than the metric's own perturbation interval is not a difference between models.
    """
    selection = _select(
        nde={"model_a": 0.25, "model_b": 0.24},
        bands={"model_a": (0.23, 0.27), "model_b": (0.22, 0.26)},
        ece={"model_a": 0.09, "model_b": 0.04},
        precision={"model_a": 0.9, "model_b": 0.5},
    )
    assert selection.n_tied_on_nde == 2
    assert selection.model_name == "model_b"
    assert selection.decided_on_axis == "ece"


def test_the_rule_falls_to_precision_when_calibration_also_ties() -> None:
    selection = _select(
        nde={"model_a": 0.25, "model_b": 0.24},
        bands={"model_a": (0.23, 0.27), "model_b": (0.22, 0.26)},
        ece={"model_a": 0.05, "model_b": 0.05},
        precision={"model_a": 0.60, "model_b": 0.70},
    )
    assert selection.model_name == "model_b"
    assert selection.decided_on_axis == "precision_at_k_1_day"


def test_the_rule_always_terminates_on_the_model_name() -> None:
    """Two models identical on every measured axis still produce one answer, deterministically."""
    selection = _select(
        nde={"model_b": 0.25, "model_a": 0.25},
        bands={"model_b": (0.23, 0.27), "model_a": (0.23, 0.27)},
        ece={"model_b": 0.05, "model_a": 0.05},
        precision={"model_b": 0.6, "model_a": 0.6},
    )
    assert selection.model_name == "model_a"
    assert selection.decided_on_axis == "model_name"


# --- 2. the discarded band, recorded rather than forgotten -------------------------


def test_the_discarded_band_outcome_is_reported_beside_the_rules() -> None:
    """The tie rule decides the deployment, and it was fixed after its inputs were read.

    ADR 0039 records that sequence. A manifest that carried only the outcome would hide that
    another defensible rule selects a different model, so both are emitted on every run.
    """
    selection = _select(
        nde={"model_a": 0.25, "model_b": 0.25 - DISCARDED_TIE_BAND * 3},
        bands={"model_a": (0.20, 0.30), "model_b": (0.19, 0.29)},
        ece={"model_a": 0.09, "model_b": 0.02},
        precision={"model_a": 0.6, "model_b": 0.6},
    )
    # Band overlap ties them, so calibration picks model_b.
    assert selection.model_name == "model_b"
    # The narrow discarded band separates them on NDE alone, so it picks model_a.
    assert selection.under_discarded_band == "model_a"


# --- 3. what the rule refuses ------------------------------------------------------


def test_a_candidate_with_no_measurement_stops_the_run() -> None:
    """Dropping it silently would let an absent number decide the deployment."""
    with pytest.raises(SelectionError, match="absent from the evaluation artifacts"):
        select(
            simulation=_simulation({"model_a": 0.25}),
            metrics=_metrics({"model_a": 0.05}, {"model_a": 0.6}),
            sensitivity=_sensitivity({"model_a": (0.2, 0.3)}),
            definition_version="v1",
            models=("model_a", "model_missing"),
        )


def test_a_candidate_missing_one_axis_stops_the_run() -> None:
    with pytest.raises(SelectionError, match="ece"):
        select(
            simulation=_simulation({"model_a": 0.25, "model_b": 0.24}),
            metrics=_metrics({"model_a": 0.05}, {"model_a": 0.6, "model_b": 0.6}),
            sensitivity=_sensitivity({"model_a": (0.2, 0.3), "model_b": (0.2, 0.3)}),
            definition_version="v1",
            models=("model_a", "model_b"),
        )


def test_an_empty_candidate_list_is_refused() -> None:
    with pytest.raises(SelectionError, match="no admissible"):
        select(
            simulation=_simulation({}),
            metrics=_metrics({}, {}),
            sensitivity=_sensitivity({}),
            definition_version="v1",
            models=(),
        )


# --- 4. the fold set the rule reads -------------------------------------------------


def test_the_shift_fold_is_not_pooled_into_the_selection() -> None:
    """One 18-month episode cannot outvote seventeen quarters.

    Component 7 measured that the shift fold orders these models differently. It is reported
    beside the rule as a named limitation, never averaged into it.
    """
    quarterly = _simulation({"model_a": 0.25, "model_b": 0.20})
    shift = _simulation({"model_a": 0.01, "model_b": 0.99}, fold_set="covid_shift")
    table = axis_table(
        simulation=pl.concat([quarterly, shift]),
        metrics=_metrics({"model_a": 0.05, "model_b": 0.05}, {"model_a": 0.6, "model_b": 0.6}),
        sensitivity=_sensitivity({"model_a": (0.23, 0.27), "model_b": (0.18, 0.22)}),
        models=("model_a", "model_b"),
    )
    values = dict(zip(table["model_name"].to_list(), table["nde"].to_list(), strict=True))
    assert values["model_a"] == pytest.approx(0.25)
    assert values["model_b"] == pytest.approx(0.20)


# --- 5. the emitted audit trail -----------------------------------------------------


def test_every_registered_model_gets_a_row_including_the_refused_one() -> None:
    """The refusal is data, so a reader of the Parquet learns why there are four and not five."""
    selection = _select(
        nde=dict.fromkeys(CANDIDATE_MODELS, 0.25),
        bands=dict.fromkeys(CANDIDATE_MODELS, (0.23, 0.27)),
        ece=dict.fromkeys(CANDIDATE_MODELS, 0.05),
        precision=dict.fromkeys(CANDIDATE_MODELS, 0.6),
    )
    names = [row["model_name"] for row in selection.rows]
    assert "xgboost_chain_embeddings_platt" in names
    refused = next(r for r in selection.rows if r["model_name"] == "xgboost_chain_embeddings_platt")
    assert refused["admissible"] is False
    assert refused["is_selected"] is False
    assert refused["admissibility_reason"]


def test_exactly_one_row_is_marked_selected() -> None:
    selection = _select(
        nde=dict.fromkeys(CANDIDATE_MODELS, 0.25),
        bands=dict.fromkeys(CANDIDATE_MODELS, (0.23, 0.27)),
        ece={name: 0.05 + index / 100 for index, name in enumerate(CANDIDATE_MODELS)},
        precision=dict.fromkeys(CANDIDATE_MODELS, 0.6),
    )
    assert sum(1 for row in selection.rows if row["is_selected"]) == 1
