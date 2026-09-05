"""Applying the frozen model-selection rule. Pure -- no filesystem, no clock.

Nine components produced five calibrated models and no decision about which one a city should
carry. MEMORY open question 13 stayed open through Components 11 and 12, both of which were
explicitly blocked from settling it, and it lands here because a queue needs exactly one model
and cannot be built from an unresolved question.

**This module applies a rule; it does not search for a winner.** The axes, their order, the
tie rule and the admissible candidates are all frozen in ``definitions.py``. What happens here
is arithmetic over Component 5 and Component 9's own published artifacts, and every input and
intermediate is written to ``policy_model_selection`` so the decision can be re-derived by
someone who does not trust it.

**The result is an operating choice, not a finding.** Axis 1 turned out to separate nothing:
under Component 5's 1,000-replication label-flip study, all four candidates' NDE intervals
overlap, so the headline metric of this entire project cannot tell them apart. The rule
therefore fell to calibration. That is a defensible basis for a deployment decision and a poor
basis for a claim about which model is better, and the manifest says so in those words.

**The tie rule was fixed after its inputs were first read.** ADR 0039 records the sequence: a
placeholder band borrowed from a different metric was replaced by Component 5's own NDE
interval once the mismatch was noticed, and the two rules select different models. Both
outcomes are emitted on every run. A rule chosen after seeing what it decides is defensible
only when the choosing is visible, so it is made visible in the artifact rather than argued
away in a comment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from sentinel.policy.definitions import (
    CANDIDATE_MODELS,
    DISCARDED_TIE_BAND,
    MODEL_CANDIDATES,
    SELECTION_AXES,
    SELECTION_FOLD_SET,
)


class SelectionError(ValueError):
    """Raised when the selection rule cannot be applied to the artifacts offered."""


@dataclass(frozen=True, slots=True)
class Selection:
    """The outcome of the rule, and everything needed to check it."""

    model_name: str
    decided_on_axis: str
    under_discarded_band: str
    n_tied_on_nde: int
    rows: tuple[dict[str, object], ...]


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise SelectionError(f"expected a numeric metric, got {type(value).__name__}")


def _require(value: float | None, model: str, axis: str) -> float:
    if value is None:
        raise SelectionError(
            f"{model}: axis {axis!r} is missing from the evaluation artifacts. The rule cannot "
            "be applied to a candidate it has no measurement for, and dropping the candidate "
            "silently would let an absent number decide the deployment"
        )
    return value


def axis_table(
    *,
    simulation: pl.DataFrame,
    metrics: pl.DataFrame,
    sensitivity: pl.DataFrame,
    models: Sequence[str] = CANDIDATE_MODELS,
) -> pl.DataFrame:
    """The three axes plus the NDE interval, per model, from Components 5 and 9's artifacts.

    Restricted to the quarterly fold set. The ``covid_shift`` fold is one 18-month episode;
    pooling it in would let a single unusual period outvote seventeen ordinary quarters, and
    Component 7 already measured that it orders these models differently.
    """
    wanted = list(models)
    nde = (
        simulation.filter(
            (pl.col("schedule_name") == "model")
            & (pl.col("fold_set") == SELECTION_FOLD_SET)
            & pl.col("model_name").is_in(wanted)
        )
        .group_by("model_name")
        .agg(pl.col("normalized_discovery_efficiency").mean().alias("nde"))
    )
    band = (
        sensitivity.filter(
            (pl.col("fold_set") == SELECTION_FOLD_SET) & pl.col("model_name").is_in(wanted)
        )
        .group_by("model_name")
        .agg(pl.col("p05").mean().alias("nde_p05"), pl.col("p95").mean().alias("nde_p95"))
    )
    quarterly = metrics.filter(
        (pl.col("fold_set") == SELECTION_FOLD_SET) & pl.col("model_name").is_in(wanted)
    )
    ece = (
        quarterly.filter(pl.col("metric") == "ece")
        .group_by("model_name")
        .agg(pl.col("value").mean().alias("ece"))
    )
    precision = (
        quarterly.filter((pl.col("metric") == "precision_at_k") & (pl.col("k_name") == "k_1_day"))
        .group_by("model_name")
        .agg(pl.col("value").mean().alias("precision_at_k_1_day"))
    )
    return (
        nde.join(band, on="model_name", how="left")
        .join(ece, on="model_name", how="left")
        .join(precision, on="model_name", how="left")
        .sort("model_name")
    )


def _bands_overlap(left: dict[str, object], right: dict[str, object]) -> bool:
    """Two NDE intervals overlap, so the metric cannot separate the two models.

    The comparison Component 6 already used to decide whether two NDE numbers differ. It is
    conservative -- overlapping intervals are weaker evidence of equality than a formal test
    -- and conservative is the right direction here, because the consequence of wrongly
    declaring a difference is choosing a city's model on noise.
    """
    left_low = _require(_as_float(left["nde_p05"]), str(left["model_name"]), "nde_p05")
    left_high = _require(_as_float(left["nde_p95"]), str(left["model_name"]), "nde_p95")
    right_low = _require(_as_float(right["nde_p05"]), str(right["model_name"]), "nde_p05")
    right_high = _require(_as_float(right["nde_p95"]), str(right["model_name"]), "nde_p95")
    return left_high >= right_low and right_high >= left_low


def select(
    *,
    simulation: pl.DataFrame,
    metrics: pl.DataFrame,
    sensitivity: pl.DataFrame,
    definition_version: str,
    models: Sequence[str] = CANDIDATE_MODELS,
) -> Selection:
    """Apply the lexicographic rule and record every step of it."""
    if not models:
        raise SelectionError("no admissible model candidates")
    table = axis_table(
        simulation=simulation, metrics=metrics, sensitivity=sensitivity, models=models
    )
    missing = sorted(set(models) - set(table["model_name"].to_list()))
    if missing:
        raise SelectionError(
            f"{', '.join(missing)}: absent from the evaluation artifacts. Every admissible "
            "candidate must be measurable, or the rule is choosing between whichever models "
            "happened to be scored"
        )

    candidates = list(table.iter_rows(named=True))
    for row in candidates:
        _require(_as_float(row["nde"]), str(row["model_name"]), "nde")
        _require(_as_float(row["ece"]), str(row["model_name"]), "ece")
        _require(
            _as_float(row["precision_at_k_1_day"]), str(row["model_name"]), "precision_at_k_1_day"
        )

    leader = sorted(
        candidates, key=lambda r: (-(_as_float(r["nde"]) or 0.0), str(r["model_name"]))
    )[0]

    tied = [row for row in candidates if _bands_overlap(row, leader)]
    decided_on = SELECTION_AXES[0][0]
    surviving = tied
    if len(surviving) > 1:
        best_ece = min(_as_float(r["ece"]) or 0.0 for r in surviving)
        narrowed = [r for r in surviving if (_as_float(r["ece"]) or 0.0) == best_ece]
        if len(narrowed) < len(surviving):
            decided_on = SELECTION_AXES[1][0]
        surviving = narrowed
    if len(surviving) > 1:
        best_precision = max(_as_float(r["precision_at_k_1_day"]) or 0.0 for r in surviving)
        narrowed = [
            r for r in surviving if (_as_float(r["precision_at_k_1_day"]) or 0.0) == best_precision
        ]
        if len(narrowed) < len(surviving):
            decided_on = SELECTION_AXES[2][0]
        surviving = narrowed
    if len(surviving) > 1:
        decided_on = SELECTION_AXES[3][0]
    selected = sorted(surviving, key=lambda r: str(r["model_name"]))[0]

    # The rule the plan started with, applied to the same numbers, so the reader can see that
    # the choice of tie rule -- not the data -- is what picks the model.
    leader_nde = _as_float(leader["nde"]) or 0.0
    discarded_tied = [
        r for r in candidates if leader_nde - (_as_float(r["nde"]) or 0.0) <= DISCARDED_TIE_BAND
    ]
    discarded_best = min(_as_float(r["ece"]) or 0.0 for r in discarded_tied)
    discarded_pick = sorted(
        [r for r in discarded_tied if (_as_float(r["ece"]) or 0.0) == discarded_best],
        key=lambda r: str(r["model_name"]),
    )[0]

    admissibility = {c.model_name: c for c in MODEL_CANDIDATES}
    rows: list[dict[str, object]] = []
    for candidate in MODEL_CANDIDATES:
        measured = next(
            (r for r in candidates if str(r["model_name"]) == candidate.model_name), None
        )
        rows.append(
            {
                "model_name": candidate.model_name,
                "admissible": candidate.admissible,
                "admissibility_reason": admissibility[candidate.model_name].reason,
                "nde": _as_float(measured["nde"]) if measured else None,
                "nde_p05": _as_float(measured["nde_p05"]) if measured else None,
                "nde_p95": _as_float(measured["nde_p95"]) if measured else None,
                "ece": _as_float(measured["ece"]) if measured else None,
                "precision_at_k_1_day": (
                    _as_float(measured["precision_at_k_1_day"]) if measured else None
                ),
                "tied_on_nde": (
                    any(str(r["model_name"]) == candidate.model_name for r in tied)
                    if measured
                    else None
                ),
                "is_selected": candidate.model_name == str(selected["model_name"]),
                "selected_under_discarded_band": (
                    candidate.model_name == str(discarded_pick["model_name"])
                ),
                "decided_on_axis": decided_on,
                "fold_set": SELECTION_FOLD_SET,
                "policy_definition_version": definition_version,
            }
        )

    return Selection(
        model_name=str(selected["model_name"]),
        decided_on_axis=decided_on,
        under_discarded_band=str(discarded_pick["model_name"]),
        n_tied_on_nde=len(tied),
        rows=tuple(rows),
    )


__all__ = ["Selection", "SelectionError", "axis_table", "select"]
