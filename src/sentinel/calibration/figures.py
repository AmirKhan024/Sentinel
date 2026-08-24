"""Reliability diagrams and the calibration-drift plot.

The only matplotlib in this package, and it follows ``neural/figures.py`` exactly: the
backend is selected before ``pyplot`` is imported, the import is lazy and confined to this
file because matplotlib ships no ``py.typed``, and a function returns ``None`` rather than
raising when a figure cannot be drawn honestly.

These are not decorative. A reliability diagram has to let a reader see the specific claim
"before calibration this model said 0.7 more often than 0.7 happened, and afterwards it did
not" -- so the bins are the same 15 equal-mass bins the ECE uses, the bin masses are
annotated, and the uncertainty band is a real bootstrap rather than a smoothing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from sentinel.calibration.definitions import (
    BOOTSTRAP_SEED,
    CANDIDATE_REGISTRY,
    STAGE_UNCALIBRATED,
    STAGES,
    CandidateSpec,
    Method,
)
from sentinel.calibration.preprocess import expit, logit
from sentinel.evaluation.metrics import DEFAULT_CALIBRATION_BINS

logger = logging.getLogger(__name__)

#: Below this a reliability diagram is not drawn: 15 equal-mass bins over fewer rows would
#: put single digits in each, and a band around that would suggest a precision the data
#: cannot support.
MIN_RELIABILITY_ROWS = 300

#: Replications for the per-bin band. Fewer than the metric bootstrap because this is a
#: visual aid rather than a reported interval.
BAND_REPLICATIONS = 400

STAGE_STYLE: dict[str, tuple[str, str]] = {
    STAGE_UNCALIBRATED: ("#9467bd", "o"),
    Method.PLATT.value: ("#1f77b4", "s"),
    Method.ISOTONIC.value: ("#d62728", "^"),
    "selected": ("#2ca02c", "D"),
}


def _matplotlib() -> tuple[Any, Any]:
    """Import matplotlib with a headless backend, chosen before pyplot loads."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


def _bins(labels: list[int], scores: list[float], n_bins: int) -> list[tuple[int, float, float]]:
    from sentinel.evaluation.metrics import calibration_bins

    return calibration_bins(labels, scores, n_bins=n_bins)


def _band(
    labels: list[int],
    scores: list[float],
    *,
    n_bins: int,
    seed_key: list[int],
    replications: int = BAND_REPLICATIONS,
) -> tuple[list[float], list[float]] | None:
    """Per-bin 2.5/97.5 percentiles of the observed rate, under a row bootstrap.

    **The bins are resampled with the rows**, not held fixed -- an equal-mass bin's
    boundaries are themselves an estimate, and a band that ignored that would be too narrow.
    The caption says so.
    """
    rng = np.random.default_rng(np.random.SeedSequence(seed_key))
    n = len(labels)
    draws: list[list[float]] = []
    for _ in range(replications):
        index = rng.integers(0, n, size=n)
        drawn_labels = [labels[int(i)] for i in index]
        if len(set(drawn_labels)) < 2:
            continue
        bins = _bins(drawn_labels, [scores[int(i)] for i in index], n_bins)
        if len(bins) != n_bins:
            continue
        draws.append([observed for _, _, observed in bins])
    if len(draws) < replications // 2:
        return None
    array = np.asarray(draws, dtype=np.float64)
    return (
        [float(v) for v in np.percentile(array, 2.5, axis=0)],
        [float(v) for v in np.percentile(array, 97.5, axis=0)],
    )


def _replay(
    parameters: pl.DataFrame,
    breakpoints: pl.DataFrame,
    *,
    model_name: str,
    fold_id: str,
    method: Method,
    scores: list[float],
) -> list[float] | None:
    """Re-apply a persisted calibrator from its parameters alone.

    Both methods are drawn on every reliability diagram, including the one the protocol did
    not freeze -- otherwise the picture would only ever show the winner and the comparison
    the brief asks for could not be read off it.

    The losing method's scores are not in ``calibrated_predictions``, by design: that table
    holds the production probability. They are reconstructed here from
    ``calibrator_parameters`` and ``calibrator_isotonic_breakpoints``, which is also a
    standing demonstration that those tables really are sufficient to reproduce the mapping
    -- the claim ADR 0024 makes for them.
    """
    if method is Method.PLATT:
        rows = parameters.filter(
            (pl.col("model_name") == model_name)
            & (pl.col("fold_id") == fold_id)
            & (pl.col("method") == method.value)
        )
        terms = dict(zip(rows["term"].to_list(), rows["value"].to_list(), strict=True))
        if "coef" not in terms or "intercept" not in terms:
            return None
        a, b = float(terms["coef"]), float(terms["intercept"])
        return [expit(a * logit(p) + b) for p in scores]

    rows = breakpoints.filter(
        (pl.col("model_name") == model_name) & (pl.col("fold_id") == fold_id)
    ).sort("breakpoint_index")
    if rows.height == 0:
        return None
    mapped = np.interp(
        np.asarray(scores, dtype=np.float64),
        np.asarray(rows["x_threshold"].to_list(), dtype=np.float64),
        np.asarray(rows["y_threshold"].to_list(), dtype=np.float64),
    )
    return [float(v) for v in mapped]


def reliability_figure(
    base_scores: pl.DataFrame,
    calibrated: pl.DataFrame,
    parameters: pl.DataFrame,
    breakpoints: pl.DataFrame,
    *,
    model_name: str,
    fold_id: str,
    path: Path,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> Path | None:
    """Three panels for one (model, fold): before, after Platt, after isotonic.

    Reads the base scores from the artifact and replays both calibrators from their
    persisted parameters, so the picture is of what was actually written rather than of a
    recomputation.
    """
    test = base_scores.filter(
        (pl.col("model_name") == model_name)
        & (pl.col("fold_id") == fold_id)
        & (pl.col("split") == "test")
    ).sort("target_inspection_id")
    if test.height < MIN_RELIABILITY_ROWS:
        logger.info(
            "Skipping reliability diagram for %s/%s: %d rows is too few for %d equal-mass bins",
            model_name,
            fold_id,
            test.height,
            n_bins,
        )
        return None

    labels = [int(v) for v in test["target"].to_list()]
    stages: dict[str, list[float]] = {STAGE_UNCALIBRATED: [float(v) for v in test["base_score"]]}

    frozen = calibrated.filter(
        (pl.col("base_model_name") == model_name) & (pl.col("fold_id") == fold_id)
    ).sort("target_inspection_id")
    chosen = str(frozen["method"][0]) if frozen.height else None

    for method in Method:
        replayed = _replay(
            parameters, breakpoints, model_name=model_name, fold_id=fold_id,
            method=method, scores=stages[STAGE_UNCALIBRATED],
        )
        if replayed is not None:
            stages[method.value] = replayed

    # A free correctness check every time a figure is drawn: the replayed winner must equal
    # the production artifact it was replayed from.
    if chosen is not None and frozen.height == test.height and chosen in stages:
        published = [float(v) for v in frozen["score"].to_list()]
        drift = max(abs(a - b) for a, b in zip(stages[chosen], published, strict=True))
        if drift > 1e-9:
            logger.warning(
                "%s/%s: replayed %s differs from the published score by %.3e",
                model_name, fold_id, chosen, drift,
            )

    _, plt = _matplotlib()
    figure, axes = plt.subplots(1, len(stages), figsize=(5.0 * len(stages), 5.0), sharey=True)
    panels = axes if isinstance(axes, np.ndarray) else np.asarray([axes])

    from sentinel.evaluation.metrics import brier, ece, mce

    for panel, (stage, scores) in zip(panels, stages.items(), strict=True):
        bins = _bins(labels, scores, n_bins)
        predicted = [p for _, p, _ in bins]
        observed = [o for _, _, o in bins]
        counts = [c for c, _, _ in bins]
        colour, marker = STAGE_STYLE.get(stage, ("#333333", "o"))

        stage_index = STAGES.index(stage) if stage in STAGES else len(STAGES)
        band = _band(labels, scores, n_bins=n_bins, seed_key=[BOOTSTRAP_SEED, stage_index])
        if band is not None:
            panel.fill_between(predicted, band[0], band[1], color=colour, alpha=0.18, linewidth=0)

        panel.plot([0, 1], [0, 1], color="0.4", linestyle="--", linewidth=1.0, label="perfect")
        panel.plot(predicted, observed, color=colour, marker=marker, markersize=5, linewidth=1.4,
                   label=stage)
        for x, y, count in zip(predicted, observed, counts, strict=True):
            panel.annotate(str(count), (x, y), textcoords="offset points", xytext=(0, 6),
                           fontsize=6, color="0.35", ha="center")

        marker_label = f"{stage} (frozen)" if stage == chosen else stage
        panel.set_title(
            f"{marker_label} -- ECE {ece(labels, scores):.4f}  MCE {mce(labels, scores):.4f}  "
            f"Brier {brier(labels, scores):.4f}",
            fontsize=9,
        )
        panel.set_xlabel("predicted probability")
        panel.grid(alpha=0.25, linewidth=0.6)
        panel.set_xlim(0, 1)
        panel.set_ylim(0, 1)
        panel.legend(fontsize=8, loc="upper left")

    panels[0].set_ylabel("observed frequency")
    figure.suptitle(f"{model_name} -- {fold_id} (test window, {test.height} rows)", fontsize=11)
    figure.text(
        0.5,
        0.005,
        f"Bins are {n_bins} equal-mass, not equal-width; the number beside each point is that "
        f"bin's row count. The band is a {BAND_REPLICATIONS}-replication percentile bootstrap "
        "of the observed rate, resampling the rows -- so the bin boundaries move with it.",
        ha="center",
        fontsize=7,
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("Wrote %s", path)
    return path


def drift_figure(
    drift: pl.DataFrame, *, path: Path, candidates: list[CandidateSpec] | None = None
) -> Path | None:
    """ECE against fold, one line per stage, one panel per model.

    ``covid_shift`` is drawn as a separate marked point and **never joined into the
    quarterly line**: it is a different regime, its calibration window is the months
    Chicago's inspection programme was suspended, and connecting it would imply a
    continuity that does not exist.
    """
    specs = candidates or list(CANDIDATE_REGISTRY)
    names = [c.name for c in specs if c.name in set(drift["model_name"].to_list())]
    if not names:
        logger.info("Skipping drift figure: no model in the drift table")
        return None

    _, plt = _matplotlib()
    figure, axes = plt.subplots(len(names), 1, figsize=(10.0, 3.2 * len(names)), sharex=True)
    panels = axes if isinstance(axes, np.ndarray) else np.asarray([axes])

    quarterly = drift.filter(pl.col("fold_set") == "quarterly").sort("fold_index")
    covid = drift.filter(pl.col("fold_set") == "covid_shift")
    labels = quarterly.filter(pl.col("model_name") == names[0]).filter(
        pl.col("stage") == STAGE_UNCALIBRATED
    )["fold_id"].to_list()

    for panel, name in zip(panels, names, strict=True):
        for stage, (colour, marker) in STAGE_STYLE.items():
            series = quarterly.filter(
                (pl.col("model_name") == name) & (pl.col("stage") == stage)
            ).sort("fold_index")
            if series.height == 0:
                continue
            panel.plot(
                range(series.height),
                [float(v) for v in series["ece"].to_list()],
                color=colour,
                marker=marker,
                markersize=4,
                linewidth=1.3,
                label=stage,
            )
        point = covid.filter((pl.col("model_name") == name) & (pl.col("stage") == "selected"))
        if point.height:
            panel.axhline(
                float(point["ece"][0]),
                color="#8c564b",
                linestyle=":",
                linewidth=1.2,
                label="covid_shift (selected, not in the series)",
            )
        panel.set_title(name, fontsize=10)
        panel.set_ylabel("ECE")
        panel.grid(alpha=0.25, linewidth=0.6)
        panel.legend(fontsize=7, loc="best", ncol=2)

    panels[-1].set_xticks(range(len(labels)))
    panels[-1].set_xticklabels([label.replace("quarterly-", "") for label in labels],
                               rotation=45, fontsize=7, ha="right")
    panels[-1].set_xlabel("test quarter")
    figure.text(
        0.5,
        0.004,
        "Each point's ECE is measured on that fold's test window, with a calibrator fitted on "
        "that fold's calibration window only -- no calibrator is refitted per test quarter. "
        "covid_shift is a different regime and is drawn as a reference line, never joined into "
        "the quarterly series.",
        ha="center",
        fontsize=7,
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("Wrote %s", path)
    return path


def render(
    tables: dict[str, pl.DataFrame],
    *,
    destination: Path,
    candidates: list[CandidateSpec] | None = None,
) -> list[Path]:
    """Every figure one run produces.

    Reliability diagrams are drawn for the latest quarterly fold and for ``covid_shift``,
    for each candidate -- the representative-fold convention ``neural/build.py`` uses, plus
    the shift fold, because the point of the shift fold is that it looks different.
    """
    specs = candidates or list(CANDIDATE_REGISTRY)
    drift = tables["calibration_drift"]
    base = tables["calibration_base_scores"]
    calibrated = tables["calibrated_predictions"]
    written: list[Path] = []

    quarterly = drift.filter(pl.col("fold_set") == "quarterly")
    latest = (
        str(quarterly.sort("fold_index")["fold_id"][-1]) if quarterly.height else None
    )
    covid = drift.filter(pl.col("fold_set") == "covid_shift")
    covid_id = str(covid["fold_id"][0]) if covid.height else None

    for spec in specs:
        for fold_id in (latest, covid_id):
            if fold_id is None:
                continue
            path = destination / f"calibration_reliability_{spec.name}_{fold_id}.png"
            result = reliability_figure(
                base,
                calibrated,
                tables["calibrator_parameters"],
                tables["calibrator_isotonic_breakpoints"],
                model_name=spec.name,
                fold_id=fold_id,
                path=path,
            )
            if result is not None:
                written.append(result)

    drift_path = drift_figure(
        drift, path=destination / "calibration_ece_drift.png", candidates=specs
    )
    if drift_path is not None:
        written.append(drift_path)
    return written


__all__ = [
    "BAND_REPLICATIONS",
    "MIN_RELIABILITY_ROWS",
    "drift_figure",
    "reliability_figure",
    "render",
]
