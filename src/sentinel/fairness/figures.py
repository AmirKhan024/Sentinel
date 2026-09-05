"""Figures for Component 12. Every one answers a stated question; none is decorative.

Drawn only from the persisted tables -- never from an in-memory object -- so a figure can be
regenerated from the artifact alone and cannot silently disagree with it.

Three rules this component needs that the others did not.

**Insufficient support is drawn, not hidden.** A chart of only the groups that qualified would
show the same picture as a chart of a city with no small neighbourhoods. Where a figure shows
groups, the count that fell below the floor is stated on the axis label or the caption.

**The display policy is stated on the figure.** Seventy-eight tiny labels is an unreadable
chart, so figures show the best-supported ``DISPLAY_TOP_N``. That makes every figure a *view*,
and the caption says so, with the full table named as the source of truth.

**No figure gets a title that reads as a verdict.** "Fairness by neighbourhood" is a sentence
this artifact cannot support. The titles name what was measured.

A figure that cannot be drawn honestly returns ``None`` and logs, rather than raising or
drawing something misleading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from sentinel.fairness.definitions import (
    DISPLAY_TOP_N,
    RELIABILITY_GROUPS,
    SUPPORT_MIN_ROWS,
    Grain,
    GroupStatus,
    Stage,
)

logger = logging.getLogger(__name__)

#: Below this a per-group panel is a handful of points pretending to be a distribution.
MIN_GROUPS = 4

DISPLAY_CAPTION = (
    f"Shows the {DISPLAY_TOP_N} best-supported groups. The full table is the source of "
    "truth; this figure is a view of it."
)
SUPPORT_CAPTION = (
    f"Only groups with >= {SUPPORT_MIN_ROWS} rows and both classes present are plotted. "
    "Groups below the floor are counted in the axis label, not dropped from the artifact."
)
NOT_A_VERDICT_CAPTION = (
    "A measured difference in model behaviour across geography. NOT evidence of "
    "discrimination, not causal, and not a protected-class finding: no demographic variable "
    "is observed anywhere in this project. See ADR 0035."
)
BASE_RATE_CAPTION = (
    "Outcome rates differ across neighbourhoods before any model is involved, so a working "
    "risk model is EXPECTED to select at different rates. Parity is not the target."
)


#: The fold set model-level figures are drawn for. See the note in ``render``.
MODEL_FOLD_SET = "quarterly"

#: Suffix Component 9 appends to a base model name when it calibrates it.
CALIBRATED_SUFFIX = "_platt"


def base_model_name(model_name: str) -> str:
    """The uncalibrated ancestor of a calibrated model name.

    Component 9 names a calibrated model ``"<base>_<method>"`` -- deliberately, so that an
    uncalibrated probability can never be described as calibrated (MEMORY invariant 71).
    Component 11 keys its attributions by the base name. So the two artifacts this component
    joins speak different names for the same model, and the translation lives here rather
    than being written out at a call site where getting it wrong produces no error at all.
    """
    return model_name.removesuffix(CALIBRATED_SUFFIX)


def _matplotlib() -> tuple[Any, Any]:
    """Import matplotlib with a headless backend, chosen before pyplot loads.

    Without it matplotlib picks a backend from the environment, which on a headless runner is
    a different one than on this machine, and a figure that renders differently per
    environment is not an artifact.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


def _save(figure: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    _, plt = _matplotlib()
    plt.close(figure)
    logger.info("Wrote %s", path)
    return path


def _caption(figure: Any, text: str) -> None:
    figure.text(0.5, -0.04, text, ha="center", va="top", fontsize=7, wrap=True, color="#444444")


def _supported(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("group_status") == GroupStatus.SUPPORTED.value)


def _pooled(frame: pl.DataFrame, fold_set: str) -> pl.DataFrame:
    return frame.filter(
        (pl.col("grain") == Grain.FOLD_SET.value) & (pl.col("fold_set") == fold_set)
    )


# --- 1. representation and support -------------------------------------------


def representation(
    support: pl.DataFrame, definition: str, fold_set: str, destination: Path
) -> Path | None:
    """How many rows each group contributes, and where the support floor cuts.

    The first figure deliberately, because it is the one that says what the rest of the audit
    could and could not measure. The floor is drawn as a line rather than applied as a filter
    so a reader sees the groups that fell below it.
    """
    _, plt = _matplotlib()
    rows = _pooled(support, fold_set).filter(pl.col("group_definition") == definition)
    if rows.height < MIN_GROUPS:
        logger.info("Too few %s groups for a representation figure", definition)
        return None
    top = rows.sort("n_rows", descending=True).head(DISPLAY_TOP_N).reverse()
    below = rows.filter(pl.col("ranking_status") != GroupStatus.SUPPORTED.value).height

    figure, axis = plt.subplots(figsize=(8, max(3.0, 0.28 * top.height)))
    colours = [
        "#4c72b0" if status == GroupStatus.SUPPORTED.value else "#c44e52"
        for status in top["ranking_status"].to_list()
    ]
    axis.barh(top["group_value"].to_list(), top["n_rows"].to_list(), color=colours)
    axis.axvline(SUPPORT_MIN_ROWS, color="#333333", linestyle="--", linewidth=1)
    axis.set_xlabel(
        f"held-out rows (dashed line = {SUPPORT_MIN_ROWS}-row floor; "
        f"{below} of {rows.height} groups fall below it)"
    )
    axis.set_ylabel(definition)
    axis.set_title(f"Evaluated population by {definition} -- {fold_set}")
    _caption(figure, DISPLAY_CAPTION + " Red = insufficient support.")
    return _save(figure, destination / f"fairness_representation_{definition}_{fold_set}.png")


# --- 2. outcome base rates ----------------------------------------------------


def base_rates(
    support: pl.DataFrame, definition: str, fold_set: str, destination: Path
) -> Path | None:
    """What the outcome rate already looks like, before any model is evaluated.

    Drawn before any model figure because it is the context every one of them needs: a
    difference in selection rate across groups whose outcome rates differ by thirty points is
    the expected behaviour of a working risk model, not a defect.
    """
    _, plt = _matplotlib()
    rows = _pooled(support, fold_set).filter(
        (pl.col("group_definition") == definition)
        & (pl.col("ranking_status") == GroupStatus.SUPPORTED.value)
    )
    if rows.height < MIN_GROUPS:
        logger.info("Too few supported %s groups for a base-rate figure", definition)
        return None
    top = rows.sort("base_rate").head(DISPLAY_TOP_N)
    overall = float(rows["n_positive"].sum()) / float(rows["n_rows"].sum())

    figure, axis = plt.subplots(figsize=(8, max(3.0, 0.28 * top.height)))
    axis.barh(top["group_value"].to_list(), top["base_rate"].to_list(), color="#55a868")
    axis.axvline(overall, color="#333333", linestyle="--", linewidth=1)
    axis.set_xlabel(f"outcome rate (dashed line = pooled {overall:.3f})")
    axis.set_ylabel(definition)
    axis.set_title(f"Priority-citation rate by {definition} -- {fold_set}")
    _caption(figure, BASE_RATE_CAPTION + " " + SUPPORT_CAPTION)
    return _save(figure, destination / f"fairness_base_rates_{definition}_{fold_set}.png")


# --- 3. ranking performance ---------------------------------------------------


def ranking_by_group(
    metrics: pl.DataFrame, model: str, definition: str, fold_set: str, destination: Path
) -> Path | None:
    """Does the ranking work equally well inside every group?

    ROC-AUC, because it is one of only two metrics here that does not move with the base
    rate -- and the base rate moves by thirty points across these groups, so a PR-AUC panel
    would mostly be a picture of prevalence.
    """
    _, plt = _matplotlib()
    rows = _supported(
        _pooled(metrics, fold_set).filter(
            (pl.col("model_name") == model)
            & (pl.col("group_definition") == definition)
            & (pl.col("metric") == "roc_auc")
            & (pl.col("stage") == Stage.CALIBRATED.value)
        )
    ).drop_nulls("value")
    if rows.height < MIN_GROUPS:
        logger.info("Too few supported %s groups for a ranking figure", definition)
        return None
    top = rows.sort("value").head(DISPLAY_TOP_N)

    figure, axis = plt.subplots(figsize=(8, max(3.0, 0.28 * top.height)))
    axis.barh(top["group_value"].to_list(), top["value"].to_list(), color="#4c72b0")
    axis.axvline(0.5, color="#c44e52", linestyle="--", linewidth=1)
    axis.set_xlabel("ROC-AUC (dashed line = 0.5, no better than random within the group)")
    axis.set_ylabel(definition)
    axis.set_title(f"Within-group ranking quality -- {model}, {definition}, {fold_set}")
    _caption(figure, NOT_A_VERDICT_CAPTION + " " + DISPLAY_CAPTION)
    return _save(figure, destination / f"fairness_ranking_{model}_{definition}_{fold_set}.png")


# --- 4. calibration before and after -----------------------------------------


def calibration_before_after(
    calibration: pl.DataFrame, model: str, definition: str, fold_set: str, destination: Path
) -> Path | None:
    """Did Component 9's global calibration improvement reach every group?

    The diagonal is the figure. A point below it is a group whose ECE fell; a point above it
    is a group that got *worse* while the global number improved -- and finding those is the
    reason section 18 of the brief exists.
    """
    _, plt = _matplotlib()
    rows = (
        _pooled(calibration, fold_set)
        .filter(
            (pl.col("model_name") == model)
            & (pl.col("group_definition") == definition)
            & (pl.col("metric") == "ece")
            & (pl.col("group_status") == GroupStatus.SUPPORTED.value)
        )
        .drop_nulls(["base_value", "calibrated_value"])
    )
    if rows.height < MIN_GROUPS:
        logger.info("Too few supported %s groups for a calibration figure", definition)
        return None

    base = rows["base_value"].to_list()
    calibrated = rows["calibrated_value"].to_list()
    worsened = rows.filter(~pl.col("improved")).height

    figure, axis = plt.subplots(figsize=(5.5, 5.5))
    axis.scatter(base, calibrated, s=22, color="#4c72b0", alpha=0.85)
    limit = max([*base, *calibrated, 0.01]) * 1.1
    axis.plot([0, limit], [0, limit], color="#333333", linestyle="--", linewidth=1)
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel("group ECE, uncalibrated base score")
    axis.set_ylabel("group ECE, Platt-calibrated probability")
    axis.set_title(f"Did calibration reach every group? -- {model}, {definition}, {fold_set}")
    _caption(
        figure,
        f"Points above the diagonal got WORSE: {worsened} of {rows.height} supported groups. "
        "A global ECE improvement does not guarantee a group-level one. " + SUPPORT_CAPTION,
    )
    return _save(figure, destination / f"fairness_calibration_{model}_{definition}_{fold_set}.png")


# --- 5. top-k representation --------------------------------------------------


def topk_representation(
    priority: pl.DataFrame,
    model: str,
    definition: str,
    fold_set: str,
    k_name: str,
    destination: Path,
) -> Path | None:
    """Who appears in the priority set, relative to their share of the population?

    A ratio rather than two bars, because the eye is bad at comparing two nearly-equal bars
    and the whole question is how far from 1.0 each group sits.
    """
    _, plt = _matplotlib()
    rows = _supported(
        _pooled(priority, fold_set).filter(
            (pl.col("model_name") == model)
            & (pl.col("group_definition") == definition)
            & (pl.col("k_name") == k_name)
            & (pl.col("stage") == Stage.CALIBRATED.value)
        )
    ).drop_nulls("selection_rate_ratio")
    if rows.height < MIN_GROUPS:
        logger.info("Too few supported %s groups for a top-k figure", definition)
        return None
    top = rows.sort("selection_rate_ratio").head(DISPLAY_TOP_N)

    figure, axis = plt.subplots(figsize=(8, max(3.0, 0.28 * top.height)))
    axis.barh(top["group_value"].to_list(), top["selection_rate_ratio"].to_list(), color="#8172b2")
    axis.axvline(1.0, color="#333333", linestyle="--", linewidth=1)
    axis.set_xlabel("selection rate / overall selection rate (1.0 = proportionate)")
    axis.set_ylabel(definition)
    axis.set_title(f"Share of the priority set at {k_name} -- {model}, {definition}, {fold_set}")
    _caption(figure, BASE_RATE_CAPTION + " " + DISPLAY_CAPTION)
    return _save(
        figure,
        destination / f"fairness_topk_{model}_{definition}_{k_name}_{fold_set}.png",
    )


# --- 6. positive-outcome capture ---------------------------------------------


def capture_by_group(
    priority: pl.DataFrame,
    model: str,
    definition: str,
    fold_set: str,
    k_name: str,
    destination: Path,
) -> Path | None:
    """Of each group's actual violations, how many did the priority set contain?

    The effectiveness question, and the one most likely to matter operationally. A group can
    be selected at an ordinary rate and still have less of its risk found -- selected often
    and selected badly -- which is invisible in the representation figure alone.
    """
    _, plt = _matplotlib()
    rows = _supported(
        _pooled(priority, fold_set).filter(
            (pl.col("model_name") == model)
            & (pl.col("group_definition") == definition)
            & (pl.col("k_name") == k_name)
            & (pl.col("stage") == Stage.CALIBRATED.value)
        )
    ).drop_nulls("capture_rate")
    if rows.height < MIN_GROUPS:
        logger.info("Too few supported %s groups for a capture figure", definition)
        return None
    top = rows.sort("capture_rate").head(DISPLAY_TOP_N)
    overall = float(rows["overall_capture_rate"].head(1).item() or 0.0)

    figure, axis = plt.subplots(figsize=(8, max(3.0, 0.28 * top.height)))
    axis.barh(top["group_value"].to_list(), top["capture_rate"].to_list(), color="#dd8452")
    axis.axvline(overall, color="#333333", linestyle="--", linewidth=1)
    axis.set_xlabel(f"share of the group's positives captured (dashed = overall {overall:.3f})")
    axis.set_ylabel(definition)
    axis.set_title(f"Positive-outcome capture at {k_name} -- {model}, {definition}, {fold_set}")
    _caption(
        figure,
        "Representation in the top k and effectiveness of the top k are different "
        "questions; this is the second. " + DISPLAY_CAPTION,
    )
    return _save(
        figure,
        destination / f"fairness_capture_{model}_{definition}_{k_name}_{fold_set}.png",
    )


# --- 7. missingness -----------------------------------------------------------


def missingness_by_group(
    missingness: pl.DataFrame, definition: str, fold_set: str, destination: Path
) -> Path | None:
    """Is the data itself distributed evenly across groups?

    Component 11 measured a missingness indicator ranking second and third in importance for
    two of four models, so an unevenly distributed absence is an unevenly distributed input.
    This figure measures the distribution and claims nothing about its cause.
    """
    _, plt = _matplotlib()
    rows = _supported(
        _pooled(missingness, fold_set).filter(pl.col("group_definition") == definition)
    )
    if rows.height < MIN_GROUPS:
        logger.info("Too few supported %s groups for a missingness figure", definition)
        return None
    indicators = sorted(rows["indicator"].unique().to_list())

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for indicator in indicators:
        subset = rows.filter(pl.col("indicator") == indicator).sort("missing_rate")
        axis.plot(
            range(subset.height),
            subset["missing_rate"].to_list(),
            marker="o",
            markersize=3,
            linewidth=1,
            label=indicator,
        )
    axis.set_xlabel(f"supported {definition} groups, sorted by rate within each indicator")
    axis.set_ylabel("share of rows with the family missing")
    axis.set_title(f"Feature missingness by {definition} -- {fold_set}")
    axis.legend(fontsize=7)
    _caption(
        figure,
        "Missingness is not unfair by definition: 'we have never inspected this place' is a "
        "true and relevant fact, and removing the feature would not undo the inequality in "
        "inspection history behind it. " + SUPPORT_CAPTION,
    )
    return _save(figure, destination / f"fairness_missingness_{definition}_{fold_set}.png")


# --- 8. disparity drift -------------------------------------------------------


def disparity_drift(
    disparity: pl.DataFrame, model: str, definition: str, destination: Path
) -> Path | None:
    """Is the gap between groups moving, quarter by quarter?

    Drawn only where the per-fold disparity was computable. The support policy means that is
    usually few folds, and the axis label says how many rather than interpolating a line
    across the quarters where no group cleared the floor.
    """
    _, plt = _matplotlib()
    rows = disparity.filter(
        (pl.col("grain") == Grain.FOLD.value)
        & (pl.col("fold_set") == "quarterly")
        & (pl.col("model_name") == model)
        & (pl.col("group_definition") == definition)
        & (pl.col("measure") == "spread")
        & (pl.col("stage") == Stage.CALIBRATED.value)
        & (pl.col("metric").is_in(["roc_auc", "ece"]))
    ).drop_nulls("value")
    if rows.height < MIN_GROUPS:
        logger.info("Too few per-fold %s disparities to draw drift", definition)
        return None

    figure, axis = plt.subplots(figsize=(8, 4))
    for metric in sorted(rows["metric"].unique().to_list()):
        subset = rows.filter(pl.col("metric") == metric).sort("fold_id")
        axis.plot(
            subset["fold_id"].to_list(),
            subset["value"].to_list(),
            marker="o",
            markersize=4,
            label=f"{metric} ({subset.height} folds measured)",
        )
    axis.set_xlabel("fold (only folds where the disparity was computable)")
    axis.set_ylabel("max - min across supported groups")
    axis.set_title(f"Group disparity over time -- {model}, {definition}, quarterly")
    axis.tick_params(axis="x", rotation=90, labelsize=7)
    axis.legend(fontsize=7)
    _caption(
        figure,
        "Gaps in the series are quarters where no group cleared the support floor, not "
        "quarters where the disparity was zero. Group shares also move across folds, so a "
        "change here has two candidate explanations.",
    )
    return _save(figure, destination / f"fairness_disparity_drift_{model}_{definition}.png")


# --- 9. covid comparison ------------------------------------------------------


def covid_comparison(support: pl.DataFrame, definition: str, destination: Path) -> Path | None:
    """How much less can be measured under the distribution shift?

    A support figure rather than a metric figure, and that is the finding: the covid_shift
    window holds more rows than any single quarter and supports far fewer groups, because the
    inspection programme was suspended and the establishments seen during it were not a
    cross-section of the city.
    """
    _, plt = _matplotlib()
    counts: dict[str, tuple[int, int]] = {}
    for fold_set in ("quarterly", "covid_shift"):
        rows = _pooled(support, fold_set).filter(pl.col("group_definition") == definition)
        if rows.is_empty():
            return None
        counts[fold_set] = (
            rows.filter(pl.col("ranking_status") == GroupStatus.SUPPORTED.value).height,
            rows.height,
        )

    figure, axis = plt.subplots(figsize=(5.5, 4))
    labels = list(counts)
    supported = [counts[label][0] for label in labels]
    observed = [counts[label][1] for label in labels]
    axis.bar(labels, observed, color="#cccccc", label="groups observed")
    axis.bar(labels, supported, color="#4c72b0", label="groups supported")
    axis.set_ylabel(f"{definition} groups")
    axis.set_title(f"Measurable groups by fold set -- {definition}")
    axis.legend(fontsize=8)
    _caption(
        figure,
        "covid_shift is reported as a separate stress-test observation and is never averaged "
        "into a quarterly mean. One fold cannot support a trend.",
    )
    return _save(figure, destination / f"fairness_covid_support_{definition}.png")


# --- 10. attribution profiles -------------------------------------------------


def attribution_divergence(
    profiles: pl.DataFrame, model: str, definition: str, fold_set: str, destination: Path
) -> Path | None:
    """How much does each group's feature profile differ from the model's overall one?

    Descriptive. A low correlation says the model's *reliance* on features differs for that
    population. It is not evidence of discrimination, it is not causal, and per ADR 0030 an
    attribution is not a quality measure in the first place.
    """
    _, plt = _matplotlib()
    rows = (
        profiles.filter(
            (pl.col("model_name") == model)
            & (pl.col("group_definition") == definition)
            & (pl.col("fold_set") == fold_set)
        )
        .drop_nulls("profile_spearman")
        .group_by("group_value")
        .agg(pl.col("profile_spearman").first(), pl.col("n_rows").first())
    )
    if rows.height < MIN_GROUPS:
        logger.info("Too few supported %s attribution profiles for %s", definition, model)
        return None
    top = rows.sort("profile_spearman").head(DISPLAY_TOP_N)

    figure, axis = plt.subplots(figsize=(8, max(3.0, 0.28 * top.height)))
    axis.barh(top["group_value"].to_list(), top["profile_spearman"].to_list(), color="#937860")
    axis.axvline(1.0, color="#333333", linestyle="--", linewidth=1)
    axis.set_xlabel("Spearman rho of the group's feature ranking against the model's overall")
    axis.set_ylabel(definition)
    axis.set_title(f"Feature-reliance divergence -- {model}, {definition}, {fold_set}")
    _caption(
        figure,
        "Differences in attribution describe model behaviour, not discrimination and not "
        "causality. " + DISPLAY_CAPTION,
    )
    return _save(figure, destination / f"fairness_attribution_{model}_{definition}_{fold_set}.png")


# --- 11. reliability diagrams -------------------------------------------------


def group_reliability(
    metrics: pl.DataFrame, model: str, definition: str, fold_set: str, destination: Path
) -> Path | None:
    """Calibration slope and ECE for the best-supported groups, side by side.

    A slope panel rather than a per-group reliability curve: the curve would need the raw
    probabilities, which are a prediction artifact rather than a fairness one, and drawing it
    from a re-read of that artifact would put a second copy of Component 9's diagnostic in
    this component. The slope is the same information in one number per group.
    """
    _, plt = _matplotlib()
    rows = _supported(
        _pooled(metrics, fold_set).filter(
            (pl.col("model_name") == model)
            & (pl.col("group_definition") == definition)
            & (pl.col("metric") == "calibration_slope")
            & (pl.col("stage") == Stage.CALIBRATED.value)
        )
    ).drop_nulls("value")
    if rows.height < RELIABILITY_GROUPS:
        logger.info("Too few supported %s groups for a slope figure", definition)
        return None
    top = rows.sort("n_rows", descending=True).head(DISPLAY_TOP_N).sort("value")

    figure, axis = plt.subplots(figsize=(8, max(3.0, 0.28 * top.height)))
    axis.barh(top["group_value"].to_list(), top["value"].to_list(), color="#64b5cd")
    axis.axvline(1.0, color="#333333", linestyle="--", linewidth=1)
    axis.set_xlabel("calibration slope (1.0 = perfectly calibrated; below = underconfident)")
    axis.set_ylabel(definition)
    axis.set_title(f"Within-group calibration slope -- {model}, {definition}, {fold_set}")
    _caption(figure, SUPPORT_CAPTION + " " + DISPLAY_CAPTION)
    return _save(figure, destination / f"fairness_slope_{model}_{definition}_{fold_set}.png")


# --- entry point --------------------------------------------------------------


def render(tables: dict[str, pl.DataFrame], *, destination: Path) -> list[Path]:
    """Draw every figure the tables can support, and skip the rest.

    ``None`` from a figure function is a normal outcome, not an error: a fold set with too few
    supported groups genuinely has no honest chart, and drawing one anyway is how a figure
    ends up saying more than the data does.
    """
    support = tables.get("fairness_group_support", pl.DataFrame())
    metrics = tables.get("fairness_group_metrics", pl.DataFrame())
    calibration = tables.get("fairness_group_calibration", pl.DataFrame())
    priority = tables.get("fairness_priority_audit", pl.DataFrame())
    missing = tables.get("fairness_group_missingness", pl.DataFrame())
    disparity_table = tables.get("fairness_disparity", pl.DataFrame())
    profiles = tables.get("fairness_attribution_profiles", pl.DataFrame())

    if support.is_empty():
        logger.info("No support table; drawing no figures")
        return []

    definitions = sorted(support["group_definition"].unique().to_list())
    fold_sets = sorted(support["fold_set"].unique().to_list())
    models = sorted(metrics["model_name"].unique().to_list()) if not metrics.is_empty() else []
    # One representative cutoff for the priority figures. The percentage cutoff rather than a
    # capacity one, because it means the same fraction in every fold set and the covid window
    # is nineteen months long.
    k_name = "k_pct_05"

    paths: list[Path] = []
    for definition in definitions:
        # Population-level figures are drawn for every fold set: what the data looks like
        # under the distribution shift is a finding in its own right.
        for fold_set in fold_sets:
            for figure in (
                representation(support, definition, fold_set, destination),
                base_rates(support, definition, fold_set, destination),
                missingness_by_group(missing, definition, fold_set, destination),
            ):
                if figure is not None:
                    paths.append(figure)
        covid = covid_comparison(support, definition, destination)
        if covid is not None:
            paths.append(covid)

        # Model-level figures are drawn for the quarterly fold set only, and that is a
        # measurement rather than a saving. Only 11 of 78 community areas and 14 of 69 ZIPs
        # clear the support floor inside the covid_shift window, so a per-model covid panel
        # would be a chart of a dozen groups presented at the same visual weight as one of
        # fifty -- and no trend may be claimed from that fold anyway. What covid *can* say is
        # in `covid_comparison`, which is about how much less is measurable there.
        for model in models:
            candidates = [
                ranking_by_group(metrics, model, definition, MODEL_FOLD_SET, destination),
                calibration_before_after(
                    calibration, model, definition, MODEL_FOLD_SET, destination
                ),
                group_reliability(metrics, model, definition, MODEL_FOLD_SET, destination),
                topk_representation(
                    priority, model, definition, MODEL_FOLD_SET, k_name, destination
                ),
                capture_by_group(priority, model, definition, MODEL_FOLD_SET, k_name, destination),
                disparity_drift(disparity_table, model, definition, destination),
            ]
            if not profiles.is_empty():
                # Component 11 keys its artifact by the BASE model name, while every table
                # here is keyed by the calibrated one -- `xgboost` against `xgboost_platt`.
                # Looking the profile up under the calibrated name silently found nothing and
                # drew no figure, which is the quietest possible failure: a missing figure
                # looks exactly like a figure the data could not support.
                candidates.append(
                    attribution_divergence(
                        profiles, base_model_name(model), definition, MODEL_FOLD_SET, destination
                    )
                )
            paths.extend(path for path in candidates if path is not None)

    return sorted(set(paths))


__all__ = [
    "CALIBRATED_SUFFIX",
    "MIN_GROUPS",
    "MODEL_FOLD_SET",
    "attribution_divergence",
    "base_model_name",
    "base_rates",
    "calibration_before_after",
    "capture_by_group",
    "covid_comparison",
    "disparity_drift",
    "group_reliability",
    "missingness_by_group",
    "ranking_by_group",
    "render",
    "representation",
    "topk_representation",
]
