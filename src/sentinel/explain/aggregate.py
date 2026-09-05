"""Global importance, stability, drift and representative cases. Pure -- no I/O.

Component 11 is not finished at a bar chart. A model can hold its ROC-AUC steady across
seventeen quarters while quietly changing which signals it leans on, and a single pooled
importance ranking would hide exactly that. So three questions are answered separately:

**What did the model lean on?** ``mean(|SHAP|)`` per feature, per fold, then aggregated
within a fold set -- with the standard deviation and the rank spread beside every mean,
because a mean importance quoted alone invites "feature X is the most important feature"
and that claim is only true if the ranks hold.

**Did that change over time?** Two metrics, chosen because they disagree usefully. Spearman
rank correlation sees the whole ranking including its noisy tail; top-k Jaccard sees only
whether the same features stayed at the top. A model can reorder its tail while keeping its
top ten, or swap two dominant features while every other rank holds, and one number would
call both of those the same thing.

**Which features moved, and by how much?** Per-feature rank travel across a fold set, with
"materially changed" decided against ``RANK_DRIFT_THRESHOLD`` -- a constant declared before
any rank was computed. A threshold chosen after seeing the ranks is a conclusion wearing a
criterion's clothes.

**``covid_shift`` is never pooled into a quarterly aggregate.** It is one fold, from a
period when the scheduling policy itself broke, and Component 6 already measured the model
ordering *inverting* on it. Everything below aggregates within a ``fold_set`` and never
across, so the separation is structural rather than a convention someone must remember.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from sentinel.explain.definitions import (
    RANK_DRIFT_THRESHOLD,
    REPRESENTATIVE_QUANTILES,
    TOP_K,
    origin_of,
)
from sentinel.explain.models import (
    DriftRow,
    FoldAttribution,
    ImportanceRow,
    RepresentativeCase,
    StabilityRow,
)

logger = logging.getLogger(__name__)

#: ``scope`` values on the importance table.
SCOPE_FOLD = "fold"
SCOPE_FOLD_SET = "fold_set"

#: The comparisons the stability table reports.
COMPARISON_CONSECUTIVE = "consecutive"
COMPARISON_FIRST_TO_LAST = "first_to_last"

#: Named in the manifest so the metric a claim rests on is never inferred.
STABILITY_METRICS: tuple[str, ...] = (
    "spearman rank correlation of mean_abs_shap ranks, ties averaged",
    f"top-{TOP_K} Jaccard overlap of the highest mean_abs_shap features",
)


# --- rank statistics ---------------------------------------------------------


def ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Descending ranks with ties averaged; the largest value gets rank 1.

    Ties are averaged rather than broken by position. Breaking them would make a rank
    depend on the order features happen to sit in the matrix, and two features with
    identical importance would then produce a spurious rank difference that the drift table
    would faithfully report as movement.
    """
    order = np.argsort(-values, kind="stable")
    out = np.empty(len(values), dtype=np.float64)
    out[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    for value in np.unique(values):
        tied = np.flatnonzero(values == value)
        if len(tied) > 1:
            out[tied] = out[tied].mean()
    return out


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation of two rank vectors.

    Zero rather than NaN when either vector is constant. A constant ranking means every
    feature tied, which is a degenerate but real answer -- emitting NaN would propagate
    through the aggregates and force every consumer to handle it.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = float(np.sqrt(float((xc**2).sum()) * float((yc**2).sum())))
    return float((xc * yc).sum() / denominator) if denominator else 0.0


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Intersection over union of two feature sets."""
    left, right = set(a), set(b)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def top_features(names: Sequence[str], values: Sequence[float], k: int) -> tuple[str, ...]:
    """The ``k`` highest-importance feature names, ties broken by name.

    The name tie-break is deliberate: without it the top-k set would depend on matrix order,
    and a Jaccard overlap computed on it would report movement that never happened.
    """
    paired = sorted(zip(names, values, strict=True), key=lambda item: (-item[1], item[0]))
    return tuple(name for name, _ in paired[:k])


# --- global importance -------------------------------------------------------


def fold_importance(attribution: FoldAttribution) -> list[ImportanceRow]:
    """Mean absolute and mean signed attribution per feature, for one (model, fold).

    Both are emitted. ``mean_abs_shap`` measures how much the model *used* a feature and is
    what the ranking is built on; ``mean_shap`` keeps the direction, which is the difference
    between "this feature moves the score a lot, both ways" and "this feature consistently
    raises risk". Reporting only the absolute value would hide the second question.
    """
    absolute = np.abs(attribution.values).mean(axis=0)
    signed = attribution.values.mean(axis=0)
    ordered = ranks(absolute)
    rows: list[ImportanceRow] = []
    for index, name in enumerate(attribution.feature_names):
        original, _ = origin_of(name)
        rows.append(
            ImportanceRow(
                model_name=attribution.model_name,
                fold_set=attribution.fold_set,
                fold_id=attribution.fold_id,
                scope=SCOPE_FOLD,
                feature_name=name,
                original_feature_name=original,
                mean_abs_shap=float(absolute[index]),
                mean_shap=float(signed[index]),
                rank=int(round(float(ordered[index]))),
                sd_abs_shap=None,
                mean_rank=None,
                sd_rank=None,
                best_rank=None,
                worst_rank=None,
                folds=1,
                rows=attribution.values.shape[0],
            )
        )
    return rows


def aggregate_importance(fold_rows: Sequence[ImportanceRow]) -> list[ImportanceRow]:
    """Fold-set level importance, aggregated within ``(model_name, fold_set)`` only.

    The aggregate rank is computed from the mean importance rather than by averaging the
    per-fold ranks, and the mean rank is reported beside it. The two can disagree, and when
    they do the disagreement is the finding: a feature that is second on most folds and
    twenty-fifth on two has a good mean rank and a poor mean importance, and a reader is
    entitled to see both rather than whichever the aggregation happened to produce.
    """
    grouped: dict[tuple[str, str], dict[str, list[ImportanceRow]]] = {}
    for row in fold_rows:
        if row.scope != SCOPE_FOLD:
            continue
        grouped.setdefault((row.model_name, row.fold_set), {}).setdefault(
            row.feature_name, []
        ).append(row)

    out: list[ImportanceRow] = []
    for (model_name, fold_set), by_feature in sorted(grouped.items()):
        names = sorted(by_feature)
        means = np.array([np.mean([r.mean_abs_shap for r in by_feature[n]]) for n in names])
        aggregate_ranks = ranks(means)
        for index, name in enumerate(names):
            members = by_feature[name]
            absolute = np.array([r.mean_abs_shap for r in members], dtype=np.float64)
            signed = np.array([r.mean_shap for r in members], dtype=np.float64)
            per_fold_ranks = np.array([r.rank for r in members], dtype=np.float64)
            original, _ = origin_of(name)
            out.append(
                ImportanceRow(
                    model_name=model_name,
                    fold_set=fold_set,
                    fold_id=None,
                    scope=SCOPE_FOLD_SET,
                    feature_name=name,
                    original_feature_name=original,
                    mean_abs_shap=float(absolute.mean()),
                    mean_shap=float(signed.mean()),
                    rank=int(round(float(aggregate_ranks[index]))),
                    # Population SD, not sample: these seventeen folds are the folds, not a
                    # draw from a larger set of them. Component 5 makes the same choice and
                    # warns that the spread is a fold-to-fold range rather than a
                    # confidence interval, because the folds share establishments.
                    sd_abs_shap=float(absolute.std()),
                    mean_rank=float(per_fold_ranks.mean()),
                    sd_rank=float(per_fold_ranks.std()),
                    best_rank=int(per_fold_ranks.min()),
                    worst_rank=int(per_fold_ranks.max()),
                    folds=len(members),
                    rows=int(sum(r.rows for r in members)),
                )
            )
    return out


# --- stability ---------------------------------------------------------------


def stability(fold_rows: Sequence[ImportanceRow]) -> list[StabilityRow]:
    """Rank correlation and top-k overlap between folds, within a fold set.

    Folds are ordered by ``fold_id``, which for the quarterly set is chronological because
    the id embeds the test quarter. A fold set with one fold -- ``covid_shift`` -- yields no
    rows rather than a self-comparison of 1.0, which would be a meaningless number that
    reads like evidence of stability.
    """
    per_fold: dict[tuple[str, str], dict[str, list[ImportanceRow]]] = {}
    for row in fold_rows:
        if row.scope != SCOPE_FOLD or row.fold_id is None:
            continue
        per_fold.setdefault((row.model_name, row.fold_set), {}).setdefault(row.fold_id, []).append(
            row
        )

    out: list[StabilityRow] = []
    for (model_name, fold_set), by_fold in sorted(per_fold.items()):
        fold_ids = sorted(by_fold)
        if len(fold_ids) < 2:
            logger.info(
                "%s/%s has %d fold(s); no stability comparison is possible",
                model_name,
                fold_set,
                len(fold_ids),
            )
            continue

        # Sorted by feature name on both sides of every comparison, so the two rank
        # vectors line up positionally without a join. Bound as a default argument rather
        # than closed over: a closure over the loop variable would silently profile the
        # last model's folds for every model.
        def profile(
            fold_id: str, folds_of: dict[str, list[ImportanceRow]] = by_fold
        ) -> tuple[list[str], list[float], list[float]]:
            rows = sorted(folds_of[fold_id], key=lambda r: r.feature_name)
            return (
                [r.feature_name for r in rows],
                [r.mean_abs_shap for r in rows],
                [float(r.rank) for r in rows],
            )

        pairs = [
            (COMPARISON_CONSECUTIVE, a, b) for a, b in zip(fold_ids, fold_ids[1:], strict=False)
        ]
        pairs.append((COMPARISON_FIRST_TO_LAST, fold_ids[0], fold_ids[-1]))

        for comparison, left, right in pairs:
            names_a, values_a, ranks_a = profile(left)
            names_b, values_b, ranks_b = profile(right)
            if names_a != names_b:
                raise ValueError(
                    f"{model_name}: folds {left} and {right} report different features, so "
                    "their rankings cannot be compared"
                )
            out.append(
                StabilityRow(
                    model_name=model_name,
                    fold_set=fold_set,
                    comparison=comparison,
                    from_fold_id=left,
                    to_fold_id=right,
                    spearman_rho=spearman(ranks_a, ranks_b),
                    top_k=TOP_K,
                    top_k_jaccard=jaccard(
                        top_features(names_a, values_a, TOP_K),
                        top_features(names_b, values_b, TOP_K),
                    ),
                    features=len(names_a),
                )
            )
    return out


# --- drift -------------------------------------------------------------------


def drift(fold_rows: Sequence[ImportanceRow]) -> list[DriftRow]:
    """Per-feature rank travel across a fold set.

    ``coefficient_of_variation`` is the SD of a feature's per-fold importance over its mean.
    It is ``None`` rather than infinity when the mean is zero, which happens for a feature
    the model never split on -- a real and informative case that a division would turn into
    a NaN the artifact would then have to explain.
    """
    grouped: dict[tuple[str, str], dict[str, list[ImportanceRow]]] = {}
    for row in fold_rows:
        if row.scope != SCOPE_FOLD or row.fold_id is None:
            continue
        grouped.setdefault((row.model_name, row.fold_set), {}).setdefault(
            row.feature_name, []
        ).append(row)

    out: list[DriftRow] = []
    for (model_name, fold_set), by_feature in sorted(grouped.items()):
        for name in sorted(by_feature):
            members = sorted(by_feature[name], key=lambda r: r.fold_id or "")
            if len(members) < 2:
                continue
            values = np.array([r.mean_abs_shap for r in members], dtype=np.float64)
            positions = np.array([r.rank for r in members], dtype=np.float64)
            mean = float(values.mean())
            sd = float(values.std())
            original, _ = origin_of(name)
            out.append(
                DriftRow(
                    model_name=model_name,
                    fold_set=fold_set,
                    feature_name=name,
                    original_feature_name=original,
                    first_fold_id=members[0].fold_id or "",
                    last_fold_id=members[-1].fold_id or "",
                    first_rank=members[0].rank,
                    last_rank=members[-1].rank,
                    best_rank=int(positions.min()),
                    worst_rank=int(positions.max()),
                    rank_range=int(positions.max() - positions.min()),
                    mean_abs_shap=mean,
                    sd_abs_shap=sd,
                    coefficient_of_variation=(sd / mean) if mean > 0 else None,
                    materially_changed=bool(
                        (positions.max() - positions.min()) >= RANK_DRIFT_THRESHOLD
                    ),
                )
            )
    return out


# --- representative cases ----------------------------------------------------


def representative_cases(
    attribution: FoldAttribution,
    *,
    base_scores: Mapping[str, float],
    calibrated: Mapping[str, tuple[float, str]] | None = None,
) -> list[RepresentativeCase]:
    """One high, one medium and one low **predicted-risk** case, deterministically.

    Selected by quantiles of the model's own committed probability, using the nearest-rank
    definition, with ``target_inspection_id`` breaking ties. Nothing about the outcome
    participates: a case chosen because the model was right about it would be storytelling
    with a reproducible rule bolted on, and the rule's reproducibility would make it more
    persuasive rather than less misleading.

    Quantiles of the *prediction*, not of ``prediction_value``: the base model's committed
    probability is the number a reader recognises, and it is a monotone transform of the
    log-odds being decomposed, so the two orderings are identical anyway.
    """
    ordered = sorted(attribution.row_ids, key=lambda row_id: (base_scores[row_id], row_id))
    if not ordered:
        return []

    out: list[RepresentativeCase] = []
    position_of = {row_id: index for index, row_id in enumerate(attribution.row_ids)}
    for tier, quantile in sorted(REPRESENTATIVE_QUANTILES.items()):
        # Nearest-rank: the smallest index whose cumulative share reaches the quantile.
        index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
        row_id = ordered[index]
        calibration = calibrated.get(row_id) if calibrated else None
        out.append(
            RepresentativeCase(
                model_name=attribution.model_name,
                fold_set=attribution.fold_set,
                fold_id=attribution.fold_id,
                tier=tier,
                quantile=quantile,
                target_inspection_id=row_id,
                base_value=attribution.base_value,
                prediction_value=float(attribution.output[position_of[row_id]]),
                base_score=float(base_scores[row_id]),
                calibrated_probability=calibration[0] if calibration else None,
                calibration_method=calibration[1] if calibration else None,
                output_space=attribution.output_space.value,
                method=attribution.method.value,
                is_exact=attribution.is_exact,
            )
        )
    return out


__all__ = [
    "COMPARISON_CONSECUTIVE",
    "COMPARISON_FIRST_TO_LAST",
    "SCOPE_FOLD",
    "SCOPE_FOLD_SET",
    "STABILITY_METRICS",
    "aggregate_importance",
    "drift",
    "fold_importance",
    "jaccard",
    "ranks",
    "representative_cases",
    "spearman",
    "stability",
    "top_features",
]
