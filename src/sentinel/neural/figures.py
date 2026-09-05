"""The two figures the project specification requires, and nothing decorative.

**Learning curves.** Training loss and validation loss per epoch, with the restored epoch
marked. The validation curve is measured on rows carved from inside the training window,
and the axis label says so -- a reader who mistook it for held-out performance would be
reading an in-sample number as an out-of-sample one, which is the error this whole project
is organised against.

**The chain embedding projection.** t-SNE over the learned chain vectors, from
scikit-learn, which is already a dependency. UMAP was not added: it would be a new runtime
dependency for a second view of the same question, and the specification asks for "t-SNE
or UMAP".

A warning that belongs in the code and not only in the findings document: **a t-SNE plot is
not evidence of semantic structure.** t-SNE will produce clusters from isotropic noise. It
has a perplexity parameter that changes the picture, it does not preserve global distances,
and inter-cluster distance in the output is not interpretable. So the projection is
rendered, the parameters are recorded on the figure, and any claim about what the clusters
*mean* has to come from a measurement made elsewhere -- which is why ``build`` also emits
the raw vectors as a table. If the projection shows nothing, that is the finding.

Points are coloured only by information legitimately available for interpretation: the
size of the chain in the training window. Colouring by outcome rate would draw a picture
of the training labels and invite exactly the circular reading the specification warns
against.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sentinel.neural.definitions import INDEPENDENT_CHAIN, UNKNOWN_CATEGORY
from sentinel.neural.models import EpochRecord, FittedNetwork

logger = logging.getLogger(__name__)

#: Below this many categories a projection is not attempted. t-SNE on a handful of points
#: produces a picture that is entirely an artifact of the algorithm.
MIN_PROJECTION_POINTS = 30

#: Recorded on the figure and in the findings, because the picture depends on them.
TSNE_PERPLEXITY = 30.0
TSNE_SEED = 20260818
TSNE_ITERATIONS = 1000


def _matplotlib() -> tuple[Any, Any]:
    """Import matplotlib with a non-interactive backend.

    ``Agg`` is selected before ``pyplot`` is imported. Without it matplotlib picks a
    backend from the environment, which on a headless CI runner is a different one than on
    this machine, and a figure that renders differently per environment is not an artifact.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


def learning_curve_figure(fitted: FittedNetwork, path: Path, *, title: str | None = None) -> Path:
    """Render one fit's training and validation loss, marking the restored epoch."""
    _, plt = _matplotlib()
    epochs: Sequence[EpochRecord] = fitted.epochs
    if not epochs:
        raise ValueError(f"{fitted.spec.name}/{fitted.fold_id}: no epochs to plot")

    xs = [e.epoch for e in epochs]
    train = [e.train_loss for e in epochs]
    validation = [e.validation_loss for e in epochs]

    figure, axes = plt.subplots(figsize=(9, 5.5))
    axes.plot(xs, train, label="training loss", linewidth=1.8)
    axes.plot(xs, validation, label="inner-validation loss (in-sample)", linewidth=1.8)
    axes.axvline(
        fitted.best_epoch,
        linestyle="--",
        linewidth=1.2,
        color="0.35",
        label=f"restored epoch ({fitted.best_epoch})",
    )
    if fitted.final_epoch != fitted.best_epoch:
        axes.axvline(
            fitted.final_epoch,
            linestyle=":",
            linewidth=1.2,
            color="0.6",
            label=f"stopped at epoch {fitted.final_epoch}",
        )

    # Every learning-rate reduction, so a kink in the curve is attributable.
    previous = epochs[0].learning_rate
    labelled = False
    for record in epochs:
        if record.learning_rate != previous:
            axes.axvline(
                record.epoch,
                linestyle="-",
                linewidth=0.8,
                color="0.8",
                zorder=0,
                label=None if labelled else "learning-rate reduction",
            )
            labelled = True
            previous = record.learning_rate

    axes.set_xlabel("epoch")
    axes.set_ylabel("BCEWithLogitsLoss")
    axes.set_title(
        title
        or (
            f"{fitted.spec.name} — {fitted.fold_id}\n"
            f"lr {fitted.learning_rate:.0e}, seed {fitted.seed}, "
            f"{fitted.inner_train_rows:,} train / {fitted.inner_validation_rows:,} validation rows"
        ),
        fontsize=10,
    )
    axes.legend(fontsize=8, loc="best")
    axes.grid(alpha=0.25, linewidth=0.6)
    figure.text(
        0.5,
        0.005,
        "Validation rows are carved from the END of the training window "
        f"(from {fitted.inner_validation_start}). This curve is in-sample: it is the "
        "early-stopping signal, not a measure of performance.",
        ha="center",
        fontsize=7,
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("Wrote %s", path)
    return path


def project_embeddings(
    vectors: NDArray[np.float64], *, perplexity: float = TSNE_PERPLEXITY
) -> NDArray[np.float64]:
    """t-SNE to two dimensions, seeded.

    Perplexity is clamped below ``n_samples / 3``, which is scikit-learn's own constraint;
    exceeding it raises, and failing on a small ablation would be worse than adjusting
    and recording the adjustment.
    """
    from sklearn.manifold import TSNE

    n = int(vectors.shape[0])
    effective = float(min(perplexity, max(5.0, (n - 1) / 3.0)))
    model = TSNE(
        n_components=2,
        perplexity=effective,
        random_state=TSNE_SEED,
        init="pca",
        max_iter=TSNE_ITERATIONS,
    )
    return np.asarray(model.fit_transform(vectors), dtype=np.float64)


def embedding_figure(
    fitted: FittedNetwork,
    path: Path,
    *,
    family: str = "chain",
    sizes: dict[str, int] | None = None,
) -> Path | None:
    """Project one family's learned vectors and render them.

    Returns ``None`` when there are too few categories to project honestly. That is a
    result, not a failure: an empty figure would be less informative than its absence, and
    the caller records which happened.
    """
    _, plt = _matplotlib()
    table = fitted.embedding_for(family)
    categories = list(table.categories)
    vectors = np.asarray(table.vectors, dtype=np.float64)

    if len(categories) < MIN_PROJECTION_POINTS:
        logger.warning(
            "%s/%s: %d %s categories is too few to project; skipping",
            fitted.spec.name,
            fitted.fold_id,
            len(categories),
            family,
        )
        return None

    projected = project_embeddings(vectors)

    # The two reserved tokens are marked rather than hidden. Where UNKNOWN lands relative
    # to the real categories is one of the few genuinely interpretable things in the
    # picture: if it sits in the middle of the cloud, the network learned little that
    # distinguishes a known chain from an unknown one.
    reserved = {UNKNOWN_CATEGORY, INDEPENDENT_CHAIN}
    is_reserved = np.array([c in reserved for c in categories])

    weights = np.array(
        [float(sizes.get(c, 1)) if sizes else 1.0 for c in categories], dtype=np.float64
    )
    scale = 12.0 + 40.0 * (weights - weights.min()) / max(float(np.ptp(weights)), 1.0)

    figure, axes = plt.subplots(figsize=(8.5, 7.5))
    axes.scatter(
        projected[~is_reserved, 0],
        projected[~is_reserved, 1],
        s=scale[~is_reserved],
        alpha=0.55,
        linewidths=0,
        label=f"{family} categories ({int((~is_reserved).sum())})",
    )
    if is_reserved.any():
        axes.scatter(
            projected[is_reserved, 0],
            projected[is_reserved, 1],
            s=90,
            marker="X",
            color="crimson",
            label="reserved tokens (UNKNOWN, INDEPENDENT)",
        )
        for index in np.flatnonzero(is_reserved):
            axes.annotate(
                categories[index].strip("_"),
                (projected[index, 0], projected[index, 1]),
                fontsize=7,
                color="crimson",
                xytext=(4, 4),
                textcoords="offset points",
            )

    axes.set_title(
        f"{fitted.spec.name} — learned {family} embeddings, {fitted.fold_id}\n"
        f"t-SNE(perplexity={TSNE_PERPLEXITY:g}, seed={TSNE_SEED}) over "
        f"{vectors.shape[1]} dimensions, {len(categories)} categories",
        fontsize=10,
    )
    axes.set_xlabel("t-SNE 1 (no units; distances are not interpretable)")
    axes.set_ylabel("t-SNE 2")
    axes.legend(fontsize=8, loc="best")
    axes.grid(alpha=0.2, linewidth=0.6)
    if sizes:
        figure.text(
            0.5,
            0.005,
            "Point size = establishments per chain in the training window. t-SNE produces "
            "clusters from noise; apparent grouping here is NOT evidence of learned "
            "semantics without a separate measurement.",
            ha="center",
            fontsize=7,
            color="0.35",
        )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("Wrote %s", path)
    return path


def learning_rate_figure(
    scores: Sequence[tuple[str, Sequence[tuple[float, float]]]], path: Path
) -> Path:
    """Mean inner-validation PR-AUC against learning rate, one line per fold set."""
    _, plt = _matplotlib()
    figure, axes = plt.subplots(figsize=(8, 5))
    for label, series in scores:
        rates = [rate for rate, _ in series]
        values = [value for _, value in series]
        axes.plot(rates, values, marker="o", label=label, linewidth=1.8)
    axes.set_xscale("log")
    axes.set_xlabel("learning rate (log scale)")
    axes.set_ylabel("mean inner-validation PR-AUC (in-sample)")
    axes.set_title(
        "Learning-rate sensitivity, selected before any test window was opened", fontsize=10
    )
    axes.legend(fontsize=8)
    axes.grid(alpha=0.25, linewidth=0.6)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("Wrote %s", path)
    return path


def nearest_neighbours(
    fitted: FittedNetwork, *, family: str = "chain", k: int = 5, limit: int = 25
) -> list[tuple[str, tuple[str, ...]]]:
    """Cosine nearest neighbours in the raw embedding space, for the largest categories.

    This is the measurement the t-SNE picture cannot supply. Neighbours are computed in
    the space the network actually learned, not in the projection, so a claim about two
    chains being "close" means something checkable.
    """
    table = fitted.embedding_for(family)
    vectors = np.asarray(table.vectors, dtype=np.float64)
    if vectors.shape[0] <= k + 1:
        return []
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalised = vectors / np.where(norms == 0.0, 1.0, norms)
    similarity = normalised @ normalised.T
    np.fill_diagonal(similarity, -np.inf)

    reserved = {UNKNOWN_CATEGORY, INDEPENDENT_CHAIN}
    out: list[tuple[str, tuple[str, ...]]] = []
    for index, category in enumerate(table.categories):
        if category in reserved:
            continue
        order = np.argsort(-similarity[index])[:k]
        out.append((category, tuple(table.categories[j] for j in order)))
        if len(out) >= limit:
            break
    return out


__all__ = [
    "MIN_PROJECTION_POINTS",
    "TSNE_ITERATIONS",
    "TSNE_PERPLEXITY",
    "TSNE_SEED",
    "embedding_figure",
    "learning_curve_figure",
    "learning_rate_figure",
    "nearest_neighbours",
    "project_embeddings",
]
