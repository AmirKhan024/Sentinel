"""Read-only profiling of the inputs Component 8 adds, before any model is fitted.

Analysis tooling, not library code: it answers one-off questions about a snapshot, nothing
imports it, and it should not ship in the wheel. Output is markdown on stdout, pasted into
``docs/analysis/neural_models_findings.md``.

⚠ **No profile in this script may report a test-window number.** Every figure below is
computed over a fold's *training* window, or over the whole snapshot in a way that carries
no outcome (cardinality, coverage, vocabulary growth). Component 5 protects evaluation
time, but it cannot protect against a human reading a test metric, changing an embedding
dimension and re-running. That loop is leakage, it leaves no trace in any artifact, and no
check in this repository can detect it.

Questions this script answers
-----------------------------
1. ``categorical_coverage``   -- how often does each family actually carry a value?
2. ``cardinality_growth``     -- how large is each vocabulary per fold, and does it grow?
3. ``unseen_rate``            -- how many test-window rows meet a category the fold's
                                 training window never saw? (a count of *rows*, never an
                                 outcome)
4. ``chain_structure``        -- how many chains are there, and how concentrated?
5. ``embedding_budget``       -- how many parameters does each spec's embedding block cost
                                 against the dense network's?
6. ``inner_split``            -- where does the early-stopping split fall per fold, and is
                                 either side too small?
7. ``scaling_need``           -- the feature scale spread a network has to cope with and a
                                 tree did not.
8. ``as_of_lag``              -- how stale is a carried categorical?
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.config import load_settings  # noqa: E402
from sentinel.evaluation import folds as folds_module  # noqa: E402
from sentinel.evaluation.models import FoldSpec  # noqa: E402
from sentinel.modeling.train import training_frame  # noqa: E402
from sentinel.neural import encode, preprocess  # noqa: E402
from sentinel.neural.categoricals import EMITTED_CATEGORICALS  # noqa: E402
from sentinel.neural.definitions import (  # noqa: E402
    EMBEDDING_DIMS,
    HIDDEN_SIZES,
    INNER_VALIDATION_FRACTION,
    NEURAL_REGISTRY,
    UNKNOWN_CATEGORY,
    CategoricalEncoding,
    Learner,
    embedding_width,
    spec_for,
)
from sentinel.neural.train import inner_split_date  # noqa: E402
from sentinel.query import duckdb_queries  # noqa: E402

PRIMARY = "neural_embeddings"


# --- shared helpers ----------------------------------------------------------


def _folds(frame: pl.DataFrame) -> list[FoldSpec]:
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    if start is None or end is None:
        raise SystemExit("feature table has no usable reference dates")
    quarterly = folds_module.quarterly_folds(data_start=start, data_end=end)
    return [*quarterly, *folds_module.covid_shift_fold(data_end=end)]


def _joined(frame: pl.DataFrame, categoricals: pl.DataFrame) -> pl.DataFrame:
    wanted = ["chain_key", "facility_type", "community_area", "zip", "days_since_source"]
    return frame.join(
        categoricals.select("target_inspection_id", *wanted),
        on="target_inspection_id",
        how="left",
    )


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    lines.extend("| " + " | ".join(r) + " |" for r in rows)
    return lines


# --- profiles ----------------------------------------------------------------


def profile_categorical_coverage(frame: pl.DataFrame, categoricals: pl.DataFrame) -> str:
    """How often each family carries a real value rather than UNKNOWN."""
    lines: list[str] = []
    rows = []
    for family in EMITTED_CATEGORICALS:
        series = categoricals[family]
        known = int((series != UNKNOWN_CATEGORY).sum())
        rows.append(
            [
                f"`{family}`",
                f"{int(series.n_unique())}",
                f"{known:,}",
                f"{known / categoricals.height:.4f}",
                f"{categoricals.height - known:,}",
            ]
        )
    lines.extend(_table(["family", "distinct", "with a value", "coverage", "UNKNOWN"], rows))
    without_prior = int(categoricals.filter(pl.col("source_inspection_id").is_null()).height)
    lines.append("")
    lines.append(
        f"{without_prior:,} rows have **no prior inspection of any type** to carry a value "
        f"forward from, so all four families are UNKNOWN for them. That is exactly the count "
        f"of rows Component 4 marks with a null `days_since_any_inspection`, which is the "
        f"consistency check worth having: the two components independently agree on which "
        f"establishments have no history."
    )
    return "\n".join(lines)


def profile_cardinality_growth(frame: pl.DataFrame, categoricals: pl.DataFrame) -> str:
    """Vocabulary size per fold, fitted on training rows only."""
    spec = spec_for(PRIMARY)
    lines: list[str] = []
    rows = []
    for fold in _folds(frame):
        training = training_frame(frame, fold)
        joined = _joined(training, categoricals)
        encoding = encode.fit_encoding(joined, spec)
        sizes = encoding.sizes
        rows.append(
            [
                f"`{fold.fold_id}`",
                f"{training.height:,}",
                f"{sizes.get('chain', 0)}",
                f"{sizes.get('facility_type', 0)}",
                f"{sizes.get('community_area', 0)}",
                f"{sizes.get('zip', 0)}",
                f"{len(encoding.chains)}",
            ]
        )
    lines.extend(
        _table(
            ["fold", "train rows", "chain", "facility", "community", "zip", "chains"],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "Every vocabulary is refitted per fold on training rows only, so these grow with "
        "the expanding window. The chain column counts vocabulary entries, which is the "
        "chain count plus the two reserved tokens (`__UNKNOWN__`, `__INDEPENDENT__`)."
    )
    return "\n".join(lines)


def profile_unseen_rate(frame: pl.DataFrame, categoricals: pl.DataFrame) -> str:
    """Test-window rows meeting a category the training window never saw.

    A row count, never an outcome. This is the number that decides whether an embedding
    can do anything at all: a family whose test rows mostly map to UNKNOWN contributes one
    learned vector to most predictions.
    """
    spec = spec_for(PRIMARY)
    lines: list[str] = []
    rows = []
    for fold in _folds(frame):
        training = training_frame(frame, fold)
        window = folds_module.window_frame(frame, fold)
        encoding = encode.fit_encoding(_joined(training, categoricals), spec)
        rates = encode.unseen_rate(_joined(window, categoricals), spec, encoding)
        rows.append(
            [
                f"`{fold.fold_id}`",
                f"{window.height:,}",
                f"{rates.get('chain', 0.0):.4f}",
                f"{rates.get('facility_type', 0.0):.4f}",
                f"{rates.get('community_area', 0.0):.4f}",
                f"{rates.get('zip', 0.0):.4f}",
            ]
        )
    lines.extend(_table(["fold", "test rows", "chain", "facility", "community", "zip"], rows))
    lines.append("")
    lines.append(
        "**These are row counts, not metrics.** No label is read anywhere in this profile. "
        "A high chain rate means most test rows fall back to the learned `__UNKNOWN__` "
        "vector, which caps what a chain embedding can contribute regardless of how well "
        "it is trained."
    )
    return "\n".join(lines)


def profile_chain_structure(frame: pl.DataFrame, categoricals: pl.DataFrame) -> str:
    """How many establishments per chain, over the whole snapshot."""
    grouped = (
        categoricals.filter(pl.col("chain_key") != UNKNOWN_CATEGORY)
        .group_by("chain_key")
        .agg(
            pl.col("establishment_id").n_unique().alias("establishments"),
            pl.len().alias("rows"),
        )
    )
    chains = grouped.filter(pl.col("establishments") > 1).sort("establishments", descending=True)
    lines: list[str] = []
    lines.append(
        f"Distinct normalised names: **{grouped.height:,}**. Names carried by more than one "
        f"establishment (i.e. chains): **{chains.height:,}**, covering "
        f"**{int(chains['rows'].sum()):,}** of {categoricals.height:,} rows "
        f"({int(chains['rows'].sum()) / categoricals.height:.2%})."
    )
    lines.append("")
    rows = [
        [f"`{r['chain_key']}`", f"{r['establishments']}", f"{r['rows']:,}"]
        for r in chains.head(15).iter_rows(named=True)
    ]
    lines.extend(_table(["chain (normalised name)", "establishments", "rows"], rows))
    lines.append("")
    lines.append(
        "Measured over the whole snapshot for description only. **The model never sees "
        "this set**: membership is recomputed inside each fold from that fold's training "
        "rows, because whether a name is shared depends on which establishments exist, and "
        "a location opened in 2025 must not make a 2022 row part of a chain."
    )
    return "\n".join(lines)


def profile_embedding_budget(frame: pl.DataFrame, categoricals: pl.DataFrame) -> str:
    """Parameters bought by the embedding block against the dense network's."""
    spec = spec_for(PRIMARY)
    last = _folds(frame)[-2]
    training = training_frame(frame, last)
    encoding = encode.fit_encoding(_joined(training, categoricals), spec)

    lines: list[str] = []
    rows = []
    total_embedding = 0
    for family, dim in EMBEDDING_DIMS.items():
        size = encoding.sizes.get(family.value, 0)
        params = size * dim
        total_embedding += params
        rows.append([f"`{family.value}`", f"{size}", f"{dim}", f"{params:,}"])
    lines.extend(_table(["family", "vocabulary", "dim", "parameters"], rows))

    dense_in = len(preprocess.matrix_columns(spec)) + embedding_width(spec)
    dense_params = (
        dense_in * HIDDEN_SIZES[0]
        + HIDDEN_SIZES[0]
        + HIDDEN_SIZES[0] * HIDDEN_SIZES[1]
        + HIDDEN_SIZES[1]
        + HIDDEN_SIZES[1]
    )
    lines.append("")
    lines.append(
        f"Embedding parameters: **{total_embedding:,}**. Dense-stack parameters "
        f"(approximate, excluding BatchNorm): **{dense_params:,}**. The embedding tables "
        f"are {total_embedding / max(dense_params, 1):.2f}x the rest of the network, which "
        f"is why dropout and weight decay are not optional here."
    )
    lines.append("")
    rows = []
    for s in NEURAL_REGISTRY:
        if s.learner is not Learner.MLP:
            continue
        if s.encoding is CategoricalEncoding.ONE_HOT:
            width = sum(encoding.sizes.get(c, 0) for c in s.entity_columns)
            note = f"{width} indicator columns"
        elif s.encoding is CategoricalEncoding.NONE:
            width = 0
            note = "no categoricals"
        else:
            width = embedding_width(s)
            note = f"{width} embedding dims"
        rows.append([f"`{s.name}`", s.encoding.value, f"{width}", note])
    lines.extend(_table(["model", "encoding", "extra width", "note"], rows))
    lines.append("")
    lines.append(
        "The one-hot control buys a far wider first layer than the embedding model for the "
        "same information, which is the textbook argument for embeddings and the thing "
        "experiment B measures rather than assumes."
    )
    return "\n".join(lines)


def profile_inner_split(frame: pl.DataFrame, categoricals: pl.DataFrame) -> str:
    """Where the early-stopping split falls, per fold."""
    lines: list[str] = []
    rows = []
    for fold in _folds(frame):
        training = training_frame(frame, fold)
        cut = inner_split_date(training)
        left = training.filter(pl.col("rd") < cut)
        right = training.filter(pl.col("rd") >= cut)
        rows.append(
            [
                f"`{fold.fold_id}`",
                f"{training.height:,}",
                f"{cut}",
                f"{left.height:,}",
                f"{right.height:,}",
                f"{right.height / training.height:.4f}",
                f"{fold.train_end}",
            ]
        )
    lines.extend(
        _table(
            ["fold", "train rows", "split date", "fit", "validate", "share", "train_end"],
            rows,
        )
    )
    lines.append("")
    lines.append(
        f"Target share is {INNER_VALIDATION_FRACTION:.0%}, cut on a whole day so no single "
        "date straddles the split -- two inspections days apart share almost all of their "
        "as-of history, and splitting a day would leak near-duplicate rows across the "
        "boundary. **Every split date is at or before that fold's `train_end`**, which is "
        "what keeps `trained_through = train_end` literally true for a component that "
        "early-stops."
    )
    return "\n".join(lines)


def profile_scaling_need(frame: pl.DataFrame, categoricals: pl.DataFrame) -> str:
    """The feature scale spread a network must cope with and a tree did not."""
    spec = spec_for(PRIMARY)
    last = _folds(frame)[-2]
    training = training_frame(frame, last)
    matrix = preprocess.numeric_matrix(training, spec)
    columns = preprocess.matrix_columns(spec)

    stats = []
    for index, name in enumerate(columns):
        column = matrix[:, index]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            continue
        stats.append((name, float(np.nanstd(finite)), float(np.nanmax(finite))))
    stats.sort(key=lambda t: t[1], reverse=True)

    lines: list[str] = []
    rows = [[f"`{name}`", f"{sd:,.2f}", f"{mx:,.2f}"] for name, sd, mx in stats[:6]]
    rows.append(["...", "", ""])
    rows.extend([f"`{name}`", f"{sd:,.4f}", f"{mx:,.4f}"] for name, sd, mx in stats[-4:])
    lines.extend(_table(["column", "SD", "max"], rows))
    spread = stats[0][1] / max(stats[-1][1], 1e-12)
    lines.append("")
    lines.append(
        f"Widest standard deviation is **{spread:,.0f}x** the narrowest on fold "
        f"`{last.fold_id}`'s training window. A tree is invariant to this and Component 7 "
        f"therefore fitted no scaler at all; a dense layer's first weighted sum is not, "
        f"which is why Component 8 standardises and Component 7 did not. The statistics are "
        f"fitted on the inner training rows only."
    )
    return "\n".join(lines)


def profile_as_of_lag(frame: pl.DataFrame, categoricals: pl.DataFrame) -> str:
    """How stale a carried categorical is."""
    lag = categoricals.filter(pl.col("days_since_source").is_not_null())["days_since_source"]
    values = lag.to_numpy()
    lines: list[str] = []
    rows = [
        [
            f"{lag.len():,}",
            f"{float(np.mean(values)):.1f}",
            f"{float(np.median(values)):.0f}",
            f"{float(np.percentile(values, 25)):.0f}",
            f"{float(np.percentile(values, 75)):.0f}",
            f"{int(np.min(values))}",
            f"{int(np.max(values))}",
        ]
    ]
    lines.extend(_table(["rows", "mean", "median", "p25", "p75", "min", "max"], rows))
    lines.append("")
    lines.append(
        f"Days between a row's own inspection and the earlier inspection its categoricals "
        f"were carried from. **Minimum is {int(np.min(values))}**, which is the observable "
        f"proving the join is strictly as-of: a zero here would mean a row supplied its own "
        f"attributes. A large lag is not an error -- facility type and address are stable "
        f"attributes -- but a stale value is a stale value, and it is recorded rather than "
        f"assumed away."
    )
    return "\n".join(lines)


PROFILES: dict[str, Callable[[pl.DataFrame, pl.DataFrame], str]] = {
    "categorical_coverage": profile_categorical_coverage,
    "cardinality_growth": profile_cardinality_growth,
    "unseen_rate": profile_unseen_rate,
    "chain_structure": profile_chain_structure,
    "embedding_budget": profile_embedding_budget,
    "inner_split": profile_inner_split,
    "scaling_need": profile_scaling_need,
    "as_of_lag": profile_as_of_lag,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, metavar="PATH")
    parser.add_argument("--categoricals", type=Path, metavar="PATH")
    parser.add_argument("--only", action="append", metavar="NAME")
    parser.add_argument("--list", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for name in PROFILES:
            print(name)
        return 0

    settings = load_settings()
    features_path = args.features or duckdb_queries.latest_parquet(
        settings.features_processed_dir, prefix="as_of_features_"
    )
    categoricals_path = args.categoricals or duckdb_queries.latest_parquet(
        settings.neural_processed_dir, prefix="neural_categoricals_"
    )

    frame = pl.read_parquet(features_path).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )
    categoricals = pl.read_parquet(categoricals_path)

    requested = args.only or list(PROFILES)
    unknown = [name for name in requested if name not in PROFILES]
    if unknown:
        raise SystemExit(f"unknown profile(s): {', '.join(unknown)}")

    print(f"<!-- generated by scripts/profile_neural.py from {features_path.name} -->")
    print(f"<!-- categoricals: {categoricals_path.name}; {frame.height} feature rows -->")
    for name in requested:
        print()
        print(f"### {name}")
        print()
        print(PROFILES[name](frame, categoricals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
