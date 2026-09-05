"""Fitting one network to one fold's training window. Pure -- no filesystem, no clock.

Four decisions carry this module's integrity.

**What "train" means is not redefined here.** The training frame comes from
``modeling.train.training_frame``, which delegates to ``evaluation.folds.assign_split``.
There is exactly one definition of the training window in the repository -- the one
Component 5's ``future_rows_never_enter_training`` check independently re-derives -- and
Component 8 inherits it rather than reimplementing it.

**Early stopping reads a window carved from the end of the training data.** This is the
decision that keeps ``trained_through`` honest, and it is worth stating precisely because
it is the one place Component 8 does something Components 6 and 7 did not.

A network needs a validation signal to stop on. The two obvious candidates are the fold's
calibration window and its test window, and *both are later than* ``train_end``. Reading
either would mean the fit learned from a date later than the horizon it declares, which is
exactly what ``trained_through`` denies -- and in the calibration window's case it would
also consume the window Component 9 exists to use. So the split is internal: training days
are sorted, the last ~15% of *rows* (cut on a whole-day boundary, so no single day
straddles the split) become the validation window, and the rest is fitted. Nothing later
than ``fold.train_end`` is read by anything.

The cost is real and is not hidden: the final model is fitted on about 85% of its fold's
training rows rather than all of them, because the weights restored at the end are the
ones from the best validation epoch. Component 7 avoided that cost by freezing a round
count and refitting on everything; doing the same here would double an already long run,
and the specification asks for per-fold early stopping and per-fold learning curves. The
trade is recorded in the findings and in the manifest rather than argued away.

**Day boundaries are never split.** Two inspections of the same establishment days apart
share almost all of their as-of history, so a split that put one on each side would leak
in the ordinary machine-learning sense -- near-duplicate rows across a validation
boundary. Cutting on a whole day is the cheapest defence and costs nothing.

**Row order is load-bearing, more than anywhere else in the project.** Component 6
measured coefficients moving by 7.049e-09 under a re-ordering; Component 7 measured a
booster's prediction moving by 0.11. A mini-batched optimiser has both exposures plus
batch composition. So rows are canonically sorted before anything, and every shuffle is
drawn from a seeded generator rather than from global state.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from sentinel.evaluation.models import FoldSpec
from sentinel.modeling.train import TRAIN_SORT_KEYS
from sentinel.neural import encode, net, preprocess
from sentinel.neural.definitions import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP_NORM,
    INNER_VALIDATION_FRACTION,
    MAX_EPOCHS,
    MIN_INNER_SPLIT_ROWS,
    SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR,
    SCHEDULER_PATIENCE,
    WEIGHT_DECAY,
    CategoricalEncoding,
    NeuralSpec,
    embedding_dim,
    learning_rate_for,
)
from sentinel.neural.encode import FoldEncoding
from sentinel.neural.models import EmbeddingTable, EpochRecord, FittedNetwork

logger = logging.getLogger(__name__)

#: Reasons a fit stopped. Recorded rather than inferred: "stopped at epoch 43" is
#: ambiguous between early stopping and exhausting the budget, and the two mean opposite
#: things about whether the budget was adequate.
STOP_EARLY = "early stopping: validation loss did not improve for the patience window"
STOP_BUDGET = "epoch budget exhausted: max_epochs reached without triggering patience"


class NeuralTrainError(RuntimeError):
    """Raised when a fold cannot be fitted."""


def inner_split_date(train: pl.DataFrame, fraction: float = INNER_VALIDATION_FRACTION) -> date:
    """The first date belonging to the inner validation window.

    Chosen so that at least ``fraction`` of training rows fall on or after it, cutting on
    a whole day. Walking the distinct dates backwards rather than taking a row quantile is
    what guarantees the whole-day property -- a quantile would land mid-day and split rows
    that share an establishment's history.
    """
    if "rd" not in train.columns:
        raise NeuralTrainError("training frame has no parsed reference date column 'rd'")
    counts = train.group_by("rd").agg(pl.len().alias("n")).sort("rd", descending=True)
    target = max(1, int(round(train.height * fraction)))
    running = 0
    chosen: date | None = None
    for row in counts.iter_rows(named=True):
        running += int(row["n"])
        chosen = row["rd"]
        if running >= target:
            break
    if chosen is None:
        raise NeuralTrainError("training window has no dates to split on")
    return chosen


def split_training_window(
    train: pl.DataFrame, spec: NeuralSpec, fold: FoldSpec
) -> tuple[pl.DataFrame, pl.DataFrame, date]:
    """Split one fold's training rows into inner-train and inner-validation.

    Both sides are checked against ``MIN_INNER_SPLIT_ROWS``. A fold that cannot produce a
    usable split is refused rather than silently fitted without an early-stopping signal:
    two folds trained under different stopping rules are not comparable, and the whole
    component is a comparison.
    """
    cut = inner_split_date(train)
    inner_train = train.filter(pl.col("rd") < cut)
    inner_validation = train.filter(pl.col("rd") >= cut)

    if inner_train.height < MIN_INNER_SPLIT_ROWS or inner_validation.height < MIN_INNER_SPLIT_ROWS:
        raise NeuralTrainError(
            f"{spec.name}: fold {fold.fold_id} splits into {inner_train.height} inner "
            f"training and {inner_validation.height} inner validation rows at {cut}, and "
            f"at least {MIN_INNER_SPLIT_ROWS} are required on each side. A fold is never "
            "fitted without an early-stopping signal, because it would not be comparable "
            "with the folds that have one."
        )
    # Re-derived rather than assumed: the filter above implies it, but this is the
    # property the whole early-stopping argument rests on, so it is checked.
    latest_train = inner_train["rd"].max()
    earliest_validation = inner_validation["rd"].min()
    if not (isinstance(latest_train, date) and isinstance(earliest_validation, date)):
        raise NeuralTrainError(
            f"{spec.name}: fold {fold.fold_id} inner split produced unparseable dates"
        )
    if earliest_validation <= latest_train:  # pragma: no cover - guard
        raise NeuralTrainError(
            f"{spec.name}: fold {fold.fold_id} inner split is not temporally ordered: "
            f"validation starts {earliest_validation}, training ends {latest_train}"
        )
    return inner_train, inner_validation, cut


def _labels(frame: pl.DataFrame, spec: NeuralSpec, fold_id: str) -> NDArray[np.int64]:
    if "target" not in frame.columns:
        raise NeuralTrainError(f"{spec.name}: fold {fold_id} frame has no target")
    values = frame["target"].to_list()
    if any(v is None for v in values):
        raise NeuralTrainError(
            f"{spec.name}: fold {fold_id} window contains a null target. Component 3 "
            "emits a target only for eligible rows; a null here means an ineligible row "
            "reached the model."
        )
    return np.asarray(values, dtype=np.int64)


def _positive_rate(frame: pl.DataFrame) -> float | None:
    if frame.height == 0:
        return None
    mean = frame["target"].mean()
    return float(mean) if isinstance(mean, (int, float)) else None


def _join_categoricals(
    frame: pl.DataFrame, categoricals: pl.DataFrame, spec: NeuralSpec
) -> pl.DataFrame:
    """Attach the experimental categorical columns to one window, preserving row order.

    A left join on ``target_inspection_id`` with the row count re-checked afterwards. The
    check is not decoration: a many-to-one join here would silently duplicate rows and
    every downstream matrix would still have a consistent shape.
    """
    if spec.encoding is CategoricalEncoding.NONE:
        return frame
    if categoricals is None or categoricals.height == 0:
        raise NeuralTrainError(
            f"{spec.name}: needs categorical inputs but none were supplied. Run "
            "`sentinel build-neural-categoricals` first."
        )
    wanted = [
        c
        for c in ("chain_key", "facility_type", "community_area", "zip", "establishment_id")
        if c in categoricals.columns
    ]
    joined = frame.join(
        categoricals.select("target_inspection_id", *wanted),
        on="target_inspection_id",
        how="left",
        suffix="_cat",
    )
    if joined.height != frame.height:
        raise NeuralTrainError(
            f"{spec.name}: categorical join changed the row count from {frame.height} to "
            f"{joined.height}; target_inspection_id must be unique in both frames"
        )
    return joined


def _embedding_sizes(spec: NeuralSpec, encoding: FoldEncoding) -> list[tuple[int, int]]:
    if spec.encoding is not CategoricalEncoding.EMBEDDING:
        return []
    return [
        (encoding.vocabulary_for(column).size, embedding_dim(column))
        for column in spec.entity_columns
    ]


def _tensors(
    frame: pl.DataFrame,
    spec: NeuralSpec,
    preprocessor: object,
    encoding: FoldEncoding,
    labels: NDArray[np.int64],
) -> tuple[Tensor, Tensor, Tensor]:
    dense = preprocess.dense_matrix(frame, spec, preprocessor, encoding)
    codes = preprocess.code_matrix(frame, spec, encoding)
    return (
        torch.from_numpy(np.ascontiguousarray(dense, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(codes, dtype=np.int64)),
        torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32)),
    )


def _evaluate_loss(
    model: nn.Module, criterion: nn.Module, dense: Tensor, codes: Tensor, targets: Tensor
) -> float:
    """Mean loss over a window, in eval mode and without gradients.

    Eval mode matters twice over: dropout is disabled, and BatchNorm switches from batch
    statistics to its running estimates. A validation loss computed in training mode would
    depend on how the window happened to be batched.
    """
    model.eval()
    with torch.no_grad():
        logits = model(dense, codes)
        return float(criterion(logits, targets).item())


def fit_fold(
    spec: NeuralSpec,
    train: pl.DataFrame,
    fold: FoldSpec,
    *,
    categoricals: pl.DataFrame | None = None,
    learning_rate: float | None = None,
    seed: int | None = None,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> FittedNetwork:
    """Fit one network on one fold's training window.

    ``train`` must already be a training frame for ``fold`` -- pass the output of
    ``modeling.train.training_frame``. It is re-sorted here anyway, because the sort is a
    correctness requirement rather than a caller's courtesy.
    """
    if train.height == 0:
        raise NeuralTrainError(
            f"{spec.name}: fold {fold.fold_id} has no training rows. A model is never "
            "fitted on an empty window, and a fold that produces one is a defect upstream "
            "rather than something to work around."
        )

    resolved_seed = spec.seed if seed is None else seed
    rate = learning_rate_for(fold.fold_set) if learning_rate is None else learning_rate
    net.seed_everything(resolved_seed)

    ordered = train.sort(list(TRAIN_SORT_KEYS))
    with_categoricals = (
        _join_categoricals(ordered, categoricals, spec) if categoricals is not None else ordered
    )
    if spec.encoding is not CategoricalEncoding.NONE and categoricals is None:
        raise NeuralTrainError(f"{spec.name}: encoding {spec.encoding.value} requires categoricals")

    labels_all = _labels(with_categoricals, spec, fold.fold_id)
    if len(set(labels_all.tolist())) < 2:
        raise NeuralTrainError(
            f"{spec.name}: fold {fold.fold_id} training window contains a single class "
            f"({int(labels_all[0])}). A fitted probability would be a constant, which is "
            "a reference schedule rather than a model."
        )

    inner_train, inner_validation, cut = split_training_window(with_categoricals, spec, fold)

    # Every fitted statistic below comes from ``inner_train`` alone. That is stricter
    # than the fold requires -- the whole training window would be legitimate -- and it
    # is deliberate: the validation loss is only an honest stopping signal if the
    # validation rows influenced neither the scaler nor the vocabulary.
    encoding = encode.fit_encoding(inner_train, spec)
    preprocessor = preprocess.build_preprocessor(spec)
    preprocessor.fit(preprocess.numeric_matrix(inner_train, spec))

    train_labels = _labels(inner_train, spec, fold.fold_id)
    validation_labels = _labels(inner_validation, spec, fold.fold_id)

    train_dense, train_codes, train_targets = _tensors(
        inner_train, spec, preprocessor, encoding, train_labels
    )
    val_dense, val_codes, val_targets = _tensors(
        inner_validation, spec, preprocessor, encoding, validation_labels
    )

    model = net.EmbeddingNet(
        dense_width=train_dense.shape[1],
        embedding_sizes=_embedding_sizes(spec, encoding),
    )

    pos_weight_value: float | None = None
    if spec.pos_weighted:
        positives = int(np.count_nonzero(train_labels == 1))
        negatives = int(np.count_nonzero(train_labels == 0))
        pos_weight_value = (negatives / positives) if positives else 1.0
        criterion: nn.Module = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32)
        )
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
    )

    generator = torch.Generator()
    generator.manual_seed(resolved_seed)

    n_rows = int(train_dense.shape[0])
    starts = net.batch_starts(n_rows, BATCH_SIZE)
    if not starts:
        raise NeuralTrainError(
            f"{spec.name}: fold {fold.fold_id} inner training window has {n_rows} rows, "
            f"too few for a single batch of at least {net.MIN_BATCH_ROWS}"
        )

    history: list[EpochRecord] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] = {}
    since_improved = 0
    previous_rate = rate
    rate_changes = 0
    stop_reason = STOP_BUDGET
    final_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        # A fresh permutation per epoch, drawn from the seeded generator rather than
        # global state, so batch composition is reproducible without being fixed.
        order = torch.randperm(n_rows, generator=generator)
        running = 0.0
        counted = 0
        for start in starts:
            index = order[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_dense[index], train_codes[index])
            loss = criterion(logits, train_targets[index])
            loss.backward()
            # Clipping before the step, on the global norm across every parameter. An
            # embedding table is sparse in effect -- a rare category's row is touched by
            # a handful of rows per epoch -- so an unlucky batch can produce a very large
            # gradient for it, and one such step can move a well-fitted network a long way.
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            running += float(loss.item()) * int(index.shape[0])
            counted += int(index.shape[0])

        train_loss = running / max(counted, 1)
        validation_loss = _evaluate_loss(model, criterion, val_dense, val_codes, val_targets)
        current_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                learning_rate=current_rate,
            )
        )
        final_epoch = epoch

        scheduler.step(validation_loss)
        stepped_rate = float(optimizer.param_groups[0]["lr"])
        if stepped_rate != previous_rate:
            rate_changes += 1
            previous_rate = stepped_rate

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                stop_reason = STOP_EARLY
                break

    if not best_state:  # pragma: no cover - only reachable if every epoch was NaN
        raise NeuralTrainError(
            f"{spec.name}: fold {fold.fold_id} never recorded an improving epoch"
        )
    # The weights that are kept are the best validation epoch's, not the last epoch's.
    # Without this, patience would detect overfitting and then return the overfitted model.
    model.load_state_dict(best_state)
    model.eval()

    scaler_mean, scaler_scale = preprocess.scaler_statistics(preprocessor)
    tables = _embedding_tables(model, spec, encoding)

    fitted = FittedNetwork(
        spec=spec,
        fold_set=fold.fold_set,
        fold_id=fold.fold_id,
        encoding=encoding,
        matrix_columns=preprocess.transformed_columns(spec, encoding),
        dense_width=int(train_dense.shape[1]),
        embedding_width=model.embedding_width,
        parameter_count=model.parameter_count,
        learning_rate=rate,
        pos_weight=pos_weight_value,
        epochs=tuple(history),
        best_epoch=best_epoch,
        final_epoch=final_epoch,
        stop_reason=stop_reason,
        learning_rate_changes=rate_changes,
        embeddings=tables,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        imputed_values=preprocess.imputed_values(preprocessor, spec),
        train_rows=with_categoricals.height,
        inner_train_rows=inner_train.height,
        inner_validation_rows=inner_validation.height,
        inner_validation_start=cut,
        train_nan_cells=preprocess.nan_cells(inner_train, spec),
        train_positive_rate=_positive_rate(with_categoricals),
        train_start=fold.train_start,
        train_end=fold.train_end,
        # The training end, not the calibration end. Early stopping read a window carved
        # from inside the training data, so no date later than train_end influenced this
        # fit. ADR 0014, ADR 0021.
        trained_through=fold.train_end,
        calibration_end_unused=fold.calibration_end,
        seed=resolved_seed,
    )
    _stash_model(fitted, model, preprocessor)

    logger.info(
        "Fitted %s on %s: %d/%d inner rows, lr %.2e, best epoch %d of %d (%s)",
        spec.name,
        fold.fold_id,
        inner_train.height,
        inner_validation.height,
        rate,
        best_epoch,
        final_epoch,
        "early stop" if stop_reason == STOP_EARLY else "budget",
    )
    return fitted


def _embedding_tables(
    model: net.EmbeddingNet, spec: NeuralSpec, encoding: FoldEncoding
) -> tuple[EmbeddingTable, ...]:
    if spec.encoding is not CategoricalEncoding.EMBEDDING:
        return ()
    tables: list[EmbeddingTable] = []
    for position, column in enumerate(spec.entity_columns):
        weight = model.embedding_weights(position).numpy()
        vocab = encoding.vocabulary_for(column)
        tables.append(
            EmbeddingTable(
                column=column,
                categories=vocab.categories,
                vectors=tuple(tuple(float(v) for v in row) for row in weight),
            )
        )
    return tuple(tables)


#: The live torch module and fitted preprocessor for each ``FittedNetwork``.
#:
#: ``FittedNetwork`` is a frozen slots dataclass carrying only plain typed Python, which
#: is what lets a leakage test compare two fits by value. The objects needed to *score* a
#: window are not plain Python and would break that, so they are kept beside the record
#: rather than inside it, keyed by identity. ``predict`` is the only reader.
#:
#: **The record itself is retained in the value, and that is not redundant.** ``id()`` is
#: only unique among *live* objects: CPython reuses an address as soon as the object at it
#: is collected. Keying on ``id`` alone, a ``FittedNetwork`` that had been garbage
#: collected could hand its slot to an unrelated record, and ``scorer_for`` would return
#: some other fold's network without raising -- scoring a window with the wrong model and
#: reporting nothing. Holding a strong reference makes the key valid for as long as the
#: entry exists. A test constructs a copy of a live record and asserts it is refused.
_MODELS: dict[int, tuple[net.EmbeddingNet, object, FittedNetwork]] = {}


def _stash_model(fitted: FittedNetwork, model: net.EmbeddingNet, preprocessor: object) -> None:
    _MODELS[id(fitted)] = (model, preprocessor, fitted)


def scorer_for(fitted: FittedNetwork) -> tuple[net.EmbeddingNet, object]:
    """The live module and preprocessor for a fit, for ``predict``."""
    entry = _MODELS.get(id(fitted))
    if entry is None or entry[2] is not fitted:
        raise NeuralTrainError(
            "this FittedNetwork has no live module: it was not produced by fit_fold in "
            "this process, so it cannot score a window"
        )
    return entry[0], entry[1]


__all__ = [
    "STOP_BUDGET",
    "STOP_EARLY",
    "NeuralTrainError",
    "fit_fold",
    "inner_split_date",
    "scorer_for",
    "split_training_window",
]
