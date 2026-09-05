"""The network. Exactly the architecture the project specification names, and no more.

    embeddings(chain 16, facility 8, community 8, zip 8)  ->|
                                                            |-- concat --> Linear(256)
    standardised numerics (26 features + 4 indicators)   ->|                BatchNorm
                                                                           ReLU
                                                                           Dropout(0.3)
                                                                           Linear(128)
                                                                           BatchNorm
                                                                           ReLU
                                                                           Dropout(0.3)
                                                                           Linear(1)

The output is a **logit**. Sigmoid is applied only where a probability is required for
reporting or evaluation -- never inside the loss, because ``BCEWithLogitsLoss`` fuses the
sigmoid into the loss for numerical stability and applying it twice would produce a model
that merely trains badly rather than one that raises.

Determinism lives here too, in :func:`seed_everything`. Component 7 pins ``n_jobs=1``
because a float reduction over threads depends on the order the threads finish in;
a network has the same exposure through its BLAS calls plus two of its own -- weight
initialisation and batch composition. All four are pinned. The cost is single-threaded
training on a machine with a GPU sitting idle, and that trade is taken deliberately: this
project's standard for "did not move" is bit-identity, and a GPU cannot offer it. See ADR
0020.
"""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from sentinel.neural.definitions import DROPOUT, HIDDEN_SIZES, OUTPUT_SIZE

logger = logging.getLogger(__name__)

#: BatchNorm computes a per-batch variance, which is undefined for a single row and
#: raises in training mode. A trailing batch of one is dropped rather than padded --
#: padding would invent a row, and one row out of tens of thousands changes nothing.
MIN_BATCH_ROWS = 2


class EmbeddingNet(nn.Module):
    """The specified architecture, parameterised by what a spec actually declares.

    ``embedding_sizes`` is a list of ``(vocabulary_size, embedding_dim)`` pairs in the
    spec's declared family order, so an ablation that drops a family produces a narrower
    first layer rather than a zeroed block. That matters: a zeroed block would still cost
    parameters and still receive gradient, and "without the chain embedding" would not
    mean what it says.
    """

    def __init__(
        self,
        *,
        dense_width: int,
        embedding_sizes: Sequence[tuple[int, int]],
        hidden_sizes: Sequence[int] = HIDDEN_SIZES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        if dense_width < 0:
            raise ValueError(f"dense_width must be non-negative, got {dense_width}")

        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings=size, embedding_dim=dim) for size, dim in embedding_sizes]
        )
        self.embedding_width = sum(dim for _, dim in embedding_sizes)
        self.dense_width = dense_width

        width = dense_width + self.embedding_width
        if width == 0:
            raise ValueError("network would have no inputs at all")

        layers: list[nn.Module] = []
        previous = width
        for size in hidden_sizes:
            layers.append(nn.Linear(previous, size))
            # BatchNorm before the nonlinearity, as the specification writes it. It
            # re-centres each layer's pre-activations per batch, which keeps the ReLU
            # from saturating one-sided as the weights move and lets a higher learning
            # rate be stable than would otherwise be.
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            # Dropout after the nonlinearity. With 30 dense inputs and up to ~13k
            # embedding rows, the embedding table holds far more parameters than the
            # rest of the network combined, so some regularisation is not optional.
            layers.append(nn.Dropout(dropout))
            previous = size
        layers.append(nn.Linear(previous, OUTPUT_SIZE))
        self.stack = nn.Sequential(*layers)

    def forward(self, dense: Tensor, codes: Tensor) -> Tensor:
        """Returns logits of shape ``(rows,)``.

        ``codes`` is ``(rows, families)`` of integer indices; column *i* is looked up in
        embedding table *i*. A zero-width ``codes`` is legitimate and means a spec with no
        categoricals.
        """
        parts: list[Tensor] = []
        if self.dense_width:
            parts.append(dense)
        for position in range(len(self.embeddings)):
            table = self._table(position)
            parts.append(table(codes[:, position]))
        combined = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        # ``nn.Sequential.__call__`` is typed as returning ``Any``; narrowing here keeps
        # that ``Any`` from escaping into every caller under strict mode.
        logits: Tensor = self.stack(combined)
        return logits.squeeze(-1)

    def _table(self, position: int) -> nn.Embedding:
        """One embedding table, narrowed.

        ``nn.ModuleList.__getitem__`` is typed as returning ``Module``, so every attribute
        beyond the base class would be ``Any``. The cast is checked at construction: this
        list only ever receives ``nn.Embedding`` instances.
        """
        table = self.embeddings[position]
        if not isinstance(table, nn.Embedding):  # pragma: no cover - construction guard
            raise TypeError(f"embedding table {position} is a {type(table).__name__}")
        return table

    def embedding_weights(self, position: int) -> Tensor:
        """The learned table for one family, detached and on the CPU.

        Detached because these are read for visualisation and for the
        embeddings-into-XGBoost experiment, and a live graph reference there would be a
        memory leak at best and an accidental gradient path at worst.
        """
        weight = self._table(position).weight
        return weight.detach().cpu().clone()

    @property
    def parameter_count(self) -> int:
        """Total learnable parameters, recorded in the manifest."""
        return sum(p.numel() for p in self.parameters())


def seed_everything(seed: int) -> None:
    """Pin every source of randomness this component can reach.

    Python's ``random`` and numpy's global state are seeded because they order things
    upstream of the fit; ``torch.manual_seed`` covers weight initialisation and dropout
    masks. ``use_deterministic_algorithms`` makes torch raise rather than silently choose
    a nondeterministic kernel, which is the behaviour worth having: a fit that cannot be
    reproduced should fail loudly rather than produce a number nobody can get back.

    ``torch.set_num_threads(1)`` is the direct analogue of Component 7's ``n_jobs=1``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    # Read by CUDA's BLAS for reproducible reductions. Set unconditionally so the
    # environment is identical whether or not a GPU is ever used; harmless on CPU.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def batch_starts(n_rows: int, batch_size: int) -> list[int]:
    """Batch offsets, dropping a trailing batch too small for BatchNorm.

    Returned as a list rather than a generator so the caller can count epochs' worth of
    steps without consuming it, and so a test can assert the drop rule directly.
    """
    if n_rows <= 0:
        return []
    starts = list(range(0, n_rows, batch_size))
    if starts and n_rows - starts[-1] < MIN_BATCH_ROWS:
        starts.pop()
    return starts


def device_name() -> str:
    """The device every fit runs on. CPU, deliberately -- see the module docstring."""
    return "cpu"


__all__ = [
    "MIN_BATCH_ROWS",
    "EmbeddingNet",
    "batch_starts",
    "device_name",
    "seed_everything",
]
