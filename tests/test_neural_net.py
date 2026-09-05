"""The module itself: shape, architecture, and the determinism switches.

These are structural tests. They assert that the network the specification describes is
the network that gets built -- the right layers in the right order, an embedding block
that narrows when a family is ablated, and a logit rather than a probability coming out
of ``forward``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from sentinel.neural import net
from sentinel.neural.definitions import DROPOUT, HIDDEN_SIZES


def _net(dense: int = 30, embeddings: list[tuple[int, int]] | None = None) -> net.EmbeddingNet:
    return net.EmbeddingNet(
        dense_width=dense, embedding_sizes=embeddings if embeddings is not None else []
    )


# --- 1. the architecture is the specification's ------------------------------


def test_the_layer_sequence_is_linear_batchnorm_relu_dropout_twice_then_a_logit() -> None:
    """The exact order the project specification writes, asserted layer by layer.

    BatchNorm sits between the linear map and the nonlinearity. Putting it after the ReLU
    would be a different (and also defensible) architecture, which is precisely why it is
    pinned rather than left to a reader's assumption.
    """
    model = _net()
    kinds = [type(layer) for layer in model.stack]
    assert kinds == [
        nn.Linear,
        nn.BatchNorm1d,
        nn.ReLU,
        nn.Dropout,
        nn.Linear,
        nn.BatchNorm1d,
        nn.ReLU,
        nn.Dropout,
        nn.Linear,
    ]


def test_the_hidden_widths_are_256_then_128() -> None:
    model = _net()
    linears = [layer for layer in model.stack if isinstance(layer, nn.Linear)]
    assert [layer.out_features for layer in linears] == [*HIDDEN_SIZES, 1]
    assert linears[0].in_features == 30


def test_dropout_is_the_specified_rate() -> None:
    model = _net()
    for layer in model.stack:
        if isinstance(layer, nn.Dropout):
            assert layer.p == DROPOUT == 0.3


def test_the_output_is_a_single_logit_per_row() -> None:
    """Not a probability. The sigmoid lives in ``predict`` and in the loss, nowhere else."""
    model = _net()
    model.eval()
    dense = torch.zeros((7, 30), dtype=torch.float32)
    codes = torch.zeros((7, 0), dtype=torch.int64)
    out = model(dense, codes)
    assert out.shape == (7,)
    # A logit is unbounded; a probability would not be. With zero input and default
    # initialisation the output is near zero, which is a logit of 0.5 rather than 0.5.
    assert out.dtype == torch.float32


# --- 2. the embedding block --------------------------------------------------


def test_the_first_layer_widens_by_exactly_the_embedding_dimensions() -> None:
    model = _net(dense=30, embeddings=[(50, 16), (10, 8)])
    first = next(layer for layer in model.stack if isinstance(layer, nn.Linear))
    assert model.embedding_width == 24
    assert first.in_features == 30 + 24


def test_an_ablated_family_narrows_the_network_rather_than_zeroing_a_block() -> None:
    """ "Without the chain embedding" must mean the parameters are gone.

    A zeroed block would still cost parameters and still receive gradient, so the
    ablation would not measure what it claims.
    """
    full = _net(dense=30, embeddings=[(50, 16), (10, 8)])
    ablated = _net(dense=30, embeddings=[(10, 8)])
    assert ablated.embedding_width == 8
    assert ablated.parameter_count < full.parameter_count
    assert len(ablated.embeddings) == 1


def test_each_family_is_looked_up_in_its_own_table() -> None:
    """Column *i* of ``codes`` must index table *i*, or every family is scrambled."""
    model = _net(dense=0, embeddings=[(4, 2), (3, 2)])
    with torch.no_grad():
        model.embeddings[0].weight.copy_(torch.arange(8, dtype=torch.float32).reshape(4, 2))
        model.embeddings[1].weight.copy_(torch.full((3, 2), -1.0))
    first = model.embedding_weights(0)
    second = model.embedding_weights(1)
    assert first.shape == (4, 2)
    assert second.shape == (3, 2)
    assert torch.allclose(second, torch.full((3, 2), -1.0))


def test_embedding_weights_are_detached_from_the_graph() -> None:
    """They are read for visualisation and for the XGBoost experiment."""
    model = _net(dense=0, embeddings=[(5, 3)])
    weight = model.embedding_weights(0)
    assert not weight.requires_grad


def test_a_network_with_no_inputs_at_all_is_refused() -> None:
    with pytest.raises(ValueError, match="no inputs at all"):
        _net(dense=0, embeddings=[])


def test_a_negative_dense_width_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _net(dense=-1)


def test_a_pure_embedding_network_is_legitimate() -> None:
    model = _net(dense=0, embeddings=[(5, 3)])
    model.eval()
    codes = torch.zeros((4, 1), dtype=torch.int64)
    out = model(torch.zeros((4, 0), dtype=torch.float32), codes)
    assert out.shape == (4,)


# --- 3. batching -------------------------------------------------------------


def test_a_trailing_batch_of_one_is_dropped() -> None:
    """BatchNorm's per-batch variance is undefined for a single row and raises."""
    starts = net.batch_starts(n_rows=1025, batch_size=512)
    assert starts == [0, 512], "the 1-row trailing batch was not dropped"


def test_a_trailing_batch_of_two_is_kept() -> None:
    starts = net.batch_starts(n_rows=1026, batch_size=512)
    assert starts == [0, 512, 1024]


def test_an_exact_multiple_keeps_every_batch() -> None:
    assert net.batch_starts(n_rows=1024, batch_size=512) == [0, 512]


def test_an_empty_window_produces_no_batches() -> None:
    assert net.batch_starts(n_rows=0, batch_size=512) == []


def test_a_window_smaller_than_one_batch_still_produces_one() -> None:
    assert net.batch_starts(n_rows=17, batch_size=512) == [0]


def test_a_window_of_one_row_produces_none() -> None:
    assert net.batch_starts(n_rows=1, batch_size=512) == []


def test_batchnorm_really_does_reject_a_single_row_in_training_mode() -> None:
    """The reason ``batch_starts`` drops a trailing singleton, driven rather than assumed."""
    model = _net(dense=4)
    model.train()
    with pytest.raises(ValueError):
        model(torch.zeros((1, 4), dtype=torch.float32), torch.zeros((1, 0), dtype=torch.int64))


# --- 4. determinism ----------------------------------------------------------


def test_seeding_makes_initialisation_reproducible() -> None:
    net.seed_everything(42)
    first = _net(dense=5, embeddings=[(7, 3)])
    net.seed_everything(42)
    second = _net(dense=5, embeddings=[(7, 3)])
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(a, b), "the same seed produced different initial weights"


def test_different_seeds_produce_different_initialisation() -> None:
    """A vacuity guard: the test above would pass if seeding did nothing at all."""
    net.seed_everything(42)
    first = _net(dense=5, embeddings=[(7, 3)])
    net.seed_everything(43)
    second = _net(dense=5, embeddings=[(7, 3)])
    differ = any(
        not torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters(), strict=True)
    )
    assert differ, "two different seeds produced identical weights, so seeding is inert"


def test_seeding_pins_one_thread_and_deterministic_algorithms() -> None:
    """The direct analogue of Component 7's ``n_jobs=1``."""
    net.seed_everything(42)
    assert torch.get_num_threads() == 1
    assert torch.are_deterministic_algorithms_enabled()


def test_seeding_also_pins_python_and_numpy() -> None:
    """Both order things upstream of the fit."""
    import random

    net.seed_everything(7)
    a_random, a_numpy = random.random(), float(np.random.random())
    net.seed_everything(7)
    b_random, b_numpy = random.random(), float(np.random.random())
    assert a_random == b_random
    assert a_numpy == b_numpy


def test_the_device_is_cpu() -> None:
    """Deliberate. A GPU cannot offer the bit-identity this project claims. ADR 0020."""
    assert net.device_name() == "cpu"
