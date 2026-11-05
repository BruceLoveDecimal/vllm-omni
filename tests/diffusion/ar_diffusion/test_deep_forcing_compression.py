# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent dense scoring oracles and storage/lifecycle compression tests."""

import math

import pytest
import torch

from tests.diffusion.ar_diffusion.test_deep_forcing import absolute_keys, df, raw

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def rows(count=12, dtype=torch.float64):
    key = torch.randn(1, count, 2, 8, generator=torch.Generator().manual_seed(71), dtype=dtype)
    return df.MemoryRows(key, key + 20, torch.arange(count) // 4, torch.arange(count))


@pytest.mark.parametrize("capacity", [8, 10, 12])
def test_pc_sum_scores_match_materialized_qk_with_protected_regions(capacity):
    source = rows()
    query = torch.randn(1, 5, 2, 8, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    indices = df.participative_indices(query, source, sink_frames=1, recent_start=2, capacity=capacity)
    scores = torch.einsum("bqhd,bkhd->bqk", query, source.key).sum(1)[0]
    middle = torch.arange(4, 8)
    selected = middle[torch.argsort(scores[middle], descending=True, stable=True)[: capacity - 8]]
    reference = torch.cat((torch.arange(4), selected, torch.arange(8, 12))).sort().values
    assert torch.equal(indices, reference)
    with pytest.raises(ValueError, match="protected"):
        df.participative_indices(query, source, sink_frames=1, recent_start=2, capacity=7)


def pc_state(**kwargs):
    return df.DeepForcing(
        df.DeepForcingConfig(window_frames=8, sink_frames=2, pc_budget_frames=6, pc_recent_frames=2, **kwargs),
        num_layers=2,
        tokens_per_frame=2,
        temporal_dim=4,
    )


def run_pc_chunk(memory, start, *, noisy_steps=3):
    keys = [
        absolute_keys(raw(start) + layer, torch.arange(start, start + 2).repeat_interleave(2)) for layer in range(2)
    ]
    selection = None
    for step in range(noisy_steps + 1):
        commit = step == noisy_steps
        with memory.forward(start, 2, commit=commit):
            for layer in range(2):
                k, v = memory.assemble(layer, raw(start) + step * 100, keys[layer], raw(start))
                assert k.shape == v.shape and k.shape[1] <= 16
                assert torch.isfinite(k).all()
            if selection is not None:
                for layer, before in selection.items():
                    assert memory._selection[layer] is before
            selection = dict(memory._selection)


def test_pc_selection_frozen_across_denoising_and_bounded_on_long_rollouts():
    memory = pc_state()
    for start in range(0, 80, 2):
        run_pc_chunk(memory, start)
        assert memory.next_frame == start + 2
        # The window (8 frames) is used in full; PC compresses to the 6-frame
        # budget only when the next chunk would overflow it.
        assert all(r.positions.numel() <= 16 for r in memory.history)
        if start >= 8:
            assert all(r.positions.numel() >= 12 for r in memory.history)
        for r in memory.history:
            assert r.positions[:4].tolist() == [0, 0, 1, 1]
            assert r.positions[-4:].tolist() == [start, start, start + 1, start + 1]
            assert r.key.untyped_storage().nbytes() == r.key.numel() * r.key.element_size()
        assert all(q.shape[1] <= 4 for q in memory._query_history.values())


def test_pc_attention_rebases_sparse_top_keys_without_changing_spatial_channels():
    memory = pc_state(rebase_sink=True, rebase_pc=True)
    for start in range(0, 12, 2):
        run_pc_chunk(memory, start)
    with memory.forward(12, 2, commit=False):
        for layer in range(2):
            key, value = memory.assemble(
                layer, raw(12), absolute_keys(raw(12), torch.arange(12, 14).repeat_interleave(2)), raw(12)
            )
            selected = memory._previous(layer)
            middle_count = selected.positions.numel() - 4
            middle_frames = math.ceil(middle_count / 2)
            targets = torch.cat(
                (
                    torch.tensor([0, 0, 1, 1]) + 12 - middle_frames - 2,
                    12 - middle_frames + torch.arange(middle_count) // 2,
                )
            )
            assert (selected.positions < 12).all()
            delta = targets - selected.positions
            expected = absolute_keys(selected.key, delta)
            torch.testing.assert_close(key[:, : selected.positions.numel()], expected)
            assert torch.equal(key[:, : selected.positions.numel(), :, 4:], selected.key[..., 4:])
            assert torch.equal(value[:, : selected.positions.numel()], selected.value)


@pytest.mark.parametrize(
    "config",
    [
        {"sink_frames": 9, "pc_budget_frames": 8},
        {"sink_frames": 9, "pc_budget_frames": 12, "pc_recent_frames": 4},
        {"pc_budget_frames": 18},
        {"pc_budget_frames": 12, "pc_recent_frames": -1},
    ],
)
def test_invalid_compression_config(config):
    with pytest.raises(ValueError):
        df.DeepForcingConfig(**config)


def test_pc_freezes_only_indices_and_preserves_history_on_failure():
    memory = pc_state()
    for start in (0, 2, 4, 6):
        run_pc_chunk(memory, start)
    original = memory.history
    with memory.forward(8, 2, commit=False):
        for layer in range(2):
            memory.assemble(layer, raw(8), raw(8), raw(8))
    assert memory.history is original and memory._selection
    for indices in memory._selection.values():
        assert isinstance(indices, torch.Tensor)
        assert indices.dtype == torch.int64 and indices.ndim == 1
        assert indices.untyped_storage().nbytes() == indices.numel() * 8
    selection = dict(memory._selection)
    with pytest.raises(RuntimeError, match="head failure"):
        with memory.forward(8, 2, commit=True):
            for layer in range(2):
                memory.assemble(layer, raw(8), raw(8), raw(8))
            raise RuntimeError("head failure")
    assert memory.history is original and memory.next_frame == 8
    assert all(memory._selection[i] is index for i, index in selection.items())
    run_pc_chunk(memory, 8)
    assert not memory._selection
    assert all(row.key.shape[1] == 12 for row in memory.history)
