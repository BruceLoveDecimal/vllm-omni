# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent dense scoring oracles and storage/lifecycle compression tests."""

import math

import pytest
import torch
import torch.nn.functional as F

from tests.diffusion.ar_diffusion.test_video_memory import absolute_keys, raw, vm
from tests.diffusion.ar_diffusion.test_video_retrieval import append as append_retrieval
from tests.diffusion.ar_diffusion.test_video_retrieval import state as retrieval_state

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def rows(count=12, dtype=torch.float64):
    key = torch.randn(1, count, 2, 8, generator=torch.Generator().manual_seed(71), dtype=dtype)
    return vm.MemoryRows(key, key + 20, torch.arange(count) // 4, torch.arange(count))


@pytest.mark.parametrize("score", ["pairwise_mean", "centroid"])
@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_world_selection_equals_dense_oracle_and_owns_exact_storage(score, ratio, dtype):
    source = rows(dtype=dtype)
    output = vm.world_compress(source, 4, ratio, score)
    flat = source.key[0].double().flatten(1)
    if score == "pairwise_mean":
        reference = (F.normalize(flat[4:], dim=-1) @ F.normalize(flat[:4], dim=-1).T).mean(-1)
    else:
        reference = F.cosine_similarity(flat[4:], flat[:4].mean(0)[None], dim=-1)
    selected = torch.argsort(reference, stable=True)[: math.ceil(8 * ratio)].sort().values + 4
    indices = torch.cat((torch.arange(4), selected))
    assert torch.equal(output.key, source.key[:, indices])
    assert torch.equal(output.value, source.value[:, indices])
    assert torch.equal(output.positions, source.positions[indices])
    assert output.key.untyped_storage().nbytes() == output.key.numel() * output.key.element_size()
    assert output.value.untyped_storage().nbytes() == output.value.numel() * output.value.element_size()


def test_paper_and_centroid_scores_are_distinct_not_silently_interchanged():
    source = rows()
    # Unequal anchor norms distinguish mean(normalize(K)) from normalize(mean(K)).
    source.key[:, 0] *= 100
    paper = vm.world_compress(source, 4, 0.25, "pairwise_mean")
    code = vm.world_compress(source, 4, 0.25, "centroid")
    assert not torch.equal(paper.key, code.key)


def test_zero_keys_ties_preserve_anchor_and_stable_original_order():
    source = vm.MemoryRows(
        torch.zeros(1, 12, 1, 4), torch.arange(48.0).view(1, 12, 1, 4), torch.arange(12) // 4, torch.arange(12)
    )
    output = vm.world_compress(source, 4, 0.25, "pairwise_mean")
    assert torch.equal(output.value, source.value[:, :6])
    assert torch.isfinite(output.key).all()


@pytest.mark.parametrize("capacity", [8, 10, 12])
def test_pc_sum_scores_match_materialized_qk_with_protected_regions(capacity):
    source = rows()
    query = torch.randn(1, 5, 2, 8, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    indices = vm.participative_indices(query, source, sink_frames=1, recent_start=2, capacity=capacity)
    scores = torch.einsum("bqhd,bkhd->bqk", query, source.key).sum(1)[0]
    middle = torch.arange(4, 8)
    selected = middle[torch.argsort(scores[middle], descending=True, stable=True)[: capacity - 8]]
    reference = torch.cat((torch.arange(4), selected, torch.arange(8, 12))).sort().values
    assert torch.equal(indices, reference)
    with pytest.raises(ValueError, match="protected"):
        vm.participative_indices(query, source, sink_frames=1, recent_start=2, capacity=7)


def pc_state(**kwargs):
    return vm.VideoMemory(
        vm.VideoMemoryConfig(window_frames=8, sink_frames=2, pc_budget_frames=6, pc_recent_frames=2, **kwargs),
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
        if start >= 4:
            assert all(r.positions.numel() <= 12 for r in memory.history)
        for r in memory.history:
            assert r.positions[:4].tolist() == [0, 0, 1, 1]
            assert r.positions[-4:].tolist() == [start, start, start + 1, start + 1]
            assert r.key.untyped_storage().nbytes() == r.key.numel() * r.key.element_size()
        assert all(q.shape[1] <= 4 for q in memory._query_history.values())


def test_pc_attention_rebases_sparse_top_keys_without_changing_spatial_channels():
    memory = pc_state()
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


def test_compressed_bank_uses_half_kv_and_per_layer_selection():
    plain = retrieval_state()
    compressed = retrieval_state(world_keep_ratio=0.0)
    for start in range(0, 8, 2):
        append_retrieval(plain, start)
        append_retrieval(compressed, start)
    assert compressed.bank_nbytes < plain.bank_nbytes
    for start, entry in compressed.bank.items():
        for layer, r in enumerate(entry.layers):
            assert r.key.shape[1] == 2
            assert torch.equal(r.key, plain.bank[start].layers[layer].key[:, :2])


@pytest.mark.parametrize(
    "config",
    [
        {"world_keep_ratio": float("nan")},
        {"world_keep_ratio": 2},
        {"world_keep_ratio": 0.25},
        {"world_score": "unknown"},
        {"pc_budget_frames": 8},
        {"pc_budget_frames": 12, "pc_recent_frames": 4},
        {"pc_budget_frames": 18},
    ],
)
def test_invalid_compression_config(config):
    with pytest.raises(ValueError):
        vm.VideoMemoryConfig(**config)


def test_pc_freezes_only_indices_and_preserves_history_on_failure():
    memory = pc_state()
    for start in (0, 2, 4):
        run_pc_chunk(memory, start)
    original = memory.history
    with memory.forward(6, 2, commit=False):
        for layer in range(2):
            memory.assemble(layer, raw(6), raw(6), raw(6))
    assert memory.history is original
    for indices in memory._selection.values():
        assert isinstance(indices, torch.Tensor)
        assert indices.dtype == torch.int64 and indices.ndim == 1
        assert indices.untyped_storage().nbytes() == indices.numel() * 8
    selection = dict(memory._selection)
    with pytest.raises(RuntimeError, match="head failure"):
        with memory.forward(6, 2, commit=True):
            for layer in range(2):
                memory.assemble(layer, raw(6), raw(6), raw(6))
            raise RuntimeError("head failure")
    assert memory.history is original and memory.next_frame == 6
    assert all(memory._selection[i] is index for i, index in selection.items())
    run_pc_chunk(memory, 6)
    assert not memory._selection
    assert all(row.key.shape[1] == 12 for row in memory.history)
