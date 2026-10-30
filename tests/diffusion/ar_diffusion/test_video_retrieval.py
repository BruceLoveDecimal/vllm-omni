# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""World retrieval ranking, attention, byte bounds, and transactional CPU tests."""

import math

import pytest
import torch
import torch.nn.functional as F

from tests.diffusion.ar_diffusion.test_video_memory import absolute_keys, raw, vm

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def pose(x=0.0, yaw=0.0):
    result = torch.eye(4, dtype=torch.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    result[:3, :3] = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)
    result[0, 3] = x
    return result


def state(**kwargs):
    return vm.VideoMemory(
        vm.VideoMemoryConfig(
            window_frames=8,
            sink_frames=2,
            retrieval_frames=2,
            bank_bytes=kwargs.pop("bank_bytes", 1 << 20),
            rebase_sink=False,
            **kwargs,
        ),
        num_layers=2,
        tokens_per_frame=2,
        temporal_dim=4,
    )


def append(memory, start, *, camera=None):
    memory.set_pose(pose(start) if camera is None else camera)
    with memory.forward(start, 2, commit=True):
        for layer in range(2):
            memory.assemble(
                layer,
                raw(start),
                absolute_keys(raw(start) + layer, torch.arange(start, start + 2).repeat_interleave(2)),
                raw(start) * 2,
            )


def test_pose_scores_against_analytic_translation_and_rotation():
    candidates = torch.stack((pose(0, 0), pose(1, math.pi / 2), pose(2, math.pi)))
    torch.testing.assert_close(
        vm.camera_distances(pose(), candidates), torch.tensor([0, 0.75, 2.0], dtype=torch.float64)
    )
    assert torch.equal(vm.camera_distances(pose(), torch.stack([pose()] * 3)), torch.zeros(3, dtype=torch.float64))


@pytest.mark.parametrize("correction", [False, True])
def test_revisit_attention_matches_manual_chunks_without_duplicates(correction):
    memory = state(rebase_retrieval=correction)
    for start in range(0, 8, 2):
        append(memory, start)
    original = memory.bank[2].layers[0].key.clone()
    for _ in range(3):
        memory.set_pose(pose(2))
        with memory.forward(8, 2, commit=False):
            assert [chunk.start for chunk in memory._retrieved] == [2]
            for layer in range(2):
                positions = torch.arange(8, 10).repeat_interleave(2)
                current = absolute_keys(raw(8) + layer, positions)
                k, v = memory.assemble(layer, raw(8), current, raw(8) * 2)
                key_parts, value_parts = [], []
                for chunk in (0, 2, 6, 8):
                    target = 4 if correction and chunk == 2 else chunk
                    key_parts.append(
                        absolute_keys(raw(chunk) + layer, torch.arange(target, target + 2).repeat_interleave(2))
                    )
                    value_parts.append(raw(chunk) * 2)
                expected_k, expected_v = torch.cat(key_parts, 1), torch.cat(value_parts, 1)
                torch.testing.assert_close(k, expected_k)
                torch.testing.assert_close(v, expected_v)
                expected = F.scaled_dot_product_attention(
                    raw(8).transpose(1, 2), expected_k.transpose(1, 2), expected_v.transpose(1, 2)
                )
                actual = F.scaled_dot_product_attention(raw(8).transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
                torch.testing.assert_close(actual, expected)
                assert k.shape[1] == memory.config.window_frames * 2
    assert torch.equal(memory.bank[2].layers[0].key, original)


def test_bank_byte_budget_eviction_and_noisy_read_only():
    sample = state()
    append(sample, 0)
    append(sample, 2)
    entry_bytes = sample.bank_nbytes
    memory = state(bank_bytes=entry_bytes * 2)
    for start in range(0, 30, 2):
        append(memory, start)
        assert memory.bank_nbytes <= entry_bytes * 2
        assert len(memory.bank) <= 2
        for entry in memory.bank.values():
            assert all(rows.key.device.type == "cpu" for rows in entry.layers)
    assert list(memory.bank) == [26, 28]
    before = memory.bank
    memory.set_pose(pose(26))
    with memory.forward(30, 2, commit=False):
        for layer in range(2):
            memory.assemble(layer, raw(30), raw(30), raw(30))
    assert memory.bank is before and memory.next_frame == 30
    memory.close()
    assert memory.bank_nbytes == 0


def test_oversized_entry_does_not_evict_existing_history():
    memory = state(bank_bytes=1)
    append(memory, 0)
    append(memory, 2)
    assert memory.bank_nbytes == 0


def test_bank_is_transactional_and_copies_camera(monkeypatch):
    memory = state()
    camera = pose(2)
    append(memory, 0)
    append(memory, 2, camera=camera)
    camera[0, 3] = 99
    assert memory.bank[2].pose[0, 3] == 2
    old_bank, old_history = memory.bank, memory.history

    def fail():
        raise RuntimeError("CPU bank copy failed")

    monkeypatch.setattr(memory, "_publish_bank", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        append(memory, 4)
    assert memory.bank is old_bank and memory.history is old_history and memory.next_frame == 4


@pytest.mark.parametrize("bad", [torch.eye(3), torch.full((4, 4), float("nan")), torch.ones(4, 4), -torch.eye(4)])
def test_invalid_camera(bad):
    with pytest.raises(ValueError):
        state().set_pose(bad)


def test_missing_pose_rejected_and_pose_required_again_after_commit():
    memory = state()
    with pytest.raises(ValueError, match="absolute camera pose"):
        with memory.forward(0, 2, commit=False):
            pass
    append(memory, 0)
    with pytest.raises(ValueError, match="absolute camera pose"):
        with memory.forward(2, 2, commit=False):
            pass


def test_ties_are_deterministic_and_sink_recent_never_retrieved():
    memory = state()
    for start in range(0, 12, 2):
        append(memory, start, camera=pose())
    memory.set_pose(pose())
    assert [chunk.start for chunk in memory._retrieve(2)] == [2]
    assert 0 not in memory.bank


@pytest.mark.parametrize(
    "config",
    [
        {"retrieval_frames": -1},
        {"retrieval_frames": 3},
        {"retrieval_frames": 9, "bank_bytes": 100},
        {"bank_bytes": True},
    ],
)
def test_invalid_retrieval_config(config):
    with pytest.raises(ValueError):
        vm.VideoMemoryConfig(**config)


def test_pc_retrieves_missing_tokens_from_partially_retained_chunk():
    memory = state(pc_budget_frames=6, pc_recent_frames=2)
    for start in (0, 2, 4, 6):
        append(memory, start)
    chunk = memory.bank[2]
    # Different layers keep different tokens of the same frame; neither
    # overlap justifies dropping the complete historical chunk.
    memory.history = [chunk.layers[0].select(torch.tensor([0])), chunk.layers[1].select(torch.tensor([1, 2]))]
    memory.set_pose(pose(2))
    selected = memory._retrieve(2)
    assert len(selected) == 1 and selected[0].start == 2
    assert selected[0].layers[0].token_ids.tolist() == [5, 6, 7]
    assert selected[0].layers[1].token_ids.tolist() == [4, 7]
    for layer in range(2):
        retrieved = selected[0].layers[layer]
        assert not torch.isin(retrieved.token_ids, memory.history[layer].token_ids).any()
    assert chunk.layers[0].token_ids.tolist() == [4, 5, 6, 7]


def test_combined_small_fifo_bank_retrieves_over_long_rollout():
    config = vm.VideoMemoryConfig(
        window_frames=18,
        sink_frames=3,
        retrieval_frames=3,
        bank_bytes=4 * (2 * 24 * (2 * 8 * 4 + 16) + 128),
        world_keep_ratio=0.25,
        pc_budget_frames=12,
        pc_recent_frames=3,
    )
    memory = vm.VideoMemory(config, num_layers=2, tokens_per_frame=16, temporal_dim=4)
    generator = torch.Generator().manual_seed(7)
    hits = 0
    for start in range(0, 120, 3):
        data = [torch.randn(1, 48, 1, 8, generator=generator) for _ in range(2)]
        memory.set_pose(pose())
        for commit in (False, True):
            with memory.forward(start, 3, commit=commit):
                hits += len(memory._retrieved) if not commit else 0
                for layer, x in enumerate(data):
                    memory.assemble(layer, x, x, x)
                    previous = memory._previous(layer)
                    if previous is not None:
                        for chunk in memory._retrieved:
                            assert not torch.isin(chunk.layers[layer].token_ids, previous.token_ids).any()
        assert len(memory.bank) <= 4
    assert hits > 0
