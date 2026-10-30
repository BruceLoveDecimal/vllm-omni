# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Factorial mechanism isolation, including disabled-path and canonical-KV checks."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from tests.diffusion.ar_diffusion.test_video_memory import absolute_keys, vm

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("rebase", [False, True], ids=["absolute", "rebase"])
@pytest.mark.parametrize("bank", ["off", "raw", "compressed"])
@pytest.mark.parametrize("pc", [False, True], ids=["window", "pc"])
def test_factorial_mechanisms_execute_only_when_enabled(rebase, bank, pc):
    config = vm.VideoMemoryConfig(
        window_frames=18,
        sink_frames=3,
        rebase_sink=rebase,
        rebase_retrieval=False,
        retrieval_frames=0 if bank == "off" else 3,
        bank_bytes=0 if bank == "off" else 1_000_000,
        world_keep_ratio=0.25 if bank == "compressed" else 1.0,
        pc_budget_frames=12 if pc else 0,
        pc_recent_frames=3,
    )
    state = vm.VideoMemory(config, num_layers=2, tokens_per_frame=2, temporal_dim=4)
    generator = torch.Generator().manual_seed(72)
    raw_keys = torch.randn(1, 120, 2, 8, generator=generator, dtype=torch.float64)
    canonical_keys = absolute_keys(raw_keys, torch.arange(120) // 2)
    values = torch.randn(1, 120, 2, 8, generator=generator, dtype=torch.float64)
    # An exact token identifier in V permits independent reconstruction of the
    # selected attention rows without using the production selection indices.
    values[0, :, 0, 0] = torch.arange(120)
    hits = 0
    with (
        patch.object(vm, "rebase_temporal_keys", wraps=vm.rebase_temporal_keys) as rotate,
        patch.object(vm, "participative_indices", wraps=vm.participative_indices) as select,
        patch.object(vm, "world_compress", wraps=vm.world_compress) as compress,
    ):
        for start in range(0, 60, 3):
            pose = torch.eye(4)
            pose[0, 3] = min(start, 60 - start) / 30
            if bank != "off":
                state.set_pose(pose)
            current = slice(start * 2, (start + 3) * 2)
            query = raw_keys[:, current]
            for commit in (False, True):
                with state.forward(start, 3, commit=commit):
                    hits += len(state._retrieved) if not commit else 0
                    for layer in range(2):
                        key, value = state.assemble(layer, query, canonical_keys[:, current], values[:, current])
                        ids = value[0, :, 0, 0].long()
                        assert len(ids.unique()) == len(ids)
                        torch.testing.assert_close(value, values[:, ids], atol=0, rtol=0)
                        if not rebase:
                            expected_key = canonical_keys[:, ids]
                            torch.testing.assert_close(key, expected_key, atol=0, rtol=0)
                            logits = torch.einsum("bqhd,bkhd->bhqk", query, expected_key) / 8**0.5
                            expected = torch.einsum("bhqk,bkhd->bqhd", logits.softmax(-1), values[:, ids])
                            actual = F.scaled_dot_product_attention(
                                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
                            ).transpose(1, 2)
                            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            for history in state.history:
                torch.testing.assert_close(history.key, canonical_keys[:, history.token_ids], atol=0, rtol=0)
            if bank == "off":
                assert not state.bank and not state._device_retrieved
            else:
                for chunk in state.bank.values():
                    for rows in chunk.layers:
                        assert rows.key.shape[1] == (3 if bank == "compressed" else 6)
                        torch.testing.assert_close(rows.key, canonical_keys[:, rows.token_ids], atol=0, rtol=0)
            assert bool(state._query_history) == pc
        assert bool(rotate.call_count) == rebase
        assert bool(select.call_count) == pc
        assert bool(compress.call_count) == (bank != "off")
        assert bool(hits) == (bank != "off")
        assert all(call.args[2] == config.world_keep_ratio for call in compress.call_args_list)
    state.close()


@pytest.mark.parametrize("window", [12, 18])
@pytest.mark.parametrize("sink", [0, 3])
def test_fixed_window_controls_match_explicit_token_ranges(window, sink):
    state = vm.VideoMemory(
        vm.VideoMemoryConfig(window_frames=window, sink_frames=sink, rebase_sink=False),
        num_layers=1,
        tokens_per_frame=2,
        temporal_dim=4,
    )
    generator = torch.Generator().manual_seed(83)
    raw_keys = torch.randn(1, 120, 2, 8, generator=generator, dtype=torch.float64)
    keys = absolute_keys(raw_keys, torch.arange(120) // 2)
    values = torch.randn(1, 120, 2, 8, generator=generator, dtype=torch.float64)
    for start in range(0, 60, 3):
        end = start + 3
        tail_start = max(sink, end - window + sink)
        ids = torch.cat((torch.arange(min(sink, start) * 2), torch.arange(min(tail_start, start) * 2, end * 2)))
        with state.forward(start, 3, commit=True):
            key, value = state.assemble(
                0, raw_keys[:, start * 2 : end * 2], keys[:, start * 2 : end * 2], values[:, start * 2 : end * 2]
            )
            torch.testing.assert_close(key, keys[:, ids], atol=0, rtol=0)
            torch.testing.assert_close(value, values[:, ids], atol=0, rtol=0)
            assert key.shape[1] <= window * 2
    state.close()
