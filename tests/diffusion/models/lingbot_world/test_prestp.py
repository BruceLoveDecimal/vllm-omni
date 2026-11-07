# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import runpy
from pathlib import Path

import pytest
import torch

from tests.diffusion.models.lingbot_world import test_lingbot_world_attention as attention_tests
from tests.diffusion.models.lingbot_world import test_lingbot_world_transformer as transformer_tests

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def state():
    module = runpy.run_path(str(Path(__file__).parents[4] / "vllm_omni/diffusion/models/lingbot_world/prestp.py"))
    return module["PreSTPState"](module["PreSTPConfig"]())


def test_warmup_static_restore_and_reanchor():
    pruning = state()
    clean = torch.ones(1, 3, 4, 8)
    pruning.begin_chunk(clean)
    pruning.update(clean, step_index=0)
    assert pruning.keep_indices().numel() == 12
    pruning.finish_chunk()
    pruning.begin_chunk(clean)
    pruning.update(clean, step_index=0)
    assert pruning.keep_indices().numel() == 0
    torch.testing.assert_close(pruning.update(clean * 5, step_index=1), clean)
    pruning.finish_chunk()
    pruning.begin_chunk(clean)
    assert not pruning.probe
    pruning.update(clean, step_index=0)
    pruning.finish_chunk()
    pruning.begin_chunk(clean)
    pruning.update(clean, step_index=0)
    pruning.finish_chunk()
    pruning.begin_chunk(clean)
    assert pruning.probe
    assert pruning.keep_indices().numel() == 12


def test_patch_roundtrip_and_frozen_values():
    pruning = state()
    latent = torch.randn(1, 16, 3, 8, 10)
    torch.testing.assert_close(pruning.unpatchify(pruning.patchify(latent), latent.shape), latent)
    clean = torch.randn(1, 3, 4, 8)
    pruning.begin_chunk(clean)
    pruning.update(clean, step_index=0)
    pruning.finish_chunk()
    pruning.begin_chunk(clean)
    pruning.update(-clean, step_index=0)
    pruning.update(-clean, step_index=1)
    assert pruning.keep_indices().numel() == 0
    torch.testing.assert_close(pruning.update(clean, step_index=2), -clean)


@torch.inference_mode()
def test_sparse_queries_match_dense_attention_at_same_kv():
    module = attention_tests._load_module()
    torch.manual_seed(1)
    attention = module.LingBotSelfAttention(dim=8, num_heads=2)
    for parameter in attention.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    cache = module.allocate_lingbot_cache(
        batch_size=1,
        num_layers=1,
        max_tokens=8,
        num_local_heads=2,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).self_attention[0]
    hidden = torch.randn(1, 8, 8)
    pruning = state()
    dense = attention(hidden, cache=cache, current_start=0, sink_tokens=0, update_cache=False)
    capture = attention(
        hidden,
        cache=cache,
        current_start=0,
        sink_tokens=0,
        update_cache=False,
        prestp=pruning,
    )
    torch.testing.assert_close(capture, dense, rtol=0, atol=0)
    keep = torch.tensor([0, 3, 7])
    sparse = attention(
        hidden[:, keep],
        cache=cache,
        current_start=0,
        sink_tokens=0,
        update_cache=False,
        prestp=pruning,
        keep_indices=keep,
    )
    torch.testing.assert_close(sparse, dense[:, keep], rtol=1e-5, atol=1e-6)


@torch.inference_mode()
def test_transformer_selects_matching_camera_time_and_rotary_positions():
    module = attention_tests._load_module()
    torch.manual_seed(8)
    model = transformer_tests._tiny_model(module, num_frames_per_block=3, sliding_window_num_frames=6).eval()
    hidden = torch.randn(1, 36, 3, 4, 4)
    camera = torch.randn(1, 384, 3, 4, 4)
    text = torch.randn(1, 3, 6)
    timestep = torch.tensor([750.0])
    cache = transformer_tests._cache(module, model, max_tokens=12)
    pruning = state()
    pruning.begin_chunk(torch.zeros(1, 3, 4, 8))
    args = (hidden, timestep, text, camera)
    dense = model(*args, cache=cache, start_frame=0, update_cache=False)
    captured = model(*args, cache=cache, start_frame=0, update_cache=False, prestp=pruning)
    torch.testing.assert_close(captured, dense, rtol=0, atol=0)
    pruning.static_mask.fill_(True)
    keep = torch.tensor([0, 3, 5, 9, 11])
    pruning.static_mask.flatten()[keep] = False
    sparse = model(*args, cache=cache, start_frame=0, update_cache=False, prestp=pruning)
    dense_patches = pruning.patchify(dense).flatten(1, 2)
    sparse_patches = pruning.patchify(sparse).flatten(1, 2)
    torch.testing.assert_close(sparse_patches[:, keep], dense_patches[:, keep], rtol=1e-4, atol=1e-5)
    assert torch.count_nonzero(sparse_patches[:, [1, 2, 4, 6, 7, 8, 10]]) == 0
    assert len(model._timestep_projection_buffers) == 1
    pruning.static_mask.fill_(True)
    skipped = model(*args, cache=cache, start_frame=0, update_cache=False, prestp=pruning)
    assert torch.count_nonzero(skipped) == 0
    committed = model(*args, cache=cache, start_frame=0, update_cache=True, prestp=pruning)
    torch.testing.assert_close(committed, dense, rtol=0, atol=0)
    for index, layer in enumerate(cache.self_attention):
        assert layer.absolute_end == 12
        assert layer.last_start == 0
        torch.testing.assert_close(layer.value[:, :12], pruning.layer_kv[index][1])
