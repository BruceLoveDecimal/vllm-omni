# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SeaCache on MiniMax-H3 through the extractor-driven hook."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.diffusion.cache.test_teacache_extractors import (
    _minimax_h3_sample_kwargs,
    _minimax_h3_small_od_config,
    _MiniMaxH3FakeAttention,
    _MiniMaxH3FakeLinear,
    _MiniMaxH3FakeMergedLinear,
    _MiniMaxH3FakeQKVLinear,
)
from vllm_omni.diffusion.attention.backends.abstract import VideoTokenLayout
from vllm_omni.diffusion.cache.seacache import SeaCacheBackend, SeaCacheConfig, SeaCacheContextHook, SeaCacheRootHook
from vllm_omni.diffusion.cache.seacache.indicators import minimax_h3_indicator
from vllm_omni.diffusion.cache.seacache.sea_filter import apply_sea_filter
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.forward_context import (
    set_forward_context,
    set_forward_context_denoise_step_idx,
    set_forward_context_denoise_total_steps,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def h3_module(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    monkeypatch.setattr(h3, "ColumnParallelLinear", _MiniMaxH3FakeLinear)
    monkeypatch.setattr(h3, "MergedColumnParallelLinear", _MiniMaxH3FakeMergedLinear)
    monkeypatch.setattr(h3, "QKVParallelLinear", _MiniMaxH3FakeQKVLinear)
    monkeypatch.setattr(h3, "RowParallelLinear", _MiniMaxH3FakeLinear)
    monkeypatch.setattr(h3, "Attention", _MiniMaxH3FakeAttention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)
    with patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=1):
        with patch("vllm.distributed.parallel_state.get_tp_group", return_value=MagicMock(world_size=1)):
            model = h3.MiniMaxH3DiTModel(_minimax_h3_small_od_config(), quant_config=None)
            for submodule in model.modules():
                if isinstance(submodule, h3.MiniMaxH3Attention):
                    submodule.rope._forward_method = submodule.rope.forward_native
            yield model.eval()


def _inputs(*, t_video: float = 0.3, t_audio: float = 0.6, seed: int = 0) -> dict:
    """Packed kwargs: text rows 0-1, a (1, 1, 2) video grid at rows 2-3, audio rows 4-5."""
    torch.manual_seed(seed)
    inputs = _minimax_h3_sample_kwargs(seq_len=8)
    inputs["video_token_layout"] = VideoTokenLayout(prefix_len=2, latent_grid=(1, 1, 2))
    timesteps = torch.full((8,), t_video)
    timesteps[4:6] = t_audio
    inputs["unique_timesteps"], inputs["inverse_indices"] = torch.unique(timesteps, sorted=True, return_inverse=True)
    return inputs


def _forward_at(module, inputs: dict, *, step: int, total: int):
    with set_forward_context(), torch.inference_mode():
        set_forward_context_denoise_step_idx(step)
        set_forward_context_denoise_total_steps(total)
        return module(**inputs)


def test_indicator_filters_each_modality_at_its_own_sigma(h3_module) -> None:
    inputs = _inputs(t_video=0.3, t_audio=0.6)
    video, audio = minimax_h3_indicator(h3_module, SeaCacheConfig(), **inputs) or []
    torch.testing.assert_close(video, apply_sea_filter(inputs["x"][0, 2:4].reshape(1, 1, 2, -1), 0.7))
    # Audio rows pack channel-major: (channels=2, t=1, C), filtered over time only.
    torch.testing.assert_close(audio, apply_sea_filter(inputs["audio_x"][0, 4:6].reshape(2, 1, -1), 0.4, dims=(1,)))
    inputs.pop("video_token_layout")
    assert minimax_h3_indicator(h3_module, SeaCacheConfig(), **inputs) is None


def test_backend_hook_reuses_residual_between_full_endpoints(h3_module, monkeypatch) -> None:
    block, block_calls = h3_module.blocks[0], []
    monkeypatch.setattr(block, "forward", lambda *a, **k: block_calls.append(1) or type(block).forward(block, *a, **k))
    pipeline = type("MiniMaxH3Pipeline", (), {"transformer": h3_module, "partition": "combined"})()
    backend = SeaCacheBackend(DiffusionCacheConfig(sea_threshold=100.0, sea_max_consecutive_cached=0))
    backend.enable(pipeline)
    hook = h3_module._hook_registry.get_hook(SeaCacheRootHook._HOOK_NAME)
    assert isinstance(hook, SeaCacheContextHook)

    total = 4
    for step in range(total):
        video, audio = _forward_at(h3_module, _inputs(t_video=0.1 * (step + 1), seed=step), step=step, total=total)
    # Step 0 and the last step always run the blocks; the two middle steps reuse the residual.
    assert len(block_calls) == 2
    assert (hook.full_count, hook.skip_count) == (2, 2)
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()

    backend.refresh(pipeline, num_inference_steps=total)
    assert (hook.full_count, hook.skip_count) == (0, 0)
    assert hook.state_manager._states == {}
