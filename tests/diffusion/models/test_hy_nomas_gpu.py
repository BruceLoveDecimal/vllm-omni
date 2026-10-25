# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CUDA component smoke with real shared SDPA; not released-weight parity.

Small random models exercise production dispatch without checkpoints or reference
source downloads. No distributed process group is needed: both components opt out
of sequence parallelism and the config uses single-rank parallelism.
"""

import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.hy_nomas.cache import HyNomasKVCache
from vllm_omni.diffusion.models.hy_nomas.decoder import PixelUMMDecoder
from vllm_omni.diffusion.models.hy_nomas.transformer import HyNomasTransformer3DModel

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": ["L4", "B200"]}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.fixture(params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def gpu_context(request, monkeypatch):
    # Exercise real single-rank linear kernels without a process group.
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: 1)
    dtype = request.param
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")
    config = OmniDiffusionConfig(
        dtype=dtype,
        diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}},
    )
    with (
        torch.random.fork_rng(devices=[torch.accelerator.current_device_index()]),
        set_current_diffusion_config(config),
        set_forward_context(omni_diffusion_config=config),
        torch.inference_mode(),
    ):
        torch.manual_seed(17)
        yield dtype


def _assert_shared_sdpa(model):
    layers = [module for module in model.modules() if isinstance(module, Attention)]
    assert layers, "test must execute the production shared attention layer"
    assert all(layer.attn_backend is SDPABackend for layer in layers)
    assert all(layer.skip_sequence_parallel for layer in layers)


def test_dit_cache_camera_and_action_cuda(gpu_context):
    dtype = gpu_context
    model = (
        HyNomasTransformer3DModel(
            in_channels=4,
            out_channels=4,
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=128,
            text_embed_dim=8,
            mlp_ratio=2,
            rope_axes_dim=(16, 56, 56),
            camera_embed_dim=8,
            camera_attention_dim=256,
            camera_num_attention_heads=2,
            camera_rope_axes_dim=(16, 56, 56),
        )
        .to(device="cuda", dtype=dtype)
        .eval()
    )
    for module in model.modules():
        if isinstance(module, LinearBase):
            torch.nn.init.normal_(module.weight, std=0.02)
            torch.nn.init.zeros_(module.bias)
    _assert_shared_sdpa(model)
    latents = torch.randn(1, 4, 2, 2, 2, device="cuda", dtype=dtype)
    text = torch.randn(1, 5, 8, device="cuda", dtype=dtype)
    mask = torch.tensor([[True, True, True, True, False]], device="cuda")
    poses = torch.eye(4, device="cuda").repeat(1, 5, 1, 1)
    poses[:, :, 0, 3] = torch.linspace(0, 1, 5, device="cuda")
    caches: list[dict[str, torch.Tensor]] = [{}, {}]
    args = dict(
        hidden_states=latents,
        encoder_hidden_states=text,
        timestep=torch.tensor([500.0], device="cuda"),
        encoder_attention_mask=mask,
        action_text_offsets=[[2, 3, 4]],
        action_frame_windows=[(0, 1, 0), (1, 2, 1)],
        cache_action_start=2,
        kv_cache=caches,
    )
    baseline = model(**args).sample
    gated = model(**args, camera_extrinsics=poses, camera_attention_mask=torch.zeros(1, device="cuda")).sample
    torch.testing.assert_close(gated, baseline)
    padded = torch.cat([poses[:, :1].repeat(1, 3, 1, 1), poses], dim=1)
    camera = model(**args, camera_extrinsics=poses).sample
    torch.testing.assert_close(camera, model(**args, camera_extrinsics=padded).sample)
    assert caches == [{}, {}]
    committed = model(**args, camera_extrinsics=poses, update_cache=True).sample
    torch.testing.assert_close(committed, camera)
    snapshots = [{name: tensor.clone() for name, tensor in cache.items()} for cache in caches]
    for cache in caches:
        assert cache["key"].shape == (1, 13, 2, 128)
        assert cache["mask"].tolist() == [[True] * 10 + [False] * 3]
    continued = model(**args, camera_extrinsics=poses, frame_offset=2).sample
    assert continued.shape == latents.shape
    assert continued.dtype == dtype
    assert torch.isfinite(continued).all()
    assert not torch.equal(continued, camera)
    for cache, snapshot in zip(caches, snapshots):
        for name in snapshot:
            torch.testing.assert_close(cache[name], snapshot[name], atol=0, rtol=0)
    reserved = [HyNomasKVCache(26) for _ in caches]
    args["kv_cache"] = reserved
    cached_commit = model(**args, camera_extrinsics=poses, update_cache=True).sample
    assert torch.equal(cached_commit, committed)
    cached_continued = model(**args, camera_extrinsics=poses, frame_offset=2).sample
    assert torch.equal(cached_continued, continued)
    for reserved_cache, snapshot in zip(reserved, snapshots):
        for name in snapshot:
            assert torch.equal(getattr(reserved_cache, name)[:, : reserved_cache.length], snapshot[name])


@pytest.mark.parametrize("tile_pixels", [0, 16])
def test_pixelumm_single_window_and_stream_cuda(gpu_context, monkeypatch, tile_pixels):
    dtype = gpu_context
    model = (
        PixelUMMDecoder(
            hidden_size=256,
            intermediate_size=384,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            head_dim=128,
            temporal_window_latents=3,
            temporal_overlap_latents=1,
            tail_hold_latents=1,
            local_attention_tile_pixels=tile_pixels,
        )
        .to(device="cuda", dtype=dtype)
        .eval()
    )
    _assert_shared_sdpa(model)
    latents = torch.randn(1, 32, 8, 1, 1, device="cuda", dtype=dtype)
    noise = model._make_noise(latents, 0, 3, model.config.seed)
    single = model(latents[:, :, :3], noise).sample
    decoded = model.decode(latents[:, :, :3], return_dict=False)[0]
    assert single.shape == (1, 3, 12, 32, 32)
    assert torch.isfinite(single).all()
    torch.testing.assert_close(single, decoded)
    # Compare native GQA dispatch against the former expanded-head SDPA path.
    # Exercise both global attention and partial spatial neighborhoods.
    with monkeypatch.context() as patch:
        for layer in model.modules():
            if isinstance(layer, Attention):
                original = layer.forward

                def expanded(query, key, value, forward=original):
                    assert key.shape[2] == 1
                    repeat = query.shape[2] // key.shape[2]
                    return forward(query, key.repeat_interleave(repeat, dim=2), value.repeat_interleave(repeat, dim=2))

                patch.setattr(layer, "forward", expanded)
        expanded_output = model.decode(latents[:, :, :3], return_dict=False)[0]
    torch.testing.assert_close(decoded, expanded_output)
    complete = model.decode(latents, return_dict=False)[0]
    session = model.create_streaming_decode_session(to_cpu=False, retain_history=False)
    first = session.push(latents[:, :, :4], pixel_frames=13)
    saved = first.clone()
    second = session.push(latents[:, :, 4:], pixel_frames=15)
    tail = session.finish()
    streamed = torch.cat([part for part in (first, second, tail) if part.numel()], dim=2)
    expected = torch.cat([complete[:, :, :13], complete[:, :, 16:31]], dim=2)
    assert streamed.shape == (1, 3, 28, 32, 32)
    assert torch.isfinite(streamed).all()
    torch.testing.assert_close(streamed, expected)
    torch.testing.assert_close(first, saved, atol=0, rtol=0)
    assert session.stable_pixel_frames == 28
    with pytest.raises(RuntimeError, match="after finish"):
        session.push(latents[:, :, :1])
