# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunk-causal streaming forward of the SANA-WM transformer on a tiny model.

CPU-only. The checks pin the two invariants the streaming path is built on:

* a first chunk with an empty cache is the bidirectional forward, bit for bit;
* for the *last* chunk of a sequence, the chunk-causal computation equals the
  bidirectional one restricted to that chunk (there is no future to isolate),
  so a one-block model run on ``[chunk0 | chunk1]`` is an exact reference for
  ``forward_streaming(chunk1, cache written by chunk0)`` — provided the state
  chunk 0 hands over does not itself depend on chunk 1. In the bidirectional
  reference the backward taps of the K short conv and the FFN temporal conv
  let chunk-0 activations peek into chunk 1, so those taps are neutralised for
  the whole-block check and covered by their own primitive tests instead.
  The whole-block check still drives the GDN main / camera state carry, the
  softmax main / camera K/V replay with absolute RoPE and the per-chunk
  camera geometry.
"""

from __future__ import annotations

import os

import pytest
import torch

from vllm_omni.diffusion.models.sana_wm.camera_control import (
    SanaWmCameraCondition,
    build_plucker_condition,
)
from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import (
    SanaWmSelfAttention,
    SanaWmTransformer3DModel,
    SanaWmWanRotaryPosEmbed,
    _bidirectional_delta_scan,
    _delta_scan,
)
from vllm_omni.diffusion.models.sana_wm.streaming_cache import SanaWmStreamingCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HIDDEN = 32
HEAD_DIM = 16
LATENT_CHANNELS = 4
PROMPT_CHANNELS = 8
TEXT_LEN = 5
LATENT_HW = 2  # 64x64 pixels / 32
CHUNK0_FRAMES = 4  # conditioning frame + one block
CHUNK1_FRAMES = 3
TOTAL_FRAMES = CHUNK0_FRAMES + CHUNK1_FRAMES


@pytest.fixture(autouse=True)
def _init_distributed():
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29517")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend="gloo")
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture(autouse=True)
def _force_sdpa_attention(monkeypatch):
    """Resolve the shared ``Attention`` layer to the device-agnostic SDPA backend.

    The platform default needs a CUDA device (and is unimplemented on CPU-only
    platforms); SDPA runs the same non-causal attention on CPU tensors.
    """
    from vllm_omni.diffusion.attention import selector
    from vllm_omni.platforms import current_omni_platform

    selector._cached_get_backend_cls.cache_clear()
    monkeypatch.setattr(
        current_omni_platform,
        "get_diffusion_attn_backend_cls",
        lambda *args, **kwargs: "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend",
    )
    if not (current_omni_platform.is_cuda() or current_omni_platform.is_rocm()):
        # ``AttentionImpl.forward`` dispatches on the platform; SDPA's CUDA
        # entry point is plain ``F.scaled_dot_product_attention`` and runs on
        # CPU tensors, so CPU-only hosts borrow it.
        monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    yield
    selector._cached_get_backend_cls.cache_clear()


@pytest.fixture(autouse=True)
def _force_default_gemm(monkeypatch):
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda: default_unquantized_gemm,
    )


def _tiny_config(*, num_blocks: int, softmax_every_n: int, streaming: bool = True) -> SanaWmConfig:
    pos_embed_type = "wan_rope"
    if streaming:
        pos_embed_type = "casual_wan_rope"
    return SanaWmConfig(
        architecture_name="tiny",
        num_blocks=num_blocks,
        hidden_size=HIDDEN,
        mlp_ratio=1.0,
        attn_type="BidirectionalGDNTriton",
        softmax_every_n=softmax_every_n,
        linear_head_dim=HEAD_DIM,
        conv_kernel_size=4,
        t_kernel_size=3,
        pos_embed_type=pos_embed_type,
        chunk_plucker_channels=48,
        chunk_plucker_post_attn_blocks=num_blocks,
        model_max_length=TEXT_LEN,
        streaming=streaming,
        chunk_size=CHUNK1_FRAMES,
    )


def _tiny_model(config: SanaWmConfig, seed: int = 0) -> SanaWmTransformer3DModel:
    torch.manual_seed(seed)
    model = SanaWmTransformer3DModel(
        config=config,
        latent_channels=LATENT_CHANNELS,
        prompt_channels=PROMPT_CHANNELS,
    )
    # Randomise every parameter (the zero-initialised camera / temporal
    # projections would otherwise leave the cached camera paths untested).
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.2)
    return model.eval()


def _camera_tensors(latent_frames: int) -> dict[str, torch.Tensor]:
    num_frames = 8 * (latent_frames - 1) + 1
    condition = SanaWmCameraCondition(
        poses=None,
        intrinsics={"fx": 32.0, "fy": 32.0, "cx": 32.0, "cy": 32.0},
        action=f"w-{num_frames - 1}",
        num_frames=num_frames,
        height=LATENT_HW * 32,
        width=LATENT_HW * 32,
        translation_speed=0.05,
        rotation_speed_deg=1.2,
    )
    tensors = build_plucker_condition(condition)
    return {
        "plucker": tensors["chunk_plucker"].float(),
        "raymap": tensors["raymap"].float(),
        "spatial_raymap": tensors["spatial_raymap"].float(),
    }


def _slice_camera(camera: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
    return {
        "plucker": camera["plucker"][:, start:end],
        "raymap": camera["raymap"][start:end],
        "spatial_raymap": camera["spatial_raymap"][:, start:end],
    }


def _inputs(seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, LATENT_CHANNELS, TOTAL_FRAMES, LATENT_HW, LATENT_HW, generator=generator)
    prompt = torch.randn(1, TEXT_LEN, PROMPT_CHANNELS, generator=generator)
    return latents, prompt


def _per_frame_timestep(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1)


def _new_cache(model: SanaWmTransformer3DModel) -> SanaWmStreamingCache:
    return SanaWmStreamingCache.new(
        num_blocks=len(model.blocks), chunk_size=CHUNK1_FRAMES, num_cached_blocks=-1, sink_token=True
    )


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_delta_scan_state_carry_equals_full_scan() -> None:
    torch.manual_seed(0)
    batch, heads, dim, frames, spatial = 1, 2, 4, 6, 3
    split = 4
    q = torch.rand(batch, heads, dim, frames * spatial)
    k = torch.rand(batch, heads, dim, frames * spatial)
    v = torch.randn(batch, heads, dim, frames * spatial)
    beta = torch.rand(batch, heads, frames, spatial)
    decay = torch.rand(batch, heads, frames)

    def part(tensor, start, end):
        width = tensor.shape[2]
        return tensor.view(batch, heads, width, frames, spatial)[:, :, :, start:end].reshape(batch, heads, width, -1)

    num_full, den_full = _delta_scan(q, k, v, beta, decay, spatial_tokens=spatial, query=q, key=k)
    num_a, den_a, (state_kv, state_z) = _delta_scan(
        part(q, 0, split),
        part(k, 0, split),
        part(v, 0, split),
        beta[:, :, :split],
        decay[:, :, :split],
        spatial_tokens=spatial,
        query=part(q, 0, split),
        key=part(k, 0, split),
        return_state=True,
    )
    num_b, den_b = _delta_scan(
        part(q, split, frames),
        part(k, split, frames),
        part(v, split, frames),
        beta[:, :, split:],
        decay[:, :, split:],
        spatial_tokens=spatial,
        query=part(q, split, frames),
        key=part(k, split, frames),
        initial_state_kv=state_kv,
        initial_state_z=state_z,
    )
    torch.testing.assert_close(torch.cat([num_a, num_b], dim=-1), num_full, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(torch.cat([den_a, den_b], dim=-1), den_full, rtol=1e-5, atol=1e-5)

    # Chunk-causal bidirectional: forward seeded, backward chunk-local. For the
    # last chunk this equals the full bidirectional scan restricted to it.
    num_bi_full, den_bi_full = _bidirectional_delta_scan(q, k, v, beta, decay, spatial_tokens=spatial, query=q, key=k)
    num_bi_b, den_bi_b, (carried_kv, _) = _bidirectional_delta_scan(
        part(q, split, frames),
        part(k, split, frames),
        part(v, split, frames),
        beta[:, :, split:],
        decay[:, :, split:],
        spatial_tokens=spatial,
        query=part(q, split, frames),
        key=part(k, split, frames),
        initial_state=(state_kv, state_z),
        return_state=True,
    )
    torch.testing.assert_close(num_bi_b, part(num_bi_full, split, frames), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(den_bi_b, part(den_bi_full, split, frames), rtol=1e-5, atol=1e-5)
    # The carried state is the forward state after the last frame.
    _, _, (state_full_kv, _) = _delta_scan(
        q, k, v, beta, decay, spatial_tokens=spatial, query=q, key=k, return_state=True
    )
    torch.testing.assert_close(carried_kv, state_full_kv, rtol=1e-5, atol=1e-5)

    # Numerator-only (camera) variant carries its state the same way.
    num_cam_full, _ = _delta_scan(q, k, v, beta, decay, spatial_tokens=spatial, skip_z=True)
    _, _, (cam_state, cam_z) = _delta_scan(
        part(q, 0, split),
        part(k, 0, split),
        part(v, 0, split),
        beta[:, :, :split],
        decay[:, :, :split],
        spatial_tokens=spatial,
        skip_z=True,
        return_state=True,
    )
    assert cam_z is None
    num_cam_b, _ = _delta_scan(
        part(q, split, frames),
        part(k, split, frames),
        part(v, split, frames),
        beta[:, :, split:],
        decay[:, :, split:],
        spatial_tokens=spatial,
        skip_z=True,
        initial_state_kv=cam_state,
    )
    torch.testing.assert_close(num_cam_b, part(num_cam_full, split, frames), rtol=1e-5, atol=1e-5)


def test_rope_frame_index_matches_contiguous_slice() -> None:
    rope = SanaWmWanRotaryPosEmbed(HEAD_DIM, max_seq_len=8)
    full = rope((TOTAL_FRAMES, LATENT_HW, LATENT_HW), torch.device("cpu"))
    chunk = rope(
        (CHUNK1_FRAMES, LATENT_HW, LATENT_HW),
        torch.device("cpu"),
        frame_index=torch.arange(CHUNK0_FRAMES, TOTAL_FRAMES),
    )
    tokens = LATENT_HW * LATENT_HW
    torch.testing.assert_close(chunk, full[:, :, CHUNK0_FRAMES * tokens :], rtol=0, atol=0)
    # Positions past the prebuilt table trigger a rebuild instead of an error.
    far = rope((2, LATENT_HW, LATENT_HW), torch.device("cpu"), frame_index=torch.tensor([40, 41]))
    assert far.shape == (1, 1, 2 * tokens, HEAD_DIM // 2)
    with pytest.raises(ValueError):
        rope((2, LATENT_HW, LATENT_HW), torch.device("cpu"), frame_index=torch.tensor([1, 2, 3]))


def test_temporal_short_conv_tail_matches_causal_conv_over_concatenation() -> None:
    config = _tiny_config(num_blocks=1, softmax_every_n=0)
    torch.manual_seed(1)
    attn = SanaWmSelfAttention(config, use_gdn=True)
    with torch.no_grad():
        attn.conv_k.weight.normal_()
    spatial = (TOTAL_FRAMES, LATENT_HW, LATENT_HW)
    tokens = LATENT_HW * LATENT_HW
    x = torch.randn(1, TOTAL_FRAMES * tokens, attn.num_heads * attn.head_dim)
    x0 = x[:, : CHUNK0_FRAMES * tokens]
    x1 = x[:, CHUNK0_FRAMES * tokens :]

    _, tail = attn._bidirectional_temporal_short_conv(
        x0, attn.conv_k, (CHUNK0_FRAMES, LATENT_HW, LATENT_HW), tail=None, return_tail=True
    )
    kernel = attn.conv_k.kernel_size[0]
    assert tail.shape == (tokens, kernel - 1, x.shape[-1])
    chunked, _ = attn._bidirectional_temporal_short_conv(
        x1, attn.conv_k, (CHUNK1_FRAMES, LATENT_HW, LATENT_HW), tail=tail, return_tail=True
    )
    # Last chunk of the sequence: the global forward pass plus a backward pass
    # that never crosses the chunk start equals the plain bidirectional conv.
    full = attn._bidirectional_temporal_short_conv(x, attn.conv_k, spatial)
    torch.testing.assert_close(chunked, full[:, CHUNK0_FRAMES * tokens :], rtol=1e-5, atol=1e-5)
    # ... and without the tail the chunk is isolated from the past.
    isolated = attn._bidirectional_temporal_short_conv(x1, attn.conv_k, (CHUNK1_FRAMES, LATENT_HW, LATENT_HW))
    assert not torch.allclose(isolated, chunked)


def test_ffn_temporal_conv_tail_matches_concatenation() -> None:
    """``SanaWmMbConvFfn`` depends on its input only, so the last chunk of the
    concatenation is an exact reference for the cached chunk."""
    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmMbConvFfn
    from vllm_omni.diffusion.models.sana_wm.streaming_cache import SanaWmBlockCache

    torch.manual_seed(2)
    ffn = SanaWmMbConvFfn(_tiny_config(num_blocks=1, softmax_every_n=0))
    with torch.no_grad():
        for param in ffn.parameters():
            param.normal_(0, 0.2)
    tokens = LATENT_HW * LATENT_HW
    x = torch.randn(1, TOTAL_FRAMES * tokens, HIDDEN)
    x0, x1 = x[:, : CHUNK0_FRAMES * tokens], x[:, CHUNK0_FRAMES * tokens :]
    with torch.no_grad():
        full = ffn(x, (TOTAL_FRAMES, LATENT_HW, LATENT_HW))
        cache = SanaWmBlockCache()
        first = ffn(x0, (CHUNK0_FRAMES, LATENT_HW, LATENT_HW), cache=cache, save_cache=True)
        assert cache.ffn_tconv_tail is not None and cache.ffn_tconv_tail.shape == (1, HIDDEN, 1, tokens)
        second = ffn(x1, (CHUNK1_FRAMES, LATENT_HW, LATENT_HW), cache=cache, save_cache=False)
        isolated = ffn(x1, (CHUNK1_FRAMES, LATENT_HW, LATENT_HW))
        plain_first = ffn(x0, (CHUNK0_FRAMES, LATENT_HW, LATENT_HW))
    # Chunk 0 with an empty tail is the plain (zero-padded) module.
    torch.testing.assert_close(first, plain_first, rtol=0, atol=0)
    torch.testing.assert_close(second, full[:, CHUNK0_FRAMES * tokens :], rtol=1e-5, atol=1e-5)
    assert not torch.allclose(isolated, second)


def _isolate_cross_chunk_taps(model: SanaWmTransformer3DModel) -> None:
    """Keep only the taps whose chunk-0 contribution cannot see chunk 1.

    The K short convs keep a random centre tap (so keys still pass through the
    conv path) and the FFN temporal conv is zeroed; both are exercised by the
    primitive tests above.
    """
    with torch.no_grad():
        for block in model.blocks:
            for conv in (block.attn.conv_k, block.attn.conv_k_cam):
                if conv is None:
                    continue
                centre = conv.weight[:, 0, -1].clone()
                conv.weight.zero_()
                conv.weight[:, 0, -1] = centre
            block.mlp.t_conv.weight.zero_()


# ---------------------------------------------------------------------------
# Whole-model checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("softmax_every_n", [0, 2], ids=["gdn", "gdn+softmax"])
def test_first_chunk_with_empty_cache_is_bidirectional_forward(softmax_every_n) -> None:
    model = _tiny_model(_tiny_config(num_blocks=2, softmax_every_n=softmax_every_n))
    latents, prompt = _inputs()
    camera = _camera_tensors(TOTAL_FRAMES)
    chunk0 = latents[:, :, :CHUNK0_FRAMES]
    timestep = _per_frame_timestep([0.0, 500.0, 500.0, 500.0])
    cam0 = _slice_camera(camera, 0, CHUNK0_FRAMES)
    with torch.no_grad():
        reference = model(chunk0, timestep, encoder_hidden_states=prompt, **cam0)
        cache = _new_cache(model)
        streamed = model.forward_streaming(
            chunk0,
            timestep,
            frame_index=torch.arange(CHUNK0_FRAMES),
            cache=cache,
            save_cache=True,
            encoder_hidden_states=prompt,
            **cam0,
        )
    torch.testing.assert_close(streamed, reference, rtol=0, atol=0)
    # The save pass populated every slot the block type owns.
    for block_index, block_cache in enumerate(cache.blocks):
        uses_gdn = model.blocks[block_index].attn.use_gdn
        assert (block_cache.gdn_state_kv is not None) == uses_gdn
        assert (block_cache.gdn_state_z is not None) == uses_gdn
        assert (block_cache.cam_state_kv is not None) == uses_gdn
        assert (block_cache.conv_k_tail is not None) == uses_gdn
        assert (block_cache.conv_k_cam_tail is not None) == uses_gdn
        assert (len(block_cache.softmax_chunks) == 1) == (not uses_gdn)
        assert block_cache.ffn_tconv_tail is not None
        if not uses_gdn:
            entry = block_cache.softmax_chunks[0]
            assert entry.chunk_index == 0 and entry.frames == CHUNK0_FRAMES
            assert entry.k_cam is not None and entry.v_cam is not None
    assert cache.nbytes() > 0


def test_denoise_pass_does_not_write_cache() -> None:
    model = _tiny_model(_tiny_config(num_blocks=2, softmax_every_n=2))
    latents, prompt = _inputs()
    camera = _camera_tensors(TOTAL_FRAMES)
    cache = _new_cache(model)
    with torch.no_grad():
        model.forward_streaming(
            latents[:, :, :CHUNK0_FRAMES],
            _per_frame_timestep([0.0, 500.0, 500.0, 500.0]),
            frame_index=torch.arange(CHUNK0_FRAMES),
            cache=cache,
            save_cache=False,
            encoder_hidden_states=prompt,
            **_slice_camera(camera, 0, CHUNK0_FRAMES),
        )
    assert cache.nbytes() == 0


@pytest.mark.parametrize("softmax_every_n", [0, 1], ids=["gdn-block", "softmax-block"])
def test_last_chunk_streaming_matches_bidirectional_reference(softmax_every_n) -> None:
    """One block: chunk 1 through the cache == bidirectional over [chunk0 | chunk1]."""
    model = _tiny_model(_tiny_config(num_blocks=1, softmax_every_n=softmax_every_n))
    _isolate_cross_chunk_taps(model)
    latents, prompt = _inputs()
    camera = _camera_tensors(TOTAL_FRAMES)
    t_chunk = 500.0
    # Chunk 0 is committed clean (t = 0 on every frame, the NVlabs save pass);
    # the reference therefore runs chunk-0 frames at t = 0 as well.
    reference_timestep = _per_frame_timestep([0.0] * CHUNK0_FRAMES + [t_chunk] * CHUNK1_FRAMES)
    with torch.no_grad():
        reference = model(latents, reference_timestep, encoder_hidden_states=prompt, **camera)
        cache = _new_cache(model)
        model.forward_streaming(
            latents[:, :, :CHUNK0_FRAMES],
            _per_frame_timestep([0.0] * CHUNK0_FRAMES),
            frame_index=torch.arange(CHUNK0_FRAMES),
            cache=cache,
            save_cache=True,
            encoder_hidden_states=prompt,
            **_slice_camera(camera, 0, CHUNK0_FRAMES),
        )
        cache.commit(CHUNK0_FRAMES)
        streamed = model.forward_streaming(
            latents[:, :, CHUNK0_FRAMES:],
            _per_frame_timestep([t_chunk] * CHUNK1_FRAMES),
            frame_index=torch.arange(CHUNK0_FRAMES, TOTAL_FRAMES),
            cache=cache,
            save_cache=False,
            encoder_hidden_states=prompt,
            **_slice_camera(camera, CHUNK0_FRAMES, TOTAL_FRAMES),
        )
        # Without the cache the chunk is cut off from its past.
        isolated = model.forward_streaming(
            latents[:, :, CHUNK0_FRAMES:],
            _per_frame_timestep([t_chunk] * CHUNK1_FRAMES),
            frame_index=torch.arange(CHUNK0_FRAMES, TOTAL_FRAMES),
            cache=_new_cache(model),
            save_cache=False,
            encoder_hidden_states=prompt,
            **_slice_camera(camera, CHUNK0_FRAMES, TOTAL_FRAMES),
        )
    torch.testing.assert_close(streamed, reference[:, :, CHUNK0_FRAMES:], rtol=1e-4, atol=1e-4)
    assert not torch.allclose(isolated, streamed, rtol=1e-3, atol=1e-3)


def test_softmax_window_trim_changes_attention_context() -> None:
    """After trimming, a chunk only sees the kept K/V (here: the sink chunk)."""
    model = _tiny_model(_tiny_config(num_blocks=1, softmax_every_n=1))
    latents, prompt = _inputs()
    frames = CHUNK0_FRAMES + 2 * CHUNK1_FRAMES
    extra = torch.randn(1, LATENT_CHANNELS, frames - TOTAL_FRAMES, LATENT_HW, LATENT_HW)
    latents = torch.cat([latents, extra], dim=2)
    camera = _camera_tensors(frames)

    def run(cache: SanaWmStreamingCache, start: int, end: int, *, save: bool) -> torch.Tensor:
        with torch.no_grad():
            out = model.forward_streaming(
                latents[:, :, start:end],
                _per_frame_timestep([0.0] * (end - start)),
                frame_index=torch.arange(start, end),
                cache=cache,
                save_cache=save,
                encoder_hidden_states=prompt,
                **_slice_camera(camera, start, end),
            )
        if save:
            cache.commit(end - start)
        return out

    full = SanaWmStreamingCache.new(num_blocks=1, chunk_size=CHUNK1_FRAMES, num_cached_blocks=-1, sink_token=True)
    sink_only = SanaWmStreamingCache.new(num_blocks=1, chunk_size=CHUNK1_FRAMES, num_cached_blocks=1, sink_token=True)
    boundaries = [0, CHUNK0_FRAMES, TOTAL_FRAMES, frames]
    for cache in (full, sink_only):
        for start, end in zip(boundaries[:-2], boundaries[1:-1]):
            run(cache, start, end, save=True)
    assert [c.chunk_index for c in full.blocks[0].softmax_chunks] == [0, 1]
    assert [c.chunk_index for c in sink_only.blocks[0].softmax_chunks] == [0]
    out_full = run(full, TOTAL_FRAMES, frames, save=False)
    out_sink = run(sink_only, TOTAL_FRAMES, frames, save=False)
    assert not torch.allclose(out_full, out_sink, rtol=1e-3, atol=1e-3)


def test_forward_streaming_rejects_bad_contracts() -> None:
    model = _tiny_model(_tiny_config(num_blocks=1, softmax_every_n=0))
    latents, prompt = _inputs()
    cache = _new_cache(model)
    with pytest.raises(ValueError, match="per-frame timestep"):
        model.forward_streaming(
            latents[:, :, :CHUNK0_FRAMES],
            torch.tensor([0.0]),
            frame_index=torch.arange(CHUNK0_FRAMES),
            cache=cache,
            save_cache=False,
            encoder_hidden_states=prompt,
        )
    with pytest.raises(ValueError, match="block slots"):
        model.forward_streaming(
            latents[:, :, :CHUNK0_FRAMES],
            _per_frame_timestep([0.0] * CHUNK0_FRAMES),
            frame_index=torch.arange(CHUNK0_FRAMES),
            cache=SanaWmStreamingCache.new(num_blocks=3, chunk_size=3, num_cached_blocks=2, sink_token=True),
            save_cache=False,
            encoder_hidden_states=prompt,
        )
