# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunk-causal LTX-2 refiner (SANA-WM Stage-2) on a tiny native LTX-2 transformer.

CPU-only. Pins the K/V-prefix contract of ``refiner.py``:

* the prefix path (sink pre-RoPE K/V re-rotated + history post-RoPE K/V
  concatenated in front of the current block) equals one plain video-only
  forward over the concatenated frames ``[sink | history | block]`` at their
  absolute positions, restricted to the block's tokens;
* the post-RoPE capture pass returns exactly the K/V a full forward computes;
* the sliding window keeps ``kv_max_frames - sink_frames`` frames;
* the Euler update matches a hand computation; the schedule validates.
"""

from __future__ import annotations

import os

import pytest
import torch

from vllm_omni.diffusion.models.ltx2.ltx2_components import create_transformer_from_config
from vllm_omni.diffusion.models.sana_wm.refiner import (
    SanaWmRefinerKvCache,
    SanaWmRefinerRunner,
    SanaWmRefinerSchedule,
    refiner_rotary_emb,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HEADS = 2
HEAD_DIM = 8
INNER = HEADS * HEAD_DIM
LATENT_CHANNELS = 4
CAPTION_CHANNELS = 6
TEXT_LEN = 3
LATENT_H, LATENT_W = 2, 3
TOKENS_PER_FRAME = LATENT_H * LATENT_W
BLOCK = 3
SINK = 1
FPS = 16.0


@pytest.fixture(autouse=True)
def _init_distributed():
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29519")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend="gloo")
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture(autouse=True)
def _force_sdpa_attention(monkeypatch):
    from vllm_omni.diffusion.attention import selector
    from vllm_omni.platforms import current_omni_platform

    selector._cached_get_backend_cls.cache_clear()
    monkeypatch.setattr(
        current_omni_platform,
        "get_diffusion_attn_backend_cls",
        lambda *args, **kwargs: "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend",
    )
    if not (current_omni_platform.is_cuda() or current_omni_platform.is_rocm()):
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


def _tiny_transformer(rope_type: str = "split", num_layers: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    config = {
        "in_channels": LATENT_CHANNELS,
        "out_channels": LATENT_CHANNELS,
        "patch_size": 1,
        "patch_size_t": 1,
        "num_attention_heads": HEADS,
        "attention_head_dim": HEAD_DIM,
        "cross_attention_dim": INNER,
        "audio_in_channels": 2,
        "audio_out_channels": 2,
        "audio_num_attention_heads": 1,
        "audio_attention_head_dim": 4,
        "audio_cross_attention_dim": 4,
        "num_layers": num_layers,
        "caption_channels": CAPTION_CHANNELS,
        "rope_type": rope_type,
        "pos_embed_max_pos": 20,
        "base_height": 64,
        "base_width": 96,
        "vae_scale_factors": (8, 32, 32),
    }
    model = create_transformer_from_config(config)
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.2)
    return model.eval()


def _runner(model, **kwargs) -> SanaWmRefinerRunner:
    return SanaWmRefinerRunner(model, sink_frames=SINK, block_size=BLOCK, kv_max_frames=kwargs.pop("kv", 11))


def _prompt(seed: int = 5):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, TEXT_LEN, CAPTION_CHANNELS, generator=generator)


def _latents(frames: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, LATENT_CHANNELS, frames, LATENT_H, LATENT_W, generator=generator)


# ---------------------------------------------------------------------------
# Schedule / cache bookkeeping
# ---------------------------------------------------------------------------


def test_schedule_defaults_and_validation() -> None:
    schedule = SanaWmRefinerSchedule()
    assert schedule.sigmas == (0.909375, 0.725, 0.421875, 0.0)
    assert schedule.num_steps == 3
    assert schedule.pairs() == [(0.909375, 0.725), (0.725, 0.421875), (0.421875, 0.0)]
    schedule.check_num_steps(None)
    schedule.check_num_steps(3)
    with pytest.raises(ValueError, match="refiner_steps must be 3"):
        schedule.check_num_steps(4)
    for bad in [(0.5,), (0.9, 0.5), (0.9, 0.95, 0.0), (1.5, 0.0)]:
        with pytest.raises(ValueError):
            SanaWmRefinerSchedule(bad)


def test_cache_window_keeps_last_frames() -> None:
    cache = SanaWmRefinerKvCache(num_layers=1, tokens_per_frame=2, sink_frames=1, block_size=3, kv_max_frames=8)
    assert cache.max_history_frames == 7
    for block in range(4):
        k = torch.full((1, 6, 4), float(block))
        cache.append_history([(k, k.clone())], 3)
    assert cache.blocks_refined == 4
    assert cache.history_frames == 7
    hk, _ = cache.history_kv_post[0]
    assert hk.shape[1] == 14  # 7 frames * 2 tokens
    # Oldest frames evicted first: the retained tokens end with block 3 and
    # start inside block 1 (block 0 fully gone, 2 of 3 frames of block 1 gone).
    assert hk[0, :, 0].tolist() == [1.0] * 2 + [2.0] * 6 + [3.0] * 6
    assert cache.cached_frames() == 7  # no sink captured yet
    with pytest.raises(ValueError, match="kv_max_frames"):
        SanaWmRefinerKvCache(num_layers=1, tokens_per_frame=2, sink_frames=1, block_size=3, kv_max_frames=3)


# ---------------------------------------------------------------------------
# Prefix attention == concatenated-sequence attention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rope_type", ["split", "interleaved"])
def test_block0_prefix_forward_matches_concatenated_forward(rope_type) -> None:
    """Sink prefix (re-rotated at its own position 0) == forward over ``[sink | block]``."""
    model = _tiny_transformer(rope_type)
    runner = _runner(model)
    prompt = _prompt()
    sink = _latents(SINK, seed=1)
    block = _latents(BLOCK, seed=2)
    cache = runner.new_cache(latent_height=LATENT_H, latent_width=LATENT_W)
    runner.capture_sink(cache, sink, encoder_hidden_states=prompt, encoder_attention_mask=None, fps=FPS)
    assert cache.sink_kv_pre is not None and len(cache.sink_kv_pre) == 2
    assert cache.sink_kv_pre[0][0].shape == (1, SINK * TOKENS_PER_FRAME, INNER)

    sigma = 0.725
    positions = list(range(SINK, SINK + BLOCK))
    rope = refiner_rotary_emb(
        model, frame_positions=positions, height=LATENT_H, width=LATENT_W, batch_size=1, device=block.device, fps=FPS
    )
    prefixes = runner._prefixes(
        cache, block_start=SINK, height=LATENT_H, width=LATENT_W, batch_size=1, device=block.device, fps=FPS
    )
    out, _ = runner.forward_video(
        runner._pack(block),
        sigma=sigma,
        encoder_hidden_states=prompt,
        encoder_attention_mask=None,
        rotary_emb=rope,
        prefixes=prefixes,
    )
    # Reference: one block-causal joint forward over [sink | block] at absolute
    # positions (sink rows at sigma 0 attending the sink only, as when their
    # K/V were captured; block rows at sigma attending everything).
    reference = _reference_forward_per_token(
        runner,
        torch.cat([sink, block], dim=2),
        [0] + positions,
        [SINK * TOKENS_PER_FRAME, BLOCK * TOKENS_PER_FRAME],
        [0.0, sigma],
        prompt,
    )

    torch.testing.assert_close(out, reference[:, SINK * TOKENS_PER_FRAME :], atol=2e-4, rtol=2e-4)


def _reference_forward_per_token(runner, latents, positions, segments, sigmas_per_segment, prompt):
    """Independent video-only forward over the whole sequence with a block-causal mask.

    ``segments`` are token counts; tokens of segment ``i`` attend to segments
    ``<= i`` and run at ``sigmas_per_segment[i]``. Sink / history rows therefore
    see exactly what they saw when their K/V were captured, so the last
    segment's rows are the reference for the prefix path.
    """
    import torch.nn.functional as F

    from vllm_omni.diffusion.models.ltx2.ltx2_transformer import LTX2AudioVideoAttnProcessor
    from vllm_omni.diffusion.models.sana_wm.refiner import _apply_rotary

    transformer = runner.transformer
    tokens = runner._pack(latents)
    batch_size, seq_len, _ = tokens.shape
    assert sum(segments) == seq_len
    rope = refiner_rotary_emb(
        transformer,
        frame_positions=positions,
        height=LATENT_H,
        width=LATENT_W,
        batch_size=1,
        device=tokens.device,
        fps=FPS,
    )
    segment_id = torch.repeat_interleave(torch.arange(len(segments)), torch.tensor(segments))
    allowed = segment_id[:, None] >= segment_id[None, :]  # [N_q, N_k]
    sigmas = torch.repeat_interleave(torch.tensor(sigmas_per_segment, dtype=torch.float32), torch.tensor(segments))

    hidden = transformer.proj_in(tokens)
    timestep = sigmas.view(1, seq_len) * runner.timestep_scale
    temb, embedded = transformer.time_embed(timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden.dtype)
    temb = temb.view(batch_size, -1, temb.size(-1))
    embedded = embedded.view(batch_size, -1, embedded.size(-1))
    enc = transformer.caption_projection(prompt).view(batch_size, -1, hidden.size(-1))

    for block in transformer.transformer_blocks:
        ada = block.get_mod_params(block.scale_shift_table, temb, batch_size)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada[:6]
        norm = block.norm1(hidden) * (1 + scale_msa) + shift_msa
        attn = block.attn1
        q, k, v = LTX2AudioVideoAttnProcessor._project_qkv(
            attn=attn, hidden_states=norm, encoder_hidden_states=norm, is_self_attention=True
        )
        q = _apply_rotary(attn, attn.norm_q(q), rope)
        k = _apply_rotary(attn, attn.norm_k(k), rope)
        q = q.unflatten(2, (attn.heads, attn.head_dim)).transpose(1, 2)
        k = k.unflatten(2, (attn.heads, attn.head_dim)).transpose(1, 2)
        v = v.unflatten(2, (attn.heads, attn.head_dim)).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[None, None])
        out = out.transpose(1, 2).flatten(2, 3)
        out = attn.to_out[1](attn.to_out[0](out))
        hidden = hidden + out * gate_msa
        hidden = hidden + block.attn2(block.norm2(hidden), encoder_hidden_states=enc, query_rotary_emb=None)
        hidden = hidden + block.ff(block.norm3(hidden) * (1 + scale_mlp) + shift_mlp) * gate_mlp
    scale_shift = transformer.scale_shift_table[None, None] + embedded[:, :, None]
    shift, scale = scale_shift[:, :, 0], scale_shift[:, :, 1]
    hidden = transformer.norm_out(hidden) * (1 + scale) + shift
    return transformer.proj_out(hidden)


def test_history_prefix_matches_concatenated_forward() -> None:
    """After one refined block, ``[sink | history | block]`` prefix == joint forward at absolute positions."""
    model = _tiny_transformer("split")
    runner = _runner(model)
    prompt = _prompt()
    sink = _latents(SINK, seed=1)
    refined0 = _latents(BLOCK, seed=7)  # stands in for the refined block 0 (clean, sigma = 0)
    block1 = _latents(BLOCK, seed=8)
    cache = runner.new_cache(latent_height=LATENT_H, latent_width=LATENT_W)
    runner.capture_sink(cache, sink, encoder_hidden_states=prompt, encoder_attention_mask=None, fps=FPS)

    # Capture block 0's post-RoPE K/V under its own prefix (sink only).
    pos0 = list(range(SINK, SINK + BLOCK))
    rope0 = refiner_rotary_emb(
        model, frame_positions=pos0, height=LATENT_H, width=LATENT_W, batch_size=1, device=sink.device, fps=FPS
    )
    prefixes0 = runner._prefixes(
        cache, block_start=SINK, height=LATENT_H, width=LATENT_W, batch_size=1, device=sink.device, fps=FPS
    )
    _, captured = runner.forward_video(
        runner._pack(refined0),
        sigma=0.0,
        encoder_hidden_states=prompt,
        encoder_attention_mask=None,
        rotary_emb=rope0,
        prefixes=prefixes0,
        capture="post_rope",
    )
    cache.append_history(captured, BLOCK)
    assert cache.history_frames == BLOCK

    sigma = 0.421875
    start1 = SINK + BLOCK
    pos1 = list(range(start1, start1 + BLOCK))
    rope1 = refiner_rotary_emb(
        model, frame_positions=pos1, height=LATENT_H, width=LATENT_W, batch_size=1, device=sink.device, fps=FPS
    )
    prefixes1 = runner._prefixes(
        cache, block_start=start1, height=LATENT_H, width=LATENT_W, batch_size=1, device=sink.device, fps=FPS
    )
    out, _ = runner.forward_video(
        runner._pack(block1),
        sigma=sigma,
        encoder_hidden_states=prompt,
        encoder_attention_mask=None,
        rotary_emb=rope1,
        prefixes=prefixes1,
    )
    # With 3 history frames the shifted sink lands at start1 - 3 - 1 = 0, i.e.
    # its true position, so the block-causal joint forward over
    # [sink | refined0 | block1] at positions [0..6] (sink and history rows at
    # sigma 0) is the reference.
    joint = torch.cat([sink, refined0, block1], dim=2)
    reference = _reference_forward_per_token(
        runner,
        joint,
        [0] + pos0 + pos1,
        [SINK * TOKENS_PER_FRAME, BLOCK * TOKENS_PER_FRAME, BLOCK * TOKENS_PER_FRAME],
        [0.0, 0.0, sigma],
        prompt,
    )
    torch.testing.assert_close(out, reference[:, (SINK + BLOCK) * TOKENS_PER_FRAME :], atol=2e-4, rtol=2e-4)


def test_capture_matches_projection_of_full_forward() -> None:
    """The K/V capture pass returns the post-RoPE K/V a plain forward computes in every layer."""
    model = _tiny_transformer("split")
    runner = _runner(model)
    prompt = _prompt()
    block = _latents(BLOCK, seed=3)
    positions = list(range(SINK, SINK + BLOCK))
    rope = refiner_rotary_emb(
        model, frame_positions=positions, height=LATENT_H, width=LATENT_W, batch_size=1, device=block.device, fps=FPS
    )
    seen: list[torch.Tensor] = []

    def record(attn):
        original = attn.attn.forward

        def wrapped(q, k, v, meta=None):
            seen.append(k.flatten(2, 3).clone())
            return original(q, k, v, meta)

        attn.attn.forward = wrapped

    for layer in model.transformer_blocks:
        record(layer.attn1)
    runner.forward_video(
        runner._pack(block),
        sigma=0.0,
        encoder_hidden_states=prompt,
        encoder_attention_mask=None,
        rotary_emb=rope,
        prefixes=None,
    )
    for layer in model.transformer_blocks:
        del layer.attn1.attn.forward  # restore the class method
    _, captured = runner.forward_video(
        runner._pack(block),
        sigma=0.0,
        encoder_hidden_states=prompt,
        encoder_attention_mask=None,
        rotary_emb=rope,
        prefixes=None,
        capture="post_rope",
    )
    assert len(seen) == len(captured) == 2
    for full_k, (cap_k, _) in zip(seen, captured):
        torch.testing.assert_close(cap_k, full_k)


# ---------------------------------------------------------------------------
# refine_block end to end
# ---------------------------------------------------------------------------


def test_refine_block_euler_and_window(monkeypatch) -> None:
    model = _tiny_transformer("split")
    runner = _runner(model, kv=SINK + 2 * BLOCK)  # window: sink + 6 frames
    prompt = _prompt()
    sink = _latents(SINK, seed=1)
    cache = runner.new_cache(latent_height=LATENT_H, latent_width=LATENT_W)
    sigmas = runner.schedule.sigmas

    # Deterministic velocity so the Euler trajectory can be reproduced by hand.
    calls: list[float] = []
    real_forward = runner.forward_video

    def fake_forward(tokens, *, sigma, capture=None, **kwargs):
        if capture is not None:
            return real_forward(tokens, sigma=sigma, capture=capture, **kwargs)
        calls.append(sigma)
        return tokens * 0.5, None

    monkeypatch.setattr(runner, "forward_video", fake_forward)

    with pytest.raises(ValueError, match="sink_latents"):
        runner.refine_block(
            cache,
            _latents(BLOCK, 2),
            block_start=SINK,
            encoder_hidden_states=prompt,
            encoder_attention_mask=None,
            fps=FPS,
            generator=None,
        )

    generator = torch.Generator().manual_seed(11)
    clean = _latents(BLOCK, seed=2)
    refined = runner.refine_block(
        cache,
        clean,
        block_start=SINK,
        encoder_hidden_states=prompt,
        encoder_attention_mask=None,
        fps=FPS,
        generator=generator,
        sink_latents=sink,
    )
    assert calls == list(sigmas[:-1])
    eps = torch.randn(clean.shape, generator=torch.Generator().manual_seed(11))
    x = (1 - sigmas[0]) * clean + sigmas[0] * eps
    for s_cur, s_next in zip(sigmas[:-1], sigmas[1:]):
        x0 = x - 0.5 * x * s_cur
        x = (s_next / s_cur) * x + (1 - s_next / s_cur) * x0
    torch.testing.assert_close(refined, x, atol=1e-5, rtol=1e-5)
    assert cache.blocks_refined == 1 and cache.history_frames == BLOCK
    assert cache.cached_frames() == SINK + BLOCK

    for block_idx in range(1, 4):
        runner.refine_block(
            cache,
            _latents(BLOCK, seed=20 + block_idx),
            block_start=SINK + block_idx * BLOCK,
            encoder_hidden_states=prompt,
            encoder_attention_mask=None,
            fps=FPS,
            generator=generator,
        )
    assert cache.blocks_refined == 4
    assert cache.history_frames == 2 * BLOCK
    assert cache.history_kv_post[0][0].shape[1] == 2 * BLOCK * TOKENS_PER_FRAME
    assert cache.nbytes() > 0
    with pytest.raises(ValueError, match="overlaps the sink"):
        runner.refine_block(
            cache,
            clean,
            block_start=0,
            encoder_hidden_states=prompt,
            encoder_attention_mask=None,
            fps=FPS,
            generator=generator,
        )


def test_rotary_positions_match_contiguous_rope() -> None:
    """Absolute-position RoPE for ``[a, b)`` equals the slice of the contiguous ``[0, b)`` table."""
    model = _tiny_transformer("split")
    full_coords = model.rope.prepare_video_coords(1, 7, LATENT_H, LATENT_W, torch.device("cpu"), fps=FPS)
    full_cos, full_sin = model.rope(full_coords, device=torch.device("cpu"))
    cos, sin = refiner_rotary_emb(
        model,
        frame_positions=range(4, 7),
        height=LATENT_H,
        width=LATENT_W,
        batch_size=1,
        device=torch.device("cpu"),
        fps=FPS,
    )
    start = 4 * TOKENS_PER_FRAME
    torch.testing.assert_close(cos, full_cos[:, :, start:])
    torch.testing.assert_close(sin, full_sin[:, :, start:])
