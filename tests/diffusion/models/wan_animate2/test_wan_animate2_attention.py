# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Numerical parity tests for Wan2.2-Animate-2 reference attention (M0).

The ground truth is the upstream ``Wan-Video/Wan-Animate-2`` implementation:
``WanAnimate2Transformer.create_mask`` (flex-attention ``BlockMask`` predicate),
``_score_mod_impl`` (distilled logit bias) and ``rope_apply``.  Those three are
transcribed verbatim below and evaluated densely, so the comparison is against
upstream's semantics rather than against a re-derivation of them.
"""

import math

import pytest
import torch

from vllm_omni.diffusion.models.wan_animate2.reference_attention import (
    ReferenceGridInfo,
    attention_with_lse,
    merge_attention_branches,
    reference_context_attention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
# Upstream reference implementations (verbatim transcriptions)
# ---------------------------------------------------------------------------


def _upstream_attention_mask_logic(q_idx, kv_idx, *, hw, q_limit, k_limit, q_total):
    """``WanAnimate2Transformer.create_mask.attention_mask_logic``, verbatim."""
    q_valid = q_idx < q_limit
    is_base_attention = kv_idx < q_limit

    q_frame = q_idx // hw
    is_first_part = kv_idx < q_total

    kv_frame_1 = kv_idx // hw
    kv_is_valid_1 = kv_idx < q_limit

    rel_kv_idx = kv_idx - q_total
    kv_frame_2 = (rel_kv_idx // hw) + 1
    kv_is_valid_2 = rel_kv_idx < k_limit

    kv_frame = torch.where(is_first_part, kv_frame_1, kv_frame_2)
    kv_is_valid = torch.where(is_first_part, kv_is_valid_1, kv_is_valid_2)

    is_cond_attention = (q_frame == kv_frame) & kv_is_valid

    return q_valid & (is_base_attention | is_cond_attention)


def _upstream_score_bias(kv_idx, *, hw, log_scale):
    """``_score_mod_impl``: constant bias on the generated frame-1 key block."""
    condition = (kv_idx >= hw) & (kv_idx < 2 * hw)
    return torch.where(
        condition,
        torch.full_like(kv_idx, log_scale, dtype=torch.float32),
        torch.zeros_like(kv_idx, dtype=torch.float32),
    )


def _upstream_rope_params(max_seq_len, dim, theta=10000, offset=0):
    """``wan_animate_2_model.rope_params``, verbatim."""
    freqs = torch.outer(
        torch.arange(max_seq_len) + offset,
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def _upstream_rope_apply(x, grid_sizes, freqs, time_stride=1):
    """``wan_animate_2_model.rope_apply``, verbatim."""
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][: f * time_stride : time_stride].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


def _upstream_reference_attention(
    query,
    key,
    value,
    reference_key,
    reference_value,
    *,
    num_frames,
    frame_tokens,
    num_reference_frames,
    padded_gen_frames,
    padded_reference_frames,
    log_scale,
    padded_frame_tokens=None,
):
    """Dense evaluation of upstream's packed flex-attention call.

    Mirrors ``Incontext_AttentionBlock.forward_gen``: scatter the real tokens
    into the padded ``[padded_frames, frame_tokens]`` grids (leaving zeros
    elsewhere), then a single masked softmax over ``[gen | reference]``.  The
    128-element alignment upstream applies on top is omitted because the mask
    excludes those slots from both the query and the key axis.
    """
    batch, _, num_heads, head_dim = query.shape
    padded_frame_tokens = padded_frame_tokens or frame_tokens
    q_total = padded_gen_frames * padded_frame_tokens
    k_extra = padded_reference_frames * padded_frame_tokens

    def _scatter(src, frames, total_frames):
        packed = torch.zeros(batch, total_frames, padded_frame_tokens, num_heads, head_dim, dtype=src.dtype)
        packed[:, :frames, :frame_tokens] = src[:, : frames * frame_tokens].view(
            batch, frames, frame_tokens, num_heads, head_dim
        )
        return packed.reshape(batch, total_frames * padded_frame_tokens, num_heads, head_dim)

    q_packed = _scatter(query, num_frames, padded_gen_frames)
    k_packed = torch.cat(
        [
            _scatter(key, num_frames, padded_gen_frames),
            _scatter(reference_key, num_reference_frames, padded_reference_frames),
        ],
        dim=1,
    )
    v_packed = torch.cat(
        [
            _scatter(value, num_frames, padded_gen_frames),
            _scatter(reference_value, num_reference_frames, padded_reference_frames),
        ],
        dim=1,
    )

    q_idx = torch.arange(q_total).view(-1, 1)
    kv_idx = torch.arange(q_total + k_extra).view(1, -1)
    mask = _upstream_attention_mask_logic(
        q_idx, kv_idx, hw=padded_frame_tokens, q_limit=q_total, k_limit=k_extra, q_total=q_total
    )
    bias = _upstream_score_bias(kv_idx, hw=padded_frame_tokens, log_scale=log_scale)

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.einsum("bqhd,bkhd->bhqk", q_packed.float(), k_packed.float()) * scale
    scores = scores + bias.view(1, 1, 1, -1)
    scores = scores.masked_fill(~mask.view(1, 1, q_total, -1), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", weights, v_packed.float())
    out = out.view(batch, padded_gen_frames, padded_frame_tokens, num_heads, head_dim)
    out = out[:, :num_frames, :frame_tokens].reshape(batch, num_frames * frame_tokens, num_heads, head_dim)
    return out.to(query.dtype), weights


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _grid_from_kwargs(kwargs):
    """Build the port's grid description from the upstream-style keyword set."""
    return ReferenceGridInfo(
        num_frames=kwargs["num_frames"],
        frame_tokens=kwargs["frame_tokens"],
        num_reference_frames=kwargs["num_reference_frames"],
        padded_frame_tokens=kwargs["padded_frame_tokens"],
        padded_gen_frames=kwargs["padded_gen_frames"],
        padded_reference_frames=kwargs["padded_reference_frames"],
    )


def _make_inputs(num_frames, num_reference_frames, frame_tokens, num_heads=4, head_dim=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, num_frames * frame_tokens, num_heads, head_dim)
    ref_shape = (1, num_reference_frames * frame_tokens, num_heads, head_dim)
    return (
        torch.randn(shape, generator=generator, dtype=torch.float32),
        torch.randn(shape, generator=generator, dtype=torch.float32),
        torch.randn(shape, generator=generator, dtype=torch.float32),
        torch.randn(ref_shape, generator=generator, dtype=torch.float32),
        torch.randn(ref_shape, generator=generator, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# A0.1 / A0.2 — LSE-merged attention vs upstream mask semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("log_scale", [0.0, -1.3])
def test_reference_attention_matches_upstream(log_scale):
    """A0.1/A0.2: dual-branch LSE merge reproduces upstream's masked softmax."""

    num_frames, num_reference_frames, frame_tokens = 6, 5, 12
    query, key, value, reference_key, reference_value = _make_inputs(num_frames, num_reference_frames, frame_tokens)

    expected, _ = _upstream_reference_attention(
        query,
        key,
        value,
        reference_key,
        reference_value,
        num_frames=num_frames,
        frame_tokens=frame_tokens,
        num_reference_frames=num_reference_frames,
        padded_gen_frames=num_frames,
        padded_reference_frames=num_reference_frames,
        padded_frame_tokens=frame_tokens,
        log_scale=log_scale,
    )

    grid = ReferenceGridInfo(
        num_frames=num_frames,
        frame_tokens=frame_tokens,
        num_reference_frames=num_reference_frames,
        padded_frame_tokens=frame_tokens,
        padded_gen_frames=num_frames,
        padded_reference_frames=num_reference_frames,
    )
    actual = reference_context_attention(
        query,
        key,
        value,
        reference_key,
        reference_value,
        grid,
        softmax_scale=1.0 / math.sqrt(query.shape[-1]),
        log_scale=log_scale,
    )

    max_abs = (actual - expected).abs().max().item()
    rel = max_abs / expected.abs().max().item()
    assert max_abs <= 1e-5, f"max abs diff {max_abs}"
    assert rel <= 1e-4, f"relative diff {rel}"


@pytest.mark.parametrize("log_scale", [0.0, -1.3])
def test_reference_attention_matches_upstream_short_trailing_segment(log_scale):
    """Trailing segments are shorter than the mask window; the zero-valued key
    slots inside that window still widen the softmax denominator upstream, and
    must do so here too."""

    num_frames, num_reference_frames, frame_tokens = 4, 3, 12
    padded_reference_frames = 5
    padded_gen_frames = padded_reference_frames + 1
    query, key, value, reference_key, reference_value = _make_inputs(
        num_frames, num_reference_frames, frame_tokens, seed=7
    )

    kwargs = dict(
        num_frames=num_frames,
        frame_tokens=frame_tokens,
        num_reference_frames=num_reference_frames,
        padded_frame_tokens=frame_tokens,
        padded_gen_frames=padded_gen_frames,
        padded_reference_frames=padded_reference_frames,
        log_scale=log_scale,
    )
    expected, _ = _upstream_reference_attention(query, key, value, reference_key, reference_value, **kwargs)
    actual = reference_context_attention(
        query,
        key,
        value,
        reference_key,
        reference_value,
        _grid_from_kwargs(kwargs),
        softmax_scale=1.0 / math.sqrt(query.shape[-1]),
        log_scale=log_scale,
    )

    assert (actual - expected).abs().max().item() <= 1e-5


def test_zero_padded_keys_change_the_result():
    """Guards the test above: the padded-window behaviour is not a no-op, so a
    regression that ignores the zero slots cannot pass by accident."""

    num_frames, num_reference_frames, frame_tokens = 4, 3, 12
    query, key, value, reference_key, reference_value = _make_inputs(
        num_frames, num_reference_frames, frame_tokens, seed=7
    )
    scale = 1.0 / math.sqrt(query.shape[-1])
    tight_grid = ReferenceGridInfo(
        num_frames=num_frames,
        frame_tokens=frame_tokens,
        num_reference_frames=num_reference_frames,
        padded_frame_tokens=frame_tokens,
        padded_gen_frames=num_frames,
        padded_reference_frames=num_reference_frames,
    )
    padded_grid = ReferenceGridInfo(
        num_frames=num_frames,
        frame_tokens=frame_tokens,
        num_reference_frames=num_reference_frames,
        padded_frame_tokens=frame_tokens,
        padded_gen_frames=6,
        padded_reference_frames=5,
    )
    tight = reference_context_attention(query, key, value, reference_key, reference_value, tight_grid, scale)
    padded = reference_context_attention(query, key, value, reference_key, reference_value, padded_grid, scale)
    assert (tight - padded).abs().max().item() > 1e-4


# ---------------------------------------------------------------------------
# A0.3 — frame-alignment semantics
# ---------------------------------------------------------------------------


def test_reference_attention_is_frame_aligned():
    """A0.3: query frame f sees only reference frame f-1; latent frame 0 (the
    reference-image slot) sees no reference token at all."""
    num_frames, num_reference_frames, frame_tokens = 5, 4, 6
    query, key, value, reference_key, reference_value = _make_inputs(
        num_frames, num_reference_frames, frame_tokens, seed=3
    )
    _, weights = _upstream_reference_attention(
        query,
        key,
        value,
        reference_key,
        reference_value,
        num_frames=num_frames,
        frame_tokens=frame_tokens,
        num_reference_frames=num_reference_frames,
        padded_gen_frames=num_frames,
        padded_reference_frames=num_reference_frames,
        padded_frame_tokens=frame_tokens,
        log_scale=0.0,
    )
    gen_span = num_frames * frame_tokens
    reference_weights = weights[..., gen_span:]  # [B, H, Sq, ref]

    for q_frame in range(num_frames):
        rows = slice(q_frame * frame_tokens, (q_frame + 1) * frame_tokens)
        for ref_frame in range(num_reference_frames):
            cols = slice(ref_frame * frame_tokens, (ref_frame + 1) * frame_tokens)
            block = reference_weights[:, :, rows, cols]
            if ref_frame == q_frame - 1:
                assert block.sum() > 0, f"q frame {q_frame} must attend reference frame {ref_frame}"
            else:
                assert torch.equal(block, torch.zeros_like(block)), (
                    f"q frame {q_frame} must not attend reference frame {ref_frame}"
                )
    assert torch.equal(reference_weights[:, :, :frame_tokens], torch.zeros_like(reference_weights[:, :, :frame_tokens]))


def test_lse_merge_equals_single_softmax():
    """The merge identity itself: splitting the key axis and recombining by
    log-sum-exp reproduces one dense softmax."""

    generator = torch.Generator().manual_seed(11)
    query = torch.randn(2, 9, 3, 8, generator=generator)
    key = torch.randn(2, 20, 3, 8, generator=generator)
    value = torch.randn(2, 20, 3, 8, generator=generator)
    scale = 1.0 / math.sqrt(8)

    whole, _ = attention_with_lse(query, key, value, scale)
    merged = merge_attention_branches(
        [
            attention_with_lse(query, key[:, :7], value[:, :7], scale),
            attention_with_lse(query, key[:, 7:], value[:, 7:], scale),
        ]
    )
    assert (whole - merged).abs().max().item() <= 1e-5


# ---------------------------------------------------------------------------
# Letterboxing: the mask window is wider than the generated grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("log_scale", [0.0, -1.3])
def test_reference_attention_matches_upstream_with_spatial_padding(log_scale):
    """``origin_area`` sizes the mask from the requested resolution, but
    letterboxing rounds the real frame down to a divisor-aligned box, so each
    frame's key block carries a zero-valued spatial remainder."""

    num_frames, num_reference_frames, frame_tokens = 5, 4, 9
    padded_frame_tokens = 12
    query, key, value, reference_key, reference_value = _make_inputs(
        num_frames, num_reference_frames, frame_tokens, seed=13
    )

    kwargs = dict(
        num_frames=num_frames,
        frame_tokens=frame_tokens,
        num_reference_frames=num_reference_frames,
        padded_frame_tokens=padded_frame_tokens,
        padded_gen_frames=num_frames,
        padded_reference_frames=num_reference_frames,
        log_scale=log_scale,
    )
    expected, _ = _upstream_reference_attention(query, key, value, reference_key, reference_value, **kwargs)
    actual = reference_context_attention(
        query,
        key,
        value,
        reference_key,
        reference_value,
        _grid_from_kwargs(kwargs),
        softmax_scale=1.0 / math.sqrt(query.shape[-1]),
        log_scale=log_scale,
    )

    assert (actual - expected).abs().max().item() <= 1e-5


@pytest.mark.parametrize("log_scale", [0.0, -1.3])
def test_reference_attention_matches_upstream_with_both_paddings(log_scale):
    """Trailing segment *and* letterboxing at once — the shape the last segment
    of a real request actually takes."""

    num_frames, num_reference_frames, frame_tokens = 4, 3, 9
    query, key, value, reference_key, reference_value = _make_inputs(
        num_frames, num_reference_frames, frame_tokens, seed=17
    )

    kwargs = dict(
        num_frames=num_frames,
        frame_tokens=frame_tokens,
        num_reference_frames=num_reference_frames,
        padded_frame_tokens=12,
        padded_gen_frames=6,
        padded_reference_frames=5,
        log_scale=log_scale,
    )
    expected, _ = _upstream_reference_attention(query, key, value, reference_key, reference_value, **kwargs)
    actual = reference_context_attention(
        query,
        key,
        value,
        reference_key,
        reference_value,
        _grid_from_kwargs(kwargs),
        softmax_scale=1.0 / math.sqrt(query.shape[-1]),
        log_scale=log_scale,
    )

    assert (actual - expected).abs().max().item() <= 1e-5
