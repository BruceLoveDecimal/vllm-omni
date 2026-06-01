# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SANA-WM camera-branch prep.

This module ports the public NVlabs ``cam_prep_func`` contract into a local
PyTorch implementation.  The fused upstream kernel performs, per token/head:

* full-channel RMSNorm on Q/K;
* ReLU on Q/K and K scaling;
* UCPE 4x4 ray-frame projection on the first half of the head channels;
* interleaved-pair RoPE on the second half;
* V projection/RoPE without RMSNorm or ReLU;
* K norm inflation reporting for Dynamic Beta Discounting.

The function returns tensors in the same layout as the NVlabs kernel:
``(B, H, D, N)`` for Q/K/V and ``(B, H, N)`` for inflation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from vllm_omni.diffusion.models.sana_wm.ucpe import (
    _invert_se3,
    _prepare_ray_apply_fns,
    _process_camera_conditions,
    _slice_rope_for_cam,
)


@dataclass(frozen=True)
class SanaWmCamPrepContext:
    """Precomputed per-request tensors needed by ``cam_prep_func``."""

    proj_q: torch.Tensor
    proj_kv: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    apply_output: Callable[[torch.Tensor], torch.Tensor]


def _prepare_ucpe_rope_tables(
    rotary_emb_cam: torch.Tensor,
    token_count: int,
    half_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert complex RoPE freqs to NVlabs' interleaved cos/sin tables.

    The output convention is ``out[d] = x[d] * cos[d] + x[d ^ 1] * sin[d]``.
    For every complex pair, ``sin[2i] = -imag(freq[i])`` and
    ``sin[2i + 1] = imag(freq[i])``.
    """
    freqs = rotary_emb_cam
    while freqs.ndim > 2 and freqs.shape[0] == 1:
        freqs = freqs.squeeze(0)
    if freqs.ndim != 2:
        raise ValueError(f"Sana-WM cam RoPE freqs must be 2D after squeeze, got {tuple(freqs.shape)}.")
    if freqs.shape[0] != token_count:
        raise ValueError(f"Sana-WM cam RoPE token count {freqs.shape[0]} != {token_count}.")
    if freqs.shape[1] * 2 != half_dim:
        raise ValueError(f"Sana-WM cam RoPE dim {freqs.shape[1] * 2} != half_dim {half_dim}.")
    cos_half = freqs.real.float()
    sin_half = freqs.imag.float()
    rope_cos = cos_half.repeat_interleave(2, dim=-1).contiguous()
    rope_sin = torch.stack((-sin_half, sin_half), dim=-1).reshape(token_count, half_dim).contiguous()
    return rope_cos, rope_sin


def prepare_cam_prep_context(
    *,
    camera_conditions: torch.Tensor,
    spatial_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    head_dim: int,
    rotary_emb: torch.Tensor | None,
) -> SanaWmCamPrepContext:
    """Precompute ray projection matrices, RoPE tables, and inverse output fn."""
    if head_dim % 2 != 0 or (head_dim // 2) % 4 != 0:
        raise ValueError(f"Sana-WM cam prep head_dim={head_dim} must satisfy D % 2 == 0 and (D/2) % 4 == 0.")

    batch = camera_conditions.shape[0]
    frames, height, width = spatial_shape
    token_count = frames * height * width
    half_dim = head_dim // 2

    raymats = _process_camera_conditions(camera_conditions, spatial_shape, patch_size=patch_size)
    proj = raymats.reshape(batch, token_count, 4, 4).contiguous()
    proj_q = proj.transpose(-1, -2).contiguous()
    proj_kv = _invert_se3(proj).contiguous()

    rotary_emb_cam = _slice_rope_for_cam(rotary_emb, head_dim, half_dim)
    if rotary_emb_cam is None:
        rope_cos = torch.ones(token_count, half_dim, device=camera_conditions.device, dtype=torch.float32)
        rope_sin = torch.zeros(token_count, half_dim, device=camera_conditions.device, dtype=torch.float32)
    else:
        rope_cos, rope_sin = _prepare_ucpe_rope_tables(rotary_emb_cam, token_count, half_dim)

    _, _, apply_output = _prepare_ray_apply_fns(head_dim, proj, proj_q, proj_kv, rotary_emb=rotary_emb_cam)
    return SanaWmCamPrepContext(
        proj_q=proj_q,
        proj_kv=proj_kv,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        apply_output=apply_output,
    )


def _validate_cam_prep_inputs(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if q_raw.shape != k_raw.shape or q_raw.shape != v_raw.shape:
        raise ValueError(f"Sana-WM cam prep q/k/v shapes must match, got {q_raw.shape}, {k_raw.shape}, {v_raw.shape}.")
    if q_raw.ndim != 4:
        raise ValueError(f"Sana-WM cam prep q/k/v must be (B,N,H,D), got {tuple(q_raw.shape)}.")
    batch, token_count, num_heads, head_dim = q_raw.shape
    if head_dim % 2 != 0 or (head_dim // 2) % 4 != 0:
        raise ValueError(f"Sana-WM cam prep head_dim={head_dim} must satisfy D % 2 == 0 and (D/2) % 4 == 0.")
    half_dim = head_dim // 2
    if proj_q.shape != (batch, token_count, 4, 4):
        raise ValueError(f"Sana-WM cam prep proj_q shape {tuple(proj_q.shape)} != {(batch, token_count, 4, 4)}.")
    if proj_kv.shape != (batch, token_count, 4, 4):
        raise ValueError(f"Sana-WM cam prep proj_kv shape {tuple(proj_kv.shape)} != {(batch, token_count, 4, 4)}.")
    if rope_cos.shape != (token_count, half_dim) or rope_sin.shape != (token_count, half_dim):
        raise ValueError(
            "Sana-WM cam prep rope table shapes must be "
            f"{(token_count, half_dim)}, got {tuple(rope_cos.shape)} and {tuple(rope_sin.shape)}."
        )
    channel_count = num_heads * head_dim
    if q_norm_weight.numel() != channel_count or k_norm_weight.numel() != channel_count:
        raise ValueError(
            "Sana-WM cam prep norm weights must have H*D elements, got "
            f"{q_norm_weight.numel()} and {k_norm_weight.numel()} for H*D={channel_count}."
        )
    return batch, token_count, num_heads, head_dim, half_dim


def _rms_norm_cam(raw: torch.Tensor, weight: torch.Tensor, norm_eps: float) -> torch.Tensor:
    batch, token_count, num_heads, head_dim = raw.shape
    inv_rms = torch.rsqrt(raw.float().square().sum(dim=(-1, -2)) / float(num_heads * head_dim) + norm_eps)
    weight_view = weight.float().reshape(num_heads, head_dim)
    return raw.float() * inv_rms.reshape(batch, token_count, 1, 1) * weight_view.reshape(1, 1, num_heads, head_dim)


def _apply_ray_projection(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    batch, token_count, num_heads, half_dim = feats.shape
    grouped = feats.reshape(batch, token_count, num_heads, half_dim // 4, 4)
    out = torch.einsum("bnij,bnhgj->bnhgi", matrix.float(), grouped)
    return out.reshape(batch, token_count, num_heads, half_dim)


def _apply_interleaved_rope(feats: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
    half_dim = feats.shape[-1]
    pair_index = torch.arange(half_dim, device=feats.device) ^ 1
    paired = feats.index_select(-1, pair_index)
    return feats * rope_cos.reshape(1, rope_cos.shape[0], 1, half_dim) + paired * rope_sin.reshape(
        1, rope_sin.shape[0], 1, half_dim
    )


def cam_prep_func(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    *,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_scale: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Native equivalent of NVlabs ``cam_prep_func``.

    Args:
        q_raw, k_raw, v_raw: ``(B, N, H, D)`` raw camera Q/K/V. K must already
            include the temporal short convolution.
        q_norm_weight, k_norm_weight: flattened ``(H*D,)`` RMSNorm weights.
        proj_q, proj_kv: ``(B, N, 4, 4)`` ray projection matrices.
        rope_cos, rope_sin: ``(N, D//2)`` interleaved-pair RoPE tables.
        k_scale: ``D^-0.5 * spatial_tokens^-0.5``.
        norm_eps: RMSNorm epsilon.

    Returns:
        ``q_trans``, ``k_trans``, ``v_trans`` in ``(B, H, D, N)`` layout, and
        ``inflation_sq`` in ``(B, H, N)``.
    """
    batch, token_count, num_heads, head_dim, half_dim = _validate_cam_prep_inputs(
        q_raw,
        k_raw,
        v_raw,
        q_norm_weight,
        k_norm_weight,
        proj_q,
        proj_kv,
        rope_cos,
        rope_sin,
    )

    q_norm = torch.relu(_rms_norm_cam(q_raw, q_norm_weight, norm_eps))
    k_norm = torch.relu(_rms_norm_cam(k_raw, k_norm_weight, norm_eps)) * float(k_scale)
    value = v_raw.float()

    q_half = _apply_ray_projection(q_norm[..., :half_dim], proj_q)
    k_half = _apply_ray_projection(k_norm[..., :half_dim], proj_kv)
    v_half = _apply_ray_projection(value[..., :half_dim], proj_kv)

    q_rope = _apply_interleaved_rope(q_norm[..., half_dim:], rope_cos.float(), rope_sin.float())
    k_rope = _apply_interleaved_rope(k_norm[..., half_dim:], rope_cos.float(), rope_sin.float())
    v_rope = _apply_interleaved_rope(value[..., half_dim:], rope_cos.float(), rope_sin.float())

    k_pre_sq = k_norm.square().sum(dim=-1).permute(0, 2, 1).contiguous()
    k_post = torch.cat((k_half, k_rope), dim=-1)
    k_post_sq = k_post.square().sum(dim=-1).permute(0, 2, 1).contiguous()
    inflation_sq = k_post_sq.clamp_min(1e-12) / k_pre_sq.clamp_min(1e-12)

    def to_bhdn(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute(0, 2, 3, 1).contiguous().to(q_raw.dtype)

    q_out = to_bhdn(torch.cat((q_half, q_rope), dim=-1))
    k_out = to_bhdn(k_post)
    v_out = to_bhdn(torch.cat((v_half, v_rope), dim=-1))
    expected = (batch, num_heads, head_dim, token_count)
    if q_out.shape != expected or k_out.shape != expected or v_out.shape != expected:
        raise RuntimeError("Sana-WM cam prep produced an unexpected output shape.")
    return q_out, k_out, v_out, inflation_sq


__all__ = [
    "SanaWmCamPrepContext",
    "cam_prep_func",
    "prepare_cam_prep_context",
]
