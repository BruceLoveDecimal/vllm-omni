# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright 2026 SolarWM Contributors.
"""Native single-device SolarWM-5B Stage2 DiT with request-owned rolling KV.

Adapted from SolarWM a3a3fac16466102a2b97df867f7703df7172cb2a.
Only the released fused-PRoPE, no-sink, flow-matching inference architecture
is supported. Cached keys are normalized but unrotated: moving the window
requires applying new RoPE coordinates before the camera projection.
"""

import math
from dataclasses import dataclass, field

import torch
from torch import nn

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.platforms import current_omni_platform

from .camera import apply_projective_transform, camera_projection


@dataclass
class SolarWMCache:
    keys: dict[int, torch.Tensor] = field(default_factory=dict)
    values: dict[int, torch.Tensor] = field(default_factory=dict)
    text: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    viewmats: torch.Tensor | None = None
    intrinsics: torch.Tensor | None = None
    next_frame: int = 0


def sinusoidal_embedding(dim: int, positions: torch.Tensor) -> torch.Tensor:
    half = dim // 2
    positions = positions.flatten().to(torch.float64)
    frequencies = torch.pow(10000, -torch.arange(half, device=positions.device).to(positions) / half)
    phases = torch.outer(positions, frequencies)
    return torch.cat([phases.cos(), phases.sin()], dim=-1)


def window_rope(x: torch.Tensor, grid: tuple[int, int, int], start_frame: int = 0) -> torch.Tensor:
    """Original Wan complex128 RoPE, with time relative to the visible window."""
    frames, height, width = grid
    dim = x.shape[-1]
    spatial = 2 * (dim // 6)
    axes = []
    for length, axis_dim, start in (
        (frames, dim - 2 * spatial, start_frame),
        (height, spatial, 0),
        (width, spatial, 0),
    ):
        positions = torch.arange(start, start + length, device=x.device, dtype=torch.float64)
        frequencies = 1 / torch.pow(
            10000, torch.arange(0, axis_dim, 2, device=x.device, dtype=torch.float64) / axis_dim
        )
        phases = torch.outer(positions, frequencies)
        axes.append(torch.polar(torch.ones_like(phases), phases))
    frequencies = torch.cat(
        [
            axes[0][:, None, None].expand(frames, height, width, -1),
            axes[1][None, :, None].expand(frames, height, width, -1),
            axes[2][None, None, :].expand(frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(1, frames * height * width, 1, dim // 2)
    pairs = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], dim // 2, 2))
    return torch.view_as_real(pairs * frequencies).flatten(-2).to(x.dtype)


class SolarWMRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Match the reference operation order. Autocast controls pow precision;
        # cast normalization back to the activation dtype before the weight.
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class SolarWMLayerNorm(nn.LayerNorm):
    def forward(self, x):
        # The reference casts autocast-promoted layer norm back to activation dtype.
        return super().forward(x).type_as(x)


class SolarWMAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float, cross: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = SolarWMRMSNorm(dim, eps)
        self.norm_k = SolarWMRMSNorm(dim, eps)
        self.attention = Attention(
            num_heads=num_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.head_dim**-0.5,
            skip_sequence_parallel=True,
            disable_kv_quant=True,
            role="cross" if cross else "self",
            qkv_layout="BSND",
        )

    def query(self, x):
        return self.norm_q(self.q(x)).unflatten(-1, (self.num_heads, self.head_dim))

    def key_value(self, x):
        return (
            self.norm_k(self.k(x)).unflatten(-1, (self.num_heads, self.head_dim)),
            self.v(x).unflatten(-1, (self.num_heads, self.head_dim)),
        )

    def forward(self, q, k, v):
        return self.attention(q, k, v)


class SolarWMBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float):
        super().__init__()
        self.norm1 = SolarWMLayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = SolarWMAttention(dim, num_heads, eps)
        self.norm3 = SolarWMLayerNorm(dim, eps=eps)
        self.cross_attn = SolarWMAttention(dim, num_heads, eps, cross=True)
        self.norm2 = SolarWMLayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.empty(1, 6, dim))

    def forward(self, x, e, context, grid, projection, inverse, cache, layer, keep_tokens, commit):
        shift, scale, gate, fshift, fscale, fgate = (self.modulation.unsqueeze(0) + e).unbind(2)
        normed = self.norm1(x) * (1 + scale) + shift
        q = self.self_attn.query(normed)
        k, v = self.self_attn.key_value(normed)
        if layer in cache.keys and keep_tokens:
            k = torch.cat([cache.keys[layer][:, -keep_tokens:], k], dim=1)
            v = torch.cat([cache.values[layer][:, -keep_tokens:], v], dim=1)
        frames, height, width = grid
        visible_frames = k.shape[1] // (height * width)
        q = window_rope(q, grid, visible_frames - frames)
        roped_k = window_rope(k, (visible_frames, height, width))
        q_projection = projection[:, -frames:]
        q = apply_projective_transform(q, q_projection.transpose(-1, -2))
        roped_k = apply_projective_transform(roped_k, inverse)
        transformed_v = apply_projective_transform(v, inverse)
        attended = self.self_attn(q, roped_k, transformed_v)
        attended = apply_projective_transform(attended, q_projection)
        x = x + self.self_attn.o(attended.flatten(2)) * gate
        if commit:
            cache.keys[layer], cache.values[layer] = k.detach(), v.detach()
        if layer not in cache.text:
            cache.text[layer] = self.cross_attn.key_value(context)
        text_k, text_v = cache.text[layer]
        q = self.cross_attn.query(self.norm3(x))
        x = x + self.cross_attn.o(self.cross_attn(q, text_k, text_v).flatten(2))
        return x + self.ffn(self.norm2(x) * (1 + fscale) + fshift) * fgate


class SolarWMHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps):
        super().__init__()
        self.norm = SolarWMLayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.empty(1, 2, dim))

    def forward(self, x, e):
        shift, scale = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).unbind(2)
        return self.head(self.norm(x) * (1 + scale) + shift)


class SolarWMTransformer(nn.Module):
    _repeated_blocks = ["SolarWMBlock"]
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(
        self,
        *,
        dim=3072,
        ffn_dim=14336,
        num_heads=24,
        num_layers=30,
        in_dim=48,
        out_dim=48,
        freq_dim=256,
        text_dim=4096,
        text_len=512,
        patch_size=(1, 2, 2),
        eps=1e-6,
        max_history_frames=15,
    ):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 4 or freq_dim % 2:
            raise ValueError("Invalid SolarWM head/time dimensions")
        if tuple(patch_size) != (1, 2, 2):
            raise ValueError("SolarWM Stage2 requires patch_size=(1,2,2)")
        self.dim, self.freq_dim, self.text_len = dim, freq_dim, text_len
        self.out_dim, self.patch_size = out_dim, tuple(patch_size)
        self.max_history_frames = max_history_frames
        self.patch_embedding = nn.Conv3d(in_dim, dim, patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.blocks = nn.ModuleList([SolarWMBlock(dim, ffn_dim, num_heads, eps) for _ in range(num_layers)])
        self.head = SolarWMHead(dim, out_dim, patch_size, eps)

    def forward(self, latents, timestep, context, viewmats, intrinsics, *, cache, start_frame, commit=False):
        # Match the release inference context, including FP32 pow in QK norm.
        with current_omni_platform.create_autocast_context(
            device_type=latents.device.type,
            dtype=torch.bfloat16,
            enabled=latents.device.type == "cuda" and latents.dtype == torch.bfloat16,
        ):
            return self._forward(
                latents, timestep, context, viewmats, intrinsics, cache=cache, start_frame=start_frame, commit=commit
            )

    def _forward(self, latents, timestep, context, viewmats, intrinsics, *, cache, start_frame, commit=False):
        """Consume BCTHW latents, BF times and BF camera matrices; return BCTHW."""
        if start_frame != cache.next_frame:
            raise ValueError(f"Out-of-order chunk: expected {cache.next_frame}, got {start_frame}")
        b, _, frames, height, width = latents.shape
        if frames != 3 or height % 2 or width % 2:
            raise ValueError("Stage2 needs three latent frames and even latent spatial dimensions")
        if timestep.shape != (b, frames) or viewmats.shape != (b, frames, 4, 4):
            raise ValueError("Timesteps and camera matrices must align with the current latent chunk")
        x = self.patch_embedding(latents)
        grid = tuple(x.shape[2:])
        tokens_per_frame = grid[1] * grid[2]
        x = x.flatten(2).transpose(1, 2)
        times = timestep.repeat_interleave(tokens_per_frame, dim=1)
        e = self.time_embedding(sinusoidal_embedding(self.freq_dim, times).type_as(x)).reshape(b, -1, self.dim)
        modulation = self.time_projection(e).unflatten(-1, (6, self.dim))
        if context.shape[1] > self.text_len:
            raise ValueError("Text context exceeds checkpoint text_len")
        context = torch.nn.functional.pad(context, (0, 0, 0, self.text_len - context.shape[1]))
        context = self.text_embedding(context)
        views, ks = viewmats, intrinsics
        if cache.viewmats is not None and self.max_history_frames:
            views = torch.cat([cache.viewmats[:, -self.max_history_frames :], views], dim=1)
            ks = torch.cat([cache.intrinsics[:, -self.max_history_frames :], ks], dim=1)
        projection, inverse = camera_projection(views, ks)
        for index, block in enumerate(self.blocks):
            x = block(
                x,
                modulation,
                context,
                grid,
                projection,
                inverse,
                cache,
                index,
                self.max_history_frames * tokens_per_frame,
                commit,
            )
        output = self.head(x, e)
        output = output.reshape(b, *grid, *self.patch_size, self.out_dim)
        output = torch.einsum("bfhwpqrc->bcfphqwr", output).reshape(b, self.out_dim, frames, height, width)
        if commit:
            cache.viewmats, cache.intrinsics = views.detach(), ks.detach()
            cache.next_frame += frames
        return output
