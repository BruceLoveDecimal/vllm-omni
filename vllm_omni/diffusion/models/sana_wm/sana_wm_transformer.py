# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SANA-WM Stage-1 transformer fallback.

This module is intentionally split from the official NVlabs backend.  It gives
vLLM-Omni a materializable, shape-compatible native path for scaffolding, weight
audits, and small smoke tests while the fused Bidirectional Gated DeltaNet
Triton operator and exact UCPE camera branch are being ported.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field, replace
from functools import reduce
from operator import mul
from typing import Any, ClassVar, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.gated_deltanet_triton import (
    SANA_WM_REQUIRE_TRITON_GDN_ENV,
    reference_bidirectional_gated_delta_net,
    triton_bidirectional_gated_delta_net_from_qkv,
)
from vllm_omni.diffusion.models.sana_wm.weight_mapping import normalize_sana_wm_stage1_weight_name

SANA_WM_TRANSFORMER_FORWARD_ERROR = (
    "Sana-WM full Stage-1 transformer forward is not implemented yet. "
    "The native fallback runs a pure PyTorch GDN smoke path; production quality "
    "still requires porting the fused Triton GDN kernel, exact camera injection, "
    "and the official denoising stack."
)

SANA_WM_STAGE1_LATENT_CHANNELS = 128
SANA_WM_STAGE1_PROMPT_CHANNELS = 2304
SANA_WM_STAGE1_TIMESTEP_CHANNELS = 256
SANA_WM_STAGE1_GDN_STATE_SIZE = 20


def _prod(values: tuple[int, ...]) -> int:
    return reduce(mul, values, 1)


def _to_3tuple(value: int | tuple[int, int] | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) == 2:
        return (1, int(value[0]), int(value[1]))
    return (int(value[0]), int(value[1]), int(value[2]))


@dataclass(frozen=True)
class SanaWmStage1LoadReport:
    total_weights: int = 0
    loaded_weights: int = 0
    unmapped_weights: tuple[str, ...] = ()
    duplicate_weights: tuple[str, ...] = ()
    loaded_names: tuple[str, ...] = field(default_factory=tuple)
    materialized_weights: int = 0
    unapplied_weights: tuple[str, ...] = ()


class SanaWmRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps).to(hidden_states.dtype)
        return hidden_states * self.weight


class SanaWmTextProjection(nn.Module):
    def __init__(self, prompt_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(prompt_channels, hidden_size)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, prompt_embeds: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(prompt_embeds)))


class SanaWmTextEmbedder(nn.Module):
    def __init__(self, prompt_channels: int, hidden_size: int, max_length: int) -> None:
        super().__init__()
        self.y_embedding = nn.Parameter(torch.zeros(max_length, prompt_channels))
        self.y_proj = SanaWmTextProjection(prompt_channels, hidden_size)

    def forward(
        self,
        prompt_embeds: torch.Tensor | None,
        *,
        batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if prompt_embeds is None:
            prompt_embeds = self.y_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        return self.y_proj(prompt_embeds.to(dtype=dtype))


class SanaWmTimestepEmbedder(nn.Module):
    def __init__(self, in_features: int, hidden_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def sinusoidal_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=timestep.device, dtype=torch.float32) / max(half, 1)
        )
        args = timestep.float().reshape(-1, 1) * freqs.reshape(1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[-1]))
        return emb

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.sinusoidal_embedding(timestep, self.mlp[0].in_features)
        return self.mlp(emb.to(dtype=self.mlp[0].weight.dtype))


class SanaWmPatchEmbedMS3D(nn.Module):
    """Official-style 3D patch embedder used by SANA-WM Stage-1."""

    def __init__(
        self,
        patch_size: tuple[int, int, int],
        in_channels: int,
        hidden_size: int,
        *,
        kernel_size: tuple[int, int, int] | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = _to_3tuple(patch_size)
        self.kernel_size = _to_3tuple(kernel_size or patch_size)
        self.flatten = True
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=self.kernel_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = nn.Identity()

    def project_with_shape(self, latents: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        hidden_states = self.norm(self.proj(latents))
        _, _, frames, height, width = hidden_states.shape
        return hidden_states.flatten(2).transpose(1, 2), (frames, height, width)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.project_with_shape(latents)[0]


class SanaWmWanRotaryPosEmbed(nn.Module):
    """Wan-style 3D RoPE table used by the official SANA-WM GDN blocks."""

    def __init__(
        self,
        attention_head_dim: int,
        *,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        self._split_sizes = (
            t_dim // 2,
            h_dim // 2,
            w_dim // 2,
        )
        self.freqs = self._build_freqs(max_seq_len)

    def _build_1d_freq(self, dim: int, positions: torch.Tensor) -> torch.Tensor:
        if dim <= 0:
            return torch.empty(positions.shape[0], 0, dtype=torch.complex128)
        freqs = 1.0 / (
            self.theta
            ** (
                torch.arange(0, dim, 2, dtype=torch.float64, device=positions.device)[: dim // 2]
                / dim
            )
        )
        phase = torch.outer(positions.to(torch.float64), freqs)
        return torch.polar(torch.ones_like(phase), phase)

    def _build_freqs(self, max_seq_len: int) -> torch.Tensor:
        positions = torch.arange(max_seq_len, dtype=torch.float64)
        dims = (
            self._split_sizes[0] * 2,
            self._split_sizes[1] * 2,
            self._split_sizes[2] * 2,
        )
        return torch.cat([self._build_1d_freq(dim, positions) for dim in dims], dim=1)

    def forward(self, spatial_shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
        frames, height, width = spatial_shape
        if max(spatial_shape) > self.freqs.shape[0]:
            self.freqs = self._build_freqs(max(spatial_shape)).to(self.freqs.device)
        freqs = self.freqs.to(device=device)
        freqs_f, freqs_h, freqs_w = freqs.split(self._split_sizes, dim=1)
        f_dim, h_dim, w_dim = self._split_sizes
        parts = [
            freqs_f[:frames].view(frames, 1, 1, f_dim).expand(frames, height, width, f_dim),
            freqs_h[:height].view(1, height, 1, h_dim).expand(frames, height, width, h_dim),
            freqs_w[:width].view(1, 1, width, w_dim).expand(frames, height, width, w_dim),
        ]
        return torch.cat(parts, dim=-1).reshape(1, 1, frames * height * width, -1)


class SanaWmCameraEmbedder(nn.Module):
    """Small camera branch used by the native smoke path."""

    def __init__(self, config: SanaWmConfig | None = None) -> None:
        super().__init__()
        config = config or SanaWmConfig()
        self.hidden_size = config.hidden_size
        self.plucker = nn.Module()
        self.plucker.proj = nn.Conv3d(config.chunk_plucker_channels, config.hidden_size, kernel_size=1)
        self.raymap = nn.Module()
        self.raymap.proj = nn.Linear(20, config.hidden_size)

    @staticmethod
    def _match_tokens(hidden_states: torch.Tensor, expected_tokens: int) -> torch.Tensor:
        if hidden_states.shape[1] == expected_tokens:
            return hidden_states
        if hidden_states.shape[1] > expected_tokens:
            return hidden_states[:, :expected_tokens]
        pad = hidden_states[:, -1:].expand(-1, expected_tokens - hidden_states.shape[1], -1)
        return torch.cat([hidden_states, pad], dim=1)

    def forward(
        self,
        *,
        plucker: torch.Tensor | None = None,
        raymap: torch.Tensor | None = None,
        spatial_shape: tuple[int, int, int],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        frames, height, width = spatial_shape
        expected_tokens = frames * height * width
        hidden_states = None

        if plucker is not None:
            if plucker.ndim == 4:
                plucker = plucker.unsqueeze(0)
            plucker = plucker.to(device=device, dtype=dtype)
            if plucker.shape[0] == 1 and batch_size > 1:
                plucker = plucker.expand(batch_size, -1, -1, -1, -1)
            plucker_hidden = self.plucker.proj(plucker).flatten(2).transpose(1, 2)
            hidden_states = self._match_tokens(plucker_hidden, expected_tokens)

        if raymap is not None:
            if raymap.ndim == 2:
                raymap = raymap.unsqueeze(0)
            raymap = raymap.to(device=device, dtype=dtype)
            if raymap.shape[0] == 1 and batch_size > 1:
                raymap = raymap.expand(batch_size, -1, -1)
            ray_hidden = self.raymap.proj(raymap)
            ray_hidden = ray_hidden.repeat_interleave(height * width, dim=1)
            ray_hidden = self._match_tokens(ray_hidden, expected_tokens)
            hidden_states = ray_hidden if hidden_states is None else hidden_states + ray_hidden

        return hidden_states


class SanaWmSelfAttention(nn.Module):
    def __init__(self, config: SanaWmConfig, *, use_gdn: bool = True) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = max(hidden_size // max(config.linear_head_dim, 1), 1)
        self.head_dim = hidden_size // self.num_heads
        self.heads = self.num_heads
        self.dim = self.head_dim
        self.in_dim = hidden_size
        self.out_dim = hidden_size
        self.eps = 1e-8
        self.attn_type = config.attn_type
        self.use_gdn = use_gdn and "GDN" in config.attn_type
        self.k_conv_only = config.k_conv_only
        self.conv_kernel_size = config.conv_kernel_size
        self.qkv = nn.Linear(hidden_size, self.num_heads * self.head_dim * 3, bias=False)
        self.proj = nn.Linear(self.num_heads * self.head_dim, hidden_size)
        norm_cls: type[nn.Module] = SanaWmRMSNorm if config.qk_norm else nn.Identity
        self.q_norm = norm_cls(self.num_heads * self.head_dim)
        self.k_norm = norm_cls(self.num_heads * self.head_dim)

        # These names mirror the official GDN checkpoint. The current forward
        # executes a pure PyTorch recurrence; the fused Triton scan is ported
        # separately.
        self.A_log = nn.Parameter(torch.zeros(self.num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.register_buffer("recall_gate", torch.zeros(1))
        self.beta_proj = nn.Linear(hidden_size, self.num_heads)
        self.gate_proj = nn.Linear(hidden_size, self.num_heads)
        self.output_gate = nn.Linear(hidden_size, hidden_size)
        self.conv_k = (
            nn.Conv1d(hidden_size, hidden_size, kernel_size=config.conv_kernel_size, groups=hidden_size)
            if config.conv_kernel_size > 0
            else None
        )
        self.conv_q = None
        self.conv_v = None

        cam_compress = max(config.cam_attn_compress, 1)
        self.cam_dim = hidden_size // cam_compress
        self.cam_heads = max(self.num_heads // cam_compress, 1)
        self.cam_head_dim = self.cam_dim // self.cam_heads
        self.q_proj_cam = nn.Linear(hidden_size, self.cam_dim, bias=True)
        self.k_proj_cam = nn.Linear(hidden_size, self.cam_dim, bias=True)
        self.v_proj_cam = nn.Linear(hidden_size, self.cam_dim, bias=True)
        self.out_proj_cam = nn.Linear(self.cam_dim, hidden_size, bias=True)
        self.q_norm_cam = norm_cls(self.cam_dim)
        self.k_norm_cam = norm_cls(self.cam_dim)
        self.conv_k_cam = (
            nn.Conv1d(self.cam_dim, self.cam_dim, kernel_size=config.conv_kernel_size, groups=self.cam_dim)
            if config.conv_kernel_size > 0
            else None
        )
        self.conv_q_cam = None
        self.conv_v_cam = None
        self._init_short_convs()

    @staticmethod
    def _init_short_conv(conv: nn.Conv1d | None) -> None:
        if conv is None:
            return
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[:, 0, -1] = 1.0
            if conv.bias is not None:
                conv.bias.zero_()

    def _init_short_convs(self) -> None:
        for conv in (self.conv_q, self.conv_k, self.conv_v, self.conv_q_cam, self.conv_k_cam, self.conv_v_cam):
            self._init_short_conv(conv)

    @staticmethod
    def _flip_and_shift(tensor: torch.Tensor, *, dim: int, shift_value: float) -> torch.Tensor:
        flipped = torch.flip(tensor, dims=[dim])
        shifted = flipped.narrow(dim, 0, tensor.shape[dim] - 1)
        pad_shape = list(tensor.shape)
        pad_shape[dim] = 1
        padding = torch.full(pad_shape, shift_value, device=tensor.device, dtype=tensor.dtype)
        return torch.cat([padding, shifted], dim=dim)

    @staticmethod
    def _apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        rotated = torch.view_as_complex(
            hidden_states.permute(0, 1, 3, 2).to(torch.float64).unflatten(3, (-1, 2))
        )
        output = torch.view_as_real(rotated * freqs).flatten(3, 4).permute(0, 1, 3, 2)
        return output.type_as(hidden_states)

    @classmethod
    def _apply_rotary_emb_to_sdpa(cls, hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        return cls._apply_rotary_emb(hidden_states.transpose(2, 3), freqs).transpose(2, 3)

    @staticmethod
    def _reshape_to_temporal(
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, int, int, int]:
        batch_size, token_count, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(f"Sana-WM temporal conv expects N=T*H*W, got N={token_count}, THW={spatial_shape}.")
        hidden_states = hidden_states.reshape(batch_size, frames, spatial_tokens, hidden_size)
        hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(batch_size * spatial_tokens, frames, hidden_size)
        return hidden_states, batch_size, spatial_tokens, frames

    @staticmethod
    def _reshape_from_temporal(
        hidden_states: torch.Tensor,
        batch_size: int,
        spatial_tokens: int,
        frames: int,
    ) -> torch.Tensor:
        hidden_size = hidden_states.shape[-1]
        return (
            hidden_states.reshape(batch_size, spatial_tokens, frames, hidden_size)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, frames * spatial_tokens, hidden_size)
        )

    @staticmethod
    def _causal_conv_1d(hidden_states: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        dtype = hidden_states.dtype
        conv_input = hidden_states.transpose(1, 2).to(conv.weight.dtype)
        conv_input = F.pad(conv_input, (conv.kernel_size[0] - 1, 0))
        output = conv(conv_input).transpose(1, 2)
        return output.to(dtype)

    def _bidirectional_temporal_short_conv(
        self,
        hidden_states: torch.Tensor,
        conv: nn.Conv1d,
        spatial_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        hidden_states, batch_size, spatial_tokens, frames = self._reshape_to_temporal(hidden_states, spatial_shape)
        forward_states = self._causal_conv_1d(hidden_states, conv)
        backward_states = self._causal_conv_1d(hidden_states.flip(1), conv).flip(1)
        center_weight = conv.weight[:, 0, -1].to(hidden_states.dtype)
        center_states = hidden_states * center_weight.view(1, 1, -1)
        output = forward_states + backward_states - center_states
        return self._reshape_from_temporal(output, batch_size, spatial_tokens, frames)

    def _compute_frame_gates(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(
                f"Sana-WM GDN token layout mismatch: N={token_count}, expected {frames * spatial_tokens}."
            )
        beta = torch.sigmoid(self.beta_proj(hidden_states))
        beta = beta.reshape(batch_size, frames, spatial_tokens, self.num_heads).permute(0, 3, 1, 2)
        frame_states = hidden_states.reshape(batch_size, frames, spatial_tokens, hidden_size).mean(dim=2)
        gate = self.gate_proj(frame_states).float()
        decay = torch.exp(
            -self.A_log.float().exp().view(1, 1, -1)
            * F.softplus(gate + self.dt_bias.float().view(1, 1, -1))
        )
        return beta, decay.transpose(1, 2)

    def _gdn_update_components(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_rot: torch.Tensor,
        key_rot: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
        *,
        spatial_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_heads, head_dim, token_count = query.shape
        frames = beta.shape[2]

        def to_frames(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, num_heads, head_dim, frames, spatial_tokens).permute(0, 1, 3, 2, 4)

        query_f = to_frames(query)
        key_f = to_frames(key)
        value_f = to_frames(value)
        query_rot_f = to_frames(query_rot)
        key_rot_f = to_frames(key_rot)
        state_kv = torch.zeros(
            batch_size,
            num_heads,
            head_dim,
            head_dim,
            device=query.device,
            dtype=query.dtype,
        )
        state_z = torch.zeros(batch_size, num_heads, head_dim, 1, device=query.device, dtype=query.dtype)
        numerators: list[torch.Tensor] = []
        denominators: list[torch.Tensor] = []

        for frame_idx in range(frames):
            query_t = query_f[:, :, frame_idx]
            key_t = key_f[:, :, frame_idx]
            value_t = value_f[:, :, frame_idx]
            query_rot_t = query_rot_f[:, :, frame_idx]
            key_rot_t = key_rot_f[:, :, frame_idx]
            beta_t = beta[:, :, frame_idx].unsqueeze(2)
            decay_t = decay[:, :, frame_idx].view(batch_size, num_heads, 1, 1)

            state_kv = state_kv * decay_t
            state_z = state_z * decay_t
            value_pred = torch.matmul(state_kv, key_rot_t)
            delta_value = (value_t - value_pred) * beta_t
            state_kv = state_kv + torch.matmul(delta_value, key_rot_t.transpose(-1, -2))

            z_pred = torch.matmul(state_z.transpose(-1, -2), key_t)
            delta_z = (1.0 - z_pred) * beta_t
            state_z = state_z + torch.matmul(key_t, delta_z.transpose(-1, -2))

            numerators.append(torch.matmul(state_kv, query_rot_t))
            denominators.append(torch.matmul(state_z.transpose(-1, -2), query_t))

        def restore(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
            stacked = torch.stack(tensors, dim=2)
            return stacked.permute(0, 1, 3, 2, 4).reshape(batch_size, num_heads, dim, token_count)

        return restore(numerators, head_dim), restore(denominators, 1)

    def _forward_gdn(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, token_count, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(f"Sana-WM GDN expects N=T*H*W, got N={token_count}, THW={spatial_shape}.")

        query, key, value = self.qkv(hidden_states).chunk(3, dim=-1)
        if self.conv_k is not None:
            key = self._bidirectional_temporal_short_conv(key, self.conv_k, spatial_shape)
        beta, decay = self._compute_frame_gates(hidden_states, spatial_shape)

        if hidden_states.is_cuda:
            qkv = torch.stack(
                (
                    query.reshape(batch_size, token_count, self.num_heads, self.head_dim),
                    key.reshape(batch_size, token_count, self.num_heads, self.head_dim),
                    value.reshape(batch_size, token_count, self.num_heads, self.head_dim),
                ),
                dim=2,
            )
            try:
                output = triton_bidirectional_gated_delta_net_from_qkv(
                    qkv,
                    beta=beta,
                    decay=decay,
                    q_norm=self.q_norm,
                    k_norm=self.k_norm,
                    spatial_tokens=spatial_tokens,
                    rotary_emb=rotary_emb,
                    k_scale=(self.head_dim**-0.5) * (spatial_tokens**-0.5),
                    eps=self.eps,
                )
                output = output.permute(0, 3, 1, 2).reshape(batch_size, token_count, hidden_size)
                gate = F.silu(self.output_gate(hidden_states).float()).to(output.dtype)
                return self.proj((output * gate).to(self.proj.weight.dtype))
            except Exception:
                if os.environ.get(SANA_WM_REQUIRE_TRITON_GDN_ENV, "").lower() in {"1", "true", "yes", "on"}:
                    raise
                # The fused path is an optimization. Keep the reference path as
                # the correctness fallback for unsupported GPUs / Triton builds.
                pass
        query = self.q_norm(query).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        key = self.k_norm(key).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, token_count, self.num_heads, self.head_dim)

        query = F.relu(query).permute(0, 2, 3, 1)
        key = F.relu(key).permute(0, 2, 3, 1)
        value = value.permute(0, 2, 3, 1)
        key = key * ((self.head_dim**-0.5) * (spatial_tokens**-0.5))
        if rotary_emb is not None:
            query_rot = self._apply_rotary_emb(query, rotary_emb)
            key_rot = self._apply_rotary_emb(key, rotary_emb)
        else:
            query_rot = query
            key_rot = key

        output = reference_bidirectional_gated_delta_net(
            query,
            key,
            value,
            beta=beta,
            decay=decay,
            spatial_tokens=spatial_tokens,
            query_rot=query_rot,
            key_rot=key_rot,
            eps=self.eps,
        )
        output = output.permute(0, 3, 1, 2).reshape(batch_size, token_count, hidden_size)
        gate = F.silu(self.output_gate(hidden_states).float()).to(output.dtype)
        return self.proj((output * gate).to(self.proj.weight.dtype))

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = tensor.shape
        tensor = tensor.reshape(batch, seq_len, self.num_heads, hidden_size // self.num_heads)
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = tensor.shape
        return tensor.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if spatial_shape is not None and self.use_gdn:
            return self._forward_gdn(hidden_states, spatial_shape, rotary_emb)
        query, key, value = self.qkv(hidden_states).chunk(3, dim=-1)
        query = self._split_heads(self.q_norm(query))
        key = self._split_heads(self.k_norm(key))
        value = self._split_heads(value)
        if rotary_emb is not None:
            query = self._apply_rotary_emb_to_sdpa(query, rotary_emb)
            key = self._apply_rotary_emb_to_sdpa(key, rotary_emb)
        attn = F.scaled_dot_product_attention(query, key, value)
        return self.proj(self._merge_heads(attn))


class SanaWmCrossAttention(nn.Module):
    def __init__(self, config: SanaWmConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = max(hidden_size // max(config.linear_head_dim, 1), 1)
        self.head_dim = hidden_size // self.num_heads
        inner = self.num_heads * self.head_dim
        self.q_linear = nn.Linear(hidden_size, inner)
        self.kv_linear = nn.Linear(hidden_size, inner * 2)
        self.proj = nn.Linear(inner, hidden_size)
        norm_cls: type[nn.Module] = SanaWmRMSNorm if config.cross_norm else nn.Identity
        self.q_norm = norm_cls(inner)
        self.k_norm = norm_cls(inner)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = tensor.shape
        tensor = tensor.reshape(batch, seq_len, self.num_heads, hidden_size // self.num_heads)
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = tensor.shape
        return tensor.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.q_norm(self.q_linear(hidden_states)))
        key, value = self.kv_linear(encoder_hidden_states).chunk(2, dim=-1)
        key = self._split_heads(self.k_norm(key))
        value = self._split_heads(value)
        attn = F.scaled_dot_product_attention(query, key, value)
        return self.proj(self._merge_heads(attn))


class _ConvWrapper(nn.Module):
    def __init__(self, conv: nn.Module) -> None:
        super().__init__()
        self.conv = conv


class SanaWmMbConvFfn(nn.Module):
    def __init__(self, config: SanaWmConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        expanded = int(hidden_size * config.mlp_ratio) * 2
        t_padding = config.t_kernel_size // 2
        self.glu_act = nn.SiLU()
        self.inverted_conv = _ConvWrapper(nn.Conv2d(hidden_size, expanded, kernel_size=1))
        self.depth_conv = _ConvWrapper(nn.Conv2d(expanded, expanded, kernel_size=3, padding=1, groups=expanded))
        self.point_conv = _ConvWrapper(nn.Conv2d(expanded // 2, hidden_size, kernel_size=1, bias=False))
        self.t_conv = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=(config.t_kernel_size, 1),
            padding=(t_padding, 0),
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
        batch, _, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        x = hidden_states.transpose(1, 2).reshape(batch, hidden_size, frames, height * width)
        x = self.inverted_conv.conv(x)
        x = self.depth_conv.conv(x)
        value, gate = x.chunk(2, dim=1)
        x = value * self.glu_act(gate)
        x = self.point_conv.conv(x)
        x = x + self.t_conv(x)
        return x.reshape(batch, hidden_size, frames * height * width).transpose(1, 2)


class SanaWmBlock(nn.Module):
    def __init__(self, config: SanaWmConfig, *, block_idx: int = 0) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        use_gdn = config.softmax_every_n <= 0 or (block_idx + 1) % config.softmax_every_n != 0
        use_plucker_proj = config.use_chunk_plucker_post_attn and (
            config.chunk_plucker_post_attn_blocks < 0 or block_idx < config.chunk_plucker_post_attn_blocks
        )
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SanaWmSelfAttention(config, use_gdn=use_gdn)
        self.cross_attn = SanaWmCrossAttention(config)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = SanaWmMbConvFfn(config)
        self.plucker_proj = nn.Linear(hidden_size, hidden_size) if use_plucker_proj else None
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))

    @staticmethod
    def _modulate(hidden_states: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return hidden_states * (1 + scale) + shift

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_modulation: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None = None,
        camera_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep_modulation.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        attn_input = self._modulate(self.norm1(hidden_states), shift_msa, scale_msa)
        attn_output = self.attn(attn_input, spatial_shape, rotary_emb)
        if camera_hidden_states is not None and self.plucker_proj is not None:
            attn_output = attn_output + self.plucker_proj(camera_hidden_states)
        hidden_states = hidden_states + gate_msa * attn_output
        hidden_states = hidden_states + self.cross_attn(hidden_states, encoder_hidden_states)
        mlp_input = self._modulate(self.norm2(hidden_states), shift_mlp, scale_mlp)
        hidden_states = hidden_states + gate_mlp * self.mlp(mlp_input, spatial_shape)
        return hidden_states


class SanaWmFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: tuple[int, int, int], out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.zeros(2, hidden_size))
        self.linear = nn.Linear(hidden_size, _prod(patch_size) * out_channels)
        self.out_channels = out_channels

    def forward(self, hidden_states: torch.Tensor, timestep_embed: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.scale_shift_table[None] + timestep_embed[:, None]).chunk(2, dim=1)
        hidden_states = self.norm_final(hidden_states) * (1 + scale) + shift
        return self.linear(hidden_states)


class SanaWmTransformer3DModel(nn.Module):
    """SANA-WM Stage-1 DiT with a runnable pure-PyTorch GDN fallback path."""

    _repeated_blocks: ClassVar[list[str]] = ["blocks"]
    _layerwise_offload_blocks_attr: ClassVar[str] = "blocks"

    def __init__(
        self,
        config: SanaWmConfig | None = None,
        *,
        materialize: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        latent_channels: int = SANA_WM_STAGE1_LATENT_CHANNELS,
        prompt_channels: int = SANA_WM_STAGE1_PROMPT_CHANNELS,
    ) -> None:
        super().__init__()
        self.config = config or SanaWmConfig()
        self._latent_channels = latent_channels
        self._prompt_channels = prompt_channels
        self._is_materialized = False
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self._loaded_parameters = nn.ParameterDict()
        self._source_to_remapped_name: dict[str, str] = {}
        self._remapped_to_storage_name: dict[str, str] = {}
        self._storage_to_remapped_name: dict[str, str] = {}
        self._materialized_loaded_names: set[str] = set()
        self.last_load_report = SanaWmStage1LoadReport()
        if materialize:
            self.materialize(
                device=device,
                dtype=dtype,
                latent_channels=latent_channels,
                prompt_channels=prompt_channels,
            )

    @property
    def is_materialized(self) -> bool:
        return self._is_materialized

    def materialize(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        latent_channels: int | None = None,
        prompt_channels: int | None = None,
    ) -> None:
        if self._is_materialized:
            return
        self._latent_channels = latent_channels or self._latent_channels
        self._prompt_channels = prompt_channels or self._prompt_channels
        self.patch_size = _to_3tuple(self.config.patch_size)
        self.x_embedder = SanaWmPatchEmbedMS3D(self.patch_size, self._latent_channels, self.config.hidden_size)
        self.y_embedder = SanaWmTextEmbedder(
            self._prompt_channels,
            self.config.hidden_size,
            self.config.model_max_length,
        )
        self.t_embedder = SanaWmTimestepEmbedder(SANA_WM_STAGE1_TIMESTEP_CHANNELS, self.config.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.config.hidden_size, 6 * self.config.hidden_size),
        )
        self.plucker_embedder = SanaWmPatchEmbedMS3D(
            self.patch_size,
            self.config.chunk_plucker_channels,
            self.config.hidden_size,
        )
        self.raymap_embedder = SanaWmPatchEmbedMS3D(self.patch_size, 3, self.config.hidden_size)
        self.blocks = nn.ModuleList([SanaWmBlock(self.config, block_idx=i) for i in range(self.config.num_blocks)])
        self.final_layer = SanaWmFinalLayer(self.config.hidden_size, self.patch_size, self._latent_channels)
        self.pos_embed = nn.Parameter(torch.zeros(1, 484, self.config.hidden_size))
        self.rope = SanaWmWanRotaryPosEmbed(self.config.linear_head_dim)
        self.attention_y_norm = SanaWmRMSNorm(self.config.hidden_size)
        self.to(device=device, dtype=dtype)
        self._is_materialized = True
        self._apply_loaded_tensors_to_materialized()

    @staticmethod
    def _storage_name(remapped_name: str, index: int) -> str:
        digest = hashlib.sha1(remapped_name.encode("utf-8")).hexdigest()[:16]
        return f"w_{index}_{digest}"

    @staticmethod
    def _local_name(remapped_name: str) -> str:
        return remapped_name.removeprefix("transformer.")

    def _copy_to_materialized_param(
        self,
        remapped_name: str,
        tensor: torch.Tensor,
        *,
        require_target: bool = False,
    ) -> bool:
        if not self._is_materialized or not remapped_name.startswith("transformer."):
            if require_target:
                raise ValueError(
                    f"Sana-WM weight {remapped_name} was remapped but is not a transformer weight "
                    "that can be consumed by the materialized Stage-1 model."
                )
            return False
        local_name = self._local_name(remapped_name)
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        target = params.get(local_name)
        if target is None:
            target = buffers.get(local_name)
        if target is None:
            if require_target:
                raise ValueError(
                    f"Sana-WM weight {remapped_name} was remapped but not consumed by the "
                    "materialized Stage-1 model."
                )
            return False
        if tuple(target.shape) != tuple(tensor.shape):
            raise ValueError(
                f"Sana-WM weight shape mismatch for {remapped_name}: "
                f"expected {tuple(target.shape)}, got {tuple(tensor.shape)}."
            )
        with torch.no_grad():
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
        self._materialized_loaded_names.add(remapped_name)
        return True

    def _store_tensor(self, remapped_name: str, tensor: torch.Tensor) -> None:
        if self._is_materialized:
            self._copy_to_materialized_param(remapped_name, tensor, require_target=True)
            return
        storage_name = self._storage_name(remapped_name, len(self._remapped_to_storage_name))
        target_device = self._device_anchor.device
        if target_device.type != "meta" and tensor.device != target_device:
            tensor = tensor.to(target_device)
        if torch.is_floating_point(tensor) or torch.is_complex(tensor):
            self._loaded_parameters[storage_name] = nn.Parameter(tensor.detach(), requires_grad=False)
        else:
            self.register_buffer(storage_name, tensor.detach(), persistent=True)
        self._remapped_to_storage_name[remapped_name] = storage_name
        self._storage_to_remapped_name[storage_name] = remapped_name

    def _apply_loaded_tensors_to_materialized(self) -> None:
        applied: list[str] = []
        unapplied: list[str] = []
        for remapped_name, storage_name in list(self._remapped_to_storage_name.items()):
            tensor = (
                self._loaded_parameters[storage_name]
                if storage_name in self._loaded_parameters
                else getattr(self, storage_name)
            )
            if self._copy_to_materialized_param(remapped_name, tensor):
                applied.append(remapped_name)
            else:
                unapplied.append(remapped_name)
        self.last_load_report = replace(
            self.last_load_report,
            materialized_weights=len(applied),
            unapplied_weights=tuple(unapplied),
        )
        if unapplied:
            raise ValueError(
                "Sana-WM Stage-1 checkpoint keys were remapped but not consumed by the "
                f"materialized model: {unapplied[:10]}"
            )

    def get_loaded_tensor(self, remapped_name: str) -> torch.Tensor:
        storage_name = self._remapped_to_storage_name.get(remapped_name)
        if storage_name is not None:
            if storage_name in self._loaded_parameters:
                return self._loaded_parameters[storage_name]
            return getattr(self, storage_name)
        if self._is_materialized and remapped_name in self._materialized_loaded_names:
            local_name = self._local_name(remapped_name)
            params = dict(self.named_parameters())
            buffers = dict(self.named_buffers())
            target = params.get(local_name)
            if target is None:
                target = buffers.get(local_name)
            if target is not None:
                return target
        raise KeyError(remapped_name)

    def _positional_embedding(self, token_count: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        pos_embed = self.pos_embed.to(device=device, dtype=dtype)
        if pos_embed.shape[1] == token_count:
            return pos_embed
        pos_embed = pos_embed.transpose(1, 2)
        pos_embed = F.interpolate(pos_embed, size=token_count, mode="linear", align_corners=False)
        return pos_embed.transpose(1, 2)

    @staticmethod
    def _match_tokens(hidden_states: torch.Tensor, expected_tokens: int) -> torch.Tensor:
        if hidden_states.shape[1] == expected_tokens:
            return hidden_states
        if hidden_states.shape[1] > expected_tokens:
            return hidden_states[:, :expected_tokens]
        pad = hidden_states[:, -1:].expand(-1, expected_tokens - hidden_states.shape[1], -1)
        return torch.cat([hidden_states, pad], dim=1)

    def _camera_hidden_states_from_conditions(
        self,
        *,
        plucker: torch.Tensor | None,
        spatial_shape: tuple[int, int, int],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if plucker is None or not hasattr(self, "plucker_embedder"):
            return None
        if plucker.ndim == 4:
            plucker = plucker.unsqueeze(0)
        plucker = plucker.to(device=device, dtype=dtype)
        if plucker.shape[0] == 1 and batch_size > 1:
            plucker = plucker.expand(batch_size, -1, -1, -1, -1)
        camera_hidden_states = self.plucker_embedder(plucker)
        expected_tokens = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
        return self._match_tokens(camera_hidden_states, expected_tokens)

    def _unpatchify(self, hidden_states: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        frames, height, width = spatial_shape
        patch_frames, patch_height, patch_width = self.patch_size
        hidden_states = hidden_states.reshape(
            batch_size,
            frames,
            height,
            width,
            patch_frames,
            patch_height,
            patch_width,
            self._latent_channels,
        )
        hidden_states = torch.einsum("nfhwopqc->ncfohpwq", hidden_states)
        return hidden_states.reshape(
            batch_size,
            self._latent_channels,
            frames * patch_frames,
            height * patch_height,
            width * patch_width,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor | float | int,
        *,
        encoder_hidden_states: torch.Tensor | None = None,
        camera_hidden_states: torch.Tensor | None = None,
        camera_encoder: SanaWmCameraEmbedder | None = None,
        plucker: torch.Tensor | None = None,
        raymap: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 5:
            raise ValueError("Sana-WM transformer expects latent input shaped [B, C, F, H, W].")
        if not self._is_materialized:
            prompt_channels = (
                int(encoder_hidden_states.shape[-1])
                if encoder_hidden_states is not None
                else SANA_WM_STAGE1_PROMPT_CHANNELS
            )
            self.materialize(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                latent_channels=int(hidden_states.shape[1]),
                prompt_channels=prompt_channels,
            )

        batch_size = hidden_states.shape[0]
        latent_shape = hidden_states.shape[2:]
        hidden_states, spatial_shape = self.x_embedder.project_with_shape(hidden_states)
        rotary_emb = None
        if self.config.pos_embed_type == "wan_rope":
            rotary_emb = self.rope(spatial_shape, hidden_states.device)
        else:
            hidden_states = hidden_states + self._positional_embedding(
                hidden_states.shape[1], hidden_states.dtype, hidden_states.device
            )

        encoder_hidden_states = self.y_embedder(
            encoder_hidden_states.to(device=hidden_states.device) if encoder_hidden_states is not None else None,
            batch_size=batch_size,
            dtype=hidden_states.dtype,
        )
        encoder_hidden_states = self.attention_y_norm(encoder_hidden_states)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=hidden_states.device, dtype=torch.float32)
        timestep = timestep.to(device=hidden_states.device)
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        elif timestep.ndim == 1 and timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        time_embed = self.t_embedder(timestep)
        timestep_modulation = self.t_block(time_embed).to(hidden_states.dtype)

        if camera_hidden_states is None:
            camera_hidden_states = self._camera_hidden_states_from_conditions(
                plucker=plucker,
                spatial_shape=spatial_shape,
                batch_size=batch_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        if camera_hidden_states is None and camera_encoder is not None:
            camera_hidden_states = camera_encoder(
                plucker=plucker,
                raymap=raymap,
                spatial_shape=spatial_shape,
                batch_size=batch_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_modulation,
                spatial_shape,
                rotary_emb,
                camera_hidden_states,
            )

        hidden_states = self.final_layer(hidden_states, time_embed.to(hidden_states.dtype))
        hidden_states = self._unpatchify(hidden_states, spatial_shape)
        if hidden_states.shape[2:] != latent_shape:
            hidden_states = F.interpolate(hidden_states, size=latent_shape, mode="trilinear", align_corners=False)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, Any]]) -> set[str]:
        loaded: set[str] = set()
        unmapped: list[str] = []
        duplicates: list[str] = []
        total = 0

        for source_name, tensor in weights:
            total += 1
            remapped_name = normalize_sana_wm_stage1_weight_name(source_name)
            if remapped_name is None:
                unmapped.append(source_name)
                continue
            if (
                remapped_name in self._remapped_to_storage_name
                or remapped_name in self._materialized_loaded_names
                or remapped_name in loaded
            ):
                duplicates.append(remapped_name)
                continue
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Sana-WM weight {source_name!r} must be a torch.Tensor, got {type(tensor).__name__}.")
            self._store_tensor(remapped_name, tensor)
            self._source_to_remapped_name[source_name] = remapped_name
            loaded.add(remapped_name)

        self.last_load_report = SanaWmStage1LoadReport(
            total_weights=total,
            loaded_weights=len(loaded),
            unmapped_weights=tuple(unmapped),
            duplicate_weights=tuple(duplicates),
            loaded_names=tuple(sorted(loaded)),
            materialized_weights=sum(1 for name in loaded if name in self._materialized_loaded_names),
            unapplied_weights=tuple(
                name for name in sorted(loaded) if self._is_materialized and name not in self._materialized_loaded_names
            ),
        )
        if unmapped or duplicates:
            details = []
            if unmapped:
                details.append(f"unmapped={unmapped[:10]}")
            if duplicates:
                details.append(f"duplicates={duplicates[:10]}")
            raise ValueError("Invalid SANA-WM Stage-1 checkpoint keys: " + "; ".join(details))
        return loaded
