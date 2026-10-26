# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright 2026 The Hy-Nomas Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native Hy-Nomas DiT with caller-owned segmented KV and Ulysses attention."""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.normalization import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.distributed.comm import all_to_all_4D
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.layers.norm import LayerNorm
from vllm_omni.diffusion.models.flux.flux_transformer import FeedForward
from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import HunyuanVideo15PatchEmbed


def _attention(heads: int, head_dim: int, role: str) -> Attention:
    return Attention(
        num_heads=heads,
        head_size=head_dim,
        causal=False,
        softmax_scale=head_dim**-0.5,
        role=role,
        role_category="cross" if role == "hy_nomas.camera" else "self",
        skip_sequence_parallel=True,
    )


def _attend(
    attention: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # Preserve the reference SDPA sequence layout: removing masked keys changes
    # kernel selection and BF16 reduction order, even with equivalent visibility.
    if mask is None:
        return attention(query, key, value)
    if mask.ndim == 2:
        mask = mask[:, None, None, :]
    return attention(query, key, value, attn_metadata=AttentionMetadata(attn_mask=mask))


@dataclass(frozen=True)
class HyNomasTransformerConfig:
    in_channels: int = 32
    out_channels: int = 32
    num_layers: int = 54
    num_attention_heads: int = 16
    attention_head_dim: int = 128
    text_embed_dim: int = 3584
    mlp_ratio: float = 4.0
    rope_axes_dim: tuple[int, int, int] = (16, 56, 56)
    rope_theta: float = 256.0
    camera_embed_dim: int = 128
    camera_attention_dim: int = 1024
    camera_num_attention_heads: int = 16
    camera_rope_axes_dim: tuple[int, int, int] = (8, 28, 28)


def _apply_rotary_emb(hidden_states, rotary_emb):
    cosine, sine = rotary_emb
    real, imaginary = hidden_states.float().unflatten(-1, (-1, 2)).unbind(-1)
    rotated = torch.stack((-imaginary, real), dim=-1).flatten(-2)
    return (hidden_states.float() * cosine[None, :, None] + rotated * sine[None, :, None]).to(hidden_states.dtype)


class HyNomasRotaryPosEmbed(nn.Module):
    def __init__(self, axes_dim, theta):
        super().__init__()
        self.axes_dim = axes_dim
        self.theta = theta

    def forward(self, num_frames, height, width, device, frame_offset=0) -> tuple[torch.Tensor, torch.Tensor]:
        # Continuation positions in the released model use CPU trigonometry. Its rounding matters before
        # casting the rotated query and key back to the model dtype.
        computation_device = "cpu" if frame_offset else device
        positions = torch.meshgrid(
            torch.arange(frame_offset, frame_offset + num_frames, device=computation_device, dtype=torch.float32),
            torch.arange(height, device=computation_device, dtype=torch.float32),
            torch.arange(width, device=computation_device, dtype=torch.float32),
            indexing="ij",
        )
        cosines, sines = [], []
        for dim, position in zip(self.axes_dim, positions):
            frequencies = 1.0 / (self.theta ** (torch.arange(0, dim, 2, device=computation_device).float() / dim))
            angles = torch.outer(position.flatten(), frequencies)
            cosines.append(angles.cos().repeat_interleave(2, dim=-1))
            sines.append(angles.sin().repeat_interleave(2, dim=-1))
        return torch.cat(cosines, dim=-1).to(device), torch.cat(sines, dim=-1).to(device)


class HyNomasTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        # CPU frequencies preserve the released model's rounding and are never cast with model weights.
        self.frequencies = tuple(
            torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32, device="cpu") / 128).tolist()
        )

    def forward(self, timestep, dtype) -> torch.Tensor:
        frequencies = torch.tensor(self.frequencies, dtype=torch.float32, device=timestep.device)
        angles = timestep[:, None].float() * frequencies[None]
        hidden_states = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return self.mlp(hidden_states.to(dtype))


class HyNomasTextProjection(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states = self.linear_1(hidden_states)
        hidden_states = F.silu(hidden_states)
        return self.linear_2(hidden_states)


class HyNomasTokenRefinerMLP(nn.Module):
    """The text refiner uses SiLU, unlike the shared HY1.5 GELU feed-forward."""

    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.fc2(hidden_states)


class HyNomasTextAttnProcessor:
    def __call__(self, attn, hidden_states, attention_mask=None):
        query, key, value = attn.self_attn_qkv(hidden_states).unflatten(-1, (3, attn.heads, -1)).unbind(-3)
        hidden_states = _attend(attn.attention, query, key, value, attention_mask)
        return attn.self_attn_proj(hidden_states.flatten(-2))


class HyNomasTextAttention(nn.Module):
    _default_processor_cls = HyNomasTextAttnProcessor

    def __init__(self, hidden_size, heads):
        super().__init__()
        self.heads = heads
        self.self_attn_qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.self_attn_proj = nn.Linear(hidden_size, hidden_size)
        self.attention = _attention(heads, hidden_size // heads, "hy_nomas.text")
        self.processor = self._default_processor_cls()

    def forward(self, hidden_states, attention_mask=None) -> torch.Tensor:
        return self.processor(self, hidden_states, attention_mask)


class HyNomasTokenRefinerBlock(nn.Module):
    def __init__(self, hidden_size, heads):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, eps=1e-6)
        self.attn = HyNomasTextAttention(hidden_size, heads)
        self.norm2 = LayerNorm(hidden_size, eps=1e-6)
        self.mlp = HyNomasTokenRefinerMLP(hidden_size)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, hidden_states, temb, attention_mask) -> torch.Tensor:
        attention_gate, mlp_gate = self.adaLN_modulation(temb).chunk(2, dim=-1)
        hidden_states = (
            hidden_states
            + self.attn(self.norm1(hidden_states).to(hidden_states.dtype), attention_mask) * attention_gate[:, None]
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states).to(hidden_states.dtype)) * mlp_gate[:, None]
        return hidden_states


class HyNomasIndividualTokenRefiner(nn.Module):
    def __init__(self, hidden_size, heads):
        super().__init__()
        self.blocks = nn.ModuleList([HyNomasTokenRefinerBlock(hidden_size, heads) for _ in range(2)])

    def forward(self, hidden_states, temb, attention_mask) -> torch.Tensor:
        if attention_mask is not None:
            attention_mask = attention_mask.clone().bool()
            attention_mask[:, 0] = True
            attention_mask = attention_mask[:, None, None, :]
        for block in self.blocks:
            hidden_states = block(hidden_states, temb, attention_mask)
        return hidden_states


class HyNomasTokenRefiner(nn.Module):
    def __init__(self, in_channels, hidden_size, heads):
        super().__init__()
        self.input_embedder = nn.Linear(in_channels, hidden_size)
        self.t_embedder = HyNomasTimestepEmbedder(hidden_size)
        self.c_embedder = HyNomasTextProjection(in_channels, hidden_size)
        self.individual_token_refiner = HyNomasIndividualTokenRefiner(hidden_size, heads)

    def forward(self, hidden_states, timestep, attention_mask=None) -> torch.Tensor:
        temb = self.t_embedder(timestep, hidden_states.dtype)
        if attention_mask is None:
            context = hidden_states.mean(dim=1)
        else:
            mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
            context = (hidden_states * mask).sum(dim=1, dtype=torch.float32) / mask.sum(
                dim=1, dtype=torch.float32
            ).clamp_min(1)
        temb = temb + self.c_embedder(context.to(hidden_states.dtype))
        hidden_states = self.input_embedder(hidden_states)
        return self.individual_token_refiner(hidden_states, temb, attention_mask)


class HyNomasAttnProcessor:
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        rotary_emb,
        encoder_attention_mask=None,
        kv_cache=None,
        update_cache=False,
        action_text_offsets=None,
        action_token_windows=None,
        cache_valid_len=0,
        cache_txt=True,
        cache_mask=None,
    ):
        query = attn.img_attn_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.img_attn_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.img_attn_v(hidden_states).unflatten(-1, (attn.heads, -1))
        query = _apply_rotary_emb(attn.img_attn_q_norm(query).to(value.dtype), rotary_emb)
        key = _apply_rotary_emb(attn.img_attn_k_norm(key).to(value.dtype), rotary_emb)
        encoder_query = attn.txt_attn_q(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        encoder_key = attn.txt_attn_k(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        encoder_value = attn.txt_attn_v(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        encoder_query = attn.txt_attn_q_norm(encoder_query).to(encoder_value.dtype)
        encoder_key = attn.txt_attn_k_norm(encoder_key).to(encoder_value.dtype)
        group = get_sp_group() if attn.sp_size > 1 else None
        if group is not None:
            query, key, value = (all_to_all_4D(x, group=group.ulysses_group) for x in (query, key, value))
            heads = attn.heads // attn.sp_size
            start = group.rank_in_group * heads
            encoder_query, encoder_key, encoder_value = (
                x[:, :, start : start + heads] for x in (encoder_query, encoder_key, encoder_value)
            )
        batch_size, video_length = query.shape[:2]
        text_length = encoder_hidden_states.shape[1]
        text_mask = encoder_attention_mask
        if text_mask is None:
            text_mask = torch.ones(batch_size, text_length, dtype=torch.bool, device=query.device)
        video_mask = torch.ones(batch_size, video_length, dtype=torch.bool, device=query.device)
        query_mask = torch.cat((video_mask, text_mask.bool()), dim=1)
        write_mask = query_mask if cache_txt else video_mask
        history_length = cache_valid_len
        include_text = cache_txt or update_cache
        current_key = torch.cat((key, encoder_key), dim=1) if include_text else key
        current_value = torch.cat((value, encoder_value), dim=1) if include_text else value
        current_mask = query_mask if include_text else video_mask
        if update_cache:
            stop = history_length + write_mask.shape[1]
            kv_cache["k"][:, history_length:stop].copy_(current_key[:, : write_mask.shape[1]])
            kv_cache["v"][:, history_length:stop].copy_(current_value[:, : write_mask.shape[1]])
            cache_mask[:, history_length:stop].copy_(write_mask)
        key = torch.cat((kv_cache["k"][:, :history_length], current_key), dim=1)
        value = torch.cat((kv_cache["v"][:, :history_length], current_value), dim=1)
        key_mask = torch.cat((cache_mask[:, :history_length], current_mask), dim=1)
        query = torch.cat((query, encoder_query), dim=1)
        output = self._compute_attention(
            attn.attention,
            query,
            key,
            value,
            key_mask,
            text_mask,
            history_length,
            video_length,
            text_length,
            text_mask,
            action_text_offsets if cache_txt else None,
            action_token_windows,
        )
        output = output * query_mask[:, :, None, None].to(output.dtype)
        image_output, text_output = output.split((video_length, text_length), dim=1)
        if group is not None:
            image_output = all_to_all_4D(
                image_output.contiguous(), scatter_idx=1, gather_idx=2, group=group.ulysses_group
            )
            text_output = group.all_gather(text_output.contiguous(), dim=2, group=group.ulysses_group)
        return attn.img_attn_proj(image_output.flatten(-2)), attn.txt_attn_proj(text_output.flatten(-2))

    def _compute_attention(
        self,
        attention,
        query,
        key,
        value,
        key_mask,
        text_mask,
        history_length,
        video_length,
        text_length,
        encoder_attention_mask,
        action_text_offsets,
        action_token_windows,
    ):
        if action_text_offsets is None:
            attention_mask = (
                key_mask[:, None, None, :] if encoder_attention_mask is not None or history_length else None
            )
            output = _attend(attention, query, key, value, attention_mask)
        else:
            # Each query group shares a visible key set; no quadratic dense mask is needed.
            outputs = []
            for batch_index, offsets in enumerate(action_text_offsets):
                sample_output = torch.zeros_like(query[batch_index : batch_index + 1])
                sample_text_mask = text_mask[batch_index].bool()
                text_positions = sample_text_mask.cumsum(dim=0) - 1
                valid_text = sample_text_mask.sum()
                if offsets is None:
                    groups = [(0, video_length + text_length, None)]
                else:
                    groups = list(action_token_windows)
                    groups.append((video_length, video_length + min(offsets[0], text_length), -1))
                    groups.extend(
                        (video_length + min(start, text_length), video_length + min(end, text_length), index)
                        for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:]))
                    )
                for start, end, action_index in groups:
                    if end <= start:
                        continue
                    visible_text = sample_text_mask
                    if offsets is not None:
                        visible_text = text_positions < offsets[0]
                        if action_index >= 0:
                            visible_text = visible_text | (
                                (text_positions >= offsets[action_index]) & (text_positions < offsets[action_index + 1])
                            )
                        visible_text = visible_text & sample_text_mask
                    attention_mask = torch.cat((key_mask[batch_index, : history_length + video_length], visible_text))[
                        None, None, None, :
                    ]
                    group_output = _attend(
                        attention,
                        query[batch_index : batch_index + 1, start:end],
                        key[batch_index : batch_index + 1],
                        value[batch_index : batch_index + 1],
                        attention_mask,
                    )
                    if offsets is not None:
                        query_mask = torch.arange(start, end, device=query.device) < video_length + valid_text
                        group_output = group_output * query_mask[None, :, None, None].to(group_output.dtype)
                    sample_output[:, start:end] = group_output
                outputs.append(sample_output)
            output = torch.cat(outputs, dim=0)
        return output


class HyNomasAttention(nn.Module):
    _default_processor_cls = HyNomasAttnProcessor

    def __init__(self, hidden_size, heads):
        super().__init__()
        self.heads = heads
        self.sp_size = 1
        self.attention = _attention(heads, hidden_size // heads, "hy_nomas.joint")
        for prefix in ("img", "txt"):
            for projection in ("q", "k", "v", "proj"):
                setattr(self, f"{prefix}_attn_{projection}", nn.Linear(hidden_size, hidden_size))
            setattr(self, f"{prefix}_attn_q_norm", RMSNorm(hidden_size // heads, eps=1e-6))
            setattr(self, f"{prefix}_attn_k_norm", RMSNorm(hidden_size // heads, eps=1e-6))
        self.processor = self._default_processor_cls()

    def forward(self, hidden_states, encoder_hidden_states, rotary_emb, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return self.processor(self, hidden_states, encoder_hidden_states, rotary_emb, **kwargs)


class HyNomasModulation(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size * 6)

    def forward(self, hidden_states) -> torch.Tensor:
        return self.linear(F.silu(hidden_states))


class HyNomasTransformerBlock(nn.Module):
    def __init__(self, hidden_size, heads, mlp_ratio):
        super().__init__()
        self.img_mod = HyNomasModulation(hidden_size)
        self.img_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False, keep_fp32_output=True)
        self.img_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False, keep_fp32_output=True)
        self.img_mlp = FeedForward(hidden_size, inner_dim=int(hidden_size * mlp_ratio))
        self.txt_mod = HyNomasModulation(hidden_size)
        self.txt_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False, keep_fp32_output=True)
        self.txt_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False, keep_fp32_output=True)
        self.txt_mlp = FeedForward(hidden_size, inner_dim=int(hidden_size * mlp_ratio))
        self.attn = HyNomasAttention(hidden_size, heads)

    def forward(
        self, hidden_states, encoder_hidden_states, temb, rotary_emb, attention_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate, mlp_shift, mlp_scale, mlp_gate = self.img_mod(temb).chunk(6, dim=-1)
        text_shift, text_scale, text_gate, text_mlp_shift, text_mlp_scale, text_mlp_gate = self.txt_mod(temb).chunk(
            6, dim=-1
        )
        norm_hidden_states = self.img_norm1(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        norm_encoder_hidden_states = (
            self.txt_norm1(encoder_hidden_states) * (1 + text_scale[:, None]) + text_shift[:, None]
        )
        attention_output, encoder_attention_output = self.attn(
            norm_hidden_states.to(hidden_states.dtype),
            norm_encoder_hidden_states.to(encoder_hidden_states.dtype),
            rotary_emb,
            **attention_kwargs,
        )
        hidden_states = hidden_states + attention_output * gate[:, None]
        hidden_states = (
            hidden_states
            + self.img_mlp(
                (self.img_norm2(hidden_states) * (1 + mlp_scale[:, None]) + mlp_shift[:, None]).to(hidden_states.dtype)
            )
            * mlp_gate[:, None]
        )
        encoder_hidden_states = encoder_hidden_states + encoder_attention_output * text_gate[:, None]
        encoder_hidden_states = (
            encoder_hidden_states
            + self.txt_mlp(
                (self.txt_norm2(encoder_hidden_states) * (1 + text_mlp_scale[:, None]) + text_mlp_shift[:, None]).to(
                    encoder_hidden_states.dtype
                )
            )
            * text_mlp_gate[:, None]
        )
        return hidden_states, encoder_hidden_states


class HyNomasCameraAttnProcessor:
    def __call__(
        self,
        attn,
        hidden_states,
        camera_hidden_states,
        rotary_emb,
        camera_attention_mask=None,
        *,
        tokens_per_frame,
        token_offset=0,
        chunk_size=1,
    ):
        query = attn.img_attn_q(hidden_states).unflatten(-1, (attn.heads, -1))
        keyboard = attn.keyboard_embed(camera_hidden_states)
        key, value = attn.keyboard_attn_kv(keyboard).unflatten(-1, (2, attn.heads, -1)).unbind(-3)
        query = attn.key_attn_q_norm(query).to(value.dtype)
        key = attn.key_attn_k_norm(key).to(value.dtype)
        frame_ids = torch.arange(token_offset, token_offset + query.shape[1], device=query.device) // tokens_per_frame
        query = _apply_rotary_emb(query, tuple(x[frame_ids] for x in rotary_emb))
        key = _apply_rotary_emb(key, rotary_emb)
        frames = key.shape[1]
        outputs = []
        for start in range(0, frames, chunk_size or frames):
            end = min(start + (chunk_size or frames), frames)
            left = max(start * tokens_per_frame - token_offset, 0)
            right = min(end * tokens_per_frame - token_offset, query.shape[1])
            if left < right:
                outputs.append(_attend(attn.attention, query[:, left:right], key[:, start:end], value[:, start:end]))
        output = attn.proj_keyboard(torch.cat(outputs, dim=1).flatten(-2))
        if camera_attention_mask is not None:
            output = output * camera_attention_mask.reshape(-1, 1, 1).to(output.dtype)
        return hidden_states + output


class HyNomasCameraAttention(nn.Module):
    _default_processor_cls = HyNomasCameraAttnProcessor

    def __init__(self, hidden_size, camera_embed_dim, camera_attention_dim, heads):
        super().__init__()
        self.heads = heads
        self.camera_attention_dim = camera_attention_dim
        self.attention = _attention(heads, camera_attention_dim // heads, "hy_nomas.camera")
        self.keyboard_embed = nn.Sequential(
            nn.Linear(64, camera_embed_dim), nn.SiLU(), nn.Linear(camera_embed_dim, camera_embed_dim)
        )
        self.key_attn_q_norm = RMSNorm(camera_attention_dim // heads, eps=1e-6)
        self.key_attn_k_norm = RMSNorm(camera_attention_dim // heads, eps=1e-6)
        self.img_attn_q = nn.Linear(hidden_size, camera_attention_dim, bias=False)
        self.keyboard_attn_kv = nn.Linear(camera_embed_dim, 2 * camera_attention_dim, bias=False)
        self.proj_keyboard = nn.Linear(camera_attention_dim, hidden_size, bias=False)
        self.processor = self._default_processor_cls()

    def forward(
        self, hidden_states, camera_hidden_states, rotary_emb, camera_attention_mask=None, **kwargs
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, camera_hidden_states, rotary_emb, camera_attention_mask, **kwargs)


class HyNomasFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False, keep_fp32_output=True)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, hidden_states, temb) -> torch.Tensor:
        dtype = hidden_states.dtype
        shift, scale = self.adaLN_modulation(temb).chunk(2, dim=-1)
        hidden_states = self.norm_final(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.linear(hidden_states.to(dtype))


class HyNomasTransformer3DModel(nn.Module):
    r"""The Hy-Nomas segment autoregressive video transformer with frame-local camera conditioning.

    Args:
        in_channels (`int`, defaults to `32`): Number of latent channels.
        out_channels (`int`, defaults to `32`): Number of output channels.
        num_layers (`int`, defaults to `54`): Number of double stream blocks.
        num_attention_heads (`int`, defaults to `16`): Number of attention heads.
        attention_head_dim (`int`, defaults to `128`): Dimension of an attention head.
        text_embed_dim (`int`, defaults to `3584`): Dimension of text encoder features.
        mlp_ratio (`float`, defaults to `4.0`): MLP expansion ratio.
        rope_axes_dim (`tuple[int]`, defaults to `(16, 56, 56)`): Temporal and spatial RoPE dimensions.
        rope_theta (`float`, defaults to `256.0`): RoPE frequency base.
        camera_embed_dim (`int`, defaults to `128`): Camera embedding dimension.
        camera_attention_dim (`int`, defaults to `1024`): Camera attention projection dimension.
        camera_num_attention_heads (`int`, defaults to `16`): Number of camera attention heads.
        camera_rope_axes_dim (`tuple[int]`, defaults to `(8, 28, 28)`): Camera RoPE dimensions.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "_mlp.fc1.": "_mlp.net.0.proj.",
            "_mlp.fc2.": "_mlp.net.2.",
        }
    )

    _no_split_modules = [
        "HyNomasTransformerBlock",
        "HyNomasTokenRefiner",
        "HyNomasCameraAttention",
        "HyNomasFinalLayer",
    ]
    _repeated_blocks = ["HyNomasTransformerBlock"]
    _skip_layerwise_casting_patterns = ["img_in", "txt_in", "time_in", "norm"]
    _layerwise_offload_blocks_attrs = ["double_blocks", "action_modules"]

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        num_layers: int = 54,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        text_embed_dim: int = 3584,
        mlp_ratio: float = 4.0,
        rope_axes_dim: tuple[int, int, int] = (16, 56, 56),
        rope_theta: float = 256.0,
        camera_embed_dim: int = 128,
        camera_attention_dim: int = 1024,
        camera_num_attention_heads: int = 16,
        camera_rope_axes_dim: tuple[int, int, int] = (8, 28, 28),
        sp_size: int = 1,
        camera_chunk_size: int = 1,
    ):
        super().__init__()
        self.config = HyNomasTransformerConfig(
            in_channels,
            out_channels,
            num_layers,
            num_attention_heads,
            attention_head_dim,
            text_embed_dim,
            mlp_ratio,
            tuple(rope_axes_dim),
            rope_theta,
            camera_embed_dim,
            camera_attention_dim,
            camera_num_attention_heads,
            tuple(camera_rope_axes_dim),
        )
        self.sp_size = sp_size
        self.camera_chunk_size = camera_chunk_size
        self.patch_size = (1, 1, 1)
        self.heads_num = num_attention_heads
        self.hidden_size = num_attention_heads * attention_head_dim
        self.camera_injection_mode = "extrinsics_attn"
        if sp_size < 1 or num_attention_heads % sp_size:
            raise ValueError("Ulysses degree must divide the attention heads.")
        if sum(rope_axes_dim) != attention_head_dim or any(dim % 2 for dim in rope_axes_dim):
            raise ValueError("Even rope_axes_dim entries must sum to attention_head_dim.")
        if camera_attention_dim != camera_num_attention_heads * sum(camera_rope_axes_dim):
            raise ValueError("camera_attention_dim must match camera_num_attention_heads * sum(camera_rope_axes_dim).")
        hidden_size = num_attention_heads * attention_head_dim
        self.img_in = HunyuanVideo15PatchEmbed((1, 1, 1), in_channels, hidden_size)
        self.txt_in = HyNomasTokenRefiner(text_embed_dim, hidden_size, num_attention_heads)
        self.time_in = HyNomasTimestepEmbedder(hidden_size)
        self.rotary_embed = HyNomasRotaryPosEmbed(rope_axes_dim, rope_theta)
        self.camera_rotary_embed = HyNomasRotaryPosEmbed(camera_rope_axes_dim, rope_theta)
        self.double_blocks = nn.ModuleList(
            [HyNomasTransformerBlock(hidden_size, num_attention_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.action_modules = nn.ModuleList(
            [
                HyNomasCameraAttention(hidden_size, camera_embed_dim, camera_attention_dim, camera_num_attention_heads)
                for _ in range(num_layers)
            ]
        )
        for block in self.double_blocks:
            block.attn.sp_size = sp_size
            if sp_size > 1:
                block.attn.attention = _attention(num_attention_heads // sp_size, attention_head_dim, "hy_nomas.joint")
        self.extrinsics_embedding_camctrl = nn.Linear(64, 64)
        self.c2ws_hidden_states_layer1 = nn.Linear(64, 64)
        self.c2ws_hidden_states_layer2 = nn.Linear(64, 64)
        self.final_layer = HyNomasFinalLayer(hidden_size, out_channels)

    @property
    def dtype(self) -> torch.dtype:
        return self.img_in.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.img_in.proj.weight.device

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the converted checkpoint without silently dropping or inventing parameters."""
        parameters = dict(self.named_parameters())
        checkpoint_mapper = WeightsMapper(
            orig_to_new_substr={new: old for old, new in self.hf_to_vllm_mapper.orig_to_new_substr.items()}
        )
        checkpoint_names = set(checkpoint_mapper.apply_list(list(parameters)))
        loaded: set[str] = set()
        seen: set[str] = set()
        for name, tensor in weights:
            if name not in checkpoint_names:
                raise ValueError(f"Unexpected Hy-Nomas transformer weight: {name}")
            if name in seen:
                raise ValueError(f"Duplicate Hy-Nomas transformer weight: {name}")
            destination = self.hf_to_vllm_mapper.apply_list([name])[0]
            default_weight_loader(parameters[destination], tensor)
            seen.add(name)
            loaded.add(destination)
        missing = checkpoint_names - seen
        if missing:
            raise ValueError(f"Missing Hy-Nomas transformer weights: {sorted(missing)}")
        return loaded

    def _preprocess_camera(self, norm_extrinsics, norm_intrinsics, action_onehot, oh, ow, img_dtype, device):
        poses = norm_extrinsics.to(device=device, dtype=img_dtype)
        padding = (-poses.shape[1]) % 4
        if padding:
            poses = torch.cat((poses[:, :1].repeat(1, padding, 1, 1), poses), dim=1)
        camera = self.extrinsics_embedding_camctrl(poses.reshape(poses.shape[0], -1, 64))
        return camera + self.c2ws_hidden_states_layer2(F.silu(self.c2ws_hidden_states_layer1(camera)))

    def forward_with_cache(
        self,
        hidden_states,
        timestep,
        text_states,
        text_mask,
        kv_cache_list,
        cache_valid_len=0,
        update_cache=False,
        global_frame_counter=0,
        guidance=None,
        cache_txt=True,
        cache_mask_list=None,
        camera_embedding=None,
        camera_has_mask=None,
        intra_seg_txt_offsets=None,
        intra_seg_vid_windows=None,
        camera_rope_time_offset=0,
        rotary_emb=None,
    ):
        batch, _, frames, height, width = hidden_states.shape
        video_tokens = frames * height * width
        written = video_tokens + (text_states.shape[1] if cache_txt else 0)
        if (
            len(kv_cache_list) != len(self.double_blocks)
            or cache_mask_list is None
            or len(cache_mask_list) != len(self.double_blocks)
        ):
            raise ValueError("Provide one caller-owned KV view and mask per transformer layer.")
        if video_tokens % self.sp_size or cache_valid_len < 0:
            raise ValueError("Invalid video token count or cache prefix for Ulysses execution.")
        expected = (batch, self.heads_num // self.sp_size, self.config.attention_head_dim)
        for cache, mask in zip(kv_cache_list, cache_mask_list):
            key, value = cache["k"], cache["v"]
            if key.ndim != 4 or key.shape != value.shape or (key.shape[0], *key.shape[2:]) != expected:
                raise ValueError("KV cache must use full tokens and rank-local heads.")
            if (
                key.shape[1] < cache_valid_len + written
                or mask.shape[0] != batch
                or mask.shape[1] < cache_valid_len + written
            ):
                raise ValueError("KV cache view does not have enough space for this chunk.")
        if rotary_emb is None:
            rotary_emb = self.rotary_embed(frames, height, width, hidden_states.device, global_frame_counter)
        image = self.img_in(hidden_states).contiguous()
        group = get_sp_group() if self.sp_size > 1 else None
        token_offset = 0
        if group is not None:
            rank = group.rank_in_group
            token_offset = rank * (video_tokens // self.sp_size)
            image = image.chunk(self.sp_size, dim=1)[rank].contiguous()
            rotary_emb = tuple(x.chunk(self.sp_size, dim=0)[rank] for x in rotary_emb)
        temb = self.time_in(timestep, image.dtype)
        text = self.txt_in(text_states, timestep, text_mask)
        camera_rope = None
        if camera_embedding is not None:
            camera_rope = self.camera_rotary_embed(frames, 1, 1, hidden_states.device, camera_rope_time_offset)
        windows = (
            None
            if intra_seg_vid_windows is None
            else [
                (start * height * width, end * height * width, action) for start, end, action in intra_seg_vid_windows
            ]
        )
        for index, block in enumerate(self.double_blocks):
            image, text = block(
                image,
                text,
                temb,
                rotary_emb,
                attention_kwargs=dict(
                    encoder_attention_mask=text_mask,
                    kv_cache=kv_cache_list[index],
                    cache_valid_len=cache_valid_len,
                    update_cache=update_cache,
                    cache_txt=cache_txt,
                    cache_mask=cache_mask_list[index],
                    action_text_offsets=intra_seg_txt_offsets,
                    action_token_windows=windows,
                ),
            )
            if camera_embedding is not None:
                image = self.action_modules[index](
                    image,
                    camera_embedding,
                    camera_rope,
                    camera_has_mask,
                    tokens_per_frame=height * width,
                    token_offset=token_offset,
                    chunk_size=self.camera_chunk_size,
                )
        if group is not None:
            image = group.all_gather(image.contiguous(), dim=1, group=group.ulysses_group)
        image = self.final_layer(image, temb)
        image = image.transpose(1, 2).reshape(batch, self.config.out_channels, frames, height, width)
        return image, written if update_cache else 0
