# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SolarWM-H3 causal student on top of the MiniMax-H3 DiT.

Parameters, checkpoint names and every non-attention layer are the MiniMax-H3
DiT's. Only the block attention changes: each forward runs one five-latent
chunk against a fixed prefix and a six-chunk window of cached raw keys and
values, with window-local RoPE and camera PRoPE applied per segment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_omni.diffusion.attention.ops.minimax_h3_modulation import (
    indexed_gate,
    indexed_gate_rms_norm_scale_shift,
    rms_norm_indexed_scale_shift,
)
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3Attention,
    MiniMaxH3DiTBlock,
    MiniMaxH3DiTModel,
)

from .camera import CameraProjection, apply_camera_projection
from .layout import (
    SOLARWM_H3_AUDIO_ROW_WIDTH,
    SOLARWM_H3_CHUNK_ROWS,
    SOLARWM_H3_HISTORY_CHUNKS,
    SOLARWM_H3_VIDEO_ROW_WIDTH,
    ChunkWindow,
)


class SolarWMKVCache:
    """Raw (post-norm, pre-RoPE) keys and values of the last five committed chunks, per layer.

    Ring slots hold one chunk each. The cache lives on ``device``; pinned host
    memory is the default so the 33B weights and a full window fit next to each
    other on one accelerator, and each layer's history is copied in on demand.
    """

    def __init__(self, *, num_layers: int, device: torch.device) -> None:
        self.num_layers = num_layers
        self.device = device
        self._keys: list[torch.Tensor | None] = [None] * num_layers
        self._values: list[torch.Tensor | None] = [None] * num_layers

    def _allocate(self, layer: int, template: torch.Tensor) -> None:
        shape = (SOLARWM_H3_HISTORY_CHUNKS, *template.shape)
        pin = self.device.type == "cpu" and torch.accelerator.is_available()
        self._keys[layer] = torch.empty(shape, dtype=template.dtype, device=self.device, pin_memory=pin)
        self._values[layer] = torch.empty(shape, dtype=template.dtype, device=self.device, pin_memory=pin)

    def commit(self, layer: int, chunk_index: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        if self._keys[layer] is None:
            self._allocate(layer, keys)
        slot = chunk_index % SOLARWM_H3_HISTORY_CHUNKS
        self._keys[layer][slot].copy_(keys, non_blocking=True)
        self._values[layer][slot].copy_(values, non_blocking=True)

    def history(
        self, layer: int, window: ChunkWindow, device: torch.device
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return the window's cached keys and values, one tensor per chunk, oldest first."""
        keys = []
        values = []
        for chunk_index in window.history_chunks:
            slot = chunk_index % SOLARWM_H3_HISTORY_CHUNKS
            keys.append(self._keys[layer][slot].to(device, non_blocking=True))
            values.append(self._values[layer][slot].to(device, non_blocking=True))
        return keys, values

    def clear(self) -> None:
        self._keys = [None] * self.num_layers
        self._values = [None] * self.num_layers


@dataclass
class SolarWMAttentionPlan:
    """Everything one chunk forward needs beyond the packed hidden states.

    ``row_*`` tensors cover the packed sequence ``[prefix | current chunk]``;
    ``window_*`` tensors cover the chunk segment's keys ``[history | current]``.
    RoPE tables are the pre-cos/sin frequencies of the native rope module.
    """

    window: ChunkWindow
    condition_len: int
    prefix_len: int
    row_freqs: torch.Tensor
    row_projection: CameraProjection
    window_freqs: torch.Tensor
    window_key_value: torch.Tensor
    cache: SolarWMKVCache
    commit: bool


class SolarWMH3Attention(MiniMaxH3Attention):
    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        query_freqs: torch.Tensor,
        key_freqs: torch.Tensor,
        projection: CameraProjection,
        key_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Native RoPE, then camera PRoPE, then full attention of ``q`` over ``k``/``v``."""
        q = apply_camera_projection(self._apply_rope(q, query_freqs), projection.query)
        k = apply_camera_projection(self._apply_rope(k, key_freqs), key_matrix)
        v = apply_camera_projection(v, key_matrix)
        attended = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).squeeze(0)
        return apply_camera_projection(attended, projection.output)

    def forward_windowed(self, x: torch.Tensor, plan: SolarWMAttentionPlan, layer_index: int) -> torch.Tensor:
        """x: ``[S, hidden]`` packed rows ``[text | anchor | audio | chunk]`` -> ``[S, hidden]``."""
        total = x.shape[0]
        qkv, _ = self.qkv_proj(x)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = self.q_norm(q.view(total, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(total, self.num_kv_heads, self.head_dim))
        v = v.view(total, self.num_kv_heads, self.head_dim)

        condition = plan.condition_len
        prefix = plan.prefix_len
        rows = plan.row_projection
        out = torch.empty_like(q)
        # Text and anchor rows see only each other.
        out[:condition] = self._attend(
            q[:condition],
            k[:condition],
            v[:condition],
            query_freqs=plan.row_freqs[:condition],
            key_freqs=plan.row_freqs[:condition],
            projection=rows.rows(0, condition),
            key_matrix=rows.key_value[:condition],
        )
        # Audio rows see the whole prefix.
        out[condition:prefix] = self._attend(
            q[condition:prefix],
            k[:prefix],
            v[:prefix],
            query_freqs=plan.row_freqs[condition:prefix],
            key_freqs=plan.row_freqs[:prefix],
            projection=rows.rows(condition, prefix),
            key_matrix=rows.key_value[:prefix],
        )
        # The current chunk sees the prefix, the cached window and itself.
        history_k, history_v = plan.cache.history(layer_index, plan.window, x.device)
        out[prefix:] = self._attend(
            q[prefix:],
            torch.cat((k[:prefix], *history_k, k[prefix:])),
            torch.cat((v[:prefix], *history_v, v[prefix:])),
            query_freqs=plan.row_freqs[prefix:],
            key_freqs=torch.cat((plan.row_freqs[:prefix], plan.window_freqs)),
            projection=rows.rows(prefix, total),
            key_matrix=torch.cat((rows.key_value[:prefix], plan.window_key_value)),
        )
        if plan.commit:
            plan.cache.commit(layer_index, plan.window.chunk_index, k[prefix:], v[prefix:])

        out, _ = self.out_proj(out.reshape(total, self.num_heads * self.head_dim))
        return out


class SolarWMH3DiTBlock(MiniMaxH3DiTBlock):
    _attention_cls = SolarWMH3Attention

    def forward_windowed(
        self,
        x: torch.Tensor,
        *,
        t_emb: torch.Tensor,
        combined_indices: torch.Tensor,
        plan: SolarWMAttentionPlan,
        layer_index: int,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        residual = x
        h = rms_norm_indexed_scale_shift(
            x,
            self.norm1.weight,
            shift_msa,
            scale_msa,
            combined_indices,
            self.norm1.variance_epsilon,
        )
        h = self.attn.forward_windowed(h, plan, layer_index)
        x, h = indexed_gate_rms_norm_scale_shift(
            residual,
            gate_msa,
            h,
            self.norm2.weight,
            shift_mlp,
            scale_mlp,
            combined_indices,
            self.norm2.variance_epsilon,
        )
        residual = x
        h = self.mlp(h)
        return indexed_gate(residual, gate_mlp, h, combined_indices)


@dataclass
class SolarWMChunkInputs:
    """Packed rows for one chunk forward; positions index the packed sequence."""

    prompt_embeds: torch.Tensor
    video_rows: torch.Tensor
    audio_rows: torch.Tensor
    text_pos: torch.Tensor
    img_pos: torch.Tensor
    audio_pos: torch.Tensor
    token_tags: torch.Tensor
    unique_timesteps: torch.Tensor
    inverse_indices: torch.Tensor


class SolarWMH3DiTModel(MiniMaxH3DiTModel):
    _block_cls = SolarWMH3DiTBlock

    def rope_frequencies(self, position_ids: torch.Tensor) -> torch.Tensor:
        """``[S, 3]`` float64 positions -> ``[S, 96]`` float32 rotary frequencies."""
        return self.rope(position_ids.to(self.rope.inv_freq.device).unsqueeze(0))

    def forward_chunk(self, inputs: SolarWMChunkInputs, plan: SolarWMAttentionPlan) -> torch.Tensor:
        """Return the fp32 velocity rows ``[chunk_rows, 96]`` of the current chunk."""
        device = inputs.token_tags.device
        seq_len = int(inputs.token_tags.shape[0])
        text_len = int(inputs.text_pos.shape[0])
        x = torch.zeros(1, seq_len, SOLARWM_H3_VIDEO_ROW_WIDTH, dtype=torch.float32, device=device)
        x[0].index_copy_(0, inputs.img_pos, inputs.video_rows)
        audio_x = torch.zeros(1, seq_len, SOLARWM_H3_AUDIO_ROW_WIDTH, dtype=torch.float32, device=device)
        audio_x[0].index_copy_(0, inputs.audio_pos, inputs.audio_rows)

        hidden, t_emb = self._embed(
            x=x,
            audio_x=audio_x,
            text_embeddings_selected=inputs.prompt_embeds,
            unique_timesteps=inputs.unique_timesteps,
            img_pos=inputs.img_pos,
            audio_pos=inputs.audio_pos,
            text_pos=inputs.text_pos,
            refiner_cu_seqlens=torch.tensor([0, text_len, text_len], dtype=torch.int32, device=device),
            refiner_max_seqlen=text_len,
            seq_len=seq_len,
            device=device,
            local_span=(0, seq_len),
        )
        combined_indices = inputs.inverse_indices * MINIMAX_H3_ADALN_MODALITY_NUM + inputs.token_tags
        for layer_index, block in enumerate(self.blocks):
            hidden = block.forward_windowed(
                hidden,
                t_emb=t_emb,
                combined_indices=combined_indices,
                plan=plan,
                layer_index=layer_index,
            )
        video_logits, _ = self.final_layer(hidden, t_emb=t_emb, inverse_indices=inputs.inverse_indices)
        return video_logits[seq_len - SOLARWM_H3_CHUNK_ROWS :]


__all__ = [
    "SolarWMAttentionPlan",
    "SolarWMChunkInputs",
    "SolarWMH3Attention",
    "SolarWMH3DiTBlock",
    "SolarWMH3DiTModel",
    "SolarWMKVCache",
]
