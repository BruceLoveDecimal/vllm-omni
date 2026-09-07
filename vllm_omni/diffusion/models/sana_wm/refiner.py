# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunk-causal LTX-2 refiner (SANA-WM Stage-2) for the streaming pipeline.

The SANA-WM streaming release ships a standard LTX-2 video transformer plus
text connectors that were trained to refine ``block_size`` clean Stage-1
latent frames at a time with a sliding-window KV cache over the frames refined
before (NVlabs ``RefinerChunkRunner``, recipe ``distilled-3step +
source-sink-1``). Per block:

1. ``x_t = (1 - sigma_0) * clean + sigma_0 * eps`` with one ``eps`` draw per
   block from the request's refiner generator.
2. Three deterministic Euler steps on the distilled sigmas
   ``0.909375, 0.725, 0.421875 -> 0``. Every self-attention layer sees
   ``[sink K/V | history K/V | current K/V]``: the sink is the raw Stage-1
   conditioning latent whose *pre*-RoPE K/V were captured once at ``sigma = 0``
   and are re-rotated to sit immediately before the history window
   (``rf_shifted_sink``); the history holds the *post*-RoPE K/V of the refined
   blocks at their absolute frame positions, trimmed to
   ``kv_max_frames - sink_frames`` frames.
3. One clean forward at ``sigma = 0`` on the refined block under the same
   prefix captures its post-RoPE K/V for the next block.

The forward is a video-only pass through vLLM-Omni's native
:class:`~vllm_omni.diffusion.models.ltx2.ltx2_transformer.LTX2VideoTransformer3DModel`
(the audio stream and the audio-video cross-attention are skipped, as in
NVlabs ``DiffusersLTX2Refiner``). The K/V prefix is passed explicitly per
layer instead of through module attributes, so the transformer module itself
carries no request state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import nn

from vllm_omni.diffusion.models.ltx2.ltx2_guidance import euler_step_from_velocity
from vllm_omni.diffusion.models.ltx2.ltx2_latents import pack_latents, unpack_latents
from vllm_omni.diffusion.models.ltx2.ltx2_transformer import (
    LTX2AudioVideoAttnProcessor,
    apply_interleaved_rotary_emb,
    apply_split_rotary_emb,
)

__all__ = [
    "SANA_WM_REFINER_BLOCK_SIZE",
    "SANA_WM_REFINER_KV_MAX_FRAMES",
    "SANA_WM_REFINER_SIGMAS",
    "SANA_WM_REFINER_SINK_FRAMES",
    "SanaWmRefinerKvCache",
    "SanaWmRefinerRunner",
    "SanaWmRefinerSchedule",
    "refiner_rotary_emb",
]

# NVlabs ``STAGE_2_DISTILLED_SIGMA_VALUES`` (``diffusers_ltx2_refiner.py``).
SANA_WM_REFINER_SIGMAS: tuple[float, ...] = (0.909375, 0.725, 0.421875, 0.0)
# ``--sink_size`` / ``--refiner_block_size`` / ``--refiner_kv_max_frames``
# defaults of ``inference_sana_wm_streaming.py``. The window must hold at
# least ``sink + block`` frames or every block loses its cross-chunk context.
SANA_WM_REFINER_SINK_FRAMES = 1
SANA_WM_REFINER_BLOCK_SIZE = 3
SANA_WM_REFINER_KV_MAX_FRAMES = 11

KvPair = tuple[torch.Tensor, torch.Tensor]
CaptureMode = Literal["pre_rope", "post_rope"]


@dataclass(frozen=True)
class SanaWmRefinerSchedule:
    """Descending sigma schedule ending at ``0`` (``len - 1`` Euler steps)."""

    sigmas: tuple[float, ...] = SANA_WM_REFINER_SIGMAS

    def __post_init__(self) -> None:
        sigmas = tuple(float(value) for value in self.sigmas)
        object.__setattr__(self, "sigmas", sigmas)
        if len(sigmas) < 2:
            raise ValueError(f"Sana-WM refiner schedule needs >= 2 sigmas, got {sigmas}.")
        if sigmas[-1] != 0.0:
            raise ValueError(f"Sana-WM refiner schedule must end at sigma = 0, got {sigmas}.")
        if not 0.0 < sigmas[0] <= 1.0:
            raise ValueError(f"Sana-WM refiner schedule must start in (0, 1], got {sigmas}.")
        if any(later >= earlier for earlier, later in zip(sigmas, sigmas[1:])):
            raise ValueError(f"Sana-WM refiner sigmas must be strictly decreasing, got {sigmas}.")

    @property
    def num_steps(self) -> int:
        return len(self.sigmas) - 1

    @property
    def sigma_max(self) -> float:
        return self.sigmas[0]

    def pairs(self) -> list[tuple[float, float]]:
        return list(zip(self.sigmas[:-1], self.sigmas[1:]))

    def check_num_steps(self, num_steps: int | None) -> None:
        if num_steps is not None and int(num_steps) != self.num_steps:
            raise ValueError(
                f"Sana-WM refiner runs the distilled {self.num_steps}-step schedule {self.sigmas}; "
                f"refiner_steps must be {self.num_steps} or omitted, got {num_steps}."
            )


@dataclass
class SanaWmRefinerKvCache:
    """Per-request rolling K/V state of the chunk-causal refiner.

    ``sink_kv_pre`` holds one pre-RoPE ``(K, V)`` per layer for the
    ``sink_frames`` raw conditioning latents (``[B, N_sink, C]`` in the
    attention layout of the layer, TP-local). ``history_kv_post`` holds the
    post-RoPE ``(K, V)`` of the most recent refined frames, at most
    ``kv_max_frames - sink_frames`` frames (``history_frames`` counts them).
    """

    num_layers: int
    tokens_per_frame: int
    sink_frames: int = SANA_WM_REFINER_SINK_FRAMES
    block_size: int = SANA_WM_REFINER_BLOCK_SIZE
    kv_max_frames: int = SANA_WM_REFINER_KV_MAX_FRAMES
    sink_kv_pre: list[KvPair] | None = None
    history_kv_post: list[KvPair | None] = field(default_factory=list)
    history_frames: int = 0
    blocks_refined: int = 0

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError(f"Sana-WM refiner cache needs num_layers > 0, got {self.num_layers}.")
        if self.tokens_per_frame <= 0:
            raise ValueError(f"Sana-WM refiner cache needs tokens_per_frame > 0, got {self.tokens_per_frame}.")
        if self.sink_frames < 0 or self.block_size <= 0:
            raise ValueError(
                f"Sana-WM refiner cache needs sink_frames >= 0 and block_size > 0, "
                f"got {self.sink_frames} / {self.block_size}."
            )
        if self.kv_max_frames < self.sink_frames + self.block_size:
            raise ValueError(
                f"Sana-WM refiner kv_max_frames ({self.kv_max_frames}) must be >= sink_frames + block_size "
                f"({self.sink_frames + self.block_size}); a smaller window drops every previous block."
            )
        if not self.history_kv_post:
            self.history_kv_post = [None] * self.num_layers

    @property
    def max_history_frames(self) -> int:
        return self.kv_max_frames - self.sink_frames

    def append_history(self, block_kv_post: Sequence[KvPair], active_frames: int) -> None:
        """Append one refined block's post-RoPE K/V and trim to the window."""
        if len(block_kv_post) != self.num_layers:
            raise ValueError(
                f"Sana-WM refiner capture has {len(block_kv_post)} layers, cache expects {self.num_layers}."
            )
        keep_tokens = self.max_history_frames * self.tokens_per_frame
        for layer_idx, (new_k, new_v) in enumerate(block_kv_post):
            old = self.history_kv_post[layer_idx]
            if old is None:
                k, v = new_k, new_v
            else:
                k = torch.cat([old[0], new_k], dim=1)
                v = torch.cat([old[1], new_v], dim=1)
            if k.shape[1] > keep_tokens:
                k, v = k[:, -keep_tokens:], v[:, -keep_tokens:]
            self.history_kv_post[layer_idx] = (k.contiguous(), v.contiguous())
        self.history_frames = min(self.history_frames + int(active_frames), self.max_history_frames)
        self.blocks_refined += 1

    def cached_frames(self) -> int:
        return (self.sink_frames if self.sink_kv_pre is not None else 0) + self.history_frames

    def nbytes(self) -> int:
        total = 0
        for pair in self.sink_kv_pre or ():
            total += pair[0].numel() * pair[0].element_size() + pair[1].numel() * pair[1].element_size()
        for pair in self.history_kv_post:
            if pair is not None:
                total += pair[0].numel() * pair[0].element_size() + pair[1].numel() * pair[1].element_size()
        return total

    def clear(self) -> None:
        self.sink_kv_pre = None
        self.history_kv_post = [None] * self.num_layers
        self.history_frames = 0


@dataclass(frozen=True)
class _LayerPrefix:
    """K/V prepended to one self-attention layer for the current forward."""

    sink_k_pre: torch.Tensor | None
    sink_v: torch.Tensor | None
    sink_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None
    history_k: torch.Tensor | None
    history_v: torch.Tensor | None


def refiner_rotary_emb(
    transformer: nn.Module,
    *,
    frame_positions: Sequence[int],
    height: int,
    width: int,
    batch_size: int,
    device: torch.device,
    fps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LTX-2 video RoPE for explicit absolute latent-frame positions.

    ``LTX2AudioVideoRotaryPosEmbed.prepare_video_coords`` assumes
    ``arange(num_frames)``; the sliding-window refiner needs each frame's
    absolute index in the clip (the history is cached post-RoPE, the sink is
    re-rotated to a shifted position). Mirrors NVlabs
    ``_build_rotary_emb_for_absolute_positions``.
    """
    rope = transformer.rope
    patch_size_t = int(rope.patch_size_t)
    patch_size = int(rope.patch_size)
    if len(frame_positions) % patch_size_t:
        raise ValueError(f"frame_positions ({len(frame_positions)}) must be a multiple of patch_size_t={patch_size_t}.")
    grid_f = torch.tensor(list(frame_positions), dtype=torch.float32, device=device)[::patch_size_t]
    grid_h = torch.arange(0, height, step=patch_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(0, width, step=patch_size, dtype=torch.float32, device=device)
    grid = torch.stack(torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij"), dim=0)  # [3, F, H, W]
    patch_delta = torch.tensor((patch_size_t, patch_size, patch_size), dtype=grid.dtype, device=device)
    patch_ends = grid + patch_delta.view(3, 1, 1, 1)
    latent_coords = torch.stack([grid, patch_ends], dim=-1).flatten(1, 3).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    scale = torch.tensor(rope.scale_factors, device=device, dtype=latent_coords.dtype)
    pixel_coords = latent_coords * scale.view(1, -1, 1, 1)
    pixel_coords[:, 0] = (pixel_coords[:, 0] + rope.causal_offset - rope.scale_factors[0]).clamp(min=0)
    pixel_coords[:, 0] = pixel_coords[:, 0] / float(fps)
    return rope(pixel_coords, device=device)


def _apply_rotary(attn: nn.Module, x: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    if attn.rope_type == "interleaved":
        return apply_interleaved_rotary_emb(x, rotary_emb)
    if attn.rope_type == "split":
        return apply_split_rotary_emb(x, rotary_emb, head_dim=attn.head_dim)
    raise ValueError(f"Unsupported LTX-2 RoPE type: {attn.rope_type}")


def _self_attention_with_prefix(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    prefix: _LayerPrefix | None,
    capture: CaptureMode | None,
    *,
    kv_only: bool = False,
) -> tuple[torch.Tensor | None, KvPair | None]:
    """Native LTX-2 self-attention over ``[prefix K/V | current K/V]``.

    Returns ``(attention output, captured (K, V))``. With ``kv_only`` only the
    projection / RoPE / capture runs (the last layer of a cache-write pass
    never needs its attention output).
    """
    gate_logits = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None
    query, key, value = LTX2AudioVideoAttnProcessor._project_qkv(
        attn=attn,
        hidden_states=hidden_states,
        encoder_hidden_states=hidden_states,
        is_self_attention=True,
    )
    query = attn.norm_q(query).to(dtype=value.dtype)
    key = attn.norm_k(key).to(dtype=value.dtype)

    captured: KvPair | None = None
    if capture == "pre_rope":
        captured = (key.detach().clone(), value.detach().clone())

    rotary_emb = LTX2AudioVideoAttnProcessor._slice_rope_for_tp(rotary_emb, attn)
    query = _apply_rotary(attn, query, rotary_emb).to(dtype=value.dtype)
    key = _apply_rotary(attn, key, rotary_emb).to(dtype=value.dtype)

    if capture == "post_rope":
        captured = (key.detach().clone(), value.detach().clone())
    if kv_only:
        return None, captured

    if prefix is not None:
        key_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        if prefix.sink_k_pre is not None and prefix.sink_v is not None and prefix.sink_k_pre.shape[1] > 0:
            if prefix.sink_rotary_emb is None:
                raise ValueError("Sana-WM refiner sink prefix requires its shifted RoPE.")
            sink_rotary_emb = LTX2AudioVideoAttnProcessor._slice_rope_for_tp(prefix.sink_rotary_emb, attn)
            key_parts.append(_apply_rotary(attn, prefix.sink_k_pre.to(key.dtype), sink_rotary_emb).to(key.dtype))
            value_parts.append(prefix.sink_v.to(value.dtype))
        if prefix.history_k is not None and prefix.history_v is not None and prefix.history_k.shape[1] > 0:
            key_parts.append(prefix.history_k.to(key.dtype))
            value_parts.append(prefix.history_v.to(value.dtype))
        if key_parts:
            key = torch.cat([*key_parts, key], dim=1)
            value = torch.cat([*value_parts, value], dim=1)

    query = query.unflatten(2, (attn.heads, attn.head_dim))
    key = key.unflatten(2, (attn.heads, attn.head_dim))
    value = value.unflatten(2, (attn.heads, attn.head_dim))
    out = attn.attn(query, key, value, None)
    out = out.flatten(2, 3).to(query.dtype)

    if gate_logits is not None:
        out = out.unflatten(2, (attn.heads, attn.head_dim))
        out = out * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
        out = out.flatten(2, 3)

    out = attn.to_out[0](out)
    if isinstance(out, tuple):
        out = out[0]
    return attn.to_out[1](out), captured


def _video_block_forward(
    block: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    *,
    temb: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    encoder_attention_mask: torch.Tensor | None,
    prefix: _LayerPrefix | None,
    capture: CaptureMode | None,
    kv_only: bool,
) -> tuple[torch.Tensor | None, KvPair | None]:
    """Video stream of ``LTX2VideoTransformerBlock.forward`` (self-attn, text cross-attn, FFN)."""
    batch_size = hidden_states.size(0)
    ada = block.get_mod_params(block.scale_shift_table, temb, batch_size)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada[:6]

    norm_hidden_states = block.norm1(hidden_states) * (1 + scale_msa) + shift_msa
    attn_out, captured = _self_attention_with_prefix(
        block.attn1, norm_hidden_states, rotary_emb, prefix, capture, kv_only=kv_only
    )
    if kv_only:
        return None, captured
    hidden_states = hidden_states + attn_out * gate_msa

    norm_hidden_states = block.norm2(hidden_states)
    if getattr(block, "video_cross_attn_adaln", False):
        shift_text_q, scale_text_q, gate_text_q = ada[6:9]
        norm_hidden_states = norm_hidden_states * (1 + scale_text_q) + shift_text_q
    attn_out = block.attn2(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        query_rotary_emb=None,
        attention_mask=encoder_attention_mask,
    )
    if getattr(block, "video_cross_attn_adaln", False):
        attn_out = attn_out * gate_text_q
    hidden_states = hidden_states + attn_out

    norm_hidden_states = block.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
    hidden_states = hidden_states + block.ff(norm_hidden_states) * gate_mlp
    return hidden_states, captured


class SanaWmRefinerRunner:
    """Stateless driver of the chunk-causal refinement over one LTX-2 transformer.

    All request state lives in the :class:`SanaWmRefinerKvCache` the caller
    passes in, so one runner serves every request of the pipeline.
    """

    def __init__(
        self,
        transformer: nn.Module,
        *,
        schedule: SanaWmRefinerSchedule | None = None,
        sink_frames: int = SANA_WM_REFINER_SINK_FRAMES,
        block_size: int = SANA_WM_REFINER_BLOCK_SIZE,
        kv_max_frames: int = SANA_WM_REFINER_KV_MAX_FRAMES,
    ) -> None:
        self.transformer = transformer
        self.schedule = schedule or SanaWmRefinerSchedule()
        self.sink_frames = int(sink_frames)
        self.block_size = int(block_size)
        self.kv_max_frames = int(kv_max_frames)
        config = transformer.config
        self.patch_size = int(config.patch_size)
        self.patch_size_t = int(config.patch_size_t)
        self.timestep_scale = float(config.timestep_scale_multiplier)
        self.num_layers = len(transformer.transformer_blocks)

    # ------------------------------------------------------------------
    # Cache / geometry helpers
    # ------------------------------------------------------------------

    def tokens_per_frame(self, height: int, width: int) -> int:
        return (int(height) // self.patch_size) * (int(width) // self.patch_size) * self.patch_size_t

    def new_cache(self, *, latent_height: int, latent_width: int) -> SanaWmRefinerKvCache:
        return SanaWmRefinerKvCache(
            num_layers=self.num_layers,
            tokens_per_frame=self.tokens_per_frame(latent_height, latent_width),
            sink_frames=self.sink_frames,
            block_size=self.block_size,
            kv_max_frames=self.kv_max_frames,
        )

    def _rotary_emb(
        self,
        frame_positions: Sequence[int],
        *,
        height: int,
        width: int,
        batch_size: int,
        device: torch.device,
        fps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return refiner_rotary_emb(
            self.transformer,
            frame_positions=frame_positions,
            height=height,
            width=width,
            batch_size=batch_size,
            device=device,
            fps=fps,
        )

    @staticmethod
    def _encoder_attention_bias(mask: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
        """2-D multiplicative text mask -> additive bias, as ``LTX2VideoTransformer3DModel.forward`` does."""
        if mask is None:
            return None
        if mask.ndim != 2:
            return mask
        if bool(mask.all()):
            return None
        return ((1 - mask.to(dtype)) * -10000.0).unsqueeze(1)

    def _prefixes(
        self,
        cache: SanaWmRefinerKvCache,
        *,
        block_start: int,
        height: int,
        width: int,
        batch_size: int,
        device: torch.device,
        fps: float,
    ) -> list[_LayerPrefix]:
        """``rf_shifted_sink`` prefix per layer for the block starting at ``block_start``.

        The sink is re-rotated to sit immediately before the history window,
        ``block_start - history_frames - sink_frames``; the history keeps the
        absolute positions it was captured at.
        """
        if cache.sink_kv_pre is None:
            raise RuntimeError("Sana-WM refiner sink K/V must be captured before the first block is refined.")
        sink_rotary_emb = None
        if cache.sink_frames > 0:
            sink_start = block_start - cache.history_frames - cache.sink_frames
            sink_rotary_emb = self._rotary_emb(
                range(sink_start, sink_start + cache.sink_frames),
                height=height,
                width=width,
                batch_size=batch_size,
                device=device,
                fps=fps,
            )
        prefixes: list[_LayerPrefix] = []
        for layer_idx in range(self.num_layers):
            sink_k, sink_v = cache.sink_kv_pre[layer_idx]
            history = cache.history_kv_post[layer_idx]
            prefixes.append(
                _LayerPrefix(
                    sink_k_pre=sink_k,
                    sink_v=sink_v,
                    sink_rotary_emb=sink_rotary_emb,
                    history_k=history[0] if history is not None else None,
                    history_v=history[1] if history is not None else None,
                )
            )
        return prefixes

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward_video(
        self,
        latent_tokens: torch.Tensor,
        *,
        sigma: float,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        prefixes: Sequence[_LayerPrefix] | None,
        capture: CaptureMode | None = None,
    ) -> tuple[torch.Tensor | None, list[KvPair] | None]:
        """Video-only LTX-2 forward on packed tokens ``[B, N, C]`` at one uniform ``sigma``.

        Returns the velocity ``[B, N, C]`` (``None`` when ``capture`` is set:
        a capture pass stops after the last layer's K/V projection) and the
        captured per-layer ``(K, V)``.
        """
        transformer = self.transformer
        batch_size, seq_len, _ = latent_tokens.shape
        if prefixes is not None and len(prefixes) != self.num_layers:
            raise ValueError(f"Sana-WM refiner got {len(prefixes)} layer prefixes for {self.num_layers} layers.")

        hidden_states = transformer.proj_in(latent_tokens)
        timestep = torch.full(
            (batch_size, seq_len), float(sigma) * self.timestep_scale, dtype=torch.float32, device=latent_tokens.device
        )
        temb, embedded_timestep = transformer.time_embed(
            timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        if hasattr(transformer, "caption_projection"):
            encoder_hidden_states = transformer.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))
        encoder_attention_mask = self._encoder_attention_bias(encoder_attention_mask, hidden_states.dtype)

        captured: list[KvPair] = []
        last_layer = self.num_layers - 1
        for layer_idx, block in enumerate(transformer.transformer_blocks):
            kv_only = capture is not None and layer_idx == last_layer
            hidden_states, layer_kv = _video_block_forward(
                block,
                hidden_states,
                encoder_hidden_states,
                temb=temb,
                rotary_emb=rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                prefix=prefixes[layer_idx] if prefixes is not None else None,
                capture=capture,
                kv_only=kv_only,
            )
            if capture is not None:
                if layer_kv is None:
                    raise RuntimeError(f"Sana-WM refiner layer {layer_idx} produced no K/V capture.")
                captured.append(layer_kv)
        if capture is not None:
            return None, captured

        scale_shift = transformer.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift[:, :, 0], scale_shift[:, :, 1]
        hidden_states = transformer.norm_out(hidden_states) * (1 + scale) + shift
        return transformer.proj_out(hidden_states), None

    def _pack(self, latents: torch.Tensor) -> torch.Tensor:
        return pack_latents(latents, patch_size=self.patch_size, patch_size_t=self.patch_size_t)

    def _unpack(self, tokens: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return unpack_latents(
            tokens,
            num_frames=int(like.shape[2]),
            height=int(like.shape[3]),
            width=int(like.shape[4]),
            patch_size=self.patch_size,
            patch_size_t=self.patch_size_t,
        )

    @torch.no_grad()
    def capture_sink(
        self,
        cache: SanaWmRefinerKvCache,
        sink_latents: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        fps: float,
    ) -> None:
        """Capture the pre-RoPE K/V of the raw conditioning latents at ``sigma = 0``, positions ``[0, sink)``."""
        if cache.sink_kv_pre is not None:
            return
        if int(sink_latents.shape[2]) != cache.sink_frames:
            raise ValueError(
                f"Sana-WM refiner sink has {sink_latents.shape[2]} frames, cache expects {cache.sink_frames}."
            )
        batch_size, _, _, height, width = sink_latents.shape
        rotary_emb = self._rotary_emb(
            range(cache.sink_frames),
            height=height,
            width=width,
            batch_size=batch_size,
            device=sink_latents.device,
            fps=fps,
        )
        _, captured = self.forward_video(
            self._pack(sink_latents),
            sigma=0.0,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            rotary_emb=rotary_emb,
            prefixes=None,
            capture="pre_rope",
        )
        cache.sink_kv_pre = list(captured or [])

    @torch.no_grad()
    def refine_block(
        self,
        cache: SanaWmRefinerKvCache,
        clean_block: torch.Tensor,
        *,
        block_start: int,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        fps: float,
        generator: torch.Generator | None,
        sink_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Refine one clean Stage-1 block ``[B, C, F, H, W]`` covering frames ``[block_start, block_start + F)``.

        ``sink_latents`` is required on the first call (the raw conditioning
        latents, frames ``[0, sink_frames)``) and ignored afterwards. The
        cache is advanced by the refined block's post-RoPE K/V.
        """
        if block_start < cache.sink_frames:
            raise ValueError(f"Sana-WM refiner block_start={block_start} overlaps the sink ({cache.sink_frames}).")
        if cache.sink_kv_pre is None:
            if sink_latents is None:
                raise ValueError("Sana-WM refiner needs sink_latents on the first refine_block call.")
            self.capture_sink(
                cache,
                sink_latents.to(dtype=clean_block.dtype),
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                fps=fps,
            )
        batch_size, _, active_frames, height, width = clean_block.shape
        device, dtype = clean_block.device, clean_block.dtype
        positions = range(int(block_start), int(block_start) + int(active_frames))
        rotary_emb = self._rotary_emb(
            positions, height=height, width=width, batch_size=batch_size, device=device, fps=fps
        )
        prefixes = self._prefixes(
            cache,
            block_start=int(block_start),
            height=height,
            width=width,
            batch_size=batch_size,
            device=device,
            fps=fps,
        )

        sigma_max = self.schedule.sigma_max
        noise_device = generator.device if isinstance(generator, torch.Generator) else device
        eps = torch.randn(clean_block.shape, generator=generator, device=noise_device, dtype=dtype).to(device=device)
        x_t = ((1.0 - sigma_max) * clean_block.float() + sigma_max * eps.float()).to(dtype)

        # Step on packed tokens: pack/unpack are pure rearrangements, so the
        # shared LTX Euler update in token space matches the official
        # latent-space update exactly.
        tokens = self._pack(x_t)
        sigmas = torch.tensor(self.schedule.sigmas, dtype=torch.float32, device=device)
        for step_index, sigma_cur in enumerate(self.schedule.sigmas[:-1]):
            velocity, _ = self.forward_video(
                tokens,
                sigma=sigma_cur,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                rotary_emb=rotary_emb,
                prefixes=prefixes,
            )
            tokens = euler_step_from_velocity(tokens, velocity, sigmas, step_index)
        x_t = self._unpack(tokens, like=clean_block)

        # Cache write: the refined block's post-RoPE K/V under the same prefix.
        _, captured = self.forward_video(
            tokens,
            sigma=0.0,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            rotary_emb=rotary_emb,
            prefixes=prefixes,
            capture="post_rope",
        )
        cache.append_history(captured or [], int(active_frames))
        return x_t

    def describe(self) -> dict[str, Any]:
        return {
            "sigmas": list(self.schedule.sigmas),
            "sink_frames": self.sink_frames,
            "block_size": self.block_size,
            "kv_max_frames": self.kv_max_frames,
            "num_layers": self.num_layers,
        }
