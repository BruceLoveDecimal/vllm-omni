# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wan2.2-Animate-2 transformer with two-phase in-context reference attention.

Ported from the official inference repository ``Wan-Video/Wan-Animate-2``
(``wanxiang/wanxiang_animate_2_arch.py``); weights are loaded from the
Diffusers layout (``WanAnimate2Transformer3DModel``).

The backbone is the ordinary Wan DiT block (self-attn + text/image cross-attn
+ FFN + adaLN), so the tensor-parallel building blocks from
:mod:`vllm_omni.diffusion.models.wan2_2.wan2_2_transformer` are reused.  What
is specific to Animate-2 is the *execution model*:

``extract_reference()``
    Runs once per segment over the driving-video latents at a fixed timestep of
    1 and records every layer's pre-RoPE key/value tensors into a
    :class:`ReferenceKVCache`.

``forward()``
    The denoising pass.  Each layer's self-attention attends jointly over the
    generated tokens and the *frame-aligned* slice of the cached reference
    keys/values, with RoPE re-applied to the cached keys on a disjoint position
    grid.

Upstream expresses that joint attention as a single flex-attention call with a
compiled ``BlockMask``.  Here it is decomposed into a small number of dense
attention branches merged by log-sum-exp, which is mathematically identical
(see :func:`reference_context_attention`) but keeps shapes static and works
with any attention backend, on CPU as well as GPU.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.conv import Conv3dLayer
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm
from vllm_omni.diffusion.layers.norm import LayerNorm, RMSNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbeddingWanS2V
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
    DistributedRMSNorm,
    OutputScaleShiftPrepare,
    WanCrossAttention,
    WanFeedForward,
    WanTimeTextImageEmbedding,
)
from vllm_omni.diffusion.models.wan_animate2.reference_attention import (
    ReferenceGridInfo,
    WanAnimate2RotaryPosEmbed,
    attention_with_lse,
    reference_context_attention,
)
from vllm_omni.diffusion.models.wan_animate2.reference_kv_cache import ReferenceKVCache

# `torch.nn.LayerNorm`'s default eps, which upstream's `MLPProj` relies on.  The
# shared `WanImageEmbedding` passes no eps and inherits this repo's 1e-6
# default, a ~9e-3 relative shift at dim=128, so the value is pinned locally.
_MLP_PROJ_LAYERNORM_EPS = 1e-5

# Block index skipped by the negative CFG branch (upstream `forward_gen`:
# `if is_uncondtion and idx == 9: continue`).
_UNCONDITION_SKIPPED_BLOCK = 9

# Width of the CLIP features fed to `img_emb` (CLIP ViT-H/14 hidden size).
_CLIP_IMAGE_DIM = 1280


@dataclass(frozen=True)
class WanAnimate2TransformerConfig:
    """Architecture constants, in the spelling of the Diffusers ``config.json``.

    Only ``log_scale`` has no counterpart in that file: the distilled
    checkpoint is trained with a constant attention bias that the Diffusers
    modular blocks pass at call time, so the pipeline sets it from the
    ``modular_model_index.json`` blocks class.
    """

    patch_size: tuple[int, int, int] = (1, 2, 2)
    dim: int = 5120
    num_heads: int = 40
    in_dim: int = 36
    out_dim: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    eps: float = 1e-6
    use_img_emb: bool = True
    refer_offset_t: int = 1
    refer_offset_h: int = 0
    refer_offset_w: int = -1
    refer_stride: int = 1
    log_scale: float = 0.0
    rope_max_seq_len: int = 512

    @property
    def attention_head_dim(self) -> int:
        return self.dim // self.num_heads

    @classmethod
    def from_dict(cls, config: Mapping[str, object], log_scale: float = 0.0) -> WanAnimate2TransformerConfig:
        """Build from the Diffusers ``transformer/config.json`` dictionary."""
        dim = int(config["dim"])
        num_heads = int(config["num_heads"])
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        return cls(
            patch_size=tuple(int(p) for p in config.get("patch_size", (1, 2, 2))),
            dim=dim,
            num_heads=num_heads,
            in_dim=int(config.get("in_dim", 36)),
            out_dim=int(config.get("out_dim", 16)),
            text_dim=int(config.get("text_dim", 4096)),
            freq_dim=int(config.get("freq_dim", 256)),
            ffn_dim=int(config["ffn_dim"]),
            num_layers=int(config["num_layers"]),
            cross_attn_norm=bool(config.get("cross_attn_norm", True)),
            eps=float(config.get("eps", 1e-6)),
            use_img_emb=bool(config.get("use_img_emb", True)),
            refer_offset_t=int(config.get("refer_offset_t", 1)),
            refer_offset_h=int(config.get("refer_offset_h", 0)),
            refer_offset_w=int(config.get("refer_offset_w", -1)),
            refer_stride=int(config.get("refer_stride", 1)),
            log_scale=log_scale,
        )


class WanAnimate2SelfAttention(nn.Module):
    """Self-attention with a reference-extraction and an in-context mode."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float,
        log_scale: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.log_scale = log_scale
        self.softmax_scale = 1.0 / (head_dim**0.5)

        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=head_dim,
            total_num_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.to_qkv",
        )
        self.num_heads = self.to_qkv.num_heads
        self.num_kv_heads = self.to_qkv.num_kv_heads
        tp_inner_dim = self.num_heads * head_dim

        if get_tensor_model_parallel_world_size() > 1:
            self.norm_q = DistributedRMSNorm(tp_inner_dim, eps=eps)
            self.norm_k = DistributedRMSNorm(tp_inner_dim, eps=eps)
        else:
            self.norm_q = RMSNorm(tp_inner_dim, eps=eps)
            self.norm_k = RMSNorm(tp_inner_dim, eps=eps)

        self.to_out = RowParallelLinear(
            self.inner_dim,
            dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_out",
        )
        self.rotary_embedding = RotaryEmbeddingWanS2V()

    def _project(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused QKV projection with QK-norm, each ``[B, S, H_local, D]``."""
        qkv, _ = self.to_qkv(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = self.norm_q(query).unflatten(2, (self.num_heads, self.head_dim))
        key = self.norm_k(key).unflatten(2, (self.num_kv_heads, self.head_dim))
        value = value.unflatten(2, (self.num_kv_heads, self.head_dim))
        return query, key, value

    def forward_reference(
        self,
        hidden_states: torch.Tensor,
        reference_freqs: torch.Tensor,
        kv_cache: ReferenceKVCache,
        layer_idx: int,
    ) -> torch.Tensor:
        """Extraction pass: cache pre-RoPE K/V, then plain self-attention."""
        query, key, value = self._project(hidden_states)

        # Cached before RoPE: the denoising pass re-applies it every step.
        kv_cache.store(layer_idx, key, value)

        query = self.rotary_embedding(query, reference_freqs)
        key = self.rotary_embedding(key, reference_freqs)

        out, _ = attention_with_lse(query, key, value, self.softmax_scale)
        return self.to_out(out.flatten(2, 3).type_as(query))

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs: torch.Tensor,
        reference_freqs: torch.Tensor,
        kv_cache: ReferenceKVCache,
        layer_idx: int,
        grid: ReferenceGridInfo,
    ) -> torch.Tensor:
        """Denoising pass: joint attention over generated + reference tokens."""
        query, key, value = self._project(hidden_states)

        query = self.rotary_embedding(query, freqs)
        key = self.rotary_embedding(key, freqs)

        reference_key, reference_value = kv_cache.get(layer_idx)
        reference_key = self.rotary_embedding(reference_key, reference_freqs)

        out = reference_context_attention(
            query,
            key,
            value,
            reference_key,
            reference_value,
            grid,
            softmax_scale=self.softmax_scale,
            log_scale=self.log_scale,
        )
        return self.to_out(out.flatten(2, 3).type_as(query))


class WanAnimate2TransformerBlock(nn.Module):
    """Wan DiT block whose self-attention carries the reference context."""

    def __init__(
        self,
        config: WanAnimate2TransformerConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        dim = config.dim
        head_dim = config.attention_head_dim

        self.norm1 = AdaLayerNorm(dim, elementwise_affine=False, eps=config.eps)
        self.attn1 = WanAnimate2SelfAttention(
            dim=dim,
            num_heads=config.num_heads,
            head_dim=head_dim,
            eps=config.eps,
            log_scale=config.log_scale,
            quant_config=quant_config,
            prefix=f"{prefix}.attn1",
        )

        # The CLIP features are projected to `dim` by `img_emb` before they
        # reach the cross-attention, so the added K/V projections are
        # dim -> dim (upstream: `self.k_img = nn.Linear(dim, dim)`).
        added_kv_proj_dim = dim if config.use_img_emb else None
        self.attn2 = WanCrossAttention(
            dim=dim,
            num_heads=config.num_heads,
            head_dim=head_dim,
            eps=config.eps,
            added_kv_proj_dim=added_kv_proj_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn2",
        )
        if config.cross_attn_norm:
            self.norm2 = LayerNorm(dim, config.eps, elementwise_affine=True)
        else:
            self.norm2 = nn.Identity()

        self.ffn = WanFeedForward(
            dim=dim,
            inner_dim=config.ffn_dim,
            dim_out=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.ffn",
        )
        self.norm3 = AdaLayerNorm(dim, elementwise_affine=False, eps=config.eps)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        reference_freqs: torch.Tensor,
        kv_cache: ReferenceKVCache,
        layer_idx: int,
        freqs: torch.Tensor | None = None,
        grid: ReferenceGridInfo | None = None,
        extract: bool = False,
    ) -> torch.Tensor:
        """Run one block in either phase.

        Both phases go through ``forward``, and therefore through
        ``Module.__call__``, because FSDP2's unshard hook and the layerwise
        CPU-offload hook are pre-forward hooks: calling a bespoke method
        directly would silently read sharded or offloaded parameters.

        Args:
            extract: reference-extraction pass, which populates ``kv_cache``
                instead of consuming it.  ``freqs`` and ``grid`` are then
                unused; the reference grid's RoPE applies to the tokens
                themselves.
        """
        shift, scale, gate, c_shift, c_scale, c_gate = (self.scale_shift_table + temb).chunk(6, dim=1)

        # 1. Self-attention (with the reference context in the denoising pass)
        norm_hidden_states = self.norm1(hidden_states, scale, shift).type_as(hidden_states)
        if extract:
            attn_output = self.attn1.forward_reference(norm_hidden_states, reference_freqs, kv_cache, layer_idx)
        else:
            if freqs is None or grid is None:
                raise ValueError("the denoising pass needs both `freqs` and `grid`")
            attn_output = self.attn1(norm_hidden_states, freqs, reference_freqs, kv_cache, layer_idx, grid)
        hidden_states = (hidden_states + attn_output * gate).type_as(hidden_states)

        # 2. Text (+ CLIP image) cross-attention
        norm_hidden_states = self.norm2(hidden_states).type_as(hidden_states)
        hidden_states = hidden_states + self.attn2(norm_hidden_states, encoder_hidden_states, None)

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states, c_scale, c_shift).type_as(hidden_states)
        hidden_states = (hidden_states + self.ffn(norm_hidden_states) * c_gate).type_as(hidden_states)
        return hidden_states


class WanAnimate2Transformer3DModel(nn.Module):
    """Wan2.2-Animate-2 DiT.

    Two entry points share the same blocks:

    * :meth:`extract_reference` fills a :class:`ReferenceKVCache` from the
      driving-video latents once per segment;
    * :meth:`forward` is the per-step denoising pass that consumes that cache.

    The cache is returned to (and owned by) the caller rather than stored on the
    module, so segment state never enters ``state_dict()``, FSDP flat
    parameters or the offload bookkeeping.
    """

    _repeated_blocks = ["WanAnimate2TransformerBlock"]
    _layerwise_offload_blocks_attrs = ["blocks"]
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
    }

    @staticmethod
    def _is_transformer_block(name: str, module: nn.Module) -> bool:
        """Match transformer blocks for HSDP sharding (e.g. blocks.0, blocks.1)."""
        return "blocks" in name and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]

    def __init__(
        self,
        config: WanAnimate2TransformerConfig,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.config = config
        self.refer_offsets = (config.refer_offset_t, config.refer_offset_h, config.refer_offset_w)

        self.rope = WanAnimate2RotaryPosEmbed(config.attention_head_dim, max_seq_len=config.rope_max_seq_len)

        self.patch_embedding = Conv3dLayer(
            in_channels=config.in_dim,
            out_channels=config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        image_embed_dim = _CLIP_IMAGE_DIM if config.use_img_emb else None
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=config.dim,
            time_freq_dim=config.freq_dim,
            time_proj_dim=config.dim * 6,
            text_embed_dim=config.text_dim,
            image_embed_dim=image_embed_dim,
        )
        if self.condition_embedder.image_embedder is not None:
            self.condition_embedder.image_embedder.norm1.eps = _MLP_PROJ_LAYERNORM_EPS
            self.condition_embedder.image_embedder.norm2.eps = _MLP_PROJ_LAYERNORM_EPS

        self.blocks = nn.ModuleList(
            [
                WanAnimate2TransformerBlock(config, quant_config=quant_config, prefix=f"blocks.{idx}")
                for idx in range(config.num_layers)
            ]
        )

        self.norm_out = AdaLayerNorm(config.dim, elementwise_affine=False, eps=config.eps)
        self.proj_out = nn.Linear(config.dim, config.out_dim * math.prod(config.patch_size))
        self.output_scale_shift_prepare = OutputScaleShiftPrepare(config.dim)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _embed_tokens(
        self,
        latents: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Channel-concatenate ``condition`` onto ``latents`` and patchify.

        Returns ``[B, F*h*w, dim]`` tokens and the ``(F, h, w)`` patch grid.
        """
        hidden_states = torch.cat([latents, condition], dim=1)
        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = (int(hidden_states.shape[2]), int(hidden_states.shape[3]), int(hidden_states.shape[4]))
        return hidden_states.flatten(2).transpose(1, 2), grid_sizes

    def _prepare_context(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Timestep and text/image conditioning shared by both phases."""
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.cat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        return temb, timestep_proj, encoder_hidden_states

    def _reference_freqs(
        self,
        reference_grid: tuple[int, int, int],
        generation_grid: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """RoPE frequencies for the reference grid, off the generation grid.

        A negative configured offset means "shift by the corresponding
        generation-grid extent", which is how the checkpoint keeps the
        reference tokens off the generated grid entirely.
        """
        offsets = []
        for axis, offset in enumerate(self.refer_offsets):
            if offset < 0:
                offsets.append(generation_grid[axis])
            else:
                offsets.append(offset)
        return self.rope(reference_grid, device, offsets=tuple(offsets), time_stride=self.config.refer_stride)

    @torch.no_grad()
    def extract_reference(
        self,
        reference_latents: torch.Tensor,
        reference_condition: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor,
        generation_grid_sizes: tuple[int, int, int],
    ) -> ReferenceKVCache:
        """Run the driving-video latents through every block, caching K/V.

        Args:
            reference_latents: ``[B, 16, T_ref, H, W]`` driving-video latents.
            reference_condition: ``[B, 20, T_ref, H, W]`` mask + latent
                conditioning, concatenated on the channel axis.
            encoder_hidden_states: embeddings of the fixed reference prompt.
            encoder_hidden_states_image: CLIP features of the driving segment's
                first frame.
            generation_grid_sizes: the *generation* patch grid, needed to place
                the reference RoPE positions off that grid.

        Returns:
            A populated :class:`ReferenceKVCache` owned by the caller.
        """
        hidden_states, reference_grid = self._embed_tokens(reference_latents, reference_condition)

        # Upstream pins the extraction timestep to 1 regardless of the sampler.
        timestep = torch.ones(hidden_states.shape[0], device=hidden_states.device, dtype=torch.long)
        _, timestep_proj, encoder_hidden_states = self._prepare_context(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        reference_freqs = self._reference_freqs(reference_grid, generation_grid_sizes, hidden_states.device)

        kv_cache = ReferenceKVCache(self.config.num_layers)
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                reference_freqs,
                kv_cache,
                layer_idx,
                extract=True,
            )
        return kv_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor,
        condition_latents: torch.Tensor,
        kv_cache: ReferenceKVCache,
        reference_grid_sizes: tuple[int, int, int],
        origin_len: int,
        origin_area: tuple[int, int],
        is_uncondition: bool = False,
        return_dict: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor] | Transformer2DModelOutput:
        """Denoising forward.

        Args:
            hidden_states: ``[B, 16, T, H, W]`` noisy latents; latent frame 0 is
                the reference-image slot.
            timestep: ``[B]`` diffusion timestep.
            encoder_hidden_states: text embeddings (positive or negative).
            encoder_hidden_states_image: CLIP features of the reference image.
            condition_latents: ``[B, 20, T, H, W]`` mask + latent conditioning.
            kv_cache: cache produced by :meth:`extract_reference` for this
                segment; shared by both CFG branches.
            reference_grid_sizes: patch grid of the reference tokens.
            origin_len: the segment length the reference mask was sized for
                (the request's nominal segment length, not the possibly shorter
                trailing segment).
            origin_area: ``(width, height)`` in pixels, sizing the mask window.
            is_uncondition: negative CFG branch, which skips block 9.
        """
        batch_size = hidden_states.shape[0]
        p_t, p_h, p_w = self.config.patch_size
        num_frames = hidden_states.shape[2] // p_t
        height = hidden_states.shape[3] // p_h
        width = hidden_states.shape[4] // p_w

        hidden_states, generation_grid = self._embed_tokens(hidden_states, condition_latents)
        temb, timestep_proj, encoder_hidden_states = self._prepare_context(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )

        freqs = self.rope(generation_grid, hidden_states.device)
        reference_freqs = self._reference_freqs(reference_grid_sizes, generation_grid, hidden_states.device)
        grid = ReferenceGridInfo.from_grids(generation_grid, reference_grid_sizes, origin_len, origin_area)

        for layer_idx, block in enumerate(self.blocks):
            if is_uncondition and layer_idx == _UNCONDITION_SKIPPED_BLOCK:
                continue
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                reference_freqs,
                kv_cache,
                layer_idx,
                freqs=freqs,
                grid=grid,
            )

        shift, scale = self.output_scale_shift_prepare(temb)
        hidden_states = self.norm_out(hidden_states, scale.unsqueeze(1), shift.unsqueeze(1)).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        # [B, F*h*w, C*p] -> [B, C, F*p_t, h*p_h, w*p_w]
        hidden_states = hidden_states.reshape(batch_size, num_frames, height, width, p_t, p_h, p_w, -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    # Diffusers checkpoint name -> module path.  Longest / most specific
    # patterns first; the first match wins.
    _BLOCK_RENAMES: tuple[tuple[str, str], ...] = (
        (".self_attn.to_out.0.", ".attn1.to_out."),
        (".self_attn.", ".attn1."),
        (".cross_attn.to_out.0.", ".attn2.to_out."),
        (".cross_attn.", ".attn2."),
        (".ffn.0.", ".ffn.net_0.proj."),
        (".ffn.2.", ".ffn.net_2."),
        (".norm3.", ".norm2."),
        (".modulation", ".scale_shift_table"),
    )

    _TOPLEVEL_RENAMES: tuple[tuple[str, str], ...] = (
        ("text_embedding.0.", "condition_embedder.text_embedder.linear_1."),
        ("text_embedding.2.", "condition_embedder.text_embedder.linear_2."),
        ("time_embedding.0.", "condition_embedder.time_embedder.linear_1."),
        ("time_embedding.2.", "condition_embedder.time_embedder.linear_2."),
        ("time_projection.1.", "condition_embedder.time_proj."),
        ("img_emb.proj.0.", "condition_embedder.image_embedder.norm1."),
        ("img_emb.proj.1.", "condition_embedder.image_embedder.ff.net.0.proj."),
        ("img_emb.proj.3.", "condition_embedder.image_embedder.ff.net.2."),
        ("img_emb.proj.4.", "condition_embedder.image_embedder.norm2."),
        ("head.head.", "proj_out."),
        ("head.modulation", "output_scale_shift_prepare.scale_shift_table"),
    )

    # RMSNorms that sit on a column-parallel output and must be sharded to match.
    _TP_SHARDED_NORMS: tuple[str, ...] = (
        ".attn1.norm_q.",
        ".attn1.norm_k.",
        ".attn2.norm_q.",
        ".attn2.norm_k.",
        ".attn2.norm_added_k.",
    )

    @classmethod
    def remap_weight_name(cls, name: str) -> str:
        """Translate a Diffusers ``WanAnimate2Transformer3DModel`` key to a module path."""
        if name.startswith("blocks."):
            for src, dst in cls._BLOCK_RENAMES:
                if src in name:
                    return name.replace(src, dst)
            return name

        for src, dst in cls._TOPLEVEL_RENAMES:
            if name.startswith(src):
                return dst + name[len(src) :]
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load Diffusers-format Animate-2 weights, fusing self-attention QKV."""
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        stacked_params_mapping = [
            (".attn1.to_qkv", ".attn1.to_q", "q"),
            (".attn1.to_qkv", ".attn1.to_k", "k"),
            (".attn1.to_qkv", ".attn1.to_v", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for original_name, loaded_weight in weights:
            name = self.remap_weight_name(original_name)

            stacked = None
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                    stacked = (name.replace(weight_name, param_name), shard_id)
                    break

            if stacked is not None:
                lookup_name, shard_id = stacked
                if lookup_name not in params_dict:
                    raise KeyError(f"unexpected weight {original_name} -> {lookup_name}")
                param = params_dict[lookup_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(lookup_name)
                continue

            if name not in params_dict:
                raise KeyError(f"unexpected weight {original_name} -> {name}")
            if tp_size > 1 and any(norm in name for norm in self._TP_SHARDED_NORMS):
                shard_size = loaded_weight.shape[0] // tp_size
                loaded_weight = loaded_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size]

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
