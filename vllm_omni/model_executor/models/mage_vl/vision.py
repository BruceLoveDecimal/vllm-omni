# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mage-ViT: the codec-native visual encoder of ``microsoft/Mage-VL``.

Structurally a Qwen2-VL-style flat-patch ViT (patches arrive pre-arranged in 2x2 block
order, a spatial merger folds 2x2 -> 1), so attention, projections, and the varlen
mechanism are the upstream ones. Two behaviours differ from the Qwen-VL family and are
documented as deliberate overrides (MODEL-INV-003):

1. **Rotary application.** Mage-VL combines an *interleaved* ``rotate_half`` with cos/sin
   built over the **full** head dim via ``cat([freqs, freqs], -1)``, so the two halves of
   an adjacent pair carry *different* frequencies. No upstream ``ApplyRotaryEmb`` mode
   expresses that: measured against the reference implementation, ``is_neox_style=False``
   reaches cosine 0.699 and ``is_neox_style=True`` 0.572, while the replication below is
   bit-exact. (The reference op is not even norm-preserving, i.e. not a rotation -- it is
   a property the checkpoint was trained with, so it is reproduced rather than
   normalized.) See ``_apply_interleaved_rotary``.

2. **Window partitioning.** ``cu_seqlens`` splits each sample every
   ``frame_windows_size`` temporal steps (remainder forms its own window) instead of one
   segment per image/video. Only the segment boundaries differ; the varlen attention
   itself is upstream ``MMEncoderAttention``.

Positions come in as an explicit ``patch_positions`` table of per-patch ``(t, h, w)``
triples rather than being derived from ``grid_thw``: under codec-native sampling the
retained patches are sparse and ``t`` is the source frame index, which is what the 3D
RoPE must encode.
"""

from collections.abc import Iterable
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.qwen2_vl import Qwen2VisionMLP, Qwen2VisionPatchMerger
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.models.vision import is_vit_use_data_parallel

from vllm_omni.transformers_utils.configs.mage_vl import MageVLVisionConfig

# 3D RoPE splits the half-width frequency vector across (t, h, w) in a 4:6:6 ratio.
_ROPE_THW_RATIO = (4, 6, 6)


def _apply_interleaved_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Mage-VL's rotary application.

    ``x`` is ``[seq, heads, head_dim]``; ``cos``/``sin`` are ``[seq, head_dim]`` (full
    width, not half). Computed in fp32 and cast back, matching the reference, which
    upcasts q/k before applying.
    """
    orig_dtype = x.dtype
    x = x.float()
    cos = cos.float().unsqueeze(-2)
    sin = sin.float().unsqueeze(-2)
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    return (x * cos + rotated * sin).to(orig_dtype)


class MageVLVisionRotaryEmbedding(nn.Module):
    """3D (t, h, w) rotary frequencies with a 4:6:6 split of the half head dim."""

    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        half = head_dim // 2
        if half % sum(_ROPE_THW_RATIO) != 0:
            raise ValueError(
                f"head_dim//2 ({half}) must be divisible by {sum(_ROPE_THW_RATIO)} "
                "to split into 4:6:6 for (t, h, w)"
            )
        unit = half // sum(_ROPE_THW_RATIO)
        self.sizes = tuple(r * unit for r in _ROPE_THW_RATIO)
        base = config.rope_theta
        for name, size in zip(("t", "h", "w"), self.sizes):
            self.register_buffer(
                f"inv_freq_{name}",
                1.0 / (base ** (torch.arange(size, dtype=torch.float32) / size)),
                persistent=False,
            )

    def forward(self, patch_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``patch_positions`` is ``[seq, 3]`` of (t, h, w); returns full-width cos/sin."""
        device = patch_positions.device
        pos = patch_positions.float()
        freqs = torch.cat(
            [
                torch.outer(pos[:, 0], self.inv_freq_t.to(device)),
                torch.outer(pos[:, 1], self.inv_freq_h.to(device)),
                torch.outer(pos[:, 2], self.inv_freq_w.to(device)),
            ],
            dim=-1,
        )
        # The reference duplicates the half-width table to full head dim; the duplication
        # is what makes adjacent pairs carry different frequencies (see module docstring).
        freqs = torch.cat([freqs, freqs], dim=-1)
        return freqs.cos(), freqs.sin()


def build_window_cu_seqlens(
    grid_thw: torch.Tensor | np.ndarray,
    total_patches: int,
    frame_windows_size: int,
) -> np.ndarray:
    """Split each sample into windows of ``frame_windows_size`` temporal steps.

    Returns cumulative sequence lengths as an ``int32`` numpy array; the caller turns it
    into backend-specific metadata via ``MMEncoderAttention``'s classmethods.
    """
    if grid_thw is None or len(grid_thw) == 0:
        return np.array([0, total_patches], dtype=np.int32)
    if isinstance(grid_thw, torch.Tensor):
        grid_thw = grid_thw.cpu().numpy()
    grid_thw = np.asarray(grid_thw)

    lengths: list[int] = []
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        patches_per_step = h * w
        if frame_windows_size and frame_windows_size > 0 and t > frame_windows_size:
            full, remainder = divmod(t, frame_windows_size)
            lengths.extend([frame_windows_size * patches_per_step] * full)
            if remainder:
                lengths.append(remainder * patches_per_step)
        else:
            lengths.append(t * patches_per_step)

    cu = np.zeros(len(lengths) + 1, dtype=np.int32)
    np.cumsum(lengths, out=cu[1:])
    if int(cu[-1]) != total_patches:
        raise ValueError(
            f"cu_seqlens mismatch: grid_thw implies {int(cu[-1])} patches but got "
            f"{total_patches}"
        )
    return cu


class MageVLVisionEmbeddings(nn.Module):
    """Conv patch embedding over pre-extracted ``[N, C, P, P]`` patches."""

    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        self.in_channels = config.num_channels
        self.patch_size = config.patch_size
        self.embed_dim = config.hidden_size
        self.patch_embedding = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        patches = patches.view(-1, self.in_channels, self.patch_size, self.patch_size)
        out = self.patch_embedding(patches.to(self.patch_embedding.weight.dtype))
        return out.view(-1, self.embed_dim)


class MageVLVisionAttention(nn.Module):
    def __init__(
        self,
        config: MageVLVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.tp_size = (
            1
            if use_data_parallel
            else parallel_state.get_tensor_model_parallel_world_size()
        )
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        self.head_dim = dist_utils.divide(embed_dim, num_heads)
        self.num_heads_per_partition = dist_utils.divide(num_heads, self.tp_size)

        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )
        self.proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )
        self.attn = MMEncoderAttention(
            num_heads=self.num_heads_per_partition,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        # x: [seq, embed_dim]
        qkv, _ = self.qkv(x)
        seq_len = qkv.shape[0]
        # QKVParallelLinear packs [q | k | v] along the last dim, each holding this
        # partition's heads only.
        qkv = qkv.view(seq_len, 3, self.num_heads_per_partition, self.head_dim)
        q, k, v = qkv.unbind(dim=1)

        q = _apply_interleaved_rotary(q, cos, sin)
        k = _apply_interleaved_rotary(k, cos, sin)

        out = self.attn(
            query=q.unsqueeze(0),
            key=k.unsqueeze(0),
            value=v.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        out = out.view(seq_len, self.num_heads_per_partition * self.head_dim)
        out, _ = self.proj(out)
        return out


def _norm_layer(config: MageVLVisionConfig) -> nn.Module:
    if config.layer_norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


class MageVLVisionEncoderLayer(nn.Module):
    def __init__(
        self,
        config: MageVLVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_norm1 = _norm_layer(config)
        self.layer_norm2 = _norm_layer(config)
        self.self_attn = MageVLVisionAttention(
            config, quant_config=quant_config, prefix=f"{prefix}.self_attn"
        )
        # Same fc1 -> act -> fc2 shape and parameter names as the Qwen2-VL encoder MLP.
        # ``act_layer`` takes a *type*, and upstream defaults it to QuickGELU, so the
        # config's activation has to be bound explicitly -- otherwise the activation is
        # silently swapped and the tower stops matching the reference.
        self.mlp = Qwen2VisionMLP(
            config.hidden_size,
            config.intermediate_size,
            act_layer=partial(get_act_fn, config.hidden_act),
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.layer_norm1(x), cu_seqlens, cos, sin, max_seqlen, sequence_lengths
        )
        x = x + self.mlp(self.layer_norm2(x))
        return x


class MageVLVisionEncoder(nn.Module):
    """Holds the encoder layers under ``encoder.layers.N``, matching the checkpoint."""

    def __init__(
        self,
        config: MageVLVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MageVLVisionEncoderLayer(
                    config, quant_config=quant_config, prefix=f"{prefix}.layers.{i}"
                )
                for i in range(config.num_hidden_layers)
            ]
        )


class MageVLVisionTransformer(nn.Module):
    """Mage-ViT tower: patch embed -> pre-norm -> windowed encoder -> 2x2 merger."""

    def __init__(
        self,
        config: MageVLVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.out_hidden_size = config.out_hidden_size
        self.frame_windows_size = config.frame_windows_size

        self.embeddings = MageVLVisionEmbeddings(config)
        self.layernorm_pre = _norm_layer(config)
        self.video_rope = MageVLVisionRotaryEmbedding(config)
        self.encoder = MageVLVisionEncoder(
            config, quant_config=quant_config, prefix=f"{prefix}.encoder"
        )
        self.layernorm_post = _norm_layer(config) if config.use_head else None
        # Same ln_q / mlp.0 / mlp.2 layout as the Qwen2-VL merger, so the checkpoint
        # keys load unchanged; only the norm epsilon comes from Mage's config.
        self.merger = Qwen2VisionPatchMerger(
            d_model=config.out_hidden_size,
            context_dim=config.hidden_size,
            norm_layer=partial(nn.LayerNorm, eps=config.layer_norm_eps),
            spatial_merge_size=config.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.embeddings.patch_embedding.weight.device

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        patch_positions: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embeddings(pixel_values)

        if patch_positions.dim() == 3:
            patch_positions = patch_positions.squeeze(0)
        cos, sin = self.video_rope(patch_positions.to(x.device))

        x = self.layernorm_pre(x)

        cu_seqlens_np = build_window_cu_seqlens(
            grid_thw, total_patches=x.shape[0], frame_windows_size=self.frame_windows_size
        )
        # Backend-specific metadata is built by MMEncoderAttention's own helpers: each
        # attention backend consumes a different shape, so hand-computing these would
        # only ever be correct for one of them.
        attn_backend = self.encoder.layers[0].self_attn.attn.attn_backend
        sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
            attn_backend, cu_seqlens_np, x.device
        )
        max_seqlen_val = MMEncoderAttention.compute_max_seqlen(attn_backend, cu_seqlens_np)
        # Kept on CPU on purpose: attention wrappers call .item() on it, and a device
        # tensor would record a wasteful D2H copy into a CUDA graph.
        max_seqlen = torch.tensor(max_seqlen_val, dtype=torch.int32)
        cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
            attn_backend,
            cu_seqlens_np,
            hidden_size=self.config.hidden_size,
            tp_size=self.encoder.layers[0].self_attn.tp_size,
            device=x.device,
        )

        for layer in self.encoder.layers:
            x = layer(x, cu_seqlens, cos, sin, max_seqlen, sequence_lengths)

        if self.layernorm_post is not None:
            x = self.layernorm_post(x)
        return self.merger(x)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the tower.

        Module names mirror the checkpoint (``encoder.layers.N.*``, ``merger.mlp.0`` ...)
        so no renaming is needed. The checkpoint stores a single fused
        ``self_attn.qkv`` matrix laid out as ``[all_q | all_k | all_v]``, which is what
        ``QKVParallelLinear``'s loader consumes when no shard id is supplied.
        """
        return AutoWeightsLoader(self).load_weights(weights)
