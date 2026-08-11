# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Wan-Animate variant of WanTransformer3DModel for character animation/replacement.

Wan-Animate reuses the Wan2.1-style DiT backbone and adds three things on top:

1. ``pose_patch_embedding`` -- a second Conv3d patch embedding for the pose
   (skeleton) video latents, whose output is added onto the noisy latent tokens
   of every frame except the leading reference frame.
2. ``motion_encoder`` / ``face_encoder`` -- turn the face video (in *pixel*
   space, not latent space) into a per-frame motion token sequence.
3. ``face_adapter`` -- temporally-aligned cross attention blocks that inject the
   motion tokens back into the hidden states after every
   ``inject_face_latents_blocks``-th transformer block.

Everything else (self/cross attention with TP, RoPE, timestep modulation,
output projection, weight loading) comes from :class:`WanTransformer3DModel`.

Ported from ``diffusers.models.transformers.transformer_wan_animate``. Module
and parameter names are kept identical to the diffusers implementation so that
:meth:`WanTransformer3DModel.load_weights` can load a stock Wan-Animate
checkpoint without a remapping table.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.layers.conv import Conv3dLayer
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_omni.diffusion.distributed.parallel_state import get_pipeline_parallel_world_size
from vllm_omni.diffusion.forward_context import get_forward_context
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
    Transformer2DModelOutput,
    WanTransformer3DModel,
)

logger = init_logger(__name__)


# Channel widths of the motion encoder's convolutional trunk, keyed by spatial
# resolution. Copied verbatim from diffusers'
# ``WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES``.
WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES = {
    "4": 512,
    "8": 512,
    "16": 512,
    "32": 512,
    "64": 256,
    "128": 128,
    "256": 64,
    "512": 32,
    "1024": 16,
}


class FusedLeakyReLU(nn.Module):
    """LeakyReLU with a channel-wise bias folded in and a constant output scale."""

    def __init__(self, negative_slope: float = 0.2, scale: float = 2**0.5, bias_channels: int | None = None):
        super().__init__()
        self.negative_slope = negative_slope
        self.scale = scale
        self.channels = bias_channels

        if self.channels is not None:
            self.bias = nn.Parameter(torch.zeros(self.channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        if self.bias is not None:
            expanded_shape = [1] * x.ndim
            expanded_shape[channel_dim] = self.bias.shape[0]
            x = x + self.bias.reshape(*expanded_shape)
        return F.leaky_relu(x, self.negative_slope) * self.scale


class MotionConv2d(nn.Module):
    """Equalized-learning-rate Conv2d with an optional FIR blur and fused activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        blur_kernel: tuple[int, ...] | None = None,
        blur_upsample_factor: int = 1,
        use_activation: bool = True,
    ):
        super().__init__()
        self.use_activation = use_activation
        self.in_channels = in_channels

        self.blur = False
        if blur_kernel is not None:
            p = (len(blur_kernel) - stride) + (kernel_size - 1)
            self.blur_padding = ((p + 1) // 2, p // 2)

            kernel = torch.tensor(blur_kernel)
            if kernel.ndim == 1:
                kernel = kernel[None, :] * kernel[:, None]
            kernel = kernel / kernel.sum()
            if blur_upsample_factor > 1:
                kernel = kernel * (blur_upsample_factor**2)
            self.register_buffer("blur_kernel", kernel, persistent=False)
            self.blur = True

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.stride = stride
        self.padding = padding

        # When an activation follows, its channel-wise bias plays the role of the conv bias.
        if bias and not self.use_activation:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        self.act_fn = FusedLeakyReLU(bias_channels=out_channels) if self.use_activation else None

    def forward(self, x: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        if self.blur:
            # The original implementation uses a 2D upfirdn with up/down rates of 1,
            # which is equivalent to a depthwise convolution with the blur kernel.
            expanded_kernel = self.blur_kernel[None, None, :, :].expand(self.in_channels, 1, -1, -1)
            x = F.conv2d(x, expanded_kernel.to(x.dtype), padding=self.blur_padding, groups=self.in_channels)

        x = x.to(self.weight.dtype)
        x = F.conv2d(x, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)

        if self.use_activation:
            x = self.act_fn(x, channel_dim=channel_dim)
        return x


class MotionLinear(nn.Module):
    """Equalized-learning-rate Linear with an optional fused activation."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, use_activation: bool = False):
        super().__init__()
        self.use_activation = use_activation

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.scale = 1 / math.sqrt(in_dim)

        if bias and not self.use_activation:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

        self.act_fn = FusedLeakyReLU(bias_channels=out_dim) if self.use_activation else None

    def forward(self, input: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        out = F.linear(input, self.weight * self.scale, bias=self.bias)
        if self.use_activation:
            out = self.act_fn(out, channel_dim=channel_dim)
        return out


class MotionEncoderResBlock(nn.Module):
    """Residual block that halves the spatial resolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        kernel_size_skip: int = 1,
        blur_kernel: tuple[int, ...] = (1, 3, 3, 1),
        downsample_factor: int = 2,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.conv1 = MotionConv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            use_activation=True,
        )
        self.conv2 = MotionConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=self.downsample_factor,
            padding=0,
            blur_kernel=blur_kernel,
            use_activation=True,
        )
        self.conv_skip = MotionConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size_skip,
            stride=self.downsample_factor,
            padding=0,
            bias=False,
            blur_kernel=blur_kernel,
            use_activation=False,
        )

    def forward(self, x: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        x_out = self.conv1(x, channel_dim)
        x_out = self.conv2(x_out, channel_dim)
        x_skip = self.conv_skip(x, channel_dim)
        return (x_out + x_skip) / math.sqrt(2)


class WanAnimateMotionEncoder(nn.Module):
    """Encode a face crop into a motion vector via Linear Motion Decomposition."""

    def __init__(
        self,
        size: int = 512,
        style_dim: int = 512,
        motion_dim: int = 20,
        out_dim: int = 512,
        motion_blocks: int = 5,
        channels: dict[str, int] | None = None,
    ):
        super().__init__()
        self.size = size

        if channels is None:
            channels = WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES

        self.conv_in = MotionConv2d(3, channels[str(size)], 1, use_activation=True)

        self.res_blocks = nn.ModuleList()
        in_channels = channels[str(size)]
        log_size = int(math.log(size, 2))
        for i in range(log_size, 2, -1):
            out_channels = channels[str(2 ** (i - 1))]
            self.res_blocks.append(MotionEncoderResBlock(in_channels, out_channels))
            in_channels = out_channels

        self.conv_out = MotionConv2d(in_channels, style_dim, 4, padding=0, bias=False, use_activation=False)

        # No activations between these linear layers -- matches the original implementation.
        linears = [MotionLinear(style_dim, style_dim) for _ in range(motion_blocks - 1)]
        linears.append(MotionLinear(style_dim, motion_dim))
        self.motion_network = nn.ModuleList(linears)

        self.motion_synthesis_weight = nn.Parameter(torch.randn(out_dim, motion_dim))

    def forward(self, face_image: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        if (face_image.shape[-2] != self.size) or (face_image.shape[-1] != self.size):
            raise ValueError(
                f"Face pixel values has resolution ({face_image.shape[-1]}, {face_image.shape[-2]}) but is expected"
                f" to have resolution ({self.size}, {self.size})"
            )

        face_image = self.conv_in(face_image, channel_dim)
        for block in self.res_blocks:
            face_image = block(face_image, channel_dim)
        face_image = self.conv_out(face_image, channel_dim)
        motion_feat = face_image.squeeze(-1).squeeze(-1)

        for linear_layer in self.motion_network:
            motion_feat = linear_layer(motion_feat, channel_dim=channel_dim)

        # Linear Motion Decomposition. The QR orthogonalization is numerically
        # sensitive, so it runs in FP32 regardless of the model dtype.
        weight = self.motion_synthesis_weight + 1e-8
        original_motion_dtype = motion_feat.dtype
        motion_feat = motion_feat.to(torch.float32)
        weight = weight.to(torch.float32)

        Q = torch.linalg.qr(weight)[0].to(device=motion_feat.device)

        motion_feat_diag = torch.diag_embed(motion_feat)
        motion_decomposition = torch.matmul(motion_feat_diag, Q.T)
        motion_vec = torch.sum(motion_decomposition, dim=1)

        return motion_vec.to(dtype=original_motion_dtype)


class WanAnimateFaceEncoder(nn.Module):
    """Turn a per-frame motion vector sequence into per-frame motion tokens.

    The two stride-2 causal Conv1d layers downsample the frame axis by 4, which
    is exactly the VAE temporal compression -- so the output frame count lines up
    with the latent frame count of the segment.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 1024,
        num_heads: int = 4,
        kernel_size: int = 3,
        eps: float = 1e-6,
        pad_mode: str = "replicate",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.time_causal_padding = (kernel_size - 1, 0)
        self.pad_mode = pad_mode

        self.act = nn.SiLU()

        self.conv1_local = nn.Conv1d(in_dim, hidden_dim * num_heads, kernel_size=kernel_size, stride=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=2)

        self.norm1 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)

        self.out_proj = nn.Linear(hidden_dim, out_dim)

        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = x.permute(0, 2, 1)
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        x = self.conv1_local(x)  # [B, C, T_pad] -> [B, N * C, T]
        x = x.unflatten(1, (self.num_heads, -1)).flatten(0, 1)  # -> [B * N, C, T]
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.act(x)

        x = x.permute(0, 2, 1)
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = self.act(x)

        x = x.permute(0, 2, 1)
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.act(x)

        x = self.out_proj(x)
        x = x.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3)  # [B * N, T, C] -> [B, T, N, C]

        padding = self.padding_tokens.repeat(batch_size, x.shape[1], 1, 1).to(device=x.device)
        return torch.cat([x, padding], dim=-2)  # [B, T, N, C] -> [B, T, N + 1, C]


class WanAnimateFaceBlockCrossAttention(nn.Module):
    """Temporally-aligned cross attention between video tokens and motion tokens.

    Each latent frame's video tokens attend only to that frame's ``N + 1`` motion
    tokens, which is implemented by folding the frame axis into the batch axis.
    This requires the token sequence length ``S`` to be divisible by the motion
    token frame count ``T`` -- see the sequence-parallel note on
    :class:`WanAnimateTransformer3DModel`.

    Kept as plain ``nn.Linear`` (no tensor parallelism), mirroring how the S2V
    audio/motion modules are handled: these blocks are a small fraction of the
    parameter count and TP-sharding them would add all-reduces on a tiny matmul.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-6,
        cross_attention_dim_head: int | None = None,
        bias: bool = True,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        # Pre-attention norms -- not present in "vanilla" Wan attention.
        self.pre_norm_q = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.pre_norm_kv = nn.LayerNorm(dim, eps, elementwise_affine=False)

        self.to_q = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, self.kv_inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, self.kv_inner_dim, bias=bias)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=bias)

        # Applied after the head reshape, so over dim_head rather than dim_head * heads.
        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.pre_norm_q(hidden_states)
        encoder_hidden_states = self.pre_norm_kv(encoder_hidden_states)

        # B: batch, T: motion frames, N: motion tokens per frame, C: dim
        B, T, N, C = encoder_hidden_states.shape
        seq_len = hidden_states.shape[1]
        if seq_len % T != 0:
            raise ValueError(
                f"Face adapter requires the token sequence length ({seq_len}) to be divisible by the motion frame "
                f"count ({T}). This normally means the pose/face conditioning length is out of sync with the latent "
                f"frame count."
            )

        encoder_hidden_states = encoder_hidden_states.flatten(1, 2)  # [B, T, N, C] -> [B, T * N, C]

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = query.unflatten(2, (self.heads, -1))  # [B, S, H * D] -> [B, S, H, D]
        key = key.view(B, T, N, self.heads, -1)
        value = value.view(B, T, N, self.heads, -1)

        query = self.norm_q(query)
        key = self.norm_k(key)

        query = query.unflatten(1, (T, -1)).flatten(0, 1)  # [B, S, H, D] -> [B * T, S / T, H, D]
        key = key.flatten(0, 1)  # [B, T, N, H, D] -> [B * T, N, H, D]
        value = value.flatten(0, 1)

        # SDPA wants [B, H, L, D]; kv length is N + 1 (5 by default) so a plain
        # SDPA call is cheaper than routing through the distributed attention layer.
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = hidden_states.unflatten(0, (B, T)).flatten(1, 2)

        hidden_states = self.to_out(hidden_states)

        if attention_mask is not None:
            # Multiplicative mask over the token axis.
            hidden_states = hidden_states * attention_mask.flatten(start_dim=1)

        return hidden_states


class WanAnimateTransformer3DModel(WanTransformer3DModel):
    """Wan-Animate DiT: Wan2.1 backbone + pose patch embedding + face adapter.

    Parallelism support
    -------------------
    * **Tensor parallel** -- inherited from :class:`WanTransformer3DModel`; the
      animate-specific modules stay replicated (they are small).
    * **CFG parallel** -- handled at the pipeline level, nothing to do here.
    * **Sequence parallel** -- *not supported yet*. The face adapter attends
      per latent frame, so it needs each rank's token shard to cover whole
      frames. The inherited ``_sp_plan`` shards the flattened token axis evenly
      (with padding), which only lands on frame boundaries when the sequence
      parallel size divides the latent frame count. Rather than silently produce
      wrong results, ``_sp_plan`` is cleared and :meth:`forward` raises.
    * **Pipeline parallel** -- not supported yet: the face adapter is indexed by
      global block index and has not been partitioned across stages.
    """

    _layerwise_offload_blocks_attrs = ["blocks"]

    # Sequence parallelism is disabled -- see the class docstring. An empty plan
    # means no sharding hooks are installed; forward() additionally raises so a
    # misconfigured run fails loudly instead of duplicating work on every rank.
    _sp_plan: dict[str, Any] = {}

    def __init__(
        self,
        *,
        latent_channels: int | None = 16,
        motion_encoder_channel_sizes: dict[str, int] | None = None,
        motion_encoder_size: int = 512,
        motion_style_dim: int = 512,
        motion_dim: int = 20,
        motion_encoder_dim: int = 512,
        face_encoder_hidden_dim: int = 1024,
        face_encoder_num_heads: int = 4,
        inject_face_latents_blocks: int = 5,
        motion_encoder_batch_size: int = 8,
        quant_config: QuantizationConfig | None = None,
        **kwargs,
    ):
        in_channels = kwargs.get("in_channels")
        # Wan-Animate concatenates [noisy latents | conditioning latents | 4 mask
        # channels] on the channel axis, so in_channels == 2 * latent_channels + 4.
        if in_channels is None and latent_channels is not None:
            in_channels = 2 * latent_channels + 4
            kwargs["in_channels"] = in_channels
        elif in_channels is not None and latent_channels is None:
            latent_channels = (in_channels - 4) // 2
        elif in_channels is None and latent_channels is None:
            raise ValueError("At least one of `in_channels` and `latent_channels` must be supplied.")
        elif in_channels != 2 * latent_channels + 4:
            raise ValueError(
                f"in_channels ({in_channels}) must equal 2 * latent_channels + 4 "
                f"(2 * {latent_channels} + 4 = {2 * latent_channels + 4})"
            )

        kwargs.setdefault("out_channels", latent_channels)

        super().__init__(quant_config=quant_config, **kwargs)

        if get_pipeline_parallel_world_size() > 1:
            raise NotImplementedError(
                "Wan-Animate does not support pipeline parallelism yet: the face adapter blocks are indexed by "
                "global transformer block index and have not been partitioned across pipeline stages."
            )

        num_layers = self.config.num_layers
        num_attention_heads = self.config.num_attention_heads
        inner_dim = num_attention_heads * self.config.attention_head_dim

        # Extend the config object created by the base class with the animate-specific fields.
        self.config.latent_channels = latent_channels
        self.config.motion_encoder_size = motion_encoder_size
        self.config.motion_style_dim = motion_style_dim
        self.config.motion_dim = motion_dim
        self.config.motion_encoder_dim = motion_encoder_dim
        self.config.face_encoder_hidden_dim = face_encoder_hidden_dim
        self.config.face_encoder_num_heads = face_encoder_num_heads
        self.config.inject_face_latents_blocks = inject_face_latents_blocks
        self.config.motion_encoder_batch_size = motion_encoder_batch_size

        self.pose_patch_embedding = Conv3dLayer(
            in_channels=latent_channels,
            out_channels=inner_dim,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )

        self.motion_encoder = WanAnimateMotionEncoder(
            size=motion_encoder_size,
            style_dim=motion_style_dim,
            motion_dim=motion_dim,
            out_dim=motion_encoder_dim,
            channels=motion_encoder_channel_sizes,
        )

        self.face_encoder = WanAnimateFaceEncoder(
            in_dim=motion_encoder_dim,
            out_dim=inner_dim,
            hidden_dim=face_encoder_hidden_dim,
            num_heads=face_encoder_num_heads,
        )

        self.face_adapter = nn.ModuleList(
            [
                WanAnimateFaceBlockCrossAttention(
                    dim=inner_dim,
                    heads=num_attention_heads,
                    dim_head=inner_dim // num_attention_heads,
                    eps=self.config.eps,
                    cross_attention_dim_head=inner_dim // num_attention_heads,
                )
                for _ in range(num_layers // inject_face_latents_blocks)
            ]
        )

        # Set for the duration of forward(); read by after_transformer_block().
        self._motion_vec: torch.Tensor | None = None

    def _check_sequence_parallel_disabled(self) -> None:
        ctx = get_forward_context()
        parallel_config = getattr(ctx.omni_diffusion_config, "parallel_config", None)
        if parallel_config is not None and parallel_config.sequence_parallel_size > 1:
            raise NotImplementedError(
                "Wan-Animate does not support sequence parallelism yet (sequence_parallel_size="
                f"{parallel_config.sequence_parallel_size}). The face adapter cross-attends per latent frame and "
                "requires frame-aligned token shards, which the generic token-axis sharding does not guarantee. "
                "Use tensor parallelism and/or CFG parallelism instead."
            )

    def encode_face_motion(
        self,
        face_pixel_values: torch.Tensor,
        motion_encode_batch_size: int | None = None,
    ) -> torch.Tensor:
        """Encode a face video (pixel space) into per-frame motion tokens.

        Args:
            face_pixel_values: ``[B, C, S, H, W]`` face crops, where ``H == W ==
                config.motion_encoder_size``.
            motion_encode_batch_size: Frames per motion-encoder micro-batch.
                Lower values trade speed for peak memory.

        Returns:
            ``[B, T + 1, N + 1, inner_dim]`` motion tokens, where the leading
            frame is a zero pad aligning the motion tokens with the latent
            sequence's leading reference frame.
        """
        batch_size, channels, num_face_frames, height, width = face_pixel_values.shape
        # [B, C, S, H, W] -> [B * S, C, H, W]
        face_pixel_values = face_pixel_values.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)

        motion_encode_batch_size = motion_encode_batch_size or self.config.motion_encoder_batch_size
        motion_vec_batches = [
            self.motion_encoder(face_batch) for face_batch in torch.split(face_pixel_values, motion_encode_batch_size)
        ]
        motion_vec = torch.cat(motion_vec_batches).view(batch_size, num_face_frames, -1)

        motion_vec = self.face_encoder(motion_vec)

        # Prepend a zero frame so motion tokens line up with the reference frame
        # that the pipeline prepends to the latent sequence.
        pad_face = torch.zeros_like(motion_vec[:, :1])
        return torch.cat([pad_face, motion_vec], dim=1)

    def after_transformer_block(self, block_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """Inject the face motion signal after every ``inject_face_latents_blocks``-th block.

        Kept as a standalone hook (rather than inlined in the block loop) to match
        the shape the S2V CacheDiT wrapper expects -- ``Wan22S2VCachedBlocks``
        calls exactly this method from inside the cached block loop, so wiring
        Cache-DiT up for Wan-Animate later does not need the loop to change.
        """
        inject_every = self.config.inject_face_latents_blocks
        if block_idx % inject_every != 0:
            return hidden_states
        if self._motion_vec is None:
            raise RuntimeError(
                "after_transformer_block() was called outside of forward(): the face motion tokens are not available."
            )
        face_adapter_output = self.face_adapter[block_idx // inject_every](hidden_states, self._motion_vec)
        return face_adapter_output.to(device=hidden_states.device) + hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        pose_hidden_states: torch.Tensor | None = None,
        face_pixel_values: torch.Tensor | None = None,
        motion_encode_batch_size: int | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """Denoise one Wan-Animate segment.

        Args:
            hidden_states: ``[B, 2C + 4, T + 1, H, W]`` noisy latents concatenated
                with the conditioning latents and the 4-channel I2V mask.
            timestep: Current denoising timestep.
            encoder_hidden_states: umT5 text embeddings.
            encoder_hidden_states_image: CLIP features of the reference character image.
            pose_hidden_states: ``[B, C, T, H, W]`` pose video latents -- one frame
                shorter than ``hidden_states`` because of the leading reference frame.
            face_pixel_values: ``[B, 3, S, H', W']`` face video in pixel space.
            motion_encode_batch_size: Motion encoder micro-batch size.
        """
        self._check_sequence_parallel_disabled()

        if pose_hidden_states is None:
            raise ValueError("Wan-Animate requires `pose_hidden_states`.")
        if face_pixel_values is None:
            raise ValueError("Wan-Animate requires `face_pixel_values`.")
        if pose_hidden_states.shape[2] + 1 != hidden_states.shape[2]:
            raise ValueError(
                f"pose_hidden_states frame dim (dim 2) is {pose_hidden_states.shape[2]} but must be one less than "
                f"hidden_states' corresponding frame dim: {hidden_states.shape[2]}"
            )

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 1. RoPE (cached per resolution+dtype -- constant across denoising steps and segments)
        current_rope_resolution = (post_patch_num_frames, post_patch_height, post_patch_width, hidden_states.dtype)
        if self._cached_rope_resolution == current_rope_resolution and self._cached_rope_emb is not None:
            rotary_emb = self._cached_rope_emb
        else:
            freqs_cos, freqs_sin = self.rope(hidden_states)
            rotary_emb = (freqs_cos[..., 0::2].to(hidden_states.dtype), freqs_sin[..., 1::2].to(hidden_states.dtype))
            self._cached_rope_emb = rotary_emb
            self._cached_rope_resolution = current_rope_resolution

        # 2. Patch embedding, with the pose latents added onto every frame but the
        #    leading reference frame.
        hidden_states = self.patch_embedding(hidden_states)
        pose_hidden_states = self.pose_patch_embedding(pose_hidden_states)
        hidden_states = torch.cat(
            [hidden_states[:, :, :1], hidden_states[:, :, 1:] + pose_hidden_states],
            dim=2,
        )
        hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()

        # 3. Condition embeddings. Wan-Animate follows Wan2.1's timestep logic
        #    (a single scalar timestep, no per-token expansion).
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=None
        )
        timestep_proj = self.timestep_proj_prepare(timestep_proj, None)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Motion tokens from the face video. Stashed on the module so
        #    after_transformer_block() can be driven from outside this loop
        #    (see the note on that method).
        self._motion_vec = self.encode_face_motion(face_pixel_values, motion_encode_batch_size)

        # 5. Transformer blocks with periodic face-adapter injection
        try:
            for block_idx, block in enumerate(self.blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, None)
                hidden_states = self.after_transformer_block(block_idx, hidden_states)
        finally:
            self._motion_vec = None

        # 6. Output norm, projection & unpatchify
        shift, scale = self.output_scale_shift_prepare(temb)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        if shift.ndim == 2:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)

        hidden_states = self.norm_out(hidden_states, scale, shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


def create_animate_transformer_from_config(
    config: dict,
    quant_config: QuantizationConfig | None = None,
) -> WanAnimateTransformer3DModel:
    """Build a :class:`WanAnimateTransformer3DModel` from a diffusers config dict."""
    kwargs: dict[str, Any] = {}

    if "patch_size" in config:
        kwargs["patch_size"] = tuple(config["patch_size"])
    for key in (
        "num_attention_heads",
        "attention_head_dim",
        "in_channels",
        "out_channels",
        "text_dim",
        "freq_dim",
        "ffn_dim",
        "num_layers",
        "cross_attn_norm",
        "eps",
        "image_dim",
        "added_kv_proj_dim",
        "rope_max_seq_len",
        "pos_embed_seq_len",
        # Animate-specific
        "latent_channels",
        "motion_encoder_channel_sizes",
        "motion_encoder_size",
        "motion_style_dim",
        "motion_dim",
        "motion_encoder_dim",
        "face_encoder_hidden_dim",
        "face_encoder_num_heads",
        "inject_face_latents_blocks",
        "motion_encoder_batch_size",
    ):
        if key in config and config[key] is not None:
            kwargs[key] = config[key]

    if "quantization_config" in config:
        from vllm_omni.quantization.factory import resolve_quant_config_from_disk

        quant_config = resolve_quant_config_from_disk(quant_config, config["quantization_config"])

    if quant_config is not None:
        kwargs["quant_config"] = quant_config

    return WanAnimateTransformer3DModel(**kwargs)
