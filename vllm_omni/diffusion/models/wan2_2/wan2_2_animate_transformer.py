# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Wan-Animate variant of WanTransformer3DModel for character animation.

The 40 backbone blocks are identical to Wan2.1-I2V-14B, so this model extends
``WanTransformer3DModel`` and only adds the Animate conditioning branches:
pose latents (added after patch embedding) and a face-motion signal injected
through cross-attention every ``inject_face_latents_blocks`` blocks.

Module and parameter names mirror the diffusers checkpoint layout so the
inherited ``load_weights`` handles the whole state dict without remapping.
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

from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
    Transformer2DModelOutput,
    WanTransformer3DModel,
)

logger = init_logger(__name__)

# Per-resolution channel widths of the motion encoder's appearance CNN.
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


def check_unsupported_parallelism(sequence_parallel_size: int, pipeline_parallel_size: int) -> None:
    """Reject parallelism modes whose token layout breaks Animate conditioning."""
    if sequence_parallel_size > 1:
        raise NotImplementedError(
            "Wan-Animate does not support sequence parallelism: the face adapter attends "
            "within each latent frame, which requires shard boundaries to fall on frame "
            "boundaries. Use tensor parallelism or CFG parallelism instead."
        )
    if pipeline_parallel_size > 1:
        raise NotImplementedError(
            "Wan-Animate does not support pipeline parallelism: the face adapter is injected "
            "between backbone blocks and would need the motion signal on every stage."
        )


class FusedLeakyReLU(nn.Module):
    """LeakyReLU with a channel-wise bias folded in and a fixed output scale."""

    def __init__(self, negative_slope: float = 0.2, scale: float = 2**0.5, bias_channels: int | None = None):
        super().__init__()
        self.negative_slope = negative_slope
        self.scale = scale
        self.bias = nn.Parameter(torch.zeros(bias_channels)) if bias_channels is not None else None

    def forward(self, x: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        if self.bias is not None:
            expanded_shape = [1] * x.ndim
            expanded_shape[channel_dim] = self.bias.shape[0]
            x = x + self.bias.reshape(*expanded_shape)
        return F.leaky_relu(x, self.negative_slope) * self.scale


class MotionConv2d(nn.Module):
    """StyleGAN2-style conv: equalized learning rate, optional FIR blur and fused bias."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        blur_kernel: tuple[int, ...] | None = None,
        use_activation: bool = True,
    ):
        super().__init__()
        self.use_activation = use_activation
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding

        self.blur = blur_kernel is not None
        if blur_kernel is not None:
            p = (len(blur_kernel) - stride) + (kernel_size - 1)
            self.blur_padding = ((p + 1) // 2, p // 2)
            kernel = torch.tensor(blur_kernel, dtype=torch.float32)
            kernel = kernel[None, :] * kernel[:, None]
            self.register_buffer("blur_kernel", kernel / kernel.sum(), persistent=False)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        # With an activation the bias lives inside FusedLeakyReLU instead.
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias and not use_activation else None
        self.act_fn = FusedLeakyReLU(bias_channels=out_channels) if use_activation else None

    def forward(self, x: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        if self.blur:
            expanded_kernel = self.blur_kernel[None, None, :, :].expand(self.in_channels, 1, -1, -1)
            x = F.conv2d(x, expanded_kernel.to(x.dtype), padding=self.blur_padding, groups=self.in_channels)

        x = x.to(self.weight.dtype)
        x = F.conv2d(x, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)

        if self.act_fn is not None:
            x = self.act_fn(x, channel_dim=channel_dim)
        return x


class MotionLinear(nn.Module):
    """Linear layer with equalized learning rate scaling."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, use_activation: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.scale = 1 / math.sqrt(in_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias and not use_activation else None
        self.act_fn = FusedLeakyReLU(bias_channels=out_dim) if use_activation else None

    def forward(self, x: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        out = F.linear(x, self.weight * self.scale, bias=self.bias)
        if self.act_fn is not None:
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
        self.conv1 = MotionConv2d(
            in_channels, in_channels, kernel_size, stride=1, padding=kernel_size // 2, use_activation=True
        )
        self.conv2 = MotionConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=downsample_factor,
            padding=0,
            blur_kernel=blur_kernel,
            use_activation=True,
        )
        self.conv_skip = MotionConv2d(
            in_channels,
            out_channels,
            kernel_size_skip,
            stride=downsample_factor,
            padding=0,
            bias=False,
            blur_kernel=blur_kernel,
            use_activation=False,
        )

    def forward(self, x: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        x_out = self.conv2(self.conv1(x, channel_dim), channel_dim)
        x_skip = self.conv_skip(x, channel_dim)
        return (x_out + x_skip) / math.sqrt(2)


class WanAnimateMotionEncoder(nn.Module):
    """Encode a face frame into a motion vector via Linear Motion Decomposition."""

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
        channels = channels or WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES

        self.conv_in = MotionConv2d(3, channels[str(size)], 1, use_activation=True)

        self.res_blocks = nn.ModuleList()
        in_channels = channels[str(size)]
        for i in range(int(math.log(size, 2)), 2, -1):
            out_channels = channels[str(2 ** (i - 1))]
            self.res_blocks.append(MotionEncoderResBlock(in_channels, out_channels))
            in_channels = out_channels

        self.conv_out = MotionConv2d(in_channels, style_dim, 4, padding=0, bias=False, use_activation=False)

        # The original model has no activation between these linear layers.
        linears = [MotionLinear(style_dim, style_dim) for _ in range(motion_blocks - 1)]
        linears.append(MotionLinear(style_dim, motion_dim))
        self.motion_network = nn.ModuleList(linears)

        self.motion_synthesis_weight = nn.Parameter(torch.randn(out_dim, motion_dim))

    def forward(self, face_image: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        if face_image.shape[-2] != self.size or face_image.shape[-1] != self.size:
            raise ValueError(
                f"Face pixel values have resolution ({face_image.shape[-1]}, {face_image.shape[-2]}) but "
                f"({self.size}, {self.size}) is required."
            )

        face_image = self.conv_in(face_image, channel_dim)
        for block in self.res_blocks:
            face_image = block(face_image, channel_dim)
        face_image = self.conv_out(face_image, channel_dim)
        motion_feat = face_image.squeeze(-1).squeeze(-1)

        for linear_layer in self.motion_network:
            motion_feat = linear_layer(motion_feat, channel_dim=channel_dim)

        # The QR orthogonalization is numerically sensitive and must stay in FP32.
        original_dtype = motion_feat.dtype
        motion_feat = motion_feat.float()
        weight = (self.motion_synthesis_weight + 1e-8).float()
        basis = torch.linalg.qr(weight)[0].to(device=motion_feat.device)
        motion_vec = torch.sum(torch.matmul(torch.diag_embed(motion_feat), basis.T), dim=1)
        return motion_vec.to(dtype=original_dtype)


class WanAnimateFaceEncoder(nn.Module):
    """Turn a motion-vector sequence into per-frame face tokens.

    ``conv2``/``conv3`` each halve the temporal length, so a segment of S face
    frames yields T' tokens with T' = ((S - 1) // 2 + 1 - 1) // 2 + 1.
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

        # Plain Conv1d rather than a causal-conv wrapper: the checkpoint stores
        # these weights directly under `conv1_local.weight` etc.
        self.conv1_local = nn.Conv1d(in_dim, hidden_dim * num_heads, kernel_size, stride=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=2)

        self.norm1 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)

        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, out_dim))

    def _causal_conv(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        return conv(F.pad(x, self.time_causal_padding, mode=self.pad_mode))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = self._causal_conv(x.permute(0, 2, 1), self.conv1_local)
        x = x.unflatten(1, (self.num_heads, -1)).flatten(0, 1)  # [B, N*C, T] -> [B*N, C, T]
        x = self.act(self.norm1(x.permute(0, 2, 1)))

        x = self._causal_conv(x.permute(0, 2, 1), self.conv2)
        x = self.act(self.norm2(x.permute(0, 2, 1)))

        x = self._causal_conv(x.permute(0, 2, 1), self.conv3)
        x = self.act(self.norm3(x.permute(0, 2, 1)))

        x = self.out_proj(x)
        x = x.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3)  # [B*N, T, C] -> [B, T, N, C]

        padding = self.padding_tokens.repeat(batch_size, x.shape[1], 1, 1).to(device=x.device, dtype=x.dtype)
        return torch.cat([x, padding], dim=-2)


class WanAnimateFaceBlockCrossAttention(nn.Module):
    """Cross-attention between video tokens and the face tokens of their own frame.

    Uses plain SDPA rather than the shared ``Attention`` layer: the key/value
    sequence is only ``face_encoder_num_heads + 1`` tokens long, and the query
    is regrouped so the batch dimension becomes ``batch * latent_frames``.
    """

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-6, bias: bool = True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.softmax_scale = dim_head**-0.5

        self.pre_norm_q = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.pre_norm_kv = nn.LayerNorm(dim, eps, elementwise_affine=False)

        self.to_q = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=bias)

        # Applied after the head split, so these normalize over dim_head.
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

    def forward(self, hidden_states: torch.Tensor, motion_vec: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, num_face_tokens, _ = motion_vec.shape
        seq_len = hidden_states.shape[1]
        if seq_len % num_frames != 0:
            raise ValueError(
                f"Face adapter needs the token sequence ({seq_len}) to divide evenly across {num_frames} latent frames."
            )

        hidden_states = self.pre_norm_q(hidden_states)
        motion_vec = self.pre_norm_kv(motion_vec).flatten(1, 2)

        query = self.to_q(hidden_states).unflatten(2, (self.heads, -1))
        key = self.to_k(motion_vec).view(batch_size, num_frames, num_face_tokens, self.heads, -1)
        value = self.to_v(motion_vec).view(batch_size, num_frames, num_face_tokens, self.heads, -1)

        query = self.norm_q(query)
        key = self.norm_k(key)

        # Regroup so every latent frame attends only to its own face tokens.
        query = query.unflatten(1, (num_frames, -1)).flatten(0, 1)  # [B*T, S/T, H, D]
        key = key.flatten(0, 1)
        value = value.flatten(0, 1)

        out = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            scale=self.softmax_scale,
        )
        out = out.transpose(1, 2).flatten(2, 3).type_as(query)
        out = out.unflatten(0, (batch_size, num_frames)).flatten(1, 2)
        return self.to_out(out)


class WanAnimateTransformer3DModel(WanTransformer3DModel):
    """Wan2.2-Animate transformer: Wan2.1-I2V backbone with pose and face conditioning."""

    # Sequence parallelism is rejected in __init__; drop the inherited plan so a
    # bypassed guard cannot silently shard the sequence.
    _sp_plan: dict = {}

    def __init__(
        self,
        *,
        latent_channels: int = 16,
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
        check_unsupported_parallelism(*_current_parallel_sizes())

        num_layers = kwargs.get("num_layers", 40)
        if num_layers % inject_face_latents_blocks != 0:
            raise ValueError(
                f"num_layers ({num_layers}) must be divisible by inject_face_latents_blocks "
                f"({inject_face_latents_blocks}) so every injection point has a face adapter."
            )

        # The input packs noisy latents, a 4-channel mask and the conditioning
        # latents, so the two channel counts imply each other.
        expected_in_channels = 2 * latent_channels + 4
        if kwargs.get("in_channels") is None:
            kwargs["in_channels"] = expected_in_channels
        elif kwargs["in_channels"] != expected_in_channels:
            raise ValueError(
                f"in_channels ({kwargs['in_channels']}) must equal 2 * latent_channels + 4 ({expected_in_channels})."
            )

        super().__init__(quant_config=quant_config, **kwargs)

        inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.inject_face_latents_blocks = inject_face_latents_blocks
        self.motion_encoder_batch_size = motion_encoder_batch_size

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
                    heads=self.config.num_attention_heads,
                    dim_head=self.config.attention_head_dim,
                    eps=self.config.eps,
                )
                for _ in range(self.config.num_layers // inject_face_latents_blocks)
            ]
        )

    def encode_face(self, face_pixel_values: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
        """Encode a segment of face frames into per-frame face tokens.

        The result depends only on the face video, so the pipeline computes it
        once per segment instead of at every denoising step.

        Args:
            face_pixel_values: Face video in pixel space ``[B, 3, S, 512, 512]``.
            batch_size: Frames per motion-encoder batch; trades speed for memory.

        Returns:
            Face tokens ``[B, T' + 1, face_tokens, inner_dim]``, where the
            leading frame is zero-padding aligned with the reference latent.
        """
        # Under HSDP this runs outside the FSDP forward hook.
        is_fsdp = hasattr(self, "unshard") and hasattr(self, "reshard")
        if is_fsdp:
            self.unshard()
        try:
            batch, channels, num_face_frames, height, width = face_pixel_values.shape
            frames = face_pixel_values.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)

            batch_size = batch_size or self.motion_encoder_batch_size
            motion_vec = torch.cat([self.motion_encoder(chunk) for chunk in torch.split(frames, batch_size)])
            motion_vec = motion_vec.view(batch, num_face_frames, -1)

            motion_vec = self.face_encoder(motion_vec)
            return torch.cat([torch.zeros_like(motion_vec[:, :1]), motion_vec], dim=1)
        finally:
            if is_fsdp:
                self.reshard()

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        pose_hidden_states: torch.Tensor | None = None,
        motion_vec: torch.Tensor | None = None,
        face_pixel_values: torch.Tensor | None = None,
        motion_encode_batch_size: int | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """Predict noise for one Animate segment.

        Args:
            hidden_states: Packed latents ``[B, 2C + 4, T + 1, H, W]``; frame 0
                is the reference-image slot.
            pose_hidden_states: Pose latents ``[B, C, T, H, W]`` (one frame
                fewer than ``hidden_states``).
            motion_vec: Face tokens from :meth:`encode_face`. When omitted,
                ``face_pixel_values`` is encoded inline.
        """
        if pose_hidden_states is None:
            raise ValueError("Wan-Animate requires pose_hidden_states.")
        if pose_hidden_states.shape[2] + 1 != hidden_states.shape[2]:
            raise ValueError(
                f"pose_hidden_states has {pose_hidden_states.shape[2]} frames but must have one fewer "
                f"than hidden_states ({hidden_states.shape[2]}) to leave room for the reference frame."
            )
        if motion_vec is None:
            if face_pixel_values is None:
                raise ValueError("Wan-Animate requires either motion_vec or face_pixel_values.")
            motion_vec = self.encode_face(face_pixel_values, motion_encode_batch_size)

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Device and dtype belong in the key: offloading can move the module
        # between forwards, and a cached tensor would land on the wrong device.
        rope_key = (
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            hidden_states.device,
            hidden_states.dtype,
        )
        if self._cached_rope_resolution == rope_key and self._cached_rope_emb is not None:
            rotary_emb = self._cached_rope_emb
        else:
            freqs_cos, freqs_sin = self.rope(hidden_states)
            rotary_emb = (freqs_cos[..., 0::2].to(hidden_states.dtype), freqs_sin[..., 1::2].to(hidden_states.dtype))
            self._cached_rope_resolution = rope_key
            self._cached_rope_emb = rotary_emb

        hidden_states = self.patch_embedding(hidden_states)
        # Pose conditions every frame except the reference slot.
        hidden_states[:, :, 1:] = hidden_states[:, :, 1:] + self.pose_patch_embedding(pose_hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=None
        )
        timestep_proj = self.timestep_proj_prepare(timestep_proj, None)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        motion_vec = motion_vec.to(hidden_states.dtype)
        for block_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, None)
            if block_idx % self.inject_face_latents_blocks == 0:
                face_output = self.face_adapter[block_idx // self.inject_face_latents_blocks](hidden_states, motion_vec)
                hidden_states = hidden_states + face_output.to(device=hidden_states.device)

        shift, scale = self.output_scale_shift_prepare(temb)
        shift = shift.to(hidden_states.device).unsqueeze(1)
        scale = scale.to(hidden_states.device).unsqueeze(1)

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


def _current_parallel_sizes() -> tuple[int, int]:
    """Return (sequence_parallel_size, pipeline_parallel_size), 1 when uninitialized."""
    from vllm_omni.diffusion.distributed.parallel_state import (
        get_pipeline_parallel_world_size,
        get_sequence_parallel_world_size,
    )

    try:
        return get_sequence_parallel_world_size(), get_pipeline_parallel_world_size()
    except AssertionError:
        return 1, 1


def create_animate_transformer_from_config(
    config: dict,
    quant_config: QuantizationConfig | None = None,
) -> WanAnimateTransformer3DModel:
    """Create WanAnimateTransformer3DModel from a diffusers config dict."""
    kwargs: dict = {}

    passthrough_keys = (
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
    )
    for key in passthrough_keys:
        if config.get(key) is not None:
            kwargs[key] = config[key]

    if "patch_size" in config:
        kwargs["patch_size"] = tuple(config["patch_size"])

    if "quantization_config" in config:
        from vllm_omni.quantization.factory import resolve_quant_config_from_disk

        quant_config = resolve_quant_config_from_disk(quant_config, config["quantization_config"])
    if quant_config is not None:
        kwargs["quant_config"] = quant_config

    return WanAnimateTransformer3DModel(**kwargs)
