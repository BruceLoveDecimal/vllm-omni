# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions adapted from Microsoft Mage:
# https://github.com/microsoft/Mage/tree/df7f84d9f8fc991d189d929f03cff623b430a4a2
# Copyright (c) Microsoft Corporation.
# Microsoft-derived portions are licensed under the MIT License.

"""Core layers for the padded request-batch Mage-Flow transformer.

The checkpoint-facing module names intentionally match the upstream Mage
implementation. The attention kernel itself is replaced with vLLM-Omni's
backend-selectable ``Attention`` layer.
"""

import math
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.normalization import RMSNorm

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention


def _request_isolated_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep BF16 GEMM shapes identical to standalone request execution."""
    if hidden_states.shape[0] == 1:
        return module(hidden_states)
    if attention_mask is None:
        return torch.cat(
            [
                module(hidden_states[index : index + 1])
                for index in range(hidden_states.shape[0])
            ],
            dim=0,
        )

    output = None
    for index in range(hidden_states.shape[0]):
        valid_tokens = attention_mask[index]
        sample_output = module(
            hidden_states[index : index + 1, valid_tokens],
        )
        if output is None:
            output = sample_output.new_zeros(
                (
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    *sample_output.shape[2:],
                )
            )
        output[index, valid_tokens] = sample_output[0]
    assert output is not None
    return output


def apply_rotary_emb_mage_flow(
    hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply Mage's adjacent-pair complex RoPE to ``[B, S, H, D]`` Q/K."""
    if hidden_states.shape[-1] % 2:
        raise ValueError("Mage-Flow RoPE requires an even attention head dimension")
    expected_unbatched = (hidden_states.shape[1], hidden_states.shape[-1] // 2)
    expected_batched = (hidden_states.shape[0], *expected_unbatched)
    if freqs_cis.shape not in (expected_unbatched, expected_batched):
        raise ValueError(
            "RoPE shape does not match image tokens: "
            f"got {tuple(freqs_cis.shape)}, expected {expected_unbatched} "
            f"or {expected_batched}"
        )
    rotated = torch.view_as_complex(hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2))
    frequencies = freqs_cis[None, :, None, :] if freqs_cis.ndim == 2 else freqs_cis[:, :, None, :]
    output = torch.view_as_real(rotated * frequencies).flatten(-2)
    return output.to(hidden_states.dtype)


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Mage's timestep embedding, including its checkpoint-critical rounding."""
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be a 1-D tensor")
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    frequencies = torch.exp(exponent).to(timesteps.dtype)
    embedding = scale * timesteps[:, None].float() * frequencies[None, :]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
    if flip_sin_to_cos:
        embedding = torch.cat([embedding[:, half_dim:], embedding[:, :half_dim]], dim=-1)
    if embedding_dim % 2:
        embedding = nn.functional.pad(embedding, (0, 1))
    return embedding


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class MageFlowTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            scale=1000,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.time_proj(timestep)
        return _request_isolated_forward(
            self.timestep_embedder,
            projected.to(dtype=hidden_states.dtype),
        )


class MageFlowEmbedRope(nn.Module):
    """Generate the upstream three-axis, centered image RoPE frequencies."""

    def __init__(
        self,
        theta: int,
        axes_dim: list[int],
        scale_rope: bool = True,
        max_positions: int = 4096,
    ) -> None:
        super().__init__()
        if any(dim % 2 for dim in axes_dim):
            raise ValueError(f"all axes_dim entries must be even, got {axes_dim}")
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope

        # DiffusersPipelineLoader may instantiate modules under a CUDA or meta
        # default-device context. Pin construction to CPU explicitly; relying
        # on torch.arange()'s ambient device would silently reintroduce the
        # CUDA pow/polar rounding that this cache is intended to avoid.
        positive_indices = torch.arange(max_positions, device="cpu")
        negative_indices = (
            torch.arange(max_positions, device="cpu").flip(0) * -1 - 1
        )

        # Match Qwen-Image and upstream Mage: build the canonical complex64
        # tables on CPU, then lazily move the complete tables to the device
        # requested by forward(). CPU and CUDA pow/polar differ in their last
        # few bits, which is enough to change BF16 rounding after RoPE.
        #
        # These must remain plain attributes rather than registered buffers.
        # The pipeline applies ``module.to(dtype=torch.bfloat16)``; applying
        # that conversion to a complex buffer discards its imaginary phase.
        self.pos_freqs = torch.cat(
            [
                self._rope_params(
                    positive_indices,
                    axis_dim,
                    self.theta,
                )
                for axis_dim in self.axes_dim
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self._rope_params(
                    negative_indices,
                    axis_dim,
                    self.theta,
                )
                for axis_dim in self.axes_dim
            ],
            dim=1,
        )

    @staticmethod
    def _rope_params(
        index: torch.Tensor,
        dim: int,
        theta: int,
    ) -> torch.Tensor:
        frequencies = torch.outer(
            index,
            1.0
            / torch.pow(
                theta,
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.float32,
                    device="cpu",
                ).div(dim),
            ),
        )
        return torch.polar(torch.ones_like(frequencies), frequencies)

    @staticmethod
    def _canonical_device(device: torch.device) -> torch.device:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            return torch.device(
                "cuda",
                torch.accelerator.current_device_index(),
            )
        return device

    @lru_cache(maxsize=16)
    def _compute_grid(
        self,
        device: torch.device,
        frame: int,
        height: int,
        width: int,
        frame_offset: int = 0,
    ) -> torch.Tensor:
        if frame < 1 or height < 1 or width < 1:
            raise ValueError(f"image grid dimensions must be positive, got {(frame, height, width)}")
        if self.pos_freqs.device != device:
            # As in Qwen-Image, mutate the single canonical table pair to the
            # active device. The final grid cache is keyed by device, so a
            # cached cuda:0 tensor can never satisfy a cuda:1 request.
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        positive = self.pos_freqs.split(
            [axis_dim // 2 for axis_dim in self.axes_dim],
            dim=1,
        )
        negative = self.neg_freqs.split(
            [axis_dim // 2 for axis_dim in self.axes_dim],
            dim=1,
        )
        if (
            frame_offset + frame > positive[0].shape[0]
            or height > positive[1].shape[0]
            or width > positive[2].shape[0]
        ):
            raise ValueError(f"image grid {(frame, height, width)} exceeds the RoPE cache")

        frame_freqs = (
            positive[0][frame_offset : frame_offset + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        )
        if self.scale_rope:
            height_freqs = torch.cat(
                [
                    negative[1][-(height - height // 2) :],
                    positive[1][: height // 2],
                ]
            )
            width_freqs = torch.cat(
                [
                    negative[2][-(width - width // 2) :],
                    positive[2][: width // 2],
                ]
            )
        else:
            height_freqs = positive[1][:height]
            width_freqs = positive[2][:width]

        height_freqs = height_freqs.view(1, height, 1, -1).expand(frame, height, width, -1)
        width_freqs = width_freqs.view(1, 1, width, -1).expand(frame, height, width, -1)
        # Cache the final contiguous device tensor, not an expanded view. This
        # bounds both recomputation and repeated CPU-to-device transfers.
        return (
            torch.cat(
                [frame_freqs, height_freqs, width_freqs],
                dim=-1,
            )
            .reshape(frame * height * width, -1)
            .clone()
            .contiguous()
        )

    def forward(
        self,
        image_grid_fhw: tuple[int, int, int] | list[tuple[int, int, int]],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        grids = [image_grid_fhw] if isinstance(image_grid_fhw, tuple) else image_grid_fhw
        if not grids:
            raise ValueError("image_grid_fhw must contain one grid")
        target_device = self._canonical_device(
            self.pos_freqs.device if device is None else device
        )
        return torch.cat(
            [
                self._compute_grid(
                    target_device,
                    frame,
                    height,
                    width,
                    frame_offset=index,
                )
                for index, (frame, height, width) in enumerate(grids)
            ],
            dim=0,
        )


class MageJointAttention(nn.Module):
    """Joint attention over padded per-request ``[text, image]`` tokens."""

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        added_kv_proj_dim: int,
        bias: bool = True,
        eps: float = 1e-6,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head

        # Names match the official Mage checkpoint.
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=bias)
        self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=bias)
        self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=bias)

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.norm_added_q = RMSNorm(dim_head, eps=eps)
        self.norm_added_k = RMSNorm(dim_head, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, query_dim, bias=True), nn.Dropout(0.0)])
        self.to_add_out = nn.Linear(self.inner_dim, added_kv_proj_dim, bias=True)

        self.attention = Attention(
            num_heads=heads,
            head_size=dim_head,
            softmax_scale=dim_head**-0.5,
            causal=False,
            prefix=prefix,
            role="mage_flow.joint",
            role_category="self",
        )

    def _project(
        self,
        hidden_states: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        q_norm: nn.Module,
        k_norm: nn.Module,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (self.heads, self.dim_head)
        query = q_norm(
            _request_isolated_forward(
                q_proj,
                hidden_states,
                attention_mask,
            ).unflatten(-1, shape)
        )
        key = k_norm(
            _request_isolated_forward(
                k_proj,
                hidden_states,
                attention_mask,
            ).unflatten(-1, shape)
        )
        value = _request_isolated_forward(
            v_proj,
            hidden_states,
            attention_mask,
        ).unflatten(-1, shape)
        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        image_attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[0] != encoder_hidden_states.shape[0]:
            raise ValueError("Mage-Flow image and text batch sizes must match")
        batch_size = hidden_states.shape[0]
        if image_attention_mask is None:
            image_attention_mask = torch.ones(
                hidden_states.shape[:2],
                dtype=torch.bool,
                device=hidden_states.device,
            )
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[:2],
                dtype=torch.bool,
                device=encoder_hidden_states.device,
            )
        if image_attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("image_attention_mask must match image token dimensions")
        if encoder_attention_mask.shape != encoder_hidden_states.shape[:2]:
            raise ValueError("encoder_attention_mask must match encoder token dimensions")
        if image_attention_mask.shape[0] != batch_size:
            raise ValueError("Mage-Flow attention masks must match the request batch")

        img_q, img_k, img_v = self._project(
            hidden_states,
            self.to_q,
            self.to_k,
            self.to_v,
            self.norm_q,
            self.norm_k,
            image_attention_mask,
        )
        txt_q, txt_k, txt_v = self._project(
            encoder_hidden_states,
            self.add_q_proj,
            self.add_k_proj,
            self.add_v_proj,
            self.norm_added_q,
            self.norm_added_k,
            encoder_attention_mask,
        )
        img_q = apply_rotary_emb_mage_flow(img_q, image_rotary_emb)
        img_k = apply_rotary_emb_mage_flow(img_k, image_rotary_emb)

        text_length = encoder_hidden_states.shape[1]
        joint_attention_mask = torch.cat(
            [encoder_attention_mask, image_attention_mask],
            dim=1,
        ).to(torch.bool)
        joint_q = torch.cat([txt_q, img_q], dim=1)
        joint_k = torch.cat([txt_k, img_k], dim=1)
        joint_v = torch.cat([txt_v, img_v], dim=1)
        if batch_size == 1:
            attn_metadata = (
                AttentionMetadata(attn_mask=joint_attention_mask) if not bool(joint_attention_mask.all()) else None
            )
            joint_output = self.attention(
                joint_q,
                joint_k,
                joint_v,
                attn_metadata=attn_metadata,
            )
        else:
            # BF16 attention kernels can select a different reduction path when
            # the request-batch dimension changes. The small per-layer drift is
            # amplified by Mage-Flow's residual stack, so execute each request
            # through the same kernel shape as standalone inference. QKV
            # projection, MLP, and residual computation remain request-batched.
            joint_output = torch.zeros_like(joint_q)
            for sample_index in range(batch_size):
                valid_tokens = joint_attention_mask[sample_index]
                sample_output = self.attention(
                    joint_q[sample_index : sample_index + 1, valid_tokens],
                    joint_k[sample_index : sample_index + 1, valid_tokens],
                    joint_v[sample_index : sample_index + 1, valid_tokens],
                )
                joint_output[sample_index, valid_tokens] = sample_output[0]
        txt_output, img_output = joint_output.split([text_length, hidden_states.shape[1]], dim=1)
        if batch_size == 1:
            img_output = self.to_out[1](self.to_out[0](img_output.flatten(2, 3)))
            txt_output = self.to_add_out(txt_output.flatten(2, 3))
        else:
            projected_img_output = hidden_states.new_zeros(
                batch_size,
                hidden_states.shape[1],
                self.inner_dim,
            )
            projected_txt_output = encoder_hidden_states.new_zeros(
                batch_size,
                encoder_hidden_states.shape[1],
                self.inner_dim,
            )
            for sample_index in range(batch_size):
                valid_image_tokens = image_attention_mask[sample_index]
                valid_text_tokens = encoder_attention_mask[sample_index]
                sample_img_output = self.to_out[1](
                    self.to_out[0](
                        img_output[
                            sample_index : sample_index + 1,
                            valid_image_tokens,
                        ].flatten(2, 3)
                    )
                )
                sample_txt_output = self.to_add_out(
                    txt_output[
                        sample_index : sample_index + 1,
                        valid_text_tokens,
                    ].flatten(2, 3)
                )
                projected_img_output[
                    sample_index,
                    valid_image_tokens,
                ] = sample_img_output[0]
                projected_txt_output[
                    sample_index,
                    valid_text_tokens,
                ] = sample_txt_output[0]
            img_output = projected_img_output
            txt_output = projected_txt_output
        img_output = img_output * image_attention_mask[..., None]
        txt_output = txt_output * encoder_attention_mask[..., None]
        return img_output, txt_output


class MageFlowDoubleStreamBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = MageJointAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            bias=True,
            eps=eps,
            prefix=f"{prefix}.attn" if prefix else "",
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
        )

        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
        )

    @staticmethod
    def _modulate(
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = modulation.chunk(3, dim=-1)
        return (
            hidden_states * (1 + scale[:, None]) + shift[:, None],
            gate[:, None],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        image_attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if joint_attention_kwargs:
            unsupported = ", ".join(sorted(joint_attention_kwargs))
            raise ValueError(f"Mage-Flow dense attention does not accept attention kwargs: {unsupported}")
        img_mod1, img_mod2 = _request_isolated_forward(
            self.img_mod,
            temb,
        ).chunk(2, dim=-1)
        txt_mod1, txt_mod2 = _request_isolated_forward(
            self.txt_mod,
            temb,
        ).chunk(2, dim=-1)

        img_normed, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1)
        txt_normed, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)
        img_attn, txt_attn = self.attn(
            img_normed,
            txt_normed,
            image_rotary_emb,
            image_attention_mask=image_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + img_gate1 * img_attn
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn
        if image_attention_mask is not None:
            hidden_states = hidden_states * image_attention_mask[..., None]
        if encoder_attention_mask is not None:
            encoder_hidden_states = encoder_hidden_states * encoder_attention_mask[..., None]

        img_normed, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2)
        txt_normed, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        img_mlp_output = _request_isolated_forward(
            self.img_mlp,
            img_normed,
            image_attention_mask,
        )
        hidden_states = hidden_states + img_gate2 * img_mlp_output
        txt_mlp_output = _request_isolated_forward(
            self.txt_mlp,
            txt_normed,
            encoder_attention_mask,
        )
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output
        if image_attention_mask is not None:
            hidden_states = hidden_states * image_attention_mask[..., None]
        if encoder_attention_mask is not None:
            encoder_hidden_states = encoder_hidden_states * encoder_attention_mask[..., None]
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clamp(-65504, 65504)
            encoder_hidden_states = encoder_hidden_states.clamp(-65504, 65504)
        return encoder_hidden_states, hidden_states


class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        self.norm = nn.LayerNorm(
            embedding_dim,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_embedding: torch.Tensor,
    ) -> torch.Tensor:
        conditioning_embedding = self.silu(conditioning_embedding).to(
            hidden_states.dtype
        )
        scale, shift = _request_isolated_forward(
            self.linear,
            conditioning_embedding,
        ).chunk(2, dim=-1)
        return self.norm(hidden_states) * (1 + scale[:, None]) + shift[:, None]
