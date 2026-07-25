# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions adapted from Microsoft Mage:
# https://github.com/microsoft/Mage/tree/df7f84d9f8fc991d189d929f03cff623b430a4a2
# Copyright (c) Microsoft Corporation.
# Microsoft-derived portions are licensed under the MIT License.

"""Native vLLM-Omni implementation of the Mage-Flow NR-MMDiT."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from diffusers.models.normalization import RMSNorm
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import OmniDiffusionConfig

from .mage_flow_layers import (
    AdaLayerNormContinuous,
    MageFlowDoubleStreamBlock,
    MageFlowEmbedRope,
    MageFlowTimestepProjEmbeddings,
    _request_isolated_forward,
)


class MageFlowTransformer2DModel(nn.Module):
    """The official Mage-Flow transformer with padded request-batch attention."""

    _repeated_blocks = ["MageFlowDoubleStreamBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]

    def __init__(
        self,
        od_config: OmniDiffusionConfig | None = None,
        *,
        in_channels: int | None = None,
        out_channels: int | None = None,
        context_in_dim: int | None = None,
        hidden_size: int | None = None,
        num_heads: int | None = None,
        depth: int | None = None,
        mlp_ratio: float = 4.0,
        depth_single_blocks: int = 0,
        axes_dim: list[int] | None = None,
        theta: int = 10000,
        patch_size: int = 1,
        qkv_bias: bool = True,
        guidance_embed: bool = False,
        rope_type: str = "msrope",
        time_type: str = "qwen_proj",
        double_block_type: str = "double_stream",
        apply_text_rotary_emb: bool = False,
        packing: bool = True,
        schedule_mode: str = "z-image",
        static_shift: float = 6.0,
        use_time_shift: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        model_config = getattr(od_config, "tf_model_config", None) if od_config is not None else None

        def resolve(name: str, explicit: object, default: object = None):
            if explicit is not None:
                return explicit
            return getattr(model_config, name, default)

        in_channels = int(resolve("in_channels", in_channels, 128))
        out_channels = int(resolve("out_channels", out_channels, 128))
        context_in_dim = int(resolve("context_in_dim", context_in_dim, 2560))
        hidden_size = int(resolve("hidden_size", hidden_size, 3072))
        num_heads = int(resolve("num_heads", num_heads, 24))
        depth = int(resolve("depth", depth, 12))
        mlp_ratio = float(resolve("mlp_ratio", mlp_ratio if model_config is None else None, 4.0))
        depth_single_blocks = int(
            resolve(
                "depth_single_blocks",
                depth_single_blocks if model_config is None else None,
                0,
            )
        )
        axes_dim = list(resolve("axes_dim", axes_dim, [16, 56, 56]))
        theta = int(resolve("theta", theta if model_config is None else None, 10000))
        patch_size = int(resolve("patch_size", patch_size if model_config is None else None, 1))
        qkv_bias = bool(resolve("qkv_bias", qkv_bias if model_config is None else None, True))
        guidance_embed = bool(
            resolve(
                "guidance_embed",
                guidance_embed if model_config is None else None,
                False,
            )
        )
        rope_type = str(resolve("rope_type", rope_type if model_config is None else None, "msrope"))
        time_type = str(resolve("time_type", time_type if model_config is None else None, "qwen_proj"))
        double_block_type = str(
            resolve(
                "double_block_type",
                double_block_type if model_config is None else None,
                "double_stream",
            )
        )
        apply_text_rotary_emb = bool(
            resolve(
                "apply_text_rotary_emb",
                apply_text_rotary_emb if model_config is None else None,
                False,
            )
        )
        packing = bool(resolve("packing", packing if model_config is None else None, True))
        schedule_mode = str(
            resolve(
                "schedule_mode",
                schedule_mode if model_config is None else None,
                "z-image",
            )
        )
        static_shift = float(
            resolve(
                "static_shift",
                static_shift if model_config is None else None,
                6.0,
            )
        )
        use_time_shift = bool(
            resolve(
                "use_time_shift",
                use_time_shift if model_config is None else None,
                False,
            )
        )

        if hidden_size % num_heads:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        head_dim = hidden_size // num_heads
        if sum(axes_dim) != head_dim:
            raise ValueError(f"sum(axes_dim) ({sum(axes_dim)}) must equal head_dim ({head_dim})")
        if in_channels != out_channels:
            raise ValueError(
                f"Mage-Flow requires matching in_channels and out_channels, got {in_channels} and {out_channels}"
            )
        if depth_single_blocks != 0:
            raise ValueError(
                f"Mage-Flow single-stream blocks are not supported; checkpoint declares {depth_single_blocks}"
            )
        if patch_size != 1:
            raise ValueError(f"Mage-Flow currently requires patch_size=1, got {patch_size}")
        unsupported_config = {
            "mlp_ratio": (mlp_ratio, 4.0),
            "qkv_bias": (qkv_bias, True),
            "guidance_embed": (guidance_embed, False),
            "rope_type": (rope_type, "msrope"),
            "time_type": (time_type, "qwen_proj"),
            "double_block_type": (double_block_type, "double_stream"),
            "apply_text_rotary_emb": (apply_text_rotary_emb, False),
            "use_time_shift": (use_time_shift, False),
        }
        mismatches = {
            name: (actual, expected) for name, (actual, expected) in unsupported_config.items() if actual != expected
        }
        if mismatches:
            raise ValueError(f"Unsupported Mage-Flow transformer config values: {mismatches}")
        if not packing:
            raise ValueError("Mage-Flow checkpoint must declare packing=true")
        if schedule_mode != "z-image":
            raise ValueError(f"Unsupported Mage-Flow schedule_mode {schedule_mode!r}; expected 'z-image'")
        if static_shift != 6.0:
            raise ValueError(f"Unsupported Mage-Flow static_shift={static_shift}; expected 6.0")

        self.od_config = od_config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_dim = hidden_size
        self.num_attention_heads = num_heads
        self.attention_head_dim = head_dim
        self.axes_dim = axes_dim
        self.patch_size = patch_size
        self.schedule_mode = schedule_mode
        self.static_shift = static_shift

        self.pos_embed = MageFlowEmbedRope(
            theta=theta,
            axes_dim=axes_dim,
            scale_rope=True,
        )
        self.img_in = nn.Linear(in_channels, hidden_size)
        self.txt_norm = RMSNorm(context_in_dim, eps=1e-6)
        self.txt_in = nn.Linear(context_in_dim, hidden_size)
        self.time_text_embed = MageFlowTimestepProjEmbeddings(hidden_size)
        self.transformer_blocks = nn.ModuleList(
            [
                MageFlowDoubleStreamBlock(
                    dim=hidden_size,
                    num_attention_heads=num_heads,
                    attention_head_dim=head_dim,
                    prefix=f"transformer_blocks.{index}",
                )
                for index in range(depth)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(
            hidden_size,
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.proj_out = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
        )

    @staticmethod
    def _normalize_image_grids(
        image_grid_hw: list[tuple[int, int]] | list[tuple[int, int, int]] | tuple[int, int],
    ) -> list[tuple[int, int, int]]:
        grids = [image_grid_hw] if isinstance(image_grid_hw, tuple) else image_grid_hw
        normalized: list[tuple[int, int, int]] = []
        for grid in grids:
            if len(grid) == 2:
                height, width = grid
                normalized.append((1, int(height), int(width)))
            elif len(grid) == 3:
                frame, height, width = grid
                normalized.append((int(frame), int(height), int(width)))
            else:
                raise ValueError(f"invalid Mage-Flow image grid: {grid}")
        return normalized

    def _normalize_batched_image_grids(
        self,
        image_grid_hw: (
            list[tuple[int, int]]
            | list[tuple[int, int, int]]
            | list[list[tuple[int, int]]]
            | list[list[tuple[int, int, int]]]
            | tuple[int, int]
        ),
        batch_size: int,
    ) -> list[list[tuple[int, int, int]]]:
        if batch_size == 1 and (
            isinstance(image_grid_hw, tuple)
            or not image_grid_hw
            or isinstance(image_grid_hw[0], tuple)
        ):
            return [self._normalize_image_grids(image_grid_hw)]
        if not isinstance(image_grid_hw, list) or len(image_grid_hw) != batch_size:
            raise ValueError(
                "batched Mage-Flow image_grid_hw must provide one grid list "
                f"per request, got batch_size={batch_size}"
            )
        return [self._normalize_image_grids(grids) for grids in image_grid_hw]

    def _prepare_batched_image_rope(
        self,
        grids_per_sample: list[list[tuple[int, int, int]]],
        max_image_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        frequencies = []
        for grids in grids_per_sample:
            # RoPE owns a bounded device-aware LRU. Pass the execution device
            # explicitly so cached GPU tensors cannot cross device boundaries.
            sample_frequencies = self.pos_embed(grids, device=device)
            padding = max_image_tokens - sample_frequencies.shape[0]
            if padding < 0:
                raise ValueError("Mage-Flow image grid exceeds the padded token length")
            if padding:
                sample_frequencies = torch.cat(
                    [
                        sample_frequencies,
                        torch.ones(
                            padding,
                            sample_frequencies.shape[1],
                            dtype=sample_frequencies.dtype,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
            frequencies.append(sample_frequencies)
        return torch.stack(frequencies)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        image_grid_hw: (
            list[tuple[int, int]]
            | list[tuple[int, int, int]]
            | list[list[tuple[int, int]]]
            | list[list[tuple[int, int, int]]]
            | tuple[int, int]
        ),
        image_attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor]:
        if hidden_states.ndim != 3 or encoder_hidden_states.ndim != 3:
            raise ValueError("Mage-Flow hidden_states and encoder_hidden_states must be 3-D")
        if hidden_states.shape[0] != encoder_hidden_states.shape[0]:
            raise ValueError("Mage-Flow image and text batch sizes must match")
        batch_size = hidden_states.shape[0]
        if image_attention_mask is None:
            image_attention_mask = torch.ones(
                hidden_states.shape[:2],
                dtype=torch.bool,
                device=hidden_states.device,
            )
        if image_attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("image_attention_mask must match image token dimensions")
        if encoder_attention_mask is not None:
            if encoder_attention_mask.shape != encoder_hidden_states.shape[:2]:
                raise ValueError("encoder_attention_mask must match encoder token dimensions")
        else:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[:2],
                dtype=torch.bool,
                device=encoder_hidden_states.device,
            )

        grids_per_sample = self._normalize_batched_image_grids(image_grid_hw, batch_size)
        expected_tokens = [
            sum(frame * height * width for frame, height, width in grids)
            for grids in grids_per_sample
        ]
        actual_tokens = image_attention_mask.to(torch.int64).sum(dim=1).tolist()
        if actual_tokens != expected_tokens:
            raise ValueError(
                "valid image token counts do not match image grids: "
                f"got {actual_tokens}, expected {expected_tokens}"
            )

        image_rotary_emb = self._prepare_batched_image_rope(
            grids_per_sample,
            hidden_states.shape[1],
            hidden_states.device,
        )
        hidden_states = _request_isolated_forward(
            self.img_in,
            hidden_states,
            image_attention_mask,
        )
        encoder_hidden_states = _request_isolated_forward(
            self.txt_in,
            self.txt_norm(encoder_hidden_states),
            encoder_attention_mask,
        )
        hidden_states = hidden_states * image_attention_mask[..., None]
        encoder_hidden_states = encoder_hidden_states * encoder_attention_mask[..., None]
        timestep = timestep.to(hidden_states.dtype)
        if timestep.shape != (batch_size,):
            raise ValueError(
                f"Mage-Flow timestep must have shape ({batch_size},), got {tuple(timestep.shape)}"
            )
        temb = self.time_text_embed(timestep, hidden_states)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                image_attention_mask=image_attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        output = _request_isolated_forward(
            self.proj_out,
            self.norm_out(hidden_states, temb),
            image_attention_mask,
        )
        output = output * image_attention_mask[..., None]
        return (output,)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
