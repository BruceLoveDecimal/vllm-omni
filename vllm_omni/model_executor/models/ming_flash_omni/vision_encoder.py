# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
# Adapted from Ming repository qwen3_moe_vit.py
# https://github.com/inclusionAI/Ming

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3Omni_VisionTransformer,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.utils.torch_utils import async_tensor_h2d

logger = init_logger(__name__)


def _adapt_vision_config(vision_config):
    # Adapt Ming's Qwen3VLMoeVisionConfig to be compatible with vLLM's
    # Qwen3Omni_VisionTransformer expectations.
    if not hasattr(vision_config, "image_size") or vision_config.image_size is None:
        if hasattr(vision_config, "num_position_embeddings") and vision_config.num_position_embeddings:
            import math

            num_grid = int(math.sqrt(vision_config.num_position_embeddings))
            vision_config.image_size = num_grid * vision_config.patch_size
        else:
            vision_config.image_size = vision_config.patch_size * 14  # fallback

    if not hasattr(vision_config, "apply_vit_abs_pos_embed"):
        vision_config.apply_vit_abs_pos_embed = True

    return vision_config


class MingVisionEncoder(nn.Module):
    """**Wrapper** around vLLM's Qwen3Omni_VisionTransformer for Ming."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "deepstack_merger_list.": "merger_list.",
            "merger.norm.": "merger.ln_q.",
            "merger.linear_fc1.": "merger.mlp.0.",
            "merger.linear_fc2.": "merger.mlp.2.",
        }
    )

    def __init__(
        self,
        vision_config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        adapted_config = _adapt_vision_config(vision_config)
        norm_eps = 1e-6
        self.encoder = Qwen3Omni_VisionTransformer(
            vision_config=adapted_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder",
        )
        self.image_emb_dim = vision_config.out_hidden_size
        self.use_deepstack = (
            hasattr(vision_config, "deepstack_visual_indexes") and vision_config.deepstack_visual_indexes is not None
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.encoder.dtype

    @property
    def device(self) -> torch.device:
        return self.encoder.device

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor | None = None,
        *,
        encoder_metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """forward method of the vision encoder.

        Args:
            pixel_values: Flattened pixel values.
            grid_thw: [num_images, 3] tensor of (t, h, w) grid sizes.

        Returns:
            If deepstack is enabled, returns concatenated multi-scale features
            along the feature dim: [seq_len, hidden_size * (1 + num_deepstack)].
            Otherwise returns [seq_len, hidden_size].
        """
        if encoder_metadata is None:
            return self.encoder(pixel_values, grid_thw=grid_thw)

        # Qwen3Omni_VisionTransformer does not yet accept precomputed metadata.
        # Reuse its modules (and unchanged checkpoint names) while keeping all
        # grid-dependent Python work outside the captured computation.
        encoder = self.encoder
        hidden_states = pixel_values.to(device=self.device, dtype=self.dtype, non_blocking=True)
        hidden_states = encoder.patch_embed(hidden_states)
        if encoder.apply_vit_abs_pos_embed:
            hidden_states = hidden_states + encoder_metadata["pos_embeds"]
        hidden_states = hidden_states.unsqueeze(1)

        deepstack_states = []
        for layer_num, block in enumerate(encoder.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=encoder_metadata["cu_seqlens"],
                rotary_pos_emb_cos=encoder_metadata["rotary_pos_emb_cos"],
                rotary_pos_emb_sin=encoder_metadata["rotary_pos_emb_sin"],
                max_seqlen=encoder_metadata["max_seqlen"],
                sequence_lengths=encoder_metadata.get("sequence_lengths"),
            )
            if encoder.deepstack_visual_indexes is not None and layer_num in encoder.deepstack_visual_indexes:
                deepstack_states.append(hidden_states)

        hidden_states = encoder.merger(hidden_states)
        if encoder.deepstack_visual_indexes is not None:
            features = [hidden_states]
            for merger, states in zip(encoder.merger_list, deepstack_states):
                features.append(merger(states))
            hidden_states = torch.cat(features, dim=1)
        return hidden_states

    def prepare_encoder_metadata(
        self,
        grid_thw: torch.Tensor,
        *,
        max_sequences: int | None = None,
        max_seqlen_override: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Prepare positions and attention metadata outside CUDA graph capture.

        Grid metadata stays on CPU through MultiModalFieldConfig.keep_on_cpu.
        Reject device grids instead of introducing an implicit D2H sync. Only
        positions and cumulative lengths needed by attention are sent to GPU.
        """
        if grid_thw.device.type != "cpu":
            raise ValueError("Ming vision grid_thw must stay on CPU; configure keep_on_cpu=True")
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3 or grid_thw.shape[0] == 0:
            raise ValueError("Ming vision grid_thw must have shape [num_items, 3] with at least one item")

        encoder = self.encoder
        grid = grid_thw.numpy()
        if np.any(grid <= 0) or np.any(grid[:, 1:] % encoder.spatial_merge_size):
            raise ValueError(
                "Ming vision grids must be positive with spatial dimensions divisible by spatial_merge_size"
            )

        lengths = np.repeat(grid[:, 1] * grid[:, 2], grid[:, 0])
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), lengths.cumsum(dtype=np.int32)])
        if max_sequences is not None:
            if len(lengths) > max_sequences:
                raise ValueError("Ming vision batch exceeds the captured attention sequence capacity")
            # Repeated terminal offsets represent empty sequences. Zero padding
            # would make cu_seqlens non-monotonic and corrupt attention.
            cu_seqlens = np.pad(cu_seqlens, (0, max_sequences - len(lengths)), mode="edge")

        max_seqlen = MMEncoderAttention.compute_max_seqlen(encoder.attn_backend, cu_seqlens)
        if max_seqlen_override is not None:
            if max_seqlen_override < max_seqlen:
                raise ValueError("Ming vision capture max_seqlen must cover every attention sequence")
            max_seqlen = max_seqlen_override

        rotary_cos, rotary_sin = encoder.rot_pos_emb(grid_thw)
        metadata = {
            "rotary_pos_emb_cos": rotary_cos,
            "rotary_pos_emb_sin": rotary_sin,
            "cu_seqlens": async_tensor_h2d(cu_seqlens, self.device),
            # Attention consumes this scalar on the host during capture. A CPU
            # tensor avoids a GPU -> CPU transfer; replay uses the capture bound.
            "max_seqlen": torch.tensor(max_seqlen, dtype=torch.int32),
        }
        if encoder.apply_vit_abs_pos_embed:
            metadata["pos_embeds"] = encoder.fast_pos_embed_interpolate(grid_thw.tolist())
        sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(encoder.attn_backend, cu_seqlens, self.device)
        if sequence_lengths is not None:
            metadata["sequence_lengths"] = sequence_lengths
        return metadata

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        import re

        def _remap_merger_list_inner(name: str) -> str:
            name = re.sub(r"(merger_list\.\d+)\.norm\.", r"\1.ln_q.", name)
            name = re.sub(r"(merger_list\.\d+)\.linear_fc1\.", r"\1.mlp.0.", name)
            name = re.sub(r"(merger_list\.\d+)\.linear_fc2\.", r"\1.mlp.2.", name)

            return name

        remapped_weights = self.hf_to_vllm_mapper.apply(weights)
        remapped_weights = ((_remap_merger_list_inner(name), tensor) for name, tensor in remapped_weights)
        loaded_params = self.encoder.load_weights(remapped_weights)

        loaded_params = {f"encoder.{loaded_param}" for loaded_param in loaded_params}

        return loaded_params
