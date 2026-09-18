# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-side adapter for vLLM's encoder CUDA graph manager.

The thinker and its registered stage wrapper share this protocol implementation.
Graph ownership, budget packing, output cloning and cache insertion stay in vLLM.
"""

from __future__ import annotations

from collections.abc import Hashable
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from vllm.v1.worker.encoder_cudagraph_defs import (
        EncoderCudaGraphCaptureInputs,
        EncoderCudaGraphConfig,
        EncoderCudaGraphReplayBuffers,
        EncoderItemSpec,
    )

    from .ming_flash_omni_thinker import MingFlashOmniThinkerForConditionalGeneration


def pad_cumulative_lengths(destination: torch.Tensor, source: torch.Tensor, *, spatial_merge_unit: int) -> None:
    """Put unused patches in an isolated padding attention sequence.

    Every patch must belong to a sequence, including the padded tail, so the
    attention backend never leaves output rows uninitialized. Each graph has
    token_budget + 2 offsets: up to token_budget real sequences plus padding.
    Derive the terminal offset from this static capacity, without reading a
    GPU scalar. Remaining offsets repeat that terminal value (empty sequences).
    """
    if source.numel() >= destination.numel():
        raise ValueError("Ming encoder attention sequences exceed the graph token budget")
    total_patches = (destination.numel() - 2) * spatial_merge_unit
    destination.fill_(total_patches)
    destination[: source.shape[0]].copy_(source)


class MingVisionCudaGraphMixin(SupportsEncoderCudaGraph):
    """Capture the final visual embeddings consumed by Ming's language model."""

    @property
    def encoder_cudagraph_model(self) -> MingFlashOmniThinkerForConditionalGeneration:
        """Resolve the thinker for both direct and wrapped stage models."""
        raise NotImplementedError

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

        thinker = self.encoder_cudagraph_model
        if thinker.vision.encoder.attn_backend != AttentionBackendEnum.FLASH_ATTN:
            raise ValueError("Ming vision CUDA graphs currently require mm_encoder_attn_backend=FLASH_ATTN")
        mm_config = thinker.model_config.multimodal_config
        if mm_config is not None and mm_config.mm_encoder_tp_mode == "data":
            # Qwen3-Omni's MLP still uses TP linear layers. Sharding images over
            # those ranks would mix unrelated items in the collective reduction.
            raise ValueError("Ming vision CUDA graphs require mm_encoder_tp_mode=weights")

        buffer_keys = ["pixel_values", "rotary_pos_emb_cos", "rotary_pos_emb_sin", "cu_seqlens", "max_seqlen"]
        if thinker.vision.encoder.apply_vit_abs_pos_embed:
            buffer_keys.append("pos_embeds")
        return EncoderCudaGraphConfig(
            modalities=["image", "video"],
            buffer_keys=buffer_keys,
            out_hidden_size=thinker.config.hidden_size,
            padding_logics={
                "cu_seqlens": partial(
                    pad_cumulative_lengths, spatial_merge_unit=thinker.vision.encoder.spatial_merge_size**2
                )
            },
            max_frames_per_video=self.get_max_frames_per_video(),
        )

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        if "image_grid_thw" in mm_kwargs:
            return "image"
        if "video_grid_thw" in mm_kwargs:
            return "video"
        raise ValueError("Ming vision CUDA graph inputs require image_grid_thw or video_grid_thw")

    def get_max_frames_per_video(self) -> int:
        # One temporal grid contributes at least one merged token. This is a
        # conservative bound; per-graph sequence storage uses its token budget.
        return self.encoder_cudagraph_model.model_config.max_model_len

    def get_encoder_cudagraph_budget_range(self, vllm_config: VllmConfig) -> tuple[int, int]:
        # Budgets count merged output tokens, not raw input patches. Keep the
        # default capture set small enough for deployment without requiring
        # Ming-specific tuning in YAML. With the normal 2048-token maximum,
        # upstream infers [512, 1024, 2048] and at most four packed items.
        runtime_limit = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.model_config.max_model_len,
        )
        max_budget = min(2048, runtime_limit)
        min_budget = min(512, max_budget)
        return min_budget, max_budget

    def get_encoder_cudagraph_inputs(self, mm_kwargs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Read flat pixels and host grids without synchronizing the GPU."""
        modality = self.get_input_modality(mm_kwargs)
        grid = mm_kwargs[f"{modality}_grid_thw"]
        if not isinstance(grid, torch.Tensor) or grid.device.type != "cpu":
            raise ValueError("Ming encoder CUDA graphs require CPU grid tensors (keep_on_cpu=True)")
        if grid.ndim != 2 or grid.shape[1] != 3:
            raise ValueError("Ming encoder grid must have shape [num_items, 3]")
        merge_size = self.encoder_cudagraph_model.vision.encoder.spatial_merge_size
        for t, h, w in grid.tolist():
            if t <= 0 or h <= 0 or w <= 0 or h % merge_size or w % merge_size:
                raise ValueError(
                    "Ming grids require positive dimensions and spatial dimensions divisible by merge size"
                )
        pixel_key = "pixel_values"
        if modality == "video":
            pixel_key = "pixel_values_videos"
        return mm_kwargs[pixel_key], grid

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, Any]) -> list[EncoderItemSpec]:
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderItemSpec

        _, grid = self.get_encoder_cudagraph_inputs(mm_kwargs)
        merge_size = self.encoder_cudagraph_model.vision.encoder.spatial_merge_size
        return [
            EncoderItemSpec(input_size=t * h * w, output_tokens=t * (h // merge_size) * (w // merge_size))
            for t, h, w in grid.tolist()
        ]

    def select_encoder_cudagraph_items(self, mm_kwargs: dict[str, Any], indices: list[int]) -> dict[str, Any]:
        pixels, grid = self.get_encoder_cudagraph_inputs(mm_kwargs)
        offsets = [0]
        for t, h, w in grid.tolist():
            offsets.append(offsets[-1] + t * h * w)
        selected_pixels = pixels[:0]
        if indices:
            selected_pixels = torch.cat([pixels[offsets[i] : offsets[i + 1]] for i in indices])
        modality = self.get_input_modality(mm_kwargs)
        if modality == "video":
            return {"pixel_values_videos": selected_pixels, "video_grid_thw": grid[indices]}
        return {"pixel_values": selected_pixels, "image_grid_thw": grid[indices]}

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ) -> EncoderCudaGraphCaptureInputs:
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphCaptureInputs

        vision = self.encoder_cudagraph_model.vision
        encoder = vision.encoder
        patch_count = token_budget * encoder.spatial_merge_size**2
        patch_embed = encoder.patch_embed
        patch_width = patch_embed.proj.in_channels * patch_embed.temporal_patch_size * patch_embed.patch_size**2

        # The dummy layout need not encode a real image. Allocate by budget
        # directly, avoiding synthetic skinny grids that exceed the RoPE cache.
        merge_size = encoder.spatial_merge_size
        seed_grid = torch.tensor([[1, merge_size, merge_size]], dtype=torch.int64)
        metadata = vision.prepare_encoder_metadata(seed_grid)
        values = {
            "pixel_values": torch.zeros(patch_count, patch_width, device=device, dtype=vision.dtype),
            # Each frame needs >= 1 merged token, so token_budget bounds the
            # number of sequences even for many tiny frames. The upstream
            # packer budgets tokens/items, not frames. Do not underallocate
            # from max_frames_per_batch and then overflow on a legal replay.
            "cu_seqlens": async_tensor_h2d([0] + [patch_count] * (token_budget + 1), device=device, dtype=torch.int32),
            "max_seqlen": torch.tensor(patch_count, dtype=torch.int32),
        }
        for key in ("pos_embeds", "rotary_pos_emb_cos", "rotary_pos_emb_sin"):
            template = metadata.get(key)
            if template is not None:
                values[key] = torch.zeros((patch_count, *template.shape[1:]), device=device, dtype=template.dtype)
        values["rotary_pos_emb_cos"].fill_(1)
        return EncoderCudaGraphCaptureInputs(values=values)

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphReplayBuffers

        pixels, grid = self.get_encoder_cudagraph_inputs(mm_kwargs)
        metadata = self.encoder_cudagraph_model.vision.prepare_encoder_metadata(grid)
        # max_seqlen is a host launch argument baked into the graph. Preserve
        # the capture-time upper bound instead of copying the request's value.
        del metadata["max_seqlen"]
        return EncoderCudaGraphReplayBuffers(values={"pixel_values": pixels, **metadata})

    def encoder_cudagraph_forward(self, inputs: dict[str, torch.Tensor], path: str = "default") -> torch.Tensor:
        return self.encoder_cudagraph_model.extract_image_feature(inputs["pixel_values"], encoder_metadata=inputs)

    def encoder_eager_forward(self, mm_kwargs: dict[str, Any], path: str = "default") -> torch.Tensor:
        pixels, grid = self.get_encoder_cudagraph_inputs(mm_kwargs)
        return self.encoder_cudagraph_model.extract_image_feature(pixels, grid)
