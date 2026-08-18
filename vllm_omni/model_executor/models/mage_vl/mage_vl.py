# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mage-VL: codec-native Mage-ViT vision tower + Qwen3-4B causal decoder.

The decoder is stock Qwen3 and is instantiated through
``init_vllm_registered_model``; only the vision tower is model-specific (see
``vision.py``). Visual features are spliced into the text embedding stream at
``<|image_pad|>`` positions, and positions are plain 1-D -- the checkpoint's
``text_config`` carries no ``mrope_section``, so no M-RoPE plumbing is involved.

The reference implementation routes video through the *image* tensors
(``pixel_values`` / ``image_grid_thw``) with one expanded block per frame, so this port
keeps a single visual path and treats video frames as images.
"""

from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Annotated, Any, Literal

import torch
import torch.nn as nn
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from vllm_omni.model_executor.models.mage_vl.vision import MageVLVisionTransformer
from vllm_omni.transformers_utils.configs.mage_vl import MageVLConfig
from vllm_omni.transformers_utils.processors.mage_vl import MageVLProcessor


class MageVLImagePixelInputs(TensorSchema):
    """
    Dimensions:
        np: total patch count across the batch
        cpp: flattened patch payload, ``num_channels * patch_size * patch_size``
        ni: number of images

    The Qwen2-VL image processor emits patches already flattened, and the tower's patch
    embedding reshapes them back, so the schema keeps the flat form.
    """

    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, TensorShape("np", "cpp")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]
    patch_positions: Annotated[torch.Tensor, TensorShape("np", 3)]


class MageVLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> MageVLConfig:
        return self.ctx.get_hf_config(MageVLConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_hf_processor(self, **kwargs: object) -> "MageVLProcessor":
        # The checkpoint's own processor is not a ``ProcessorMixin``, which vLLM's
        # processor cache requires, so the vendored one is requested explicitly.
        return self.ctx.get_hf_processor(MageVLProcessor, **kwargs)

    def get_image_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).image_processor


class MageVLDummyInputsBuilder(BaseDummyInputsBuilder[MageVLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<|vision_start|><|image_pad|><|vision_end|>" * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ):
        num_images = mm_counts.get("image", 0)
        target_width, target_height = 448, 448
        return {
            "image": self._get_dummy_images(
                width=target_width, height=target_height, num_images=num_images
            )
        }


class MageVLMultiModalProcessor(BaseMultiModalProcessor[MageVLProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed = super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)
        if "pixel_values" in processed and "patch_positions" not in processed:
            from vllm_omni.transformers_utils.processors.mage_vl import (
                build_patch_positions,
            )

            merge_size = self.info.get_hf_config().vision_config.spatial_merge_size
            processed["patch_positions"] = build_patch_positions(
                processed["image_grid_thw"], spatial_merge_size=merge_size
            )
        return processed

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        # pixel_values and patch_positions are flat per-patch tables, so both are sliced
        # by the same per-image patch counts.
        patch_sizes = grid_thw.prod(-1)
        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes("image", patch_sizes),
            "patch_positions": MultiModalFieldConfig.flat_from_sizes("image", patch_sizes),
            "image_grid_thw": MultiModalFieldConfig.batched("image"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        config = self.info.get_hf_config()
        merge_length = config.vision_config.spatial_merge_size**2

        def get_replacement(item_idx: int):
            grid_thw = out_mm_kwargs["image"][item_idx]["image_grid_thw"].data
            num_tokens = int(grid_thw.prod()) // merge_length
            return [config.image_token_id] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[config.image_token_id],
                replacement=get_replacement,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    MageVLMultiModalProcessor,
    info=MageVLProcessingInfo,
    dummy_inputs=MageVLDummyInputsBuilder,
)
class MageVLForConditionalGeneration(nn.Module, SupportsMultiModal):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    merge_by_field_config = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: MageVLConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.visual = MageVLVisionTransformer(
            config.vision_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen3ForCausalLM"],
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_image_input(self, **kwargs: object) -> MageVLImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None
        return MageVLImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_thw=kwargs.pop("image_grid_thw"),
            patch_positions=kwargs.pop("patch_positions"),
        )

    def _process_image_input(self, image_input: MageVLImagePixelInputs) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        embeds = self.visual(
            pixel_values,
            grid_thw=grid_thw,
            patch_positions=image_input["patch_positions"],
        )
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // (merge_size**2)).tolist()
        return embeds.split(sizes)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_image_input(**kwargs)
        if image_input is None:
            return []
        return self._process_image_input(image_input)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Positions are plain 1-D: the checkpoint's text config has no mrope section."""
        if intermediate_tensors is not None:
            inputs_embeds = None

        return self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
