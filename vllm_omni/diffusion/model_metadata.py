# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusionModelMetadata:
    # Keep serving-facing capability metadata in a lightweight shared module so
    # config/model plumbing can read it without importing concrete pipelines.
    supports_multimodal_inputs: bool = False
    max_multimodal_image_inputs: int | None = None


QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES = 4
MAGE_FLOW_MAX_INPUT_IMAGES = 3
# Upstream HunyuanImage-3.0 "Multi-Image Fusion" caps reference images at 3.
HUNYUAN_IMAGE3_MAX_INPUT_IMAGES = 3
# Boogu-Image editing (TI2I) supports a single reference image for now.
BOOGU_IMAGE_MAX_INPUT_IMAGES = 1

# Request-scoped controls MageFlowPipeline accepts through
# OmniDiffusionSamplingParams.extra_args. This lives here rather than in
# model_extras so that the entrypoint layer (which advertises the keys) and the
# pipeline (which rejects everything else) cannot drift apart; a drift would let
# a key pass request validation and then fail mid-request.
MAGE_FLOW_EXTRA_BODY_PARAMS = frozenset(
    {
        "mage_enable_safety_check",
        "mage_enable_watermark",
        "mage_vision_long_edge",
    }
)


_DIFFUSION_MODEL_METADATA: dict[str, DiffusionModelMetadata] = {
    "MageFlowPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=MAGE_FLOW_MAX_INPUT_IMAGES,
    ),
    "QwenImageEditPlusPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES,
    ),
    "HunyuanImage3Pipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=HUNYUAN_IMAGE3_MAX_INPUT_IMAGES,
    ),
    # Shared by the Base (text-to-image) and Edit (TI2I) checkpoints, which use
    # the same ``BooguImagePipeline`` class. Text-to-image requests simply carry
    # no reference image.
    "BooguImagePipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=BOOGU_IMAGE_MAX_INPUT_IMAGES,
    ),
}


def get_diffusion_model_metadata(model_class_name: str | None) -> DiffusionModelMetadata:
    # Unknown models fall back to "no special multimodal capabilities" so new
    # pipelines do not accidentally inherit limits meant for other models.
    if model_class_name is None:
        return DiffusionModelMetadata()
    return _DIFFUSION_MODEL_METADATA.get(model_class_name, DiffusionModelMetadata())
