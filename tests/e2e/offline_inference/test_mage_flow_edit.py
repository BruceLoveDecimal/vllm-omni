# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline inference smoke test for native multi-reference Mage-Flow Edit."""

import pytest
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "microsoft/Mage-Flow-Edit-Turbo"


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_mage_flow_edit(
    omni_runner_handler: OmniRunnerHandler,
) -> None:
    reference_images = [
        Image.fromarray(generate_synthetic_image(512, 512)["np_array"]),
        Image.fromarray(generate_synthetic_image(384, 512)["np_array"]),
    ]
    sampling = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=2,
        guidance_scale=1.0,
        seed=42,
        extra_args={
            "mage_enable_safety_check": False,
            "mage_enable_watermark": False,
            "mage_vision_long_edge": 384,
        },
    )
    omni_runner_handler.send_diffusion_request(
        {
            "model": MODEL,
            "prompt": "Use the subject from image one in the setting from image two.",
            "multi_modal_data": {"image": reference_images},
            "sampling_params": sampling,
        }
    )
