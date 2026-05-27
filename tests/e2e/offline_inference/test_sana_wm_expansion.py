# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
L2 offline expansion test for SANA-WM I2V.

send_diffusion_request does not thread the sana_wm key through to the prompt
dict, so this test calls runner.generate() directly with the full prompt dict.
"""

import json

import pytest
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner, OmniRunnerHandler
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "Efficient-Large-Model/SANA-WM_bidirectional"
SANA_WM_STAGE_CONFIG = get_deploy_config_path("sana_wm.yaml")

PROMPT = "A mountain range at dusk, cinematic aerial shot."
NEGATIVE_PROMPT = "low quality, blurry, distorted, watermark"

_H, _W = 480, 848
_NUM_FRAMES = 9
_NUM_STEPS = 2

_OMNI_RUNNER_PARAM = (MODEL, SANA_WM_STAGE_CONFIG)


def _synthetic_first_frame(height: int, width: int) -> Image.Image:
    return Image.new("RGB", (width, height), color=(100, 140, 180))


pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_sana_wm_i2v_single_card(omni_runner_handler: OmniRunnerHandler) -> None:
    """Single-GPU I2V: pipeline runs end-to-end and returns video frames."""
    first_frame = _synthetic_first_frame(_H, _W)
    prompt = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "multi_modal_data": {"image": first_frame},
        "sana_wm": {
            "action": "w-8",
            "num_frames": _NUM_FRAMES,
            "height": _H,
            "width": _W,
        },
    }
    sampling_params = OmniDiffusionSamplingParams(
        height=_H,
        width=_W,
        num_frames=_NUM_FRAMES,
        num_inference_steps=_NUM_STEPS,
        seed=42,
    )
    outputs = omni_runner_handler.runner.generate([prompt], [sampling_params])
    assert outputs, "generate() returned empty list"
    assert outputs[0].images, "No frames in output"
