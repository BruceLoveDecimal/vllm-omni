# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Released SolarWM checkpoint smoke using the shared Omni test client."""

import os

import numpy as np
import pytest
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = os.environ.get("SOLARWM_MODEL")
IMAGE = os.environ.get("SOLARWM_IMAGE")
pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.skipif(not MODEL or not IMAGE, reason="Set SOLARWM_MODEL (assembled index) and SOLARWM_IMAGE"),
    pytest.mark.parametrize(
        "omni_runner",
        [(MODEL or "SolarWM-Omni", None, {"enforce_eager": True, "attention_backend": "TORCH_SDPA"})],
        indirect=True,
    ),
]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_solarwm_window_rollout(omni_runner_handler: OmniRunnerHandler):
    # 93 pixel frames cover 24 latents, including window eviction at chunk 7.
    image = Image.open(IMAGE).convert("RGB")
    response = omni_runner_handler.send_diffusion_request(
        {
            "prompt": "A cinematic view with gentle natural motion.",
            "modalities": ["video"],
            "images": image,
            "sampling_params": OmniDiffusionSamplingParams(
                height=480,
                width=864,
                num_frames=93,
                num_inference_steps=4,
                guidance_scale=1.0,
                seed=42,
                fps=16,
                output_type="np",
            ),
        }
    )
    assert response.success and response.images
    video = np.asarray(response.images[0]).reshape(-1, 480, 864, 3)
    assert video.shape[0] == 93
    assert np.isfinite(video).all() and np.ptp(video) > 0
    assert np.mean(np.abs(np.diff(video.astype(np.float32), axis=0))) > 0
