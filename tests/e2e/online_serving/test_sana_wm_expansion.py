# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
L2 online expansion test for SANA-WM I2V.

Coverage (H100, 1 GPU):
- Basic I2V via /v1/videos with sana_wm form field and base64 first frame.
"""

import json

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

MODEL = "Efficient-Large-Model/SANA-WM_bidirectional"
SANA_WM_STAGE_CONFIG = get_deploy_config_path("sana_wm.yaml")

PROMPT = "A mountain range at dusk, cinematic aerial shot."
NEGATIVE_PROMPT = "low quality, blurry, distorted, watermark"

_H, _W = 480, 848
_NUM_FRAMES = 9

SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_sana_wm_cases(model: str, stage_config: str):
    return [
        pytest.param(
            OmniServerParams(model=model, stage_config_path=stage_config),
            id="single_card_i2v",
            marks=SINGLE_CARD_MARKS,
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_sana_wm_cases(MODEL, SANA_WM_STAGE_CONFIG),
    indirect=True,
)
def test_sana_wm_i2v_online(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    """Online I2V: /v1/videos returns completed video for a single SANA-WM request."""
    first_frame_b64 = generate_synthetic_image(_W, _H)["base64"]

    form_data = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "height": _H,
        "width": _W,
        "num_frames": _NUM_FRAMES,
        "num_inference_steps": 2,
        "seed": 42,
        "sana_wm": json.dumps({
            "action": "w-8",
            "num_frames": _NUM_FRAMES,
            "height": _H,
            "width": _W,
        }),
    }

    request_config = {
        "model": omni_server.model,
        "form_data": form_data,
        "image_reference": f"data:image/jpeg;base64,{first_frame_b64}",
    }

    openai_client.send_video_diffusion_request(request_config)
