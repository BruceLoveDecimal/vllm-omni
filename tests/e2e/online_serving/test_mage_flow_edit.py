# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Online serving smoke test for native Mage-Flow Edit."""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)

MODEL = "microsoft/Mage-Flow-Edit-Turbo"
SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(model=MODEL),
            id="turbo",
            marks=SINGLE_CARD_MARKS,
        )
    ],
    indirect=True,
)
def test_mage_flow_edit(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    image_data_url = "data:image/jpeg;base64," + generate_synthetic_image(512, 512)["base64"]
    request_config = {
        "model": omni_server.model,
        "messages": dummy_messages_from_mix_data(
            image_data_url=image_data_url,
            content_text="Replace the background with a quiet snowy forest.",
        ),
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "seed": 42,
        },
    }
    openai_client.send_diffusion_request(request_config)
