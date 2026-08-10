# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Online serving smoke test for the native Mage-Flow T2I pipeline."""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)

MODEL = "microsoft/Mage-Flow-Turbo"
PROMPT = "A small red fox sitting in fresh snow, soft morning light."
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
def test_mage_flow_t2i(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    request_config = {
        "model": omni_server.model,
        "messages": dummy_messages_from_mix_data(content_text=PROMPT),
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "seed": 42,
        },
    }
    openai_client.send_diffusion_request(request_config)
