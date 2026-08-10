# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline inference smoke test for the native Mage-Flow T2I pipeline."""

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "microsoft/Mage-Flow-Turbo"
PROMPT = "A small red fox sitting in fresh snow, soft morning light."


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_mage_flow_t2i(
    omni_runner_handler: OmniRunnerHandler,
) -> None:
    sampling = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=2,
        guidance_scale=1.0,
        seed=42,
        extra_args={
        },
    )
    omni_runner_handler.send_diffusion_request(
        {
            "model": MODEL,
            "prompt": PROMPT,
            "sampling_params": sampling,
        }
    )
