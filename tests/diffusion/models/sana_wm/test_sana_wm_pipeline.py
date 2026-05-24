# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_sana_wm_stage1_pipeline_declares_components() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmPipeline

    assert SanaWmPipeline._dit_modules == ["transformer"]
    assert SanaWmPipeline._encoder_modules == ["text_encoder", "camera_encoder"]
    assert SanaWmPipeline._vae_modules == ["vae"]


def test_sana_wm_stage1_preprocess_accepts_action() -> None:
    from vllm_omni.diffusion.models.sana_wm import get_sana_wm_pre_process_func

    request = SimpleNamespace(
        prompts=[
            {
                "prompt": "drive forward",
                "multi_modal_data": {"image": object()},
                "sana_wm": {"action": "w-1", "num_frames": 1},
            }
        ],
        sampling_params=SimpleNamespace(height=64, width=64, num_frames=1),
    )

    get_sana_wm_pre_process_func(SimpleNamespace())(request)
    payload = request.prompts[0]["additional_information"]["sana_wm"]

    assert payload["action"] == "w-1"
    assert payload["num_frames"] == 1
