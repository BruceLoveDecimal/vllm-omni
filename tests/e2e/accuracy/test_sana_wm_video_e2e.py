# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os

import pytest


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.diffusion,
    pytest.mark.gpu,
    pytest.mark.skipif(os.environ.get("SANA_WM_E2E") != "1", reason="Set SANA_WM_E2E=1 to run Sana-WM GPU e2e."),
]


def test_sana_wm_official_backend_generates_video() -> None:
    import numpy as np
    import torch
    from PIL import Image

    from vllm_omni.diffusion.models.sana_wm import (
        SANA_WM_MODEL_ID,
        SANA_WM_OFFICIAL_REPO_ENV,
        SANA_WM_OUTPUT_HEIGHT,
        SANA_WM_OUTPUT_WIDTH,
    )
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput

    if not torch.cuda.is_available():
        pytest.skip("Sana-WM e2e requires CUDA.")
    if not os.environ.get(SANA_WM_OFFICIAL_REPO_ENV):
        pytest.skip(f"Set {SANA_WM_OFFICIAL_REPO_ENV} to an NVlabs/Sana checkout.")

    model = os.environ.get("SANA_WM_E2E_MODEL", SANA_WM_MODEL_ID)
    model_class_name = os.environ.get("SANA_WM_E2E_MODEL_CLASS", "SanaWmPipeline")
    num_frames = int(os.environ.get("SANA_WM_E2E_NUM_FRAMES", "9"))
    image = Image.new("RGB", (SANA_WM_OUTPUT_WIDTH, SANA_WM_OUTPUT_HEIGHT), (96, 128, 160))
    omni = Omni(
        model=model,
        model_class_name=model_class_name,
        enforce_eager=True,
    )
    output = omni.generate(
        {
            "prompt": "A slow forward camera move through a quiet city street.",
            "multi_modal_data": {"image": image},
            "sana_wm": {
                "action": "w-16",
                "num_frames": num_frames,
                "translation_speed": 0.055,
                "rotation_speed_deg": 1.2,
                # Avoid the optional Pi3X dependency in the official runner by
                # passing deterministic camera intrinsics directly.
                "intrinsics": {
                    "fx": SANA_WM_OUTPUT_WIDTH / 2,
                    "fy": SANA_WM_OUTPUT_WIDTH / 2,
                    "cx": SANA_WM_OUTPUT_WIDTH / 2,
                    "cy": SANA_WM_OUTPUT_HEIGHT / 2,
                },
            },
        },
        OmniDiffusionSamplingParams(
            height=SANA_WM_OUTPUT_HEIGHT,
            width=SANA_WM_OUTPUT_WIDTH,
            num_frames=num_frames,
            seed=0,
            fps=16,
            num_inference_steps=1,
            guidance_scale=1.0,
            guidance_scale_provided=True,
            extra_args={"sana_wm_sampling_algo": "flow_euler_ltx", "sana_wm_offload_vae": True},
        ),
    )

    request_output = output[0] if isinstance(output, list) else output
    assert isinstance(request_output, OmniRequestOutput)
    assert request_output.error is None
    assert request_output.images

    frames = request_output.images[0]
    if isinstance(frames, list):
        frames = frames[0]
    frames = np.asarray(frames)
    assert frames.ndim == 4
    assert 0 < frames.shape[0] <= num_frames
    assert frames.shape[-1] in (3, 4)
