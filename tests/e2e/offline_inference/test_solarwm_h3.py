# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline SolarWM-H3 smoke: two causal chunks from a synthetic first frame.

The 33B checkpoint is gated and 150 GB, so the test reads its location from
``SOLARWM_H3_E2E_MODEL`` (the repository root holding ``SolarWM-h3-33B-base``
and ``SolarWM-h3-33B-sgf-stage2-158f``) and skips when it is absent.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.advanced_model,
    pytest.mark.diffusion,
    pytest.mark.gpu,
]

# 22 frames is the shortest valid request: two five-latent chunks, seven
# decoded latents, one window with history.
_E2E_NUM_FRAMES = 22


def _static_camera(num_frames: int) -> list[list[list[float]]]:
    return np.repeat(np.eye(4, dtype=np.float32)[None], num_frames, axis=0).tolist()


def _coerce_video_array(video: Any) -> np.ndarray:
    import torch

    if isinstance(video, list):
        assert video, "SolarWM-H3 e2e produced an empty video list."
        video = video[0]
    if isinstance(video, dict):
        video = video["video"]
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().float().numpy()
    video_array = np.asarray(video)
    if video_array.ndim == 5:
        assert video_array.shape[0] == 1
        video_array = video_array[0]
    assert video_array.ndim == 4
    return video_array


def _run_solarwm_h3_e2e(*, output_type: str) -> np.ndarray:
    import torch
    from PIL import Image

    from vllm_omni.diffusion.models.solarwm_h3.layout import (
        SOLARWM_H3_PIXEL_HEIGHT,
        SOLARWM_H3_PIXEL_WIDTH,
    )
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput

    if not torch.cuda.is_available():
        pytest.skip("SolarWM-H3 e2e requires CUDA.")
    model = os.environ.get("SOLARWM_H3_E2E_MODEL")
    if not model:
        pytest.skip("Set SOLARWM_H3_E2E_MODEL to the SolarWM-H3-33B repository root.")

    num_frames = int(os.environ.get("SOLARWM_H3_E2E_NUM_FRAMES", str(_E2E_NUM_FRAMES)))
    omni = Omni(
        model=model,
        enforce_eager=True,
        tensor_parallel_size=int(os.environ.get("SOLARWM_H3_E2E_TP", "1")),
        # The Qwen3-VL encoder does not fit next to the 33B DiT on one 96 GiB GPU.
        diffusion_offload_config={"mode": "layer", "components": ["text_encoder"]},
    )
    image = Image.new("RGB", (SOLARWM_H3_PIXEL_WIDTH, SOLARWM_H3_PIXEL_HEIGHT), (96, 128, 160))
    output = omni.generate(
        {
            "prompt": "A calm blue-grey wall under soft light, the camera holds still.",
            "multi_modal_data": {"image": image},
        },
        OmniDiffusionSamplingParams(
            height=SOLARWM_H3_PIXEL_HEIGHT,
            width=SOLARWM_H3_PIXEL_WIDTH,
            num_frames=num_frames,
            fps=24,
            seed=0,
            output_type=output_type,
            extra_args={"camera_c2w": _static_camera(num_frames)},
        ),
    )

    request_output = output[0] if isinstance(output, list) else output
    assert isinstance(request_output, OmniRequestOutput)
    assert request_output.error is None
    assert request_output.images
    return _coerce_video_array(request_output.images[0])


def test_solarwm_h3_generates_latents() -> None:
    from vllm_omni.diffusion.models.solarwm_h3.layout import (
        SOLARWM_H3_LATENT_CHANNELS,
        SOLARWM_H3_LATENT_HEIGHT,
        SOLARWM_H3_LATENT_WIDTH,
        rollout_geometry,
    )

    num_frames = int(os.environ.get("SOLARWM_H3_E2E_NUM_FRAMES", str(_E2E_NUM_FRAMES)))
    latents = _run_solarwm_h3_e2e(output_type="latent")
    expected = (
        SOLARWM_H3_LATENT_CHANNELS,
        rollout_geometry(num_frames).decode_latents,
        SOLARWM_H3_LATENT_HEIGHT,
        SOLARWM_H3_LATENT_WIDTH,
    )
    assert latents.shape == expected
    assert np.isfinite(latents).all()


def test_solarwm_h3_generates_video() -> None:
    num_frames = int(os.environ.get("SOLARWM_H3_E2E_NUM_FRAMES", str(_E2E_NUM_FRAMES)))
    video = _run_solarwm_h3_e2e(output_type="np")
    assert video.shape[0] == num_frames
    assert video.shape[-1] == 3
