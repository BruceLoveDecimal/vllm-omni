# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.diffusion,
    pytest.mark.gpu,
    pytest.mark.skipif(os.environ.get("SANA_WM_E2E") != "1", reason="Set SANA_WM_E2E=1 to run Sana-WM GPU e2e."),
]


def _coerce_video_array(video: Any) -> np.ndarray:
    import torch

    if isinstance(video, list):
        assert video, "SANA-WM e2e produced an empty video list."
        video = video[0]
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().float().numpy()
    video_array = np.asarray(video)
    if video_array.ndim == 5:
        assert video_array.shape[0] == 1
        video_array = video_array[0]
    assert video_array.ndim == 4
    return video_array


def _run_sana_wm_e2e(
    *,
    inprocess_refiner: bool,
    output_type: str,
    refiner_steps: int,
) -> np.ndarray:
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
    extra_args = {"sana_wm_sampling_algo": "flow_euler_ltx", "sana_wm_offload_vae": True}
    if inprocess_refiner:
        extra_args.update(
            {
                "sana_wm_inprocess_refiner": True,
                "sana_wm_refiner_output_type": output_type,
                "sana_wm_inprocess_refiner_steps": refiner_steps,
            }
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
            extra_args=extra_args,
        ),
    )

    request_output = output[0] if isinstance(output, list) else output
    assert isinstance(request_output, OmniRequestOutput)
    assert request_output.error is None
    assert request_output.images

    frames = request_output.images[0]
    video = _coerce_video_array(frames)
    assert 0 < video.shape[0] <= num_frames
    return video


def _assert_video_reference_alignment(
    *,
    prediction: np.ndarray,
    reference: np.ndarray,
    max_mean_abs_error: float,
) -> None:
    assert prediction.shape == reference.shape
    prediction_f = prediction.astype(np.float32)
    reference_f = reference.astype(np.float32)
    prediction_max = float(np.max(prediction_f)) if prediction_f.size else 0.0
    reference_max = float(np.max(reference_f)) if reference_f.size else 0.0
    if prediction_max <= 1.0 and reference_max > 1.0:
        prediction_f *= 255.0
    if reference_max <= 1.0 and prediction_max > 1.0:
        reference_f *= 255.0
    mean_abs_error = float(np.mean(np.abs(prediction_f - reference_f)))
    print(f"SANA-WM reference-alignment MAE={mean_abs_error:.6f}, threshold<={max_mean_abs_error:.6f}")
    assert mean_abs_error <= max_mean_abs_error


def test_sana_wm_official_backend_generates_video() -> None:
    output_type = os.environ.get("SANA_WM_E2E_OUTPUT_TYPE", "np")
    refiner_steps = int(os.environ.get("SANA_WM_E2E_REFINER_STEPS", "1"))
    video = _run_sana_wm_e2e(
        inprocess_refiner=os.environ.get("SANA_WM_E2E_INPROCESS_REFINER") == "1",
        output_type=output_type,
        refiner_steps=refiner_steps,
    )
    if output_type == "latent":
        assert video.ndim == 4
    else:
        assert video.shape[-1] in (3, 4)


def test_sana_wm_inprocess_refiner_aligns_with_official_bridge() -> None:
    if os.environ.get("SANA_WM_E2E_REFERENCE_ALIGNMENT") != "1":
        pytest.skip("Set SANA_WM_E2E_REFERENCE_ALIGNMENT=1 to run Sana-WM reference alignment.")

    refiner_steps = int(os.environ.get("SANA_WM_E2E_REFINER_STEPS", "1"))
    official = _run_sana_wm_e2e(inprocess_refiner=False, output_type="np", refiner_steps=refiner_steps)
    inprocess = _run_sana_wm_e2e(inprocess_refiner=True, output_type="np", refiner_steps=refiner_steps)
    _assert_video_reference_alignment(
        prediction=inprocess,
        reference=official,
        max_mean_abs_error=float(os.environ.get("SANA_WM_E2E_REFERENCE_MAX_MAE", "255.0")),
    )
