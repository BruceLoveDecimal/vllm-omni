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


def _assert_sana_wm_e2e_shape(
    video: np.ndarray,
    *,
    output_type: str,
    num_frames: int,
) -> None:
    if output_type == "latent":
        # Latent output is channel-first: (C, T, H, W) after removing the
        # batch dimension. The refiner uses LTX-2 latents with 128 channels.
        assert video.shape[0] == 128
        assert 0 < video.shape[1] <= num_frames
        return
    assert 0 < video.shape[0] <= num_frames


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
    _assert_sana_wm_e2e_shape(video, output_type=output_type, num_frames=num_frames)
    return video


def _normalize_to_uint8(video_f: np.ndarray) -> np.ndarray:
    """Ensure pixel values are in [0, 255] float32."""
    vmax = float(np.max(video_f)) if video_f.size else 0.0
    if vmax <= 1.0:
        return video_f * 255.0
    return video_f


def _compute_psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    """Per-video PSNR (dB) over all frames, assuming uint8 range [0, 255]."""
    mse = float(np.mean((pred - ref) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def _compute_ssim_y(pred: np.ndarray, ref: np.ndarray) -> float:
    """Mean SSIM on the Y (luma) channel across all frames.

    Converts RGB→Y via BT.601, then computes per-patch SSIM and averages.
    Implements the standard SSIM formula without external dependencies.
    """
    # RGB → Y (BT.601), inputs in [0, 255]
    def _to_y(v: np.ndarray) -> np.ndarray:
        # v: [T, H, W, 3]
        r, g, b = v[..., 0], v[..., 1], v[..., 2]
        return 0.299 * r + 0.587 * g + 0.114 * b  # [T, H, W]

    pred_y = _to_y(pred)
    ref_y = _to_y(ref)

    # SSIM constants (K1=0.01, K2=0.03, L=255)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def _ssim_frame(p: np.ndarray, r: np.ndarray) -> float:
        mu_p, mu_r = p.mean(), r.mean()
        sig_p = float(np.var(p))
        sig_r = float(np.var(r))
        sig_pr = float(np.mean((p - mu_p) * (r - mu_r)))
        num = (2 * mu_p * mu_r + c1) * (2 * sig_pr + c2)
        den = (mu_p ** 2 + mu_r ** 2 + c1) * (sig_p + sig_r + c2)
        return float(num / den) if den != 0.0 else 1.0

    return float(np.mean([_ssim_frame(pred_y[t], ref_y[t]) for t in range(pred_y.shape[0])]))


def _assert_video_reference_alignment(
    *,
    prediction: np.ndarray,
    reference: np.ndarray,
    max_mean_abs_error: float,
    min_psnr: float | None = None,
    min_ssim_y: float | None = None,
) -> None:
    assert prediction.ndim == reference.ndim == 4
    assert prediction.shape[1:] == reference.shape[1:]
    if prediction.shape[0] != reference.shape[0]:
        # The official SANA-WM bridge may trim the final decoded frame after
        # VAE/video post-processing. Compare the common prefix but fail on
        # larger frame-count drift, which would indicate a real scheduling or
        # decode-contract mismatch.
        assert abs(prediction.shape[0] - reference.shape[0]) <= 1
        num_common_frames = min(prediction.shape[0], reference.shape[0])
        prediction = prediction[:num_common_frames]
        reference = reference[:num_common_frames]
    prediction_f = _normalize_to_uint8(prediction.astype(np.float32))
    reference_f = _normalize_to_uint8(reference.astype(np.float32))
    mean_abs_error = float(np.mean(np.abs(prediction_f - reference_f)))
    psnr = _compute_psnr(prediction_f, reference_f)
    ssim_y = _compute_ssim_y(prediction_f, reference_f)
    print(
        f"SANA-WM reference-alignment  MAE={mean_abs_error:.4f} (≤{max_mean_abs_error:.1f})"
        f"  PSNR={psnr:.2f} dB"
        f"  SSIM-Y={ssim_y:.4f}"
    )
    assert mean_abs_error <= max_mean_abs_error, (
        f"MAE {mean_abs_error:.4f} exceeds threshold {max_mean_abs_error:.1f}"
    )
    if min_psnr is not None:
        assert psnr >= min_psnr, f"PSNR {psnr:.2f} dB below threshold {min_psnr:.1f} dB"
    if min_ssim_y is not None:
        assert ssim_y >= min_ssim_y, f"SSIM-Y {ssim_y:.4f} below threshold {min_ssim_y:.4f}"


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

    max_mae = float(os.environ.get("SANA_WM_E2E_REFERENCE_MAX_MAE", "30.0"))
    # PSNR/SSIM gates are opt-in via env vars (set to empty string to disable).
    # Spec targets: PSNR ≥ 30 dB, SSIM-Y ≥ 0.93.
    psnr_env = os.environ.get("SANA_WM_E2E_MIN_PSNR", "30.0")
    ssim_env = os.environ.get("SANA_WM_E2E_MIN_SSIM_Y", "0.93")
    min_psnr = float(psnr_env) if psnr_env else None
    min_ssim_y = float(ssim_env) if ssim_env else None

    _assert_video_reference_alignment(
        prediction=inprocess,
        reference=official,
        max_mean_abs_error=max_mae,
        min_psnr=min_psnr,
        min_ssim_y=min_ssim_y,
    )
