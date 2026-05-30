# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.diffusion,
    pytest.mark.gpu,
    pytest.mark.skipif(os.environ.get("SANA_WM_E2E") != "1", reason="Set SANA_WM_E2E=1 to run Sana-WM GPU e2e."),
]


def _make_sana_wm_e2e_image() -> Any:
    from PIL import Image

    from vllm_omni.diffusion.models.sana_wm import SANA_WM_OUTPUT_HEIGHT, SANA_WM_OUTPUT_WIDTH

    return Image.new("RGB", (SANA_WM_OUTPUT_WIDTH, SANA_WM_OUTPUT_HEIGHT), (96, 128, 160))


def _make_sana_wm_e2e_first_frame() -> np.ndarray:
    return np.asarray(_make_sana_wm_e2e_image(), dtype=np.uint8)


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
    stage1_steps = int(
        os.environ.get(
            "SANA_WM_E2E_STAGE1_STEPS",
            os.environ.get("SANA_WM_E2E_NUM_INFERENCE_STEPS", "1"),
        )
    )
    action = os.environ.get("SANA_WM_E2E_ACTION", f"w-{max(num_frames - 1, 1)}")
    image = _make_sana_wm_e2e_image()
    omni = Omni(
        model=model,
        model_class_name=model_class_name,
        enforce_eager=True,
    )
    extra_args = {"sana_wm_sampling_algo": "flow_euler_ltx", "sana_wm_offload_vae": True}
    native_smoke_max_tokens = os.environ.get("SANA_WM_E2E_NATIVE_SMOKE_MAX_TOKENS", "").strip()
    if native_smoke_max_tokens:
        extra_args["sana_wm_native_smoke_max_tokens"] = int(native_smoke_max_tokens)
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
                "action": action,
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
            num_inference_steps=stage1_steps,
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


def _prepend_first_frame(video: np.ndarray, first_frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(first_frame, dtype=np.float32)
    assert frame.shape == video.shape[1:]
    video_max = float(np.max(video)) if video.size else 0.0
    if np.issubdtype(video.dtype, np.floating) and video_max <= 1.0:
        frame = frame / 255.0
    return np.concatenate([frame[None, ...].astype(video.dtype, copy=False), video], axis=0)


def _align_sana_wm_video_frames(
    *,
    prediction: np.ndarray,
    reference: np.ndarray,
    first_frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    mode = os.environ.get("SANA_WM_E2E_FRAME_ALIGNMENT", "auto").strip().lower()
    if mode in {"none", "common_prefix"}:
        return prediction, reference, mode
    if mode in {"prepend_reference_frame0", "prepend_reference_first", "reference_add_frame0"}:
        return prediction, _prepend_first_frame(reference, first_frame), "prepend_reference_frame0"
    if mode in {"prepend_prediction_frame0", "prepend_prediction_first", "prediction_add_frame0"}:
        return _prepend_first_frame(prediction, first_frame), reference, "prepend_prediction_frame0"
    if mode in {"drop_prediction_frame0", "drop_native_frame0", "prediction_skip_frame0"}:
        return prediction[1:], reference, "drop_prediction_frame0"
    if mode in {"drop_reference_frame0", "reference_skip_frame0"}:
        return prediction, reference[1:], "drop_reference_frame0"
    if mode != "auto":
        raise ValueError(
            "Unsupported SANA_WM_E2E_FRAME_ALIGNMENT value "
            f"{mode!r}; use auto, none, prepend_reference_frame0, "
            "prepend_prediction_frame0, drop_prediction_frame0, or drop_reference_frame0."
        )

    if prediction.shape[0] == reference.shape[0] + 1:
        return prediction, _prepend_first_frame(reference, first_frame), "auto_prepend_reference_frame0"
    if reference.shape[0] == prediction.shape[0] + 1:
        return _prepend_first_frame(prediction, first_frame), reference, "auto_prepend_prediction_frame0"
    return prediction, reference, "auto_none"


def _assert_video_reference_alignment(
    *,
    prediction: np.ndarray,
    reference: np.ndarray,
    first_frame: np.ndarray,
    max_mean_abs_error: float,
    min_psnr: float | None = None,
    min_ssim_y: float | None = None,
    min_frame0_psnr: float | None = 30.0,
) -> dict[str, float | int | str]:
    assert prediction.ndim == reference.ndim == 4
    assert prediction.shape[1:] == reference.shape[1:]
    assert first_frame.shape == prediction.shape[1:]
    original_prediction_frames = prediction.shape[0]
    original_reference_frames = reference.shape[0]
    prediction, reference, frame_alignment = _align_sana_wm_video_frames(
        prediction=prediction,
        reference=reference,
        first_frame=first_frame,
    )
    aligned_prediction_frames = prediction.shape[0]
    aligned_reference_frames = reference.shape[0]
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
    frame0_psnr = _compute_psnr(prediction_f[:1], reference_f[:1])
    frame0_mae = float(np.mean(np.abs(prediction_f[:1] - reference_f[:1])))
    generated_prediction_f = prediction_f[1:] if prediction_f.shape[0] > 1 else prediction_f[:0]
    generated_reference_f = reference_f[1:] if reference_f.shape[0] > 1 else reference_f[:0]
    if generated_prediction_f.shape[0] > 0:
        generated_mae = float(np.mean(np.abs(generated_prediction_f - generated_reference_f)))
        generated_psnr = _compute_psnr(generated_prediction_f, generated_reference_f)
        generated_ssim_y = _compute_ssim_y(generated_prediction_f, generated_reference_f)
    else:
        generated_mae = 0.0
        generated_psnr = float("inf")
        generated_ssim_y = 1.0
    metrics: dict[str, float | int | str] = {
        "prediction_frames": original_prediction_frames,
        "reference_frames": original_reference_frames,
        "aligned_prediction_frames": aligned_prediction_frames,
        "aligned_reference_frames": aligned_reference_frames,
        "common_frames": int(prediction_f.shape[0]),
        "mae": mean_abs_error,
        "psnr": psnr,
        "ssim_y": ssim_y,
        "frame0_mae": frame0_mae,
        "frame0_psnr": frame0_psnr,
        "generated_frames": int(generated_prediction_f.shape[0]),
        "generated_mae": generated_mae,
        "generated_psnr": generated_psnr,
        "generated_ssim_y": generated_ssim_y,
        "frame_alignment": frame_alignment,
    }
    print(
        f"SANA-WM reference-alignment  MAE={mean_abs_error:.4f} (≤{max_mean_abs_error:.1f})"
        f"  PSNR={psnr:.2f} dB"
        f"  SSIM-Y={ssim_y:.4f}"
        f"  frame0_PSNR={frame0_psnr:.2f} dB"
        f"  generated_MAE={generated_mae:.4f}"
        f"  generated_PSNR={generated_psnr:.2f} dB"
        f"  generated_SSIM-Y={generated_ssim_y:.4f}"
        f"  frame_alignment={frame_alignment}"
    )
    if min_frame0_psnr is not None:
        assert frame0_psnr >= min_frame0_psnr, (
            f"Frame-0 PSNR {frame0_psnr:.2f} dB below threshold {min_frame0_psnr:.1f} dB; "
            "the SANA-WM e2e frame-index alignment is likely broken."
        )
    assert mean_abs_error <= max_mean_abs_error, (
        f"MAE {mean_abs_error:.4f} exceeds threshold {max_mean_abs_error:.1f}"
    )
    if min_psnr is not None:
        assert psnr >= min_psnr, f"PSNR {psnr:.2f} dB below threshold {min_psnr:.1f} dB"
    if min_ssim_y is not None:
        assert ssim_y >= min_ssim_y, f"SSIM-Y {ssim_y:.4f} below threshold {min_ssim_y:.4f}"
    return metrics


def _maybe_write_reference_metrics(metrics: dict[str, float | int | str]) -> None:
    metrics_path = os.environ.get("SANA_WM_E2E_METRICS_JSON", "").strip()
    if not metrics_path:
        return
    payload = dict(metrics)
    payload.update(
        {
            "num_frames": int(os.environ.get("SANA_WM_E2E_NUM_FRAMES", "9")),
            "stage1_steps": int(
                os.environ.get(
                    "SANA_WM_E2E_STAGE1_STEPS",
                    os.environ.get("SANA_WM_E2E_NUM_INFERENCE_STEPS", "1"),
                )
            ),
            "refiner_steps": int(os.environ.get("SANA_WM_E2E_REFINER_STEPS", "1")),
            "action": os.environ.get(
                "SANA_WM_E2E_ACTION",
                f"w-{max(int(os.environ.get('SANA_WM_E2E_NUM_FRAMES', '9')) - 1, 1)}",
            ),
        }
    )
    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    first_frame = _make_sana_wm_e2e_first_frame()

    max_mae = float(os.environ.get("SANA_WM_E2E_REFERENCE_MAX_MAE", "30.0"))
    # PSNR/SSIM gates are opt-in via env vars (set to empty string to disable).
    # Spec targets: PSNR ≥ 30 dB, SSIM-Y ≥ 0.93.
    psnr_env = os.environ.get("SANA_WM_E2E_MIN_PSNR", "30.0")
    ssim_env = os.environ.get("SANA_WM_E2E_MIN_SSIM_Y", "0.93")
    frame0_psnr_env = os.environ.get("SANA_WM_E2E_FRAME0_MIN_PSNR", "30.0")
    min_psnr = float(psnr_env) if psnr_env else None
    min_ssim_y = float(ssim_env) if ssim_env else None
    min_frame0_psnr = float(frame0_psnr_env) if frame0_psnr_env else None

    metrics = _assert_video_reference_alignment(
        prediction=inprocess,
        reference=official,
        first_frame=first_frame,
        max_mean_abs_error=max_mae,
        min_psnr=min_psnr,
        min_ssim_y=min_ssim_y,
        min_frame0_psnr=min_frame0_psnr,
    )
    _maybe_write_reference_metrics(metrics)
