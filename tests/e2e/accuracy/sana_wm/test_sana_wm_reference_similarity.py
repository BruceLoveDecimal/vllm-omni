# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SSIM/PSNR similarity: native vLLM-Omni SANA-WM vs the NVlabs reference.

Runs the same first-frame / camera / prompt / resolution / steps through both:
  * native  — ``vllm_omni`` ``SanaWmPipeline`` (Stage-1 DiT + SANA VAE decode),
  * reference — the upstream NVlabs ``SanaWMPipeline`` imported from
    ``VLLM_OMNI_SANA_WM_OFFICIAL_REPO`` (see ``run_sana_wm_reference``),

then compares the decoded RGB frames with SSIM/PSNR.

Gated: requires CUDA, the SANA-WM weights (``SANA_WM_E2E_MODEL``, plus
``SANA_WM_REF_MODEL`` pointing at an original-layout snapshot for the
reference side once ``SANA_WM_E2E_MODEL`` is a converted diffusers tree), and
the official checkout (``VLLM_OMNI_SANA_WM_OFFICIAL_REPO``); skipped otherwise.

Note on thresholds: native runs in the Omni worker subprocess while the
reference runs in-process, so the two seed *different* RNG streams and the
initial latents are not bit-identical (same limitation documented in
``test_ltx2_3_video_similarity``). The assertions are therefore loose
structural gates that catch gross regressions (e.g. the all-noise output from
the pre-fix 1-step refiner); the per-frame SSIM/PSNR values are printed as the
reported parity evidence.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.e2e.accuracy.helpers import compute_image_ssim_psnr
from tests.e2e.accuracy.sana_wm.run_sana_wm_reference import (
    CFG_SCALE,
    FPS,
    PROMPT,
    SEED,
    generate_reference,
    make_camera,
    reference_model_root,
)

pytestmark = [
    pytest.mark.advanced_model,
    pytest.mark.diffusion,
    pytest.mark.gpu,
]

_NUM_FRAMES = int(os.environ.get("SANA_WM_REF_NUM_FRAMES", "9"))
_STEPS = int(os.environ.get("SANA_WM_REF_STEPS", "20"))
_MIN_SSIM = float(os.environ.get("SANA_WM_REF_MIN_SSIM", "0.30"))
_MIN_PSNR = float(os.environ.get("SANA_WM_REF_MIN_PSNR", "8.0"))


def _require_env() -> str:
    if not __import__("torch").cuda.is_available():
        pytest.skip("SANA-WM reference similarity requires CUDA.")
    model = os.environ.get("SANA_WM_E2E_MODEL")
    if not model:
        pytest.skip("Set SANA_WM_E2E_MODEL to the SANA-WM checkpoint path.")
    if not os.environ.get("VLLM_OMNI_SANA_WM_OFFICIAL_REPO"):
        pytest.skip("Set VLLM_OMNI_SANA_WM_OFFICIAL_REPO to the NVlabs-Sana checkout.")
    if reference_model_root() is None:
        pytest.skip(
            "Set SANA_WM_REF_MODEL to an original-layout (config.yaml + dit/) "
            "SANA-WM snapshot for the NVlabs reference side."
        )
    return model


def _native_video(model: str, num_frames: int, steps: int) -> np.ndarray:
    """vLLM-Omni native Stage-1 + SANA VAE decode. Returns (T, H, W, 3)."""
    from tests.e2e.accuracy.sana_wm.run_sana_wm_reference import _first_frame_image
    from vllm_omni.diffusion.models.sana_wm import SANA_WM_OUTPUT_HEIGHT, SANA_WM_OUTPUT_WIDTH
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput

    omni = Omni(model=model, model_class_name="SanaWmPipeline", enforce_eager=True)
    try:
        out = omni.generate(
            {
                "prompt": PROMPT,
                "multi_modal_data": {"image": _first_frame_image(None)},
                "sana_wm": {
                    # The vLLM payload follows the JSON contract (nested lists);
                    # make_camera returns an ndarray, which the payload validator
                    # rejects as "not a sequence". Serialize to lists here. The
                    # reference side keeps the ndarray for the NVlabs pipeline.
                    "camera": {"poses": make_camera(num_frames).tolist()},
                    "num_frames": num_frames,
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
                seed=SEED,
                fps=FPS,
                num_inference_steps=steps,
                guidance_scale=CFG_SCALE,
                guidance_scale_provided=True,
                extra_args={
                    "sana_wm_sampling_algo": "flow_euler_ltx",
                    "sana_wm_offload_vae": True,
                    "sana_wm_output_type": "np",
                    "sana_wm_native_max_tokens": 30000,
                },
            ),
        )
        req = out[0] if isinstance(out, list) else out
        assert isinstance(req, OmniRequestOutput) and req.error is None, getattr(req, "error", "bad output")
        frames = req.images[0]
        if isinstance(frames, list):
            frames = frames[0]
        video = np.squeeze(np.asarray(frames))
        assert video.ndim == 4 and video.shape[-1] == 3, f"unexpected native video shape {video.shape}"
        return video
    finally:
        shutdown = getattr(omni, "shutdown", None)
        if callable(shutdown):
            shutdown()


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    scaled = np.clip(frame, 0.0, 1.0) * 255.0 if float(frame.max()) <= 1.0 else frame
    return scaled.astype("uint8")


def test_sana_wm_native_matches_nvlabs_reference() -> None:
    from PIL import Image

    model = _require_env()

    native = _native_video(model, _NUM_FRAMES, _STEPS)
    reference = generate_reference(num_frames=_NUM_FRAMES, steps=_STEPS)

    n = min(native.shape[0], reference.shape[0])
    assert n > 0, "no frames produced"
    # Frame counts may differ by the sink anchor; compare the shared prefix.
    ssims, psnrs = [], []
    for i in range(n):
        ssim, psnr = compute_image_ssim_psnr(
            prediction=Image.fromarray(_to_uint8(native[i])),
            reference=Image.fromarray(_to_uint8(reference[i])),
        )
        ssims.append(ssim)
        psnrs.append(psnr)

    mean_ssim = float(np.mean(ssims))
    mean_psnr = float(np.mean(psnrs))
    print(
        f"\n[sana-wm parity] frames={n} native={native.shape} reference={reference.shape} "
        f"mean_ssim={mean_ssim:.4f} mean_psnr={mean_psnr:.2f}dB "
        f"(cfg={CFG_SCALE} steps={_STEPS} num_frames={_NUM_FRAMES})",
        flush=True,
    )

    assert mean_ssim >= _MIN_SSIM, f"native vs reference SSIM {mean_ssim:.4f} < {_MIN_SSIM}"
    assert mean_psnr >= _MIN_PSNR, f"native vs reference PSNR {mean_psnr:.2f} < {_MIN_PSNR}"
