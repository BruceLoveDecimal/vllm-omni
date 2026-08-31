# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SSIM/PSNR similarity: native vLLM-Omni SANA-WM vs the NVlabs reference.

Runs the same first-frame / camera / prompt / resolution / steps through both:
  * native  — ``vllm_omni`` ``SanaWmTwoStagesPipeline`` (Stage-1 DiT plus the
    LTX-2 refiner, decoded through the SANA VAE),
  * reference — the upstream NVlabs ``SanaWMPipeline`` with its own
    ``RefinerSettings``, imported from ``VLLM_OMNI_SANA_WM_OFFICIAL_REPO``
    (see ``run_sana_wm_reference``),

then compares the decoded RGB frames with SSIM/PSNR.

Two-stage is the only parity case: it drives the same Stage-1 sampler the
Stage-1-only pipeline does, so a Stage-1 regression shows up here too.

Gated: requires CUDA, the SANA-WM weights (``SANA_WM_E2E_MODEL``, plus
``SANA_WM_REF_MODEL`` pointing at an original-layout snapshot for the
reference side once ``SANA_WM_E2E_MODEL`` is a converted diffusers tree), and
the official checkout (``VLLM_OMNI_SANA_WM_OFFICIAL_REPO``); skipped otherwise.

Note on thresholds: native runs in the Omni worker subprocess while the
reference runs in-process, so the two seed *different* RNG streams and the
initial latents are not bit-identical (same limitation documented in
``test_ltx2_3_video_similarity``). The two implementations therefore never
converge to a bit-exact match, and the gates are set just under the worst
measured profile rather than at a value the two could only hit by agreeing
exactly. The per-frame SSIM/PSNR values are still printed as the reported
parity evidence.
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
    REF_REFINER_SEED,
    REF_REFINER_SINK_SIZE,
    SEED,
    generate_reference,
    make_camera,
    reference_model_root,
    reference_refiner_root,
)

pytestmark = [
    pytest.mark.advanced_model,
    pytest.mark.diffusion,
    pytest.mark.gpu,
]

_NUM_FRAMES = int(os.environ.get("SANA_WM_REF_NUM_FRAMES", "9"))
_STEPS = int(os.environ.get("SANA_WM_REF_STEPS", "20"))
# The tightest measured pair is the two-stage production profile (161 frames,
# 60 steps): 0.8234 SSIM / 21.08 dB. Everything else clears these by a wide
# margin -- the default 9-frame / 20-step profile scores ~0.977 / ~38 dB. So
# 0.80 / 20.0 dB sits just under the worst real result rather than at the
# noise floor; raising them further would gate on that one profile's headroom.
_MIN_SSIM = float(os.environ.get("SANA_WM_REF_MIN_SSIM", "0.80"))
_MIN_PSNR = float(os.environ.get("SANA_WM_REF_MIN_PSNR", "20.0"))


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


def _native_video(
    model: str,
    num_frames: int,
    steps: int,
    *,
    model_class_name: str = "SanaWmTwoStagesPipeline",
    extra_args: dict | None = None,
) -> np.ndarray:
    """vLLM-Omni native generation. Returns (T, H, W, 3).

    ``model_class_name="SanaWmPipeline"`` stops after Stage-1 + SANA VAE decode,
    which is what the reference does with ``use_refiner=False``.
    """
    from tests.e2e.accuracy.sana_wm.run_sana_wm_reference import _first_frame_image
    from vllm_omni.diffusion.models.sana_wm import SANA_WM_OUTPUT_HEIGHT, SANA_WM_OUTPUT_WIDTH
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput

    omni = Omni(model=model, model_class_name=model_class_name, enforce_eager=True)
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
                # 30000 clears the 9-frame 704x1280 latent
                # (2 * 22 * 40 = 1760 tokens) with headroom.
                extra_args={"sana_wm_native_max_tokens": 30000, **(extra_args or {})},
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


def test_sana_wm_two_stages_matches_nvlabs_reference() -> None:
    """SanaWmTwoStagesPipeline (Stage 1 + LTX-2 refiner) vs the same NVlabs path."""
    from PIL import Image

    model = _require_env()
    if reference_refiner_root() is None:
        pytest.skip(
            "Set SANA_WM_REF_REFINER_ROOT (or use a reference snapshot carrying "
            "refiner/transformer/config.json) for the two-stage baseline."
        )

    native = _native_video(
        model,
        _NUM_FRAMES,
        _STEPS,
        model_class_name="SanaWmTwoStagesPipeline",
        # Pin the refiner knobs to the reference's own defaults rather than
        # trusting the two implementations to agree by coincidence.
        extra_args={
            "sana_wm_refiner_sink_size": REF_REFINER_SINK_SIZE,
            "sana_wm_refiner_seed": REF_REFINER_SEED,
            "sana_wm_inprocess_refiner_steps": 3,
        },
    )
    reference = generate_reference(num_frames=_NUM_FRAMES, steps=_STEPS, use_refiner=True)

    # The reference drops the decoded sink anchor (``SanaWMPipeline._refine``
    # returns ``video[1:]``) while the native pipeline keeps it; realign before
    # comparing or every frame is off by one.
    assert native.shape[0] == reference.shape[0] + 1, (
        f"expected the native clip to carry one extra sink frame; native={native.shape} reference={reference.shape}"
    )
    native_aligned = native[1:]

    n = reference.shape[0]
    assert n > 0, "no frames produced"
    ssims, psnrs = [], []
    for i in range(n):
        ssim, psnr = compute_image_ssim_psnr(
            prediction=Image.fromarray(_to_uint8(native_aligned[i])),
            reference=Image.fromarray(_to_uint8(reference[i])),
        )
        ssims.append(ssim)
        psnrs.append(psnr)

    mean_ssim = float(np.mean(ssims))
    mean_psnr = float(np.mean(psnrs))
    print(
        f"\n[sana-wm two-stage parity] frames={n} native={native.shape} reference={reference.shape} "
        f"mean_ssim={mean_ssim:.4f} mean_psnr={mean_psnr:.2f}dB "
        f"(cfg={CFG_SCALE} steps={_STEPS} num_frames={_NUM_FRAMES})",
        flush=True,
    )

    assert mean_ssim >= _MIN_SSIM, f"two-stage vs reference SSIM {mean_ssim:.4f} < {_MIN_SSIM}"
    assert mean_psnr >= _MIN_PSNR, f"two-stage vs reference PSNR {mean_psnr:.2f} < {_MIN_PSNR}"
