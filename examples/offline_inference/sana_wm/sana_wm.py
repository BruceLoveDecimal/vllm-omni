# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sana-WM image-to-video smoke example.

This example uses vLLM-Omni's Sana-WM pipeline surface and the official NVlabs
Sana-WM CLI backend. Set ``--official-repo`` to a Sana checkout containing
``inference_video_scripts/inference_sana_wm.py``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sana-WM I2V through vLLM-Omni.")
    parser.add_argument("--model", default=SANA_WM_MODEL_ID, help="HF model id or local Sana-WM snapshot.")
    parser.add_argument("--official-repo", required=True, help="NVlabs/Sana checkout with inference_sana_wm.py.")
    parser.add_argument("--image", required=True, help="First-frame image path.")
    parser.add_argument("--prompt", required=True, help="Prompt text or path to a UTF-8 prompt file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--action", help='Official WASD/IJKL action DSL, e.g. "w-80,jw-40,w-40".')
    group.add_argument("--camera", help="NumPy .npy camera-to-world poses with shape (F,4,4).")
    parser.add_argument("--intrinsics", help="Optional NumPy .npy intrinsics, shape (3,3), (F,3,3), or (4,).")
    parser.add_argument("--translation-speed", type=float, default=0.055)
    parser.add_argument("--rotation-speed-deg", type=float, default=1.2)
    parser.add_argument("--num-frames", type=int, default=321)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument(
        "--no-refiner",
        action="store_true",
        help="Run Stage 1 only with the official --no_refiner flag.",
    )
    parser.add_argument(
        "--inprocess-refiner",
        action="store_true",
        help="Run the vLLM-Omni in-process LTX-2 refiner path instead of the official CLI refiner.",
    )
    parser.add_argument("--inprocess-refiner-steps", type=int, default=1)
    parser.add_argument("--output", default="sana_wm_output.mp4", help="Output mp4 path.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile in vLLM-Omni.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel degree.")
    return parser.parse_args()


def _read_prompt(value: str) -> str:
    path = Path(value)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return value


def _extract_video(output: Any) -> Any:
    if isinstance(output, list):
        output = output[0] if output else None
    if isinstance(output, OmniRequestOutput):
        if output.images:
            video = output.images[0]
            if isinstance(video, list) and len(video) == 1:
                return video[0]
            return video
        raise ValueError(f"Sana-WM generation failed: {output.error or 'no images returned'}")
    return output


def main() -> None:
    args = parse_args()
    os.environ[SANA_WM_OFFICIAL_REPO_ENV] = args.official_repo

    image = Image.open(args.image).convert("RGB")
    sana_wm: dict[str, Any] = {
        "num_frames": args.num_frames,
        "translation_speed": args.translation_speed,
        "rotation_speed_deg": args.rotation_speed_deg,
    }
    if args.action:
        sana_wm["action"] = args.action
    else:
        sana_wm["camera"] = {"poses": np.load(args.camera)}
    if args.intrinsics:
        sana_wm["intrinsics"] = np.load(args.intrinsics)

    model_class_name = "SanaWmPipeline" if args.no_refiner else "SanaWmTwoStagesPipeline"
    omni = Omni(
        model=args.model,
        model_class_name=model_class_name,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    output = omni.generate(
        {
            "prompt": _read_prompt(args.prompt),
            "multi_modal_data": {"image": image},
            "sana_wm": sana_wm,
        },
        OmniDiffusionSamplingParams(
            height=SANA_WM_OUTPUT_HEIGHT,
            width=SANA_WM_OUTPUT_WIDTH,
            num_frames=args.num_frames,
            seed=args.seed,
            extra_args=(
                {
                    "sana_wm_inprocess_refiner": True,
                    "sana_wm_inprocess_refiner_steps": args.inprocess_refiner_steps,
                }
                if args.inprocess_refiner
                else {}
            ),
        ),
    )

    video = _extract_video(output)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise ImportError("diffusers is required to export the Sana-WM output video.") from exc

    if isinstance(video, np.ndarray) and video.ndim == 4 and np.issubdtype(video.dtype, np.integer):
        video = video.astype(np.float32) / 255.0
    export_to_video(video, str(output_path), fps=args.fps)
    print(f"Saved generated video to {output_path}")


if __name__ == "__main__":
    main()
