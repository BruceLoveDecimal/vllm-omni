# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sana-WM camera-controlled image-to-video example (native vLLM-Omni path).

Sana-WM is a *camera-conditioned world model*: unlike a plain text-to-video
model, every request needs (1) a first-frame image and (2) a camera trajectory.
This script mirrors the Wan2.2 ``text_to_video.py`` CLI so a reviewer can run it
with a single command, while exposing the extra camera knobs Sana-WM requires.

------------------------------------------------------------------------------
Quick start (no assets needed — a placeholder first frame is synthesized):

    python sana_wm.py --demo forward_push --output sana_wm_forward.mp4

Wan-style explicit command (camera injected via --action):

    python sana_wm.py \
        --image first_frame.png \
        --prompt "A slow forward camera move through a quiet city street." \
        --negative_prompt "blurry, low quality, distorted, watermark" \
        --action "w-32" \
        --height 704 --width 1280 \
        --num_frames 33 \
        --guidance_scale 5.0 \
        --num_inference_steps 40 \
        --fps 16 \
        --output sana_wm_out.mp4

------------------------------------------------------------------------------
Camera injection (the Sana-WM-specific part)
============================================
The camera trajectory is injected through the ``sana_wm`` block of the request
prompt. Provide *exactly one* of:

  * ``--action``   WASD/IJKL action DSL, segments ``<keys>-<frames>`` joined by
                   commas. Keys: w/s = forward/back, a/d = left/right strafe,
                   i/k = pitch up/down, j/l = yaw left/right; keys combine, e.g.
                   ``jw-32`` = yaw-left while moving forward. Segment frame
                   counts must sum to ``num_frames - 1``.
  * ``--camera``   path to a ``.npy`` of camera-to-world poses, shape (F, 4, 4).

Optionally pass ``--intrinsics`` (a ``.npy`` of shape (3,3)/(F,3,3)/(4,)); if
omitted, deterministic pinhole intrinsics are derived from the frame size so the
run needs no external calibration. ``--translation-speed`` / ``--rotation-speed-deg``
tune the action-DSL motion magnitude.

Note: ``--guidance_scale_high`` is accepted for CLI parity with Wan2.2 (MoE
high/low-noise experts) but Sana-WM uses a single CFG branch, so it is ignored.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vllm_omni.diffusion.models.sana_wm import (
    SANA_WM_MODEL_ID,
    SANA_WM_OUTPUT_HEIGHT,
    SANA_WM_OUTPUT_WIDTH,
)
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

# Latent geometry: Sana-WM downsamples by 32 spatially and 8 temporally
# (latent_frames = (num_frames - 1) // 8 + 1). Keep these for size validation.
SPATIAL_DOWNSAMPLE = 32
TEMPORAL_DOWNSAMPLE = 8


# ---------------------------------------------------------------------------
# Demo presets — pick one with --demo; explicit flags still override.
# action templates are functions of num_frames so the DSL stays valid.
# ---------------------------------------------------------------------------
DEMOS: dict[str, dict[str, Any]] = {
    "forward_push": {
        "prompt": "A slow forward camera move through a quiet city street at golden hour.",
        "action": lambda nf: f"w-{nf - 1}",
        "num_frames": 33,
    },
    "orbit_left": {
        "prompt": "A smooth orbit to the left around a stone fountain in a garden.",
        "action": lambda nf: f"jw-{nf - 1}",
        "num_frames": 33,
    },
    "pan_right": {
        "prompt": "A steady pan to the right across a misty mountain valley.",
        "action": lambda nf: f"l-{nf - 1}",
        "num_frames": 33,
    },
    "crane_dolly": {
        "prompt": "A cinematic crane-up then dolly-forward over a calm lake at dawn.",
        "action": lambda nf: f"i-{(nf - 1) // 2},w-{nf - 1 - (nf - 1) // 2}",
        "num_frames": 33,
    },
}

DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark, static, jpeg artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sana-WM camera-controlled I2V demo (Wan-style CLI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- model / pipeline ---
    parser.add_argument("--model", default=SANA_WM_MODEL_ID, help="HF model id or local Sana-WM snapshot.")
    parser.add_argument(
        "--model-class-name",
        "--model_class_name",
        default=None,
        help="Pipeline class. Default: SanaWmTwoStagesPipeline (Stage-1 + LTX-2 refiner); "
        "use SanaWmPipeline for Stage-1 only.",
    )
    parser.add_argument("--no-refiner", "--no_refiner", action="store_true", help="Stage-1 only (no LTX-2 refiner).")
    parser.add_argument("--inprocess-refiner", "--inprocess_refiner", action="store_true",
                        help="Enable in-process LTX-2 refiner steps on the two-stage pipeline.")
    parser.add_argument("--inprocess-refiner-steps", "--inprocess_refiner_steps", type=int, default=1)

    # --- demo / inputs ---
    parser.add_argument("--demo", choices=sorted(DEMOS), default=None,
                        help="Run a named preset (prompt + camera action). Explicit flags override it.")
    parser.add_argument("--image", default=None,
                        help="First-frame image path. If omitted, a placeholder frame is synthesized.")
    parser.add_argument("--prompt", default=None, help="Text prompt.")
    parser.add_argument("--negative-prompt", "--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)

    # --- camera injection (Sana-WM specific) ---
    parser.add_argument("--action", default=None,
                        help='Action DSL, e.g. "w-32" or "w-16,jw-16". Mutually exclusive with --camera.')
    parser.add_argument("--camera", default=None, help="Path to .npy camera-to-world poses, shape (F,4,4).")
    parser.add_argument("--intrinsics", default=None, help="Optional .npy intrinsics (3,3)/(F,3,3)/(4,).")
    parser.add_argument("--translation-speed", "--translation_speed", type=float, default=0.055)
    parser.add_argument("--rotation-speed-deg", "--rotation_speed_deg", type=float, default=1.2)

    # --- generation (Wan-style) ---
    parser.add_argument("--height", type=int, default=SANA_WM_OUTPUT_HEIGHT, help="Must be divisible by 32.")
    parser.add_argument("--width", type=int, default=SANA_WM_OUTPUT_WIDTH, help="Must be divisible by 32.")
    parser.add_argument("--num-frames", "--num_frames", type=int, default=None,
                        help="Frame count; (num_frames-1) should be a multiple of 8 (e.g. 9, 17, 33).")
    parser.add_argument("--guidance-scale", "--guidance_scale", type=float, default=5.0, help="CFG scale.")
    parser.add_argument("--guidance-scale-high", "--guidance_scale_high", type=float, default=None,
                        help="Accepted for Wan-CLI parity; IGNORED by Sana-WM (single CFG branch).")
    parser.add_argument("--num-inference-steps", "--num_inference_steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--native-max-tokens", "--native_max_tokens", type=int, default=None,
                        help="Override the native latent-token cap. Auto-sized to the request when omitted.")

    # --- runtime ---
    parser.add_argument("--tensor-parallel-size", "--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--enforce-eager", "--enforce_eager", action="store_true", default=True,
                        help="Disable torch.compile (default on for Sana-WM).")
    parser.add_argument("--output", default="sana_wm_out.mp4", help="Output mp4 path.")
    return parser.parse_args()


def _read_prompt(value: str | None) -> str:
    if value is None:
        return ""
    path = Path(value)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return value


def _synth_first_frame(width: int, height: int) -> Image.Image:
    """Deterministic placeholder first frame so the demo runs without assets."""
    yy, xx = np.meshgrid(np.linspace(0, 1, height), np.linspace(0, 1, width), indexing="ij")
    rgb = np.stack(
        [(0.30 + 0.5 * xx), (0.40 + 0.4 * yy), (0.55 + 0.3 * (1 - xx))],
        axis=-1,
    )
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), mode="RGB")


def build_camera_payload(args: argparse.Namespace, *, num_frames: int) -> dict[str, Any]:
    """Assemble the ``sana_wm`` request block that carries camera conditioning.

    This is where camera control is injected: the pipeline consumes exactly one
    of ``action`` (DSL) or ``camera={"poses": (F,4,4)}`` plus optional
    ``intrinsics``, and turns them into Plücker / raymap conditioning internally.
    """
    if args.action is not None and args.camera is not None:
        raise SystemExit("Pass only one of --action or --camera, not both.")

    sana_wm: dict[str, Any] = {
        "num_frames": num_frames,
        "translation_speed": args.translation_speed,
        "rotation_speed_deg": args.rotation_speed_deg,
    }
    if args.camera is not None:
        sana_wm["camera"] = {"poses": np.load(args.camera)}
    else:
        sana_wm["action"] = args.action  # already resolved (demo or explicit)

    if args.intrinsics is not None:
        sana_wm["intrinsics"] = np.load(args.intrinsics)
    else:
        # Deterministic pinhole intrinsics derived from the frame size — keeps
        # the demo self-contained (no external calibration / Pi3X dependency).
        sana_wm["intrinsics"] = {
            "fx": args.width / 2,
            "fy": args.width / 2,
            "cx": args.width / 2,
            "cy": args.height / 2,
        }
    return sana_wm


def _resolve_demo(args: argparse.Namespace) -> None:
    """Fill prompt/action/num_frames from a --demo preset where not explicit."""
    preset = DEMOS.get(args.demo) if args.demo else None
    if args.num_frames is None:
        args.num_frames = (preset or {}).get("num_frames", 33)
    if args.prompt is None:
        args.prompt = (preset or {}).get("prompt", "A slow forward camera move through a quiet city street.")
    if args.action is None and args.camera is None:
        if preset is not None:
            args.action = preset["action"](args.num_frames)
        else:
            args.action = f"w-{args.num_frames - 1}"  # sensible default trajectory


def _validate_geometry(args: argparse.Namespace) -> int:
    if args.height % SPATIAL_DOWNSAMPLE or args.width % SPATIAL_DOWNSAMPLE:
        raise SystemExit(f"--height/--width must be divisible by {SPATIAL_DOWNSAMPLE} (got {args.height}x{args.width}).")
    if (args.num_frames - 1) % TEMPORAL_DOWNSAMPLE:
        print(
            f"[warn] (num_frames-1)={args.num_frames - 1} is not a multiple of {TEMPORAL_DOWNSAMPLE}; "
            "latent temporal length will be rounded. Prefer 9/17/33/... for clean alignment."
        )
    latent_frames = (args.num_frames - 1) // TEMPORAL_DOWNSAMPLE + 1
    return latent_frames * (args.height // SPATIAL_DOWNSAMPLE) * (args.width // SPATIAL_DOWNSAMPLE)


def _extract_frames(output: Any) -> Any:
    if isinstance(output, list):
        output = output[0] if output else None
    if isinstance(output, OmniRequestOutput):
        if output.error is not None:
            raise RuntimeError(f"Sana-WM generation failed: {output.error}")
        if not output.images:
            raise ValueError("No frames in OmniRequestOutput.")
        video = output.images[0]
        if isinstance(video, list) and len(video) == 1:
            return video[0]
        return video
    return output


def _to_frame_list(frames: Any) -> list[np.ndarray]:
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().float().numpy()
    arr = np.asarray(frames)
    if arr.ndim == 5:  # (B, C, T, H, W) or (B, T, H, W, C)
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[0] in (3, 4):  # (C, T, H, W) -> (T, H, W, C)
        arr = np.transpose(arr, (1, 2, 3, 0))
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = np.clip(arr, 0.0, 1.0)
    return [arr[t] for t in range(arr.shape[0])]


def main() -> None:
    args = parse_args()
    _resolve_demo(args)
    token_count = _validate_geometry(args)

    if args.guidance_scale_high is not None:
        print("[note] --guidance_scale_high is ignored by Sana-WM (single CFG branch).")

    # First frame: load or synthesize.
    if args.image is not None:
        image = Image.open(args.image).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
    else:
        image = _synth_first_frame(args.width, args.height)
        print(f"[info] No --image given; synthesized a {args.width}x{args.height} placeholder first frame.")

    sana_wm = build_camera_payload(args, num_frames=args.num_frames)

    model_class_name = args.model_class_name or ("SanaWmPipeline" if args.no_refiner else "SanaWmTwoStagesPipeline")

    # Native Stage-1 is latent-token capped (default 4096); auto-raise with
    # headroom for the requested size unless the user pinned it explicitly.
    max_tokens = args.native_max_tokens if args.native_max_tokens is not None else max(4096, token_count * 2)
    extra_args: dict[str, Any] = {"sana_wm_native_smoke_max_tokens": max_tokens}
    if args.inprocess_refiner and not args.no_refiner:
        extra_args["sana_wm_inprocess_refiner"] = True
        extra_args["sana_wm_inprocess_refiner_steps"] = args.inprocess_refiner_steps

    print(f"\n{'=' * 60}")
    print("Sana-WM generation")
    print(f"  demo            : {args.demo or '(custom)'}")
    print(f"  pipeline        : {model_class_name}")
    print(f"  size            : {args.width}x{args.height}  frames={args.num_frames}  fps={args.fps}")
    print(f"  camera          : {'poses=' + args.camera if args.camera else 'action=' + args.action}")
    print(f"  cfg / steps     : {args.guidance_scale} / {args.num_inference_steps}")
    print(f"  latent tokens   : {token_count}  (cap={max_tokens})")
    print(f"{'=' * 60}\n")

    omni = Omni(
        model=args.model,
        model_class_name=model_class_name,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    start = time.perf_counter()
    output = omni.generate(
        {
            "prompt": _read_prompt(args.prompt),
            "negative_prompt": args.negative_prompt,
            "multi_modal_data": {"image": image},
            "sana_wm": sana_wm,
        },
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed,
            fps=args.fps,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            guidance_scale_provided=True,
            extra_args=extra_args,
        ),
    )
    print(f"[time] generation took {time.perf_counter() - start:.2f}s")

    frames = _to_frame_list(_extract_frames(output))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise ImportError("diffusers is required to export the Sana-WM video.") from exc
    export_to_video(frames, str(output_path), fps=args.fps)
    print(f"Saved generated video to {output_path}")


if __name__ == "__main__":
    main()
