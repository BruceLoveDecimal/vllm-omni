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
import json
import os
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
                        help="Run eager (default). torch.compile reuse was measured net-negative for "
                        "Sana-WM (compute-bound; GDN/attention run eager, refiner not compiled): "
                        "~8%% slower than eager AND it perturbs the output (PSNR ~24.5 dB vs eager). "
                        "Pass --no-enforce-eager only to experiment with the compiled path.")
    parser.add_argument("--no-enforce-eager", "--no_enforce_eager", dest="enforce_eager",
                        action="store_false", help="Opt into the (slower, output-perturbing) compiled path.")
    parser.add_argument("--fp32-stage1", "--fp32_stage1", action="store_true",
                        help="High-fidelity mode: run the Stage-1 DiT in fp32 (refiner stays bf16). "
                        "The 60-step flow-matching trajectory is chaotic and bf16 per-step rounding "
                        "is amplified; fp32 Stage-1 is the precision-stable trajectory. Measured "
                        "+0.69 dB PSNR / -2.5 MAE vs the NVlabs fp32 reference (latent cos 0.838->0.909) "
                        "at a Stage-1 latency cost (~1.4x). Sets SANA_WM_FORCE_FP32_TRANSFORMER=1.")
    parser.add_argument("--output", default="sana_wm_out.mp4", help="Output mp4 path.")
    # --- evaluation / metrics (optional) ---
    parser.add_argument("--reference", default=None,
                        help="Reference video (.mp4) or frames (.npy) to compute MAE/PSNR/SSIM-Y against. "
                        "E.g. an eager run's output to validate the compiled path, or a frozen golden.")
    parser.add_argument("--metrics-json", "--metrics_json", default=None,
                        help="Write run metadata + metrics to this JSON path (persisted to disk).")
    parser.add_argument("--save-frames-npy", "--save_frames_npy", default=None,
                        help="Also dump generated frames losslessly as (T,H,W,3) uint8 .npy. Use this "
                        "as a clean --reference for the next run (avoids mp4 codec noise in metrics).")
    parser.add_argument("--frame-align", "--frame_align", default="drop_pred_frame0",
                        choices=["drop_pred_frame0", "none", "drop_ref_frame0",
                                 "drop_pred_last", "drop_ref_last", "auto"],
                        help="Frame alignment for --reference metrics. Default 'drop_pred_frame0' is "
                        "the fixed I2V conditioning-frame convention (pred[1:] vs ref): our output "
                        "includes the given first frame at index 0, the reference does not. "
                        "'auto' (lowest-MSE search) is a DIAGNOSTIC only — do not report its number.")
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


# ---------------------------------------------------------------------------
# Metrics (optional, vs a --reference video/frames). All in float [0, 255].
# ---------------------------------------------------------------------------
def _stack_frames_255(frames: list[np.ndarray]) -> np.ndarray:
    """List of (H,W,C) frames -> (T,H,W,3) float32 in [0, 255]."""
    arr = np.stack([np.asarray(f, dtype=np.float32) for f in frames], axis=0)
    if arr.size and float(arr.max()) <= 1.0 + 1e-6:
        arr = arr * 255.0
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _load_reference_255(path: str) -> np.ndarray:
    """Load a reference video (.mp4) or frames (.npy) as (T,H,W,3) float32 [0,255]."""
    p = Path(path)
    if p.suffix == ".npy":
        a = np.asarray(np.load(p), dtype=np.float32)
        if a.ndim == 5:
            a = a[0]
        if a.ndim == 4 and a.shape[0] in (3, 4):  # (C,T,H,W) -> (T,H,W,C)
            a = np.transpose(a, (1, 2, 3, 0))
        if a.size and float(a.max()) <= 1.0 + 1e-6:
            a = a * 255.0
        return a[..., :3] if a.shape[-1] == 4 else a
    import imageio.v3 as iio

    a = np.asarray(iio.imread(p), dtype=np.float32)  # (T,H,W,C)
    return a[..., :3] if a.shape[-1] == 4 else a


def _to_luma(v: np.ndarray) -> np.ndarray:  # (T,H,W,3) -> (T,H,W) BT.601
    return 0.299 * v[..., 0] + 0.587 * v[..., 1] + 0.114 * v[..., 2]


def _frame_align_candidates(pred: np.ndarray, ref: np.ndarray, mode: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Resolve the (pred, ref) pairing for the requested frame alignment.

    I2V conditioning-frame convention: the first frame of an image-to-video
    output is the *given* conditioning image, not a generated frame. SANA-WM
    emits it at index 0; the NVlabs reference does not. So the principled,
    fixed comparison is pred[1:] vs ref ("drop_pred_frame0") — that is the
    default and the number to report.

    The other modes (incl. ``auto``, which searches for the lowest-MSE shift)
    exist only as opt-in diagnostics for investigating an unexpected offset.
    auto must NOT be used for reported metrics: picking the best-scoring shift
    is cherry-picking and can mask a real misalignment bug.
    """
    explicit = {
        "none": (pred, ref),
        "drop_pred_frame0": (pred[1:], ref),
        "drop_ref_frame0": (pred, ref[1:]),
        "drop_pred_last": (pred[:-1], ref),
        "drop_ref_last": (pred, ref[:-1]),
    }
    if mode != "auto":
        return {mode: explicit[mode]}
    cands = {"none": (pred, ref)}
    if pred.shape[0] == ref.shape[0] + 1:
        cands["drop_pred_frame0"] = (pred[1:], ref)
        cands["drop_pred_last"] = (pred[:-1], ref)
    elif ref.shape[0] == pred.shape[0] + 1:
        cands["drop_ref_frame0"] = (pred, ref[1:])
        cands["drop_ref_last"] = (pred, ref[:-1])
    return cands


def _compute_video_metrics(pred_frames: list[np.ndarray], reference: str, frame_align: str = "drop_pred_frame0") -> dict[str, Any]:
    pred = _stack_frames_255(pred_frames)
    ref = _load_reference_255(reference)
    if pred.shape[1:3] != ref.shape[1:3]:
        return {
            "reference": reference,
            "metric_error": f"spatial size mismatch pred={pred.shape[1:3]} ref={ref.shape[1:3]}",
        }
    orig_pred_frames, orig_ref_frames = int(pred.shape[0]), int(ref.shape[0])
    # Pick the frame alignment that minimizes MSE (auto), or the requested one.
    best = None  # (mse, name, pred_a, ref_a)
    for name, (pa, ra) in _frame_align_candidates(pred, ref, frame_align).items():
        k = min(pa.shape[0], ra.shape[0])
        if k == 0:
            continue
        m = float(np.mean((pa[:k] - ra[:k]) ** 2))
        if best is None or m < best[0]:
            best = (m, name, pa[:k], ra[:k])
    _, alignment, pred, ref = best
    mse = float(np.mean((pred - ref) ** 2))
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10((255.0 ** 2) / mse))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    py, ry = _to_luma(pred), _to_luma(ref)
    ssim_vals = []
    for t in range(py.shape[0]):
        p, r = py[t], ry[t]
        mp, mr = float(p.mean()), float(r.mean())
        vp, vr = float(np.var(p)), float(np.var(r))
        vpr = float(np.mean((p - mp) * (r - mr)))
        num = (2 * mp * mr + c1) * (2 * vpr + c2)
        den = (mp ** 2 + mr ** 2 + c1) * (vp + vr + c2)
        ssim_vals.append(float(num / den) if den else 1.0)
    return {
        "reference": reference,
        "frame_alignment": alignment,
        "frames_compared": int(pred.shape[0]),
        "pred_frames": orig_pred_frames,
        "ref_frames": orig_ref_frames,
        "mae": round(float(np.mean(np.abs(pred - ref))), 6),
        "psnr_db": round(psnr, 4) if psnr != float("inf") else "inf",
        "ssim_y": round(float(np.mean(ssim_vals)), 6),
    }


def main() -> None:
    args = parse_args()
    _resolve_demo(args)
    token_count = _validate_geometry(args)

    if args.fp32_stage1:
        # High-fidelity mode: the transformer reads this at materialize/forward
        # time, so it must be set before the pipeline is constructed below.
        os.environ["SANA_WM_FORCE_FP32_TRANSFORMER"] = "1"
        print("[info] --fp32-stage1: running Stage-1 DiT in fp32 (refiner stays bf16).")

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
    # Always decode to RGB frames (np) so the output is a viewable mp4. Without
    # this the native Stage-1 path returns raw 128-channel latents.
    extra_args: dict[str, Any] = {
        "sana_wm_native_smoke_max_tokens": max_tokens,
        "sana_wm_output_type": "np",
    }
    if not args.no_refiner:
        # Two-stage (SanaWmTwoStagesPipeline): run the in-process LTX-2 refiner
        # to produce the production RGB video. This is the NVlabs default path.
        extra_args["sana_wm_inprocess_refiner"] = True
        extra_args["sana_wm_inprocess_refiner_steps"] = args.inprocess_refiner_steps
        extra_args["sana_wm_refiner_output_type"] = "np"

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
    gen_seconds = time.perf_counter() - start
    print(f"[time] generation took {gen_seconds:.2f}s")

    frames = _to_frame_list(_extract_frames(output))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise ImportError("diffusers is required to export the Sana-WM video.") from exc
    export_to_video(frames, str(output_path), fps=args.fps)
    print(f"Saved generated video to {output_path}")

    if args.save_frames_npy:
        npy_path = Path(args.save_frames_npy)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        frames_u8 = np.clip(_stack_frames_255(frames), 0, 255).round().astype(np.uint8)
        np.save(npy_path, frames_u8)
        print(f"Saved lossless frames to {npy_path}  shape={frames_u8.shape}")

    # --- metrics (optional) ---
    metrics: dict[str, Any] = {
        "demo": args.demo,
        "pipeline": model_class_name,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "latent_tokens": token_count,
        "generation_seconds": round(gen_seconds, 3),
        "output": str(output_path),
    }
    if args.reference:
        vm = _compute_video_metrics(frames, args.reference, frame_align=args.frame_align)
        metrics.update(vm)
        if "metric_error" in vm:
            print(f"[metrics] skipped: {vm['metric_error']}")
        else:
            print(
                f"[metrics] vs {args.reference}: MAE={vm['mae']:.4f}  "
                f"PSNR={vm['psnr_db']} dB  SSIM-Y={vm['ssim_y']:.4f}  "
                f"(frames={vm['frames_compared']}, align={vm['frame_alignment']})"
            )
    else:
        print("[metrics] no --reference; MAE/PSNR/SSIM skipped (run config + timing still recorded)")

    if args.metrics_json:
        mj = Path(args.metrics_json)
        mj.parent.mkdir(parents=True, exist_ok=True)
        mj.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[metrics] wrote {mj}")


if __name__ == "__main__":
    main()
