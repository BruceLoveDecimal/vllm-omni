# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Character animation / replacement example using Wan2.2-Animate.

Two modes, selected by which inputs you pass:

- Animation: reference character image + pose video + face video. The character
  is animated to follow the driving motion; the background is generated.
- Replacement: additionally pass --background-video and --mask-video. The
  character in the background video is swapped for the reference character,
  keeping the original background.

The pose and face videos must be pre-extracted (e.g. with the Wan-Animate
``preprocess_data.py`` tooling) -- this pipeline does not run pose estimation or
face retargeting itself.

Usage:
    python character_animation.py \
        --model /path/to/Wan2.2-Animate-14B-Diffusers \
        --image character.png \
        --pose-video pose.mp4 \
        --face-video face.mp4 \
        --prompt "a person dancing in a studio"
"""

import argparse
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate or replace a character with Wan2.2-Animate.")
    parser.add_argument("--model", required=True, help="Path or HF ID of Wan2.2-Animate-14B-Diffusers.")
    parser.add_argument("--image", required=True, help="Reference character image.")
    parser.add_argument("--pose-video", required=True, help="Pre-extracted pose/skeleton driving video.")
    parser.add_argument("--face-video", required=True, help="Pre-extracted face crop driving video.")
    parser.add_argument(
        "--background-video",
        default=None,
        help="Background video (replacement mode). Must be paired with --mask-video.",
    )
    parser.add_argument(
        "--mask-video",
        default=None,
        help="Character mask video (replacement mode). Must be paired with --background-video.",
    )
    parser.add_argument("--prompt", default="a person moving naturally", help="Text prompt.")
    parser.add_argument("--negative-prompt", default=None, help="Negative prompt (model default if unset).")
    parser.add_argument("--height", type=int, default=None, help="Output height (from pose video if unset).")
    parser.add_argument("--width", type=int, default=None, help="Output width (from pose video if unset).")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Truncate the driving videos to this many frames (default: use all of them).",
    )
    parser.add_argument("--num-inference-steps", type=int, default=20, help="Denoising steps per segment.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="CFG scale. Wan-Animate is trained to run without CFG; 1.0 disables it.",
    )
    parser.add_argument("--flow-shift", type=float, default=5.0, help="Scheduler flow shift.")
    parser.add_argument(
        "--segment-frame-length",
        type=int,
        default=77,
        help="Frames generated per segment. Should be of the form 4N + 1.",
    )
    parser.add_argument(
        "--prev-segment-conditioning-frames",
        type=int,
        default=1,
        help="Frames of the previous segment re-used as temporal guidance (1 or 5 recommended).",
    )
    parser.add_argument(
        "--motion-encode-batch-size",
        type=int,
        default=None,
        help="Face frames per motion-encoder micro-batch. Lower values reduce peak memory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fps", type=int, default=16, help="FPS of the saved video.")
    parser.add_argument("--output", default="wan_animate_output.mp4", help="Output video path.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="TP size inside the DiT.")
    parser.add_argument(
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2],
        help="CFG parallel size (only useful with --guidance-scale > 1).",
    )
    parser.add_argument("--vae-patch-parallel-size", type=int, default=1, help="VAE decode patch parallelism.")
    parser.add_argument("--vae-use-slicing", action="store_true", help="Enable VAE slicing.")
    parser.add_argument("--vae-use-tiling", action="store_true", help="Enable VAE tiling.")
    parser.add_argument("--enable-cpu-offload", action="store_true", help="Enable CPU offloading.")
    parser.add_argument("--enable-layerwise-offload", action="store_true", help="Enable layerwise DiT offloading.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    return parser.parse_args()


def _to_frame_array(data) -> np.ndarray:
    """Unwrap the pipeline output into a single (T, H, W, C) uint8 array."""
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Unexpected video output type: {type(data)}")
    if data.ndim == 5:  # (B, T, H, W, C)
        data = data[0]
    if data.dtype != np.uint8:
        data = (np.clip(data, 0, 1) * 255).astype(np.uint8)
    return data


def main() -> None:
    args = parse_args()

    if (args.background_video is None) != (args.mask_video is None):
        raise SystemExit("Replacement mode needs both --background-video and --mask-video.")

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    image = PIL.Image.open(args.image).convert("RGB")

    parallel_config = DiffusionParallelConfig(
        tensor_parallel_size=args.tensor_parallel_size,
        cfg_parallel_size=args.cfg_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
    )

    omni = Omni(
        model=args.model,
        model_class_name="WanAnimatePipeline",
        flow_shift=args.flow_shift,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_layerwise_offload=args.enable_layerwise_offload,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
    )

    multi_modal_data = {
        "image": image,
        "pose_video": args.pose_video,
        "face_video": args.face_video,
    }
    if args.background_video is not None:
        multi_modal_data["video"] = args.background_video
        multi_modal_data["mask"] = args.mask_video

    extra_args = {
        "segment_frame_length": args.segment_frame_length,
        "prev_segment_conditioning_frames": args.prev_segment_conditioning_frames,
    }
    if args.motion_encode_batch_size is not None:
        extra_args["motion_encode_batch_size"] = args.motion_encode_batch_size

    mode = "replacement" if args.background_video is not None else "animation"
    print(f"Running Wan2.2-Animate in {mode} mode ({args.num_inference_steps} steps/segment)")

    start = time.perf_counter()
    result = omni.generate(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "multi_modal_data": multi_modal_data,
        },
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            generator=generator,
            extra_args=extra_args,
        ),
    )
    print(f"Total generation time: {time.perf_counter() - start:.2f}s")

    output = OmniRequestOutput.unwrap_result(result)
    if not output.images:
        raise ValueError("No video frames found in OmniRequestOutput.")

    frames = _to_frame_array(output.images)

    from diffusers.utils import export_to_video

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(list(frames), str(output_path), fps=args.fps)
    print(f"Saved {frames.shape[0]} frames to {output_path}")


if __name__ == "__main__":
    main()
