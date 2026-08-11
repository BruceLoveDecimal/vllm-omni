# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Character animation example using Wan2.2-Animate-14B (animation mode).

Drives a reference character image with a pre-processed pose (skeleton) video
and face video. Both driving videos must be produced beforehand by the upstream
Wan-Animate `preprocess_data.py`; this pipeline does not run pose estimation or
face retargeting itself.

Usage:
    python wan_animate.py \
        --model Wan-AI/Wan2.2-Animate-14B-Diffusers \
        --image character.png \
        --pose-video pose.mp4 \
        --face-video face.mp4 \
        --prompt "A person dancing in a studio"
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
    parser = argparse.ArgumentParser(description="Animate a character image with pose and face videos (Wan-Animate).")
    parser.add_argument("--model", required=True, help="Wan2.2-Animate diffusers model (local path or HF ID).")
    parser.add_argument("--image", required=True, help="Path to the reference character image.")
    parser.add_argument("--pose-video", required=True, help="Path to the pre-processed pose/skeleton video.")
    parser.add_argument("--face-video", required=True, help="Path to the pre-processed face video.")
    parser.add_argument("--prompt", default="A person performing a motion", help="Text prompt for the scene.")
    parser.add_argument("--negative-prompt", default=None, help="Negative prompt (uses the Wan default if unset).")
    parser.add_argument("--output", default="animate_output.mp4", help="Output video path.")

    parser.add_argument("--height", type=int, default=None, help="Video height (derived from the image if unset).")
    parser.add_argument("--width", type=int, default=None, help="Video width (derived from the image if unset).")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Frames to generate; defaults to the length of the driving videos.",
    )
    parser.add_argument("--segment-frame-length", type=int, default=77, help="Frames per segment (must be 4n+1).")
    parser.add_argument(
        "--prev-segment-conditioning-frames",
        type=int,
        default=1,
        choices=(1, 5),
        help="Frames carried over from the previous segment.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=20, help="Denoising steps per segment.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="CFG scale; Wan-Animate runs without CFG by default.",
    )
    parser.add_argument("--flow-shift", type=float, default=5.0, help="Flow matching shift.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fps", type=int, default=30, help="Output frame rate.")

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cfg-parallel-size", type=int, default=1)
    parser.add_argument("--vae-patch-parallel-size", type=int, default=1)
    parser.add_argument("--use-hsdp", action="store_true")
    parser.add_argument("--hsdp-shard-size", type=int, default=None)
    parser.add_argument("--hsdp-replicate-size", type=int, default=None)

    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enable-layerwise-offload", action="store_true")
    parser.add_argument("--vae-use-slicing", action="store_true")
    parser.add_argument("--vae-use-tiling", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def to_frame_list(frames) -> list[np.ndarray]:
    """Normalize the pipeline output into a list of HWC float frames in [0, 1]."""
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().float().numpy()
    if isinstance(frames, list):
        if len(frames) == 0:
            raise ValueError("No video frames found in the output.")
        frames = np.asarray(frames[0]) if isinstance(frames[0], np.ndarray) else np.asarray(frames)
    if frames.ndim == 5:
        frames = frames[0]
    if np.issubdtype(frames.dtype, np.integer):
        frames = frames.astype(np.float32) / 255.0
    return list(frames)


def main() -> None:
    args = parse_args()
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    image = PIL.Image.open(args.image).convert("RGB")

    parallel_config = DiffusionParallelConfig(
        tensor_parallel_size=args.tensor_parallel_size,
        cfg_parallel_size=args.cfg_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
        use_hsdp=args.use_hsdp,
        hsdp_shard_size=args.hsdp_shard_size,
        hsdp_replicate_size=args.hsdp_replicate_size,
    )

    omni = Omni(
        model=args.model,
        flow_shift=args.flow_shift,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_layerwise_offload=args.enable_layerwise_offload,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
    )

    print(f"\n{'=' * 60}")
    print("Generation configuration:")
    print(f"  Model: {args.model}")
    print(f"  Reference image: {args.image}")
    print(f"  Pose video: {args.pose_video}")
    print(f"  Face video: {args.face_video}")
    print(f"  Segment length: {args.segment_frame_length} (carry-over {args.prev_segment_conditioning_frames})")
    print(f"  Steps per segment: {args.num_inference_steps}, guidance {args.guidance_scale}")
    size = f"{args.width}x{args.height}" if args.height and args.width else "auto (from reference image)"
    print(f"  Video size: {size}")
    print(f"{'=' * 60}\n")

    start = time.perf_counter()
    result = omni.generate(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "multi_modal_data": {
                "image": image,
                "pose_video": args.pose_video,
                "face_video": args.face_video,
            },
        },
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            generator=generator,
            extra_args={
                "segment_frame_length": args.segment_frame_length,
                "prev_segment_conditioning_frames": args.prev_segment_conditioning_frames,
            },
        ),
    )
    elapsed = time.perf_counter() - start
    print(f"Total generation time: {elapsed:.2f} s")

    output = OmniRequestOutput.unwrap_result(result)
    if not output.images:
        raise ValueError("No video frames found in OmniRequestOutput.")

    from diffusers.utils import export_to_video

    frames = to_frame_list(output.images)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=args.fps)

    print(f"Saved generated video to {output_path}")
    print(f"Video has {len(frames)} frames at {args.fps} fps ({len(frames) / args.fps:.1f}s)")


if __name__ == "__main__":
    main()
