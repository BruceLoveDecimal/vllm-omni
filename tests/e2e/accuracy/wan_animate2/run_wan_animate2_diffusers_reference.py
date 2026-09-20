# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Produce the Wan2.2-Animate-2 reference video with diffusers' modular pipeline.

The reference implementation (``WanAnimate2ModularPipeline``, diffusers >= 0.40)
is a *baseline-only* dependency: it is not installed in the vLLM-Omni runtime
environment, so this runs in a separate environment and writes an MP4 plus a
metadata JSON that the similarity test compares against the vLLM-Omni output.

Use the **base** checkpoint for cross-implementation parity: diffusers applies
no attention bias for the distilled release (upstream's ``log_scale=-1.3``), so
the distilled outputs are not expected to match between the two.

Example::

    python tests/e2e/accuracy/wan_animate2/run_wan_animate2_diffusers_reference.py \\
        --model Wan-AI/Wan2.2-Animate-2-14B-Diffusers --image ref.png --video drive.mp4 \\
        --size 640x800 --fps 24 --num-frames 81 --num-inference-steps 40 --guidance-scale 3.0 \\
        --seed 42 --output ref.mp4 --metadata-output ref.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
from diffusers.modular_pipelines import WanAnimate2ModularPipeline
from diffusers.utils import export_to_video
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the diffusers Wan2.2-Animate-2 reference generation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True, help="Reference image path.")
    parser.add_argument("--video", required=True, help="Driving video path.")
    parser.add_argument("--prompt", default="static background.")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--size", default="640x800", help="Target area as WIDTHxHEIGHT.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-frames", type=int, default=None, help="Cap on driving frames after resampling.")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--segment-frame-length", type=int, default=81)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    return parser.parse_args()


def _decode_video(path: str) -> tuple[list[Image.Image], float]:
    with av.open(path) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate or stream.guessed_rate)
        frames = [Image.fromarray(frame.to_ndarray(format="rgb24")) for frame in container.decode(stream)]
    return frames, rate


def main() -> None:
    args = _parse_args()
    width, height = (int(value) for value in args.size.lower().split("x", 1))
    device = torch.device("cuda:0")

    pipeline = WanAnimate2ModularPipeline.from_pretrained(args.model)
    pipeline.load_components(torch_dtype=torch.bfloat16)
    pipeline.to(device)

    frames, source_fps = _decode_video(args.video)
    if args.num_frames is not None:
        # The modular pipeline resamples to `fps` itself; cap the *source* so
        # the resampled clip has at most `num_frames` frames.
        frames = frames[: int(np.ceil(args.num_frames * source_fps / args.fps))]

    inputs = {
        "image": Image.open(args.image).convert("RGB"),
        "driving_video": frames,
        "driving_video_fps": source_fps,
        "prompt": args.prompt,
        "height": height,
        "width": width,
        "fps": args.fps,
        "segment_frame_length": args.segment_frame_length,
        "num_inference_steps": args.num_inference_steps,
        "generator": torch.Generator(device=device).manual_seed(args.seed),
        "output_type": "np",
    }
    if args.negative_prompt is not None:
        inputs["negative_prompt"] = args.negative_prompt
    if hasattr(pipeline, "guider") and pipeline.guider is not None:
        pipeline.update_components(guider=pipeline.guider.new(guidance_scale=args.guidance_scale))

    videos = pipeline(**inputs, output="videos")
    video = videos[0]
    if args.num_frames is not None:
        video = video[: args.num_frames]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(list(video), str(output), fps=args.fps)
    Path(args.metadata_output).write_text(
        json.dumps(
            {
                "model": args.model,
                "num_frames": int(len(video)),
                "height": int(video[0].shape[0]),
                "width": int(video[0].shape[1]),
                "fps": args.fps,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "source_fps": source_fps,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
