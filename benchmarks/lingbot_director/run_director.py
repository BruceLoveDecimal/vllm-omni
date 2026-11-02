# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run one Director rollout against the served models.

Start the models with ``serve_stack.sh``, then for example::

    python benchmarks/lingbot_director/run_director.py \
        --task door-open --arm rebase --seed 11 --chunks 24 \
        --image /path/door.jpg --width 480 --height 832 \
        --world-url ws://127.0.0.1:8000/v1/realtime/video \
        --world-url ws://127.0.0.1:8004/v1/realtime/video \
        --output /path/runs
"""

from __future__ import annotations

# ruff: noqa: E402, I001
import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.lingbot_director.clients import (
    EditClient,
    SegmentClient,
    VlmClient,
    WorldSession,
    data_url,
    fit_image,
)
from benchmarks.lingbot_director.director import ARMS, Director
from benchmarks.lingbot_director.tasks import TASKS
from benchmarks.lingbot_world.workload import build_camera_script, num_frames_for_chunks

WORLD_MODEL = "robbyant/lingbot-world-v2-14b-causal-fast-diffusers"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=24)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--max-rebases", type=int, default=1)
    parser.add_argument(
        "--world-url",
        action="append",
        default=[],
        help="Repeat to add a standby world, so that restarts overlap the running segment.",
    )
    parser.add_argument("--world-model", default=WORLD_MODEL)
    parser.add_argument("--vlm-url", default="http://127.0.0.1:8001")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--edit-url", default="http://127.0.0.1:8002")
    parser.add_argument("--segment-url", default="http://127.0.0.1:8003")
    args = parser.parse_args(argv)
    if not args.world_url:
        args.world_url = ["ws://127.0.0.1:8000/v1/realtime/video"]
    return args


def session_start(args: argparse.Namespace, image: np.ndarray, prompt: str, num_chunks: int) -> dict:
    return {
        "type": "session.start",
        "model": args.world_model,
        "prompt": prompt,
        "image_reference": {"image_url": data_url(image)},
        "width": args.width,
        "height": args.height,
        "num_frames": num_frames_for_chunks(num_chunks),
        "fps": args.fps,
        "seed": args.seed,
        "extra_params": {"flow_shift": 5.0, "camera_action_script": build_camera_script(num_chunks, "hold")},
    }


def save_rollout(director: Director, directory: Path, fps: int) -> None:
    import imageio.v2 as imageio

    directory.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(directory / "rollout.mp4"), fps=fps, codec="libx264", quality=7) as writer:
        for frames in director.chunk_frames:
            for frame in frames:
                writer.append_data(frame)
    for number, keyframe in enumerate(director.keyframes):
        Image.fromarray(keyframe).save(directory / f"keyframe-{number}.png")
    (directory / "trace.json").write_text(json.dumps(director.trace.to_dict(), indent=2, ensure_ascii=False))


async def run(args: argparse.Namespace) -> dict:
    reference = fit_image(Image.open(args.image), args.width, args.height)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as http:
        director = Director(
            task=TASKS[args.task],
            arm=args.arm,
            seed=args.seed,
            total_chunks=args.chunks,
            world_urls=args.world_url,
            open_world=lambda url, image, prompt, num_chunks: WorldSession(
                url, session_start(args, image, prompt, num_chunks)
            ),
            vlm=VlmClient(http, args.vlm_url, args.vlm_model),
            segmenter=SegmentClient(http, args.segment_url),
            editor=EditClient(http, args.edit_url),
            max_rebases=args.max_rebases,
        )
        await director.run(reference)
    save_rollout(director, args.output / f"{args.task}-{args.seed}-{args.arm}", args.fps)
    return director.trace.to_dict()["summary"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(asyncio.run(run(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
