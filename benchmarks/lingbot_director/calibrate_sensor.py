# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Score the served SAM 3 sensor against per-chunk human labels, before trusting it online.

``labels.json`` maps a blind code to one ``{"achieved": bool}`` per chunk and
``mapping.json`` maps that code to an episode named ``<task>-<seed>-<arm>``.
Every episode directory holds a ``rollout.mp4``. The report sweeps the ratio
limit, because the limit a task ships with was fitted to another SAM version.
"""

from __future__ import annotations

# ruff: noqa: E402, I001
import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.lingbot_director.clients import SegmentClient, decode_frames
from benchmarks.lingbot_director.tasks import TASKS, mask_span
from benchmarks.lingbot_world.workload import pixel_frames_in_chunk


def chunk_end_frames(num_chunks: int) -> list[int]:
    """Index of the last pixel frame of every chunk."""
    ends = []
    total = 0
    for chunk in range(num_chunks):
        total += pixel_frames_in_chunk(chunk)
        ends.append(total - 1)
    return ends


def balanced_accuracy(truth: Sequence[bool], predicted: Sequence[bool]) -> float:
    positive = [p for t, p in zip(truth, predicted) if t]
    negative = [p for t, p in zip(truth, predicted) if not t]
    recall = sum(positive) / len(positive)
    false_alarm = sum(negative) / len(negative)
    return (recall + 1 - false_alarm) / 2


async def episode_ratios(client: SegmentClient, video: Path, task_name: str, num_chunks: int) -> list[float]:
    task = TASKS[task_name]
    frames = decode_frames(video.read_bytes())
    reference = mask_span(await client.best_mask(frames[0], task.concept), task.axis)
    ratios = []
    for end in chunk_end_frames(num_chunks):
        mask = await client.best_mask(frames[end], task.concept)
        ratios.append(mask_span(mask, task.axis) / reference)
    return ratios


async def run(args: argparse.Namespace) -> None:
    labels = json.loads(args.labels.read_text())
    mapping = json.loads(args.mapping.read_text())
    ratios: list[float] = []
    truth: list[bool] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as http:
        client = SegmentClient(http, args.segment_url)
        for code, episode in sorted(mapping.items()):
            if not episode.startswith(f"{args.task}-"):
                continue
            chunks = labels[code]
            ratios.extend(await episode_ratios(client, args.runs / episode / "rollout.mp4", args.task, len(chunks)))
            truth.extend(bool(chunk["achieved"]) for chunk in chunks)
    task = TASKS[args.task]
    print(f"{args.task}: {len(truth)} chunks, {sum(truth)} labelled reached, shipped limit {task.ratio_limit}")
    for step in range(5, 21):
        limit = step / 10 if not task.shrinks else step / 20
        predicted = [ratio <= limit if task.shrinks else ratio >= limit for ratio in ratios]
        print(f"limit {limit:.2f}: balanced accuracy {balanced_accuracy(truth, predicted):.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--segment-url", default="http://127.0.0.1:8003")
    asyncio.run(run(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
