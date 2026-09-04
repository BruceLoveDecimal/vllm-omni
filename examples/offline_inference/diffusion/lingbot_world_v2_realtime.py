# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run LingBot-World 2.0 realtime generation in process.

This example exercises internal APIs. It is not a public HTTP or WebSocket
protocol. Two modes share one engine and one output layout:

- ``--mode tick`` drives an AR-Diffusion session where each JSONL event is one
  ``generate()`` call producing one three-latent-frame AR block.
- ``--mode stepwise`` issues a single ``generate()`` with step execution and
  streaming output; each streamed output is one AR block.

Both modes write one latent plus identity metadata per chunk. The current
implementation supports at most ten chunks per world because the image
condition is bounded to 117 pixel frames; in tick mode, reset or create a
session to start a new world.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_MODEL = "robbyant/lingbot-world-v2-14b-causal-fast-diffusers"
_CAMERA_ACTION_SCHEMA = "lingbot.camera_actions.v1"
_FRAMES_PER_BLOCK = 3
_TEMPORAL_COMPRESSION = 4
# ((117 pixel frames - 1) / VAE temporal factor 4 + 1) / 3 latent frames.
_MAX_CHUNKS = 10


def _camera_event_data(frames: list[list[str]]) -> dict[str, Any]:
    """Build the event-side script consumed by LingBotCameraControlReducer."""
    return {"mode": "script", "frames": frames}


def _idle_frames() -> list[list[str]]:
    return [[] for _ in range(_FRAMES_PER_BLOCK)]


def _num_frames_for_chunks(num_chunks: int) -> int:
    latent_frames = num_chunks * _FRAMES_PER_BLOCK
    return (latent_frames - 1) * _TEMPORAL_COMPRESSION + 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run in-process realtime LingBot-World 2.0 generation.")
    parser.add_argument(
        "--mode",
        choices=("tick", "stepwise"),
        default="tick",
        help="tick: one generate() per AR block through the session manager; "
        "stepwise: one generate() streaming every AR block.",
    )
    parser.add_argument("--model", default=_MODEL, help="Hugging Face model ID or local checkpoint path.")
    parser.add_argument("--image", required=True, help="Initial RGB image.")
    parser.add_argument("--prompt", required=True, help="Initial scene prompt.")
    parser.add_argument(
        "--events",
        help="JSONL file with one event per AR block; at most 10 events. "
        "Required in tick mode. In stepwise mode only frames are used and "
        "the file defaults to --chunks idle camera chunks.",
    )
    parser.add_argument("--chunks", type=int, default=3, help="Stepwise AR block count when --events is omitted.")
    parser.add_argument("--output-dir", required=True, help="Directory for chunk latents and metadata.")
    parser.add_argument(
        "--session-id",
        default="lingbot-world",
        help="Persistent world session identifier; also the request id in stepwise mode.",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args(argv)


def _load_events(path: Path, *, mode: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"events line {line_number} is not valid JSON: {exc.msg}.") from None
        if not isinstance(value, dict):
            raise ValueError(f"events line {line_number} must be a JSON object.")
        event_id = value.get("event_id")
        prompt = value.get("prompt")
        frames = value.get("frames")
        if mode == "tick":
            if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id < 0:
                raise ValueError(f"events line {line_number} requires a non-negative integer event_id.")
            if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
                raise ValueError(f"events line {line_number} prompt must be a non-empty string.")
            if prompt is None and frames is None:
                raise ValueError(f"events line {line_number} must update prompt and/or frames.")
        else:
            if prompt is not None:
                raise ValueError(f"events line {line_number}: stepwise mode does not support per-chunk prompt updates.")
            if frames is None:
                frames = _idle_frames()
        if frames is not None:
            if not isinstance(frames, list) or len(frames) != _FRAMES_PER_BLOCK:
                raise ValueError(f"events line {line_number} frames must contain exactly three lists.")
            for frame in frames:
                if not isinstance(frame, list) or any(not isinstance(action, str) for action in frame):
                    raise ValueError(f"events line {line_number} frames must contain only action strings.")
        events.append({"event_id": event_id, "prompt": prompt, "frames": frames})
    if not events:
        raise ValueError("events file must contain at least one event.")
    if len(events) > _MAX_CHUNKS:
        raise ValueError(
            "events file must contain at most "
            f"{_MAX_CHUNKS} events because the current LingBot realtime "
            "image-condition horizon is 117 pixel frames."
        )
    if mode == "tick":
        event_ids = [event["event_id"] for event in events]
        if event_ids != sorted(set(event_ids)):
            raise ValueError("event_id values must be unique and strictly increasing.")
    return events


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path | None, Path]:
    image = Path(args.image).expanduser().resolve()
    events = Path(args.events).expanduser().resolve() if args.events else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not image.is_file():
        raise ValueError("--image must point to an existing file.")
    if events is None and args.mode == "tick":
        raise ValueError("--events is required in tick mode.")
    if events is not None and not events.is_file():
        raise ValueError("--events must point to an existing JSONL file.")
    if events is None and (args.chunks <= 0 or args.chunks > _MAX_CHUNKS):
        raise ValueError(f"--chunks must be between 1 and {_MAX_CHUNKS}.")
    if not args.prompt.strip():
        raise ValueError("--prompt must contain non-whitespace text.")
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must be positive multiples of 16.")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive.")
    if not math.isfinite(args.gpu_memory_fraction) or not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be in (0, 1].")
    return image, events, output_dir


def _build_engine(args: argparse.Namespace):
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    stepwise = args.mode == "stepwise"
    return AsyncOmni(
        model=args.model,
        engine_backend="vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine",
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=1,
        step_execution=stepwise,
        diffusion_streaming_output=stepwise,
        model_config={
            "ar_diffusion_height": args.height,
            "ar_diffusion_width": args.width,
            "ar_diffusion_kv_config": {
                "gpu_memory_fraction": args.gpu_memory_fraction,
                "warmup_cudagraph": True,
            },
        },
    )


def _build_sampling(args: argparse.Namespace, *, num_frames: int, extra_args: dict[str, Any]):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    return OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=num_frames,
        num_inference_steps=4,
        max_sequence_length=512,
        seed=args.seed,
        output_type="latent",
        extra_args={"flow_shift": 5.0, **extra_args},
    )


def _record_chunk(
    output_dir: Path,
    chunk_index: int,
    output: Any,
    metadata: dict[str, Any],
    **fields: Any,
) -> dict[str, Any]:
    import torch

    images = getattr(output, "images", None)
    if not images or len(images) != 1 or not isinstance(images[0], torch.Tensor):
        raise RuntimeError("Expected one latent tensor from each realtime LingBot chunk.")
    latent = images[0].detach().float().cpu()
    torch.save(latent, output_dir / f"chunk_{chunk_index:03d}.pt")
    (output_dir / f"chunk_{chunk_index:03d}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    measurement = {
        "chunk_index": chunk_index,
        "shape": list(latent.shape),
        "finite": bool(torch.isfinite(latent).all()),
        "metadata": metadata,
        **fields,
    }
    print(json.dumps(measurement, sort_keys=True), flush=True)
    return measurement


async def _run_tick(args: argparse.Namespace, image: Path, events: list[dict[str, Any]], output_dir: Path):
    from vllm_omni.diffusion.models.lingbot_world.actions import LingBotCameraControlReducer
    from vllm_omni.experimental.ar_diffusion.consumer import ARDiffusionOmniTickConsumer
    from vllm_omni.experimental.ar_diffusion.session import (
        ARDiffusionSessionEvent,
        ARDiffusionSessionManager,
        ARDiffusionWorkerLifecycle,
    )
    from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionControlInput

    engine = _build_engine(args)
    sampling = _build_sampling(args, num_frames=_num_frames_for_chunks(1), extra_args={})
    consumer = ARDiffusionOmniTickConsumer(
        engine,
        prompt_provider=lambda tick: {
            "prompt": tick.prompt,
            "multi_modal_data": {"image": str(image)},
        },
        sampling_params_list=[sampling],
        diffusion_stage_id=0,
    )
    manager = ARDiffusionSessionManager(
        tick_consumer=consumer,
        lifecycle=ARDiffusionWorkerLifecycle(engine, stage_ids=[0], timeout=180.0),
        max_pending_events=32,
        control_reducer_factory=LingBotCameraControlReducer,
    )
    session = await manager.create_session(args.session_id)
    measurements: list[dict[str, Any]] = []
    try:
        for chunk_index, event in enumerate(events):
            controls = ()
            if event["frames"] is not None:
                controls = (
                    ARDiffusionControlInput(
                        track="camera",
                        schema=_CAMERA_ACTION_SCHEMA,
                        data=_camera_event_data(event["frames"]),
                    ),
                )
            await session.accept_event(
                ARDiffusionSessionEvent(
                    event_id=event["event_id"],
                    prompt=event["prompt"]
                    if event["prompt"] is not None
                    else (args.prompt if chunk_index == 0 else None),
                    controls=controls,
                )
            )
            started = time.perf_counter()
            output = await session.next_chunk()
            elapsed = time.perf_counter() - started
            metadata = consumer.chunk_metadata(output).to_dict()
            measurements.append(_record_chunk(output_dir, chunk_index, output, metadata, latency_seconds=elapsed))
    finally:
        try:
            await manager.close_session(args.session_id)
        finally:
            engine.shutdown()
    return measurements


def _chunk_metadata(output: Any) -> dict[str, Any]:
    multimodal = getattr(output, "multimodal_output", None) or {}
    metadata = multimodal.get("metadata") if isinstance(multimodal, dict) else None
    if not isinstance(metadata, dict) or not isinstance(metadata.get("ar_diffusion"), dict):
        raise RuntimeError("Expected ar_diffusion metadata on each streamed LingBot chunk.")
    return dict(metadata["ar_diffusion"])


async def _run_stepwise(args: argparse.Namespace, image: Path, events: list[dict[str, Any]], output_dir: Path):
    script = [event["frames"] for event in events]
    engine = _build_engine(args)
    sampling = _build_sampling(
        args,
        num_frames=_num_frames_for_chunks(len(script)),
        extra_args={"camera_action_script": script},
    )
    prompt = {"prompt": args.prompt, "multi_modal_data": {"image": str(image)}}
    measurements: list[dict[str, Any]] = []
    try:
        async for output in engine.generate(prompt, sampling, request_id=args.session_id):
            if not getattr(output, "images", None):
                continue
            chunk_index = len(measurements)
            measurements.append(
                _record_chunk(
                    output_dir,
                    chunk_index,
                    output,
                    _chunk_metadata(output),
                    finished=bool(getattr(output, "finished", False)),
                )
            )
    finally:
        engine.shutdown()
    if len(measurements) != len(script):
        raise RuntimeError(f"Expected {len(script)} streamed chunks, got {len(measurements)}.")
    return measurements


async def run(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    image, events_path, output_dir = _validate_args(args)
    if events_path is not None:
        events = _load_events(events_path, mode=args.mode)
    else:
        events = [{"event_id": None, "prompt": None, "frames": _idle_frames()} for _ in range(args.chunks)]
    output_dir.mkdir(parents=True, exist_ok=True)

    # vllm_omni imports stay inside the mode runners so --help and helper
    # tests do not require a CUDA-enabled vLLM installation.
    if args.mode == "tick":
        measurements = await _run_tick(args, image, events, output_dir)
    else:
        measurements = await _run_stepwise(args, image, events, output_dir)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"chunks": measurements}, indent=2, sort_keys=True) + "\n")
    return summary_path


def main(argv: Sequence[str] | None = None) -> Path:
    return asyncio.run(run(argv))


if __name__ == "__main__":
    main()
