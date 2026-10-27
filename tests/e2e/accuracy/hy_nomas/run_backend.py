# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One isolated backend run; invoked by test_native_similarity.py."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import psutil
import pynvml


class MemorySampler:
    """Device-wide sampled usage, not allocator peak; run on an idle single GPU."""

    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get("HY_NOMAS_NVML_INDEX", "0")))
        self.stop = threading.Event()
        self.gpu_bytes = 0
        self.cpu_rss_bytes = 0
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            self.gpu_bytes = max(self.gpu_bytes, pynvml.nvmlDeviceGetMemoryInfo(self.handle).used)
            processes = [psutil.Process(), *psutil.Process().children(recursive=True)]
            rss = 0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            self.cpu_rss_bytes = max(self.cpu_rss_bytes, rss)
            self.stop.wait(0.05)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        pynvml.nvmlShutdown()


def record(args, index, videos, turns, elapsed, memory):
    video = np.concatenate(videos)
    expected = args.frames * (3 if args.scenario == "session" else 1)
    assert video.shape == (expected, args.height, args.width, 3), video.shape
    assert video.dtype == np.uint8
    measured = index >= args.warmups
    result = {
        "iteration": index,
        "measured": measured,
        "elapsed_s": elapsed,
        "turns": turns,
        "frames": len(video),
        "rgb_sha256": hashlib.sha256(video).hexdigest(),
        "gpu_sampled_peak_bytes": memory.gpu_bytes,
        "process_tree_rss_peak_bytes": memory.cpu_rss_bytes,
    }
    # Keep the first measured clip; hashes expose variation in later repeats.
    if index == args.warmups:
        np.save(args.output / f"video_{index - args.warmups}.npy", video)
    with (args.output / "runs.jsonl").open("a") as stream:
        stream.write(json.dumps(result) + "\n")
    print(json.dumps(result), flush=True)


def run_video(args):
    import av

    from tests.e2e.accuracy.helpers import send_video_request_with_timeout
    from tests.helpers.runtime import OmniServer, OnlineOmniClient

    serve_args = [
        "--deploy-config",
        "vllm_omni/deploy/hy_nomas.yaml",
        "--custom-pipeline-args",
        json.dumps({f"{args.component}_backend": args.backend}),
    ]
    if args.offload:
        serve_args.append("--enable-cpu-offload")
    started = time.perf_counter()
    with OmniServer(args.model, serve_args, startup_timeout=1200) as server:
        (args.output / "startup.json").write_text(json.dumps({"seconds": time.perf_counter() - started}))
        client = OnlineOmniClient(port=server.port)
        for index in range(args.warmups + args.repeats):
            with MemorySampler() as memory:
                started = time.perf_counter()
                content = send_video_request_with_timeout(
                    client,
                    {
                        "form_data": {
                            "prompt": "A forest beside a clear river, photorealistic, daylight.",
                            "size": f"{args.width}x{args.height}",
                            "num_frames": args.frames,
                            "fps": 24,
                            "num_inference_steps": args.steps,
                            "guidance_scale": 1,
                            "seed": args.seed,
                        }
                    },
                    timeout_seconds=3600,
                    poll_interval_seconds=0.1,
                )
                elapsed = time.perf_counter() - started
            if index == args.warmups:
                (args.output / "video.mp4").write_bytes(content)
            with av.open(BytesIO(content)) as container:
                video = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
            record(args, index, [video], [], elapsed, memory)


async def run_session(args):
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.experimental.ar_diffusion.consumer import ARDiffusionOmniTickConsumer
    from vllm_omni.experimental.ar_diffusion.session import ARDiffusionWorkerLifecycle
    from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionControlInput, ARDiffusionTickRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    started = time.perf_counter()
    engine = AsyncOmni(
        model=args.model,
        enforce_eager=True,
        dtype="bfloat16",
        distributed_executor_backend="mp",
        max_num_seqs=1,
        enable_cpu_offload=args.offload,
        engine_backend="vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine",
        step_execution=False,
        parallel_config={"ulysses_degree": 1},
        custom_pipeline_args={
            f"{args.component}_backend": args.backend,
            "ar_height": args.height,
            "ar_width": args.width,
            "ar_max_latent_frames": 48,
            "ar_max_segments": 4,
        },
        model_config={"ar_diffusion_kv_config": {"gpu_memory_fraction": 0.02, "warmup_cudagraph": False}},
    )
    lifecycle = ARDiffusionWorkerLifecycle(engine, stage_ids=[0], timeout=180)
    consumer = ARDiffusionOmniTickConsumer(
        engine,
        prompt_provider=lambda tick: tick.prompt,
        sampling_params_list=[
            OmniDiffusionSamplingParams(
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                seed=args.seed,
                num_inference_steps=args.steps,
                guidance_scale=1,
                output_type="np",
            )
        ],
        diffusion_stage_id=0,
        output_modalities=["video"],
    )
    (args.output / "startup.json").write_text(json.dumps({"seconds": time.perf_counter() - started}))
    try:
        for index in range(args.warmups + args.repeats):
            session_id = f"comparison-{index}"
            videos: list[np.ndarray] = []
            turns = []
            try:
                with MemorySampler() as memory:
                    started = time.perf_counter()
                    for turn, prompt in enumerate(
                        [
                            "A forest beside a clear river, photorealistic, daylight.",
                            "The camera approaches a wooden bridge beside the river.",
                            "The camera approaches a wooden bridge beside the river.",
                        ]
                    ):
                        controls = (
                            ()
                            if turn < 2
                            else (
                                ARDiffusionControlInput(
                                    "camera", "hy_nomas.camera.v1", {"camera_trajectory": "pan_left"}
                                ),
                            )
                        )
                        tick = ARDiffusionTickRequest(
                            session_id,
                            f"{session_id}-{turn}",
                            turn,
                            prompt=prompt,
                            controls=controls,
                            applied_event_ids=(turn,),
                            reset=turn == 0,
                            close_session=turn == 2,
                        )
                        turn_start = time.perf_counter()
                        output = await consumer.execute_tick(tick)
                        video = output.images[0]
                        metadata = output.multimodal_output["metadata"]["hy_nomas"]
                        assert metadata["frame_start"] == sum(len(v) for v in videos)
                        if turn == 2:
                            assert metadata["withheld_frames"] == 0
                        turns.append(
                            {
                                "turn": turn,
                                "elapsed_s": time.perf_counter() - turn_start,
                                "available_s": time.perf_counter() - started,
                                "frames": len(video),
                                "metadata": metadata,
                            }
                        )
                        videos.append(video)
                    elapsed = time.perf_counter() - started
                record(args, index, videos, turns, elapsed, memory)
            finally:
                await lifecycle.close_session(session_id)
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--component", choices=("dit", "decoder"), required=True)
    parser.add_argument("--backend", choices=("reference", "native"), required=True)
    parser.add_argument("--scenario", choices=("video", "session"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--offload", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "request.json").write_text(json.dumps(vars(args), default=str, indent=2))
    environment = {name: importlib.metadata.version(name) for name in ("torch", "vllm", "diffusers", "transformers")}
    import hymm

    environment["reference_root"] = str(next(iter(hymm.__path__)))
    environment["python"] = sys.version
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get("HY_NOMAS_NVML_INDEX", "0")))
    environment["gpu"] = pynvml.nvmlDeviceGetName(handle)
    environment["driver"] = pynvml.nvmlSystemGetDriverVersion()
    pynvml.nvmlShutdown()
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2))
    if args.scenario == "video":
        run_video(args)
    else:
        asyncio.run(run_session(args))


if __name__ == "__main__":
    main()
