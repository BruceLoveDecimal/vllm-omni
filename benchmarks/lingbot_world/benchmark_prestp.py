# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Matched offline LingBot runs with optional PreSTP and lossless pixel outputs."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from examples.offline_inference.diffusion import lingbot_world_v2 as recipe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prestp", action="store_true")
    parser.add_argument("--spatial-threshold", type=float, default=0.97)
    parser.add_argument("--temporal-threshold", type=float, default=0.997)
    parser.add_argument("--reuse-threshold", type=float, default=0.5)
    parser.add_argument("--reanchor-period", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--result-dir", type=Path, required=True)
    options, recipe_args = parser.parse_known_args()
    if options.repetitions < 1:
        parser.error("repetitions must be positive")
    args = recipe.parse_args(recipe_args)
    paths = recipe.resolve_cli_paths(args)
    prompt, sampling = recipe.build_request(args, paths)
    config = recipe.build_omni_kwargs(args, paths)
    config["enforce_eager"] = True
    config["max_num_seqs"] = 1
    pruning = None
    if options.prestp:
        pruning = {
            "spatial_threshold": options.spatial_threshold,
            "temporal_threshold": options.temporal_threshold,
            "reuse_threshold": options.reuse_threshold,
            "reanchor_period": options.reanchor_period,
        }
        config["model_config"]["lingbot_prestp"] = pruning
    config.update({f"stage_0_{key}": value for key, value in list(config.items()) if key != "model"})
    options.result_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import vllm
    from diffusers.utils import export_to_video

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    report = {
        "prestp": pruning,
        "model": config["model"],
        "sampling": sampling,
        "prompt": args.prompt,
        "image": str(paths.image),
        "action_path": str(paths.action_root / paths.action_relative),
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "runs": [],
    }
    engine = Omni(**config)
    try:
        for repetition in range(options.repetitions):
            start = time.perf_counter()
            output = engine.generate(prompt, OmniDiffusionSamplingParams(**sampling), use_tqdm=False)
            elapsed = time.perf_counter() - start
            frames = recipe.extract_video_array(output)
            if not np.isfinite(frames).all():
                raise RuntimeError("Generated video contains nonfinite pixels")
            np.save(options.result_dir / f"pixels-{repetition}.npy", frames)
            export_to_video(frames, str(options.result_dir / f"video-{repetition}.mp4"), fps=args.fps)
            report["runs"].append({"repetition": repetition, "generate_seconds": elapsed, "shape": list(frames.shape)})
            (options.result_dir / "results.json").write_text(json.dumps(report, indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
