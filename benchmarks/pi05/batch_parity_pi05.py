#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0.5 request-batch parity: one batched forward against each request alone.

For every batch size B, B requests that differ in prompt, state, images,
camera layout and seed go through ``Pi05Pipeline.forward`` once as one batch
and once one by one. The report gives max |batched - alone| per dtype and B,
and whether a lone request through the batch path is bit-identical to the
pre-batching call (``sample_actions(generator=g)`` on that request).

With ``--model`` it builds the real pipeline from a checkpoint and deploy
config, eager (batches of more than one run eagerly). Without it, it builds a
random-weight model at the real widths with two layers per tower, which runs
on CPU and checks the batching logic rather than the checkpoint.

    python batch_parity_pi05.py                                   # random weights, CPU
    python batch_parity_pi05.py --model lerobot/pi05_base --batch-sizes 1 2 4 8 16 32

Exits non-zero when a float32 row exceeds ``--atol``. bfloat16 is reported
only: its tolerance for batching is an open question in RFC #6867 (R2).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

PROMPTS = ["pick up the red block", "open the drawer", "", "place the cup on the plate and push it left"]


class _CharTokenizer:
    """Per-character ids, so different prompts reach the random model as different tokens."""

    def __call__(self, text, **kwargs):
        length = kwargs["max_length"]
        ids = [1 + ord(char) % 255 for char in text][:length]
        pad = length - len(ids)
        return {"input_ids": ids + [0] * pad, "attention_mask": [1] * len(ids) + [0] * pad}


def build_random_pipeline():
    """Real widths, two layers per tower, every weight perturbed off its init."""
    from vllm_omni.diffusion.models.pi05 import modeling_pi05
    from vllm_omni.diffusion.models.pi05.config import Pi05Config
    from vllm_omni.diffusion.models.pi05.modeling_pi05 import GemmaVariantConfig, Pi05ForActionPrediction
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline
    from vllm_omni.diffusion.models.pi05.processor_pi05 import Pi05Processor

    shrunk = {
        "gemma_2b": GemmaVariantConfig(2048, 2, 1024, 8, 1, 256),
        "gemma_300m": GemmaVariantConfig(1024, 2, 512, 8, 1, 256),
    }
    modeling_pi05.get_gemma_config = lambda variant: shrunk[variant]
    config = Pi05Config(
        max_action_dim=8,
        max_state_dim=8,
        chunk_size=50,
        n_action_steps=50,
        image_feature_keys=[
            "observation.images.base_0_rgb",
            "observation.images.left_wrist_0_rgb",
            "observation.images.right_wrist_0_rgb",
        ],
    )
    torch.manual_seed(0)
    model = Pi05ForActionPrediction(config).eval()
    # Zero-initialized parts would otherwise leave layers inert.
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.02)

    pipeline = object.__new__(Pi05Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.config = config
    pipeline.model = model
    pipeline._device = torch.device("cpu")
    pipeline.processor = Pi05Processor(config, _CharTokenizer(), pipeline._device)
    return pipeline


def build_checkpoint_pipeline(model: str, deploy_config: str, dtype: str):
    """The pipeline the server builds, eager, as ``bench_pi05.build_model`` does."""
    import yaml
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline

    if not os.path.isdir(model):
        from huggingface_hub import snapshot_download

        model = snapshot_download(repo_id=model, repo_type="model")
    with open(deploy_config, encoding="utf-8") as f:
        (stage,) = yaml.safe_load(f)["stages"]
    od_config = OmniDiffusionConfig(model=model, dtype=dtype, model_config=stage["model_config"], enforce_eager=True)
    with set_default_torch_dtype(od_config.dtype):
        return Pi05Pipeline(od_config=od_config)


def make_request(pipeline, seed: int, num_steps: int):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    cameras = list(pipeline.config.image_feature_keys)
    layouts = [cameras, cameras[::2], cameras[:1], cameras[1:]]
    rng = np.random.RandomState(seed)
    robot_obs = {
        "prompt": PROMPTS[seed % len(PROMPTS)],
        "state": rng.uniform(-1.0, 1.0, pipeline.config.state_dim).astype(np.float32),
        "images": {key: rng.randint(0, 256, (48, 64, 3), dtype=np.uint8) for key in layouts[seed % len(layouts)]},
    }
    return OmniDiffusionRequest(
        prompt="",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={"robot_obs": robot_obs},
            num_inference_steps=num_steps,
            generator=torch.Generator(device=pipeline._device).manual_seed(1000 + seed),
        ),
        request_id=f"row-{seed}",
    )


def pre_batching_call(pipeline, request) -> np.ndarray:
    """One request as the pipeline served it before request batching."""
    robot_obs = request.sampling_params.extra_args["robot_obs"]
    images, image_masks, lang_tokens, lang_masks = pipeline.processor.build_model_inputs(robot_obs)
    actions = pipeline.model.sample_actions(
        images=images,
        image_masks=image_masks,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        num_steps=request.sampling_params.num_inference_steps,
        generator=request.sampling_params.generator,
    )
    return pipeline.processor.build_model_outputs(actions, robot_obs)


def _synchronize(pipeline) -> None:
    if pipeline._device.type == "cuda":
        torch.accelerator.synchronize()


def measure(pipeline, dtype: str, batch_sizes: list[int], num_steps: int) -> list[dict]:
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

    rows = []
    for batch_size in batch_sizes:
        seeds = range(batch_size)
        start = time.perf_counter()
        batched = pipeline.forward(DiffusionRequestBatch([make_request(pipeline, s, num_steps) for s in seeds]))
        _synchronize(pipeline)
        batched_s = time.perf_counter() - start
        start = time.perf_counter()
        alone = [pipeline.forward(DiffusionRequestBatch([make_request(pipeline, s, num_steps)]))[0] for s in seeds]
        _synchronize(pipeline)
        alone_s = time.perf_counter() - start
        errors = [output.error for output in batched + alone if output.error]
        if errors:
            raise RuntimeError(f"B={batch_size}: {errors}")

        diffs = [float(np.abs(x.output["actions"] - y.output["actions"]).max()) for x, y in zip(batched, alone)]
        rows.append(
            {
                "dtype": dtype,
                "batch_size": batch_size,
                "max_abs_diff": max(diffs),
                "per_row_max_abs_diff": diffs,
                "max_abs_action": max(float(np.abs(y.output["actions"]).max()) for y in alone),
                "lone_request_matches_pre_batching_call": all(
                    np.array_equal(
                        pre_batching_call(pipeline, make_request(pipeline, s, num_steps)), y.output["actions"]
                    )
                    for s, y in zip(seeds, alone)
                ),
                "rows_distinct": len({y.output["actions"].tobytes() for y in alone}) == batch_size,
                # Wall clock of this check, not a benchmark: see bench_pi05.py.
                "batched_s": round(batched_s, 2),
                "alone_s": round(alone_s, 2),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="Checkpoint dir or HF repo id; omit for random weights on CPU")
    parser.add_argument("--deploy-config", default=None, help="Default: the package's vllm_omni/deploy/pi05.yaml")
    parser.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16"], choices=["float32", "bfloat16"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1e-4, help="float32 bound on max |batched - alone|")
    parser.add_argument("--out", default=None, help="Write the rows as JSON here")
    args = parser.parse_args()

    if args.deploy_config is None:
        import vllm_omni

        args.deploy_config = str(Path(vllm_omni.__file__).parent / "deploy" / "pi05.yaml")

    rows = []
    if args.model is None:
        from vllm_omni.diffusion.models.pi05.pipeline_pi05 import _set_inference_dtype

        pipeline = build_random_pipeline()
        for dtype in args.dtypes:
            _set_inference_dtype(pipeline.model, getattr(torch, dtype))
            rows += measure(pipeline, dtype, args.batch_sizes, args.steps)
    else:
        for dtype in args.dtypes:
            pipeline = build_checkpoint_pipeline(args.model, args.deploy_config, dtype)
            rows += measure(pipeline, dtype, args.batch_sizes, args.steps)
            del pipeline
            if torch.cuda.is_available():
                torch.accelerator.empty_cache()

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    failed = [
        row
        for row in rows
        if not row["lone_request_matches_pre_batching_call"]
        or (row["dtype"] == "float32" and row["max_abs_diff"] > args.atol)
    ]
    print(
        f"\n{len(rows) - len(failed)}/{len(rows)} rows within bounds (float32 atol={args.atol}; bfloat16 reported only)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
