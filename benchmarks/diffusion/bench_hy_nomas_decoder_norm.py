# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Precision and latency A/B for PixelUMM norms, optionally with real decoder weights.

Run from the repository root with CUDA/Triton. --model selects the directory
containing decoder/config.json and its safetensors index. Without it, only
operator microbenchmarks run. Every variant uses the same process and inputs;
quality checks and CPU copies are outside measured CUDA-event intervals.
"""

import argparse
import json
import statistics
from contextlib import contextmanager
from pathlib import Path

import torch

from vllm_omni.diffusion.models.hy_nomas import decoder as d
from vllm_omni.diffusion.models.hy_nomas import decoder_kernels as k


@contextmanager
def kernel_variant(name):
    """Select eager/norm/qk/both without adding a production configuration knob."""
    norm, qk = k.rms_norm, k.split_norm_rope
    k.rms_norm = norm if name in ("norm", "both") else lambda *args: None
    k.split_norm_rope = qk if name in ("qk", "both") else lambda *args: None
    try:
        yield
    finally:
        k.rms_norm, k.split_norm_rope = norm, qk


def measure(fn, warmups, repeats):
    """Return median/min/max CUDA-event milliseconds, excluding warmup."""
    for _ in range(warmups):
        fn()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del result
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def compare(name, fn, variants, args):
    """Check exact output parity and measure isolated variants of one operation."""
    with kernel_variant("eager"):
        expected = fn().cpu()
    results = []
    for variant in variants:
        with kernel_variant(variant):
            actual = fn().cpu()
            exact = torch.equal(actual, expected)
            row = {"operation": name, "variant": variant, "exact": exact}
            row["max_abs"] = (actual.float() - expected.float()).abs().max().item()
            assert exact, row
            row.update(measure(fn, args.warmups, args.repeats))
            results.append(row)
            print(json.dumps(row), flush=True)
    return results


def microbench(args):
    """Exercise the production norm and chunked QK preparation at window sizes."""
    norm = d.PixelUMMRMSNorm(4096, 1e-6).to(device="cuda", dtype=torch.bfloat16)
    norm.weight.copy_(torch.randn_like(norm.weight))
    x = torch.randn(args.tokens, 4096, device="cuda", dtype=torch.bfloat16)
    results = compare("rmsnorm", lambda x=x: norm(x), ("eager", "norm"), args)
    del x
    cos, sin = d.mrope_tables(d.PixelUMMArchitecture(), (1, 1, args.tokens), "cuda")

    class Geometry:
        arch = d.PixelUMMArchitecture()

    for heads in (8, 32):
        q = torch.randn(1, args.tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
        nt = d.PixelUMMRMSNorm(64, 1e-6).to(device="cuda", dtype=torch.bfloat16)
        nh = d.PixelUMMRMSNorm(64, 1e-6).to(device="cuda", dtype=torch.bfloat16)
        nt.weight.copy_(torch.randn_like(nt.weight))
        nh.weight.copy_(torch.randn_like(nh.weight))
        results += compare(
            f"qk_heads_{heads}",
            lambda: d.PixelUMMAttention._prepare_qk(Geometry(), q, cos, sin, nt, nh, q.dtype),
            ("eager", "qk"),
            args,
        )
    return results


def decoder_bench(args):
    """Compare complete decoding with identical checkpoint weights and latent input."""
    from safetensors.torch import load_file

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import build_decoder

    root = args.model / "decoder"
    config = json.loads((root / "config.json").read_text())
    cfg = OmniDiffusionConfig(model=str(args.model), model_class_name="HyNomasPipeline", dtype=torch.bfloat16)
    with set_current_diffusion_config(cfg), set_forward_context(omni_diffusion_config=cfg):
        model = build_decoder(config, torch.device("meta"), backend="native")
        model.to_empty(device="cuda")
        index = json.loads((root / "diffusion_pytorch_model.safetensors.index.json").read_text())
        state = {}
        for shard in sorted(set(index["weight_map"].values())):
            state.update(load_file(str(root / shard)))
        model.load_state_dict(state, strict=True)
        del state
        latent = torch.randn(1, 32, 10, 22, 40, device="cuda", dtype=torch.bfloat16)
        return compare(
            "decoder_704x1280_40frames", lambda: model.decode(latent).sample, ("eager", "norm", "qk", "both"), args
        )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tokens", type=int, default=35200)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens < 16 or args.warmups < 1 or args.repeats < 1:
        parser.error("tokens >= 16 and positive warmups/repeats are required")
    torch.manual_seed(args.seed)
    results = decoder_bench(args) if args.model else microbench(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "results": results}, indent=2)
    )


if __name__ == "__main__":
    main()
