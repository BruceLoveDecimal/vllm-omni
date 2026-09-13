# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sequential released-checkpoint CUDA DiT parity, including cache eviction."""

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from .compare_reference import REFERENCE_REVISION


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    assert revision == REFERENCE_REVISION
    sys.path.insert(0, str(args.reference / "src"))
    from solarwm.backends.wan22.runtime.modeling import causal_model, model

    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
    from vllm_omni.diffusion.models.solarwm.components import load_transformer
    from vllm_omni.diffusion.models.solarwm.transformer import SolarWMCache
    from vllm_omni.platforms import current_omni_platform

    device = torch.device("cuda:0")
    config = json.loads((args.base / "transformer/config.json").read_text())
    config = {key: value for key, value in config.items() if not key.startswith("_")}
    with torch.device("meta"):
        reference = causal_model.CausalWanModel(
            **config,
            local_attn_size=18,
            sink_size=0,
            add_control_adapter=True,
            camera_attention_mode="fused_prope",
            use_echorope=False,
            frame_seq_length=405,
        )
    state = torch.load(args.stage / "model.pt", weights_only=True, mmap=True, map_location="cpu")["generator_ema"]
    reference.load_state_dict(
        {key.removeprefix("model."): value for key, value in state.items()}, strict=True, assign=True
    )
    reference = reference.to(device=device, dtype=torch.bfloat16).eval()
    d = config["dim"] // config["num_heads"]
    reference.freqs = torch.cat(
        [model.rope_params(1024, n) for n in (d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6))], dim=1
    )
    del state
    ref_cache = [
        {
            "k": torch.zeros(1, 7290, 24, 128, device=device, dtype=torch.bfloat16),
            "v": torch.zeros(1, 7290, 24, 128, device=device, dtype=torch.bfloat16),
            "global_end_index": torch.tensor(0, device=device),
            "local_end_index": torch.tensor(0, device=device),
        }
        for _ in range(30)
    ]
    cross_cache = [{"is_init": False} for _ in range(30)]
    generator = torch.Generator(device=device).manual_seed(42)
    context = torch.randn(1, 32, 4096, generator=generator, device=device, dtype=torch.bfloat16)
    cases = []

    def sdpa(q, k, v, **kwargs):
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    with (
        patch.object(causal_model, "attention", sdpa),
        patch.object(model, "flash_attention", sdpa),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        for start in range(0, 24, 3):
            x = torch.randn(1, 48, 3, 30, 54, generator=generator, device=device, dtype=torch.bfloat16)
            views = torch.eye(4, device=device).repeat(1, 3, 1, 1)
            views[:, :, 2, 3] = torch.arange(start, start + 3, device=device) * -0.02
            ks = torch.eye(3, device=device).repeat(1, 3, 1, 1)
            ks[..., 0, 0], ks[..., 1, 1] = 0.5050505, 0.8978676
            for step in (937.5, 0.0):
                t = torch.full((1, 3), step, device=device)
                expected = reference(
                    list(x),
                    t.repeat_interleave(405, dim=1),
                    list(context),
                    seq_len=1215,
                    y_camera={"viewmats": views.repeat_interleave(405, dim=1), "K": ks.repeat_interleave(405, dim=1)},
                    kv_cache=ref_cache,
                    crossattn_cache=cross_cache,
                    current_start=start * 405,
                    cache_update_policy="commit_detached" if step == 0 else "none",
                )
                cases.append((start, step, x.cpu(), views.cpu(), ks.cpu(), expected.cpu()))
            print(f"Reference chunk {start // 3 + 1}/8", flush=True)
    del reference, ref_cache, cross_cache
    gc.collect()
    current_omni_platform.empty_cache()
    rows = []
    with patch("vllm_omni.diffusion.attention.layer.get_attn_backend_for_role", return_value=(SDPABackend, None)):
        native = load_transformer(args.base, args.stage, device, torch.bfloat16)
        cache = SolarWMCache()
        for start, step, x, views, ks, expected in cases:
            actual = native(
                x.to(device),
                torch.full((1, 3), step, device=device),
                context,
                views.to(device),
                ks.to(device),
                cache=cache,
                start_frame=start,
                commit=step == 0,
            ).cpu()
            error = (actual.float() - expected.float()).abs()
            rows.append({"start": start, "step": step, "max_abs": error.max().item(), "mean_abs": error.mean().item()})
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
            print(rows[-1], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"reference": revision, "weight_role": "ema", "comparisons": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
