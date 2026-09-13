# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare native rollout with the pinned reference's actual Stage2 sampler.

The reference sampler runs through an adapter to the already-qualified native
DiT. This isolates schedule, RNG order, first-frame restoration, and commits.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from .compare_reference import REFERENCE_REVISION


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-frames", type=int, default=24)
    args = parser.parse_args()
    assert (
        subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
        == REFERENCE_REVISION
    )
    sys.path.insert(0, str(args.reference / "src"))
    from solarwm.backends.wan22.runtime.components import WanDiffusion
    from solarwm.backends.wan22.runtime.scheduler import FlowMatchScheduler
    from solarwm.backends.wan22.runtime.stage2 import _stage2_self_forcing_latents

    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.diffusion.models.solarwm.pipeline import SolarWMStage2Pipeline, prepare_camera, prepare_image
    from vllm_omni.diffusion.models.solarwm.transformer import SolarWMCache
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    config = SimpleNamespace(
        model=str(args.model),
        parallel_config=DiffusionParallelConfig(),
        cache_backend=None,
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        quantization_config=None,
    )
    with patch("vllm_omni.diffusion.attention.layer.get_attn_backend_for_role", return_value=(SDPABackend, None)):
        pipe = SolarWMStage2Pipeline(od_config=config)
    device = pipe.device
    image = Image.open(args.image).convert("RGB")
    prompt = "A cat in sunglasses on a boat, gentle natural motion."
    frames = args.latent_frames
    if frames < 3 or frames % 3:
        parser.error("--latent-frames must be a positive multiple of 3")
    sampling = OmniDiffusionSamplingParams(
        height=480,
        width=864,
        num_frames=4 * frames - 3,
        num_inference_steps=4,
        guidance_scale=1.0,
        seed=42,
        output_type="latent",
    )
    actual = pipe(
        SimpleNamespace(prompts=[{"prompt": prompt, "multi_modal_data": {"image": image}}], sampling_params=sampling)
    ).output
    views, ks = prepare_camera(None, frames, device)
    pixels = prepare_image(image, 480, 864, device)
    mean = torch.tensor(pipe.vae.config.latents_mean, device=device).view(1, 48, 1, 1, 1)
    std = torch.tensor(pipe.vae.config.latents_std, device=device).view(1, 48, 1, 1, 1)
    first = ((pipe.vae.encode(pixels).latent_dist.mode() - mean) * (1 / std)).to(torch.bfloat16).permute(0, 2, 1, 3, 4)
    tokens = pipe.tokenizer([prompt], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    context = pipe.text_encoder(**tokens).last_hidden_state
    context = context.masked_fill(~tokens["attention_mask"].bool().unsqueeze(-1), 0).to(torch.bfloat16)

    class Adapter:
        scheduler = FlowMatchScheduler(shift=5.0)
        flow_to_x0 = staticmethod(WanDiffusion.flow_to_x0)

        def __call__(self, x, condition, camera, t, **kwargs):
            out = pipe.transformer(
                x.permute(0, 2, 1, 3, 4),
                t[:, ::405],
                condition["prompt_embeds"],
                camera["viewmats"][:, ::405],
                camera["K"][:, ::405],
                cache=kwargs["kv_cache"][0]["native"],
                start_frame=kwargs["current_start"] // 405,
                commit=kwargs["cache_update_policy"] == "commit_detached",
            )
            return out.permute(0, 2, 1, 3, 4)

    provider = SimpleNamespace(
        diffusion=Adapter(),
        device=device,
        config={
            "model": {"num_frame_per_block": 3, "latent_channels": 48, "frame_sequence_length": 405},
            "data": {"latent_shape": [48, frames, 30, 54]},
            "train": {"denoising_step_list": [1000, 750, 500, 250], "num_train_timesteps": 1000},
        },
        allocate_kv_cache=lambda *a, **k: [{"_fused_prope_camera_metadata": True, "native": SolarWMCache()}],
        allocate_crossattn_cache=lambda *a, **k: [],
        _noise=lambda shape, generator: torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16),
    )
    expected, schedule = _stage2_self_forcing_latents(
        provider,
        SimpleNamespace(solver="self_forcing", num_inference_steps=4, rollout_latent_frames=frames),
        first,
        {"prompt_embeds": context},
        {"viewmats": views.repeat_interleave(405, 1), "K": ks.repeat_interleave(405, 1)},
        torch.Generator(device=device).manual_seed(42),
    )
    expected = expected.permute(0, 2, 1, 3, 4)
    result = {
        "reference": REFERENCE_REVISION,
        "max_abs": (actual - expected).abs().max().item(),
        "mean_abs": (actual - expected).abs().mean().item(),
        "latent_frames": frames,
        "schedule": schedule,
    }
    print(json.dumps(result, indent=2))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
