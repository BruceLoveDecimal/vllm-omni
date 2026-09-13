# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Released-weight auxiliary-component parity on one CUDA device."""

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import torch

from .compare_reference import REFERENCE_REVISION


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-frames", type=int, default=2)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    assert revision == REFERENCE_REVISION
    sys.path.insert(0, str(args.reference / "src"))
    from solarwm.backends.wan22.runtime.components import Wan5BVAE, WanTextEncoder
    from transformers import AutoTokenizer

    from vllm_omni.diffusion.models.solarwm.components import load_text_encoder, load_vae
    from vllm_omni.platforms import current_omni_platform

    device = torch.device("cuda:0")
    results = {"reference": revision, "torch": torch.__version__, "latent_frames": args.latent_frames}
    pixels = (
        torch.randn(1, 3, 4 * args.latent_frames - 3, 64, 96, generator=torch.Generator().manual_seed(42))
        .clamp(-1, 1)
        .to(device)
    )
    reference = Wan5BVAE(args.base / "vae/Wan2.2_VAE.pth").to(device)
    latent = reference.encode(pixels)
    expected = reference.decode_streaming(latent.to(torch.bfloat16), chunk_latent_frames=60).permute(0, 2, 1, 3, 4)
    latent = latent.cpu()
    del reference
    gc.collect()
    current_omni_platform.empty_cache()
    native = load_vae(args.base, device)
    mean = torch.tensor(native.config.latents_mean, device=device).view(1, 48, 1, 1, 1)
    std = torch.tensor(native.config.latents_std, device=device).view(1, 48, 1, 1, 1)
    actual_latent = ((native.encode(pixels).latent_dist.mode() - mean) / std).permute(0, 2, 1, 3, 4).cpu()
    results["vae_encoder_max_abs"] = (actual_latent - latent).abs().max().item()
    torch.testing.assert_close(actual_latent, latent, rtol=1e-4, atol=1e-4)
    decoded = []
    # Autocast promotes the division; keep denormalization inside its scope.
    z = latent.to(device=device, dtype=torch.bfloat16).permute(0, 2, 1, 3, 4)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = z / (1 / std.to(z.dtype)) + mean.to(z.dtype)
        native.decode_with_chunks(z, on_chunk=lambda chunk: decoded.append(chunk.float().cpu()))
    actual = torch.cat(decoded, dim=2).clamp(-1, 1)
    results["vae_decoder_max_abs"] = (actual - expected).abs().max().item()
    results["vae_decoder_mean_abs"] = (actual - expected).abs().mean().item()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    del native
    decoded.clear()
    gc.collect()
    current_omni_platform.empty_cache()
    text = "A cinematic view of a quiet mountain village, the camera slowly moves forward."
    reference = WanTextEncoder(args.base / "text_encoder/models_t5_umt5-xxl-enc-bf16.pth", args.base / "tokenizer").to(
        device
    )
    expected = reference([text])["prompt_embeds"].cpu()
    del reference
    gc.collect()
    current_omni_platform.empty_cache()
    native = load_text_encoder(args.base, device)
    tokenizer = AutoTokenizer.from_pretrained(args.base / "tokenizer", local_files_only=True)
    tokens = tokenizer([text], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    tokens = {key: value.to(device) for key, value in tokens.items()}
    actual = native(**tokens).last_hidden_state
    actual = actual.masked_fill(~tokens["attention_mask"].bool().unsqueeze(-1), 0).cpu()
    results["text_max_abs"] = (actual - expected).abs().max().item()
    results["text_mean_abs"] = (actual - expected).abs().mean().item()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
