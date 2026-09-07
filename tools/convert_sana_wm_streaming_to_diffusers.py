#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert the NVlabs ``SANA-WM_streaming`` release into the diffusers layout
``SanaWmStreamingPipeline`` loads.

The upstream release (``hf://Efficient-Large-Model/SANA-WM_streaming``) ships
the distilled chunk-causal Stage-1 as a training checkpoint
``sana_dit/model.pt`` holding ``generator`` / ``critic`` state dicts plus
their optimizers. Only the generator is served. Its parameter names are the
bidirectional Stage-1 names with a ``model.`` prefix, so the conversion is a
prefix strip plus a one-to-one key check against a bidirectional diffusers
transformer (the architecture is identical; only the attention / FFN
behaviour differs, which the streaming fields of ``transformer/config.json``
select at runtime).

Output tree::

    <output>/
      model_index.json                     -> SanaWmStreamingPipeline
      transformer/config.json              -> bidirectional config + streaming fields
      transformer/diffusion_pytorch_model.safetensors
      vae/                                 -> copied (or symlinked) from --vae-dir

The VAE is the Stage-1 LTX-2 VAE of the bidirectional conversion: the
streaming release's causal VAE decoder is a follow-up (chunks are decoded
with a one-latent-frame overlap, see the design doc).

Example::

    python tools/convert_sana_wm_streaming_to_diffusers.py \\
        --checkpoint /models/SANA-WM_streaming/sana_dit/model.pt \\
        --reference /models/SANA-WM_bidirectional-stage1-diffusers \\
        --output /models/SANA-WM_streaming-stage1-diffusers
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GENERATOR_PREFIX = "model."

# Fields that turn the bidirectional transformer/config.json into the
# streaming one. Names mirror configs/sana_wm/sana_wm_streaming_1600m_720p.yaml
# and the streaming inference CLI defaults in NVlabs/Sana.
STREAMING_CONFIG_OVERRIDES: dict[str, object] = {
    "architecture_name": "SanaMSVideoCamCtrlStreaming_1600M_P1_D20",
    "streaming": True,
    "ffn_type": "CachedGLUMBConvTemp",
    "pos_embed_type": "casual_wan_rope",
    "chunk_size": 3,
    "chunk_split_strategy": "first_chunk_plus_one",
    "num_cached_blocks": 2,
    "sink_token": True,
    "denoising_step_list": [1000, 960, 889, 727, 0],
    "inference_flow_shift": 8.0,
    "scheduler_type": "self_forcing_flow_euler",
}

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _load_generator(checkpoint: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if isinstance(payload, dict) and "generator" in payload:
        state = payload["generator"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    elif isinstance(payload, dict):
        state = payload
    else:
        raise ValueError(f"Unrecognised checkpoint container in {checkpoint}: {type(payload).__name__}")
    converted: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Non-tensor entry {name!r} in generator state dict.")
        target = name.removeprefix(GENERATOR_PREFIX)
        if target in converted:
            raise ValueError(f"Duplicate parameter after prefix strip: {target!r}")
        converted[target] = tensor
    if not converted:
        raise ValueError(f"No generator tensors found in {checkpoint}.")
    return converted


def _reference_keys(reference_transformer: Path) -> dict[str, tuple[int, ...]]:
    with safe_open(str(reference_transformer), "pt") as handle:
        return {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}


def _check_keys(converted: dict[str, torch.Tensor], reference: dict[str, tuple[int, ...]]) -> None:
    missing = sorted(set(reference) - set(converted))
    unexpected = sorted(set(converted) - set(reference))
    mismatched = sorted(key for key in converted if key in reference and tuple(converted[key].shape) != reference[key])
    if missing or unexpected or mismatched:
        raise ValueError(
            "Streaming checkpoint does not map one-to-one onto the bidirectional transformer: "
            f"missing={missing[:10]} unexpected={unexpected[:10]} shape_mismatch={mismatched[:10]}"
        )


def _copy_or_link(source: Path, target: Path, *, symlink: bool) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{target} already exists.")
    if symlink:
        os.symlink(source.resolve(), target)
    else:
        shutil.copytree(source, target)


def convert(
    *,
    checkpoint: Path,
    reference: Path,
    output: Path,
    dtype: torch.dtype,
    symlink_vae: bool,
) -> None:
    reference_transformer = reference / "transformer" / "diffusion_pytorch_model.safetensors"
    reference_config = reference / "transformer" / "config.json"
    reference_vae = reference / "vae"
    for path in (checkpoint, reference_transformer, reference_config, reference_vae):
        if not path.exists():
            raise FileNotFoundError(path)

    output.mkdir(parents=True, exist_ok=True)
    transformer_dir = output / "transformer"
    transformer_dir.mkdir(exist_ok=True)

    converted = _load_generator(checkpoint)
    _check_keys(converted, _reference_keys(reference_transformer))
    tensors = {key: value.detach().to(dtype).contiguous() for key, value in converted.items()}
    save_file(tensors, str(transformer_dir / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})

    config = json.loads(reference_config.read_text(encoding="utf-8"))
    config.update(STREAMING_CONFIG_OVERRIDES)
    (transformer_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    _copy_or_link(reference_vae, output / "vae", symlink=symlink_vae)

    model_index = {
        "_class_name": "SanaWmStreamingPipeline",
        "_diffusers_version": config.get("_diffusers_version", "0.35.0"),
        "transformer": ["vllm_omni.diffusion.models.sana_wm.sana_wm_transformer", "SanaWmTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLLTX2Video"],
    }
    (output / "model_index.json").write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(tensors)} tensors ({dtype}) to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="sana_dit/model.pt of the streaming release")
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Bidirectional Stage-1 diffusers tree (transformer/ + vae/) used for the key check and the VAE",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--symlink-vae", action="store_true", help="Symlink vae/ instead of copying it")
    args = parser.parse_args()
    convert(
        checkpoint=args.checkpoint,
        reference=args.reference,
        output=args.output,
        dtype=DTYPES[args.dtype],
        symlink_vae=args.symlink_vae,
    )


if __name__ == "__main__":
    main()
