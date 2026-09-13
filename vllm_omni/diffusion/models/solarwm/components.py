# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict original-checkpoint loading using shared Wan VAE and UMT5 modules."""

import inspect
import json
from pathlib import Path

import torch
from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm
from transformers import UMT5EncoderModel

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import OmniAutoencoderKLWan
from vllm_omni.diffusion.models.wan2_2.text_encoder import _WAN_UMT5_CONFIG, _load_wan_t5_as_umt5

from .transformer import SolarWMTransformer


def load_transformer(base: Path, stage: Path, device: torch.device, dtype: torch.dtype):
    release = json.loads((stage / "release-manifest.json").read_text())
    contract = json.loads((stage / "checkpoint-manifest.json").read_text())["contract"]
    if (contract["family"], contract["stage"], contract["objective"], contract["camera_translation_transform"]) != (
        "wan22_ti2v_5b",
        "stage2",
        "flow_matching",
        "linear",
    ):
        raise ValueError("Expected a SolarWM-5B Stage2 flow-matching checkpoint with linear camera translation")
    attention = contract["extras"]["attention"]
    if (
        attention["sink_size"] != 0
        or attention["use_echorope"]
        or attention["local_attn_size"] != 18
        or attention["max_prior_clean_chunks"] != 5
    ):
        raise ValueError("Unsupported SolarWM attention contract")
    if contract["extras"]["denoising_step_list"] != [1000, 750, 500, 250]:
        raise ValueError("Unsupported SolarWM denoising schedule")
    role = release["load"]["default_weights"]
    role_key = {"live": "generator", "ema": "generator_ema"}.get(role)
    if role_key is None:
        raise ValueError(f"Unsupported checkpoint weight role {role!r}")
    config = json.loads((base / "transformer/config.json").read_text())
    parameters = inspect.signature(SolarWMTransformer).parameters
    kwargs = {key: value for key, value in config.items() if key in parameters}
    with torch.device("meta"):
        transformer = SolarWMTransformer(**kwargs)
    state = torch.load(stage / "model.pt", map_location="cpu", weights_only=True, mmap=True)[role_key]
    state = {name.removeprefix("model."): value for name, value in state.items()}
    transformer.load_state_dict(state, strict=True, assign=True)
    return transformer.to(device=device, dtype=dtype).eval().requires_grad_(False)


def load_text_encoder(base: Path, device: torch.device):
    with torch.device("meta"):
        encoder = UMT5EncoderModel(_WAN_UMT5_CONFIG)
    encoder = _load_wan_t5_as_umt5(
        encoder, str(base / "text_encoder/models_t5_umt5-xxl-enc-bf16.pth"), dtype=torch.float32
    )
    return encoder.to(device).eval().requires_grad_(False)


def convert_vae_weights(state):
    # Diffusers handles the common quant/mid/head naming. Wan2.2 residual
    # down/up blocks have an extra nesting level absent from the 2.1 mapper.
    common = {
        key: value for key, value in state.items() if not key.startswith(("encoder.downsamples.", "decoder.upsamples."))
    }
    converted = convert_wan_vae_to_diffusers(common)
    for key, value in state.items():
        if not key.startswith(("encoder.downsamples.", "decoder.upsamples.")):
            continue
        direction = "down" if key.startswith("encoder") else "up"
        new_key = key.replace(f".{direction}samples.", f".{direction}_blocks.", 1)
        if "residual" in key or "shortcut" in key:
            new_key = new_key.replace(f".{direction}samples.", ".resnets.")
            for source, target in (
                ("residual.0", "norm1"),
                ("residual.2", "conv1"),
                ("residual.3", "norm2"),
                ("residual.6", "conv2"),
                ("shortcut", "conv_shortcut"),
            ):
                new_key = new_key.replace(f".{source}.", f".{target}.")
        else:
            parts = new_key.split(".")
            if parts[3] == f"{direction}samples":
                new_key = ".".join(parts[:3] + [f"{direction}sampler"] + parts[5:])
        if new_key in converted:
            raise ValueError(f"Duplicate VAE checkpoint key {new_key}")
        converted[new_key] = value
    return converted


class SolarWMVAENorm(WanRMS_norm):
    """Preserve the released VAE's low-precision normalization arithmetic."""

    def forward(self, x):
        normalized = torch.nn.functional.normalize(x, dim=1 if self.channel_first else -1) * self.scale * self.gamma
        return normalized + (0.0 if self.bias is None else self.bias)


def _restore_reference_vae_norms(module):
    for name, child in module.named_children():
        if isinstance(child, WanRMS_norm):
            setattr(
                module,
                name,
                SolarWMVAENorm(
                    child.gamma.shape[0],
                    channel_first=child.channel_first,
                    images=child.gamma.ndim == 3,
                    bias=isinstance(child.bias, torch.nn.Parameter),
                ),
            )
        else:
            _restore_reference_vae_norms(child)


def load_vae(base: Path, device: torch.device):
    with torch.device("meta"):
        vae = OmniAutoencoderKLWan(**VAE_CONFIG)
        _restore_reference_vae_norms(vae)
    state = torch.load(base / "vae/Wan2.2_VAE.pth", map_location="cpu", weights_only=True, mmap=True)
    vae.load_state_dict(convert_vae_weights(state), strict=True, assign=True)
    return vae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)


# Published Wan2.2 TI2V-5B VAE configuration and normalization statistics.
VAE_CONFIG = {
    "base_dim": 160,
    "z_dim": 48,
    "is_residual": True,
    "in_channels": 12,
    "out_channels": 12,
    "decoder_base_dim": 256,
    "scale_factor_temporal": 4,
    "scale_factor_spatial": 16,
    "patch_size": 2,
    "latents_mean": [
        -0.2289,
        -0.0052,
        -0.1323,
        -0.2339,
        -0.2799,
        0.0174,
        0.1838,
        0.1557,
        -0.1382,
        0.0542,
        0.2813,
        0.0891,
        0.157,
        -0.0098,
        0.0375,
        -0.1825,
        -0.2246,
        -0.1207,
        -0.0698,
        0.5109,
        0.2665,
        -0.2108,
        -0.2158,
        0.2502,
        -0.2055,
        -0.0322,
        0.1109,
        0.1567,
        -0.0729,
        0.0899,
        -0.2799,
        -0.123,
        -0.0313,
        -0.1649,
        0.0117,
        0.0723,
        -0.2839,
        -0.2083,
        -0.052,
        0.3748,
        0.0152,
        0.1957,
        0.1433,
        -0.2944,
        0.3573,
        -0.0548,
        -0.1681,
        -0.0667,
    ],
    "latents_std": [
        0.4765,
        1.0364,
        0.4514,
        1.1677,
        0.5313,
        0.499,
        0.4818,
        0.5013,
        0.8158,
        1.0344,
        0.5894,
        1.0901,
        0.6885,
        0.6165,
        0.8454,
        0.4978,
        0.5759,
        0.3523,
        0.7135,
        0.6804,
        0.5833,
        1.4146,
        0.8986,
        0.5659,
        0.7069,
        0.5338,
        0.4889,
        0.4917,
        0.4069,
        0.4999,
        0.6866,
        0.4093,
        0.5709,
        0.6065,
        0.6415,
        0.4944,
        0.5726,
        1.2042,
        0.5458,
        1.6887,
        0.3971,
        1.06,
        0.3943,
        0.5537,
        0.5444,
        0.4089,
        0.7468,
        0.7744,
    ],
}
