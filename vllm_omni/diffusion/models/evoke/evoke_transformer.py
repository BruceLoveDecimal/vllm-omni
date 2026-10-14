# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Alaya-EVOKE (https://github.com/AlayaLab/Evoke @ 1d271be),
# evoke/modules/transformer_evoke.py, Apache-2.0.
"""Evoke DiT: the Helios backbone plus visibility-filtered history tokens.

The released Evoke checkpoints share Helios' 40-layer DiT, parameter names and
multi-term memory patches. What Evoke adds on the inference path that the
shipped real-time checkpoint (``stage3_post_distillation``) exercises is a
per-token *visibility filter* on the history tiers: the short tier carries the
camera-warped frames, and warp tokens that fall in disoccluded regions are
removed before attention instead of being fed as black pixels.

Features of the upstream module that no released inference recipe turns on
(restricted self-attention + history KV cache, camera AdaLN, additive Plucker,
GAN heads, restrict-LoRA, synchronized pyramid warp, raw sink frames) are
rejected at construction so a checkpoint that needs them fails loudly instead
of silently running a different model.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.helios.helios_transformer import HeliosTransformer3DModel, pad_for_3d_conv

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

# Patch kernel of each history tier; the visibility mask is pooled with the same
# kernel so one mask value describes exactly one token.
HISTORY_TIER_PATCH = {
    "short": (1, 2, 2),
    "mid": (2, 4, 4),
    "long": (4, 8, 8),
}

# Upstream ``_keep_in_fp32_modules``. diffusers keeps these in fp32 when the
# rest of the model is loaded in bf16; the omni loader builds the whole model in
# the configured dtype, so the transformer restores them itself.
EVOKE_FP32_MODULES = ("time_embedder", "scale_shift_table", "norm1", "norm2", "norm3")

# Config keys that only change behaviour when set to a non-default value the
# shipped real-time recipe never uses.
_UNSUPPORTED_FLAGS = {
    "restrict_self_attn": "restricted self-attention (Stage-1 recipe)",
    "restrict_lora": "restricted-attention LoRA",
    "is_train_restrict_lora": "restricted-attention LoRA",
    "enable_cam_control": "camera AdaLN control",
    "geo_warp_plucker_enabled": "additive Plucker camera embedding",
    "is_use_gan": "GAN discriminator heads",
}

# Config keys that are accepted and ignored: either metadata, or features whose
# parameters are absent from the released checkpoints when the flag is off.
_IGNORED_KEYS = frozenset(
    {
        "_class_name",
        "_diffusers_version",
        "_name_or_path",
        "image_dim",
        "cam_rank",
        "cam_ctrl_layers",
        "restrict_lora_rank",
        "is_use_gan_hooks",
        "is_use_gan_final",
        "gan_cond_map_dim",
        "gan_hooks",
        # The raw-sink layout is a launcher choice in the shipped recipe: every
        # upstream launcher passes --no_raw_sink_frames, overriding config.json.
        "use_raw_sink_frames",
    }
)

_HELIOS_KEYS = (
    "patch_size",
    "num_attention_heads",
    "attention_head_dim",
    "in_channels",
    "out_channels",
    "text_dim",
    "freq_dim",
    "ffn_dim",
    "num_layers",
    "cross_attn_norm",
    "qk_norm",
    "eps",
    "added_kv_proj_dim",
    "rope_dim",
    "rope_theta",
    "guidance_cross_attn",
    "zero_history_timestep",
    "has_multi_term_memory_patch",
    "is_amplify_history",
    "history_scale_mode",
)


def should_keep_in_fp32(name: str) -> bool:
    return any(module_name in name.split(".") for module_name in EVOKE_FP32_MODULES)


def pool_history_visible_mask(mask: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Average a latent-resolution visibility mask over each token's patch footprint."""
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    if mask.ndim != 5:
        raise ValueError(f"history visible mask must be 4D/5D, got {tuple(mask.shape)}")
    mask = pad_for_3d_conv(mask.float(), patch_size)
    return F.avg_pool3d(mask, kernel_size=patch_size, stride=patch_size)


def resolve_history_keep_mask(pooled: torch.Tensor, threshold: float) -> torch.Tensor:
    """Threshold a pooled ``[B, 1, T, H, W]`` mask to one shared boolean token mask."""
    if pooled.ndim == 5:
        if pooled.shape[1] != 1:
            raise ValueError(f"history visible mask channel dimension must be 1, got {tuple(pooled.shape)}")
        pooled = pooled[:, 0]
    keep = pooled.flatten(1) >= float(threshold)
    if keep.shape[0] > 1 and not torch.equal(keep, keep[0:1].expand_as(keep)):
        raise ValueError("history visibility masking requires the same mask for every sample in the batch.")
    return keep[0]


def filter_history_tokens(
    tokens: torch.Tensor,
    rotary_emb: torch.Tensor,
    visible_mask: torch.Tensor,
    patch_size: tuple[int, int, int],
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop the tokens whose patch is (mostly) invisible, keeping RoPE aligned."""
    keep = resolve_history_keep_mask(pool_history_visible_mask(visible_mask, patch_size), threshold)
    if keep.numel() != tokens.shape[1]:
        raise ValueError(
            f"history visibility mask covers {keep.numel()} tokens but the tier has {tokens.shape[1]} tokens."
        )
    keep = keep.to(tokens.device)
    if bool(keep.all()):
        return tokens, rotary_emb
    return tokens[:, keep], rotary_emb[:, keep]


class EvokeTransformer3DModel(HeliosTransformer3DModel):
    """Helios DiT with visibility-filtered history tiers."""

    def __init__(self, quant_config: QuantizationConfig | None = None, **config: Any) -> None:
        for key, feature in _UNSUPPORTED_FLAGS.items():
            if config.get(key):
                raise NotImplementedError(f"Evoke transformer config enables {feature} ({key}=true), not supported.")
        unknown = sorted(set(config) - set(_HELIOS_KEYS) - set(_UNSUPPORTED_FLAGS) - _IGNORED_KEYS)
        if unknown:
            raise ValueError(f"Unrecognized Evoke transformer config keys: {unknown}")
        helios_kwargs = {key: config[key] for key in _HELIOS_KEYS if key in config}
        for key in ("patch_size", "rope_dim"):
            if isinstance(helios_kwargs.get(key), list):
                helios_kwargs[key] = tuple(helios_kwargs[key])
        super().__init__(quant_config=quant_config, **helios_kwargs)
        for name, param in self.named_parameters():
            if should_keep_in_fp32(name) and param.dtype != torch.float32:
                param.data = param.data.float()

    def _select_history_tokens(
        self,
        tier: str,
        tokens: torch.Tensor,
        rotary_emb: torch.Tensor,
        attention_kwargs: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = (attention_kwargs or {}).get(f"history_visible_mask_{tier}")
        if mask is None:
            return tokens, rotary_emb
        threshold = float(attention_kwargs.get("history_visible_token_threshold", 0.5))
        return filter_history_tokens(tokens, rotary_emb, mask, HISTORY_TIER_PATCH[tier], threshold)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load strictly: an unknown checkpoint tensor means a different architecture."""
        expected = {name for name, _ in self.named_parameters()}
        unexpected: list[str] = []

        def _checked(items: Iterable[tuple[str, torch.Tensor]]):
            for name, tensor in items:
                if _lookup_name(name) not in expected:
                    unexpected.append(name)
                    continue
                yield name, tensor

        loaded = super().load_weights(_checked(weights))
        if unexpected:
            raise ValueError(
                f"Evoke checkpoint carries {len(unexpected)} tensors with no matching parameter "
                f"(e.g. {sorted(unexpected)[:6]}); this checkpoint needs features the Evoke port does not implement."
            )
        return loaded


def _lookup_name(name: str) -> str:
    """Mirror ``HeliosTransformer3DModel.load_weights`` checkpoint -> parameter renames."""
    for weight_name in (".attn1.to_q", ".attn1.to_k", ".attn1.to_v"):
        if weight_name in name:
            return name.replace(weight_name, ".attn1.to_qkv")
    name = name.replace(".ffn.net.0.", ".ffn.net_0.").replace(".ffn.net.2.", ".ffn.net_2.")
    return name.replace(".to_out.0.", ".to_out.")
