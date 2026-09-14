# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint adaptation for SolarWM-H3.

SolarWM ships MiniMax-H3 in the Diffusers layout (``transformer_blocks.0.attn.to_q``)
plus a rank-384 PEFT LoRA over every attention and feed-forward projection of
the 50 DiT blocks and the 2 token-refiner blocks. vLLM-Omni's H3 DiT loads the
native layout (``blocks.0.attn.qkv_proj`` with grouped Q/K/V and a gate-first
fused MLP), so the checkpoint stream is rewritten on the way in:

1. ``W += lora_B @ lora_A`` on the Diffusers-named tensor (alpha == rank, scale 1);
2. rename to the native parameter and re-pack Q/K/V or swap the MLP halves.

The rotary inverse frequencies are a non-persistent buffer in Diffusers and a
persistent one natively, so they are synthesized at the end of the stream.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

SOLARWM_H3_LORA_RANK = 384
SOLARWM_H3_LORA_TARGET_COUNT = 312
SOLARWM_H3_EMA_FILE = "ema.pt"
SOLARWM_H3_EMA_SCHEMA = "solarwm.minimax-h3-ema.v1"

_LORA_A_SUFFIX = ".lora_A.weight"
_LORA_B_SUFFIX = ".lora_B.weight"
_FSDP_WRAPPER_PARTS = frozenset({"_fsdp_wrapped_module", "_checkpoint_wrapped_module"})

_MODEL_LEVEL_NAMES = {
    "proj_in": "video_patch_proj",
    "audio_proj_in": "audio_patch_proj",
    "context_embedder": "condition_proj",
    "time_embedder.linear_1": "time_embedder.proj_in",
    "time_embedder.linear_2": "time_embedder.proj_out",
    "norm_out.norm": "final_layer.norm",
    "norm_out.linear": "final_layer.adaln_proj.linear",
    "proj_out": "final_layer.video_out",
    "audio_proj_out": "final_layer.audio_out",
    "token_refiner.final_norm": "token_refiner.final_norm",
}
_BLOCK_PREFIXES = (
    ("transformer_blocks.", "blocks."),
    ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
)
_BLOCK_MEMBER_NAMES = {
    "attn.to_out.0": "attn.out_proj",
    "attn.norm_q": "attn.q_norm",
    "attn.norm_k": "attn.k_norm",
    "ff.net.0.proj": "mlp.fc1",
    "ff.net.2": "mlp.fc2",
    "norm1": "norm1",
    "norm2": "norm2",
    "adaln_proj.linear": "adaln_proj.linear",
}
_QKV_MEMBERS = ("attn.to_q", "attn.to_k", "attn.to_v")


def canonical_lora_key(key: str) -> str:
    """Strip the PEFT wrapper prefix, FSDP wrappers and the adapter name from a SolarWM LoRA key."""
    parts = [part for part in key.removeprefix("base_model.model.").split(".") if part not in _FSDP_WRAPPER_PARTS]
    return (
        ".".join(parts)
        .replace(".lora_A.default.weight", _LORA_A_SUFFIX)
        .replace(".lora_B.default.weight", _LORA_B_SUFFIX)
    )


def load_solarwm_lora(adapter_dir: str | Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Read the EMA LoRA of a SolarWM stage package as ``{diffusers module: (A, B)}``."""
    path = Path(adapter_dir) / SOLARWM_H3_EMA_FILE
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if payload.get("schema") != SOLARWM_H3_EMA_SCHEMA:
        raise ValueError(f"{path} is not a SolarWM H3 EMA package (schema={payload.get('schema')!r})")
    factors: dict[str, dict[str, torch.Tensor]] = {}
    for raw_key, tensor in payload["shadow"].items():
        key = canonical_lora_key(raw_key)
        if key.endswith(_LORA_A_SUFFIX):
            factors.setdefault(key[: -len(_LORA_A_SUFFIX)], {})["A"] = tensor
        elif key.endswith(_LORA_B_SUFFIX):
            factors.setdefault(key[: -len(_LORA_B_SUFFIX)], {})["B"] = tensor
        else:
            raise ValueError(f"unexpected SolarWM LoRA tensor {raw_key!r}")
    if len(factors) != SOLARWM_H3_LORA_TARGET_COUNT:
        raise ValueError(f"SolarWM H3 LoRA must cover {SOLARWM_H3_LORA_TARGET_COUNT} modules, got {len(factors)}")
    lora: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for module, pair in factors.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"SolarWM H3 LoRA module {module!r} is missing its A or B factor")
        if pair["A"].shape[0] != SOLARWM_H3_LORA_RANK or pair["B"].shape[1] != SOLARWM_H3_LORA_RANK:
            raise ValueError(f"SolarWM H3 LoRA module {module!r} is not rank {SOLARWM_H3_LORA_RANK}")
        lora[module] = (pair["A"], pair["B"])
    logger.info("SolarWM-H3 LoRA %s: %d modules, rank %d", path, len(lora), SOLARWM_H3_LORA_RANK)
    return lora


def native_parameter_name(diffusers_name: str) -> str:
    """Map a Diffusers H3 parameter name to the native one; Q/K/V map to the fused ``qkv_proj``."""
    module, _, leaf = diffusers_name.rpartition(".")
    if module in _MODEL_LEVEL_NAMES:
        return f"{_MODEL_LEVEL_NAMES[module]}.{leaf}"
    for diffusers_prefix, native_prefix in _BLOCK_PREFIXES:
        if not module.startswith(diffusers_prefix):
            continue
        block_index, _, member = module[len(diffusers_prefix) :].partition(".")
        if member in _QKV_MEMBERS:
            return f"{native_prefix}{block_index}.attn.qkv_proj.{leaf}"
        return f"{native_prefix}{block_index}.{_BLOCK_MEMBER_NAMES[member]}.{leaf}"
    raise KeyError(f"unknown Diffusers MiniMax-H3 parameter {diffusers_name!r}")


def group_qkv_per_head(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, head_dim: int) -> torch.Tensor:
    """Interleave separate Q/K/V projections into the native ``[q_h, k_h, v_h]`` per-head layout."""
    heads = q.shape[0] // head_dim
    stacked = torch.stack(
        (
            q.reshape(heads, head_dim, -1),
            k.reshape(heads, head_dim, -1),
            v.reshape(heads, head_dim, -1),
        ),
        dim=1,
    )
    return stacked.reshape(heads * 3 * head_dim, -1)


def swap_fused_halves(weight: torch.Tensor) -> torch.Tensor:
    """Turn Diffusers' value-first SwiGLU projection into the native gate-first layout."""
    value, gate = weight.chunk(2, dim=0)
    return torch.cat((gate, value), dim=0)


def rope_inverse_frequencies(rope_freq_dim: int, rope_theta: float) -> torch.Tensor:
    exponent = torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim)
    return 1.0 / (rope_theta**exponent)


class SolarWMWeightAdapter:
    """Merge the SolarWM LoRA and rename a Diffusers H3 stream into the native layout."""

    def __init__(
        self,
        lora: dict[str, tuple[torch.Tensor, torch.Tensor]],
        *,
        head_dim: int,
        rope_inv_freq: torch.Tensor,
    ) -> None:
        self._lora = lora
        self._head_dim = head_dim
        self._rope_inv_freq = rope_inv_freq
        self._merged: set[str] = set()
        self._pending_qkv: dict[str, dict[str, torch.Tensor]] = {}
        # The rank-384 products run on the accelerator; the loader copies the
        # result into parameters that live there anyway.
        self._device = current_omni_platform.get_torch_device()

    def _merge_lora(self, diffusers_name: str, weight: torch.Tensor) -> torch.Tensor:
        module, _, leaf = diffusers_name.rpartition(".")
        if leaf != "weight" or module not in self._lora:
            return weight
        lora_a, lora_b = self._lora[module]
        delta = lora_b.to(self._device).float() @ lora_a.to(self._device).float()
        merged = delta.add_(weight.to(self._device).float()).to(weight.dtype)
        self._merged.add(module)
        return merged

    def apply(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        for diffusers_name, weight in weights:
            weight = self._merge_lora(diffusers_name, weight)
            native_name = native_parameter_name(diffusers_name)
            if not native_name.endswith(".attn.qkv_proj.weight"):
                if native_name.endswith(".mlp.fc1.weight"):
                    weight = swap_fused_halves(weight)
                yield native_name, weight
                continue
            module = diffusers_name.rpartition(".")[0]
            slot = module.rsplit(".", 1)[1]  # to_q / to_k / to_v
            pending = self._pending_qkv.setdefault(native_name, {})
            pending[slot] = weight.to(self._device)
            if len(pending) == 3:
                del self._pending_qkv[native_name]
                yield (
                    native_name,
                    group_qkv_per_head(pending["to_q"], pending["to_k"], pending["to_v"], head_dim=self._head_dim),
                )
        yield "rope.inv_freq", self._rope_inv_freq

    def validate_fully_applied(self) -> None:
        if self._pending_qkv:
            raise ValueError(f"incomplete Q/K/V projections in the checkpoint: {sorted(self._pending_qkv)[:5]}")
        missing = sorted(set(self._lora) - self._merged)
        if missing:
            raise ValueError(f"SolarWM LoRA modules never met their base weight: {missing[:5]}")
        self._lora = {}


__all__ = [
    "SOLARWM_H3_EMA_FILE",
    "SOLARWM_H3_LORA_RANK",
    "SOLARWM_H3_LORA_TARGET_COUNT",
    "SolarWMWeightAdapter",
    "canonical_lora_key",
    "group_qkv_per_head",
    "load_solarwm_lora",
    "native_parameter_name",
    "rope_inverse_frequencies",
    "swap_fused_halves",
]
