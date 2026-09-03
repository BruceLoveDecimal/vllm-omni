# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A synthetic VDN-H3 release, shaped exactly like the published one.

The real ``stage-dmd-step-250`` is 5.5 GB, so the tests build a tiny release
with the same file layout, the same key spellings and the same per-block tensor
inventory (16 branch tensors, the Stage-B LoRA over q/k/v/o, the turbo LoRA over
rather more with rank 16 on the AdaLN projections). Every count and spelling
here was read off the published safetensors headers.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.minimax_h3.vdn.linear_branch import GATE_BOTTLENECK, SHORT_CONV_KERNEL

HIDDEN = 8
HEADS = 2
HEAD_DIM = 4
CHANNELS = HEADS * HEAD_DIM
FFN = 6
TIME_EMBED_DIM = 5
ADALN_OUT = 18 * HIDDEN
FINAL_ADALN_OUT = 2 * HIDDEN
RANK = 2
ADALN_RANK = 1

#: The published transform config, verbatim.
TRANSFORM_CONFIG = {
    "anchor_frames": "both",
    "enable_softmax_gate": True,
    "linear_attention": {
        "a_fp32": True,
        "bridge": "alpha",
        "delta_rule": "vdn_solve",
        "enable_text_state": True,
        "linear_head_dim": HEAD_DIM,
        "short_conv": {"targets": ["k", "v"]},
    },
    "softmax_attention": {"chunk": 5, "radius": 1},
}


def _seeded(*shape: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=torch.float32)


def branch_tensors(num_blocks: int) -> dict[str, torch.Tensor]:
    """The 16 tensors per block the released ``linear_branch`` carries."""
    tensors: dict[str, torch.Tensor] = {}
    for block in range(num_blocks):
        prefix = f"transformer_blocks.{block}.attn."
        linear = f"{prefix}linear_attention."
        shapes = {
            f"{linear}alpha.A_log": (HEADS,),
            f"{linear}alpha.dt_bias": (CHANNELS,),
            f"{linear}alpha.down.weight": (GATE_BOTTLENECK, HIDDEN),
            f"{linear}alpha.up.weight": (CHANNELS, GATE_BOTTLENECK),
            f"{linear}beta_proj.weight": (HEADS, HIDDEN),
            f"{linear}norm.weight": (HEAD_DIM,),
            f"{linear}output_gate.down.weight": (GATE_BOTTLENECK, HIDDEN),
            f"{linear}output_gate.up.weight": (CHANNELS, GATE_BOTTLENECK),
            f"{linear}output_gate.up.bias": (CHANNELS,),
            f"{linear}short_conv.k_sp.weight": (CHANNELS, 1, SHORT_CONV_KERNEL, SHORT_CONV_KERNEL),
            f"{linear}short_conv.k_tm.weight": (CHANNELS, 1, SHORT_CONV_KERNEL),
            f"{linear}short_conv.v_sp.weight": (CHANNELS, 1, SHORT_CONV_KERNEL, SHORT_CONV_KERNEL),
            f"{linear}short_conv.v_tm.weight": (CHANNELS, 1, SHORT_CONV_KERNEL),
            f"{prefix}softmax_gate.up.weight": (HEADS, HIDDEN),
            f"{prefix}softmax_gate.up.bias": (HEADS,),
            f"{prefix}to_out_linear.weight": (HIDDEN, CHANNELS),
        }
        for index, (name, shape) in enumerate(sorted(shapes.items())):
            tensors[name] = _seeded(*shape, seed=1000 + block * 100 + index)
    return tensors


#: ``(adapter module suffix, in features, out features)`` for the Stage-B LoRA.
_DEFAULT_SUFFIXES = (
    ("attn.orig.to_q", HIDDEN, CHANNELS),
    ("attn.orig.to_k", HIDDEN, CHANNELS),
    ("attn.orig.to_v", HIDDEN, CHANNELS),
    ("attn.orig.to_out.0", CHANNELS, HIDDEN),
)
_REFINER_SUFFIXES = (
    ("attn.to_q", HIDDEN, CHANNELS),
    ("attn.to_k", HIDDEN, CHANNELS),
    ("attn.to_v", HIDDEN, CHANNELS),
    ("attn.to_out.0", CHANNELS, HIDDEN),
)
#: What the turbo adapter adds on top, per DiT block.
_TURBO_EXTRA = (
    ("adaln_proj.linear", TIME_EMBED_DIM, ADALN_OUT, ADALN_RANK),
    ("ff.net.0.proj", HIDDEN, 2 * FFN, RANK),
    ("ff.net.2", FFN, HIDDEN, RANK),
)
_TURBO_REFINER_EXTRA = (
    ("ff.net.0.proj", HIDDEN, 2 * FFN, RANK),
    ("ff.net.2", FFN, HIDDEN, RANK),
)


def _pair(module: str, adapter: str, in_features: int, out_features: int, rank: int, seed: int):
    return {
        f"{module}.lora_A.{adapter}.weight": _seeded(rank, in_features, seed=seed),
        f"{module}.lora_B.{adapter}.weight": _seeded(out_features, rank, seed=seed + 1),
    }


def adapter_tensors(
    adapter: str,
    *,
    num_blocks: int,
    num_refiner_blocks: int,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """One adapter's tensors and the module paths it edits."""
    tensors: dict[str, torch.Tensor] = {}
    modules: list[str] = []
    seed = 7000 if adapter == "default" else 9000
    for block in range(num_blocks):
        entries = list(_DEFAULT_SUFFIXES)
        if adapter == "turbo":
            entries += [(name, fan_in, fan_out) for name, fan_in, fan_out, _ in _TURBO_EXTRA]
        for suffix, fan_in, fan_out in entries:
            rank = next((r for name, _, _, r in _TURBO_EXTRA if name == suffix), RANK)
            module = f"transformer_blocks.{block}.{suffix}"
            tensors.update(_pair(module, adapter, fan_in, fan_out, rank, seed))
            modules.append(module)
            seed += 2
    for block in range(num_refiner_blocks):
        entries = list(_REFINER_SUFFIXES)
        if adapter == "turbo":
            entries += [(name, fan_in, fan_out) for name, fan_in, fan_out, _ in _TURBO_REFINER_EXTRA]
        for suffix, fan_in, fan_out in entries:
            module = f"token_refiner.refiner_blocks.{block}.{suffix}"
            tensors.update(_pair(module, adapter, fan_in, fan_out, RANK, seed))
            modules.append(module)
            seed += 2
    if adapter == "turbo":
        tensors.update(_pair("norm_out.linear", adapter, TIME_EMBED_DIM, FINAL_ADALN_OUT, ADALN_RANK, seed))
        modules.append("norm_out.linear")
    return tensors, modules


def write_release(
    root: Path,
    *,
    num_blocks: int = 1,
    num_refiner_blocks: int = 1,
    adapters: tuple[str, ...] = ("default", "turbo"),
    transform_config: dict | None = None,
    metadata: dict | None = None,
    spec_overrides: dict | None = None,
    drop_branch_key: str | None = None,
    drop_adapter_key: str | None = None,
    drop_adapter_module: str | None = None,
) -> Path:
    """Write a VDN release directory and return its path."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "linear_branch").mkdir(exist_ok=True)

    branch = branch_tensors(num_blocks)
    if drop_branch_key is not None:
        branch.pop(drop_branch_key)
    save_file(branch, str(root / "linear_branch" / "model.safetensors"))

    declared = []
    for adapter in adapters:
        tensors, modules = adapter_tensors(adapter, num_blocks=num_blocks, num_refiner_blocks=num_refiner_blocks)
        if drop_adapter_key is not None:
            tensors.pop(drop_adapter_key, None)
        if drop_adapter_module is not None:
            # Both factors, so the module leaves no trace at all -- which is
            # what a truncated re-export looks like.
            for side in ("A", "B"):
                tensors.pop(f"{drop_adapter_module}.lora_{side}.{adapter}.weight", None)
        directory = root / "adapters" / adapter
        directory.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(directory / "adapter_model.safetensors"))
        config: dict = {"rank": RANK, "alpha": RANK}
        if adapter == "turbo":
            patterns = {
                module: ADALN_RANK
                for module in modules
                if module.endswith("adaln_proj.linear") or module == "norm_out.linear"
            }
            config.update(
                name="turbo",
                exact_targets=True,
                targets=sorted(modules),
                rank_pattern=patterns,
                alpha_pattern=dict(patterns),
            )
        else:
            config.update(targets=["attn.orig.to_q", "attn.orig.to_k", "attn.orig.to_v", "attn.orig.to_out.0"])
        declared.append({"type": "lora", "version": 1, "config": config})

    model_spec = {
        "format_version": 2,
        "base": {
            "library": "diffusers",
            "class_name": "MiniMaxH3Transformer3DModel",
            "source": "ckpts/h3-base",
            "subfolder": "transformer",
            "revision": "939557dc319dd91227e30195a763f272ba7f8765",
            "resolved_config": {},
            "config_hash": "0" * 64,
        },
        "transforms": [
            {
                "type": "hybrid_attention",
                "version": 2,
                "config": transform_config if transform_config is not None else TRANSFORM_CONFIG,
            }
        ],
        "adapters": declared,
    }
    model_spec.update(spec_overrides or {})
    (root / "model_spec.json").write_text(json.dumps(model_spec, indent=2), encoding="utf-8")
    (root / "metadata.json").write_text(
        json.dumps(
            metadata
            if metadata is not None
            else {"kind": "weights", "checkpoint_format_version": 2, "weights_dtype": "bfloat16", "metadata": {}}
        ),
        encoding="utf-8",
    )
    return root


__all__ = [
    "ADALN_RANK",
    "CHANNELS",
    "FFN",
    "HEADS",
    "HEAD_DIM",
    "HIDDEN",
    "RANK",
    "TIME_EMBED_DIM",
    "TRANSFORM_CONFIG",
    "adapter_tensors",
    "branch_tensors",
    "write_release",
]
