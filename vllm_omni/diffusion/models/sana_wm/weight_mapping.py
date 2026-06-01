# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-name helpers for the SANA-WM Stage-1 checkpoint."""

from __future__ import annotations

SANA_WM_STAGE1_PREFIX_MAP: tuple[tuple[str, str], ...] = (
    ("y_embedder.", "transformer.y_embedder."),
    ("t_embedder.", "transformer.t_embedder."),
    ("t_block.", "transformer.t_block."),
    ("x_embedder.", "transformer.x_embedder."),
    ("final_layer.", "transformer.final_layer."),
    ("plucker_embedder.", "transformer.plucker_embedder."),
    ("raymap_embedder.", "transformer.raymap_embedder."),
    ("attention_y_norm.", "transformer.attention_y_norm."),
)


def normalize_sana_wm_stage1_weight_name(name: str) -> str | None:
    """Map released Stage-1 checkpoint keys to planned vLLM-Omni names.

    Per-block mappings are intentionally conservative because the exact module
    tree must be finalized while porting the transformer implementation.
    """

    if name == "pos_embed":
        return "transformer.pos_embed"
    if name.startswith("blocks."):
        parts = name.split(".", 3)
        if len(parts) == 3 and parts[2] == "scale_shift_table":
            return f"transformer.blocks.{parts[1]}.scale_shift_table"
        if len(parts) < 4:
            return None
        _, block_idx, block_module, suffix = parts
        block_prefix = f"transformer.blocks.{block_idx}."
        if block_module == "attn":
            return f"{block_prefix}attn.{suffix}"
        if block_module == "cross_attn":
            return f"{block_prefix}cross_attn.{suffix}"
        if block_module == "mlp":
            return f"{block_prefix}mlp.{suffix}"
        if block_module == "plucker_post":
            return f"{block_prefix}plucker_proj.{suffix}"
        return f"{block_prefix}{block_module}.{suffix}"

    for source_prefix, target_prefix in SANA_WM_STAGE1_PREFIX_MAP:
        if name.startswith(source_prefix):
            return f"{target_prefix}{name.removeprefix(source_prefix)}"
    return None
