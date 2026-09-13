# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared native Wan UMT5 checkpoint conversion."""

import torch
from transformers import UMT5Config, UMT5EncoderModel
from vllm.logger import init_logger

logger = init_logger(__name__)

# UMT5Config matching the Wan2.2 T5 encoder architecture (umt5-xxl variant).
_WAN_UMT5_CONFIG = UMT5Config(
    vocab_size=256384,
    d_model=4096,
    d_kv=64,
    d_ff=10240,
    num_heads=64,
    num_layers=24,
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
    dense_act_fn="gelu_new",
    is_gated_act=True,
    is_encoder_decoder=False,
)


def _convert_wan_t5_state_dict(wan_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a Wan2.2 T5 encoder state dict to HuggingFace UMT5 format.

    The Wan checkpoint uses flat ``blocks.{i}.attn.*`` naming whereas
    HuggingFace ``UMT5EncoderModel`` expects ``encoder.block.{i}.layer.*``
    prefixed keys.  All tensor shapes are identical; only key names change.
    """
    hf_sd: dict[str, torch.Tensor] = {}

    # Embeddings (Wan has 1 copy; UMT5 ties shared + encoder.embed_tokens)
    hf_sd["shared.weight"] = wan_sd["token_embedding.weight"]
    hf_sd["encoder.embed_tokens.weight"] = wan_sd["token_embedding.weight"]

    # Final layer norm
    hf_sd["encoder.final_layer_norm.weight"] = wan_sd["norm.weight"]

    # Per-block weights
    num_layers = _WAN_UMT5_CONFIG.num_layers
    for i in range(num_layers):
        src = f"blocks.{i}"
        dst = f"encoder.block.{i}"

        # Self-attention
        for proj in ("q", "k", "v", "o"):
            hf_sd[f"{dst}.layer.0.SelfAttention.{proj}.weight"] = wan_sd[f"{src}.attn.{proj}.weight"]
        hf_sd[f"{dst}.layer.0.SelfAttention.relative_attention_bias.weight"] = wan_sd[
            f"{src}.pos_embedding.embedding.weight"
        ]
        hf_sd[f"{dst}.layer.0.layer_norm.weight"] = wan_sd[f"{src}.norm1.weight"]

        # Gated feed-forward (gate.0 → wi_0, fc1 → wi_1, fc2 → wo)
        hf_sd[f"{dst}.layer.1.DenseReluDense.wi_0.weight"] = wan_sd[f"{src}.ffn.gate.0.weight"]
        hf_sd[f"{dst}.layer.1.DenseReluDense.wi_1.weight"] = wan_sd[f"{src}.ffn.fc1.weight"]
        hf_sd[f"{dst}.layer.1.DenseReluDense.wo.weight"] = wan_sd[f"{src}.ffn.fc2.weight"]
        hf_sd[f"{dst}.layer.1.layer_norm.weight"] = wan_sd[f"{src}.norm2.weight"]

    return hf_sd


def _load_wan_t5_as_umt5(
    model: UMT5EncoderModel,
    checkpoint_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> UMT5EncoderModel:
    """Load a Wan2.2 T5 ``.pth`` checkpoint as a ``UMT5EncoderModel``."""
    logger.info("Loading Wan T5 checkpoint: %s", checkpoint_path)
    wan_sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    hf_sd = _convert_wan_t5_state_dict(wan_sd)

    model.load_state_dict(hf_sd, assign=True)
    model = model.to(dtype=dtype).eval()

    return model
