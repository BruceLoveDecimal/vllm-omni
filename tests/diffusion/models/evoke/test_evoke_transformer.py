# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Evoke transformer: config contract, strict weight loading, visibility filtering."""

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# evoke/stage3_post_distillation/transformer/config.json @ AlayaLab/Evoke 7fa34ec
RELEASED_CONFIG = {
    "_class_name": "EvokeTransformer3DModel",
    "_diffusers_version": "0.39.0.dev0",
    "added_kv_proj_dim": None,
    "attention_head_dim": 128,
    "cam_ctrl_layers": None,
    "cam_rank": 128,
    "cross_attn_norm": True,
    "enable_cam_control": False,
    "eps": 1e-06,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "gan_cond_map_dim": 768,
    "gan_hooks": [5, 15, 25, 35],
    "guidance_cross_attn": True,
    "has_multi_term_memory_patch": True,
    "history_scale_mode": "per_head",
    "image_dim": None,
    "in_channels": 16,
    "is_amplify_history": False,
    "is_train_restrict_lora": False,
    "is_use_gan": False,
    "is_use_gan_final": False,
    "is_use_gan_hooks": False,
    "num_attention_heads": 40,
    "num_layers": 40,
    "out_channels": 16,
    "patch_size": [1, 2, 2],
    "qk_norm": "rms_norm_across_heads",
    "restrict_lora": False,
    "restrict_lora_rank": 128,
    "restrict_self_attn": False,
    "rope_dim": [44, 42, 42],
    "rope_theta": 10000.0,
    "text_dim": 4096,
    "use_raw_sink_frames": True,
    "zero_history_timestep": True,
}

TINY = {
    **RELEASED_CONFIG,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "ffn_dim": 32,
    "freq_dim": 8,
    "num_layers": 2,
    "in_channels": 4,
    "out_channels": 4,
    "rope_dim": [4, 2, 2],
    "text_dim": 12,
}

# Released checkpoint key layout (1101 tensors = 21 top-level + 40 x 27 per block).
TOP_LEVEL_KEYS = [
    "condition_embedder.text_embedder.linear_1.bias",
    "condition_embedder.text_embedder.linear_1.weight",
    "condition_embedder.text_embedder.linear_2.bias",
    "condition_embedder.text_embedder.linear_2.weight",
    "condition_embedder.time_embedder.linear_1.bias",
    "condition_embedder.time_embedder.linear_1.weight",
    "condition_embedder.time_embedder.linear_2.bias",
    "condition_embedder.time_embedder.linear_2.weight",
    "condition_embedder.time_proj.bias",
    "condition_embedder.time_proj.weight",
    "norm_out.scale_shift_table",
    "patch_embedding.bias",
    "patch_embedding.weight",
    "patch_long.bias",
    "patch_long.weight",
    "patch_mid.bias",
    "patch_mid.weight",
    "patch_short.bias",
    "patch_short.weight",
    "proj_out.bias",
    "proj_out.weight",
]
BLOCK_KEYS = [
    f"{module}.{suffix}"
    for module, suffixes in {
        "attn1.norm_k": ["weight"],
        "attn1.norm_q": ["weight"],
        "attn1.to_k": ["bias", "weight"],
        "attn1.to_out.0": ["bias", "weight"],
        "attn1.to_q": ["bias", "weight"],
        "attn1.to_v": ["bias", "weight"],
        "attn2.norm_k": ["weight"],
        "attn2.norm_q": ["weight"],
        "attn2.to_k": ["bias", "weight"],
        "attn2.to_out.0": ["bias", "weight"],
        "attn2.to_q": ["bias", "weight"],
        "attn2.to_v": ["bias", "weight"],
        "ffn.net.0.proj": ["bias", "weight"],
        "ffn.net.2": ["bias", "weight"],
        "norm2": ["bias", "weight"],
    }.items()
    for suffix in suffixes
] + ["scale_shift_table"]


@pytest.fixture(autouse=True)
def _init_distributed(monkeypatch):
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda *_args, **_kwargs: default_unquantized_gemm,
    )
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29514")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


def _sdpa_config():
    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

    return SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
        parallel_config=SimpleNamespace(ring_degree=1, allgather_degree=1),
    )


def _build(config=TINY, dtype=torch.float32):
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.models.evoke.evoke_transformer import EvokeTransformer3DModel

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with set_current_diffusion_config(_sdpa_config()):
            model = EvokeTransformer3DModel(**config)
    finally:
        torch.set_default_dtype(previous)
    return model.eval()


@pytest.mark.parametrize("flag", ["restrict_self_attn", "enable_cam_control", "is_use_gan", "restrict_lora"])
def test_unsupported_features_are_rejected(flag):
    with pytest.raises(NotImplementedError):
        _build({**TINY, flag: True})


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="history_invisible_token"):
        _build({**TINY, "history_invisible_token": True})


def test_upstream_fp32_modules_survive_bf16_construction():
    model = _build(dtype=torch.bfloat16)
    dtypes = dict((name, p.dtype) for name, p in model.named_parameters())
    assert dtypes["blocks.0.scale_shift_table"] == torch.float32
    assert dtypes["norm_out.scale_shift_table"] == torch.float32
    assert dtypes["condition_embedder.time_embedder.linear_1.weight"] == torch.float32
    assert dtypes["blocks.0.norm2.weight"] == torch.float32
    assert dtypes["blocks.0.attn1.to_qkv.weight"] == torch.bfloat16


def _checkpoint(model):
    from vllm_omni.diffusion.models.evoke.evoke_transformer import _lookup_name

    params = dict(model.named_parameters())
    names = TOP_LEVEL_KEYS + [f"blocks.{i}.{key}" for i in range(2) for key in BLOCK_KEYS]
    weights = []
    for name in names:
        target = params[_lookup_name(name)]
        shape = target.shape
        if ".attn1.to_q" in name or ".attn1.to_k" in name or ".attn1.to_v" in name:
            shape = (shape[0] // 3, *shape[1:])
        weights.append((name, torch.randn(shape, dtype=torch.float32)))
    return weights


def test_released_key_layout_loads_completely():
    model = _build()
    loaded = model.load_weights(iter(_checkpoint(model)))
    assert {name for name, _ in model.named_parameters()} <= loaded


def test_stage1_camera_tensors_are_rejected_not_skipped():
    model = _build()
    extra = [("patch_embedding_wancamctrl.weight", torch.zeros(1)), ("c2ws_hidden_states_layer1.bias", torch.zeros(1))]
    with pytest.raises(ValueError, match="no matching parameter"):
        model.load_weights(iter(_checkpoint(model) + extra))


def test_filter_drops_invisible_tokens_and_keeps_rope_aligned():
    from vllm_omni.diffusion.models.evoke.evoke_transformer import filter_history_tokens

    tokens = torch.arange(12).float().view(1, 12, 1)
    rope = tokens.clone() * 10
    mask = torch.ones(1, 1, 3, 4, 4)
    mask[:, :, 1] = 0
    kept, kept_rope = filter_history_tokens(tokens, rope, mask, (1, 2, 2), 0.5)
    assert kept.flatten().tolist() == [0, 1, 2, 3, 8, 9, 10, 11]
    assert torch.equal(kept_rope, kept * 10)
    with pytest.raises(ValueError, match="covers"):
        filter_history_tokens(tokens[:, :6], rope[:, :6], mask, (1, 2, 2), 0.5)


def _inputs():
    torch.manual_seed(0)
    indices = torch.arange(0, 1 + 4 + 2 + 1 + 3)
    prefix, long_idx, mid_idx, one_x, hidden = indices.split([1, 4, 2, 1, 3])
    return {
        "hidden_states": torch.randn(1, 4, 3, 4, 4),
        "timestep": torch.tensor([500]),
        "encoder_hidden_states": torch.randn(1, 5, 12),
        "indices_hidden_states": hidden[None],
        "indices_latents_history_short": torch.cat([prefix, hidden, one_x])[None],
        "indices_latents_history_mid": mid_idx[None],
        "indices_latents_history_long": long_idx[None],
        "latents_history_short": torch.randn(1, 4, 5, 4, 4),
        "latents_history_mid": torch.randn(1, 4, 2, 4, 4),
        "latents_history_long": torch.randn(1, 4, 4, 4, 4),
        "return_dict": False,
    }


def _forward(model, attention_kwargs):
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.forward_context import set_forward_context

    config = _sdpa_config()
    with torch.no_grad(), set_current_diffusion_config(config), set_forward_context(omni_diffusion_config=config):
        return model(**_inputs(), attention_kwargs=attention_kwargs)[0]


def test_visibility_mask_changes_attention_only_where_tokens_are_dropped():
    model = _build()
    for _, param in model.named_parameters():
        torch.nn.init.normal_(param, std=0.05)
    baseline = _forward(model, None)
    visible = torch.ones(1, 1, 5, 4, 4)
    assert torch.equal(_forward(model, {"history_visible_mask_short": visible}), baseline)
    occluded = visible.clone()
    occluded[:, :, 1:4] = 0
    filtered = _forward(model, {"history_visible_mask_short": occluded, "history_visible_token_threshold": 0.5})
    assert filtered.shape == baseline.shape
    assert not torch.allclose(filtered, baseline)
