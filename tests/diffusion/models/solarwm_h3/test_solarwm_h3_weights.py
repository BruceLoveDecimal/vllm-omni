# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.solarwm_h3.weights import (
    SolarWMWeightAdapter,
    canonical_lora_key,
    group_qkv_per_head,
    native_parameter_name,
    rope_inverse_frequencies,
    swap_fused_halves,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_canonical_lora_key_strips_wrappers_and_adapter_name():
    key = "_fsdp_wrapped_module.transformer_blocks.3._checkpoint_wrapped_module.attn.to_q.lora_A.default.weight"
    assert canonical_lora_key(key) == "transformer_blocks.3.attn.to_q.lora_A.weight"
    # The released EMA packages keep PEFT's wrapper prefix.
    assert canonical_lora_key("base_model.model.token_refiner.refiner_blocks.0.attn.to_k.lora_B.weight") == (
        "token_refiner.refiner_blocks.0.attn.to_k.lora_B.weight"
    )


@pytest.mark.parametrize(
    ("diffusers_name", "native_name"),
    [
        ("proj_in.weight", "video_patch_proj.weight"),
        ("time_embedder.linear_2.bias", "time_embedder.proj_out.bias"),
        ("norm_out.linear.weight", "final_layer.adaln_proj.linear.weight"),
        ("transformer_blocks.12.attn.to_k.weight", "blocks.12.attn.qkv_proj.weight"),
        ("transformer_blocks.12.attn.norm_q.weight", "blocks.12.attn.q_norm.weight"),
        ("transformer_blocks.0.ff.net.0.proj.weight", "blocks.0.mlp.fc1.weight"),
        ("token_refiner.refiner_blocks.1.attn.to_out.0.weight", "token_refiner.blocks.1.attn.out_proj.weight"),
        ("token_refiner.final_norm.weight", "token_refiner.final_norm.weight"),
    ],
)
def test_native_parameter_name(diffusers_name, native_name):
    assert native_parameter_name(diffusers_name) == native_name


def test_group_qkv_per_head_interleaves_heads():
    head_dim, heads, hidden = 2, 3, 4
    q = torch.arange(heads * head_dim * hidden, dtype=torch.float32).reshape(heads * head_dim, hidden)
    k = q + 100
    v = q + 200
    grouped = group_qkv_per_head(q, k, v, head_dim=head_dim).reshape(heads, 3, head_dim, hidden)
    torch.testing.assert_close(grouped[:, 0], q.reshape(heads, head_dim, hidden))
    torch.testing.assert_close(grouped[:, 1], k.reshape(heads, head_dim, hidden))
    torch.testing.assert_close(grouped[:, 2], v.reshape(heads, head_dim, hidden))


def test_swap_fused_halves():
    weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    torch.testing.assert_close(swap_fused_halves(weight), torch.cat((weight[2:], weight[:2])))


def test_rope_inverse_frequencies_match_the_diffusers_buffer():
    inv_freq = rope_inverse_frequencies(16, 10000.0)
    expected = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
    torch.testing.assert_close(inv_freq, expected)


def test_weight_adapter_merges_lora_then_repacks_the_native_layout():
    head_dim, hidden, rank = 2, 4, 1
    inner = 2 * head_dim
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(inner, hidden, generator=generator)
    k = torch.randn(inner, hidden, generator=generator)
    v = torch.randn(inner, hidden, generator=generator)
    fc1 = torch.randn(6, hidden, generator=generator)
    lora_a = torch.randn(rank, hidden, generator=generator)
    lora_b = torch.randn(inner, rank, generator=generator)
    fc1_b = torch.randn(6, rank, generator=generator)
    lora = {
        "transformer_blocks.0.attn.to_q": (lora_a, lora_b),
        "transformer_blocks.0.ff.net.0.proj": (lora_a, fc1_b),
    }
    adapter = SolarWMWeightAdapter(lora, head_dim=head_dim, rope_inv_freq=torch.ones(3))
    stream = [
        ("transformer_blocks.0.attn.to_v.weight", v),
        ("transformer_blocks.0.ff.net.0.proj.weight", fc1),
        ("transformer_blocks.0.attn.to_q.weight", q),
        ("transformer_blocks.0.attn.to_k.weight", k),
        ("proj_in.bias", torch.zeros(hidden)),
    ]
    converted = {name: tensor.cpu() for name, tensor in adapter.apply(stream)}
    adapter.validate_fully_applied()

    assert set(converted) == {
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.mlp.fc1.weight",
        "video_patch_proj.bias",
        "rope.inv_freq",
    }
    merged_q = q + lora_b @ lora_a
    torch.testing.assert_close(
        converted["blocks.0.attn.qkv_proj.weight"],
        group_qkv_per_head(merged_q, k, v, head_dim=head_dim),
    )
    torch.testing.assert_close(converted["blocks.0.mlp.fc1.weight"], swap_fused_halves(fc1 + fc1_b @ lora_a))


def test_weight_adapter_reports_unmerged_lora_modules():
    lora = {"transformer_blocks.0.attn.to_k": (torch.zeros(1, 4), torch.zeros(4, 1))}
    adapter = SolarWMWeightAdapter(lora, head_dim=2, rope_inv_freq=torch.ones(3))
    list(adapter.apply([("proj_in.weight", torch.zeros(2, 2))]))
    with pytest.raises(ValueError, match="never met"):
        adapter.validate_fully_applied()
