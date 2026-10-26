# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU math contracts; shared backend integration needs a separate GPU smoke."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.layers.linear import LinearBase

from vllm_omni.diffusion.models.hy_nomas import transformer as model_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _SDPA(nn.Module):
    def forward(self, query, key, value, attn_metadata=None):
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=None if attn_metadata is None else attn_metadata.attn_mask,
        ).transpose(1, 2)


@pytest.fixture
def tiny_model(monkeypatch):
    # Select a real CPU math kernel without constructing platform/distributed state.
    monkeypatch.setattr(model_module, "_attention", lambda *args: _SDPA())
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: 1)
    torch.manual_seed(13)
    model = model_module.HyNomasTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=6,
        text_embed_dim=8,
        mlp_ratio=2,
        rope_axes_dim=(2, 2, 2),
        camera_embed_dim=8,
        camera_attention_dim=12,
        camera_num_attention_heads=2,
        camera_rope_axes_dim=(2, 2, 2),
    ).eval()
    for module in model.modules():
        if isinstance(module, LinearBase):
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
    return model


def test_action_groups_match_dense_visibility_oracle():
    torch.manual_seed(7)
    history, video, text = 3, 4, 5
    query = torch.randn(1, video + text, 2, 6)
    key = torch.randn(1, history + video + text, 2, 6)
    value = torch.randn_like(key)
    key_mask = torch.ones(1, key.shape[1], dtype=torch.bool)
    key_mask[:, 1] = False
    key_mask[:, -1] = False
    text_mask = key_mask[:, -text:]
    output = model_module.HyNomasAttnProcessor()._compute_attention(
        _SDPA(),
        query,
        key,
        value,
        key_mask,
        text_mask,
        history,
        video,
        text,
        text_mask,
        [[2, 3, 4]],
        [(0, 2, 0), (2, 4, 1)],
    )
    # History + all current video + head text; each action sees only its own text.
    dense_mask = torch.zeros(1, 1, video + text, history + video + text, dtype=torch.bool)
    dense_mask[..., : history + video + 2] = True
    dense_mask[..., 0:2, history + video + 2] = True
    dense_mask[..., 2:4, history + video + 3] = True
    dense_mask[..., video + 2, history + video + 2] = True
    dense_mask[..., video + 3, history + video + 3] = True
    dense_mask &= key_mask[:, None, None, :]
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=dense_mask
    ).transpose(1, 2)
    expected[:, -1] = 0  # Padding query is excluded by the grouped contract.
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)


@torch.no_grad()
def test_cache_commit_is_explicit_and_masks_action_history(tiny_model):
    latents = torch.randn(1, 4, 2, 2, 2)
    text = torch.randn(1, 5, 8)
    mask = torch.tensor([[True, True, True, True, False]])
    caches: list[dict[str, torch.Tensor]] = [{} for _ in range(2)]
    args = dict(
        hidden_states=latents,
        encoder_hidden_states=text,
        timestep=torch.tensor([500.0]),
        encoder_attention_mask=mask,
        kv_cache=caches,
        cache_action_start=2,
    )
    baseline = tiny_model(**args).sample
    assert caches == [{}, {}]
    committed = tiny_model(**args, update_cache=True).sample
    torch.testing.assert_close(committed, baseline)
    for cache in caches:
        assert cache["key"].shape == (1, 13, 2, 6)
        assert cache["mask"].tolist() == [[True] * 10 + [False] * 3]
    before = [cache["key"].clone() for cache in caches]
    continued = tiny_model(**args, frame_offset=2).sample
    assert torch.isfinite(continued).all()
    assert not torch.allclose(continued, baseline)
    for saved, cache in zip(before, caches):
        torch.testing.assert_close(saved, cache["key"])


@torch.no_grad()
def test_camera_gate_and_front_padding(tiny_model):
    latents = torch.randn(1, 4, 2, 2, 2)
    text = torch.randn(1, 4, 8)
    args = dict(hidden_states=latents, encoder_hidden_states=text, timestep=torch.tensor([200.0]))
    poses = torch.eye(4).repeat(1, 5, 1, 1)
    baseline = tiny_model(**args).sample
    gated = tiny_model(**args, camera_extrinsics=poses, camera_attention_mask=torch.zeros(1)).sample
    torch.testing.assert_close(gated, baseline)
    padded = torch.cat((poses[:, :1].repeat(1, 3, 1, 1), poses), dim=1)
    result = tiny_model(**args, camera_extrinsics=poses).sample
    torch.testing.assert_close(result, tiny_model(**args, camera_extrinsics=padded).sample)
    with pytest.raises(ValueError, match="align"):
        tiny_model(**args, camera_extrinsics=poses[:, :4])


def _checkpoint_names(name):
    """Map a parameter to its converted-checkpoint names, splitting fused q/k/v."""
    name = name.replace("_mlp.net.0.proj.", "_mlp.fc1.").replace("_mlp.net.2.", "_mlp.fc2.")
    fused = model_module._FUSED_QKV.fullmatch(name)
    return [name] if fused is None else [f"{fused[1]}{projection}.{fused[2]}" for projection in "qkv"]


def _checkpoint(state):
    weights = []
    for name, tensor in state.items():
        names = _checkpoint_names(name)
        weights.extend(zip(names, (part.clone() for part in tensor.chunk(len(names)))))
    return weights


def test_strict_checkpoint_loading(tiny_model):
    weights = _checkpoint(tiny_model.state_dict())
    weights.reverse()
    checkpoint = dict(weights)
    for parameter in tiny_model.parameters():
        nn.init.zeros_(parameter)
    assert tiny_model.load_weights(weights) == set(tiny_model.state_dict())
    for prefix in ("img", "txt"):
        base = f"double_blocks.1.attn.{prefix}_attn_"
        for field in ("weight", "bias"):
            fused = getattr(getattr(tiny_model.double_blocks[1].attn, f"{prefix}_attn_qkv"), field)
            expected = torch.cat([checkpoint[f"{base}{projection}.{field}"] for projection in "qkv"])
            assert torch.equal(fused, expected)
    for prefix in ("img", "txt"):
        checkpoint = dict(weights)
        inputs = torch.randn(1, 5, 12)
        mlp_base = f"double_blocks.0.{prefix}_mlp."
        expected_mlp = F.linear(
            F.gelu(
                F.linear(inputs, checkpoint[mlp_base + "fc1.weight"], checkpoint[mlp_base + "fc1.bias"]),
                approximate="tanh",
            ),
            checkpoint[mlp_base + "fc2.weight"],
            checkpoint[mlp_base + "fc2.bias"],
        )
        torch.testing.assert_close(getattr(tiny_model.double_blocks[0], f"{prefix}_mlp")(inputs), expected_mlp)
    mapped_name = "double_blocks.0.img_mlp.fc1.weight"
    with pytest.raises(ValueError, match="Missing"):
        tiny_model.load_weights([(name, tensor) for name, tensor in weights if name != mapped_name])
    with pytest.raises(ValueError, match="Duplicate"):
        tiny_model.load_weights(weights + [(mapped_name, dict(weights)[mapped_name])])
    with pytest.raises(ValueError, match="Missing"):
        tiny_model.load_weights(weights[:-1])
    with pytest.raises(ValueError, match="Unexpected"):
        tiny_model.load_weights([("not_a_parameter", torch.zeros(1))])
    with pytest.raises(ValueError, match="Duplicate"):
        tiny_model.load_weights([weights[0], weights[0]])


@pytest.mark.parametrize("mask_ndim", [None, 2, 4])
def test_attention_preserves_reference_key_layout(mask_ndim):
    # Equivalent visibility is insufficient for BF16 parity: keep the exact
    # sequence lengths and broadcast mask rather than compacting visible keys.
    query = torch.randn(2, 3, 2, 6)
    key = torch.randn(2, 5, 2, 6)
    value = torch.randn_like(key)
    mask = torch.tensor([[True, False, True, False, True], [False, True, True, True, False]])
    expected_mask = None if mask_ndim is None else mask[:, None, None, :]
    supplied_mask = None if mask_ndim is None else mask if mask_ndim == 2 else expected_mask
    calls = []

    def observe(q, k, v, attn_metadata=None):
        assert q is query and k is key and v is value
        actual_mask = None if attn_metadata is None else attn_metadata.attn_mask
        if expected_mask is None:
            assert actual_mask is None
        else:
            assert actual_mask is not None
            assert actual_mask.shape == (2, 1, 1, 5)
            torch.testing.assert_close(actual_mask, expected_mask)
        calls.append((q.shape, k.shape, v.shape))
        return _SDPA()(q, k, v, attn_metadata=attn_metadata)

    actual = model_module._attend(observe, query, key, value, supplied_mask)
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=expected_mask
    ).transpose(1, 2)
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@torch.no_grad()
@pytest.mark.parametrize("bounded", [False, True])
def test_preallocated_history_matches_concat_with_actions_camera_and_short_tail(tiny_model, bounded):
    from vllm_omni.diffusion.models.hy_nomas.cache import HyNomasKVCache

    dynamic: list[dict[str, torch.Tensor]] = [{}, {}]
    reserved = [HyNomasKVCache(39), HyNomasKVCache(39)]
    offset = 0
    for frames in (2, 2, 1):
        pointers = None
        latents = torch.randn(1, 4, frames, 2, 2)
        text = torch.randn(1, 5, 8)
        args = dict(
            hidden_states=latents,
            encoder_hidden_states=text,
            timestep=torch.tensor([500.0]),
            encoder_attention_mask=torch.tensor([[True, True, True, True, False]]),
            frame_offset=offset,
            camera_extrinsics=torch.eye(4).repeat(1, frames * 4, 1, 1),
            action_text_offsets=[[2, 3, 4]],
            action_frame_windows=[(0, frames, 0)],
            cache_action_start=2,
        )
        for commit in (False, False, True):
            if bounded and commit and offset:
                for layer in reserved:
                    layer.evict(layer.length)
                for layer in dynamic:
                    count = layer["key"].shape[1]
                    for name in ("key", "value", "mask"):
                        layer[name] = layer[name][:, count:].clone()
            expected = tiny_model(**args, kv_cache=dynamic, update_cache=commit).sample
            rotary = tiny_model.rotary_embed(frames, 2, 2, latents.device, offset)
            actual = tiny_model(**args, kv_cache=reserved, update_cache=commit, rotary_emb=rotary).sample
            # Compaction drops invisible keys: equal math, different reduction order.
            torch.testing.assert_close(actual, expected)
            current = [layer.key.data_ptr() for layer in reserved]
            if pointers is None:
                pointers = current
            assert current == pointers
            if commit:
                for baseline, cached in zip(dynamic, reserved):
                    visible = baseline["mask"][0]
                    assert cached.mask[:, : cached.length].all()
                    for name in ("key", "value"):
                        torch.testing.assert_close(
                            baseline[name][:, visible], getattr(cached, name)[:, : cached.length]
                        )
        offset += frames


@torch.no_grad()
def test_trimmed_text_and_compact_history_match_padded_mask(tiny_model, monkeypatch):
    from vllm_omni.diffusion.models.hy_nomas.cache import HyNomasKVCache

    masks = []
    attend = model_module._attend

    def observe(attention, query, key, value, mask=None):
        masks.append(mask)
        return attend(attention, query, key, value, mask)

    monkeypatch.setattr(model_module, "_attend", observe)
    text = torch.randn(1, 5, 8)
    text_mask = torch.tensor([[True] * 3 + [False] * 2])
    padded = [HyNomasKVCache(30), HyNomasKVCache(30)]
    trimmed = [HyNomasKVCache(30), HyNomasKVCache(30)]
    for offset in (0, 2):
        args = dict(hidden_states=torch.randn(1, 4, 2, 2, 2), timestep=torch.tensor([500.0]), frame_offset=offset)
        for commit in (False, True):
            expected = tiny_model(
                **args,
                encoder_hidden_states=text,
                encoder_attention_mask=text_mask,
                kv_cache=padded,
                update_cache=commit,
            ).sample
            masks.clear()
            actual = tiny_model(**args, encoder_hidden_states=text[:, :3], kv_cache=trimmed, update_cache=commit).sample
            torch.testing.assert_close(actual, expected)
            # Neither text refinement nor joint attention over history needs a mask.
            assert masks and all(mask is None for mask in masks)
    for masked, compact in zip(padded, trimmed):
        assert masked.length == compact.length == 2 * (8 + 3)
        for name in ("key", "value"):
            torch.testing.assert_close(
                getattr(masked, name)[:, : masked.length], getattr(compact, name)[:, : compact.length]
            )


def test_preallocated_capacity_rejects_overflow_without_advancing_history():
    from vllm_omni.diffusion.models.hy_nomas.cache import HyNomasKVCache

    cache = HyNomasKVCache(3)
    key = torch.randn(1, 2, 2, 4)
    mask = torch.tensor([[True, True]])
    cache.stage((key,), (key,), mask)
    assert cache.length == 0
    cache.commit(mask)
    before = cache.key.clone()
    with pytest.raises(ValueError, match="capacity"):
        cache.stage((key,), (key,), mask)
    assert cache.length == 2
    assert torch.equal(cache.key, before)


def test_commit_keeps_only_visible_tokens():
    from vllm_omni.diffusion.models.hy_nomas.cache import HyNomasKVCache

    key = torch.arange(4.0).reshape(1, 4, 1, 1)
    gathered = HyNomasKVCache(8)
    gathered.stage((key,), (key + 10,), torch.ones(1, 4, dtype=torch.bool))
    gathered.commit(torch.tensor([[True, False, True, False]]))
    assert gathered.length == 2
    assert gathered.key[0, :2, 0, 0].tolist() == [0.0, 2.0]
    assert gathered.value[0, :2, 0, 0].tolist() == [10.0, 12.0]
    prefix = HyNomasKVCache(8)
    prefix.stage((key,), (key,), torch.ones(1, 4, dtype=torch.bool))
    prefix.commit(torch.tensor([[True, True, True, False]]), visible_prefix=3)
    assert prefix.length == 3 and prefix.mask[:, :3].all()
    with pytest.raises(ValueError, match="visible_prefix"):
        prefix.commit(torch.ones(1, 1, dtype=torch.bool), visible_prefix=2)


def test_shared_patch_embed_preserves_reference_layout(tiny_model):
    from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import HunyuanVideo15PatchEmbed

    assert isinstance(tiny_model.img_in, HunyuanVideo15PatchEmbed)
    x = torch.randn(1, 4, 2, 3, 4)
    expected = (
        F.conv3d(x, tiny_model.img_in.proj.weight, tiny_model.img_in.proj.bias).flatten(2).transpose(1, 2).contiguous()
    )
    actual = tiny_model.img_in(x).contiguous()
    assert actual.stride() == expected.stride()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_full_checkpoint_mapping_coverage(monkeypatch):
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(model_module, "_attention", lambda *args: _SDPA())
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: 1)
    with torch.device("meta"):
        model = model_module.HyNomasTransformer3DModel()
    parameters = dict(model.named_parameters())
    # Each fused q/k/v projection replaces three checkpoint projections.
    assert len(parameters) == 2268 - 2 * 2 * 2 * 54
    checkpoint = [(name, target) for target in parameters for name in _checkpoint_names(target)]
    assert len(checkpoint) == len({name for name, _ in checkpoint}) == 2268
    weights = []
    for name, target in checkpoint:
        tensor = parameters[target]
        if model_module._FUSED_QKV.fullmatch(target):
            tensor = tensor[: tensor.shape[0] // 3]
        weights.append((name, tensor))
    assert model.load_weights(weights) == set(parameters)
