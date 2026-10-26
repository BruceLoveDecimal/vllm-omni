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


def _make_model(monkeypatch, sp_size=1):
    # Select a real CPU math kernel without constructing platform/distributed state.
    monkeypatch.setattr(model_module, "_attention", lambda *args: _SDPA())
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: 1)
    # Exercise shared layer arithmetic without platform-specific packed CPU kernels.
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.UnquantizedLinearMethod.apply",
        lambda self, layer, x, bias=None: F.linear(x, layer.weight, bias),
    )
    torch.manual_seed(13)
    model = model_module.HyNomasTransformer3DModel(
        sp_size=sp_size,
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


@pytest.fixture
def tiny_model(monkeypatch):
    return _make_model(monkeypatch)


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


def test_strict_checkpoint_loading(tiny_model):
    weights = []
    for name, tensor in tiny_model.state_dict().items():
        name = name.replace("_mlp.net.0.proj.", "_mlp.fc1.").replace("_mlp.net.2.", "_mlp.fc2.")
        weights.append((name, tensor.clone()))
    weights.reverse()
    assert tiny_model.load_weights(weights) == set(tiny_model.state_dict())
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
    checkpoint = [
        name.replace("_mlp.net.0.proj.", "_mlp.fc1.").replace("_mlp.net.2.", "_mlp.fc2.") for name in parameters
    ]
    mapped = model.hf_to_vllm_mapper.apply_list(checkpoint)
    assert len(mapped) == len(set(mapped)) == len(parameters) == 2268
    assert set(mapped) == set(parameters)
    assert model.load_weights((name, parameters[target]) for name, target in zip(checkpoint, mapped)) == set(parameters)


def _inputs(model, *, frames=2, history=0, cache_text=True):
    video_tokens = frames * 4
    count = history + video_tokens + (5 if cache_text else 0)
    return dict(
        hidden_states=torch.randn(1, 4, frames, 2, 2),
        text_states=torch.randn(1, 5, 8),
        timestep=torch.tensor([500.0]),
        text_mask=torch.tensor([[True, True, True, True, False]]),
        kv_cache_list=[{k: torch.zeros(1, count, 2 // model.sp_size, 6) for k in ("k", "v")} for _ in range(2)],
        cache_mask_list=[torch.zeros(1, count, dtype=torch.bool) for _ in range(2)],
        cache_valid_len=history,
        cache_txt=cache_text,
    )


@torch.no_grad()
def test_denoising_does_not_commit_and_clean_forward_preserves_history(tiny_model):
    args = _inputs(tiny_model, history=3)
    for cache, mask in zip(args["kv_cache_list"], args["cache_mask_list"]):
        cache["k"][:, :3].normal_()
        cache["v"][:, :3].normal_()
        mask[:, :3] = torch.tensor([True, False, True])
    old = [{key: value.clone() for key, value in cache.items()} for cache in args["kv_cache_list"]]
    old_masks = [mask.clone() for mask in args["cache_mask_list"]]
    output, written = tiny_model.forward_with_cache(**args)
    assert written == 0
    for before, after, mask, old_mask in zip(old, args["kv_cache_list"], args["cache_mask_list"], old_masks):
        assert torch.equal(before["k"], after["k"]) and torch.equal(before["v"], after["v"])
        assert torch.equal(mask, old_mask)
    committed, written = tiny_model.forward_with_cache(**args, update_cache=True)
    torch.testing.assert_close(output, committed, rtol=0, atol=0)
    assert written == 13
    for before, after, mask in zip(old, args["kv_cache_list"], args["cache_mask_list"]):
        assert torch.equal(before["k"][:, :3], after["k"][:, :3])
        assert mask[0].tolist() == [True, False, True] + [True] * 12 + [False]


@torch.no_grad()
def test_continuation_does_not_duplicate_text_keys(tiny_model):
    args = _inputs(tiny_model, history=3, cache_text=False)
    args["cache_mask_list"][0][:, :3] = True
    args["cache_mask_list"][1][:, :3] = True
    lengths = []
    attention = tiny_model.double_blocks[0].attn.attention
    handle = attention.register_forward_pre_hook(
        lambda _, values: lengths.append((values[0].shape[1], values[1].shape[1]))
    )
    try:
        _, written = tiny_model.forward_with_cache(**args)
        assert written == 0
        _, written = tiny_model.forward_with_cache(**args, update_cache=True)
        assert written == 8
    finally:
        handle.remove()
    # Denoising reuses cached text. The clean forward also attends current text,
    # while committing only image tokens, as required by the chunk protocol.
    assert lengths == [(13, 11), (13, 16)]


@torch.no_grad()
def test_camera_disabled_and_prepadded_inputs_match(tiny_model):
    args = _inputs(tiny_model)
    poses = torch.eye(4).repeat(1, 5, 1, 1)
    camera = tiny_model._preprocess_camera(poses, None, None, 2, 2, torch.float32, torch.device("cpu"))
    padded = torch.cat((poses[:, :1].repeat(1, 3, 1, 1), poses), dim=1)
    expected = tiny_model._preprocess_camera(padded, None, None, 2, 2, torch.float32, torch.device("cpu"))
    torch.testing.assert_close(camera, expected, rtol=0, atol=0)
    baseline = tiny_model.forward_with_cache(**args)[0]
    gated = tiny_model.forward_with_cache(**args, camera_embedding=camera, camera_has_mask=torch.zeros(1))[0]
    torch.testing.assert_close(baseline, gated, rtol=0, atol=0)


def test_invalid_cache_rejected_before_any_write(tiny_model):
    args = _inputs(tiny_model)
    args["kv_cache_list"][-1]["v"] = torch.empty(1, 1, 1, 1)
    before = args["kv_cache_list"][0]["k"].clone()
    with pytest.raises(ValueError, match="rank-local heads"):
        tiny_model.forward_with_cache(**args, update_cache=True)
    assert torch.equal(before, args["kv_cache_list"][0]["k"])


def _sp_worker(rank, rendezvous):
    from datetime import timedelta
    from types import SimpleNamespace

    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    patch = pytest.MonkeyPatch()
    try:

        def gather(x, dim=0, group=None):
            peers = [torch.empty_like(x) for _ in range(2)]
            dist.all_gather(peers, x.contiguous(), group=group)
            return torch.cat(peers, dim=dim)

        group = SimpleNamespace(ulysses_group=dist.group.WORLD, rank_in_group=rank, all_gather=gather)
        patch.setattr(model_module, "get_sp_group", lambda: group)
        single = _make_model(patch)
        parallel = _make_model(patch, sp_size=2)
        parallel.load_state_dict(single.state_dict())
        previous = previous_masks = None
        history = frame_offset = 0
        with torch.no_grad():
            for frames, cache_text in [(3, True), (1, False), (2, True)]:
                # Twelve tokens on two ranks splits a frame across ranks.
                torch.manual_seed(101 + frame_offset)
                args = _inputs(single, frames=frames, history=history, cache_text=cache_text)
                if previous is not None:
                    for dest, src, mask, old_mask in zip(
                        args["kv_cache_list"], previous, args["cache_mask_list"], previous_masks
                    ):
                        for key in ("k", "v"):
                            dest[key][:, :history].copy_(src[key])
                        mask[:, :history].copy_(old_mask)
                local = [
                    {key: tensor[:, :, rank : rank + 1].clone() for key, tensor in layer.items()}
                    for layer in args["kv_cache_list"]
                ]
                local_masks = [mask.clone() for mask in args["cache_mask_list"]]
                camera = torch.randn(1, frames, 64)
                extras = dict(camera_embedding=camera, global_frame_counter=frame_offset)
                if cache_text:
                    extras.update(intra_seg_txt_offsets=[[2, 3, 4]], intra_seg_vid_windows=[(0, frames, 0)])
                for commit in (False, True):
                    expected, written = single.forward_with_cache(**args, **extras, update_cache=commit)
                    actual, local_written = parallel.forward_with_cache(
                        **{**args, "kv_cache_list": local, "cache_mask_list": local_masks},
                        **extras,
                        update_cache=commit,
                    )
                    assert written == local_written
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                for full, shard, mask, shard_mask in zip(
                    args["kv_cache_list"], local, args["cache_mask_list"], local_masks
                ):
                    for key in ("k", "v"):
                        torch.testing.assert_close(shard[key], full[key][:, :, rank : rank + 1], atol=2e-6, rtol=2e-5)
                    assert torch.equal(mask, shard_mask)
                previous, previous_masks = args["kv_cache_list"], args["cache_mask_list"]
                history += written
                frame_offset += frames
    finally:
        patch.undo()
        dist.destroy_process_group()


def test_two_process_ulysses_matches_single_rank_with_continuation_and_camera(tmp_path):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_gloo_available():
        pytest.skip("Gloo is required for CPU Ulysses validation")
    mp.spawn(_sp_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
