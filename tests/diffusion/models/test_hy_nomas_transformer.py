# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU math contracts; shared backend integration needs a separate GPU smoke."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.diffusion.models.hy_nomas import transformer as model_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _SDPA(nn.Module):
    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)


@pytest.fixture
def tiny_model(monkeypatch):
    # Select a real CPU math kernel without constructing platform/distributed state.
    monkeypatch.setattr(model_module, "_attention", lambda *args: _SDPA())
    torch.manual_seed(13)
    return model_module.HyNomasTransformer3DModel(
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


def test_strict_checkpoint_loading(tiny_model):
    weights = list(tiny_model.state_dict().items())
    assert tiny_model.load_weights(weights) == set(tiny_model.state_dict())
    with pytest.raises(ValueError, match="Missing"):
        tiny_model.load_weights(weights[:-1])
    with pytest.raises(ValueError, match="Unexpected"):
        tiny_model.load_weights([("not_a_parameter", torch.zeros(1))])
    with pytest.raises(ValueError, match="Duplicate"):
        tiny_model.load_weights([weights[0], weights[0]])
