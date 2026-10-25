# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.hy_nomas import decoder as decoder_module
from vllm_omni.diffusion.models.hy_nomas.decoder import (
    PixelUMMDecoder,
    _window_starts,
    local_attention_layout,
    patchify,
    unpatchify,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _SDPA(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), enable_gqa=True
        ).transpose(1, 2)


@pytest.fixture(autouse=True)
def cpu_attention(monkeypatch):
    # CPU math coverage does not initialize a CUDA/distributed backend.
    monkeypatch.setattr(decoder_module, "Attention", _SDPA)


def tiny_decoder(**kwargs):
    return PixelUMMDecoder(
        hidden_size=16,
        intermediate_size=24,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        temporal_window_latents=3,
        temporal_overlap_latents=1,
        **kwargs,
    ).eval()


def test_patchify_preserves_tubelet_order():
    pixels = torch.arange(3 * 8 * 32 * 48).reshape(1, 3, 8, 32, 48)
    tokens = patchify(pixels)
    assert torch.equal(tokens[0, 0], pixels[0, :, :4, :16, :16].permute(1, 2, 3, 0).flatten())
    assert torch.equal(unpatchify(tokens, (2, 2, 3)), pixels)
    with pytest.raises(ValueError, match="multiple"):
        patchify(pixels[:, :, :7])


def test_spatial_neighborhood_includes_partial_edge_tiles():
    permutation, inverse, regions = local_attention_layout((2, 7, 9), 2, "cpu")
    assert torch.equal(permutation[inverse], torch.arange(126))
    assert sum(end - start for (start, end), _ in regions) == 126
    # A corner attends a full 3 x 3 neighborhood, including the second-nearest tile.
    assert len(regions[0][1]) == 9
    assert regions[-1][0][1] == 126
    assert all(0 <= start < end <= 126 for _, neighbors in regions for start, end in neighbors)


def test_window_noise_is_absolute_and_does_not_advance_generator():
    model = tiny_decoder()
    latent = torch.zeros(1, 32, 5, 1, 1)
    all_noise = model._make_noise(latent, 0, 5, 13)
    overlap = model._make_noise(latent, 2, 4, 13)
    assert torch.equal(overlap, all_noise[:, :, 8:16])
    assert _window_starts(8, 3, 2) == [0, 2, 4, 5]


def test_streaming_stable_prefix_matches_full_decode(monkeypatch):
    model = tiny_decoder()

    # A window-sensitive decoder makes wrong overlap placement observable without
    # a released 8B checkpoint or stochastic attention kernels.
    def decode_window(latent, noise):
        frames = latent[:, :1].repeat_interleave(4, dim=2)
        return frames + latent[:, :1].mean(dim=2, keepdim=True) + noise[:, :1, :, :1, :1] * 0.01

    monkeypatch.setattr(model.decoder, "forward", decode_window)
    latents = torch.arange(8).reshape(1, 1, 8, 1, 1).expand(1, 32, 8, 1, 1).float()
    generator = torch.Generator().manual_seed(23)
    before = generator.get_state().clone()
    complete = model.decode(latents, generator=generator, return_dict=False)[0]
    session = model.create_streaming_decode_session(to_cpu=True, generator=generator)
    first = session.push(latents[:, :, :4], pixel_frames=13)
    old_prefix = first.clone()
    second = session.push(latents[:, :, 4:], pixel_frames=15)
    tail = session.finish()
    output = torch.cat([first, second, tail], dim=2)
    expected = torch.cat([complete[:, :, :13], complete[:, :, 16:31]], dim=2)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.equal(first, old_prefix)
    assert torch.equal(generator.get_state(), before)
    assert session.stable_pixel_frames == 28
    torch.testing.assert_close(session.full_video(), expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="after finish"):
        session.push(latents[:, :, :1])


def test_decoder_load_rejects_missing_and_unknown_tensors():
    model = tiny_decoder()
    weights = list(model.state_dict().items())
    assert model.load_weights(weights) == set(model.state_dict())
    with pytest.raises(ValueError, match="Missing"):
        model.load_weights(weights[:-1])
    with pytest.raises(ValueError, match="Unexpected"):
        model.load_weights([("invalid", torch.ones(1))])
