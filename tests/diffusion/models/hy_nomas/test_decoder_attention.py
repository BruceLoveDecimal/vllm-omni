# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

reference = pytest.importorskip("hymm.models.vae.pixelumm_decoder.model")
from vllm_omni.diffusion.models.hy_nomas import decoder_attention  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class CPUAttention(nn.Module):
    use_shared_sdpa = False

    def __init__(self, **kwargs):
        super().__init__()
        self.heads = (kwargs["num_heads"], kwargs["num_kv_heads"])
        self.calls = 0

    def forward(self, query, key, value):
        assert (query.shape[2], key.shape[2]) == self.heads
        assert value.shape[2] == self.heads[1]
        self.calls += 1
        if self.use_shared_sdpa:
            from vllm_omni.diffusion.attention.backends.sdpa import _sdpa_attention

            return _sdpa_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                None,
                False,
                query.shape[-1] ** -0.5,
            ).transpose(1, 2)
        return F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), enable_gqa=True
        ).transpose(1, 2)


@pytest.fixture(params=[False, True], ids=["torch_gqa", "shared_sdpa"])
def decoder(monkeypatch, request):
    monkeypatch.setattr(decoder_attention, "Attention", CPUAttention)
    monkeypatch.setattr(CPUAttention, "use_shared_sdpa", request.param)
    torch.manual_seed(42)
    arch = reference.GenExpertArch(
        hidden_size=32, intermediate_size=48, num_layers=2, num_heads=4, num_kv_heads=1, head_dim=8
    )
    return reference.DCAEVPixelUMMDecoder(arch).eval()


@torch.no_grad()
def test_compact_decoder_matches_reference_and_preserves_weight_names(decoder):
    expected_model = copy.deepcopy(decoder)
    before = dict(decoder.named_parameters())
    decoder_attention.enable_compact_gqa(decoder)
    decoder_attention.enable_compact_gqa(decoder)
    after = dict(decoder.named_parameters())
    assert before.keys() == after.keys()
    assert all(after[name] is tensor for name, tensor in before.items())
    decoder.load_state_dict(expected_model.state_dict(), strict=True)
    for frames in (1, 2):
        latents = torch.randn(1, 32, frames, 1, 1)
        pixels = torch.randn(1, 3, frames * 4, 32, 32)
        expected = expected_model(latents, pixels)
        actual = decoder(latents, pixels)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert all(layer.attn.attention.calls == 2 for layer in decoder.layers)


@torch.no_grad()
def test_local_attention_keeps_reference_flex_gqa(decoder, monkeypatch):
    marker = object()
    calls = []

    def flex(query, key, value, block_mask):
        assert block_mask is marker
        assert query.shape[1] == 4 and key.shape[1] == value.shape[1] == 1
        calls.append(True)
        return F.scaled_dot_product_attention(query, key, value, enable_gqa=True)

    monkeypatch.setattr(reference, "_flex_attention", flex)
    source = decoder.layers[0].attn
    candidate = decoder_attention.HyNomasDecoderAttention(source)
    x = torch.randn(1, 8, 32)
    cosine, sine = reference.mrope_tables(decoder.arch, (2, 2, 2), x.device)
    expected = source(x, cosine, sine, marker)
    actual = candidate(x, cosine, sine, marker)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == [True, True]
    assert candidate.attention.calls == 0


@torch.no_grad()
def test_streaming_chunks_and_final_flush_match_reference(decoder):
    autoencoder = pytest.importorskip("hymm.models.vae.pixelumm_decoder.autoencoder")
    source = copy.deepcopy(decoder)
    decoder_attention.enable_compact_gqa(decoder)
    sessions = [
        autoencoder.PixelUMMDCAELatentDecoder(
            model,
            temporal_window_latents=2,
            temporal_overlap_latents=1,
            decoder_tile_pixels=32,
            decoder_tile_overlap_pixels=0,
        ).create_streaming_decode_session(to_cpu=True, retain_history=False)
        for model in (source, decoder)
    ]
    for _ in range(3):
        latent = torch.randn(1, 32, 1, 1, 1)
        expected, actual = [session.push(latent, pixel_frames=4) for session in sessions]
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        assert sessions[0].available_pixel_frames == sessions[1].available_pixel_frames
    expected, actual = [session.finish() for session in sessions]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
