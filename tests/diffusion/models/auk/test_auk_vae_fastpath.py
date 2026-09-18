# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The AuK codec's decode fast paths keep the reference numerics.

Four paths are checked against the plain eager decode: the shared Snake
activation with precomputed exponent caches, the per-channel FIR filter cache,
the per-length CUDA graph tier and the torch.compile bucket tier (both fall
back to eager off CUDA).
"""

import pytest
import torch

from vllm_omni.diffusion.models.auk.auk_vae import AuKVAE, LowPass, SnakeBeta, Upsample
from vllm_omni.diffusion.models.auk.vae_cudagraph import AuKVAEDecodeGraph

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _small_vae() -> AuKVAE:
    torch.manual_seed(3)
    # Six halvings of the initial width must leave at least two channels.
    vae = AuKVAE(
        upsample_initial_channel=128,
        downsample_channels=(4, 4, 8, 8, 8, 16, 16),
        latent_dim=8,
    ).eval()
    # Give the activations non-trivial parameters so the paths are exercised.
    with torch.no_grad():
        for module in vae.modules():
            if isinstance(module, SnakeBeta):
                module.alpha.normal_(0.0, 0.3)
                module.beta.normal_(0.0, 0.3)
    vae.remove_weight_norm()
    return vae


def _reference_snake(module: SnakeBeta, x: torch.Tensor) -> torch.Tensor:
    """The formula as the reference implementation writes it."""
    alpha = module.alpha.unsqueeze(0).unsqueeze(-1)
    beta = module.beta.unsqueeze(0).unsqueeze(-1)
    if module.alpha_logscale:
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
    return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).pow(2)


@torch.inference_mode()
def test_eager_snake_matches_the_reference_formula() -> None:
    module = SnakeBeta(6, alpha_logscale=True)
    with torch.no_grad():
        module.alpha.normal_()
        module.beta.normal_()
    module.precompute_exp_cache()
    x = torch.randn(2, 6, 17)

    module.fused = False
    assert torch.equal(module(x), _reference_snake(module, x))
    # On CPU the fused flag has no kernel to reach and takes the same path.
    module.fused = True
    assert torch.equal(module(x), _reference_snake(module, x))


@torch.inference_mode()
def test_filter_cache_leaves_the_filters_and_output_unchanged() -> None:
    lowpass = LowPass(cutoff=0.25, half_width=0.3, stride=2, kernel_size=12, causal=True)
    upsample = Upsample(ratio=2, kernel_size=12)
    x = torch.randn(1, 5, 40)

    for module in (lowpass, upsample):
        module.cache_filters = False
        plain = module(x)
        module.cache_filters = True
        cached = module(x)
        assert torch.equal(cached, plain)
        assert module._expanded is not None and module._expanded.shape == (5, 1, 12)
        assert module._expanded.is_contiguous()
        # A different channel count rebuilds the cache rather than reusing it.
        module(torch.randn(1, 3, 40))
        assert module._expanded.shape == (3, 1, 12)


@torch.inference_mode()
def test_decode_fast_paths_reproduce_the_plain_decode() -> None:
    vae = _small_vae()
    latents = torch.randn(1, 6, vae.latent_dim)

    vae.set_decode_fast_paths(fused_snake=False, cached_filters=False)
    plain = vae.decode(latents)
    assert plain.shape == (1, 6 * vae.hop_size)

    vae.set_decode_fast_paths(fused_snake=True, cached_filters=True)
    fast = vae.decode(latents)
    assert torch.equal(fast, plain)
    assert all(module._cached for module in vae.modules() if isinstance(module, SnakeBeta))


@torch.inference_mode()
def test_graph_wrapper_falls_back_to_eager_off_cuda(mocker) -> None:
    vae = _small_vae()
    wrapper = AuKVAEDecodeGraph(vae, frame_alignment=64, compile_shapes=(8,))
    capture_spy = mocker.spy(wrapper, "_capture")
    compile_spy = mocker.spy(torch, "compile")
    latents = torch.randn(1, 6, vae.latent_dim)

    wrapper.warmup(torch.device("cpu"))
    assert torch.equal(wrapper(latents), vae.decode(latents))
    assert wrapper.last_mode == "eager"
    capture_spy.assert_not_called()
    compile_spy.assert_not_called()
    assert not wrapper._cache and not wrapper._compiled


def test_compiled_bucket_is_the_smallest_captured_one_that_fits() -> None:
    wrapper = AuKVAEDecodeGraph(_small_vae(), compile_shapes=(16, 8, 8))
    assert wrapper.compile_shapes == [8, 16]
    # Nothing captured yet: every length goes to the per-length graph tier.
    assert wrapper.compiled_bucket(6) is None
    wrapper._compiled = {8: object(), 16: object()}  # type: ignore[dict-item]
    assert [wrapper.compiled_bucket(frames) for frames in (6, 8, 9, 16, 17)] == [8, 8, 16, 16, None]
    # A bucket whose capture failed is skipped, not padded to.
    wrapper._compiled = {16: object()}  # type: ignore[dict-item]
    assert wrapper.compiled_bucket(6) == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
def test_graph_replay_matches_eager_per_length() -> None:
    vae = _small_vae().to("cuda")
    wrapper = AuKVAEDecodeGraph(vae, max_graphs=2)
    for frames in (6, 9, 6, 12):
        latents = torch.randn(1, frames, vae.latent_dim, device="cuda")
        eager = vae.decode(latents)
        replay = wrapper(latents)
        # Exact-length graphs replay the very same kernels: bit-identical.
        assert torch.equal(replay, eager), frames
    # LRU: three distinct lengths seen, two graphs kept, the oldest evicted.
    assert list(wrapper._cache) == [6, 12] or list(wrapper._cache) == [9, 12]
    assert len(wrapper._cache) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
def test_bucketed_graph_only_disturbs_the_tail() -> None:
    vae = _small_vae().to("cuda")
    wrapper = AuKVAEDecodeGraph(vae, frame_alignment=8)
    latents = torch.randn(1, 6, vae.latent_dim, device="cuda")
    eager = vae.decode(latents)
    replay = wrapper(latents)
    assert replay.shape == eager.shape and list(wrapper._cache) == [8]
    # The zero padding leaks in through the non-causal conv_pre and the
    # alias-free upsamplers, whose lookahead accumulates through the stack, so
    # a bucketed replay is close to but not identical with the eager decode.
    # That is why frame_alignment defaults to 1.
    assert torch.isfinite(replay).all()
    assert not torch.equal(replay, eager)
    torch.testing.assert_close(replay, eager, atol=0.1, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile + CUDA graph capture requires CUDA")
@torch.inference_mode()
def test_compiled_buckets_replay_within_fusion_tolerance_and_leave_longer_clips_to_plain_graphs() -> None:
    vae = _small_vae().to("cuda")
    wrapper = AuKVAEDecodeGraph(vae, compile_shapes=(8,))
    wrapper.warmup(torch.device("cuda"))
    assert list(wrapper._compiled) == [8]
    # The fused Triton Snake is only disabled while Inductor traces.
    assert all(module.fused for module in vae.modules() if isinstance(module, SnakeBeta))

    exact = torch.randn(1, 8, vae.latent_dim, device="cuda")
    eager = vae.decode(exact)
    replay = wrapper(exact)
    assert wrapper.last_mode == "compiled"
    assert replay.shape == eager.shape
    # Same formula, different fusion order: not bit-identical, but close.
    torch.testing.assert_close(replay, eager, atol=1e-4, rtol=0.0)

    short = torch.randn(1, 5, vae.latent_dim, device="cuda")
    padded = wrapper(short)
    assert wrapper.last_mode == "compiled" and padded.shape == (1, 5 * vae.hop_size)
    torch.testing.assert_close(padded, vae.decode(short), atol=0.1, rtol=0.0)

    long = torch.randn(1, 12, vae.latent_dim, device="cuda")
    assert torch.equal(wrapper(long), vae.decode(long))
    assert wrapper.last_mode == "graph" and list(wrapper._cache) == [12]
