# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the batched HiFT vocoder path.

``inference_batch`` pads variable-length mels and runs one conv-trunk pass;
every row must match ``inference`` run on that row alone. Uses the real
CosyVoice3 topology (upsample rates / istft params / look-right convs) with
shrunken channel widths so it runs on CPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import (
    CausalConvRNNF0Predictor,
    CausalHiFTGenerator,
)
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import (
    CosyVoice3Code2Wav,
    plan_vocoder_buckets,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tiny_hift() -> CausalHiFTGenerator:
    torch.manual_seed(0)
    hift = CausalHiFTGenerator(
        in_channels=80,
        base_channels=16,
        nb_harmonics=8,
        sampling_rate=24000,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        conv_pre_look_right=4,
        f0_predictor=CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=16),
    )
    return hift.float().eval()


@pytest.mark.parametrize("finalize", [True, False])
def test_inference_batch_matches_per_row(finalize: bool) -> None:
    hift = _tiny_hift()
    torch.manual_seed(1)
    mels = [torch.randn(1, 80, t) for t in (33, 17, 48)]

    with torch.inference_mode():
        expected = [hift.inference(speech_feat=mel, finalize=finalize)[0] for mel in mels]
        actual = hift.inference_batch(mels, finalize=finalize)

    assert len(actual) == len(expected)
    for row, reference in zip(actual, expected):
        assert row.shape == reference.shape
        torch.testing.assert_close(row, reference, rtol=1e-4, atol=1e-5)


def test_f0_left_context_matches_condnet_geometry() -> None:
    hift = _tiny_hift()
    # condnet[0] looks right only; the four convs after it look left.
    assert hift.f0_predictor.condnet[0].causal_padding == 3
    assert hift.f0_left_context == 8


@pytest.mark.parametrize("finalize", [True, False])
def test_predict_f0_incremental_matches_full_pass(finalize: bool) -> None:
    """Growing the mel chunk by chunk must give the same f0 as one full pass."""
    hift = _tiny_hift()
    torch.manual_seed(5)
    mel = torch.randn(1, 80, 200)
    chunk = 30

    with torch.inference_mode():
        expected = hift.predict_f0(mel, finalize=finalize)

        cached = None
        # Every chunk but the last is mid-stream, so it predicts with
        # finalize=False and leaves the look-right frames for the next one.
        for end in range(chunk, mel.shape[-1], chunk):
            cached = hift.predict_f0(mel[:, :, :end], finalize=False, cached_f0=cached)
        actual = hift.predict_f0(mel, finalize=finalize, cached_f0=cached)

    assert actual.shape == expected.shape
    # Bit-exact, not merely close: float64 prediction is what buys this, and a
    # regression to float32 would show up here as drift rather than a failure
    # at some arbitrary tolerance.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_predict_f0_runs_in_float64_on_cpu() -> None:
    """Precision is load-bearing: float32 slicing noise grows into audible drift."""
    hift = _tiny_hift()
    torch.manual_seed(6)
    mel = torch.randn(1, 80, 64)

    with torch.inference_mode():
        actual = hift.predict_f0(mel, finalize=True)
        assert next(hift.f0_predictor.parameters()).dtype is torch.float64
        assert actual.dtype is mel.dtype

        expected = hift.f0_predictor(mel.double(), finalize=True).to(mel)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_predict_f0_short_cache_is_rebuilt_not_trusted() -> None:
    """A cache shorter than the warmup still yields full-pass values."""
    hift = _tiny_hift()
    torch.manual_seed(7)
    mel = torch.randn(1, 80, 90)

    with torch.inference_mode():
        expected = hift.predict_f0(mel, finalize=True)
        cached = hift.predict_f0(mel[:, :, :5], finalize=False)  # 2 frames
        actual = hift.predict_f0(mel, finalize=True, cached_f0=cached)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_inference_accepts_precomputed_f0() -> None:
    hift = _tiny_hift()
    torch.manual_seed(8)
    mel = torch.randn(1, 80, 40)

    with torch.inference_mode():
        expected, _ = hift.inference(speech_feat=mel, finalize=True)
        f0 = hift.predict_f0(mel, finalize=True)
        actual, _ = hift.inference(speech_feat=mel, finalize=True, f0=f0)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("finalize", [True, False])
def test_inference_batch_accepts_precomputed_f0(finalize: bool) -> None:
    hift = _tiny_hift()
    torch.manual_seed(9)
    mels = [torch.randn(1, 80, t) for t in (36, 22)]

    with torch.inference_mode():
        expected = hift.inference_batch(mels, finalize=finalize)
        f0s = [hift.predict_f0(mel, finalize=finalize) for mel in mels]
        actual = hift.inference_batch(mels, finalize=finalize, f0s=f0s)

    for row, reference in zip(actual, expected):
        torch.testing.assert_close(row, reference, rtol=0, atol=0)


def test_inference_batch_single_row_matches_inference() -> None:
    hift = _tiny_hift()
    torch.manual_seed(2)
    mel = torch.randn(1, 80, 25)

    with torch.inference_mode():
        expected, _ = hift.inference(speech_feat=mel, finalize=True)
        (actual,) = hift.inference_batch([mel], finalize=True)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_streaming_f0_cache_matches_recomputing_each_chunk() -> None:
    """The whole point: caching f0 must not change a single emitted sample."""
    code2wav = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(code2wav)
    code2wav.hift = _tiny_hift()

    torch.manual_seed(10)
    mel_full = torch.randn(1, 80, 150)
    chunk = 30

    with torch.inference_mode():
        cached_f0 = None
        for end in range(chunk, mel_full.shape[-1] + 1, chunk):
            finalize = end >= mel_full.shape[-1]
            mel = mel_full[:, :, :end]

            cached_f0 = code2wav.hift.predict_f0(mel, finalize=finalize, cached_f0=cached_f0)
            with_cache = code2wav.vocode_batch([mel], [finalize], [cached_f0])[0]
            # No f0 supplied -> HiFT predicts over the whole mel, as before.
            without_cache = code2wav.vocode_batch([mel], [finalize])[0]

            assert with_cache.shape == without_cache.shape
            torch.testing.assert_close(with_cache, without_cache, rtol=0, atol=0)


def test_vocode_batch_mixed_finalize_and_empty_rows() -> None:
    code2wav = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(code2wav)
    code2wav.hift = _tiny_hift()

    torch.manual_seed(3)
    mels = [
        torch.randn(1, 80, 20),
        torch.zeros(1, 80, 0),
        torch.randn(1, 80, 31),
        torch.randn(1, 80, 12),
    ]
    finalize_flags = [True, True, False, True]

    with torch.inference_mode():
        results = code2wav.vocode_batch(mels, finalize_flags)
        expected = [
            None if mel.shape[-1] == 0 else code2wav.hift.inference(speech_feat=mel, finalize=fin)[0]
            for mel, fin in zip(mels, finalize_flags)
        ]

    assert results[1].numel() == 0
    for row, reference in zip(results, expected):
        if reference is None:
            continue
        assert row.shape == reference.shape
        torch.testing.assert_close(row, reference, rtol=1e-4, atol=1e-5)


def test_vocode_batch_falls_back_to_per_row_for_long_mels(monkeypatch: pytest.MonkeyPatch) -> None:
    code2wav = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(code2wav)
    code2wav.hift = _tiny_hift()

    # Keep the tensors small by lowering the cap instead of building a
    # 1280-frame mel; the routing decision is what matters here.
    monkeypatch.setattr("vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav.MAX_BATCHED_VOCODER_FRAMES", 20)
    batched: list[int] = []
    original = code2wav.hift.inference_batch
    code2wav.hift.inference_batch = lambda feats, finalize, f0s=None: (
        batched.append(len(feats)) or original(feats, finalize, f0s)
    )

    torch.manual_seed(4)
    mels = [torch.randn(1, 80, t) for t in (18, 40, 15)]

    with torch.inference_mode():
        results = code2wav.vocode_batch(mels, [True, True, True])
        expected = [code2wav.hift.inference(speech_feat=mel, finalize=True)[0] for mel in mels]

    # Only the two short rows batch together; the 40-frame row runs alone.
    assert batched == [2]
    for row, reference in zip(results, expected):
        torch.testing.assert_close(row, reference, rtol=1e-4, atol=1e-5)


def test_plan_vocoder_buckets_covers_each_row_once() -> None:
    lengths = [400, 30, 395, 60, 388, 10]
    buckets = plan_vocoder_buckets(lengths)

    seen = sorted(i for bucket in buckets for i in bucket)
    assert seen == list(range(len(lengths)))


def test_plan_vocoder_buckets_splits_on_padding_waste() -> None:
    # 100/98 pad cheaply together; adding the length-10 row would cost
    # 100*3 padded frames for 208 real ones (waste 1.44 > 1.3).
    assert plan_vocoder_buckets([100, 98, 10], waste_threshold=1.3) == [[0, 1], [2]]


def test_plan_vocoder_buckets_keeps_similar_lengths_together() -> None:
    assert plan_vocoder_buckets([90, 100, 95, 92], waste_threshold=1.3) == [[1, 2, 3, 0]]
