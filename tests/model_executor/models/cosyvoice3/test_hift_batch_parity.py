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
    _slice_causal_noise,
)
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import (
    VOCODER_LEFT_CONTEXT_FRAMES,
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


def _source_full(hift, f0):
    s = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
    s, _, _, carry = hift.m_source(s)
    return s.transpose(1, 2), carry


def _source_windowed(hift, f0, chunk):
    """Synthesize the same excitation one chunk at a time."""
    pieces = []
    carry = None
    offset = 0
    for start in range(0, f0.shape[-1], chunk):
        part = f0[:, start : start + chunk]
        s = hift.f0_upsamp(part[:, None]).transpose(1, 2)
        s, _, _, carry = hift.m_source(s, sample_offset=offset, phase_carry=carry)
        pieces.append(s.transpose(1, 2))
        offset += s.shape[1]
    return torch.cat(pieces, dim=-1)


def _source_reference(f0: torch.Tensor) -> torch.Tensor:
    """Full-utterance excitation in float64 — the ground truth both paths approximate."""
    hift64 = _tiny_hift().double()
    with torch.inference_mode():
        s, _ = _source_full(hift64, f0.double())
    return s.float()


def test_windowed_source_is_no_worse_than_full_utterance() -> None:
    """The excitation must be reproducible window by window.

    SineGen2 integrates phase with a global cumsum and reads its noise tables
    from absolute position 0, so before the carry/offset plumbing every window
    restarted both. That is what pinned windowed HiFT at ~-44 dBFS against full
    replay no matter how much left context it was given.

    The windowed result is *not* bit-identical to the full-utterance one, and
    should not be: full replay lets the phase grow to thousands of radians,
    where float32 ``sin`` loses precision, while the windowed path wraps the
    carry to one period. Both are approximations of the float64 result, so that
    is what they are measured against.
    """
    hift = _tiny_hift()
    torch.manual_seed(11)
    f0 = torch.rand(1, 200) * 300.0
    reference = _source_reference(f0)

    with torch.inference_mode():
        full, _ = _source_full(hift, f0)
        full_err = (full - reference).abs().max()

        for chunk in (25, 40, 100):
            actual = _source_windowed(hift, f0, chunk)
            assert actual.shape == reference.shape
            windowed_err = (actual - reference).abs().max()
            assert windowed_err <= full_err, f"chunk={chunk}: windowed {windowed_err:.3e} > full {full_err:.3e}"


def test_windowed_source_needs_both_phase_and_offset() -> None:
    """Dropping either piece of state must visibly break the excitation."""
    hift = _tiny_hift()
    torch.manual_seed(12)
    f0 = torch.rand(1, 120) * 300.0
    chunk = 40
    reference = _source_reference(f0)

    with torch.inference_mode():

        def windowed(use_phase: bool, use_offset: bool):
            pieces, carry, offset = [], None, 0
            for start in range(0, f0.shape[-1], chunk):
                s = hift.f0_upsamp(f0[:, None, start : start + chunk]).transpose(1, 2)
                s, _, _, carry = hift.m_source(
                    s, sample_offset=offset if use_offset else 0, phase_carry=carry if use_phase else None
                )
                pieces.append(s.transpose(1, 2))
                offset += s.shape[1]
            return torch.cat(pieces, dim=-1)

        # Each omission is catastrophic, four orders of magnitude worse than the
        # ~1e-5 float32 noise floor the correct path sits at.
        assert (windowed(True, False) - reference).abs().max() > 1e-3
        assert (windowed(False, True) - reference).abs().max() > 1e-3
        assert (windowed(True, True) - reference).abs().max() < 1e-4


def _run_bounded_window_stream(hift, mel, chunk: int, left_context: int) -> torch.Tensor:
    """Decode ``mel`` chunk by chunk, keeping only ``left_context`` frames of state."""
    spf = hift.source_samples_per_frame
    look_right = hift.f0_predictor.condnet[0].causal_padding
    total_frames = mel.shape[-1]

    with torch.inference_mode():
        emitted: list[torch.Tensor] = []
        f0_cache: torch.Tensor | None = None
        src_cache = torch.zeros(1, 1, 0)
        carry = None
        # All counters are absolute frame indices into the utterance.
        ctx_start = 0  # first frame held in the caches
        src_end = 0  # frames whose excitation exists
        emitted_frames = 0  # frames already emitted as audio

        for end in range(chunk, total_frames + 1, chunk):
            finalize = end >= total_frames
            window = mel[:, :, ctx_start:end]

            # f0 over the window, extending the cached prefix rather than redoing it
            f0_window = hift.predict_f0(window, finalize=True, cached_f0=f0_cache)

            # Mid-stream the last few frames have no look-right yet, so the
            # excitation stops short of them and picks them up next round.
            target = end if finalize else end - look_right
            new_f0 = f0_window[:, src_end - ctx_start : target - ctx_start]
            new_src, carry = hift.synthesize_source(new_f0, sample_offset=src_end * spf, phase_carry=carry)
            source = torch.cat([src_cache, new_src], dim=-1)

            audio, _ = hift.inference(speech_feat=window, finalize=finalize, source=source)
            # Mid-stream, decode() withholds conv_pre's look-right frames plus
            # one more for the istft tail, so the audio covers
            # [ctx_start, target - look_right_conv - 1). Emit only that.
            covered = target if finalize else target - hift.conv_pre_look_right - 1
            emitted.append(audio[..., (emitted_frames - ctx_start) * spf : (covered - ctx_start) * spf])

            src_end, emitted_frames = target, covered
            # Slide the caches down to ``left_context`` frames of history.
            # Only frames below ``target`` are cached: the ones above it were
            # predicted with a zero look-right and must be redone next round,
            # or the provisional values would be frozen into the stream.
            new_ctx_start = max(0, end - left_context)
            drop = new_ctx_start - ctx_start
            f0_cache = f0_window[:, drop : target - ctx_start]
            src_cache = source[:, :, drop * spf :]
            ctx_start = new_ctx_start

        return torch.cat(emitted, dim=-1).reshape(1, -1)


@pytest.mark.parametrize("chunk", [15, 30])
def test_bounded_window_streaming_matches_full_replay(chunk: int) -> None:
    """A bounded left-context window must reproduce full-replay audio.

    This is what the windowed design rests on: per-chunk cost becomes
    O(window) instead of O(history), so it must not change the output. The
    window carries three things — mel left context for the conv trunk, the
    excitation over that same span, and the source generator's phase/offset.
    With enough context the two agree to float32 round-off, which is only
    possible because the phase and noise-table position are carried across
    windows rather than restarted.
    """
    hift = _tiny_hift()
    torch.manual_seed(13)
    mel = torch.randn(1, 80, 240)

    with torch.inference_mode():
        expected, _ = hift.inference(speech_feat=mel, finalize=True)
    expected = expected.reshape(1, -1)
    actual = _run_bounded_window_stream(hift, mel, chunk=chunk, left_context=48)

    assert actual.shape == expected.shape, (
        f"windowed stream emitted {actual.shape[-1]} samples, full replay {expected.shape[-1]}"
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_windowed_stream_never_caches_provisional_f0() -> None:
    """Frames predicted with a zero look-right must not be cached as final.

    ``predict_f0(finalize=True)`` pads the right edge with zeros, so the last
    few frames of every window are provisional. Caching them would freeze the
    provisional values into the stream — a drift of ~8e-3 Hz per chunk here,
    which the phase integral downstream amplifies.
    """
    hift = _tiny_hift()
    torch.manual_seed(13)
    mel = torch.randn(1, 80, 240)
    chunk, left_context = 30, 48
    look_right = hift.f0_predictor.condnet[0].causal_padding

    with torch.inference_mode():
        truth = hift.predict_f0(mel, finalize=True)

        used = torch.zeros(1, 0)
        f0_cache, ctx_start = None, 0
        for end in range(chunk, mel.shape[-1] + 1, chunk):
            finalize = end >= mel.shape[-1]
            f0_window = hift.predict_f0(mel[:, :, ctx_start:end], finalize=True, cached_f0=f0_cache)
            target = end if finalize else end - look_right
            used = torch.cat([used, f0_window[:, used.shape[-1] - ctx_start : target - ctx_start]], dim=-1)
            new_ctx_start = max(0, end - left_context)
            f0_cache = f0_window[:, new_ctx_start - ctx_start : target - ctx_start]
            ctx_start = new_ctx_start

    assert used.shape == truth.shape
    torch.testing.assert_close(used, truth, rtol=0, atol=0)


def test_bounded_window_needs_enough_left_context() -> None:
    """Pin the measured context requirement, and that too little really breaks.

    Measured on the real conv topology (kernel sizes, dilations and upsample
    rates are the production ones; only channel widths are shrunk, which does
    not affect receptive field): 32 frames already reaches float32 round-off
    and more buys nothing, while 16 frames is audibly wrong at -50 dBFS.
    """
    hift = _tiny_hift()
    torch.manual_seed(13)
    mel = torch.randn(1, 80, 240)

    with torch.inference_mode():
        expected, _ = hift.inference(speech_feat=mel, finalize=True)
    expected = expected.reshape(1, -1)

    errors = {
        lc: (_run_bounded_window_stream(hift, mel, chunk=30, left_context=lc) - expected).abs().max().item()
        for lc in (16, 32, 96)
    }
    assert errors[16] > 1e-3, errors
    assert errors[32] < 1e-6, errors
    # Beyond the receptive field extra context is pure cost: tripling it must
    # not buy even an order of magnitude. (Not equality — the two land on the
    # float32 floor and differ by backend-dependent rounding, so this passed on
    # one machine and failed on another when it asserted they matched exactly.)
    assert errors[96] < 1e-6, errors
    assert errors[32] < errors[96] * 10, errors


def test_causal_noise_table_exhaustion_is_reported() -> None:
    hift = _tiny_hift()
    with pytest.raises(ValueError, match="Causal HiFT cannot synthesize past that length"):
        hift.m_source.l_sin_gen.sine_waves[:, : 300 * 24000]  # table is this long
        _slice_causal_noise(hift.m_source.uv, 300 * 24000 - 10, 100, "SourceModuleHnNSF.uv")


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
    # 512-frame mel; the routing decision is what matters here.
    monkeypatch.setattr("vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav.MAX_BATCHED_VOCODER_FRAMES", 20)
    batched: list[int] = []
    original = code2wav.hift.inference_batch
    code2wav.hift.inference_batch = lambda feats, finalize, f0s=None, sources=None: (
        batched.append(len(feats)) or original(feats, finalize, f0s, sources)
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


def test_code2wav_streaming_window_matches_one_shot() -> None:
    """Drive the production streaming helpers and compare against one-shot decode.

    Exercises ``prepare_streaming_mel`` → ``vocode_batch`` → ``emit_streaming``
    with the flow model stubbed out, so it covers the real cache hand-off,
    frame accounting and sliding — the parts that are easy to get subtly wrong.
    """
    code2wav = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(code2wav)
    code2wav.hift = _tiny_hift()

    torch.manual_seed(17)
    total_frames, chunk = 180, 30
    full_mel = torch.randn(1, 80, total_frames)

    # Stand in for flow: hand back the next slice of the utterance.
    produced = {"frames": 0}

    def fake_forward_mel(**kwargs):
        start = produced["frames"]
        produced["frames"] = min(start + chunk, total_frames)
        return full_mel[:, :, start : produced["frames"]]

    code2wav._forward_mel = fake_forward_mel

    with torch.inference_mode():
        expected, _ = code2wav.hift.inference(speech_feat=full_mel, finalize=True)

        emitted, cache = [], None
        for step in range(total_frames // chunk):
            finalize = step == total_frames // chunk - 1
            mel, window = code2wav.prepare_streaming_mel(
                token=torch.zeros(1, 1, dtype=torch.long),
                prompt_token=torch.zeros(1, 1, dtype=torch.long),
                prompt_feat=torch.zeros(1, 1, 80),
                embedding=torch.zeros(1, 1),
                cache_state=cache,
                finalize=finalize,
            )
            speech = code2wav.vocode_batch([mel], [finalize], [window.f0], [window.source])[0]
            audio, cache = code2wav.emit_streaming(speech, window, finalize)
            emitted.append(audio.reshape(1, -1))

            # The window must stay bounded no matter how long the stream runs.
            assert mel.shape[-1] <= VOCODER_LEFT_CONTEXT_FRAMES + chunk

        actual = torch.cat(emitted, dim=-1)

    assert actual.shape[-1] == expected.reshape(1, -1).shape[-1]
    torch.testing.assert_close(actual, expected.reshape(1, -1), rtol=0, atol=1e-6)


def test_trunk_graph_on_cpu_stays_eager_and_exact() -> None:
    """Off-CUDA the wrapper must be a transparent pass-through."""
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.trunk_graph import HiFTTrunkGraph

    hift = _tiny_hift()
    torch.manual_seed(20)
    mels = [torch.randn(1, 80, t) for t in (30, 24)]

    with torch.inference_mode():
        expected = hift.inference_batch(mels, finalize=True)
        hift.trunk_graph = HiFTTrunkGraph(lambda mel, s_stft: hift._decode_trunk(hift.conv_pre(mel), s_stft))
        actual = hift.inference_batch(mels, finalize=True)

    assert not hift.trunk_graph._graphs, "no graphs may be captured on CPU"
    for row, reference in zip(actual, expected):
        torch.testing.assert_close(row, reference, rtol=0, atol=0)


def test_vocode_batch_routes_single_rows_through_graph_path() -> None:
    """With graph replay installed, a lone stream must also reach inference_batch."""
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.trunk_graph import HiFTTrunkGraph

    code2wav = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(code2wav)
    code2wav.hift = _tiny_hift()

    torch.manual_seed(21)
    mel = torch.randn(1, 80, 26)
    batched: list[int] = []
    original = code2wav.hift.inference_batch
    code2wav.hift.inference_batch = lambda feats, finalize, f0s=None, sources=None: (
        batched.append(len(feats)) or original(feats, finalize, f0s, sources)
    )

    with torch.inference_mode():
        code2wav.vocode_batch([mel], [True])
        assert batched == [], "without a graph, single rows keep the inference path"

        code2wav.hift.trunk_graph = HiFTTrunkGraph(
            lambda m, s: code2wav.hift._decode_trunk(code2wav.hift.conv_pre(m), s)
        )
        code2wav.vocode_batch([mel], [True])
        assert batched == [1], "with a graph, single rows use inference_batch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
def test_trunk_graph_replay_matches_eager_on_cuda() -> None:
    """Captured replay must reproduce the eager trunk, and reuse its graph."""
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.trunk_graph import HiFTTrunkGraph

    hift = _tiny_hift().cuda()
    torch.manual_seed(22)
    mels = [torch.randn(1, 80, 30, device="cuda") for _ in range(2)]

    with torch.inference_mode():
        eager = hift.inference_batch(mels, finalize=True)

        hift.trunk_graph = HiFTTrunkGraph(lambda mel, s_stft: hift._decode_trunk(hift.conv_pre(mel), s_stft))
        first = hift.inference_batch(mels, finalize=True)
        assert len(hift.trunk_graph._graphs) == 1

        # Fresh inputs through the replayed graph. The eager reference must use
        # the SAME instance (graph toggled off): SineGen2's noise tables are
        # drawn at construction and not part of the state_dict, so a second
        # instance decodes with different noise and any comparison is void.
        mels2 = [torch.randn(1, 80, 30, device="cuda") for _ in range(2)]
        saved_graph = hift.trunk_graph
        hift.trunk_graph = None
        eager2 = hift.inference_batch(mels2, finalize=True)
        hift.trunk_graph = saved_graph
        replayed2 = hift.inference_batch(mels2, finalize=True)
        assert len(hift.trunk_graph._graphs) == 1, "same shape must reuse the captured graph"

    for got, ref in zip(first, eager):
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)
    for got, ref in zip(replayed2, eager2):
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)
