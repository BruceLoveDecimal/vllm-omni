# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The batched streaming HiFT step must emit what the per-request path emits.

Streams start at different hops so one batch mixes first windows, short
histories and full windows with F0 margins, plus a final hop per stream.
Batches of one to four windows pad to row slots and frame steps.
"""

import pytest
import torch

from tests.model_executor.models.cosyvoice3.test_cosyvoice3_incremental_hift import _make_hift, _make_model
from vllm_omni.model_executor.models.cosyvoice3 import cosyvoice3_code2wav

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.usefixtures("single_threaded_cpu")]

# Batch size and padded width change convolution rounding, which builds up to
# ~4e-5 over HiFT's layers on CPU and GPU. A wrong offset, margin or phase
# carry moves samples by 4e-3 or more, so 2e-4 (~6.5 PCM16 steps) separates them.
ATOL, RTOL = 2e-4, 1e-4
# (first hop frames, later hop frames, hops, start step) per stream
STREAMS = [(56, 50, 4, 0), (70, 100, 3, 1), (30, 24, 5, 0), (100, 50, 2, 2)]


def _voiced_model(window_len: int):
    """A small HiFT with speech-like F0 (median ~110 Hz, a few unvoiced frames),
    so the harmonic phase carry and the first window's random phase reach the
    output; the untrained predictor's F0 sits far below the voicing threshold."""
    hift = _make_hift()
    with torch.no_grad():
        hift.f0_predictor.classifier.weight.mul_(3000.0)
        hift.f0_predictor.classifier.bias.zero_()
    return _make_model(hift, window_len)


def _stream_chunks(seed: int, first: int, hop: int, hops: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    total = first + hop * (hops - 1)
    mel = torch.randn((1, 80, total), generator=generator)
    bounds = [0, first] + [first + hop * i for i in range(1, hops)]
    return [mel[:, :, start:end] for start, end in zip(bounds, bounds[1:])]


def _run(model, batched: bool) -> list[list[torch.Tensor]]:
    """Step every live stream once per step; the last hop of each is final."""
    chunks = [_stream_chunks(seed, *spec[:3]) for seed, spec in enumerate(STREAMS)]
    states: list = [None] * len(STREAMS)
    emitted: list[list[torch.Tensor]] = [[] for _ in STREAMS]
    hop_of = [0] * len(STREAMS)
    for step in range(max(spec[3] + spec[2] for spec in STREAMS)):
        for finalize in (False, True):
            live = [
                index
                for index, (_, _, hops, start) in enumerate(STREAMS)
                if start <= step and hop_of[index] < hops and (hop_of[index] == hops - 1) == finalize
            ]
            if not live:
                continue
            feats = [chunks[index][hop_of[index]] for index in live]
            cached = [states[index] for index in live]
            if batched:
                values = model._stream_hift(feats, cached, finalize=finalize)
            else:
                values = [
                    model._stream_hift_from_feat(feat, cache_state=state, finalize=finalize)
                    for feat, state in zip(feats, cached)
                ]
            for index, (speech, state) in zip(live, values):
                emitted[index].append(speech)
                states[index] = state
                hop_of[index] += 1
    return emitted


@pytest.mark.parametrize("window_len", [64, 16])
@torch.inference_mode()
def test_batched_streaming_hift_matches_per_request(monkeypatch, window_len):
    monkeypatch.setattr(cosyvoice3_code2wav, "cosyvoice3_packed_streaming_enabled", lambda: True)
    model = _voiced_model(window_len)
    expected = _run(model, batched=False)
    monkeypatch.setattr(cosyvoice3_code2wav, "cosyvoice3_hift_batch_enabled", lambda: True)
    actual = _run(model, batched=True)
    for stream_expected, stream_actual in zip(expected, actual):
        assert len(stream_expected) == len(stream_actual)
        for want, got in zip(stream_expected, stream_actual):
            assert got.shape == want.shape
            torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL)


@torch.inference_mode()
def test_final_window_runs_alone():
    model = _voiced_model(64)
    window, _ = model._hift_stream_window(torch.randn(1, 80, 60), None, finalize=True)
    with pytest.raises(ValueError, match="exact width"):
        model._get_hift_batcher().run([window, window], finalize=True)
