# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Minimal regression coverage for reference-code caching and single-flight."""

import threading
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.fish_omni import reference_audio
from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _encode(cache, wav, sample_rate=44100, model_path="model-a", **kwargs):
    return reference_audio.encode_reference_audio_codes_cached(
        model_path, wav, sample_rate, device=torch.device("cpu"), cache=cache, **kwargs
    )


@pytest.fixture
def encoding_calls(monkeypatch):
    calls = []

    def encode(model_path, wav, sample_rate, *, device):
        calls.append((model_path, wav.copy(), sample_rate))
        return torch.full((2, 10), len(calls), dtype=torch.long)

    monkeypatch.setattr(reference_audio, "encode_reference_audio_codes", encode)
    return calls


def test_inline_content_is_reused_across_tensor_and_array_inputs(encoding_calls):
    cache = SpeakerEmbeddingCache(max_bytes=1024)
    wav = np.linspace(-0.5, 0.5, 4096, dtype=np.float32)
    first = _encode(cache, wav)
    assert torch.equal(_encode(cache, torch.from_numpy(wav.copy())), first)
    assert torch.equal(_encode(cache, np.repeat(wav, 2)[::2]), first)
    assert torch.equal(_encode(cache, torch.from_numpy(wav).double()), first)
    assert len(encoding_calls) == 1
    model_path, samples, rate = encoding_calls[0]
    assert model_path == "model-a" and rate == 44100
    np.testing.assert_array_equal(samples, wav)
    assert samples.dtype == np.float32 and samples.flags.c_contiguous
    assert torch.equal(first, torch.ones((2, 10), dtype=torch.long))
    assert first.dtype == torch.int64 and first.device.type == "cpu"
    assert cache.memory_bytes() == first.numel() * first.element_size()


@pytest.mark.parametrize("change", ["content", "sample_rate", "shape", "model", "revision"])
def test_different_encoder_inputs_do_not_alias(encoding_calls, change):
    cache = SpeakerEmbeddingCache(max_bytes=1024)
    wav = np.linspace(-0.5, 0.5, 4096, dtype=np.float32)
    _encode(cache, wav, revision="r1")
    kwargs = {"revision": "r1"}
    other = wav.copy()
    if change == "content":
        other[123] += 0.25
    elif change == "sample_rate":
        kwargs["sample_rate"] = 16000
    elif change == "shape":
        other = other.reshape(2, -1)
    elif change == "model":
        kwargs["model_path"] = "model-b"
    else:
        kwargs["revision"] = "r2"
    _encode(cache, other, **kwargs)
    assert len(encoding_calls) == 2


def test_uploaded_voice_invalidation_and_generations(encoding_calls):
    cache = SpeakerEmbeddingCache(max_bytes=2048)
    wav = np.zeros(4096, dtype=np.float32)
    _encode(cache, wav, voice_name="Alice", voice_created_at=1)
    _encode(cache, wav.copy(), voice_name="alice", voice_created_at=1)
    assert len(encoding_calls) == 1
    _encode(cache, wav, voice_name="alice", voice_created_at=2)
    assert len(encoding_calls) == 2
    # Even if the name/generation is reused, different audio must not hit.
    _encode(cache, wav + 0.1, voice_name="alice", voice_created_at=2)
    _encode(cache, wav)
    assert cache.clear("ALICE") == 3
    _encode(cache, wav)
    assert len(encoding_calls) == 4  # Inline entry survives voice invalidation.
    _encode(cache, wav, voice_name="alice", voice_created_at=2)
    assert len(encoding_calls) == 5
    _encode(cache, wav, voice_name="reference_audio")
    assert len(encoding_calls) == 6


@pytest.fixture
def wait_for_followers(monkeypatch):
    """Observe actual flight joins, without sleeps or thread-scheduling guesses."""

    def install(count):
        joined = threading.Barrier(count + 1)

        class ObservedFuture(Future):
            def result(self, timeout=None):
                joined.wait(timeout=5)
                return super().result(timeout=5)

        monkeypatch.setattr(reference_audio, "Future", ObservedFuture)
        return joined

    return install


@pytest.mark.parametrize("max_bytes", [0, 1024])
def test_concurrent_misses_encode_once(monkeypatch, wait_for_followers, max_bytes):
    cache = SpeakerEmbeddingCache(max_bytes=max_bytes)
    wav = np.zeros(4096, dtype=np.float32)
    entered, release = threading.Event(), threading.Event()
    joined = wait_for_followers(1)
    calls = []

    def encode(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return torch.full((2, 10), 7, dtype=torch.long)

    monkeypatch.setattr(reference_audio, "encode_reference_audio_codes", encode)
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(_encode, cache, wav)
        try:
            assert entered.wait(5)
            follower = pool.submit(_encode, cache, wav.copy())
            joined.wait(timeout=5)
            assert len(calls) == 1
        finally:
            release.set()
        results = [f.result(timeout=5) for f in (owner, follower)]
    assert all(torch.equal(result, torch.full((2, 10), 7, dtype=torch.long)) for result in results)
    assert len(calls) == 1
    _encode(cache, wav)
    # In-flight sharing works even when the result cannot be retained.
    assert len(calls) == (1 if max_bytes >= 160 else 2)


def test_concurrent_failure_reaches_waiters_and_can_retry(monkeypatch, wait_for_followers):
    cache = SpeakerEmbeddingCache(max_bytes=1024)
    wav = np.zeros(4096, dtype=np.float32)
    entered, release = threading.Event(), threading.Event()
    joined = wait_for_followers(1)
    calls = []

    def encode(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
            raise ValueError("encoder failed")
        return torch.zeros((2, 10), dtype=torch.long)

    monkeypatch.setattr(reference_audio, "encode_reference_audio_codes", encode)
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(_encode, cache, wav)
        try:
            assert entered.wait(5)
            follower = pool.submit(_encode, cache, wav)
            joined.wait(timeout=5)
        finally:
            release.set()
        for future in (owner, follower):
            with pytest.raises(ValueError, match="encoder failed"):
                future.result(timeout=5)
    assert len(calls) == 1 and cache.memory_bytes() == 0
    _encode(cache, wav)
    _encode(cache, wav)
    assert len(calls) == 2


@pytest.mark.parametrize("difference", ["voice", "cache"])
def test_distinct_references_encode_independently(monkeypatch, difference):
    cache = SpeakerEmbeddingCache(max_bytes=1024)
    other_cache = SpeakerEmbeddingCache(max_bytes=1024) if difference == "cache" else cache
    other_voice = "Alice" if difference == "cache" else "Bob"
    wav = np.zeros(4096, dtype=np.float32)
    entered = threading.Barrier(3)
    release = threading.Event()

    def encode(*args, **kwargs):
        entered.wait(timeout=5)
        assert release.wait(5)
        return torch.zeros((2, 10), dtype=torch.long)

    monkeypatch.setattr(reference_audio, "encode_reference_audio_codes", encode)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_encode, cache, wav, voice_name="Alice"),
            pool.submit(_encode, other_cache, wav, voice_name=other_voice),
        ]
        try:
            # Both encoders must enter before either completes.
            entered.wait(timeout=5)
        finally:
            release.set()
        for future in futures:
            future.result(timeout=5)
