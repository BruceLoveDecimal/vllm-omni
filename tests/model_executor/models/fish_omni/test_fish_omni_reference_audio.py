# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded reference caching and exact equivalence through the real CPU DAC."""

import threading
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.fish_omni import reference_audio
from vllm_omni.model_executor.models.fish_omni.prompt import Reference, build_prompt_frames
from vllm_omni.model_executor.models.fish_speech import dac_encoder
from vllm_omni.utils import speaker_cache
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


@pytest.mark.parametrize("max_bytes", [0, 159, 160])
def test_cache_budget_and_eviction(encoding_calls, max_bytes):
    # Each fake result is 2 * 10 * 8 = 160 bytes.
    cache = SpeakerEmbeddingCache(max_bytes=max_bytes)
    wav = np.zeros(4096, dtype=np.float32)
    _encode(cache, wav)
    _encode(cache, wav + 0.1)
    _encode(cache, wav)
    assert len(encoding_calls) == 3
    assert cache.memory_bytes() <= max_bytes


def test_failed_encoding_is_not_cached(monkeypatch):
    cache = SpeakerEmbeddingCache(max_bytes=1024)
    calls = 0

    def encode(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("invalid reference")
        return torch.zeros((2, 10), dtype=torch.long)

    monkeypatch.setattr(reference_audio, "encode_reference_audio_codes", encode)
    wav = np.zeros(4096, dtype=np.float32)
    with pytest.raises(ValueError, match="invalid reference"):
        _encode(cache, wav)
    assert cache.memory_bytes() == 0
    _encode(cache, wav)
    _encode(cache, wav)
    assert calls == 2


@pytest.fixture
def wait_for_followers(monkeypatch):
    """Observe actual flight joins, without sleeps or thread-scheduling guesses."""

    def install(count):
        joined = threading.Barrier(count + 1)

        class ObservedFuture(Future):
            def result(self, timeout=None):
                joined.wait(timeout=5)
                return super().result(timeout=5)

        monkeypatch.setattr(speaker_cache, "Future", ObservedFuture)
        return joined

    return install


@pytest.mark.parametrize("max_bytes", [0, 159, 1024])
def test_concurrent_misses_encode_once(monkeypatch, wait_for_followers, max_bytes):
    cache = SpeakerEmbeddingCache(max_bytes=max_bytes)
    wav = np.zeros(4096, dtype=np.float32)
    entered, release = threading.Event(), threading.Event()
    joined = wait_for_followers(3)
    calls = []

    def encode(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return torch.full((2, 10), 7, dtype=torch.long)

    monkeypatch.setattr(reference_audio, "encode_reference_audio_codes", encode)
    with ThreadPoolExecutor(max_workers=4) as pool:
        owner = pool.submit(_encode, cache, wav)
        try:
            assert entered.wait(5)
            followers = [pool.submit(_encode, cache, wav.copy()) for _ in range(3)]
            joined.wait(timeout=5)
            assert len(calls) == 1
        finally:
            release.set()
        results = [f.result(timeout=5) for f in [owner, *followers]]
    assert all(torch.equal(result, torch.full((2, 10), 7, dtype=torch.long)) for result in results)
    assert len(calls) == 1
    _encode(cache, wav)
    # In-flight sharing works even when the result cannot be retained.
    assert len(calls) == (1 if max_bytes >= 160 else 2)


class _EncoderInterrupted(BaseException):
    pass


@pytest.mark.parametrize("error_type", [ValueError, _EncoderInterrupted])
def test_concurrent_failure_reaches_waiters_and_can_retry(monkeypatch, wait_for_followers, error_type):
    cache = SpeakerEmbeddingCache(max_bytes=1024)
    wav = np.zeros(4096, dtype=np.float32)
    entered, release = threading.Event(), threading.Event()
    joined = wait_for_followers(2)
    calls = []

    def encode(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
            raise error_type("encoder failed")
        return torch.zeros((2, 10), dtype=torch.long)

    monkeypatch.setattr(reference_audio, "encode_reference_audio_codes", encode)
    with ThreadPoolExecutor(max_workers=3) as pool:
        owner = pool.submit(_encode, cache, wav)
        try:
            assert entered.wait(5)
            followers = [pool.submit(_encode, cache, wav) for _ in range(2)]
            joined.wait(timeout=5)
        finally:
            release.set()
        for future in [owner, *followers]:
            with pytest.raises(error_type, match="encoder failed"):
                future.result(timeout=5)
    assert len(calls) == 1 and cache.memory_bytes() == 0
    _encode(cache, wav)
    _encode(cache, wav)
    assert len(calls) == 2


@pytest.mark.parametrize("difference", ["voice", "voice_and_audio", "inline_audio", "generation", "namespace"])
def test_concurrent_conditions_have_independent_flights(monkeypatch, wait_for_followers, difference):
    cache = SpeakerEmbeddingCache(max_bytes=2048)
    wav_a = np.zeros(4096, dtype=np.float32)
    wav_b = wav_a.copy()
    kwargs_a = {"voice_name": "Alice", "voice_created_at": 1}
    kwargs_b = {"voice_name": "Bob", "voice_created_at": 1}
    if difference == "voice_and_audio":
        wav_b += 0.25
    elif difference == "inline_audio":
        kwargs_a, kwargs_b = {}, {}
        wav_b += 0.25
    elif difference == "generation":
        kwargs_b = {"voice_name": "Alice", "voice_created_at": 2}
    elif difference == "namespace":
        kwargs_a, kwargs_b = {"voice_name": "reference_audio"}, {}
    entered = threading.Barrier(3)
    release = threading.Event()
    joined = wait_for_followers(2)
    calls = []
    lock = threading.Lock()

    def encode(*args, **kwargs):
        with lock:
            calls.append(1)
            value = len(calls)
        entered.wait(timeout=5)
        assert release.wait(5)
        return torch.full((2, 10), value, dtype=torch.long)

    monkeypatch.setattr(reference_audio, "encode_reference_audio_codes", encode)
    with ThreadPoolExecutor(max_workers=4) as pool:
        owner_a = pool.submit(_encode, cache, wav_a, **kwargs_a)
        owner_b = pool.submit(_encode, cache, wav_b, **kwargs_b)
        try:
            # Both encoders must start before either completes: no global
            # compute lock, and no shared flight across voice identities.
            entered.wait(timeout=5)
            follower_b = pool.submit(_encode, cache, wav_b.copy(), **kwargs_b)
            follower_a = pool.submit(_encode, cache, wav_a.copy(), **kwargs_a)
            joined.wait(timeout=5)
        finally:
            release.set()
        a, b = owner_a.result(timeout=5), owner_b.result(timeout=5)
        assert not torch.equal(a, b)
        assert torch.equal(a, follower_a.result(timeout=5))
        assert torch.equal(b, follower_b.result(timeout=5))
    assert torch.equal(a, _encode(cache, wav_a, **kwargs_a))
    assert torch.equal(b, _encode(cache, wav_b, **kwargs_b))
    assert len(calls) == 2


@pytest.mark.parametrize("clear_all", [False, True])
def test_clear_during_encoding_cannot_publish_stale_codes(monkeypatch, wait_for_followers, clear_all):
    cache = SpeakerEmbeddingCache(max_bytes=2048)
    wav = np.zeros(4096, dtype=np.float32)
    entered, release = threading.Event(), threading.Event()
    joined = wait_for_followers(1)
    calls = []

    def encode(*args, **kwargs):
        calls.append(1)
        value = len(calls)
        if value == 1:
            entered.set()
            assert release.wait(5)
        return torch.full((2, 10), value, dtype=torch.long)

    monkeypatch.setattr(reference_audio, "encode_reference_audio_codes", encode)
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(_encode, cache, wav, voice_name="Alice")
        try:
            assert entered.wait(5)
            follower = pool.submit(_encode, cache, wav, voice_name="alice")
            joined.wait(timeout=5)
            bob = _encode(cache, wav, voice_name="Bob")
            cache.clear(None if clear_all else "ALICE")
            fresh = _encode(cache, wav, voice_name="alice")
            assert torch.equal(fresh, torch.full((2, 10), 3, dtype=torch.long))
            if not clear_all:
                assert torch.equal(bob, _encode(cache, wav, voice_name="Bob"))
        finally:
            release.set()
        assert torch.equal(owner.result(timeout=5), torch.full((2, 10), 1, dtype=torch.long))
        assert torch.equal(follower.result(timeout=5), owner.result(timeout=5))
    assert torch.equal(_encode(cache, wav, voice_name="Alice"), fresh)
    assert len(calls) == 3


@pytest.mark.parametrize("old_fails", [False, True])
def test_detached_flight_cannot_remove_a_newer_flight(monkeypatch, wait_for_followers, old_fails):
    cache = SpeakerEmbeddingCache(max_bytes=1024)
    wav = np.zeros(4096, dtype=np.float32)
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    joined = wait_for_followers(1)
    calls = []
    lock = threading.Lock()

    def encode(*args, **kwargs):
        with lock:
            index = len(calls)
            calls.append(1)
        assert index < 2, "new caller lost the replacement flight"
        entered[index].set()
        assert release[index].wait(5)
        if index == 0 and old_fails:
            raise ValueError("old encoding failed")
        return torch.full((2, 10), index + 1, dtype=torch.long)

    monkeypatch.setattr(reference_audio, "encode_reference_audio_codes", encode)
    with ThreadPoolExecutor(max_workers=3) as pool:
        old = pool.submit(_encode, cache, wav, voice_name="Alice")
        try:
            assert entered[0].wait(5)
            cache.clear("alice")
            new = pool.submit(_encode, cache, wav, voice_name="Alice")
            assert entered[1].wait(5)
            release[0].set()
            if old_fails:
                with pytest.raises(ValueError, match="old encoding failed"):
                    old.result(timeout=5)
            else:
                old.result(timeout=5)
            follower = pool.submit(_encode, cache, wav, voice_name="Alice")
            joined.wait(timeout=5)
            assert len(calls) == 2
        finally:
            for event in release:
                event.set()
        assert torch.equal(new.result(timeout=5), follower.result(timeout=5))
    assert torch.equal(_encode(cache, wav, voice_name="Alice"), torch.full((2, 10), 2, dtype=torch.long))


@pytest.fixture(scope="module")
def random_codec():
    pytest.importorskip("torchaudio")
    from vllm_omni.model_executor.models.fish_speech.dac_utils import build_dac_codec

    # Exercise the real architecture, resampler and quantizer, without a model
    # download. This is numerical regression coverage, not a speech-quality test.
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        codec = build_dac_codec()
    codec.decoder = None
    try:
        yield codec.eval()
    finally:
        torch.set_num_threads(old_threads)


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


@pytest.mark.parametrize(
    ("sample_rate", "shape"), [(44100, (4097,)), (16000, (1601,)), (44100, (2, 4097)), (24000, (2401, 2))]
)
def test_cached_codes_and_prompt_equal_uncached_dac(monkeypatch, random_codec, sample_rate, shape):
    monkeypatch.setattr(dac_encoder, "_load_dac_codec", lambda *args, **kwargs: random_codec)
    wav = np.random.default_rng(7).uniform(-0.5, 0.5, shape).astype(np.float32)
    original = wav.copy()
    expected = dac_encoder.encode_reference_audio_codes("unused", wav, sample_rate, device="cpu")
    calls = 0

    def encode(*args, **kwargs):
        nonlocal calls
        calls += 1
        return dac_encoder.encode_reference_audio_codes(*args, **kwargs)

    monkeypatch.setattr(reference_audio, "encode_reference_audio_codes", encode)
    cache = SpeakerEmbeddingCache(max_bytes=1024**2)
    miss = _encode(cache, wav, sample_rate, model_path="unused")
    hit = _encode(cache, torch.from_numpy(wav.copy()), sample_rate, model_path="unused")
    assert calls == 1
    assert torch.equal(expected, miss)
    assert torch.equal(expected, hit)
    np.testing.assert_array_equal(wav, original)

    def prompt(codes):
        return build_prompt_frames(
            _Tokenizer(), "Hello.", [Reference("Reference.", codes)], semantic_begin_id=248099, num_codebooks=10
        )

    assert torch.equal(prompt(expected), prompt(hit))
