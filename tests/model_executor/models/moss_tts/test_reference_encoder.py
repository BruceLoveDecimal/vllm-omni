# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MossReferenceEncoder: content-addressed caching,
single-flight, and micro-batched encoding."""

import asyncio
import threading
import time

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.reference_encoder import (
    MossReferenceEncoder,
    _RefEncodeBatcher,
)
from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

_SR = 24000
_N_VQ = 4


class _FakeProcessor:
    """Deterministic stand-in for the MOSS AutoProcessor."""

    def __init__(self, *, n_vq=_N_VQ, sampling_rate=_SR, delay_s=0.0, fail_if_batched=False):
        self.model_config = type("_Cfg", (), {"n_vq": n_vq, "sampling_rate": sampling_rate})()
        self.attempt_sizes: list[int] = []
        self.total_items = 0
        self._delay_s = delay_s
        self._fail_if_batched = fail_if_batched
        self._fail_values: set[int] = set()
        self._lock = threading.Lock()

    def set_fail_values(self, values) -> None:
        self._fail_values = set(values)

    def encode_audios_from_wav(self, wav_list, sampling_rate, n_vq=None):
        if self._delay_s:
            time.sleep(self._delay_s)
        with self._lock:
            self.attempt_sizes.append(len(wav_list))
        if self._fail_if_batched and len(wav_list) > 1:
            raise RuntimeError("batched forward not supported")
        nq = n_vq or self.model_config.n_vq
        out = []
        for wav in wav_list:
            cid = round(float(wav.reshape(-1)[0].item()))
            if cid in self._fail_values:
                raise RuntimeError(f"encode failed for content {cid}")
            out.append(torch.full((3, nq), cid, dtype=torch.long))
        with self._lock:
            self.total_items += len(wav_list)
        return out


class _FakeAudio:
    """Fake resolve + content-hash side, keyed by request-side ref string."""

    def __init__(self, *, resolve_delay_s=0.0):
        self._wav: dict[str, tuple] = {}
        self._artifact: dict[str, str | None] = {}
        self.resolve_calls: list[str] = []
        self._resolve_delay_s = resolve_delay_s
        self._lock = threading.Lock()

    def register(self, ref_str, content_id, *, artifact=..., sr=_SR, length=100, wav=...):
        wav_list = [float(content_id)] * length if wav is ... else wav
        self._wav[ref_str] = (wav_list, sr)
        # Default: content hash keyed by content_id (same content -> same key).
        self._artifact[ref_str] = f"c{content_id}" if artifact is ... else artifact

    async def resolve(self, ref_str):
        with self._lock:
            self.resolve_calls.append(ref_str)
        if self._resolve_delay_s:
            await asyncio.sleep(self._resolve_delay_s)
        return self._wav[ref_str]

    def artifact_key(self, ref_str):
        return self._artifact.get(ref_str)


@pytest.fixture
async def make_encoder():
    """Factory yielding encoders; tears each one's drainer down after the test."""
    created: list[MossReferenceEncoder] = []

    def _make(processor, *, cache=None, window_ms=10.0, max_batch=8, variant="local"):
        enc = MossReferenceEncoder(
            processor,
            variant=variant,
            n_vq=_N_VQ,
            sr_target=_SR,
            speaker_cache=cache or SpeakerEmbeddingCache(max_bytes=8 * 1024**2),
            window_ms=window_ms,
            max_batch=max_batch,
        )
        created.append(enc)
        return enc

    yield _make
    for enc in created:
        await enc.aclose()


async def _encode(enc, audio, ref, **kw):
    return await enc.encode(
        ref,
        resolve_ref_audio=audio.resolve,
        get_artifact_key=audio.artifact_key,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Group A — content-addressed cache (R1)                                       #
# --------------------------------------------------------------------------- #


async def test_same_reference_encodes_once(make_encoder):
    proc, audio = _FakeProcessor(), _FakeAudio()
    audio.register("a", 7)
    enc = make_encoder(proc)

    r1 = await _encode(enc, audio, "a")
    r2 = await _encode(enc, audio, "a")

    assert proc.total_items == 1  # second call is a cache hit
    assert r1.dtype == torch.int64 and r2.dtype == torch.int64
    assert int(r1[0, 0]) == 7 and int(r2[0, 0]) == 7


async def test_returned_tensor_does_not_alias_cache(make_encoder):
    proc, audio = _FakeProcessor(), _FakeAudio()
    audio.register("a", 7)
    enc = make_encoder(proc)

    r1 = await _encode(enc, audio, "a")
    r1[:] = 999  # caller mutates its copy
    r2 = await _encode(enc, audio, "a")

    assert int(r2[0, 0]) == 7  # cache untouched
    assert proc.total_items == 1


async def test_cached_dtype_is_int32(make_encoder):
    cache = SpeakerEmbeddingCache(max_bytes=8 * 1024**2)
    proc, audio = _FakeProcessor(), _FakeAudio()
    audio.register("a", 7)
    enc = make_encoder(proc, cache=cache)

    out = await _encode(enc, audio, "a")

    key = cache.make_cache_key("ref:c7", model_type=f"moss_tts_local_nq{_N_VQ}")
    stored = cache.get(key)
    assert stored is not None
    assert stored["codes"].dtype == torch.int32  # compact on-disk
    assert out.dtype == torch.int64  # int64 to the caller


async def test_same_content_via_different_refs_shares_cache(make_encoder):
    proc, audio = _FakeProcessor(), _FakeAudio()
    # base64 vs URL for the *same* clip resolve to the same content hash.
    audio.register("data:base64,xxx", 7, artifact="c7")
    audio.register("http://x/clip.wav", 7, artifact="c7")
    enc = make_encoder(proc)

    await _encode(enc, audio, "data:base64,xxx")
    await _encode(enc, audio, "http://x/clip.wav")

    assert proc.total_items == 1  # encoded once, shared by content hash


async def test_named_voice_created_at_isolation(make_encoder):
    proc, audio = _FakeProcessor(), _FakeAudio()
    audio.register("ref", 7)
    enc = make_encoder(proc)

    await _encode(enc, audio, "ref", voice_name="alice", voice_created_at=1)
    await _encode(enc, audio, "ref", voice_name="alice", voice_created_at=2)
    assert proc.total_items == 2  # re-upload (new created_at) re-encodes

    await _encode(enc, audio, "ref", voice_name="alice", voice_created_at=1)
    assert proc.total_items == 2  # original generation still cached


async def test_unregistered_default_voice_uses_ref_content_key(make_encoder):
    proc, audio = _FakeProcessor(), _FakeAudio()
    audio.register("first", 7, artifact="c7")
    audio.register("second", 8, artifact="c8")
    enc = make_encoder(proc)

    first = await _encode(enc, audio, "first", voice_name="default", voice_created_at=0)
    second = await _encode(enc, audio, "second", voice_name="default", voice_created_at=0)
    first_again = await _encode(enc, audio, "first", voice_name="default", voice_created_at=0)

    assert int(first[0, 0]) == 7
    assert int(second[0, 0]) == 8
    assert int(first_again[0, 0]) == 7
    assert proc.total_items == 2  # no collision on the placeholder voice name


async def test_artifact_key_unavailable_falls_back_to_ref_hash(make_encoder):
    proc, audio = _FakeProcessor(), _FakeAudio()
    # artifact=None simulates a cold/disabled resolve cache.
    audio.register("ref", 7, artifact=None)
    enc = make_encoder(proc)

    await _encode(enc, audio, "ref")
    await _encode(enc, audio, "ref")

    # Falls back to a stable sha1(ref_str) key, so the second call still hits.
    assert proc.total_items == 1


# --------------------------------------------------------------------------- #
# Group B — single-flight (R2)                                                 #
# --------------------------------------------------------------------------- #


async def test_concurrent_same_ref_encodes_once(make_encoder):
    proc, audio = _FakeProcessor(delay_s=0.05), _FakeAudio()
    audio.register("a", 7)
    enc = make_encoder(proc)

    results = await asyncio.gather(*[_encode(enc, audio, "a") for _ in range(10)])

    assert proc.total_items == 1  # single-flight collapsed 10 -> 1 encode
    assert audio.resolve_calls == ["a"]  # and one resolve/download
    assert all(int(r[0, 0]) == 7 for r in results)


async def test_concurrent_results_are_isolated(make_encoder):
    proc, audio = _FakeProcessor(delay_s=0.02), _FakeAudio()
    audio.register("a", 7)
    enc = make_encoder(proc)

    r1, r2 = await asyncio.gather(_encode(enc, audio, "a"), _encode(enc, audio, "a"))
    r1[:] = 111
    assert int(r2[0, 0]) == 7  # separate storage per waiter


async def test_flight_exception_propagates_and_is_not_cached(make_encoder):
    proc, audio = _FakeProcessor(delay_s=0.02), _FakeAudio()
    audio.register("a", 9)
    proc.set_fail_values({9})
    enc = make_encoder(proc)

    results = await asyncio.gather(*[_encode(enc, audio, "a") for _ in range(6)], return_exceptions=True)
    assert all(isinstance(r, Exception) for r in results)  # every waiter sees it

    proc.set_fail_values(set())  # failure was not cached: a retry now succeeds
    ok = await _encode(enc, audio, "a")
    assert int(ok[0, 0]) == 9


async def test_initiator_cancel_does_not_break_other_waiter(make_encoder):
    proc, audio = _FakeProcessor(delay_s=0.1), _FakeAudio()
    audio.register("a", 7)
    enc = make_encoder(proc)

    a = asyncio.create_task(_encode(enc, audio, "a"))
    await asyncio.sleep(0.02)  # let A create the flight
    b = asyncio.create_task(_encode(enc, audio, "a"))
    await asyncio.sleep(0.02)  # let B join the same flight
    a.cancel()

    res_b = await b
    assert int(res_b[0, 0]) == 7  # B still completes off the shared flight
    assert proc.total_items == 1
    with pytest.raises(asyncio.CancelledError):
        await a


# --------------------------------------------------------------------------- #
# Group C — micro-batched encoding (R3)                                        #
# --------------------------------------------------------------------------- #


async def test_distinct_refs_coalesce_into_one_batch(make_encoder):
    proc, audio = _FakeProcessor(), _FakeAudio()
    for i, ref in enumerate(("a", "b", "c")):
        audio.register(ref, 10 + i)
    enc = make_encoder(proc, window_ms=100.0, max_batch=8)

    results = await asyncio.gather(*[_encode(enc, audio, r) for r in ("a", "b", "c")])

    assert proc.attempt_sizes == [3]  # one batched forward of all three
    assert sorted(int(r[0, 0]) for r in results) == [10, 11, 12]


async def test_max_batch_splits_work():
    # Drive the batcher directly: with everything already queued, max_batch=2
    # must split 4 submissions into batches of at most 2.
    seen: list[int] = []

    def encode_batch(payload):
        seen.append(len(payload))
        return [torch.zeros(1) for _ in payload]

    batcher = _RefEncodeBatcher(encode_batch, window_ms=5.0, max_batch=2)
    try:
        await asyncio.gather(*[batcher.submit([1.0], _SR) for _ in range(4)])
    finally:
        await batcher.aclose()

    assert sum(seen) == 4
    assert max(seen) == 2 and all(s <= 2 for s in seen)


async def test_single_submission_still_runs(make_encoder):
    proc, audio = _FakeProcessor(), _FakeAudio()
    audio.register("a", 7)
    enc = make_encoder(proc, window_ms=20.0)

    out = await _encode(enc, audio, "a")  # not starved waiting for a full batch
    assert int(out[0, 0]) == 7
    assert proc.attempt_sizes == [1]


async def test_window_zero_runs_immediately():
    seen: list[int] = []

    def encode_batch(payload):
        seen.append(len(payload))
        return [torch.zeros(1) for _ in payload]

    batcher = _RefEncodeBatcher(encode_batch, window_ms=0.0, max_batch=8)
    try:
        out = await batcher.submit([1.0], _SR)
    finally:
        await batcher.aclose()
    assert out is not None
    assert seen == [1]


def test_prep_failure_is_isolated(make_encoder_sync):
    enc = make_encoder_sync()
    good = [1.0] * 100
    bad = None  # torch.tensor(None) raises inside prep

    results = enc._encode_batch_sync([(good, _SR), (bad, _SR), (good, _SR)])

    assert isinstance(results[0], torch.Tensor)
    assert isinstance(results[1], Exception)  # only the bad clip fails
    assert isinstance(results[2], torch.Tensor)


def test_batch_forward_failure_falls_back_per_item(make_encoder_sync):
    proc = _FakeProcessor(fail_if_batched=True)
    enc = make_encoder_sync(proc)

    results = enc._encode_batch_sync([([float(i)] * 100, _SR) for i in (1, 2, 3)])

    assert all(isinstance(r, torch.Tensor) for r in results)
    assert [int(r[0, 0]) for r in results] == [1, 2, 3]
    assert any(s > 1 for s in proc.attempt_sizes)  # a batched attempt happened
    assert proc.attempt_sizes.count(1) == 3  # then per-item fallback


@pytest.fixture
def make_encoder_sync():
    """Encoder factory for the synchronous ``_encode_batch_sync`` tests."""

    def _make(processor=None):
        return MossReferenceEncoder(
            processor or _FakeProcessor(),
            variant="local",
            n_vq=_N_VQ,
            sr_target=_SR,
            speaker_cache=SpeakerEmbeddingCache(max_bytes=8 * 1024**2),
        )

    return _make


# --------------------------------------------------------------------------- #
# Group D — integration with the real SpeakerEmbeddingCache                    #
# --------------------------------------------------------------------------- #


async def test_model_type_namespaces_cache(make_encoder):
    # Two variants sharing one cache must not collide on the same ref.
    cache = SpeakerEmbeddingCache(max_bytes=8 * 1024**2)
    proc, audio = _FakeProcessor(), _FakeAudio()
    audio.register("a", 7)
    enc_tts = make_encoder(proc, cache=cache, variant="tts")
    enc_ttsd = make_encoder(proc, cache=cache, variant="ttsd")

    await _encode(enc_tts, audio, "a")
    await _encode(enc_ttsd, audio, "a")

    assert proc.total_items == 2  # distinct model_type keys -> no cross-hit
    assert cache.stats()["entries"] == 2
