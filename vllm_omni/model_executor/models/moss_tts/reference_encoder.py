"""Reference-audio encoding + speaker cache for the MOSS-TTS-family talker.

This lives in the model package (not the shared serving layer) so all
MOSS-specific reference handling stays with the model — mirroring how Fish
Speech (``dac_encoder.encode_reference_audio_codes``), CosyVoice3, and
Qwen3-TTS keep their reference/speaker extraction next to the model rather than
in ``serving_speech.py``. The serving layer constructs one
:class:`MossReferenceEncoder` per server (lazily, alongside the upstream MOSS
processor) and calls :meth:`MossReferenceEncoder.encode` with its generic
helpers (the audio resolver, the content-hash lookup, and the process-wide
speaker cache), which adds content-addressed caching, single-flight, and
micro-batched encoding on top.

Kept import-light (only ``asyncio`` / ``hashlib`` / ``torch`` plus the logger)
so importing it from the API-server process does not pull the talker/codec.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Coalescing defaults. Kept as module constants (not env/CLI) to match the
# ``_REF_AUDIO_RESOLVE_CACHE_MAX_*`` convention in serving_speech.py.
_REF_ENCODE_BATCH_WINDOW_MS = 10.0
_REF_ENCODE_MAX_BATCH = 8

_INT32_MAX = 2**31


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def _prep_wav_sync(wav_list: list, sr: int, sr_target: int) -> torch.Tensor:
    """Tensor-ise + resample one clip to ``sr_target`` (the blocking prep)."""
    wav = torch.tensor(wav_list, dtype=torch.float32)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if sr != sr_target:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, sr_target)
    return wav


# --- codes storage convention ----------------------------------------------
# The speaker cache always stores the compact form (int32, falling back to
# int64 only when values overflow); every caller of encode() receives its own
# independent int64 copy, so mutating a returned tensor never aliases the
# tensor owned by the cache.


def _to_compact_codes(codes: torch.Tensor) -> torch.Tensor:
    """Downcast RVQ codes to int32 for compact caching."""
    codes = codes.detach().cpu().contiguous()
    if codes.numel() == 0:
        return codes.to(torch.int32)
    hi = int(codes.max().item())
    lo = int(codes.min().item())
    if -_INT32_MAX <= lo and hi < _INT32_MAX:
        return codes.to(torch.int32)
    logger.warning("MOSS ref codes out of int32 range (min=%d max=%d); caching as int64", lo, hi)
    return codes.to(torch.int64)


def _clone_for_caller(codes: torch.Tensor) -> torch.Tensor:
    """Return an independent int64 copy for the caller."""
    return codes.to(torch.int64, copy=True)


@dataclass(frozen=True)
class _RefIdentity:
    """Normalized identity of one reference-audio request.

    Owns voice-name normalization and the flight-key derivation so the hot
    path and the flight body in :class:`MossReferenceEncoder` cannot drift
    apart.
    """

    ref_str: str
    voice_name: str | None
    created_at: int
    flight_key: str

    @classmethod
    def build(cls, ref_str: str, voice_name: str | None, voice_created_at: int) -> _RefIdentity:
        # The OpenAI speech API requires a ``voice`` field, and callers often
        # send placeholders such as "default" for ref-audio voice cloning. Only
        # registered uploaded voices have a positive created_at timestamp; other
        # names must not key the cache because the timbre comes from ref_audio.
        name = voice_name.strip() if isinstance(voice_name, str) and voice_name.strip() else None
        created_at = int(voice_created_at) if name else 0
        if name and created_at <= 0:
            name = None
            created_at = 0
        # The anonymous flight key is the request-side reference (not the
        # content hash), so concurrent requests for the same ref_str also share
        # the resolve/download, not just the encode.
        flight_key = f"voice:{name.lower()}:{created_at}" if name else "ref:" + _sha1(ref_str)
        return cls(ref_str=ref_str, voice_name=name, created_at=created_at, flight_key=flight_key)


class _EncodeJob(NamedTuple):
    """One queued reference-encode submission."""

    wav_list: list
    sr: int
    fut: asyncio.Future


class _RefEncodeBatcher:
    """Coalesce cold reference encodes into batched processor forwards."""

    def __init__(
        self,
        encode_batch_fn: Callable[[list[tuple[list, int]]], list],
        *,
        window_ms: float,
        max_batch: int,
    ):
        # encode_batch_fn: sync, takes [(wav_list, sr), ...], returns a list of
        # (codes_tensor | Exception) aligned to the input order.
        self._encode_batch_fn = encode_batch_fn
        self._window_s = max(0.0, float(window_ms) / 1000.0)
        self._max_batch = max(1, int(max_batch))
        self._queue: asyncio.Queue[_EncodeJob] | None = None
        self._drainer: asyncio.Task | None = None

    def _ensure_started(self) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._drainer is None or self._drainer.done():
            self._drainer = asyncio.create_task(self._drain_loop())

    async def submit(self, wav_list: list, sr: int) -> torch.Tensor:
        self._ensure_started()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_EncodeJob(wav_list, sr, fut))  # type: ignore[union-attr]
        return await fut

    async def _drain_loop(self) -> None:
        assert self._queue is not None
        while True:
            first = await self._queue.get()
            await self._run_batch(await self._collect_batch(first))

    async def _collect_batch(self, first: _EncodeJob) -> list[_EncodeJob]:
        """Grow ``first`` into a batch: wait up to the window, cap at max_batch."""
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        jobs = [first]
        deadline = loop.time() + self._window_s
        while len(jobs) < self._max_batch:
            if self._window_s > 0:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    jobs.append(await asyncio.wait_for(self._queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            else:
                # window=0: coalesce only what is already queued, never wait.
                try:
                    jobs.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
        return jobs

    async def _run_batch(self, jobs: list[_EncodeJob]) -> None:
        payload = [(job.wav_list, job.sr) for job in jobs]
        try:
            results = await asyncio.to_thread(self._encode_batch_fn, payload)
        except Exception as exc:  # noqa: BLE001 — propagate the batch failure to every waiter
            for job in jobs:
                if not job.fut.done():
                    job.fut.set_exception(exc)
            return
        for job, res in zip(jobs, results):
            if job.fut.done():
                continue
            if isinstance(res, BaseException):
                job.fut.set_exception(res)
            else:
                job.fut.set_result(res)

    async def aclose(self) -> None:
        if self._drainer is not None:
            self._drainer.cancel()
            try:
                await self._drainer
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._drainer = None


class MossReferenceEncoder:
    """Content-addressed, single-flight, micro-batched reference-audio encoder."""

    def __init__(
        self,
        processor: Any,
        *,
        variant: str,
        n_vq: int,
        sr_target: int,
        speaker_cache: Any,
        window_ms: float = _REF_ENCODE_BATCH_WINDOW_MS,
        max_batch: int = _REF_ENCODE_MAX_BATCH,
    ):
        self._processor = processor
        self._n_vq = int(n_vq)
        self._sr_target = int(sr_target)
        self._speaker_cache = speaker_cache
        # ``created_at`` and the audio-content name vary per request; the
        # model_type namespaces the whole family so a moss_tts server never
        # collides with another model's speaker-cache entries.
        self._model_type = f"moss_tts_{variant}_nq{int(n_vq)}"
        self._inflight: dict[str, asyncio.Task] = {}
        self._batcher = _RefEncodeBatcher(self._encode_batch_sync, window_ms=window_ms, max_batch=max_batch)

    def _cache_key(
        self,
        identity: _RefIdentity,
        get_artifact_key: Callable[[str], str | None],
        *,
        after_resolve: bool,
    ) -> tuple | None:
        """Speaker-cache key for ``identity`` (the only key-derivation site).

        A named voice always has a stable key. An anonymous ref is keyed by
        its content hash, which is only knowable once the clip has been
        resolved: before resolve (``after_resolve=False``, the hot path) we
        can key it only if a *prior* resolve already made the hash visible via
        ``get_artifact_key``, otherwise there is no stable key yet and we
        return ``None``; after resolve the hash is available, falling back to
        the request-side ref hash when the resolve cache is cold/disabled.
        """
        if identity.voice_name:
            return self._speaker_cache.make_cache_key(
                identity.voice_name, model_type=self._model_type, created_at=identity.created_at
            )
        artifact_key = get_artifact_key(identity.ref_str)
        if artifact_key:
            name = "ref:" + artifact_key
        elif after_resolve:
            name = "ref:" + _sha1(identity.ref_str)
        else:
            return None
        return self._speaker_cache.make_cache_key(name, model_type=self._model_type, created_at=0)

    async def encode(
        self,
        ref_str: str,
        *,
        resolve_ref_audio: Callable[[str], Awaitable[tuple[list, int]]],
        get_artifact_key: Callable[[str], str | None],
        voice_name: str | None = None,
        voice_created_at: int = 0,
    ) -> torch.Tensor:
        """Encode one reference clip into MOSS RVQ codes, reusing the cache."""
        identity = _RefIdentity.build(ref_str, voice_name, voice_created_at)

        # Hot path: named voices always have a stable key; an anonymous ref
        # only once a prior resolve made its content hash known.
        key = self._cache_key(identity, get_artifact_key, after_resolve=False)
        if key is not None:
            cached = self._speaker_cache.get(key)
            if cached is not None:
                return _clone_for_caller(cached["codes"])

        flight_result = await self._single_flight(identity, resolve_ref_audio, get_artifact_key)
        return _clone_for_caller(flight_result)

    async def _single_flight(
        self,
        identity: _RefIdentity,
        resolve_ref_audio: Callable[[str], Awaitable[tuple[list, int]]],
        get_artifact_key: Callable[[str], str | None],
    ) -> torch.Tensor:
        """Run one encode per uncached reference at a time.

        ``shield`` so a caller cancelling its own request does not cancel the
        shared flight (asyncio would otherwise propagate the cancel into the
        awaited task and take down every other waiter with it).
        """
        task = self._inflight.get(identity.flight_key)
        if task is not None:
            return await asyncio.shield(task)

        task = asyncio.create_task(self._resolve_and_encode(identity, resolve_ref_audio, get_artifact_key))
        self._inflight[identity.flight_key] = task
        try:
            return await asyncio.shield(task)
        finally:
            # Identity guard: only drop the slot if it still holds *our* task
            # (a later request may have replaced it after ours completed).
            if self._inflight.get(identity.flight_key) is task:
                self._inflight.pop(identity.flight_key, None)

    async def _resolve_and_encode(
        self,
        identity: _RefIdentity,
        resolve_ref_audio: Callable[[str], Awaitable[tuple[list, int]]],
        get_artifact_key: Callable[[str], str | None],
    ) -> torch.Tensor:
        """Flight body: resolve → re-check cache by content hash → batch-encode."""
        wav_list, sr = await resolve_ref_audio(identity.ref_str)

        # The content hash is available now that the clip is resolved; this also
        # catches the case where another flight populated the cache in between.
        key = self._cache_key(identity, get_artifact_key, after_resolve=True)
        cached = self._speaker_cache.get(key)
        if cached is not None:
            return cached["codes"]

        codes = await self._batcher.submit(wav_list, sr)
        compact = _to_compact_codes(codes)
        self._speaker_cache.put(key, {"codes": compact})
        logger.debug(
            "MOSS ref encode STORE key=%s shape=%s dtype=%s",
            key,
            tuple(compact.shape),
            compact.dtype,
        )
        return compact

    def _encode_batch_sync(self, payload: list[tuple[list, int]]) -> list:
        """Worker-thread body: prep each clip, then one batched forward."""
        n = len(payload)
        results: list = [None] * n
        prepared: list[torch.Tensor] = []
        prepared_idx: list[int] = []
        for i, (wav_list, sr) in enumerate(payload):
            try:
                prepared.append(_prep_wav_sync(wav_list, sr, self._sr_target))
                prepared_idx.append(i)
            except Exception as exc:  # noqa: BLE001 — isolate this clip's failure
                results[i] = exc
        if not prepared:
            return results

        try:
            codes_list = self._encode_prepared(prepared)
            for local_i, orig_i in enumerate(prepared_idx):
                results[orig_i] = codes_list[local_i]
        except Exception:  # noqa: BLE001 — batch forward failed; retry item-by-item
            logger.warning("MOSS ref batch encode (n=%d) failed; falling back to per-item", len(prepared))
            for local_i, orig_i in enumerate(prepared_idx):
                try:
                    results[orig_i] = self._encode_prepared([prepared[local_i]])[0]
                except Exception as exc:  # noqa: BLE001 — isolate this clip's failure
                    results[orig_i] = exc
        return results

    def _encode_prepared(self, prepared: list[torch.Tensor]) -> list[torch.Tensor]:
        with torch.no_grad():
            return self._processor.encode_audios_from_wav(prepared, sampling_rate=self._sr_target, n_vq=self._n_vq)

    async def aclose(self) -> None:
        """Release the batcher drainer + any in-flight encodes (tests/shutdown)."""
        for task in list(self._inflight.values()):
            task.cancel()
        self._inflight.clear()
        await self._batcher.aclose()
