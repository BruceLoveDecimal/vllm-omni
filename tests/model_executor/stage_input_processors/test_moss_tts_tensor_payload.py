# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MOSS-TTS raw async-chunk processor: 2-D tensor payload (RFC #4676, W3).

Stage 0 ships each codec chunk as a ``(T, n_vq)`` tensor by default so the
Stage-1 codec reshapes it directly and the connector keeps it a tensor (the
``ndim >= 2`` receive fast path). Flattening to a 1-D ``[n_vq * T]`` stream --
which the receiver materialises into a ``list[int]`` prompt -- is the fallback,
selected via the ``moss_codec_tensor_payload`` connector flag.

CPU-only and weight-free: the processor is driven with a fake transfer_manager.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vllm_omni.model_executor.stage_input_processors.moss_tts import talker2codec_raw_async_chunk

N_VQ = 4


def _tm(extra: dict):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        put_req_chunk=defaultdict(int),
        connector=SimpleNamespace(config={"extra": extra}),
    )


def _req(rid="r1"):
    return SimpleNamespace(external_req_id=rid, request_id=rid, is_finished=lambda: False)


def _frames(t):
    # (t, N_VQ) distinct in-range codes, no all-pad rows.
    return {"codes": {"audio": torch.arange(t * N_VQ, dtype=torch.long).reshape(t, N_VQ) % 1000}}


def test_default_emits_2d_tensor_chunk():
    tm = _tm({"codec_chunk_frames": 3, "initial_codec_chunk_frames": 1})
    payload = talker2codec_raw_async_chunk(tm, _frames(1), _req(), is_finished=False)
    assert payload is not None
    audio = payload.codes.audio
    assert isinstance(audio, torch.Tensor)
    assert audio.ndim == 2 and tuple(audio.shape) == (1, N_VQ)  # (T, n_vq)
    assert audio.dtype == torch.long
    assert payload.meta.code_flat_numel == N_VQ
    assert payload.meta.codec_streaming is True
    assert int(payload.meta.codec_chunk_frames) == 1


def test_steady_chunk_threshold_2d():
    tm = _tm({"codec_chunk_frames": 3, "initial_codec_chunk_frames": 1})
    tm.put_req_chunk["r1"] = 1  # first chunk already emitted -> steady threshold=3
    assert talker2codec_raw_async_chunk(tm, _frames(2), _req(), is_finished=False) is None  # 2 < 3, buffer
    payload = talker2codec_raw_async_chunk(tm, _frames(1), _req(), is_finished=False)  # now 3 pending
    assert payload is not None
    assert tuple(payload.codes.audio.shape) == (3, N_VQ)


def test_flat_fallback_when_disabled():
    tm = _tm({"codec_chunk_frames": 3, "initial_codec_chunk_frames": 1, "moss_codec_tensor_payload": False})
    payload = talker2codec_raw_async_chunk(tm, _frames(1), _req(), is_finished=False)
    assert payload is not None
    audio = payload.codes.audio
    assert audio.ndim == 1 and tuple(audio.shape) == (N_VQ,)  # flat [n_vq * T]
    assert payload.meta.code_flat_numel == N_VQ


def test_finish_control_packet_stays_1d_sentinel():
    tm = _tm({"codec_chunk_frames": 3, "initial_codec_chunk_frames": 1})
    # No pending frames + finish -> control-only finish packet (not a 2-D chunk).
    payload = talker2codec_raw_async_chunk(tm, None, _req(), is_finished=True)
    assert payload is not None
    assert payload.codes.audio.ndim == 1
    assert int(payload.meta.code_flat_numel) == 0
    assert bool(payload.meta.stream_finished.item()) is True


def test_finish_flushes_remaining_as_2d():
    tm = _tm({"codec_chunk_frames": 10, "initial_codec_chunk_frames": 10})
    assert talker2codec_raw_async_chunk(tm, _frames(2), _req(), is_finished=False) is None  # 2 < 10
    payload = talker2codec_raw_async_chunk(tm, _frames(1), _req(), is_finished=True)  # flush 3
    assert payload is not None
    assert tuple(payload.codes.audio.shape) == (3, N_VQ)
    assert bool(payload.meta.stream_finished.item()) is True
