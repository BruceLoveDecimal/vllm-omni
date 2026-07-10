# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MOSS-TTS codec 2-D tensor code path (RFC #4676, W3).

The codec prefers a ``(T, n_vq)`` tensor delivered via
``additional_information["codes"]["audio"]`` over the flat ``input_ids`` stream,
transposing to ``(n_vq, T)`` and range-validating (dropping out-of-range frames)
instead of silently clamping. Falls back to ``None`` (flat path) when no usable
2-D tensor is present.

CPU-only and weight-free: ``_extract_frame_tensor_codes`` is exercised on a bare
codec with a stub codebook_size.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

N_VQ = 4
CODEBOOK_SIZE = 1024
CPU = torch.device("cpu")


def _bare_codec():
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import MossTTSCodecDecoder

    codec = MossTTSCodecDecoder.__new__(MossTTSCodecDecoder)
    codec._n_vq = N_VQ
    codec._codec = SimpleNamespace(config=SimpleNamespace(codebook_size=CODEBOOK_SIZE))
    return codec


def _info(audio):
    return {"codes": {"audio": audio}}


def test_transposes_T_nvq_to_nvq_T():
    codec = _bare_codec()
    frames = torch.arange(3 * N_VQ, dtype=torch.long).reshape(3, N_VQ) % CODEBOOK_SIZE  # (T=3, n_vq)
    out = codec._extract_frame_tensor_codes(_info(frames), CPU)
    assert out is not None
    assert tuple(out.shape) == (N_VQ, 3)  # (n_vq, T)
    assert torch.equal(out, frames.transpose(0, 1))


def test_accepts_already_nvq_T_when_unambiguous():
    codec = _bare_codec()
    frames = torch.arange(N_VQ * 5, dtype=torch.long).reshape(N_VQ, 5) % CODEBOOK_SIZE  # (n_vq, T=5)
    out = codec._extract_frame_tensor_codes(_info(frames), CPU)
    # shape[1] != n_vq (5 != 4) but shape[0] == n_vq -> treated as (n_vq, T)
    assert tuple(out.shape) == (N_VQ, 5)


def test_none_for_missing_or_1d_or_empty():
    codec = _bare_codec()
    assert codec._extract_frame_tensor_codes({}, CPU) is None
    assert codec._extract_frame_tensor_codes(_info(None), CPU) is None
    assert codec._extract_frame_tensor_codes(_info(torch.arange(N_VQ)), CPU) is None  # 1-D
    assert codec._extract_frame_tensor_codes(_info(torch.empty((0, N_VQ), dtype=torch.long)), CPU) is None


def test_drops_out_of_range_frames_instead_of_clamping():
    codec = _bare_codec()
    frames = torch.zeros((3, N_VQ), dtype=torch.long)
    frames[1, 0] = CODEBOOK_SIZE + 2  # leaked stop/pad code in the middle frame
    out = codec._extract_frame_tensor_codes(_info(frames), CPU)
    assert tuple(out.shape) == (N_VQ, 2)  # middle frame dropped, not clamped
    assert (out < CODEBOOK_SIZE).all() and (out >= 0).all()


def test_all_invalid_returns_empty():
    codec = _bare_codec()
    frames = torch.full((2, N_VQ), CODEBOOK_SIZE + 1, dtype=torch.long)
    out = codec._extract_frame_tensor_codes(_info(frames), CPU)
    assert out is not None and out.numel() == 0  # caller skips on numel()==0


def test_matches_flat_reshape_for_same_codes():
    """Tensor path and the flat fallback must yield identical (n_vq, T) codes."""
    codec = _bare_codec()
    frames = (torch.arange(6 * N_VQ, dtype=torch.long) % CODEBOOK_SIZE).reshape(6, N_VQ)  # (T=6, n_vq)
    tensor_out = codec._extract_frame_tensor_codes(_info(frames), CPU)
    # Flat fallback: sender sends transpose(0,1).reshape(-1); codec reshapes (n_vq, T).
    flat = frames.transpose(0, 1).contiguous().reshape(-1)
    flat_out = flat.reshape(N_VQ, 6)
    assert torch.equal(tensor_out, flat_out)
