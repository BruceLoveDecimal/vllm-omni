# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batched streaming HiFT on CUDA, on a side stream and with decode graphs,
must emit what the per-request path emits on the current stream."""

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.model_executor.models.cosyvoice3.test_cosyvoice3_hift_stream_batch import (
    ATOL,
    RTOL,
    STREAMS,
    _run,
    _voiced_model,
)
from vllm_omni.model_executor.models.cosyvoice3 import cosyvoice3_code2wav
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA"),
]


def _side_stream_hift(model):
    """Wrap _stream_hift as _forward_packed_streaming runs it: inputs made on
    the current stream, HiFT on its own stream, results joined back."""
    per_group = model._stream_hift
    side = torch.cuda.Stream()

    def run(feats, cache_states, *, finalize):
        feats = [feat.cuda() for feat in feats]
        event = torch.cuda.Event()
        event.record()
        with torch.cuda.stream(side):
            side.wait_event(event)
            for feat in feats:
                feat.record_stream(side)
            values = per_group(feats, cache_states, finalize=finalize)
        current = torch.cuda.current_stream()
        current.wait_stream(side)
        for speech, _ in values:
            speech.record_stream(current)
        return values

    return run


@pytest.mark.nvidia_only
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("decode_graphs", [False, True])
@torch.inference_mode()
def test_side_stream_batched_hift_matches_per_request(monkeypatch, decode_graphs):
    monkeypatch.setattr(cosyvoice3_code2wav, "cosyvoice3_packed_streaming_enabled", lambda: True)
    # Batch size changes cuDNN's algorithm; TF32 would add rounding beyond the shared tolerance.
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    model = _voiced_model(64)
    model.hift.cuda()
    model.hift.enable_cached_istft()

    class OnCuda:
        """Feed _run's CPU chunks to the per-request path on CUDA."""

        def __init__(self, inner):
            self.inner = inner

        def _stream_hift_from_feat(self, feat, *, cache_state, finalize):
            return self.inner._stream_hift_from_feat(feat.cuda(), cache_state=cache_state, finalize=finalize)

    expected = _run(OnCuda(model), batched=False)
    if decode_graphs:
        model.hift.enable_decode_graphs()
    monkeypatch.setattr(cosyvoice3_code2wav, "cosyvoice3_hift_batch_enabled", lambda: True)
    monkeypatch.setattr(model, "_stream_hift", _side_stream_hift(model))
    # Repeat so recurring bucketed shapes reach graph capture and replay.
    for _ in range(3 if decode_graphs else 1):
        actual = _run(model, batched=True)
        torch.accelerator.synchronize()
        assert len(actual) == len(STREAMS)
        for stream_expected, stream_actual in zip(expected, actual):
            for want, got in zip(stream_expected, stream_actual):
                torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL)
