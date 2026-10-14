# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Same-shape vocoder decode batches preserve independent streaming state."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.model_executor.models.cosyvoice3.test_cosyvoice3_incremental_hift import (
    LEGACY_CONFIG,
    REAL_CONFIG,
    _make_hift,
    _make_model,
)

pytestmark = [pytest.mark.core_model]


def _check_streams(device, config, graphs=False, atol=1e-6):
    hift = _make_hift(config).to(device)
    if graphs:
        hift.remove_weight_norm()
        hift.f0_predictor.remove_weight_norm()
        hift.enable_decode_graphs()
    model = _make_model(hift, window_len=64)
    generator = torch.Generator(device=device).manual_seed(28)
    actual_states = [None] * 3
    expected_states = [None] * 3
    retained = []
    for step in range(9):
        finalize = step == 8
        # Two compatible streams, one independent tail, and an empty request.
        widths = [0, 0, 17] if finalize else [24, 24, 28]
        feats = [torch.randn(1, 80, width, device=device, generator=generator) for width in widths]
        flags = [finalize, finalize, True]
        expected = [
            model._stream_hift_from_feat(feat, cache_state=state, finalize=flag)
            for feat, state, flag in zip(feats, expected_states, flags)
        ]
        order = [2, 1, 0] if step % 2 else [0, 2, 1]
        inputs = [(feats[i], actual_states[i], flags[i]) for i in order]
        inputs.insert(1, (feats[0][..., :0], None, finalize))
        with patch.object(hift, "decode", wraps=hift.decode) as decode:
            actual = model._stream_hift_batch(inputs)
        assert sorted(call.args[0].shape[0] for call in decode.call_args_list) == [1, 2]
        empty, empty_state = actual.pop(1)
        assert empty.numel() == 0
        assert (empty_state is None) == finalize
        for i, (audio, state) in zip(order, actual):
            ref_audio, ref_state = expected[i]
            torch.testing.assert_close(audio, ref_audio, rtol=1e-5, atol=atol)
            assert (state is None) == (ref_state is None)
            if state is not None:
                assert state["mel_offset"] == ref_state["mel_offset"]
                torch.testing.assert_close(state["mel"], ref_state["mel"], rtol=0, atol=0)
                torch.testing.assert_close(state["phase_acc"], ref_state["phase_acc"], rtol=0, atol=0)
            actual_states[i], expected_states[i] = state, ref_state
            retained.append((audio, audio.clone()))
        # Each result owns its trimmed waveform, including the two batched rows.
        by_request = dict(zip(order, actual))
        assert by_request[0][0].untyped_storage().data_ptr() != by_request[1][0].untyped_storage().data_ptr()
        for audio, saved in retained:
            torch.testing.assert_close(audio, saved, rtol=0, atol=0)
    assert model._stream_hift_batch([]) == []
    if graphs:
        assert any(entry.mel.shape[0] == 2 for entry in hift._decode_graphs.graphs.values())


@pytest.mark.cpu
@pytest.mark.usefixtures("single_threaded_cpu")
@pytest.mark.parametrize("config", [REAL_CONFIG, LEGACY_CONFIG])
@torch.inference_mode()
def test_hift_decode_batch_preserves_stream_state(config):
    _check_streams("cpu", config)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("tf32", [False, True])
@torch.inference_mode()
def test_hift_decode_batch_cuda(graphs, tf32):
    # Batch size can select different TF32 convolution kernels. Also verify
    # strict FP32 parity independently of this lower-precision rounding.
    with torch.backends.cudnn.flags(allow_tf32=tf32):
        _check_streams("cuda", REAL_CONFIG, graphs=graphs, atol=1e-4 if tf32 else 1e-6)


@pytest.mark.cpu
@pytest.mark.usefixtures("single_threaded_cpu")
@pytest.mark.parametrize("batch_flow", [False, True])
@torch.inference_mode()
def test_hift_decode_batches_across_flow_groups(monkeypatch, batch_flow):
    monkeypatch.setenv("COSYVOICE3_PACKED_STREAMING", "0")
    monkeypatch.setenv("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "0")
    model = _make_model(_make_hift(), window_len=64)
    model.flow_model = SimpleNamespace(pre_lookahead_len=1, token_mel_ratio=2)
    generator = torch.Generator().manual_seed(11)
    mels = torch.randn(3, 80, 24, generator=generator)
    items = [
        dict(
            token=torch.full((1, 13), i, dtype=torch.int32),
            prompt_token=torch.ones(1, prompt, dtype=torch.int32),
            prompt_feat=torch.ones(1, prompt * 2, 80),
            embedding=torch.ones(1, 192),
            finalize=False,
        )
        for i, prompt in enumerate([4, 7, 4])
    ]

    def flow(**kwargs):
        return mels[kwargs["token"][:, 0].long()]

    monkeypatch.setattr(model, "_forward_mel", flow)
    expected = [model.forward_streaming_batch([item])[0] for item in items]
    with (
        patch.object(model, "_forward_mel", wraps=flow) as flow_calls,
        patch.object(model.hift, "decode", wraps=model.hift.decode) as decode,
    ):
        actual = model.forward_streaming_batch(items, batch_flow=batch_flow)
    assert len(flow_calls.call_args_list) == (2 if batch_flow else 3)
    assert [call.args[0].shape[0] for call in decode.call_args_list] == [3]
    for (audio, state), (reference, ref_state) in zip(actual, expected):
        torch.testing.assert_close(audio, reference, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(state["phase_acc"], ref_state["phase_acc"], rtol=0, atol=0)
