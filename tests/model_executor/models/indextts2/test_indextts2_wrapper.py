# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.indextts2.configuration_indextts2 import (
    IndexTTS2Config,
)
from vllm_omni.model_executor.models.indextts2.modeling_indextts2 import (
    IndexTTS2ForGeneration,
    _to_mono_1d,
)


class _FakeIndexTTS2Runtime:
    def infer_generator(self, **kwargs):
        text = kwargs["text"]
        value = 1.0 if text == "request a" else 2.0
        yield torch.full((4,), value, dtype=torch.float32)


def _make_model() -> IndexTTS2ForGeneration:
    cfg = IndexTTS2Config(hidden_size=8, vocab_size=16)
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=cfg, model="/tmp/indextts2"))
    model = IndexTTS2ForGeneration(vllm_config=vllm_config)
    model._runtime = _FakeIndexTTS2Runtime()
    model._device = torch.device("cpu")
    return model


def test_to_mono_1d_mixes_channels_and_scales_int16_range():
    waveform = torch.tensor([[32767.0, 0.0], [32767.0, -32767.0]])

    result = _to_mono_1d(waveform)

    torch.testing.assert_close(result, torch.tensor([1.0, -0.5]), atol=1e-4, rtol=1e-4)


def test_forward_uses_independent_generators_per_request():
    model = _make_model()
    infos = [
        {
            "global_request_id": "req-a",
            "text": "request a",
            "spk_audio_prompt": "/tmp/ref_a.wav",
        },
        {
            "global_request_id": "req-b",
            "text": "request b",
            "spk_audio_prompt": "/tmp/ref_b.wav",
        },
    ]

    first = model.forward(
        input_ids=torch.ones((2, 1), dtype=torch.long),
        runtime_additional_information=infos,
    )

    chunks = first.multimodal_outputs["model_outputs"]
    torch.testing.assert_close(chunks[0], torch.ones(4))
    torch.testing.assert_close(chunks[1], torch.full((4,), 2.0))
    assert set(model._stream_gens) == {"req-a", "req-b"}

    logits = model.compute_logits(first)
    assert torch.argmax(logits[0]).item() != 2
    assert torch.argmax(logits[1]).item() != 2

    second = model.forward(
        input_ids=torch.ones((2, 1), dtype=torch.long),
        runtime_additional_information=infos,
    )

    chunks = second.multimodal_outputs["model_outputs"]
    assert chunks[0].numel() == 0
    assert chunks[1].numel() == 0
    assert model._stream_gens == {}

    logits = model.compute_logits(second)
    assert torch.argmax(logits[0]).item() == 2
    assert torch.argmax(logits[1]).item() == 2


def test_dummy_forward_finishes_immediately():
    model = _make_model()

    output = model.forward(runtime_additional_information=[{"_is_dummy": True}])

    assert output.multimodal_outputs["model_outputs"][0].numel() == 0
    assert int(output.multimodal_outputs["sr"][0].item()) == 22050
    logits = model.compute_logits(output)
    assert torch.argmax(logits[0]).item() == 2
