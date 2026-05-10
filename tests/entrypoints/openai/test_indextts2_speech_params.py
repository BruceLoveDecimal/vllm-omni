# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech


def _service() -> OmniOpenAIServingSpeech:
    return OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)


def _request(**kwargs) -> OpenAICreateSpeechRequest:
    base = {
        "input": "hello",
        "ref_audio": "data:audio/wav;base64,AAAA",
    }
    base.update(kwargs)
    return OpenAICreateSpeechRequest(**base)


def test_validate_indextts2_requires_text_and_reference_audio():
    service = _service()

    missing_audio = OpenAICreateSpeechRequest(input="hello")
    assert "requires 'ref_audio'" in service._validate_indextts2_request(missing_audio)

    empty_text = OpenAICreateSpeechRequest(input="", ref_audio="data:audio/wav;base64,AAAA")
    assert service._validate_indextts2_request(empty_text) == "Input text cannot be empty"


def test_validate_indextts2_rejects_bad_extra_params():
    service = _service()

    assert "Unsupported IndexTTS2 extra_params" in service._validate_indextts2_request(
        _request(extra_params={"unknown": 1})
    )
    assert "'emo_vector' must be a list of 8 numbers" == service._validate_indextts2_request(
        _request(extra_params={"emo_vector": [0.1, 0.2]})
    )
    assert "'emo_alpha' must be between 0 and 1" == service._validate_indextts2_request(
        _request(extra_params={"emo_alpha": 2.0})
    )


@pytest.mark.asyncio
async def test_build_indextts2_params_resolves_reference_and_emotion_audio():
    service = _service()

    async def _resolve(ref_audio: str):
        if ref_audio == "data:audio/wav;base64,EMO=":
            return [0.5, 0.25], 16000
        return [0.1, -0.1], 24000

    service._resolve_ref_audio = _resolve
    request = _request(
        instructions="happy",
        max_new_tokens=123,
        extra_params={
            "emo_audio": "data:audio/wav;base64,EMO=",
            "emo_alpha": 0.7,
            "top_p": 0.9,
        },
    )

    params = await service._build_indextts2_params(request)

    assert params["text"] == ["hello"]
    assert params["spk_audio_array"] == [[[0.1, -0.1], 24000]]
    assert params["emo_audio_array"] == [[[0.5, 0.25], 16000]]
    assert params["use_emo_text"] == [True]
    assert params["emo_text"] == ["happy"]
    assert params["max_mel_tokens"] == [123]
    assert params["emo_alpha"] == [0.7]
    assert params["top_p"] == [0.9]
