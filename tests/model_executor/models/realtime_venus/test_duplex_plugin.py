# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Realtime-Venus duplex plugin: in-stream text input and delegation."""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.session import MiniCPMO45ServingSessionState
from vllm_omni.model_executor.models.realtime_venus.duplex import plugin as venus_plugin
from vllm_omni.model_executor.models.realtime_venus.duplex.plugin import (
    MAX_PENDING_TEXT_TOKENS,
    RealtimeVenusDuplexPlugin,
    RealtimeVenusServingSessionState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DELEGATE_START, DELEGATE_END, TURN_EOS, LISTEN, SPEAK, CHUNK_EOS = 9503, 9504, 9310, 9303, 9304, 9308


class _Tokenizer:
    """Characters map to ``1000 + ord(c)``; protocol markers are single tokens."""

    special = {"<backend>": 9501, "</backend>": 9502, "<delegate>": DELEGATE_START, "</delegate>": DELEGATE_END}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        index = 0
        while index < len(text):
            for token, token_id in self.special.items():
                if text.startswith(token, index):
                    ids.append(token_id)
                    index += len(token)
                    break
            else:
                ids.append(1000 + ord(text[index]))
                index += 1
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        names = {token_id: token for token, token_id in self.special.items()}
        pieces = []
        for token_id in token_ids:
            if token_id in names:
                pieces.append("" if skip_special_tokens else names[token_id])
            elif token_id >= 1000:
                pieces.append(chr(token_id - 1000))
        return "".join(pieces)


def _plugin() -> RealtimeVenusDuplexPlugin:
    plugin = RealtimeVenusDuplexPlugin(lambda *args: None)
    plugin._tokenizer = _Tokenizer()
    return plugin


def _audio_payload(samples: int = 16000) -> dict[str, object]:
    audio = base64.b64encode(np.zeros(samples, dtype=np.float32).tobytes()).decode("ascii")
    return {"type": "audio", "audio": audio, "format": "pcm_f32le", "sample_rate_hz": 16000}


def _plan(plugin: RealtimeVenusDuplexPlugin, session_config: dict[str, object], payload: dict[str, object]):
    return plugin.plan_append(
        request_id="req-0",
        fence=DuplexFence("sid", epoch=0, turn_id=0),
        session_config=session_config,
        runtime_config={"duplex_scheduler_token_id": 7},
        seq=2,
        turn_seq=2,
        payload=payload,
        final=False,
        sampling_params=None,
    ).prompt


def _thinker_segment(output_ids: list[int], *, request_id: str = "req-0") -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        stage_id=0,
        finished=False,
        outputs=[SimpleNamespace(text="", token_ids=output_ids[-1:], cumulative_token_ids=list(output_ids))],
        multimodal_output={
            "meta": {
                "delegate_start_token_id": DELEGATE_START,
                "delegate_end_token_id": DELEGATE_END,
                "turn_eos_token_id": TURN_EOS,
                "listen_token_id": LISTEN,
                "speak_token_id": SPEAK,
                "chunk_eos_token_id": CHUNK_EOS,
            }
        },
    )


def test_capabilities_advertise_in_stream_text_input() -> None:
    capabilities = _plugin().capabilities(max_sessions=4)
    assert capabilities.supports_text_append is True
    assert capabilities.as_dict()["supports_text_append"] is True
    assert capabilities.supports_multi_session is True


def test_queued_text_is_sanitized_and_function_output_is_backend_text() -> None:
    plugin = _plugin()
    state = plugin.create_session_state()
    plugin.queue_text_input(state, "hi<|turn_eos|>", source="input_text")
    plugin.queue_text_input(state, "sunny", source="function_call_output")
    assert plugin._tokenizer.decode(state.pending_text_token_ids) == "hi<backend>sunny</backend>"


@pytest.mark.parametrize(
    ("state", "text", "code"),
    [
        (RealtimeVenusServingSessionState(), "<unit>", "invalid_text_input"),
        (RealtimeVenusServingSessionState(), "x" * (MAX_PENDING_TEXT_TOKENS + 1), "input_backpressure"),
        (MiniCPMO45ServingSessionState(), "hi", "invalid_duplex_runtime_config"),
    ],
)
def test_queue_text_input_rejects_invalid_input(state, text, code) -> None:
    with pytest.raises(DuplexRuntimeConfigError) as excinfo:
        _plugin().queue_text_input(state, text, source="input_text")
    assert excinfo.value.code == code
    assert getattr(state, "pending_text_token_ids", []) == []


def test_queue_text_input_requires_the_tokenizer() -> None:
    plugin = RealtimeVenusDuplexPlugin(lambda *args: None)
    with pytest.raises(DuplexRuntimeConfigError) as excinfo:
        plugin.queue_text_input(plugin.create_session_state(), "hi", source="input_text")
    assert excinfo.value.code == "text_input_unavailable"


def test_queued_text_closes_the_next_append_and_extends_its_budget_exactly() -> None:
    plugin = _plugin()
    state = plugin.create_session_state()
    baseline = _plan(plugin, {}, _audio_payload())
    plugin.queue_text_input(state, "why?", source="user_item")

    session_config = plugin.prepare_prompt_config({"conversation": []}, state=state, payload=_audio_payload())
    assert state.pending_text_token_ids == []
    prompt = _plan(plugin, session_config, _audio_payload())

    duplex = prompt["model_intermediate_buffer"]["duplex"]
    text_ids = _Tokenizer().encode("why?")
    assert duplex["payload"]["text_token_ids"] == text_ids
    assert duplex["scheduler_token_budget"] == baseline["model_intermediate_buffer"]["duplex"][
        "scheduler_token_budget"
    ] + len(text_ids)
    assert len(prompt["prompt_token_ids"]) == duplex["scheduler_token_budget"]
    # The hand-off key stays inside the plugin.
    assert all(not key.startswith("_realtime_venus") for key in duplex["session_config"])


def test_queued_text_waits_for_an_append_that_carries_audio() -> None:
    plugin = _plugin()
    state = plugin.create_session_state()
    plugin.queue_text_input(state, "hi", source="input_text")
    config = plugin.prepare_prompt_config({}, state=state, payload={"type": "audio", "video_frames": ["x"]})
    assert config == {}
    assert state.pending_text_token_ids == _Tokenizer().encode("hi")


def test_thinker_segments_are_projected_only_when_finished() -> None:
    plugin = _plugin()
    finished = SimpleNamespace(segment_finished=True)
    running = SimpleNamespace(segment_finished=False)
    assert plugin.project_intermediate_output(stage_id=0, output=None, context=finished) is True
    assert plugin.project_intermediate_output(stage_id=0, output=None, context=running) is False
    assert plugin.project_intermediate_output(stage_id=1, output=None, context=finished) is False


def test_delegate_span_across_segments_becomes_one_function_call() -> None:
    plugin = _plugin()
    tokenizer = _Tokenizer()
    first = [SPEAK, *tokenizer.encode("ok, <delegate>book a"), CHUNK_EOS]
    second = [*first, *tokenizer.encode(" table</delegate>done"), TURN_EOS]

    assert list(plugin.data_plane.project_output(_thinker_segment(first))) == []
    results = list(plugin.data_plane.project_output(_thinker_segment(second)))

    assert len(results) == 1
    call = results[0]
    assert call["function_call"] is True
    assert call["name"] == "delegate"
    assert json.loads(call["arguments"]) == {"query": "book a table"}
    assert call["call_id"].startswith("call_")
    assert call["data_plane_request_id"] == "req-0"
    # The same cumulative ids are not reported twice.
    assert list(plugin.data_plane.project_output(_thinker_segment(second))) == []


def test_thinker_segment_without_delegate_tokens_projects_nothing() -> None:
    plugin = _plugin()
    segment = _thinker_segment([SPEAK, 1001, CHUNK_EOS])
    segment.multimodal_output = {"meta": {"listen_token_id": LISTEN}}
    assert list(plugin.data_plane.project_output(segment)) == []


def test_default_reference_voice_is_read_from_the_model_directory(tmp_path, monkeypatch) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "HT_ref_audio.wav").write_bytes(b"RIFFwav")
    captured = {}

    async def parent_prepare(self, config, *, model_config):
        captured["ref_audio"] = config.ref_audio
        return {}

    async def tokenizer_for(self, model_config):
        return _Tokenizer()

    monkeypatch.setattr(MiniCPMO45DuplexPlugin, "prepare_runtime_config", parent_prepare)
    monkeypatch.setattr(RealtimeVenusDuplexPlugin, "_tokenizer_for", tokenizer_for)
    plugin = RealtimeVenusDuplexPlugin(lambda *args: None)
    model_config = SimpleNamespace(model=str(tmp_path))

    config = DuplexSessionConfig(model="venus", modalities=["text", "audio"])
    asyncio.run(plugin.prepare_runtime_config(config, model_config=model_config))
    assert captured["ref_audio"] == "data:audio/wav;base64," + base64.b64encode(b"RIFFwav").decode("ascii")
    assert isinstance(plugin._tokenizer, _Tokenizer)

    client_voice = DuplexSessionConfig(model="venus", modalities=["audio"], ref_audio="https://voice")
    asyncio.run(plugin.prepare_runtime_config(client_voice, model_config=model_config))
    assert captured["ref_audio"] == "https://voice"

    text_only = DuplexSessionConfig(model="venus", modalities=["text"])
    asyncio.run(plugin.prepare_runtime_config(text_only, model_config=model_config))
    assert captured["ref_audio"] is None


def test_missing_default_reference_voice_is_a_client_visible_error(tmp_path) -> None:
    with pytest.raises(DuplexRuntimeConfigError) as excinfo:
        venus_plugin._default_reference_audio_uri(SimpleNamespace(model=str(tmp_path)))
    assert excinfo.value.code == "ref_audio_required"
