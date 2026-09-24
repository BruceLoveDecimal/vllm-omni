# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Realtime-Venus full-duplex plugin.

Realtime-Venus runs on the MiniCPM-o 4.5 duplex runtime and only adds session
policy on top of it:

- Client text (``input.text.append``, user text items, and function-call
  outputs) is read in-stream: it closes the model's next input unit, after the
  unit's audio, so the listen/speak decision is sampled after the text.
- ``<delegate>query</delegate>`` spans in the Thinker output reach the client
  as ``delegate`` function calls and are never spoken. The client answers with
  a ``function_call_output`` item, which the model reads as
  ``<backend>answer</backend>``.
- The checkpoint's packaged reference voice is the default Talker voice.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pybase64 as base64

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan
from vllm_omni.engine.duplex.plugin import DuplexModelSessionState, DuplexRuntimeConfigError
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.capabilities import minicpmo45_native_capabilities
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.data_plane import (
    MiniCPMO45DataPlaneContext,
    MiniCPMO45DataPlaneSession,
    _runtime_result,
    _special_token_ids,
    coerce_int,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.session import MiniCPMO45ServingSessionState
from vllm_omni.model_executor.models.realtime_venus.duplex.protocol import (
    DELEGATE_FUNCTION_NAME,
    DelegateSpanTracker,
    DelegateStreamCursor,
    backend_text,
    strip_protocol_tokens,
)
from vllm_omni.outputs.duplex import get_duplex_output_decision

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from vllm.config import ModelConfig

    from vllm_omni.engine.duplex.plugin import EncodeAudio

#: Reference voice packaged with both Realtime-Venus checkpoints.
DEFAULT_REFERENCE_AUDIO = "assets/HT_ref_audio.wav"
#: Client text waiting for the next input unit, in tokens. The text becomes
#: part of the Stage-0 context, so an unbounded queue could fill it.
MAX_PENDING_TEXT_TOKENS = 2048
# Session-config key that carries queued text from ``prepare_prompt_config``
# (which owns the session state) to ``plan_append`` (which owns the payload).
_TEXT_TOKEN_IDS_KEY = "_realtime_venus_text_token_ids"


@dataclass(slots=True)
class RealtimeVenusServingSessionState(MiniCPMO45ServingSessionState):
    """MiniCPM-o 4.5 session state plus client text queued for the next unit."""

    pending_text_token_ids: list[int] = field(default_factory=list)


class RealtimeVenusDataPlaneSession(MiniCPMO45DataPlaneSession):
    """MiniCPM-o 4.5 projector that also turns delegate spans into function calls."""

    def __init__(self, encode_audio: EncodeAudio, plugin: RealtimeVenusDuplexPlugin) -> None:
        super().__init__(encode_audio)
        self._plugin = plugin
        self._delegate_cursors: dict[str, DelegateStreamCursor] = {}

    def close_request(self, request_id: str) -> None:
        super().close_request(request_id)
        self._delegate_cursors.pop(request_id, None)

    def project_output(
        self,
        output: object,
        *,
        context: MiniCPMO45DataPlaneContext | None = None,
    ) -> Iterator[dict[str, object]]:
        if getattr(output, "stage_id", None) == 0:
            yield from self._project_delegate_calls(output)
            if get_duplex_output_decision(output) is None:
                # A Thinker segment projected only for its delegate spans; its
                # text and hidden states still flow to the Talker.
                return
        yield from super().project_output(output, context=context)

    def _project_delegate_calls(self, output: object) -> Iterator[dict[str, object]]:
        request_id = getattr(output, "request_id", None)
        outputs = getattr(output, "outputs", None)
        completion = outputs[0] if isinstance(outputs, list) and outputs else None
        if not isinstance(request_id, str) or not request_id or completion is None:
            return
        output_ids = getattr(completion, "cumulative_token_ids", None) or getattr(completion, "token_ids", None)
        if not output_ids:
            return
        decision = get_duplex_output_decision(output)
        mm_output = getattr(decision, "metadata", None) if decision is not None else None
        if not isinstance(mm_output, Mapping):
            mm_output = getattr(output, "multimodal_output", None) or getattr(completion, "multimodal_output", None)
        cursor = self._delegate_cursor(request_id, _special_token_ids(dict(mm_output or {})))
        if cursor is None:
            return
        output_ids = [token_id for value in output_ids if (token_id := coerce_int(value)) is not None]
        for query in cursor.feed_cumulative(output_ids, self._plugin.decode_text):
            yield _runtime_result(
                stage_role="llm",
                data_plane_request_id=request_id,
                function_call=True,
                call_id=f"call_{uuid4().hex}",
                name=DELEGATE_FUNCTION_NAME,
                arguments=json.dumps({"query": query}, ensure_ascii=False),
            )

    def _delegate_cursor(self, request_id: str, special_token_ids: dict[str, int]) -> DelegateStreamCursor | None:
        cursor = self._delegate_cursors.get(request_id)
        if cursor is not None:
            return cursor
        start_id = special_token_ids.get("delegate_start_token_id")
        end_id = special_token_ids.get("delegate_end_token_id")
        if start_id is None or end_id is None:
            return None
        tracker = DelegateSpanTracker(
            start_id=start_id,
            end_id=end_id,
            turn_eos_id=special_token_ids.get("turn_eos_token_id"),
            ignored_ids=frozenset(special_token_ids.values()),
        )
        cursor = DelegateStreamCursor(tracker=tracker)
        self._delegate_cursors[request_id] = cursor
        return cursor


class RealtimeVenusDuplexPlugin(MiniCPMO45DuplexPlugin):
    """MiniCPM-o 4.5 duplex policy plus Realtime-Venus text input and delegation."""

    plugin_id = "realtime_venus"

    def __init__(self, encode_audio: EncodeAudio) -> None:
        super().__init__(encode_audio)
        self.data_plane = RealtimeVenusDataPlaneSession(encode_audio, self)
        # One model per engine: the tokenizer the sessions were prepared with.
        self._tokenizer: PreTrainedTokenizerBase | None = None

    # ---- text input ----

    def create_session_state(self) -> RealtimeVenusServingSessionState:
        return RealtimeVenusServingSessionState()

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        return replace(minicpmo45_native_capabilities(max_sessions=max_sessions), supports_text_append=True)

    def queue_text_input(self, state: DuplexModelSessionState, text: str, *, source: str) -> None:
        if not isinstance(state, RealtimeVenusServingSessionState):
            raise DuplexRuntimeConfigError("Realtime-Venus text input requires a Realtime-Venus session")
        if self._tokenizer is None:
            raise DuplexRuntimeConfigError(
                "Realtime-Venus text input needs the model tokenizer, which failed to load",
                code="text_input_unavailable",
            )
        if source == "function_call_output":
            model_text = backend_text(text)
        else:
            model_text = strip_protocol_tokens(text)
        if not model_text:
            raise DuplexRuntimeConfigError("text input is empty", code="invalid_text_input")
        token_ids = [int(token_id) for token_id in self._tokenizer.encode(model_text, add_special_tokens=False)]
        if len(state.pending_text_token_ids) + len(token_ids) > MAX_PENDING_TEXT_TOKENS:
            raise DuplexRuntimeConfigError(
                f"pending text input exceeds {MAX_PENDING_TEXT_TOKENS} tokens; send audio so the model reads it",
                code="input_backpressure",
            )
        state.pending_text_token_ids.extend(token_ids)

    def prepare_prompt_config(
        self,
        config: dict[str, object],
        *,
        state: DuplexModelSessionState,
        payload: dict[str, object],
    ) -> dict[str, object]:
        # Queued text closes the next unit; every append carries audio (real
        # input or a silence continuation), so any append can take it.
        if not isinstance(state, RealtimeVenusServingSessionState) or not state.pending_text_token_ids:
            return config
        if not payload.get("audio"):
            return config
        token_ids = list(state.pending_text_token_ids)
        state.pending_text_token_ids.clear()
        return {**config, _TEXT_TOKEN_IDS_KEY: token_ids}

    def plan_append(self, *, session_config: dict[str, object], payload: object, **kwargs) -> DuplexAppendPlan:
        token_ids = session_config.get(_TEXT_TOKEN_IDS_KEY)
        if isinstance(token_ids, list) and isinstance(payload, dict):
            session_config = {key: value for key, value in session_config.items() if key != _TEXT_TOKEN_IDS_KEY}
            payload = {**payload, "text_token_ids": token_ids}
        return super().plan_append(session_config=session_config, payload=payload, **kwargs)

    def decode_text(self, token_ids: list[int]) -> str:
        if self._tokenizer is None:
            return ""
        return str(self._tokenizer.decode(token_ids, skip_special_tokens=True))

    # ---- delegation ----

    def project_intermediate_output(self, *, stage_id: int, output: object, context: object) -> bool:
        # Every finished Thinker segment passes the data plane, which emits a
        # function call for each delegate span it closes. Listen decisions
        # already reach the data plane as direct responses.
        del output
        segment_finished = bool(getattr(context, "segment_finished", False))
        return stage_id == 0 and segment_finished and self._tokenizer is not None

    # ---- session policy ----

    async def prepare_runtime_config(
        self, config: DuplexSessionConfig, *, model_config: ModelConfig | None
    ) -> dict[str, object]:
        self._tokenizer = await self._tokenizer_for(model_config)
        if _requests_audio(config) and not _has_client_ref_audio(config):
            config.ref_audio = _default_reference_audio_uri(model_config)
        return await super().prepare_runtime_config(config, model_config=model_config)


def _requests_audio(config: DuplexSessionConfig) -> bool:
    return any(str(modality).lower() == "audio" for modality in config.modalities)


def _has_client_ref_audio(config: DuplexSessionConfig) -> bool:
    extra_body = config.extra_body if isinstance(config.extra_body, dict) else {}
    return config.ref_audio is not None or any(
        isinstance(extra_body.get(key), str) for key in ("ref_audio", "tts_ref_audio")
    )


def _default_reference_audio_uri(model_config: ModelConfig | None) -> str:
    """The packaged reference voice as a data URI, read from the model directory."""
    model_path = getattr(model_config, "model", None)
    path = Path(model_path) / DEFAULT_REFERENCE_AUDIO if isinstance(model_path, str) else None
    if path is None or not path.is_file():
        raise DuplexRuntimeConfigError(
            f"Realtime-Venus audio output needs ref_audio or the packaged voice {DEFAULT_REFERENCE_AUDIO} "
            "in the local model directory",
            code="ref_audio_required",
        )
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{data}"


__all__ = [
    "RealtimeVenusDataPlaneSession",
    "RealtimeVenusDuplexPlugin",
    "RealtimeVenusServingSessionState",
]
