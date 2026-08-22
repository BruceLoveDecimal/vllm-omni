# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mage-VL's duplex WebSocket endpoint, over a real socket with the model stubbed.

``test_adapter.py`` covers the streaming policy against the runtime directly; this covers
what a client actually sees on the wire -- decision events, proactive responses, barge-in,
and the session-slot rules that keep a long-lived video stream from wedging the server.
"""

import asyncio

import pytest

from vllm_omni.experimental.fullduplex.core import protocol as ev
from vllm_omni.experimental.fullduplex.mage_vl import MageVLDuplexAdapter, MageVLStreamConfig
from vllm_omni.experimental.fullduplex.mage_vl.serving import (
    MageVLDuplexServer,
    MageVLServingConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

fastapi = pytest.importorskip("fastapi")
from starlette.testclient import TestClient  # noqa: E402


def _client(scores, *, config=None, serving=None, chunks=("hello", " world"), slow=False):
    """A server whose gate replays ``scores`` and whose decoder replays ``chunks``."""
    from vllm_omni.experimental.fullduplex.mage_vl.serving import create_app

    def adapter_factory(emit):
        pending = list(scores)

        def score_segment(_segment):
            return pending.pop(0) if pending else 0.0

        async def generate(_messages):
            for chunk in chunks:
                if slow:
                    await asyncio.sleep(0.02)
                yield chunk

        return MageVLDuplexAdapter(score_segment, generate, config=config, on_decision=emit)

    server = MageVLDuplexServer(adapter_factory, serving or MageVLServingConfig())
    return TestClient(create_app(server)), server


def _drain(socket, *, until, limit=40):
    """Collect frames until one of ``until`` arrives, so tests never hang on a slow stream."""
    seen = []
    for _ in range(limit):
        event = socket.receive_json()
        seen.append(event)
        if event["type"] in until:
            break
    return seen


def test_a_segment_over_the_threshold_answers_on_the_socket():
    client, _ = _client([0.9])

    with client.websocket_connect("/v1/duplex/mage-vl?session_id=s1") as socket:
        socket.send_json({"type": ev.INPUT_APPEND, "modality": "video", "data": "seg-0"})
        events = _drain(socket, until={ev.RESPONSE_DONE})
        socket.send_json({"type": ev.CLOSE})

    types = [e["type"] for e in events]
    assert types[0] == "response.speak", "the gate's decision reaches the client first"
    assert events[0]["metadata"]["p_speak"] == pytest.approx(0.9)
    assert ev.RESPONSE_CREATED in types and ev.RESPONSE_DONE in types
    assert "".join(e["data"] for e in events if e["type"] == ev.RESPONSE_DELTA) == "hello world"


def test_a_quiet_stream_only_reports_listening():
    client, _ = _client([0.1, 0.2])

    with client.websocket_connect("/v1/duplex/mage-vl?session_id=s2") as socket:
        for index in range(2):
            socket.send_json(
                {"type": ev.INPUT_APPEND, "modality": "video", "data": f"seg-{index}"}
            )
        events = [socket.receive_json() for _ in range(2)]
        socket.send_json({"type": ev.CLOSE})

    assert [e["type"] for e in events] == ["response.listen", "response.listen"]
    assert all(e["model_speak"] is False for e in events)


def test_barge_in_over_the_socket_stops_the_answer():
    client, _ = _client([0.9], chunks=[f"c{i}" for i in range(40)], slow=True)

    with client.websocket_connect("/v1/duplex/mage-vl?session_id=s3") as socket:
        socket.send_json({"type": ev.INPUT_APPEND, "modality": "video", "data": "seg-0"})
        # Wait for the answer to actually start before interrupting it.
        _drain(socket, until={ev.RESPONSE_DELTA})
        socket.send_json({"type": ev.RESPONSE_CANCEL})
        tail = _drain(socket, until={ev.RESPONSE_CANCELLED})
        socket.send_json({"type": ev.CLOSE})

    types = [e["type"] for e in tail]
    assert ev.RESPONSE_CANCELLED in types
    assert ev.RESPONSE_DONE not in types
    assert types[types.index(ev.RESPONSE_CANCELLED) :] == [ev.RESPONSE_CANCELLED], (
        "nothing may follow the cancellation on the wire"
    )


def test_capacity_is_refused_rather_than_queued():
    """A late-admitted video stream has already missed what it would answer about."""
    client, server = _client([0.1] * 4, serving=MageVLServingConfig(max_sessions=1))

    with client.websocket_connect("/v1/duplex/mage-vl?session_id=first") as first:
        first.send_json({"type": ev.INPUT_APPEND, "modality": "video", "data": "seg-0"})
        first.receive_json()
        assert server.active_sessions == 1

        with client.websocket_connect("/v1/duplex/mage-vl?session_id=second") as second:
            refusal = second.receive_json()

        assert refusal["type"] == ev.ERROR
        assert "session slot" in refusal["message"]
        first.send_json({"type": ev.CLOSE})

    # The slot comes back once the stream ends, so a client can reconnect.
    with client.websocket_connect("/v1/duplex/mage-vl?session_id=third") as third:
        third.send_json({"type": ev.INPUT_APPEND, "modality": "video", "data": "seg-0"})
        assert third.receive_json()["type"] in {"response.listen", "response.speak"}
        third.send_json({"type": ev.CLOSE})


def test_a_malformed_frame_is_reported_and_the_stream_continues():
    client, _ = _client([0.9])

    with client.websocket_connect("/v1/duplex/mage-vl?session_id=s4") as socket:
        socket.send_json({"no": "type"})
        complaint = socket.receive_json()
        socket.send_json({"type": ev.INPUT_APPEND, "modality": "video", "data": "seg-0"})
        events = _drain(socket, until={ev.RESPONSE_DONE})
        socket.send_json({"type": ev.CLOSE})

    assert complaint["type"] == ev.ERROR
    assert ev.RESPONSE_DONE in [e["type"] for e in events], "a bad frame must not end the stream"


def test_a_question_over_the_socket_answers_a_quiet_stream():
    client, _ = _client([0.1])

    with client.websocket_connect("/v1/duplex/mage-vl?session_id=s5") as socket:
        socket.send_json({"type": ev.INPUT_APPEND, "modality": "video", "data": "seg-0"})
        assert socket.receive_json()["type"] == "response.listen"
        socket.send_json({"type": ev.INPUT_APPEND, "modality": "text", "data": "what is that?"})
        events = _drain(socket, until={ev.RESPONSE_DONE})
        socket.send_json({"type": ev.CLOSE})

    assert ev.RESPONSE_DONE in [e["type"] for e in events]


def test_serving_config_rejects_nonsense():
    with pytest.raises(ValueError, match="max_sessions"):
        MageVLServingConfig(max_sessions=0)
    with pytest.raises(ValueError, match="idle_timeout_s"):
        MageVLServingConfig(idle_timeout_s=0)


def test_stream_config_from_the_server_reaches_the_policy():
    tuned = MageVLStreamConfig(speak_threshold=0.95)
    client, _ = _client([0.9], serving=MageVLServingConfig(stream_config=tuned))

    with client.websocket_connect("/v1/duplex/mage-vl?session_id=s6") as socket:
        socket.send_json({"type": ev.INPUT_APPEND, "modality": "video", "data": "seg-0"})
        decision = socket.receive_json()
        socket.send_json({"type": ev.CLOSE})

    assert decision["type"] == "response.listen", "0.9 is below the server's 0.95 threshold"
    assert decision["metadata"]["threshold"] == pytest.approx(0.95)
