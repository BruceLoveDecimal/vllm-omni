# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for Mage-VL's proactive video streaming.

These drive the real ``core.DuplexRuntime`` with the model stubbed out: the gate is a list
of scores and the decoder is a list of text chunks. What is under test is the streaming
behaviour -- when a proactive response fires, what it looks at, what happens when the user
interrupts -- not the model's numbers, which are measured in
``tests/model_executor/models/mage_vl/parity/run_gate_parity.py``.
"""

import asyncio

import pytest

from vllm_omni.experimental.fullduplex.core import protocol as ev
from vllm_omni.experimental.fullduplex.core.runtime import DuplexRuntime
from vllm_omni.experimental.fullduplex.core.session import DuplexSession, DuplexSessionConfig
from vllm_omni.experimental.fullduplex.mage_vl import (
    Decision,
    MageVLDuplexAdapter,
    MageVLGatePolicy,
    MageVLStreamConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _adapter(scores, *, config=None, chunks=("hello", " there"), slow=False):
    """An adapter whose gate replays ``scores`` and whose decoder replays ``chunks``."""
    pending = list(scores)
    decisions: list[dict] = []
    prompts: list[list[dict]] = []

    def score_segment(_segment):
        return pending.pop(0) if pending else 0.0

    async def generate(messages):
        prompts.append(messages)
        for chunk in chunks:
            if slow:
                await asyncio.sleep(0.01)
            yield chunk

    async def on_decision(event):
        decisions.append(event)

    adapter = MageVLDuplexAdapter(
        score_segment, generate, config=config, on_decision=on_decision
    )
    return adapter, decisions, prompts


def _drive(adapter, events):
    session = DuplexSession(
        session_id="mage", config=DuplexSessionConfig(input_modalities=("video",), proactive=True)
    )
    runtime = DuplexRuntime(session, adapter)

    async def stream():
        for event in events:
            await asyncio.sleep(0.005)
            yield event

    emitted: list[dict] = []

    async def emit(event):
        emitted.append(event)

    asyncio.run(runtime.run(stream(), emit))
    return emitted, session


def _segments(count, start=0):
    return [
        {"type": ev.INPUT_APPEND, "modality": "video", "data": f"segment-{i}"}
        for i in range(start, start + count)
    ]


def test_quiet_stream_produces_no_response():
    """Below the threshold the model stays silent -- no response, no output at all."""
    adapter, decisions, prompts = _adapter([0.1, 0.2, 0.05])

    emitted, _ = _drive(adapter, [*_segments(3), {"type": ev.CLOSE}])

    assert [e["type"] for e in emitted] == []
    assert prompts == []
    assert [d["type"] for d in decisions] == ["response.listen"] * 3


def test_gate_crossing_the_threshold_speaks_on_its_own():
    """No commit, no response.create -- the segment itself triggers the answer."""
    adapter, decisions, prompts = _adapter([0.1, 0.9])

    emitted, _ = _drive(adapter, [*_segments(2), {"type": ev.CLOSE}])

    types = [e["type"] for e in emitted]
    assert types.count(ev.RESPONSE_CREATED) == 1
    assert types.count(ev.RESPONSE_DONE) == 1
    assert [e["data"] for e in emitted if e["type"] == ev.RESPONSE_DELTA] == ["hello", " there"]
    assert [d["type"] for d in decisions] == ["response.listen", "response.speak"]
    assert len(prompts) == 1


def test_decision_events_carry_the_score_and_reuse_existing_types():
    """The gate reports on ``response.listen`` / ``response.speak``, not a new event type."""
    adapter, decisions, _ = _adapter([0.9], config=MageVLStreamConfig(speak_threshold=0.5))

    _drive(adapter, [*_segments(1), {"type": ev.CLOSE}])

    speak = decisions[0]
    assert speak["type"] == "response.speak"
    assert speak["model_speak"] is True
    assert speak["metadata"]["p_speak"] == pytest.approx(0.9)
    assert speak["metadata"]["threshold"] == pytest.approx(0.5)
    assert speak["metadata"]["segment_index"] == 0
    assert speak["session_id"] == "mage"


def test_cooldown_stops_the_model_talking_over_itself():
    """The gate stays confident while the scene lasts; the cooldown is what shuts it up."""
    adapter, decisions, prompts = _adapter([0.9, 0.9, 0.9], config=MageVLStreamConfig(cooldown_segments=1))

    emitted, _ = _drive(adapter, [*_segments(3), {"type": ev.CLOSE}])

    assert len(prompts) == 2, "segments 1 and 3 answer; segment 2 falls in the cooldown"
    assert [d["type"] for d in decisions] == [
        "response.speak",
        "response.listen",
        "response.speak",
    ]
    assert [e["type"] for e in emitted].count(ev.RESPONSE_CREATED) == 2


def test_a_question_mid_stream_answers_without_waiting_for_the_gate():
    adapter, _, prompts = _adapter([0.1, 0.1])

    _drive(
        adapter,
        [
            *_segments(2),
            {"type": ev.INPUT_APPEND, "modality": "text", "data": "what is he holding?"},
            {"type": ev.RESPONSE_CREATE},
            {"type": ev.CLOSE},
        ],
    )

    assert len(prompts) == 1
    assert prompts[0][0]["content"][-1] == {"type": "text", "text": "what is he holding?"}


def test_response_looks_at_the_recent_window_only():
    """The gate's own state spans the session; the prompt is bounded."""
    adapter, _, prompts = _adapter([0.1] * 4 + [0.9], config=MageVLStreamConfig(window_segments=3))

    _drive(adapter, [*_segments(5), {"type": ev.CLOSE}])

    content = prompts[0][0]["content"]
    images = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
    assert images == ["segment-2", "segment-3", "segment-4"], "oldest segments fall out"
    assert content[-1]["type"] == "text", "the question comes after what it is about"


def test_barge_in_drops_the_in_flight_answer_and_the_standing_trigger():
    """Interrupting mid-answer must not leak stale deltas or immediately re-fire."""
    adapter, _, prompts = _adapter([0.9], chunks=[f"chunk-{i}" for i in range(50)], slow=True)

    emitted, session = _drive(
        adapter,
        [*_segments(1), {"type": ev.RESPONSE_CANCEL}, {"type": ev.CLOSE}],
    )

    types = [e["type"] for e in emitted]
    assert ev.RESPONSE_CANCELLED in types
    assert ev.RESPONSE_DONE not in types, "a cancelled response must not report done"
    cancel_at = types.index(ev.RESPONSE_CANCELLED)
    assert ev.RESPONSE_DELTA not in types[cancel_at:], "no output after the interruption"
    assert session.epoch == 1, "barge-in advances the epoch exactly once"
    assert adapter.policy.decision is Decision.LISTEN
    assert len(prompts) == 1, "the cancelled response is not retried on its own"


def test_barge_in_keeps_the_window_the_user_is_interrupting_about():
    adapter, _, _ = _adapter([0.9, 0.1])

    _drive(
        adapter,
        [*_segments(1), {"type": ev.RESPONSE_CANCEL}, *_segments(1, start=1), {"type": ev.CLOSE}],
    )

    assert adapter.policy.window() == ["segment-0", "segment-1"]


def test_unsupported_modality_is_refused_not_ignored():
    adapter, _, _ = _adapter([0.9])

    emitted, _ = _drive(
        adapter,
        [{"type": ev.INPUT_APPEND, "modality": "audio", "data": b""}, {"type": ev.CLOSE}],
    )

    assert [e["type"] for e in emitted] == [ev.ERROR]


# --- policy, on its own -------------------------------------------------------------


def test_threshold_is_inclusive_and_cooldown_counts_segments():
    policy = MageVLGatePolicy(MageVLStreamConfig(speak_threshold=0.5, cooldown_segments=2))

    assert policy.observe("a", 0.5) is Decision.SPEAK, "at the threshold counts as speak"
    policy.note_response_started()
    assert policy.observe("b", 0.99) is Decision.LISTEN
    assert policy.observe("c", 0.99) is Decision.LISTEN
    assert policy.observe("d", 0.99) is Decision.SPEAK, "cooldown covers exactly two segments"


def test_policy_rejects_nonsense_configuration():
    with pytest.raises(ValueError, match="speak_threshold"):
        MageVLStreamConfig(speak_threshold=1.5)
    with pytest.raises(ValueError, match="window_segments"):
        MageVLStreamConfig(window_segments=0)
    with pytest.raises(ValueError, match="cooldown_segments"):
        MageVLStreamConfig(cooldown_segments=-1)


def test_reset_returns_a_fresh_session():
    policy = MageVLGatePolicy(MageVLStreamConfig(window_segments=2))
    policy.observe("a", 0.9)
    policy.reset()

    assert policy.window() == []
    assert policy.segments_seen == 0
    assert policy.last_score is None
    assert policy.decision is Decision.LISTEN
