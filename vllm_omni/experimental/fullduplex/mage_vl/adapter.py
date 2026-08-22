# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mage-VL's duplex adapter: a video stream in, proactive answers out.

Mage-VL is proactive rather than turn-taking -- nobody presses "send". Its cognition gate
scores each incoming segment, and when the score crosses the threshold the model speaks on
its own. That maps onto the core contracts without extending them: ``core.DuplexRuntime``
already starts a response when a session is ``proactive`` and the adapter's
``should_respond`` returns true, so the gate *is* ``should_respond``.

Everything model-heavy is injected. ``score_segment`` runs the vision tower and gate;
``generate`` runs the decoder. Keeping them as callables is what lets the whole streaming
contract -- proactive triggers, cooldown, barge-in, staleness -- be tested on CPU against
the real runtime, with the model stubbed.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from vllm_omni.experimental.fullduplex.core.adapter import (
    DuplexAdapter,
    DuplexCapability,
    OutputChunk,
)
from vllm_omni.experimental.fullduplex.core.session import DuplexSession
from vllm_omni.experimental.fullduplex.mage_vl.policy import (
    Decision,
    MageVLGatePolicy,
    MageVLStreamConfig,
)

ScoreSegment = Callable[[Any], float | Awaitable[float]]
Generate = Callable[[list[dict[str, Any]]], AsyncIterator[str]]
EmitDecision = Callable[[dict[str, Any]], Awaitable[None]]

DEFAULT_PROMPT = "Describe what is happening right now."


class MageVLDuplexAdapter(DuplexAdapter):
    """Gate-driven proactive streaming for Mage-VL."""

    def __init__(
        self,
        score_segment: ScoreSegment,
        generate: Generate,
        *,
        config: MageVLStreamConfig | None = None,
        on_decision: EmitDecision | None = None,
        default_prompt: str = DEFAULT_PROMPT,
    ) -> None:
        self._score_segment = score_segment
        self._generate = generate
        self._on_decision = on_decision
        self._default_prompt = default_prompt
        self.policy = MageVLGatePolicy(config)
        self._pending_query: str | None = None

    def capabilities(self) -> DuplexCapability:
        return DuplexCapability(
            frozenset({"video", "text"}), frozenset({"text"}), proactive=True
        )

    async def on_input(self, session: DuplexSession, modality: str, data: Any) -> None:
        if modality == "text":
            # A question asked mid-stream answers at the next opportunity rather than
            # waiting on the gate: the user speaking is itself the trigger.
            self._pending_query = data
            return

        score = self._score_segment(data)
        if inspect.isawaitable(score):
            score = await score
        decision = self.policy.observe(data, float(score))
        await self._report(session, decision, float(score))

    def should_respond(self, session: DuplexSession) -> bool:
        return self._pending_query is not None or self.policy.decision is Decision.SPEAK

    async def respond(self, session: DuplexSession) -> AsyncIterator[OutputChunk]:
        query = self._pending_query or self._default_prompt
        self._pending_query = None
        window = self.policy.window()
        self.policy.note_response_started()

        async for text in self._generate(self._build_messages(window, query)):
            yield OutputChunk("text", text)

    async def on_barge_in(self, session: DuplexSession) -> None:
        self._pending_query = None
        self.policy.note_interrupted()

    def _build_messages(self, window: list[Any], query: str) -> list[dict[str, Any]]:
        """The sliding window as one user turn: segments in order, then the question.

        Segments enter as images because that is how video reaches this model -- one
        expanded block per frame or canvas, never a separate video tensor.
        """
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": segment}} for segment in window
        ]
        content.append({"type": "text", "text": query})
        return [{"role": "user", "content": content}]

    async def _report(self, session: DuplexSession, decision: Decision, score: float) -> None:
        """Publish the gate's verdict on the existing decision events.

        ``response.speak`` / ``response.listen`` already exist in the Realtime projection
        and carry decision metadata rather than transcript text, which is exactly what a
        gate produces -- so the gate does not get an event type of its own.
        """
        if self._on_decision is None:
            return
        await self._on_decision(
            {
                "type": "response.speak" if decision is Decision.SPEAK else "response.listen",
                "session_id": session.session_id,
                "epoch": session.epoch,
                "model_speak": decision is Decision.SPEAK,
                "metadata": {
                    "p_speak": score,
                    "threshold": self.policy.config.speak_threshold,
                    "segment_index": self.policy.segments_seen - 1,
                    "segment_seconds": self.policy.config.segment_seconds,
                },
            }
        )


__all__ = ["MageVLDuplexAdapter", "DEFAULT_PROMPT"]
