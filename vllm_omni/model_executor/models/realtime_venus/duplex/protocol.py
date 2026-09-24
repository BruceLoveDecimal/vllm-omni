# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Realtime-Venus in-stream delegation protocol.

The model asks an external backend for help by writing
``<delegate>query</delegate>`` into its output stream while the conversation
keeps going. The backend's answer comes back later as
``<backend>answer</backend>`` text at the end of an input unit, and the model
decides on its own when and how to deliver it.

On the Realtime endpoint a delegation is a ``delegate`` function call, and the
client answers it with a ``function_call_output`` item.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

DELEGATE_FUNCTION_NAME = "delegate"

#: Control tokens of the duplex format. Client text must not carry them: a
#: backend answer or a user question could otherwise open a second delegation,
#: close the current unit, or switch the model's listen/speak state.
RESERVED_PROTOCOL_TOKENS = (
    "<|speak|>",
    "<|listen|>",
    "<|turn_eos|>",
    "<|chunk_eos|>",
    "<|chunk_tts_eos|>",
    "<|tts_bos|>",
    "<|tts_eos|>",
    "<unit>",
    "</unit>",
    "<image>",
    "</image>",
    "<delegate>",
    "</delegate>",
    "<backend>",
    "</backend>",
)

#: Longest delegation query the tracker buffers before dropping the span.
MAX_DELEGATE_QUERY_TOKENS = 1024


def strip_protocol_tokens(text: str) -> str:
    """Remove every reserved control token, including ones formed by removal."""
    while True:
        cleaned = text
        for token in RESERVED_PROTOCOL_TOKENS:
            cleaned = cleaned.replace(token, "")
        if cleaned == text:
            return cleaned.strip()
        text = cleaned


def backend_text(output: str) -> str:
    """The model-native form of a backend answer to a delegation."""
    return f"<backend>{strip_protocol_tokens(output)}</backend>"


@dataclass(slots=True)
class DelegateSpanTracker:
    """Collect ``<delegate>`` spans from a stream of generated token ids.

    A span may straddle units; the duplex control tokens between its units
    (``ignored_ids``) are not part of the query. A new ``<delegate>`` restarts
    the span, and ``<|turn_eos|>`` drops an unterminated one so it cannot
    swallow the next turn. A span longer than ``max_query_tokens`` is dropped.
    """

    start_id: int
    end_id: int
    turn_eos_id: int | None = None
    ignored_ids: frozenset[int] = frozenset()
    max_query_tokens: int = MAX_DELEGATE_QUERY_TOKENS
    inside: bool = False
    body: list[int] = field(default_factory=list)

    def feed(self, token_ids: Sequence[int]) -> list[list[int]]:
        """Consume generated ids; return the bodies of the spans they close."""
        completed: list[list[int]] = []
        for token_id in token_ids:
            if token_id == self.start_id:
                self.inside = True
                self.body = []
            elif token_id == self.end_id:
                if self.inside and self.body:
                    completed.append(self.body)
                self.inside = False
                self.body = []
            elif token_id == self.turn_eos_id:
                self.inside = False
                self.body = []
            elif self.inside and token_id not in self.ignored_ids:
                if len(self.body) >= self.max_query_tokens:
                    self.inside = False
                    self.body = []
                else:
                    self.body.append(int(token_id))
        return completed


@dataclass(slots=True)
class DelegateStreamCursor:
    """Turn cumulative Stage-0 output ids into delegation queries.

    The Thinker's resumable request reports cumulative output ids. The cursor
    feeds each id to the tracker once, and restarts when the cumulative ids no
    longer extend what it has seen (an epoch reset after barge-in).
    """

    tracker: DelegateSpanTracker
    seen_ids: list[int] = field(default_factory=list)

    def feed_cumulative(self, output_ids: Sequence[int], decode: Callable[[list[int]], str]) -> list[str]:
        seen = len(self.seen_ids)
        if len(output_ids) < seen or list(output_ids[:seen]) != self.seen_ids:
            seen = 0
            self.tracker.inside = False
            self.tracker.body = []
        self.seen_ids = [int(token_id) for token_id in output_ids]
        queries = []
        for body in self.tracker.feed(self.seen_ids[seen:]):
            query = decode(body).strip()
            if query:
                queries.append(query)
        return queries


__all__ = [
    "DELEGATE_FUNCTION_NAME",
    "MAX_DELEGATE_QUERY_TOKENS",
    "RESERVED_PROTOCOL_TOKENS",
    "DelegateSpanTracker",
    "DelegateStreamCursor",
    "backend_text",
    "strip_protocol_tokens",
]
