# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Realtime-Venus delegation protocol helpers."""

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.realtime_venus.duplex.protocol import (
    DelegateSpanTracker,
    DelegateStreamCursor,
    backend_text,
    strip_protocol_tokens,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

START, END, TURN_EOS = 900, 901, 902


def _tracker(**kwargs) -> DelegateSpanTracker:
    return DelegateSpanTracker(start_id=START, end_id=END, turn_eos_id=TURN_EOS, **kwargs)


def test_strip_protocol_tokens_removes_markers_formed_by_removal() -> None:
    assert strip_protocol_tokens("  <back<delegate>end>answer</backend> ") == "answer"
    assert strip_protocol_tokens("<|turn_eos|>hi<unit>") == "hi"


def test_backend_text_wraps_a_sanitized_answer() -> None:
    assert backend_text("it is <delegate>sunny</delegate>") == "<backend>it is sunny</backend>"
    assert backend_text("") == "<backend></backend>"


def test_tracker_collects_a_span_that_straddles_units() -> None:
    chunk_eos, speak = 903, 904
    tracker = _tracker(ignored_ids=frozenset({chunk_eos, speak}))
    assert tracker.feed([1, START, 10, 11, chunk_eos]) == []
    assert tracker.inside is True
    assert tracker.feed([speak, 12, END, 2]) == [[10, 11, 12]]
    assert tracker.inside is False


def test_tracker_drops_unterminated_and_empty_spans() -> None:
    tracker = _tracker()
    assert tracker.feed([START, 10, TURN_EOS, 11, END]) == []
    assert tracker.feed([START, END]) == []
    # A second opening restarts the span.
    assert tracker.feed([START, 10, START, 11, END]) == [[11]]


def test_tracker_drops_a_span_longer_than_its_limit() -> None:
    tracker = _tracker(max_query_tokens=2)
    assert tracker.feed([START, 10, 11, 12, END]) == []
    assert tracker.feed([START, 13, END]) == [[13]]


def test_cursor_feeds_each_cumulative_id_once_and_restarts_on_rewind() -> None:
    cursor = DelegateStreamCursor(tracker=_tracker())

    def decode(ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in ids)

    assert cursor.feed_cumulative([1, START, 10], decode) == []
    assert cursor.feed_cumulative([1, START, 10, 11, END], decode) == ["10 11"]
    # Replaying the same cumulative ids does not emit the span again.
    assert cursor.feed_cumulative([1, START, 10, 11, END], decode) == []
    # A shorter or diverging history is a new request incarnation.
    assert cursor.feed_cumulative([START, 20, END], decode) == ["20"]
