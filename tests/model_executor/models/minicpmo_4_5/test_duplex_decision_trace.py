# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the opt-in MiniCPM-o 4.5 native duplex decision trace.

The trace exists to attribute a missing soft-interrupt handoff: a mid-turn
``listen`` is rewritten to ``tts_bos`` so the turn keeps speaking, and only
``turn_eos`` leaves a turn. It must stay off unless explicitly enabled, and it
must report the rewrite that a raw sampled-token log would hide.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MINICPMO45_DUPLEX_TRACE_ENV,
    MiniCPMO45OmniForConditionalGeneration,
    minicpmo45_duplex_decision_name,
    minicpmo45_duplex_trace_enabled,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TOKEN_IDS = {
    "listen_token_id": 10,
    "speak_token_id": 11,
    "tts_bos_token_id": 12,
    "turn_eos_token_id": 13,
    "chunk_eos_token_id": 14,
}


def _bare_model() -> MiniCPMO45OmniForConditionalGeneration:
    """An uninitialized instance: the trace helpers own all state they touch."""
    return MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), ("", False), ("0", False), ("false", False), ("off", False), ("1", True), ("true", True)],
)
def test_trace_is_opt_in(monkeypatch, value: str | None, expected: bool) -> None:
    monkeypatch.delenv(MINICPMO45_DUPLEX_TRACE_ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(MINICPMO45_DUPLEX_TRACE_ENV, value)

    assert minicpmo45_duplex_trace_enabled() is expected


def test_decision_name_maps_special_tokens_and_falls_back_to_text():
    assert minicpmo45_duplex_decision_name(13, TOKEN_IDS) == "turn_eos"
    assert minicpmo45_duplex_decision_name(12, TOKEN_IDS) == "tts_bos"
    # Ordinary vocabulary tokens are the model speaking, not a unit decision.
    assert minicpmo45_duplex_decision_name(9999, TOKEN_IDS) == "text:9999"


def test_trace_reports_the_listen_to_tts_bos_rewrite(caplog):
    model = _bare_model()
    state = SimpleNamespace(session_id="sess-a", audio_chunk_idx=10, current_turn_ended=False)
    model._minicpmo45_duplex_listen_rewrites()[0] = True

    with caplog.at_level(logging.INFO):
        model._trace_minicpmo45_duplex_decision(
            0,
            TOKEN_IDS["tts_bos_token_id"],
            TOKEN_IDS,
            state=state,
            payload={"is_speech": True},
        )

    record = caplog.text
    assert "unit=10" in record
    assert "decision=tts_bos" in record
    assert "rewritten_from_listen=True" in record
    assert "is_speech=True" in record


def test_rewrite_flag_does_not_leak_into_the_next_decision(caplog):
    model = _bare_model()
    state = SimpleNamespace(session_id="sess-a", audio_chunk_idx=11, current_turn_ended=False)
    model._minicpmo45_duplex_listen_rewrites()[0] = True

    with caplog.at_level(logging.INFO):
        model._trace_minicpmo45_duplex_decision(0, TOKEN_IDS["tts_bos_token_id"], TOKEN_IDS, state=state, payload=None)
        caplog.clear()
        model._trace_minicpmo45_duplex_decision(0, TOKEN_IDS["turn_eos_token_id"], TOKEN_IDS, state=state, payload=None)

    assert "decision=turn_eos" in caplog.text
    assert "rewritten_from_listen=False" in caplog.text
