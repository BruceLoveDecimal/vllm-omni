# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MiniCPM-o 4.5 streaming audio cache compatibility."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.whisper.modeling_whisper import WhisperConfig

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    WHISPER_ATTENTION_CLASSES,
    MiniCPMWhisperEncoderLayer,
    _get_audio_cache_length,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _LegacyWhisperAttention(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.seen_cache = None

    def forward(self, hidden_states, past_key_value=None, **kwargs):
        self.seen_cache = past_key_value
        return hidden_states, None, past_key_value


class _CurrentWhisperAttention(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.seen_cache = None

    def forward(self, hidden_states, past_key_values=None, **kwargs):
        self.seen_cache = past_key_values
        return hidden_states, None


@pytest.mark.parametrize(
    ("implementation", "attention_cls"),
    [
        ("test_legacy_cache", _LegacyWhisperAttention),
        ("test_current_cache", _CurrentWhisperAttention),
    ],
)
def test_whisper_attention_preserves_streaming_cache(
    monkeypatch: pytest.MonkeyPatch,
    implementation: str,
    attention_cls: type[nn.Module],
) -> None:
    monkeypatch.setitem(WHISPER_ATTENTION_CLASSES, implementation, attention_cls)
    config = WhisperConfig(
        d_model=4,
        encoder_attention_heads=1,
        encoder_ffn_dim=8,
        encoder_layers=1,
    )
    config._attn_implementation = implementation
    layer = MiniCPMWhisperEncoderLayer(config, layer_idx=0)
    cache = object()

    outputs = layer(
        torch.randn(1, 2, config.d_model),
        attention_mask=None,
        layer_head_mask=None,
        past_key_values=cache,
        use_cache=True,
    )

    assert layer.self_attn.seen_cache is cache
    assert outputs[-1] is cache


def test_audio_cache_length_uses_current_cache_api() -> None:
    dynamic_cache = SimpleNamespace(get_seq_length=lambda: 7)
    encoder_decoder_cache = SimpleNamespace(self_attention_cache=dynamic_cache)

    assert _get_audio_cache_length(encoder_decoder_cache) == 7


def test_audio_cache_length_supports_legacy_cache() -> None:
    key = torch.zeros(1, 1, 5, 2)
    value = torch.zeros_like(key)

    assert _get_audio_cache_length(((key, value),)) == 5
