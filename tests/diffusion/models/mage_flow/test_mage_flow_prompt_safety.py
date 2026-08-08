# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prompt encoding and content screening share one code path per concern.

Both the T2I/Edit prompt encoders and the text/edit screeners were merged into
single implementations. These tests pin the behaviour each branch must keep:
the exact conditioning slice, the vision-key filtering, and fail-closed
screening.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from PIL import Image

from vllm_omni.diffusion.models.mage_flow import content_policy, prompt_utils

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = None

    def __init__(self, raw: str):
        self.raw = raw

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return "rendered text policy prompt"

    def __call__(self, text, **kwargs):
        del text, kwargs
        return {
            "input_ids": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }

    def decode(self, token_ids, **kwargs):
        del token_ids, kwargs
        return self.raw


class _FakeSafetyProcessor:
    def __init__(self, tokenizer: _FakeTokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return "rendered edit policy prompt"

    def __call__(self, **kwargs):
        del kwargs
        return {
            "input_ids": torch.tensor([[20, 21, 22, 23]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "pixel_values": torch.ones(1, 3, 2, 2),
            "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
            "mm_token_type_ids": torch.ones(1, 4, dtype=torch.long),
            "ignored_processor_value": torch.ones(1),
        }


class _FakeSafetyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        suffix = kwargs["input_ids"].new_tensor([[99]])
        return torch.cat([kwargs["input_ids"], suffix], dim=1)


def test_text_and_edit_screeners_share_generation_and_schema_path():
    raw = '{"violates": false, "categories": [], "reason": "allowed"}'
    tokenizer = _FakeTokenizer(raw)
    processor = _FakeSafetyProcessor(tokenizer)
    encoder = _FakeSafetyEncoder()

    text_verdict = content_policy.screen_mage_flow_prompt(
        encoder,
        processor,
        "a landscape",
    )
    edit_verdict = content_policy.screen_mage_flow_edit_prompt(
        encoder,
        processor,
        "make it brighter",
        [Image.new("RGB", (4, 4))],
    )

    assert text_verdict == content_policy.FilterVerdict(False, [], "allowed", raw=raw)
    assert edit_verdict == content_policy.FilterVerdict(False, [], "allowed", raw=raw)
    assert len(encoder.generate_calls) == 2
    assert encoder.generate_calls[0]["max_new_tokens"] == 160
    assert encoder.generate_calls[1]["max_new_tokens"] == 192
    # Only the edit branch conditions on vision, and neither branch forwards
    # processor keys the model does not accept.
    assert "pixel_values" not in encoder.generate_calls[0]
    assert "pixel_values" in encoder.generate_calls[1]
    assert "ignored_processor_value" not in encoder.generate_calls[1]


@pytest.mark.parametrize("screen", ["text", "edit"])
def test_screeners_remain_fail_closed(screen: str):
    """A checker failure must never turn into an unscreened generation."""
    tokenizer = _FakeTokenizer('{"violates": "no", "categories": [], "reason": "bad schema"}')
    processor = _FakeSafetyProcessor(tokenizer)
    encoder = _FakeSafetyEncoder()

    if screen == "text":
        verdict = content_policy.screen_mage_flow_prompt(encoder, processor, "a landscape")
    else:
        verdict = content_policy.screen_mage_flow_edit_prompt(
            encoder,
            processor,
            "make it brighter",
            [Image.new("RGB", (4, 4))],
        )

    assert verdict.violates
    assert verdict.categories == ["safety_check_error"]


def test_edit_screening_falls_back_to_text_when_no_references():
    raw = '{"violates": false, "categories": [], "reason": "allowed"}'
    processor = _FakeSafetyProcessor(_FakeTokenizer(raw))
    encoder = _FakeSafetyEncoder()

    verdict = content_policy.screen_mage_flow_edit_prompt(encoder, processor, "a landscape", [])

    assert verdict == content_policy.FilterVerdict(False, [], "allowed", raw=raw)
    assert "pixel_values" not in encoder.generate_calls[0]


class _FakeConditioningModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        sequence_length = kwargs["input_ids"].shape[1]
        hidden_states = torch.arange(sequence_length * 2, dtype=torch.float32).reshape(1, sequence_length, 2)
        return SimpleNamespace(last_hidden_state=hidden_states)


class _FakePromptProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        sequence_length = 70
        inputs = {
            "input_ids": torch.arange(sequence_length).reshape(1, -1),
            "attention_mask": torch.tensor([[1] * 68 + [0, 0]]),
            "position_ids": torch.arange(sequence_length).reshape(1, -1),
            "ignored_processor_value": torch.ones(1),
        }
        if "images" in kwargs:
            inputs.update(
                {
                    "pixel_values": torch.ones(1, 3, 2, 2),
                    "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
                    "mm_token_type_ids": torch.ones(1, sequence_length, dtype=torch.long),
                }
            )
        return inputs


def test_prompt_encoder_preserves_exact_slices_for_both_branches():
    """One encoder serves T2I and Edit; each keeps its own template and slice."""
    model = _FakeConditioningModel()
    text_encoder = SimpleNamespace(model=model)
    processor = _FakePromptProcessor()
    image = Image.new("RGB", (4, 4))

    text_result = prompt_utils.encode_mage_flow_prompt(
        text_encoder,
        processor,
        "a landscape",
        device=torch.device("cpu"),
    )
    edit_result = prompt_utils.encode_mage_flow_prompt(
        text_encoder,
        processor,
        "make it brighter",
        device=torch.device("cpu"),
        reference_images=[image],
    )

    expected = torch.arange(140, dtype=torch.float32).reshape(1, 70, 2)
    # Slice runs from the template's prefix length to the last valid token, so
    # padding never reaches the conditioning tensor.
    torch.testing.assert_close(
        text_result,
        expected[:, prompt_utils.MAGE_FLOW_PROMPT_START_INDEX : 68],
    )
    torch.testing.assert_close(
        edit_result,
        expected[:, prompt_utils.MAGE_FLOW_EDIT_PROMPT_START_INDEX : 68],
    )
    # Vision keys reach the model only on the Edit branch, and processor keys
    # the model does not accept reach it on neither.
    assert "pixel_values" not in model.calls[0]
    assert "pixel_values" in model.calls[1]
    assert "ignored_processor_value" not in model.calls[0]
    assert "ignored_processor_value" not in model.calls[1]
    assert processor.calls[0]["text"] == [prompt_utils.format_mage_flow_prompt("a landscape")]
    assert processor.calls[1]["text"] == [prompt_utils.format_mage_flow_edit_prompt("make it brighter", 1)]
