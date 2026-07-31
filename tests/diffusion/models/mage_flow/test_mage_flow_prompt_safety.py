# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


@pytest.mark.parametrize(
    ("raw", "valid"),
    [
        (
            "classifier preamble ```json\n"
            '{"violates": false, "categories": [], '
            '"reason": "literal } and { braces are harmless"}\n```',
            True,
        ),
        (
            '{malformed} {"violates": false, "categories": [], '
            '"reason": "allowed"}',
            False,
        ),
        (
            '{"violates": false, "categories": [], '
            '"reason": "allowed"} denied',
            False,
        ),
        (
            '{"violates": false, "categories": ["violence"], '
            '"reason": "contradiction"}',
            False,
        ),
        (
            '{"violates": true, "categories": [], '
            '"reason": "contradiction"}',
            False,
        ),
        (
            '{"violates": true, "categories": ["unknown"], '
            '"reason": "unsupported"}',
            False,
        ),
    ],
)
def test_filter_verdict_requires_one_consistent_json_object(
    raw: str,
    valid: bool,
):
    if not valid:
        with pytest.raises(ValueError):
            content_policy._parse_filter_verdict(raw)
        return

    verdict = content_policy._parse_filter_verdict(raw)
    assert not verdict.violates
    assert verdict.categories == []
    assert verdict.reason == "literal } and { braces are harmless"


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
    assert "pixel_values" in encoder.generate_calls[1]
    assert "ignored_processor_value" not in encoder.generate_calls[1]


@pytest.mark.parametrize("screen", ["text", "edit"])
def test_screeners_remain_fail_closed_without_leaking_checker_errors(screen: str):
    tokenizer = _FakeTokenizer('{"violates": "no", "categories": [], "reason": "bad schema"}')
    processor = _FakeSafetyProcessor(tokenizer)
    encoder = _FakeSafetyEncoder()

    with pytest.raises(
        content_policy.MageFlowSafetyCheckError,
        match="Mage-Flow content safety check failed",
    ) as exc_info:
        if screen == "text":
            content_policy.screen_mage_flow_prompt(encoder, processor, "a landscape")
        else:
            content_policy.screen_mage_flow_edit_prompt(
                encoder,
                processor,
                "make it brighter",
                [Image.new("RGB", (4, 4))],
            )
    assert "bad schema" not in str(exc_info.value)


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


def test_prompt_encoders_share_model_call_and_preserve_exact_slices():
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
    edit_result = prompt_utils.encode_mage_flow_edit_prompt(
        text_encoder,
        processor,
        "make it brighter",
        [image],
        device=torch.device("cpu"),
    )

    expected = torch.arange(140, dtype=torch.float32).reshape(1, 70, 2)
    torch.testing.assert_close(
        text_result,
        expected[:, prompt_utils.MAGE_FLOW_PROMPT_START_INDEX : 68],
    )
    torch.testing.assert_close(
        edit_result,
        expected[:, prompt_utils.MAGE_FLOW_EDIT_PROMPT_START_INDEX : 68],
    )
    assert "pixel_values" not in model.calls[0]
    assert "pixel_values" in model.calls[1]
    assert "ignored_processor_value" not in model.calls[0]
    assert "ignored_processor_value" not in model.calls[1]
    assert processor.calls[0]["text"] == [prompt_utils.format_mage_flow_prompt("a landscape")]
    assert processor.calls[1]["text"] == [prompt_utils.format_mage_flow_edit_prompt("make it brighter", 1)]


def test_edit_rejects_more_than_three_reference_images():
    with pytest.raises(ValueError, match="between 1 and 3"):
        prompt_utils.validate_reference_image_count([None] * 4)
