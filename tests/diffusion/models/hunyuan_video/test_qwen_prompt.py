# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
import inspect
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import BatchEncoding

from vllm_omni.diffusion.models.hunyuan_video import pipeline_hunyuan_video_1_5 as hv
from vllm_omni.diffusion.models.hunyuan_video.qwen_prompt import (
    encode_qwen_chat_prompt,
    qwen_chat_crop_start,
    tokenize_qwen_chat_prompt,
)
from vllm_omni.diffusion.models.hy_nomas.prompt import SYSTEM_PROMPT

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_system_prompts_match_exactly():
    # Read the literal without constructing/loading the HY1.5 pipeline.
    tree = ast.parse(inspect.getsource(hv))
    assignments = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and any(isinstance(target, ast.Attribute) and target.attr == "system_message" for target in node.targets)
    ]
    assert assignments == [SYSTEM_PROMPT]


@pytest.mark.parametrize("prompts,content", [("", ""), ([""], " "), (["fox", ""], "fox")])
def test_chat_template_preserves_empty_and_batched_inputs(prompts, content):
    tokenizer = Mock()
    tokenize_qwen_chat_prompt(tokenizer, prompts, SYSTEM_PROMPT, 120)
    args, kwargs = tokenizer.apply_chat_template.call_args
    messages = args[0] if isinstance(prompts, str) else args[0][0]
    assert messages == [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
    assert kwargs == dict(
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        truncation=True,
        max_length=120,
        padding="max_length",
    )


def test_dynamic_crop_uses_marker_and_rejects_missing_marker():
    tokenizer = Mock(return_value={"input_ids": [7, 8]})
    tokenizer.apply_chat_template.return_value = {"input_ids": torch.tensor([[1, 2, 7, 8, 9]])}
    assert qwen_chat_crop_start(tokenizer, SYSTEM_PROMPT) == 4
    tokenizer.apply_chat_template.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    with pytest.raises(ValueError, match="user marker"):
        qwen_chat_crop_start(tokenizer, SYSTEM_PROMPT)


@pytest.mark.parametrize("use_cache", [None, False])
@pytest.mark.parametrize("skip_layers", [0, 2])
def test_encoder_preserves_fields_slicing_and_cache_default(use_cache, skip_layers):
    tokens = {"input_ids": torch.arange(12).reshape(2, 6), "attention_mask": torch.ones(2, 6)}
    hidden = [torch.randn(2, 6, 4) for _ in range(4)]
    encoder = Mock(return_value=SimpleNamespace(hidden_states=hidden))
    embeds, mask = encode_qwen_chat_prompt(encoder, tokens, 2, skip_layers, use_cache=use_cache)
    torch.testing.assert_close(embeds, hidden[-(skip_layers + 1)][:, 2:], atol=0, rtol=0)
    assert torch.equal(mask, tokens["attention_mask"][:, 2:])
    assert encoder.call_args.kwargs.keys() == tokens.keys() | {"output_hidden_states"} | (
        set() if use_cache is None else {"use_cache"}
    )
    if use_cache is False:
        assert encoder.call_args.kwargs["use_cache"] is False


def test_hy15_wrapper_keeps_fixed_crop_dtype_and_encoder_fields():
    tokens = BatchEncoding(
        {"input_ids": torch.arange(112).reshape(1, 112), "attention_mask": torch.ones(1, 112), "unused": torch.zeros(1)}
    )
    hidden = [torch.randn(1, 112, 4) for _ in range(4)]
    pipeline = SimpleNamespace(
        tokenizer=Mock(),
        text_encoder=Mock(return_value=SimpleNamespace(hidden_states=hidden)),
        system_message=SYSTEM_PROMPT,
        tokenizer_max_length=4,
        prompt_template_encode_start_idx=108,
    )
    pipeline.tokenizer.apply_chat_template.return_value = tokens
    embeds, mask = hv.HunyuanVideo15Pipeline._get_mllm_prompt_embeds(
        pipeline, ["fox"], torch.device("cpu"), torch.bfloat16
    )
    torch.testing.assert_close(embeds, hidden[-3][:, 108:].to(torch.bfloat16), atol=0, rtol=0)
    assert torch.equal(mask, tokens.attention_mask[:, 108:])
    assert set(pipeline.text_encoder.call_args.kwargs) == {"input_ids", "attention_mask", "output_hidden_states"}


def test_real_qwen_template_crop():
    from transformers import Qwen2Tokenizer

    model = os.environ.get("HY_NOMAS_MODEL_PATH")
    if model is None:
        pytest.skip("Set HY_NOMAS_MODEL_PATH to the local released checkpoint")
    tokenizer = Qwen2Tokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=True)
    assert qwen_chat_crop_start(tokenizer, SYSTEM_PROMPT) == 108
