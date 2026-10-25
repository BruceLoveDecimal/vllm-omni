# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared Qwen chat tokenization and hidden-state extraction for video models."""


def format_text_input(prompt: list[str], system_message: str) -> list[list[dict[str, str]]]:
    return [
        [{"role": "system", "content": system_message}, {"role": "user", "content": p if p else " "}] for p in prompt
    ]


def tokenize_qwen_chat_prompt(tokenizer, prompts: str | list[str], system_message: str, max_length: int | None = None):
    # Keep the single-prompt template (including an empty user message) distinct
    # from HY1.5's batched template, which substitutes a space for empty prompts.
    messages = (
        [{"role": "system", "content": system_message}, {"role": "user", "content": prompts}]
        if isinstance(prompts, str)
        else format_text_input(prompts, system_message)
    )
    kwargs = {} if max_length is None else {"truncation": True, "max_length": max_length, "padding": "max_length"}
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **kwargs
    )


def qwen_chat_crop_start(tokenizer, system_message: str) -> int:
    marker = tokenizer("<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    ids = tokenize_qwen_chat_prompt(tokenizer, "", system_message)["input_ids"][0].tolist()
    crop = next(
        (i + len(marker) for i in range(len(ids) - len(marker) + 1) if ids[i : i + len(marker)] == marker), None
    )
    if crop is None:
        raise ValueError("The tokenizer chat template must contain the Qwen user marker.")
    return crop


def encode_qwen_chat_prompt(
    text_encoder, tokens, crop_start: int, skip_layers: int = 2, *, use_cache: bool | None = None
):
    """Encode prepared tokens, preserving the caller's fields, device and dtype.

    Tokenization is separate so action-conditioned callers can reuse the exact
    token IDs. Omitting use_cache preserves the encoder's configured default.
    """
    kwargs = {} if use_cache is None else {"use_cache": use_cache}
    embeds = text_encoder(**tokens, output_hidden_states=True, **kwargs).hidden_states[-(skip_layers + 1)]
    return embeds[:, crop_start:], tokens["attention_mask"][:, crop_start:]
