# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright 2026 The HuggingFace Team. All rights reserved.
"""Prompt and action timing semantics from the pinned Hy-Nomas reference.

Source: jiaxiangc/diffusers@12722ef11acf8e335f96cfd4faea17de78cd8909.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm_omni.diffusion.models.hunyuan_video.qwen_prompt import (
    encode_qwen_chat_prompt,
    qwen_chat_crop_start,
    tokenize_qwen_chat_prompt,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, Qwen2_5_VLTextModel

SYSTEM_PROMPT = (
    "You are a helpful assistant. Describe the video by detailing the following aspects: "
    "        1. The main content and theme of the video. "
    "        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
    "        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
    "        4. background environment, light, style and atmosphere. "
    "        5. camera angles, movements, and transitions used in the video."
)


def _tokenize(tokenizer, prompt, max_length=None):
    return tokenize_qwen_chat_prompt(tokenizer, prompt, SYSTEM_PROMPT, max_length)


def _action_variants(prompt):
    try:
        obj = json.loads(prompt)
    except (ValueError, TypeError):
        return []
    segments = obj.get("segments") if isinstance(obj, dict) else None
    if not isinstance(segments, list) or not segments or not isinstance(segments[0], dict):
        return []
    actions = segments[0].get("contained_actions")
    if not isinstance(actions, (dict, list)) or not actions:
        return []
    variants = []
    for count in range(len(actions)):
        segment = dict(segments[0])
        if count == 0:
            segment.pop("contained_actions")
        elif isinstance(actions, dict):
            segment["contained_actions"] = dict(list(actions.items())[:count])
        else:
            segment["contained_actions"] = actions[:count]
        variants.append(
            json.dumps(
                {key: [segment] if key == "segments" else value for key, value in obj.items()}, ensure_ascii=False
            )
        )
    return variants


def _timestamp(value):
    parts = str(value).split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid action timestamp: {value}")
    return sum(float(part) * 60**i for i, part in enumerate(reversed(parts)))


def _action_windows(prompt, frames, fps):
    try:
        obj = json.loads(prompt)
    except (ValueError, TypeError):
        return None
    segments = obj.get("segments") if isinstance(obj, dict) else None
    if not isinstance(segments, list) or not segments or not isinstance(segments[0], dict):
        return None
    actions = segments[0].get("contained_actions", [])
    if isinstance(actions, dict):
        actions = list(actions.values())
    if not isinstance(actions, list) or not actions:
        return None
    spans = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or "start_time" not in action or "end_time" not in action:
            return None
        start, end = [
            max(0, min(frames, int(max(_timestamp(action[key]), 0) * fps) // 4)) for key in ("start_time", "end_time")
        ]
        if end > start:
            spans.append((start, end, index))
    if not spans:
        return None
    spans.sort()
    cursor, windows = 0, []
    for start, end, index in spans:
        if start < cursor:
            raise ValueError("Action time windows must not overlap.")
        if start > cursor:
            windows.append((cursor, start, -1))
        windows.append((start, end, index))
        cursor = end
    if cursor < frames:
        windows.append((cursor, frames, -1))
    return None if len(windows) == len(actions) == 1 else windows


@dataclass
class EncodedPrompt:
    embeds: torch.Tensor
    # None when every token is valid.
    mask: torch.Tensor | None
    action_offsets: list[int] | None
    cache_action_start: int | None


@torch.no_grad()
def encode_segment_prompt(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    text_encoder: Qwen2_5_VLTextModel,
    device: torch.device,
    max_sequence_length: int = 1280,
) -> EncodedPrompt:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Hy-Nomas requires a nonempty segment prompt.")
    if max_sequence_length < 1:
        raise ValueError("max_sequence_length must be positive.")
    crop = qwen_chat_crop_start(tokenizer, SYSTEM_PROMPT)
    length = max_sequence_length
    tokens = _tokenize(tokenizer, prompt, length + crop).to(device)
    embeds, mask = encode_qwen_chat_prompt(text_encoder, tokens, crop, use_cache=False)
    full_ids = tokens["input_ids"][0].tolist()
    boundaries = []
    for variant in _action_variants(prompt):
        variant_ids = _tokenize(tokenizer, variant, length + crop)["input_ids"][0].tolist()
        common = next(
            (i for i, (a, b) in enumerate(zip(full_ids, variant_ids)) if a != b),
            min(len(full_ids), len(variant_ids)),
        )
        boundaries.append(max(boundaries[-1] if boundaries else 0, min(length - 1, common - crop)))
    # Right padding leaves the valid text as a prefix. Dropping the padded tail
    # is equivalent under the key mask and lets attention run without a mask.
    valid = int(mask.sum())
    if bool(mask[0, :valid].all()):
        embeds, mask = embeds[:, :valid], None
    return EncodedPrompt(
        embeds=embeds,
        mask=mask,
        action_offsets=boundaries + [min(length, int(tokens["attention_mask"].sum()))] if boundaries else None,
        cache_action_start=boundaries[0] if boundaries else None,
    )
