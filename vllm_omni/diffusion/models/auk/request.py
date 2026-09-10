# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AuK request parsing shared by offline generation and speech serving."""

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class AuKRequest:
    instruction: str
    audio: Any
    seconds: float | None


def parse_request(prompt: str | dict, extra: dict) -> AuKRequest:
    """Explicit instruction means editing; input/text means speech synthesis."""
    if isinstance(prompt, str):
        prompt = {"input": prompt}
    if not isinstance(prompt, dict):
        raise ValueError("AuK expects a text string or a structured prompt")
    mm = prompt.get("multi_modal_data") or {}
    options = {**(prompt.get("mm_processor_kwargs") or {}), **extra}
    audio = prompt.get("ref_audio")
    if audio is None:
        audio = mm.get("audio")
    instruction = prompt.get("instruction")
    if instruction is None:
        text = prompt.get("input") or prompt.get("text") or prompt.get("prompt")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("AuK requires non-empty input text or an instruction")
        style = prompt.get("instruct") or options.get("instruct")
        if style:
            instruction = f'Based on the description: "{style}", generate speech saying: "{text}".'
        elif audio is not None:
            instruction = f'Say the following with the same voice: "{text}"'
        else:
            instruction = f'Generate speech saying: "{text}".'
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    seconds = prompt.get("gen_seconds", options.get("gen_seconds"))
    if seconds is not None:
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("gen_seconds must be a finite positive number")
        seconds = float(seconds)
    if seconds is None and audio is None:
        raise ValueError("gen_seconds is required when no reference audio is supplied")
    return AuKRequest(instruction, audio, seconds)
