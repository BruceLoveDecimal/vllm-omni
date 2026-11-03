# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prompt interaction handler for LingBot-World stepwise sessions.

LingBot pads every prompt to ``max_sequence_length`` and zeroes the padding,
so prompts of any length share one shape and the generic handler's embedding
interpolation applies unchanged. Only the encoder differs: LingBot's
``encode_prompt`` takes the text plus a sequence length and dtype and returns
one tensor, without negative prompts or classifier-free guidance.
"""

from __future__ import annotations

from typing import Any

import torch
from typing_extensions import override

from vllm_omni.diffusion.interaction.modality_handlers.prompt import PromptInteractionHandler


class LingBotPromptInteractionHandler(PromptInteractionHandler):
    @classmethod
    @override
    def from_pipeline(cls, pipeline: Any) -> LingBotPromptInteractionHandler:
        def encode_prompt(*, prompt: str, max_sequence_length: int | None, dtype: torch.dtype, **_: Any):
            # The generic handler also passes CFG and batching arguments that LingBot has no use for.
            if max_sequence_length is None:
                # A served request leaves the length unset and LingBot then uses its fixed one, as
                # _parse_request does; the update has to match the length the session started with.
                from vllm_omni.diffusion.models.lingbot_world.pipeline import _MAX_SEQUENCE_LENGTH

                max_sequence_length = _MAX_SEQUENCE_LENGTH
            return pipeline.encode_prompt(prompt, max_sequence_length=max_sequence_length, dtype=dtype), None

        return cls(encode_prompt=encode_prompt, device=pipeline.device, dtype=pipeline.transformer.dtype)
