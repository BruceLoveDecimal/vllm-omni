# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bridge rejections for the MAMMOTH-MODA2 AR -> DiT stage input processor."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.errors import StageInputProcessingError
from vllm_omni.model_executor.stage_input_processors.mammoth_moda2 import ar2dit

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _ar_output(multimodal_output):
    completion = SimpleNamespace(
        cumulative_token_ids=[11, 12, 13],
        multimodal_output=multimodal_output,
    )
    return SimpleNamespace(request_id="req-mammoth", prompt_token_ids=[1, 2], outputs=[completion])


def test_ar2dit_without_latent_raises_stage_input_error() -> None:
    """The DiT stage cannot be conditioned without the AR latent. Typing the
    refusal keeps the processor's own message in the client-facing error instead
    of it being reported as an internal error (a processor bug)."""
    with pytest.raises(StageInputProcessingError, match="missing latent multimodal output"):
        ar2dit([_ar_output(None)], prompts={})


def test_ar2dit_with_non_mapping_multimodal_output_raises_stage_input_error() -> None:
    with pytest.raises(StageInputProcessingError, match="missing latent multimodal output"):
        ar2dit([_ar_output(["not-a-mapping"])], prompts={})


def test_ar2dit_accepts_a_well_formed_latent() -> None:
    # prompt_token_ids + cumulative_token_ids[:-1] == 2 + 2 rows of hidden state.
    dit_inputs = ar2dit([_ar_output({"latent": torch.zeros((4, 8))})], prompts={})

    assert len(dit_inputs) == 1
