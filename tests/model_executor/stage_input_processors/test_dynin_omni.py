# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bridge rejections for the DynIn-Omni stage input processors."""

from types import SimpleNamespace

import pytest

from vllm_omni.errors import StageInputProcessingError
from vllm_omni.model_executor.stage_input_processors.dynin_omni import (
    token2image_to_token2audio,
    token2text_to_token2image,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _source_output(*, token_ids=None, cumulative_token_ids=None):
    completion = SimpleNamespace(
        multimodal_output={"token_ids": token_ids} if token_ids is not None else None,
        cumulative_token_ids=cumulative_token_ids or [],
    )
    return SimpleNamespace(request_id="req-dynin", outputs=[completion])


@pytest.mark.parametrize("bridge", [token2text_to_token2image, token2image_to_token2audio])
def test_bridge_without_token_ids_raises_stage_input_error(bridge) -> None:
    """Both registered bridges share _bridge_tokens, so both must refuse an
    output carrying no tokens the same way — as a typed rejection that reports
    the processor's own reason rather than an internal error."""
    with pytest.raises(StageInputProcessingError, match="has no token_ids"):
        bridge([_source_output()], prompt={})


@pytest.mark.parametrize("bridge", [token2text_to_token2image, token2image_to_token2audio])
def test_bridge_falls_back_to_cumulative_token_ids(bridge) -> None:
    next_inputs = bridge([_source_output(cumulative_token_ids=[5, 6])], prompt={})

    assert len(next_inputs) == 1
    assert next_inputs[0]["prompt_token_ids"] == [5, 6]
