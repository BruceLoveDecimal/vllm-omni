# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Component-discovery contract for the two-stage SANA-WM pipeline."""

import pytest

from vllm_omni.diffusion.models.ltx2.ltx2_transformer import LTX2VideoTransformer3DModel
from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm_two_stages import SanaWmTwoStagesPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_refiner_transformer_is_discovered_as_a_dit():
    """The refiner is the largest component; it must be offloadable and shardable.

    ``_resident_modules`` pins a module whole on the device, while ``_dit_modules``
    is what layerwise offload, HSDP and SP all walk. Declaring the refiner as
    resident made the one component that most needs streaming the one component
    that could not stream.
    """

    assert SanaWmTwoStagesPipeline._dit_modules == ["transformer", "refiner_transformer"]
    assert SanaWmTwoStagesPipeline._resident_modules == []


def test_refiner_transformer_declares_the_hooks_the_backends_need():
    assert LTX2VideoTransformer3DModel._layerwise_offload_blocks_attrs == ["transformer_blocks"]
    assert LTX2VideoTransformer3DModel._hsdp_shard_conditions
