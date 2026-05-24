# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_sana_wm_two_stage_declares_isolated_refiner_components() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmTwoStagesPipeline

    pipe = SanaWmTwoStagesPipeline(od_config=None)

    assert "refiner_text_encoder" in pipe._encoder_modules
    assert "refiner_connectors" in pipe._encoder_modules
    assert pipe._resident_modules == ["refiner_transformer"]
    assert pipe.refiner_transformer is None
    assert pipe.refiner_text_encoder is None
    assert pipe.refiner_connectors is None


def test_sana_wm_two_stage_refiner_loader_requires_checkpoint() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmTwoStagesPipeline

    pipe = SanaWmTwoStagesPipeline(od_config=None)

    with pytest.raises(ValueError, match="checkpoint resolution"):
        pipe.ensure_refiner_components()
