# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_sana_wm_pipeline_inherits_cfg_parallel_mixin() -> None:
    from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
    from vllm_omni.diffusion.models.sana_wm import SanaWmPipeline, SanaWmTwoStagesPipeline

    assert issubclass(SanaWmPipeline, CFGParallelMixin)
    assert issubclass(SanaWmTwoStagesPipeline, CFGParallelMixin)


def test_sana_wm_cfg_parallel_uses_standard_noise_combine() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmPipeline

    pipe = SanaWmPipeline(od_config=None)
    positive = torch.tensor([[[2.0, 4.0]]])
    negative = torch.tensor([[[1.0, 1.0]]])

    combined = pipe.combine_cfg_noise(positive, negative, true_cfg_scale=0.5, cfg_normalize=False)

    assert torch.equal(combined, torch.tensor([[[1.5, 2.5]]]))
