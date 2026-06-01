# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_sana_wm_exposes_hsdp_shard_conditions_for_stage1_blocks() -> None:
    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmTransformer3DModel

    model = object.__new__(SanaWmTransformer3DModel)
    nn.Module.__init__(model)
    model.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
    model.final_layer = nn.Linear(4, 4)

    conditions = getattr(model, "_hsdp_shard_conditions", None)

    assert conditions is not None
    assert len(conditions) == 1

    matched = []
    for name, module in model.named_modules():
        if any(cond(name, module) for cond in conditions):
            matched.append(name)

    assert matched == ["blocks.0", "blocks.1"]
