# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_sana_wm_camera_action_builds_plucker_condition() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmCameraCondition, build_plucker_condition

    condition = SanaWmCameraCondition(action="w-8", num_frames=9, height=128, width=192)
    tensors = build_plucker_condition(condition)

    assert tensors["raymap"].shape[1] == 20
    assert tensors["chunk_plucker"].shape[0] == 48
