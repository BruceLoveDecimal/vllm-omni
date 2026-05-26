# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_sana_wm_compute_raymap_identity_pose_numeric_reference() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm.camera_control import compute_raymap

    intrinsics = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    pose = torch.eye(4).unsqueeze(0)

    raymap = compute_raymap(intrinsics, pose, height=1, width=1, use_plucker=True)

    assert raymap.shape == (1, 1, 1, 6)
    assert torch.allclose(raymap[0, 0, 0, :3], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(raymap[0, 0, 0, 3:], torch.zeros(3))


def test_sana_wm_compute_raymap_translated_pose_numeric_reference() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm.camera_control import compute_raymap

    intrinsics = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    pose = torch.eye(4).unsqueeze(0)
    pose[0, 0, 3] = 1.0

    raymap = compute_raymap(intrinsics, pose, height=1, width=1, use_plucker=True)

    assert torch.allclose(raymap[0, 0, 0, :3], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(raymap[0, 0, 0, 3:], torch.tensor([0.0, -1.0, 0.0]))


def test_sana_wm_camera_action_builds_plucker_condition() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmCameraCondition, build_plucker_condition

    condition = SanaWmCameraCondition(action="w-8", num_frames=9, height=128, width=192)
    tensors = build_plucker_condition(condition)

    assert tensors["raymap"].shape[1] == 20
    assert tensors["chunk_plucker"].shape[0] == 48
