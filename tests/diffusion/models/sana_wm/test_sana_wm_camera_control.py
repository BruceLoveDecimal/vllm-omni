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


def test_sana_wm_compute_raymap_origin_direction_use_plucker_false() -> None:
    """use_plucker=False path returns [origin, direction] per pixel."""
    import torch

    from vllm_omni.diffusion.models.sana_wm.camera_control import compute_raymap

    # Identity camera at origin: origin=[0,0,0], direction=[0,0,1] for center pixel.
    intrinsics = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    pose = torch.eye(4).unsqueeze(0)

    raymap = compute_raymap(intrinsics, pose, height=1, width=1, use_plucker=False)

    assert raymap.shape == (1, 1, 1, 6)
    # Origin is the camera translation (zeros for identity).
    assert torch.allclose(raymap[0, 0, 0, :3], torch.zeros(3), atol=1e-6)
    # Direction is normalised camera-to-world forward ray.
    assert torch.allclose(raymap[0, 0, 0, 3:], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)


def test_sana_wm_compute_raymap_rotation_changes_direction() -> None:
    """90° yaw rotation: camera looks along +X in world, not +Z."""
    import math

    import torch

    from vllm_omni.diffusion.models.sana_wm.camera_control import compute_raymap

    intrinsics = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    # 90° CCW around Y: cam +Z maps to world +X.
    c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
    R = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    pose = torch.eye(4).unsqueeze(0)
    pose[0, :3, :3] = R

    raymap = compute_raymap(intrinsics, pose, height=1, width=1, use_plucker=False)

    direction = raymap[0, 0, 0, 3:]
    assert abs(float(direction[0]) - 1.0) < 1e-5, "Camera should look along world +X"
    assert abs(float(direction[1])) < 1e-5
    assert abs(float(direction[2])) < 1e-5


def test_sana_wm_compute_raymap_multi_pixel_varies() -> None:
    """Ray directions must differ across pixels (non-trivial spatial variation)."""
    import torch

    from vllm_omni.diffusion.models.sana_wm.camera_control import compute_raymap

    intrinsics = torch.tensor([[2.0, 2.0, 1.5, 1.5]])  # fx=fy=2, cx=cy=1.5 for 4x4
    pose = torch.eye(4).unsqueeze(0)

    raymap = compute_raymap(intrinsics, pose, height=4, width=4, use_plucker=False)

    directions = raymap[0, :, :, 3:]  # [4, 4, 3]
    center = directions[2, 2]  # pixel (2,2): offset=(0.5/2, 0.5/2) → not exactly [0,0,1]
    corner = directions[0, 0]  # pixel (0,0): large offset from principal point
    assert not torch.allclose(center, corner, atol=1e-3), "Different pixels must have different directions"


def test_sana_wm_build_plucker_condition_spatial_raymap_deterministic() -> None:
    """Same camera condition → identical spatial_raymap (determinism check)."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmCameraCondition, build_plucker_condition

    condition = SanaWmCameraCondition(action="w-1", num_frames=3, height=64, width=64)
    t1 = build_plucker_condition(condition)["spatial_raymap"]
    t2 = build_plucker_condition(condition)["spatial_raymap"]

    assert t1 is not None and t2 is not None
    assert torch.allclose(t1, t2, atol=0.0), "spatial_raymap must be deterministic"


def test_sana_wm_build_plucker_condition_spatial_raymap_forward_direction() -> None:
    """action='w-1' (forward drive): center pixel directions should point mostly forward."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmCameraCondition, build_plucker_condition

    condition = SanaWmCameraCondition(action="w-1", num_frames=3, height=64, width=64)
    spatial_raymap = build_plucker_condition(condition)["spatial_raymap"]

    assert spatial_raymap is not None
    # spatial_raymap: [3, F, H, W] — channels are (x, y, z) direction components
    # For a forward-driving camera the z-component should dominate (|z| > |x|, |y|).
    F, H, W = spatial_raymap.shape[1], spatial_raymap.shape[2], spatial_raymap.shape[3]
    center_x = spatial_raymap[0, :, H // 2, W // 2]  # x-component
    center_y = spatial_raymap[1, :, H // 2, W // 2]  # y-component
    center_z = spatial_raymap[2, :, H // 2, W // 2]  # z-component
    assert (center_z.abs() > center_x.abs()).all(), "Forward camera: z > x at center pixel"
    assert (center_z.abs() > center_y.abs()).all(), "Forward camera: z > y at center pixel"


def test_sana_wm_camera_action_builds_plucker_condition() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmCameraCondition, build_plucker_condition

    condition = SanaWmCameraCondition(action="w-8", num_frames=9, height=128, width=192)
    tensors = build_plucker_condition(condition)

    assert tensors["raymap"].shape[1] == 20
    assert tensors["chunk_plucker"].shape[0] == 48
