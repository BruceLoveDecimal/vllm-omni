# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Evoke camera conventions and the media-grid camera handler."""

import math

import pytest
import torch

from vllm_omni.diffusion.interaction.modality_handlers.camera import CameraPose, CameraSession, _quat_to_rotmat
from vllm_omni.diffusion.models.evoke.camera import (
    EvokeCameraHandler,
    default_intrinsics,
    unity_to_opencv,
)
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _state() -> StepRequestState:
    return StepRequestState(request_id="r", sampling=None)


def test_unity_to_opencv_keeps_forward_and_flips_up():
    pose = CameraPose(translation=(0.5, 2.0, 3.0)).as_matrix()
    cv = unity_to_opencv(pose)
    assert torch.allclose(cv[:3, 3], torch.tensor([0.5, -2.0, 3.0], dtype=torch.float64))
    # Pitching up in Unity (+x rotation lifts the view toward +y) looks toward -y in OpenCV.
    half = math.radians(-10) / 2
    look_up = CameraPose(rotation=(math.sin(half), 0.0, 0.0, math.cos(half))).as_matrix()
    forward = unity_to_opencv(look_up)[:3, 2]
    assert forward[1] < 0 and forward[2] > 0
    assert torch.allclose(torch.det(unity_to_opencv(look_up)[:3, :3]), torch.tensor(1.0, dtype=torch.float64))


def test_default_intrinsics_follow_the_upstream_ui_camera():
    fx, fy, cx, cy = default_intrinsics(384, 640)
    focal = 640.0 / math.tan(math.radians(35.0)) * (384 / 720)
    assert fx == pytest.approx(focal) and fy == pytest.approx(focal)
    assert cx == pytest.approx(320.0, abs=0.5) and cy == pytest.approx(192.0)
    with pytest.raises(ValueError):
        default_intrinsics(384, 640, hfov_degrees=0.0)


def _velocity_event(handler, state, dz):
    handler.enqueue(
        state,
        event_id="v",
        received_at=0.0,
        payload={"mode": "velocity", "data": {"translation": [0.0, 0.0, dz]}},
        transition_chunks=None,
    )


def test_first_chunk_starts_at_origin_and_velocity_integrates_per_media_frame():
    handler = EvokeCameraHandler()
    state = _state()
    _velocity_event(handler, state, 0.01)
    handler.apply_at_chunk_boundary(state, boundary_at=1.0, num_media_frames=36, fps=24.0, num_latent_frames=9)
    session = state.interaction_sessions["camera"]
    assert isinstance(session, CameraSession)
    first = session.last_absolute_poses
    assert first.shape == (36, 4, 4)
    assert torch.equal(first[0], torch.eye(4, dtype=torch.float64))
    assert first[-1, 2, 3].item() == pytest.approx(0.35)

    handler.apply_at_chunk_boundary(state, boundary_at=2.5, num_media_frames=36, fps=24.0, num_latent_frames=9)
    second = session.last_absolute_poses
    assert second.shape == (36, 4, 4)
    assert second[0, 2, 3].item() == pytest.approx(0.36)
    assert second[-1, 2, 3].item() == pytest.approx(0.71)


def test_idle_camera_holds_the_origin():
    handler = EvokeCameraHandler()
    state = _state()
    handler.apply_at_chunk_boundary(state, boundary_at=0.0, num_media_frames=36, fps=24.0, num_latent_frames=9)
    poses = state.interaction_sessions["camera"].last_absolute_poses
    assert torch.equal(poses, torch.eye(4, dtype=torch.float64).expand(36, 4, 4))
    assert _quat_to_rotmat((0.0, 0.0, 0.0, 1.0)).equal(torch.eye(3, dtype=torch.float64))
