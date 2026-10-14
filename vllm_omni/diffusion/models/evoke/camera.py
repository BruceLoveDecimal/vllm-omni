# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Evoke camera conventions and the realtime camera handler.

Evoke conditions on camera motion only through the geometry warp, which is
rendered for every 24 fps output frame. Its poses are OpenCV camera-to-world
matrices (x right, y down, z forward) relative to the first frame of the
session -- the convention of the upstream interactive UI.

The shared interaction timeline speaks Unity (x right, y up, z forward), so
poses are converted by flipping the y axis of both the world and camera frames.
"""

from __future__ import annotations

import math
from typing import ClassVar

import torch
from typing_extensions import override

from vllm_omni.diffusion.interaction.modality_handlers.camera import CameraPose, CameraSession, SE3DeltaCameraHandler
from vllm_omni.diffusion.interaction.types import InteractionChunkMetadata
from vllm_omni.diffusion.worker.utils import StepRequestState

# Upstream UI camera: 70 degree horizontal FOV on a 1280x720 source, which the
# training pipeline maps to the output size with resize + centre crop.
DEFAULT_HFOV_DEGREES = 70.0
DEFAULT_SOURCE_HW = (720, 1280)

_Y_FLIP = torch.diag(torch.tensor([1.0, -1.0, 1.0, 1.0], dtype=torch.float64))


def unity_to_opencv(c2w: torch.Tensor) -> torch.Tensor:
    """Unity c2w ``[..., 4, 4]`` -> OpenCV c2w in the y-flipped world frame."""
    flip = _Y_FLIP.to(device=c2w.device, dtype=c2w.dtype)
    return flip @ c2w @ flip


def crop_resize_intrinsics(
    fx: float, fy: float, cx: float, cy: float, source_hw: tuple[int, int], target_hw: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Pixel intrinsics after a uniform resize that covers ``target_hw`` and a centre crop."""
    (h_src, w_src), (h_dst, w_dst) = source_hw, target_hw
    scale = max(h_dst / h_src, w_dst / w_src)
    crop_x = (round(w_src * scale) - w_dst) / 2.0
    crop_y = (round(h_src * scale) - h_dst) / 2.0
    return fx * scale, fy * scale, cx * scale - crop_x, cy * scale - crop_y


def default_intrinsics(
    height: int,
    width: int,
    hfov_degrees: float = DEFAULT_HFOV_DEGREES,
    source_hw: tuple[int, int] = DEFAULT_SOURCE_HW,
) -> tuple[float, float, float, float]:
    """``[fx, fy, cx, cy]`` at the output size for a pinhole with ``hfov_degrees`` on ``source_hw``."""
    if not 1.0 <= float(hfov_degrees) <= 170.0:
        raise ValueError(f"camera hfov must be within [1, 170] degrees, got {hfov_degrees}")
    h_src, w_src = source_hw
    focal = (w_src / 2.0) / math.tan(math.radians(float(hfov_degrees)) / 2.0)
    return crop_resize_intrinsics(focal, focal, w_src / 2.0, h_src / 2.0, source_hw, (int(height), int(width)))


class EvokeCameraHandler(SE3DeltaCameraHandler):
    """Camera timeline sampled on the 24 fps media grid instead of the latent grid.

    Each chunk yields one absolute pose per decoded media frame. ``velocity``
    deltas therefore integrate once per media frame (1/24 s), and the first
    chunk starts exactly at the session origin, the pose of the conditioning
    image. ``last_absolute_poses`` holds Unity-convention c2w matrices; the
    pipeline converts them with :func:`unity_to_opencv`.
    """

    modality: ClassVar[str] = "camera"

    @override
    def apply_at_chunk_boundary(
        self,
        state: StepRequestState,
        *,
        boundary_at: float,
        chunk_index: int | None = None,
        num_media_frames: int | None = None,
        fps: float | None = None,
        num_latent_frames: int | None = None,
    ) -> InteractionChunkMetadata | None:
        del chunk_index, num_latent_frames
        if num_media_frames is None or fps is None:
            raise ValueError("EvokeCameraHandler requires chunk num_media_frames and fps")
        session = state.interaction_sessions.setdefault("camera", CameraSession())
        assert isinstance(session, CameraSession)
        with session.lock:
            is_first = session.last_absolute_poses is None
            steps = int(num_media_frames) - 1 if is_first else int(num_media_frames)
            origin = [session.current_pose.clone()] if is_first else []
            stepped: list[CameraPose] = []
            started: list[str] = []
            active: list[str] = []
            completed: list[str] = []
            if steps > 0:
                stepped, started, active, completed = self._step_one_chunk(
                    session,
                    num_media_frames=steps,
                    fps=fps,
                    num_latent_frames=steps,
                    boundary_at=boundary_at,
                )
            else:
                session.last_boundary_at = boundary_at
            poses = origin + stepped
            session.last_absolute_poses = torch.stack([pose.as_matrix() for pose in poses], dim=0)
        return InteractionChunkMetadata(
            started_event_ids=started,
            active_event_ids=active,
            completed_event_ids=completed,
        )


__all__ = [
    "DEFAULT_HFOV_DEGREES",
    "DEFAULT_SOURCE_HW",
    "EvokeCameraHandler",
    "crop_resize_intrinsics",
    "default_intrinsics",
    "unity_to_opencv",
]
