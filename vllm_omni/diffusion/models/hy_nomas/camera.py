# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared Unity timeline to Hy-Nomas framewise OpenCV camera conditions."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


def normalize_camera(poses: torch.Tensor, pixel_frames: int) -> torch.Tensor:
    """Resample C2W poses, change basis, then apply native segment normalization.

    Native reference 911cea6: normalization.py and rotation_scale.py.
    Both world and camera bases change, so rotations stay in SO(3).
    """
    source = poses.detach().cpu().double().numpy()
    if source.ndim != 3 or source.shape[1:] != (4, 4) or len(source) == 0 or pixel_frames < 1:
        raise ValueError("Camera requires nonempty [T,4,4] poses and positive pixel_frames.")
    if not np.isfinite(source).all():
        raise ValueError("Camera poses must be finite.")
    basis = np.diag([1.0, -1.0, 1.0, 1.0])
    source = basis @ source @ basis
    times = np.linspace(0, len(source) - 1, pixel_frames)
    result = np.broadcast_to(np.eye(4), (pixel_frames, 4, 4)).copy()
    if len(source) == 1:
        result[:] = source[0]
    else:
        result[:, :3, :3] = Slerp(np.arange(len(source)), Rotation.from_matrix(source[:, :3, :3]))(times).as_matrix()
        for axis in range(3):
            result[:, axis, 3] = np.interp(times, np.arange(len(source)), source[:, axis, 3])
    # Float64 first-frame reanchor, then analytic relative-to-previous transform.
    result = np.linalg.inv(result[0]) @ result
    result[0] = np.eye(4)
    inverse = np.broadcast_to(np.eye(4), result[:-1].shape).copy()
    inverse[:, :3, :3] = result[:-1, :3, :3].transpose(0, 2, 1)
    inverse[:, :3, 3:4] = -inverse[:, :3, :3] @ result[:-1, :3, 3:4]
    result[1:] = inverse @ result[1:]
    if pixel_frames > 1:
        scale = max(float(np.quantile(np.linalg.norm(result[1:, :3, 3], axis=1), 0.99)), 1e-6)
        result[:, :3, 3] /= scale
    # Native prepare_turn rounds normalized poses to float32 before the
    # rotation-gain helper promotes their rotations to float64. Moving this
    # rounding boundary can cross a BF16 threshold in the camera condition.
    result = result.astype(np.float32)
    rotvec = Rotation.from_matrix(result[:, :3, :3].astype(np.float64)).as_rotvec() * 2.5
    if np.linalg.norm(rotvec, axis=1).max() >= np.pi - 1e-7:
        raise ValueError("Camera rotation gain reaches 180 degrees; reduce the requested camera increment.")
    result[:, :3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    return torch.from_numpy(result).unsqueeze(0)
