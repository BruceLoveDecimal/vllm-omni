# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Camera conditioning for SolarWM-H3.

SolarWM keeps MiniMax-H3's native MM-RoPE on head dimensions ``[0, 96)`` and
adds a projective positional encoding (PRoPE) on the remaining ``[96, 128)``
dimensions. Poses are first-frame-relative world-to-camera matrices whose
translation is compressed with ``t * log1p(|t|) / (4 |t|)``; intrinsics are the
fixed normalized Wan focal lengths the adapter was trained with, so the
user-supplied intrinsics are never read.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SOLARWM_H3_HEAD_DIM = 128
SOLARWM_H3_PROPE_DIM_START = 96
# Normalized focal lengths shared with SolarWM's Wan camera training.
SOLARWM_H3_FIXED_FX = 969.6969696969696 / (960.0 * 2.0)
SOLARWM_H3_FIXED_FY = 969.6969696969696 / (540.0 * 2.0)


@dataclass(frozen=True)
class CameraProjection:
    """Per-row 4x4 projective transforms for queries, keys/values and outputs."""

    query: torch.Tensor
    key_value: torch.Tensor
    output: torch.Tensor

    def rows(self, start: int, stop: int) -> CameraProjection:
        return CameraProjection(
            query=self.query[start:stop],
            key_value=self.key_value[start:stop],
            output=self.output[start:stop],
        )


def invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert rigid ``[..., 4, 4]`` transforms without a general matrix inverse."""
    rotation_t = transforms[..., :3, :3].transpose(-1, -2)
    output = torch.zeros_like(transforms)
    output[..., :3, :3] = rotation_t
    output[..., :3, 3] = -torch.einsum("...ij,...j->...i", rotation_t, transforms[..., :3, 3])
    output[..., 3, 3] = 1.0
    return output


def first_frame_relative_w2c(c2w: torch.Tensor) -> torch.Tensor:
    """Convert absolute camera-to-world poses ``[T, 4, 4]`` to first-frame-relative world-to-camera."""
    poses = c2w.to(torch.float32)
    relative_c2w = torch.matmul(invert_se3(poses[:1]), poses)
    relative_c2w[0] = torch.eye(4, dtype=torch.float32)
    return invert_se3(relative_c2w).contiguous()


def logd4_translation(viewmats: torch.Tensor) -> torch.Tensor:
    """Compress the translation column with ``t * log1p(|t|) / (4 |t|)``, zero-safe."""
    translation = viewmats[..., :3, 3].to(torch.float32)
    norm = torch.linalg.vector_norm(translation, dim=-1, keepdim=True)
    safe_norm = norm.clamp_min(torch.finfo(torch.float32).tiny)
    scale = torch.where(norm > 0, torch.log1p(norm) / (4.0 * safe_norm), torch.zeros_like(norm))
    output = viewmats.clone()
    output[..., :3, 3] = (translation * scale).to(viewmats.dtype)
    return output


def _lifted_focal(reference: torch.Tensor, fx: float, fy: float) -> torch.Tensor:
    lifted = torch.zeros_like(reference)
    lifted[..., 0, 0] = fx
    lifted[..., 1, 1] = fy
    lifted[..., 2, 2] = 1.0
    lifted[..., 3, 3] = 1.0
    return lifted


def camera_projection(viewmats: torch.Tensor, dtype: torch.dtype) -> CameraProjection:
    """Build PRoPE matrices for ``[N, 4, 4]`` world-to-camera poses.

    The reference applies this arithmetic in the attention dtype (bfloat16), so
    the poses are cast first and every product below runs in ``dtype``.
    """
    views = logd4_translation(viewmats.to(dtype))
    focal = _lifted_focal(views, SOLARWM_H3_FIXED_FX, SOLARWM_H3_FIXED_FY)
    # The reference inverts the (already rounded) focal entries in ``dtype``.
    inverse_focal = torch.zeros_like(focal)
    inverse_focal[..., 0, 0] = 1.0 / focal[..., 0, 0]
    inverse_focal[..., 1, 1] = 1.0 / focal[..., 1, 1]
    inverse_focal[..., 2, 2] = 1.0
    inverse_focal[..., 3, 3] = 1.0
    projection = torch.matmul(focal, views)
    return CameraProjection(
        query=projection.transpose(-1, -2).contiguous(),
        key_value=torch.matmul(invert_se3(views), inverse_focal),
        output=projection,
    )


def apply_camera_projection(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Project the ``[96, 128)`` suffix of ``[S, heads, 128]`` features by per-row ``[S, 4, 4]`` matrices."""
    native = x[..., :SOLARWM_H3_PROPE_DIM_START]
    camera = x[..., SOLARWM_H3_PROPE_DIM_START:]
    rows, heads, suffix = camera.shape
    tiled = camera.reshape(rows, heads, suffix // 4, 4)
    projected = torch.einsum("sij,shpj->shpi", matrix.to(x.dtype), tiled)
    return torch.cat((native, projected.reshape(rows, heads, suffix)), dim=-1)


__all__ = [
    "SOLARWM_H3_FIXED_FX",
    "SOLARWM_H3_FIXED_FY",
    "SOLARWM_H3_HEAD_DIM",
    "SOLARWM_H3_PROPE_DIM_START",
    "CameraProjection",
    "apply_camera_projection",
    "camera_projection",
    "first_frame_relative_w2c",
    "invert_se3",
    "logd4_translation",
]
