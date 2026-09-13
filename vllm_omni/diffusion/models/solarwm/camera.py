# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright 2026 SolarWM Contributors
# Copyright (c) the PRoPE authors. See NOTICE in this directory.
# Portions adapted under MIT; the upstream permission notice is retained.
"""SolarWM projective attention in the shared [B, L, H, D] layout.

Adapted from SolarWM a3a3fac16466102a2b97df867f7703df7172cb2a.
Inputs are first-frame-relative world-to-camera matrices and normalized
intrinsics, already aligned to latent tokens. Raw C2W is not this contract.
"""

import torch


def transform_relative_viewmats(viewmats: torch.Tensor, transform: str = "linear") -> torch.Tensor:
    if viewmats.ndim != 4 or viewmats.shape[-2:] != (4, 4) or not viewmats.is_floating_point():
        raise ValueError("viewmats must be floating [B, cameras, 4, 4]")
    if transform == "linear":
        return viewmats
    if transform != "logd4":
        raise ValueError("camera translation transform must be linear or logd4")
    dtype = torch.float64 if viewmats.dtype == torch.float64 else torch.float32
    translation = viewmats[..., :3, 3].to(dtype)
    norm = torch.linalg.vector_norm(translation, dim=-1, keepdim=True)
    scale = torch.where(
        norm > 0,
        torch.log1p(norm) / (4 * norm.clamp_min(torch.finfo(dtype).tiny)),
        torch.zeros_like(norm),
    )
    result = viewmats.clone()
    result[..., :3, 3] = (translation * scale).to(viewmats.dtype)
    return result


def camera_projection(
    viewmats: torch.Tensor, intrinsics: torch.Tensor | None, translation_transform: str = "linear"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return P and P^-1, retaining camera precision until activation use."""
    viewmats = transform_relative_viewmats(viewmats, translation_transform)
    rotation_inv = viewmats[..., :3, :3].transpose(-1, -2)
    inverse = torch.zeros_like(viewmats)
    inverse[..., :3, :3] = rotation_inv
    inverse[..., :3, 3] = -torch.einsum("...ij,...j->...i", rotation_inv, viewmats[..., :3, 3])
    inverse[..., 3, 3] = 1
    if intrinsics is None:
        return viewmats, inverse
    if intrinsics.shape != (*viewmats.shape[:2], 3, 3):
        raise ValueError("intrinsics must be [B, cameras, 3, 3] aligned with viewmats")
    if intrinsics.device != viewmats.device or intrinsics.dtype != viewmats.dtype:
        raise ValueError("intrinsics and viewmats must share device and floating dtype")
    # SolarWM intentionally discards principal point and skew here. Intrinsics
    # are normalized upstream; do not divide them by image resolution again.
    lifted = torch.zeros_like(viewmats)
    lifted[..., 0, 0] = intrinsics[..., 0, 0]
    lifted[..., 1, 1] = intrinsics[..., 1, 1]
    lifted[..., 2, 2] = lifted[..., 3, 3] = 1
    lifted_inv = lifted.clone()
    lifted_inv[..., 0, 0] = 1 / intrinsics[..., 0, 0]
    lifted_inv[..., 1, 1] = 1 / intrinsics[..., 1, 1]
    return lifted @ viewmats, inverse @ lifted_inv


def apply_projective_transform(features: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply a tiled 4x4 transform per camera or per token to [B, L, H, D]."""
    if features.ndim != 4 or features.shape[-1] % 4:
        raise ValueError("features must be [B, L, H, D] with D divisible by 4")
    b, length, heads, dim = features.shape
    if matrix.ndim != 4 or matrix.shape[0] != b or matrix.shape[-2:] != (4, 4):
        raise ValueError("projection must be [B, cameras, 4, 4]")
    cameras = matrix.shape[1]
    if cameras == 0 or length == 0 or length % cameras:
        raise ValueError("token length must be a positive multiple of the camera count")
    tiled = features.reshape(b, cameras, length // cameras, heads, dim // 4, 4)
    return torch.einsum("bcij,bcpnkj->bcpnki", matrix.to(features.dtype), tiled).reshape_as(features)
