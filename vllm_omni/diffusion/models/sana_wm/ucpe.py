# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SANA-WM UCPE (Unified Camera Pose Embedding) per-block attention transforms.

Ported from NVlabs/Sana ``sana_camctrl_blocks.py`` for the inference-only
SANA-WM bidirectional 1600M release. Scope is intentionally narrow:

* Pinhole camera only (xi=0); the UCM ``xi`` parameter is fixed.
* Inference path only — no training-time camera-branch dropout.
* Online computation only — the precomputed ``cam_pos_embeds`` shortcut
  used by NVlabs' fused kernels is omitted because vLLM-Omni recomputes
  the matrices once per request and shares the resulting closures across
  all blocks via the per-block ``precomputed_gates`` plumbing.
* ``apply_vo=True`` only — the SANA-WM checkpoint always uses the
  inverse-output transform.

Public surface:
    ``prepare_prope_fns(head_dim, camera_conditions, HW, patch_size,
                        rotary_emb=None)`` -> ``(apply_q, apply_kv, apply_o)``

Each closure transforms a tensor of shape ``(B, num_heads, N, D)`` where
``N = T * H * W`` (matching the GDN token layout) and ``D == head_dim``,
returning a tensor of the same shape.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pinhole ray-grid construction
# ---------------------------------------------------------------------------


def _make_pixel_grid(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``(H, W, 3)`` grid ``(x, y, 1)`` of pixel coordinates."""
    xs = torch.arange(width, device=device, dtype=dtype)
    ys = torch.arange(height, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((x_grid, y_grid, torch.ones_like(x_grid)), dim=-1)


def _unproject_pinhole(
    fx: torch.Tensor,
    fy: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Unproject a pixel grid into camera-frame direction vectors.

    Args:
        fx, fy, cx, cy: shape ``(B, T)`` intrinsics in latent-pixel units
            (caller divides ``cx``/``cy`` by patch size before calling).
        height, width: latent grid size.

    Returns:
        ``(B, T, H, W, 3)`` direction tensor (NOT normalised).
    """
    B, T = fx.shape
    device = fx.device
    dtype = fx.dtype
    grid = _make_pixel_grid(height, width, device, dtype)  # (H, W, 3)
    u = grid[..., 0]  # (H, W)
    v = grid[..., 1]
    fx_e = fx.view(B, T, 1, 1)
    fy_e = fy.view(B, T, 1, 1)
    cx_e = cx.view(B, T, 1, 1)
    cy_e = cy.view(B, T, 1, 1)
    x = (u - cx_e) / fx_e
    y = (v - cy_e) / fy_e
    z = torch.ones_like(x)
    # Match NVlabs UCM normalisation at xi=0: alpha = 1, gamma = 1/(1+r^2),
    # X = gamma*x, Y = gamma*y, Z = gamma. The resulting vector has unit
    # norm. Computed compactly via direct stack + normalise.
    d_cam = torch.stack((x, y, z), dim=-1)  # (B, T, H, W, 3)
    d_cam = d_cam / d_cam.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return d_cam


def _world_to_ray_mats(
    d_cam: torch.Tensor,  # (B, T, H, W, 3) unit rays in camera frame
    c2w: torch.Tensor,  # (B, T, 4, 4) camera-to-world SE(3)
) -> torch.Tensor:
    """Build per-pixel ``ray <- world`` SE(3) transforms ``(B, T, H, W, 4, 4)``.

    The ray frame is anchored at the camera origin with the +Z axis along
    the per-pixel viewing direction and a +Y axis defined by the camera-Y
    cross product. This matches NVlabs ``world_to_ray_mats``.
    """
    B, T, H, W, _ = d_cam.shape
    device = d_cam.device
    dtype = d_cam.dtype
    rot_cam = c2w[..., :3, :3]
    trans_cam = c2w[..., :3, 3]
    d_world = torch.einsum("btij,bthwj->bthwi", rot_cam, d_cam)
    cam_y = rot_cam[..., :, 1].unsqueeze(2).unsqueeze(3).expand(B, T, H, W, 3)
    z_ray = F.normalize(d_world, dim=-1, eps=1e-6)
    x_ray = F.normalize(torch.cross(cam_y, z_ray, dim=-1), dim=-1, eps=1e-6)
    y_ray = F.normalize(torch.cross(z_ray, x_ray, dim=-1), dim=-1, eps=1e-6)
    rot_local_to_world = torch.stack((x_ray, y_ray, z_ray), dim=-1)  # (B,T,H,W,3,3)
    rot_world_to_local = rot_local_to_world.transpose(-1, -2)
    t_world = trans_cam.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, 3)
    t_w2l = -torch.einsum("bthwij,bthwj->bthwi", rot_world_to_local, t_world)
    raymats = torch.zeros(B, T, H, W, 4, 4, device=device, dtype=dtype)
    raymats[..., :3, :3] = rot_world_to_local
    raymats[..., :3, 3] = t_w2l
    raymats[..., 3, 3] = 1.0
    # Guard NaNs from degenerate pixels (e.g. nan direction at infinity).
    nan_mask = torch.isnan(d_world).any(-1)
    if nan_mask.any():
        eye = torch.eye(4, device=device, dtype=dtype)
        raymats[nan_mask] = eye
    return raymats


def _process_camera_conditions(
    camera_conditions: torch.Tensor,  # (B, T, 20)
    HW: tuple[int, int, int],
) -> torch.Tensor:
    """Convert ``(B, T, 20)`` SANA-WM camera conditions to ``(B, T, H, W, 4, 4)``
    ``ray <- world`` transforms.

    Layout of the last axis is ``[c2w_flat (16), fx, fy, cx, cy]``.
    Intrinsics are expected in **latent-pixel units** — matching the output
    of ``camera_control._pack_camera_conditions`` which scales the original
    image-pixel intrinsics by ``latent_width / image_width`` and
    ``latent_height / image_height`` before packing into the ``raymap``
    tensor. The mathematics is identical to the NVlabs reference: NVlabs
    accepts image-pixel intrinsics and converts to latent via a FoV
    round-trip, but the FoV round-trip is a no-op when both ``fx`` and
    ``width`` scale together.

    Args:
        camera_conditions: ``(B, T, 20)`` from ``_pack_camera_conditions``.
        HW: ``(T_latent, H_latent, W_latent)`` latent-grid shape.
    """
    B, T_cond, last = camera_conditions.shape
    if last != 20:
        raise ValueError(f"SANA-WM camera_conditions last dim must be 20, got {last}.")
    T_latent, H_latent, W_latent = HW
    if T_cond != T_latent:
        raise ValueError(
            f"SANA-WM camera_conditions frames {T_cond} != latent frames {T_latent}."
        )
    c2w = camera_conditions[..., :16].reshape(B, T_latent, 4, 4)
    fx = camera_conditions[..., 16]
    fy = camera_conditions[..., 17]
    cx = camera_conditions[..., 18]
    cy = camera_conditions[..., 19]
    d_cam = _unproject_pinhole(fx, fy, cx, cy, H_latent, W_latent)
    raymats = _world_to_ray_mats(d_cam, c2w)
    return raymats


def _invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of a stack of 4x4 SE(3) matrices."""
    if transforms.shape[-2:] != (4, 4):
        raise ValueError(f"SE3 inverse expects (..., 4, 4), got {tuple(transforms.shape)}.")
    rot_inv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = rot_inv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", rot_inv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


# ---------------------------------------------------------------------------
# Per-token transform primitives
# ---------------------------------------------------------------------------


def _apply_ray_projmat(
    feats: torch.Tensor,  # (B, num_heads, N, D)
    matrix: torch.Tensor,  # (B, N, 4, 4)
) -> torch.Tensor:
    """Apply a per-token 4x4 projection to features grouped by 4 channels.

    ``D`` must be a multiple of 4. Out-of-place; preserves ``feats.shape``.
    """
    B, num_heads, N, D = feats.shape
    if D % 4 != 0:
        raise ValueError(f"UCPE projmat expects D divisible by 4, got D={D}.")
    feats_grouped = feats.reshape(B, num_heads, N, D // 4, 4)
    # matrix: (B, N, 4, 4); feats_grouped: (B, num_heads, N, K, 4)
    out = torch.einsum("bnij,bhnkj->bhnki", matrix, feats_grouped)
    return out.reshape(B, num_heads, N, D)


def _apply_complex_rope(
    hidden_states: torch.Tensor,
    freqs: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply complex RoPE: treat last-dim pairs as ``(real, imag)`` and multiply
    by ``freqs``. Computation runs in fp64 to match the NVlabs reference.

    Args:
        hidden_states: ``(..., D)`` with ``D % 2 == 0``.
        freqs: complex tensor broadcastable to ``(..., D // 2)``.
        inverse: when ``True`` apply ``freqs.conj()`` (used for ``apply_o``).
    """
    x_real = hidden_states.to(torch.float64).contiguous()
    x_complex = torch.view_as_complex(x_real.unflatten(-1, (-1, 2)))
    if inverse:
        freqs = freqs.conj()
    x_out = torch.view_as_real(x_complex * freqs).flatten(-2, -1)
    return x_out.to(hidden_states.dtype)


def _apply_block_diagonal(
    feats: torch.Tensor,
    func_size_pairs: list[tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Split ``feats`` along the last axis by ``block_sizes`` and apply one
    callable per block, then concatenate. Preserves ``feats.shape``.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    if feats.shape[-1] != sum(block_sizes):
        raise ValueError(
            f"UCPE block-diagonal block sum {sum(block_sizes)} != feats D {feats.shape[-1]}."
        )
    blocks = torch.split(feats, list(block_sizes), dim=-1)
    out = torch.cat([func(block) for func, block in zip(funcs, blocks)], dim=-1)
    if out.shape != feats.shape:
        raise RuntimeError(
            f"UCPE block-diagonal output shape {tuple(out.shape)} != input {tuple(feats.shape)}."
        )
    return out


def _slice_rope_for_cam(
    rotary_emb: torch.Tensor | None,
    head_dim: int,
    rope_dim: int,
) -> torch.Tensor | None:
    """Re-slice WAN-style RoPE frequencies for a smaller ``rope_dim`` using the
    same ``(T, H, W)`` split as the main branch. Mirrors NVlabs slicing logic.
    """
    if rotary_emb is None:
        return None
    orig_t = head_dim // 2 - 2 * (head_dim // 6)
    orig_h = head_dim // 6
    new_t = rope_dim // 2 - 2 * (rope_dim // 6)
    new_h = rope_dim // 6
    new_w = rope_dim // 6
    t_part = rotary_emb[..., :new_t]
    h_part = rotary_emb[..., orig_t : orig_t + new_h]
    w_part = rotary_emb[..., orig_t + orig_h : orig_t + orig_h + new_w]
    return torch.cat([t_part, h_part, w_part], dim=-1)


# ---------------------------------------------------------------------------
# Apply-fn preparation
# ---------------------------------------------------------------------------


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _prepare_ray_apply_fns(
    head_dim: int,
    P: torch.Tensor,  # (B, N, 4, 4) ray <- world
    P_T: torch.Tensor,  # (B, N, 4, 4) ray <- world transpose
    P_inv: torch.Tensor,  # (B, N, 4, 4) world <- ray
    rotary_emb: torch.Tensor | None = None,
) -> tuple[Callable, Callable, Callable]:
    """Build ``(apply_q, apply_kv, apply_o)`` block-diagonal callables.

    Each callable transforms a ``(B, num_heads, N, head_dim)`` tensor:
    the first ``head_dim // 2`` channels go through the per-token 4x4
    projection, the remaining ``head_dim // 2`` channels go through complex
    RoPE (or identity when ``rotary_emb`` is None).
    """
    if rotary_emb is not None:
        rope_fwd = partial(_apply_complex_rope, freqs=rotary_emb, inverse=False)
        rope_inv = partial(_apply_complex_rope, freqs=rotary_emb, inverse=True)
    else:
        rope_fwd = _identity
        rope_inv = _identity

    half = head_dim // 2

    transforms_q = [
        (partial(_apply_ray_projmat, matrix=P_T), half),
        (rope_fwd, half),
    ]
    transforms_kv = [
        (partial(_apply_ray_projmat, matrix=P_inv), half),
        (rope_fwd, half),
    ]
    transforms_o = [
        (partial(_apply_ray_projmat, matrix=P), half),
        (rope_inv, half),
    ]

    apply_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_q, apply_kv, apply_o


def prepare_prope_fns(
    head_dim: int,
    camera_conditions: torch.Tensor,
    HW: tuple[int, int, int],
    rotary_emb: torch.Tensor | None = None,
) -> tuple[Callable, Callable, Callable]:
    """Precompute the UCPE ``(apply_q, apply_kv, apply_o)`` callables once
    per request. Shared across all transformer blocks via the per-block
    ``precomputed_gates`` plumbing in ``SanaWmSelfAttention``.

    Args:
        head_dim: per-head channel count. Must be divisible by 4.
        camera_conditions: ``(B, T, 20)`` SANA-WM camera payload with
            intrinsics in latent-pixel units (see ``_process_camera_conditions``).
        HW: ``(T_latent, H_latent, W_latent)`` latent-grid shape.
        rotary_emb: complex rotary embeddings ``(1, 1, N, D//2)`` produced
            by the main-branch RoPE module, or ``None`` to skip the rope
            block in the block-diagonal transform.
    """
    if head_dim % 4 != 0:
        raise ValueError(f"UCPE head_dim must be divisible by 4, got {head_dim}.")
    B = camera_conditions.shape[0]
    raymats = _process_camera_conditions(camera_conditions, HW)
    raymats_flat = raymats.reshape(B, -1, 4, 4)
    P = raymats_flat
    P_T = P.transpose(-1, -2)
    P_inv = _invert_se3(P)
    rotary_emb_cam = _slice_rope_for_cam(rotary_emb, head_dim, head_dim // 2)
    return _prepare_ray_apply_fns(head_dim, P, P_T, P_inv, rotary_emb=rotary_emb_cam)


__all__ = [
    "prepare_prope_fns",
]
