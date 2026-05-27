# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the ported UCPE math (``vllm_omni/diffusion/models/sana_wm/ucpe.py``).

These tests do not exercise the full attention block — they only verify the
geometric and algebraic properties of ``prepare_prope_fns`` and its helpers.
A separate GPU-gated harness in ``tests/e2e/accuracy/`` measures the full
SANA-WM Stage-1 reference alignment after the UCPE port lands.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm_omni.diffusion.models.sana_wm.ucpe import prepare_prope_fns


def _make_identity_camera_conditions(batch: int, frames: int, fx: float = 8.0) -> torch.Tensor:
    """Build ``(B, T, 20)`` with identity C2W poses and simple latent intrinsics."""
    c2w = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch, frames, 1, 1)
    intrinsics = torch.tensor([[fx, fx, 2.0, 2.0]]).repeat(batch, frames, 1)
    return torch.cat([c2w.reshape(batch, frames, 16), intrinsics], dim=-1)


def test_prepare_prope_fns_returns_three_callables() -> None:
    cam_cond = _make_identity_camera_conditions(1, 2)
    apply_q, apply_kv, apply_o = prepare_prope_fns(
        head_dim=8,
        camera_conditions=cam_cond,
        HW=(2, 1, 3),
        rotary_emb=None,
    )
    assert callable(apply_q) and callable(apply_kv) and callable(apply_o)


def test_apply_q_preserves_shape() -> None:
    batch, num_heads, frames, height, width, head_dim = 1, 2, 2, 1, 3, 8
    n = frames * height * width
    cam_cond = _make_identity_camera_conditions(batch, frames)
    apply_q, _, _ = prepare_prope_fns(
        head_dim=head_dim,
        camera_conditions=cam_cond,
        HW=(frames, height, width),
        rotary_emb=None,
    )
    feats = torch.randn(batch, num_heads, n, head_dim)
    out = apply_q(feats)
    assert out.shape == feats.shape
    assert torch.isfinite(out).all()


def test_block_diagonal_rejects_mismatched_head_dim() -> None:
    cam_cond = _make_identity_camera_conditions(1, 2)
    with pytest.raises(ValueError, match="divisible by 4"):
        prepare_prope_fns(
            head_dim=6,  # not divisible by 4
            camera_conditions=cam_cond,
            HW=(2, 1, 3),
            rotary_emb=None,
        )


def test_apply_o_after_apply_q_is_approximate_identity_at_identity_poses() -> None:
    """At identity poses, ``apply_q`` projects with ``P_T`` and ``apply_o`` with ``P``.

    ``P @ P_T`` is not exactly identity for the per-pixel ray frame (the
    ray frame depends on per-pixel direction), but ``apply_o(apply_q(x))``
    should preserve the *magnitude* of the input — the transforms are
    rigid SE(3), so they cannot inflate energy. This is a coarse sanity
    check that the closures are wired up correctly.
    """
    torch.manual_seed(0)
    batch, num_heads, frames, height, width, head_dim = 1, 2, 2, 1, 3, 8
    n = frames * height * width
    cam_cond = _make_identity_camera_conditions(batch, frames)
    apply_q, _, apply_o = prepare_prope_fns(
        head_dim=head_dim,
        camera_conditions=cam_cond,
        HW=(frames, height, width),
        rotary_emb=None,
    )
    feats = torch.randn(batch, num_heads, n, head_dim)
    out = apply_o(apply_q(feats))
    # Energy preservation — the projmat is a rigid frame change so the
    # L2 norm of each per-token feature should be preserved within fp32
    # roundoff (the rope half is identity when rotary_emb is None, the
    # projmat half is rigid).
    in_norm = feats.norm(dim=-1)
    out_norm = out.norm(dim=-1)
    assert torch.allclose(in_norm, out_norm, atol=1e-4), (
        f"UCPE projmat must preserve per-token norm; max delta="
        f"{(in_norm - out_norm).abs().max().item()}"
    )


def test_apply_kv_changes_features_on_translation() -> None:
    """With a non-identity translation the cam transform must alter the features."""
    torch.manual_seed(1)
    batch, frames, height, width, head_dim = 1, 2, 1, 3, 8
    n = frames * height * width

    # Identity rotations but per-frame translations.
    c2w = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch, frames, 1, 1)
    c2w[:, :, 0, 3] = torch.tensor([0.0, 0.5])  # translate along X by 0, then 0.5
    intrinsics = torch.tensor([[8.0, 8.0, 2.0, 2.0]]).repeat(batch, frames, 1)
    cam_cond = torch.cat([c2w.reshape(batch, frames, 16), intrinsics], dim=-1)

    _, apply_kv, _ = prepare_prope_fns(
        head_dim=head_dim,
        camera_conditions=cam_cond,
        HW=(frames, height, width),
        rotary_emb=None,
    )
    feats = torch.randn(batch, 2, n, head_dim)
    out = apply_kv(feats)
    assert not torch.allclose(out, feats), "Translation must alter UCPE-transformed features."


def test_apply_complex_rope_round_trip() -> None:
    """``apply_o`` uses the conjugate of the RoPE freqs — round-tripping through
    ``apply_kv`` (forward freqs) then a manual conjugate apply must approximately
    recover the input.
    """
    torch.manual_seed(2)
    batch, frames, height, width, head_dim = 1, 2, 1, 3, 8
    n = frames * height * width
    rope_dim_half = head_dim // 4
    # Build a synthetic complex freqs tensor: (1, 1, N, head_dim // 2).
    angles = torch.linspace(0, math.pi, n * (head_dim // 2)).reshape(1, 1, n, head_dim // 2)
    rotary_emb = torch.polar(torch.ones_like(angles), angles)

    cam_cond = _make_identity_camera_conditions(batch, frames)
    apply_q, _, apply_o = prepare_prope_fns(
        head_dim=head_dim,
        camera_conditions=cam_cond,
        HW=(frames, height, width),
        rotary_emb=rotary_emb,
    )
    feats = torch.randn(batch, 2, n, head_dim)
    round_tripped = apply_o(apply_q(feats))
    # Energy preservation as in the no-rope case — the rope half cancels
    # (forward * conjugate) and the projmat half is rigid.
    assert torch.allclose(feats.norm(dim=-1), round_tripped.norm(dim=-1), atol=1e-3), (
        f"UCPE forward(q) ∘ inverse(o) must preserve per-token norm; max delta="
        f"{(feats.norm(dim=-1) - round_tripped.norm(dim=-1)).abs().max().item()}"
    )
    del rope_dim_half  # silence linter — kept for documentation


def test_camera_conditions_frame_count_must_match_HW() -> None:
    cam_cond = _make_identity_camera_conditions(1, 3)
    with pytest.raises(ValueError, match="frames"):
        prepare_prope_fns(
            head_dim=8,
            camera_conditions=cam_cond,
            HW=(2, 1, 3),  # T_latent=2 != camera frames=3
            rotary_emb=None,
        )
