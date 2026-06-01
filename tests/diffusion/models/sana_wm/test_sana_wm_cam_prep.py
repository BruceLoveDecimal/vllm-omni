# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import sys

import pytest
import torch

from vllm_omni.diffusion.models.sana_wm.cam_prep import cam_prep_func, prepare_cam_prep_context


pytestmark = [pytest.mark.diffusion]


def _make_identity_camera_conditions(batch: int, frames: int, fx: float = 8.0) -> torch.Tensor:
    c2w = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch, frames, 1, 1)
    intrinsics = torch.tensor([[fx, fx, 2.0, 2.0]]).repeat(batch, frames, 1)
    return torch.cat([c2w.reshape(batch, frames, 16), intrinsics], dim=-1)


def _manual_rms_norm(raw: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    batch, token_count, num_heads, head_dim = raw.shape
    inv_rms = torch.rsqrt(raw.float().square().sum(dim=(-1, -2)) / float(num_heads * head_dim) + eps)
    return raw.float() * inv_rms.reshape(batch, token_count, 1, 1) * weight.reshape(num_heads, head_dim)


def _manual_rope(feats: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
    half_dim = feats.shape[-1]
    pair_index = torch.arange(half_dim, device=feats.device) ^ 1
    cos = rope_cos.reshape(1, rope_cos.shape[0], 1, half_dim)
    sin = rope_sin.reshape(1, rope_sin.shape[0], 1, half_dim)
    return feats * cos + feats.index_select(-1, pair_index) * sin


def _manual_cam_prep(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    *,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_scale: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    head_dim = q_raw.shape[-1]
    half_dim = head_dim // 2
    q_norm = torch.relu(_manual_rms_norm(q_raw, q_norm_weight, norm_eps))
    k_norm = torch.relu(_manual_rms_norm(k_raw, k_norm_weight, norm_eps)) * k_scale
    value = v_raw.float()

    def project(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        grouped = feats.reshape(*feats.shape[:-1], half_dim // 4, 4)
        return torch.einsum("bnij,bnhgj->bnhgi", matrix.float(), grouped).reshape_as(feats)

    q_out = torch.cat(
        [
            project(q_norm[..., :half_dim], proj_q),
            _manual_rope(q_norm[..., half_dim:], rope_cos, rope_sin),
        ],
        dim=-1,
    )
    k_out = torch.cat(
        [
            project(k_norm[..., :half_dim], proj_kv),
            _manual_rope(k_norm[..., half_dim:], rope_cos, rope_sin),
        ],
        dim=-1,
    )
    v_out = torch.cat(
        [
            project(value[..., :half_dim], proj_kv),
            _manual_rope(value[..., half_dim:], rope_cos, rope_sin),
        ],
        dim=-1,
    )
    k_pre_sq = k_norm.square().sum(dim=-1).permute(0, 2, 1).contiguous()
    k_post_sq = k_out.square().sum(dim=-1).permute(0, 2, 1).contiguous()
    inflation_sq = k_post_sq.clamp_min(1e-12) / k_pre_sq.clamp_min(1e-12)
    return (
        q_out.permute(0, 2, 3, 1).contiguous().to(q_raw.dtype),
        k_out.permute(0, 2, 3, 1).contiguous().to(q_raw.dtype),
        v_out.permute(0, 2, 3, 1).contiguous().to(q_raw.dtype),
        inflation_sq,
    )


def test_cam_prep_func_matches_manual_reference() -> None:
    torch.manual_seed(0)
    batch, frames, height, width, num_heads, head_dim = 2, 2, 1, 3, 3, 8
    token_count = frames * height * width
    q_raw = torch.randn(batch, token_count, num_heads, head_dim)
    k_raw = torch.randn_like(q_raw)
    v_raw = torch.randn_like(q_raw)
    q_norm_weight = torch.randn(num_heads * head_dim).float()
    k_norm_weight = torch.randn(num_heads * head_dim).float()

    proj_q = torch.randn(batch, token_count, 4, 4).contiguous()
    proj_kv = torch.randn(batch, token_count, 4, 4).contiguous()
    angles = torch.linspace(0.0, 1.0, token_count * (head_dim // 4)).reshape(token_count, head_dim // 4)
    rope_cos = torch.cos(angles).repeat_interleave(2, dim=-1).contiguous()
    rope_sin = torch.stack(
        [-torch.sin(angles), torch.sin(angles)],
        dim=-1,
    ).reshape(token_count, head_dim // 2).contiguous()

    actual = cam_prep_func(
        q_raw,
        k_raw,
        v_raw,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        proj_q=proj_q,
        proj_kv=proj_kv,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        k_scale=0.125,
        norm_eps=1e-6,
    )
    expected = _manual_cam_prep(
        q_raw,
        k_raw,
        v_raw,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        proj_q=proj_q,
        proj_kv=proj_kv,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        k_scale=0.125,
        norm_eps=1e-6,
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=1e-6, rtol=1e-6)


def test_prepare_cam_prep_context_shapes_and_output_fn() -> None:
    torch.manual_seed(1)
    batch, frames, height, width, head_dim = 1, 2, 1, 3, 8
    token_count = frames * height * width
    camera_conditions = _make_identity_camera_conditions(batch, frames)
    context = prepare_cam_prep_context(
        camera_conditions=camera_conditions,
        spatial_shape=(frames, height, width),
        patch_size=(1, 1, 1),
        head_dim=head_dim,
        rotary_emb=None,
    )
    assert context.proj_q.shape == (batch, token_count, 4, 4)
    assert context.proj_kv.shape == (batch, token_count, 4, 4)
    assert context.rope_cos.shape == (token_count, head_dim // 2)
    assert context.rope_sin.shape == (token_count, head_dim // 2)
    feats = torch.randn(batch, 2, token_count, head_dim)
    assert context.apply_output(feats).shape == feats.shape


def test_cam_prep_func_rejects_invalid_head_dim() -> None:
    q_raw = torch.zeros(1, 1, 1, 6)
    with pytest.raises(ValueError, match="head_dim"):
        cam_prep_func(
            q_raw,
            q_raw,
            q_raw,
            q_norm_weight=torch.ones(6),
            k_norm_weight=torch.ones(6),
            proj_q=torch.eye(4).reshape(1, 1, 4, 4),
            proj_kv=torch.eye(4).reshape(1, 1, 4, 4),
            rope_cos=torch.ones(1, 3),
            rope_sin=torch.zeros(1, 3),
            k_scale=1.0,
            norm_eps=1e-6,
        )


@pytest.mark.gpu
def test_cam_prep_func_matches_nvlabs_kernel_when_available() -> None:
    repo = os.environ.get("SANA_WM_OFFICIAL_REPO", "").strip()
    if not repo:
        pytest.skip("Set SANA_WM_OFFICIAL_REPO to run NVlabs cam_prep_func parity.")
    if not torch.cuda.is_available():
        pytest.skip("NVlabs cam_prep_func parity requires CUDA.")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from diffusion.model.ops.fused_cam_gdn import cam_prep_func as nvlabs_cam_prep_func
    except Exception as exc:
        pytest.skip(f"Could not import NVlabs cam_prep_func: {exc}")

    torch.manual_seed(2)
    batch, frames, height, width, num_heads, head_dim = 1, 2, 1, 3, 2, 8
    token_count = frames * height * width
    q_raw = torch.randn(batch, token_count, num_heads, head_dim, device="cuda", dtype=torch.float32).contiguous()
    k_raw = torch.randn_like(q_raw).contiguous()
    v_raw = torch.randn_like(q_raw).contiguous()
    q_norm_weight = torch.randn(num_heads * head_dim, device="cuda", dtype=torch.float32).contiguous()
    k_norm_weight = torch.randn(num_heads * head_dim, device="cuda", dtype=torch.float32).contiguous()
    proj_q = torch.randn(batch, token_count, 4, 4, device="cuda", dtype=torch.float32).contiguous()
    proj_kv = torch.randn(batch, token_count, 4, 4, device="cuda", dtype=torch.float32).contiguous()
    angles = torch.linspace(0.0, 1.0, token_count * (head_dim // 4), device="cuda").reshape(token_count, head_dim // 4)
    rope_cos = torch.cos(angles).repeat_interleave(2, dim=-1).contiguous()
    rope_sin = torch.stack(
        [-torch.sin(angles), torch.sin(angles)],
        dim=-1,
    ).reshape(token_count, head_dim // 2).contiguous()

    actual = cam_prep_func(
        q_raw,
        k_raw,
        v_raw,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        proj_q=proj_q,
        proj_kv=proj_kv,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        k_scale=0.125,
        norm_eps=1e-6,
    )
    expected = nvlabs_cam_prep_func(
        q_raw,
        k_raw,
        v_raw,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        proj_q=proj_q,
        proj_kv=proj_kv,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        k_scale=0.125,
        norm_eps=1e-6,
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor.float(), expected_tensor.float(), atol=2e-5, rtol=2e-5)
