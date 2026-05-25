# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytestmark = [pytest.mark.diffusion, pytest.mark.gpu]


def _make_norm(num_heads: int, head_dim: int):
    import torch
    from torch import nn

    class Norm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(num_heads * head_dim, device="cuda"))
            self.eps = 1e-6

    return Norm()


def _assert_sana_wm_triton_gdn_matches_reference(
    *,
    batch_size: int,
    frames: int,
    spatial_tokens: int,
    num_heads: int,
    head_dim: int,
    dtype,
) -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm.gated_deltanet_triton import (
        reference_bidirectional_gated_delta_net,
        triton_bidirectional_gated_delta_net_from_qkv,
    )

    if not torch.cuda.is_available():
        pytest.skip("Sana-WM Triton GDN parity test requires CUDA.")

    torch.manual_seed(0)
    token_count = frames * spatial_tokens
    qkv = torch.randn(
        batch_size,
        token_count,
        3,
        num_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    beta = torch.sigmoid(torch.randn(batch_size, num_heads, frames, spatial_tokens, device="cuda"))
    decay = torch.sigmoid(torch.randn(batch_size, num_heads, frames, device="cuda"))
    q_norm = _make_norm(num_heads, head_dim)
    k_norm = _make_norm(num_heads, head_dim)
    q_raw, k_raw, value = qkv.unbind(dim=2)

    def rms_norm(raw: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        flattened = raw.reshape(batch_size, token_count, num_heads * head_dim)
        normalized = flattened * torch.rsqrt(flattened.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return (normalized.to(raw.dtype) * weight.to(raw.dtype)).reshape(batch_size, token_count, num_heads, head_dim)

    k_scale = (head_dim**-0.5) * (spatial_tokens**-0.5)
    query = torch.relu(rms_norm(q_raw, q_norm.weight)).permute(0, 2, 3, 1)
    key = torch.relu(rms_norm(k_raw, k_norm.weight)).permute(0, 2, 3, 1) * k_scale
    value = value.permute(0, 2, 3, 1)

    reference = reference_bidirectional_gated_delta_net(
        query,
        key,
        value,
        beta=beta,
        decay=decay,
        spatial_tokens=spatial_tokens,
    )
    fused = triton_bidirectional_gated_delta_net_from_qkv(
        qkv,
        beta=beta,
        decay=decay,
        q_norm=q_norm,
        k_norm=k_norm,
        spatial_tokens=spatial_tokens,
        k_scale=k_scale,
    )

    torch.testing.assert_close(fused.float(), reference.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "batch_size,frames,spatial_tokens,num_heads,head_dim",
    [
        (1, 1, 1, 1, 16),
        (1, 2, 4, 2, 16),
        (2, 3, 5, 2, 32),
    ],
)
def test_sana_wm_triton_gdn_matches_reference_shapes(
    batch_size: int,
    frames: int,
    spatial_tokens: int,
    num_heads: int,
    head_dim: int,
) -> None:
    import torch

    _assert_sana_wm_triton_gdn_matches_reference(
        batch_size=batch_size,
        frames=frames,
        spatial_tokens=spatial_tokens,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
    )


def test_sana_wm_triton_gdn_matches_reference_fp32() -> None:
    import torch

    _assert_sana_wm_triton_gdn_matches_reference(
        batch_size=1,
        frames=2,
        spatial_tokens=4,
        num_heads=2,
        head_dim=16,
        dtype=torch.float32,
    )


def test_sana_wm_triton_gdn_disable_env_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm.gated_deltanet_triton import (
        SANA_WM_DISABLE_TRITON_GDN_ENV,
        triton_bidirectional_gated_delta_net_from_qkv,
    )

    if not torch.cuda.is_available():
        pytest.skip("Sana-WM Triton GDN fail-fast test requires CUDA.")

    monkeypatch.setenv(SANA_WM_DISABLE_TRITON_GDN_ENV, "1")
    qkv = torch.zeros(1, 1, 3, 1, 16, device="cuda", dtype=torch.bfloat16)
    beta = torch.full((1, 1, 1, 1), 0.5, device="cuda")
    decay = torch.full((1, 1, 1), 0.9, device="cuda")
    norm = _make_norm(num_heads=1, head_dim=16)
    with pytest.raises(RuntimeError, match=SANA_WM_DISABLE_TRITON_GDN_ENV):
        triton_bidirectional_gated_delta_net_from_qkv(
            qkv,
            beta=beta,
            decay=decay,
            q_norm=norm,
            k_norm=norm,
            spatial_tokens=1,
            k_scale=16**-0.5,
        )


def test_sana_wm_triton_gdn_invalid_spatial_tokens() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm.gated_deltanet_triton import (
        triton_bidirectional_gated_delta_net_from_qkv,
    )

    if not torch.cuda.is_available():
        pytest.skip("Sana-WM Triton GDN validation test requires CUDA.")

    qkv = torch.zeros(1, 3, 3, 1, 16, device="cuda", dtype=torch.bfloat16)
    beta = torch.full((1, 1, 1, 2), 0.5, device="cuda")
    decay = torch.full((1, 1, 1), 0.9, device="cuda")
    norm = _make_norm(num_heads=1, head_dim=16)
    with pytest.raises(ValueError, match="not divisible"):
        triton_bidirectional_gated_delta_net_from_qkv(
            qkv,
            beta=beta,
            decay=decay,
            q_norm=norm,
            k_norm=norm,
            spatial_tokens=2,
            k_scale=16**-0.5,
        )


def test_sana_wm_triton_gdn_warmup_runs() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm.gated_deltanet_triton import (
        warmup_sana_wm_gdn_kernel,
    )

    if not torch.cuda.is_available():
        pytest.skip("Sana-WM Triton GDN warmup test requires CUDA.")

    assert warmup_sana_wm_gdn_kernel(frames=1, spatial_tokens=1, num_heads=1, head_dim=16)
