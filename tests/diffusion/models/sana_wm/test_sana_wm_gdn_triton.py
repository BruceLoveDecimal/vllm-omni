# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytestmark = [pytest.mark.diffusion, pytest.mark.gpu]


def test_sana_wm_triton_gdn_matches_reference_small() -> None:
    import torch
    from torch import nn

    from vllm_omni.diffusion.models.sana_wm.gated_deltanet_triton import (
        reference_bidirectional_gated_delta_net,
        triton_bidirectional_gated_delta_net_from_qkv,
    )

    if not torch.cuda.is_available():
        pytest.skip("Sana-WM Triton GDN parity test requires CUDA.")

    torch.manual_seed(0)
    batch_size, token_count, num_heads, head_dim = 1, 8, 2, 16
    frames, spatial_tokens = 2, 4
    qkv = torch.randn(
        batch_size,
        token_count,
        3,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    beta = torch.sigmoid(torch.randn(batch_size, num_heads, frames, spatial_tokens, device="cuda"))
    decay = torch.sigmoid(torch.randn(batch_size, num_heads, frames, device="cuda"))

    class Norm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(num_heads * head_dim, device="cuda"))
            self.eps = 1e-6

    q_norm = Norm()
    k_norm = Norm()
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
