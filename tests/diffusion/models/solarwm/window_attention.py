# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cache-free comparison harness for the pinned SolarWM window predicate."""

import torch
from torch import nn

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.models.solarwm.camera import apply_projective_transform, camera_projection


class SolarWMWindowAttention(nn.Module):
    """Fused PRoPE after ordinary RoPE, followed by chunk-causal attention.

    Q and K must already have QK normalization and window-relative Wan RoPE.
    V must be raw. A chunk is internally bidirectional, so token-causal attention
    would be incorrect. Splitting query chunks avoids a quadratic dense mask.
    This correctness implementation recomputes clean history; it is not a KV cache.
    """

    def __init__(self, num_heads: int, head_dim: int, prefix: str = ""):
        super().__init__()
        if head_dim % 4:
            raise ValueError("SolarWM PRoPE requires head_dim divisible by 4")
        self.attention = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            causal=False,
            softmax_scale=head_dim**-0.5,
            prefix=prefix,
            skip_sequence_parallel=True,
            disable_kv_quant=True,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        viewmats: torch.Tensor,
        intrinsics: torch.Tensor,
        tokens_per_chunk: int | None = None,
        translation_transform: str = "linear",
    ) -> torch.Tensor:
        if q.shape != k.shape or k.shape != v.shape:
            raise ValueError("cache-free attention requires matching Q/K/V shapes")
        if tokens_per_chunk is not None and tokens_per_chunk <= 0:
            raise ValueError("tokens_per_chunk must be positive")
        projection, inverse = camera_projection(viewmats, intrinsics, translation_transform)
        q = apply_projective_transform(q, projection.transpose(-1, -2))
        k = apply_projective_transform(k, inverse)
        v = apply_projective_transform(v, inverse)
        chunk = tokens_per_chunk or q.shape[1]
        pieces = []
        for start in range(0, q.shape[1], chunk):
            end = min(start + chunk, q.shape[1])
            pieces.append(self.attention(q[:, start:end], k[:, :end], v[:, :end]))
        return apply_projective_transform(torch.cat(pieces, dim=1), projection)
