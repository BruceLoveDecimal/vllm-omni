# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PixelUMM normalization kernels preserving ATen reductions and FP32 MRoPE.

The ordinary norm rounds to BF16 before scaling. Split QK norms instead keep
FP32 through weighting and each T/H/W rotation, then round once for attention.
The shared diffusion QK fusion rounds normalized values before rotation and
cannot express this split-norm contract. Unsupported layouts use eager math.
"""

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

if HAS_TRITON:

    @triton.jit
    def _norm4096(x_ptr, w_ptr, out_ptr, rows, eps: tl.constexpr, rows_per_block: tl.constexpr):
        row = tl.program_id(0) * rows_per_block + tl.arange(0, rows_per_block)
        lane = tl.arange(0, 32)
        base = row[:, None] * 4096 + lane[None, :] * 4
        a = tl.full((rows_per_block, 32), 0, tl.float32)
        b = tl.full((rows_per_block, 32), 0, tl.float32)
        c = tl.full((rows_per_block, 32), 0, tl.float32)
        d = tl.full((rows_per_block, 32), 0, tl.float32)
        # ATen mean on contiguous FP32 rows uses four independent accumulators
        # per lane, then combines them before descending warp shuffles.
        for i in range(32):
            x0 = tl.load(x_ptr + base + i * 128, row[:, None] < rows, 0).to(tl.float32)
            x1 = tl.load(x_ptr + base + i * 128 + 1, row[:, None] < rows, 0).to(tl.float32)
            x2 = tl.load(x_ptr + base + i * 128 + 2, row[:, None] < rows, 0).to(tl.float32)
            x3 = tl.load(x_ptr + base + i * 128 + 3, row[:, None] < rows, 0).to(tl.float32)
            a += x0 * x0
            b += x1 * x1
            c += x2 * x2
            d += x3 * x3
        variance = tl.sum(((a + b) + c) + d, 1) * (1.0 / 4096)
        inv = tldevice.rsqrt(variance + eps)
        col = tl.arange(0, 4096)
        x = tl.load(x_ptr + row[:, None] * 4096 + col[None, :], row[:, None] < rows, 0).to(tl.float32)
        weight = tl.load(w_ptr + col).to(tl.float32)
        normalized = (x * inv[:, None]).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + row[:, None] * 4096 + col[None, :], normalized * weight[None, :], row[:, None] < rows)

    @triton.jit
    def _split_norm_rope(
        x_ptr,
        wt_ptr,
        whw_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        rows,
        heads: tl.constexpr,
        eps: tl.constexpr,
        rows_per_block: tl.constexpr,
    ):
        row = tl.program_id(0) * rows_per_block + tl.arange(0, rows_per_block)
        lane = tl.arange(0, 32)
        base = row[:, None] * 128
        t0 = tl.load(x_ptr + base + lane[None, :], row[:, None] < rows, 0).to(tl.float32)
        t1 = tl.load(x_ptr + base + lane[None, :] + 32, row[:, None] < rows, 0).to(tl.float32)
        h0 = tl.load(x_ptr + base + lane[None, :] + 64, row[:, None] < rows, 0).to(tl.float32)
        h1 = tl.load(x_ptr + base + lane[None, :] + 96, row[:, None] < rows, 0).to(tl.float32)
        inv_t = tldevice.rsqrt(tl.sum(t0 * t0 + t1 * t1, 1) * (1.0 / 64) + eps)
        inv_hw = tldevice.rsqrt(tl.sum(h0 * h0 + h1 * h1, 1) * (1.0 / 64) + eps)
        col = tl.arange(0, 128)
        # Pair within T(64), H(32), W(32), rather than across the whole head.
        half = tl.where(col < 64, 32, 16)
        pair = col ^ half
        sign = tl.where((col & half) == 0, -1.0, 1.0)
        inv = tl.where(col[None, :] < 64, inv_t[:, None], inv_hw[:, None])
        weight = tl.where(col < 64, tl.load(wt_ptr + col % 64), tl.load(whw_ptr + col % 64)).to(tl.float32)
        pair_weight = tl.where(pair < 64, tl.load(wt_ptr + pair % 64), tl.load(whw_ptr + pair % 64)).to(tl.float32)
        x = tl.load(x_ptr + base + col[None, :], row[:, None] < rows, 0).to(tl.float32)
        other = tl.load(x_ptr + base + pair[None, :], row[:, None] < rows, 0).to(tl.float32)
        normalized = (x * inv) * weight[None, :]
        rotated = ((other * inv) * pair_weight[None, :]) * sign[None, :]
        # cos/sin have one row per token, shared by all heads and batches.
        token = (row // heads) % (rows // heads)
        cosine = tl.load(cos_ptr + token[:, None] * 128 + col[None, :], row[:, None] < rows, 0)
        sine = tl.load(sin_ptr + token[:, None] * 128 + col[None, :], row[:, None] < rows, 0)
        tl.store(out_ptr + base + col[None, :], normalized * cosine + rotated * sine, row[:, None] < rows)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor | None:
    """Return cast-before-scale BF16 RMSNorm, or decline unsupported inputs."""
    if (
        not HAS_TRITON
        or not current_platform.is_cuda()
        or x.device.type != "cuda"
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or x.dtype != torch.bfloat16
        or weight.dtype != x.dtype
        or x.shape[-1] != 4096
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or weight.shape != (4096,)
        or weight.device != x.device
        or x.numel() // 4096 < 16
    ):
        return None
    out = torch.empty_like(x)
    rows = x.numel() // 4096
    with torch.get_device_module().device(x.device):
        _norm4096[(triton.cdiv(rows, 4),)](
            x, weight, out, rows, eps, rows_per_block=4, num_warps=4, enable_fp_fusion=False
        )
    return out


def split_norm_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, weight_t: torch.Tensor, weight_hw: torch.Tensor, eps: float
) -> torch.Tensor | None:
    """Fuse PixelUMM's two FP32 norms and three rotations for BF16 D=128 Q/K."""
    if (
        not HAS_TRITON
        or not current_platform.is_cuda()
        or x.device.type != "cuda"
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or x.dtype != torch.bfloat16
        or x.ndim != 4
        or x.shape[0] != 1
        or x.shape[-1] != 128
        or not x.is_contiguous()
        or not cos.is_contiguous()
        or not sin.is_contiguous()
        or cos.dtype != torch.float32
        or sin.dtype != torch.float32
        or cos.shape != (1, x.shape[1], 1, 128)
        or sin.shape != cos.shape
        or cos.device != x.device
        or sin.device != x.device
        or weight_t.shape != (64,)
        or weight_hw.shape != (64,)
        or weight_t.dtype != x.dtype
        or weight_hw.dtype != x.dtype
        or weight_t.device != x.device
        or weight_hw.device != x.device
        or not weight_t.is_contiguous()
        or not weight_hw.is_contiguous()
        or x.numel() == 0
    ):
        return None
    out = torch.empty_like(x)
    rows = x.numel() // 128
    with torch.get_device_module().device(x.device):
        _split_norm_rope[(triton.cdiv(rows, 4),)](
            x,
            weight_t,
            weight_hw,
            cos,
            sin,
            out,
            rows,
            x.shape[2],
            eps,
            rows_per_block=4,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
