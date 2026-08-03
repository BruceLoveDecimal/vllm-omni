from importlib.util import find_spec

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger

from vllm_omni.diffusion.layers.custom_op import CustomOp

logger = init_logger(__name__)

_HAS_MINDIESD = find_spec("mindiesd") is not None


class LayerNorm(nn.LayerNorm, CustomOp):
    """
    LayerNorm implementation that inherits from both ``nn.LayerNorm`` and ``CustomOp``.
    NPU:
        Uses ``mindiesd.fast_layernorm(self, x)`` when MindIE-SD is installed.
    CUDA / HIP / XPU / native:
        Falls back to FP32 nn.LayerNorm implementation.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__(normalized_shape=dim, eps=eps, elementwise_affine=elementwise_affine)
        # CustomOp.__init__ cannot be called here because it would re-run
        # nn.Module initialization and clear LayerNorm parameters.
        self._forward_method = CustomOp.dispatch_forward(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_method(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_MINDIESD:
            try:
                from mindiesd import fast_layernorm

                return fast_layernorm(self, x)
            except ImportError as e:
                logger.warning_once(
                    "mindiesd.fast_layernorm import failed, falling back to FP32 layer_norm: %s",
                    e,
                )

        return self.forward_native(x)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        origin_dtype = x.dtype
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class RMSNorm(CustomOp):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        from vllm._custom_ops import rms_norm as fused_rms_norm

        orig_shape = x.shape
        hidden_size = orig_shape[-1]
        x_2d = x.reshape(-1, hidden_size)
        out = torch.empty_like(x_2d)
        fused_rms_norm(out, x_2d, self.weight.data, self.variance_epsilon)
        return out.reshape(orig_shape)

    def forward_cuda(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # During torch.compile tracing, fused_rms_norm writes to `out` in-place
        # (returns None) and accesses self.weight.data, which is a DTensor under
        # HSDP. Both patterns confuse inductor's compute_ancestors scheduler.
        # Fall back to forward_native so inductor can fuse the pure-PyTorch ops
        # itself.
        if torch.compiler.is_compiling():
            return self.forward_native(x)
        try:
            return self._forward_fused(x)
        except Exception:
            return self.forward_native(x)

    def forward_hip(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return self.forward_native(x)
        try:
            return self._forward_fused(x)
        except Exception:
            return self.forward_native(x)

    def forward_npu(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        import torch_npu

        output = torch_npu.npu_rms_norm(x, gamma=self.weight, epsilon=self.variance_epsilon)[0]

        return output

    def forward_xpu(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_native(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        out = x * torch.rsqrt(variance + self.variance_epsilon)
        out = self.weight.to(torch.float32) * out
        return out.to(input_dtype)


class RMSNormVAE(CustomOp):
    """Root Mean Square Layer Normalization for Channel-First or Last"""

    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None
        self.epsilon = epsilon

        self.gamma_rmsnorm = None

    def forward_cuda(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_hip(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_npu(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        import torch_npu

        if self.gamma_rmsnorm is None:
            self.gamma_rmsnorm = self.gamma.reshape(-1)

        if self.channel_first:
            x = x.transpose(1, -1)
            out = torch_npu.npu_rms_norm(x, self.gamma_rmsnorm, epsilon=self.epsilon)[0].transpose(1, -1)
        else:
            out = torch_npu.npu_rms_norm(x, self.gamma_rmsnorm, epsilon=self.epsilon)[0]

        if self.bias is not None:
            out = out + self.bias
        return out

    def forward_xpu(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_native(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        out = (
            F.normalize(
                x,
                dim=(1 if self.channel_first else -1),
                eps=self.epsilon,
            )
            * self.scale
            * self.gamma
        )
        if self.bias is not None:
            out = out + self.bias
        return out


class TensorParallelRMSNorm(nn.Module):
    """RMSNorm whose statistics span a tensor-parallel sharded last dimension.

    Models that normalize q/k across *all* heads (``qk_norm="rms_norm_across_heads"``)
    cannot use a plain RMSNorm once the q/k projections are column-parallel: each
    rank would then compute the RMS over its local head shard only. This
    all-reduces the squared sum so the denominator matches the global width,
    while the affine weight stays local.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = max(int(tp_size), 1)
        self.global_hidden_size = hidden_size * self.tp_size
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def _local_sum_sq(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_float = x.float()
        return x_float, x_float.pow(2).sum(dim=-1, keepdim=True)

    def _scale(self, x_float: torch.Tensor, global_sum_sq: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        out = x_float * torch.rsqrt(global_sum_sq / self.global_hidden_size + self.eps)
        if self.weight is not None:
            out = out * self.weight.float()
        return out.to(dtype=input_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float, sum_sq = self._local_sum_sq(x)
        if self.tp_size > 1:
            sum_sq = tensor_model_parallel_all_reduce(sum_sq)
        return self._scale(x_float, sum_sq, x.dtype)


def fused_qk_rms_norm(
    norm_q: nn.Module,
    norm_k: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply q/k :class:`TensorParallelRMSNorm` with a SINGLE fused all-reduce.

    Self-attention normalizes q and k every step, and under TP each norm issues
    its own tiny all-reduce of the per-token sum-of-squares. These collectives
    are latency-bound, so packing both sums into one tensor halves the count
    (2 collectives -> 1) for free.

    Numerically identical to ``norm_q(q), norm_k(k)``: all-reduce is
    elementwise, so packing along the last dim reduces each slice independently
    with the same fp32 accumulation. Falls back to independent application when
    either norm is not a :class:`TensorParallelRMSNorm` (e.g. ``nn.Identity``
    when qk_norm is off, or plain ``RMSNorm`` at TP=1).
    """
    if not (isinstance(norm_q, TensorParallelRMSNorm) and isinstance(norm_k, TensorParallelRMSNorm)):
        return norm_q(q), norm_k(k)
    if norm_q.tp_size <= 1:
        return norm_q(q), norm_k(k)

    q_float, q_sum_sq = norm_q._local_sum_sq(q)
    k_float, k_sum_sq = norm_k._local_sum_sq(k)
    packed = tensor_model_parallel_all_reduce(torch.cat([q_sum_sq, k_sum_sq], dim=-1))
    q_sum_sq, k_sum_sq = packed[..., 0:1], packed[..., 1:2]
    return norm_q._scale(q_float, q_sum_sq, q.dtype), norm_k._scale(k_float, k_sum_sq, k.dtype)
