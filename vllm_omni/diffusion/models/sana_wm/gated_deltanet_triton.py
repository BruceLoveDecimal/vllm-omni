# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-local Bidirectional Gated DeltaNet kernels for SANA-WM."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

SANA_WM_GDN_ERROR = (
    "Sana-WM fused Bidirectional Gated DeltaNet Triton kernel is unavailable. "
    "This module falls back to the pure PyTorch reference recurrence when the "
    "fused NVlabs recurrence cannot be launched."
)
SANA_WM_DISABLE_TRITON_GDN_ENV = "VLLM_OMNI_SANA_WM_DISABLE_TRITON_GDN"
SANA_WM_REQUIRE_TRITON_GDN_ENV = "VLLM_OMNI_SANA_WM_REQUIRE_TRITON_GDN"


def _validate_gdn_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    spatial_tokens: int,
) -> tuple[int, int, int, int, int]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Sana-WM GDN query/key/value must be shaped [B, H, D, N].")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            "Sana-WM GDN query/key/value shapes must match, got "
            f"{tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}."
        )
    if beta.ndim != 4:
        raise ValueError("Sana-WM GDN beta must be shaped [B, H, T, S].")
    if decay.ndim != 3:
        raise ValueError("Sana-WM GDN decay must be shaped [B, H, T].")
    if spatial_tokens <= 0:
        raise ValueError("Sana-WM GDN spatial_tokens must be positive.")

    batch_size, num_heads, head_dim, token_count = query.shape
    if token_count % spatial_tokens != 0:
        raise ValueError(
            f"Sana-WM GDN token count {token_count} is not divisible by spatial_tokens={spatial_tokens}."
        )
    frames = token_count // spatial_tokens
    if beta.shape != (batch_size, num_heads, frames, spatial_tokens):
        raise ValueError(
            "Sana-WM GDN beta shape mismatch: expected "
            f"{(batch_size, num_heads, frames, spatial_tokens)}, got {tuple(beta.shape)}."
        )
    if decay.shape != (batch_size, num_heads, frames):
        raise ValueError(
            "Sana-WM GDN decay shape mismatch: expected "
            f"{(batch_size, num_heads, frames)}, got {tuple(decay.shape)}."
        )
    return batch_size, num_heads, head_dim, token_count, frames


def _delta_scan(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_rot: torch.Tensor,
    key_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    spatial_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_heads, head_dim, token_count = query.shape
    frames = beta.shape[2]

    def to_frames(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch_size, num_heads, head_dim, frames, spatial_tokens).permute(0, 1, 3, 2, 4)

    query_f = to_frames(query)
    key_f = to_frames(key)
    value_f = to_frames(value)
    query_rot_f = to_frames(query_rot)
    key_rot_f = to_frames(key_rot)
    state_kv = torch.zeros(batch_size, num_heads, head_dim, head_dim, device=query.device, dtype=query.dtype)
    state_z = torch.zeros(batch_size, num_heads, head_dim, 1, device=query.device, dtype=query.dtype)
    numerators: list[torch.Tensor] = []
    denominators: list[torch.Tensor] = []

    for frame_idx in range(frames):
        query_t = query_f[:, :, frame_idx]
        key_t = key_f[:, :, frame_idx]
        value_t = value_f[:, :, frame_idx]
        query_rot_t = query_rot_f[:, :, frame_idx]
        key_rot_t = key_rot_f[:, :, frame_idx]
        beta_t = beta[:, :, frame_idx].unsqueeze(2)
        decay_t = decay[:, :, frame_idx].view(batch_size, num_heads, 1, 1)

        state_kv = state_kv * decay_t
        state_z = state_z * decay_t
        value_pred = torch.matmul(state_kv, key_rot_t)
        delta_value = (value_t - value_pred) * beta_t
        state_kv = state_kv + torch.matmul(delta_value, key_rot_t.transpose(-1, -2))

        z_pred = torch.matmul(state_z.transpose(-1, -2), key_t)
        delta_z = (1.0 - z_pred) * beta_t
        state_z = state_z + torch.matmul(key_t, delta_z.transpose(-1, -2))

        numerators.append(torch.matmul(state_kv, query_rot_t))
        denominators.append(torch.matmul(state_z.transpose(-1, -2), query_t))

    def restore(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
        stacked = torch.stack(tensors, dim=2)
        return stacked.permute(0, 1, 3, 2, 4).reshape(batch_size, num_heads, dim, token_count)

    return restore(numerators, head_dim), restore(denominators, 1)


def _flip_and_shift(tensor: torch.Tensor, *, dim: int, shift_value: float) -> torch.Tensor:
    flipped = torch.flip(tensor, dims=[dim])
    shifted = flipped.narrow(dim, 0, tensor.shape[dim] - 1)
    pad_shape = list(tensor.shape)
    pad_shape[dim] = 1
    padding = torch.full(pad_shape, shift_value, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([padding, shifted], dim=dim)


def reference_bidirectional_gated_delta_net(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    beta: torch.Tensor,
    decay: torch.Tensor,
    spatial_tokens: int,
    query_rot: torch.Tensor | None = None,
    key_rot: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Run the SANA-WM bidirectional gated delta recurrence in PyTorch.

    Inputs follow the layout used by the official Stage-1 operator:
    ``query/key/value/query_rot/key_rot`` are ``[B, H, D, T*S]``, ``beta`` is
    ``[B, H, T, S]``, and ``decay`` is ``[B, H, T]``.
    """

    batch_size, num_heads, head_dim, token_count, frames = _validate_gdn_inputs(
        query, key, value, beta, decay, spatial_tokens
    )
    if query_rot is None:
        query_rot = query
    if key_rot is None:
        key_rot = key
    if query_rot.shape != query.shape or key_rot.shape != key.shape:
        raise ValueError("Sana-WM GDN rotary query/key shapes must match query/key.")

    dtype_orig = query.dtype
    query = query.float()
    key = key.float()
    value = value.float()
    query_rot = query_rot.float()
    key_rot = key_rot.float()
    beta = beta.float()
    decay = decay.float()

    num_fwd, den_fwd = _delta_scan(
        query,
        key,
        value,
        query_rot,
        key_rot,
        beta,
        decay,
        spatial_tokens=spatial_tokens,
    )

    def to_time(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch_size, num_heads, head_dim, frames, spatial_tokens).permute(0, 1, 3, 2, 4)

    def from_time(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute(0, 1, 3, 2, 4).reshape(batch_size, num_heads, head_dim, token_count)

    query_t = to_time(query)
    key_t = to_time(key)
    value_t = to_time(value)
    query_rot_t = to_time(query_rot)
    key_rot_t = to_time(key_rot)
    num_bwd_flipped, den_bwd_flipped = _delta_scan(
        from_time(torch.flip(query_t, dims=[2])),
        from_time(_flip_and_shift(key_t, dim=2, shift_value=0.0)),
        from_time(_flip_and_shift(value_t, dim=2, shift_value=0.0)),
        from_time(torch.flip(query_rot_t, dims=[2])),
        from_time(_flip_and_shift(key_rot_t, dim=2, shift_value=0.0)),
        _flip_and_shift(beta, dim=2, shift_value=0.0),
        _flip_and_shift(decay, dim=2, shift_value=1.0),
        spatial_tokens=spatial_tokens,
    )

    def flip_back(tensor: torch.Tensor) -> torch.Tensor:
        dim = tensor.shape[2]
        tensor = tensor.view(batch_size, num_heads, dim, frames, spatial_tokens)
        return torch.flip(tensor, dims=[3]).reshape(batch_size, num_heads, dim, token_count)

    output = (num_fwd + flip_back(num_bwd_flipped)) / (den_fwd + flip_back(den_bwd_flipped) + eps)
    return output.to(dtype_orig)


def _triton_disabled() -> bool:
    return os.environ.get(SANA_WM_DISABLE_TRITON_GDN_ENV, "").lower() in {"1", "true", "yes", "on"}


def _rms_norm_weight(module: nn.Module, hidden_size: int, *, device: torch.device) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight.float().contiguous()
    return torch.ones(hidden_size, device=device, dtype=torch.float32)


def triton_bidirectional_gated_delta_net_from_qkv(
    qkv: torch.Tensor,
    *,
    beta: torch.Tensor,
    decay: torch.Tensor,
    q_norm: nn.Module,
    k_norm: nn.Module,
    spatial_tokens: int,
    rotary_emb: torch.Tensor | None = None,
    k_scale: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Run the fused NVlabs bidirectional GDN kernel from raw QKV.

    Args:
      qkv: ``[B, N, 3, H, D]`` raw projected Q/K/V after temporal K-conv.
      beta: ``[B, H, T, S]`` token gates.
      decay: ``[B, H, T]`` per-frame decay.

    Returns:
      Tensor shaped ``[B, H, D, N]`` to match
      :func:`reference_bidirectional_gated_delta_net`.
    """

    if _triton_disabled():
        raise RuntimeError(f"{SANA_WM_DISABLE_TRITON_GDN_ENV} disables the fused Sana-WM GDN kernel.")
    if not qkv.is_cuda:
        raise RuntimeError("Sana-WM fused GDN requires CUDA tensors.")
    if qkv.ndim != 5 or qkv.shape[2] != 3:
        raise ValueError(f"Sana-WM fused GDN expects qkv [B, N, 3, H, D], got {tuple(qkv.shape)}.")
    if spatial_tokens <= 0:
        raise ValueError("Sana-WM fused GDN spatial_tokens must be positive.")

    batch_size, token_count, _, num_heads, head_dim = qkv.shape
    if token_count % spatial_tokens != 0:
        raise ValueError(
            f"Sana-WM fused GDN token count {token_count} is not divisible by spatial_tokens={spatial_tokens}."
        )
    frames = token_count // spatial_tokens
    if beta.shape != (batch_size, num_heads, frames, spatial_tokens):
        raise ValueError(
            "Sana-WM fused GDN beta shape mismatch: expected "
            f"{(batch_size, num_heads, frames, spatial_tokens)}, got {tuple(beta.shape)}."
        )
    if decay.shape != (batch_size, num_heads, frames):
        raise ValueError(
            "Sana-WM fused GDN decay shape mismatch: expected "
            f"{(batch_size, num_heads, frames)}, got {tuple(decay.shape)}."
        )

    from vllm_omni.diffusion.models.sana_wm.fused_gdn import (
        fused_bigdn_func,
        fused_qk_inv_rms,
        prepare_rope_tables,
    )

    qkv = qkv.contiguous()
    q_inv_rms, k_inv_rms = fused_qk_inv_rms(qkv, eps=float(getattr(q_norm, "eps", 1e-5)))
    hidden_size = num_heads * head_dim
    q_norm_weight = _rms_norm_weight(q_norm, hidden_size, device=qkv.device)
    k_norm_weight = _rms_norm_weight(k_norm, hidden_size, device=qkv.device)
    rope_cos, rope_sin = prepare_rope_tables(rotary_emb, token_count, head_dim, qkv.device)
    output = fused_bigdn_func(
        qkv,
        q_inv_rms,
        k_inv_rms,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        beta=beta.contiguous(),
        decay=decay.contiguous(),
        F=frames,
        S=spatial_tokens,
        k_scale=k_scale,
        eps=eps,
    )
    return output.permute(0, 2, 3, 1).contiguous()


class BidirectionalGatedDeltaNetTriton(nn.Module):
    """Reference-compatible wrapper for SANA-WM's model-local GDN operator.

    The fused path is intentionally model-local: SANA-WM uses a bidirectional
    video-latent recurrence instead of the autoregressive GDN cache contract
    implemented by vLLM's Qwen3-Next layers.
    """

    def __init__(self, *, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.triton_available = True

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        try:
            beta = kwargs.pop("beta")
            decay = kwargs.pop("decay")
            spatial_tokens = int(kwargs.pop("spatial_tokens"))
        except KeyError as exc:
            raise TypeError(f"Sana-WM GDN missing required argument: {exc.args[0]}") from exc
        if kwargs.keys() - {"query_rot", "key_rot", "eps"}:
            raise TypeError(f"Unexpected Sana-WM GDN arguments: {sorted(kwargs)}")
        return reference_bidirectional_gated_delta_net(
            query,
            key,
            value,
            beta=beta,
            decay=decay,
            spatial_tokens=spatial_tokens,
            query_rot=kwargs.pop("query_rot", None),
            key_rot=kwargs.pop("key_rot", None),
            eps=float(kwargs.pop("eps", self.eps)),
        )


def warmup_sana_wm_gdn_kernel(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 1,
    frames: int = 2,
    spatial_tokens: int = 4,
    num_heads: int = 2,
    head_dim: int = 16,
) -> bool:
    """Compile/autotune the fused SANA-WM GDN kernel with a small workload.

    vLLM's Qwen3-Next GDN path warms its chunked prefill kernels before real
    inference to avoid first-request autotune after cache allocation. SANA-WM
    does not use the autoregressive GDN cache path, but the same operational
    concern applies to the model-local Triton recurrence.

    Returns ``True`` when the warmup ran. Returns ``False`` when CUDA is not
    available or the fused path is explicitly disabled.
    """

    if _triton_disabled() or not torch.cuda.is_available():
        return False
    device = torch.device(device or "cuda")
    token_count = frames * spatial_tokens
    qkv = torch.zeros(
        batch_size,
        token_count,
        3,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    beta = torch.full(
        (batch_size, num_heads, frames, spatial_tokens),
        0.5,
        device=device,
        dtype=torch.float32,
    )
    decay = torch.full(
        (batch_size, num_heads, frames),
        0.9,
        device=device,
        dtype=torch.float32,
    )

    class _WarmupNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.eps = 1e-6
            self.weight = nn.Parameter(torch.ones(num_heads * head_dim, device=device))

    triton_bidirectional_gated_delta_net_from_qkv(
        qkv,
        beta=beta,
        decay=decay,
        q_norm=_WarmupNorm(),
        k_norm=_WarmupNorm(),
        spatial_tokens=spatial_tokens,
        k_scale=(head_dim**-0.5) * (spatial_tokens**-0.5),
    )
    torch.cuda.synchronize(device)
    return True


__all__ = [
    "SANA_WM_GDN_ERROR",
    "SANA_WM_DISABLE_TRITON_GDN_ENV",
    "SANA_WM_REQUIRE_TRITON_GDN_ENV",
    "BidirectionalGatedDeltaNetTriton",
    "reference_bidirectional_gated_delta_net",
    "triton_bidirectional_gated_delta_net_from_qkv",
    "warmup_sana_wm_gdn_kernel",
]
