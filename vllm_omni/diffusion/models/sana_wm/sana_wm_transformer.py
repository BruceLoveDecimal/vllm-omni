# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SANA-WM Stage-1 transformer fallback.

This module is intentionally split from the official NVlabs backend.  It gives
vLLM-Omni a materializable, shape-compatible native path for scaffolding, weight
audits, and small smoke tests while the fused Bidirectional Gated DeltaNet
Triton operator and exact UCPE camera branch are being ported.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import dataclass, field, replace
from functools import reduce
from operator import mul
from typing import Any, ClassVar, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.gated_deltanet_triton import (
    SANA_WM_REQUIRE_TRITON_GDN_ENV,
    _flip_and_shift,
    reference_bidirectional_gated_delta_net,
    triton_bidirectional_gated_delta_net_from_qkv,
)
from vllm_omni.diffusion.models.sana_wm.ucpe import prepare_prope_fns
from vllm_omni.diffusion.models.sana_wm.weight_mapping import normalize_sana_wm_stage1_weight_name

try:
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    from vllm_omni.diffusion.attention.layer import Attention
    from vllm_omni.diffusion.distributed.sp_plan import (
        SequenceParallelInput,
        SequenceParallelOutput,
    )
except ImportError:  # pragma: no cover - local macOS/dev environments may not install vLLM.
    Attention = None
    ColumnParallelLinear = None
    QKVParallelLinear = None
    QuantizationConfig = Any  # type: ignore[misc, assignment]
    RowParallelLinear = None
    SequenceParallelInput = None
    SequenceParallelOutput = None
    VllmRMSNorm = None
    get_tensor_model_parallel_rank = None
    get_tensor_model_parallel_world_size = None

SANA_WM_TRANSFORMER_FORWARD_ERROR = (
    "Sana-WM full Stage-1 transformer forward is not implemented yet. "
    "The native fallback runs a pure PyTorch GDN smoke path; production quality "
    "still requires porting the fused Triton GDN kernel, exact camera injection, "
    "and the official denoising stack."
)

SANA_WM_STAGE1_LATENT_CHANNELS = 128
SANA_WM_STAGE1_PROMPT_CHANNELS = 2304
SANA_WM_STAGE1_TIMESTEP_CHANNELS = 256
SANA_WM_STAGE1_GDN_STATE_SIZE = 20
SANA_WM_DISABLE_VLLM_OPS_ENV = "VLLM_OMNI_SANA_WM_DISABLE_VLLM_OPS"


def _get_tp_rank() -> int:
    if get_tensor_model_parallel_rank is None:
        return 0
    try:
        return int(get_tensor_model_parallel_rank())
    except Exception:
        return 0


def _get_tp_world_size() -> int:
    if get_tensor_model_parallel_world_size is None:
        return 1
    try:
        return int(get_tensor_model_parallel_world_size())
    except Exception:
        return 1


def _vllm_tp_group_ready() -> bool:
    if get_tensor_model_parallel_world_size is None:
        return False
    try:
        get_tensor_model_parallel_world_size()
    except Exception:
        return False
    return True


def _sana_wm_disable_vllm_ops() -> bool:
    return os.environ.get(SANA_WM_DISABLE_VLLM_OPS_ENV, "").lower() in {"1", "true", "yes", "on"}


def _vllm_parallel_layers_available() -> bool:
    if _sana_wm_disable_vllm_ops():
        return False
    return (
        ColumnParallelLinear is not None
        and QKVParallelLinear is not None
        and RowParallelLinear is not None
        and _vllm_tp_group_ready()
    )


def _vllm_attention_available() -> bool:
    if _sana_wm_disable_vllm_ops():
        return False
    return Attention is not None


def _sequence_parallel_plan_available() -> bool:
    return SequenceParallelInput is not None and SequenceParallelOutput is not None


def _disable_tp_for_vllm_layer() -> bool:
    return not _vllm_tp_group_ready()


def _linear_output(output: torch.Tensor | tuple[torch.Tensor, Any]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _linear_weight_dtype(module: nn.Module) -> torch.dtype:
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight.dtype
    for param in module.parameters(recurse=False):
        return param.dtype
    return torch.float32


def _maybe_make_vllm_rms_norm(hidden_size: int, eps: float = 1e-6) -> nn.Module:
    if VllmRMSNorm is None or _sana_wm_disable_vllm_ops():
        return SanaWmRMSNorm(hidden_size, eps=eps)
    return VllmRMSNorm(hidden_size, eps=eps)


_CAM_TRITON_SENTINEL: object = object()
_cam_triton_fn: Any = _CAM_TRITON_SENTINEL


def _import_nvlabs_cam_scan_bidi() -> Any:
    """Lazy import of NVlabs ``cam_scan_bidi_chunkwise``. Cached after first call.

    Requires ``VLLM_OMNI_SANA_WM_OFFICIAL_REPO`` to point at the NVlabs source
    tree. Returns ``None`` on import failure (caller falls back to Python path).
    """
    global _cam_triton_fn
    if _cam_triton_fn is not _CAM_TRITON_SENTINEL:
        return _cam_triton_fn
    repo = os.environ.get("VLLM_OMNI_SANA_WM_OFFICIAL_REPO", "")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from diffusion.model.ops.fused_gdn_chunkwise import cam_scan_bidi_chunkwise  # type: ignore
        _cam_triton_fn = cam_scan_bidi_chunkwise
        print(f"[sana-wm cam-triton] imported from {repo}", flush=True)
    except Exception as exc:
        print(f"[sana-wm cam-triton] import failed ({exc}); falling back to Python path", flush=True)
        _cam_triton_fn = None
    return _cam_triton_fn


def _is_sana_wm_transformer_block(name: str, module: Any) -> bool:
    del module
    parts = name.split(".")
    return len(parts) == 2 and parts[0] == "blocks" and parts[1].isdigit()


def _prod(values: tuple[int, ...]) -> int:
    return reduce(mul, values, 1)


def _build_sana_wm_sp_plan() -> dict[str, Any] | None:
    """Declare Sana-WM Stage-1 sequence sharding boundaries.

    The first transformer block is the earliest point where video latents have
    already been patchified to ``[B, seq, hidden]``. The final layer is the last
    sequence-shaped output before unpatchify, so it is the natural gather point.
    RoPE is kept unsharded for now because the native GDN path consumes temporal
    layout information and needs a separate multi-card correctness pass.
    """

    if not _sequence_parallel_plan_available():
        return None
    assert SequenceParallelInput is not None
    assert SequenceParallelOutput is not None
    return {
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=True),
            "camera_hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=True),
        },
        "final_layer": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }


def _to_3tuple(value: int | tuple[int, int] | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) == 2:
        return (1, int(value[0]), int(value[1]))
    return (int(value[0]), int(value[1]), int(value[2]))


@dataclass(frozen=True)
class SanaWmStage1LoadReport:
    total_weights: int = 0
    loaded_weights: int = 0
    unmapped_weights: tuple[str, ...] = ()
    duplicate_weights: tuple[str, ...] = ()
    loaded_names: tuple[str, ...] = field(default_factory=tuple)
    materialized_weights: int = 0
    unapplied_weights: tuple[str, ...] = ()


class SanaWmRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps).to(hidden_states.dtype)
        return hidden_states * self.weight


class SanaWmTextProjection(nn.Module):
    def __init__(
        self,
        prompt_channels: int,
        hidden_size: int,
        *,
        quant_config: QuantizationConfig | None = None,
        use_vllm_parallel_layers: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.uses_vllm_parallel_layers = use_vllm_parallel_layers and _vllm_parallel_layers_available()
        if self.uses_vllm_parallel_layers:
            assert ColumnParallelLinear is not None
            assert RowParallelLinear is not None
            self.fc1 = ColumnParallelLinear(
                prompt_channels,
                hidden_size,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1" if prefix else "fc1",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.fc2 = RowParallelLinear(
                hidden_size,
                hidden_size,
                bias=True,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fc2" if prefix else "fc2",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
        else:
            self.fc1 = nn.Linear(prompt_channels, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, prompt_embeds: torch.Tensor) -> torch.Tensor:
        return _linear_output(self.fc2(self.act(_linear_output(self.fc1(prompt_embeds)))))


class SanaWmTextEmbedder(nn.Module):
    def __init__(
        self,
        prompt_channels: int,
        hidden_size: int,
        max_length: int,
        *,
        quant_config: QuantizationConfig | None = None,
        use_vllm_parallel_layers: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.y_embedding = nn.Parameter(torch.zeros(max_length, prompt_channels))
        self.y_proj = SanaWmTextProjection(
            prompt_channels,
            hidden_size,
            quant_config=quant_config,
            use_vllm_parallel_layers=use_vllm_parallel_layers,
            prefix=f"{prefix}.y_proj" if prefix else "y_proj",
        )

    def forward(
        self,
        prompt_embeds: torch.Tensor | None,
        *,
        batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if prompt_embeds is None:
            prompt_embeds = self.y_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        return self.y_proj(prompt_embeds.to(dtype=dtype))


class SanaWmTimestepEmbedder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        *,
        quant_config: QuantizationConfig | None = None,
        use_vllm_parallel_layers: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.uses_vllm_parallel_layers = use_vllm_parallel_layers and _vllm_parallel_layers_available()
        if self.uses_vllm_parallel_layers:
            assert ColumnParallelLinear is not None
            assert RowParallelLinear is not None
            linear_1 = ColumnParallelLinear(
                in_features,
                hidden_size,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.0" if prefix else "mlp.0",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            linear_2 = RowParallelLinear(
                hidden_size,
                hidden_size,
                bias=True,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.2" if prefix else "mlp.2",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
        else:
            linear_1 = nn.Linear(in_features, hidden_size)
            linear_2 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        self.mlp = nn.ModuleList([linear_1, self.act, linear_2])

    @staticmethod
    def sinusoidal_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=timestep.device, dtype=torch.float32) / max(half, 1)
        )
        args = timestep.float().reshape(-1, 1) * freqs.reshape(1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[-1]))
        return emb

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.sinusoidal_embedding(timestep, self.in_features)
        hidden_states = _linear_output(self.mlp[0](emb.to(dtype=_linear_weight_dtype(self.mlp[0]))))
        hidden_states = self.act(hidden_states)
        return _linear_output(self.mlp[2](hidden_states))


class SanaWmPatchEmbedMS3D(nn.Module):
    """Official-style 3D patch embedder used by SANA-WM Stage-1."""

    def __init__(
        self,
        patch_size: tuple[int, int, int],
        in_channels: int,
        hidden_size: int,
        *,
        kernel_size: tuple[int, int, int] | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = _to_3tuple(patch_size)
        self.kernel_size = _to_3tuple(kernel_size or patch_size)
        self.flatten = True
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=self.kernel_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = nn.Identity()

    def project_with_shape(self, latents: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        hidden_states = self.norm(self.proj(latents))
        _, _, frames, height, width = hidden_states.shape
        return hidden_states.flatten(2).transpose(1, 2), (frames, height, width)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.project_with_shape(latents)[0]


class SanaWmWanRotaryPosEmbed(nn.Module):
    """Wan-style 3D RoPE table used by the official SANA-WM GDN blocks."""

    def __init__(
        self,
        attention_head_dim: int,
        *,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        self._split_sizes = (
            t_dim // 2,
            h_dim // 2,
            w_dim // 2,
        )
        self.freqs = self._build_freqs(max_seq_len)

    def _build_1d_freq(self, dim: int, positions: torch.Tensor) -> torch.Tensor:
        if dim <= 0:
            return torch.empty(positions.shape[0], 0, dtype=torch.complex128)
        freqs = 1.0 / (
            self.theta
            ** (
                torch.arange(0, dim, 2, dtype=torch.float64, device=positions.device)[: dim // 2]
                / dim
            )
        )
        phase = torch.outer(positions.to(torch.float64), freqs)
        return torch.polar(torch.ones_like(phase), phase)

    def _build_freqs(self, max_seq_len: int) -> torch.Tensor:
        positions = torch.arange(max_seq_len, dtype=torch.float64)
        dims = (
            self._split_sizes[0] * 2,
            self._split_sizes[1] * 2,
            self._split_sizes[2] * 2,
        )
        return torch.cat([self._build_1d_freq(dim, positions) for dim in dims], dim=1)

    def forward(self, spatial_shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
        frames, height, width = spatial_shape
        if max(spatial_shape) > self.freqs.shape[0]:
            self.freqs = self._build_freqs(max(spatial_shape)).to(self.freqs.device)
        freqs = self.freqs.to(device=device)
        freqs_f, freqs_h, freqs_w = freqs.split(self._split_sizes, dim=1)
        f_dim, h_dim, w_dim = self._split_sizes
        parts = [
            freqs_f[:frames].view(frames, 1, 1, f_dim).expand(frames, height, width, f_dim),
            freqs_h[:height].view(1, height, 1, h_dim).expand(frames, height, width, h_dim),
            freqs_w[:width].view(1, 1, width, w_dim).expand(frames, height, width, w_dim),
        ]
        return torch.cat(parts, dim=-1).reshape(1, 1, frames * height * width, -1)


class SanaWmCameraEmbedder(nn.Module):
    """Small camera branch used by the native smoke path."""

    def __init__(
        self,
        config: SanaWmConfig | None = None,
        *,
        use_vllm_parallel_layers: bool = True,
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = config or SanaWmConfig()
        self.hidden_size = config.hidden_size
        self.plucker = nn.Module()
        # Conv3d has no vLLM parallel equivalent — kept as nn.Conv3d.
        self.plucker.proj = nn.Conv3d(config.chunk_plucker_channels, config.hidden_size, kernel_size=1)
        self.raymap = nn.Module()
        # raymap.proj is a plain Linear over 20 Plücker features; use
        # ColumnParallelLinear when vLLM TP layers are available.
        if use_vllm_parallel_layers and _vllm_parallel_layers_available() and ColumnParallelLinear is not None:
            self.raymap.proj: nn.Module = ColumnParallelLinear(
                20,
                config.hidden_size,
                bias=True,
                gather_output=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.raymap.proj" if prefix else "raymap.proj",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
        else:
            self.raymap.proj = nn.Linear(20, config.hidden_size)

    @staticmethod
    def _match_tokens(hidden_states: torch.Tensor, expected_tokens: int) -> torch.Tensor:
        if hidden_states.shape[1] == expected_tokens:
            return hidden_states
        if hidden_states.shape[1] > expected_tokens:
            return hidden_states[:, :expected_tokens]
        pad = hidden_states[:, -1:].expand(-1, expected_tokens - hidden_states.shape[1], -1)
        return torch.cat([hidden_states, pad], dim=1)

    def forward(
        self,
        *,
        plucker: torch.Tensor | None = None,
        raymap: torch.Tensor | None = None,
        spatial_shape: tuple[int, int, int],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        frames, height, width = spatial_shape
        expected_tokens = frames * height * width
        hidden_states = None

        if plucker is not None:
            if plucker.ndim == 4:
                plucker = plucker.unsqueeze(0)
            plucker = plucker.to(device=device, dtype=dtype)
            if plucker.shape[0] == 1 and batch_size > 1:
                plucker = plucker.expand(batch_size, -1, -1, -1, -1)
            plucker_hidden = self.plucker.proj(plucker).flatten(2).transpose(1, 2)
            hidden_states = self._match_tokens(plucker_hidden, expected_tokens)

        if raymap is not None:
            if raymap.ndim == 2:
                raymap = raymap.unsqueeze(0)
            raymap = raymap.to(device=device, dtype=dtype)
            if raymap.shape[0] == 1 and batch_size > 1:
                raymap = raymap.expand(batch_size, -1, -1)
            ray_hidden = self.raymap.proj(raymap)
            ray_hidden = ray_hidden.repeat_interleave(height * width, dim=1)
            ray_hidden = self._match_tokens(ray_hidden, expected_tokens)
            hidden_states = ray_hidden if hidden_states is None else hidden_states + ray_hidden

        return hidden_states


class SanaWmSelfAttention(nn.Module):
    def __init__(
        self,
        config: SanaWmConfig,
        *,
        use_gdn: bool = True,
        quant_config: QuantizationConfig | None = None,
        use_vllm_parallel_layers: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.total_num_heads = max(hidden_size // max(config.linear_head_dim, 1), 1)
        self.head_dim = hidden_size // self.total_num_heads
        self.num_heads = self.total_num_heads
        self.num_kv_heads = self.total_num_heads
        self.heads = self.total_num_heads
        self.dim = self.head_dim
        self.in_dim = hidden_size
        self.out_dim = hidden_size
        self.eps = 1e-8
        self.attn_type = config.attn_type
        self.use_gdn = use_gdn and "GDN" in config.attn_type
        self.use_vllm_attention = (not self.use_gdn) and _vllm_attention_available()
        self.k_conv_only = config.k_conv_only
        self.conv_kernel_size = config.conv_kernel_size
        self.patch_size = _to_3tuple(config.patch_size)
        self.uses_vllm_parallel_layers = use_vllm_parallel_layers and _vllm_parallel_layers_available()
        # cam_dim must be known before the TP/fallback branch uses it for layer sizes.
        self.cam_dim = hidden_size // max(config.cam_attn_compress, 1)
        if self.uses_vllm_parallel_layers:
            assert ColumnParallelLinear is not None
            assert QKVParallelLinear is not None
            assert RowParallelLinear is not None
            self.qkv = QKVParallelLinear(
                hidden_size=hidden_size,
                head_size=self.head_dim,
                total_num_heads=self.total_num_heads,
                bias=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv" if prefix else "qkv",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.num_heads = self.qkv.num_heads
            self.num_kv_heads = self.qkv.num_kv_heads
            self.proj = RowParallelLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=True,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.proj" if prefix else "proj",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.beta_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.beta_proj" if prefix else "beta_proj",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_proj" if prefix else "gate_proj",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.output_gate = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads * self.head_dim,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.output_gate" if prefix else "output_gate",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.q_proj_cam = ColumnParallelLinear(
                hidden_size,
                self.cam_dim,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj_cam" if prefix else "q_proj_cam",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.k_proj_cam = ColumnParallelLinear(
                hidden_size,
                self.cam_dim,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.k_proj_cam" if prefix else "k_proj_cam",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.v_proj_cam = ColumnParallelLinear(
                hidden_size,
                self.cam_dim,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.v_proj_cam" if prefix else "v_proj_cam",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.out_proj_cam = RowParallelLinear(
                self.cam_dim,
                hidden_size,
                bias=True,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.out_proj_cam" if prefix else "out_proj_cam",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
        else:
            self.qkv = nn.Linear(hidden_size, self.num_heads * self.head_dim * 3, bias=False)
            self.proj = nn.Linear(self.num_heads * self.head_dim, hidden_size)
            self.beta_proj = nn.Linear(hidden_size, self.num_heads)
            self.gate_proj = nn.Linear(hidden_size, self.num_heads)
            self.output_gate = nn.Linear(hidden_size, hidden_size)
            self.q_proj_cam = nn.Linear(hidden_size, self.cam_dim, bias=True)
            self.k_proj_cam = nn.Linear(hidden_size, self.cam_dim, bias=True)
            self.v_proj_cam = nn.Linear(hidden_size, self.cam_dim, bias=True)
            self.out_proj_cam = nn.Linear(self.cam_dim, hidden_size, bias=True)
        local_inner_dim = self.num_heads * self.head_dim
        norm_cls = _maybe_make_vllm_rms_norm if config.qk_norm else (lambda *_args, **_kwargs: nn.Identity())
        self.q_norm = norm_cls(local_inner_dim)
        self.k_norm = norm_cls(local_inner_dim)
        self.softmax_attn = (
            Attention(
                num_heads=self.num_heads,
                head_size=self.head_dim,
                num_kv_heads=self.num_kv_heads,
                softmax_scale=1.0 / (self.head_dim**0.5),
                causal=False,
                role="self",
                qkv_layout="BSND",
                prefix=prefix,
            )
            if self.use_vllm_attention
            else None
        )

        # These names mirror the official GDN checkpoint. The current forward
        # executes a pure PyTorch recurrence; the fused Triton scan is ported
        # separately.
        self.A_log = nn.Parameter(torch.zeros(self.num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.register_buffer("recall_gate", torch.zeros(1))
        self.conv_k = (
            nn.Conv1d(local_inner_dim, local_inner_dim, kernel_size=config.conv_kernel_size, groups=local_inner_dim)
            if config.conv_kernel_size > 0
            else None
        )
        self.conv_q = None
        self.conv_v = None

        cam_compress = max(config.cam_attn_compress, 1)
        self.cam_heads = max(self.num_heads // cam_compress, 1)
        self.cam_head_dim = self.cam_dim // self.cam_heads
        # q_proj_cam / k_proj_cam / v_proj_cam / out_proj_cam are initialised
        # in the TP/fallback conditional above so they follow the same
        # ColumnParallelLinear / RowParallelLinear pattern as qkv and proj.
        self.q_norm_cam = norm_cls(self.cam_dim)
        self.k_norm_cam = norm_cls(self.cam_dim)
        self.conv_k_cam = (
            nn.Conv1d(self.cam_dim, self.cam_dim, kernel_size=config.conv_kernel_size, groups=self.cam_dim)
            if config.conv_kernel_size > 0
            else None
        )
        self.conv_q_cam = None
        self.conv_v_cam = None
        self._init_short_convs()

    @staticmethod
    def _init_short_conv(conv: nn.Conv1d | None) -> None:
        if conv is None:
            return
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[:, 0, -1] = 1.0
            if conv.bias is not None:
                conv.bias.zero_()

    def _init_short_convs(self) -> None:
        for conv in (self.conv_q, self.conv_k, self.conv_v, self.conv_q_cam, self.conv_k_cam, self.conv_v_cam):
            self._init_short_conv(conv)

    @staticmethod
    def _flip_and_shift(tensor: torch.Tensor, *, dim: int, shift_value: float) -> torch.Tensor:
        flipped = torch.flip(tensor, dims=[dim])
        shifted = flipped.narrow(dim, 0, tensor.shape[dim] - 1)
        pad_shape = list(tensor.shape)
        pad_shape[dim] = 1
        padding = torch.full(pad_shape, shift_value, device=tensor.device, dtype=tensor.dtype)
        return torch.cat([padding, shifted], dim=dim)

    @staticmethod
    def _apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        rotated = torch.view_as_complex(
            hidden_states.permute(0, 1, 3, 2).to(torch.float64).unflatten(3, (-1, 2))
        )
        output = torch.view_as_real(rotated * freqs).flatten(3, 4).permute(0, 1, 3, 2)
        return output.type_as(hidden_states)

    @classmethod
    def _apply_rotary_emb_to_sdpa(cls, hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        return cls._apply_rotary_emb(hidden_states.transpose(2, 3), freqs).transpose(2, 3)

    @staticmethod
    def _reshape_to_temporal(
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, int, int, int]:
        batch_size, token_count, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(f"Sana-WM temporal conv expects N=T*H*W, got N={token_count}, THW={spatial_shape}.")
        hidden_states = hidden_states.reshape(batch_size, frames, spatial_tokens, hidden_size)
        hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(batch_size * spatial_tokens, frames, hidden_size)
        return hidden_states, batch_size, spatial_tokens, frames

    @staticmethod
    def _reshape_from_temporal(
        hidden_states: torch.Tensor,
        batch_size: int,
        spatial_tokens: int,
        frames: int,
    ) -> torch.Tensor:
        hidden_size = hidden_states.shape[-1]
        return (
            hidden_states.reshape(batch_size, spatial_tokens, frames, hidden_size)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, frames * spatial_tokens, hidden_size)
        )

    @staticmethod
    def _causal_conv_1d(hidden_states: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        dtype = hidden_states.dtype
        conv_input = hidden_states.transpose(1, 2).to(conv.weight.dtype)
        conv_input = F.pad(conv_input, (conv.kernel_size[0] - 1, 0))
        output = conv(conv_input).transpose(1, 2)
        return output.to(dtype)

    def _bidirectional_temporal_short_conv(
        self,
        hidden_states: torch.Tensor,
        conv: nn.Conv1d,
        spatial_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        hidden_states, batch_size, spatial_tokens, frames = self._reshape_to_temporal(hidden_states, spatial_shape)
        forward_states = self._causal_conv_1d(hidden_states, conv)
        backward_states = self._causal_conv_1d(hidden_states.flip(1), conv).flip(1)
        center_weight = conv.weight[:, 0, -1].to(hidden_states.dtype)
        center_states = hidden_states * center_weight.view(1, 1, -1)
        output = forward_states + backward_states - center_states
        return self._reshape_from_temporal(output, batch_size, spatial_tokens, frames)

    def _compute_frame_gates(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(
                f"Sana-WM GDN token layout mismatch: N={token_count}, expected {frames * spatial_tokens}."
            )
        beta = torch.sigmoid(_linear_output(self.beta_proj(hidden_states)))
        beta = beta.reshape(batch_size, frames, spatial_tokens, self.num_heads).permute(0, 3, 1, 2)
        frame_states = hidden_states.reshape(batch_size, frames, spatial_tokens, hidden_size).mean(dim=2)
        gate = _linear_output(self.gate_proj(frame_states)).float()
        decay = torch.exp(
            -self.A_log.float().exp().view(1, 1, -1)
            * F.softplus(gate + self.dt_bias.float().view(1, 1, -1))
        )
        return beta, decay.transpose(1, 2)

    def _gdn_update_components(
        self,
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
        state_kv = torch.zeros(
            batch_size,
            num_heads,
            head_dim,
            head_dim,
            device=query.device,
            dtype=query.dtype,
        )
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

    def _forward_gdn_raw(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
        *,
        precomputed_gates: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run main-branch bidirectional GDN and return the (B, N, q_size)
        raw output BEFORE ``output_gate`` and ``proj`` are applied.

        Returns ``(raw_output, (beta, decay))`` so the caller can share the
        beta/decay gates with the camera branch (matches NVlabs' shared
        ``precomputed_gates`` plumbing).
        """
        batch_size, token_count, _hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(f"Sana-WM GDN expects N=T*H*W, got N={token_count}, THW={spatial_shape}.")

        qkv = _linear_output(self.qkv(hidden_states))
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)
        if self.conv_k is not None:
            key = self._bidirectional_temporal_short_conv(key, self.conv_k, spatial_shape)
        if precomputed_gates is None:
            beta, decay = self._compute_frame_gates(hidden_states, spatial_shape)
        else:
            beta, decay = precomputed_gates

        if hidden_states.is_cuda:
            qkv = torch.stack(
                (
                    query.reshape(batch_size, token_count, self.num_heads, self.head_dim),
                    key.reshape(batch_size, token_count, self.num_heads, self.head_dim),
                    value.reshape(batch_size, token_count, self.num_heads, self.head_dim),
                ),
                dim=2,
            )
            try:
                output = triton_bidirectional_gated_delta_net_from_qkv(
                    qkv,
                    beta=beta,
                    decay=decay,
                    q_norm=self.q_norm,
                    k_norm=self.k_norm,
                    spatial_tokens=spatial_tokens,
                    rotary_emb=rotary_emb,
                    k_scale=(self.head_dim**-0.5) * (spatial_tokens**-0.5),
                    eps=self.eps,
                )
                output = output.permute(0, 3, 1, 2).reshape(batch_size, token_count, q_size)
                return output, (beta, decay)
            except Exception:
                if os.environ.get(SANA_WM_REQUIRE_TRITON_GDN_ENV, "").lower() in {"1", "true", "yes", "on"}:
                    raise
                # The fused path is an optimization. Keep the reference path as
                # the correctness fallback for unsupported GPUs / Triton builds.
                pass
        query = self.q_norm(query).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        key = self.k_norm(key).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, token_count, self.num_heads, self.head_dim)

        query = F.relu(query).permute(0, 2, 3, 1)
        key = F.relu(key).permute(0, 2, 3, 1)
        value = value.permute(0, 2, 3, 1)
        key = key * ((self.head_dim**-0.5) * (spatial_tokens**-0.5))
        if rotary_emb is not None:
            query_rot = self._apply_rotary_emb(query, rotary_emb)
            key_rot = self._apply_rotary_emb(key, rotary_emb)
        else:
            query_rot = query
            key_rot = key

        output = reference_bidirectional_gated_delta_net(
            query,
            key,
            value,
            beta=beta,
            decay=decay,
            spatial_tokens=spatial_tokens,
            query_rot=query_rot,
            key_rot=key_rot,
            eps=self.eps,
        )
        output = output.permute(0, 3, 1, 2).reshape(batch_size, token_count, q_size)
        return output, (beta, decay)

    def _apply_output_gate_and_proj(
        self,
        combined: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SiLU output gate (driven by hidden_states) + final projection.

        Centralised so the same shared gate/proj path is applied whether or
        not the camera branch contributes to ``combined``.
        """
        gate = F.silu(_linear_output(self.output_gate(hidden_states)).float())
        gated = combined * gate
        return _linear_output(self.proj(gated.to(_linear_weight_dtype(self.proj))))

    def _forward_gdn(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        """Main-branch only GDN forward with output_gate + proj applied."""
        raw, _ = self._forward_gdn_raw(hidden_states, spatial_shape, rotary_emb)
        return self._apply_output_gate_and_proj(raw, hidden_states)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = tensor.shape
        tensor = tensor.reshape(batch, seq_len, self.num_heads, hidden_size // self.num_heads)
        return tensor.transpose(1, 2)

    def _reshape_to_seq_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = tensor.shape
        return tensor.reshape(batch, seq_len, self.num_heads, hidden_size // self.num_heads)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = tensor.shape
        return tensor.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)

    @staticmethod
    def _downscale_to_reference_rms(
        ref: torch.Tensor,
        transformed: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Downscale ``transformed`` so its per-token channel RMS does not
        exceed ``ref``'s. Mirrors NVlabs ``_downscale_to_reference_rms``.

        Inputs are ``(B, H, N, D)`` (channel dim last). The RMS-scale is
        clamped at 1.0 — the function only ever shrinks, never amplifies.
        """
        ref_rms = ref.square().mean(dim=-1, keepdim=True).add(eps).sqrt()
        tr_rms = transformed.square().mean(dim=-1, keepdim=True).add(eps).sqrt()
        scale = (ref_rms / tr_rms.clamp_min(eps)).clamp(max=1.0)
        return transformed * scale

    @staticmethod
    def _ucpe_rotary_freqs(rotary_emb: torch.Tensor | None) -> torch.Tensor | None:
        if rotary_emb is None:
            return None
        rotary_emb_freqs = rotary_emb.squeeze(0).squeeze(0)
        if rotary_emb_freqs.ndim == 3:
            # vLLM-Omni packs RoPE as (1, 1, N, D//2) complex. Some
            # call-sites may pass an already squeezed (N, D//2) tensor.
            rotary_emb_freqs = rotary_emb_freqs.squeeze(0)
        return rotary_emb_freqs

    def _forward_softmax_raw(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        """Softmax self-attention raw output before output_gate/proj.

        Mirrors NVlabs ``_forward_softmax_attn(..., apply_output_gate=False)``
        for every-N-th hybrid blocks. GDN-specific gates are intentionally not
        used here.
        """
        batch_size, token_count, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        if token_count != frames * height * width:
            raise ValueError(
                f"Sana-WM softmax attention expects N=T*H*W, got N={token_count}, THW={spatial_shape}."
            )

        qkv = _linear_output(self.qkv(hidden_states))
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = self.q_norm(query).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        key = self.k_norm(key).reshape(batch_size, token_count, self.num_kv_heads, self.head_dim)
        value = value.reshape(batch_size, token_count, self.num_kv_heads, self.head_dim)

        if rotary_emb is not None:
            query = self._apply_rotary_emb(query.permute(0, 2, 3, 1), rotary_emb).permute(0, 3, 1, 2)
            key = self._apply_rotary_emb(key.permute(0, 2, 3, 1), rotary_emb).permute(0, 3, 1, 2)

        query = query.transpose(1, 2)  # (B, H, N, D)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        dtype_orig = hidden_states.dtype
        if query.dtype == torch.float32:
            query = query.bfloat16()
            key = key.bfloat16()
            value = value.bfloat16()
        attn = F.scaled_dot_product_attention(query, key, value)
        return attn.transpose(1, 2).reshape(batch_size, token_count, q_size).to(dtype_orig)

    def _forward_softmax_cam_branch(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        """UCPE camera branch for softmax hybrid blocks.

        Matches NVlabs ``_forward_cam_branch_softmax``: camera Q/K/V are
        projected from the same hidden stream, transformed by UCPE, run through
        SDPA, inverse-transformed, then returned as raw ``cam_dim`` features.
        """
        batch_size, token_count, _ = hidden_states.shape
        frames, height, width = spatial_shape
        if token_count != frames * height * width:
            raise ValueError(
                f"Sana-WM softmax cam branch expects N=T*H*W, got N={token_count}, THW={spatial_shape}."
            )

        q_cam = _linear_output(self.q_proj_cam(hidden_states))
        k_cam = _linear_output(self.k_proj_cam(hidden_states))
        v_cam = _linear_output(self.v_proj_cam(hidden_states))

        if self.conv_q_cam is not None:
            q_cam = self._bidirectional_temporal_short_conv(q_cam, self.conv_q_cam, spatial_shape)
        if self.conv_k_cam is not None:
            k_cam = self._bidirectional_temporal_short_conv(k_cam, self.conv_k_cam, spatial_shape)
        if self.conv_v_cam is not None:
            v_cam = self._bidirectional_temporal_short_conv(v_cam, self.conv_v_cam, spatial_shape)

        q_cam = self.q_norm_cam(q_cam).reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim)
        k_cam = self.k_norm_cam(k_cam).reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim)
        v_cam = v_cam.reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim)

        q_cam = q_cam.transpose(1, 2).contiguous()  # (B, H_cam, N, D)
        k_cam = k_cam.transpose(1, 2).contiguous()
        v_cam = v_cam.transpose(1, 2).contiguous()

        apply_fn_q, apply_fn_kv, apply_fn_o = prepare_prope_fns(
            head_dim=self.cam_head_dim,
            camera_conditions=camera_conditions,
            HW=spatial_shape,
            patch_size=self.patch_size,
            rotary_emb=self._ucpe_rotary_freqs(rotary_emb),
        )
        q_cam_trans = apply_fn_q(q_cam)
        kv_cam_trans = apply_fn_kv(torch.cat([k_cam, v_cam], dim=1))
        k_cam_trans, v_cam_trans = torch.chunk(kv_cam_trans, chunks=2, dim=1)

        q_cam_trans = self._downscale_to_reference_rms(q_cam, q_cam_trans)
        k_cam_trans = self._downscale_to_reference_rms(k_cam, k_cam_trans)
        v_cam_trans = self._downscale_to_reference_rms(v_cam, v_cam_trans)

        dtype_orig = hidden_states.dtype
        query = q_cam_trans
        key = k_cam_trans
        value = v_cam_trans
        if getattr(self, "fp32_attention", True):
            query = query.float()
            key = key.float()
            value = value.float()
        if query.dtype == torch.float32:
            query = query.bfloat16()
            key = key.bfloat16()
            value = value.bfloat16()

        head_dim = query.shape[-1]
        needs_pad = head_dim not in (32, 64, 128, 256) and head_dim < 256
        if needs_pad:
            pad_to = 128 if head_dim <= 128 else 256
            pad_size = pad_to - head_dim
            query = F.pad(query, (0, pad_size))
            key = F.pad(key, (0, pad_size))
            value = F.pad(value, (0, pad_size))

        out = F.scaled_dot_product_attention(query, key, value)
        if needs_pad:
            out = out[..., :head_dim]
        if out.dtype != dtype_orig:
            out = out.to(dtype_orig)
        out = apply_fn_o(out)
        return out.transpose(1, 2).reshape(batch_size, token_count, self.cam_dim)

    @staticmethod
    def _cam_single_path_delta_scan(
        q_rot: torch.Tensor,  # (B, H, D, N)
        k_rot: torch.Tensor,  # (B, H, D, N)
        value: torch.Tensor,  # (B, H, D, N)
        beta: torch.Tensor,  # (B, H, T, S)
        decay: torch.Tensor,  # (B, H, T)
        *,
        spatial_tokens: int,
    ) -> torch.Tensor:
        """Numerator-only delta-rule recurrence used by the SANA-WM cam branch.

        Matches NVlabs ``torch_recurrent_cam_single_path_delta_rule``: no Z
        denominator stream, no final divide. Output is ``state_kv @ q_rot``
        per frame, stacked back into ``(B, H, D, N)`` layout.

        Runs in fp32 internally for numerical stability (matches NVlabs
        ``fp32_attention=True``) and casts the result back to the input dtype.
        """
        dtype_orig = q_rot.dtype
        q_rot = q_rot.float()
        k_rot = k_rot.float()
        value = value.float()
        beta = beta.float()
        decay = decay.float()

        batch_size, num_heads, head_dim, token_count = q_rot.shape
        frames = beta.shape[2]
        if token_count != frames * spatial_tokens:
            raise ValueError(
                f"single-path scan token_count={token_count} != frames*S={frames * spatial_tokens}."
            )

        def to_frames(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch_size, num_heads, head_dim, frames, spatial_tokens).permute(0, 1, 3, 2, 4)

        q_rot_f = to_frames(q_rot)
        k_rot_f = to_frames(k_rot)
        value_f = to_frames(value)
        state_kv = torch.zeros(
            batch_size, num_heads, head_dim, head_dim, device=q_rot.device, dtype=torch.float32
        )
        outputs: list[torch.Tensor] = []
        for frame_idx in range(frames):
            qrt = q_rot_f[:, :, frame_idx]
            krt = k_rot_f[:, :, frame_idx]
            vt = value_f[:, :, frame_idx]
            bt = beta[:, :, frame_idx].unsqueeze(2)
            gt = decay[:, :, frame_idx].view(batch_size, num_heads, 1, 1)

            state_kv = state_kv * gt
            v_pred = torch.matmul(state_kv, krt)
            delta_v = (vt - v_pred) * bt
            state_kv = state_kv + torch.matmul(delta_v, krt.transpose(-1, -2))
            outputs.append(torch.matmul(state_kv, qrt))

        stacked = torch.stack(outputs, dim=2)
        out = stacked.permute(0, 1, 3, 2, 4).reshape(batch_size, num_heads, head_dim, token_count)
        return out.to(dtype_orig)

    def _bidi_single_path(
        self,
        q_rot: torch.Tensor,
        k_rot: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
        *,
        spatial_tokens: int,
    ) -> torch.Tensor:
        """Bidirectional single-path scan: forward + backward, simple sum.

        Backward direction shifts K/V/beta by one frame (with zero pad) and
        the decay by one frame (with neutral 1.0 pad), matching NVlabs
        ``flip_and_shift`` conventions.

        Set ``SANA_WM_CAM_TRITON=1`` to dispatch to NVlabs
        ``cam_scan_bidi_chunkwise`` from
        ``diffusion.model.ops.fused_gdn_chunkwise``; falls back to the
        Python reference path on import failure.
        """
        if os.environ.get("SANA_WM_CAM_TRITON", "").lower() in {"1", "true", "yes", "on"}:
            triton_fn = _import_nvlabs_cam_scan_bidi()
            if triton_fn is not None:
                dtype_orig = q_rot.dtype
                B, H, D, N = q_rot.shape
                F = beta.shape[2]
                if N != F * spatial_tokens:
                    raise ValueError(
                        f"cam triton path: N={N} != F*S={F * spatial_tokens}"
                    )
                # NVlabs requires fp32 contiguous q/k/v, beta (B,H,F,S), decay (B,H,F).
                q_f = q_rot.float().contiguous()
                k_f = k_rot.float().contiguous()
                v_f = value.float().contiguous()
                if beta.ndim == 3:
                    beta_f = beta.unsqueeze(-1).expand(B, H, F, spatial_tokens).contiguous()
                else:
                    beta_f = beta.float().contiguous()
                decay_f = decay.float().contiguous()
                out = triton_fn(q_f, k_f, v_f, beta_f, decay_f)
                return out.to(dtype_orig)
        batch_size, num_heads, head_dim, token_count = q_rot.shape
        frames = beta.shape[2]

        out_fwd = self._cam_single_path_delta_scan(
            q_rot, k_rot, value, beta, decay, spatial_tokens=spatial_tokens
        )

        def to_time(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch_size, num_heads, head_dim, frames, spatial_tokens).permute(0, 1, 3, 2, 4)

        def from_time(t: torch.Tensor) -> torch.Tensor:
            return t.permute(0, 1, 3, 2, 4).reshape(batch_size, num_heads, head_dim, token_count)

        q_T = to_time(q_rot)
        k_T = to_time(k_rot)
        v_T = to_time(value)
        q_bwd = torch.flip(q_T, dims=[2])
        k_bwd = _flip_and_shift(k_T, dim=2, shift_value=0.0)
        v_bwd = _flip_and_shift(v_T, dim=2, shift_value=0.0)
        beta_bwd = _flip_and_shift(beta, dim=2, shift_value=0.0)
        decay_bwd = _flip_and_shift(decay, dim=2, shift_value=1.0)

        out_bwd_flipped = self._cam_single_path_delta_scan(
            from_time(q_bwd),
            from_time(k_bwd),
            from_time(v_bwd),
            beta_bwd,
            decay_bwd,
            spatial_tokens=spatial_tokens,
        )
        out_bwd = torch.flip(
            out_bwd_flipped.view(batch_size, num_heads, head_dim, frames, spatial_tokens),
            dims=[3],
        ).reshape(batch_size, num_heads, head_dim, token_count)
        return out_fwd + out_bwd

    def _forward_cam_branch(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        *,
        precomputed_gates: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """UCPE camera branch — matches the NVlabs production
        ``BidirectionalGDNUCPESinglePathLiteLABothTriton`` variant declared
        in the SANA-WM 1600M release config.

        Pipeline (mirrors the released NVlabs
        ``BidirectionalGDNUCPESinglePathLiteLABothTriton`` path):

        1. Project Q/K/V from ``hidden_states`` (all three from x — UCPE is
           NOT cross-attention; camera info enters via per-pixel projection
           matrices).
        2. Short conv on K (and optionally Q/V).
        3. ``q_norm_cam`` / ``k_norm_cam`` (RMSNorm over cam_dim).
        4. ReLU kernel + ``k_scale = D^-0.5 * S^-0.5``.
        5. UCPE per-token block-diagonal transforms via ``apply_fn_q`` and
           ``apply_fn_kv`` (per-ray 4x4 projection on first half + sliced
           complex RoPE on second half).
        6. Dynamic Beta Discounting: β_cam = β / frame_inflation_sq.clamp_min(1).
           The released NVlabs ``BothTriton`` cam path does not apply the
           Python ``PostUCPERenorm`` shrink step before the scan; it uses the
           raw UCPE-transformed Q/K/V and discounts beta from their raw K
           inflation.
        7. **Single-path** delta-rule recurrence (numerator only, no Z
           denominator) — forward + backward, simple sum.
        8. **Inverse output transform**: ``apply_fn_o`` brings the recurrence
           output from the per-ray frame back to the world frame.

        Returns ``(B, N, cam_dim)`` raw — caller applies ``out_proj_cam`` and
        the shared ``output_gate`` + ``proj``.
        """
        batch_size, token_count, _ = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(
                f"Sana-WM cam branch expects N=T*H*W, got N={token_count}, THW={spatial_shape}."
            )

        # Build the UCPE apply closures once per call. Future optimisation:
        # hoist this to the block level so the same closures are reused
        # across all 20 transformer blocks (NVlabs caches via prope_fns).
        rotary_emb_freqs = self._ucpe_rotary_freqs(rotary_emb)
        apply_fn_q, apply_fn_kv, apply_fn_o = prepare_prope_fns(
            head_dim=self.cam_head_dim,
            camera_conditions=camera_conditions,
            HW=spatial_shape,
            patch_size=self.patch_size,
            rotary_emb=rotary_emb_freqs,
        )

        # 1. Projections (all from x).
        q_cam = _linear_output(self.q_proj_cam(hidden_states))
        k_cam = _linear_output(self.k_proj_cam(hidden_states))
        v_cam = _linear_output(self.v_proj_cam(hidden_states))

        # 2. Short conv on K (and optionally Q, V).
        if self.conv_k_cam is not None:
            k_cam = self._bidirectional_temporal_short_conv(k_cam, self.conv_k_cam, spatial_shape)
        if self.conv_q_cam is not None:
            q_cam = self._bidirectional_temporal_short_conv(q_cam, self.conv_q_cam, spatial_shape)
        if self.conv_v_cam is not None:
            v_cam = self._bidirectional_temporal_short_conv(v_cam, self.conv_v_cam, spatial_shape)

        # 3. Q/K norm on flattened channel dim.
        q_cam = self.q_norm_cam(q_cam)
        k_cam = self.k_norm_cam(k_cam)

        # Reshape to (B, N, H_cam, D_cam) then to (B, H_cam, N, D_cam) for
        # UCPE apply (UCPE closures expect (B, num_heads, N, D)).
        q_cam = q_cam.reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim).transpose(1, 2)
        k_cam = k_cam.reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim).transpose(1, 2)
        v_cam = v_cam.reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim).transpose(1, 2)

        # 4. ReLU kernel + k_scale.
        q_cam = F.relu(q_cam)
        k_cam = F.relu(k_cam)
        k_scale = (self.cam_head_dim**-0.5) * (spatial_tokens**-0.5)
        k_cam = k_cam * k_scale

        # 5. UCPE per-token transforms.
        pre_ucpe_k_norm = torch.linalg.vector_norm(k_cam, dim=-1, keepdim=True).clamp_min(1e-6)
        q_cam_trans = apply_fn_q(q_cam)
        kv_cam = torch.cat([k_cam, v_cam], dim=1)
        kv_cam_trans = apply_fn_kv(kv_cam)
        k_cam_trans, v_cam_trans = torch.chunk(kv_cam_trans, chunks=2, dim=1)

        # 6. Match NVlabs' production BothTriton path: do not apply the
        # Python PostUCPERenorm shrink here. `cam_prep_func` feeds raw
        # UCPE-transformed Q/K/V into `cam_scan_bidi_chunkwise` and computes
        # beta discounting from the raw K inflation.
        post_ucpe_k_norm = torch.linalg.vector_norm(k_cam_trans, dim=-1, keepdim=True).clamp_min(1e-6)
        inflation_sq = (post_ucpe_k_norm / pre_ucpe_k_norm) ** 2  # (B, H, N, 1)

        # 7. Dynamic Beta Discounting (per-frame mean over spatial tokens).
        beta, decay = precomputed_gates
        inflation_per_token = inflation_sq.squeeze(-1).reshape(
            batch_size, self.cam_heads, frames, spatial_tokens
        )
        frame_inflation_sq = inflation_per_token.mean(dim=-1)  # (B, H_cam, T)
        if beta.shape[1] != self.cam_heads:
            repeat_factor = self.cam_heads // beta.shape[1]
            if repeat_factor * beta.shape[1] != self.cam_heads:
                raise ValueError(
                    "Sana-WM cam branch expects cam_heads to be a multiple of main heads,"
                    f" got cam_heads={self.cam_heads} main_heads={beta.shape[1]}."
                )
            beta_cam = beta.repeat_interleave(repeat_factor, dim=1)
            decay_cam = decay.repeat_interleave(repeat_factor, dim=1)
        else:
            beta_cam = beta
            decay_cam = decay
        beta_cam = beta_cam / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        # 8. Single-path bidirectional delta-rule recurrence on the UCPE-
        # transformed Q/K/V (no Z denominator, no final divide).
        q_rot_bhdn = q_cam_trans.permute(0, 1, 3, 2).contiguous()
        k_rot_bhdn = k_cam_trans.permute(0, 1, 3, 2).contiguous()
        v_bhdn = v_cam_trans.permute(0, 1, 3, 2).contiguous()
        cam_out_bhdn = self._bidi_single_path(
            q_rot_bhdn, k_rot_bhdn, v_bhdn, beta_cam, decay_cam, spatial_tokens=spatial_tokens
        )

        # 9. Apply inverse UCPE transform on the recurrence output.
        # ``apply_fn_o`` expects (B, H, N, D); we currently have
        # (B, H, D, N) — transpose, apply, transpose back.
        cam_out_bhnd = cam_out_bhdn.transpose(-1, -2).contiguous()
        cam_out_bhnd = apply_fn_o(cam_out_bhnd)
        cam_out_bhdn = cam_out_bhnd.transpose(-1, -2).contiguous()

        # (B, H_cam, D_cam, N) → (B, N, cam_dim)
        return cam_out_bhdn.permute(0, 3, 1, 2).reshape(batch_size, token_count, self.cam_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        camera_conditions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with dual-branch GDN+UCPE when ``camera_conditions`` is
        provided, otherwise main-branch-only GDN or softmax fallback.

        ``camera_conditions`` is the raw ``(B, T_latent, 20)`` raymap tensor
        produced by :func:`camera_control._pack_camera_conditions` — NOT the
        embedder output. The cam branch consumes ``hidden_states`` as the
        sole Q/K/V source and uses ``camera_conditions`` to build per-pixel
        4x4 transforms that are applied to the Q/K/V channels via UCPE.
        """
        if spatial_shape is not None and self.use_gdn:
            cam_disabled = os.environ.get("VLLM_OMNI_SANA_WM_DISABLE_CAM_BRANCH", "").lower() in {
                "1", "true", "yes", "on",
            }
            if camera_conditions is None or self.q_proj_cam is None or cam_disabled:
                # Main branch only — preserves legacy path when the request
                # has no camera info or the cam projections were not built.
                return self._forward_gdn(hidden_states, spatial_shape, rotary_emb)
            # Dual-branch: precompute β/decay once, run main raw, cam raw,
            # combine, then apply shared output_gate + proj exactly once.
            beta, decay = self._compute_frame_gates(hidden_states, spatial_shape)
            main_raw, _ = self._forward_gdn_raw(
                hidden_states,
                spatial_shape,
                rotary_emb,
                precomputed_gates=(beta, decay),
            )
            cam_raw = self._forward_cam_branch(
                hidden_states,
                spatial_shape,
                camera_conditions,
                rotary_emb,
                precomputed_gates=(beta, decay),
            )
            cam_contrib = _linear_output(self.out_proj_cam(cam_raw))
            combined = main_raw + cam_contrib.to(main_raw.dtype)
            # Audit §6.13i probe hook: dump attn sub-stages for block-0.
            # First call wins via file existence guard.
            _attn_dump = os.environ.get("SANA_WM_DUMP_ATTN0", "")
            if _attn_dump and not os.path.exists(f"{_attn_dump}_done.flag"):
                torch.save(
                    {
                        "attn_in": hidden_states.detach().cpu(),
                        "main_raw": main_raw.detach().cpu(),
                        "cam_raw": cam_raw.detach().cpu(),
                        "cam_contrib": cam_contrib.detach().cpu(),
                        "combined": combined.detach().cpu(),
                        "beta": beta.detach().cpu(),
                        "decay": decay.detach().cpu(),
                    },
                    f"{_attn_dump}_internals.pt",
                )
            attn_out = self._apply_output_gate_and_proj(combined, hidden_states)
            if _attn_dump and not os.path.exists(f"{_attn_dump}_done.flag"):
                gate = F.silu(
                    _linear_output(self.output_gate(hidden_states)).float()
                ).to(combined.dtype)
                torch.save(
                    {
                        "output_gate": gate.detach().cpu(),
                        "pre_proj": (combined * gate).detach().cpu(),
                        "attn_out": attn_out.detach().cpu(),
                    },
                    f"{_attn_dump}_post.pt",
                )
                with open(f"{_attn_dump}_done.flag", "w") as _f:
                    _f.write("done\n")
                print(
                    f"[sana-wm probe] saved native attn-0 dump prefix={_attn_dump} "
                    f"main_raw_norm={main_raw.float().norm().item():.2f} "
                    f"cam_contrib_norm={cam_contrib.float().norm().item():.2f} "
                    f"combined_norm={combined.float().norm().item():.2f} "
                    f"attn_out_norm={attn_out.float().norm().item():.2f}",
                    flush=True,
                )
            return attn_out

        if spatial_shape is not None:
            cam_disabled = os.environ.get("VLLM_OMNI_SANA_WM_DISABLE_CAM_BRANCH", "").lower() in {
                "1", "true", "yes", "on",
            }
            main_raw = self._forward_softmax_raw(hidden_states, spatial_shape, rotary_emb)
            if camera_conditions is not None and self.q_proj_cam is not None and not cam_disabled:
                cam_raw = self._forward_softmax_cam_branch(
                    hidden_states,
                    spatial_shape,
                    camera_conditions,
                    rotary_emb,
                )
                cam_contrib = _linear_output(self.out_proj_cam(cam_raw))
                main_raw = main_raw + cam_contrib.to(main_raw.dtype)
            return self._apply_output_gate_and_proj(main_raw, hidden_states)

        # Shape-only fallback for legacy callers that do not provide THW.
        qkv = _linear_output(self.qkv(hidden_states))
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)
        query = self._split_heads(self.q_norm(query))
        key = self._split_heads(self.k_norm(key))
        value = self._split_heads(value)
        if rotary_emb is not None:
            query = self._apply_rotary_emb_to_sdpa(query, rotary_emb)
            key = self._apply_rotary_emb_to_sdpa(key, rotary_emb)
        attn = F.scaled_dot_product_attention(query, key, value)
        return self._apply_output_gate_and_proj(self._merge_heads(attn), hidden_states)


class SanaWmCrossAttention(nn.Module):
    def __init__(
        self,
        config: SanaWmConfig,
        *,
        quant_config: QuantizationConfig | None = None,
        use_vllm_parallel_layers: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.total_num_heads = max(hidden_size // max(config.linear_head_dim, 1), 1)
        self.head_dim = hidden_size // self.total_num_heads
        self.num_heads = self.total_num_heads
        inner = self.total_num_heads * self.head_dim
        self.uses_vllm_parallel_layers = use_vllm_parallel_layers and _vllm_parallel_layers_available()
        if self.uses_vllm_parallel_layers:
            assert ColumnParallelLinear is not None
            assert RowParallelLinear is not None
            self.q_linear = ColumnParallelLinear(
                hidden_size,
                inner,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_linear" if prefix else "q_linear",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.kv_linear = ColumnParallelLinear(
                hidden_size,
                inner * 2,
                bias=True,
                gather_output=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_linear" if prefix else "kv_linear",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
            self.num_heads = inner // max(_get_tp_world_size(), 1) // self.head_dim
            self.proj = RowParallelLinear(
                inner,
                hidden_size,
                bias=True,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.proj" if prefix else "proj",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
        else:
            self.q_linear = nn.Linear(hidden_size, inner)
            self.kv_linear = nn.Linear(hidden_size, inner * 2)
            self.proj = nn.Linear(inner, hidden_size)
        local_inner = self.num_heads * self.head_dim
        norm_cls = _maybe_make_vllm_rms_norm if config.cross_norm else (lambda *_args, **_kwargs: nn.Identity())
        self.q_norm = norm_cls(local_inner)
        self.k_norm = norm_cls(local_inner)
        self.softmax_attn = (
            Attention(
                num_heads=self.num_heads,
                head_size=self.head_dim,
                num_kv_heads=self.num_heads,
                softmax_scale=1.0 / (self.head_dim**0.5),
                causal=False,
                role="cross",
                qkv_layout="BSND",
                skip_sequence_parallel=True,
                disable_kv_quant=True,
                prefix=prefix,
            )
            if _vllm_attention_available()
            else None
        )

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = tensor.shape
        tensor = tensor.reshape(batch, seq_len, self.num_heads, hidden_size // self.num_heads)
        return tensor.transpose(1, 2)

    def _reshape_to_seq_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = tensor.shape
        return tensor.reshape(batch, seq_len, self.num_heads, hidden_size // self.num_heads)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = tensor.shape
        return tensor.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.q_norm(_linear_output(self.q_linear(hidden_states)))
        key, value = _linear_output(self.kv_linear(encoder_hidden_states)).chunk(2, dim=-1)
        key = self.k_norm(key)
        if self.softmax_attn is not None and attention_mask is None:
            query = self._reshape_to_seq_heads(query)
            key = self._reshape_to_seq_heads(key)
            value = self._reshape_to_seq_heads(value)
            attn = self.softmax_attn(query, key, value)
            return _linear_output(self.proj(attn.flatten(2, 3)))
        # vLLM Attention path does not currently take the SANA-WM prompt
        # padding mask; fall back to SDPA for exact NVlabs semantics when a
        # mask is supplied.
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(
                    "Sana-WM cross-attention mask must be shaped [B, text_len], "
                    f"got {tuple(attention_mask.shape)}."
                )
            attn_mask = (1 - attention_mask.to(query.dtype)) * -10000.0
            attn_mask = attn_mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        attn = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        return _linear_output(self.proj(self._merge_heads(attn)))


class _ConvWrapper(nn.Module):
    def __init__(self, conv: nn.Module) -> None:
        super().__init__()
        self.conv = conv


class SanaWmMbConvFfn(nn.Module):
    def __init__(self, config: SanaWmConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        expanded = int(hidden_size * config.mlp_ratio) * 2
        t_padding = config.t_kernel_size // 2
        self.glu_act = nn.SiLU()
        self.inverted_conv = _ConvWrapper(nn.Conv2d(hidden_size, expanded, kernel_size=1))
        self.depth_conv = _ConvWrapper(nn.Conv2d(expanded, expanded, kernel_size=3, padding=1, groups=expanded))
        self.point_conv = _ConvWrapper(nn.Conv2d(expanded // 2, hidden_size, kernel_size=1, bias=False))
        self.t_conv = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=(config.t_kernel_size, 1),
            padding=(t_padding, 0),
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
        """Match NVlabs ``GLUMBConvTemp.forward`` exactly (audit §6.13h):

        1. Reshape ``(B, N=F*H*W, C) → (B*F, H, W, C) → (B*F, C, H, W)``
           so the spatial MBConv runs PER FRAME with proper 2D (H, W)
           geometry. Our previous reshape ``(B, C, F, H*W)`` made the
           3×3 ``depth_conv`` span the (F, H*W) plane treating F as
           height and the flattened H*W as a 1D width — wrong axis,
           producing garbage (cosine ≈ 0.04 vs NVlabs in the
           block-0 parity probe).
        2. Apply expand → depthwise 3×3 → GLU → contract on per-frame
           (B*F, C, H, W).
        3. Temporal aggregation: reshape back to ``(B, C, F, H*W)``
           and add ``t_conv`` (kernel ``(3, 1)``) along the F axis.
        4. Reshape to ``(B, N, C)``.
        """
        batch, _, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        # (B, F*H*W, C) → (B*F, H, W, C) → (B*F, C, H, W)
        x = (
            hidden_states.reshape(batch * frames, spatial_tokens, hidden_size)
            .reshape(batch * frames, height, width, hidden_size)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        # NVlabs ``ConvLayer`` wraps Conv2d in `Conv → norm → act`. For SANA-WM
        # `act=("silu", "silu", None)`: inverted_conv applies SiLU, depth_conv
        # and point_conv don't. Our ``_ConvWrapper`` only stores the raw
        # Conv2d to match the checkpoint key (`inverted_conv.conv.weight`),
        # so we need to apply the SiLU explicitly here — skipping it left
        # the expanded features unactivated and inflated the downstream
        # GLU output magnitude by ~12× (audit §6.13h block-0 probe).
        x_inv = self.glu_act(self.inverted_conv.conv(x))
        x_dep = self.depth_conv.conv(x_inv)
        value, gate = x_dep.chunk(2, dim=1)
        x_glu = value * self.glu_act(gate)
        x_pt = self.point_conv.conv(x_glu)  # (B*F, hidden, H, W)
        # Audit §6.13i/§6.13n probe hook: dump MLP sub-stages, optionally
        # for a specific block index (SANA_WM_DUMP_MLP_BLOCK_IDX, default 0).
        # File name suffixed with _block{idx} so concurrent dumps don't collide.
        _mlp_dump = os.environ.get("SANA_WM_DUMP_MLP0", "")
        _mlp_target_idx = int(os.environ.get("SANA_WM_DUMP_MLP_BLOCK_IDX", "0"))
        _mlp_idx = getattr(self, "block_idx", 0)
        _mlp_per = f"{_mlp_dump}_block{_mlp_idx}" if _mlp_dump and _mlp_idx == _mlp_target_idx else ""
        if _mlp_per and not os.path.exists(f"{_mlp_per}_done.flag"):
            # Also include min/max/has_nan/has_inf per stage to catch overflow.
            def _stats(t):
                tf = t.float()
                return {
                    "norm": float(tf.norm().item()),
                    "absmax": float(tf.abs().max().item()),
                    "has_nan": bool(tf.isnan().any().item()),
                    "has_inf": bool(tf.isinf().any().item()),
                }
            torch.save(
                {
                    "mlp_in_4d": x.detach().cpu(),
                    "after_inverted_silu": x_inv.detach().cpu(),
                    "after_depth": x_dep.detach().cpu(),
                    "after_glu": x_glu.detach().cpu(),
                    "after_point": x_pt.detach().cpu(),
                    "stats": {
                        "mlp_in_4d": _stats(x),
                        "after_inverted_silu": _stats(x_inv),
                        "after_depth": _stats(x_dep),
                        "after_glu": _stats(x_glu),
                        "after_point": _stats(x_pt),
                    },
                    "block_idx": _mlp_idx,
                    "dtype_input": str(x.dtype),
                },
                f"{_mlp_per}_stages.pt",
            )
            with open(f"{_mlp_per}_done.flag", "w") as _f:
                _f.write("done\n")
            print(
                f"[sana-wm probe] block{_mlp_idx} MLP stage dump "
                f"inv_norm={x_inv.float().norm().item():.2f} absmax={x_inv.float().abs().max().item():.2f} "
                f"depth_absmax={x_dep.float().abs().max().item():.2f} "
                f"glu_absmax={x_glu.float().abs().max().item():.2f} "
                f"point_absmax={x_pt.float().abs().max().item():.2f}",
                flush=True,
            )
        x = x_pt
        # Temporal aggregation: (B*F, C, H, W) → (B, F, C, H*W) → (B, C, F, H*W)
        x_temporal = (
            x.reshape(batch, frames, hidden_size, spatial_tokens).permute(0, 2, 1, 3)
        )
        t_conv_out = self.t_conv(x_temporal)
        x_temporal = x_temporal + t_conv_out
        # → (B, F, H*W, C) → (B, N, C)
        final = (
            x_temporal.permute(0, 2, 3, 1)
            .reshape(batch, frames * spatial_tokens, hidden_size)
            .contiguous()
        )
        # Extra dump: t_conv output + final mlp output (§6.13j/§6.13n).
        # Gated by SANA_WM_DUMP_MLP_FINAL prefix + SANA_WM_DUMP_MLP_BLOCK_IDX
        # (default 0). Per-block filename suffix so concurrent dumps don't collide.
        _mlp_final_dump = os.environ.get("SANA_WM_DUMP_MLP_FINAL", "")
        _mlpf_target = int(os.environ.get("SANA_WM_DUMP_MLP_BLOCK_IDX", "0"))
        _mlpf_idx = getattr(self, "block_idx", 0)
        _mlpf_per = (
            f"{_mlp_final_dump}_block{_mlpf_idx}"
            if _mlp_final_dump and _mlpf_idx == _mlpf_target
            else ""
        )
        if _mlpf_per and not os.path.exists(f"{_mlpf_per}_done.flag"):
            torch.save(
                {
                    "t_conv_in": (x_temporal - t_conv_out).detach().cpu(),
                    "t_conv_out": t_conv_out.detach().cpu(),
                    "after_tconv_add": x_temporal.detach().cpu(),
                    "final_mlp_out": final.detach().cpu(),
                    "t_conv_weight_norm": float(self.t_conv.weight.detach().float().norm().item()),
                    "block_idx": _mlpf_idx,
                },
                f"{_mlpf_per}_final.pt",
            )
            with open(f"{_mlpf_per}_done.flag", "w") as _f:
                _f.write("done\n")
            print(
                f"[sana-wm probe] block{_mlpf_idx} MLP-final dump "
                f"t_conv_w_norm={self.t_conv.weight.float().norm().item():.4f} "
                f"t_conv_in_norm={(x_temporal - t_conv_out).float().norm().item():.2f} "
                f"t_conv_out_norm={t_conv_out.float().norm().item():.2f} "
                f"final_mlp_out_norm={final.float().norm().item():.2f}",
                flush=True,
            )
        return final


class SanaWmBlock(nn.Module):
    def __init__(
        self,
        config: SanaWmConfig,
        *,
        block_idx: int = 0,
        quant_config: QuantizationConfig | None = None,
        use_vllm_parallel_layers: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.block_idx = block_idx
        # Attach block_idx to attn/mlp so per-block probes can read it.
        # (Set after they're created below — see end of __init__.)
        hidden_size = config.hidden_size
        use_gdn = config.softmax_every_n <= 0 or (block_idx + 1) % config.softmax_every_n != 0
        use_plucker_proj = config.use_chunk_plucker_post_attn and (
            config.chunk_plucker_post_attn_blocks < 0 or block_idx < config.chunk_plucker_post_attn_blocks
        )
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SanaWmSelfAttention(
            config,
            use_gdn=use_gdn,
            quant_config=quant_config,
            use_vllm_parallel_layers=use_vllm_parallel_layers,
            prefix=f"{prefix}.attn" if prefix else "attn",
        )
        self.cross_attn = SanaWmCrossAttention(
            config,
            quant_config=quant_config,
            use_vllm_parallel_layers=use_vllm_parallel_layers,
            prefix=f"{prefix}.cross_attn" if prefix else "cross_attn",
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = SanaWmMbConvFfn(config)
        self.mlp.block_idx = block_idx
        self.attn.block_idx = block_idx
        if use_plucker_proj:
            if use_vllm_parallel_layers and _vllm_parallel_layers_available() and ColumnParallelLinear is not None:
                self.plucker_proj: nn.Module | None = ColumnParallelLinear(
                    hidden_size,
                    hidden_size,
                    bias=True,
                    gather_output=True,
                    return_bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.plucker_proj" if prefix else "plucker_proj",
                    disable_tp=_disable_tp_for_vllm_layer(),
                )
            else:
                self.plucker_proj = nn.Linear(hidden_size, hidden_size)
        else:
            self.plucker_proj = None
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))

    @staticmethod
    def _modulate(hidden_states: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return hidden_states * (1 + scale) + shift

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_modulation: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None = None,
        camera_hidden_states: torch.Tensor | None = None,
        camera_conditions: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # NVlabs dispatch convention (audit §6.13a): when timestep
        # modulation carries a frame axis (ndim > 2), route through the
        # frame-aware path so shift/scale/gate are applied per frame.
        if timestep_modulation.ndim > 2:
            return self._forward_frame_aware(
                hidden_states,
                encoder_hidden_states,
                timestep_modulation,
                spatial_shape,
                rotary_emb,
                camera_hidden_states,
                camera_conditions,
                encoder_attention_mask,
            )
        batch_size = hidden_states.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep_modulation.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        attn_input = self._modulate(self.norm1(hidden_states), shift_msa, scale_msa)
        attn_output = self.attn(attn_input, spatial_shape, rotary_emb, camera_conditions)
        plucker_proj_disabled = os.environ.get(
            "VLLM_OMNI_SANA_WM_DISABLE_PLUCKER_PROJ", ""
        ).lower() in {"1", "true", "yes", "on"}
        if (
            camera_hidden_states is not None
            and self.plucker_proj is not None
            and not plucker_proj_disabled
        ):
            attn_output = attn_output + _linear_output(self.plucker_proj(camera_hidden_states))
        hidden_states = hidden_states + gate_msa * attn_output
        hidden_states = hidden_states + self.cross_attn(
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask,
        )
        mlp_input = self._modulate(self.norm2(hidden_states), shift_mlp, scale_mlp)
        hidden_states = hidden_states + gate_mlp * self.mlp(mlp_input, spatial_shape)
        return hidden_states

    def _forward_frame_aware(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_modulation: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
        camera_hidden_states: torch.Tensor | None,
        camera_conditions: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-frame timestep modulation path matching NVlabs
        ``SanaVideoMSCamCtrlBlock.forward_frame_aware``.

        ``timestep_modulation`` is ``(B, 1, F, 6*D)``. We split into the
        six per-frame ``(B, F, 1, D)`` chunks via ``scale_shift_table``
        and apply ``shift/scale/gate`` per frame, broadcasting over the
        spatial tokens within each frame.
        """
        batch_size, token_count, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(
                f"Sana-WM frame-aware block expects N=T*H*W, got N={token_count}, THW={spatial_shape}."
            )
        if timestep_modulation.shape[2] != frames:
            raise ValueError(
                "Sana-WM frame-aware block: timestep frame axis must match spatial frames, "
                f"got modulation frames={timestep_modulation.shape[2]} vs spatial frames={frames}."
            )

        # (B, 1, F, 6*D) -> (B, F, 6, D), then add the (1, 1, 6, D) table.
        t_per_frame = timestep_modulation.reshape(batch_size, frames, 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None, :, :] + t_per_frame
        ).chunk(6, dim=-2)  # each: (B, F, 1, D)

        # Audit §6.13h/§6.13m probe hook: dump per-block intermediates for
        # parity vs NVlabs. Gated by SANA_WM_DUMP_BLOCK0 (env path prefix)
        # plus SANA_WM_DUMP_BLOCK_IDXS (comma-separated indices; defaults
        # to "0" for backwards compat). Writes per-block files:
        #   {prefix}_block{idx}_input.pt {prefix}_block{idx}_post_attn.pt
        #   {prefix}_block{idx}_post_cross_attn.pt {prefix}_block{idx}_output.pt
        _dump_prefix = os.environ.get("SANA_WM_DUMP_BLOCK0", "")
        _dump_idxs_env = os.environ.get("SANA_WM_DUMP_BLOCK_IDXS", "0")
        _dump_idx_set = {int(s) for s in _dump_idxs_env.split(",") if s.strip()}
        _per_block_prefix = (
            f"{_dump_prefix}_block{self.block_idx}"
            if _dump_prefix and self.block_idx in _dump_idx_set
            else ""
        )
        # First-write-wins so we only capture the first stage-1 step (the
        # only step with clean controlled input).
        _do_dump = bool(_per_block_prefix) and not os.path.exists(
            f"{_per_block_prefix}_input.pt"
        )
        if _do_dump:
            torch.save(hidden_states.detach().cpu(), f"{_per_block_prefix}_input.pt")
            torch.save(
                {
                    "shift_msa": shift_msa.detach().cpu(),
                    "scale_msa": scale_msa.detach().cpu(),
                    "gate_msa": gate_msa.detach().cpu(),
                    "shift_mlp": shift_mlp.detach().cpu(),
                    "scale_mlp": scale_mlp.detach().cpu(),
                    "gate_mlp": gate_mlp.detach().cpu(),
                    "scale_shift_table": self.scale_shift_table.detach().cpu(),
                    "encoder_hidden_states": encoder_hidden_states.detach().cpu(),
                },
                f"{_per_block_prefix}_modulation.pt",
            )

        # Apply per-frame modulation by reshaping x to (B, F, S, D), broadcasting
        # scale/shift over the spatial axis, then flattening back.
        x_norm1 = self.norm1(hidden_states).reshape(batch_size, frames, spatial_tokens, hidden_size)
        x_msa_in = (x_norm1 * (1 + scale_msa) + shift_msa).reshape(batch_size, token_count, hidden_size)
        attn_output = self.attn(x_msa_in, spatial_shape, rotary_emb, camera_conditions)
        plucker_proj_disabled = os.environ.get(
            "VLLM_OMNI_SANA_WM_DISABLE_PLUCKER_PROJ", ""
        ).lower() in {"1", "true", "yes", "on"}
        if (
            camera_hidden_states is not None
            and self.plucker_proj is not None
            and not plucker_proj_disabled
        ):
            attn_output = attn_output + _linear_output(self.plucker_proj(camera_hidden_states))
        attn_output_4d = attn_output.reshape(batch_size, frames, spatial_tokens, hidden_size)
        hidden_states = hidden_states + (gate_msa * attn_output_4d).reshape(
            batch_size, token_count, hidden_size
        )
        if _do_dump:
            torch.save(
                {
                    "x_msa_in": x_msa_in.detach().cpu(),
                    "attn_output": attn_output.detach().cpu(),
                    "post_attn_residual": hidden_states.detach().cpu(),
                },
                f"{_per_block_prefix}_post_attn.pt",
            )

        hidden_states = hidden_states + self.cross_attn(
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask,
        )
        if _do_dump:
            torch.save(hidden_states.detach().cpu(), f"{_per_block_prefix}_post_cross_attn.pt")

        x_norm2 = self.norm2(hidden_states).reshape(batch_size, frames, spatial_tokens, hidden_size)
        x_mlp_in = (x_norm2 * (1 + scale_mlp) + shift_mlp).reshape(batch_size, token_count, hidden_size)
        mlp_out = self.mlp(x_mlp_in, spatial_shape)
        mlp_out_4d = mlp_out.reshape(batch_size, frames, spatial_tokens, hidden_size)
        hidden_states = hidden_states + (gate_mlp * mlp_out_4d).reshape(
            batch_size, token_count, hidden_size
        )
        if _do_dump:
            torch.save(
                {
                    "x_mlp_in": x_mlp_in.detach().cpu(),
                    "mlp_out": mlp_out.detach().cpu(),
                    "block_output": hidden_states.detach().cpu(),
                },
                f"{_per_block_prefix}_output.pt",
            )
            print(
                f"[sana-wm probe] saved native block-{self.block_idx} dump "
                f"input_norm={torch.load(f'{_per_block_prefix}_input.pt').float().norm().item():.3f} "
                f"attn_out_norm={attn_output.float().norm().item():.3f} "
                f"mlp_out_norm={mlp_out.float().norm().item():.3f} "
                f"block_out_norm={hidden_states.float().norm().item():.3f}",
                flush=True,
            )
        return hidden_states


class SanaWmFinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: tuple[int, int, int],
        out_channels: int,
        *,
        quant_config: Any = None,
        use_vllm_parallel_layers: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.zeros(2, hidden_size))
        self.out_channels = out_channels
        out_features = _prod(patch_size) * out_channels
        if use_vllm_parallel_layers and _vllm_parallel_layers_available() and ColumnParallelLinear is not None:
            self.linear: nn.Module = ColumnParallelLinear(
                hidden_size,
                out_features,
                bias=True,
                gather_output=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.linear" if prefix else "linear",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
        else:
            self.linear = nn.Linear(hidden_size, out_features)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embed: torch.Tensor,
        spatial_shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        # Per-frame timestep contract (audit §6.13a) dispatch.
        if timestep_embed.ndim > 2:
            return self._forward_frame_aware(hidden_states, timestep_embed, spatial_shape)
        shift, scale = (self.scale_shift_table[None] + timestep_embed[:, None]).chunk(2, dim=1)
        hidden_states = self.norm_final(hidden_states) * (1 + scale) + shift
        return _linear_output(self.linear(hidden_states))

    def _forward_frame_aware(
        self,
        hidden_states: torch.Tensor,
        timestep_embed: torch.Tensor,
        spatial_shape: tuple[int, int, int] | None,
    ) -> torch.Tensor:
        """Per-frame final-layer modulation matching NVlabs
        ``T2IFinalLayer.forward_frame_aware``.

        ``timestep_embed`` is ``(B, 1, F, D)``. We transpose to
        ``(B, F, 1, D)`` so it adds correctly to the ``(1, 1, 2, D)``
        ``scale_shift_table`` and produces per-frame ``(B, F, 1, D)``
        ``shift`` / ``scale`` that broadcast over the spatial tokens
        within each frame.
        """
        batch_size, token_count, hidden_size = hidden_states.shape
        frames = timestep_embed.shape[2]
        if spatial_shape is not None:
            spatial_tokens = spatial_shape[1] * spatial_shape[2]
        else:
            spatial_tokens = token_count // frames
        if frames * spatial_tokens != token_count:
            raise ValueError(
                "Sana-WM frame-aware final layer: token count mismatch "
                f"(N={token_count}, F={frames}, S={spatial_tokens})."
            )
        # (B, 1, F, D) -> (B, F, 1, D); add (1, 1, 2, D); chunk into shift/scale: each (B, F, 1, D).
        t_per_frame = timestep_embed.transpose(1, 2)
        shift, scale = (self.scale_shift_table[None, None, :, :] + t_per_frame).chunk(2, dim=-2)
        x_norm = self.norm_final(hidden_states).reshape(batch_size, frames, spatial_tokens, hidden_size)
        x_mod = (x_norm * (1 + scale) + shift).reshape(batch_size, token_count, hidden_size)
        return _linear_output(self.linear(x_mod))


class SanaWmTransformer3DModel(nn.Module):
    """SANA-WM Stage-1 DiT with a runnable pure-PyTorch GDN fallback path."""

    _repeated_blocks: ClassVar[list[str]] = ["blocks"]
    _layerwise_offload_blocks_attr: ClassVar[str] = "blocks"
    _hsdp_shard_conditions: ClassVar[list[Any]] = [_is_sana_wm_transformer_block]
    _sp_plan: ClassVar[dict[str, Any] | None] = _build_sana_wm_sp_plan()

    def __init__(
        self,
        config: SanaWmConfig | None = None,
        *,
        materialize: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        latent_channels: int = SANA_WM_STAGE1_LATENT_CHANNELS,
        prompt_channels: int = SANA_WM_STAGE1_PROMPT_CHANNELS,
        quant_config: QuantizationConfig | None = None,
        use_vllm_parallel_layers: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config or SanaWmConfig()
        self.quant_config = quant_config
        self.prefix = prefix
        self.use_vllm_parallel_layers = use_vllm_parallel_layers and _vllm_parallel_layers_available()
        self._latent_channels = latent_channels
        self._prompt_channels = prompt_channels
        self._is_materialized = False
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self._loaded_parameters = nn.ParameterDict()
        self._source_to_remapped_name: dict[str, str] = {}
        self._remapped_to_storage_name: dict[str, str] = {}
        self._storage_to_remapped_name: dict[str, str] = {}
        self._materialized_loaded_names: set[str] = set()
        self.last_load_report = SanaWmStage1LoadReport()
        if materialize:
            self.materialize(
                device=device,
                dtype=dtype,
                latent_channels=latent_channels,
                prompt_channels=prompt_channels,
            )

    @property
    def is_materialized(self) -> bool:
        return self._is_materialized

    def materialize(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        latent_channels: int | None = None,
        prompt_channels: int | None = None,
    ) -> None:
        if self._is_materialized:
            return
        self._latent_channels = latent_channels or self._latent_channels
        self._prompt_channels = prompt_channels or self._prompt_channels
        self.patch_size = _to_3tuple(self.config.patch_size)
        self.x_embedder = SanaWmPatchEmbedMS3D(self.patch_size, self._latent_channels, self.config.hidden_size)
        self.y_embedder = SanaWmTextEmbedder(
            self._prompt_channels,
            self.config.hidden_size,
            self.config.model_max_length,
            quant_config=self.quant_config,
            use_vllm_parallel_layers=self.use_vllm_parallel_layers,
            prefix=f"{self.prefix}.y_embedder" if self.prefix else "y_embedder",
        )
        self.t_embedder = SanaWmTimestepEmbedder(
            SANA_WM_STAGE1_TIMESTEP_CHANNELS,
            self.config.hidden_size,
            quant_config=self.quant_config,
            use_vllm_parallel_layers=self.use_vllm_parallel_layers,
            prefix=f"{self.prefix}.t_embedder" if self.prefix else "t_embedder",
        )
        # t_block: SiLU → Linear(hidden, 6*hidden).  Weight key is t_block.1.{weight,bias}
        # to match the Stage-1 checkpoint layout (nn.Sequential index 1).
        if self.use_vllm_parallel_layers and ColumnParallelLinear is not None:
            _t_block_linear: nn.Module = ColumnParallelLinear(
                self.config.hidden_size,
                6 * self.config.hidden_size,
                bias=True,
                gather_output=True,
                return_bias=False,
                quant_config=self.quant_config,
                prefix=f"{self.prefix}.t_block.1" if self.prefix else "t_block.1",
                disable_tp=_disable_tp_for_vllm_layer(),
            )
        else:
            _t_block_linear = nn.Linear(self.config.hidden_size, 6 * self.config.hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), _t_block_linear)
        self.plucker_embedder = SanaWmPatchEmbedMS3D(
            self.patch_size,
            self.config.chunk_plucker_channels,
            self.config.hidden_size,
        )
        self.raymap_embedder = SanaWmPatchEmbedMS3D(self.patch_size, 3, self.config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                SanaWmBlock(
                    self.config,
                    block_idx=i,
                    quant_config=self.quant_config,
                    use_vllm_parallel_layers=self.use_vllm_parallel_layers,
                    prefix=f"{self.prefix}.blocks.{i}" if self.prefix else f"blocks.{i}",
                )
                for i in range(self.config.num_blocks)
            ]
        )
        self.final_layer = SanaWmFinalLayer(
            self.config.hidden_size,
            self.patch_size,
            self._latent_channels,
            quant_config=self.quant_config,
            use_vllm_parallel_layers=self.use_vllm_parallel_layers,
            prefix=f"{self.prefix}.final_layer" if self.prefix else "final_layer",
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 484, self.config.hidden_size))
        self.rope = SanaWmWanRotaryPosEmbed(self.config.linear_head_dim)
        self.attention_y_norm = _maybe_make_vllm_rms_norm(self.config.hidden_size)
        self.to(device=device, dtype=dtype)
        self._is_materialized = True
        self._apply_loaded_tensors_to_materialized()

    @staticmethod
    def _storage_name(remapped_name: str, index: int) -> str:
        digest = hashlib.sha1(remapped_name.encode("utf-8")).hexdigest()[:16]
        return f"w_{index}_{digest}"

    @staticmethod
    def _local_name(remapped_name: str) -> str:
        return remapped_name.removeprefix("transformer.")

    def _copy_to_materialized_param(
        self,
        remapped_name: str,
        tensor: torch.Tensor,
        *,
        require_target: bool = False,
    ) -> bool:
        if not self._is_materialized or not remapped_name.startswith("transformer."):
            if require_target:
                raise ValueError(
                    f"Sana-WM weight {remapped_name} was remapped but is not a transformer weight "
                    "that can be consumed by the materialized Stage-1 model."
                )
            return False
        local_name = self._local_name(remapped_name)
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        target = params.get(local_name)
        if target is None:
            target = buffers.get(local_name)
        if target is None:
            if require_target:
                raise ValueError(
                    f"Sana-WM weight {remapped_name} was remapped but not consumed by the "
                    "materialized Stage-1 model."
                )
            return False
        if tuple(target.shape) != tuple(tensor.shape):
            weight_loader = getattr(target, "weight_loader", None)
            if callable(weight_loader):
                weight_loader(target, tensor.to(device=target.device, dtype=target.dtype))
                self._materialized_loaded_names.add(remapped_name)
                return True
            if tuple(target.shape) != tuple(tensor.shape):
                tensor = self._maybe_shard_loaded_tensor(remapped_name, tensor, target)
            if tuple(target.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"Sana-WM weight shape mismatch for {remapped_name}: "
                    f"expected {tuple(target.shape)}, got {tuple(tensor.shape)}."
                )
        with torch.no_grad():
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
        self._materialized_loaded_names.add(remapped_name)
        return True

    @staticmethod
    def _maybe_shard_loaded_tensor(
        remapped_name: str,
        tensor: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Shard simple non-vLLM parameters that become TP-local.

        vLLM parallel linear parameters have their own `weight_loader`.
        SANA-WM also has model-local GDN vectors and depthwise temporal
        convolution weights (`A_log`, `dt_bias`, `conv_k`) whose checkpoint
        tensors are full-sized. Narrow those along the single changed dimension
        when TP makes the materialized parameter local.
        """

        if tensor.ndim != target.ndim:
            return tensor
        differing_dims = [
            dim
            for dim, (target_size, loaded_size) in enumerate(zip(target.shape, tensor.shape, strict=True))
            if target_size != loaded_size
        ]
        if len(differing_dims) != 1:
            return tensor
        dim = differing_dims[0]
        target_size = target.shape[dim]
        loaded_size = tensor.shape[dim]
        tp_size = _get_tp_world_size()
        tp_rank = _get_tp_rank()
        if tp_size <= 1 or target_size * tp_size != loaded_size:
            return tensor
        if not any(token in remapped_name for token in (".A_log", ".dt_bias", ".conv_k.", ".q_norm.", ".k_norm.")):
            return tensor
        start = tp_rank * target_size
        return tensor.narrow(dim, start, target_size)

    def _store_tensor(self, remapped_name: str, tensor: torch.Tensor) -> None:
        if self._is_materialized:
            self._copy_to_materialized_param(remapped_name, tensor, require_target=True)
            return
        storage_name = self._storage_name(remapped_name, len(self._remapped_to_storage_name))
        target_device = self._device_anchor.device
        if target_device.type != "meta" and tensor.device != target_device:
            tensor = tensor.to(target_device)
        if torch.is_floating_point(tensor) or torch.is_complex(tensor):
            self._loaded_parameters[storage_name] = nn.Parameter(tensor.detach(), requires_grad=False)
        else:
            self.register_buffer(storage_name, tensor.detach(), persistent=True)
        self._remapped_to_storage_name[remapped_name] = storage_name
        self._storage_to_remapped_name[storage_name] = remapped_name

    def _apply_loaded_tensors_to_materialized(self) -> None:
        applied: list[str] = []
        unapplied: list[str] = []
        for remapped_name, storage_name in list(self._remapped_to_storage_name.items()):
            tensor = (
                self._loaded_parameters[storage_name]
                if storage_name in self._loaded_parameters
                else getattr(self, storage_name)
            )
            if self._copy_to_materialized_param(remapped_name, tensor):
                applied.append(remapped_name)
            else:
                unapplied.append(remapped_name)
        self.last_load_report = replace(
            self.last_load_report,
            materialized_weights=len(applied),
            unapplied_weights=tuple(unapplied),
        )
        if unapplied:
            raise ValueError(
                "Sana-WM Stage-1 checkpoint keys were remapped but not consumed by the "
                f"materialized model: {unapplied[:10]}"
            )

    def get_loaded_tensor(self, remapped_name: str) -> torch.Tensor:
        storage_name = self._remapped_to_storage_name.get(remapped_name)
        if storage_name is not None:
            if storage_name in self._loaded_parameters:
                return self._loaded_parameters[storage_name]
            return getattr(self, storage_name)
        if self._is_materialized and remapped_name in self._materialized_loaded_names:
            local_name = self._local_name(remapped_name)
            params = dict(self.named_parameters())
            buffers = dict(self.named_buffers())
            target = params.get(local_name)
            if target is None:
                target = buffers.get(local_name)
            if target is not None:
                return target
        raise KeyError(remapped_name)

    def _positional_embedding(self, token_count: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        pos_embed = self.pos_embed.to(device=device, dtype=dtype)
        if pos_embed.shape[1] == token_count:
            return pos_embed
        pos_embed = pos_embed.transpose(1, 2)
        pos_embed = F.interpolate(pos_embed, size=token_count, mode="linear", align_corners=False)
        return pos_embed.transpose(1, 2)

    @staticmethod
    def _match_tokens(hidden_states: torch.Tensor, expected_tokens: int) -> torch.Tensor:
        if hidden_states.shape[1] == expected_tokens:
            return hidden_states
        if hidden_states.shape[1] > expected_tokens:
            return hidden_states[:, :expected_tokens]
        pad = hidden_states[:, -1:].expand(-1, expected_tokens - hidden_states.shape[1], -1)
        return torch.cat([hidden_states, pad], dim=1)

    def _camera_hidden_states_from_conditions(
        self,
        *,
        plucker: torch.Tensor | None,
        spatial_raymap: torch.Tensor | None,
        spatial_shape: tuple[int, int, int],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if plucker is None or not hasattr(self, "plucker_embedder"):
            return None
        if plucker.ndim == 4:
            plucker = plucker.unsqueeze(0)
        plucker = plucker.to(device=device, dtype=dtype)
        if plucker.shape[0] == 1 and batch_size > 1:
            plucker = plucker.expand(batch_size, -1, -1, -1, -1)
        camera_hidden_states = self.plucker_embedder(plucker)
        expected_tokens = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
        camera_hidden_states = self._match_tokens(camera_hidden_states, expected_tokens)

        # Fuse per-pixel ray-direction map via raymap_embedder only on the
        # non-plucker path. NVlabs skips the absmap/raymap embedder whenever
        # chunk-plucker input or post-attention injection is enabled.
        use_chunk_plucker = bool(
            getattr(self.config, "use_chunk_plucker_input", False)
            or getattr(self.config, "use_chunk_plucker_post_attn", False)
        )
        if spatial_raymap is not None and not use_chunk_plucker and hasattr(self, "raymap_embedder"):
            if spatial_raymap.ndim == 4:  # [C, F, H, W] → [1, C, F, H, W]
                spatial_raymap = spatial_raymap.unsqueeze(0)
            spatial_raymap = spatial_raymap.to(device=device, dtype=dtype)
            if spatial_raymap.shape[0] == 1 and batch_size > 1:
                spatial_raymap = spatial_raymap.expand(batch_size, -1, -1, -1, -1)
            ray_hidden = self._match_tokens(self.raymap_embedder(spatial_raymap), expected_tokens)
            camera_hidden_states = camera_hidden_states + ray_hidden

        return camera_hidden_states

    def _unpatchify(self, hidden_states: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        frames, height, width = spatial_shape
        patch_frames, patch_height, patch_width = self.patch_size
        hidden_states = hidden_states.reshape(
            batch_size,
            frames,
            height,
            width,
            patch_frames,
            patch_height,
            patch_width,
            self._latent_channels,
        )
        hidden_states = torch.einsum("nfhwopqc->ncfohpwq", hidden_states)
        return hidden_states.reshape(
            batch_size,
            self._latent_channels,
            frames * patch_frames,
            height * patch_height,
            width * patch_width,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor | float | int,
        *,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        camera_hidden_states: torch.Tensor | None = None,
        camera_encoder: SanaWmCameraEmbedder | None = None,
        plucker: torch.Tensor | None = None,
        raymap: torch.Tensor | None = None,
        spatial_raymap: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 5:
            raise ValueError("Sana-WM transformer expects latent input shaped [B, C, F, H, W].")
        # Audit §6.13n: env-gated force-fp32 transformer forward for the
        # bf16-precision-drift upper-bound experiment. Casts inputs to fp32,
        # also upcasts model params (one-shot), and casts noise_pred back to
        # the original input dtype before returning.
        _force_fp32 = os.environ.get("SANA_WM_FORCE_FP32_TRANSFORMER", "").lower() in {
            "1", "true", "yes", "on",
        }
        _input_dtype = hidden_states.dtype
        if _force_fp32:
            if not getattr(self, "_fp32_upcasted", False):
                self.to(torch.float32)
                self._fp32_upcasted = True
            hidden_states = hidden_states.float()
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.float()
            if encoder_attention_mask is not None:
                encoder_attention_mask = encoder_attention_mask.to(device=hidden_states.device)
            if camera_hidden_states is not None:
                camera_hidden_states = camera_hidden_states.float()
            if plucker is not None:
                plucker = plucker.float()
            if raymap is not None:
                raymap = raymap.float()
            if spatial_raymap is not None:
                spatial_raymap = spatial_raymap.float()
        if not self._is_materialized:
            prompt_channels = (
                int(encoder_hidden_states.shape[-1])
                if encoder_hidden_states is not None
                else SANA_WM_STAGE1_PROMPT_CHANNELS
            )
            self.materialize(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                latent_channels=int(hidden_states.shape[1]),
                prompt_channels=prompt_channels,
            )

        batch_size = hidden_states.shape[0]
        latent_shape = hidden_states.shape[2:]
        hidden_states, spatial_shape = self.x_embedder.project_with_shape(hidden_states)
        rotary_emb = None
        if self.config.pos_embed_type == "wan_rope":
            rotary_emb = self.rope(spatial_shape, hidden_states.device)
        else:
            hidden_states = hidden_states + self._positional_embedding(
                hidden_states.shape[1], hidden_states.dtype, hidden_states.device
            )

        encoder_hidden_states = self.y_embedder(
            encoder_hidden_states.to(device=hidden_states.device) if encoder_hidden_states is not None else None,
            batch_size=batch_size,
            dtype=hidden_states.dtype,
        )
        encoder_hidden_states = self.attention_y_norm(encoder_hidden_states)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[:2],
                device=hidden_states.device,
                dtype=torch.float32,
            )
        else:
            encoder_attention_mask = encoder_attention_mask.to(device=hidden_states.device)
            if encoder_attention_mask.shape != encoder_hidden_states.shape[:2]:
                raise ValueError(
                    "Sana-WM encoder_attention_mask must match text tokens "
                    f"{tuple(encoder_hidden_states.shape[:2])}, got {tuple(encoder_attention_mask.shape)}."
                )

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=hidden_states.device, dtype=torch.float32)
        timestep = timestep.to(device=hidden_states.device)
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        elif timestep.ndim == 1 and timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        # Per-frame timestep contract (audit §6.13a): when ``timestep`` is
        # rank > 1 (e.g. ``(B, 1, F)`` from the LTX flow-matching sampler),
        # the per-frame embedding is computed on the flattened token axis
        # and unflattened back to the input rank. ``time_embed`` then has
        # an extra ``hidden_size`` axis appended (``(B, 1, F, D)``) and
        # ``timestep_modulation`` has ``6*hidden_size`` (``(B, 1, F, 6*D)``).
        # SanaWmBlock / SanaWmFinalLayer dispatch on ``ndim > 2`` to the
        # frame-aware paths that broadcast per-frame modulation over the
        # spatial tokens within each frame.
        timestep_shape = tuple(timestep.shape)
        time_embed = self.t_embedder(timestep)  # (numel, D)
        # t_block is Sequential(SiLU, Linear|ColumnParallelLinear); index explicitly
        # so _linear_output can unwrap the (tensor, None) tuple from parallel layers.
        _t_silu = self.t_block[0](time_embed)
        timestep_modulation = _linear_output(self.t_block[1](_t_silu)).to(hidden_states.dtype)
        if len(timestep_shape) > 1:
            time_embed = time_embed.unflatten(0, timestep_shape)
            timestep_modulation = timestep_modulation.unflatten(0, timestep_shape)

        if camera_hidden_states is None:
            camera_hidden_states = self._camera_hidden_states_from_conditions(
                plucker=plucker,
                spatial_raymap=spatial_raymap,
                spatial_shape=spatial_shape,
                batch_size=batch_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        if camera_hidden_states is None and camera_encoder is not None:
            camera_hidden_states = camera_encoder(
                plucker=plucker,
                raymap=raymap,
                spatial_shape=spatial_shape,
                batch_size=batch_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        # Build (B, T_latent, 20) camera_conditions from the raw raymap so
        # the per-block UCPE branch can compute its per-pixel apply_fns.
        # When the caller passes pre-embedded camera_hidden_states without
        # raymap, the cam branch becomes a no-op (main-branch only).
        camera_conditions: torch.Tensor | None = None
        if raymap is not None:
            camera_conditions = raymap if raymap.ndim == 3 else raymap.unsqueeze(0)
            if camera_conditions.shape[0] == 1 and batch_size > 1:
                camera_conditions = camera_conditions.expand(batch_size, -1, -1)
            camera_conditions = camera_conditions.to(device=hidden_states.device, dtype=hidden_states.dtype)

        # Audit §6.13n runtime weight probe: env-gated one-time dump of
        # block.mlp.* parameter norms for load_weights verification.
        _rtw_path = os.environ.get("SANA_WM_RUNTIME_WEIGHT_DUMP", "")
        if _rtw_path and not os.path.exists(_rtw_path):
            _rtw_dump: dict = {}
            for _bi, _blk in enumerate(self.blocks):
                for _pn, _pp in _blk.mlp.named_parameters():
                    _rtw_dump[f"block{_bi}.mlp.{_pn}"] = float(
                        _pp.detach().float().norm().item()
                    )
                    _rtw_dump[f"block{_bi}.mlp.{_pn}.shape"] = tuple(_pp.shape)
            torch.save(_rtw_dump, _rtw_path)
            print(
                f"[runtime weight probe] dumped {len(_rtw_dump)//2} mlp params -> {_rtw_path}",
                flush=True,
            )

        # Audit §6.13m compounding probe: env-gated dump of block-N outputs
        # (default 0,5,10,15,19), final-layer in/out, and unpatchify output.
        # Hooks are no-ops unless SANA_WM_DUMP_BLOCKS_PREFIX is set.
        _blocks_prefix = os.environ.get("SANA_WM_DUMP_BLOCKS_PREFIX", "")
        _blocks_target = os.environ.get("SANA_WM_DUMP_BLOCKS_INDICES", "0,5,10,15,19")
        _blocks_idx_set = (
            {int(s) for s in _blocks_target.split(",") if s.strip()}
            if _blocks_prefix
            else set()
        )
        _blocks_done_flag = f"{_blocks_prefix}_done.flag" if _blocks_prefix else ""
        _blocks_active = bool(_blocks_prefix) and not os.path.exists(_blocks_done_flag)
        _blocks_dump: dict = {}

        for block_i, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_modulation,
                spatial_shape,
                rotary_emb,
                camera_hidden_states,
                camera_conditions,
                encoder_attention_mask,
            )
            if _blocks_active and block_i in _blocks_idx_set:
                _blocks_dump[f"block{block_i}_out"] = hidden_states.detach().cpu()

        if _blocks_active:
            _blocks_dump["final_layer_in"] = hidden_states.detach().cpu()
        hidden_states = self.final_layer(
            hidden_states, time_embed.to(hidden_states.dtype), spatial_shape
        )
        if _blocks_active:
            _blocks_dump["final_layer_out"] = hidden_states.detach().cpu()
        hidden_states = self._unpatchify(hidden_states, spatial_shape)
        if _blocks_active:
            _blocks_dump["unpatchify_out"] = hidden_states.detach().cpu()
            torch.save(_blocks_dump, f"{_blocks_prefix}_blocks.pt")
            with open(_blocks_done_flag, "w") as _bf:
                _bf.write("done\n")
            print(
                f"[sana-wm probe] native blocks dump idx={sorted(_blocks_idx_set)} -> "
                f"{_blocks_prefix}_blocks.pt (final_out_norm="
                f"{hidden_states.float().norm().item():.2f})",
                flush=True,
            )
        if hidden_states.shape[2:] != latent_shape:
            hidden_states = F.interpolate(hidden_states, size=latent_shape, mode="trilinear", align_corners=False)
        if _force_fp32 and hidden_states.dtype != _input_dtype:
            hidden_states = hidden_states.to(_input_dtype)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, Any]]) -> set[str]:
        loaded: set[str] = set()
        unmapped: list[str] = []
        duplicates: list[str] = []
        total = 0

        for source_name, tensor in weights:
            total += 1
            remapped_name = normalize_sana_wm_stage1_weight_name(source_name)
            if remapped_name is None:
                unmapped.append(source_name)
                continue
            if (
                remapped_name in self._remapped_to_storage_name
                or remapped_name in self._materialized_loaded_names
                or remapped_name in loaded
            ):
                duplicates.append(remapped_name)
                continue
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Sana-WM weight {source_name!r} must be a torch.Tensor, got {type(tensor).__name__}.")
            self._store_tensor(remapped_name, tensor)
            self._source_to_remapped_name[source_name] = remapped_name
            loaded.add(remapped_name)

        self.last_load_report = SanaWmStage1LoadReport(
            total_weights=total,
            loaded_weights=len(loaded),
            unmapped_weights=tuple(unmapped),
            duplicate_weights=tuple(duplicates),
            loaded_names=tuple(sorted(loaded)),
            materialized_weights=sum(1 for name in loaded if name in self._materialized_loaded_names),
            unapplied_weights=tuple(
                name for name in sorted(loaded) if self._is_materialized and name not in self._materialized_loaded_names
            ),
        )
        if unmapped or duplicates:
            details = []
            if unmapped:
                details.append(f"unmapped={unmapped[:10]}")
            if duplicates:
                details.append(f"duplicates={duplicates[:10]}")
            raise ValueError("Invalid SANA-WM Stage-1 checkpoint keys: " + "; ".join(details))
        return loaded
