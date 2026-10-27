# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Triton kernel arguments keep Triton's uppercase names for pointers and constexprs.
# ruff: noqa: N803
"""Inference-only π0.5 VLA math kernel for vllm-omni.

Only the math that turns a robot observation into an action chunk; no serving or
request glue. Deliberately shaped like ``models/pi0/modeling_pi0.py`` so the two
can later be factored into a shared Pi-family module (RFC step 2) on a
"behaviour unchanged" review.

π0.5 = PaliGemma (SigLIP vision + Gemma 2B LM) prefix + Gemma 300M action expert
suffix + flow-matching head. Where π0 projects the robot state through
``state_proj``, π0.5 discretizes it into prompt tokens — so there is no
``state_proj`` layer, ``sample_actions`` takes no ``state`` argument, and the
suffix is action tokens only, which drops π0's leading state-token boundary from
the suffix attention mask and leaves ``[1] + [0] * (horizon - 1)``.

π0.5 is nonetheless the *larger* model: the 37 AdaRMS ``dense`` projections add
~116M parameters against the ~8K that ``state_proj`` saves. (LeRobot's README
says otherwise; the checkpoint disagrees.)

The layers are vLLM-Omni's: both Gemma towers run on vLLM's packed parallel
linear layers (``QKVParallelLinear``, ``MergedColumnParallelLinear``,
``RowParallelLinear``), ``GemmaRMSNorm`` and ``GeluAndMul``, with the omni
``RotaryEmbedding``; SigLIP is vLLM's ``SiglipVisionModel``. Only the
PaliGemma language model takes a ``quant_config``, and only when one is set.
The AdaRMS norm vLLM lacks is a ``CustomOp`` of its own.

The optimized path's fused Triton kernels live here too, next to the eager
layers they replace; ``Pi05ForActionPrediction.enable_fused_kernels`` switches
the denoising step onto them. The eager baseline never runs them.

Reference implementations:
   - OpenPI: openpi/src/openpi/models_pytorch/pi0_pytorch.py, gemma_pytorch.py
   - LeRobot: lerobot/src/lerobot/policies/pi05/modeling_pi05.py
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.auto import CONFIG_MAPPING
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

from vllm_omni.diffusion.layers.custom_op import CustomOp
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.pi05.cuda_graph_pi05 import Pi05CUDAGraphs

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
DEFAULT_ACTION_DIM = 32
DEFAULT_ACTION_HORIZON = 50
DEFAULT_MAX_TOKEN_LEN = 200  # π0 uses 48
DEFAULT_NUM_INFERENCE_STEPS = 10
DEFAULT_IMAGE_RESOLUTION = (224, 224)  # openpi/models/model.py IMAGE_RESOLUTION
DEFAULT_STATE_NUM_BINS = 256  # openpi PaliGemmaTokenizer.tokenize()

# Checkpoint names → (packed vLLM parameter, shard) in both Gemma towers.
_PACKED_PARAMS = (
    (".self_attn.qkv_proj.", ".self_attn.q_proj.", "q"),
    (".self_attn.qkv_proj.", ".self_attn.k_proj.", "k"),
    (".self_attn.qkv_proj.", ".self_attn.v_proj.", "v"),
    (".mlp.gate_up_proj.", ".mlp.gate_proj.", 0),
    (".mlp.gate_up_proj.", ".mlp.up_proj.", 1),
)
_PACKED_SHARDS = {"qkv_proj": {"q", "k", "v"}, "gate_up_proj": {0, 1}}
_UNUSED_EXPERT_LM_HEAD = "paligemma_with_expert.gemma_expert.lm_head.weight"

# Large negative value to fill masked-out positions in a float attention mask.
# Matches OpenPI's constant exactly so that numerics line up during parity.
# Ref: openpi/src/openpi/models/gemma.py
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38


# ──────────────────────────────────────────────────────────────────────
# Gemma variant configs (matches openpi/models/gemma.py get_config)
# ──────────────────────────────────────────────────────────────────────
class GemmaVariantConfig:
    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaVariantConfig:
    if variant == "gemma_2b":
        return GemmaVariantConfig(2048, 18, 16384, 8, 1, 256)
    elif variant == "gemma_300m":
        return GemmaVariantConfig(1024, 18, 4096, 8, 1, 256)
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ──────────────────────────────────────────────────────────────────────
# Utility functions (match openpi/models_pytorch/pi0_pytorch.py)
# ──────────────────────────────────────────────────────────────────────
def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Compute a sine/cosine positional embedding for scalar timesteps.

    Ref: openpi/models_pytorch/pi0_pytorch.py create_sinusoidal_pos_embedding
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time tensor must be 1-D (batch_size,)")
    if device is None:
        device = time.device

    # Use float64 for the log-linear sweep and for the inner products, to
    # match the reference implementation's numerical behaviour exactly.
    dtype = torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(dtype)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Build a 2D attention mask from a padding mask and an autoregressive mask.

    Ref: openpi/models_pytorch/pi0_pytorch.py make_att_2d_masks
    """
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must be 2-D, got {att_masks.ndim}-D")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must be 2-D, got {pad_masks.ndim}-D")

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def prepare_attention_masks_4d(att_2d_masks: torch.Tensor) -> torch.Tensor:
    """Convert ``(B, S, S)`` bool masks to ``(B, 1, S, S)`` float masks.

    ``True`` → 0.0 (attend), ``False`` → ``OPENPI_ATTENTION_MASK_VALUE``.
    """
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)


class Pi05AdaRMSNorm(CustomOp):
    """Adaptive RMSNorm conditioned on the flow-matching timestep.

    π0 conditions on time by *concatenating* a time embedding onto each action
    embedding. π0.5 instead feeds the time embedding into every action-expert
    norm, which produces a per-layer ``(scale, shift, gate)`` triple::

        y    = norm(x) * (1 + scale) + shift
        out  = residual + gate * sublayer(y)

    ``dense`` is zero-initialized, so an untrained model starts as the identity
    modulation with a closed gate — matching OpenPI's parameterization.

    Note the shape of the unconditioned branch: ``normed * (1 + weight)`` with
    ``weight`` zero-initialized, which is exactly vLLM's ``GemmaRMSNorm``. That
    equivalence is why only the *expert* norms need replacing here and the
    PaliGemma prefix keeps ``GemmaRMSNorm``.

    vLLM has no conditioned equivalent, so this is a ``CustomOp`` of its own:
    ``forward_native`` is the eager formula, and on CUDA a conditioned batch-1
    call on CUDA tensors runs R0's fused Triton kernel (``_rms_norm_kernel``),
    which differs from it by the order of the variance sum only. Any other
    call, a compiled one, or a platform without Triton, falls back to
    ``forward_native``.
    """

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
            nn.init.zeros_(self.dense.bias)
            self.weight = None
        else:
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        return x.float() * torch.rsqrt(var + self.eps)

    def _check_cond(self, cond: torch.Tensor | None) -> None:
        if cond is None:
            # A conditioned norm silently falling back to an unconditioned one
            # would drop the entire timestep signal and still return a
            # well-shaped, finite tensor.
            raise ValueError(
                "Pi05AdaRMSNorm was built with cond_dim="
                f"{self.cond_dim} but called without an AdaRMS conditioning vector."
            )
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected AdaRMS cond dim {self.cond_dim}, got {cond.shape[-1]}")

    def forward_cuda(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not HAS_TRITON or not x.is_cuda or self.dense is None or cond is None:
            return self.forward_native(x, cond)
        # Under torch.compile inductor fuses the eager formula itself; it cannot
        # re-emit this kernel, whose source reads ``tldevice``.
        if torch.compiler.is_compiling():
            return self.forward_native(x, cond)
        if cond.shape[0] != 1 or x.shape[0] != 1:
            return self.forward_native(x, cond)
        self._check_cond(cond)
        modulation = _adarms_modulation(self, cond)
        rows = x.reshape(-1, self.dim)
        normed = torch.empty_like(rows)
        _fused_rms_norm(rows, self, normed, modulation=modulation)
        return normed.view(x.shape), modulation[2 * self.dim :].to(x.dtype)

    def forward_native(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(normed, gate)``; ``gate`` is ``None`` in the unconditioned case."""
        dtype = x.dtype
        normed = self._norm(x)
        if self.dense is None:
            normed = normed * (1.0 + self.weight.float())
            return normed.to(dtype), None

        self._check_cond(cond)
        modulation = self.dense(cond.to(self.dense.weight.dtype))
        if x.ndim == 3:
            # (B, 3*dim) → (B, 1, 3*dim), broadcast across the token axis: the
            # timestep is a per-sample scalar, identical for every action token.
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)


def _gated_residual(residual: torch.Tensor, out: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    if gate is None:
        return residual + out
    return residual + gate * out


# ──────────────────────────────────────────────────────────────────────
# Dual-backbone: PaliGemma + AdaRMS action expert
# ──────────────────────────────────────────────────────────────────────
def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for grouped-query attention.

    ``(B, num_kv_heads, S, D) → (B, num_kv_heads * n_rep, S, D)``.
    """
    if n_rep == 1:
        return hidden_states
    b, nh, s, d = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, nh, n_rep, s, d)
    return hidden_states.reshape(b, nh * n_rep, s, d)


def _attend(query_states, key_states, value_states, attention_mask, num_kv_groups, scaling):
    """Manual eager attention: ``softmax(Q Kᵀ · scale + mask) · V``.

    The mask is sliced to ``key_states.shape[-2]`` so the same
    ``(B, 1, Q, prefix+suffix)`` mask works in both the prefix pass (K length =
    prefix) and the suffix pass (K length = prefix + suffix).
    """
    k = _repeat_kv(key_states, num_kv_groups)
    v = _repeat_kv(value_states, num_kv_groups)
    attn_weights = torch.matmul(query_states, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : k.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return torch.matmul(attn_weights, v)


@dataclass
class Pi05RequestPrefixKV:
    """One request's prefix K/V: computed once per observation, read by every
    denoising step of that request and by no other request.

    ``key`` and ``value`` are ``(num_layers, num_kv_heads, prefix_len,
    head_dim)``, the post-RoPE K/V of each PaliGemma layer in the action
    expert's K/V dtype; ``pad_mask`` is the ``(prefix_len,)`` prefix padding
    mask the denoising steps attend through. The request owns the storage: no
    :class:`Pi05StepKVWorkspace` or CUDA Graph buffer aliases it, so a later
    request's prefix pass cannot change it, and it is freed with the request.
    Nothing carries over to the request's next action chunk, which encodes a
    new observation.
    """

    key: torch.Tensor
    value: torch.Tensor
    pad_mask: torch.Tensor

    @property
    def prefix_len(self) -> int:
        return self.key.shape[2]


class Pi05StepKVWorkspace:
    """The per-layer K/V that one denoising step attends over, for every row of its batch.

    ``key`` and ``value`` are ``(num_layers, batch, num_kv_heads, prefix_len +
    suffix_len, head_dim)``. Along the token axis each layer holds ``[prefix |
    suffix]``: row ``i``'s prefix slots hold the prefix of the request in batch
    row ``i``, and each denoising step overwrites the suffix slots, so a suffix
    layer attends over one contiguous view instead of a fresh ``torch.cat`` of
    the two. The addresses never change, which is what CUDA Graph replay needs.

    The prefix slots are filled one of two ways. ``sample_actions`` runs the
    prefix pass straight into them (``write_prefix``), since its batch is fixed
    for the whole call. When the rows of a step change from one step to the
    next, as they do when requests join and leave a running batch, each request
    keeps its own :class:`Pi05RequestPrefixKV` (``export_prefix``) and the step
    copies the current rows' prefixes in first (``load_prefixes``).

    The model owns the workspace, not a request: the suffix slots are scratch,
    and nothing is written back from the workspace into a request's prefix.
    Every write is a copy on the current stream, ordered after the GPU work
    that last read the slots, so the workspace can take the next batch as soon
    as the host has queued the previous one's reads; a finished or cancelled
    request leaves nothing behind in it that the next load does not overwrite.

    The dtype is the action expert's K/V dtype: writing the prefix K/V casts it
    exactly as the ``.to(k_suf.dtype)`` before the concatenation it replaces.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        prefix_len: int,
        suffix_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        shape = (num_layers, batch_size, num_kv_heads, prefix_len + suffix_len, head_dim)
        self.key = torch.empty(shape, dtype=dtype, device=device)
        self.value = torch.empty(shape, dtype=dtype, device=device)
        self.batch_size = batch_size
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len

    def fits(self, batch_size: int, prefix_len: int) -> bool:
        return batch_size == self.batch_size and prefix_len == self.prefix_len

    def write_prefix(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self.key[layer_idx, :, :, : self.prefix_len].copy_(key)
        self.value[layer_idx, :, :, : self.prefix_len].copy_(value)

    def export_prefix(self, row: int, pad_mask: torch.Tensor) -> Pi05RequestPrefixKV:
        """A request-owned copy of row ``row``'s prefix, with its ``(prefix_len,)`` pad mask."""
        return Pi05RequestPrefixKV(
            key=self.key[:, row, :, : self.prefix_len].clone(),
            value=self.value[:, row, :, : self.prefix_len].clone(),
            pad_mask=pad_mask.clone(),
        )

    def load_prefixes(self, prefixes: list[Pi05RequestPrefixKV]) -> torch.Tensor:
        """Copy request ``i``'s prefix into row ``i``; return the rows' ``(batch, prefix_len)`` pad masks.

        ``prefixes`` is in batch order, which is the only row identity: a
        request may sit in another row than it did on its previous step.
        """
        if len(prefixes) != self.batch_size:
            raise ValueError(f"The KV workspace has {self.batch_size} rows, got {len(prefixes)} request prefixes.")
        for row, prefix in enumerate(prefixes):
            if prefix.prefix_len != self.prefix_len:
                raise ValueError(
                    f"Request prefix {row} has {prefix.prefix_len} tokens, the KV workspace {self.prefix_len}."
                )
            self.key[:, row, :, : self.prefix_len].copy_(prefix.key)
            self.value[:, row, :, : self.prefix_len].copy_(prefix.value)
        return torch.stack([prefix.pad_mask for prefix in prefixes])


def _linear_dtype(linear: nn.Module) -> torch.dtype:
    """The dtype ``linear`` computes in: its weight's, unless quantization stores the weight narrower.

    A quantized vLLM linear (fp8, say) keeps the activation dtype it was built
    for in ``params_dtype``; an unquantized one, and ``nn.Linear``, compute in
    their weight's dtype, which the serving dtype layout may have changed
    after construction.
    """
    weight = linear.weight
    if weight.dtype in (torch.float32, torch.bfloat16, torch.float16):
        return weight.dtype
    return linear.params_dtype


def _match(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Cast ``tensor`` to the dtype ``module`` computes in."""
    dtype = _linear_dtype(module)
    return tensor.to(dtype) if tensor.dtype != dtype else tensor


class Pi05GemmaRotaryEmbedding(nn.Module):
    """Gemma's RoPE angles: the half-split ``(cos, sin)`` table for each position.

    The same ``inv_freq`` as ``transformers``' ``GemmaRotaryEmbedding``, kept a
    buffer so the serving dtype layout casts it as it casts that one. Applying
    the rotation is ``Pi05GemmaAttention.rotary``'s job.
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        exponents = torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim
        self.register_buffer("inv_freq", 1.0 / (base**exponents), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, S)`` positions → float32 ``(B, S, head_dim // 2)`` ``cos`` and ``sin``."""
        freqs = position_ids[..., None].float() * self.inv_freq.float()
        return freqs.cos(), freqs.sin()


class Pi05GemmaAttention(nn.Module):
    """Gemma self-attention projections on vLLM layers: packed Q/K/V, RoPE and ``o_proj``.

    The attention itself is not here: the prefix pass and the suffix pass
    attend over different K/V (``Pi05GemmaDecoderLayer`` /
    ``Pi05ExpertDecoderLayer``). ``quant_config`` reaches the projections; it
    is ``None`` except in the PaliGemma prefix with quantization enabled.
    """

    def __init__(self, config, quant_config: QuantizationConfig | None, prefix: str):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=config.num_attention_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        # Gemma rotates the two halves of each head (NeoX style).
        self.rotary = RotaryEmbedding(is_neox_style=True)

    def project(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(B, S, width)`` → post-RoPE ``q`` ``(B, S, heads, head_dim)`` and ``k``, ``v``
        ``(B, S, kv_heads, head_dim)``. ``cos``/``sin`` are the float32 ``(B, S,
        head_dim // 2)`` tables, rounded here to the projections' output dtype
        as ``transformers`` rounds its table.

        The rotary op shares one table across its batch, while π0.5's
        positions differ per row, so the batch is flattened into the token axis.
        """
        batch_size, seq_len = x.shape[:2]
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        tokens = batch_size * seq_len
        cos = cos.reshape(tokens, -1).to(qkv.dtype)
        sin = sin.reshape(tokens, -1).to(qkv.dtype)
        # The rotary op dispatches on the platform, and its CUDA kernel takes
        # CUDA tensors only; tensors elsewhere rotate natively.
        rotate = self.rotary if x.is_cuda else self.rotary.forward_native
        q = rotate(q.reshape(1, tokens, self.num_heads, self.head_dim), cos, sin)
        k = rotate(k.reshape(1, tokens, self.num_kv_heads, self.head_dim), cos, sin)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        return q, k, v


class Pi05GemmaMLP(nn.Module):
    """GeGLU MLP on vLLM layers: packed gate/up projection, ``gelu_tanh(gate) · up``, ``down_proj``."""

    def __init__(self, config, quant_config: QuantizationConfig | None, prefix: str):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if config.hidden_act != "gelu_pytorch_tanh":
            raise ValueError(f"π0.5's Gemma MLP uses gelu_pytorch_tanh, got {config.hidden_act!r}.")
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        out, _ = self.down_proj(self.act_fn(gate_up))
        return out


class Pi05GemmaDecoderLayer(nn.Module):
    """One PaliGemma language-model layer on the prefix.

    Identical to π0: the prefix backbone is unchanged in π0.5 (no AdaRMS —
    there is no timestep in the prefix).
    """

    def __init__(self, config, quant_config: QuantizationConfig | None, prefix: str, cond_dim: int | None = None):
        super().__init__()
        self.self_attn = Pi05GemmaAttention(config, quant_config, prefix=f"{prefix}.self_attn")
        self.mlp = Pi05GemmaMLP(config, quant_config, prefix=f"{prefix}.mlp")
        if cond_dim is None:
            self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = Pi05AdaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)
            self.post_attention_layernorm = Pi05AdaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the layer output and its post-RoPE ``(k, v)``, ``(B, kv_heads, S, head_dim)``,
        which the suffix pass attends over."""
        attn = self.self_attn
        residual = hidden_states
        x = _match(self.input_layernorm(hidden_states), attn.qkv_proj)

        q, k, v = attn.project(x, cos, sin)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        att = _attend(q.transpose(1, 2), k, v, attention_mask, attn.num_key_value_groups, attn.scaling)
        att = att.transpose(1, 2).reshape(q.shape[0], q.shape[1], attn.q_size)

        out, _ = attn.o_proj(_match(att, attn.o_proj))
        out = out + residual
        after_resid = out
        normed = self.post_attention_layernorm(out)
        out = self.mlp(_match(normed, self.mlp.gate_up_proj)) + after_resid
        return out, k, v


class Pi05ExpertDecoderLayer(Pi05GemmaDecoderLayer):
    """One action-expert layer on the suffix, with AdaRMS conditioning.

    This is where π0.5 diverges from π0. Both norms are
    :class:`Pi05AdaRMSNorm` and each returns a gate that scales its sublayer's
    contribution to the residual stream.
    """

    def __init__(self, config, prefix: str, cond_dim: int):
        super().__init__(config, quant_config=None, prefix=prefix, cond_dim=cond_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        adarms_cond: torch.Tensor,
        key_slots: torch.Tensor,
        value_slots: torch.Tensor,
    ) -> torch.Tensor:
        """``key_slots``/``value_slots`` are this layer's ``[prefix | suffix]`` rows of the
        :class:`Pi05StepKVWorkspace`, ``(B, kv_heads, prefix_len + suffix_len, head_dim)``:
        the suffix K/V is written behind the cached prefix, and the layer attends
        over the whole row rather than over a fresh concatenation."""
        attn = self.self_attn
        prefix_len = key_slots.shape[2] - hidden_states.shape[1]

        residual = hidden_states
        x, gate = self.input_layernorm(hidden_states, adarms_cond)
        x = _match(x, attn.qkv_proj)

        # RoPE frequencies are shared between PaliGemma and the expert.
        q, k_suf, v_suf = attn.project(x, cos, sin)
        key_slots[:, :, prefix_len:].copy_(k_suf.transpose(1, 2))
        value_slots[:, :, prefix_len:].copy_(v_suf.transpose(1, 2))

        att = _attend(
            q.transpose(1, 2), key_slots, value_slots, attention_mask, attn.num_key_value_groups, attn.scaling
        )
        att = att.transpose(1, 2).reshape(q.shape[0], q.shape[1], attn.q_size)

        out, _ = attn.o_proj(_match(att, attn.o_proj))
        hidden_states = _gated_residual(residual, out, gate)

        residual = hidden_states
        x, gate = self.post_attention_layernorm(hidden_states, adarms_cond)
        return _gated_residual(residual, self.mlp(_match(x, self.mlp.gate_up_proj)), gate)


class Pi05GemmaModel(nn.Module):
    """A Gemma tower: the PaliGemma language model, or (``cond_dim`` set) the AdaRMS action expert.

    The module names follow ``transformers``' ``GemmaModel`` so the checkpoint
    keys map one to one, apart from the packed ``qkv_proj`` / ``gate_up_proj``
    that ``Pi05ForActionPrediction.load_weights`` stacks. ``embed_tokens``
    exists in the prefix only; the expert embeds no tokens.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cond_dim: int | None = None,
        embed_tokens: bool = True,
    ):
        super().__init__()
        self.config = config
        if embed_tokens:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            # As ``GemmaTextScaledWordEmbedding`` scales: a buffer, cast with the others.
            self.register_buffer("embed_scale", torch.tensor(config.hidden_size**0.5), persistent=False)
        else:
            self.embed_tokens = None
        if cond_dim is None:
            layers = [
                Pi05GemmaDecoderLayer(config, quant_config, prefix=f"{prefix}.layers.{index}")
                for index in range(config.num_hidden_layers)
            ]
        else:
            layers = [
                Pi05ExpertDecoderLayer(config, prefix=f"{prefix}.layers.{index}", cond_dim=cond_dim)
                for index in range(config.num_hidden_layers)
            ]
        self.layers = nn.ModuleList(layers)
        if cond_dim is None:
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = Pi05AdaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)
        self.rotary_emb = Pi05GemmaRotaryEmbedding(config.head_dim, base=getattr(config, "rope_theta", 10000.0))


class _SiglipSDPA(nn.Module):
    """SigLIP's bidirectional multi-head attention as one SDPA call.

    What vLLM's ``vit_torch_sdpa_wrapper`` runs, called directly rather than
    through its custom op, which is registered for the platform's device only.
    Takes and returns ``(B, S, heads · head_dim)``.
    """

    def __init__(self, num_heads: int, head_size: int, scale: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = query.shape[:2]
        heads = (batch_size, seq_len, self.num_heads, self.head_size)
        q, k, v = (tensor.view(heads).transpose(1, 2) for tensor in (query, key, value))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale)
        return out.transpose(1, 2).reshape(batch_size, seq_len, -1)


def _siglip_vision_tower(vision_config) -> SiglipVisionModel:
    """vLLM's SigLIP, its attention on SDPA.

    SigLIP serves in float32 in both layouts (``_to_bfloat16_for_inference``
    keeps it there, as LeRobot does), and vLLM picks the ViT attention kernel
    for the default dtype at construction: FlashAttention under a bfloat16
    default, which rejects float32, or the Triton kernel, whose TF32 dots move
    the image features by ~2e-3 relative L2 against ``transformers``. SDPA
    matches ``transformers`` to float32 rounding. There is no ``model_config``
    here to carry vLLM's ``mm_encoder_attn_backend`` override, so each layer's
    attention is replaced by :class:`_SiglipSDPA`.
    """
    vision_tower = SiglipVisionModel(vision_config, quant_config=None, prefix="vision_tower")
    for layer in vision_tower.vision_model.encoder.layers:
        attn = layer.self_attn.attn
        layer.self_attn.attn = _SiglipSDPA(attn.num_heads, attn.head_size, attn.scale)
    return vision_tower


class _PaliGemmaProjector(nn.Module):
    """PaliGemma's image projector: SigLIP width → text width. Never quantized."""

    def __init__(self, vision_width: int, text_width: int):
        super().__init__()
        self.linear = nn.Linear(vision_width, text_width, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.linear(image_features)


# ──────────────────────────────────────────────────────────────────────
# Fused Triton kernels: the optimized path of the denoising step
# ──────────────────────────────────────────────────────────────────────
# ``Pi05Pipeline`` enables them together with its CUDA graphs (the stage leaves
# ``enforce_eager`` unset) through ``Pi05ForActionPrediction.enable_fused_kernels``;
# the eager baseline never runs them. The prefix pass runs its eager layers on
# either path, so the prefix K/V every denoising step reads is the same. In the
# action expert, whose GEMMs have only ``chunk_size`` rows, they replace whole
# layers and the output head: in bfloat16 including the GEMMs, in float32
# around cuBLAS GEMMs. They read every weight in place, the packed vLLM
# ``qkv_proj`` / ``gate_up_proj`` weights as row blocks, and write K/V straight
# into the ``Pi05StepKVWorkspace`` slots the eager path copies into.
#
# Numerics: each kernel performs the eager path's operations in its order, in
# its dtypes and with its rounding points. A value eager materializes in a
# narrower dtype is rounded there too (``_round``), and kernels launch with
# ``enable_fp_fusion=False`` so no multiply-add is contracted into an FMA that
# eager does not have (the one it has, in torch's GELU, is spelled out). What
# still differs is the order of reductions (GEMM accumulation, the RMS variance
# and the softmax sums) and in RoPE, where the eager rotary kernel may contract
# its products into FMAs, the last float32 bit. The kernels differ from eager
# by that rounding only.
# ``tests/diffusion/models/pi05/test_pi05_fused_kernels.py`` holds each kernel
# to that.

if HAS_TRITON:
    _TL_DTYPES = {torch.float32: tl.float32, torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}

    @triton.jit
    def _round(x, dtype: tl.constexpr):
        """Round ``x`` to ``dtype`` where eager materializes it; continue in float32."""
        return x.to(dtype).to(tl.float32)

    @triton.jit
    def _gelu_tanh(x):
        """torch's CUDA ``gelu(approximate="tanh")`` in float32, including its FMA."""
        x_cube = x * x * x
        inner = 0.7978845608028654 * tl.fma(0.044715, x_cube, x)
        return 0.5 * x * (1.0 + tldevice.tanh(inner))

    @triton.jit
    def _rms_norm_kernel(
        X,
        ADD,
        RES_OUT,
        WEIGHT,
        MOD,
        OUT,
        n_cols,
        x_stride,
        add_stride,
        res_stride,
        out_stride,
        eps,
        inv_n_cols,
        NORM_DTYPE: tl.constexpr,
        HAS_ADD: tl.constexpr,
        STORE_RES: tl.constexpr,
        ADARMS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One row of ``[residual add +] Gemma RMSNorm | AdaRMS``.

        With ``HAS_ADD`` the row is ``ADD + X`` first (eager's ``sublayer(x) +
        residual``), stored to ``RES_OUT`` with ``STORE_RES``. The norm is
        ``Pi05AdaRMSNorm`` (``x̂ · (1 + scale) + shift``, scale and shift read from
        its ``dense`` output ``MOD``) with ``ADARMS``, else ``GemmaRMSNorm``
        (``x̂ · (1 + weight)``). Its result is rounded to ``NORM_DTYPE``, the
        norm input's dtype, and stored in ``OUT``'s, the next GEMM's.
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(X + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_ADD:
            add = tl.load(ADD + row * add_stride + cols, mask=mask, other=0.0).to(tl.float32)
            x = _round(add + x, NORM_DTYPE)
            if STORE_RES:
                tl.store(RES_OUT + row * res_stride + cols, x, mask=mask)
        variance = tl.sum(x * x, axis=0) * inv_n_cols
        normed = x * tldevice.rsqrt(variance + eps)
        if ADARMS:
            scale = tl.load(MOD + cols, mask=mask, other=0.0).to(tl.float32)
            shift = tl.load(MOD + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
            y = normed * (1.0 + scale) + shift
        else:
            weight = tl.load(WEIGHT + cols, mask=mask, other=0.0).to(tl.float32)
            y = normed * (1.0 + weight)
        tl.store(OUT + row * out_stride + cols, _round(y, NORM_DTYPE), mask=mask)

    @triton.jit
    def _qkv_rope_kernel(
        X,
        WQ,
        WK,
        WV,
        COS,
        SIN,
        Q_OUT,
        K_OUT,
        V_OUT,
        n_rows,
        n_in,
        x_stride,
        proj_stride,
        rope_stride,
        kv_stride,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        PROJ_DTYPE: tl.constexpr,
        FUSED_GEMM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Q/K/V projections of one head's column tile, Gemma's half-split RoPE on Q and K.

        Program ``(m, n)`` covers rows ``m`` of head ``n // SPLITS`` (``NUM_HEADS``
        query heads, then the key head, then the value head) and one
        ``BLOCK_H`` tile of each half of it, which RoPE pairs. Q is stored
        ``(NUM_HEADS, n_rows, HEAD_DIM)``, K and V to ``(n_rows, HEAD_DIM)`` rows
        ``kv_stride`` apart: their ``Pi05StepKVWorkspace`` slots. ``COS`` and
        ``SIN`` are the half-split ``(n_rows, HEAD_DIM // 2)`` tables.

        With ``FUSED_GEMM`` the projections are computed here from ``X`` and the
        row blocks ``WQ``/``WK``/``WV`` of the packed ``qkv_proj`` weight.
        Without it ``WQ``/``WK``/``WV`` point at the Q, K and V columns of the
        packed projection's output, rows ``proj_stride`` apart, and only RoPE
        and the layout run here.
        """
        HALF: tl.constexpr = HEAD_DIM // 2
        SPLITS: tl.constexpr = HALF // BLOCK_H
        head = tl.program_id(1) // SPLITS
        cols = (tl.program_id(1) % SPLITS) * BLOCK_H + tl.arange(0, BLOCK_H)
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows[:, None] < n_rows
        if FUSED_GEMM:
            if head < NUM_HEADS:
                w = WQ + head * HEAD_DIM * n_in
            elif head == NUM_HEADS:
                w = WK
            else:
                w = WV
            acc_lo = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
            acc_hi = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
            for k0 in range(0, n_in, BLOCK_K):
                ks = k0 + tl.arange(0, BLOCK_K)
                k_mask = ks < n_in
                x = tl.load(X + rows[:, None] * x_stride + ks[None, :], mask=row_mask & k_mask[None, :], other=0.0)
                w_lo = tl.load(w + cols[None, :] * n_in + ks[:, None], mask=k_mask[:, None], other=0.0)
                w_hi = tl.load(w + (cols[None, :] + HALF) * n_in + ks[:, None], mask=k_mask[:, None], other=0.0)
                acc_lo = tl.dot(x, w_lo, acc_lo)
                acc_hi = tl.dot(x, w_hi, acc_hi)
            # The projections' outputs, as the eager qkv_proj returns them.
            lo = _round(acc_lo, PROJ_DTYPE)
            hi = _round(acc_hi, PROJ_DTYPE)
        else:
            if head < NUM_HEADS:
                proj = WQ + head * HEAD_DIM + rows[:, None] * proj_stride + cols[None, :]
            elif head == NUM_HEADS:
                proj = WK + rows[:, None] * proj_stride + cols[None, :]
            else:
                proj = WV + rows[:, None] * proj_stride + cols[None, :]
            lo = tl.load(proj, mask=row_mask, other=0.0).to(tl.float32)
            hi = tl.load(proj + HALF, mask=row_mask, other=0.0).to(tl.float32)

        out_lo = rows[:, None] * HEAD_DIM + cols[None, :]
        if head <= NUM_HEADS:
            # The rotary op eager runs (``Pi05GemmaAttention.project``):
            # computed in float32 from the rounded table, rounded once.
            rope = rows[:, None] * rope_stride + cols[None, :]
            cos = tl.load(COS + rope, mask=row_mask, other=0.0).to(tl.float32)
            sin = tl.load(SIN + rope, mask=row_mask, other=0.0).to(tl.float32)
            rot_lo = _round(lo * cos - hi * sin, PROJ_DTYPE)
            rot_hi = _round(hi * cos + lo * sin, PROJ_DTYPE)
            if head < NUM_HEADS:
                q_out = Q_OUT + head * n_rows * HEAD_DIM + out_lo
                tl.store(q_out, rot_lo, mask=row_mask)
                tl.store(q_out + HALF, rot_hi, mask=row_mask)
            else:
                k_out = K_OUT + rows[:, None] * kv_stride + cols[None, :]
                tl.store(k_out, rot_lo, mask=row_mask)
                tl.store(k_out + HALF, rot_hi, mask=row_mask)
        else:
            v_out = V_OUT + rows[:, None] * kv_stride + cols[None, :]
            tl.store(v_out, lo, mask=row_mask)
            tl.store(v_out + HALF, hi, mask=row_mask)

    @triton.jit
    def _attn_scores_kernel(
        Q,
        K,
        S_OUT,
        n_rows,
        n_keys,
        k_stride,
        s_stride,
        scale,
        HEAD_DIM: tl.constexpr,
        SCORE_DTYPE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """``(Q Kᵀ) · scale`` for single-KV-head attention, heads flattened into rows."""
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        keys = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows[:, None] < n_rows
        key_mask = keys[None, :] < n_keys
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d0 in range(0, HEAD_DIM, BLOCK_K):
            ds = d0 + tl.arange(0, BLOCK_K)
            q = tl.load(Q + rows[:, None] * HEAD_DIM + ds[None, :], mask=row_mask, other=0.0)
            k = tl.load(K + keys[None, :] * k_stride + ds[:, None], mask=key_mask, other=0.0)
            acc = tl.dot(q, k, acc)
        scores = _round(_round(acc, SCORE_DTYPE) * scale, SCORE_DTYPE)
        tl.store(S_OUT + rows[:, None] * s_stride + keys[None, :], scores, mask=row_mask & key_mask)

    @triton.jit
    def _masked_softmax_kernel(
        S,
        MASK,
        n_keys,
        q_len,
        s_stride,
        mask_stride,
        scale,
        SCORE_DTYPE: tl.constexpr,
        SUM_DTYPE: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``softmax([scores · scale] + mask)`` over one row, in float32, written back in place.

        Row ``r`` is query token ``r % q_len`` of head ``r // q_len``; ``MASK`` is
        the eager path's float attention mask, one row per query token.
        ``HAS_SCALE`` applies the score scale here, for scores a GEMM left
        unscaled.
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        valid = cols < n_keys
        scores = tl.load(S + row * s_stride + cols, mask=valid, other=0.0).to(tl.float32)
        if HAS_SCALE:
            scores = _round(scores * scale, SCORE_DTYPE)
        bias = tl.load(MASK + (row % q_len) * mask_stride + cols, mask=valid, other=0.0).to(tl.float32)
        x = tl.where(valid, _round(scores + bias, SUM_DTYPE), -float("inf"))
        e = tldevice.exp(x - tl.max(x, axis=0))
        probs = tl.math.div_rn(e, tl.sum(e, axis=0))
        tl.store(S + row * s_stride + cols, probs, mask=valid)

    @triton.jit
    def _attn_values_kernel(
        P,
        V,
        OUT,
        n_rows,
        n_keys,
        q_len,
        p_stride,
        v_stride,
        out_stride,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """``P V`` stored as ``(q_len, heads · HEAD_DIM)``, the layout ``o_proj`` reads."""
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        ds = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        row_mask = rows[:, None] < n_rows
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for k0 in range(0, n_keys, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            k_mask = ks < n_keys
            p = tl.load(P + rows[:, None] * p_stride + ks[None, :], mask=row_mask & k_mask[None, :], other=0.0)
            v = tl.load(V + ks[:, None] * v_stride + ds[None, :], mask=k_mask[:, None], other=0.0)
            acc = tl.dot(p, v, acc)
        out = OUT + (rows % q_len)[:, None] * out_stride + (rows // q_len)[:, None] * HEAD_DIM + ds[None, :]
        tl.store(out, acc, mask=row_mask)

    @triton.jit
    def _linear_residual_kernel(
        X,
        W,
        RES,
        GATE,
        OUT,
        n_rows,
        n_out,
        n_in,
        x_stride,
        res_stride,
        out_stride,
        PROJ_DTYPE: tl.constexpr,
        RES_DTYPE: tl.constexpr,
        HAS_GATE: tl.constexpr,
        FUSED_GEMM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """``RES + [GATE ·] (X Wᵀ)``: a bias-free ``nn.Linear`` and eager's ``_gated_residual``.

        Without ``FUSED_GEMM``, ``X`` is the linear's output already (rows
        ``x_stride`` apart) and only the residual runs here.
        """
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows[:, None] < n_rows
        col_mask = cols < n_out
        mask = row_mask & col_mask[None, :]
        if FUSED_GEMM:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, n_in, BLOCK_K):
                ks = k0 + tl.arange(0, BLOCK_K)
                k_mask = ks < n_in
                x = tl.load(X + rows[:, None] * x_stride + ks[None, :], mask=row_mask & k_mask[None, :], other=0.0)
                w = tl.load(W + cols[None, :] * n_in + ks[:, None], mask=col_mask[None, :] & k_mask[:, None], other=0.0)
                acc = tl.dot(x, w, acc)
            out = _round(acc, PROJ_DTYPE)
        else:
            out = tl.load(X + rows[:, None] * x_stride + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        res = tl.load(RES + rows[:, None] * res_stride + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        if HAS_GATE:
            gate = _round(tl.load(GATE + cols, mask=col_mask, other=0.0).to(tl.float32), RES_DTYPE)
            out = _round(gate[None, :] * out, RES_DTYPE)
        tl.store(OUT + rows[:, None] * out_stride + cols[None, :], _round(res + out, RES_DTYPE), mask=mask)

    @triton.jit
    def _linear_gelu_mul_kernel(
        X,
        W_GATE,
        W_UP,
        OUT,
        n_rows,
        n_out,
        n_in,
        x_stride,
        out_stride,
        PROJ_DTYPE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """``gelu_tanh(X W_gateᵀ) · (X W_upᵀ)``: the ``GemmaMLP`` input half."""
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows[:, None] < n_rows
        col_mask = cols < n_out
        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, n_in, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            k_mask = ks < n_in
            x = tl.load(X + rows[:, None] * x_stride + ks[None, :], mask=row_mask & k_mask[None, :], other=0.0)
            w_mask = col_mask[None, :] & k_mask[:, None]
            w_gate = tl.load(W_GATE + cols[None, :] * n_in + ks[:, None], mask=w_mask, other=0.0)
            w_up = tl.load(W_UP + cols[None, :] * n_in + ks[:, None], mask=w_mask, other=0.0)
            acc_gate = tl.dot(x, w_gate, acc_gate)
            acc_up = tl.dot(x, w_up, acc_up)
        act = _round(_gelu_tanh(_round(acc_gate, PROJ_DTYPE)), PROJ_DTYPE)
        out = act * _round(acc_up, PROJ_DTYPE)
        tl.store(OUT + rows[:, None] * out_stride + cols[None, :], out, mask=row_mask & col_mask[None, :])

    @triton.jit
    def _final_head_kernel(
        X,
        MOD,
        W,
        B,
        OUT,
        n_cols,
        n_out,
        x_stride,
        out_stride,
        eps,
        inv_n_cols,
        NORM_DTYPE: tl.constexpr,
        HEAD_DTYPE: tl.constexpr,
        BLOCK: tl.constexpr,
        BLOCK_O: tl.constexpr,
    ):
        """One row of the expert's final AdaRMS norm followed by ``action_out_proj``."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(X + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) * inv_n_cols
        normed = x * tldevice.rsqrt(variance + eps)
        scale = tl.load(MOD + cols, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(MOD + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        y = _round(_round(normed * (1.0 + scale) + shift, NORM_DTYPE), HEAD_DTYPE)
        for o0 in range(0, n_out, BLOCK_O):
            outs = o0 + tl.arange(0, BLOCK_O)
            out_mask = outs < n_out
            w = tl.load(W + outs[:, None] * n_cols + cols[None, :], mask=out_mask[:, None] & mask[None, :], other=0.0)
            acc = tl.sum(y[None, :] * w.to(tl.float32), axis=1)
            bias = tl.load(B + outs, mask=out_mask, other=0.0).to(tl.float32)
            tl.store(OUT + row * out_stride + outs, acc + bias, mask=out_mask)


@dataclass(frozen=True)
class _Tiles:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int = 4
    num_stages: int = 3


# In the action expert, GEMMs run in Triton, with their epilogues fused in, only
# for 16-bit weights: Triton's IEEE float32 ``tl.dot`` runs on FMA units at about
# a third of cuBLAS's SIMT speed on these shapes. float32 keeps cuBLAS for every
# GEMM and runs the same epilogues after it (``FUSED_GEMM=False``), as the prefix
# does in either dtype.
_FUSED_DTYPES = (torch.float32, torch.bfloat16)
_TRITON_GEMM_DTYPES = (torch.bfloat16,)

# Tile sizes of the Triton GEMMs, which only the action expert runs, measured on
# its deployed ``chunk_size`` rows. They serve any chunk length: the kernels mask
# the rows a tile overhangs. Fixed rather than autotuned, so a restart cannot
# pick other ones and change the reduction order.
_TILES: dict[tuple[str, torch.dtype], _Tiles] = {
    # (kernel, dtype): tiles. ``block_n`` is the RoPE half-tile for ``qkv`` and
    # the head-dim tile for ``values``.
    ("qkv", torch.bfloat16): _Tiles(64, 16, 128),
    ("scores", torch.bfloat16): _Tiles(32, 32, 128, num_warps=8),
    ("values", torch.bfloat16): _Tiles(16, 64, 64, num_warps=8),
    ("linear", torch.bfloat16): _Tiles(32, 32, 256, num_stages=4),
    ("gelu_linear", torch.bfloat16): _Tiles(64, 32, 64),
}
# The elementwise epilogues that follow a cuBLAS GEMM: rows × columns.
_EPILOGUE_TILES = _Tiles(32, 64, 0)


def _triton_gemms(dtype: torch.dtype) -> bool:
    return dtype in _TRITON_GEMM_DTYPES


def _tiles(kernel: str, dtype: torch.dtype) -> _Tiles:
    return _TILES[(kernel, dtype)] if _triton_gemms(dtype) else _EPILOGUE_TILES


def _fused_rms_norm(
    x: torch.Tensor,
    norm: nn.Module,
    out: torch.Tensor,
    *,
    add: torch.Tensor | None = None,
    res_out: torch.Tensor | None = None,
    modulation: torch.Tensor | None = None,
) -> None:
    """``out = norm([add +] x)`` for 2D rows; ``res_out`` receives the sum."""
    n_rows, n_cols = x.shape
    norm_dtype = torch.promote_types(x.dtype, add.dtype) if add is not None else x.dtype
    adarms = modulation is not None
    eps = norm.eps if adarms else norm.variance_epsilon
    _rms_norm_kernel[(n_rows,)](
        x,
        add if add is not None else x,
        res_out if res_out is not None else x,
        x if adarms else norm.weight,
        modulation if adarms else x,
        out,
        n_cols,
        x.stride(0),
        add.stride(0) if add is not None else 0,
        res_out.stride(0) if res_out is not None else 0,
        out.stride(0),
        eps,
        1.0 / n_cols,
        NORM_DTYPE=_TL_DTYPES[norm_dtype],
        HAS_ADD=add is not None,
        STORE_RES=res_out is not None,
        ADARMS=adarms,
        BLOCK=triton.next_power_of_2(n_cols),
        num_warps=8 if n_cols >= 2048 else 4,
        enable_fp_fusion=False,
    )


def _fused_qkv_rope(
    x: torch.Tensor,
    attn: Pi05GemmaAttention,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
) -> None:
    """The packed Q/K/V projection of ``x`` with RoPE: Q into ``q_out``, K and V into their workspace slots.

    ``cos``/``sin`` are the half-split ``(rows, head_dim // 2)`` tables in the
    projection's dtype. In float32 the projection stays on cuBLAS, as eager runs it.
    """
    n_rows, n_in = x.shape
    num_heads, _, head_dim = q_out.shape
    dtype = _linear_dtype(attn.qkv_proj)
    fused_gemm = _triton_gemms(dtype)
    if fused_gemm:
        # Row blocks of the packed weight: contiguous, so the kernel reads them in place.
        weights = attn.qkv_proj.weight.split([attn.q_size, attn.kv_size, attn.kv_size], dim=0)
        proj_stride = 0
    else:
        qkv = attn.qkv_proj(x[None])[0][0]
        weights = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        proj_stride = qkv.stride(0)
    tiles = _tiles("qkv", dtype)
    splits = head_dim // 2 // tiles.block_n
    _qkv_rope_kernel[(triton.cdiv(n_rows, tiles.block_m), (num_heads + 2) * splits)](
        x,
        *weights,
        cos,
        sin,
        q_out,
        k_out,
        v_out,
        n_rows,
        n_in,
        x.stride(0),
        proj_stride,
        cos.stride(0),
        k_out.stride(0),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        PROJ_DTYPE=_TL_DTYPES[dtype],
        FUSED_GEMM=fused_gemm,
        BLOCK_M=tiles.block_m,
        BLOCK_H=tiles.block_n,
        BLOCK_K=tiles.block_k,
        num_warps=tiles.num_warps,
        num_stages=tiles.num_stages,
        enable_fp_fusion=False,
    )


def _fused_attention(
    q: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
    scores: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Eager ``_attend`` for one KV head, into ``out`` as ``(q_len, heads · head_dim)``.

    ``q`` is ``(heads, q_len, head_dim)``, ``key``/``value`` are ``(n_keys,
    head_dim)``, ``attention_mask`` is ``(q_len, ≥ n_keys)`` and ``scores`` is
    ``(heads · q_len, n_keys)`` scratch, which the softmax overwrites.
    """
    num_heads, q_len, head_dim = q.shape
    n_rows, n_keys = num_heads * q_len, key.shape[0]
    dtype = q.dtype
    fused_gemm = _triton_gemms(dtype)

    if fused_gemm:
        tiles = _tiles("scores", dtype)
        _attn_scores_kernel[(triton.cdiv(n_rows, tiles.block_m), triton.cdiv(n_keys, tiles.block_n))](
            q,
            key,
            scores,
            n_rows,
            n_keys,
            key.stride(0),
            scores.stride(0),
            scaling,
            HEAD_DIM=head_dim,
            SCORE_DTYPE=_TL_DTYPES[dtype],
            BLOCK_M=tiles.block_m,
            BLOCK_N=tiles.block_n,
            BLOCK_K=tiles.block_k,
            num_warps=tiles.num_warps,
            num_stages=tiles.num_stages,
            enable_fp_fusion=False,
        )
    else:
        torch.matmul(q.view(n_rows, head_dim), key.t(), out=scores)
    _masked_softmax_kernel[(n_rows,)](
        scores,
        attention_mask,
        n_keys,
        q_len,
        scores.stride(0),
        attention_mask.stride(0),
        scaling,
        SCORE_DTYPE=_TL_DTYPES[dtype],
        SUM_DTYPE=_TL_DTYPES[torch.promote_types(dtype, attention_mask.dtype)],
        HAS_SCALE=not fused_gemm,
        BLOCK=triton.next_power_of_2(n_keys),
        num_warps=4,
        enable_fp_fusion=False,
    )
    if fused_gemm:
        tiles = _tiles("values", dtype)
        _attn_values_kernel[(triton.cdiv(n_rows, tiles.block_m), triton.cdiv(head_dim, tiles.block_n))](
            scores,
            value,
            out,
            n_rows,
            n_keys,
            q_len,
            scores.stride(0),
            value.stride(0),
            out.stride(0),
            HEAD_DIM=head_dim,
            BLOCK_M=tiles.block_m,
            BLOCK_D=tiles.block_n,
            BLOCK_K=tiles.block_k,
            num_warps=tiles.num_warps,
            num_stages=tiles.num_stages,
            enable_fp_fusion=False,
        )
    else:
        values = torch.matmul(scores, value).view(num_heads, q_len, head_dim)
        out.view(q_len, num_heads, head_dim).copy_(values.transpose(0, 1))


def _fused_linear_residual(
    x: torch.Tensor,
    linear: RowParallelLinear,
    residual: torch.Tensor,
    out: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> None:
    """``out = residual + [gate ·] linear(x)``; ``out`` may be ``residual``."""
    n_rows, n_in = x.shape
    n_out = linear.weight.shape[0]
    dtype = linear.weight.dtype
    fused_gemm = _triton_gemms(dtype)
    if not fused_gemm:
        x = linear(x[None])[0][0]
    tiles = _tiles("linear", dtype)
    _linear_residual_kernel[(triton.cdiv(n_rows, tiles.block_m), triton.cdiv(n_out, tiles.block_n))](
        x,
        linear.weight,
        residual,
        gate if gate is not None else residual,
        out,
        n_rows,
        n_out,
        n_in,
        x.stride(0),
        residual.stride(0),
        out.stride(0),
        PROJ_DTYPE=_TL_DTYPES[dtype],
        RES_DTYPE=_TL_DTYPES[out.dtype],
        HAS_GATE=gate is not None,
        FUSED_GEMM=fused_gemm,
        BLOCK_M=tiles.block_m,
        BLOCK_N=tiles.block_n,
        BLOCK_K=tiles.block_k,
        num_warps=tiles.num_warps,
        num_stages=tiles.num_stages,
        enable_fp_fusion=False,
    )


def _fused_linear_gelu_mul(x: torch.Tensor, mlp: Pi05GemmaMLP, out: torch.Tensor) -> None:
    """``out = gelu_tanh(gate) · up`` of the packed ``gate_up_proj(x)``: the MLP input half.

    In float32 the GEMM stays on cuBLAS and the activation is the eager
    ``GeluAndMul`` itself.
    """
    n_rows, n_in = x.shape
    gate_up = mlp.gate_up_proj
    dtype = gate_up.weight.dtype
    if not _triton_gemms(dtype):
        out.copy_(mlp.act_fn(gate_up(x[None])[0][0]))
        return
    n_out = gate_up.weight.shape[0] // 2
    # Row blocks of the packed weight: contiguous, so the kernel reads them in place.
    w_gate, w_up = gate_up.weight.split([n_out, n_out], dim=0)
    tiles = _tiles("gelu_linear", dtype)
    _linear_gelu_mul_kernel[(triton.cdiv(n_rows, tiles.block_m), triton.cdiv(n_out, tiles.block_n))](
        x,
        w_gate,
        w_up,
        out,
        n_rows,
        n_out,
        n_in,
        x.stride(0),
        out.stride(0),
        PROJ_DTYPE=_TL_DTYPES[dtype],
        BLOCK_M=tiles.block_m,
        BLOCK_N=tiles.block_n,
        BLOCK_K=tiles.block_k,
        num_warps=tiles.num_warps,
        num_stages=tiles.num_stages,
        enable_fp_fusion=False,
    )


def _fused_final_head(
    x: torch.Tensor, norm: nn.Module, modulation: torch.Tensor, head: nn.Linear, out: torch.Tensor
) -> None:
    n_rows, n_cols = x.shape
    _final_head_kernel[(n_rows,)](
        x,
        modulation,
        head.weight,
        head.bias,
        out,
        n_cols,
        head.out_features,
        x.stride(0),
        out.stride(0),
        norm.eps,
        1.0 / n_cols,
        NORM_DTYPE=_TL_DTYPES[x.dtype],
        HEAD_DTYPE=_TL_DTYPES[head.weight.dtype],
        BLOCK=triton.next_power_of_2(n_cols),
        BLOCK_O=8,
        num_warps=8,
        enable_fp_fusion=False,
    )


def _adarms_modulation(norm: Pi05AdaRMSNorm, adarms_cond: torch.Tensor) -> torch.Tensor:
    """``Pi05AdaRMSNorm``'s ``(scale | shift | gate)`` for batch 1, computed as it computes it."""
    return norm.dense(adarms_cond.to(norm.dense.weight.dtype))[0]


def _fused_denoise_forward(
    gemma_expert: nn.Module,
    suffix_embs: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    kv_workspace: Pi05StepKVWorkspace,
    adarms_cond: torch.Tensor,
    action_out_proj: nn.Linear,
) -> torch.Tensor:
    """The suffix pass and output head of ``denoise_step``, batch 1, with fused kernels.

    Per layer: AdaRMS norm, QKV projection + RoPE writing the suffix K/V
    behind the cached prefix, attention over the whole workspace row, and the
    ``o_proj`` and MLP GEMMs with their gated residuals and GELU; then the final
    AdaRMS norm fused with ``action_out_proj``. In bfloat16 the GEMMs run in
    Triton with what follows them fused in, reading the packed vLLM weights in
    place; in float32 they stay on cuBLAS and only what follows them is fused.
    Returns ``v_t``.
    """
    expert = gemma_expert.model
    x_in = suffix_embs[0]
    seq_len, width = x_in.shape
    first_attn = expert.layers[0].self_attn
    proj_dtype = _linear_dtype(first_attn.qkv_proj)
    num_heads = first_attn.num_heads
    head_dim = first_attn.head_dim
    prefix_len = kv_workspace.prefix_len
    n_keys = prefix_len + seq_len
    device = x_in.device

    hidden_states = torch.empty(seq_len, width, dtype=torch.promote_types(x_in.dtype, proj_dtype), device=device)
    x_normed = torch.empty(seq_len, width, dtype=proj_dtype, device=device)
    q = torch.empty(num_heads, seq_len, head_dim, dtype=proj_dtype, device=device)
    scores = torch.empty(num_heads * seq_len, n_keys, dtype=proj_dtype, device=device)
    att = torch.empty(seq_len, num_heads * head_dim, dtype=proj_dtype, device=device)
    mlp_hidden = torch.empty(seq_len, expert.config.intermediate_size, dtype=proj_dtype, device=device)
    cos, sin = expert.rotary_emb(position_ids)
    cos, sin = cos[0].to(proj_dtype), sin[0].to(proj_dtype)
    mask = attention_mask[0, 0]

    residual = x_in
    for layer_idx, layer in enumerate(expert.layers):
        attn = layer.self_attn
        key, value = kv_workspace.key[layer_idx, 0, 0], kv_workspace.value[layer_idx, 0, 0]

        modulation = _adarms_modulation(layer.input_layernorm, adarms_cond)
        _fused_rms_norm(residual, layer.input_layernorm, x_normed, modulation=modulation)
        _fused_qkv_rope(x_normed, attn, cos, sin, q, key[prefix_len:], value[prefix_len:])
        _fused_attention(q, key, value, mask, attn.scaling, scores, att)
        _fused_linear_residual(att, attn.o_proj, residual, hidden_states, gate=modulation[2 * width :])
        residual = hidden_states

        modulation = _adarms_modulation(layer.post_attention_layernorm, adarms_cond)
        _fused_rms_norm(hidden_states, layer.post_attention_layernorm, x_normed, modulation=modulation)
        _fused_linear_gelu_mul(x_normed, layer.mlp, mlp_hidden)
        _fused_linear_residual(
            mlp_hidden, layer.mlp.down_proj, hidden_states, hidden_states, gate=modulation[2 * width :]
        )

    v_t = torch.empty(1, seq_len, action_out_proj.out_features, dtype=action_out_proj.weight.dtype, device=device)
    _fused_final_head(hidden_states, expert.norm, _adarms_modulation(expert.norm, adarms_cond), action_out_proj, v_t[0])
    return v_t


def _fused_kernels_serve(backbone: PaliGemmaWithActionExpertPi05, kv_workspace, batch_size: int) -> bool:
    """Whether a denoising step runs the fused kernels: they are enabled, and it is batch 1
    on a single-KV-head ``Pi05StepKVWorkspace`` in the action expert's K/V dtype. Any
    other call runs eagerly, which also keeps eager's errors for bad workspaces."""
    return (
        backbone.fused_kernels
        and batch_size == 1
        and isinstance(kv_workspace, Pi05StepKVWorkspace)
        and kv_workspace.batch_size == 1
        and kv_workspace.key.shape[2] == 1
        and kv_workspace.key.dtype == backbone.gemma_expert.model.layers[0].self_attn.qkv_proj.weight.dtype
    )


class _Container(nn.Module):
    """A plain module holder, so the module paths stay the checkpoint's
    (``paligemma.model.language_model.*``, ``gemma_expert.model.*``)."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config


class PaliGemmaWithActionExpertPi05(nn.Module):
    """Dual-backbone transformer: PaliGemma (Gemma 2B) + AdaRMS expert (300M).

    Same two-mode dispatch as π0 (``prefix_only`` / ``suffix_only``). Both
    Gemma towers are :class:`Pi05GemmaModel`, on vLLM's packed linear layers,
    ``GemmaRMSNorm`` and ``GeluAndMul``; the vision tower is vLLM's
    ``SiglipVisionModel``. The one structural difference between the towers is
    π0.5's: every action-expert norm is a :class:`Pi05AdaRMSNorm` carrying a
    ``dense`` conditioning projection. The module paths follow the
    ``transformers`` PaliGemma/Gemma layout the checkpoint was saved from.

    ``quant_config`` reaches the PaliGemma language model only: SigLIP, the
    projector and the action expert always run in the serving dtype.
    """

    def __init__(self, vlm_config, action_expert_config, quant_config: QuantizationConfig | None = None):
        super().__init__()

        # PaliGemma prefix: identical to π0. It sees no timestep, so no AdaRMS.
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.hidden_act = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            hidden_act="gelu_pytorch_tanh",
            dtype="float32",
        )

        self.adarms_cond_dim = action_expert_config.width

        self.paligemma = _Container(vlm_config_hf)
        self.paligemma.model = _Container()
        vision_config = vlm_config_hf.vision_config
        self.paligemma.model.vision_tower = _siglip_vision_tower(vision_config)
        self.paligemma.model.multi_modal_projector = _PaliGemmaProjector(
            vision_config.hidden_size, vlm_config_hf.text_config.hidden_size
        )
        self.paligemma.model.language_model = Pi05GemmaModel(
            vlm_config_hf.text_config, quant_config=quant_config, prefix="language_model"
        )

        # The action expert doesn't embed tokens — it only consumes the
        # suffix action embeddings we feed in — and has no LM head.
        self.gemma_expert = _Container(action_expert_config_hf)
        self.gemma_expert.model = Pi05GemmaModel(
            action_expert_config_hf, prefix="gemma_expert", cond_dim=self.adarms_cond_dim, embed_tokens=False
        )
        # Set by ``Pi05ForActionPrediction.enable_fused_kernels`` on the
        # optimized path; ``False`` is the eager baseline.
        self.fused_kernels = False

    def embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images with SigLIP vision tower + PaliGemma projector.

        The projector output is not divided by ``sqrt(text_hidden_size)`` as
        ``PaliGemmaModel.get_image_features`` divides it: π0 and π0.5 feed it
        unscaled (SigLIP is unchanged in π0.5).
        """
        # Shapes: pixel_values (B, 3, 224, 224) → SigLIP (B, 256, 1152)
        #                                      → projector (B, 256, 2048)
        image_features = self.paligemma.model.vision_tower(pixel_values)
        return self.paligemma.model.multi_modal_projector(image_features)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed language tokens, returning the ``* sqrt(hidden)``-scaled embedding,
        scaled as ``GemmaTextScaledWordEmbedding`` scales it."""
        language_model = self.paligemma.model.language_model
        embed_tokens = language_model.embed_tokens
        return embed_tokens(tokens) * language_model.embed_scale.to(embed_tokens.weight.dtype)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: list[torch.Tensor | None] | None = None,
        use_cache: bool = False,
        adarms_cond: torch.Tensor | None = None,
    ):
        """Dispatch to prefix_only / suffix_only and return
        ``([prefix_out, suffix_out], past_key_values_or_None)``.

        ``past_key_values`` is a :class:`Pi05StepKVWorkspace` in both modes:
        prefix_only with ``use_cache`` writes the prefix K/V into it and returns
        it, and suffix_only reads it. The prefix pass always runs these eager
        layers, also on the optimized path, so its K/V is the same either way.
        """
        pali_lm = self.paligemma.model.language_model
        expert_lm = self.gemma_expert.model

        if (use_cache or inputs_embeds[1] is not None) and not isinstance(past_key_values, Pi05StepKVWorkspace):
            raise TypeError(
                "PaliGemmaWithActionExpertPi05.forward expects past_key_values to be "
                f"a Pi05StepKVWorkspace to write (prefix_only) or read (suffix_only); got {type(past_key_values)}"
            )

        if inputs_embeds[1] is None:
            hidden_states = inputs_embeds[0]
            cos, sin = pali_lm.rotary_emb(position_ids)
            for layer_idx, layer in enumerate(pali_lm.layers):
                hidden_states, k, v = layer(hidden_states, attention_mask, cos, sin)
                if use_cache:
                    past_key_values.write_prefix(layer_idx, k, v)
            hidden_states = pali_lm.norm(hidden_states)
            return [hidden_states, None], (past_key_values if use_cache else None)

        if inputs_embeds[0] is not None:
            raise ValueError(
                "PaliGemmaWithActionExpertPi05.forward only supports prefix-only "
                "or suffix-only dispatch; got both inputs_embeds populated."
            )
        hidden_states = inputs_embeds[1]
        suffix_dtype = _linear_dtype(expert_lm.layers[0].self_attn.qkv_proj)
        if past_key_values.key.dtype != suffix_dtype:
            raise TypeError(f"Suffix K/V is {suffix_dtype}, but the KV workspace holds {past_key_values.key.dtype}.")
        cos, sin = expert_lm.rotary_emb(position_ids)
        for layer_idx, layer in enumerate(expert_lm.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask,
                cos,
                sin,
                adarms_cond,
                past_key_values.key[layer_idx],
                past_key_values.value[layer_idx],
            )
        hidden_states, _ = expert_lm.norm(hidden_states, adarms_cond)
        return [None, hidden_states], None


# ──────────────────────────────────────────────────────────────────────
# Main π0.5 Model
# ──────────────────────────────────────────────────────────────────────
class Pi05ForActionPrediction(nn.Module):
    """π0.5 VLA model for robot action prediction via flow matching.

    Inference flow:
      1. Embed prefix (images + language, where the language already carries the
         discretized state) → prefix tokens.
      2. Forward prefix through PaliGemma → per-layer prefix K/V in the step KV workspace.
      3. For each denoising step ``t = 1.0, 1-dt, ..., 0``:
         a. Embed the timestep → an AdaRMS conditioning vector.
         b. Embed the suffix (action tokens only).
         c. Forward the suffix through the AdaRMS action expert.
         d. ``x_t = x_t + dt * v_t`` (Euler integration).
      4. Return ``x_0`` as the predicted action chunk.
    """

    # The blocks ``Pi05Pipeline.setup_compile`` compiles one by one (``regionally_compile``).
    _repeated_blocks = ["Pi05GemmaDecoderLayer", "Pi05ExpertDecoderLayer"]

    def __init__(self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""):
        super().__init__()
        # ``quant_config`` reaches the PaliGemma language model's linear layers
        # only (``PaliGemmaWithActionExpertPi05``); ``None`` serves in the
        # serving dtype throughout, the default.
        self.config = config

        self.action_dim = getattr(config, "max_action_dim", DEFAULT_ACTION_DIM)
        self.max_state_dim = getattr(config, "max_state_dim", self.action_dim)
        self.action_horizon = getattr(config, "chunk_size", DEFAULT_ACTION_HORIZON)
        self.num_inference_steps = getattr(config, "num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)

        paligemma_variant = getattr(config, "paligemma_variant", "gemma_2b")
        action_expert_variant = getattr(config, "action_expert_variant", "gemma_300m")
        vlm_config = get_gemma_config(paligemma_variant)
        expert_config = get_gemma_config(action_expert_variant)
        self.vlm_width = vlm_config.width
        self.expert_width = expert_config.width

        # Dual backbone
        self.paligemma_with_expert = PaliGemmaWithActionExpertPi05(vlm_config, expert_config, quant_config)

        # Action chunk projections.
        self.action_in_proj = nn.Linear(self.action_dim, self.expert_width)
        self.action_out_proj = nn.Linear(self.expert_width, self.action_dim)

        # π0.5 timestep MLP: (W → W → W) with SiLU, feeding AdaRMS.
        # π0 instead has action_time_mlp_{in,out} of shape (2W → W → W) because
        # it concatenates the time embedding onto the action embedding.
        # NOTE: there is deliberately **no** ``state_proj`` here — that is the
        # π0-only continuous-state path.
        self.time_mlp_in = nn.Linear(self.expert_width, self.expert_width)
        self.time_mlp_out = nn.Linear(self.expert_width, self.expert_width)

        # ``None`` runs ``sample_actions`` eagerly (the baseline). ``Pi05Pipeline``
        # installs a ``Pi05CUDAGraphs`` unless the stage sets ``enforce_eager``.
        self.cuda_graphs: Pi05CUDAGraphs | None = None
        # Allocated once by ``Pi05Pipeline`` after the model reaches its device
        # and dtype. ``sample_actions`` falls back to a one-off workspace when this
        # is unset or sized for another batch or prefix length.
        self.kv_workspace: Pi05StepKVWorkspace | None = None

    def new_kv_workspace(self, batch_size: int, prefix_len: int | None = None) -> Pi05StepKVWorkspace:
        """Create a new `Pi05StepKVWorkspace` object on this model's device, in the expert's K/V dtype.

        ``prefix_len`` defaults to the deployed prefix: ``max_cameras`` slots of
        SigLIP image tokens plus the ``tokenizer_max_length`` text block.
        """
        text_config = self.paligemma_with_expert.paligemma.config.text_config
        expert_qkv_proj = self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.qkv_proj
        if prefix_len is None:
            vision_config = self.paligemma_with_expert.paligemma.config.vision_config
            tokens_per_image = (vision_config.image_size // vision_config.patch_size) ** 2
            prefix_len = int(self.config.max_cameras) * tokens_per_image + int(self.config.tokenizer_max_length)
        return Pi05StepKVWorkspace(
            num_layers=text_config.num_hidden_layers,
            batch_size=batch_size,
            num_kv_heads=text_config.num_key_value_heads,
            prefix_len=prefix_len,
            suffix_len=self.action_horizon,
            head_dim=text_config.head_dim,
            dtype=expert_qkv_proj.weight.dtype,
            device=expert_qkv_proj.weight.device,
        )

    def enable_fused_kernels(self) -> None:
        """Run each denoising step's action expert through the fused Triton kernels.

        The optimized path's kernels (``Pi05Pipeline`` enables them together
        with its CUDA graphs). The prefix pass keeps its eager layers, so the
        prefix K/V is the same on either path. A step still runs eagerly when
        it is not batch 1 on a single-KV-head ``Pi05StepKVWorkspace``. Raises if
        the kernels cannot serve this model at all.
        """
        problems = []
        if not HAS_TRITON:
            problems.append("Triton is not installed")
        if self.action_in_proj.weight.device.type != "cuda":
            problems.append(f"the model is on {self.action_in_proj.weight.device}, not a CUDA device")
        if self.action_out_proj.bias is None:
            problems.append("action_out_proj has no bias")
        expert = self.paligemma_with_expert.gemma_expert.model
        config = expert.config
        if config.num_key_value_heads != 1:
            problems.append(f"the action expert has {config.num_key_value_heads} KV heads, not 1")
        head_dim = expert.layers[0].self_attn.head_dim
        proj_dtype = _linear_dtype(expert.layers[0].self_attn.qkv_proj)
        if proj_dtype not in _FUSED_DTYPES:
            problems.append(f"the action expert runs in {proj_dtype}, not one of {_FUSED_DTYPES}")
        else:
            # The tiles that span head_dim unmasked: the RoPE half-tiles, and
            # with Triton GEMMs the score reduction and the value columns.
            spans = [2 * _tiles("qkv", proj_dtype).block_n]
            if _triton_gemms(proj_dtype):
                spans += [_tiles("scores", proj_dtype).block_k, _tiles("values", proj_dtype).block_n]
            if any(head_dim % span for span in spans):
                problems.append(f"the action expert head_dim {head_dim} is not a multiple of the kernel tiles {spans}")
        if problems:
            raise RuntimeError("π0.5's fused kernels cannot run this model: " + "; ".join(problems) + ".")
        self.paligemma_with_expert.fused_kernels = True

    # ── Prefix embedding ─────────────────────────────────────────────
    def embed_prefix(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build prefix embeddings, per-token padding mask, and AR mask.

        Prefix tokens form ``[img_cam_0..., ..., lang_tokens...]`` with fully
        bidirectional attention (all-zero ``att_masks``). Identical to π0 — the
        state is inside ``lang_tokens``, so nothing here changes shape-wise.

        Cameras are embedded one at a time; each call is ``(B, 3, 224, 224)``.
        The number of slots is fixed by ``config.max_cameras`` for the deployed
        model; missing cameras occupy their slot with a false image mask.
        """
        num_views = len(images)
        if len(image_masks) != num_views:
            raise ValueError(
                f"images and image_masks must contain the same number of views, got {num_views} and {len(image_masks)}."
            )
        max_cameras = int(self.config.max_cameras)
        if num_views != max_cameras:
            raise ValueError(f"Expected exactly max_cameras={max_cameras} image views, got {num_views}.")

        embs: list[torch.Tensor] = []
        pad_masks: list[torch.Tensor] = []

        for img, img_mask in zip(images, image_masks):
            img_emb = self.paligemma_with_expert.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        # All zeros, built on the device: a host list here is a pageable
        # host-to-device copy that syncs the stream and fails CUDA Graph capture.
        att_masks = torch.zeros(pad_masks.shape, dtype=torch.bool, device=embs.device)

        return embs, pad_masks, att_masks

    # ── Timestep + suffix embedding ──────────────────────────────────
    def embed_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """Timestep → AdaRMS conditioning vector ``(B, expert_width)``.

        ``silu(time_mlp_out(silu(time_mlp_in(sinusoid(t)))))``. The trailing
        SiLU is part of the reference implementation — dropping it is a silent
        numerical error, not a crash.
        """
        model_dtype = self.action_in_proj.weight.dtype
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=getattr(self.config, "min_period", 4e-3),
            max_period=getattr(self.config, "max_period", 4.0),
            device=timestep.device,
        ).to(dtype=model_dtype)
        time_cond = self.time_mlp_in(time_emb)
        time_cond = F.silu(time_cond)
        time_cond = self.time_mlp_out(time_cond)
        return F.silu(time_cond)

    def embed_suffix(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the suffix: **action tokens only**, plus the AdaRMS condition.

        π0's suffix is ``[state_token, action_tokens×H]`` with an AR mask of
        ``[1, 1, 0...]``. π0.5 has no state token, so the suffix is
        ``[action_tokens×H]`` and the mask is ``[1] + [0]*(H-1)``: the first
        action token opens a causal block and the rest attend bidirectionally
        within it.
        """
        model_dtype = self.action_in_proj.weight.dtype
        noisy_actions = noisy_actions.to(dtype=model_dtype)

        time_cond = self.embed_timestep(timestep)
        action_emb = self.action_in_proj(noisy_actions)  # (B, H, W)

        bsize, action_len = action_emb.shape[:2]
        pad_masks = torch.ones(bsize, action_len, dtype=torch.bool, device=action_emb.device)
        # ``[1] + [0] * (H - 1)``, built on the device: a host list here is a
        # pageable host-to-device copy on every step that syncs the stream and
        # fails CUDA Graph capture.
        att_masks = torch.zeros(self.action_horizon, dtype=action_emb.dtype, device=action_emb.device)
        att_masks[:1].fill_(1)
        att_masks = att_masks[None, :].expand(bsize, -1)
        return action_emb, pad_masks, att_masks, time_cond

    # ── Denoising step ───────────────────────────────────────────────
    def denoise_step(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one flow-matching denoising step: predict ``v_t`` from ``x_t``.

        Signature differs from π0's by exactly one argument: no ``state``. With
        the fused kernels enabled, a batch-1 step runs the action expert and
        its output head on them (``_fused_denoise_forward``).
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, time_cond = self.embed_suffix(x_t, timestep)

        batch_size = prefix_pad_masks.shape[0]
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # Position IDs continue from where the prefix's last valid token left off.
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)

        if (
            _fused_kernels_serve(self.paligemma_with_expert, past_key_values, batch_size)
            and past_key_values.fits(batch_size, prefix_len)
            and past_key_values.suffix_len == suffix_len
        ):
            return _fused_denoise_forward(
                self.paligemma_with_expert.gemma_expert,
                suffix_embs,
                full_att_2d_masks_4d,
                position_ids,
                past_key_values,
                time_cond,
                self.action_out_proj,
            )

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=time_cond,
        )

        # Every suffix token is an action token in π0.5 (no state token to drop).
        suffix_out = outputs_embeds[1][:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    # ── Prefix pass ──────────────────────────────────────────────────
    def _prefix_pass(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, Pi05StepKVWorkspace]:
        """Embed the prefix and run it through PaliGemma into the step KV workspace.

        Returns the ``(B, prefix_len)`` prefix pad masks and the workspace whose
        prefix slots now hold every row's per-layer prefix K/V: ``kv_workspace``
        when it fits this batch, else a one-off workspace. Both regions run
        through CUDA graphs when installed; each falls back to its eager call
        on its own.
        """
        graphs = self.cuda_graphs
        embed_prefix = self.embed_prefix if graphs is None else graphs.embed_prefix
        prefix_forward = self.paligemma_with_expert.forward if graphs is None else graphs.prefix_forward

        prefix_embs, prefix_pad_masks, prefix_att_masks = embed_prefix(images, image_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)

        batch_size, prefix_len = prefix_embs.shape[:2]
        kv_workspace = self.kv_workspace
        if kv_workspace is None or not kv_workspace.fits(batch_size, prefix_len):
            kv_workspace = self.new_kv_workspace(batch_size, prefix_len)
        _, kv_workspace = prefix_forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=kv_workspace,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return prefix_pad_masks, kv_workspace

    @torch.no_grad()
    def encode_prefix(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> list[Pi05RequestPrefixKV]:
        """Encode each row's observation into a prefix K/V its request owns.

        For execution that denoises a changing batch one step at a time: each
        request keeps its :class:`Pi05RequestPrefixKV` for all of its denoising
        steps, and a step loads the prefixes of the rows it runs into a
        :class:`Pi05StepKVWorkspace` (``load_prefixes``) before
        ``denoise_step``. The prefix pass itself runs exactly as in
        ``sample_actions``, so the returned K/V are the ones it would attend over.
        """
        prefix_pad_masks, kv_workspace = self._prefix_pass(images, image_masks, lang_tokens, lang_masks)
        return [kv_workspace.export_prefix(row, prefix_pad_masks[row]) for row in range(prefix_pad_masks.shape[0])]

    # ── Full action generation ───────────────────────────────────────
    @torch.no_grad()
    def sample_actions(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> torch.Tensor:
        """Generate an action chunk via iterative flow-matching denoising.

        Convention: ``t=1`` is noise, ``t=0`` is the target — opposite of the
        published π0 paper but matching both OpenPI and LeRobot.

        Takes no ``state``: π0.5's state rides inside ``lang_tokens``.
        """
        if num_steps is None:
            num_steps = self.num_inference_steps

        bsize = lang_tokens.shape[0]
        device = lang_tokens.device
        if noise is None:
            noise_shape = (self.action_horizon, self.action_dim)
            if isinstance(generator, list):
                if len(generator) != bsize:
                    raise ValueError(f"Expected {bsize} generators, got {len(generator)}.")
                noise = torch.stack(
                    [torch.randn(noise_shape, dtype=torch.float32, device=device, generator=item) for item in generator]
                )
            else:
                noise = torch.randn(
                    bsize,
                    *noise_shape,
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                )

        # 1-2. Prefix embedding and forward, into the step KV workspace.
        prefix_pad_masks, past_key_values = self._prefix_pass(images, image_masks, lang_tokens, lang_masks)

        # 3. Euler-integrated denoising from t=1 down to t=0. Its CUDA graph
        #    falls back to the eager step on its own, as the prefix regions do.
        graphs = self.cuda_graphs
        denoise_step = self.denoise_step if graphs is None else graphs.denoise_step
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t = 1.0 + step * dt
            time_tensor = torch.full((bsize,), t, dtype=torch.float32, device=device)
            v_t = denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t
        return x_t

    # ── Weight loading ───────────────────────────────────────────────
    def process_weights_after_loading(self) -> None:
        """Finish every vLLM linear's weights once the checkpoint is in, as the
        diffusion loader does after loading.

        ``Pi05Pipeline`` loads its own checkpoint inside ``__init__`` and
        captures its CUDA graphs right after, so this cannot wait for the
        loader: a graph captured before, say, an fp8 weight is repacked would
        replay stale storage. The flag it sets makes the loader's own pass skip
        these layers. Call it once, after ``load_weights``: on some CPUs it
        repacks an unquantized weight for vLLM's CPU GEMM and drops ``weight``.
        """
        for module in self.modules():
            quant_method = getattr(module, "quant_method", None)
            if not isinstance(quant_method, QuantizeMethodBase):
                continue
            if getattr(module, "_already_called_process_weights_after_loading", False):
                continue
            quant_method.process_weights_after_loading(module)
            module._already_called_process_weights_after_loading = True

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        strict: bool = True,
    ):
        """Load and audit a LeRobot π0.5 safetensors checkpoint.

        Same remap rules as π0 (strip the ``model.`` prefix, flatten→nested
        PaliGemma submodules, tied ``lm_head`` → ``embed_tokens``, version-robust
        SigLIP nesting), plus the ones vLLM's layers and π0.5 need:

          * each Gemma tower's ``q/k/v_proj`` and ``gate/up_proj`` load as
            shards of the packed ``qkv_proj`` / ``gate_up_proj``, and a packed
            parameter counts as loaded only once every shard has; SigLIP loads
            through vLLM's own ``SiglipVisionTransformer.load_weights``.
          * the action expert's ``lm_head`` is dropped: the expert decodes no
            tokens, so π0.5 never reads it.

          * ``action_time_mlp_{in,out}`` → ``time_mlp_{in,out}``: some
            checkpoints were exported under the π0 parameter names.
          * ``state_proj.*`` is reported, not silently dropped. A π0.5
            checkpoint should not contain it; its presence usually means a π0
            checkpoint was pointed at the π0.5 model class, which would
            otherwise run happily with a randomly-initialized action expert.

        The action-expert norms are AdaRMS here, so they expose ``dense.weight``
        / ``dense.bias`` and no plain ``weight``. A checkpoint that carries a
        plain expert-norm ``weight`` is a π0-shaped checkpoint; that too is
        rejected rather than skipped. ``strict=False`` exists only for focused
        remapping unit tests that intentionally provide a partial state dict;
        the serving path always uses the strict default.
        """
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())

        _PALIGEMMA_SUBMODULES = ("vision_tower", "multi_modal_projector", "language_model")
        _EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model."

        def _remap(name: str) -> str:
            # Strip the leading "model." that LeRobot's PI05Policy wrapper adds.
            if name.startswith("model."):
                name = name[len("model.") :]

            # π0-style timestep MLP names → π0.5 names.
            if name.startswith("action_time_mlp_in."):
                name = "time_mlp_in." + name[len("action_time_mlp_in.") :]
            elif name.startswith("action_time_mlp_out."):
                name = "time_mlp_out." + name[len("action_time_mlp_out.") :]

            # Nested PaliGemma layout.
            for sub in _PALIGEMMA_SUBMODULES:
                flat = f"paligemma_with_expert.paligemma.{sub}."
                nested = f"paligemma_with_expert.paligemma.model.{sub}."
                if name.startswith(flat) and not name.startswith(nested):
                    return nested + name[len(flat) :]

            if name == "paligemma_with_expert.paligemma.lm_head.weight":
                return "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"

            return name

        _VISION_PREFIX = "paligemma_with_expert.paligemma.model.vision_tower."

        def _vision_model_name(name: str) -> str:
            """The name inside ``SiglipVisionModel.vision_model``. Reconciles
            SigLIP nesting across transformers versions (≤5.3 wraps the
            encoder in ``vision_tower.vision_model.*``; ≥5.4 flattens it)."""
            rest = name[len(_VISION_PREFIX) :]
            if rest.startswith("vision_model."):
                return rest[len("vision_model.") :]
            return rest

        loaded = 0
        skipped: list[str] = []
        pi0_shaped: list[str] = []
        filled_params: set = set()
        vision_weights: list[tuple[str, torch.Tensor]] = []
        # The packed parameters' shards seen so far; each needs all of them.
        packed_shards: dict[str, set] = {}

        for name, loaded_weight in weights:
            mapped = _remap(name)

            # Diagnose π0-shaped keys instead of dropping them quietly.
            is_expert_norm_weight = mapped.startswith(_EXPERT_PREFIX) and (
                mapped.endswith("input_layernorm.weight")
                or mapped.endswith("post_attention_layernorm.weight")
                or mapped == _EXPERT_PREFIX + "norm.weight"
            )
            if mapped.startswith("state_proj.") or is_expert_norm_weight:
                pi0_shaped.append(mapped)
                continue
            if mapped == _UNUSED_EXPERT_LM_HEAD:
                continue
            if mapped.startswith(_VISION_PREFIX):
                vision_weights.append((_vision_model_name(mapped), loaded_weight))
                continue

            packed = None
            for packed_name, checkpoint_name, shard_id in _PACKED_PARAMS:
                if checkpoint_name in mapped:
                    packed = (mapped.replace(checkpoint_name, packed_name), shard_id)
                    break
            if packed is not None and packed[0] in params_dict:
                packed_name, shard_id = packed
                param = params_dict[packed_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded += 1
                packed_shards.setdefault(packed_name, set()).add(shard_id)
            elif mapped in params_dict:
                param = params_dict[mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded += 1
                filled_params.add(mapped)
            elif mapped in buffers_dict:
                buffers_dict[mapped].copy_(loaded_weight)
                loaded += 1
                filled_params.add(mapped)
            else:
                skipped.append(mapped)

        for packed_name, shards in packed_shards.items():
            if shards == _PACKED_SHARDS[packed_name.rsplit(".", 2)[-2]]:
                filled_params.add(packed_name)

        if vision_weights:
            vision_model = self.paligemma_with_expert.paligemma.model.vision_tower.vision_model
            vision_loaded = vision_model.load_weights(vision_weights)
            loaded += len(vision_weights)
            filled_params.update(_VISION_PREFIX + "vision_model." + vision_name for vision_name in vision_loaded)

        # Reverse audit: any model param that got no checkpoint tensor at all
        # would be running with random init.
        missing_params = [pname for pname in params_dict if pname not in filled_params]

        parts: list[str] = []
        if pi0_shaped:
            parts.append(
                f"{len(pi0_shaped)} checkpoint key(s) are π0-shaped, not π0.5-shaped (first 5: {pi0_shaped[:5]})"
            )
        if skipped:
            parts.append(f"{len(skipped)} checkpoint key(s) had no model target (first 5: {skipped[:5]})")
        if missing_params:
            parts.append(f"{len(missing_params)} model param(s) received no weight (first 5: {missing_params[:5]})")

        if parts and strict:
            raise RuntimeError("Incomplete or incompatible π0.5 checkpoint: " + "; ".join(parts))
        if parts:
            logger.debug("π0.5 partial test load: %d tensors loaded — %s.", loaded, "; ".join(parts))
        else:
            logger.info("π0.5 load_weights: %d tensors loaded, 0 skipped, 0 missing.", loaded)
        return filled_params


EntryClass = Pi05ForActionPrediction
