# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sol-Attn: training-free on-the-fly sparse attention for video DiTs.

Sol-Attn (NVlabs/Sana ``sol-engine``, arXiv:2607.24027) routes 64-token KV
blocks inside one online-softmax pass: every block gets a lightweight proxy
score, blocks above a per-query threshold are evaluated exactly, and the rest
reuse their proxy score instead of being dropped. No routing map is ever
materialized, so the kernel needs nothing from the model beyond contiguous
BF16 ``[B, T, H, 128]`` tensors.

The integration contract, taken from the released MiniMax-H3 policy in
Sol-Engine:

* A contiguous *prefix* of the packed sequence (text, visual conditions,
  audio: everything before the target-video tail) is marked as an exact KV
  sink, so every query attends those keys exactly.
* The prefix's own *query* rows are recomputed densely afterwards; the sink
  keeps the prefix exact as keys only.
* The first ``dense_steps`` denoise steps and the DiT blocks listed in
  ``dense_layers`` stay dense.

``AttentionMetadata.video_layout`` supplies the prefix length, and the packed
padding metadata supplies the valid length of document 0, so a model that
already publishes both (MiniMax-H3 does) needs no code changes. Anything the
sparse path cannot serve delegates to ``FLASH_ATTN`` with the same metadata.

The kernel is an optional external package (``pip install -e
techniques/sparse_backends`` from the ``sol-engine`` branch). It is imported
lazily inside a ``torch.ops.vllm_omni`` custom op so regional compilation sees a
single opaque Tensor->Tensor boundary, mirroring FASTVIDEO_VSA.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    VideoTokenLayout,
)
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

logger = init_logger(__name__)

_HEAD_DIM = 128
_BLOCK_SIZE = 64
_INPUT_LAYOUT = "BSND"
_VALID_THRESH_TYPES = ("diag", "exact")
_VALID_KV_SPLITS = (1, 2, 4)
_VALID_SINK_MODES = ("prefix", "none")
# Sol-Engine selects split-KV only on SM90 for sequences of at least this many tokens.
_SM90_SPLIT_KV_MIN_TOKENS = 65536
_INSTALL_HINT = (
    "Install the released kernel from the NVlabs/Sana `sol-engine` branch: "
    "`git clone -b sol-engine https://github.com/NVlabs/Sana && "
    "pip install -e Sana/techniques/sparse_backends` (CuTe DSL / cutlass and "
    "cuda-python are needed for the SM89/SM90/SM100/SM120 kernels; other "
    "architectures use the Triton reference)."
)


# Keep the external CuTe DSL / Triton kernel opaque to torch.compile. Dynamo
# sees one Tensor->Tensor boundary and can schedule the surrounding DiT block
# normally instead of tracing into the lazily compiled kernel launcher.
if not hasattr(torch.ops.vllm_omni, "sol_attn"):

    @torch.library.custom_op("vllm_omni::sol_attn", mutates_args=())
    def _sol_attn_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        tau: float,
        thresh_type: str,
        kv_splits: int,
        sink_start: int,
        sink_tokens: int,
    ) -> torch.Tensor:
        from sol_attn import sol_attn

        return sol_attn(
            query,
            key,
            value,
            scale=scale,
            tau=tau,
            thresh_type=thresh_type,
            kv_splits=kv_splits,
            sink_start=sink_start,
            sink_tokens=sink_tokens,
        )

    @_sol_attn_op.register_fake
    def _(query, key, value, scale, tau, thresh_type, kv_splits, sink_start, sink_tokens):
        del key, value, scale, tau, thresh_type, kv_splits, sink_start, sink_tokens
        return torch.empty_like(query)


_sol_attn_op = torch.ops.vllm_omni.sol_attn


def _try_extract_layer_index(prefix: str) -> int | None:
    if not prefix:
        return None
    try:
        return extract_layer_index(prefix)
    except (AssertionError, ValueError):
        return None


@dataclass(frozen=True)
class SolAttnConfig:
    """Resolved Sol-Attn controls for one attention layer.

    ``tau`` scales the routing threshold (larger routes fewer blocks exactly);
    ``thresh_type`` picks the diagonal or full-covariance threshold estimate.
    ``kv_splits`` is the split-KV factor, ``None`` meaning the Sol-Engine
    policy (4 on SM90 for >= 65536 tokens, else 1). ``dense_steps`` and
    ``dense_layers`` are the accuracy knobs. ``sink_mode`` selects whether the
    published prefix is kept as an exact KV sink. ``strict`` turns silent dense
    fallbacks on kernel errors into exceptions.
    """

    tau: float = 1.0
    thresh_type: str = "diag"
    kv_splits: int | None = None
    dense_steps: int = 10
    dense_layers: frozenset[int] = frozenset({0, 1})
    sink_mode: str = "prefix"
    strict: bool = False

    @classmethod
    def from_backend_kwargs(cls, backend_kwargs: Mapping[str, Any] | None) -> SolAttnConfig:
        bk = backend_kwargs or {}
        raw_kv_splits = bk.get("kv_splits")
        kv_splits: int | None
        if raw_kv_splits is None or raw_kv_splits == "auto":
            kv_splits = None
        else:
            kv_splits = int(raw_kv_splits)
            if kv_splits not in _VALID_KV_SPLITS:
                raise ValueError(f"SOL_ATTN kv_splits must be one of {_VALID_KV_SPLITS} or 'auto', got {kv_splits}")
        thresh_type = str(bk.get("thresh_type", "diag"))
        if thresh_type not in _VALID_THRESH_TYPES:
            raise ValueError(f"SOL_ATTN thresh_type must be one of {_VALID_THRESH_TYPES}, got {thresh_type!r}")
        sink_mode = str(bk.get("sink_mode", "prefix"))
        if sink_mode not in _VALID_SINK_MODES:
            raise ValueError(f"SOL_ATTN sink_mode must be one of {_VALID_SINK_MODES}, got {sink_mode!r}")
        tau = float(bk.get("tau", 1.0))
        if not math.isfinite(tau):
            raise ValueError(f"SOL_ATTN tau must be finite, got {tau}")
        dense_steps = int(bk.get("dense_steps", 10))
        if dense_steps < 0:
            raise ValueError(f"SOL_ATTN dense_steps must be >= 0, got {dense_steps}")
        dense_layers = bk.get("dense_layers")
        return cls(
            tau=tau,
            thresh_type=thresh_type,
            kv_splits=kv_splits,
            dense_steps=dense_steps,
            dense_layers=frozenset(int(i) for i in dense_layers) if dense_layers is not None else frozenset({0, 1}),
            sink_mode=sink_mode,
            strict=bool(bk.get("strict", False)),
        )


@dataclass(frozen=True)
class SolAttnPlan:
    """Per-forward geometry handed to the kernel."""

    used_len: int
    sink_start: int
    sink_tokens: int


class SolAttnBackend(AttentionBackend):
    supported_platforms: tuple[str, ...] = ("cuda",)

    @classmethod
    def validate_available(cls) -> None:
        if importlib.util.find_spec("sol_attn") is None:
            raise ImportError(f"SOL_ATTN requires the `sol_attn` package, which is not importable. {_INSTALL_HINT}")

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        # The sparse path reads the valid length of document 0 off the packed
        # metadata and never touches attn_mask; the dense fallback is
        # FLASH_ATTN, whose packed varlen path has the same property.
        return FlashAttentionBackend.supports_packed_mask_free()

    @classmethod
    def supports_multi_doc_packed_varlen(cls) -> bool:
        # Routing and the exact sink are defined over one document. A model
        # that packs several requests into a forward must run them one at a
        # time on this backend; advertising False makes MiniMax-H3 do exactly
        # that instead of silently attending across request boundaries.
        return False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [_HEAD_DIM]

    @staticmethod
    def get_name() -> str:
        return "SOL_ATTN"

    @staticmethod
    def get_impl_cls() -> type[SolAttnImpl]:
        return SolAttnImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError


class SolAttnImpl(AttentionImpl):
    """On-the-fly block-sparse self-attention via the external ``sol_attn`` kernel.

    Every call that the kernel cannot serve — warmup denoise steps, dense
    layers, a missing valid-length contract, masks, joint or piecewise
    attention, non-BF16 activations, a kernel failure in non-strict mode —
    delegates to FlashAttention with the untouched metadata, so a model can
    select this backend unconditionally.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: Mapping[str, Any] | None = None,
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_size
        self.softmax_scale = float(softmax_scale)
        self.causal = causal
        self.qkv_layout = qkv_layout
        self.config = SolAttnConfig.from_backend_kwargs(backend_kwargs)
        self.layer_idx = _try_extract_layer_index(prefix)

        if causal:
            raise ValueError(
                "SOL_ATTN does not support causal attention: the kernel is noncausal and routes "
                "KV blocks by proxy score. Select FLASH_ATTN for causal roles."
            )
        if head_size != _HEAD_DIM:
            raise ValueError(
                f"SOL_ATTN requires head_size={_HEAD_DIM}, got {head_size}. Select FLASH_ATTN for this role."
            )
        if qkv_layout is not None and qkv_layout.upper() != _INPUT_LAYOUT:
            raise ValueError(
                f"SOL_ATTN needs {_INPUT_LAYOUT} tensors to locate the sequence axis, but this layer "
                f"declares qkv_layout={qkv_layout!r}. Select FLASH_ATTN for this role."
            )
        self._validate_parallel_config()

        self.dense_fallback = FlashAttentionBackend.get_impl_cls()(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            qkv_layout=qkv_layout,
        )
        # Every layer shares one spec, so this prints once per process and is
        # the line to grep for when checking which backend a run selected. The
        # per-forward "SOL_ATTN active" line below confirms the sparse path was
        # actually taken; without it the run was dense.
        logger.info_once(
            "SOL_ATTN configured: tau=%.3f, thresh_type=%s, kv_splits=%s, dense_steps=%d, dense_layers=%s, "
            "sink_mode=%s, strict=%s (dense fallback: FLASH_ATTN).",
            self.config.tau,
            self.config.thresh_type,
            "auto" if self.config.kv_splits is None else self.config.kv_splits,
            self.config.dense_steps,
            sorted(self.config.dense_layers),
            self.config.sink_mode,
            self.config.strict,
        )

    @staticmethod
    def _validate_parallel_config() -> None:
        config = get_current_diffusion_config_or_none()
        parallel_config = getattr(config, "parallel_config", None)
        ring_degree = getattr(parallel_config, "ring_degree", 1)
        if ring_degree > 1:
            # Ring hands each rank a slice of the sequence and bypasses the
            # backend entirely (see Attention._run_ring_attention). Ulysses is
            # fine: after the all-to-all every rank holds the whole sequence for
            # its own heads, and routing is decided per (query, head).
            raise ValueError(
                "SOL_ATTN is not compatible with ring sequence parallelism "
                f"(ring_degree={ring_degree}): block routing needs the whole key sequence. "
                "Use Ulysses SP (ring_degree=1) instead."
            )

    # -- dispatch ---------------------------------------------------------------

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        plan = self._resolve_plan(query, key, value, attn_metadata)
        if isinstance(plan, str):
            return self._dense(query, key, value, attn_metadata, plan)
        try:
            return self._forward_sparse(query, key, value, plan)
        except Exception as exc:
            if self.config.strict:
                raise
            return self._dense(query, key, value, attn_metadata, f"kernel failed: {type(exc).__name__}: {exc}")

    def forward_npu(self, query, key, value, attn_metadata=None):
        raise NotImplementedError("SOL_ATTN runs on CUDA only")

    def forward_xpu(self, query, key, value, attn_metadata=None):
        raise NotImplementedError("SOL_ATTN runs on CUDA only")

    def _dense(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        reason: str,
    ) -> torch.Tensor:
        if reason not in ("warmup_step", "dense_layer"):
            logger.warning_once("SOL_ATTN staying dense: %s", reason)
        return self.dense_fallback.forward(query, key, value, attn_metadata)

    # -- plan -------------------------------------------------------------------

    def _resolve_plan(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> SolAttnPlan | str:
        """Return the kernel geometry, or the reason this forward stays dense."""
        cfg = self.config
        if self.layer_idx is not None and self.layer_idx in cfg.dense_layers:
            return "dense_layer"
        if cfg.dense_layers and self.layer_idx is None:
            logger.warning_once(
                "SOL_ATTN cannot resolve a layer index from this attention prefix; dense_layers=%s is ignored.",
                sorted(cfg.dense_layers),
            )
        if cfg.dense_steps > 0:
            step_idx = get_forward_context().denoise_step_idx if is_forward_context_available() else None
            if step_idx is None:
                logger.warning_once(
                    "SOL_ATTN dense_steps=%d is ignored: the pipeline does not publish the denoise "
                    "step index on the forward context, so every step runs sparse.",
                    cfg.dense_steps,
                )
            elif step_idx < cfg.dense_steps:
                return "warmup_step"

        if self.qkv_layout is None:
            return f"this layer does not declare qkv_layout; SOL_ATTN needs {_INPUT_LAYOUT} to locate the sequence axis"
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            return f"expected [B, S, H, D] tensors, got {tuple(query.shape)}"
        if query.shape != key.shape or query.shape != value.shape:
            return (
                "self-attention with identical q/k/v shapes is required (GQA/MQA and cross-attention are not supported)"
            )
        if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            return f"the kernel requires bfloat16 activations, got {query.dtype}"
        if query.shape[-1] != _HEAD_DIM:
            return f"the kernel requires head_dim {_HEAD_DIM}, got {query.shape[-1]}"
        if query.device.type != "cuda":
            return f"q/k/v must be CUDA tensors, got {query.device.type}"

        batch, total_len = int(query.shape[0]), int(query.shape[1])
        if attn_metadata is not None:
            if attn_metadata.attn_mask is not None:
                return "attention masks are not supported"
            if attn_metadata.full_attn_spans is not None:
                return "piecewise/full attention spans are not supported"
            if attn_metadata.joint_query is not None or attn_metadata.joint_key is not None:
                return "joint attention metadata is not supported"

        used_len = self._resolve_used_len(attn_metadata, total_len)
        if isinstance(used_len, str):
            return used_len
        if used_len < total_len and batch != 1:
            return f"padded packed sequences require batch size 1, got {batch}"
        if used_len < 2 * _BLOCK_SIZE:
            return f"sequence length {used_len} is too short to route ({2 * _BLOCK_SIZE} rows minimum)"

        sink = self._resolve_sink(attn_metadata.video_layout if attn_metadata is not None else None, used_len)
        if isinstance(sink, str):
            return sink
        sink_start, sink_tokens = sink
        plan = SolAttnPlan(used_len=used_len, sink_start=sink_start, sink_tokens=sink_tokens)
        logger.info_once(
            "SOL_ATTN active: tau=%.3f, thresh_type=%s, dense_steps=%d, dense_layers=%s, sink_mode=%s, "
            "used_len=%d, sink=[%d, %d), total_len=%d, heads=%d.",
            cfg.tau,
            cfg.thresh_type,
            cfg.dense_steps,
            sorted(cfg.dense_layers),
            cfg.sink_mode,
            used_len,
            sink_start,
            sink_start + sink_tokens,
            total_len,
            int(query.shape[2]),
        )
        return plan

    @staticmethod
    def _resolve_used_len(attn_metadata: AttentionMetadata | None, total_len: int) -> int | str:
        """Valid length of packed document 0, without reading device scalars.

        Producers describe alignment padding in three equivalent ways; all are
        plain ints. Absent every one of them, the whole sequence is valid.
        """
        if attn_metadata is None:
            return total_len
        used: int | None = None
        packed = attn_metadata.packed_padding
        if packed is not None:
            if packed.q_length != packed.kv_length:
                return f"packed_padding q_length {packed.q_length} != kv_length {packed.kv_length}"
            used = int(packed.q_length)
        else:
            valid_kv = attn_metadata.extra.get("valid_kv_length")
            if isinstance(valid_kv, int) and not isinstance(valid_kv, bool):
                used = valid_kv
            else:
                layout = attn_metadata.video_layout
                if layout is not None:
                    if layout.used_len is not None:
                        used = int(layout.used_len)
                    elif layout.prefix_len is not None and layout.latent_grid is not None:
                        used = int(layout.prefix_len) + math.prod(int(d) for d in layout.latent_grid)
        if used is None:
            used = total_len
        if not 0 < used <= total_len:
            return f"valid length {used} is outside (0, {total_len}]"
        max_seqlen_q = attn_metadata.extra.get("max_seqlen_q")
        if max_seqlen_q is not None and int(max_seqlen_q) != used:
            # A longer document than the valid prefix means the packing is not
            # the single-document [real, pad] contract this backend serves.
            return f"max_seqlen_q {int(max_seqlen_q)} does not match the valid document length {used}"
        return used

    def _resolve_sink(self, layout: VideoTokenLayout | None, used_len: int) -> tuple[int, int] | str:
        """Exact KV range: everything before the target-video rows.

        The kernel applies exactness at 64-token block granularity, rounding
        outward, so a few target-video keys next to the boundary become exact
        too. The sink never changes query routing; ``_forward_sparse``
        recomputes the sink's own query rows densely.
        """
        if self.config.sink_mode == "none":
            return 0, 0
        if layout is None:
            logger.warning_once(
                "SOL_ATTN: no AttentionMetadata.video_layout was published for this role, so the whole "
                "sequence is routed sparsely without an exact prefix sink."
            )
            return 0, 0
        if layout.video_spans:
            targets = [span for span in layout.video_spans if span.role == "target"]
            if len(targets) != 1:
                return f"video layout must contain exactly one target video span, got {len(targets)}"
            target = targets[0]
            if not 0 <= target.start <= used_len - target.length:
                end = target.start + target.length
                return f"target video span [{target.start}, {end}) exceeds valid length {used_len}"
            prefix_len = int(target.start)
        elif layout.prefix_len is not None:
            prefix_len = int(layout.prefix_len)
        else:
            return "video layout has neither a prefix length nor video spans"
        if prefix_len < 0 or prefix_len >= used_len:
            return f"prefix length {prefix_len} leaves no video rows to sparsify in {used_len}"
        return 0, prefix_len

    # -- kernel -----------------------------------------------------------------

    def _resolve_kv_splits(self, device: torch.device, tokens: int) -> int:
        if self.config.kv_splits is not None:
            return self.config.kv_splits
        from sol_attn import get_sol_attn_backend

        if get_sol_attn_backend(device) == "cute_sm90" and tokens >= _SM90_SPLIT_KV_MIN_TOKENS:
            return 4
        return 1

    def _forward_sparse(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        plan: SolAttnPlan,
    ) -> torch.Tensor:
        cfg = self.config
        total_len = int(query.shape[1])
        used = plan.used_len
        # The kernel wants contiguous BTHD over the valid rows only; the
        # trailing alignment padding never reaches it.
        q = query[:, :used].contiguous()
        k = key[:, :used].contiguous()
        v = value[:, :used].contiguous()

        out = _sol_attn_op(
            q,
            k,
            v,
            self.softmax_scale,
            cfg.tau,
            cfg.thresh_type,
            self._resolve_kv_splits(q.device, used),
            plan.sink_start,
            plan.sink_tokens,
        )
        if plan.sink_tokens:
            # The sink makes the prefix exact as keys. Its own queries still
            # route sparsely, and the release is explicit that an MMDiT
            # integration must recompute them densely.
            lo, hi = plan.sink_start, plan.sink_start + plan.sink_tokens
            out[:, lo:hi] = self._dense_rows(q[:, lo:hi], k, v)
        if used == total_len:
            return out
        padded = query.new_zeros(query.shape)
        padded[:, :used] = out
        return padded

    def _dense_rows(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Dense attention for a slice of query rows over the full valid K/V. [B, S, H, D] in and out."""
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=self.softmax_scale,
        ).transpose(1, 2)
