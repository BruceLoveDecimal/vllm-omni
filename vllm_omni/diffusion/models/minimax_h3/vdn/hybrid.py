# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The hybrid attention branch of one MiniMax-H3 DiT block.

    softmax = to_out( softmax_gate(x) * window_softmax(q, k, v) )
    linear  = to_out_linear( output_gate(x) * RMSNorm( branch(q_raw, k_raw, v) ) )
    out     = softmax;  out[video rows] += linear

Both branches share the block's QKV projection: the window sees the QK-normed,
RoPE'd tensors the dense model would have seen, and the linear branch sees the
raw ones and applies its own NoPE post-processing. Nothing about the dense path
changes - the projections, their names, their quantization prefixes and their
tensor-parallel split are the block's own - which is why this is a submodule of
``MiniMaxH3Attention`` rather than an attention backend or a second model.

Under Ulysses this module runs its own two all-to-alls rather than the shared
sequence-parallel strategy. It has to: the window needs whole frames and the
branch is a recurrence over all of them, so a rank holding a slice of the rows
can compute neither. One exchange turns rows-sharded/all-heads into
all-rows/heads-sharded, and one turns the two branch outputs back.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_omni.diffusion.attention.backends.abstract import VideoTokenLayout
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D

from .config import MiniMaxH3HybridAttentionConfig, MiniMaxH3HybridGeometry
from .linear_branch import MiniMaxH3LinearBranch, _shard_param_across_tp
from .window import WindowPlan, build_window_plan, window_softmax

#: The attention role the window runs under. Distinct from the dense ``self``
#: role so an operator can point the window at a different backend without
#: moving the token refiner, and so a block-sparse backend selected for ``self``
#: never reaches a mask this module has already decomposed.
VDN_WINDOW_ROLE = "minimax_h3.vdn_window"


@dataclass(frozen=True)
class _SequenceShard:
    """This rank's slice of the packed rows, when Ulysses is active."""

    group: dist.ProcessGroup
    world_size: int
    rank: int
    local_rows: int

    @property
    def local_start(self) -> int:
        return self.rank * self.local_rows


class MiniMaxH3HybridAttention(nn.Module):
    """VDN's window softmax and linear branch for one attention layer."""

    def __init__(
        self,
        config: MiniMaxH3HybridAttentionConfig,
        *,
        hidden_size: int,
        total_num_heads: int,
        num_heads: int,
        head_dim: int,
        ulysses_degree: int = 1,
        quant_config: object | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if num_heads % ulysses_degree:
            raise ValueError(
                f"the hybrid branch splits {num_heads} tensor-parallel heads across "
                f"{ulysses_degree} Ulysses ranks, which does not divide"
            )
        self.config = config
        # ``num_heads`` is this tensor-parallel rank's share; ``total_num_heads``
        # is the checkpoint's. vLLM's parallel layers are built from the global
        # width and shard it themselves, while the plain parameters below are
        # built local and narrowed by their weight loader.
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ulysses_degree = ulysses_degree
        self.branch_heads = num_heads // ulysses_degree

        # Per-head mass gate on the softmax branch. The window renormalises to
        # one no matter how little mass it saw, so this scales the branch back
        # toward the share it actually captured - a property of a distribution,
        # hence one value per head rather than per channel.
        self.softmax_gate = nn.ModuleDict({"up": nn.Linear(hidden_size, num_heads, bias=True)})
        _shard_param_across_tp(self.softmax_gate["up"].weight)
        _shard_param_across_tp(self.softmax_gate["up"].bias)

        # Built for every tensor-parallel head, not for the Ulysses share:
        # sequence parallelism redistributes heads at run time and shards no
        # parameter, so each rank holds them all and reads its own slice.
        self.linear = MiniMaxH3LinearBranch(
            config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        # The branch's own output projection. Wide on both sides, so online fp8
        # quantizes it exactly as it does the block's other big matmuls; the
        # gates and the per-head vectors deliberately stay in bf16.
        self.to_out_linear = RowParallelLinear(
            total_num_heads * head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix=f"{prefix}.to_out_linear",
        )
        self.window_attn = Attention(
            num_heads=self.branch_heads,
            num_kv_heads=self.branch_heads,
            head_size=head_dim,
            softmax_scale=head_dim**-0.5,
            causal=False,
            qkv_layout="BSND",
            role=VDN_WINDOW_ROLE,
            role_category="self",
            # The window does its own sequence-parallel exchange, because the
            # linear branch beside it needs the whole sequence too.
            skip_sequence_parallel=True,
            prefix=f"{prefix}.window_attn",
        )
        self._plans: dict[tuple, WindowPlan] = {}

    # -- geometry -------------------------------------------------------------

    def plan(self, geometry: MiniMaxH3HybridGeometry, device: torch.device) -> WindowPlan:
        """The decomposed window for this geometry, built once per request.

        Keyed on the geometry and device rather than cached globally: a server
        answers many shapes, and the index tensors are device-resident.
        """
        key = (
            geometry.seq_len,
            geometry.used_len,
            geometry.video_start,
            geometry.num_frames,
            geometry.frame_height,
            geometry.frame_width,
            str(device),
        )
        plan = self._plans.get(key)
        if plan is None:
            plan = build_window_plan(geometry, self.config, device)
            # One geometry per request, and a server answers a handful of
            # shapes; keep the newest few rather than growing without bound.
            if len(self._plans) >= 4:
                self._plans.pop(next(iter(self._plans)))
            self._plans[key] = plan
        return plan

    # -- forward --------------------------------------------------------------

    @torch.compiler.disable
    def forward(
        self,
        *,
        x: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_raw: torch.Tensor,
        key_raw: torch.Tensor,
        video_layout: VideoTokenLayout | None,
        packed_total: int,
        out_proj: nn.Module,
    ) -> torch.Tensor | None:
        """Run both branches and return the block's attention output.

        ``x`` is the residual stream this rank owns; ``query``/``key`` are
        QK-normed and RoPE'd, ``query_raw``/``key_raw`` are not. ``out_proj`` is
        the block's own output projection, applied here so the softmax branch
        keeps the dense model's parameter, prefix and tensor-parallel split.

        Returns ``None`` for a clip short enough that every frame sees every
        chunk: the window is then full attention, the branch would double count,
        and the caller runs the block's ordinary dense path instead.
        """
        geometry = MiniMaxH3HybridGeometry.from_video_layout(video_layout, packed_total=packed_total)
        if self.config.covers_all_frames(geometry.num_frames):
            return None
        shard = self._sequence_shard(x.shape[0], geometry)
        plan = self.plan(geometry, x.device)
        bounds = list(plan.bounds)

        # Everything that reads the residual stream is computed while the rows
        # are local: sending these beside Q/K/V is far cheaper than exchanging
        # the 5376-wide stream itself.
        softmax_gate = torch.sigmoid(self.softmax_gate["up"](x))
        beta = self.linear.beta(x)
        gate_hidden = self.linear.gate_hidden(x)
        frame_sums = self._frame_sums(x, geometry, shard)

        if shard is None:
            frame_mean = frame_sums / geometry.tokens_per_frame
        else:
            work = dist.all_reduce(frame_sums, group=shard.group, async_op=True)
            query, key, value, query_raw, key_raw, softmax_gate, beta = self._to_heads(
                shard, query, key, value, query_raw, key_raw, softmax_gate, beta
            )
            gate_hidden = self._gather_rows(shard, gate_hidden, geometry.seq_len)
            work.wait()
            frame_mean = frame_sums / geometry.tokens_per_frame

        attended = window_softmax(
            self.window_attn,
            query,
            key,
            value,
            plan,
            group_batch=self.config.window_group_batch,
            varlen=self._use_varlen(),
        )
        attended = attended * softmax_gate.unsqueeze(-1).to(attended.dtype)

        linear_out = None
        if self.config.linear_attention_enabled:
            video = slice(geometry.video_start, geometry.video_end)
            text = slice(geometry.text_start, geometry.text_start + geometry.text_len)
            # Which heads this rank received from the exchange above. The
            # all-to-all hands rank r the r-th head block.
            local_heads = None
            if shard is not None:
                first = shard.rank * self.branch_heads
                local_heads = slice(first, first + self.branch_heads)
            readout = self.linear(
                video_x=None,
                video_qkv=(query_raw[video], key_raw[video], value[video]),
                text_qkv=(query_raw[text], key_raw[text], value[text]),
                num_frames=geometry.num_frames,
                tokens_per_frame=geometry.tokens_per_frame,
                frame_size=geometry.frame_size,
                bounds=bounds,
                beta=beta[video],
                gate=self.linear.gate_from_hidden(gate_hidden[video], heads=local_heads),
                frame_mean=frame_mean,
                text_beta=beta[text],
                heads=local_heads,
            )
            linear_out = attended.new_zeros(attended.shape[0], self.branch_heads, self.head_dim)
            linear_out[video] = readout.view(-1, self.branch_heads, self.head_dim)

        if shard is not None:
            attended, linear_out = self._to_rows(shard, attended, linear_out)

        rows = attended.shape[0]
        out, _ = out_proj(attended.reshape(rows, -1))
        if linear_out is not None:
            local_video = self._local_video_rows(geometry, shard, rows)
            if local_video is not None:
                start, length = local_video
                contribution, _ = self.to_out_linear(linear_out.narrow(0, start, length).reshape(length, -1))
                out.narrow(0, start, length).add_(contribution.to(out.dtype))
        return out

    # -- sequence parallelism -------------------------------------------------

    def _sequence_shard(self, local_rows: int, geometry: MiniMaxH3HybridGeometry) -> _SequenceShard | None:
        """Describe this rank's row slice, or ``None`` when it holds them all."""
        if local_rows == geometry.seq_len:
            return None
        from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

        if local_rows <= 0 or geometry.seq_len % local_rows:
            raise ValueError(
                f"the hybrid branch received {local_rows} of {geometry.seq_len} packed rows, which is not an "
                "equal shard; it supports strict Ulysses sequence parallelism only"
            )
        world_size = geometry.seq_len // local_rows
        sp_group = get_sp_group()
        if sp_group.ulysses_world_size != world_size:
            raise ValueError(
                f"the hybrid branch inferred {world_size} sequence-parallel ranks from the row shard, but "
                f"the process group has ulysses_world_size={sp_group.ulysses_world_size}"
            )
        return _SequenceShard(
            group=sp_group.ulysses_group,
            world_size=world_size,
            rank=sp_group.ulysses_rank,
            local_rows=local_rows,
        )

    def _to_heads(
        self,
        shard: _SequenceShard,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_raw: torch.Tensor,
        key_raw: torch.Tensor,
        softmax_gate: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Rows-sharded/all-heads -> all-rows/heads-sharded, in one exchange.

        Both branches' inputs travel together because they are needed together
        and a second all-to-all would cost another latency for the same bytes.
        The two per-head scalars ride along in the same buffer.
        """
        head_dim = self.head_dim
        packed = torch.cat(
            [
                query,
                key,
                value,
                query_raw,
                key_raw,
                softmax_gate.unsqueeze(-1),
                beta.unsqueeze(-1),
            ],
            dim=-1,
        )
        packed = SeqAllToAll4D.apply(shard.group, packed.unsqueeze(0), 2, 1).squeeze(0)
        widths = (head_dim, head_dim, head_dim, head_dim, head_dim, 1, 1)
        query, key, value, query_raw, key_raw, softmax_gate, beta = packed.split(widths, dim=-1)
        return (
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            query_raw.contiguous(),
            key_raw.contiguous(),
            softmax_gate.squeeze(-1),
            beta.squeeze(-1),
        )

    def _to_rows(
        self,
        shard: _SequenceShard,
        attended: torch.Tensor,
        linear_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """All-rows/heads-sharded -> rows-sharded/all-heads, in one exchange.

        The reverse exchange is also the branch merge: every row owner gets both
        branches' heads and applies the two output projections locally, which
        keeps them independent - each has its own activation scale under fp8.
        """
        head_dim = self.head_dim
        if linear_out is None:
            restored = SeqAllToAll4D.apply(shard.group, attended.unsqueeze(0), 1, 2).squeeze(0)
            return restored, None
        packed = torch.cat([attended, linear_out], dim=-1)
        restored = SeqAllToAll4D.apply(shard.group, packed.unsqueeze(0), 1, 2).squeeze(0)
        softmax_local, linear_local = restored.split((head_dim, head_dim), dim=-1)
        return softmax_local.contiguous(), linear_local.contiguous()

    def _gather_rows(self, shard: _SequenceShard, local: torch.Tensor, seq_len: int) -> torch.Tensor:
        """All-gather a row-sharded ``[rows, width]`` tensor.

        Only the low-rank half of the linear output gate travels this way: the
        receiving rank applies ``up`` for its own heads, so the payload is 128
        wide instead of ``heads * dim``.
        """
        gathered = local.new_empty((seq_len, local.shape[-1]))
        dist.all_gather_into_tensor(gathered, local.contiguous(), group=shard.group)
        return gathered

    def _frame_sums(
        self,
        x: torch.Tensor,
        geometry: MiniMaxH3HybridGeometry,
        shard: _SequenceShard | None,
    ) -> torch.Tensor:
        """Per-frame sums of the residual stream over this rank's rows.

        fp32 on the reduction itself: the stream is bf16, so summing in bf16
        would round before the gate's fp32 island ever starts, and the gate is
        multiplied across every frame by the scan.
        """
        if shard is None:
            video = x.narrow(0, geometry.video_start, geometry.num_video_rows)
            return video.view(geometry.num_frames, geometry.tokens_per_frame, -1).sum(dim=1, dtype=torch.float32)

        sums = torch.zeros(geometry.num_frames, x.shape[-1], device=x.device, dtype=torch.float32)
        start = shard.local_start
        low = max(start, geometry.video_start)
        high = min(start + shard.local_rows, geometry.video_end)
        if low < high:
            rows = x.narrow(0, low - start, high - low).float()
            frames = torch.arange(low, high, device=x.device)
            frames = (frames - geometry.video_start).div(geometry.tokens_per_frame, rounding_mode="floor")
            sums.index_add_(0, frames, rows)
        return sums

    def _local_video_rows(
        self,
        geometry: MiniMaxH3HybridGeometry,
        shard: _SequenceShard | None,
        rows: int,
    ) -> tuple[int, int] | None:
        """This rank's video rows as ``(start, length)`` in its own shard."""
        if shard is None:
            return geometry.video_start, geometry.num_video_rows
        start = shard.local_start
        low = max(start, geometry.video_start)
        high = min(start + rows, geometry.video_end)
        return (low - start, high - low) if low < high else None

    def _use_varlen(self) -> bool:
        """Whether the packed-varlen window path is available and wanted.

        ``auto`` takes it only on a backend that consumes ``cu_seqlens`` as a
        genuine block-diagonal plan. It materialises every group's keys at once
        where the grouped path bounds that gather, so it is the faster
        arrangement exactly where the kernel that needs it exists.
        """
        if self.config.window_impl == "grouped":
            return False
        supported = self.window_attn.attn_backend.supports_multi_doc_packed_varlen()
        if self.config.window_impl == "varlen":
            if not supported:
                raise ValueError(
                    f"vdn.window_impl='varlen' needs an attention backend that isolates packed "
                    f"multi-document cu_seqlens; {self.window_attn.attn_backend.get_name()} does not"
                )
            return True
        return bool(supported)


__all__ = [
    "VDN_WINDOW_ROLE",
    "MiniMaxH3HybridAttention",
]
