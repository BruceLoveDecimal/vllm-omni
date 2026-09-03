# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The VDN window softmax, as a union of dense attentions.

VDN's mask is not arbitrary sparsity. Every kept pair lies in one of a few dense
rectangles::

  video query in chunk c  ->  keys in chunks [c - r, c + r], plus the two anchor
                              frames, plus every text/audio row
  anchor-frame query      ->  every row (the anchors see the whole sequence)
  text/audio query        ->  every row

So the query rows split into groups whose kept key set is identical, and each
group is a plain dense attention. Groups that agree on both lengths batch into
one call, which turns the whole mask into a handful of ordinary attention
launches.

Written this way on purpose rather than as a block-sparse kernel or a packed
varlen call: it runs on **every** backend this repository resolves, including
the CUDNN_ATTN that Blackwell consumer cards (sm_120) default to and the Sage
backends that refuse a mask outright. A packed-varlen path exists underneath the
same plan for backends that isolate multi-document cu_seqlens, but nothing
depends on it being available.

The arithmetic is the same in both: each query's softmax spans exactly its kept
set in one pass, so the result differs from a masked reference by reduction
order only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

from .config import MiniMaxH3HybridAttentionConfig, MiniMaxH3HybridGeometry


def window_bounds(num_frames: int, *, chunk: int, radius: int) -> list[tuple[int, int]]:
    """Per-frame inclusive chunk-aligned window ``[lo, hi]``, unclamped.

    Frame ``t`` belongs to chunk ``t // chunk`` and sees whole chunks
    ``[c - radius, c + radius]``. The window is a property of the CHUNK, not of
    the frame, which is the point: the VAE encodes every ``chunk`` latent frames
    independently, so a frame that saw only part of a neighbouring chunk would
    see a fragment of a unit that was never coded as separable. A centred
    per-frame window cannot express that for every frame at once.

    The last chunk is short when ``chunk`` does not divide ``num_frames``
    (102 = 20*5 + 2), which is fine - it is still a whole chunk.
    """
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    return [(((t // chunk) - radius) * chunk, ((t // chunk) + radius + 1) * chunk - 1) for t in range(num_frames)]


@dataclass(frozen=True)
class WindowShapeClass:
    """Window groups that agree on both lengths, so one call serves them all.

    ``q_start`` is set when the class's query rows are one contiguous run, which
    is the common case (the interior chunks are consecutive frames): the batch
    is then a view of the packed rows rather than a gather.
    """

    q_len: int
    kv_len: int
    num_groups: int
    kv_index: torch.Tensor
    q_index: torch.Tensor | None = None
    q_start: int | None = None

    def query_batch(self, query: torch.Tensor, group_slice: slice) -> torch.Tensor:
        """``[groups, q_len, heads, dim]`` for the groups in ``group_slice``."""
        groups = group_slice.stop - group_slice.start
        if self.q_start is not None:
            start = self.q_start + group_slice.start * self.q_len
            return query.narrow(0, start, groups * self.q_len).view(groups, self.q_len, *query.shape[1:])
        assert self.q_index is not None
        rows = self.q_index.narrow(0, group_slice.start * self.q_len, groups * self.q_len)
        return query.index_select(0, rows).view(groups, self.q_len, *query.shape[1:])

    def query_rows(self, group_slice: slice) -> torch.Tensor | tuple[int, int]:
        """Where this batch's output belongs, as rows or a contiguous range."""
        groups = group_slice.stop - group_slice.start
        if self.q_start is not None:
            return self.q_start + group_slice.start * self.q_len, groups * self.q_len
        assert self.q_index is not None
        return self.q_index.narrow(0, group_slice.start * self.q_len, groups * self.q_len)

    def key_batch(self, key: torch.Tensor, group_slice: slice) -> torch.Tensor:
        groups = group_slice.stop - group_slice.start
        rows = self.kv_index.narrow(0, group_slice.start * self.kv_len, groups * self.kv_len)
        return key.index_select(0, rows).view(groups, self.kv_len, *key.shape[1:])


@dataclass(frozen=True)
class WindowPlan:
    """Every dense call the mask decomposes into, for one packed geometry."""

    bounds: tuple[tuple[int, int], ...]
    used_len: int
    seq_len: int
    dense_q_index: torch.Tensor
    classes: tuple[WindowShapeClass, ...]

    @property
    def num_shape_classes(self) -> int:
        return len(self.classes)

    @property
    def covered_rows(self) -> int:
        return int(self.dense_q_index.numel()) + sum(cls.q_len * cls.num_groups for cls in self.classes)


def build_window_plan(
    geometry: MiniMaxH3HybridGeometry,
    config: MiniMaxH3HybridAttentionConfig,
    device: torch.device,
) -> WindowPlan:
    """Decompose the c1 mask for one packed sequence into dense calls."""
    num_frames = geometry.num_frames
    bounds = window_bounds(num_frames, chunk=config.chunk, radius=config.radius)
    anchors = {0, num_frames - 1}

    # Rows that are not video at all: text and audio. Both directions stay
    # dense for them, so they are keys for every group and queries for one.
    global_ranges = _merge([(0, geometry.video_start), (geometry.video_end, geometry.used_len)])

    dense_ranges = _merge(global_ranges + [geometry.frame_rows(frame) for frame in sorted(anchors)])
    dense_q_index = _index_of(dense_ranges, device)

    groups: list[list[int]] = []
    for frame in range(num_frames):
        if frame in anchors:
            continue
        if groups and groups[-1][-1] == frame - 1 and bounds[groups[-1][-1]] == bounds[frame]:
            groups[-1].append(frame)
        else:
            groups.append([frame])

    by_shape: dict[tuple[int, int], list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]] = {}
    for frames in groups:
        low, high = bounds[frames[0]]
        kv_frames = sorted(set(range(max(low, 0), min(high + 1, num_frames))) | anchors)
        q_ranges = _merge([geometry.frame_rows(frame) for frame in frames])
        kv_ranges = _merge(global_ranges + [geometry.frame_rows(frame) for frame in kv_frames])
        q_len = sum(stop - start for start, stop in q_ranges)
        kv_len = sum(stop - start for start, stop in kv_ranges)
        by_shape.setdefault((q_len, kv_len), []).append((q_ranges, kv_ranges))

    classes = []
    for (q_len, kv_len), members in by_shape.items():
        q_starts = [ranges[0][0] for ranges, _ in members]
        contiguous = all(
            len(ranges) == 1 and start == q_starts[0] + position * q_len
            for position, ((ranges, _), start) in enumerate(zip(members, q_starts, strict=True))
        )
        kv_index = torch.cat([_index_of(kv_ranges, device) for _, kv_ranges in members])
        classes.append(
            WindowShapeClass(
                q_len=q_len,
                kv_len=kv_len,
                num_groups=len(members),
                kv_index=kv_index,
                q_index=None if contiguous else torch.cat([_index_of(ranges, device) for ranges, _ in members]),
                q_start=q_starts[0] if contiguous else None,
            )
        )

    plan = WindowPlan(
        bounds=tuple(bounds),
        used_len=geometry.used_len,
        seq_len=geometry.seq_len,
        dense_q_index=dense_q_index,
        classes=tuple(classes),
    )
    if plan.covered_rows != geometry.used_len:
        raise ValueError(
            f"the window decomposition covers {plan.covered_rows} of {geometry.used_len} content rows; "
            "every query row must belong to exactly one call"
        )
    return plan


def window_softmax(
    attention: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: WindowPlan,
    *,
    group_batch: int = 4,
    varlen: bool = False,
) -> torch.Tensor:
    """``[rows, heads, dim]`` -> the window softmax over the same rows.

    Padding rows belong to no group and are zeroed rather than attended: they
    carry no content, and a backend handed them would still have to produce
    something for them.
    """
    out = torch.empty_like(query)
    if plan.used_len < plan.seq_len:
        out.narrow(0, plan.used_len, plan.seq_len - plan.used_len).zero_()

    # The globals and the two anchor frames see the whole sequence, so their
    # keys are the content rows themselves - no gather.
    dense_key = key.narrow(0, 0, plan.used_len)
    dense_value = value.narrow(0, 0, plan.used_len)
    dense_query = query.index_select(0, plan.dense_q_index)
    dense_out = attention(dense_query.unsqueeze(0), dense_key.unsqueeze(0), dense_value.unsqueeze(0))
    out.index_copy_(0, plan.dense_q_index, dense_out.squeeze(0))

    if varlen:
        _window_softmax_varlen(attention, query, key, value, plan, out)
        return out

    for shape in plan.classes:
        for start in range(0, shape.num_groups, group_batch):
            group_slice = slice(start, min(start + group_batch, shape.num_groups))
            attended = attention(
                shape.query_batch(query, group_slice),
                shape.key_batch(key, group_slice),
                shape.key_batch(value, group_slice),
            )
            attended = attended.reshape(-1, *query.shape[1:])
            rows = shape.query_rows(group_slice)
            if isinstance(rows, tuple):
                out.narrow(0, rows[0], rows[1]).copy_(attended)
            else:
                out.index_copy_(0, rows, attended)
    return out


def _window_softmax_varlen(
    attention: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: WindowPlan,
    out: torch.Tensor,
) -> None:
    """Every window group in one packed variable-length call.

    Available only on a backend that consumes ``cu_seqlens`` as a genuine
    block-diagonal plan. It trades the grouped path's bounded gather for one
    that materialises every group's keys at once, which VDN measured as the
    faster arrangement on Blackwell data-centre parts.
    """
    q_rows: list[torch.Tensor] = []
    kv_rows: list[torch.Tensor] = []
    q_lens: list[int] = []
    kv_lens: list[int] = []
    for shape in plan.classes:
        for group in range(shape.num_groups):
            group_slice = slice(group, group + 1)
            rows = shape.query_rows(group_slice)
            if isinstance(rows, tuple):
                q_rows.append(torch.arange(rows[0], rows[0] + rows[1], device=query.device))
            else:
                q_rows.append(rows)
            kv_rows.append(shape.kv_index.narrow(0, group * shape.kv_len, shape.kv_len))
            q_lens.append(shape.q_len)
            kv_lens.append(shape.kv_len)

    q_index = torch.cat(q_rows)
    kv_index = torch.cat(kv_rows)
    cu_q = _cumulative(q_lens, query.device)
    cu_k = _cumulative(kv_lens, query.device)
    metadata = AttentionMetadata(
        extra={
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": max(q_lens),
            "max_seqlen_k": max(kv_lens),
        }
    )
    attended = attention(
        query.index_select(0, q_index).unsqueeze(0),
        key.index_select(0, kv_index).unsqueeze(0),
        value.index_select(0, kv_index).unsqueeze(0),
        metadata,
    )
    out.index_copy_(0, q_index, attended.squeeze(0))


def _cumulative(lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, dtype=torch.int32, device=device)


def _merge(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort, drop empties, and join ranges that touch."""
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(span for span in ranges if span[0] < span[1]):
        if merged and merged[-1][1] >= start:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def _index_of(ranges: Sequence[tuple[int, int]], device: torch.device) -> torch.Tensor:
    if not ranges:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat([torch.arange(start, stop, device=device) for start, stop in ranges])


__all__ = [
    "WindowPlan",
    "WindowShapeClass",
    "build_window_plan",
    "window_bounds",
    "window_softmax",
]
