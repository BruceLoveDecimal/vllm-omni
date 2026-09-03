# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The window softmax, against the mask it claims to be.

``window.py`` turns VDN's c1 mask into a handful of dense attention calls. The
oracle here is the mask written out in full and applied to one masked softmax,
which is what the decomposition has to equal - and what a mis-grouped or
mis-scattered row would visibly differ from.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.diffusion.models.minimax_h3.vdn_release import HEAD_DIM, TRANSFORM_CONFIG
from vllm_omni.diffusion.models.minimax_h3.vdn.config import (
    MiniMaxH3HybridAttentionConfig,
    MiniMaxH3HybridGeometry,
)
from vllm_omni.diffusion.models.minimax_h3.vdn.window import (
    WindowShapeClass,
    build_window_plan,
    window_bounds,
    window_softmax,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HEADS = 2
DIM = 8
TEXT = 3
AUDIO = 4
GRID = (2, 3)
TOKENS_PER_FRAME = GRID[0] * GRID[1]
SCALE = DIM**-0.5


class _DenseAttention(nn.Module):
    """A stand-in for the resolved backend: plain dense attention on [B,S,H,D]."""

    def forward(self, query, key, value, metadata=None):
        assert metadata is None, "the grouped path never sends attention metadata"
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scale=SCALE
        )
        return attended.transpose(1, 2)


class _VarlenAttention(nn.Module):
    """A stand-in for a backend that honours packed ``cu_seqlens``."""

    def forward(self, query, key, value, metadata=None):
        if metadata is None:
            return _DenseAttention().forward(query, key, value)
        cu_q = metadata.extra["cu_seqlens_q"].tolist()
        cu_k = metadata.extra["cu_seqlens_k"].tolist()
        out = torch.empty_like(query)
        for start_q, stop_q, start_k, stop_k in zip(cu_q[:-1], cu_q[1:], cu_k[:-1], cu_k[1:], strict=True):
            out[:, start_q:stop_q] = _DenseAttention().forward(
                query[:, start_q:stop_q], key[:, start_k:stop_k], value[:, start_k:stop_k]
            )
        return out


def _geometry(num_frames: int, *, pad: int = 5) -> MiniMaxH3HybridGeometry:
    video_start = TEXT + AUDIO
    used = video_start + num_frames * TOKENS_PER_FRAME
    return MiniMaxH3HybridGeometry(
        seq_len=used + pad,
        used_len=used,
        text_start=0,
        text_len=TEXT,
        video_start=video_start,
        num_frames=num_frames,
        frame_height=GRID[0],
        frame_width=GRID[1],
    )


def _config(**overrides) -> MiniMaxH3HybridAttentionConfig:
    return MiniMaxH3HybridAttentionConfig.from_transform_config(
        TRANSFORM_CONFIG, attention_head_dim=HEAD_DIM, **overrides
    )


def _reference(query, key, value, geometry, config):
    """The mask, written out, applied as one masked softmax per head.

    Keep every pair unless BOTH sides are video and the key's frame falls
    outside the query frame's chunk window - with frames 0 and F-1 dense as both
    rows and columns, and the alignment padding attended by nobody.
    """
    bounds = window_bounds(geometry.num_frames, chunk=config.chunk, radius=config.radius)
    rows = geometry.seq_len
    frame_of = torch.full((rows,), -1, dtype=torch.long)
    for frame in range(geometry.num_frames):
        start, stop = geometry.frame_rows(frame)
        frame_of[start:stop] = frame

    keep = torch.zeros(rows, rows, dtype=torch.bool)
    anchors = {0, geometry.num_frames - 1}
    for q_row in range(geometry.used_len):
        q_frame = int(frame_of[q_row])
        for k_row in range(geometry.used_len):
            k_frame = int(frame_of[k_row])
            if q_frame < 0 or k_frame < 0:
                keep[q_row, k_row] = True
                continue
            low, high = bounds[q_frame]
            keep[q_row, k_row] = low <= k_frame <= high or k_frame in anchors or q_frame in anchors

    out = torch.zeros_like(query)
    for head in range(query.shape[1]):
        scores = (query[:, head].float() @ key[:, head].float().T) * SCALE
        scores = scores.masked_fill(~keep, float("-inf"))
        weights = torch.softmax(scores[: geometry.used_len], dim=-1)
        out[: geometry.used_len, head] = weights @ value[:, head].float()
    return out


def _qkv(geometry, seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    shape = (geometry.seq_len, HEADS, DIM)
    return tuple(torch.randn(*shape, generator=generator, dtype=torch.float32) for _ in range(3))


@pytest.mark.parametrize("num_frames", [16, 17, 21, 32, 37])
@pytest.mark.parametrize("group_batch", [1, 4])
def test_the_decomposition_equals_the_mask_it_claims_to_be(num_frames, group_batch):
    geometry = _geometry(num_frames)
    config = _config(window_group_batch=group_batch)
    plan = build_window_plan(geometry, config, torch.device("cpu"))
    query, key, value = _qkv(geometry)

    out = window_softmax(_DenseAttention(), query, key, value, plan, group_batch=group_batch)

    torch.testing.assert_close(
        out[: geometry.used_len],
        _reference(query, key, value, geometry, config)[: geometry.used_len],
        rtol=1e-5,
        atol=1e-5,
    )


def test_the_varlen_path_agrees_with_the_grouped_one():
    """Two arrangements of the same plan; a backend choice must not move output."""
    geometry = _geometry(32)
    config = _config()
    plan = build_window_plan(geometry, config, torch.device("cpu"))
    query, key, value = _qkv(geometry)

    grouped = window_softmax(_DenseAttention(), query, key, value, plan)
    varlen = window_softmax(_VarlenAttention(), query, key, value, plan, varlen=True)

    torch.testing.assert_close(grouped, varlen, rtol=1e-5, atol=1e-5)


def test_padding_rows_are_zeroed_rather_than_attended():
    geometry = _geometry(21, pad=7)
    plan = build_window_plan(geometry, _config(), torch.device("cpu"))
    query, key, value = _qkv(geometry)

    out = window_softmax(_DenseAttention(), query, key, value, plan)

    assert torch.count_nonzero(out[geometry.used_len :]) == 0


def test_every_content_row_belongs_to_exactly_one_call():
    geometry = _geometry(32)
    plan = build_window_plan(geometry, _config(), torch.device("cpu"))

    rows = plan.dense_q_index.tolist()
    for shape in plan.classes:
        for group in range(shape.num_groups):
            claimed = shape.query_rows(slice(group, group + 1))
            if isinstance(claimed, tuple):
                rows.extend(range(claimed[0], claimed[0] + claimed[1]))
            else:
                rows.extend(claimed.tolist())

    assert sorted(rows) == list(range(geometry.used_len))
    assert plan.covered_rows == geometry.used_len


def test_the_dense_leg_is_the_globals_and_the_two_anchor_frames():
    geometry = _geometry(32)
    plan = build_window_plan(geometry, _config(), torch.device("cpu"))

    expected = list(range(geometry.video_start))
    expected += list(range(*geometry.frame_rows(0)))
    expected += list(range(*geometry.frame_rows(geometry.num_frames - 1)))

    assert plan.dense_q_index.tolist() == sorted(expected)


def test_the_mask_collapses_to_a_handful_of_shapes():
    """The point of the decomposition: a whole clip is a few dense calls.

    Five classes at c1 - the opening chunk, the one that still reaches frame 0,
    the interior, the one that reaches the last frame, and the short tail.
    """
    for num_frames in (32, 102):
        plan = build_window_plan(_geometry(num_frames), _config(), torch.device("cpu"))
        assert plan.num_shape_classes == 5, num_frames
    # 102 frames is the 14.4 s / 768p workload: 21 chunks, five shapes.
    plan = build_window_plan(_geometry(102), _config(), torch.device("cpu"))
    assert sum(shape.num_groups for shape in plan.classes) == 21


def test_window_bounds_are_chunk_aligned_not_frame_centred():
    bounds = window_bounds(12, chunk=5, radius=1)

    # Frames 5 and 9 share chunk 1, so they share a window: the VAE coded that
    # chunk as one unit, and a frame that saw half of it would see a fragment.
    assert bounds[5] == bounds[9] == (0, 14)
    assert bounds[0] == bounds[4] == (-5, 9)
    # The last chunk is short when the frame count does not divide; still whole.
    assert bounds[10] == bounds[11] == (5, 19)


def test_a_shape_class_can_gather_query_rows_that_are_not_contiguous():
    """Groups of one shape need not be adjacent; the gather path serves those."""
    rows = torch.tensor([2, 3, 9, 10], dtype=torch.long)
    shape = WindowShapeClass(
        q_len=2,
        kv_len=1,
        num_groups=2,
        kv_index=torch.tensor([0, 1], dtype=torch.long),
        q_index=rows,
    )
    query = torch.arange(12, dtype=torch.float32).view(12, 1, 1)

    batch = shape.query_batch(query, slice(0, 2))

    assert batch.shape == (2, 2, 1, 1)
    assert batch.flatten().tolist() == [2.0, 3.0, 9.0, 10.0]
    assert shape.query_rows(slice(1, 2)).tolist() == [9, 10]
