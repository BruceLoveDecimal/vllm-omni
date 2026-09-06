# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bookkeeping of the SANA-WM streaming cache (window rule, sink, bytes)."""

import pytest
import torch

from vllm_omni.diffusion.models.sana_wm.streaming_cache import (
    SanaWmSoftmaxChunk,
    SanaWmStreamingCache,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _cache(num_cached_blocks: int, sink_token: bool, num_blocks: int = 2) -> SanaWmStreamingCache:
    return SanaWmStreamingCache.new(
        num_blocks=num_blocks, chunk_size=3, num_cached_blocks=num_cached_blocks, sink_token=sink_token
    )


def _commit_softmax_chunk(cache: SanaWmStreamingCache, frames: int) -> None:
    index = cache.chunks_committed
    for block in cache.blocks:
        block.softmax_chunks.append(
            SanaWmSoftmaxChunk(
                chunk_index=index,
                frames=frames,
                k_main=torch.zeros(1, frames, 2, 4),
                v_main=torch.zeros(1, frames, 2, 4),
                k_cam=torch.zeros(1, frames, 2, 4),
                v_cam=torch.zeros(1, frames, 2, 4),
            )
        )
    cache.commit(frames)


@pytest.mark.parametrize(
    ("num_cached_blocks", "sink_token", "expected"),
    [
        # NVlabs default: chunk 0 pinned + the last (num_cached_blocks - 1) chunks.
        (2, True, {1: [0], 2: [0, 1], 3: [0, 2], 4: [0, 3]}),
        # No sink: plain sliding window of num_cached_blocks chunks.
        (2, False, {1: [0], 2: [0, 1], 3: [1, 2], 4: [2, 3]}),
        # -1 keeps everything.
        (-1, True, {1: [0], 2: [0, 1], 3: [0, 1, 2], 4: [0, 1, 2, 3]}),
        (3, True, {1: [0], 2: [0, 1], 3: [0, 1, 2], 4: [0, 2, 3], 5: [0, 3, 4]}),
    ],
)
def test_kept_chunk_indices_match_nvlabs_window(num_cached_blocks, sink_token, expected) -> None:
    cache = _cache(num_cached_blocks, sink_token)
    assert cache.kept_chunk_indices(0) == []
    for next_chunk, kept in expected.items():
        assert cache.kept_chunk_indices(next_chunk) == kept


def test_commit_trims_softmax_entries_and_tracks_frames() -> None:
    cache = _cache(num_cached_blocks=2, sink_token=True)
    _commit_softmax_chunk(cache, 4)  # chunk 0: conditioning frame + 3
    assert cache.chunks_committed == 1 and cache.frames_committed == 4
    assert [c.chunk_index for c in cache.blocks[0].softmax_chunks] == [0]
    assert cache.cached_frames() == 4

    _commit_softmax_chunk(cache, 3)  # chunk 1
    assert [c.chunk_index for c in cache.blocks[0].softmax_chunks] == [0, 1]
    assert cache.cached_frames() == 7

    _commit_softmax_chunk(cache, 3)  # chunk 2 -> window for chunk 3 is {0, 2}
    for block in cache.blocks:
        assert [c.chunk_index for c in block.softmax_chunks] == [0, 2]
    assert cache.frames_committed == 10
    assert cache.cached_frames() == 7
    assert len(cache.blocks[0].cached_main_kv()) == 2
    assert len(cache.blocks[0].cached_cam_kv()) == 2


def test_unbounded_window_never_trims() -> None:
    cache = _cache(num_cached_blocks=-1, sink_token=True)
    for frames in (4, 3, 3, 3):
        _commit_softmax_chunk(cache, frames)
    assert [c.chunk_index for c in cache.blocks[1].softmax_chunks] == [0, 1, 2, 3]
    assert cache.cached_frames() == 13


def test_nbytes_and_clear() -> None:
    cache = _cache(num_cached_blocks=2, sink_token=False, num_blocks=1)
    assert cache.nbytes() == 0
    block = cache.blocks[0]
    block.gdn_state_kv = torch.zeros(1, 2, 4, 4, dtype=torch.float32)
    block.ffn_tconv_tail = torch.zeros(1, 8, 1, 6, dtype=torch.bfloat16)
    _commit_softmax_chunk(cache, 3)
    expected = 1 * 2 * 4 * 4 * 4 + 1 * 8 * 1 * 6 * 2 + 4 * (1 * 3 * 2 * 4 * 4)
    assert cache.nbytes() == expected
    cache.clear()
    assert cache.nbytes() == 0
    assert block.softmax_chunks == []


def test_new_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        SanaWmStreamingCache.new(num_blocks=0, chunk_size=3, num_cached_blocks=2, sink_token=True)
    with pytest.raises(ValueError):
        SanaWmStreamingCache.new(num_blocks=1, chunk_size=0, num_cached_blocks=2, sink_token=True)
    cache = _cache(2, True)
    with pytest.raises(ValueError):
        cache.commit(0)
