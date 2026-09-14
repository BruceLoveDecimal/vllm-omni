# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.solarwm_h3.layout import ChunkWindow
from vllm_omni.diffusion.models.solarwm_h3.solarwm_h3_transformer import SolarWMKVCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _commit(cache: SolarWMKVCache, chunk_index: int) -> None:
    keys = torch.full((4, 2, 3), float(chunk_index))
    cache.commit(0, chunk_index, keys, -keys)


def _history_ids(cache: SolarWMKVCache, chunk_index: int) -> tuple[list[int], list[int]]:
    keys, values = cache.history(0, ChunkWindow(chunk_index), torch.device("cpu"))
    return [int(part[0, 0, 0]) for part in keys], [int(part[0, 0, 0]) for part in values]


def test_kv_cache_returns_the_window_history_oldest_first():
    cache = SolarWMKVCache(num_layers=1, device=torch.device("cpu"))
    assert cache.history(0, ChunkWindow(0), torch.device("cpu")) == ([], [])

    # The rollout commits chunks in order and reads the window before the next commit.
    _commit(cache, 0)
    _commit(cache, 1)
    assert _history_ids(cache, 2) == ([0, 1], [0, -1])

    for chunk_index in range(2, 8):
        _commit(cache, chunk_index)
    assert _history_ids(cache, 8) == ([3, 4, 5, 6, 7], [-3, -4, -5, -6, -7])

    cache.clear()
    assert cache._keys == [None]
