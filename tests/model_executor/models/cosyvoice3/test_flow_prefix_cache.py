# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for admission and exact prefix matching (no vLLM/CUDA import)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

path = (
    Path(__file__).resolve().parents[4]
    / "vllm_omni/model_executor/models/cosyvoice3/code2wav_core/flow_prefix_cache.py"
)
spec = importlib.util.spec_from_file_location("_cosyvoice_flow_prefix_cache", path)
cache_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cache_module
spec.loader.exec_module(cache_module)
FlowPrefixCache = cache_module.FlowPrefixCache
FlowPrefixEntry = cache_module.FlowPrefixEntry
FlowPrefixStep = cache_module.FlowPrefixStep


def entry(signature):
    length = signature.shape[0]
    return FlowPrefixEntry(signature.clone(), [FlowPrefixStep((torch.zeros(2, length, 8),), torch.zeros(2, length, 3))])


def test_prefix_matching_rounds_down_and_isolates_reordered_rows():
    cache = FlowPrefixCache(1 << 20)
    first, second = torch.randn(16, 12), torch.randn(16, 12)
    a, b = entry(first), entry(second)
    cache.put(a)
    cache.put(b)
    grown = torch.cat((first, torch.randn(4, 12)))
    changed = torch.cat((second, torch.randn(4, 12)))
    changed[10, 0] += 1
    matches = cache.lookup([changed, grown, torch.randn(20, 12)], chunk_size=4)
    assert matches[0][0] is b and matches[0][1] == 8
    assert matches[1][0] is a and matches[1][1] == 16
    assert matches[2] == (None, 0)
    assert cache.hit_frames == 24
    assert cache.query_frames == 36


@pytest.mark.parametrize("column", [0, 3, 6, 9])
def test_noise_mu_condition_and_speaker_changes_invalidate(column):
    cache = FlowPrefixCache(1 << 20)
    signature = torch.randn(16, 12)
    cache.put(entry(signature))
    changed = signature.clone()
    changed[0, column] += 1
    assert cache.lookup([changed], 4) == [(None, 0)]


def test_identical_request_keeps_nonempty_suffix_and_short_rows_miss():
    cache = FlowPrefixCache(1 << 20)
    signature = torch.randn(16, 12)
    cache.put(entry(signature))
    matches = cache.lookup([signature, signature[:3]], 4)
    assert matches[0][1] == 12
    assert matches[1] == (None, 0)


def test_lru_budget_schedule_reset_and_oversized_admission():
    a, b, c = [entry(torch.randn(16, 12)) for _ in range(3)]
    cache = FlowPrefixCache(a.nbytes * 2)
    cache.reset((10, "cosine"))
    cache.put(a)
    cache.put(b)
    cache.lookup([a.signature], 4)
    cache.put(c)
    assert list(cache.entries.values())[0] is a
    assert list(cache.entries.values())[1] is c
    assert cache.nbytes == a.nbytes * 2
    cache.put(entry(torch.randn(100, 12)))
    assert len(cache.entries) == 2
    cache.reset((10, "cosine"))
    assert len(cache.entries) == 2
    cache.reset((4, "cosine"))
    assert not cache.entries and cache.nbytes == 0


def test_entry_limit_and_zero_budget():
    cache = FlowPrefixCache(1 << 20, max_entries=1)
    cache.put(entry(torch.ones(4, 12)))
    cache.put(entry(torch.zeros(4, 12)))
    assert len(cache.entries) == 1
    disabled = FlowPrefixCache(0)
    disabled.put(entry(torch.ones(4, 12)))
    assert disabled.nbytes == 0


def test_packed_prefix_solver_cpu_numerical_reference():
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("_flow_prefix_reference.py"))],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_growing_entry_replaces_old_storage_without_duplicate_retention():
    cache = FlowPrefixCache(1 << 20)
    old = entry(torch.randn(8, 12))
    grown = entry(torch.cat((old.signature, torch.randn(8, 12))))
    cache.put(old)
    cache.put(grown, replaces=old)
    assert len(cache.entries) == 1
    assert cache.nbytes == grown.nbytes
