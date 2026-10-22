# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prefix-cache attention on CUDA must follow the packed Flow's FlashAttention
version and read the paged prefix exactly as the eager reference does."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.cosyvoice3 import runtime
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core import flow_prefix_cache
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA"),
]

HEADS, HEAD_DIM, BLOCKS = 2, 64, 6


@pytest.mark.parametrize("fa_version", [2, 3])
@torch.inference_mode()
def test_prefix_attention_matches_reference(monkeypatch, fa_version):
    if not runtime._fa_version_supported(fa_version):
        pytest.skip(f"FlashAttention {fa_version} is not supported on this GPU")
    page = flow_prefix_cache.PAGE_FRAMES
    device, dtype = torch.device("cuda"), torch.bfloat16
    torch.manual_seed(0)
    pool = torch.randn((1, 2, BLOCKS, page, HEADS, HEAD_DIM), device=device, dtype=dtype)
    monkeypatch.setattr(
        flow_prefix_cache, "_ACTIVE_CACHE", SimpleNamespace(kv=pool, estimator=SimpleNamespace(fa_version=fa_version))
    )
    # Segment 0 appends 4 frames to a 16-frame prefix (pages 1, 2); segment 1
    # writes its first 9 frames into page 3.
    page_table = torch.tensor([[1, 2], [3, 0]], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([page + 4, 9], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 4, 13], dtype=torch.int32, device=device)
    slots = torch.cat((2 * page + torch.arange(4), 3 * page + torch.arange(9))).to(device)
    q, k, v = (torch.randn((13, HEADS, HEAD_DIM), device=device, dtype=dtype) for _ in range(3))

    output = flow_prefix_cache.prefix_attention(q, k, v, 0, slots, cache_seqlens, page_table, cu_seqlens_q, 9)

    key_cache, value_cache = pool[0]
    expected = flow_prefix_cache.paged_attention_reference(
        q, key_cache, value_cache, cache_seqlens, page_table, cu_seqlens_q
    )
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
