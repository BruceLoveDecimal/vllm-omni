# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU contract for sharing text packing metadata without changing FA2 output."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.models.hy_nomas.attention import HyNomasTextAttentionKernel

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "H100"}, num_cards=1)]


def test_shared_text_packing_matches_unpad():
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA FlashAttention 2")
    flash_attn = pytest.importorskip("flash_attn")
    from flash_attn.bert_padding import pad_input, unpad_input

    torch.manual_seed(42)
    kernel = HyNomasTextAttentionKernel()
    # Different requests have different, non-prefix visibility and batch lengths.
    for valid_lengths in ((18, 37), (21, 9)):
        mask = torch.zeros(2, 128, dtype=torch.bool, device="cuda")
        for batch, length in enumerate(valid_lengths):
            mask[batch, 1 : 2 * length : 2] = True
        packing: dict[str, torch.Tensor | int] = {}
        for _ in range(2):  # Two refiner blocks share the same plan, not QKV values.
            qkv = [torch.randn(2, 128, 2, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
            packed = []
            for tensor in qkv:
                data, indices, cumulative, maximum, _ = unpad_input(tensor.flatten(2), mask)
                packed.append(data.unflatten(-1, (2, 64)))
            expected = flash_attn.flash_attn_varlen_func(
                *packed, cumulative, cumulative, maximum, maximum, causal=False
            )
            expected = pad_input(expected.flatten(1), indices, 2, 128).unflatten(-1, (2, 64))
            actual = kernel(*qkv, attn_metadata=AttentionMetadata(attn_mask=mask, extra=packing))
            assert torch.equal(actual, expected)
