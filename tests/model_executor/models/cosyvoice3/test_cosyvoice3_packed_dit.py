# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _attention(dim=64, heads=4, dim_head=16, dtype=torch.float32):
    attn = torch.nn.Module()
    attn.heads, attn.inner_dim = heads, heads * dim_head
    for name in ("to_q", "to_k", "to_v"):
        setattr(attn, name, torch.nn.Linear(dim, attn.inner_dim).to(dtype))
    return attn


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_fused_qkv_matches_separate_projections(dtype):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import fuse_qkv, project_qkv

    torch.manual_seed(0)
    attn = _attention(dtype=dtype)
    x = torch.randn(1, 37, 64, dtype=dtype)
    expected = (attn.to_q(x), attn.to_k(x), attn.to_v(x))
    keys = set(attn.state_dict())

    fuse_qkv(attn)
    actual = project_qkv(attn, x)

    for output, reference in zip(actual, expected):
        assert torch.equal(output, reference)
    # The modules keep views of the fused weight: no copy, unchanged checkpoint keys.
    storage = attn.qkv_weight.untyped_storage().data_ptr()
    assert all(getattr(attn, name).weight.untyped_storage().data_ptr() == storage for name in ("to_q", "to_k", "to_v"))
    assert set(attn.state_dict()) == keys
    fuse_qkv(attn)  # idempotent
    assert attn.to_q.weight.untyped_storage().data_ptr() == storage


@pytest.mark.parametrize(
    ("max_rows", "slots"),
    [(1, (1,)), (8, (1, 2, 4, 8)), (12, (1, 2, 4, 8, 12)), (32, (1, 2, 4, 8, 16, 32))],
)
def test_flow_graph_slots(max_rows, slots):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import flow_graph_slots

    assert flow_graph_slots(max_rows) == slots


def test_flow_graph_slots_rejects_zero_rows():
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import flow_graph_slots

    with pytest.raises(ValueError, match="at least 1"):
        flow_graph_slots(0)
