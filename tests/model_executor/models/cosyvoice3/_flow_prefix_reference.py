# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Isolated CPU numerical oracle for the packed prefix solver.

Run in a subprocess so stub FA3 imports and custom-op registrations cannot
contaminate GPU tests. Real convolution, RoPE, DiT blocks and Euler code execute;
only the CUDA-only attention/convolution boundaries are replaced with CPU ops.
"""

import importlib
import sys
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

root = Path(__file__).resolve().parents[4] / "vllm_omni/model_executor/models/cosyvoice3/code2wav_core"
pkg = types.ModuleType("prefix_test")
pkg.__path__ = [str(root)]
sys.modules[pkg.__name__] = pkg
stub = types.ModuleType("vllm.vllm_flash_attn")
stub.flash_attn_varlen_func = lambda *a, **kw: None
sys.modules[stub.__name__] = stub
conv = types.ModuleType("prefix_test.packed_conv")
conv.pack_conv_weight = lambda x: None
conv.packed_causal_conv_mish = lambda *a: None
sys.modules[conv.__name__] = conv
pd = importlib.import_module("prefix_test.packed_dit")
pp = importlib.import_module("prefix_test.packed_prefix")

pd.native_mish = F.mish


def attention(q, k, v, a, max_seqlen_q):
    outputs = []
    for i in range(len(a.cache_seqlens)):
        start, end = a.cu_seqlens_q[i : i + 2].tolist()
        indices = a.page_table[i, : a.cache_seqlens[i]].long()
        qq = q[:, start:end].reshape(1, -1, a.heads, a.head_dim).transpose(1, 2)
        kk = k[:, indices].reshape(1, -1, a.heads, a.head_dim).transpose(1, 2)
        vv = v[:, indices].reshape(1, -1, a.heads, a.head_dim).transpose(1, 2)
        outputs.append(F.scaled_dot_product_attention(qq, kk, vv).transpose(1, 2).flatten(2))
    return torch.cat(outputs, dim=1)


pd.ragged_attention_tensor_geometry = attention
pp.ragged_attention_tensor_geometry = attention


class Obj(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)


class Rope:
    def forward_from_seq_len(self, n):
        return torch.arange(n).reshape(1, n, 1).float().expand(1, n, 4) * 0.03, 1


def make_estimator():
    blocks = []
    for _ in range(2):
        attn = Obj(
            heads=2,
            dim_head=4,
            inner_dim=8,
            to_q=nn.Linear(8, 8),
            to_k=nn.Linear(8, 8),
            to_v=nn.Linear(8, 8),
            to_out=nn.Sequential(nn.Linear(8, 8), nn.Identity()),
        )
        blocks.append(
            Obj(
                attn=attn,
                attn_norm=Obj(linear=nn.Linear(8, 48), silu=nn.SiLU(), norm=nn.LayerNorm(8)),
                ff_norm=nn.LayerNorm(8),
                ff=nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8)),
            )
        )
    dit = Obj(
        input_embed=Obj(
            proj=nn.Linear(12, 8),
            conv_pos_embed=Obj(
                kernel_size=3, conv1=nn.Sequential(nn.Conv1d(8, 8, 3)), conv2=nn.Sequential(nn.Conv1d(8, 8, 3))
            ),
        ),
        transformer_blocks=nn.ModuleList(blocks),
        norm_out=Obj(linear=nn.Linear(8, 16), silu=nn.SiLU(), norm=nn.LayerNorm(8)),
        proj_out=nn.Linear(8, 3),
        long_skip_connection=nn.Linear(16, 8),
        rotary_embed=Rope(),
        static_chunk_size=4,
        time_embed=nn.Sequential(nn.Unflatten(0, (1, 1)), nn.Linear(1, 8)),
    )
    e = pd.PackedDiT(dit.eval())
    e.compiled_full_forward = e.forward_full
    e.compiled_prefix_forward = pp.forward_prefix
    return e


torch.manual_seed(1)
e = make_estimator()
noise = torch.randn(2, 32, 3)
mu = torch.randn(2, 32, 3)
cond = torch.randn(2, 32, 3)
spks = torch.randn(2, 3)
times = torch.tensor([0.0, 0.3, 1.0])


@torch.inference_mode()
def run(order, lengths):
    rows = pd.pack_rows(lengths, torch.device("cpu"))
    inputs = [pd.gather_rows(x[order], rows) for x in (noise, mu, cond)]
    args = (e, inputs[0], times, inputs[1], spks[order], inputs[2], rows)
    expected = pd.solve_flow_euler_packed(*args, cfg_rate=0.7, streaming=True, modulation_key=(2, "linear"))
    actual = pd.solve_flow_euler_packed(
        *args, cfg_rate=0.7, streaming=True, modulation_key=(2, "linear"), prefix_cache_bytes=1 << 20
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    print(
        "PASS", order, lengths, "max error", float((actual - expected).abs().max()), "hits", e.prefix_cache.hit_frames
    )


run([0, 1], [12, 17])
run([1, 0], [24, 20])
mu[0, 10] += 1
run([0, 1], [24, 28])
spks[0] += 1
run([0], [32])


# Dynamic fullgraph tracing uses a fake CPU FA3 op with the same shape contract.
# Its CPU implementation consumes metadata outside the captured graph.
@torch.library.custom_op("prefix_cpu::attn", mutates_args=())
def fa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor,
    pages: torch.Tensor,
    cu: torch.Tensor,
    heads: int,
    dim: int,
) -> torch.Tensor:
    return attention(
        q,
        k,
        v,
        types.SimpleNamespace(cache_seqlens=lens, page_table=pages, cu_seqlens_q=cu, heads=heads, head_dim=dim),
        0,
    )


@fa.register_fake
def _(q, k, v, lens, pages, cu, heads, dim):
    return torch.empty_like(q)


def traced_attention(q, k, v, a, max_seqlen_q):
    return fa(q, k, v, a.cache_seqlens, a.page_table, a.cu_seqlens_q, a.heads, a.head_dim)


pp.ragged_attention_tensor_geometry = traced_attention
e.compiled_prefix_forward = torch.compile(pp.forward_prefix, backend="eager", dynamic=True, fullgraph=True)
run([0, 1], [28, 31])
run([1], [32])
run([0, 1], [27, 31])  # reuse without saving duplicate prefix tensors
print("CPU numerical and fullgraph tracing checks passed")
