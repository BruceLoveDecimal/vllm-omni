# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0.5's fused Triton kernels against the eager ops they replace, one kernel at a time.

Each kernel runs on the same inputs as its eager counterpart, in both serving
dtypes and in the deployed mixed-dtype layout: bfloat16 GEMM weights, K/V and
GEMM inputs, with the float32 residual stream, norms and head that
``_to_bfloat16_for_inference`` keeps. The shapes are the deployed ones: a
968-token prefix on the Gemma 2B widths and a 50-token action chunk on the
Gemma 300M ones, with 8 query heads, 1 KV head and head_dim 256. The action
expert's GEMM kernels also run a 256-token chunk: every chunk length uses the
same tiles.

The kernels perform eager's operations in its order and round where it
rounds, so what a test expects depends on whether a kernel reorders a
reduction:

* It does not: bit-exact. That is the residual sums and every epilogue a
  GEMM feeds (the V slot writes, the gated residual, the GELU product) when
  the GEMM is cuBLAS's: float32 serves that way, and the bfloat16 action
  expert is run that way here too to pin its rounding points.
* It does (a Triton GEMM's accumulation, the RMS variance, softmax sums), or
  it is RoPE, whose eager rotary kernel may contract its products into FMAs:
  the result may differ where that flips a rounding, and must be as accurate
  as eager against a float64 evaluation of the same math: within 1.25x
  eager's relative L2 error and 1.5x its largest error. Eager's cuBLAS
  split-K GEMMs round twice in bfloat16, so the fused path is often the more
  accurate one.

The eager ops are π0.5's own modules on vLLM's layers (``Pi05GemmaAttention``,
``Pi05GemmaMLP``, ``RowParallelLinear``, ``GemmaRMSNorm``) and
``Pi05AdaRMSNorm.forward_native``: on CUDA ``Pi05AdaRMSNorm`` itself dispatches
a batch-1 call to the kernel under test.

Needs a CUDA GPU; no checkpoint. ``-s`` prints each fused kernel's max absolute
and relative L2 error against its eager op::

    python -m pytest tests/diffusion/models/pi05/test_pi05_fused_kernels.py -v -s
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from transformers import GemmaConfig
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_omni.diffusion.models.pi05 import modeling_pi05 as M

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="The fused kernels need a CUDA GPU."),
    pytest.mark.skipif(not M.HAS_TRITON, reason="The fused kernels need Triton."),
]

DTYPES = [torch.float32, torch.bfloat16]
DEVICE = "cuda"
HEADS, HEAD_DIM = 8, 256
# (tokens, width, MLP width): the prefix on Gemma 2B, the action chunk on Gemma 300M.
SHAPES = {"prefix": (968, 2048, 16384), "suffix": (50, 1024, 4096)}
# Action-chunk lengths for the GEMM kernels: the deployed one and a long one. A
# chunk of 256 tokens or more once looked up tiles that did not exist.
CHUNKS = [SHAPES["suffix"][0], 256]


def _dtype_id(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _generator(seed: int = 0) -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _randn(*shape, dtype=torch.float32, scale=1.0, seed=0) -> torch.Tensor:
    return (torch.randn(*shape, device=DEVICE, generator=_generator(seed)) * scale).to(dtype)


def _gemma_config(width: int, mlp_width: int) -> GemmaConfig:
    return GemmaConfig(
        hidden_size=width,
        intermediate_size=mlp_width,
        num_attention_heads=HEADS,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        hidden_act="gelu_pytorch_tanh",
        num_hidden_layers=1,
    )


def _randomize(module: nn.Module, seed: int) -> nn.Module:
    generator = torch.Generator().manual_seed(seed)
    for param in module.parameters():
        param.data = torch.randn(param.shape, generator=generator) * param.shape[-1] ** -0.5
    return module


def _proj_dtype_module(module: nn.Module, dtype: torch.dtype) -> nn.Module:
    """GEMM weights in the serving dtype, as ``_set_inference_dtype`` lays them out."""
    return module.to(device=DEVICE, dtype=dtype)


_DTYPE_NAMES = {torch.float32: "fp32", torch.bfloat16: "bf16"}
_MODULATION_NOTE = "The scale, shift and gate modulation is still computed by the torch Linear norm.dense, as in eager."


def _print_vs_eager(actual, expected, kernel: str, eager: str, dtype: torch.dtype, note: str) -> None:
    """Print the fused kernel's distance from its eager op (shown with ``-s``)."""
    diff = actual.double() - expected.double()
    max_abs = diff.abs().max().item()
    rel_l2 = (diff.norm() / expected.double().norm()).item()
    print(
        f"\nFused {kernel} kernel versus eager {eager} (dtype={_DTYPE_NAMES[dtype]}): "
        f"max absolute error = {max_abs:.3e}; relative L2 error = {rel_l2:.3e}." + (f" {note}" if note else "")
    )


def _assert_as_accurate_as_eager(
    actual: torch.Tensor,
    expected: torch.Tensor,
    reference: torch.Tensor,
    kernel: str,
    eager: str,
    dtype: torch.dtype,
    note: str = "",
) -> None:
    """``actual`` (fused) is as close to the float64 ``reference`` as ``expected`` (eager) is."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    _print_vs_eager(actual, expected, kernel, eager, dtype, note)
    reference = reference.double()

    def errors(t):
        diff = t.double() - reference
        return (diff.norm() / reference.norm()).item(), diff.abs().max().item()

    fused_rel, fused_max = errors(actual)
    eager_rel, eager_max = errors(expected)
    assert fused_rel <= 1.25 * eager_rel and fused_max <= 1.5 * eager_max, (
        f"fused error vs float64: rel {fused_rel:.3e}, max {fused_max:.3e}; "
        f"eager: rel {eager_rel:.3e}, max {eager_max:.3e}"
    )


def _assert_bit_exact(
    actual: torch.Tensor, expected: torch.Tensor, kernel: str, eager: str, dtype: torch.dtype, note: str = ""
) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    _print_vs_eager(actual, expected, kernel, eager, dtype, note)
    mismatches = int((actual != expected).sum())
    assert mismatches == 0, f"{mismatches} of {actual.numel()} elements differ from eager"


def _prefix_pad_masks(seq_len: int) -> torch.Tensor:
    """One real camera, two missing ones, then 120 live text tokens: ``embed_prefix``'s layout."""
    pad = torch.zeros(1, seq_len, dtype=torch.bool, device=DEVICE)
    pad[:, :256] = True
    pad[:, 768 : 768 + 120] = True
    return pad


# ── Norms: the RMS variance is a reordered reduction ──────────────────
def _rms_norm_reference(x: torch.Tensor, eps: float, weight=None, scale=None, shift=None) -> torch.Tensor:
    x = x.double()
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return normed * (1.0 + weight.double())
    return normed * (1.0 + scale.double()) + shift.double()


@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("residual_add", [False, True], ids=["norm", "residual_add_norm"])
def test_rms_norm_matches_gemma_rms_norm(dtype, residual_add):
    """``_match(norm([sublayer_out +] residual), next_proj)`` at the prefix's
    shape (the served prefix keeps eager's norm): the residual sum is
    bit-exact, the norm as accurate as eager's."""
    tokens, width, _ = SHAPES["prefix"]
    norm = _randomize(GemmaRMSNorm(width), seed=3).to(DEVICE)  # float32 in both layouts
    residual = _randn(tokens, width, scale=4.0, seed=4)
    sublayer_out = _randn(tokens, width, dtype=dtype, scale=2.0, seed=5) if residual_add else None

    summed = sublayer_out + residual if residual_add else residual
    expected = norm(summed).to(dtype)

    actual = torch.empty(tokens, width, dtype=dtype, device=DEVICE)
    summed_out = torch.full_like(residual, float("nan")) if residual_add else None
    M._fused_rms_norm(residual, norm, actual, add=sublayer_out, res_out=summed_out)
    kernel = "residual-add RMSNorm" if residual_add else "RMSNorm"
    if residual_add:
        _assert_bit_exact(summed_out, summed, kernel, "sublayer_out + residual", dtype)
    _assert_as_accurate_as_eager(
        actual,
        expected,
        _rms_norm_reference(summed, norm.variance_epsilon, weight=norm.weight),
        kernel,
        "GemmaRMSNorm",
        dtype,
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
def test_final_rms_norm_keeps_the_residual_dtype(dtype):
    """The final Gemma norm at the prefix's shape returns the residual stream's
    dtype, as eager's does (the served prefix keeps eager's final norm)."""
    tokens, width, _ = SHAPES["prefix"]
    norm = _randomize(GemmaRMSNorm(width), seed=6).to(DEVICE)
    residual = _randn(tokens, width, scale=4.0, seed=7)
    mlp_out = _randn(tokens, width, dtype=dtype, seed=8)

    summed = mlp_out + residual
    expected = norm(summed)
    actual = torch.empty_like(residual)
    M._fused_rms_norm(residual, norm, actual, add=mlp_out)
    reference = _rms_norm_reference(summed, norm.variance_epsilon, weight=norm.weight)
    _assert_as_accurate_as_eager(
        actual, expected, reference, "residual-add final RMSNorm", "GemmaRMSNorm (model.norm)", dtype
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
def test_adarms_norm_matches_pi05_adarms_norm(dtype):
    """The action expert's ``_match(Pi05AdaRMSNorm(x, cond)[0], next_proj)``."""
    tokens, width, _ = SHAPES["suffix"]
    norm = _randomize(M.Pi05AdaRMSNorm(width, cond_dim=width), seed=9).to(DEVICE)
    x = _randn(tokens, width, scale=4.0, seed=10)
    cond = _randn(1, width, seed=11)

    expected = norm.forward_native(x[None], cond)[0][0].to(dtype)
    modulation = M._adarms_modulation(norm, cond)
    actual = torch.empty(tokens, width, dtype=dtype, device=DEVICE)
    M._fused_rms_norm(x, norm, actual, modulation=modulation)
    scale, shift, _ = modulation.chunk(3)
    reference = _rms_norm_reference(x, norm.eps, scale=scale, shift=shift)
    _assert_as_accurate_as_eager(actual, expected, reference, "AdaRMSNorm", "Pi05AdaRMSNorm", dtype, _MODULATION_NOTE)


def test_adarms_norm_custom_op_runs_the_kernel_for_batch_one():
    """On CUDA ``Pi05AdaRMSNorm`` is the kernel for a batch-1 call and its
    eager formula for any other; both return the same gate."""
    tokens, width, _ = SHAPES["suffix"]
    norm = _randomize(M.Pi05AdaRMSNorm(width, cond_dim=width), seed=9).to(DEVICE)
    x = _randn(1, tokens, width, scale=4.0, seed=10)
    cond = _randn(1, width, seed=11)

    normed, gate = norm(x, cond)
    kernel_out = torch.empty(tokens, width, device=DEVICE)
    M._fused_rms_norm(x[0], norm, kernel_out, modulation=M._adarms_modulation(norm, cond))
    assert torch.equal(normed[0], kernel_out)
    eager_normed, eager_gate = norm.forward_native(x, cond)
    assert torch.equal(gate.expand_as(eager_gate), eager_gate)
    torch.testing.assert_close(normed, eager_normed)

    batch_x, batch_cond = x.expand(2, -1, -1), cond.expand(2, -1)
    batch_normed, _ = norm(batch_x, batch_cond)
    assert torch.equal(batch_normed, norm.forward_native(batch_x, batch_cond)[0])


# ── GEMMs with their epilogues ─────────────────────────────────────────
@pytest.fixture(params=["cublas", "triton"])
def gemm(request, monkeypatch):
    """Which GEMM feeds the epilogues: cuBLAS (float32's, and bfloat16 forced
    onto it to pin its rounding points) or the Triton one bfloat16 serves with."""
    if request.param == "cublas":
        monkeypatch.setattr(M, "_TRITON_GEMM_DTYPES", ())
    return request.param


def _check_gemm_epilogue(
    gemm: str, dtype: torch.dtype, actual, expected, reference, kernel: str, eager: str, cublas_note: str
) -> None:
    """``cublas_note`` says what stays in torch when cuBLAS runs the GEMM."""
    if gemm == "cublas":
        _assert_bit_exact(actual, expected, kernel, eager, dtype, cublas_note)
    else:
        _assert_as_accurate_as_eager(actual, expected, reference, kernel, eager, dtype)


def _skip_unserved(gemm: str, dtype: torch.dtype) -> None:
    if gemm == "triton" and dtype not in M._TRITON_GEMM_DTYPES:
        pytest.skip(f"{dtype} GEMMs stay on cuBLAS")


def _rope64(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Gemma's half-split RoPE in float64; ``x`` ``(S, heads, D)``, ``cos``/``sin`` ``(S, D // 2)``."""
    lo, hi = x.double().chunk(2, dim=-1)
    cos, sin = cos.double()[:, None], sin.double()[:, None]
    return torch.cat([lo * cos - hi * sin, hi * cos + lo * sin], dim=-1)


@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
def test_qkv_rope_matches_eager(gemm, dtype):
    """The packed ``qkv_proj`` then RoPE, as ``Pi05GemmaAttention.project`` runs
    them on a denoising step's chunk; K and V land in their workspace rows and
    nowhere else."""
    _skip_unserved(gemm, dtype)
    tokens, width, mlp_width = SHAPES["suffix"]
    attn = M.Pi05GemmaAttention(_gemma_config(width, mlp_width), None, prefix="self_attn")
    attn = _proj_dtype_module(_randomize(attn, seed=12), dtype)
    x = _randn(tokens, width, dtype=dtype, seed=13)
    position_ids = 376 + torch.arange(tokens, device=DEVICE)[None]
    cos, sin = M.Pi05GemmaRotaryEmbedding(HEAD_DIM).to(DEVICE)(position_ids)

    q, k, v = attn.project(x[None], cos, sin)
    # Eager's table, rounded to the projection dtype: an input of the kernel, not part of its math.
    cos, sin = cos[0].to(dtype), sin[0].to(dtype)
    w_q, w_k, w_v = attn.qkv_proj.weight.double().split([attn.q_size, attn.kv_size, attn.kv_size])
    x64 = x.double()
    q64 = _rope64((x64 @ w_q.t()).view(tokens, HEADS, HEAD_DIM), cos, sin)
    k64 = _rope64((x64 @ w_k.t()).view(tokens, 1, HEAD_DIM), cos, sin)
    v64 = x64 @ w_v.t()

    offset = 7
    key_cache = torch.full((offset + tokens + 5, HEAD_DIM), float("nan"), dtype=dtype, device=DEVICE)
    value_cache = torch.full_like(key_cache, float("nan"))
    q_out = torch.empty(HEADS, tokens, HEAD_DIM, dtype=dtype, device=DEVICE)
    rows = slice(offset, offset + tokens)
    M._fused_qkv_rope(x, attn, cos, sin, q_out, key_cache[rows], value_cache[rows])

    for cache in (key_cache, value_cache):
        assert torch.isnan(cache[:offset]).all() and torch.isnan(cache[offset + tokens :]).all()
    eager_q = q[0].transpose(0, 1).contiguous()
    _assert_as_accurate_as_eager(q_out, eager_q, q64.transpose(0, 1), "QKV+RoPE", "qkv_proj + RoPE (q)", dtype)
    _assert_as_accurate_as_eager(key_cache[rows], k[0, :, 0], k64[:, 0], "QKV+RoPE", "qkv_proj + RoPE (k)", dtype)
    note = "The packed qkv_proj GEMM runs in torch (cuBLAS); the Triton kernel only writes V."
    _check_gemm_epilogue(gemm, dtype, value_cache[rows], v[0, :, 0], v64, "QKV+RoPE", "qkv_proj (v)", note)


@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("gated", [True, False], ids=["gated", "plain"])
@pytest.mark.parametrize("n_in", [2048, 4096], ids=["o_proj", "down_proj"])
@pytest.mark.parametrize("tokens", CHUNKS, ids=lambda tokens: f"chunk={tokens}")
def test_linear_residual_matches_gated_residual(gemm, dtype, gated, n_in, tokens):
    """``_gated_residual(residual, linear(x), gate)``, the float32 residual stream
    updated in place as the fused denoising step runs it."""
    _skip_unserved(gemm, dtype)
    _, width, _ = SHAPES["suffix"]
    linear = _proj_dtype_module(_randomize(RowParallelLinear(n_in, width, bias=False), seed=17), dtype)
    x = _randn(tokens, n_in, dtype=dtype, seed=18)
    residual = _randn(tokens, width, scale=4.0, seed=19)
    gate = _randn(width, seed=20) if gated else None

    expected = M._gated_residual(residual, linear(x[None])[0][0], gate)
    projection64 = x.double() @ linear.weight.double().t()
    reference = residual.double() + (gate.double() * projection64 if gated else projection64)
    actual = residual.clone()
    M._fused_linear_residual(x, linear, actual, actual, gate=gate)
    proj = {2048: "o_proj", 4096: "down_proj"}[n_in]
    residual_add = "gated residual add" if gated else "residual add"
    note = f"The {proj} GEMM runs in torch (cuBLAS); the Triton kernel only applies the {residual_add}."
    _check_gemm_epilogue(gemm, dtype, actual, expected, reference, "linear-residual", f"{proj} + {residual_add}", note)


@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("tokens", CHUNKS, ids=lambda tokens: f"chunk={tokens}")
def test_linear_gelu_mul_matches_gemma_mlp_input(gemm, dtype, tokens):
    """``GeluAndMul(gate_up_proj(x))``, the action expert's MLP up to ``down_proj``."""
    _skip_unserved(gemm, dtype)
    _, width, mlp_width = SHAPES["suffix"]
    mlp = M.Pi05GemmaMLP(_gemma_config(width, mlp_width), None, prefix="mlp")
    mlp = _proj_dtype_module(_randomize(mlp, seed=21), dtype)
    x = _randn(tokens, width, dtype=dtype, seed=22)

    expected = mlp.act_fn(mlp.gate_up_proj(x[None])[0])[0]
    x64 = x.double()
    w_gate, w_up = mlp.gate_up_proj.weight.double().chunk(2)
    reference = torch.nn.functional.gelu(x64 @ w_gate.t(), approximate="tanh") * (x64 @ w_up.t())
    actual = torch.empty_like(expected)
    M._fused_linear_gelu_mul(x, mlp, actual)
    eager = "Pi05GemmaMLP GeluAndMul(gate_up_proj(x))"
    note = "The gate_up_proj GEMM runs in torch (cuBLAS); the activation is the eager GeluAndMul too."
    _check_gemm_epilogue(gemm, dtype, actual, expected, reference, "linear GELU-multiply", eager, note)


# ── Reordered reductions: attention and the output head ────────────────
@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("part", SHAPES, ids=str)
def test_attention_matches_eager_attend(dtype, part):
    """``_attend`` over one KV head with eager's float mask: a prefix-shaped
    bidirectional mask with padded (fully masked) query rows (the served prefix
    keeps eager's ``_attend``), and a denoising step's queries over the
    ``[prefix | suffix]`` cache row."""
    tokens = SHAPES[part][0]
    prefix_pad = _prefix_pad_masks(SHAPES["prefix"][0])
    if part == "prefix":
        n_keys = tokens
        att_2d = M.make_att_2d_masks(prefix_pad, torch.zeros_like(prefix_pad))
    else:
        n_keys = prefix_pad.shape[1] + tokens
        suffix_pad = torch.ones(1, tokens, dtype=torch.bool, device=DEVICE)
        suffix_att = torch.zeros(1, tokens, device=DEVICE)
        suffix_att[:, 0] = 1
        att_2d = torch.cat(
            [prefix_pad[:, None, :].expand(1, tokens, -1), M.make_att_2d_masks(suffix_pad, suffix_att)], dim=2
        )
    mask = M.prepare_attention_masks_4d(att_2d)
    q = _randn(1, HEADS, tokens, HEAD_DIM, dtype=dtype, scale=2.0, seed=14)
    key = _randn(n_keys, HEAD_DIM, dtype=dtype, scale=2.0, seed=15)
    value = _randn(n_keys, HEAD_DIM, dtype=dtype, seed=16)
    scaling = 1.0 / math.sqrt(HEAD_DIM)

    def attend(q, key, value, mask):
        out = M._attend(q, key[None, None], value[None, None], mask, num_kv_groups=HEADS, scaling=scaling)
        return out.transpose(1, 2).reshape(tokens, HEADS * HEAD_DIM)

    expected = attend(q, key, value, mask)
    reference = attend(q.double(), key.double(), value.double(), mask.double())
    actual = torch.empty_like(expected)
    scores = torch.empty(HEADS * tokens, n_keys, dtype=dtype, device=DEVICE)
    M._fused_attention(q[0], key, value, mask[0, 0], scaling, scores, actual)
    if M._triton_gemms(dtype):
        note = (
            "It runs as three Triton kernels (scores, masked softmax, values) that keep the full score matrix "
            "in memory, not as one flash-attention kernel."
        )
    else:
        note = "The QK^T and PV matmuls run in torch (cuBLAS); only the masked softmax is a Triton kernel."
    _assert_as_accurate_as_eager(actual, expected, reference, "attention", f"_attend on the {part}", dtype, note)


@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
def test_final_head_matches_norm_and_action_out_proj(dtype):
    """The expert's final AdaRMS norm and ``action_out_proj``, both float32 in
    either layout, so ``dtype`` only names the layout."""
    tokens, width, _ = SHAPES["suffix"]
    norm = _randomize(M.Pi05AdaRMSNorm(width, cond_dim=width), seed=23).to(DEVICE)
    head = _randomize(nn.Linear(width, 32), seed=24).to(DEVICE)
    x = _randn(tokens, width, scale=4.0, seed=25)
    cond = _randn(1, width, seed=26)

    normed, _ = norm.forward_native(x[None], cond)
    expected = head(normed.to(head.weight.dtype))[0]
    modulation = M._adarms_modulation(norm, cond)
    scale, shift, _ = modulation.chunk(3)
    normed64 = _rms_norm_reference(x, norm.eps, scale=scale, shift=shift)
    reference = normed64 @ head.weight.double().t() + head.bias.double()
    actual = torch.empty_like(expected)
    M._fused_final_head(x, norm, modulation, head, actual)
    _assert_as_accurate_as_eager(
        actual,
        expected,
        reference,
        "final AdaRMSNorm + action_out_proj",
        "Pi05AdaRMSNorm + action_out_proj",
        dtype,
        _MODULATION_NOTE,
    )
