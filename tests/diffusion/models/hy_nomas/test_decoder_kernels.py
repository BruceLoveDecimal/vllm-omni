# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA parity for PixelUMM's cast-before-scale norm and FP32 split MRoPE."""

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.hy_nomas import decoder as d
from vllm_omni.diffusion.models.hy_nomas import decoder_kernels as k

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
@torch.inference_mode()
def test_decoder_rms_norm_exact(monkeypatch):
    torch.manual_seed(42)
    # Odd row count exercises the masked tail; zero inputs exercise epsilon.
    x = torch.randn(257, 4096, device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    norm = d.PixelUMMRMSNorm(4096, 1e-6).to(device="cuda", dtype=torch.bfloat16)
    norm.weight.copy_(torch.randn_like(norm.weight))
    actual = k.rms_norm(x, norm.weight, norm.eps)
    assert actual is not None
    torch.testing.assert_close(norm(x), actual, rtol=0, atol=0)
    monkeypatch.setattr(k, "rms_norm", lambda *args: None)
    expected = norm(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
@pytest.mark.parametrize("heads", [8, 32])
@torch.inference_mode()
def test_decoder_split_qk_norm_rope_exact(heads, monkeypatch):
    torch.manual_seed(42)
    grid = (3, 10, 14)
    x = torch.randn(1, 420, heads, 128, device="cuda", dtype=torch.bfloat16)
    x[:, 0].zero_()
    norm_t = d.PixelUMMRMSNorm(64, 1e-6).to(device="cuda", dtype=torch.bfloat16)
    norm_hw = d.PixelUMMRMSNorm(64, 1e-6).to(device="cuda", dtype=torch.bfloat16)
    norm_t.weight.copy_(torch.randn_like(norm_t.weight))
    norm_hw.weight.copy_(torch.randn_like(norm_hw.weight))
    cos, sin = d.mrope_tables(d.PixelUMMArchitecture(), grid, "cuda")
    actual = k.split_norm_rope(x, cos, sin, norm_t.weight, norm_hw.weight, 1e-6)
    assert actual is not None

    class Geometry:
        arch = d.PixelUMMArchitecture()

    monkeypatch.setattr(k, "split_norm_rope", lambda *args: None)
    expected = d.PixelUMMAttention._prepare_qk(Geometry(), x, cos, sin, norm_t, norm_hw, x.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
