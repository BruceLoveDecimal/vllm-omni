# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Dispatch contract of SOL_ATTN, exercised without the external kernel.

The kernel is replaced by a fake ``sol_attn`` module so the tests pin down
what the backend hands it: only the valid rows of packed document 0, the exact
prefix sink derived from ``video_layout``, and the dense recompute of the
sink's own query rows afterwards. The dense/sparse gating (warmup steps and
dense layers) is checked against the forward context the denoise loops
publish.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends import sol_attn as sol_attn_module
from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    PackedPaddingMetadata,
    VideoTokenLayout,
    VideoTokenSpan,
)
from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.backends.sol_attn import (
    SolAttnBackend,
    SolAttnConfig,
    SolAttnImpl,
    SolAttnPlan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

HEAD_DIM = 128
PREFIX_ROWS = 710  # 14 text rows + 696 audio rows (MiniMax-H3 FL2VA, 8.7s)
GRID = (62, 24, 40)  # 1280x768 -> 59520 video rows
VIDEO_ROWS = GRID[0] * GRID[1] * GRID[2]
USED_LEN = PREFIX_ROWS + VIDEO_ROWS


def make_impl(prefix: str = "transformer.blocks.5.attn", **backend_kwargs: Any) -> SolAttnImpl:
    return SolAttnImpl(
        num_heads=8,
        head_size=HEAD_DIM,
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
        prefix=prefix,
        qkv_layout="BSND",
        backend_kwargs=backend_kwargs,
    )


def h3_metadata(total_len: int = USED_LEN, used_len: int = USED_LEN) -> AttentionMetadata:
    """The single-request packed metadata MiniMaxH3Attention builds on a mask-free backend."""
    cu = torch.tensor([0, used_len, total_len], dtype=torch.int32)
    return AttentionMetadata(
        packed_padding=PackedPaddingMetadata(
            q_length=used_len, kv_length=used_len, cu_seqlens_q=cu[:2], cu_seqlens_k=cu[:2]
        ),
        extra={
            "cu_seqlens_q": cu,
            "cu_seqlens_k": cu,
            "max_seqlen_q": used_len,
            "max_seqlen_k": used_len,
            "valid_kv_length": used_len,
        },
        video_layout=VideoTokenLayout(prefix_len=PREFIX_ROWS, latent_grid=GRID),
    )


def set_denoise_step(monkeypatch: pytest.MonkeyPatch, step_idx: int | None) -> None:
    monkeypatch.setattr(sol_attn_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(sol_attn_module, "get_forward_context", lambda: SimpleNamespace(denoise_step_idx=step_idx))


# -- registration and capabilities ------------------------------------------------


def test_sol_attn_backend_is_registered():
    assert DiffusionAttentionBackendEnum.SOL_ATTN.get_path().endswith("sol_attn.SolAttnBackend")
    assert DiffusionAttentionBackendEnum["SOL_ATTN"].get_class() is SolAttnBackend


def test_backend_capabilities():
    assert SolAttnBackend.get_name() == "SOL_ATTN"
    assert SolAttnBackend.get_supported_head_sizes() == [HEAD_DIM]
    assert SolAttnBackend.supported_platforms == ("cuda",)
    # One document per forward: MiniMax-H3 must not pack co-batched requests.
    assert SolAttnBackend.supports_multi_doc_packed_varlen() is False
    assert SolAttnBackend.get_impl_cls() is SolAttnImpl


def test_validate_available_requires_kernel_package(monkeypatch):
    monkeypatch.setattr(sol_attn_module.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match="sol_attn"):
        SolAttnBackend.validate_available()
    monkeypatch.setattr(sol_attn_module.importlib.util, "find_spec", lambda name: object())
    SolAttnBackend.validate_available()


# -- configuration ---------------------------------------------------------------


def test_config_defaults_follow_sol_engine_h3_policy():
    cfg = SolAttnConfig.from_backend_kwargs(None)
    assert cfg == SolAttnConfig(
        tau=1.0,
        thresh_type="diag",
        kv_splits=None,
        dense_steps=10,
        dense_layers=frozenset({0, 1}),
        sink_mode="prefix",
        strict=False,
    )


def test_config_parses_serialized_spec_kwargs():
    cfg = SolAttnConfig.from_backend_kwargs(
        {
            "tau": 1.5,
            "thresh_type": "exact",
            "kv_splits": 4,
            "dense_steps": 0,
            "dense_layers": [3, 4],
            "sink_mode": "none",
            "strict": True,
        }
    )
    assert cfg.tau == 1.5
    assert cfg.thresh_type == "exact"
    assert cfg.kv_splits == 4
    assert cfg.dense_steps == 0
    assert cfg.dense_layers == frozenset({3, 4})
    assert cfg.sink_mode == "none"
    assert cfg.strict is True
    assert SolAttnConfig.from_backend_kwargs({"kv_splits": "auto"}).kv_splits is None


@pytest.mark.parametrize(
    "backend_kwargs",
    [
        {"thresh_type": "mean"},
        {"kv_splits": 3},
        {"sink_mode": "text"},
        {"dense_steps": -1},
        {"tau": float("inf")},
    ],
)
def test_config_rejects_invalid_values(backend_kwargs):
    with pytest.raises(ValueError):
        SolAttnConfig.from_backend_kwargs(backend_kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"causal": True}, "causal"),
        ({"head_size": 64}, "head_size=128"),
        ({"qkv_layout": "BNSD"}, "BSND"),
    ],
)
def test_impl_rejects_unsupported_layer_shapes(kwargs, match):
    init_kwargs: dict[str, Any] = {
        "num_heads": 8,
        "head_size": HEAD_DIM,
        "softmax_scale": HEAD_DIM**-0.5,
        "causal": False,
        "prefix": "transformer.blocks.5.attn",
        "qkv_layout": "BSND",
    }
    init_kwargs.update(kwargs)
    with pytest.raises(ValueError, match=match):
        SolAttnImpl(**init_kwargs)


# -- dense/sparse gating ---------------------------------------------------------


def test_dense_layers_keep_the_listed_blocks_dense(monkeypatch):
    set_denoise_step(monkeypatch, 20)
    q = torch.zeros(1, USED_LEN, 8, HEAD_DIM, dtype=torch.bfloat16)
    assert make_impl(prefix="transformer.blocks.1.attn")._resolve_plan(q, q, q, h3_metadata()) == "dense_layer"
    plan = make_impl(prefix="transformer.blocks.2.attn", dense_layers=[2])._resolve_plan(q, q, q, h3_metadata())
    assert plan == "dense_layer"


def test_warmup_steps_stay_dense_until_dense_steps(monkeypatch):
    q = torch.zeros(1, USED_LEN, 8, HEAD_DIM, dtype=torch.bfloat16)
    impl = make_impl()
    set_denoise_step(monkeypatch, 9)
    assert impl._resolve_plan(q, q, q, h3_metadata()) == "warmup_step"
    set_denoise_step(monkeypatch, 10)
    reason = impl._resolve_plan(q, q, q, h3_metadata())
    # Past the gate on CPU, the next contract check (a CUDA tensor) is the one that declines.
    assert isinstance(reason, str) and "CUDA" in reason


def test_missing_step_index_runs_sparse(monkeypatch):
    set_denoise_step(monkeypatch, None)
    q = torch.zeros(1, USED_LEN, 8, HEAD_DIM, dtype=torch.bfloat16)
    reason = make_impl()._resolve_plan(q, q, q, h3_metadata())
    assert reason != "warmup_step"


def test_metadata_contract_declines(monkeypatch):
    set_denoise_step(monkeypatch, 20)
    impl = make_impl(dense_steps=0)
    q = torch.zeros(1, USED_LEN, 8, HEAD_DIM, dtype=torch.bfloat16)
    masked = h3_metadata()
    masked.attn_mask = torch.ones(USED_LEN, dtype=torch.bool)
    assert "mask" in impl._resolve_plan(q, q, q, masked)
    joint = h3_metadata()
    joint.joint_query = q
    assert "joint" in impl._resolve_plan(q, q, q, joint)
    q16 = q.to(torch.float16)
    assert "bfloat16" in impl._resolve_plan(q16, q16, q16, h3_metadata())
    kv = torch.zeros(1, USED_LEN, 4, HEAD_DIM, dtype=torch.bfloat16)
    assert "GQA" in impl._resolve_plan(q, kv, kv, h3_metadata())


# -- geometry ---------------------------------------------------------------------


def test_used_len_prefers_packed_padding_then_valid_kv_then_layout():
    total = USED_LEN + 64
    assert SolAttnImpl._resolve_used_len(h3_metadata(total), total) == USED_LEN
    assert SolAttnImpl._resolve_used_len(AttentionMetadata(extra={"valid_kv_length": 4096}), total) == 4096
    legacy = AttentionMetadata(video_layout=VideoTokenLayout(prefix_len=PREFIX_ROWS, latent_grid=GRID))
    assert SolAttnImpl._resolve_used_len(legacy, total) == USED_LEN
    spans = AttentionMetadata(video_layout=VideoTokenLayout(used_len=1234, video_spans=()))
    assert SolAttnImpl._resolve_used_len(spans, total) == 1234
    assert SolAttnImpl._resolve_used_len(None, total) == total
    assert SolAttnImpl._resolve_used_len(AttentionMetadata(), total) == total


def test_used_len_rejects_inconsistent_packing():
    mismatched = AttentionMetadata(extra={"valid_kv_length": 4096, "max_seqlen_q": 8192})
    assert "max_seqlen_q" in SolAttnImpl._resolve_used_len(mismatched, 8192)
    beyond = AttentionMetadata(extra={"valid_kv_length": 9000})
    assert "outside" in SolAttnImpl._resolve_used_len(beyond, 8192)
    uneven = AttentionMetadata(
        packed_padding=PackedPaddingMetadata(
            q_length=10, kv_length=20, cu_seqlens_q=torch.tensor([0, 10]), cu_seqlens_k=torch.tensor([0, 20])
        )
    )
    assert "kv_length" in SolAttnImpl._resolve_used_len(uneven, 8192)


def test_sink_covers_everything_before_the_target_video():
    impl = make_impl()
    legacy = VideoTokenLayout(prefix_len=PREFIX_ROWS, latent_grid=GRID)
    assert impl._resolve_sink(legacy, USED_LEN) == (0, PREFIX_ROWS)

    # Ref2VA: [text | reference video | audio | target video]; the reference
    # clip and the audio between the clips stay exact as keys.
    ref_grid = (5, 24, 40)
    ref_rows = 5 * 24 * 40
    target_start = 14 + ref_rows + 696
    spans = (
        VideoTokenSpan(start=14, latent_grid=ref_grid, role="reference"),
        VideoTokenSpan(start=target_start, latent_grid=GRID, role="target"),
    )
    layout = VideoTokenLayout(used_len=target_start + VIDEO_ROWS, video_spans=spans)
    assert impl._resolve_sink(layout, target_start + VIDEO_ROWS) == (0, target_start)


def test_sink_declines_ambiguous_or_empty_video_geometry():
    impl = make_impl()
    two_targets = VideoTokenLayout(
        used_len=2 * VIDEO_ROWS,
        video_spans=(
            VideoTokenSpan(start=0, latent_grid=GRID, role="target"),
            VideoTokenSpan(start=VIDEO_ROWS, latent_grid=GRID, role="target"),
        ),
    )
    assert "exactly one target" in impl._resolve_sink(two_targets, 2 * VIDEO_ROWS)
    overflow = VideoTokenLayout(
        used_len=VIDEO_ROWS, video_spans=(VideoTokenSpan(start=64, latent_grid=GRID, role="target"),)
    )
    assert "exceeds" in impl._resolve_sink(overflow, VIDEO_ROWS)
    assert "no video rows" in impl._resolve_sink(VideoTokenLayout(prefix_len=USED_LEN, latent_grid=GRID), USED_LEN)
    assert "neither" in impl._resolve_sink(VideoTokenLayout(), USED_LEN)


def test_sink_modes_without_layout_or_disabled():
    assert make_impl()._resolve_sink(None, USED_LEN) == (0, 0)
    legacy = VideoTokenLayout(prefix_len=PREFIX_ROWS, latent_grid=GRID)
    assert make_impl(sink_mode="none")._resolve_sink(legacy, USED_LEN) == (0, 0)


# -- kernel dispatch ---------------------------------------------------------------


def install_fake_kernel(monkeypatch: pytest.MonkeyPatch, calls: dict[str, Any], *, fail: bool = False) -> None:
    fake = types.ModuleType("sol_attn")

    def fake_sol_attn(q, k, v, *, scale, tau, thresh_type, kv_splits, sink_tokens, sink_start):
        if fail:
            raise RuntimeError("kernel exploded")
        calls.update(
            q_shape=tuple(q.shape),
            contiguous=q.is_contiguous() and k.is_contiguous() and v.is_contiguous(),
            scale=scale,
            tau=tau,
            thresh_type=thresh_type,
            kv_splits=kv_splits,
            sink_start=sink_start,
            sink_tokens=sink_tokens,
        )
        return torch.ones_like(q)

    setattr(fake, "sol_attn", fake_sol_attn)
    setattr(fake, "get_sol_attn_backend", lambda device: calls.setdefault("backend", "cute_sm90"))
    monkeypatch.setitem(sys.modules, "sol_attn", fake)


def test_sparse_path_slices_padding_sinks_prefix_and_recomputes_prefix_queries(monkeypatch):
    calls: dict[str, Any] = {}
    install_fake_kernel(monkeypatch, calls)
    torch.manual_seed(0)
    prefix, used, total, heads = 96, 320, 384, 2
    query = torch.randn(1, total, heads, HEAD_DIM)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    impl = make_impl(tau=1.25, thresh_type="exact", kv_splits=1, dense_steps=0)
    plan = SolAttnPlan(used_len=used, sink_start=0, sink_tokens=prefix)
    monkeypatch.setattr(impl, "_resolve_plan", lambda *args: plan)

    out = impl.forward_cuda(query, key, value, h3_metadata(total, used))

    assert calls["q_shape"] == (1, used, heads, HEAD_DIM)
    assert calls["contiguous"]
    assert calls == {
        **calls,
        "scale": pytest.approx(HEAD_DIM**-0.5),
        "tau": 1.25,
        "thresh_type": "exact",
        "kv_splits": 1,
        "sink_start": 0,
        "sink_tokens": prefix,
    }
    assert out.shape == query.shape
    expected_prefix = F.scaled_dot_product_attention(
        query[:, :prefix].transpose(1, 2),
        key[:, :used].transpose(1, 2),
        value[:, :used].transpose(1, 2),
        scale=HEAD_DIM**-0.5,
    ).transpose(1, 2)
    torch.testing.assert_close(out[:, :prefix], expected_prefix)
    assert torch.equal(out[:, prefix:used], torch.ones(1, used - prefix, heads, HEAD_DIM))
    assert torch.equal(out[:, used:], torch.zeros(1, total - used, heads, HEAD_DIM))


def test_auto_kv_splits_follows_sol_engine_policy(monkeypatch):
    calls: dict[str, Any] = {}
    install_fake_kernel(monkeypatch, calls)
    impl = make_impl()
    device = torch.device("cpu")
    assert impl._resolve_kv_splits(device, 65536) == 4
    assert impl._resolve_kv_splits(device, 65535) == 1
    calls["backend"] = "cute_sm100"
    assert impl._resolve_kv_splits(device, 1 << 20) == 1
    assert make_impl(kv_splits=2)._resolve_kv_splits(device, 1 << 20) == 2


def test_kernel_failure_falls_back_to_dense_unless_strict(monkeypatch):
    calls: dict[str, Any] = {}
    install_fake_kernel(monkeypatch, calls, fail=True)
    query = torch.randn(1, 256, 2, HEAD_DIM)
    metadata = h3_metadata(256, 256)
    plan = SolAttnPlan(used_len=256, sink_start=0, sink_tokens=64)

    seen: dict[str, Any] = {}

    def fake_dense(q, k, v, attn_metadata):
        seen.update(q=q, metadata=attn_metadata)
        return q * 2

    impl = make_impl(kv_splits=1)
    monkeypatch.setattr(impl, "_resolve_plan", lambda *args: plan)
    monkeypatch.setattr(impl, "dense_fallback", SimpleNamespace(forward=fake_dense))
    out = impl.forward_cuda(query, query, query, metadata)
    assert seen["q"] is query and seen["metadata"] is metadata
    assert torch.equal(out, query * 2)

    strict = make_impl(kv_splits=1, strict=True)
    monkeypatch.setattr(strict, "_resolve_plan", lambda *args: plan)
    with pytest.raises(RuntimeError, match="kernel exploded"):
        strict.forward_cuda(query, query, query, metadata)


def test_declined_forward_hands_untouched_metadata_to_flash_attention(monkeypatch):
    set_denoise_step(monkeypatch, 0)
    query = torch.zeros(1, 256, 2, HEAD_DIM, dtype=torch.bfloat16)
    metadata = h3_metadata(256, 256)
    seen: dict[str, Any] = {}
    impl = make_impl()
    monkeypatch.setattr(
        impl,
        "dense_fallback",
        SimpleNamespace(forward=lambda q, k, v, m: seen.update(metadata=m) or q),
    )
    impl.forward_cuda(query, query, query, metadata)
    assert seen["metadata"] is metadata
