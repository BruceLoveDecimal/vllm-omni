# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamo guard-stability tests for the LingBot attention block.

The 40 ``LingBotAttentionBlock`` instances share one code object and one
dynamo recompile budget (``torch._dynamo.config.recompile_limit``, default 8).
Reading the dense cache's rolling cursors (``last_start``/``end``/
``sink_end``) or branching on rollout progress inside the compiled region
installs per-chunk value guards whose conjunctions exhaust that budget right
as the sliding window fills — after which dynamo silently falls back to
eager for the rest of the rollout (nondeterministic outputs across processes
plus a performance cliff, observed on H20 at 81 frames).

These tests drive a real block through a multi-chunk rollout the way the
transformer forward does (cache update planned OUTSIDE the compiled region)
and assert the compiled-frame count reaches steady state.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

DIM = 32
HEADS = 2
HEAD_DIM = 16
BLOCK = 8
SINK = 8
WINDOW_CHUNKS = 3
CAPACITY = WINDOW_CHUNKS * BLOCK


@pytest.fixture(autouse=True)
def _single_rank_tp():
    """Minimal distributed environment for the block's TP projections."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.device import DeviceConfig
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29527")
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
        initialize_model_parallel()
        yield
        cleanup_dist_env_and_memory()


@pytest.fixture(autouse=True)
def _cpu_friendly_kernels(monkeypatch):
    """Route platform-dispatched kernels to device-agnostic implementations."""
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    import vllm_omni.diffusion.layers.custom_op as custom_op
    import vllm_omni.platforms as platforms

    monkeypatch.setattr(
        platforms.current_omni_platform,
        "get_diffusion_attn_backend_cls",
        lambda *a, **k: "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend",
        raising=False,
    )
    monkeypatch.setattr(platforms.current_omni_platform, "is_cuda", lambda: True, raising=False)
    monkeypatch.setattr(custom_op.CustomOp, "dispatch_forward", lambda self: self.forward_native)
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda: default_unquantized_gemm,
    )


def _make_block():
    from vllm_omni.diffusion.models.lingbot_world.transformer import LingBotAttentionBlock

    torch.manual_seed(0)
    block = LingBotAttentionBlock(DIM, HEADS)
    block.eval()
    return block


def _chunk_inputs(text_needed):
    hs = torch.randn(1, BLOCK, DIM)
    tproj = torch.randn(1, 1, 6, DIM)
    cam = torch.randn(1, BLOCK, DIM)
    text = torch.randn(1, 16, DIM) if text_needed else None
    cos = torch.randn(1, BLOCK, 1, HEAD_DIM // 2)
    sin = torch.randn(1, BLOCK, 1, HEAD_DIM // 2)
    return hs, text, tproj, cam, (cos, sin)


def _dense_cache():
    from vllm_omni.diffusion.models.lingbot_world.transformer import allocate_lingbot_cache

    return allocate_lingbot_cache(
        batch_size=1,
        num_layers=1,
        max_tokens=CAPACITY,
        num_local_heads=HEADS,
        head_dim=HEAD_DIM,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_dense_rollout_compiled_frames_reach_steady_state():
    """Externally-planned dense-cache rollouts must stop recompiling once the
    sliding window fills; unbounded per-chunk guard growth is the bug that
    exhausted the shared recompile budget."""
    import torch._dynamo
    from torch._dynamo.testing import CompileCounter

    from vllm_omni.diffusion.models.lingbot_world.transformer import (
        apply_dense_cache_plan,
        plan_dense_cache_update,
    )

    block = _make_block()
    counter = CompileCounter()
    torch._dynamo.reset()
    try:
        block.forward = torch.compile(block.forward, backend=counter, dynamic=True)
        cache = _dense_cache()
        cross_cache = None
        frames_at_window_fill = None
        with torch.inference_mode():
            for chunk in range(8):
                for step in range(5):  # four DMD probes then one commit
                    commit = step == 4
                    plan = plan_dense_cache_update(
                        cache.self_attention[0],
                        current_start=chunk * BLOCK,
                        chunk_tokens=BLOCK,
                        sink_tokens=SINK,
                        update_cache=commit,
                    )
                    hs, text, tproj, cam, rope = _chunk_inputs(cross_cache is None)
                    _, cross_cache = block(
                        hs,
                        text,
                        tproj,
                        cam,
                        self_cache=cache.self_attention[0],
                        cross_cache=cross_cache,
                        current_start=chunk * BLOCK,
                        sink_tokens=SINK,
                        update_cache=commit,
                        rotary_emb=rope,
                        dense_plan=plan,
                    )
                    if plan.write_back:
                        apply_dense_cache_plan(cache.self_attention[0], plan)
                if chunk == WINDOW_CHUNKS:
                    frames_at_window_fill = counter.frame_count
        assert frames_at_window_fill is not None
        assert counter.frame_count == frames_at_window_fill, (
            "compiled frames kept growing after the sliding window filled: "
            f"{frames_at_window_fill} -> {counter.frame_count}"
        )
    finally:
        torch._dynamo.reset()


def test_dense_plan_keeps_cursor_reads_out_of_compiled_region():
    """No dense-cache cursor guard may reach dynamo when a plan is supplied:
    with guard logging enabled, recompile reasons must never mention the
    rolling cursors."""
    import logging

    import torch._dynamo
    from torch._dynamo.testing import CompileCounter

    from vllm_omni.diffusion.models.lingbot_world.transformer import (
        apply_dense_cache_plan,
        plan_dense_cache_update,
    )

    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())  # type: ignore[method-assign]
    guard_logger = logging.getLogger("torch._dynamo.guards.__recompiles")
    guard_logger.addHandler(handler)
    guard_logger.setLevel(logging.DEBUG)

    block = _make_block()
    torch._dynamo.reset()
    try:
        block.forward = torch.compile(block.forward, backend=CompileCounter(), dynamic=True)
        cache = _dense_cache()
        cross_cache = None
        with torch.inference_mode():
            for chunk in range(4):
                commit_plan = plan_dense_cache_update(
                    cache.self_attention[0],
                    current_start=chunk * BLOCK,
                    chunk_tokens=BLOCK,
                    sink_tokens=SINK,
                    update_cache=True,
                )
                hs, text, tproj, cam, rope = _chunk_inputs(cross_cache is None)
                _, cross_cache = block(
                    hs,
                    text,
                    tproj,
                    cam,
                    self_cache=cache.self_attention[0],
                    cross_cache=cross_cache,
                    current_start=chunk * BLOCK,
                    sink_tokens=SINK,
                    update_cache=True,
                    rotary_emb=rope,
                    dense_plan=commit_plan,
                )
                apply_dense_cache_plan(cache.self_attention[0], commit_plan)
        offenders = [
            line
            for line in records
            if "last_start" in line or "sink_end" in line or "absolute_end" in line or "cache.end" in line
        ]
        assert not offenders, f"cursor guards leaked into the compiled region: {offenders[:3]}"
    finally:
        guard_logger.removeHandler(handler)
        torch._dynamo.reset()
