# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.model_executor.models.cosyvoice3.test_cosyvoice3_full_batch import (
    PACKED_BACKENDS,
    MelOutput,
    _items,
    require_packed_backend,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA")]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("backend", PACKED_BACKENDS)
@pytest.mark.parametrize("streaming, seed", [(False, 0), (False, 15), (True, 0), (True, 36)])
@torch.inference_mode()
def test_flow_graph_replay_matches_eager_solve(monkeypatch, streaming, backend, seed):
    require_packed_backend(monkeypatch, backend)
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav as module
    from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    monkeypatch.setenv("COSYVOICE3_FLOW_GRAPH", "1")
    monkeypatch.setenv("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "0" if streaming else "1")
    monkeypatch.setenv("COSYVOICE3_PACKED_STREAMING", "1" if streaming else "0")
    monkeypatch.setattr(module, "CausalHiFTGenerator", MelOutput)
    # Keep seed 0 plus one historical failing weight initialization per mode:
    # 15 for full context, 36 for streaming. The original scan used different
    # noise, so these retain weight coverage rather than reproduce that scan.
    # run() fixes the solve noise independently.
    # Start with the PyTorch default: solves must apply and restore the precision policy.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", True)
    torch.manual_seed(seed)
    config = CosyVoice3Config()
    config.flow["pre_lookahead_layer"]["channels"] = 32
    # 256 channels in 16 conv groups keep the packed convolution (and graphs).
    config.flow["decoder"]["estimator"].update(dim=256, depth=2, heads=4, dim_head=64)
    model = module.CosyVoice3Code2Wav(config).cuda().bfloat16().eval()
    if streaming:
        monkeypatch.setattr(
            model, "_stream_hift_from_feat", lambda mel, cache_state, finalize: (mel, None if finalize else cache_state)
        )
    else:
        model.hift.upsample_rates = [1]
        model.hift.istft_params = {"hop_len": 1}

    def run(items, graph):
        estimator = getattr(model, "_packed_estimator", None)
        runner = None if estimator is None else estimator.graph_runner
        if estimator is not None and not graph:
            estimator.graph_runner = None
        torch.manual_seed(3)
        try:
            if streaming:
                return [audio for audio, _ in model.forward_streaming_batch(items, n_timesteps=3)]
            return model.forward_batch(items, n_timesteps=3)
        finally:
            if estimator is not None:
                estimator.graph_runner = runner

    extra = dict(token_offset_tokens=2, finalize=False, cache_state=None) if streaming else {}
    # The second batch reuses the first one's padded layout with new lengths.
    batches = [_items([(7, 60), (13, 41), (3, 97)], **extra), _items([(5, 71), (9, 30), (11, 88)], **extra)]
    # FA2 still has shape-dependent BF16 rounding after precise GEMM reductions.
    # A 40-seed scan using this test's noise on RTX PRO 6000 Blackwell required
    # at most 0.014 absolute slack at rtol=0.01; retain a margin for kernel choice.
    # FlashInfer's padded graph plan can select different attention tiles; keep
    # its existing tolerance (and FA3's) separate from the tighter FA2 check.
    # Neither tolerance is an end-to-end audio quality guarantee.
    rtol, atol = (1e-2, 2e-2) if backend == "fa2" else (5e-2, 5e-2)
    run(batches[0], graph=False)  # compile
    for items in batches:
        actual = [output.clone() for output in run(items, graph=True)]
        replayed = run(items, graph=True)
        for output, replay in zip(actual, replayed):
            torch.testing.assert_close(output, replay, rtol=0, atol=0)
        expected = run(items, graph=False)
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)
            assert (output - reference).float().norm() <= 1e-2 * reference.float().norm()
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    runner = model._packed_estimator.graph_runner
    assert len(runner.graphs) == 1


def _tiny_packed_streaming(monkeypatch, *, prefix_gib: str, graphs: str):
    """A small bf16 packed streaming code2wav whose HiFT stand-in returns the
    new mel and keeps an empty streaming state."""
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the opt-in packed backend requires Hopper FA3")
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav as module
    from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    monkeypatch.setenv("COSYVOICE3_FLOW_GRAPH", graphs)
    monkeypatch.setenv("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "0")
    monkeypatch.setenv("COSYVOICE3_PACKED_STREAMING", "1")
    monkeypatch.setenv("COSYVOICE3_FLOW_PREFIX_CACHE_GIB", prefix_gib)
    monkeypatch.setattr(module, "CausalHiFTGenerator", MelOutput)
    config = CosyVoice3Config()
    config.flow["pre_lookahead_layer"]["channels"] = 32
    config.flow["decoder"]["estimator"].update(dim=256, depth=2, heads=4, dim_head=64)
    torch.manual_seed(0)  # the same weights on every call
    model = module.CosyVoice3Code2Wav(config).cuda().bfloat16().eval()
    monkeypatch.setattr(
        model, "_stream_hift_from_feat", lambda mel, cache_state, finalize: (mel, None if finalize else {})
    )
    return model


def _stream(model, requests):
    """Two non-final hops (20 and 40 tokens), then the final hop; every hop's mel."""
    states, emitted, mels = [None] * len(requests), [0] * len(requests), []
    for hop in (20, 40, None):
        items = []
        for request, state, offset in zip(requests, states, emitted):
            token = request["token"] if hop is None else request["token"][:, :hop]
            items.append(
                dict(request, token=token, token_offset_tokens=offset, finalize=hop is None, cache_state=state)
            )
        values = model.forward_streaming_batch(items, n_timesteps=3)
        mels += [mel for mel, _ in values]
        states = [state for _, state in values]
        emitted = [0 if hop is None else hop - 3] * len(requests)
    return mels


# The second request's second hop resumes from 50 cached frames (13 prompt
# tokens and 20 generated cover one attention chunk); the others recompute.
_REQUESTS = [(7, 60), (13, 41), (3, 97)]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@torch.inference_mode()
def test_compiled_prefix_hops_read_only_their_own_cache(monkeypatch):
    """Compiled prefix-cache hops match full recompute although the pool starts
    full of NaN, as reused pages hold other requests' state: a hop reads only
    what this request's earlier hops stored, the first hop's included."""
    torch.manual_seed(5)
    requests = _items(_REQUESTS)
    expected = _stream(_tiny_packed_streaming(monkeypatch, prefix_gib="0", graphs="0"), requests)
    model = _tiny_packed_streaming(monkeypatch, prefix_gib="0.25", graphs="0")
    cache = model._get_flow_prefix_cache(3)
    cache.kv.fill_(float("nan"))
    cache.conv_input.fill_(float("nan"))
    solved = []
    solve = cache.solve
    monkeypatch.setattr(
        cache, "solve", lambda queries, *a, **k: solved.extend(q.start for q in queries) or solve(queries, *a, **k)
    )
    actual = _stream(model, requests)
    assert max(solved) > 0
    # The two paths pack rows differently, so bf16 rounding differs by an ulp
    # or two; a stale read moves the mel by 0.3 or more, or makes it NaN.
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=2e-2, atol=5e-2)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@torch.inference_mode()
def test_captured_flow_graphs_serve_a_stream_without_recording(monkeypatch):
    """Graphs captured at startup serve prefix-cache hops and final hops of
    short streams: nothing records on demand and the mel matches eager Flow."""
    model = _tiny_packed_streaming(monkeypatch, prefix_gib="0.25", graphs="1")
    # The HiFT stand-in has no FFT plans to warm.
    monkeypatch.setattr(model, "_get_hift_batcher", lambda: SimpleNamespace(warm_fft_plans=lambda frames: 0))
    # Three requests, 128 frames a row at most: one row slot of 4, up to 512 frames.
    model.capture_graphs(n_timesteps=3, row_frames=128)
    runner = model._packed_estimator.graph_runner
    prefix = model._flow_prefix_cache
    assert len(runner.graphs) == len(prefix.graphs) == 1 + 2 + 4 + 8
    torch.manual_seed(5)
    requests = _items(_REQUESTS)
    graphs = (runner, prefix.graphs)
    model._packed_estimator.graph_runner, prefix.graphs = None, None
    expected = _stream(model, requests)
    model._packed_estimator.graph_runner, prefix.graphs = graphs
    counts = (len(runner.graphs), len(prefix.graphs))
    actual = _stream(model, requests)
    assert (len(runner.graphs), len(prefix.graphs)) == counts
    assert prefix.used_pages == 0
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@torch.inference_mode()
def test_full_prefix_pool_hops_recompute_without_recording(monkeypatch):
    """Hops that miss a full prefix pool recompute without the graph runner:
    no layout records on demand and the mel matches eager Flow."""
    model = _tiny_packed_streaming(monkeypatch, prefix_gib="0.25", graphs="1")
    monkeypatch.setattr(model, "_get_hift_batcher", lambda: SimpleNamespace(warm_fft_plans=lambda frames: 0))
    model.capture_graphs(n_timesteps=3, row_frames=128)
    runner = model._packed_estimator.graph_runner
    monkeypatch.setattr(model._flow_prefix_cache, "reserve", lambda state, frames: False)
    torch.manual_seed(5)
    requests = _items(_REQUESTS)
    model._packed_estimator.graph_runner = None
    expected = _stream(model, requests)
    model._packed_estimator.graph_runner = runner
    layouts = set(runner.graphs)
    actual = _stream(model, requests)
    assert set(runner.graphs) == layouts
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@torch.inference_mode()
def test_flow_graph_runner_retires_a_generation_at_its_limit(monkeypatch):
    """Without a prefix pool every hop records its own layout; at the limit the
    runner retires all of its graphs, and the next generation still matches
    eager Flow on a replayed stream."""
    monkeypatch.setenv("COSYVOICE3_FLOW_GRAPH_MAX", "2")
    model = _tiny_packed_streaming(monkeypatch, prefix_gib="0", graphs="1")
    estimator = model._get_packed_estimator(torch.bfloat16)
    runner = estimator.graph_runner
    torch.manual_seed(5)
    requests = _items(_REQUESTS)
    estimator.graph_runner = None
    expected = _stream(model, requests)
    estimator.graph_runner = runner
    # Two non-final hop layouts and the final layout: three for a limit of two.
    for _ in range(2):
        actual = _stream(model, requests)
        assert len(runner.graphs) <= 2
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
    assert runner.flushes >= 2


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@torch.inference_mode()
def test_precise_bf16_gemms_keep_a_padded_projection_stable():
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import precise_bf16_gemms

    # Real-model projection dimensions: graph padding from M=908 to 1024
    # previously selected a reduced-precision bias GEMM (max error 0.5).
    generator = torch.Generator(device="cuda").manual_seed(0)
    inputs = torch.randn(1024, 320, device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = torch.randn(1024, 320, device="cuda", dtype=torch.bfloat16, generator=generator)
    bias = torch.randn(1024, device="cuda", dtype=torch.bfloat16, generator=generator)
    with precise_bf16_gemms():
        unpadded = torch.nn.functional.linear(inputs[:908], weight, bias)
        padded = torch.nn.functional.linear(inputs, weight, bias)[:908]
    torch.testing.assert_close(unpadded, padded, rtol=1e-2, atol=1e-2)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("streaming", [False, True])
@torch.inference_mode()
def test_flow_graph_metadata_retains_loaded_buffers(monkeypatch, backend, streaming):
    """Metadata passed by the production solver must observe later buffer loads."""
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core import packed_dit

    require_packed_backend(monkeypatch, backend)
    estimator = SimpleNamespace(
        backend=backend,
        dit=SimpleNamespace(
            transformer_blocks=[SimpleNamespace(attn=SimpleNamespace(heads=2, inner_dim=128))],
            static_chunk_size=50,
        ),
        step_modulations=lambda *_args: None,
        compiled_full_forward=None,
    )
    runner = packed_dit.FlowGraphRunner(estimator)
    noise = torch.zeros(1, 50, 80, device="cuda", dtype=torch.bfloat16)
    spks = torch.zeros(2, 80, device="cuda", dtype=torch.bfloat16)
    time_span = torch.tensor([0.0, 1.0], device="cuda")
    key, *layout = runner._layout((17, 33), streaming)
    state = runner._allocate(key, noise, spks, time_span)
    runner._load(state, *layout, noise, noise, noise, spks, time_span)
    observed = []

    def observe_metadata(forward, noise, time_span, mu, spks, cond, flow_time, rows, attention, modulations, cfg_rate):
        observed.append(attention)
        return noise

    # Stop at the solver's consumer boundary, after real metadata construction.
    monkeypatch.setattr(packed_dit, "euler_steps", observe_metadata)
    runner._solve(state, 0.7, key + ((),))
    attention = observed[0]
    snapshots = {}
    for name in ("cache_seqlens", "cu_seqlens_q", "page_table", "slot_frames"):
        buffer = getattr(state, name)
        assert getattr(attention, name) is buffer
        if buffer is not None:
            snapshots[name] = buffer.clone()

    next_key, *next_layout = runner._layout((19, 31), streaming)
    assert next_key == key
    runner._load(state, *next_layout, noise, noise, noise, spks, time_span)
    # Row boundaries must refresh even when one-frame slots keep packed order.
    assert not torch.equal(attention.cu_seqlens_q, snapshots["cu_seqlens_q"])
    for name, before in snapshots.items():
        # Check the metadata already handed to the consumer, not a new object.
        assert getattr(attention, name) is getattr(state, name)
        if attention.page_size == 1 and name == "slot_frames":
            # One-frame pages stay in packed order when only row boundaries
            # change; the graph allocation and padding slots are unchanged.
            assert torch.equal(attention.slot_frames, before)
        elif name != "cu_seqlens_q":
            assert not torch.equal(getattr(attention, name), before)
