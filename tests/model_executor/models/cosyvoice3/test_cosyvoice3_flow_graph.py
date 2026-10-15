# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.model_executor.models.cosyvoice3.test_cosyvoice3_full_batch import MelOutput, _items
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA")]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("streaming", [False, True])
@torch.inference_mode()
def test_flow_graph_replay_matches_eager_solve(monkeypatch, streaming):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the opt-in packed backend requires Hopper FA3")
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav as module
    from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    monkeypatch.setenv("COSYVOICE3_FLOW_GRAPH", "1")
    monkeypatch.setenv("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "0" if streaming else "1")
    monkeypatch.setenv("COSYVOICE3_PACKED_STREAMING", "1" if streaming else "0")
    monkeypatch.setattr(module, "CausalHiFTGenerator", MelOutput)
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
    run(batches[0], graph=False)  # compile
    for items in batches:
        actual = run(items, graph=True)
        expected = run(items, graph=False)
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
    runner = model._packed_estimator.graph_runner
    assert len(runner.graphs) == 1


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@torch.inference_mode()
def test_packed_prefix_cache_matches_uncached_streaming(monkeypatch):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the opt-in packed backend requires Hopper FA3")
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav as module
    from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    monkeypatch.setenv("COSYVOICE3_PACKED_STREAMING", "1")
    monkeypatch.setenv("COSYVOICE3_FLOW_GRAPH", "0")
    monkeypatch.setattr(module, "CausalHiFTGenerator", MelOutput)
    config = CosyVoice3Config()
    config.flow["pre_lookahead_layer"]["channels"] = 32
    config.flow["decoder"]["estimator"].update(dim=256, depth=2, heads=4, dim_head=64)
    model = module.CosyVoice3Code2Wav(config, flow_prefix_cache_mb=32).cuda().bfloat16().eval()
    monkeypatch.setattr(model, "_stream_hift_from_feat", lambda mel, cache_state, finalize: (mel, None))
    items = _items([(60, 55), (80, 45)], finalize=False, token_offset_tokens=0, cache_state=None)

    def check(batch, steps=3):
        model.flow_prefix_cache_bytes = 0
        expected = model.forward_streaming_batch(batch, n_timesteps=steps)
        model.flow_prefix_cache_bytes = 32 * 1024 * 1024
        actual = model.forward_streaming_batch(batch, n_timesteps=steps)
        for (output, _), (reference, _) in zip(actual, expected):
            torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)

    check(items)  # cold cache
    cache = model._packed_estimator.prefix_cache
    assert cache.hit_frames == 0
    # Reorder the batch and append different continuation lengths.
    grown = [
        {**item, "token": torch.cat((item["token"], torch.randint(0, 64, (1, n), device="cuda")), dim=1)}
        for item, n in zip(items[::-1], (25, 50))
    ]
    check(grown)
    assert cache.hit_frames > 0
    hits = cache.hit_frames
    # A new request shares the reference but has unrelated generated tokens.
    shared = {**items[0], "token": torch.randint(0, 64, (1, 60), device="cuda")}
    check([shared])
    assert cache.hit_frames > hits
    check([{**shared, "embedding": torch.randn_like(shared["embedding"])}])
    hits = cache.hit_frames
    check([{**item, "finalize": True} for item in grown])
    assert cache.hit_frames == hits  # full-context finalization must bypass reuse
    check(grown, steps=2)  # changed Euler schedule invalidates all entries
    assert cache.nbytes <= cache.max_bytes
