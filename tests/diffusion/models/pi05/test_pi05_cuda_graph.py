# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA-graph and torch.compile tests for the π0.5 kernel.

Runs the shrunken backbones from ``test_pi05_units.py`` on one GPU. What the
graph path has to prove:

* replaying the captured prefix and denoise graphs reproduces the eager action
  chunk **bit for bit** — replay runs the same kernels, so anything else means
  a static buffer was stale or aliased;
* a second observation, different from the one the graphs were captured with,
  still matches eager — i.e. the input copies actually land;
* the torch.compile'd layer functions stay within a tight tolerance of eager
  (fusion reorders float reductions, so this one is not bit-exact).

The full-size numerical acceptance is the LeRobot parity oracle in
``test_pi05_parity.py`` (eager) and the OpenPI e2e in
``tests/e2e/online_serving/test_pi05_expansion.py`` (graph mode).
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.pi05 import modeling_pi05
from vllm_omni.diffusion.models.pi05.config import Pi05Config
from vllm_omni.diffusion.models.pi05.cuda_graph import Pi05CudaGraphRunner, serving_shaped_inputs
from vllm_omni.diffusion.models.pi05.modeling_pi05 import GemmaVariantConfig, Pi05ForActionPrediction

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a CUDA device"),
]

# Same shrink as test_pi05_units.py: width and the SigLIP tower stay full-size.
_SHRUNK_GEMMA = {
    "gemma_2b": GemmaVariantConfig(2048, 2, 1024, 8, 1, 256),
    "gemma_300m": GemmaVariantConfig(1024, 2, 512, 8, 1, 256),
}


@pytest.fixture(scope="module", autouse=True)
def _shrink_gemma_backbones():
    patch = pytest.MonkeyPatch()
    patch.setattr(modeling_pi05, "get_gemma_config", lambda variant: _SHRUNK_GEMMA[variant])
    yield
    patch.undo()


@pytest.fixture(scope="module")
def model():
    config = Pi05Config(max_action_dim=8, max_state_dim=8, chunk_size=4, n_action_steps=4)
    model = Pi05ForActionPrediction(config).to("cuda").eval()
    # The AdaRMS projections are zero-initialized, which would make every
    # denoise step ignore the timestep; give them something to do.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".dense." in name:
                param.normal_(std=0.02)
    return model


def _observation(seed: int, num_live_cameras: int, batch_size: int = 1):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    images = [torch.rand(batch_size, 3, 224, 224, device="cuda", generator=gen) * 2 - 1 for _ in range(3)]
    camera_counts = (torch.arange(batch_size, device="cuda") + num_live_cameras - 1) % 3 + 1
    masks = [i < camera_counts for i in range(3)]
    lang = torch.randint(0, 1000, (batch_size, 200), device="cuda", generator=gen)
    prompt_lengths = 40 + seed + torch.arange(batch_size, device="cuda")
    lang_mask = torch.arange(200, device="cuda")[None, :] < prompt_lengths[:, None]
    noise = torch.randn(batch_size, 4, 8, device="cuda", generator=gen)
    return dict(images=images, image_masks=masks, lang_tokens=lang, lang_masks=lang_mask, noise=noise, num_steps=3)


@pytest.mark.parametrize("precapture", [True, False])
def test_cuda_graph_replay_is_bit_identical_to_eager(model, precapture):
    observations = [_observation(1, 3), _observation(2, 1)]
    with torch.inference_mode():
        eager = [model.sample_actions(**obs) for obs in observations]

        runner = Pi05CudaGraphRunner(model)
        if precapture:
            runner.capture(batch_size=1)
        model.cuda_graph_runner = runner
        try:
            graphed = [model.sample_actions(**obs) for obs in observations]
            # And once more in reverse: the buffers must not remember the last obs.
            graphed_again = [model.sample_actions(**obs) for obs in reversed(observations)]
        finally:
            model.cuda_graph_runner = None

    for reference, replayed in zip(eager, graphed):
        assert torch.equal(reference, replayed)
    for reference, replayed in zip(reversed(eager), graphed_again):
        assert torch.equal(reference, replayed)
    assert len(runner._prefix) == 1 and len(runner._denoise) == 1


def test_denoise_step_rejects_a_foreign_kv_cache(model):
    runner = Pi05CudaGraphRunner(model)
    inputs = serving_shaped_inputs(model, batch_size=1)
    with torch.inference_mode():
        prefix_pad_masks, past_key_values = model.encode_prefix(
            inputs["images"], inputs["image_masks"], inputs["lang_tokens"], inputs["lang_masks"]
        )
        timestep = torch.ones(1, device="cuda")
        with pytest.raises(RuntimeError, match="encode_prefix"):
            runner.denoise_step(prefix_pad_masks, past_key_values, inputs["noise"], timestep)


def test_cuda_graph_replays_multiple_batch_sizes_without_recapture(model):
    observations = {
        bsize: [_observation(10 + bsize, 1, bsize), _observation(20 + bsize, 3, bsize)] for bsize in (1, 2, 3)
    }
    with torch.inference_mode():
        eager = {bsize: [model.sample_actions(**obs) for obs in cases] for bsize, cases in observations.items()}

        runner = Pi05CudaGraphRunner(model)
        # Exercise pre-capture for two non-adjacent sizes and lazy capture for
        # the remaining one when the interleaved replay sequence reaches it.
        runner.capture(batch_size=1)
        runner.capture(batch_size=3)
        model.cuda_graph_runner = runner
        try:
            replay_order = [(3, 0), (1, 0), (2, 0), (1, 1), (3, 1), (2, 1)]
            first = {(bsize, case): model.sample_actions(**observations[bsize][case]) for bsize, case in replay_order}
            graph_ids = {
                bsize: (id(runner._prefix[bsize].graph), id(runner._denoise[bsize].graph)) for bsize in (1, 2, 3)
            }

            # Replay new inputs in a different order. A stale static buffer or
            # a graph lookup keyed incorrectly by batch size breaks equality.
            second_order = [(2, 1), (3, 0), (1, 1), (2, 0), (1, 0), (3, 1)]
            second = {(bsize, case): model.sample_actions(**observations[bsize][case]) for bsize, case in second_order}
        finally:
            model.cuda_graph_runner = None

    assert set(runner._prefix) == {1, 2, 3}
    assert set(runner._denoise) == {1, 2, 3}
    assert graph_ids == {
        bsize: (id(runner._prefix[bsize].graph), id(runner._denoise[bsize].graph)) for bsize in (1, 2, 3)
    }
    for results in (first, second):
        for (bsize, case), replayed in results.items():
            assert torch.equal(eager[bsize][case], replayed)


def test_torch_compile_tracks_eager_and_captures(model):
    obs = _observation(3, 2)
    backbone = model.paligemma_with_expert
    with torch.inference_mode():
        eager = model.sample_actions(**obs)
        backbone.enable_torch_compile(dynamic=False, fullgraph=True)
        try:
            compiled = model.sample_actions(**obs)
            runner = Pi05CudaGraphRunner(model)
            runner.capture(batch_size=1)
            model.cuda_graph_runner = runner
            try:
                compiled_graphed = model.sample_actions(**obs)
            finally:
                model.cuda_graph_runner = None
        finally:
            backbone.disable_torch_compile()

    torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-4)
    assert torch.equal(compiled_graphed, compiled)
