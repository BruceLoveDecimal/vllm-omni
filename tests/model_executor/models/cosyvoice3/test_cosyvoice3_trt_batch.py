# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batched flow calls on a TensorRT estimator whose profile is narrower than the batch."""

import types

import pytest
import torch
from omegaconf import DictConfig

from vllm_omni.model_executor.models.cosyvoice3 import flow_estimator_trt
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import CausalConditionalCFM

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CFM_PARAMS = DictConfig(
    {
        "sigma_min": 1e-06,
        "solver": "euler",
        "t_scheduler": "cosine",
        "training_cfg_rate": 0.2,
        "inference_cfg_rate": 0.7,
    }
)
FRAMES = 60


class _FakeStream:
    cuda_stream = 0

    def wait_stream(self, other):
        pass


class _FakeTrtContext:
    """Accepts batches up to ``max_batch`` rows, like a TensorRT profile."""

    def __init__(self, max_batch: int):
        self.max_batch = max_batch
        self.batches: list[int] = []

    def set_input_shape(self, name, shape):
        if name == "x":
            self.batches.append(int(shape[0]))
        return int(shape[0]) <= self.max_batch

    def set_tensor_address(self, name, address):
        pass

    def execute_async_v3(self, stream):
        return True


class _FakeTrtEstimator:
    def __init__(self, *, engine_max_batch: int, advertised_max_batch: int | None = None):
        self.io_dtype = torch.float32
        self.supports_attn_mask = True
        self.static_chunk_size = 50
        self.max_batch = engine_max_batch if advertised_max_batch is None else advertised_max_batch
        self.context = _FakeTrtContext(engine_max_batch)
        self.released = 0

    def max_batch_for(self, frames):
        return self.max_batch

    def acquire_estimator(self):
        return [self.context, _FakeStream()], object()

    def release_estimator(self, context, stream):
        self.released += 1


def _cfm(estimator) -> CausalConditionalCFM:
    return CausalConditionalCFM(in_channels=240, cfm_params=CFM_PARAMS, n_spks=1, spk_emb_dim=80, estimator=estimator)


def _inputs(rows: int):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(rows, 80, FRAMES, generator=g)
    mask = torch.ones(rows, 1, FRAMES)
    mu = torch.randn(rows, 80, FRAMES, generator=g)
    t = torch.rand(rows, generator=g)
    spks = torch.randn(rows, 80, generator=g)
    cond = torch.randn(rows, 80, FRAMES, generator=g)
    return x, mask, mu, t, spks, cond


@pytest.fixture
def fake_cuda_streams(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: _FakeStream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: torch.no_grad())


def test_oversized_batch_runs_in_engine_sized_slices_in_order(fake_cuda_streams):
    cfm = _cfm(_FakeTrtEstimator(engine_max_batch=2))
    seen = []

    def fake_run(self, x, mask, mu, t, spks, cond, attn_mask):
        seen.append((x.shape[0], attn_mask.shape[0]))
        # Each row's output is its own x plus its own t, so a misplaced row shows.
        return x + t.view(-1, 1, 1)

    cfm._run_trt_estimator = types.MethodType(fake_run, cfm)
    x, mask, mu, t, spks, cond = _inputs(6)  # three requests, CFG-doubled
    out = cfm.forward_estimator(x, mask, mu, t, spks, cond, streaming=True)

    assert seen == [(2, 2), (2, 2), (2, 2)]
    torch.testing.assert_close(out, x + t.view(-1, 1, 1), rtol=0, atol=0)


def test_batch_within_profile_is_one_engine_call(fake_cuda_streams):
    cfm = _cfm(_FakeTrtEstimator(engine_max_batch=8))
    cfm.forward_estimator(*_inputs(6), streaming=True)
    assert cfm.estimator.context.batches == [6]
    assert cfm.estimator.released == 1


def test_rejected_shape_raises_and_returns_the_context(fake_cuda_streams):
    # An engine that advertises more than its profile holds: the shape is
    # rejected, which must surface as an error, not a failed assert on
    # execute, and the context must go back to the pool.
    estimator = _FakeTrtEstimator(engine_max_batch=2, advertised_max_batch=8)
    cfm = _cfm(estimator)
    with pytest.raises(RuntimeError, match="optimization profile"):
        cfm.forward_estimator(*_inputs(4), streaming=True)
    assert estimator.released == 1


def test_profile_limits_pick_the_widest_batch_that_reaches_the_length():
    class _Engine:
        num_optimization_profiles = 2

        def get_tensor_profile_shape(self, name, profile):
            assert name == "x"
            return [((2, 80, 4), (2, 80, 500), (2, 80, 3000)), ((4, 80, 4), (16, 80, 500), (16, 80, 1000))][profile]

    wrapper = object.__new__(flow_estimator_trt.TrtContextWrapper)
    wrapper.profile_limits = flow_estimator_trt._engine_profile_limits(_Engine())
    assert wrapper.profile_limits == [(2, 3000), (16, 1000)]
    assert wrapper.max_batch_for(800) == 16
    assert wrapper.max_batch_for(2000) == 2
    assert wrapper.max_batch_for(4000) == 0
