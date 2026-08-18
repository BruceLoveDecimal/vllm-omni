# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HELPERS_PATH = (
    Path(__file__).resolve().parents[4]
    / "vllm_omni"
    / "model_executor"
    / "models"
    / "voxtral_tts"
    / "flow_matching_seed.py"
)


@functools.lru_cache(maxsize=1)
def _seed_module():
    spec = importlib.util.spec_from_file_location("voxtral_tts_flow_matching_seed", _HELPERS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seeded_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def test_sample_flow_matching_noise_same_seed_reproduces():
    sample_flow_matching_noise = _seed_module().sample_flow_matching_noise
    first = sample_flow_matching_noise(
        2, 7, device="cpu", dtype=torch.float32, generators=[_seeded_generator(42), _seeded_generator(7)]
    )
    second = sample_flow_matching_noise(
        2, 7, device="cpu", dtype=torch.float32, generators=[_seeded_generator(42), _seeded_generator(7)]
    )

    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_resolve_flow_matching_generators_reuses_per_request_stream():
    cache: dict[str, tuple[int, torch.Generator]] = {}
    device = torch.device("cpu")
    resolve = _seed_module().resolve_flow_matching_generators
    first = resolve([42], ["req-a"], device=device, cache=cache, active_req_ids=["req-a"])
    second = resolve([42], ["req-a"], device=device, cache=cache, active_req_ids=["req-a"])

    assert first is not None and second is not None
    assert first[0] is second[0]


def test_resolve_flow_matching_generators_isolates_same_seed_across_requests():
    helpers = _seed_module()
    cache: dict[str, tuple[int, torch.Generator]] = {}
    device = torch.device("cpu")
    generators = helpers.resolve_flow_matching_generators(
        [42, 42],
        ["req-a", "req-b"],
        device=device,
        cache=cache,
        active_req_ids=["req-a", "req-b"],
    )

    assert generators is not None
    assert generators[0] is not generators[1]
    first_a = helpers.sample_flow_matching_noise(1, 5, device=device, dtype=torch.float32, generators=[generators[0]])
    second_a = helpers.sample_flow_matching_noise(1, 5, device=device, dtype=torch.float32, generators=[generators[0]])
    first_b = helpers.sample_flow_matching_noise(1, 5, device=device, dtype=torch.float32, generators=[generators[1]])
    standalone = helpers.sample_flow_matching_noise(
        1, 5, device=device, dtype=torch.float32, generators=[_seeded_generator(42)]
    )
    torch.testing.assert_close(first_b, standalone, atol=0, rtol=0)
    torch.testing.assert_close(first_a, standalone, atol=0, rtol=0)
    assert not torch.equal(first_a, second_a)


@functools.lru_cache(maxsize=1)
def _voxtral_tts_model_cls():
    pytest.importorskip("vllm")
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()
    import vllm.model_executor.models.utils  # noqa: F401

    from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts import (
        VoxtralTTSForConditionalGeneration,
    )

    return VoxtralTTSForConditionalGeneration


def test_make_omni_output_forwards_seeded_generators():
    model_cls = _voxtral_tts_model_cls()
    model = model_cls.__new__(model_cls)
    model.model_stage = "audio_generation"
    model._cudagraph_acoustic_transformer = None
    model._tts_noise_generators = {}
    captured: dict[str, object] = {}

    class FakeAudioGeneration:
        def compute_mm_logits(self, hidden_states, cfg_alpha, generators=None):
            captured["generators"] = generators
            batch = hidden_states.shape[0]
            return torch.zeros(batch), {"codes": {"audio": [torch.zeros(1, 8) for _ in range(batch)]}}

    model.model = FakeAudioGeneration()
    hidden = torch.zeros(2, 4)

    model.make_omni_output(
        hidden,
        logits_index=torch.tensor([0, 1]),
        sampling_extra_args=[{"tts_local_seed": 11}, {"tts_local_seed": 22}],
        req_ids=["r1", "r2"],
        active_req_ids=["r1", "r2"],
    )

    generators = captured["generators"]
    assert isinstance(generators, list)
    assert generators[0] is not None
    assert generators[1] is not None
    assert generators[0] is not generators[1]
