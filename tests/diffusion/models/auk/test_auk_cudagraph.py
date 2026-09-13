# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-request eager and CUDA-graph AuK DiT sampling equivalence."""

import pytest
import torch

from vllm_omni.diffusion.models.auk.auk_transformer import AuKTransformer, sample_latents
from vllm_omni.diffusion.models.auk.cudagraph_wrapper import AuKCUDAGraphWrapper


def _make_dit(device: str) -> AuKTransformer:
    torch.manual_seed(12)
    return (
        AuKTransformer(
            dim=32,
            heads=2,
            dim_head=16,
            ff_mult=2,
            latent_dim=4,
            text_hidden_dim=8,
            num_layers=2,
            num_single_layers=2,
        )
        .eval()
        .to(device)
    )


def _sample_inputs(device: str, offset: float = 0.0) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(42)
    return {
        "text": torch.randn(1, 7, 8, generator=generator, device=device) + offset,
        "c_mask": torch.ones(1, 7, dtype=torch.bool, device=device),
        "ref": torch.randn(1, 4, 4, generator=generator, device=device) - offset,
        "ref_mask": torch.ones(1, 4, dtype=torch.bool, device=device),
    }


def _assert_graph_matches_eager(
    dit: AuKTransformer,
    wrapper: AuKCUDAGraphWrapper,
    *,
    gen_frames: int,
) -> dict[str, torch.Tensor]:
    inputs = _sample_inputs("cuda")
    common = dict(
        **inputs,
        gen_frames=gen_frames,
        t_grid=[0.0, 0.4, 1.0],
        cfg_strength=0.0,
    )
    eager = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7))
    graph = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7), sampler=wrapper)
    torch.testing.assert_close(graph, eager, atol=3e-6, rtol=3e-5)
    return inputs


@pytest.mark.core_model
@pytest.mark.cpu
@torch.inference_mode()
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_single_request_graph_wrapper_cpu_falls_back_to_eager(cfg_strength: float, mocker) -> None:
    dit = _make_dit("cpu")
    wrapper = AuKCUDAGraphWrapper(dit)
    run_spy = mocker.spy(wrapper, "_run")
    run_cfg_spy = mocker.spy(wrapper, "_run_cfg")
    capture_spy = mocker.spy(wrapper, "_capture")
    inputs = _sample_inputs("cpu")
    common = dict(
        **inputs,
        gen_frames=9,
        t_grid=[0.0, 0.4, 1.0],
        cfg_strength=cfg_strength,
    )
    eager = sample_latents(dit, **common, generator=torch.Generator().manual_seed(7))
    graph = sample_latents(dit, **common, generator=torch.Generator().manual_seed(7), sampler=wrapper)
    torch.testing.assert_close(graph, eager)

    if cfg_strength >= 1e-5:
        run_spy.assert_not_called()
        assert run_cfg_spy.call_count == 2
    else:
        assert run_spy.call_count == 2
        run_cfg_spy.assert_not_called()

    capture_spy.assert_not_called()
    assert not wrapper._cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_single_request_graph_replay_matches_eager_and_updates_inputs(cfg_strength: float) -> None:
    dit = _make_dit("cuda")
    wrapper = AuKCUDAGraphWrapper(dit)

    for offset in (0.0, 0.25):
        inputs = _sample_inputs("cuda", offset)
        common = dict(
            **inputs,
            gen_frames=9,
            t_grid=[0.0, 0.4, 1.0],
            cfg_strength=cfg_strength,
        )
        eager = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7))
        graph = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7), sampler=wrapper)
        torch.testing.assert_close(graph, eager, atol=3e-6, rtol=3e-5)

        key = wrapper._key(torch.empty(1, 9, 4, device="cuda"), inputs["text"], inputs["ref"], cfg_strength >= 1e-5)
        assert key in wrapper._cache

        entry = wrapper._cache[key]
        torch.testing.assert_close(entry.static_text, inputs["text"])
        torch.testing.assert_close(entry.static_c_mask, inputs["c_mask"])
        torch.testing.assert_close(entry.static_ref, inputs["ref"])
        torch.testing.assert_close(entry.static_ref_mask, inputs["ref_mask"])
        torch.testing.assert_close(entry.static_timestep, torch.tensor(0.4, device="cuda"))

    assert len(wrapper._cache) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
def test_graph_lru_eviction_keeps_retained_entries_replayable() -> None:
    dit = _make_dit("cuda")
    wrapper = AuKCUDAGraphWrapper(dit, max_graphs=2)

    inputs_a = _assert_graph_matches_eager(dit, wrapper, gen_frames=9)
    key_a = wrapper._key(torch.empty(1, 9, 4, device="cuda"), inputs_a["text"], inputs_a["ref"], False)
    assert key_a in wrapper._cache

    inputs_b = _assert_graph_matches_eager(dit, wrapper, gen_frames=10)
    key_b = wrapper._key(torch.empty(1, 10, 4, device="cuda"), inputs_b["text"], inputs_b["ref"], False)
    assert key_b in wrapper._cache

    inputs_c = _assert_graph_matches_eager(dit, wrapper, gen_frames=11)
    key_c = wrapper._key(torch.empty(1, 11, 4, device="cuda"), inputs_c["text"], inputs_c["ref"], False)

    assert key_a not in wrapper._cache
    assert set(wrapper._cache) == {key_b, key_c}

    _assert_graph_matches_eager(dit, wrapper, gen_frames=10)
    _assert_graph_matches_eager(dit, wrapper, gen_frames=11)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
def test_graph_capture_failure_keeps_existing_entries_replayable(mocker) -> None:
    dit = _make_dit("cuda")
    wrapper = AuKCUDAGraphWrapper(dit, max_graphs=2)

    inputs_a = _assert_graph_matches_eager(dit, wrapper, gen_frames=9)
    key_a = wrapper._key(torch.empty(1, 9, 4, device="cuda"), inputs_a["text"], inputs_a["ref"], False)
    assert key_a in wrapper._cache

    original_capture = wrapper._capture

    def fail_shape_b(*inputs: torch.Tensor, **kwargs):
        if inputs[0].shape[1] == 10:
            return None
        return original_capture(*inputs, **kwargs)

    mocker.patch.object(wrapper, "_capture", side_effect=fail_shape_b)

    inputs_b = _assert_graph_matches_eager(dit, wrapper, gen_frames=10)
    key_b = wrapper._key(torch.empty(1, 10, 4, device="cuda"), inputs_b["text"], inputs_b["ref"], False)

    assert key_b in wrapper._failed_keys
    assert key_b not in wrapper._cache
    assert key_a in wrapper._cache

    _assert_graph_matches_eager(dit, wrapper, gen_frames=9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
def test_failed_graph_key_does_not_retry_capture(mocker) -> None:
    dit = _make_dit("cuda")
    wrapper = AuKCUDAGraphWrapper(dit)
    attempts = 0

    def fail_capture(*_inputs: torch.Tensor, **_kwargs) -> None:
        nonlocal attempts
        attempts += 1
        return None

    mocker.patch.object(wrapper, "_capture", side_effect=fail_capture)

    _assert_graph_matches_eager(dit, wrapper, gen_frames=10)
    _assert_graph_matches_eager(dit, wrapper, gen_frames=10)

    assert attempts == 1
    assert len(wrapper._failed_keys) == 1
