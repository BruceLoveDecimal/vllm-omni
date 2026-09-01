# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.cache.seacache import (
    DenoiseStepTracker,
    SeaCacheBackend,
    SeaCacheConfig,
    SeaCacheRootHook,
    apply_sea_cache_hook,
)
from vllm_omni.diffusion.cache.seacache.sea_filter import (
    apply_sea_filter,
    extrapolate_residual,
    indicator_distance,
)
from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.data import DiffusionCacheConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TinyCosmos3Transformer(torch.nn.Module):
    """Small model that implements the SeaCache forward-control contract."""

    def _run_gen_layers(self, hidden_gen: torch.Tensor) -> torch.Tensor:
        return hidden_gen * 2

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        video_shape: tuple[int, int, int] | None = None,
        noisy_frame_mask: torch.Tensor | None = None,
        control_latents: list[torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timestep, text_ids, text_mask, video_shape, noisy_frame_mask
        controls = (
            []
            if control_latents is None
            else [control_latents]
            if isinstance(control_latents, torch.Tensor)
            else list(control_latents)
        )
        inputs = [*controls, hidden_states]
        gen_input = torch.cat(
            [value.movedim(1, -1).flatten(1, 3) for value in inputs],
            dim=1,
        )
        residual = getattr(self, "_seacache_residual", None)
        if getattr(self, "_seacache_skip", False) and isinstance(residual, torch.Tensor):
            return gen_input + residual
        output = self._run_gen_layers(gen_input)
        if getattr(self, "_seacache_record", False):
            self._seacache_last_residual = output - gen_input
        return output


class Cosmos3OmniDiffusersPipeline:
    def __init__(self) -> None:
        self.transformer = TinyCosmos3Transformer()
        self._num_timesteps = 0
        self.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))


def _latent(value: float) -> torch.Tensor:
    return torch.full((1, 2, 2, 2, 2), value)


def _run_step(
    transformer: TinyCosmos3Transformer,
    timestep: int,
    value: float,
    *,
    control: float | None = None,
    noisy_frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    controls = None if control is None else [_latent(control)]
    with torch.inference_mode():
        return transformer(
            hidden_states=_latent(value),
            timestep=torch.tensor([timestep]),
            noisy_frame_mask=noisy_frame_mask,
            control_latents=controls,
        )


def test_config_validation() -> None:
    assert SeaCacheConfig().threshold == 0.25
    with pytest.raises(ValueError, match="residual_order"):
        SeaCacheConfig(residual_order=-1)
    with pytest.raises(ValueError, match="max_consecutive_cached"):
        SeaCacheConfig(max_consecutive_cached=-1)


def test_sea_filter_matches_reference_equation() -> None:
    hidden = torch.randn(3, 4, 5, 2, dtype=torch.float32)
    sigma = 0.4
    power_exp = 3.0

    spectrum = torch.fft.fftn(hidden, dim=(0, 1, 2))
    gain = None
    for axis in (0, 1, 2):
        frequencies = torch.fft.fftfreq(hidden.shape[axis], dtype=torch.float32)
        clean_power = 1.0 / (frequencies.abs().pow(power_exp) + 1e-16)
        axis_gain = (1.0 - sigma) * clean_power / ((1.0 - sigma) ** 2 * clean_power + sigma**2 + 1e-16)
        shape = [1] * hidden.ndim
        shape[axis] = hidden.shape[axis]
        gain = axis_gain.reshape(shape) if gain is None else gain * axis_gain.reshape(shape)
    assert gain is not None
    gain = gain / gain.mean()
    expected = torch.fft.ifftn(spectrum * gain, dim=(0, 1, 2)).real

    torch.testing.assert_close(
        apply_sea_filter(hidden, sigma=sigma, power_exp=power_exp),
        expected,
    )


def test_indicator_distance_and_linear_extrapolation() -> None:
    previous = [torch.ones(2, 2)]
    current = [torch.full((2, 2), 1.5)]
    assert indicator_distance(current, previous) == pytest.approx(0.5)
    assert indicator_distance([torch.ones(3)], previous) == float("inf")

    history = [
        (0, torch.full((2, 2), 2.0)),
        (2, torch.full((2, 2), 6.0)),
    ]
    torch.testing.assert_close(
        extrapolate_residual(history, step=3, order=1),
        torch.full((2, 2), 8.0),
    )
    torch.testing.assert_close(
        extrapolate_residual(history, step=3, order=0),
        torch.full((2, 2), 6.0),
    )
    quadratic_history = [
        (0, torch.zeros(2, 2)),
        (1, torch.ones(2, 2)),
        (2, torch.full((2, 2), 4.0)),
    ]
    torch.testing.assert_close(
        extrapolate_residual(quadratic_history, step=3, order=2),
        torch.full((2, 2), 9.0),
    )


def test_tracker_handles_cfg_transfer_and_new_sample() -> None:
    tracker = DenoiseStepTracker()
    assert tracker.advance(1000.0) == (True, False)
    assert tracker.pass_name == "cfg0"
    assert tracker.advance(1000.0) == (False, False)
    assert tracker.pass_name == "cfg1"
    assert tracker.advance(1000.0) == (False, False)
    assert tracker.pass_name == "cfg2"
    assert tracker.advance(750.0) == (True, False)
    assert tracker.step == 1
    assert tracker.pass_name == "cfg0"
    assert tracker.advance(1000.0) == (True, True)


def test_hook_skips_middle_steps_and_forces_endpoints() -> None:
    transformer = TinyCosmos3Transformer()
    hook = apply_sea_cache_hook(
        transformer,
        SeaCacheConfig(threshold=100.0, max_consecutive_cached=2),
    )
    hook.refresh(transformer, num_inference_steps=4)

    for step, timestep in enumerate((1000, 750, 500, 250)):
        _run_step(transformer, timestep, 1.0 - step * 0.01)

    assert hook.full_count == 2
    assert hook.skip_count == 2
    assert [step for step, _ in hook.state_manager._states["cfg0"].history] == [0, 3]


def test_hook_keeps_three_transfer_branches_separate() -> None:
    transformer = TinyCosmos3Transformer()
    hook = apply_sea_cache_hook(
        transformer,
        SeaCacheConfig(threshold=100.0),
    )
    hook.refresh(transformer, num_inference_steps=3)

    for timestep, value in ((1000, 1.0), (500, 0.9)):
        _run_step(transformer, timestep, value, control=0.5)
        _run_step(transformer, timestep, value)
        _run_step(transformer, timestep, value, control=0.5)

    assert set(hook.state_manager._states) == {"cfg0", "cfg1", "cfg2"}
    assert hook.full_count == 3
    assert hook.skip_count == 3
    assert len(hook.state_manager._states["cfg0"].previous_indicator) == 2
    assert len(hook.state_manager._states["cfg1"].previous_indicator) == 1


def test_hook_fails_open_without_noisy_vision() -> None:
    transformer = TinyCosmos3Transformer()
    hook = apply_sea_cache_hook(transformer, SeaCacheConfig(threshold=100.0))
    hook.refresh(transformer, num_inference_steps=3)
    all_clean = torch.zeros(1, 1, 2, 1, 1)

    _run_step(transformer, 1000, 1.0, noisy_frame_mask=all_clean)
    _run_step(transformer, 500, 0.9, noisy_frame_mask=all_clean)

    assert hook.full_count == 0
    assert hook.skip_count == 0


def test_backend_selector_and_refresh() -> None:
    backend = get_cache_backend(
        "sea_cache",
        {
            "sea_threshold": 0.4,
            "sea_residual_order": 0,
        },
    )
    assert isinstance(backend, SeaCacheBackend)
    assert backend.config.sea_threshold == 0.4

    pipeline = Cosmos3OmniDiffusersPipeline()
    backend.enable(pipeline)
    backend.refresh(pipeline, num_inference_steps=7)
    hook = pipeline.transformer._hook_registry.get_hook(SeaCacheRootHook._HOOK_NAME)
    assert hook.num_inference_steps == 7


def test_backend_defers_pipeline_default_step_count() -> None:
    pipeline = Cosmos3OmniDiffusersPipeline()
    backend = SeaCacheBackend(DiffusionCacheConfig())
    backend.enable(pipeline)
    backend.refresh(pipeline, num_inference_steps=0)
    pipeline._num_timesteps = 35

    _run_step(pipeline.transformer, 1000, 1.0)

    hook = pipeline.transformer._hook_registry.get_hook(SeaCacheRootHook._HOOK_NAME)
    assert hook.num_inference_steps == 35
    assert hook.full_count == 1

    # The next request is refreshed before Cosmos3 resolves its mode-specific
    # default. It must adopt 50 rather than retaining the previous 35.
    backend.refresh(pipeline, num_inference_steps=0)
    pipeline._num_timesteps = 50
    _run_step(pipeline.transformer, 1000, 1.0)
    assert hook.num_inference_steps == 50
    assert hook.full_count == 1


def test_shared_config_defaults() -> None:
    config = DiffusionCacheConfig()
    assert config.sea_threshold == 0.25
    assert config.sea_residual_order == 1
    assert config.sea_max_consecutive_cached == 2
    assert config.sea_power_exp == 3.0
