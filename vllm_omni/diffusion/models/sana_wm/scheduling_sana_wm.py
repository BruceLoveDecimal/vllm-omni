# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SANA-WM Stage-1 flow scheduler helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

SANA_WM_DEFAULT_INFERENCE_FLOW_SHIFT = 9.8


def shift_flow_timestep(
    timestep: torch.Tensor,
    shift: float = SANA_WM_DEFAULT_INFERENCE_FLOW_SHIFT,
) -> torch.Tensor:
    """Apply the inference flow-shift transform used by SANA-WM."""

    return shift * timestep / (1.0 + (shift - 1.0) * timestep)


class SanaWmFlowMatchScheduler:
    """Production flow-DPM-Solver++ scheduler wrapping diffusers.

    Replaces the shifted-Euler smoke with the production ``vis_sampler:
    flow_dpm-solver`` path (``inference_flow_shift=9.8``, 20-30 steps).
    Provides ``add_noise`` for first-frame latent conditioning (A.1).
    """

    def __init__(
        self,
        num_inference_steps: int,
        shift: float = SANA_WM_DEFAULT_INFERENCE_FLOW_SHIFT,
    ) -> None:
        if num_inference_steps <= 0:
            raise ValueError("Sana-WM scheduler num_inference_steps must be positive.")
        self.num_inference_steps = num_inference_steps
        self.shift = shift
        from diffusers import FlowMatchDPMSolverMultistepScheduler

        self._sched = FlowMatchDPMSolverMultistepScheduler(shift=shift)
        self._timesteps_device: torch.device | None = None

    def _ensure_timesteps(self, device: torch.device) -> None:
        if self._timesteps_device != device:
            self._sched.set_timesteps(self.num_inference_steps, device=device)
            self._timesteps_device = device

    def timesteps(self, *, device: torch.device) -> torch.Tensor:
        self._ensure_timesteps(device)
        return self._sched.timesteps

    def step(
        self,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        return self._sched.step(noise_pred, timestep, latents).prev_sample

    def add_noise(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Linear flow interpolation: sigma*noise + (1-sigma)*sample.

        ``sigma = timestep / num_train_timesteps`` maps the scheduler's integer
        timestep (0–1000 range) back to the [0, 1] noise level.
        """
        num_train = float(self._sched.config.num_train_timesteps)
        sigma = (timestep.float() / num_train).clamp(0.0, 1.0)
        while sigma.ndim < sample.ndim:
            sigma = sigma.unsqueeze(-1)
        return (sigma * noise + (1.0 - sigma) * sample).to(sample.dtype)


@dataclass(frozen=True)
class SanaWmFlowDpmScheduler:
    """Shifted-Euler smoke scheduler — kept for backward-compat with existing tests.

    New code should use :class:`SanaWmFlowMatchScheduler`.
    """

    num_inference_steps: int
    shift: float = SANA_WM_DEFAULT_INFERENCE_FLOW_SHIFT

    def __post_init__(self) -> None:
        if self.num_inference_steps <= 0:
            raise ValueError("Sana-WM scheduler num_inference_steps must be positive.")

    def timesteps(self, *, device: torch.device) -> torch.Tensor:
        base = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=device, dtype=torch.float32)[:-1]
        return shift_flow_timestep(base, self.shift)

    def deltas(self, *, device: torch.device) -> torch.Tensor:
        base = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=device, dtype=torch.float32)
        shifted = shift_flow_timestep(base, self.shift)
        return shifted[:-1] - shifted[1:]

    def step(self, latents: torch.Tensor, noise_pred: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        while delta.ndim < latents.ndim:
            delta = delta.unsqueeze(-1)
        return latents - delta.to(device=latents.device, dtype=latents.dtype) * noise_pred
