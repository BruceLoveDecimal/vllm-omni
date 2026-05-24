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


@dataclass(frozen=True)
class SanaWmFlowDpmScheduler:
    """Dependency-light flow scheduler for native SANA-WM smoke tests.

    The official release uses ``vis_sampler: flow_dpm-solver`` with
    ``inference_flow_shift=9.8``. This class preserves the public timestep
    contract for shape/e2e plumbing; exact numerical parity still belongs to the
    official backend until the upstream solver is ported.
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
