# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-forcing sampler pieces for the distilled SANA-WM streaming Stage-1.

The distilled student is sampled with a fixed timestep list
(``denoising_step_list``, e.g. ``1000, 960, 889, 727, 0``) through the plain
flow-matching Euler update — NVlabs ``SelfForcingFlowEulerCamCtrl`` builds a
``FlowMatchEulerDiscreteScheduler(shift=1.0)`` on exactly these sigmas and
calls ``scheduler.step(-v, t, x, per_token_timesteps=...)``. No fresh noise is
mixed in between steps; the conditioning frame is held at ``t = 0`` and
restored after every step. Chunk boundaries come from
``create_autoregressive_segments``: the first chunk absorbs the remainder
(the conditioning frame), every later chunk is ``chunk_size`` frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig

SANA_WM_NUM_TRAIN_TIMESTEPS = 1000


@dataclass(frozen=True)
class SanaWmSelfForcingSchedule:
    """Distilled-student timestep schedule (``denoising_step_list[:-1]``)."""

    timesteps: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.timesteps) < 1:
            raise ValueError("Sana-WM self-forcing schedule needs at least one timestep.")
        if any(t <= 0 or t > SANA_WM_NUM_TRAIN_TIMESTEPS for t in self.timesteps):
            raise ValueError(
                f"Sana-WM self-forcing timesteps must lie in (0, {SANA_WM_NUM_TRAIN_TIMESTEPS}], got {self.timesteps}."
            )
        if self.timesteps[0] != SANA_WM_NUM_TRAIN_TIMESTEPS:
            raise ValueError(
                f"Sana-WM self-forcing schedule must start at {SANA_WM_NUM_TRAIN_TIMESTEPS}, got {self.timesteps}."
            )
        if any(later >= earlier for earlier, later in zip(self.timesteps, self.timesteps[1:])):
            raise ValueError(f"Sana-WM self-forcing timesteps must be strictly decreasing, got {self.timesteps}.")

    @classmethod
    def from_step_list(cls, denoising_step_list: Sequence[int]) -> SanaWmSelfForcingSchedule:
        """Build from the NVlabs ``denoising_step_list`` (must end with 0)."""
        steps = tuple(int(step) for step in denoising_step_list)
        if len(steps) < 2 or steps[-1] != 0:
            raise ValueError(f"Sana-WM denoising_step_list must have >= 2 entries and end with 0, got {steps}.")
        return cls(timesteps=steps[:-1])

    @classmethod
    def from_config(
        cls,
        config: SanaWmConfig,
        override: Sequence[int] | None = None,
    ) -> SanaWmSelfForcingSchedule:
        return cls.from_step_list(override if override is not None else config.denoising_step_list)

    @property
    def num_steps(self) -> int:
        return len(self.timesteps)

    @property
    def sigmas(self) -> tuple[float, ...]:
        return tuple(t / SANA_WM_NUM_TRAIN_TIMESTEPS for t in self.timesteps)

    def sigma_pair(self, step: int) -> tuple[float, float]:
        """``(sigma_step, sigma_next)``; the last step lands on ``sigma = 0``."""
        if step < 0 or step >= self.num_steps:
            raise IndexError(f"Sana-WM self-forcing step {step} outside [0, {self.num_steps}).")
        sigmas = self.sigmas
        sigma_next = sigmas[step + 1] if step + 1 < self.num_steps else 0.0
        return sigmas[step], sigma_next

    def timesteps_tensor(self, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.tensor(self.timesteps, dtype=torch.float32, device=device)

    def check_num_inference_steps(self, num_inference_steps: int | None) -> None:
        """Reject a request that asks for a different step count.

        The distilled student only produces sensible output on the schedule it
        was trained on, so ``num_inference_steps`` is validated rather than
        honoured (same contract as LingBot-World's DMD student).
        """
        if num_inference_steps is None:
            return
        if int(num_inference_steps) != self.num_steps:
            raise ValueError(
                f"Sana-WM streaming is a distilled student with a fixed {self.num_steps}-step schedule "
                f"{list(self.timesteps)}; num_inference_steps must be {self.num_steps} or omitted, "
                f"got {num_inference_steps}."
            )


SANA_WM_CHUNK_SPLIT_STRATEGY = "first_chunk_plus_one"


def validate_chunk_split_strategy(strategy: str) -> None:
    """Reject a ``chunk_split_strategy`` other than the implemented one.

    Only the NVlabs ``first_chunk_plus_one`` rule (chunk 0 absorbs the
    remainder) is implemented; a checkpoint declaring another strategy would
    otherwise be chunked silently with the wrong rule.
    """
    if strategy != SANA_WM_CHUNK_SPLIT_STRATEGY:
        raise ValueError(
            f"Sana-WM streaming only implements chunk_split_strategy={SANA_WM_CHUNK_SPLIT_STRATEGY!r}, "
            f"got {strategy!r}."
        )


def create_autoregressive_segments(
    total_frames: int,
    chunk_size: int,
    *,
    strategy: str = SANA_WM_CHUNK_SPLIT_STRATEGY,
) -> list[int]:
    """Chunk boundaries ``[0, c1, ..., total_frames]`` (NVlabs semantics).

    ``chunk_size``-frame chunks left to right; the first chunk absorbs the
    remainder, so with ``total_frames = 1 + chunk_size * k`` chunk 0 covers the
    conditioning frame plus one generated block and every later chunk is one
    block. ``strategy`` must name that rule (see
    :func:`validate_chunk_split_strategy`).
    """
    validate_chunk_split_strategy(strategy)
    if chunk_size <= 0:
        raise ValueError(f"Sana-WM chunk_size must be positive, got {chunk_size}.")
    if total_frames <= chunk_size:
        raise ValueError(
            f"Sana-WM streaming needs more than one chunk: latent frames {total_frames} <= chunk_size {chunk_size}."
        )
    remainder = total_frames % chunk_size
    boundaries = [0]
    for index in range(total_frames // chunk_size):
        end = boundaries[-1] + chunk_size
        if index == 0:
            end += remainder
        boundaries.append(end)
    return boundaries


def self_forcing_euler_step(
    latents: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma: float,
    sigma_next: float,
) -> torch.Tensor:
    """One flow-matching Euler step on the generated frames.

    Matches the sign convention of the per-token
    ``FlowMatchEulerDiscreteScheduler.step(-v, ...)`` call NVlabs makes (and
    the bidirectional ``SanaWmPipeline`` documents): the model output ``v``
    satisfies ``x_next = x - (sigma - sigma_next) * v``; the last step
    (``sigma_next = 0``) yields the clean ``x0``. The math runs in fp32 and is
    cast back to the latent dtype, which is what writing the scheduler's fp32
    result into NVlabs' bf16 latent buffer does.
    """
    if latents.shape != velocity.shape:
        raise ValueError(
            f"Sana-WM self-forcing step needs matching latent/velocity shapes, got {tuple(latents.shape)} "
            f"vs {tuple(velocity.shape)}."
        )
    delta = float(sigma) - float(sigma_next)
    stepped = latents.float() - delta * velocity.float()
    return stepped.to(latents.dtype)
