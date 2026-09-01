# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class DenoiseStepTracker:
    """Infer diffusion step and positional branch from successive timesteps."""

    last_timestep_key: float | None = None
    step: int = -1
    pass_idx: int = 0

    def advance(self, timestep_key: float) -> tuple[bool, bool]:
        if self.last_timestep_key is None:
            self.last_timestep_key = timestep_key
            self.step = 0
            self.pass_idx = 0
            return True, False

        if timestep_key == self.last_timestep_key:
            self.pass_idx += 1
            return False, False

        is_new_sample = timestep_key > self.last_timestep_key
        self.last_timestep_key = timestep_key
        self.step += 1
        self.pass_idx = 0
        return True, is_new_sample

    def reset_for_new_sample(self) -> None:
        self.step = 0

    @property
    def pass_name(self) -> str:
        return f"cfg{self.pass_idx}"

    def reset(self) -> None:
        self.last_timestep_key = None
        self.step = -1
        self.pass_idx = 0


@dataclass(slots=True)
class SeaCacheState:
    """Per-positional-branch trajectory state."""

    accumulated_distance: float = 0.0
    previous_indicator: list[torch.Tensor] | None = None
    history: list[tuple[int, torch.Tensor]] = field(default_factory=list)
    consecutive_cached: int = 0

    def reset(self) -> None:
        self.accumulated_distance = 0.0
        self.previous_indicator = None
        self.history.clear()
        self.consecutive_cached = 0
